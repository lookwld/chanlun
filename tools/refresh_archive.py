"""把 chanbook 的行情快照刷到最新。

为什么需要它: `chan-quant/data_archive/` 是本项目的**静态快照**, 全部 91 个现货
1h 文件的末根都停在 **2026-08-31 23:00 UTC**(北京 09-01 07:00)。而扫描器
(`scan_resonance.py`) 读的就是这份快照 —— 所以「扫当前市场」实际上扫的是**快照日**
那一天的市场, 不是今天。快照与真实「今天」之间的缺口, 本工具用币安公开端点补上。

做法: 只读打开源归档的 1h, 拉最近 --days 天的 1h K 线, **合并**成一份 chanbook
本地的 arch(默认 `out/arch/`), 再由 `CHANBOOK_ARCH` 环境变量喂给扫描器/回测。
4h / 1d 不落盘, 让 `data.load()` 继续从 1h 现场聚合 —— 这样反而比原来更一致
(原归档里只有 BTCUSDT 有一份原生 4h, 其余 90 个都是 1h 合成的, 两种口径混用)。

数据源: https://data-api.binance.vision  —— 币安公开 CDN, 无需 API key, 不吃 REST 限频。
  实测本机: data-api.binance.vision / data.binance.vision **可达**;
  api.binance.com / api1.binance.com / fapi.binance.com **超时不可达**。

硬约束: **绝不写 `chan-quant/`**。源归档只读打开, 全部产物落 `--dest`。

用法:
  python tools/refresh_archive.py --market spot --days 45 --workers 8
  CHANBOOK_ARCH=$PWD/out/arch python tools/scan_resonance.py --out out/resonance_scan.json
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

BASE = "https://data-api.binance.vision"        # 公开数据端点(唯一可达的币安域)
HDR = ["ts", "open", "high", "low", "close", "volume", "quote_volume", "trades"]
MS_HOUR = 3_600_000
DEF_SRC = os.path.join(os.path.dirname(ROOT), "chan-quant", "data_archive")
DEF_DEST = os.path.join(ROOT, "out", "arch")


# ------------------------------------------------------------------ 抓取
def _get(url: str, tries: int = 3, pause: float = 1.2) -> bytes:
    last = None
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                return r.read()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last = e
            if i < tries - 1:
                time.sleep(pause * (i + 1))
    raise RuntimeError(f"抓取失败: {last}")


def fetch_klines(sym: str, interval: str, start_ms: int, limit: int = 1000) -> list[list[str]]:
    """返回 [[ts, o, h, l, c, v, qv, trades], ...](只保留归档需要的 8 列)。"""
    url = (f"{BASE}/api/v3/klines?symbol={sym}&interval={interval}"
           f"&startTime={start_ms}&limit={limit}")
    raw = json.loads(_get(url).decode("utf-8"))
    out = []
    for k in raw:
        out.append([str(int(k[0])), k[1], k[2], k[3], k[4], k[5], k[7], str(int(k[8]))])
    return out


def fetch_since(sym: str, start_ms: int, max_bars: int, page: int = 1000) -> list[list[str]]:
    """从 start_ms 开始翻页拉到 max_bars 根。

    ⚠ REST 的最后一根是**未收盘**K线(盘中价), 必须丢掉 —— 归档约定是只存已收盘 bar,
      否则等于把"未来"写进了历史, 回测/扫描都会偷看。
    """
    out: list[list[str]] = []
    cur = start_ms
    now_ms = int(time.time() * 1000)
    while len(out) < max_bars:
        batch = fetch_klines(sym, "1h", cur, min(page, max_bars - len(out)))
        if not batch:
            break
        out.extend(batch)
        if len(batch) < page:
            break
        cur = int(batch[-1][0]) + MS_HOUR
    return [r for r in out if int(r[0]) + MS_HOUR <= now_ms]


# ------------------------------------------------------------------ 读写
def read_rows(path: str) -> list[list[str]]:
    if not os.path.exists(path):
        return []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rdr = csv.reader(fh)
        next(rdr, None)
        return [r for r in rdr if r]


def write_rows(path: str, rows: list[list[str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HDR)
        w.writerows(rows)
    os.replace(tmp, path)


def symbols_of(src: str, market: str) -> list[str]:
    for name in (f"{market}_symbols.txt" if market == "spot" else "all_symbols.txt",
                 "all_symbols.txt"):
        p = os.path.join(src, name)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as fh:
                return [ln.strip() for ln in fh if ln.strip()]
    return []


# ------------------------------------------------------------------ 单标的
def one(task: tuple) -> dict:
    sym, src, dest, days, allow_new = task
    sp = os.path.join(src, "spot", "k", f"{sym}_1h.csv.gz")
    dp = os.path.join(dest, "spot", "k", f"{sym}_1h.csv.gz")
    old = read_rows(sp)
    if not old and not allow_new:
        return {"sym": sym, "err": "源归档无 1h(新上市? 用 --allow-new 才拉)", "added": 0}
    last = int(old[-1][0]) if old else 0
    start = last + MS_HOUR                       # 只补缺口, 不重拉历史
    max_bars = max(1, days * 24) if days > 0 else 4000
    try:
        new = fetch_since(sym, start, max_bars)
    except Exception as e:                          # noqa: BLE001
        return {"sym": sym, "err": f"{e}", "added": 0,
                "old_last": last, "kept": len(old)}
    if not new:
        return {"sym": sym, "err": "", "added": 0, "old_last": last,
                "new_last": last, "kept": len(old), "merged": len(old)}
    merged = old + new
    write_rows(dp, merged)
    return {"sym": sym, "err": "", "added": len(new), "old_last": last,
            "new_last": int(new[-1][0]), "kept": len(old), "merged": len(merged)}


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=["spot"])
    ap.add_argument("--src", default=DEF_SRC, help="只读源归档(默认 chan-quant/data_archive)")
    ap.add_argument("--dest", default=DEF_DEST, help="产物目录(默认 chanbook/out/arch)")
    ap.add_argument("--days", type=int, default=45,
                    help="补最近多少个自然日的 1h(1h 上限 1000 根≈41 天, 超出会截断)")
    ap.add_argument("--symbols", default="", help="逗号分隔; 留空=清单全量")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--allow-new", action="store_true",
                    help="源归档里没有的标的也拉(只有最近 1000 根, 过不了预热, 仅供看看)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "refresh_report.json"))
    args = ap.parse_args()

    src = os.path.abspath(args.src)
    dest = os.path.abspath(args.dest)
    if "chan-quant" in dest.replace("\\", "/"):        # 硬约束护栏
        print("拒绝: --dest 不能落在 chan-quant/ 内", file=sys.stderr)
        return 2

    syms = ([s.strip() for s in args.symbols.split(",") if s.strip()]
            or symbols_of(src, args.market))
    if not syms:
        print(f"没读到标的清单({src})", file=sys.stderr)
        return 2

    tasks = [(s, src, dest, args.days, args.allow_new) for s in syms]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        rows = list(ex.map(one, tasks))

    # 清单文件一并搬过去, 让 data.symbols() 在本地 arch 下也能工作
    slab = "spot_symbols.txt" if args.market == "spot" else "all_symbols.txt"
    sp = os.path.join(src, slab)
    if os.path.exists(sp):
        with open(sp, encoding="utf-8") as fh:
            txt = fh.read()
        os.makedirs(dest, exist_ok=True)
        with open(os.path.join(dest, slab), "w", encoding="utf-8") as fh:
            fh.write(txt)
    p0 = os.path.join(src, f"{args.market}", "k")
    for extra in ("universe_spot.json", "universe_um.json", "all_symbols.txt"):
        if os.path.exists(os.path.join(src, extra)):
            os.makedirs(dest, exist_ok=True)
            with open(os.path.join(src, extra), "rb") as fi, \
                 open(os.path.join(dest, extra), "wb") as fo:
                fo.write(fi.read())

    ok = [r for r in rows if not r["err"]]
    bad = [r for r in rows if r["err"]]
    added = sum(r["added"] for r in ok)
    ends = sorted({r.get("new_last") or r.get("old_last") for r in ok if r.get("new_last")})
    span = time.time() - t0

    print(f"■ 刷新 {args.market}:{len(rows)} 个标的, 成功 {len(ok)} / 失败 {len(bad)}, "
          f"新增 {added} 根 1h, 耗时 {span:.1f}s")
    if ends:
        D = __import__("chanbook.data", fromlist=["fmt"])
        print(f"  合并后数据末端: {D.fmt(ends[-1])} ~ {D.fmt(ends[0])} (北京)   "
              f"覆盖 {len(ends)} 种末端时间")
    worse = [r for r in ok if r["added"] == 0]
    if worse:
        print(f"  无新增 {len(worse)} 个: " + ", ".join(r["sym"] for r in worse[:12])
              + (" ..." if len(worse) > 12 else ""))
    if bad:
        for r in bad[:10]:
            print(f"  ✗ {r['sym']}: {r['err']}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"src": src, "dest": dest, "days": args.days, "rows": rows},
                  fh, ensure_ascii=False, indent=1)
    print(f"  明细 -> {args.out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
