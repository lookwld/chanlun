"""缠论全市场扫描 —— 输出每个币每个周期当前处于哪个买卖点(B1/B2/B3/S1/S2/S3)。

数据: out/scan/klines/{SYM}_{tf}.json(由 download_all_klines.py 拉取)。

口径: 最新收盘快照。喂完该周期全部 K 线后, 在"最后一根已冻结笔"上扫一次:
  * 买点(向下笔冻结): Level.scan_points() -> B1/B2/B3
  * 卖点(向上笔冻结): SellScanner.scan() -> S1/S2/S3
  生产默认口径(BookCfg.legacy 的 scanner_kw), 不动任何策略开关。

⚠ 买卖点都是"事件触发"的: 只在笔冻结那一刻产生。所以"当前处于 X 点"的语义是
  "最近一根已冻结的笔上出现了 X 点", 不是"价格正在 X 点位置"。若最后一根笔
  冻结时没触发任何点, 则该周期该币输出"无信号"。

输出: out/scan/result.txt(纯文本) + out/scan/result.csv(表格)。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1, B2, B3                            # noqa: E402
from chanbook.sell import S1, S2, S3, SellScanner                 # noqa: E402
from chanbook.strategy import BookCfg                             # noqa: E402

TFS = ["5m", "15m", "1h", "4h", "1d"]
MIN_BARS = 500                    # 与 level.MIN_KLINES["op"] 一致
WINDOW_BI = 10                    # 有效窗口: 最近 N 根已冻结笔内出现的信号
WINDOW_HOURS = 72                 # 有效窗口: 最近 N 小时内(取两个窗口的交集)

BUY_LABEL = {B1: "B1 买①背驰", B2: "B2 买②确认", B3: "B3 买③三买"}
SELL_LABEL = {S1: "S1 卖①背驰", S2: "S2 卖②", S3: "S3 卖③"}


def _kl_from_json(rows: list) -> D.Kl:
    kl = D.Kl()
    ts, o, h, l, c, v, qv = [], [], [], [], [], [], []
    for r in rows:
        ts.append(int(r[0]))
        o.append(float(r[1]))
        h.append(float(r[2]))
        l.append(float(r[3]))
        c.append(float(r[4]))
        v.append(float(r[5]))
        qv.append(float(r[6]))
    # 直接填 array 字段(与 Kl.__init__ 同款列式, 避免重复解析)
    from array import array
    kl.ts = array("q", ts)
    kl.o = array("d", o)
    kl.h = array("d", h)
    kl.l = array("d", l)
    kl.c = array("d", c)
    kl.v = array("d", v)
    kl.qv = array("d", qv)
    kl.n = len(ts)
    kl._dl = None
    return kl


def scan_symbol_tf(sym: str, tf: str, kl_dir: str, scanner_kw: dict):
    """返回该标的该周期最近窗口内仍有效的买卖点命中列表。

    逐根 advance, 每根后调 scan_points() + SellScanner.scan() 累积所有历史命中
    (带笔下标 leg_i 与 bar 时间戳), 最后只保留「最近 WINDOW_BI 根冻结笔 且
    最近 WINDOW_HOURS 小时」内的命中。这样 B2/S2 这类"前后两笔配合"的确认点
    也能被捕获(只要出现在窗口内)。
    """
    path = os.path.join(kl_dir, f"{sym}_{tf}.json")
    if not os.path.exists(path):
        return sym, tf, None, None, "无数据"
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except Exception:
        return sym, tf, None, None, "读取失败"
    if not rows:
        return sym, tf, None, None, "空数据"
    kl = _kl_from_json(rows)
    if kl.n < MIN_BARS:
        return sym, tf, None, None, f"根数不足({kl.n})"

    tf_ms = D.TF_MS[tf]
    lv = Level("op", tf, "bi", scanner_kw=scanner_kw).attach(kl)
    sscan = SellScanner(pull_bi=scanner_kw.get("pull_bi", 1),
                        tol=0.0,
                        beichi_tol=scanner_kw.get("beichi_tol", 0.02),
                        beichi_strict=scanner_kw.get("beichi_strict", False),
                        s2_max_bi=scanner_kw.get("b2_max_bi", 20),
                        s1_max_lag=scanner_kw.get("b1_max_lag", 12),
                        s1_zs_window=scanner_kw.get("b1_zs_window", 1),
                        s3_zs_window=scanner_kw.get("b3_zs_window", 2),
                        s1_allow_no_zs=scanner_kw.get("b1_allow_no_zs", True),
                        enable=(S1, S2, S3))

    # 逐根喂入, 累积命中 (kind, price, leg_i, bar_ts)
    hits = []
    last_ts = int(kl.ts[-1])
    for i in range(kl.n):
        close_ts = int(kl.ts[i]) + tf_ms
        lv.advance(close_ts)
        for p in lv.scan_points():
            hits.append((p.kind, float(p.price), int(p.leg_i), int(kl.ts[i])))
        for p in sscan.scan(lv.bib.bis, lv.zss_view(), lv.macd, lv.bib.n_final):
            hits.append((p.kind, float(p.price), int(p.leg_i), int(kl.ts[i])))

    if not hits:
        return sym, tf, [], float(kl.c[-1]), None

    # 窗口过滤: 最近 WINDOW_BI 根冻结笔 且 最近 WINDOW_HOURS 小时
    n_final = lv.bib.n_final
    leg_min = max(0, n_final - WINDOW_BI)
    ts_cut = last_ts - WINDOW_HOURS * 3600_000
    kept = []
    for kind, price, leg_i, bar_ts in hits:
        if leg_i >= leg_min and bar_ts >= ts_cut:
            kept.append((kind, price, leg_i, bar_ts))
    # 去重(同类型同笔只留一次), 按类型排序输出
    seen = set()
    out = []
    for kind, price, leg_i, bar_ts in kept:
        key = (kind, leg_i)
        if key in seen:
            continue
        seen.add(key)
        label = BUY_LABEL.get(kind, SELL_LABEL.get(kind, kind))
        out.append((kind, price, label, leg_i, bar_ts))
    out.sort(key=lambda x: (x[3], x[0]))
    return sym, tf, out, float(kl.c[-1]), None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl_dir", default="out/scan/klines")
    ap.add_argument("--out", default="out/scan/result")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--tfs", default="5m,15m,1h,4h,1d")
    ap.add_argument("--syms", default="", help="只扫这些标的(逗号分隔, 调试用)")
    args = ap.parse_args()

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    if args.syms:
        syms = [s.strip() for s in args.syms.split(",") if s.strip()]
    else:
        # 从 klines 目录里反推标的(去重)
        seen = set()
        syms = []
        for fn in sorted(os.listdir(args.kl_dir)):
            if not fn.endswith(".json"):
                continue
            sym = fn.rsplit("_", 1)[0]
            if sym not in seen:
                seen.add(sym)
                syms.append(sym)

    scanner_kw = BookCfg.legacy(market="spot", kinds=(B1, B2, B3)).scanner_kw()

    tasks = [(s, tf) for s in syms for tf in tfs]
    print(f"扫描 {len(syms)} 标的 × {len(tfs)} 周期 = {len(tasks)} 组合", flush=True)

    results = []
    t0 = time.time()
    done = 0
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(scan_symbol_tf, s, tf, args.kl_dir, scanner_kw): (s, tf)
                    for s, tf in tasks}
            for f in as_completed(futs):
                sym, tf, hits, close, err = f.result()
                results.append((sym, tf, hits, close, err))
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    else:
        for s, tf in tasks:
            r = scan_symbol_tf(s, tf, args.kl_dir, scanner_kw)
            results.append(r)
            done += 1

    # ---- 汇总 ----
    hits_only = [r for r in results if r[2]]
    print(f"\n扫描完成, 共 {len(results)} 组合, {len(hits_only)} 个命中", flush=True)

    txt_lines = []
    txt_lines.append("# 缠论全市场扫描结果(最近窗口内有效信号)")
    txt_lines.append(f"# 标的 {len(syms)} × 周期 {len(tfs)} · 生成于 "
                     f"{time.strftime('%Y-%m-%d %H:%M')}")
    txt_lines.append(f"# 语义: 最近 {WINDOW_BI} 根冻结笔 且 最近 {WINDOW_HOURS} 小时内"
                     "出现的买卖点; 空 = 无信号")
    txt_lines.append("")

    csv_rows = [["symbol", "tf", "kind", "price", "close", "label",
                 "bar_ts"]]
    # 按 标的 -> 周期 排序
    tf_order = {t: i for i, t in enumerate(tfs)}
    results.sort(key=lambda r: (r[0], tf_order.get(r[1], 99)))

    cur_sym = None
    for sym, tf, hits, close, err in results:
        if sym != cur_sym:
            if cur_sym is not None:
                txt_lines.append("")
            cur_sym = sym
            txt_lines.append(f"## {sym}")
        if err:
            txt_lines.append(f"  {tf:<4} {err}")
            continue
        if not hits:
            txt_lines.append(f"  {tf:<4}  —")
            continue
        for kind, price, label, leg_i, bar_ts in hits:
            when = time.strftime("%m-%d %H:%M",
                                 time.localtime(bar_ts / 1000))
            txt_lines.append(f"  {tf:<4} {kind}  {label}  点位 {price:.6g}  "
                             f"现价 {close:.6g}  ({when})")
            csv_rows.append([sym, tf, kind, f"{price:.6g}", f"{close:.6g}",
                             label, bar_ts])

    txt = "\n".join(txt_lines) + "\n"
    with open(args.out + ".txt", "w", encoding="utf-8") as fh:
        fh.write(txt)
    with open(args.out + ".csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerows(csv_rows)

    print(f"\n结果已写入 {args.out}.txt / {args.out}.csv")
    # 统计各类型命中数
    from collections import Counter
    cnt = Counter()
    for r in hits_only:
        for kind, *_ in r[2]:
            cnt[kind] += 1
    print("各类型命中数:", dict(cnt))
    return 0


if __name__ == "__main__":
    sys.exit(main())
