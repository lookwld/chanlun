"""币安全市场现货 K 线拉取 —— 供缠论全市场扫描用。

只拉 USDT 现货(TRADING 且 isSpotTradingAllowed), 默认周期 **8 档**:
月线 1M / 周线 1w / 3日线 3d / 日线 1d / 4小时 4h / 1小时 1h / 30分钟 30m / 15分钟 15m
—— 与 `tools/scan_spot_b1b2b3.py::LEVELS` 严格一致(那边扫什么, 这边就必须拉什么;
两边不一致的后果是某些整列"根数不足"全空, 而报告不会报错)。

⚠ 默认**不再拉 5m**: 扫描口径里 5m 已被 15m 取代(入场周期越细越亏, 见 `bt_results/`
的多周期回测结论), 留着只会让每轮多花 ~1/4 的时间。
⚠ `1M`/`1w`/`3d` 是**非均匀周期**, 只能这样直接读交易所原始 K 线, 不能由细周期聚合
(见 `chanbook/data.py::TF_MS` 的注释)。币安 2017-08 才上线 ⇒ 1M 只有 111 根、
1w 只有 477 根、3d 封顶 1000 根, **不随日期增长**。

每个周期 1000 根(币安单请求上限, 足够缠论 WARMUP+MIN_BARS)。

端点为 data-api.binance.vision(api.binance.com 被墙, 与下载 top20 时同款)。

输出: out/scan/klines/{SYM}_{tf}.json  (list of [ts,o,h,l,c,v,qv])

===============================================================================
2026-10-08 提速改造（服务器实测: 下载步原本 241s, 占整轮 361s 的 67%）
===============================================================================

1. **连接保活**。原来每个请求都 `urllib.urlopen` 新建 HTTPS 连接 —— 4048 个请求
   = 4048 次 TLS 握手, **那才是大头**, 不是数据量。改成每线程一条长连接:
   320 请求实测 16.8s → 5.4s(**3.1x**)。
2. **`--incremental` 增量**: 校验本地缓存 → 从「缓存末根」起取 → 按 ts 覆盖合并 →
   截回 1000 根。叠加保活后 320 请求 **3.1s(5.4x)**。

   实测 224 组抽样(头部蓝筹 / 中段 / 尾部新上线币): **已收盘的 K 线与全量下载
   逐根 100% 一致** —— 增量是**等价变换, 不是近似**。差异只可能落在最后一根
   「进行中」的 K 线上, 那根本来就在被实盘 tick 改写。

⚠ `--incremental` **默认关闭**: 研究口径保持"每次都从交易所重新取全量"的可复现性。
  运营路径(看板手动扫描)由 `tools/rescan_market.py --incremental` 显式打开。
⚠ 写文件改成「临时文件 + os.replace」: 中断不会留下半截 JSON 让下一轮的校验误判
  (每轮开跑前会扫掉残留的 `.tmp`)。
⚠ 请求带 `Accept-Encoding: gzip`, 全量路径的响应体可缩小数倍。
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

# `chanbook/data.py` 是零三方依赖的, 直接复用它取 TF_MS —— 保证与扫描器**同一口径**,
# 不会再出现"两处各写一份周期表、改一处忘一处"的漂移。
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from chanbook import data as D                                    # noqa: E402

BASE = "https://data-api.binance.vision/api/v3"
TFS = ["1M", "1w", "3d", "1d", "4h", "1h", "30m", "15m"]
LIMIT = 1000                      # 币安单请求上限, 也是滚动缓存保留的根数
MS_DAY = 86_400_000
BARS_STATE = "_bars.json"         # 上一轮各文件的根数(见 load_bars_state 的注释)


# --------------------------------------------------------------- HTTP(连接保活)

class Pool:
    """每线程一条长连接 —— 实测只做这一项, 下载步就快 3.1 倍(TLS 握手才是大头)。

    标准库没给现成的连接池, 用 `threading.local()` 手搓: 每个下载线程复用同一条
    `HTTPSConnection`, 出错就换一条新的重试。全部连接登记在 `_all` 里, 结束时统一关。
    """

    def __init__(self, host: str, timeout: int = 30):
        self.host, self.timeout = host, timeout
        self._t = threading.local()
        self._all: list = []
        self._lk = threading.Lock()

    def _new(self) -> http.client.HTTPSConnection:
        c = http.client.HTTPSConnection(self.host, timeout=self.timeout)
        with self._lk:
            self._all.append(c)
        return c

    def _conn(self) -> http.client.HTTPSConnection:
        c = getattr(self._t, "c", None)
        if c is None:
            c = self._new()
            self._t.c = c
        return c

    def _reconnect(self) -> http.client.HTTPSConnection:
        try:
            self._conn().close()
        except Exception:                                         # noqa: BLE001
            pass
        c = self._new()
        self._t.c = c
        return c

    def get(self, path: str, retries: int = 3) -> bytes:
        last: Exception | None = None
        for a in range(retries):
            c = self._conn()
            try:
                c.request("GET", path, headers={
                    "User-Agent": "chanbook-scan/2.0",
                    "Accept-Encoding": "gzip",
                    "Connection": "keep-alive",
                })
                r = c.getresponse()
                body = r.read()
                if r.status in (429, 418):          # 限频: 退避后重试
                    try:
                        wait = float(r.headers.get("Retry-After") or 0)
                    except (TypeError, ValueError):
                        wait = 0.0
                    time.sleep(min(wait or (1.5 * (a + 1)), 15))
                    last = RuntimeError(f"HTTP {r.status} 限频")
                    continue
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status} {body[:120]!r}")
                if (r.headers.get("Content-Encoding") or "").lower() == "gzip":
                    body = gzip.decompress(body)
                return body
            except Exception as e:                                # noqa: BLE001
                last = e
                if a == retries - 1:
                    break
                time.sleep(0.5 * (a + 1))
                self._reconnect()
        raise last if last else RuntimeError("请求失败")

    def close(self) -> None:
        with self._lk:
            for c in self._all:
                try:
                    c.close()
                except Exception:                                 # noqa: BLE001
                    pass
            self._all.clear()


def split_base(base: str) -> tuple[str, str]:
    """`--base` → (host, 端点路径)。

    ⚠ `--base` 要的是**端点基址(含 /api/v3)**, 不是站点根。传成站点根会让每次请求都
      404 —— 症状是"全部失败"而非报错, 调用方如果只看"跑完了没有"会误判成网络问题。
      已知踩过: rescan_market.py 一开始就少传了 /api/v3。
    """
    u = urllib.parse.urlsplit(base)
    if u.scheme != "https" or not u.netloc:
        raise ValueError(f"--base 必须是 https 端点基址, 收到 {base!r}")
    path = u.path.rstrip("/")
    if "/api" not in path:
        raise ValueError(f"--base 必须是端点基址(含 /api/v3), 收到 {base!r}")
    return u.netloc, path


# --------------------------------------------------------------- 数据校验

def grid_of(tf: str) -> int:
    """校验用的网格步长 = `min(周期长度, 1 天)`。

    ⚠ **别改回 `tf_ms`**（2026-10-08 实测定的这条, 差点写错）:
      - `1M` 的网格是**日历月**(每月 1 日 00:00 UTC), 间隔 28~31 天;
      - `1w` 的网格是**周一 00:00 UTC** —— epoch 是周四, 所以 ts % 7天 == 4 天;
      - `3d` 的网格相位也不与 epoch 对齐(ts % 3天 == 1 天)。
    拿 `ts % tf_ms == 0` 去校验的话, **503 个 1M 文件 + 503 个 1w + 337 个 3d
    会全部判为不合格** —— 症状是每份缓存都"校验不通过"、静默退回全量下载,
    提速效果归零**而且不报错**。
    `min(tf, 1天)` 这一条在 4024 个真实文件上实测 **0 例外**。
    """
    return min(D.TF_MS[tf], MS_DAY)


def validate_cache(rows, tf: str, now_ms: int) -> tuple[bool, str]:
    """增量前的本地数据校验。任一条不过 ⇒ **仅该文件**退回全量下载, 不波及其它。

    这是增量方案**唯一**的正确性依赖: 增量只在这份缓存可信时才是等价变换。

    注意**不校验"有没有缺口"**: 真实数据里缺口是物理事实(FTT / LUNA 下架停牌,
    实测 1w 有 5 个文件、3d 有 8 个、1d 有 1 个)。只要求「间隔是网格步长的正整数倍」——
    缺口能过, 而乱序 / 重复 / 相位错乱过不去。
    """
    if not isinstance(rows, list) or not rows:
        return False, "空文件"
    for i, r in enumerate(rows):
        if not isinstance(r, (list, tuple)) or len(r) < 7:
            return False, f"第{i}行列数不足"
        try:
            if min(float(r[1]), float(r[2]), float(r[3]), float(r[4])) <= 0:
                return False, f"第{i}行价格非正"
        except (TypeError, ValueError):
            return False, f"第{i}行非数值"
    g = grid_of(tf)
    prev = None
    for i, r in enumerate(rows):
        try:
            t = int(r[0])
        except (TypeError, ValueError):
            return False, f"第{i}行 ts 非整数"
        if t % g:
            return False, f"第{i}根 ts 未落在 {g}ms 网格上"
        if prev is not None:
            d = t - prev
            if d <= 0:
                return False, f"第{i}根 ts 未严格递增(乱序或重复)"
            if d % g:
                return False, f"第{i}根间隔 {d} 不是 {g} 的整数倍"
        prev = t
    if int(rows[-1][0]) > now_ms + D.TF_MS[tf]:
        return False, "末根 ts 落在未来"
    return True, f"{len(rows)} 根"


def merge_append(old: list, new: list, keep: int) -> list:
    """新根并入旧序列: **末根一律由 new 覆盖**, 其余按 ts 去重后升序, 截回最近 keep 根。

    - 覆盖末根是必须的: 缓存里那根是"进行中"的半截 K 线, 而 `startTime=末根` 的请求
      会把**它自己也返回**(实测首根 ts == 传入值) —— 覆盖掉它, 末尾就变成终值, 不会
      留下"永远停在半小时前的半根 K 线"被当已收盘根用。
    - 截断到 keep 是**等价性**的关键: 全量下载拿到的就是"最近 1000 根", 滚动缓存留
      最近 1000 根 = 同一个窗口 ⇒ 两种做法的结果可比。
    """
    last = int(old[-1][0])
    m = {int(r[0]): r for r in old if int(r[0]) < last}
    for r in new:
        m[int(r[0])] = r
    return [m[k] for k in sorted(m)][-keep:]


# --------------------------------------------------------------- 抓取

def load_symbols(exch_path: str) -> list[str]:
    d = json.load(open(exch_path, encoding="utf-8"))
    out = []
    for s in d.get("symbols", []):
        if (s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
                and s.get("isSpotTradingAllowed", True)):
            out.append(s["symbol"])
    return sorted(out)


def _norm(rows) -> list:
    """只保留 ts,o,h,l,c,v,qv(与既有落盘格式一致)。"""
    return [[int(r[0]), float(r[1]), float(r[2]), float(r[3]),
             float(r[4]), float(r[5]), float(r[7])] for r in rows]


def fetch_klines(pool: Pool, api: str, sym: str, tf: str,
                 start_ms: int | None = None) -> list | None:
    q = {"symbol": sym, "interval": tf, "limit": LIMIT}
    if start_ms is not None:
        q["startTime"] = int(start_ms)
    path = f"{api}/klines?" + urllib.parse.urlencode(q)
    try:
        return _norm(json.loads(pool.get(path).decode("utf-8")))
    except Exception as e:                                        # noqa: BLE001
        print(f"  ! {sym} {tf} 失败: {e}", file=sys.stderr, flush=True)
        return None


def write_atomic(path: str, rows: list) -> None:
    """临时文件 + `os.replace` —— 原子替换。中断只会留下 `.tmp`, 不会留下半截 JSON
    (半截 JSON 会被下一轮的校验当成"坏缓存", 症状是无声退回全量, 查起来很费劲)。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(rows, fh)
    os.replace(tmp, path)


def read_cache(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:                                             # noqa: BLE001
        return None


def load_bars_state(outdir: str) -> dict:
    """上一轮各文件的根数。用来发现**"缓存被截短"**这一类静默损坏。

    为什么非要有它: 增量只会**追加**。一份被截短的缓存(写到一半、被手工改小、
    被别的脚本覆盖)在纯追加下**永远不能自愈** —— 它会一直保持两三根, 而
    `validate_cache` 本身看不出问题（2 根也是合法的 2 根）。后果是该 (标的,周期)
    静默变成"根数不足"从表里消失, 报告照常生成。
    ⚠ 这不是假想: 2026-10-08 实测把某文件截到 2 根后重跑, 当时所有校验全部放行。
    """
    try:
        d = json.load(open(os.path.join(outdir, BARS_STATE), encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except Exception:                                             # noqa: BLE001
        return {}


def save_bars_state(outdir: str, state: dict) -> None:
    tmp = os.path.join(outdir, BARS_STATE + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(state, fh, separators=(",", ":"))
        os.replace(tmp, os.path.join(outdir, BARS_STATE))
    except OSError:
        pass


def update_one(pool: Pool, api: str, sym: str, tf: str, path: str,
               keep: int, incremental: bool, now_ms: int,
               prev_bars: dict | None = None) -> dict:
    """抓 / 更新**一个** (标的, 周期) 文件。返回统计用的状态字典。"""
    r = {"sym": sym, "tf": tf, "ok": False, "mode": "full",
         "reason": "", "bars_new": 0, "bars_total": 0, "secs": 0.0}

    def full(reason: str) -> dict:
        rows = fetch_klines(pool, api, sym, tf)
        if rows is None:
            return r
        r.update(ok=True, mode="full", reason=reason, bars_new=len(rows),
                 bars_total=len(rows))
        try:
            write_atomic(path, rows)
        except OSError:
            r["ok"] = False
        return r

    if not incremental:
        return full("")

    old = read_cache(path)
    if old is None:
        return full("无缓存")
    okv, why = validate_cache(old, tf, now_ms)
    if not okv:
        return full(f"校验不过({why})")

    # 缓存比上一轮**变小**了 ⇒ 被截短/被覆盖。增量只会追加, 靠它自己永远长不回来,
    # 必须整份重下（详见 load_bars_state 的注释）。
    prev_n = (prev_bars or {}).get(f"{sym}_{tf}")
    if prev_n is not None and len(old) < min(int(prev_n), keep):
        return full(f"缓存比上轮少({len(old)}<{min(int(prev_n), keep)})")

    last = int(old[-1][0])
    tfm = D.TF_MS[tf]
    # ★ 缺口溢出护栏 —— 本方案最危险的一条, 单独拦:
    #   缓存太旧时, 从末根起的增量请求会被 limit=1000 截断, 而币安是**从 startTime
    #   往后**返回的, 于是拿到的是缺口里**最老**的 1000 根、**最新那批反而丢了**。
    #   结果是一份看起来正常、其实停在很久以前的快照。宁可整份重下。
    if now_ms - last > keep * tfm:
        return full("缓存过旧(增量会被截断)")

    new = fetch_klines(pool, api, sym, tf, start_ms=last)
    if new is None:
        return r
    if not new:
        r.update(ok=True, mode="incr", reason="无新根", bars_new=0,
                 bars_total=len(old))
        return r
    if int(new[0][0]) != last:
        # 增量返回的首根不是缓存末根 ⇒ 缓存与交易所对不上(被污染 / 被换过)
        return full("首根与缓存末根不匹配")
    # 二次确认截断: 取满了 limit 根却仍远落后于现在 ⇒ 确实被截断了
    if len(new) >= keep and int(new[-1][0]) < now_ms - 2 * tfm:
        return full("增量被截断")

    merged = merge_append(old, new, keep)
    # ⚠ **不**对 merged 整份复检(2026-10-08 实测后改): 那是 O(行数) 的, 单这一项在全市场
    #   上就是几秒 CPU, 而它证明不了任何东西 —— old 刚校验过、new 是刚取回的、`new[0][0]`
    #   又等于 `old[-1][0]`, 所以 merged 唯一的新接缝就是「old 去掉末根」与 `new` 的衔接,
    #   O(1) 判一下单调性即可。改成只校验 new(长度很小)。
    if len(old) > 1 and int(old[-2][0]) >= int(new[0][0]):
        return full("合并后接缝不单调")
    okn, whyn = validate_cache(new, tf, now_ms)
    if not okn:
        return full(f"增量数据校验不过({whyn})")
    r.update(ok=True, mode="incr", bars_new=max(0, len(new) - 1),
             bars_total=len(merged))
    try:
        write_atomic(path, merged)
    except OSError:
        r["ok"] = False
    return r


# --------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exch", default="out/scan/exchangeInfo.json")
    ap.add_argument("--outdir", default="out/scan/klines")
    ap.add_argument("--limit", type=int, default=0, help="只拉前 N 个标的(调试)")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--tfs", default=",".join(TFS),
                    help="要拉的周期(逗号分隔); 默认与扫描器 LEVELS 一致")
    ap.add_argument("--base", default=BASE,
                    help="币安现货端点(默认 data-api.binance.vision)")
    ap.add_argument("--incremental", action="store_true",
                    help="校验本地缓存后只拉增量(默认关; 等价性见模块 docstring)")
    ap.add_argument("--keep-bars", type=int, default=LIMIT,
                    help="滚动缓存保留的根数(默认 1000 = 币安单请求上限)")
    args = ap.parse_args()

    try:
        host, api = split_base(args.base)
    except ValueError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2

    os.makedirs(args.outdir, exist_ok=True)
    # 清掉上一轮崩溃残留的半截文件(既占空间, 又会被当成坏缓存)
    stale = 0
    for fn in os.listdir(args.outdir):
        if fn.endswith(".tmp"):
            try:
                os.remove(os.path.join(args.outdir, fn))
                stale += 1
            except OSError:
                pass
    if stale:
        print(f"清掉 {stale} 个残留临时文件", flush=True)

    syms = load_symbols(args.exch)
    if args.limit:
        syms = syms[: args.limit]
    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    total = len(syms) * len(tfs)
    print(f"标的 {len(syms)} × 周期 {len(tfs)} = {total} 个文件  "
          f"模式={'增量' if args.incremental else '全量'}", flush=True)

    tasks = [(s, tf) for s in syms for tf in tfs]
    now_ms = int(time.time() * 1000)
    prev_bars = load_bars_state(args.outdir)
    pool = Pool(host)
    done = fail = 0
    modes: Counter = Counter()
    reasons: Counter = Counter()
    bars_state: dict = {}
    bars_new = 0
    t0 = time.time()

    def work(task):
        sym, tf = task
        path = os.path.join(args.outdir, f"{sym}_{tf}.json")
        return update_one(pool, api, sym, tf, path, args.keep_bars,
                          args.incremental, now_ms, prev_bars)

    try:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(work, t) for t in tasks]
            for f in as_completed(futs):
                st = f.result()
                done += 1
                if not st["ok"]:
                    fail += 1
                else:
                    modes[st["mode"]] += 1
                    bars_new += st["bars_new"]
                    bars_state[f"{st['sym']}_{st['tf']}"] = st["bars_total"]
                    if st["mode"] == "full" and args.incremental:
                        reasons[st["reason"] or "无缓存"] += 1
                if done % 200 == 0:
                    el = time.time() - t0
                    print(f"  {done}/{total} ({el:.0f}s, 失败{fail})", flush=True)
    finally:
        pool.close()

    # 只更新本轮碰过的条目, 其余沿用旧值(--limit / --tfs 子集不能把别处记录抹掉)
    if bars_state:
        merged_state = dict(prev_bars)
        merged_state.update(bars_state)
        save_bars_state(args.outdir, merged_state)

    el = time.time() - t0
    # ↓ 这一行被 `tools/rescan_market.py` 用正则解析, **改格式要同步改那边**
    print(f"完成 {done} 个, 失败 {fail} 个, 用时 {el:.0f}s", flush=True)
    if args.incremental:
        n_inc, n_full = modes.get("incr", 0), modes.get("full", 0)
        top = ", ".join(f"{k}×{v}" for k, v in reasons.most_common(5)) or "无"
        print(f"增量: 增量更新 {n_inc} 个(新增 {bars_new} 根 K 线), "
              f"全量回退 {n_full} 个; 回退原因: {top}", flush=True)
    else:
        print(f"全量: 写入 {modes.get('full', 0)} 个文件", flush=True)
    # 机器可读的一行 —— 编排器优先解析它, 解析不到才退回上面那行正则
    print("SUMMARY " + json.dumps({
        "files_ok": modes.get("full", 0) + modes.get("incr", 0),
        "files_fail": fail, "incr_ok": modes.get("incr", 0),
        "full_fallback": modes.get("full", 0) if args.incremental else 0,
        "bars_new": bars_new, "secs": round(el, 1),
        "mode": "incremental" if args.incremental else "full",
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
