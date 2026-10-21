"""现货主引擎 · 多变体回测器 —— tf × 构造(kinds) × 离场 × 8 个新增开关。

## 为什么单独一个工具

`tools/backtest_b12_tf.py` 是 2026-09-28 那一轮的产物, 它把「入场 = B1/B2」与
「离场 = 结构止损 + S1」**写死在代码里**; 本轮要测的是一批**可开关的引擎改造**
(见 `THEORY_GAP_2026-09-29.md`), 需要把它们全部参数化。B12 工具**一行未改**,
4h/2h/1h 的历史报告可原样复现。

## 本工具测什么

8 项改造, 全部默认关闭(`BookCfg` 的默认值 = 改动前行为):

| 代号 | 开关 | 对应审计项 | 含义 |
|---|---|---|---|
| E1 | `sell_exit` | B1 类 | 卖点(S1/S2/S3)接入主引擎作为第 5 条离场分支 |
| E2 | `b1_dir` | A2 | B1 方向闸门: off / no_up(排除上涨) / down(只要下跌) |
| E3 | `pending_out` | A3 | 把中阴(PENDING)移出 B2/B3 放行集合 |
| E4 | `b1_path` + `b1_trend_adjacent` | A1 | 一买路径取舍; 趋势背驰改回"紧邻前一中枢下移" |
| E5 | `vol_gate` | B2 | 量能判据: 后段均量/前段均量 < 阈值 |
| E6 | `ma_mode` | B6 | 两条均线: 否决型(ma/close/both) 或 仓位型(size) |
| E7 | `pos_layer` | B8 | 仓位分层: 按 amp_ratio / ratio_area / 买点类型 |
| E8 | `zs_merge` | B3 | 中枢扩展(相邻同向重叠中枢合并 ⇒ 级别上移) |

## 统计口径(本轮新增两项, 直接针对上一轮翻出的口径级问题)

上一轮的教训是 **"所有配置的期望只来自 ≤10 笔"** —— 均值会被极少数大赢家拉动,
而中位数在「低胜率 + 高盈亏比」的系统里天然为负、没有区分力。所以本工具每行都同时给:

  * `剔尾5%均值` —— 两侧各剔 5% 分位后的均值(对单笔极端值不敏感)
  * `中位` —— 分位数视角
  * `vs基准` —— **分标的配对**: 先算每个标的的均值, 再对"配对差"做符号检验
    (多少比例的标的变好)。跨标的的配对比合并池的均值稳健得多, 因为它不让
    某一个标的的 3 笔大赢家主导结论。

## 无未来函数

与 B12 工具同款: 每根 bar 先 `advance(bar_ts + tf_ms)` 把"本根已收盘"喂进去,
两级共用同一个 `now`(2026-09-29 由 `tests/check_timebase.py` 静态守住)。

运行示例:
  python tools/backtest_engine.py --round r1 --jobs 10 --out out/eng_r1.txt
"""

from __future__ import annotations

import argparse
import bisect
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook import report as R                                  # noqa: E402
from chanbook import risk                                         # noqa: E402
from chanbook.engine import (Diag, Trade, _close, _regime_at,     # noqa: E402
                             manage, open_pos, sell_hit, struct_line)
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1, B2, B3                            # noqa: E402
from chanbook.sell import S1, S2, S3, SellScanner                 # noqa: E402
from chanbook.strategy import BookCfg, Strategy                   # noqa: E402
from chanbook.trend import PENDING, TrendState                    # noqa: E402

# 2h 不在 data.TF_MS 里 —— 只在本进程内补齐(与 probe_levels.py / B12 同款处理)。
D.TF_MS.setdefault("2h", 2 * D.MS_HOUR)

WARMUP = 300                       # 与引擎一致
MIN_BARS = 500                     # 与 level.MIN_KLINES["op"] 一致
TREND_TF = "4h"                    # 方向闸门用的高一级别(与 B12 一致)
TREND_MIN_BARS = 200               # 与 level.MIN_KLINES["trend"] 一致

TF_HOURS = {"1h": 1, "2h": 2, "3h": 3, "4h": 4, "8h": 8, "1d": 24}
BAR_MIN = {"1h": 60, "2h": 120, "3h": 180, "4h": 240, "8h": 480, "1d": 1440}

KIND_MAP = {"b1": B1, "b2": B2, "b3": B3}
KIND_ALIAS = {(B1,): "b1", (B1, B2): "b1b2", (B1, B2, B3): "b1b2b3",
              (B2,): "b2", (B3,): "b3"}


# ------------------------------------------------------------------ 离场口径
# mgmt: "manage+struct" = 追踪止损 + RR目标 + 结构破位(引擎默认路径)
#       "stop"          = 只有结构止损(灾难点), 与 B12 的 chan1_stop 同款
# sell: 卖点输出过滤器(() = 不接卖点)
#
# ⚠ 2026-09-29 第二轮: 主引擎现在**原生支持**这两个 mgmt 口径
#   (`BookCfg.exit_mode` = "old" / "stop"), 语义与本表一一对应。本工具仍保留
#   自己的内联实现(它要自由指定操作级别周期, 走的是自建的 `Level`/`_LS`),
#   因此**两处实现必须同步**: 动 `engine.run_symbol` 的离场分支时, 这里也要改,
#   反之亦然。主引擎路径下的实测结果见 `out/eng_path.txt`
#   (`tools/backtest_engine_path.py`)。
EXITS = {
    "old": dict(sell=(), mgmt="manage+struct",
                why="引擎默认: 追踪3.0×risk + RR2目标 + 结构破位"),
    "stop": dict(sell=(), mgmt="stop",
                 why="只有结构止损(灾难点), 无追踪/无目标/无卖点"),
    "s1": dict(sell=(S1,), mgmt="stop",
               why="结构止损 + 一卖 S1(B12 的 chan1_stop)"),
    "s123": dict(sell=(S1, S2, S3), mgmt="stop",
                 why="结构止损 + 全部三类卖点(B12 的 chan123_stop)"),
    "s1_trail": dict(sell=(S1,), mgmt="manage+struct",
                     why="追踪+RR+结构破位 **再加** 一卖 S1(新组合)"),
    "s123_trail": dict(sell=(S1, S2, S3), mgmt="manage+struct",
                       why="追踪+RR+结构破位 **再加** 三类卖点(新组合)"),
}


# ------------------------------------------------------------------ 数据
def load_tf(market: str, sym: str, tf: str):
    """统一走 1h 底档; tf='1h' 直接用, 其余现场聚合(口径同源)。"""
    base = D.load(market, sym, "1h")
    if base is None or base.n == 0:
        return None
    if tf == "1h":
        return base
    return D.resample(base, D.TF_MS[tf])


def build_cfg(market: str, kinds: tuple, v: dict) -> BookCfg:
    """把变体字典变成 `BookCfg`。**未列出的字段一律取改动前行为。**

    ⚠ 2026-09-29 升级后必须以 `legacy` 为基底: 本工具的 74 个变体就是升级依据
      本身, 必须逐位可复现。`exit_mode` 由 `EXITS[...]["mgmt"]` 显式映射 ——
      本工具自己实现离场(不读 `cfg.exit_mode`), 但显式写出来可防止未来误读。
    """
    keys = ("b1_dir", "pending_out", "b1_path", "b1_trend_adjacent", "vol_gate",
            "ma_mode", "ma_fast", "ma_slow", "pos_layer", "zs_merge",
            "beichi_strict", "pull_bi", "b1_allow_no_zs")
    kw = {k: v[k] for k in keys if k in v}
    ex = EXITS[v["exit"]]
    # top20 数据本质是现货, 风控/策略语义复用 spot(market 只影响 risk.CALIB 等参数表)
    cfg_market = "spot" if market == "top20" else market
    return BookCfg.legacy(market=cfg_market, kinds=tuple(kinds),
                          sell_exit=tuple(ex["sell"]),
                          exit_mode=("stop" if ex["mgmt"] == "stop" else "old"),
                          **kw)


# ------------------------------------------------------------------ 单标的
class _LS:
    """`Strategy.pick` 需要的最小接口适配器(op / trend / sym)。

    为什么不直接用 `LevelSet`: 本工具要自由指定操作级别的时间周期(1h/2h/4h),
    而 `LEVEL_TF["spot"]` 把 op 级别钉在 1h 上。用一个三字段的适配器复用
    `Strategy` 的全部闸门逻辑, 比复制一遍闸门代码安全得多。
    """

    __slots__ = ("sym", "op", "trend")

    def __init__(self, sym: str, op: Level, trend) -> None:
        self.sym = sym
        self.op = op
        self.trend = trend


class _NoTrend:
    """闸门关闭时的占位趋势级别: 恒为 `PENDING`(中阴/未完成)。

    语义: "没有方向信息" ⇒ 用默认 `dir_gate` 时 B2/B3 仍放行(`PENDING` 在放行集合里),
    与 B12 工具 `--gate off` 的行为一致。
    """

    __slots__ = ("state",)

    def __init__(self) -> None:
        self.state = TrendState(kind=PENDING, incomplete=True, why="闸门关闭")


def _sell_scanner(cfg: BookCfg) -> SellScanner | None:
    if not cfg.sell_exit:
        return None
    skw = cfg.scanner_kw()
    return SellScanner(
        pull_bi=skw["pull_bi"], tol=0.0, beichi_tol=skw["beichi_tol"],
        beichi_strict=skw["beichi_strict"], s2_max_bi=skw["b2_max_bi"],
        s1_max_lag=skw["b1_max_lag"], s1_zs_window=skw["b1_zs_window"],
        s3_zs_window=skw["b3_zs_window"], s1_allow_no_zs=skw["b1_allow_no_zs"],
        enable=tuple(cfg.sell_exit))


def _trend_level(sym: str, market: str):
    tkl = load_tf(market, sym, TREND_TF)
    if tkl is None or tkl.n < TREND_MIN_BARS:
        return None
    return Level("trend", TREND_TF, "seg").attach(tkl)


def run_symbol(sym: str, v: dict, tf: str, market: str, kinds: tuple,
               gate: bool, t_start: int = 0, t_end: int = 0):
    kl = load_tf(market, sym, tf)
    dg = Diag(sym=sym)
    if kl is None:
        dg.err = "无 1h 底档"
        return [], dg
    if kl.n < MIN_BARS:
        dg.err = f"{tf} 根数不足 {kl.n}"
        return [], dg
    cfg = build_cfg(market, kinds, v)
    extra = cfg.op_extra()
    lv = Level("op", tf, "bi", scanner_kw=cfg.scanner_kw(), **extra).attach(kl)
    tlv = _trend_level(sym, market) if gate else None
    if gate and tlv is None:
        dg.err = f"趋势级别({TREND_TF})样本不足"
        return [], dg
    trend_lv = tlv if tlv is not None else _NoTrend()
    sscan = _sell_scanner(cfg)
    ex = EXITS[v["exit"]]
    mgmt = ex["mgmt"]
    sell_kinds = tuple(ex["sell"])
    strat = Strategy(cfg)
    ls = _LS(sym, lv, trend_lv)
    # top20 数据本质是现货, 风控校准复用 spot 口径(risk.CALIB 只有 spot/um 键)
    _risk_mkt = "spot" if market == "top20" else market
    tm = risk.trail_mult_of(_risk_mkt)
    trades: list[Trade] = []
    pos = None
    start = WARMUP
    if t_start:
        start = max(start, bisect.bisect_right(kl.ts, t_start - 1))
    dg.n_op_bars = max(0, kl.n - start)
    for i in range(start, kl.n):
        bar_ts = int(kl.ts[i])
        if t_end and bar_ts > t_end:
            break
        # ⚠ **唯一时钟**: 两级必须共用同一个"当前时刻"(见 check_timebase.py)。
        now = bar_ts + lv.tf_ms
        lv.advance(now)
        if tlv is not None:
            tlv.advance(now)
        bar = {"ts": bar_ts, "open": kl.o[i], "high": kl.h[i],
               "low": kl.l[i], "close": kl.c[i]}
        if pos is not None:
            closed = None
            if sscan is not None:
                h = sell_hit(sscan, lv, sell_kinds)
                if h is not None:
                    closed = _close(pos, kl.c[i], bar_ts, "sell_" + h.kind)
                    pos = None
            if closed is None and mgmt == "manage+struct":
                pos, closed = manage(pos, bar, tm)
                if closed is None and cfg.struct_on:
                    line = struct_line(lv.bib.bis, pos["entry_ts"], 1)
                    if line is not None and kl.c[i] < line:
                        closed = _close(pos, kl.c[i], bar_ts, "structure_break")
                        pos = None
            elif closed is None and mgmt == "stop":
                if bar["low"] <= pos["stop"]:
                    closed = _close(pos, pos["stop"], bar_ts, "stop_loss")
                    pos = None
            if closed is not None:
                trades.append(closed)
            continue
        pts = lv.scan_points()
        sig = strat.pick(ls, pts)
        if sig is None:
            continue
        dg.n_signals += 1
        pos = open_pos(sig, _regime_at(market, now), bar_ts, kl.c[i])
    if pos is not None:
        last = kl.n - 1
        trades.append(_close(pos, kl.c[last], int(kl.ts[last]), "end_of_data"))
    dg.n_seen = dict(strat.n_seen)
    dg.n_blocked = dict(strat.n_blocked)
    dg.n_reject = int(strat.n_reject)
    dg.n_ma_block = int(strat.n_ma_block)
    sn = lv.snapshot()
    dg.n_bi, dg.n_seg, dg.n_zs = sn.n_bi, sn.n_seg, sn.n_zs
    dg.trend_at_end = trend_lv.state.label()
    return trades, dg


def _mp_run(task):
    sym = task[0]
    tr, dg = run_symbol(*task)
    return sym, tr, dg


def run_variant(syms, v, tf, market, kinds, gate, t0, t1, jobs, verbose=True):
    tasks = [(s, v, tf, market, kinds, gate, t0, t1) for s in syms]
    tr_all: list[Trade] = []
    dg_all: list[Diag] = []
    t0t = time.time()
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(int(jobs)) as pool:
            for k, (sym, tr, dg) in enumerate(pool.imap_unordered(_mp_run, tasks,
                                                                  chunksize=1)):
                tr_all.extend(tr)
                dg_all.append(dg)
                if verbose and (k + 1) % 40 == 0:
                    print(f"    {v['name']} {k + 1}/{len(syms)} 笔={len(tr_all)} "
                          f"{time.time() - t0t:.0f}s", file=sys.stderr, flush=True)
    else:
        for k, t in enumerate(tasks):
            sym, tr, dg = _mp_run(t)
            tr_all.extend(tr)
            dg_all.append(dg)
    dg_all.sort(key=lambda d: d.sym)
    return tr_all, dg_all


# ------------------------------------------------------------------ 统计
def per_day(s: dict) -> float:
    """每持仓日收益(线性) = 单笔毛期望 / 平均持仓天数。

    跨离场口径唯一可比的量: 离场一变持仓时长就变, 单笔期望本身不可比。
    """
    d = s["hold_min"] / 1440.0
    return (s["exp_pct"] / d) if d > 0 else 0.0


def trimmed(s: dict, frac: float = 0.05) -> tuple[float, float]:
    """返回 (剔尾均值, 中位数)。`frac` 为两侧各剔掉的比例。

    ⚠ 这是上一轮翻出的口径问题的直接对策: 均值会被极少数大赢家拉动
    (实测"所有配置的期望只来自 ≤10 笔"), 而中位数在低胜率高盈亏比的系统里
    天然为负、没有区分力 —— 两个一起给才有信息。
    """
    xs = s.get("_pnl") or []
    n = len(xs)
    if not n:
        return 0.0, 0.0
    k = int(n * frac)
    if n - 2 * k >= 5:
        xs = sorted(xs)[k:n - k]
    return sum(xs) / len(xs), statistics.median(xs)


def by_sym_means(trades: list) -> dict:
    """{sym: 该标的的单笔毛期望(均值)} —— 配对检验的原子。"""
    acc: dict[str, list] = {}
    for t in trades:
        acc.setdefault(t.sym, []).append(t.pnl_pct)
    return {s: sum(v) / len(v) for s, v in acc.items()}


def signed_pairs(a: dict, b: dict, min_n: int = 3) -> tuple[int, int, float]:
    """配对比较: 公共标的中 a 的均值 > b 的均值 的比例。

    返回 (可比标的数, a 更好的标的数, 比例)。为什么用配对而不是合并均值:
    合并均值让"某个标的恰好抓到 3 笔 600% 的行情"主导整个结论; 分标的先平均,
    每个标的只贡献一票, 结论稳健得多。
    """
    common = [s for s in a if s in b]
    if not common:
        return 0, 0, 0.0
    better = sum(1 for s in common if a[s] > b[s])
    return len(common), better, better / len(common)


def risk_block(tr: list) -> dict:
    """候选池口径风险画像(等权、按离场时间累加)。**非**真实账户回撤。"""
    srt = sorted(tr, key=lambda t: t.exit_ts)
    eq = peak = mdd = 0.0
    for t in srt:
        eq += t.pnl_pct
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
    return {"mdd": mdd, "final": eq}


def pool_index(syms, tf, market, t_start, t_end, rebal_bars: int = 1,
               min_n: int = 5) -> dict:
    """同池**等权**指数。`rebal_bars=24/小时数` 即"逐日再平衡"(tf 无关的主基准)。"""
    master = load_tf(market, "BTCUSDT", tf)
    if master is None or master.n < MIN_BARS:
        return {}
    grid = [int(t) for t in master.ts
            if (not t_start or t >= t_start) and (not t_end or t <= t_end)]
    if len(grid) < rebal_bars + 2:
        return {}
    pts = grid[::max(1, int(rebal_bars))]
    series = []
    for sym in syms:
        kl = load_tf(market, sym, tf)
        if kl is not None and kl.n >= MIN_BARS:
            series.append(kl)
    eq = 1.0
    n_seg = 0
    for j in range(len(pts) - 1):
        ta_, tb_ = pts[j], pts[j + 1]
        rs = []
        for kl in series:
            i0 = bisect.bisect_right(kl.ts, ta_) - 1
            i1 = bisect.bisect_right(kl.ts, tb_) - 1
            if i0 < 0 or i1 <= i0:
                continue
            if kl.c[i0] > 0:
                rs.append(kl.c[i1] / kl.c[i0] - 1.0)
        if len(rs) < min_n:
            continue
        eq *= 1.0 + sum(rs) / len(rs)
        n_seg += 1
    if n_seg == 0:
        return {}
    days = (pts[-1] - pts[0]) / D.MS_DAY
    pd_ = eq ** (1.0 / days) - 1.0 if days > 0 else 0.0
    return {"total": eq - 1.0, "days": days, "per_day": pd_,
            "ann": (1.0 + pd_) ** 365.25 - 1.0, "n_seg": n_seg,
            "rebal_bars": rebal_bars}


# ------------------------------------------------------------------ 输出
def ms_of(ym: str, end: bool = False) -> int:
    y, m = (int(x) for x in ym.split("-"))
    if end:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)


def stats_row(tr: list, market: str, tf: str, dg: list, label: str) -> dict:
    s = R.summarize(tr, market)
    if not s["n"]:
        return {"label": label, "n": 0}
    s["_pnl"] = [t.pnl_pct for t in tr]
    tm_, med = trimmed(s)
    used = sum(t.hold_min for t in tr)
    avail = sum(d.n_op_bars for d in dg if not d.err) * BAR_MIN[tf]
    share = (used / avail) if avail else 0.0
    pdl = per_day(s)
    # 最高 3 笔占"全部正收益"的比例 —— 上一轮翻出的口径问题的直接读数:
    # 若最赚的 3 笔就贡献了大半的毛收益, 那这条结论对样本外的鲁棒性极低。
    pos_sum = sum(x for x in s["_pnl"] if x > 0)
    top3 = sum(sorted((x for x in s["_pnl"] if x > 0), reverse=True)[:3])
    return {"label": label, "n": s["n"], "win": s["win_rate"], "exp": s["exp_pct"],
            "exp_t": tm_, "med": med, "pf": s["pf"], "rr": s["rr"],
            "net": s["net_exp_pct"], "hold_d": s["hold_min"] / 1440.0,
            "per_day": pdl, "share": share, "fold": pdl * 365 * share,
            "top3": (top3 / pos_sum if pos_sum > 0 else 0.0),
            "n_seen": sum(d.n_seen.get("B1", 0) + d.n_seen.get("B2", 0)
                          + d.n_seen.get("B3", 0) for d in dg),
            "n_blocked": sum(sum((d.n_blocked or {}).values()) for d in dg),
            "n_ma_block": sum(d.n_ma_block for d in dg),
            "n_volcap_block": sum(getattr(d, "n_volcap_block", 0) for d in dg),
            "n_regime_block": sum(getattr(d, "n_regime_block", 0) for d in dg),
            "n_overheat_block": sum(getattr(d, "n_overheat_block", 0) for d in dg),
            "n_resistance_block": sum(getattr(d, "n_resistance_block", 0) for d in dg),
            "n_fib_block": sum(getattr(d, "n_fib_block", 0) for d in dg),
            "n_correl_block": sum(getattr(d, "n_correl_block", 0) for d in dg),
            "n_reject": sum(d.n_reject for d in dg),
            "trades": tr, "by_sym": by_sym_means(tr)}


HDR = (f"{'变体':<26} {'笔数':>6} {'胜率':>6} {'单笔毛':>9} {'单笔净':>9} "
       f"{'剔尾5%':>9} {'中位':>9} {'PF':>5} {'均持仓':>7} {'每持仓日':>10} "
       f"{'在仓':>7} {'折算年化':>10} {'前3笔占比':>9}")


def fmt_row(r: dict) -> str:
    """⚠ `单笔毛` 是**非加权**的 pnl_pct 均值; `单笔净` 是**名义额加权**后再扣双边费。

    两个都必须给: 仓位分层类改造(如 E7)只动名义额, 不动价格收益率 ⇒
    只看 `单笔毛` 会以为"什么都没发生"。净期望 = 加权毛期望 − 双边费(0.2% 现货)。
    """
    if not r.get("n"):
        return f"{r['label']:<26} {0:>6}   (无成交)"
    return (f"{r['label']:<26} {r['n']:>6} {r['win'] * 100:>5.1f}% "
            f"{r['exp'] * 100:>+8.3f}% {r['net'] * 100:>+8.3f}% "
            f"{r['exp_t'] * 100:>+8.3f}% "
            f"{r['med'] * 100:>+8.3f}% {r['pf']:>5.2f} {r['hold_d']:>6.2f}天 "
            f"{r['per_day'] * 100:>+9.3f}% {r['share'] * 100:>6.2f}% "
            f"{r['fold'] * 100:>+9.1f}% {r['top3'] * 100:>8.0f}%")


# ------------------------------------------------------------------ 轮次定义
def _mk(name, tf, kinds, exit_, **kw) -> dict:
    d = {"name": name, "tf": tf, "kinds": kinds, "exit": exit_}
    d.update(kw)
    return d


def round_r1(tfs=("1h", "2h", "4h")):
    """第一轮 · 基准盘: tf × 入场构造(kinds) × 离场口径。**不加任何新开关。**"""
    out = []
    for tf in tfs:
        for kd in ("b1", "b1,b2"):
            for ex in ("old", "stop", "s1", "s1_trail"):
                out.append(_mk(f"R1 {tf} {kd} {ex}", tf, kd, ex))
    return out


def round_r2(tfs=("1h", "2h")):
    """第二轮 · 逐项改造 A/B: 每个开关单独打开, 与 R1 同格对照。

    对照格 = 同 tf、同 kinds、同离场口径的 R1 变体(基准: `s1` 离场 + `b1,b2`)。
    """
    out = []
    for tf in tfs:
        for nm, kw in (
            ("E2 b1dir_no_up", dict(b1_dir="no_up")),
            ("E2 b1dir_down", dict(b1_dir="down")),
            ("E3 pending_out", dict(pending_out=True)),
            ("E4 path_zs_only", dict(b1_path="zs_only")),
            ("E4 path_trend_only", dict(b1_path="trend_only",
                                        b1_trend_adjacent=True)),
            ("E4 path_macd_only", dict(b1_path="macd_only")),
            ("E4 trend_adjacent", dict(b1_trend_adjacent=True)),
            ("E5 vol_gate", dict(vol_gate=1.0)),
            ("E6 ma_veto(5/10)", dict(ma_mode="ma", ma_fast=5, ma_slow=10)),
            ("E6 ma_veto(13/21)", dict(ma_mode="ma", ma_fast=13, ma_slow=21)),
            ("E6 ma_close", dict(ma_mode="close", ma_fast=5, ma_slow=10)),
            ("E6 ma_size", dict(ma_mode="size", ma_fast=5, ma_slow=10)),
            ("E7 pos_amp", dict(pos_layer="amp")),
            ("E7 pos_ratio", dict(pos_layer="ratio")),
            ("E7 pos_kind", dict(pos_layer="kind")),
            ("E8 zs_merge", dict(zs_merge=True)),
        ):
            out.append(_mk(f"R2 {tf} {nm}", tf, "b1,b2", "s1", **kw))
    return out


def round_r3(tfs=("1h", "2h", "4h")):
    """第三轮 · 构件组合: 离场口径 × 入场构造 × 开关叠加。"""
    out = []
    for tf in tfs:
        out += [
            _mk(f"R3 {tf} b1b2b3 s1", tf, "b1,b2,b3", "s1"),
            _mk(f"R3 {tf} b1b2b3 s123", tf, "b1,b2,b3", "s123"),
            _mk(f"R3 {tf} b1 s1", tf, "b1", "s1"),
            _mk(f"R3 {tf} b1,b2 s123", tf, "b1,b2", "s123"),
            _mk(f"R3 {tf} b1,b2 s123_trail", tf, "b1,b2", "s123_trail"),
            _mk(f"R3 {tf} b1dir+pos", tf, "b1,b2", "s1",
                b1_dir="no_up", pos_layer="amp"),
        ]
    return out


ROUNDS = {"r1": round_r1, "r2": round_r2, "r3": round_r3}


# ------------------------------------------------------------------ 主流程
def _flush(path: str, lines: list) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as e:                                        # noqa: BLE001
        print(f"写盘失败 {path}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=["spot"])
    ap.add_argument("--round", default="r1", choices=sorted(ROUNDS) + ["all"])
    ap.add_argument("--tf", default="", help="只跑这些 tf(逗号分隔); 空=轮次默认")
    ap.add_argument("--gate", default="on", choices=["on", "off"])
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/eng.txt")
    ap.add_argument("--only", default="", help="只跑名字含该子串的变体(调试用)")
    ap.add_argument("--by-year", action="store_true", help="追加按年单笔毛期望矩阵")
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    market = args.market
    gate = args.gate == "on"
    t0, t1 = ms_of(args.start), ms_of(args.end, end=True)
    syms = D.symbols(market)
    if args.limit:
        syms = syms[:args.limit]
    tfs = tuple(x.strip() for x in args.tf.split(",") if x.strip()) or None

    rounds = sorted(ROUNDS) if args.round == "all" else [args.round]
    variants: list[dict] = []
    for r in rounds:
        variants.extend(ROUNDS[r](*( (tfs,) if tfs else () )))
    if args.only:
        variants = [v for v in variants if args.only in v["name"]]

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(str(s))
        print(s, flush=True)
        # 增量落盘: 一个变体的 worker 抛异常(踩过: 漏 import 导致 mp.Pool 在
        # 迭代中途抛 NameError)会让整个进程死掉。没有增量落盘, 前面跑完的
        # 十几分钟全部白费。
        if len(lines) % 20 == 0:
            _flush(args.out, lines)

    emit(f"# chanbook 现货主引擎多变体回测   spot   {args.start} ~ {args.end}   "
         f"({len(syms)} 个标的, 门={args.gate})")
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"# 口径: 只做多 · 预热 {WARMUP} 根 · 最少 {MIN_BARS} 根 · "
         f"方向闸门用高一级别 {TREND_TF} · 多进程 {args.jobs}")
    emit("# 改造开关默认全部关闭 ⇒ 未列出的字段一律等于改动前行为。")
    emit("")

    rows: list[dict] = []
    for v in variants:
        kinds = tuple(KIND_MAP[k.strip()] for k in v["kinds"].split(","))
        tf = v["tf"]
        ex = EXITS[v["exit"]]
        emit(f"[{time.strftime('%H:%M:%S')}] {v['name']}  —  {ex['why']}")
        ta = time.time()
        tr, dg = run_variant(syms, v, tf, market, kinds, gate, t0, t1, args.jobs)
        r = stats_row(tr, market, tf, dg, v["name"])
        r.update(tf=tf, kinds=v["kinds"], exit=v["exit"], secs=time.time() - ta)
        rows.append(r)
        emit("  " + fmt_row(r) + f"   [{r['secs']:.0f}s]")
        emit("")

    # ---- 汇总 ----
    emit("=" * 118)
    emit(f"{'轮次汇总':<20} {'tf':>4} {'构造':>8} {'离场':>10} " + HDR)
    for r in rows:
        emit(f"{'':<20} {r['tf']:>4} {r['kinds']:>8} {r['exit']:>10} "
             + fmt_row(r))
    emit("")
    emit("字段说明: 单笔毛 = 非加权毛价格期望 · 单笔净 = 名义额加权毛 − 双边费"
         "(现货 0.20%) · 剔尾5% = 两侧各剔 5% 分位后的均值 · "
         "每持仓日 = 单笔毛/均持仓天数 · 前3笔占比 = 最赚 3 笔占全部正收益的比例")
    emit("          在仓 = Σ持仓分钟 / Σ可用分钟(同标的同时至多一仓) · "
         "折算年化 = 每持仓日 × 365 × 在仓占比(线性, 非复利)")

    # ---- 池基准(逐日再平衡, tf 无关) ----
    emit("")
    emit("=" * 118)
    emit("基准: 同池等权指数(逐日再平衡 —— tf 无关, 跨周期唯一可比的基准)")
    for tf in sorted({r["tf"] for r in rows}, key=lambda x: TF_HOURS[x]):
        bpd = int(round(24.0 / TF_HOURS[tf]))
        pi = pool_index(syms, tf, market, t0, t1, rebal_bars=bpd)
        if pi:
            emit(f"  {tf}: 全程 {pi['total'] * 100:+.1f}%  "
                 f"每日 {pi['per_day'] * 100:+.4f}%  年化 {pi['ann'] * 100:+.1f}%  "
                 f"({pi['n_seg']:,} 段 / {pi['days']:.0f} 天)")
        pit = pool_index(syms, tf, market, t0, t1, rebal_bars=1)
        if pit:
            emit(f"    (上界·逐 {tf} 再平衡 —— 含再平衡溢价, 不可当市场基准: "
                 f"年化 {pit['ann'] * 100:+.1f}%)")

    # ---- 配对对照矩阵(每个 tf 内, 以 s1 离场 + b1,b2 为基准) ----
    emit("")
    emit("=" * 118)
    emit("分标的配对(以同 tf 的 `b1,b2 + s1` 为基准; 只统计两边都有成交的标的)")
    emit(f"{'tf':>4} {'变体':<28} {'可比标的':>8} {'变好的':>7} {'比例':>7} "
         f"{'基准均笔毛':>10} {'本变体均笔毛':>12} {'Δ':>9}")
    for tf in sorted({r["tf"] for r in rows}, key=lambda x: TF_HOURS[x]):
        base = next((r for r in rows
                     if r["label"] == f"R1 {tf} b1,b2 s1" and r.get("n")), None)
        if base is None:
            base = next((r for r in rows if r["tf"] == tf and r.get("n")
                         and r["kinds"] == "b1,b2" and r["exit"] == "s1"
                         and "b1dir" not in r["label"]), None)
        if base is None or not base.get("n"):
            continue
        for r in rows:
            if r["tf"] != tf or not r.get("n") or r is base:
                continue
            n, bet, share = signed_pairs(r["by_sym"], base["by_sym"])
            if not n:
                continue
            emit(f"{tf:>4} {r['label']:<28} {n:>8} {bet:>7} {share * 100:>6.0f}% "
                 f"{base['exp'] * 100:>+9.3f}% {r['exp'] * 100:>+11.3f}% "
                 f"{(r['exp'] - base['exp']) * 100:>+8.3f}%")
    emit("")
    emit("读法: 「变好的比例」≈ 50% ⇒ 该改造对标的层面的期望没有系统性影响; "
         "显著高于/低于 50% 才值得进一步看。")

    # ---- 按年稳定性 ----
    if args.by_year:
        emit("")
        emit("=" * 118)
        emit("按年 · 单笔毛期望(入场年; 空格 = 该年无成交)。用途: 看结论是被少数"
             "年份撑着, 还是逐年都成立。")
        yrs = sorted({datetime.fromtimestamp(t.entry_ts / 1000, timezone.utc).year
                      for r in rows if r.get("n") for t in r["trades"]})
        emit(f"{'变体':<28} {'笔数':>6} " +
             " ".join(f"{y:>7}" for y in yrs))
        for r in rows:
            if not r.get("n"):
                continue
            acc: dict = {}
            for t in r["trades"]:
                y = datetime.fromtimestamp(t.entry_ts / 1000, timezone.utc).year
                acc.setdefault(y, []).append(t.pnl_pct)
            cells = []
            for y in yrs:
                v = acc.get(y)
                cells.append(f"{sum(v) / len(v) * 100:>+6.2f}%" if v
                             else f"{'—':>7}")
            emit(f"{r['label']:<28} {r['n']:>6} " + " ".join(cells))
        emit("")
        emit("⚠ 每格是**该年的单笔毛期望**, 不是该年收益。样本少的年份(如 2018/2019)"
             "只有十几笔, 单笔期望的噪声很大, 不要逐格读, 看的是**符号一致性**。")

    _flush(args.out, lines)
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
