"""1h / 2h 级别 B1 + B2 入场 + 「结构止损 + 一卖 S1」离场 回测。

用户口径:
  * 操作级别 = **1 小时 / 2 小时**(`--tf`);
  * 入场 = **只做 B1(第一类买点/背驰) 与 B2(第二类买点/一买后首次回调不破低)**(`--kinds`);
  * 离场 = **结构止损(灾难止损) + S1(一卖背驰)** —— 即 `--modes chan1_stop`;
  * 只做多, 不做空。

## 为什么要这一轮

2026-09-28 的 4h-B1 实验(`B1_4H_2026-09-28.md`)把结论定位到:
**入场判别不重要(随机入场对照 p≈0.20)、出口规则有价值(单位暴露收益 10.7 倍于同池市场)、
但瓶颈是暴露不足(在仓仅 8.43%)**。本轮就是那个"下一步":保持同一套出口规则不变,
**把入场频率提上去**(级别从 4h 下沉到 1h/2h、买点从只有 B1 扩到 B1+B2),
看每持仓日的优势能不能兑现成资金回报率。

## 与 `backtest_b1_4h.py` 的关系

同一套离场机制与同一套统计口径, 三处扩展:
1. `--tf` 支持 1h / 2h(2h 由 1h 现场聚合); 2. `--kinds` 支持 B1+B2; 3. `--jobs` 多进程(1h 数据量是 4h 的 4 倍)。
既有 `tools/backtest_b1_4h.py` **一行未改**, 4h 报告可原样复现。

## 两个必须显式声明的口径决定

1. **方向闸门用"高一级别"的趋势, 不用本级别的**。`strategy.DIR_GATE` 规定
   B2/B3(顺势)要求大级别不是下跌趋势。这里高一级别固定取 **4h**(`TREND_TF`)——
   与 `LEVEL_TF["spot"] = ("4h","1h","15m")` 的"大级别"一致。
   ⚠ 为什么不能用本级别自己的趋势状态: B2 按定义**必然出现在一段下跌之后**
   (它是一买之后的第一次回调), 本级别趋势此时多数仍是"下跌" ⇒ 闸门会把 B2 近乎全部否决,
   语义上自相矛盾。`--gate off` 可关掉闸门做敏感性对照。
2. **2h 不在归档里**(归档只有 5m/15m/1h/4h/1d), 由 1h 现场聚合;
   `2h` 只在本进程内补进 `D.TF_MS`, **不改 `chanbook/data.py`**。
   对齐正确: `ts // 7200000` 落在 00/02/04/…/22 UTC, 与币安一致(4h 同理;
   ⚠ 3d/1w 会错位到"周四起算", 本工具不用)。

## 无未来函数

每根 bar 先 `advance(bar_ts + tf_ms)` 把"本根已收盘"的事实喂进去, 再按该 bar 的**收盘价**成交;
买点用已冻结的笔、卖点用已冻结的向上笔。与 4h 工具同款。

## 随机入场对照(关键纪律)

正结果必须证明"不是随便买也赚"。对照设置: 每 tick 以该标的**实测入场频率**随机决定入场,
其余(至多一仓 / 结构止损 / S1 卖点 / `risk.plan` 拒单)**完全一致**;
结构止损锚点取"最近一根冻结向下笔低点", 正是 B1/B2 用的那个基准(`Point.price`)。
**入场类别按真实成交的 B1:B2 比例抽样**, 保证风险几何(一买走 `b1_wide` 宽止损, 二买不走)同分布。

运行:
  python tools/backtest_b12_tf.py --market spot --tf 1h --kinds b1,b2 --jobs 10 \\
      --out out/b12_1h_spot.txt
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys
import time
import zlib
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook import report as R                                  # noqa: E402
from chanbook import risk                                         # noqa: E402
from chanbook.engine import (Diag, Trade, _close, _regime_at,     # noqa: E402
                             manage, open_pos, struct_line)
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1, B2, B3                            # noqa: E402
from chanbook.sell import S1, S2, S3, SellScanner                 # noqa: E402
from chanbook.strategy import DIR_GATE, PRIORITY, BookCfg, Signal  # noqa: E402
from chanbook.trend import PENDING                                # noqa: E402

# 2h 不在 data.TF_MS 里 —— 只在本进程内补齐(与 probe_levels.py 同款处理)。
D.TF_MS.setdefault("2h", 2 * D.MS_HOUR)

WARMUP = 300                       # 与引擎一致
MIN_BARS = 500                     # 与 level.MIN_KLINES["op"] 一致
TREND_TF = "4h"                    # 方向闸门用的高一级别
TREND_MIN_BARS = 200               # 与 level.MIN_KLINES["trend"] 一致

TF_HOURS = {"1h": 1, "2h": 2, "4h": 4, "15m": 0.25, "5m": 1 / 12}
BAR_MIN = {"1h": 60, "2h": 120, "4h": 240}

KIND_MAP = {"b1": B1, "b2": B2, "b3": B3}

EXIT_MODES = {
    "chan1_stop": "**仅一卖 S1 + 结构止损**(本轮用户指定的出口规则)",
    "old": "旧离场(追踪3.0×risk + RR2目标 + 结构破位) —— 同一批信号的对照",
    "chan123_stop": "S1/S2/S3 + 结构止损(对照: 多开 S2/S3 是否更好)",
    "chan1": "仅一卖 S1, 不设止损(对照: 止损的代价)",
}
SELL_KINDS = {
    "chan1_stop": (S1,),
    "old": (),
    "chan123_stop": (S1, S2, S3),
    "chan1": (S1,),
}
PRIO = {S1: 0, S2: 1, S3: 2}


# ------------------------------------------------------------------ 数据
def load_tf(market: str, sym: str, tf: str):
    """统一走 1h 底档; tf='1h' 直接用, 其余现场聚合(口径同源, 见模块 docstring)。"""
    base = D.load(market, sym, "1h")
    if base is None or base.n == 0:
        return None
    if tf == "1h":
        return base
    return D.resample(base, D.TF_MS[tf])


def scanner_kw(market: str, kinds: tuple) -> dict:
    return BookCfg.legacy(market=market, kinds=tuple(kinds)).scanner_kw()


def _sell_kw(mode: str) -> dict:
    """卖点参数 = 买点参数的镜像默认值(不调参, 与买点共用同一套阈值)。"""
    return {"pull_bi": 1, "beichi_tol": 0.02, "beichi_strict": False,
            "s2_max_bi": 20, "s1_max_lag": 12, "s1_zs_window": 1,
            "s3_zs_window": 2, "s1_allow_no_zs": True,
            "enable": SELL_KINDS[mode] or (S1, S2, S3)}


def _sell_hit(sscan, lv, kinds):
    pts = sscan.scan(lv.bib.bis, lv.zb.all_zss(), lv.macd, lv.bib.n_final)
    if not pts:
        return None
    pts = [p for p in pts if p.kind in kinds]
    if not pts:
        return None
    pts.sort(key=lambda p: PRIO.get(p.kind, 9))
    return pts[0]


# ------------------------------------------------------------------ 单标的
def run_symbol(sym: str, mode: str, tf: str, market: str, kinds: tuple,
               gate: bool, t_start: int = 0, t_end: int = 0):
    kl = load_tf(market, sym, tf)
    dg = Diag(sym=sym)
    if kl is None:
        dg.err = "无 1h 底档"
        return [], dg
    if kl.n < MIN_BARS:
        dg.err = f"{tf} 根数不足 {kl.n}"
        return [], dg
    lv = Level("op", tf, "bi", scanner_kw=scanner_kw(market, kinds)).attach(kl)
    tlv = None
    if gate:
        tkl = load_tf(market, sym, TREND_TF)
        if tkl is None or tkl.n < TREND_MIN_BARS:
            dg.err = f"趋势级别({TREND_TF})样本不足"
            return [], dg
        tlv = Level("trend", TREND_TF, "seg").attach(tkl)

    sk = SELL_KINDS[mode]
    sscan = SellScanner(**_sell_kw(mode)) if sk else None
    tm = risk.trail_mult_of(market)
    use_stop = mode.endswith("_stop")
    order = {k: n for n, k in enumerate(PRIORITY)}
    trades: list[Trade] = []
    pos = None
    tk = PENDING
    start = WARMUP
    if t_start:
        start = max(start, bisect.bisect_right(kl.ts, t_start - 1))
    dg.n_op_bars = max(0, kl.n - start)
    for i in range(start, kl.n):
        bar_ts = int(kl.ts[i])
        if t_end and bar_ts > t_end:
            break
        # ⚠ **唯一时钟**: 「本根(操作级别)K线已收盘」的那一刻。两级必须共用它。
        # `Level.advance(close_ts)` 喂的是「收盘 ≤ close_ts」的根, 所以给趋势级别
        # 传 `bar_ts + tlv.tf_ms` 等于宣称「开盘于 bar_ts 的那根 4h 已经收盘」——
        # 那根要到 `bar_ts + 4h` 才收盘 ⇒ 偷看最多 1 整根 4h(实测领先 3 小时)。
        # 2026-09-29 由 `tests/check_timebase.py` 静态扫出并改正。
        now = bar_ts + lv.tf_ms
        lv.advance(now)
        if tlv is not None:
            tlv.advance(now)
        bar = {"ts": bar_ts, "open": kl.o[i], "high": kl.h[i],
               "low": kl.l[i], "close": kl.c[i]}
        if pos is not None:
            closed = None
            if mode == "old":
                pos, closed = manage(pos, bar, tm)
                if closed is None:
                    line = struct_line(lv.bib.bis, pos["entry_ts"], 1)
                    if line is not None and kl.c[i] < line:
                        closed = _close(pos, kl.c[i], bar_ts, "structure_break")
                        pos = None
            else:
                h = _sell_hit(sscan, lv, sk)
                if h is not None:
                    closed = _close(pos, kl.c[i], bar_ts, "sell_" + h.kind)
                    pos = None
                elif use_stop and bar["low"] <= pos["stop"]:
                    closed = _close(pos, pos["stop"], bar_ts, "stop_loss")
                    pos = None
            if closed is not None:
                trades.append(closed)
            continue
        pts = lv.scan_points()                        # enable=(kinds,) ⇒ 只返回这些买点
        if not pts:
            continue
        # 同一 bar 命中多个买点时按项目优先级取(B2 > B3 > B1, 见 strategy.PRIORITY);
        # 被闸门/风控否决的候选取下一个 —— 与 Strategy.pick 同义。
        pts = sorted(pts, key=lambda p: order.get(p.kind, 99))
        tk = tlv.state.kind if tlv is not None else PENDING
        entry = float(kl.c[i])
        a = lv.atr()
        for p in pts:
            dg.n_seen[p.kind] = dg.n_seen.get(p.kind, 0) + 1
            allow = DIR_GATE.get(p.kind)
            if allow is not None and tk not in allow:
                dg.n_blocked[p.kind] = dg.n_blocked.get(p.kind, 0) + 1
                continue
            pl = risk.plan(p.side, entry, float(p.price), a, market, p.kind)
            if not pl.ok:
                dg.n_reject += 1
                continue
            sig = Signal(
                sym=sym, side=p.side, kind=p.kind, ts=bar_ts, i=i, entry=entry,
                stop=pl.stop, target=pl.target, atr=pl.atr, risk_pct=pl.risk_pct,
                notional=pl.notional, trend=tk, counter=bool(p.kind == B1),
                zs_i=int(p.zs_i), beichi=(p.beichi.short() if p.beichi else ""),
                reasons=list(p.reasons), feats=dict(p.feats),
            )
            dg.n_signals += 1
            pos = open_pos(sig, _regime_at(market, bar_ts + lv.tf_ms), bar_ts, entry)
            break
    if pos is not None:
        last = kl.n - 1
        trades.append(_close(pos, kl.c[last], int(kl.ts[last]), "end_of_data"))
    sn = lv.snapshot()
    dg.n_bi, dg.n_seg, dg.n_zs = sn.n_bi, sn.n_seg, sn.n_zs
    dg.trend_at_end = tk
    return trades, dg


def _mp_run(task):
    tr, dg = run_symbol(*task)
    return task[0], tr, dg


def run_all(syms, mode, tf, market, kinds, gate, t0, t1, jobs, label="") -> tuple:
    tasks = [(s, mode, tf, market, kinds, gate, t0, t1) for s in syms]
    tr_all: list[Trade] = []
    dg_all: list[Diag] = []
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(int(jobs)) as pool:
            for k, (sym, tr, dg) in enumerate(pool.imap_unordered(_mp_run, tasks,
                                                                  chunksize=1)):
                tr_all.extend(tr)
                dg_all.append(dg)
                if (k + 1) % 30 == 0:
                    print(f"  {label}{k + 1}/{len(syms)} 笔={len(tr_all)}",
                          file=sys.stderr, flush=True)
    else:
        for k, t in enumerate(tasks):
            sym, tr, dg = _mp_run(t)
            tr_all.extend(tr)
            dg_all.append(dg)
            if (k + 1) % 30 == 0:
                print(f"  {label}{k + 1}/{len(syms)} 笔={len(tr_all)}",
                      file=sys.stderr, flush=True)
    dg_all.sort(key=lambda d: d.sym)
    return tr_all, dg_all


# ------------------------------------------------------------------ 统计口径
def per_day(s: dict) -> float:
    """每持仓日收益(线性, 非复利) = 单笔毛期望 / 平均持仓天数。

    跨变体的唯一可比量: 离场口径一变, 持仓时长就变, 单笔期望本身不可比。
    """
    d = s["hold_min"] / 1440.0
    return (s["exp_pct"] / d) if d > 0 else 0.0


def risk_block(tr: list) -> dict:
    """候选池口径风险画像(等权、按离场时间累加)。**非**真实账户回撤。"""
    srt = sorted(tr, key=lambda t: t.exit_ts)
    eq = peak = mdd = 0.0
    cur_lose = max_lose = 0
    rets = []
    for t in srt:
        eq += t.pnl_pct
        rets.append(t.pnl_pct)
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
        if t.pnl_pct <= 0:
            cur_lose += 1
            max_lose = max(max_lose, cur_lose)
        else:
            cur_lose = 0
    return {"best": max(rets) if rets else 0.0,
            "worst": min(rets) if rets else 0.0,
            "mdd": mdd, "max_lose": max_lose, "final": eq}


def _bar_rets(sym: str, tf: str, market: str, t_start: int, t_end: int) -> dict:
    """某标的的逐 bar 收益 {ts: ret}(诊断用)。"""
    kl = load_tf(market, sym, tf)
    out = {}
    if kl is None or kl.n < MIN_BARS:
        return out
    for i in range(1, kl.n):
        ts = int(kl.ts[i])
        if t_start and ts < t_start:
            continue
        if t_end and ts > t_end:
            break
        c0, c1 = kl.c[i - 1], kl.c[i]
        if c0 > 0:
            out[ts] = c1 / c0 - 1.0
    return out


def pool_index(syms, tf, market, t_start, t_end, rebal_bars: int = 1,
               min_n: int = 5) -> dict:
    """同池**等权**指数: 每 `rebal_bars` 根 bar 把权重拉回等权, 期间持有。

    为什么不用 BTC: BTC 是池内最强资产(9 年 +1684%), 而池内**中位**标的亏钱。
    拿 BTC 当基准会把"跑输"说轻。等权池指数才对应"在这池子里随机持有一篮子"。

    ⚠ **再平衡频率本身会产生收益**(高频再平衡在波动+均值回复的币池里 =
    波动率收割), 所以"逐 tf-bar 再平衡"的指数收益**不是** tf 无关的市场基准:
    同一个池子 4h 再平衡年化 8.9%, 1h 再平衡能到 68%(纯再平衡溢价放大)。
    因此本函数默认另出一档 **`rebal_bars = 每日`** 的口径 —— 它与 tf 无关,
    是跨 tf 唯一可比的基准。逐 tf-bar 那档作为"上界(含再平衡溢价)"单独列出。

    复利按 `index *= 1 + mean_sym(close_{t+k}/close_t - 1)`(等权、区间内持有),
    而不是把逐 bar 均值幂次化 —— 后者等于"每根 bar 都再平衡", 溢价更大。
    """
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
            c0 = kl.c[i0]
            if c0 > 0:
                rs.append(kl.c[i1] / c0 - 1.0)
        if len(rs) < min_n:
            continue
        eq *= 1.0 + sum(rs) / len(rs)
        n_seg += 1
    if n_seg == 0:
        return {}
    days = (pts[-1] - pts[0]) / D.MS_DAY
    per_day = eq ** (1.0 / days) - 1.0 if days > 0 else 0.0
    bpd = 24.0 / TF_HOURS[tf]
    return {"total": eq - 1.0, "n_seg": n_seg, "days": days, "per_day": per_day,
            "ann": (1.0 + per_day) ** 365.25 - 1.0,
            "per_bar": (1.0 + per_day) ** (1.0 / bpd) - 1.0,
            "rebal_bars": rebal_bars}


def benchmark(syms, tf, market, t_start, t_end) -> list[dict]:
    out = []
    for sym in syms:
        kl = load_tf(market, sym, tf)
        if kl is None or kl.n < MIN_BARS:
            continue
        a = WARMUP
        if t_start:
            a = max(a, bisect.bisect_right(kl.ts, t_start - 1))
        b = kl.n - 1
        if t_end:
            b = bisect.bisect_right(kl.ts, t_end) - 1
        if b <= a or not kl.c[a]:
            continue
        days = (int(kl.ts[b]) - int(kl.ts[a])) / D.MS_DAY
        tot = kl.c[b] / kl.c[a] - 1.0
        out.append({"sym": sym, "days": days, "total": tot,
                    "ann": ((1 + tot) ** (365.0 / days) - 1) if days > 0 else 0.0})
    return out


# ------------------------------------------------------------------ 随机入场对照
def level_schedule(sym: str, exit_mode: str, tf: str, market: str,
                   t_start: int = 0, t_end: int = 0):
    """**一次**遍历级别, 把随机对照需要的逐 bar 信息全部录下来。

    这样 K 个种子 × 若干轮频率校准都只是纯 Python 循环, 不必反复推进级别
    (推进级别是本工具里唯一昂贵的操作)。

    卖点日程与持仓无关(只依赖级别结构) ⇒ 录一次全种子共享。
    结构止损位取"最近一根冻结向下笔的低点" —— 正是 B1/B2 用的那个基准
    (`Point.price` 就是触发它的那根向下笔低点 / 二买的回调低点)。
    """
    kl = load_tf(market, sym, tf)
    if kl is None or kl.n < MIN_BARS:
        return None
    lv = Level("op", tf, "bi",
               scanner_kw=scanner_kw(market, (B1, B2))).attach(kl)
    sscan = SellScanner(**_sell_kw(exit_mode))
    ks = SELL_KINDS[exit_mode]
    start = WARMUP
    if t_start:
        start = max(start, bisect.bisect_right(kl.ts, t_start - 1))
    rows: list[tuple] = []
    for i in range(start, kl.n):
        bar_ts = int(kl.ts[i])
        if t_end and bar_ts > t_end:
            break
        lv.advance(bar_ts + lv.tf_ms)
        hit = _sell_hit(sscan, lv, ks)
        bis = lv.bib.bis
        struct = None
        for k in range(lv.bib.n_final - 1, -1, -1):   # 笔交替 ⇒ 至多两步
            if bis[k].direction < 0:
                struct = float(bis[k].low)
                break
        rows.append((bar_ts, float(kl.c[i]), float(kl.l[i]), struct, lv.atr(),
                     (hit.kind if hit is not None else "")))
    return rows


def seeds_pass(rows: list, seed0: int, rate: float, seeds: int, kind_mix: list,
               market: str = "spot", n_iter: int = 4, target: int = 0) -> list[list]:
    """给定"每 bar 入场概率"跑 K 个随机种子, 并按目标笔数校准频率。

    为什么要校准: 若把 rate 直接取成"真实入场的**已成交**频率", 随机侧还要再被
    `risk.plan` 拒单 + 占仓阻塞吃掉一部分 ⇒ 笔数对不齐, 比较失真。
    这里迭代调整 rate 使**中位笔数**逼近真实值(最多 n_iter 轮, 容差 10%)。

    入场类别按真实成交的 B1:B2 比例抽样 —— 一买走 `b1_wide`(宽止损 + 名义额减半),
    二买不走, 不按比例抽会让随机侧的风险几何与真实侧不同分布。
    """
    reg = ""
    pool = kind_mix or [B1]
    out: list[list] = []
    for _ in range(max(1, n_iter)):
        rngs = [random.Random(seed0 * 1000003 + s * 2654435761) for s in range(seeds)]
        positions: list = [None] * seeds
        out = [[] for _ in range(seeds)]
        for (ts, c, lo, struct, atr, sell_kind) in rows:
            for s in range(seeds):
                pos = positions[s]
                if pos is not None:
                    if sell_kind:
                        out[s].append(_close(pos, c, ts, "sell_" + sell_kind))
                        positions[s] = None
                    elif lo <= pos["stop"]:
                        out[s].append(_close(pos, pos["stop"], ts, "stop_loss"))
                        positions[s] = None
                    continue
                if rngs[s].random() >= rate:
                    continue
                kd = pool[int(rngs[s].random() * len(pool)) % len(pool)]
                pl = risk.plan("long", c, struct, atr, market, kd)
                if not pl.ok:
                    continue
                sig = SimpleNamespace(sym="", side="long", kind=kd, stop=pl.stop,
                                      target=pl.target, trend="", counter=int(kd == B1),
                                      confirm="", risk_pct=pl.risk_pct,
                                      notional=pl.notional, feats={})
                positions[s] = open_pos(sig, reg, ts, c)
        if not target:
            break
        got = sorted(len(o) for o in out)[seeds // 2]
        if got <= 0 or abs(got - target) / target <= 0.10:
            break
        rate = min(0.9, rate * target / got)
    return out


# ------------------------------------------------------------------ 主流程
def ms_of(ym: str, end: bool = False) -> int:
    y, m = (int(x) for x in ym.split("-"))
    if end:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)


def _flush(path: str, lines: list) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as e:                                        # noqa: BLE001
        print(f"写盘失败 {path}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=["spot", "um"])
    ap.add_argument("--tf", default="1h", choices=["1h", "2h", "4h"])
    ap.add_argument("--kinds", default="b1,b2")
    ap.add_argument("--modes", default="chan1_stop")
    ap.add_argument("--gate", default="on", choices=["on", "off"])
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", default="chan1_stop",
                    help="随机入场对照挂在哪个离场口径上('' = 关闭)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="out/b12_1h_spot.txt")
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    market, tf = args.market, args.tf
    gate = args.gate == "on"
    kinds = tuple(KIND_MAP[k.strip().lower()] for k in args.kinds.split(",")
                  if k.strip())
    if not kinds:
        print("--kinds 为空")
        return 2
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in EXIT_MODES]
    if bad:
        print(f"未知离场口径 {bad}; 可用 {list(EXIT_MODES)}")
        return 2
    t0, t1 = ms_of(args.start), ms_of(args.end, end=True)
    syms = D.symbols(market)
    if args.limit:
        syms = syms[:args.limit]

    thr = 0.0012 if market == "um" else 0.0020
    bmin = BAR_MIN.get(tf, 60)
    out_lines: list[str] = []

    def emit(s: str = "") -> None:
        out_lines.append(str(s))
        print(s, flush=True)

    hdr = (f"# chanbook {tf} 级 {'+'.join(kinds)} 入场 · "
           f"{'+'.join(modes)} 离场回测   {market}   {args.start} ~ {args.end}   "
           f"({len(syms)} 个标的)")
    emit(hdr)
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"# 口径: 操作级别={tf} · 入场=只做 {'/'.join(kinds)} · 方向闸门="
         f"{'开(高一级别 ' + TREND_TF + ')' if gate else '关'} · 只做多 · "
         f"{tf} 由 1h 现场聚合 · 预热 {WARMUP} 根 · 最少 {MIN_BARS} 根 · "
         f"多进程 {args.jobs}")
    emit("")

    res: dict[str, list] = {}
    diags: dict[str, list] = {}
    for mode in modes:
        emit(f"[{time.strftime('%H:%M:%S')}] 开始 {mode}: {EXIT_MODES[mode]} ...")
        ta = time.time()
        tr_all, dg_all = run_all(syms, mode, tf, market, kinds, gate, t0, t1,
                                 args.jobs, label=f"{mode} ")
        res[mode] = tr_all
        diags[mode] = dg_all
        emit(R.full_report(market, f"{mode} — {EXIT_MODES[mode]}", tr_all, dg_all))
        emit(f"# {mode} 耗时 {time.time() - ta:.0f}s")
        emit("")
        if args.cache_dir:
            try:
                R.save_trades(os.path.join(args.cache_dir,
                                           f"{market}_{tf}_{mode}.json"),
                              tr_all, dg_all)
            except Exception as e:                                # noqa: BLE001
                print(f"缓存失败: {e}")

    # ---- 汇总 ----
    emit("=" * 96)
    emit("汇总(候选池口径; 单笔期望 = 笔数加权毛期望; 「每持仓日」= 单笔毛期望/均持仓天数)")
    emit(f"{'离场口径':>13} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} {'单笔净':>9} "
         f"{'盈亏比':>7} {'PF':>6} {'均持仓':>9} {'每持仓日':>10} {'仅在仓年化':>10}")
    for mode in modes:
        s = R.summarize(res[mode], market)
        if not s["n"]:
            emit(f"{mode:>13} {0:>6}   (无成交)")
            continue
        pd_ = per_day(s)
        emit(f"{mode:>13} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
             f"{s['exp_pct'] * 100:>+8.3f}% {s['net_exp_pct'] * 100:>+8.3f}% "
             f"{s['rr']:>7.2f} {s['pf']:>6.2f} "
             f"{s['hold_min'] / 1440:>7.2f}天 {pd_ * 100:>+9.3f}% "
             f"{pd_ * 365 * 100:>+9.1f}%")
    emit("")
    emit(f"门槛: 单笔**毛**期望需 > {thr * 100:.2f}%"
         f"({'合约双边 0.10%' if market == 'um' else '现货双边 0.20%'} + 滑点余量); "
         f"净期望 = 名义额加权毛期望 − 双边费")
    emit("⚠ 「每持仓日」是跨变体唯一可与买入持有「每日收益」直接对照的量; "
         "单笔毛期望高 ≠ 更优 —— 拿得久自然高。")
    emit("")

    # ---- 分买点(本轮重点: B1 vs B2 谁在赚钱) ----
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        emit(f"【{mode}】分买点(入场类别)")
        emit(f"{'类别':>8} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} {'PF':>6} "
             f"{'均持仓':>9} {'每持仓日':>10}")
        for kd in kinds:
            sub = [t for t in tr if t.kind == kd]
            s = R.summarize(sub, market)
            if not s["n"]:
                emit(f"{kd:>8} {0:>6}   (无成交)")
                continue
            emit(f"{kd:>8} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
                 f"{s['exp_pct'] * 100:>+8.3f}% {s['pf']:>6.2f} "
                 f"{s['hold_min'] / 1440:>7.2f}天 {per_day(s) * 100:>+9.3f}%")
        emit("")

    # ---- 敏感性: 剔除数据末尾强平 ----
    emit("敏感性: 剔除 end_of_data(数据末尾强平)后的单笔毛期望与中位")
    emit(f"{'离场口径':>13} {'笔数':>6} {'单笔毛':>9} {'中位':>9} {'被剔':>6} "
         f"{'被剔那笔':>16}")
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        eod = [t for t in tr if t.reason == "end_of_data"]
        keep = [t for t in tr if t.reason != "end_of_data"]
        if not keep:
            emit(f"{mode:>13} {0:>6}   (全被剔除)")
            continue
        exp = sum(t.pnl_pct for t in keep) / len(keep)
        srt = sorted(t.pnl_pct for t in keep)
        med = srt[len(srt) // 2]
        eod_s = "、".join(f"{t.pnl_pct * 100:+.1f}%" for t in eod) or "—"
        emit(f"{mode:>13} {len(keep):>6} {exp * 100:>+8.3f}% {med * 100:>+8.3f}% "
             f"{len(eod):>6} {eod_s:>16}")
    emit("")

    # ---- 结构诊断 ----
    emit("结构诊断(全池):")
    d0 = diags[modes[0]]
    ok = [d for d in d0 if not d.err]
    emit(f"  * 可用标的 {len(ok)}/{len(d0)}"
         + (f"; 跳过 {len(d0) - len(ok)} 个(样本不足)" if len(ok) != len(d0) else ""))
    emit(f"  * {tf} 笔合计 {sum(d.n_bi for d in ok):,}  "
         f"中枢合计 {sum(d.n_zs for d in ok):,}")
    seen: dict = {}
    blocked: dict = {}
    for d in ok:
        for k, v in (d.n_seen or {}).items():
            seen[k] = seen.get(k, 0) + v
        for k, v in (d.n_blocked or {}).items():
            blocked[k] = blocked.get(k, 0) + v
    emit(f"  * 结构命中 {seen}  闸门拦掉 {blocked}")
    emit(f"  * 入场成交(过 risk.plan) {sum(d.n_signals for d in ok):,}  "
         f"被风控拒单 {sum(d.n_reject for d in ok):,}")
    emit("")

    # ---- 离场归因 + 按年 ----
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        emit(f"【{mode}】离场原因分布")
        for ln in R.by_reason(tr, market):
            emit("  " + ln)
        emit("")
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        emit(f"【{mode}】按年")
        for ln in R.by_year(tr, market):
            emit("  " + ln)
        emit("")

    # ---- 风险画像 ----
    emit("风险画像(每笔等权 1 单位、按离场时间累加; **非**真实账户回撤, 仅候选池口径)")
    emit(f"{'离场口径':>13} {'最好一笔':>10} {'最差一笔':>10} {'累加最大回撤':>14} "
         f"{'最长连亏':>9} {'累加总和':>10}")
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        rk = risk_block(tr)
        emit(f"{mode:>13} {rk['best'] * 100:>+9.2f}% {rk['worst'] * 100:>+9.2f}% "
             f"{rk['mdd'] * 100:>+13.2f}% {rk['max_lose']:>7}笔 "
             f"{rk['final']:>+9.1f}")
    emit("")

    # ---- 仓位占用 ----
    emit(f"仓位占用(「同标的同时至多一仓」下的实际在仓时间占比; {tf} 一根 = {bmin} 分钟)")
    emit(f"{'离场口径':>13} {'在仓分钟':>13} {'可用分钟':>14} {'在仓占比':>9} "
         f"{'折算全年暴露':>13} {'折算年化(线性)':>15}")
    for mode in modes:
        used = sum(t.hold_min for t in res[mode])
        avail = sum(d.n_op_bars for d in diags[mode] if not d.err) * bmin
        share = (used / avail) if avail else 0.0
        s = R.summarize(res[mode], market)
        fold = per_day(s) * 365 * share if s["n"] else 0.0
        emit(f"{mode:>13} {used:>13,} {avail:>14,} {share * 100:>8.2f}% "
             f"{share * 365:>11.1f}天 {fold * 100:>+14.1f}%")
    emit("")

    # ---- 基准 ----
    bm = benchmark(syms, tf, market, t0, t1)
    if bm:
        bm.sort(key=lambda x: x["total"])
        n = len(bm)
        med = bm[n // 2]
        mean_tot = sum(x["total"] for x in bm) / n
        mean_ann = sum(x["ann"] for x in bm) / n
        pd_all = sorted(x["total"] / x["days"] for x in bm if x["days"] > 0)
        med_pd = pd_all[len(pd_all) // 2] if pd_all else 0.0
        mean_pd = sum(pd_all) / len(pd_all) if pd_all else 0.0
        btc = next((x for x in bm if x["sym"] == "BTCUSDT"), None)
        emit("=" * 96)
        emit("基准(同池、同区间、同预热)")
        emit("① 同池等权指数 ← 公平基准(对应「在这池子里随机持有一篮子」)")
        bpd = int(round(24.0 / TF_HOURS[tf]))
        pi_d = pool_index(syms, tf, market, t0, t1, rebal_bars=bpd)
        pi_t = pool_index(syms, tf, market, t0, t1, rebal_bars=1)
        if pi_d:
            emit(f"   * **逐日再平衡**(tf 无关, 主基准): 全程 {pi_d['total'] * 100:+.1f}% "
                 f"({pi_d['n_seg']:,} 段 / {pi_d['days']:.0f} 天)  "
                 f"每日 {pi_d['per_day'] * 100:+.4f}%  年化 {pi_d['ann'] * 100:+.1f}%")
        if pi_t:
            emit(f"   * 逐 {tf} 再平衡(**上界**, 含高频再平衡溢价): 全程 "
                 f"{pi_t['total'] * 100:+.1f}%  每日 {pi_t['per_day'] * 100:+.4f}%  "
                 f"年化 {pi_t['ann'] * 100:+.1f}%")
        emit("   * ⚠ 再平衡频率本身就是收益来源(波动率收割): 同一个池子 4h 再平衡年化 8.9%, "
             "1h 再平衡能到 68%。跨 tf 比较必须用**逐日再平衡**那一档; "
             "逐 tf 那一档只作上界参考, 不可当作「市场基准」。")
        emit("② 买入持有(单标的, 预热后第一根 → 区间末) —— 无再平衡、tf 无关")
        emit(f"   * 全池 {n} 个标的: 全程 中位 {med['total'] * 100:+.0f}% / "
             f"均值 {mean_tot * 100:+.0f}%  年化(复利) 均值 {mean_ann * 100:+.1f}%")
        emit(f"   * **每日收益(线性)**: 中位 {med_pd * 100:+.3f}% / "
             f"均值 {mean_pd * 100:+.3f}%  ← 与策略「每持仓日」同口径")
        if btc:
            emit(f"   * BTCUSDT(池内最强): 全程 {btc['total'] * 100:+.0f}% "
                 f"({btc['days']:.0f} 天, 年化 {btc['ann'] * 100:+.0f}%, "
                 f"每日 {btc['total'] / btc['days'] * 100:+.3f}%)")
        emit("   * ⚠ 买入持有是**满仓全程持有**, 策略是**多笔短持仓 + 至多一仓**; "
             "两者不可直接相减。可比的只有单位时间收益, 且策略还要乘在仓占比。")
        emit("")

    # ---- 随机入场对照 ----
    ctrl = args.control.strip()
    if ctrl in res and res[ctrl]:
        seeds = int(args.seeds)
        emit("=" * 96)
        emit(f"随机入场对照 —— 同离场口径({ctrl})、同止损口径、同入场频率, 只把"
             f"「{'/'.join(kinds)} 择时」换成「随机择时」({seeds} 个种子)")
        ref_diag = {d.sym: d for d in diags[ctrl]}
        real = R.summarize(res[ctrl], market)
        n1 = sum(1 for t in res[ctrl] if t.kind == B1)
        n2 = sum(1 for t in res[ctrl] if t.kind == B2)
        kind_mix = [B1] * n1 + [B2] * n2 or [B1]
        emit(f"  * 真实入场类别比例 {B1}:{B2} = {n1}:{n2} ⇒ 随机侧按此比例抽类别")
        tc = time.time()
        tasks = []
        for sym in syms:
            d = ref_diag.get(sym)
            if d is None or d.err or not d.n_op_bars or not d.n_signals:
                continue
            tasks.append((sym, ctrl, tf, market, t0, t1, d.n_signals, kind_mix, seeds))
        pool_tr: list[list] = [[] for _ in range(seeds)]
        used = 0
        if args.jobs and args.jobs > 1:
            import multiprocessing as mp
            with mp.Pool(int(args.jobs)) as pool:
                it = pool.imap_unordered(_mp_ctrl, tasks, chunksize=1)
                for k, r in enumerate(it):
                    if r is not None:
                        used += 1
                        for s in range(seeds):
                            pool_tr[s].extend(r[1][s])
                    if (k + 1) % 30 == 0:
                        print(f"  对照 {k + 1}/{len(tasks)}", file=sys.stderr, flush=True)
        else:
            for k, t in enumerate(tasks):
                r = _mp_ctrl(t)
                if r is not None:
                    used += 1
                    for s in range(seeds):
                        pool_tr[s].extend(r[1][s])
                if (k + 1) % 30 == 0:
                    print(f"  对照 {k + 1}/{len(tasks)}", file=sys.stderr, flush=True)
        summ = []
        for s in range(seeds):
            su = R.summarize(pool_tr[s], market)
            if su["n"]:
                summ.append(su)
        if summ:
            def col(f):
                return sorted(f(r) for r in summ)

            def show(label, vals, real_v, pct=True):
                n_r = len(vals)
                lo, hi = vals[n_r // 20], vals[-1 - n_r // 20]
                better = (sum(1 for v in vals if v >= real_v) / n_r) if pct else 0.0
                f = (lambda v: f"{v * 100:+.3f}%") if pct else (lambda v: f"{v:.2f}")
                emit(f"  {label:<10} 随机中位 {f(vals[n_r // 2])}  "
                     f"[P5 {f(lo)} , P95 {f(hi)}]   真实 {f(real_v)}   "
                     f"随机里 ≥ 真实的比例 {better:>5.0%}")
            emit(f"  * 参与对照标的 {used} 个;  随机池笔数中位 {summ[0]['n']}  "
                 f"(真实 {real['n']})")
            show("单笔毛期望", col(lambda r: r["exp_pct"]), real["exp_pct"])
            show("每持仓日", col(lambda r: per_day(r)), per_day(real))
            show("胜率", col(lambda r: r["win_rate"]), real["win_rate"])
            show("盈亏比", col(lambda r: r["rr"]), real["rr"], pct=False)
            show("PF", col(lambda r: r["pf"]), real["pf"], pct=False)
            emit("")
            emit("  读法: 「随机里 ≥ 真实的比例」≈ 0.5 ⇒ 真实买卖点与随机不可区分; "
                 "≈ 0(接近 0%) ⇒ 显著优于随机。只比单笔毛期望会被持仓时长污染, "
                 "请以「每持仓日」为主。")
            emit("")
        emit(f"  # 对照耗时 {time.time() - tc:.0f}s")
        emit("")

    _flush(args.out, out_lines)
    print(f"\n报告已写入 {args.out}")
    return 0


def _mp_ctrl(task):
    """随机对照的一个标的: 录日程 + 跑 K 个种子(全在 worker 内完成, 便于并行)。"""
    sym, mode, tf, market, t0, t1, target, kind_mix, seeds = task
    rows = level_schedule(sym, mode, tf, market, t0, t1)
    if not rows:
        return None
    rate0 = target / max(1, len(rows))
    outs = seeds_pass(rows, zlib.crc32(sym.encode("utf-8")), rate0, seeds,
                      kind_mix, market=market, target=target)
    return sym, outs


if __name__ == "__main__":
    sys.exit(main())
