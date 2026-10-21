#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""十字星策略 v2 —— 「新低低位十字进场 + 同级别上涨后的顶信号离场」。

相对 v1(tools/backtest_doji.py, 判决见 DOJI_2026-10-01.md)的三处修改(用户指定):

  [周期] 剔除短周期, 只跑 2h 及以上: {2h, 4h, 1d}。
         2h 无归档原始档, 由 1h 现场聚合(D.resample, UTC 向下取整);
         um 的 4h/1d 由 data.load 自动从 1h 聚合(与旧口径一致)。

  [入场] 在 v1 的「两轮下跌中枢 + 标准十字星(entry < z2.zd)」之上, 新增两条:
         - 新低: 十字星那根的 low <= 前 NEWLOW_N(默认 50) 根的最低 low;
         - 低位: 十字星收盘 <= 近 NEWLOW_N 根收盘的 LOW_Q(默认 30%) 分位。
         (口径注: 「新低之后在低位形成十字」取「当根即创新低」的严格解读;
          NEWLOW_N 可用 --newlow 调, 越大越接近「一轮下跌末段」的宽松概念。)

  [离场] 止损不变(破十字星最低价, 以星低离场)。止盈整体替换为顶信号:
         在 bar j **收盘确认**、以 close[j] 离场(与入场同口径, 无未来函数):
         - 前提一(同级别上涨后): close[j] > entry(相对入场有正收益);
         - 前提二(高位): close[j] >= 近 HI_N(默认 50) 根收盘的 HI_Q(默认 80%) 分位;
         - 信号(满足任一即离场):
           a) top_doji    高位标准十字星(实体比 <= BODY_MAX, 与入场镜像);
           b) top_shadow  长上影: (high - max(open, close)) / (high - low) >= 0.60;
           c) top_fractal_vol 顶部放量顶分形: (j-2, j-1, j) 三根简化顶分形
              (h[j-1] > h[j-2] 且 h[j-1] > h[j]), 且 h[j-1] 是近 HI_N 根最高高点,
              且 vol[j-1] >= VOL_MULT(默认 1.5) x (前 20 根均量, 不含 j-1)。
         同根 bar 既破止损又出顶信号 -> 止损优先(保守, 与 v1 一致)。
         无时间/回撤兜底: 等顶信号或止损或数据尾(报告局限节有讨论)。

两个变体: core = 上述核心规则; full = core + chanbook 全部信号层约束
(只拦入场不改离场, 与 v1 相同的 gate_block 链)。

跑法:
  python tools/backtest_doji2.py --market spot --tf 2h --jobs 10
  python tools/backtest_doji2.py --selfcheck
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook.engine import Diag, Trade, WARMUP, _regime_at       # noqa: E402
from chanbook.filters import ATR1D, ma_veto_ok                    # noqa: E402
from chanbook.level import Level                                  # noqa: E402
from chanbook import market_filters as MF                          # noqa: E402
from chanbook import risk                                          # noqa: E402

import tools.backtest_engine as BT                                 # noqa: E402

__all__ = ["run_symbol", "main"]

BODY_MAX = 0.10      # 十字星实体比上限(进场与高位十字离场共用口径)
MIN_BARS = 500       # 操作级别最小根数(与主引擎一致)
NEWLOW_N = 50        # 「新低/低位」回看根数
LOW_Q = 0.30         # 低位分位(收盘 <= 该分位)
HI_N = 50            # 「高位」回看根数
HI_Q = 0.80          # 高位分位(收盘 >= 该分位)
SHADOW_MIN = 0.60     # 长上影: 上影占全根比例下限
VOL_MULT = 1.5       # 顶分形放量: 中间根量 / 前 20 根均量
VOL_WIN = 20         # 均量窗口(不含分形中间根)


# ------------------------------------------------------------------ 数据
_RS: dict = {}


def _load_tf(market: str, sym: str, tf: str):
    """tf 加载器: 2h 无原始档, 由 1h 现场聚合(与 data.load 的 4h/1d 兜底同源)。"""
    if tf != "2h":
        return D.load(market, sym, tf)
    key = (market, sym)
    if key not in _RS:
        base = D.load(market, sym, "1h")
        _RS[key] = D.resample(base, 2 * D.MS_HOUR) if base is not None else None
    return _RS[key]


# ------------------------------------------------------------------ 规则构件
def is_doji(kl: "D.Kl", i: int, body_max: float) -> tuple[bool, float]:
    """标准十字星: high>low 且 |close-open|/(high-low) <= body_max。"""
    hi, lo = kl.h[i], kl.l[i]
    if hi <= lo:
        return False, 1.0
    body = abs(kl.c[i] - kl.o[i])
    ratio = body / (hi - lo)
    return (ratio <= body_max), ratio


def two_down_zs(zss: list):
    """最后两个已完成中枢是否构成「两轮下跌」(v1 同款, 逐字保留)。"""
    if len(zss) < 2:
        return None
    z1, z2 = zss[-2], zss[-1]
    if z1.enter_dir != -1 or z2.enter_dir != -1:
        return None
    if not (z2.zg < z1.zd):
        return None
    return z1, z2


def _q(xs, q: float) -> float:
    """经验分位(升序取整插值, 窗口小, 足够)。"""
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def low_ok(kl: "D.Kl", i: int, n: int = NEWLOW_N, q: float = LOW_Q) -> bool:
    """入场侧「新低 + 低位」: 当根 low 创前 n 根新低, 且收盘处于近 n 根低位分位。

    窗口不足 n/2 视为证据不够 -> 拒绝(保守)。
    """
    j0 = max(0, i - n)
    if i - j0 < n // 2:
        return False
    if kl.l[i] > min(kl.l[j0:i]):                 # 新低(<= 计入)
        return False
    win = kl.c[max(0, i - n + 1):i + 1]
    if len(win) < 10:
        return False
    return kl.c[i] <= _q(win, q)


def top_signal(kl: "D.Kl", i: int, entry: float, body_max: float,
               hi_n: int = HI_N, hi_q: float = HI_Q,
               shadow_min: float = SHADOW_MIN,
               vol_mult: float = VOL_MULT) -> str | None:
    """离场侧顶信号: 在 bar i 收盘判定, 命中返回标签。只读 <=i 的数据。

    前提: close > entry(同级别上涨后) 且 close 处于近 hi_n 根收盘高位分位。
    """
    close = kl.c[i]
    if not (close > entry):
        return None
    j0 = max(0, i - hi_n + 1)
    if i - j0 < 10:                               # 高位证据不足 -> 不触发
        return None
    if close < _q(kl.c[j0:i + 1], hi_q):
        return None
    # a) 高位十字星
    if is_doji(kl, i, body_max)[0]:
        return "top_doji"
    # b) 长上影
    rng = kl.h[i] - kl.l[i]
    if rng > 0 and (kl.h[i] - max(kl.o[i], close)) / rng >= shadow_min:
        return "top_shadow"
    # c) 顶部放量顶分形: (i-2, i-1, i) 三根简化顶分形 + 近期最高高点 + 放量
    if i >= 2:
        hh = kl.h[i - 1]
        if hh > kl.h[i - 2] and hh > kl.h[i]:
            if hh >= max(kl.h[j0:i + 1]):
                v0 = max(0, i - 1 - VOL_WIN)
                vwin = kl.v[v0:i - 1]
                if len(vwin) >= 5:
                    mean_v = sum(vwin) / len(vwin)
                    if mean_v > 0 and kl.v[i - 1] >= vol_mult * mean_v:
                        return "top_fractal_vol"
    return None


def _mk(pos: dict, exit_px: float, exit_ts: int, reason: str) -> Trade:
    """整仓成交记录(v2 无减仓, 不再有 half)。"""
    return Trade(
        sym=pos["sym"], entry_ts=pos["entry_ts"], exit_ts=int(exit_ts),
        side="long", kind="DOJI2", trend="", regime=pos["regime"], counter=0,
        confirm="", entry=pos["entry"], exit=float(exit_px),
        stop0=pos["stop"], target=0.0,
        pnl_pct=(float(exit_px) - pos["entry"]) / pos["entry"],
        risk_pct=pos["risk_pct"], notional=pos["notional"], reason=reason,
        hold_min=int((int(exit_ts) - pos["entry_ts"]) / D.MS_MIN),
        feats=dict(pos["feats"]),
    )


def manage(pos: dict, kl: "D.Kl", i: int, body_max: float) -> tuple[list[Trade], bool]:
    """v2 持仓单 bar 管理: 止损优先, 顶信号收盘离场。"""
    bar_ts = int(kl.ts[i])
    # 1) 止损(不变): 破十字星最低价 => 全部以星低离场
    if kl.l[i] <= pos["stop"]:
        return ([_mk(pos, pos["stop"], bar_ts, "stop_loss")], False)
    # 2) 顶信号: 收盘确认, 以收盘价离场
    sig = top_signal(kl, i, pos["entry"], body_max)
    if sig:
        pos["feats"]["exit_hi_q"] = round(_q(
            kl.c[max(0, i - HI_N + 1):i + 1], HI_Q) / pos["entry"], 6)
        return ([_mk(pos, kl.c[i], bar_ts, sig)], False)
    return [], True


# ------------------------------------------------------------------ FULL 约束
class GateView:
    """FULL 变体的约束视图(全部 fail-open: 数据不足放行, 与 chanbook 一致)。

    与 v1 同款; 2h 操作级别时 Correl 用 1h 归档, 其余构件用 1d/4h 原档。
    """

    __slots__ = ("overheat", "resistance", "fib", "correl", "atr1d")

    def __init__(self, market: str, sym: str, op_kl) -> None:
        self.overheat = self.resistance = self.fib = self.correl = self.atr1d = None
        try:
            self.overheat = MF.Overheat(D.load(market, sym, "1d"),
                                        D.load(market, sym, "4h"))
        except Exception:
            pass
        try:
            self.resistance = MF.Resistance(D.load(market, sym, "1h"))
        except Exception:
            pass
        try:
            self.fib = MF.Fib()
        except Exception:
            pass
        try:
            self.correl = MF.Correl(op_kl if op_kl.tf == "1h" else
                                    D.load(market, sym, "1h"),
                                    D.load(market, "BTCUSDT", "1h"),
                                    D.load(market, "ETHUSDT", "1h"))
        except Exception:
            pass
        try:
            self.atr1d = ATR1D(D.load(market, sym, "1d"), 14)
        except Exception:
            pass


def gate_block(market: str, gv: "GateView", op: "Level", entry: float,
               close_ts: int, dg: Diag, vol_gate: float) -> bool:
    """chanbook 信号层约束链(与 v1 逐字同款, 顺序与 Strategy._make 一致)。"""
    if _regime_at(market, close_ts) == "bear":
        dg.n_regime_block += 1
        return True
    if gv.overheat is not None and gv.overheat.verdict(close_ts, "long")[0]:
        dg.n_overheat_block += 1
        return True
    if gv.resistance is not None and gv.resistance.verdict(close_ts, "long",
                                                            entry)[0]:
        dg.n_resistance_block += 1
        return True
    if gv.fib is not None:
        blocked, _why, _d = gv.fib.verdict(op.kl, int(op.cur), "long", entry)
        if blocked:
            dg.n_fib_block += 1
            return True
    if gv.correl is not None and gv.correl.verdict(close_ts, "long")[0]:
        dg.n_correl_block += 1
        return True
    if gv.atr1d is not None and gv.atr1d.ok(close_ts, 0.05) is False:
        dg.n_volcap_block += 1
        return True
    if op.ma is not None:
        ok = ma_veto_ok(op.ma, entry, "ma", 5, 10)
        if ok is False:
            dg.n_ma_block += 1
            return True
    if vol_gate > 0.0 and op.vol is not None:
        fb = op.bib.final_bis
        if len(fb) >= 2:
            b_prev, b_last = fb[-2], fb[-1]
            r = op.vol.ratio(int(b_prev.start_ts), int(b_prev.end_ts),
                             int(b_last.start_ts), int(b_last.end_ts))
            if r is not None and r >= vol_gate:
                dg.n_blocked["vol_gate"] = dg.n_blocked.get("vol_gate", 0) + 1
                return True
    return False


# ------------------------------------------------------------------ 单标的
def run_symbol(market: str, sym: str, tf: str, full: bool,
               body_max: float = BODY_MAX, t_end: int = 0,
               newlow_n: int = NEWLOW_N
               ) -> tuple[list[Trade], Diag]:
    """单标的信号层回测。full=True 时叠加 chanbook 全部信号层约束。"""
    dg = Diag(sym=sym)
    kl = _load_tf(market, sym, tf)
    if kl is None or kl.n < MIN_BARS:
        dg.err = f"{tf} 根数不足 {0 if kl is None else kl.n}"
        return [], dg
    notional = float(risk.CALIB[market]["notional"])
    op = Level("op", tf, "bi",
               ma_periods=((5, 10) if full else None),
               vol_on=bool(full)).attach(kl)
    gv = GateView(market, sym, op.kl) if full else None
    trades: list[Trade] = []
    pos: dict | None = None
    start = WARMUP
    dg.n_op_bars = max(0, kl.n - start)
    for i in range(start, kl.n):
        bar_ts = int(kl.ts[i])
        if t_end and bar_ts > t_end:
            if pos is not None:
                trades.append(_mk(pos, kl.c[i], bar_ts, "segment_end"))
                pos = None
            break
        close_ts = bar_ts + op.tf_ms
        op.advance(close_ts)
        # ---- 先管理既有持仓(至多一仓) ----
        if pos is not None:
            closed, alive = manage(pos, kl, i, body_max)
            trades.extend(closed)
            if not alive:
                pos = None
            continue
        # ---- 空仓: 找入场(v1 条件 + 新低/低位) ----
        ok, body_ratio = is_doji(kl, i, body_max)
        if not ok:
            continue
        pair = two_down_zs(op.zb.zss)
        if pair is None:
            continue
        z1, z2 = pair
        entry = float(kl.c[i])
        if not (bar_ts > int(z2.end_ts)):
            continue
        if not (entry < z2.zd):
            continue
        if not low_ok(kl, i, newlow_n):           # v2 新增: 新低 + 低位
            continue
        dg.n_seen["DOJI2"] = dg.n_seen.get("DOJI2", 0) + 1
        # ---- FULL: chanbook 约束链(只拦入场) ----
        if full and gate_block(market, gv, op, entry, close_ts, dg, 1.0):
            continue
        stop = float(kl.l[i])
        feats = {
            "body_ratio": round(body_ratio, 4),
            "range_pct": round((kl.h[i] - kl.l[i]) / entry, 6),
            "stop_pct": round((entry - stop) / entry, 6),
            "newlow_n": newlow_n,
            "full": int(bool(full)),
        }
        pos = {"sym": sym, "entry": entry, "entry_ts": bar_ts, "stop": stop,
               "regime": _regime_at(market, close_ts), "notional": notional,
               "risk_pct": (entry - stop) / entry, "feats": feats}
    if pos is not None:
        last = kl.n - 1
        trades.append(_mk(pos, kl.c[last], int(kl.ts[last]), "end_of_data"))
    dg.n_bi = len(op.bib.bis)
    dg.n_zs = len(op.zb.zss)
    return trades, dg


# ------------------------------------------------------------------ 批量
def _mp_run(task):
    market, sym, tf, full, body_max, t_end = task
    tr, dg = run_symbol(market, sym, tf, full, body_max, t_end)
    return sym, tr, dg


def run_pool(market: str, tf: str, full: bool, syms: list, jobs: int,
             body_max: float = BODY_MAX):
    tasks = [(market, s, tf, full, body_max, 0) for s in syms]
    tr_all: list[Trade] = []
    dg_all: list[Diag] = []
    t0 = time.time()
    if jobs > 1:
        import multiprocessing as mp
        with mp.Pool(jobs) as pool:
            for k, (sym, tr, dg) in enumerate(pool.imap_unordered(_mp_run, tasks,
                                                                  chunksize=1)):
                tr_all.extend(tr)
                dg_all.append(dg)
                if (k + 1) % 20 == 0:
                    print(f"    [{tf} {'full' if full else 'core'}] {k+1}/"
                          f"{len(syms)} 笔={len(tr_all)} {time.time()-t0:.0f}s",
                          file=sys.stderr, flush=True)
    else:
        for t in tasks:
            sym, tr, dg = _mp_run(t)
            tr_all.extend(tr)
            dg_all.append(dg)
    dg_all.sort(key=lambda d: d.sym)
    return tr_all, dg_all


# ------------------------------------------------------------------ 自检
def _selfcheck() -> int:
    """A 形态判定 / B 离场优先级与门槛 / C 截断一致性(无未来, 2h 聚合路径)
    / D 新低低位判定 / E 2h 聚合正确性。"""
    from chanbook.data import Kl
    from chanbook.zhongshu import ZS

    fails = []

    def ck(cond, msg):
        print(("  PASS " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # A) 形态判定
    rows = [(0, 100.0, 101.0, 99.0, 100.05, 1.0),
            (0, 100.0, 101.0, 99.0, 100.5, 1.0),
            (0, 100.0, 100.0, 100.0, 100.0, 1.0)]
    kl = Kl(rows=rows)
    ck(is_doji(kl, 0, 0.1)[0] is True, "A1 实体2.5% = 十字星")
    ck(is_doji(kl, 1, 0.1)[0] is False, "A2 实体25% ≠ 十字星")
    ck(is_doji(kl, 2, 0.1)[0] is False, "A3 零振幅 ≠ 十字星")

    # B) 顶信号: 前提与三类信号
    def bar(ts, o, h, l, c, v):
        return (ts, o, h, l, c, v)
    # 上涨后的高位 K 线(实体 12% 避免误判十字)
    bars = [bar(0, 100 + k, 100 + k, 100 + k, 100 + k, 10.0) for k in range(20)]
    bars.append(bar(20, 128.0, 130.0, 127.0, 128.0, 10.0))
    bars.append(bar(21, 127.5, 128.9, 127.0, 127.8, 10.0))   # 上影 1.1/1.9=58%
    bars.append(bar(22, 127.5, 129.5, 127.0, 127.8, 10.0))   # 上影 1.7/2.5=68%
    kl2 = Kl(rows=bars)
    ck(top_signal(kl2, 21, 100.0, 0.1) is None, "B1 上影58% < 60% 不触发")
    ck(top_signal(kl2, 22, 100.0, 0.1) == "top_shadow", "B2 长上影68% -> top_shadow")
    ck(top_signal(kl2, 22, 130.0, 0.1) is None, "B3 未上涨(close<=entry) 不触发")
    ck(top_signal(kl2, 5, 100.0, 0.1) is None, "B4 低位(未到80分位) 不触发")
    # 高位十字
    bars3 = list(bars[:22])
    bars3.append(bar(22, 128.0, 129.0, 127.0, 128.05, 10.0))  # 实体0.05/2=2.5%
    kl3 = Kl(rows=bars3)
    ck(top_signal(kl3, 22, 100.0, 0.1) == "top_doji", "B5 高位十字星 -> top_doji")
    # 顶部放量顶分形: i-1 是最高高点且放量 2x
    bars4 = [bar(0, 100 + k, 100 + k, 100 + k, 100 + k, 10.0) for k in range(25)]
    bars4.append(bar(25, 124.0, 125.0, 123.0, 124.5, 30.0))  # 中间根, 量3x
    bars4.append(bar(26, 124.0, 124.5, 123.0, 124.4, 10.0))  # 右肩更低高点
    kl4 = Kl(rows=bars4)
    ck(top_signal(kl4, 26, 100.0, 0.1) == "top_fractal_vol",
       "B6 顶分形+放量2x -> top_fractal_vol")
    bars5 = list(bars4)
    bars5[26] = bar(26, 124.0, 125.5, 123.0, 124.4, 10.0)    # 右肩更高 -> 非顶分形
    kl5 = Kl(rows=bars5)
    ck(top_signal(kl5, 26, 100.0, 0.1) is None, "B7 右肩更高 -> 非顶分形")

    # B2) manage 优先级: 同根破止损 + 顶信号 -> 止损优先
    pos = {"sym": "X", "entry": 100.0, "entry_ts": 0, "stop": 98.0,
           "regime": "range", "notional": 5.0, "risk_pct": 0.02, "feats": {}}
    klB = Kl(rows=[(0, 100.0, 130.0, 97.5, 129.0, 10.0)])
    out, alive = manage(dict(pos), klB, 0, 0.1)
    ck(len(out) == 1 and not alive and out[0].reason == "stop_loss"
       and abs(out[0].exit - 98.0) < 1e-12, "B8 同根止损与顶信号 -> 止损优先")

    # D) 新低/低位判定
    def mk_kl(lows, closes):
        return Kl(rows=[(k, c, c + 0.5, lo, c, 1.0)
                        for k, (lo, c) in enumerate(zip(lows, closes))])
    lows = [100.0 - k * 0.1 for k in range(49)] + [95.0]      # 前 49 根递降, 当根新低
    closes = [100.0 - k * 0.1 for k in range(50)]
    ck(low_ok(mk_kl(lows, closes), 49, 50) is True, "D1 当根创新低且低位 -> 通过")
    lows2 = [100.0 - k * 0.1 for k in range(49)] + [96.0]     # 当根 > 前一根(未创新低)
    ck(low_ok(mk_kl(lows2, closes), 49, 50) is False, "D2 未创新低 -> 拒绝")
    closes3 = [90.0 + k * 0.2 for k in range(50)]            # 收盘一路走高
    ck(low_ok(mk_kl(lows, closes3), 49, 50) is False, "D3 新低但收盘不低位 -> 拒绝")

    # E) 2h 聚合正确性
    import bisect as _bs
    klh = D.load("spot", "BTCUSDT", "1h")
    if klh is None or klh.n < 100:
        print("  SKIP E (1h 数据不足)")
    else:
        kl2h = _load_tf("spot", "BTCUSDT", "2h")
        ok1 = kl2h is not None and 0 < kl2h.n <= klh.n
        # 每根 2h 的 ts 必须落在偶数小时边界, 且相邻 ts 差是 2h 的整数倍
        ok2 = kl2h is not None and all(
            (int(t) // D.MS_HOUR) % 2 == 0 for t in kl2h.ts[:50])
        diffs = ({int(b) - int(a) for a, b in zip(kl2h.ts, kl2h.ts[1:])}
                 if kl2h is not None else set())
        # 原始 1h 归档存在缺口(如 32h), 聚合后缺口原样保留是正确行为,
        # 只要求步长是 2h 的正整数倍。
        ok3 = bool(diffs) and all(d > 0 and d % (2 * D.MS_HOUR) == 0
                                  for d in diffs)
        ck(ok1 and ok2 and ok3,
           f"E1 2h 序列合法: n={0 if kl2h is None else kl2h.n}, "
           f"ts 偶时对齐且步长为 2h 正整数倍(缺口步长 {sorted(diffs)})")
        if kl2h is not None and kl2h.n:
            k = min(10, kl2h.n - 1)
            i0 = _bs.bisect_left(klh.ts, kl2h.ts[k])
            i1 = min(i0 + 1, klh.n - 1)
            ck(abs(kl2h.h[k] - max(klh.h[i0], klh.h[i1])) < 1e-9,
               "E2 2h 高 = 对应 1h 两根(或末根一根)的最大值")

    # C) 截断一致性(无未来函数, 走 2h 聚合路径; ADAUSDT 在冒烟中有成交)
    if klh is None or klh.n < MIN_BARS * 2 + 1000:
        print("  SKIP C (1h 数据不足 2h 截断检验)")
    else:
        market, sym, tf = "spot", "ADAUSDT", "2h"
        kl = _load_tf(market, sym, tf)
        cut = int(kl.ts[kl.n // 2])
        tr_full, _ = run_symbol(market, sym, tf, full=False)
        tr_cut, _ = run_symbol(market, sym, tf, full=False, t_end=cut)
        a = [t for t in tr_full if t.exit_ts < cut]
        b = [t for t in tr_cut if t.reason != "segment_end"]
        same = len(a) == len(b) and all(
            x.to_dict() == y.to_dict() for x, y in zip(a, b))
        ck(same, f"C1 截断一致性: 全量前半 {len(a)} 笔 == 截断跑 {len(b)} 笔(逐字段)")

    print(f"\n自检结果: {'全部通过' if not fails else f'{len(fails)} 项失败'}")
    return 0 if not fails else 1


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=("spot", "um"))
    ap.add_argument("--tf", default="4h", choices=("2h", "4h", "1d"))
    ap.add_argument("--variant", default="both", choices=("core", "full", "both"))
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--body", type=float, default=BODY_MAX)
    ap.add_argument("--newlow", type=int, default=NEWLOW_N)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sym", default="")
    ap.add_argument("--spot-pool", action="store_true",
                    help="用 spot 93 主流标的池 ∩ 本市场有 1h 档(um 4h/1d/2h 均由"
                         "1h 聚合, 无原始档可查, 故按 1h 判存在)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    syms = ([a.sym] if a.sym else D.symbols(a.market))
    if a.spot_pool and not a.sym:
        keep = []
        for s in D.symbols("spot"):
            p = os.path.join(D.ARCH, a.market, "k", f"{s}_1h.csv.gz")
            if os.path.exists(p):
                keep.append(s)
        syms = keep
    if a.limit:
        syms = syms[:a.limit]
    out_path = a.out or f"out/doji2_{a.market}_{a.tf}.txt"
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    emit(f"# 十字星v2(新低低位进场+顶信号离场) · {a.market} {a.tf} · "
         f"body_max={a.body} newlow_n={a.newlow} · {stamp} · {len(syms)} 标的")
    emit("# 入场: 两轮下跌中枢后标准十字星(收盘<z2.zd) + 当根创近N根新低 + 收盘低位分位")
    emit("# 离场: 破星低止损(不变) / 同级别上涨后高位: 十字星|长上影>=60%|"
         "顶部放量顶分形(收盘确认)")
    emit("")
    emit(BT.HDR)

    rows: dict[str, dict] = {}
    for name, full in (("core", False), ("full", True)):
        if a.variant not in ("both", name):
            continue
        t0 = time.time()
        tr, dg = run_pool(a.market, a.tf, full, syms, a.jobs, a.body)
        label = f"{a.tf} {name}"
        r = BT.stats_row(tr, a.market, a.tf, dg, label)
        rows[name] = r
        emit(BT.fmt_row(r) + f"   [{time.time() - t0:.0f}s]")
    emit("")

    for name, r in rows.items():
        if not r.get("n"):
            continue
        emit(f"## {name} 分离场原因")
        for ln in BT.R.by_reason(r["trades"], a.market):
            emit("  " + ln)
        emit(f"## {name} 分年")
        for ln in BT.R.by_year(r["trades"], a.market):
            emit("  " + ln)
        rb = BT.risk_block(r["trades"])
        emit(f"  候选池等权风险画像: 期末 {rb['final']*100:+.2f}%  "
             f"最大回撤 {rb['mdd']*100:+.2f}%")
        emit("")
    if "core" in rows and "full" in rows and rows["core"].get("n") \
            and rows["full"].get("n"):
        n, b, p = BT.signed_pairs(rows["full"]["by_sym"], rows["core"]["by_sym"])
        emit(f"## 分标的配对: full 更好的 {b}/{n} = {p:.0%}(基准 = core)")
        emit("")
    pi = BT.pool_index(syms, "1h", a.market, 0, 0, rebal_bars=24)
    if pi:
        emit(f"## 同池等权指数(1h 网格, 逐日再平衡): 总 {pi['total']:+.1%} "
             f"每日 {pi['per_day']:+.4%} 年化 {pi['ann']:+.1%}")
        for name, r in rows.items():
            if not r.get("n"):
                continue
            pdn = r["per_day"] * r["share"]
            emit(f"  {name:<6} 每自然日 {pdn:+.4%}  池每日 {pi['per_day']:+.4%}"
                 f"  {pdn / pi['per_day'] if pi['per_day'] else 0:.2f}x")
        emit("")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
