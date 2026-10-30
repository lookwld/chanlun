#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""十字星拐点策略回测 —— 「两轮下跌中枢后的标准十字星」。

使用者给定的核心交易规则(**回测中逐字保持, 不做任何改动**):

  入场(换线时刻 = 十字星那根 bar 收盘):
    1. 操作级别上, 最后两个**已结束**中枢 z1(先/高)、z2(后/低)满足:
       - 两者均为下跌进入(z1.enter_dir == z2.enter_dir == -1);
       - z2 整体低于 z1(z2.zg < z1.zd, 即"中枢依次下移" = 两轮下跌);
       - 当前 bar 时间在 z2 结束之后(bar_ts > z2.end_ts);
       - 十字星收盘价 < z2.zd(仍在最近中枢下方, 还没到第一目标)。
    2. 当前 bar 是**标准十字星**: high > low 且 |close-open|/(high-low) <= body_max。
    3. 入场价 = 十字星收盘价。
  离场(优先级从高到低, 同根 bar 同时触发按**止损优先**, 保守):
    - 止损: 任何后续 bar 的 low <= 十字星最低价 => 以十字星最低价离场;
    - 减仓: 首次 high >= z2.zd(离十字星最近的中枢下沿) => 减仓一半;
    - 清仓: 减仓之后 high >= z1.zd(第二个中枢下沿) => 清仓(同根到达 t1/t2
      时只执行 t1, t2 留到下一根, 同样保守);
    - 数据末尾 => 最后一根收盘价强平。
  持仓: 同标的同时至多一仓; 持仓期间不找新信号; 空仓后条件仍满足可再入场。

两个变体:
  core : 只跑上面的核心规则(纯策略);
  full : 核心规则 + chanbook 的全部**信号层**交易约束(只拦入场, 不改离场):
         regime(BTC bear 停机) / overheat(4h+1d 过热) / resistance(1h 前高压制)
         / fib(回撤深度) / correl(BTC/ETH 联动) / atr1d_cap(日线 ATR14%<=5%)
         / ma_mode="ma"(两条均线 5/10 否决, 即 BookCfg.enhanced 的口径)
         / vol_gate=1.0(最后两根冻结笔"后段/前段"缩量)。
  ⚠ 方向闸门(dir_gate)按 chanbook 生产口径对 B1 类逆势信号**不启用**(本策略
    的"下跌末端十字星"与 B1 同类); 熔断护栏(guards)/资金池/名额是**组合层**
    构件, 按 chanbook 铁律不进信号层 A/B(见报告局限节)。

无未来函数: 结构状态由 `Level.advance(close_ts)` 的"已收盘"时钟保证
(与主引擎同一套); 十字星判定只读当前 bar; 中枢只取已完成(zb.zss)。

跑法:
  python tools/backtest_doji.py --market spot --tf 1h --jobs 10 --out out/doji_spot_1h.txt
  python tools/backtest_doji.py --market um --tf 5m --jobs 10
  python tools/backtest_doji.py --selfcheck            # 口径自检(合成数据+截断一致性)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook import engine as E                                  # noqa: E402
from chanbook.engine import Diag, Trade, WARMUP, _regime_at       # noqa: E402
from chanbook.filters import ATR1D, ma_veto_ok                    # noqa: E402
from chanbook.level import Level                                  # noqa: E402
from chanbook import market_filters as MF                         # noqa: E402
from chanbook import risk                                         # noqa: E402

import tools.backtest_engine as BT                                 # noqa: E402

# BT.stats_row 的在仓率换算表只列了旁路引擎用过的周期; 本工具要跑 5m/15m/30m,
# 在这里补齐(不改共享文件, 只补本进程的映射)。
BT.BAR_MIN.setdefault("5m", 5)
BT.BAR_MIN.setdefault("15m", 15)
BT.BAR_MIN.setdefault("30m", 30)

__all__ = ["run_symbol", "main"]

# 十字星判定: 实体占全幅比例的上限(标准十字星口径)
BODY_MAX = 0.10
# 操作级别最小根数(与主引擎 LevelSet.ok 的 op 门槛一致)
MIN_BARS = 500


# ------------------------------------------------------------------ 规则构件
def is_doji(kl: "D.Kl", i: int, body_max: float) -> tuple[bool, float]:
    """标准十字星: high>low 且 |close-open|/(high-low) <= body_max。返回 (是否, 实体比)。"""
    hi, lo = kl.h[i], kl.l[i]
    if hi <= lo:
        return False, 1.0
    body = abs(kl.c[i] - kl.o[i])
    ratio = body / (hi - lo)
    return (ratio <= body_max), ratio


def two_down_zs(zss: list):
    """最后两个已完成中枢是否构成「两轮下跌」。

    条件: 两个都下跌进入(enter_dir==-1), 且后者整体在前者之下(z2.zg < z1.zd)。
    相邻中枢按 ZSBuilder 构造本就零重叠, 这里显式再判一次"下移"方向。
    返回 (z1, z2) 或 None。
    """
    if len(zss) < 2:
        return None
    z1, z2 = zss[-2], zss[-1]
    if z1.enter_dir != -1 or z2.enter_dir != -1:
        return None
    if not (z2.zg < z1.zd):          # 第二轮中枢整体更低 = 依次下移
        return None
    return z1, z2


def _mk(pos: dict, exit_px: float, exit_ts: int, reason: str, half: bool) -> Trade:
    """按整仓/半仓生成一条成交记录(半仓名义额减半, pnl_pct 口径不变)。"""
    nm = pos["notional"] * (0.5 if half else 1.0)
    return Trade(
        sym=pos["sym"], entry_ts=pos["entry_ts"], exit_ts=int(exit_ts),
        side="long", kind="DOJI", trend="", regime=pos["regime"], counter=0,
        confirm="", entry=pos["entry"], exit=float(exit_px),
        stop0=pos["stop"], target=pos["t2"],
        pnl_pct=(float(exit_px) - pos["entry"]) / pos["entry"],
        risk_pct=pos["risk_pct"], notional=nm, reason=reason,
        hold_min=int((int(exit_ts) - pos["entry_ts"]) / D.MS_MIN),
        feats=dict(pos["feats"]),
    )


def manage_doji(pos: dict, kl: "D.Kl", i: int) -> tuple[list[Trade], bool]:
    """十字星持仓的单 bar 管理。返回 (本次离场成交, 是否继续持仓)。

    优先级(保守): 止损 > 减仓(t1) > 清仓(t2); 同根同时够到 t1/t2 时只执行 t1,
    t2 留到下一根 —— 与主引擎"同根同时触发按止损优先"同一保守取向。
    """
    bar_ts = int(kl.ts[i])
    # 1) 止损优先: 破十字星最低价 => 剩余仓位全部离场
    if kl.l[i] <= pos["stop"]:
        return ([_mk(pos, pos["stop"], bar_ts, "stop_loss", half=pos["reduced"])],
                False)
    # 2) 到最近中枢(z2.zd)减仓一半
    if not pos["reduced"] and kl.h[i] >= pos["t1"]:
        pos["reduced"] = True
        return ([_mk(pos, pos["t1"], bar_ts, "zs1_half", half=True)], True)
    # 3) 减仓后到第二个中枢(z1.zd)清仓
    if pos["reduced"] and kl.h[i] >= pos["t2"]:
        return ([_mk(pos, pos["t2"], bar_ts, "zs2_exit", half=True)], False)
    return [], True


# ------------------------------------------------------------------ FULL 约束
class GateView:
    """FULL 变体的约束视图(全部 fail-open: 数据不足放行, 与 chanbook 一致)。"""

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
    """chanbook 信号层约束链(顺序与 Strategy._make 一致)。返回是否拦截。"""
    # regime: BTC bear 停机
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
    # 日线波动率上限(与 chan-quant vol_gate 同源)
    if gv.atr1d is not None and gv.atr1d.ok(close_ts, 0.05) is False:
        dg.n_volcap_block += 1
        return True
    # 两条均线(5/10)否决型 —— BookCfg.enhanced 的口径
    if op.ma is not None:
        ok = ma_veto_ok(op.ma, entry, "ma", 5, 10)
        if ok is False:                       # None(未就绪)放行
            dg.n_ma_block += 1
            return True
    # 量能: 最后两根冻结笔 后段均量/前段均量 < vol_gate(缩量放行)
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


_MKT = {}


# ------------------------------------------------------------------ 单标的
def run_symbol(market: str, sym: str, tf: str, full: bool,
               body_max: float = BODY_MAX, t_end: int = 0
               ) -> tuple[list[Trade], Diag]:
    """单标的信号层回测。full=True 时叠加 chanbook 全部信号层约束。"""
    dg = Diag(sym=sym)
    kl = D.load(market, sym, tf)
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
            if pos is not None:                      # 截断边界 = 离场(自检用)
                trades.append(_mk(pos, kl.c[i], bar_ts, "segment_end",
                                  half=pos["reduced"]))
                pos = None
            break
        close_ts = bar_ts + op.tf_ms                 # 唯一时钟(已收盘判据)
        op.advance(close_ts)
        # ---- 先管理既有持仓(至多一仓) ----
        if pos is not None:
            closed, alive = manage_doji(pos, kl, i)
            trades.extend(closed)
            if not alive:
                pos = None
            continue
        # ---- 空仓: 找入场 ----
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
        if not (entry < z2.zd):                      # 仍在最近中枢下方
            continue
        dg.n_seen["DOJI"] = dg.n_seen.get("DOJI", 0) + 1
        # ---- FULL: chanbook 约束链(只拦入场, 不改离场) ----
        if full and gate_block(market, gv, op, entry, close_ts, dg, 1.0):
            continue
        stop = float(kl.l[i])
        feats = {
            "body_ratio": round(body_ratio, 4),
            "range_pct": round((kl.h[i] - kl.l[i]) / entry, 6),
            "stop_pct": round((entry - stop) / entry, 6),
            "t1_pct": round((z2.zd - entry) / entry, 6),
            "t2_pct": round((z1.zd - entry) / entry, 6),
            "z1_h_pct": round(z1.height / entry, 6),
            "z2_h_pct": round(z2.height / entry, 6),
            "full": int(bool(full)),
        }
        pos = {"sym": sym, "entry": entry, "entry_ts": bar_ts, "stop": stop,
               "t1": float(z2.zd), "t2": float(z1.zd), "reduced": False,
               "regime": _regime_at(market, close_ts), "notional": notional,
               "risk_pct": (entry - stop) / entry, "feats": feats}
    if pos is not None:
        last = kl.n - 1
        trades.append(_mk(pos, kl.c[last], int(kl.ts[last]), "end_of_data",
                          half=pos["reduced"]))
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
    """四类口径自检: A 十字星判定 / B 离场优先级 / C 截断一致性(无未来函数)
    / D 边界(两轮下跌中枢判定)。"""
    from chanbook.data import Kl
    from chanbook.zhongshu import ZS

    fails = []

    def ck(cond, msg):
        print(("  PASS " if cond else "  FAIL ") + msg)
        if not cond:
            fails.append(msg)

    # A) 十字星判定
    rows = [(0, 100.0, 101.0, 99.0, 100.05, 1.0),    # 实体 0.05/2 = 2.5% -> 十字星
            (0, 100.0, 101.0, 99.0, 100.5, 1.0),     # 25% -> 非
            (0, 100.0, 100.0, 100.0, 100.0, 1.0)]    # high==low -> 非
    kl = Kl(rows=rows)
    ck(is_doji(kl, 0, 0.1)[0] is True, "A1 实体2.5%判定为十字星")
    ck(is_doji(kl, 1, 0.1)[0] is False, "A2 实体25%不是十字星")
    ck(is_doji(kl, 2, 0.1)[0] is False, "A3 零振幅 bar 不是十字星")

    # B) 离场优先级: 同根既破止损又到 t1 -> 止损优先; 同根 t1+t2 -> 只执行 t1
    pos = {"sym": "X", "entry": 100.0, "entry_ts": 0, "stop": 98.0,
           "t1": 101.0, "t2": 102.0, "reduced": False, "regime": "range",
           "notional": 5.0, "risk_pct": 0.02, "feats": {}}
    kl2 = Kl(rows=[(0, 100.0, 102.0, 97.5, 101.0, 1.0)])     # 破止损且够 t1/t2
    out, alive = manage_doji(dict(pos), kl2, 0)
    ck(len(out) == 1 and not alive and out[0].reason == "stop_loss"
       and abs(out[0].exit - 98.0) < 1e-12, "B1 同根止损与目标冲突 -> 止损优先")
    pos_r = dict(pos)
    kl3 = Kl(rows=[(0, 100.0, 102.5, 99.5, 101.5, 1.0)])     # 够 t1 也够 t2
    out, alive = manage_doji(pos_r, kl3, 0)
    ck(len(out) == 1 and alive and out[0].reason == "zs1_half"
       and pos_r["reduced"] is True, "B2 同根到 t1+t2 -> 只执行减仓 t1")
    out2, alive2 = manage_doji(pos_r, Kl(rows=[(0, 100.0, 103.0, 99.5, 102.0, 1.0)]), 0)
    ck(len(out2) == 1 and not alive2 and out2[0].reason == "zs2_exit", "B3 减仓后到 t2 清仓")
    pos_s = dict(pos)
    out3, alive3 = manage_doji(pos_s, Kl(rows=[(0, 100.0, 100.5, 97.9, 98.5, 1.0)]), 0)
    ck(out3 and not alive3 and out3[0].reason == "stop_loss",
       "B4 未减仓直接破止损 -> 全仓离场")

    # D) 两轮下跌中枢判定
    def mk_zs(zg, zd, edir):
        return ZS(i=0, level=1, i0=0, i1=2, zg=zg, zd=zd, gg=zg, dd=zd,
                  start_ts=0, end_ts=1, enter_i=-1, leave_i=0,
                  enter_dir=edir)
    z_hi, z_lo = mk_zs(110.0, 105.0, -1), mk_zs(100.0, 95.0, -1)
    ck(two_down_zs([z_hi, z_lo]) is not None, "D1 两个下移的下跌中枢 -> 通过")
    ck(two_down_zs([mk_zs(110.0, 105.0, 1), z_lo]) is None, "D2 第一个上涨进入 -> 拒绝")
    ck(two_down_zs([mk_zs(100.0, 95.0, -1), mk_zs(110.0, 105.0, -1)]) is None,
       "D3 第二个中枢更高(上移) -> 拒绝")
    ck(two_down_zs([z_lo]) is None, "D4 只有一个中枢 -> 拒绝")

    # C) 截断一致性(无未来函数的端到端检验):
    #    全量跑出的成交中, 离场时间 < 截断点的那些, 必须与"只喂到截断点"跑出的
    #    成交逐字段一致 —— 否则说明入场/离场用到了截断点之后的数据。
    market, sym, tf = "spot", "BTCUSDT", "1h"
    kl = D.load(market, sym, tf)
    if kl is None or kl.n < MIN_BARS + 1000:
        print("  SKIP C (数据不足)")
    else:
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
    ap.add_argument("--tf", default="1h",
                    choices=("5m", "15m", "30m", "1h", "2h", "4h", "1d"))
    ap.add_argument("--variant", default="both", choices=("core", "full", "both"))
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--body", type=float, default=BODY_MAX)
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sym", default="")
    ap.add_argument("--spot-pool", action="store_true",
                    help="用 spot 93 主流标的池 ∩ 本市场有数据(um 全池 534 个太重, "
                         "且与 spot 不同池不可比)")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()

    if a.selfcheck:
        return _selfcheck()

    syms = ([a.sym] if a.sym else D.symbols(a.market))
    if a.spot_pool and not a.sym:
        keep = []
        for s in D.symbols("spot"):
            p = os.path.join(D.ARCH, a.market, "k", f"{s}_{a.tf}.csv.gz")
            if os.path.exists(p):
                keep.append(s)
        syms = keep
    if a.limit:
        syms = syms[:a.limit]
    out_path = a.out or f"out/doji_{a.market}_{a.tf.replace(' ', '')}.txt"
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    emit(f"# 十字星拐点策略(两轮下跌中枢后的标准十字星) · {a.market} {a.tf} · "
         f"body_max={a.body} · {stamp} · {len(syms)} 标的 · jobs={a.jobs}")
    emit("# 核心规则: 换线十字星入场 / 破星低点止损 / 到 z2.zd 减半 / 到 z1.zd 清仓")
    emit("# full = 核心规则 + chanbook 全部信号层约束(regime/过热/前高/斐波/联动/"
         "atr1d/两条均线/量能), 只拦入场不改离场")
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

    # 五读数补充: 分离场原因 / 分年 / 配对 / 池基准
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
