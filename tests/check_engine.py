"""引擎 / 风控 / 报告 自检。

运行: python tests/check_engine.py

重点验证三件事:
  1. 离开管理(追踪止损/止损/目标/结构破位)与旧引擎口径一致;
  2. 引擎**没有未来函数**: 把数据截断到某时刻, 该时刻之前的成交(入场时间与
     入场价)必须逐字不变;
  3. 报告指标算术正确(单笔期望/盈亏比/PF/名义净额)。
"""

from __future__ import annotations

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                       # noqa: E402
from chanbook import engine as E                     # noqa: E402
from chanbook import report as R                     # noqa: E402
from chanbook import risk                            # noqa: E402
from chanbook.bi import Bi, BiBuilder                # noqa: E402
from chanbook.ckline import KLineMerger              # noqa: E402
from chanbook.fenxing import scan_fenxing_at         # noqa: E402
from chanbook.strategy import BookCfg, Strategy      # noqa: E402

N_PASS = 0
FAILS: list[str] = []


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


# ------------------------------------------------------------------ 1. ATR
def test_atr() -> None:
    # 构造已知 TR 序列: 收盘贴着最高价、区间恒为 1、且无跳空 ⇒ TR 恒为 1, ATR 恒为 1
    h = [10.0 + i for i in range(40)]
    l = [9.0 + i for i in range(40)]
    c = list(h)
    a = risk.atr(h, l, c, 14)
    ck(a is not None and abs(a - 1.0) < 1e-9, f"平稳序列 ATR 应为 1, 实际 {a}")
    ck(risk.atr(h[:10], l[:10], c[:10], 14) is None, "数据不足应返回 None")
    # 每根都向上跳空 5、区间恒为 1 ⇒ TR 恒为 5
    h2 = [10.0 + 5 * i for i in range(40)]
    l2 = [9.0 + 5 * i for i in range(40)]
    c2 = list(h2)
    a2 = risk.atr(h2, l2, c2, 14)
    ck(a2 is not None and abs(a2 - 5.0) < 1e-9,
       f"连续跳空 5 时 ATR 应为 5, 实际 {a2}")


def test_atr_at_window() -> None:
    kl = D.Kl(rows=[[i * 900000, 10 + i, 11 + i, 9 + i, 10.5 + i] for i in range(50)])
    a1 = risk.atr_at(kl, 49, 14, 300)
    a2 = risk.atr_at(kl, 20, 14, 300)
    ck(a1 is not None and a2 is not None, "窗口内应能算出 ATR")
    # atr_at 只用 [i-window+1, i] —— 第 20 根时不该"看见"后面的数据
    only = risk.atr(kl.h[0:21], kl.l[0:21], kl.c[0:21], 14)
    ck(abs(a2 - only) < 1e-12, "atr_at 不得使用 i 之后的数据")


# ------------------------------------------------------------------ 2. 风控
def test_plan_long() -> None:
    from chanbook.risk import plan
    # stop_buf="old" 复刻旧引擎: 有效止损 = 结构点下方 1×ATR(atr_mult 被 max 钳掉)
    p = plan("long", 100.0, 97.0, 1.0, "um", "B3", b1_wide=False)
    ck(p.ok, f"正常信号应通过: {p.why}")
    ck(abs(p.stop - 96.0) < 1e-9,
       f"旧口径止损应为 结构-1ATR=96.0(而非 结构-1.5ATR=95.5), 实际 {p.stop}")
    ck(abs(p.target - (100.0 + 2.0 * (100.0 - p.stop))) < 1e-9, "目标应为 RR2")
    # stop_buf="min" 才是原注释"至少保留 1×ATR 缓冲"的本意
    q = plan("long", 100.0, 97.0, 1.0, "um", "B3", b1_wide=False, stop_buf="min")
    ck(q.ok and abs(q.stop - 95.5) < 1e-9,
       f"min 口径止损应为 结构-1.5ATR=95.5, 实际 {q.stop}")
    ck(q.stop < p.stop, "min 口径的止损必须更远(缓冲更大)")
    # 止损距离过大 -> 拒单
    bad = plan("long", 100.0, 94.0, 1.0, "um", "B3", b1_wide=False)
    ck(not bad.ok and "止损距离过大" in bad.why, f"应被风控拒绝: {bad.why}")
    # 结构点位异常(止损不低于入场价) -> 拒单
    weird = plan("long", 100.0, 101.0, 1.0, "um", "B3", b1_wide=False)
    ck(not weird.ok and "结构点位异常" in weird.why, f"应拒绝异常结构: {weird.why}")


def test_plan_b1_wide() -> None:
    """一买特例: 同一结构下止损相同(被钳位), 但可接受的损失上限更宽 + 名义额减半。"""
    from chanbook.risk import plan
    a = plan("long", 100.0, 94.0, 1.0, "um", "B3", b1_wide=False)
    b = plan("long", 100.0, 94.0, 1.0, "um", "B1", b1_wide=True)
    ck(not a.ok, "顺势买点在这个深度应被 5% 上限拒掉")
    ck(b.ok, f"一买放宽到 8% 上限后应通过: {b.why}")
    ck(abs(b.max_loss_pct - 0.05 * 1.6) < 1e-12, f"一买上限应为 8%, 实际 {b.max_loss_pct}")
    a2 = plan("long", 100.0, 97.0, 1.0, "um", "B3", b1_wide=False)
    b2 = plan("long", 100.0, 97.0, 1.0, "um", "B1", b1_wide=True)
    ck(abs(b2.stop - a2.stop) < 1e-12, "止损位置应由结构决定, 一买不靠放宽止损距离")
    ck(b2.notional < a2.notional, f"一买名义额应减半: {b2.notional} vs {a2.notional}")


# ------------------------------------------------------------------ 3. 离开管理
def _pos(**kw) -> dict:
    d = dict(sym="X", side="long", kind="B3", entry=100.0, stop=98.0, stop0=98.0,
             target=104.0, highest=100.0, lowest=100.0, trail=False,
             entry_ts=0, trend="up", regime="bull", counter=0, confirm="",
             risk_pct=0.02, notional=15.0)
    d.update(kw)
    return d


def test_manage_stop() -> None:
    p = _pos()
    p2, t = E.manage(p, {"ts": 1, "high": 100.5, "low": 97.9, "close": 98.5}, 2.0)
    ck(p2 is None and t is not None and t.reason == "stop_loss" and t.exit == 98.0,
       f"应触发止损, 实际 {t}")


def test_manage_target() -> None:
    """止盈分支: 低点不得先碰到追踪线(否则追踪止损会先触发)。"""
    p = _pos()
    p2, t = E.manage(p, {"ts": 1, "high": 104.5, "low": 103.0, "close": 104.0}, 2.0)
    ck(p2 is None and t is not None and t.reason == "take_profit" and t.exit == 104.0,
       f"应触发止盈@104, 实际 {t}")


def test_manage_trail_beats_target() -> None:
    """同一根里追踪线已被打到、且最高价越过目标 ⇒ 按旧引擎口径先判止损。"""
    p = _pos()
    p2, t = E.manage(p, {"ts": 1, "high": 104.5, "low": 99.5, "close": 104.0}, 2.0)
    ck(t is not None and t.reason == "trail_stop",
       f"高点 104.5 越过目标但低点打到追踪线 100.5 ⇒ 应先判止损, 实际 {t}")


def test_manage_stop_priority() -> None:
    """同根同时触发止损与止盈时按止损优先(保守口径)。"""
    p = _pos()
    p2, t = E.manage(p, {"ts": 1, "high": 105.0, "low": 97.0, "close": 100.0}, 2.0)
    ck(t is not None and t.reason == "stop_loss", f"应止损优先, 实际 {t}")


def test_manage_trail() -> None:
    """浮盈达 1R 激活追踪; 之后 `risk` 用**当前 stop** ⇒ 追踪线逐轮收紧。

    这是旧引擎与实盘都有的行为, 刻意复刻(见 `engine.manage` docstring)——
    不复刻这一点持仓时间会明显偏长, A/B 就不可比了。
    """
    p = _pos()                       # entry 100, stop 98 ⇒ risk 2, 1R = 2
    p, t = E.manage(p, {"ts": 1, "high": 102.0, "low": 100.0, "close": 102.0}, 2.0)
    ck(t is None and p is not None and p["trail"], "浮盈达 1R 应激活追踪")
    ck(abs(p["stop"] - 98.0) < 1e-9,
       f"首次激活时追踪线 102-2×2=98 与原点位重合, 实际 {p['stop']}")
    p, t = E.manage(p, {"ts": 2, "high": 103.0, "low": 101.0, "close": 102.9}, 2.0)
    ck(t is None, "继续上涨不应平仓")
    ck(abs(p["stop"] - 99.0) < 1e-9, f"追踪线应上移到 103-2×2=99, 实际 {p['stop']}")
    lvl = p["stop"]
    p2, t = E.manage(p, {"ts": 3, "high": 103.0, "low": 98.0, "close": 102.0}, 2.0)
    ck(t is not None and t.reason == "trail_stop", f"应触发追踪止损, 实际 {t}")
    ck(t.exit >= lvl - 1e-9,
       f"同一次调用里 risk 会用当前 stop 再次收紧 ⇒ 成交价应不低于 {lvl}, "
       f"实际 {t.exit}")
    ck(t.pnl_pct > 0, "追踪止损离场应是盈利的")


# ------------------------------------------------------------------ 4. 结构破位
def _mkbis(pts) -> list:
    """pts: [(kind, price, ts)] -> 笔序列(kind='t'顶 'b'底)。"""
    bb = BiBuilder(4)
    from chanbook.fenxing import FX
    for k, (kind, px, ts) in enumerate(pts):
        bb.push(FX(kind=1 if kind == "t" else -1, idx=k * 5, high=px if kind == "t" else px - 1,
                   low=px + 1 if kind == "t" else px, ts=ts, extreme=px,
                   prominence=1.0, reversal=1.0, rng=1.0))
    return bb.bis


def test_struct_line() -> None:
    """结构线 = 入场之后冻结的向下笔的低点, 取其中**最高**的那个(抬高低点)。

    注意最后一笔是可延伸笔(provisional), 不参与 —— 引擎与 `bis[:-1]` 同约定。
    """
    # b100 → t110 → b105 → t115 → b108 → t120 → b112
    # 笔序列: bi0↑(100→110) bi1↓(110→105) bi2↑(105→115) bi3↓(115→108) bi4↑ ...
    bis = _mkbis([("b", 100.0, 0), ("t", 110.0, 1000), ("b", 105.0, 2000),
                  ("t", 115.0, 3000), ("b", 108.0, 4000), ("t", 120.0, 5000),
                  ("b", 112.0, 6000)])
    ck(bis[-1].direction < 0, "最后一笔应为向下笔(可延伸, 不计入结构线)")
    line = E.struct_line(bis, 500, -1)
    ck(line is not None and abs(line - 108.0) < 1e-9,
       f"结构线应取最高的抬高低点 108(bi1=105, bi3=108), 实际 {line}")
    line2 = E.struct_line(bis, 3000, -1)
    ck(line2 is not None and abs(line2 - 108.0) < 1e-9,
       f"入场在 3000 之后只应看到 bi3=108, 实际 {line2}")
    ck(E.struct_line(bis, 5000, -1) is None, "6000 之后没有冻结的向下笔 => None")


# ------------------------------------------------------------------ 5. 行情分桶
def test_regime_labels() -> None:
    ts, lab = E.btc_regime("um")
    ck(len(ts) == len(lab) and len(ts) > 10000, "行情分桶序列长度应与 1h K线一致")
    ck(set(lab) <= {"bull", "range", "bear"}, f"标签集合异常: {set(lab)}")
    ck("bull" in lab and "bear" in lab, "六年数据里应同时出现牛市与熊市桶")
    ck(all(l == "range" for l in lab[:59]), "前 60 根应为 range(数据不足)")


# ------------------------------------------------------------------ 6. 报告算术
class _T:
    def __init__(self, pnl, kind="B3", notional=15.0, ts=0, reason="stop_loss",
                 regime="bull", trend="up"):
        self.kind, self.notional, self.entry_ts = kind, notional, ts
        self.reason, self.regime, self.trend = reason, regime, trend
        self.pnl_pct, self.hold_min = pnl, 60
        self.sym, self.exit_ts, self.side = "X", ts + 3600000, "long"
        self.entry, self.exit, self.counter, self.confirm = 100.0, 100.0 + pnl * 100, 0, ""

    def to_dict(self):
        return dict(self.__dict__)


def test_report_math() -> None:
    tr = [_T(0.02), _T(-0.01), _T(0.03), _T(-0.01)]
    s = R.summarize(tr, "um")
    ck(s["n"] == 4, "笔数应为 4")
    ck(abs(s["exp_pct"] - 0.0075) < 1e-12, f"单笔期望应为 0.75%, 实际 {s['exp_pct']}")
    ck(abs(s["avg_win"] - 0.025) < 1e-12, f"均赢应为 2.5%, 实际 {s['avg_win']}")
    ck(abs(s["avg_loss"] + 0.01) < 1e-12, f"均亏应为 -1%, 实际 {s['avg_loss']}")
    ck(abs(s["rr"] - 2.5) < 1e-12, f"盈亏比应为 2.5, 实际 {s['rr']}")
    ck(abs(s["pf"] - 0.05 / 0.02) < 1e-12, f"PF 应为 2.5, 实际 {s['pf']}")
    ck(abs(s["win_rate"] - 0.5) < 1e-12, "胜率应为 50%")
    ck(abs(s["net_u"] - (0.0075 * 4 * 15 - 4 * 15 * 0.0005 * 2)) < 1e-9,
       f"名义净额计算错误: {s['net_u']}")


def test_report_group() -> None:
    tr = [_T(0.02, kind="B1"), _T(-0.01, kind="B1"), _T(0.03, kind="B3")]
    ks = R.by_kind(tr, "um")
    ck(len(ks) == 3, "分买点应输出三行")
    ck("B1" in ks[0] and "2笔" in ks[0], f"B1 行应含 2 笔: {ks[0]}")
    ck("无成交" in ks[1], f"B2 无成交应显式标注: {ks[1]}")
    ck(sum(1 for l in R.by_year(tr, "um")) == 1, "所有成交都在同一年")


# ------------------------------------------------------------------ 7. 引擎无未来函数
def _syn_kl(n=3000):
    """合成行情。

    用**确定性**的多尺度正弦叠加(不用随机数), 参数是实测定下来的: 必须能产出
    足够多的笔/线段/中枢, 且信号不至于全被 5% 止损上限拒掉 —— 否则"截断一致性"
    这类断言会因为样本太少而失去意义。当前参数: 笔≈190 / 中枢≈11 / 成交≈15。
    """
    a1, p1, a2, p2, dr, w = 0.025, 5.0, 0.02, 17.0, 0.0002, 0.002

    def f(x: float) -> float:
        return 100.0 * (1 + a1 * math.sin(x / p1) + a2 * math.sin(x / p2 + 1.1)
                        + dr * x)

    rows = []
    for i in range(n):
        o, c = f(i - 0.5), f(i)
        rows.append([i * 900000, o, max(o, c) * (1 + w),
                     min(o, c) * (1 - w), c, 1.0, 100.0])
    return D.Kl(rows=rows)


def _patch_data():
    """把 chanbook.data.load 换成合成数据提供者(按周期做真实聚合/切片)。

    ⚠ 必须同时清空 `engine._REG_CACHE`: 引擎在信号产生时会调用 `btc_regime()`,
    若不清缓存, 合成行情算出的行情分桶会被缓存下来, 污染后续用真实数据做的断言。
    """
    base = _syn_kl()
    orig = D.load
    E._REG_CACHE.clear()

    def fake(market, sym, tf):
        if tf in ("15m", "5m"):
            return base
        if tf == "1h":
            return D.resample(base, D.TF_MS["1h"])
        return D.resample(base, D.TF_MS[tf])

    D.load = fake
    return orig


def _unpatch_data(orig) -> None:
    D.load = orig
    E._REG_CACHE.clear()


def test_engine_no_lookahead() -> None:
    orig = _patch_data()
    try:
        cfg = BookCfg.legacy(market="um")
        full, dg = E.run_symbol("um", "SYN", cfg)
        ck(dg.err == "", f"合成数据不应报错: {dg.err}")
        ck(len(full) >= 5, f"合成行情应产生足够成交以便检验, 实际 {len(full)}")
        ck(any(t.entry_ts < 1000 * 900000 for t in full), "应有第 1000 根之前的成交")
        ck(any(t.entry_ts < 2000 * 900000 for t in full), "应有第 2000 根之前的成交")
        for cut in (1000, 2000):
            part, _ = E.run_symbol("um", "SYN", cfg, t_end=cut * 900000 - 1)
            a = [(t.entry_ts, t.kind, round(t.entry, 9)) for t in full
                 if t.entry_ts < cut * 900000]
            b = [(t.entry_ts, t.kind, round(t.entry, 9)) for t in part]
            ck(a[:len(b)] == b,
               f"截断到第 {cut} 根时入场不一致:\n  完整={a[:6]}\n  截断={b[:6]}")
    finally:
        _unpatch_data(orig)


def test_engine_diag() -> None:
    orig = _patch_data()
    try:
        cfg = BookCfg.legacy(market="um")
        tr, dg = E.run_symbol("um", "SYN", cfg)
        ck(dg.n_signals == len(tr),
           f"信号数({dg.n_signals})应等于候选池笔数({len(tr)})")
        ck(dg.n_bi > 50 and dg.n_zs > 2,
           f"应识别出足够多的笔与中枢: bi={dg.n_bi} zs={dg.n_zs}")
        ck(dg.trend_at_end != "", "应给出末态走势")
        ck(sum(dg.n_seen.values()) >= dg.n_signals,
           "结构命中数应不少于成交数(含被闸门拦掉的)")
        ck(all(t.reason and t.kind for t in tr), "每笔都应有买点类型与离场原因")
        ck(all(t.entry > 0 and t.exit > 0 for t in tr), "入场/离场价应为正")
        ck(all(abs(t.pnl_pct - (t.exit - t.entry) / t.entry) < 1e-12 for t in tr),
           "pnl_pct 应为 (exit-entry)/entry")
    finally:
        _unpatch_data(orig)


def test_engine_offline_parity() -> None:
    """只开 B3 时应只有 B3 成交; 关掉方向闸门后不得出现别的类型。"""
    orig = _patch_data()
    try:
        cfg = BookCfg.legacy(market="um", kinds=("B3",),
                             dir_gate={"B1": None, "B2": None, "B3": None})
        tr, dg = E.run_symbol("um", "SYN", cfg)
        ck(all(t.kind == "B3" for t in tr), "只开 B3 时应只有 B3 成交")
        ck(all(t.reason for t in tr), "每笔都应有离场原因")
    finally:
        _unpatch_data(orig)


def test_multi_symbol() -> None:
    """多标的入口的调度与诊断汇总(走串行分支 —— 合成数据无法跨进程传递,
    多进程分支由真实 A/B 运行覆盖)。"""
    orig = _patch_data()
    try:
        cfg = BookCfg.legacy(market="um")
        tr, dgs = E.run_all("um", cfg, symbols=["A", "B", "C"], jobs=1)
        ck(len(dgs) == 3, f"应返回 3 个诊断, 实际 {len(dgs)}")
        ck(all(d.sym in ("A", "B", "C") for d in dgs), "诊断的标的应正确")
        ck({d.sym for d in dgs} == {"A", "B", "C"}, "不应漏掉任何标的")
    finally:
        _unpatch_data(orig)


class _FakeBi:
    """鸭子类型的"笔"桩: `_b1/_b2/_b3` 只用到这几个字段。"""

    def __init__(self, i, s, e, d, low, high):
        self.i, self.start_ts, self.end_ts, self.direction = i, s, e, d
        self.low, self.high = low, high
        self.amp = abs(high - low)


class _FakeMacd:
    """让"越晚的段力度越小"恒成立 —— 于是背驰判据必然是 ok。

    只需实现 `verdict()` 用到的四个成员: n / n_bars / summary / hist_peak。
    """

    n = 100000

    @staticmethod
    def n_bars(a0, a1):
        return 5

    @staticmethod
    def _scale(a0):
        return 1000.0 / (1.0 + a0)          # 起点越晚 => 力度越小

    def summary(self, a0, a1, sign):
        k = self._scale(a0)
        return {"area": k, "per": k, "peak": k}

    def hist_peak(self, a0, a1, sign):
        return self._scale(a0)


def _b2_scenario():
    """构造一个**确定性**的一买 -> 回调 -> 二买序列(不依赖任何真实数据)。

    两条硬约束(写错就退化成"空等于空"的假绿):
      * 笔必须**方向交替**(否则 `verdict` 会判"两段区间重叠, 不可比较");
      * 无中枢路径要求 `last >= 4` 且 `bis[last-2]` 是向下笔 ⇒ 一买只能落在下标 5。

    bis[3] 与 bis[5] 是两段同向下跌, bis[5] 创新低(80 < 90)且力度更小 ⇒ 一买;
    bis[7] 是其后第一次回调且低点(85) 未破一买低点 ⇒ 二买。
    zss 传空 ⇒ `_b1` 只走"无中枢语境的 MACD 底背驰"路径, 便于独立验证。
    """
    out = []
    rows = [(+1, 100.0, 110.0), (-1, 95.0, 110.0),
            (+1, 95.0, 105.0), (-1, 90.0, 105.0),      # bis[3] 参照下跌段
            (+1, 90.0, 100.0), (-1, 80.0, 98.0),       # bis[5] 一买(创新低+力度更小)
            (+1, 80.0, 92.0), (-1, 85.0, 92.0)]        # bis[7] 二买(不破 80)
    for i, (d, lo, hi) in enumerate(rows):
        out.append(_FakeBi(i, 100 * i, 100 * (i + 1), d, lo, hi))
    return out


def test_scanner_enable_keeps_state_machine() -> None:
    """`enable` 必须是"输出过滤器", 绝不能把 B1/B2 共享的 `last_b1` 状态机掐断。

    事故背景(2026-09-27): 老实现把 `_b1` 整段放在 `if B1 in self.enable` 里,
    于是"只做二买"(kinds=(B2,)) 永远拿不到 `last_b1` ⇒ 全量 534 个标的 **0 笔成交**。
    这条断言就是那次事故的应有防线: 它必须在老实现上失败、在新实现上通过。
    """
    from chanbook.points import B1, B2, B3, PointScanner

    bis, macd = _b2_scenario(), _FakeMacd()
    full = PointScanner(enable=(B1, B2, B3))
    only2 = PointScanner(enable=(B2,))
    seq_full, seq_only2 = [], []
    for nf in range(2, 9):
        seq_full.extend(full.scan(bis, [], macd, nf))
        seq_only2.extend(only2.scan(bis, [], macd, nf))

    kinds = [(p.kind, round(float(p.price), 6)) for p in seq_full]
    ck(("B1", 80.0) in kinds, f"桩数据应定出一买, 实际 {kinds}")
    ck(("B2", 85.0) in kinds, f"桩数据应定出二买, 实际 {kinds}")
    ck(len(seq_only2) > 0,
       "只开 B2 仍须产出二买 —— enable 是输出过滤器, 不是关闭检测")
    ck([(p.kind, round(float(p.price), 6)) for p in seq_only2]
       == [x for x in kinds if x[0] == B2],
       "只开 B2 的点序列必须逐字等于全开里的 B2 子序列")

    # 单开 B1 / 单开 B3 也必须与全开下的对应子序列一致
    for kind in (B1, B3):
        s = PointScanner(enable=(kind,))
        got = []
        for nf in range(2, 9):
            got.extend(s.scan(bis, [], macd, nf))
        ck([(p.kind, round(float(p.price), 6)) for p in got]
           == [x for x in kinds if x[0] == kind],
           f"只开 {kind} 的点序列必须不变")


def test_report_notional_weighting() -> None:
    """名义额按买点分档(一买小仓 7.5U / 二三买 15U) ⇒ 净期望必须以
    `Σ名义额` 为基, 不能用"第一笔的名义额"。

    事故背景(2026-09-27): `all` 变体第一笔恰好是一买(名义 7.5U), 于是净期望被
    报成 -0.175%, 真值 -0.125% —— 成本被放大了 40%, 足以误导判决。
    """
    ts = 1_700_000_000_000
    def mk(pnl, nm):
        return E.Trade(sym="X", entry_ts=ts, exit_ts=ts + 1, side="long", kind="B1",
                       trend="up", regime="bull", counter=False, confirm="",
                       entry=1.0, exit=1.0 + pnl, stop0=0.9, target=1.2,
                       pnl_pct=pnl, risk_pct=0.05, notional=nm, reason="take_profit",
                       hold_min=10.0)

    tr = [mk(+0.01, 7.5), mk(-0.02, 15.0), mk(+0.03, 15.0)]
    s = R.summarize(tr, "um")
    tot = 7.5 + 15.0 + 15.0
    ck(abs(s["notional"] - tot / 3) < 1e-9,
       f"名义额应为平均值 {tot / 3}, 实际 {s['notional']}")
    ck(abs(s["net_exp_pct"] - (s["gross_u"] / tot - 2 * R.FEE["um"])) < 1e-12,
       "净期望必须等于 名义额加权毛期望 - 双边费")
    ck(abs(s["w_exp_pct"] - s["gross_u"] / tot) < 1e-12, "加权毛期望口径错误")
    # 单一名义额时, 加权与非加权必须重合(内部一致性)
    s2 = R.summarize([mk(+0.01, 15.0), mk(-0.02, 15.0)], "um")
    ck(abs(s2["w_exp_pct"] - s2["exp_pct"]) < 1e-12,
       "名义额一致时加权与非加权毛期望必须相等")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:                     # noqa: BLE001
            FAILS.append(f"{t.__name__} 抛异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n== 引擎/风控/报告 自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
