"""止损 / 目标 / 仓位 —— 交易执行层的风控口径。

口径必须与旧引擎(实盘 `config.yaml`)对齐, A/B 才有意义:

  | 项 | 合约 um | 现货 spot |
  |---|---|---|
  | ATR 倍数(配置值) | 1.5 | 2.0 |
  | 单笔最大损失占比 | 5% | 6% |
  | 盈亏比 RR | 2.0 | 2.0 |
  | 追踪倍数(相对当前 risk) | 2.0 | 3.0 |
  | 名义额 | 15U(5U 保证金 × 3x) | 5U |

ATR: Wilder(14), 输入窗口 = **操作级别最近 300 根已收盘K线**(旧引擎 `ek =
e.bars(i-299, i+1)` 同窗口), 保证同口径。

⚠ **旧引擎里 `atr_mult` 实际是失效的**(对抄口径时核对发现, 这里显式声明并保留行为):

    stop = struct - atr_mult*ATR
    if stop < entry*(1-max_loss_pct): 放弃
    stop = max(stop, struct - ATR)        # ← 这一行把 atr_mult 钳掉了

  因为 atr_mult >= 1 时恒有 `struct - atr_mult*ATR <= struct - ATR`, 取 max 之后
  有效止损距离永远是 **结构点下方 1×ATR**; 配置的 1.5/2.0 只在"拒单阈值"里起作用。
  工程上保留两套: `stop_buf="old"` 复刻旧行为(A/B 归因要求), `"min"` 是原注释
  "至少保留 1×ATR 缓冲"的本意(止损落在结构点下方 atr_mult×ATR)。

一买的风控特例(DESIGN §4): B1 天然逆势, 按"小仓 + 宽止损"处理 ——
`b1_wide` 把**可接受的单笔最大损失上限**放宽 1.6 倍, 同时**名义额减半**,
使单笔风险敞口与顺势买点同一量级, 而不是靠"偷偷放宽止损距离"制造隐藏风险。
这不是调参寻优, 而是原文对一买的一贯要求(小仓、宽止损、只做一次、不补仓);
具体倍数在 A/B 里作为独立维度检验。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CALIB", "Plan", "atr", "atr_at", "plan", "trail_mult_of",
           "B1_WIDE", "STOP_BUF"]

# market -> 基准口径
CALIB = {
    "um": {"atr_mult": 1.5, "max_loss_pct": 0.05, "trail_mult": 2.0,
           "rr": 2.0, "notional": 15.0, "fee": 0.0005},
    "spot": {"atr_mult": 2.0, "max_loss_pct": 0.06, "trail_mult": 3.0,
             "rr": 2.0, "notional": 5.0, "fee": 0.001},
}

# 一买特例: 放宽可接受的单笔最大损失上限, 同时缩小名义额
B1_WIDE = {"loss_mult": 1.6, "notional_scale": 0.5}

STOP_BUF = ("old", "min")     # 见模块 docstring


def atr(highs, lows, closes, n: int = 14) -> float | None:
    """Wilder 平滑 ATR(与实盘 `app.indicators.atr` 同式, 便于同口径比较)。"""
    m = len(closes)
    if m < n + 1:
        return None
    trs = []
    prev = closes[0]
    for i in range(1, m):
        hi, lo = highs[i], lows[i]
        tr = hi - lo
        d1 = hi - prev
        if d1 > tr:
            tr = d1
        d2 = prev - lo
        if d2 > tr:
            tr = d2
        trs.append(tr)
        prev = closes[i]
    a = sum(trs[:n]) / n
    for tr in trs[n:]:
        a = (a * (n - 1) + tr) / n
    return a


def atr_at(kl, i: int, n: int = 14, win: int = 300) -> float | None:
    """第 i 根(含)为止、窗口 win 根的 Wilder ATR。只用到 i 及之前的数据。"""
    lo = max(0, i - win + 1)
    return atr(kl.h[lo:i + 1], kl.l[lo:i + 1], kl.c[lo:i + 1], n)


def trail_mult_of(market: str) -> float:
    return float(CALIB[market]["trail_mult"])


@dataclass
class Plan:
    ok: bool
    side: str
    entry: float
    stop: float
    target: float
    atr: float
    rr: float
    risk_pct: float
    notional: float
    max_loss_pct: float = 0.0
    why: str = ""


def plan(side: str, entry: float, struct_level: float | None, a: float,
         market: str = "um", kind: str = "B3", *,
         atr_mult: float | None = None, max_loss_pct: float | None = None,
         rr: float | None = None, notional: float | None = None,
         b1_wide: bool = True, stop_buf: str = "old",
         nm_scale: float = 1.0) -> Plan:
    """由结构点位构造止损/目标/仓位。

    struct_level: 结构止损基准(做多=下跌段落低点/中枢下沿; 做空镜像)。
    止损距离超出可接受上限(超出即放弃该信号, 与旧引擎 `evaluate` 一致)。

    nm_scale: 名义额倍数(2026-09-29 新增)。由 `filters.pos_scale`(仓位分层)与
    `filters.ma_size_scale`(仓位型均线)相乘得到; 默认 1.0 ⇒ 与改动前逐位一致。
    ⚠ 它**只缩名义额, 不动止损/目标** —— 止损距离由结构点位与 ATR 决定, 与仓位无关。
    这样"仓位分层"改的是暴露而不是风险几何, 才不会把入场判据与风控口径搅在一起。
    """
    c = CALIB[market]
    am = float(c["atr_mult"] if atr_mult is None else atr_mult)
    ml = float(c["max_loss_pct"] if max_loss_pct is None else max_loss_pct)
    r = float(c["rr"] if rr is None else rr)
    nm = float(c["notional"] if notional is None else notional) * float(nm_scale)
    if a is None or a <= 0:
        return Plan(False, side, entry, 0.0, 0.0, 0.0, r, 0.0, 0.0, ml, "ATR 不可用")
    if stop_buf not in STOP_BUF:
        raise ValueError(f"未知 stop_buf: {stop_buf}")
    d = 1 if side == "long" else -1
    if struct_level is None or struct_level <= 0:
        struct_level = entry - d * am * a
    ml_cap = ml
    if kind == "B1" and b1_wide:
        ml_cap = ml * float(B1_WIDE["loss_mult"])
        nm *= float(B1_WIDE["notional_scale"])

    if d == 1:
        stop = float(struct_level) - am * a
        buf = float(struct_level) - a
        stop = min(stop, buf) if stop_buf == "min" else max(stop, buf)
        if stop >= entry:
            return Plan(False, side, entry, stop, 0.0, a, r, 0.0, 0.0, ml_cap,
                        "止损不低于入场价(结构点位异常)")
        if stop < entry * (1.0 - ml_cap):
            return Plan(False, side, entry, stop, 0.0, a, r, 0.0, 0.0, ml_cap,
                        f"止损距离过大({(entry - stop) / entry:.2%} "
                        f"> 上限{ml_cap:.1%})")
        target = entry + r * (entry - stop)
    else:
        stop = float(struct_level) + am * a
        buf = float(struct_level) + a
        stop = max(stop, buf) if stop_buf == "min" else min(stop, buf)
        if stop <= entry:
            return Plan(False, side, entry, stop, 0.0, a, r, 0.0, 0.0, ml_cap,
                        "止损不高于入场价(结构点位异常)")
        if stop > entry * (1.0 + ml_cap):
            return Plan(False, side, entry, stop, 0.0, a, r, 0.0, 0.0, ml_cap,
                        f"止损距离过大({(stop - entry) / entry:.2%} "
                        f"> 上限{ml_cap:.1%})")
        target = entry - r * (stop - entry)
    risk_pct = abs(entry - stop) / entry
    return Plan(True, side, entry, stop, target, a, r, risk_pct, nm, ml_cap, "ok")
