"""背驰判定 —— 全书体系里"力度比较"的量化落地。

原文思想: 无论趋势还是盘整, 同向的两段走势之间可以比较"力度";
后一段虽然创出了新的极值(更高或更低), 但力度明显小于前一段 => **背驰**。
背驰是第一类买卖点的唯一依据, 也是"该停手"的判据。

判据(下跌方向为例, 上涨镜像):
    ① 面积衰减:   Σ|HIST|_后 < Σ|HIST|_前            (最核心)
    ② 单位力度衰减: 面积/根数_后 < 面积/根数_前        (排除"后段更长"的机械放大)
    ③ DIFF 峰值衰减: |DIFF|max_后 < |DIFF|max_前      (黄白线未能回拉到同样深度)
    ④ 柱体峰值衰减: |HIST|max_后 < |HIST|max_前

判决规则(可配):
    宽松(默认): ① 且 ③ 成立, 再加 ② 或 ④ 之一 => 背驰成立
    严格(--strict): 四项全部成立
另外记录 `new_extreme`: 后段是否真的创出了新极值 —— 这是"背驰"与"盘整内
普通回落"的区别, 策略层可用它区分趋势背驰与盘整背驰。

无未来函数: 所有比较只用到"后段已经走完"的数据; 调用方必须保证后段已确认
(即其右端是已冻结的走势端点)。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .macd import MACD

__all__ = ["BeichiResult", "verdict", "leg_beichi", "trend_beichi"]

TOL = 0.02          # 衰减容差: 比值需低于 1-TOL 才算"明显衰减"


@dataclass
class BeichiResult:
    ok: bool = False
    score: int = 0
    new_extreme: bool = False
    strict: bool = False
    a: dict = field(default_factory=dict)
    c: dict = field(default_factory=dict)
    ratios: dict = field(default_factory=dict)
    why: str = ""

    def short(self) -> str:
        if not self.ok:
            return "无背驰"
        return ("趋势背驰" if self.new_extreme else "盘整背驰") + f"(力度{self.score}/4)"


def verdict(macd: MACD, a0: int, a1: int, c0: int, c1: int, direction: int,
            px_a: float | None = None, px_c: float | None = None,
            tol: float = TOL, strict: bool = False,
            norm_a: float = 1.0, norm_c: float = 1.0) -> BeichiResult:
    """比较两段走势 [a0,a1] 与 [c0,c1] 的力度。

    direction: +1 = 上涨段比较高点, -1 = 下跌段比较低点。
    norm_a/norm_c: 两段的**价格尺度**归一因子(通常取各段价格中值)。

    ⚠ 为什么必须归一化: MACD 是价格之差, 其量纲就是价格。下跌趋势中后一段的价格
    水平天然更低, 于是 |HIST| 面积、|DIFF| 峰值都会系统性变小 —— 不归一化的话,
    "价格越低越像背驰"这种伪结论会淹没真实信号。除以各段自己的价格尺度之后,
    四个判据都变成无量纲比值, 不同价位的两段才真正可比。
    """
    sign = 1 if direction > 0 else -1
    r = BeichiResult(strict=strict)
    na = macd.n_bars(a0, a1)
    nc = macd.n_bars(c0, c1)
    if na < 2 or nc < 2 or macd.n < 30:
        r.why = "MACD 数据不足"
        return r
    if c0 <= a1:
        r.why = "两段区间重叠, 不可比较"
        return r

    A = macd.summary(a0, a1, sign)
    C = macd.summary(c0, c1, sign)
    A["hist_peak"] = macd.hist_peak(a0, a1, sign)
    C["hist_peak"] = macd.hist_peak(c0, c1, sign)
    sa = norm_a if norm_a > 0 else 1.0
    sc = norm_c if norm_c > 0 else 1.0
    for k in ("area", "per", "peak", "hist_peak"):
        A[k] = A[k] / sa
        C[k] = C[k] / sc
    r.a, r.c = A, C

    def ratio(k: str) -> float:
        va, vc = A[k], C[k]
        if va <= 1e-15:
            return 1e9
        return vc / va

    ra, rp, rk, rh = (ratio("area"), ratio("per"), ratio("peak"), ratio("hist_peak"))
    r.ratios = {"area": round(ra, 4), "per": round(rp, 4),
                "peak": round(rk, 4), "hist_peak": round(rh, 4),
                "bars": [na, nc]}
    c1v = ra < (1 - tol)        # ①
    c2v = rp < (1 - tol)        # ②
    c3v = rk < (1 - tol)        # ③
    c4v = rh < (1 - tol)        # ④
    r.score = sum([c1v, c2v, c3v, c4v])

    if px_a is not None and px_c is not None:
        r.new_extreme = (px_c < px_a * (1 - 1e-9)) if direction < 0 else \
                        (px_c > px_a * (1 + 1e-9))
    if strict:
        r.ok = all([c1v, c2v, c3v, c4v])
    else:
        r.ok = bool(c1v and c3v and (c2v or c4v))

    if r.ok:
        r.why = (f"力度衰减: 面积×{ra:.2f} 单位×{rp:.2f} 峰值×{rk:.2f} 柱峰×{rh:.2f}"
                 f" | 段长 {na}→{nc} 根"
                 + (" | 创出新极值(趋势背驰)" if r.new_extreme else " | 未创新极值(盘整背驰)"))
    else:
        miss = []
        if not c1v:
            miss.append(f"面积未衰减(×{ra:.2f})")
        if not c3v:
            miss.append(f"DIFF峰值未衰减(×{rk:.2f})")
        if not (c2v or c4v):
            miss.append(f"单位/柱峰均未衰减(×{rp:.2f}, ×{rh:.2f})")
        r.why = "不成立: " + ", ".join(miss)
    return r


def leg_beichi(macd: MACD, leg_a, leg_c, direction: int,
               tol: float = TOL, strict: bool = False) -> BeichiResult:
    """两段"走势对象"(笔/线段)之间的背驰判定。

    极值取法: 下跌方向比低点, 上涨方向比高点 —— 与方向一致的那一端。
    归一因子取各段价格区间中值, 消除"价位不同导致 MACD 量级不同"的系统偏差。
    """
    if direction < 0:
        px_a, px_c = float(leg_a.low), float(leg_c.low)
    else:
        px_a, px_c = float(leg_a.high), float(leg_c.high)
    na = (float(leg_a.high) + float(leg_a.low)) / 2.0
    nc = (float(leg_c.high) + float(leg_c.low)) / 2.0
    return verdict(macd, int(leg_a.start_ts), int(leg_a.end_ts),
                   int(leg_c.start_ts), int(leg_c.end_ts), direction,
                   px_a=px_a, px_c=px_c, tol=tol, strict=strict,
                   norm_a=na, norm_c=nc)


def trend_beichi(macd: MACD, zs, legs: list, direction: int,
                 tol: float = TOL, strict: bool = False) -> BeichiResult:
    """趋势背驰: 比较"进入中枢的段"与"离开中枢的段"。

    zs 需带 enter_i / leave_i(见 zhongshu.ZS)。缺任一段时返回 ok=False。
    """
    ent, lv = int(zs.enter_i), int(zs.leave_i)
    if ent < 0 or lv < 0 or ent >= len(legs) or lv >= len(legs):
        return BeichiResult(ok=False, why="缺少进入段或离开段")
    return leg_beichi(macd, legs[ent], legs[lv], direction, tol, strict)
