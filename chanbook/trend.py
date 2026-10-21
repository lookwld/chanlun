"""走势类型 —— 盘整 / 趋势 / 中阴, 以及走势分解与"走势必完美"。

原文定义:
  **盘整** = 只包含一个走势中枢的走势。
  **趋势** = 至少包含两个**同向且不重叠**的走势中枢的走势:
      上涨趋势: 后中枢的 ZD > 前中枢的 ZG (第二个中枢整体更高, 中间有缺口)
      下跌趋势: 后中枢的 ZG < 前中枢的 ZD
      中枢区间有重叠 => 不构成趋势(属于更大级别的盘整 / 中枢扩展)。
  **走势必完美** = 任何走势最终都会完成; 反过来说, 一个"走势"必须至少包含一个中枢,
      没有中枢的线段序列只是"未完成的走势", 不能作为操作依据。
  **中阴阶段** = 前一个中枢已被离开、但新的同级别中枢尚未确立的过渡区间。
      此阶段方向不明, 原文建议"观望或按次级别操作", 本模块把它显式标记出来,
      供策略层决定是否降级仓位。
  **同级别分解** = 把走势序列拆成一段段"盘整/趋势", 使得每一段都是同级别的完整走势。

流式语义: `update(zss, cur_zs)` 接受已结束的中枢列表与进行中的中枢快照,
返回当前的走势判断 —— 全程只依赖已发生的中枢, 无未来函数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["Trend", "TrendBuilder", "decompose", "TrendState"]

UP, DOWN, RANGE, PENDING = "up", "down", "range", "pending"


@dataclass
class Trend:
    """一段走势(盘整或趋势)。"""

    i: int
    kind: str                     # up / down / range
    z0: int                       # 起始中枢下标(在 zss 中)
    z1: int                       # 结束中枢下标(含)
    zg: float                     # 走势上界
    zd: float                     # 走势下界
    start_ts: int
    end_ts: int

    @property
    def n_zs(self) -> int:
        return self.z1 - self.z0 + 1

    def label(self) -> str:
        return {"up": "上涨趋势", "down": "下跌趋势", "range": "盘整"}[self.kind]

    def __repr__(self) -> str:      # pragma: no cover
        return (f"Trend#{self.i} {self.kind} ZS[{self.z0}-{self.z1}] "
                f"[{self.zd:.6g},{self.zg:.6g}]")


@dataclass
class TrendState:
    """当前走势判断结果。"""

    kind: str = PENDING           # up / down / range / pending(中阴)
    n_zs: int = 0
    zg: float = 0.0
    zd: float = 0.0
    zhongyin: bool = False        # 是否处于中阴阶段
    incomplete: bool = True       # 是否"尚未构成完整走势"(缺中枢)
    why: str = ""

    def label(self) -> str:
        if self.zhongyin:
            return "中阴阶段"
        if self.incomplete:
            return "走势未完成"
        return {"up": "上涨趋势", "down": "下跌趋势", "range": "盘整"}.get(self.kind, self.kind)


def _link(a, b) -> str:
    """两个相邻中枢的关系: up / down / overlap。"""
    if b.zd > a.zg:
        return UP
    if b.zg < a.zd:
        return DOWN
    return "overlap"


def decompose(zss: list) -> list[Trend]:
    """同级别分解: 把中枢序列拆成盘整/趋势段。

    规则: 相邻中枢的关系决定这一段走势的性质 ——
      连续同向不重叠 => 上涨/下跌趋势(≥2 个中枢)
      区间重叠        => 盘整(中枢扩展, 归为同段)
      孤立中枢        => 盘整(1 个中枢)
    实现即对"关系序列"做游程合并。
    """
    out: list[Trend] = []
    if not zss:
        return out
    s = 0
    cur: str | None = None
    for k in range(1, len(zss)):
        lk = _link(zss[k - 1], zss[k])
        eff = lk if lk in (UP, DOWN) else "range"
        if cur is None:
            cur = eff
        elif eff != cur:
            out.append(_mk(len(out), cur, zss, s, k - 1))
            s = k
            cur = eff
    out.append(_mk(len(out), cur or "range", zss, s, len(zss) - 1))
    return out


def _mk(i: int, kind: str, zss: list, z0: int, z1: int) -> Trend:
    zg = max(z.zg for z in zss[z0:z1 + 1])
    zd = min(z.zd for z in zss[z0:z1 + 1])
    return Trend(i, kind, z0, z1, zg, zd, zss[z0].start_ts, zss[z1].end_ts)


class TrendBuilder:
    """流式走势判定。"""

    def __init__(self) -> None:
        self._last_z1 = -1
        self.state = TrendState()

    def update(self, zss: list, cur_zs=None, leg_high: float | None = None,
               leg_low: float | None = None) -> TrendState:
        """zss 为已结束中枢, cur_zs 为进行中的中枢快照(可 None)。

        leg_high/leg_low: 最后一个中枢之后、当前价格所处区间的极值
        (用于判断"是否已向上/向下离开最后一个中枢")。
        """
        if not zss:
            self.state = TrendState(kind=PENDING, incomplete=True,
                                    why="尚无任何中枢")
            return self.state
        # 只回看最近 4 个中枢: 更早的与当下方向判断无关, 且恒定内存
        win = zss[-4:]
        kinds = [_link(win[k - 1], win[k]) for k in range(1, len(win))]
        kind = "range"
        n = 1
        for lk in reversed(kinds):
            if lk in (UP, DOWN) and (kind == "range" or kind == lk):
                kind = lk
                n += 1
            else:
                break
        last = win[-1]
        zg = max(z.zg for z in win[-n:])
        zd = min(z.zd for z in win[-n:])
        # 中阴: 已离开最后中枢, 但新中枢尚未成立
        zy = False
        if leg_high is not None and leg_high > last.zg + (last.zg - last.zd) * 0.1:
            zy = True
        if leg_low is not None and leg_low < last.zd - (last.zg - last.zd) * 0.1:
            zy = True
        if cur_zs is not None and cur_zs.overlaps(last):
            zy = False
        st = TrendState(kind=(kind if kind != "range" or n == 1 else "range"),
                        n_zs=n, zg=zg, zd=zd, zhongyin=zy, incomplete=False)
        st.why = (f"最近{n}个中枢" +
                  (f"同向不重叠 => {('上涨' if kind == UP else '下跌')}趋势"
                   if kind in (UP, DOWN) and n >= 2 else "区间重叠 => 盘整"))
        self.state = st
        return st
