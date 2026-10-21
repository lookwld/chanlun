"""分型 —— 合并K线上的三根结构。

原文定义:
  顶分型 = 中间K线的高点最高 且 低点最高 (中间那根"凸"出来)
  底分型 = 中间K线的低点最低 且 高点最低 (中间那根"凹"进去)

要点补充(全书体系):
  1. 分型必须在**包含处理之后**的序列上判定, 否则会被包含关系污染
     —— 包含关系会让同一价位同时出现"顶"和"底"的矛盾结构。
  2. 分型**需要右侧至少一根合并K线才能确认** —— 这是天然的滞后, 也是所有
     缠论实现无法回避的"确认成本"。本模块以 `idx <= len(cks) - 2` 作为可判边界,
     绝不使用尚未收盘的K线。
  3. 分型的**力度**决定"能否成笔"与"能否被破坏", 用两个正交量刻画:
       prominence 突出度 = (中根极值 - 两侧同侧极值) / 中根振幅     —— 顶端/底端"凸"得多厉害
       reversal   回撤深度 = (中根极值 - 第3根反向极值) / 中根振幅  —— 第3根反向走了多远
     强度 = prominence + reversal。注意: 任何合法顶分型都有 c.low < b.low
     (这是顶分型定义的一部分), 所以"第3根是否反破第2根"是恒真的判据, 无区分度
     —— 必须用**深度**而不是**有无**来度量, 这是本模块与直觉写法的一个关键差别。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .ckline import CK

__all__ = ["FX", "TOP", "BOTTOM", "find_fenxing", "scan_fenxing_at"]

TOP = 1
BOTTOM = -1


@dataclass
class FX:
    """分型。"""

    kind: int            # TOP=1 / BOTTOM=-1
    idx: int             # 中间合并K线在 cks 中的下标
    high: float
    low: float
    ts: int              # 中间合并K线的起始时间
    extreme: float       # 顶分型=high, 底分型=low
    prominence: float = 0.0     # 突出度 >= 0
    reversal: float = 0.0       # 第3根反向回撤深度 >= 0
    rng: float = 0.0            # 中根振幅(供外部归一化用)

    @property
    def strength(self) -> float:
        return round(self.prominence + self.reversal, 6)

    @property
    def is_top(self) -> bool:
        return self.kind == TOP

    @property
    def confirmed_idx(self) -> int:
        """确认该分型所需的最小 cks 下标(含)。"""
        return self.idx + 1

    def label(self) -> str:
        return "顶分型" if self.kind == TOP else "底分型"

    def __repr__(self) -> str:      # pragma: no cover
        return f"FX({'顶' if self.kind == TOP else '底'}@{self.idx} {self.extreme:.6g})"


def _metrics(a: CK, b: CK, c: CK, kind: int) -> tuple[float, float, float]:
    """返回 (prominence, reversal, rng)。"""
    rng = max(b.high - b.low, 1e-12)
    if kind == TOP:
        prom = max(b.high - max(a.high, c.high), 0.0) / rng
        rev = max(b.low - c.low, 0.0) / rng
    else:
        prom = max(min(a.low, c.low) - b.low, 0.0) / rng
        rev = max(c.high - b.high, 0.0) / rng
    return round(prom, 6), round(rev, 6), rng


def scan_fenxing_at(cks: list[CK], j: int) -> FX | None:
    """判定以 cks[j] 为中间K线的分型(需 j 前后各至少 1 根)。"""
    if j < 1 or j + 1 >= len(cks):
        return None
    a, b, c = cks[j - 1], cks[j], cks[j + 1]
    if b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low:
        p, r, g = _metrics(a, b, c, TOP)
        return FX(TOP, j, b.high, b.low, b.ts0, b.high, p, r, g)
    if b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high:
        p, r, g = _metrics(a, b, c, BOTTOM)
        return FX(BOTTOM, j, b.high, b.low, b.ts0, b.low, p, r, g)
    return None


def find_fenxing(cks: list[CK], start: int = 1) -> list[FX]:
    """扫描 [start, len-2] 上的全部分型(离线用)。

    同一根合并K线不可能同时是顶分型和底分型; 但**相邻同类**分型很常见
    (连续同向斜率时), 由笔的构造器负责去重取更极端者。
    """
    out: list[FX] = []
    for j in range(max(1, start), len(cks) - 1):
        f = scan_fenxing_at(cks, j)
        if f is not None:
            out.append(f)
    return out
