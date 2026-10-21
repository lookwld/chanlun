"""笔 —— 连接相邻顶底分型的"最小走势单元"。

原文定义(新笔口径, 2007 年后的标准):
  1. 一笔的两端必须是**方向相反**的分型(顶→底 或 底→顶);
  2. 两个分型之间(不含分型自身占据的K线)**至少要有 1 根独立合并K线**;
     等价于两个分型中间K线的下标差 >= 4;
  3. 同一方向上连续出现的同类分型, 只有**更极端**的那个才有意义
     (顶取更高、底取更低) —— 这构成"笔的延伸"。

流式语义(关键, 决定有无未来函数):
  - `bis[k] = Bi(seq[k], seq[k+1])`。
  - 任何时候只有 `seq[-1]` 会被"同类更极端"替换, 因此
    **`bis[-1]` 是可延伸的(provisional), `bis[:-1]` 才是已冻结的**。
  - 消费者要判断"已完成的笔", 必须用 `final_bis` / `n_final`, 不得直接读 `bis[-1]`,
    否则会用到尚未定型的笔端点 —— 这就是旧内核每根 bar 重算全局结构所回避、
    但代价是 O(n²) 的那个问题。
"""

from __future__ import annotations

from dataclasses import dataclass

from .fenxing import BOTTOM, TOP, FX

__all__ = ["Bi", "BiBuilder", "build_bi"]

MIN_GAP = 4          # 分型中心下标差下限: 至少 1 根独立合并K线


@dataclass
class Bi:
    """一笔。"""

    i: int               # 序号
    start: FX
    end: FX

    @property
    def direction(self) -> int:
        return 1 if self.end.kind == TOP else -1

    @property
    def start_price(self) -> float:
        return self.start.extreme

    @property
    def end_price(self) -> float:
        return self.end.extreme

    @property
    def high(self) -> float:
        return max(self.start.extreme, self.end.extreme)

    @property
    def low(self) -> float:
        return min(self.start.extreme, self.end.extreme)

    @property
    def i0(self) -> int:
        """起始合并K线下标(分型左侧那根起算, 覆盖分型整体)。"""
        return max(0, self.start.idx - 1)

    @property
    def i1(self) -> int:
        return self.end.idx + 1

    @property
    def start_ts(self) -> int:
        return self.start.ts

    @property
    def end_ts(self) -> int:
        return self.end.ts

    @property
    def n_bars(self) -> int:
        """跨越的合并K线根数。"""
        return max(1, self.end.idx - self.start.idx + 1)

    @property
    def amp(self) -> float:
        return abs(self.end_price - self.start_price)

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        d = "↑" if self.direction > 0 else "↓"
        return (f"Bi#{self.i}{d} {self.start.extreme:.6g}→{self.end.extreme:.6g} "
                f"[ck {self.start.idx}→{self.end.idx}]")


def _more_extreme(new: FX, old: FX) -> bool:
    return (new.extreme >= old.extreme) if new.kind == TOP else (new.extreme <= old.extreme)


class BiBuilder:
    """流式笔构造器。"""

    __slots__ = ("min_gap", "seq", "bis", "n_dropped")

    def __init__(self, min_gap: int = MIN_GAP) -> None:
        self.min_gap = int(min_gap)
        self.seq: list[FX] = []      # 净化后的顶底交替分型序列
        self.bis: list[Bi] = []
        self.n_dropped = 0           # 因间隔不足被丢弃的分型数

    # ------------------------------------------------------------------
    def push(self, fx: FX) -> Bi | None:
        """喂一个已确认分型。返回本次新成的一笔, 否则 None。"""
        seq = self.seq
        if not seq:
            seq.append(fx)
            return None
        last = seq[-1]
        if fx.kind == last.kind:
            if _more_extreme(fx, last):
                seq[-1] = fx
                if self.bis:
                    self.bis[-1].end = fx      # 笔延伸: 端点后移且更极端
            return None
        if fx.idx - last.idx >= self.min_gap:
            seq.append(fx)
            b = Bi(len(self.bis), seq[-2], fx)
            self.bis.append(b)
            return b
        self.n_dropped += 1
        return None

    # ------------------------------------------------------------------
    @property
    def final_bis(self) -> list[Bi]:
        """已冻结的笔(最后一笔仍可能延伸, 故排除)。"""
        return self.bis[:-1] if self.bis else []

    @property
    def n_final(self) -> int:
        return max(0, len(self.bis) - 1)

    @property
    def last(self) -> Bi | None:
        return self.bis[-1] if self.bis else None

    @property
    def last_final(self) -> Bi | None:
        return self.bis[-2] if len(self.bis) >= 2 else None


def build_bi(fxs: list[FX], min_gap: int = MIN_GAP) -> list[Bi]:
    """离线批量成笔。"""
    b = BiBuilder(min_gap)
    for f in fxs:
        b.push(f)
    return b.bis


def validate(bis: list[Bi], min_gap: int = MIN_GAP) -> list[str]:
    """自检: 返回问题清单(空 = 通过)。"""
    errs: list[str] = []
    for k, b in enumerate(bis):
        if b.start.kind == b.end.kind:
            errs.append(f"#{k} 两端同类型分型")
        if b.end.idx - b.start.idx < min_gap:
            errs.append(f"#{k} 间隔不足({b.end.idx - b.start.idx})")
        if b.direction > 0 and not (b.start.kind == BOTTOM and b.end.kind == TOP):
            errs.append(f"#{k} 向上笔端点方向错")
        if k:
            p = bis[k - 1]
            if p.end is not b.start and p.end.idx != b.start.idx:
                errs.append(f"#{k} 与前一笔不连续")
            if p.direction == b.direction:
                errs.append(f"#{k} 与前一笔同向")
    return errs
