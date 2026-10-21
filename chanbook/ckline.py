"""K线包含处理 —— 全书体系的第一块基石。

原文要点（对应《教你炒股票》K线包含处理一节）:
  1. 相邻两根K线, 若一根的高低点区间完全覆盖另一根(含端点相等), 称为"包含关系"。
  2. 存在包含关系时必须先合并, 合并后的序列才可用于分型判定 —— 否则同一价位上
     会同时出现"顶"与"底"的矛盾结构。
  3. 合并方向由**被合并K线之前的那一根K线所确定的方向**决定:
       向上处理 -> 高点取两者较高, 低点取两者较高 (高高)
       向下处理 -> 高点取两者较低, 低点取两者较低 (低低)
  4. 方向的定义: 新的合并K线的高点高于前一根合并K线的高点 => 向上, 反之向下。

工程约定:
  - 这里是**流式**实现(一根一根喂), 由 `KLineMerger.push()` 维护;
    纯函数式入口 `merge_klines()` 保留给离线分析/测试。
  - 合并K线记录原始下标区间 [i0, i1] 与时间区间 [ts0, ts1], 便于回溯定位。
  - 全程只用已收盘K线; 本模块不涉及未来函数。

与旧内核(chan-quant/app/chan.py)的差异:
  - 旧版在"包含合并"时不更新方向、在"非包含"时才更新方向, 逻辑等价但
    初始方向硬编码为 1; 新版显式维护 `direction` 并在首根后按定义确定。
  - 旧版 `CK` 只有 idx/end/high/low/ts; 新版额外记录 ts1/n(原始K线根数),
    以便分型/线段回溯原始K线。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["CK", "KLineMerger", "merge_klines"]


@dataclass(slots=True)
class CK:
    """合并后的缠论K线。"""

    i0: int          # 原始K线起始下标
    i1: int          # 原始K线结束下标(含)
    ts0: int         # 起始K线开盘时间(毫秒)
    ts1: int         # 结束K线开盘时间(毫秒)
    high: float
    low: float
    n: int = 1       # 被合并的原始K线根数

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    def contains(self, high: float, low: float) -> bool:
        """本K线是否与新K线构成包含关系(含端点相等)。"""
        return (high <= self.high and low >= self.low) or \
               (high >= self.high and low <= self.low)


class KLineMerger:
    """流式包含处理。"""

    __slots__ = ("cks", "direction")

    def __init__(self) -> None:
        self.cks: list[CK] = []
        self.direction: int = 0        # 1 向上 / -1 向下 / 0 未定

    # ------------------------------------------------------------------
    def push(self, i: int, ts: int, high: float, low: float) -> CK:
        """喂入一根原始K线, 返回被更新/新建的合并K线。"""
        cks = self.cks
        if not cks:
            c = CK(i, i, ts, ts, high, low, 1)
            cks.append(c)
            return c
        last = cks[-1]
        if last.contains(high, low):
            # 包含关系 -> 沿既有方向合并; 方向本身不变
            d = self.direction or 1
            if d > 0:
                last.high = max(last.high, high)
                last.low = max(last.low, low)
            else:
                last.high = min(last.high, high)
                last.low = min(last.low, low)
            last.i1 = i
            last.ts1 = ts
            last.n += 1
            return last
        # 非包含 -> 定方向并新增
        self.direction = 1 if high > last.high else -1
        c = CK(i, i, ts, ts, high, low, 1)
        cks.append(c)
        return c


def merge_klines(klines: list[dict]) -> list[CK]:
    """离线批量包含处理。klines 元素需含 ts/high/low。"""
    m = KLineMerger()
    for i, k in enumerate(klines):
        m.push(i, int(k["ts"]), float(k["high"]), float(k["low"]))
    return m.cks
