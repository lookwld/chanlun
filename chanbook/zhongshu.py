"""走势中枢 —— 三个连续次级别走势的重叠区间。

原文定义:
  某级别走势中枢 = **至少三个连续次级别走势类型**的重叠部分。
  记第 i 个次级别走势的高点为 g_i、低点为 d_i, 则
      ZG = min(g1, g2, g3)     中枢上沿
      ZD = max(d1, d2, d3)     中枢下沿
      中枢区间 = [ZD, ZG]      (必须 ZG > ZD, 否则不存在重叠)
  另记 GG = max(全部 g_i)、DD = min(全部 d_i) 为中枢的"最高/最低触及"。

中枢的三种演化(原文均给定了名字, 工程上必须区分, 否则级别会算错):
  1. **延伸**: 后续次级别走势仍与 [ZD, ZG] 有重叠 => 中枢延续, **区间不变**,
     GG/DD 随之扩张; 延伸段数达到 9 段(3+3+3) 时视为**扩张**, 级别上移一级。
  2. **新生**: 出现与中枢无重叠的次级别走势 => 当前中枢结束, 该走势开始新中枢。
  3. **扩展**: 相邻两个**同级别同向**中枢的区间有重叠 => 合并为更高级别中枢。

工程约定(与 DESIGN.md §6 一致):
  - 次级别走势用"笔"或"线段"代理(取决于 `level`), 这是工程近似而非原文的严格递归;
  - 中枢的 ZG/ZD 一旦由前三段确定**不再改动**(忠于原文), 延伸只更新 GG/DD 与段数;
  - `leave_i` 记录"离开中枢的那一段"的下标 —— 背驰判定需要它(进入段 vs 离开段)。

流式语义:
  - 输入必须是**已冻结的**次级别走势序列(笔/线段), 调用方负责剔除可能延伸的最后一个;
  - `zss` = 已结束的中枢; `cur` = 进行中的中枢(可能延伸), 通过 `cur_snapshot()` 查看。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["ZS", "ZSBuilder", "build_zhongshu", "merge_extensions"]

MIN_LEGS = 3
EXPAND_LEGS = 9          # 3+3+3: 延伸到此段数视为"扩张", 级别上移


@dataclass
class ZS:
    """走势中枢。"""

    i: int                # 序号
    level: int            # 级别(1=笔中枢, 2=线段中枢, ...)
    i0: int               # 首个组成走势的下标
    i1: int               # 最后一个组成走势的下标(含)
    zg: float             # 中枢上沿 = min(前三个走势的高点)
    zd: float             # 中枢下沿 = max(前三个走势的低点)
    gg: float             # 最高触及
    dd: float             # 最低触及
    start_ts: int
    end_ts: int
    enter_i: int          # 进入中枢的那一段下标(-1 表示无, 即起点就在中枢内)
    leave_i: int          # 离开中枢的那一段下标(-1 表示尚未离开)
    kind: str = "done"    # done 已结束 / cur 进行中
    expanded: bool = False
    enter_dir: int = 0    # 进入段方向: 1 上涨进入 / -1 下跌进入

    # ---------------------------------------------------------------
    @property
    def n_legs(self) -> int:
        return self.i1 - self.i0 + 1

    @property
    def mid(self) -> float:
        return (self.zg + self.zd) / 2.0

    @property
    def height(self) -> float:
        return self.zg - self.zd

    def contains_price(self, px: float) -> bool:
        return self.zd <= px <= self.zg

    def overlaps(self, other: "ZS") -> bool:
        return not (self.zd > other.zg or other.zd > self.zg)

    def label(self) -> str:
        return f"中枢#{self.i}(L{self.level}) [{self.zd:.6g}, {self.zg:.6g}]"

    def __repr__(self) -> str:      # pragma: no cover
        return (f"ZS#{self.i} L{self.level} [{self.zd:.6g},{self.zg:.6g}] "
                f"legs {self.i0}-{self.i1} {self.kind}")


class ZSBuilder:
    """流式中枢构造器。输入为已冻结的次级别走势序列(笔或线段)。"""

    def __init__(self, level: int = 1, min_legs: int = MIN_LEGS) -> None:
        self.level = int(level)
        self.min_legs = int(min_legs)
        self.zss: list[ZS] = []
        self.cur: dict | None = None
        self._cursor = 0
        self.n_new = 0            # 新生次数

    # ------------------------------------------------------------------
    def update(self, legs: list, n_frozen: int | None = None) -> list[ZS]:
        """消化新走势。返回本次新**结束**的中枢。

        单闭环状态机: 延伸 / 新生 / 成中枢 三类动作轮转, 每次动作都必须让
        `_cursor` 严格前进(延伸 +1; 成中枢 i+min_legs; 结束中枢后从 leave 段
        重新尝试, 失败则 +1), 因此不会死循环。

        n_frozen: 已冻结走势数(默认 len(legs))。传该值可避免流式调用方每根K线
        都复制一份 `legs[:-1]`(长回测里会退化成 O(n²))。
        """
        out: list[ZS] = []
        n = len(legs) if n_frozen is None else int(n_frozen)
        if n <= 0:
            return out
        while True:
            if self.cur is not None:
                if self._cursor >= n:
                    break
                leg = legs[self._cursor]
                c = self.cur
                if leg.low <= c["zg"] and leg.high >= c["zd"]:
                    c["i1"] = self._cursor
                    c["gg"] = max(c["gg"], leg.high)
                    c["dd"] = min(c["dd"], leg.low)
                    c["end_ts"] = leg.end_ts
                    self._cursor += 1
                    continue
                out.append(self._finish(legs, self._cursor))
                continue
            if self._cursor + self.min_legs > n:
                break
            i = self._cursor
            grp = legs[i:i + self.min_legs]
            zg = min(x.high for x in grp)
            zd = max(x.low for x in grp)
            if zg > zd:
                self.cur = {
                    "i0": i, "i1": i + self.min_legs - 1, "zg": zg, "zd": zd,
                    "gg": max(x.high for x in grp), "dd": min(x.low for x in grp),
                    "start_ts": grp[0].start_ts, "end_ts": grp[-1].end_ts,
                }
                self._cursor = i + self.min_legs
                self.n_new += 1
                continue
            self._cursor += 1
        return out

    # ------------------------------------------------------------------
    def all_zss(self) -> list[ZS]:
        """已结束中枢 + 进行中的中枢快照(进行中的排最后, kind='cur')。"""
        cur = self.cur_snapshot()
        return self.zss + ([cur] if cur is not None else [])

    # ------------------------------------------------------------------
    def _finish(self, legs: list, leave_k: int) -> ZS:
        c = self.cur
        i0 = c["i0"]
        ent = i0 - 1
        zs = ZS(
            i=len(self.zss), level=self.level, i0=i0, i1=c["i1"],
            zg=c["zg"], zd=c["zd"], gg=c["gg"], dd=c["dd"],
            start_ts=c["start_ts"], end_ts=c["end_ts"],
            enter_i=ent, leave_i=leave_k, kind="done",
            expanded=(c["i1"] - i0 + 1) >= EXPAND_LEGS,
            enter_dir=-int(getattr(legs[ent], "direction", 0)) if ent >= 0 else 0,
        )
        self.zss.append(zs)
        self.cur = None
        self._cursor = leave_k        # 离开的那一段开始新中枢(新生)
        return zs

    # ------------------------------------------------------------------
    def cur_snapshot(self) -> ZS | None:
        """进行中的中枢(可能继续延伸)。"""
        c = self.cur
        if c is None:
            return None
        i0 = c["i0"]
        ent = i0 - 1
        return ZS(i=len(self.zss), level=self.level, i0=i0, i1=c["i1"],
                  zg=c["zg"], zd=c["zd"], gg=c["gg"], dd=c["dd"],
                  start_ts=c["start_ts"], end_ts=c["end_ts"],
                  enter_i=ent, leave_i=-1, kind="cur",
                  expanded=(c["i1"] - i0 + 1) >= EXPAND_LEGS,
                  enter_dir=0)


def build_zhongshu(legs: list, level: int = 1, min_legs: int = MIN_LEGS) -> list[ZS]:
    """离线批量建中枢: 返回**已结束 + 进行中**的全部中枢(后者 kind='cur')。

    只返回已结束中枢会漏掉"当前正在延伸的那个", 而策略恰恰最需要它
    (三买判定就是相对当前中枢而言的)。
    """
    b = ZSBuilder(level, min_legs)
    b.update(legs)
    return b.all_zss()


def merge_extensions(zss: list[ZS]) -> list[ZS]:
    """中枢扩展: 相邻同级别中枢若区间重叠 => 合并为更高级别中枢。

    仅用于"级别上移"的判定, 返回合并后的中枢列表(不修改入参)。

    ⚠ **实测: 本函数在本实现的数据上不会触发**(2026-09-29 实测)。原因是结构性的,
    分两种口径:

      * **ZG/ZD 口径**(本函数的 `overlaps`): `ZSBuilder._finish` 只在"某一段与
        原中枢无重叠"时才结束中枢, 而新中枢正是从那一端起算 ⇒ 相邻中枢
        **按构造零重叠**。BTCUSDT 1h 前 2 万根实测 0 对重叠 / 220 对相邻对。
      * **GG/DD 口径**(波动区间): 由于相邻中枢首尾各含一个共享的边界段,
        实测 327/327 对**全部重叠** ⇒ 无区分力。

    另外 `ZS.expanded`(延伸 ≥9 段 = 3+3+3 的"扩张")在本数据上命中 104/328,
    但 `ZS.level` 只进 `Point.level`, **不参与任何判定**。

    ⇒ 结论: "中枢扩展 ⇒ 级别上移"在本实现里既不会改变中枢数, 也不会改变任何
    一笔成交。它的价值是**记录理论缺口**, 不是待生效的开关。
    """
    out: list[ZS] = []
    for z in zss:
        if out and out[-1].overlaps(z) and out[-1].enter_dir == z.enter_dir:
            p = out.pop()
            merged = ZS(
                i=p.i, level=max(p.level, z.level) + 1, i0=p.i0, i1=z.i1,
                zg=min(p.zg, z.zg), zd=max(p.zd, z.zd),
                gg=max(p.gg, z.gg), dd=min(p.dd, z.dd),
                start_ts=p.start_ts, end_ts=z.end_ts,
                enter_i=p.enter_i, leave_i=z.leave_i, kind=z.kind,
                expanded=True, enter_dir=p.enter_dir,
            )
            out.append(merged)
        else:
            out.append(z)
    return out
