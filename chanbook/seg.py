"""线段 —— 由至少三笔构成，用"特征序列"的破坏判定来确定端点。

原文要点:
  1. 线段至少由三笔构成, 且第一笔方向与线段方向一致。
  2. 线段的破坏用**特征序列**判定:
       向上线段的特征序列 = 该线段内所有**向下笔**(反向笔), 按出现顺序排列;
       向下线段镜像(特征序列 = 向上笔)。
  3. 特征序列元素先做**包含处理**(与K线包含处理同一套规则, 方向由序列自身决定),
     再看是否出现**分型**:
       向上线段的特征序列出现**顶分型** => 该线段结束于该顶分型的高点;
       向下线段的特征序列出现**底分型** => 该线段结束于该底分型的低点。
  4. 两种破坏情形:
       第一种情况: 特征序列分型的**第1、2元素之间无缺口** => 分型成立即结束;
       第二种情况: 第1、2元素之间**有缺口** => 需要"从该分型最高点开始的
                   反向走势"自身的特征序列出现**底分型**(镜像: 顶分型)才确认。
     (缺口定义, 向上线段: 第2元素的最低点 > 第1元素的最高点。)

工程妥协(已在 DESIGN.md §6 声明):
  - 第二种情况用"反向走势内部的反向笔序列(即原方向的笔)出现镜像分型"来确认,
    对原文中"缺口套缺口"的极端嵌套不再无限递推;
  - 另设**强制破坏**兜底: 反向笔跌破(升破)线段起点 => 线段无条件结束
    (原文体系下这是必然被破坏的充分情形, 用它可以避免极端行情下线段的过度延展)。

流式语义(与 bi.py 一致):
  - 只处理**已冻结的笔**(`bi.py` 中 `final_bis`), 因此线段结论不会被未定型的笔污染;
  - `segs` = 已确认线段; 当前进行中的线段通过 `cur_snapshot()` 查看, 是 provisional。
  - 线段结束点是在"确认那一刻"才公布的(滞后确认), 回测/实盘都只能用当时已知的结论
    —— 结构本身不偷看未来, 但确认成本是真实存在的。
"""

from __future__ import annotations

from dataclasses import dataclass

from .bi import Bi

__all__ = ["Seg", "SegBuilder", "build_segs", "validate"]

MIN_BI = 3


@dataclass
class Seg:
    """线段。"""

    i: int                # 序号
    direction: int        # 1 向上 / -1 向下
    b0: int               # 起始笔下标(在 bis 序列中)
    b1: int               # 结束笔下标(含)
    start_price: float
    end_price: float
    high: float
    low: float
    start_ts: int
    end_ts: int
    how: str              # first / second / force
    confirmed_bi: int     # 确认时已处理到的笔下标(滞后量)

    @property
    def n_bi(self) -> int:
        return self.b1 - self.b0 + 1

    @property
    def amp(self) -> float:
        return abs(self.end_price - self.start_price)

    def label(self) -> str:
        return "向上线段" if self.direction > 0 else "向下线段"

    def __repr__(self) -> str:      # pragma: no cover
        d = "↑" if self.direction > 0 else "↓"
        return (f"Seg#{self.i}{d} {self.start_price:.6g}→{self.end_price:.6g} "
                f"[bi {self.b0}-{self.b1}, {self.how}]")


# ---------------------------------------------------------------- 特征序列工具
def _feat_push(feat: list[list], el: list) -> None:
    """把特征序列元素 [bi_idx, high, low] 推入, 做包含处理。

    包含时方向由**序列最后两个元素**的高低关系决定(与K线包含处理一致)。
    """
    if not feat:
        feat.append(el)
        return
    last = feat[-1]
    if (el[1] <= last[1] and el[2] >= last[2]) or (el[1] >= last[1] and el[2] <= last[2]):
        d = 1
        if len(feat) >= 2:
            d = 1 if feat[-1][1] > feat[-2][1] else -1
        if d > 0:
            last[1] = max(last[1], el[1])
            last[2] = max(last[2], el[2])
        else:
            last[1] = min(last[1], el[1])
            last[2] = min(last[2], el[2])
        last[0] = el[0]          # 定位用: 保留最新的笔下标
    else:
        feat.append(el)


def _fx_at_tail(feat: list[list], top: bool):
    """检查特征序列末尾三元素是否构成分型, 返回中间元素或 None。"""
    if len(feat) < 3:
        return None
    a, b, c = feat[-3], feat[-2], feat[-1]
    if top:
        if b[1] > a[1] and b[1] > c[1] and b[2] > a[2] and b[2] > c[2]:
            return b
    else:
        if b[2] < a[2] and b[2] < c[2] and b[1] < a[1] and b[1] < c[1]:
            return b
    return None


class SegBuilder:
    """流式线段构造器。"""

    def __init__(self, min_bi: int = MIN_BI) -> None:
        self.min_bi = int(min_bi)
        self.segs: list[Seg] = []
        self._cursor = 0
        self.cur: dict | None = None      # 进行中的线段
        self.feat: list[list] = []        # 当前线段特征序列
        self.pending: dict | None = None  # 第二种情况待确认
        self.n_first = 0
        self.n_second = 0
        self.n_force = 0

    # ------------------------------------------------------------------
    def update(self, bis: list[Bi], n_frozen: int | None = None) -> list[Seg]:
        """消化新笔。返回本次新确认的线段。

        线段在 `end_bi` 结束、新线段自 `end_bi+1` 起 —— 若 `end_bi+1 <= 当前笔`,
        说明结束点之前的笔被重新归属, **游标回退重放**, 否则嵌套破坏会被漏判。
        回退不会死循环: 新线段起点 `b0 = end_bi+1`, 再次成段至少需要 min_bi 笔,
        故必然 `end_bi' >= end_bi + min_bi > end_bi`, 单调推进。

        n_frozen: 已冻结笔数(默认 len(bis)-1)。流式调用方传入该值即可避免
        每根K线都做一次 `bis[:-1]` 的 O(n) 切片 —— 长回测里这是 O(n²) 的元凶。
        """
        out: list[Seg] = []
        nf = (len(bis) - 1) if n_frozen is None else int(n_frozen)
        if nf <= 0:
            return out
        guard = 0
        while self._cursor < nf:
            k = self._cursor
            seg = self._step(bis, k)
            if seg is not None:
                out.append(seg)
                back = seg.b1 + 1
                if back <= k:
                    self._cursor = back
                    guard += 1
                    if guard > 4 * (nf + 8):      # 兜底熔断, 理论上不可达
                        break
                    continue
            self._cursor += 1
        return out

    # ------------------------------------------------------------------
    def _start(self, bis: list[Bi], k: int) -> None:
        bi = bis[k]
        self.cur = {"dir": bi.direction, "b0": k, "b1": k,
                    "high": bi.high, "low": bi.low}
        self.feat = []
        self.pending = None

    def _finalize(self, bis: list[Bi], end_bi: int, end_px: float,
                  how: str, k: int) -> Seg:
        cur = self.cur
        b0 = cur["b0"]
        seg = Seg(
            i=len(self.segs), direction=cur["dir"], b0=b0, b1=end_bi,
            start_price=bis[b0].start_price, end_price=end_px,
            high=max(bis[b0].start_price, end_px),
            low=min(bis[b0].start_price, end_px),
            start_ts=bis[b0].start_ts, end_ts=bis[end_bi].end_ts,
            how=how, confirmed_bi=k,
        )
        for j in range(b0, end_bi + 1):
            seg.high = max(seg.high, bis[j].high)
            seg.low = min(seg.low, bis[j].low)
        self.segs.append(seg)
        if how == "first":
            self.n_first += 1
        elif how == "second":
            self.n_second += 1
        else:
            self.n_force += 1
        # 新线段从 end_bi+1 起; 由 update() 回退游标后经 _step 重建状态
        self.cur = None
        self.feat = []
        self.pending = None
        return seg

    # ------------------------------------------------------------------
    def _step(self, bis: list[Bi], k: int) -> Seg | None:
        if self.cur is None:
            self._start(bis, k)
            return None
        cur = self.cur
        bi = bis[k]
        up = cur["dir"] > 0

        # ---------- 同向笔: 线段延伸 ----------
        if bi.direction == cur["dir"]:
            cur["b1"] = k
            cur["high"] = max(cur["high"], bi.high)
            cur["low"] = min(cur["low"], bi.low)
            if self.pending is not None:
                px = self.pending["price"]
                if (up and bi.high > px) or ((not up) and bi.low < px):
                    self.pending = None                    # 创出更极端值 => 线段未结束
                else:
                    # 反向走势内部的反向笔 => 第二种情况的确认序列
                    _feat_push(self.pending["feat"], [k, bi.high, bi.low])
                    if _fx_at_tail(self.pending["feat"], top=(not up)) is not None:
                        mid = self.pending["mid_bi"]
                        return self._finalize(bis, mid - 1, self.pending["price"],
                                              "second", k)
            return None

        # ---------- 反向笔 ----------
        # 强制破坏兜底: 反向笔彻底打穿线段起点
        if (up and bi.end_price < cur["low"]) or ((not up) and bi.end_price > cur["high"]):
            if cur["b1"] - cur["b0"] + 1 >= self.min_bi:
                return self._finalize(bis, cur["b1"], bis[cur["b1"]].end_price,
                                      "force", k)

        _feat_push(self.feat, [k, bi.high, bi.low])
        if self.pending is None:
            fx = _fx_at_tail(self.feat, top=up)
            if fx is not None:
                mid_bi = fx[0]
                if mid_bi - cur["b0"] >= self.min_bi:      # 结束笔 = mid_bi-1
                    top_px = bis[mid_bi].start_price
                    a, b = self.feat[-3], self.feat[-2]
                    gap = (b[2] > a[1]) if up else (b[1] < a[2])
                    if not gap:
                        return self._finalize(bis, mid_bi - 1, top_px, "first", k)
                    self.pending = {"price": top_px, "mid_bi": mid_bi,
                                    "feat": [], "at": k}
        return None

    # ------------------------------------------------------------------
    def cur_snapshot(self, bis: list[Bi]) -> Seg | None:
        """当前未完成线段(provisional)。"""
        cur = self.cur
        if cur is None:
            return None
        b0 = cur["b0"]
        hi, lo = bis[b0].start_price, bis[b0].start_price
        for j in range(b0, min(cur["b1"], len(bis) - 1) + 1):
            hi = max(hi, bis[j].high)
            lo = min(lo, bis[j].low)
        return Seg(i=len(self.segs), direction=cur["dir"], b0=b0, b1=cur["b1"],
                   start_price=bis[b0].start_price, end_price=bis[cur["b1"]].end_price,
                   high=hi, low=lo, start_ts=bis[b0].start_ts,
                   end_ts=bis[cur["b1"]].end_ts, how="cur", confirmed_bi=-1)


def build_segs(bis: list[Bi], min_bi: int = MIN_BI) -> list[Seg]:
    b = SegBuilder(min_bi)
    b.update(bis)
    return b.segs


def validate(segs: list[Seg], min_bi: int = MIN_BI) -> list[str]:
    errs: list[str] = []
    for k, s in enumerate(segs):
        if s.n_bi < min_bi:
            errs.append(f"#{k} 笔数不足({s.n_bi})")
        if s.high < s.low:
            errs.append(f"#{k} 高低点颠倒")
        if (s.direction > 0) != (s.end_price > s.start_price):
            errs.append(f"#{k} 方向与首尾价格矛盾")
        if k:
            p = segs[k - 1]
            if p.direction == s.direction:
                errs.append(f"#{k} 与上一线段同向")
            if s.b0 != p.b1 + 1:
                errs.append(f"#{k} 笔区间不连续({p.b1}→{s.b0})")
    return errs
