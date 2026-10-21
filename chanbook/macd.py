"""MACD 与"段力度"度量 —— 背驰判定的量化基础。

原文用 MACD 判背驰, 但没有给出唯一定义; 工程上被广泛使用的口径是:
    DIFF = EMA(close, 12) - EMA(close, 26)
    DEA  = EMA(DIFF, 9)
    HIST = 2 * (DIFF - DEA)
一段走势的"力度"用以下量刻画(背驰 = 后段力度小于前段):
    area   段内 HIST 的绝对面积和   Σ|HIST|          —— 最常用
    per    单位时间力度             area / 段内根数   —— 排除"后段更长"带来的机械放大
    peak   段内 |DIFF| 的最大值                       —— 黄白线回拉幅度的代理
    pull   段内 HIST 同号极值        max|HIST|

本模块为**流式增量**实现:
  - EMA 逐根更新, O(1);
  - 历史 DIFF/HIST 存成与时间戳对齐的数组, 供任意区间聚合(二分定位);
  - 序列化接口 `state()/load()` 便于回测断点与实盘重启。

初始化约定: EMA 以**首个 close 播种**(而非 SMA 播种)。这会带来开头的短暂偏差,
但背驰判定只做**区间之间的相对比较**, 起点偏差在两端同时存在、相互抵消;
换成 SMA 播种需回看 26 根, 不利于流式启动。此约定显式声明。
"""

from __future__ import annotations

from bisect import bisect_left

__all__ = ["MACD", "ema_series"]


class MACD:
    """增量 MACD + 区间力度聚合。"""

    __slots__ = ("fast", "slow", "signal", "kf", "ks", "kd",
                 "ema_f", "ema_s", "dea", "diff", "hist", "n",
                 "ts", "d", "h")

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9) -> None:
        self.fast, self.slow, self.signal = fast, slow, signal
        self.kf = 2.0 / (fast + 1)
        self.ks = 2.0 / (slow + 1)
        self.kd = 2.0 / (signal + 1)
        self.ema_f = None
        self.ema_s = None
        self.dea = 0.0
        self.diff = 0.0
        self.hist = 0.0
        self.n = 0
        self.ts: list[int] = []
        self.d: list[float] = []       # DIFF
        self.h: list[float] = []       # HIST

    # ------------------------------------------------------------------
    def update(self, ts: int, close: float) -> float:
        """喂一根已收盘K线, 返回该根的 HIST。"""
        c = float(close)
        if self.ema_f is None:
            self.ema_f = self.ema_s = c
            self.dea = 0.0
        else:
            self.ema_f += self.kf * (c - self.ema_f)
            self.ema_s += self.ks * (c - self.ema_s)
        diff = self.ema_f - self.ema_s
        self.dea += self.kd * (diff - self.dea)
        hist = 2.0 * (diff - self.dea)
        self.diff, self.hist = diff, hist
        self.n += 1
        self.ts.append(int(ts))
        self.d.append(diff)
        self.h.append(hist)
        return hist

    # ------------------------------------------------------------------
    def _range(self, ts_a: int, ts_b: int) -> tuple[int, int]:
        """返回 [ts_a, ts_b] 覆盖的下标区间(左闭右开)。"""
        i = bisect_left(self.ts, int(ts_a))
        j = bisect_left(self.ts, int(ts_b) + 1)
        if j <= i:
            j = min(i + 1, len(self.ts))
        return i, j

    def area(self, ts_a: int, ts_b: int, sign: int = 0) -> float:
        """段内 |HIST| 面积。sign=1 只统计正柱, -1 只统计负柱, 0 全算。"""
        i, j = self._range(ts_a, ts_b)
        tot = 0.0
        h = self.h
        if sign == 0:
            for k in range(i, j):
                tot += abs(h[k])
        elif sign > 0:
            for k in range(i, j):
                if h[k] > 0:
                    tot += h[k]
        else:
            for k in range(i, j):
                if h[k] < 0:
                    tot += -h[k]
        return tot

    def peak(self, ts_a: int, ts_b: int, sign: int = 0) -> float:
        """段内 |DIFF| 峰值。sign 语义同 area。"""
        i, j = self._range(ts_a, ts_b)
        best = 0.0
        d = self.d
        for k in range(i, j):
            v = d[k]
            if sign > 0 and v <= 0:
                continue
            if sign < 0 and v >= 0:
                continue
            if abs(v) > best:
                best = abs(v)
        return best

    def hist_peak(self, ts_a: int, ts_b: int, sign: int = 0) -> float:
        """段内 |HIST| 峰值(柱体最高/最低)。"""
        i, j = self._range(ts_a, ts_b)
        best = 0.0
        h = self.h
        for k in range(i, j):
            v = h[k]
            if sign > 0 and v <= 0:
                continue
            if sign < 0 and v >= 0:
                continue
            if abs(v) > best:
                best = abs(v)
        return best

    def n_bars(self, ts_a: int, ts_b: int) -> int:
        i, j = self._range(ts_a, ts_b)
        return max(1, j - i)

    def macd_area(self, ts_a: int, ts_b: int) -> tuple[float, float]:
        """(正柱面积, 负柱面积) 分别返回, 供同时需要两侧的场景。"""
        return self.area(ts_a, ts_b, 1), self.area(ts_a, ts_b, -1)

    def cross_state(self, ts_a: int, ts_b: int) -> int:
        """段末 DIFF 与 DEA 的相对位置: 1 = DIFF 在上, -1 = 在下, 0 = 无数据。"""
        i, j = self._range(ts_a, ts_b)
        if j <= 0:
            return 0
        k = j - 1
        return 1 if self.d[k] >= self.dea else -1

    # ------------------------------------------------------------------
    def summary(self, ts_a: int, ts_b: int, sign: int = 0) -> dict:
        a = self.area(ts_a, ts_b, sign)
        n = self.n_bars(ts_a, ts_b)
        return {"area": a, "bars": n, "per": a / n,
                "peak": self.peak(ts_a, ts_b, sign),
                "cross": self.cross_state(ts_a, ts_b)}

    def state(self) -> dict:
        return {"fast": self.fast, "slow": self.slow, "signal": self.signal,
                "ema_f": self.ema_f, "ema_s": self.ema_s, "dea": self.dea,
                "n": self.n}

    def load(self, st: dict) -> None:
        self.ema_f = st.get("ema_f")
        self.ema_s = st.get("ema_s")
        self.dea = st.get("dea", 0.0)
        self.n = st.get("n", 0)


def ema_series(values: list[float], n: int) -> list[float]:
    """离线 EMA(用于测试与对拍)。"""
    k = 2.0 / (n + 1)
    out: list[float] = []
    e = None
    for v in values:
        e = v if e is None else e + k * (v - e)
        out.append(e)
    return out
