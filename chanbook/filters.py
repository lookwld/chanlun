"""可选构件 —— 量能 / 均线 / 仓位分层。**默认全部关闭**; 关闭时不创建对象、主流程零开销。

本模块是 108 课理论缺口审计里 B2 / B6 / B8 三项的落地:

  * **量能**(VOL, 对应 B2): 原文里量能是"力度"的辅助判据 —— 背驰与中枢突破的
    有效性都要有量的配合。数据其实早已读进 `data.Kl.v`(`data.py:61/87`),
    但整个 `chanbook/` 包内**零使用**。
  * **均线**(MA, 对应 B6): 缠论 B1 的通用辅助确认是"**两条**均线"(5/10 或 13/21)
    的位置关系; 这与旧引擎被 A/B 证伪的"**四条**均线多头排列闸门"
    (`ma5>ma10>ma20>ma60`)是**两回事** —— 后者把一买整类排除掉了, 前者没被检验过。
  * **仓位分层**(POS, 对应 B8): 原文按"级别 + 买卖点类型"分档定仓; 项目此前
    只有"一买名义额减半"这一条硬编码例外(`risk.B1_WIDE`)。

⚠ 三者都是**纯新增**: 构造参数取默认值时本模块的对象不会被创建, 既有 A/B 结果
   逐位不变。这是本项目一直守着的纪律(见 `DESIGN §0` 硬约束)。

⚠ 反面提醒(必须与实现一起读): 审计的 §6 已说明 —— 量能/均线都属于**入场侧**,
   而项目 13 轮撞墙绝大多数发生在入场侧的精细化上。本模块提供的是"把零件装上
   去测一次"的能力, 不是"装上就会更好"的承诺。
"""

from __future__ import annotations

from array import array
from bisect import bisect_left, bisect_right
from collections import deque

__all__ = ["MA", "MA_MODES", "VolCtx", "POS_LAYERS", "ma_size_scale",
           "ma_spread", "ma_veto_ok", "pos_scale", "ATR1D"]

# 均线模式(见 `ma_veto_ok` / `ma_spread` 的说明)
MA_MODES = ("off", "ma", "close", "both", "size")

# 仓位分层模式
POS_LAYERS = ("off", "amp", "ratio", "kind")

MS_DAY = 86_400_000


# ------------------------------------------------------------------ 均线
class MA:
    """收盘价均线(流式)。只保留最近 `max(periods)` 个收盘价。

    为什么自带 buf 而不去读 `Kl.c`: 均线要按**已喂入**的根数算(等长于级别视图),
    而 `Kl` 是整段归档 —— 直接切 `kl.c[-n:]` 会把"还没喂进来"的未来根也算进去。
    这里由 `Level._push` 每根喂一次, 天然与级别视图同步、无未来函数。
    """

    __slots__ = ("periods", "_buf")

    def __init__(self, periods=(5, 10, 20, 60)) -> None:
        ps = tuple(sorted({int(p) for p in periods if int(p) > 0}))
        if not ps:
            raise ValueError("MA 需要至少一个正周期")
        self.periods = ps
        self._buf: deque = deque(maxlen=max(ps))

    def push(self, close: float) -> None:
        self._buf.append(float(close))

    def value(self, p: int) -> float | None:
        if len(self._buf) < p:
            return None
        b = self._buf
        s = 0.0
        for k in range(1, p + 1):
            s += b[-k]
        return s / p

    def ready(self) -> bool:
        return len(self._buf) >= max(self.periods)


def ma_spread(ma: MA, fast: int, slow: int) -> float | None:
    """`(ma_fast - ma_slow) / ma_slow` —— 均线间距的相对值(无量纲, 跨标的可比)。

    正 = 短期均线在长期之上(金叉后); 负 = 空头压制。
    """
    f, s = ma.value(fast), ma.value(slow)
    if f is None or s is None or s <= 0:
        return None
    return (f - s) / s


def ma_veto_ok(ma: MA, close: float, mode: str,
               fast: int = 5, slow: int = 10, tol: float = 0.001) -> bool | None:
    """**否决型**均线闸门。返回 None = 均线尚未就绪(调用方按"不拦截"处理)。

    三种强度(都只做多, 因此都是"短期不再被压制"的读法):
      `ma`    —— `ma_fast >= ma_slow`(已金叉或走平)。最接近原文"两条均线"的用法。
      `close` —— `close >= ma_fast`(收盘价站上短期均线)。比金叉更早。
      `both`  —— 两者同时成立。最严。
    `tol` 是金叉判据的容差(允许 ma_fast 比 ma_slow 低 tol 以内, 避免卡在临界点)。

    ⚠ 这是**滞后**确认: 一买出现时价格刚创新低, `ma5>=ma10` 通常还没发生 ——
    所以本闸门会砍掉大量一买。那正是这次实验要测的东西(上一轮要测的"否决型"路子)。
    """
    if mode == "ma":
        sp = ma_spread(ma, fast, slow)
        return None if sp is None else bool(sp >= -tol)
    if mode == "close":
        f = ma.value(fast)
        return None if f is None else bool(close >= f)
    if mode == "both":
        sp = ma_spread(ma, fast, slow)
        f = ma.value(fast)
        if sp is None or f is None:
            return None
        return bool(sp >= -tol and close >= f)
    return None


# ------------------------------------------------------------------ 量能
class VolCtx:
    """操作级别成交量的**前缀和**视图 —— 供"量能配合"判据用。

    只在 `Level.attach` 时构造一次(O(n)), 之后区间均量是两次二分 + 一次相减。
    区间由调用方给出(通常是两段笔的 [start_ts, end_ts]) —— 笔都是**已冻结**的,
    所以区间右端必然 <= 当前bar, 无未来函数。
    """

    __slots__ = ("ts", "_cum", "n")

    def __init__(self, kl) -> None:
        n = int(kl.n)
        cum = array("d", [0.0]) * (n + 1)
        s = 0.0
        v = kl.v
        for i in range(n):
            s += float(v[i])
            cum[i + 1] = s
        self.ts = kl.ts
        self._cum = cum
        self.n = n

    def mean(self, ts0: int, ts1: int) -> float | None:
        """[ts0, ts1] 闭区间内各根成交量的均值。区间内无根则返回 None。"""
        if ts0 > ts1:
            ts0, ts1 = ts1, ts0
        a = bisect_left(self.ts, int(ts0))
        b = bisect_right(self.ts, int(ts1))
        if b <= a:
            return None
        return (self._cum[b] - self._cum[a]) / (b - a)

    def ratio(self, ts0_a: int, ts1_a: int, ts0_c: int, ts1_c: int) -> float | None:
        """后段均量 / 前段均量。`< 1` = 后段缩量(背驰应有的量能配合)。"""
        va = self.mean(ts0_a, ts1_a)
        vc = self.mean(ts0_c, ts1_c)
        if va is None or vc is None or va <= 0:
            return None
        return vc / va


# 仓位分档表: (上界, 倍数)。间隔 = (ma_fast - ma_slow)/ma_slow。
# 单调递增: 均线间距越"多头", 仓位越重。这是原文"顺级别/顺买点加仓"的
# 最粗糙版本 —— 只作为"仓位分层是否比入场精细化更有价值"的一次对照。
_MA_SPREAD_BUCKETS = ((-0.01, 0.50), (0.0, 0.75), (0.02, 1.00),
                      (0.05, 1.25), (float("inf"), 1.50))


def ma_size_scale(spread: float | None) -> float:
    """**仓位型**均线: 由均线间距决定名义额倍数。spread 缺失 ⇒ 1.0(不缩放)。"""
    if spread is None:
        return 1.0
    for edge, sc in _MA_SPREAD_BUCKETS:
        if spread <= edge:
            return float(sc)
    return 1.0


# ------------------------------------------------------------------ 仓位分层
# 分档表: (上界, 倍数)。区间是"<= 上界"。顺序扫描, 最后一档用 inf 兜底。
# 方向: 背驰越"陡" —— 后段幅度/面积远小于前段 —— 越是原文意义上的强背驰 ⇒ 越重。
_AMP_BUCKETS = ((0.50, 1.50), (1.00, 1.00), (1.50, 0.60), (float("inf"), 0.35))
_RATIO_BUCKETS = ((0.50, 1.50), (0.80, 1.00), (1.00, 0.60), (float("inf"), 0.35))
_KIND_SCALE = {"B1": 1.00, "B2": 1.25, "B3": 1.50}


def pos_scale(layer: str, kind: str, feats: dict) -> float:
    """名义额倍数。`off` 恒为 1.0 ⇒ 与改动前逐位一致。

    `amp`   —— 按 `feats["amp_ratio"]`(后段幅度/前段幅度)分档
    `ratio` —— 按 `feats["ratio_area"]`(MACD 面积比)分档
    `kind`  —— 按买点类型分档(B3 最重, 因为它是趋势确认后才出)

    ⚠ 不能用 `beichi_score` 分档: 审计 A4 已证明它在判据结构上退化为
    `{3,4}` 的二值(实测 4 分占 90.1%), 没有分档能力。用连续比值。
    """
    if not layer or layer == "off":
        return 1.0
    if layer == "kind":
        return float(_KIND_SCALE.get(kind, 1.0))
    key = "amp_ratio" if layer == "amp" else "ratio_area"
    try:
        v = float(feats.get(key))
    except (TypeError, ValueError):
        return 1.0
    for edge, sc in (_AMP_BUCKETS if layer == "amp" else _RATIO_BUCKETS):
        if v <= edge:
            return sc
    return 1.0


# ------------------------------------------------------------------ 波动率(日线)
class ATR1D:
    """日线 ATR% 的**查表**视图 —— 移植 `chan-quant/app/vol.py` 的波动上限闸门。

    为什么单独建这个: 本模块的 `VolCtx` 是**成交量**(量能配合), 而这个是
    **波动率绝对水平** —— 两个维度完全不同, 只是 chan-quant 把后者也叫 `vol_gate`
    (名字撞车, 见 DESIGN.md §6)。

    口径与 `chan-quant/app/vol.py` 的 `atr1d_pct()` **严格同源**:
      * 取最后 `n+1` 根**已收盘**日线 ⇒ `n` 个 TR (n 默认 14);
      * `TR = max(high-low, |high-prev_close|, |low-prev_close|)`;
      * `ATR% = mean(TR) / 最后一根收盘价`;
      * 数据不足 `n+1` 根 ⇒ 返回 None(调用方 **fail-open 放行**, 与 chan-quant 一致)。

    为什么是"查表"而不是"流式 push": 日线是与操作级别**不同周期**的序列, 用
    `Level._push` 喂不进去; 而它又必须是"某一时刻**已收盘**的日线"。
    预计算 O(n) 之后, `at(close_ts)` 是两次二分 —— 且**天然无未来函数**
    (只碰 `ts + 1d <= close_ts` 的根)。
    """

    __slots__ = ("ts", "_atr", "n", "tf_ms", "m")

    def __init__(self, kl, n: int = 14, tf_ms: int = MS_DAY) -> None:
        self.n = int(n)
        self.tf_ms = int(tf_ms)
        self.ts = kl.ts
        m = int(kl.n)
        self.m = m
        tr = array("d", [0.0]) * m
        h, l, c = kl.h, kl.l, kl.c
        for i in range(1, m):                    # tr[i] = 第 i 根的真实波幅
            pc = float(c[i - 1])
            hi, lo = float(h[i]), float(l[i])
            tr[i] = max(hi - lo, abs(hi - pc), abs(lo - pc))
        out = array("d", [0.0]) * m
        k = self.n
        s = 0.0
        for i in range(1, m):
            s += tr[i]
            if i > k:
                s -= tr[i - k]
            if i >= k:                           # 已有完整 k 个 TR
                cl = float(c[i])
                out[i] = (s / k / cl) if cl > 0 else 0.0
        self._atr = out

    def at(self, close_ts: int) -> float | None:
        """`close_ts` 时刻**已收盘**日线的 ATR%。不足窗口返回 None(放行)。"""
        j = bisect_right(self.ts, int(close_ts) - self.tf_ms)
        if j < self.n + 1:
            return None
        v = float(self._atr[j - 1])
        return v if v > 0.0 else None

    def ok(self, close_ts: int, cap: float) -> bool | None:
        """`cap <= 0` ⇒ None(闸门关闭, 调用方放行); 数据不足 ⇒ None(放行)。"""
        if cap <= 0.0:
            return None
        v = self.at(close_ts)
        if v is None:
            return None
        return bool(v <= cap)
