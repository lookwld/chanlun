"""卖点(离场) —— 做多持仓的缠论离场判据。

原文(做多离场侧; 与 `points.py` 的买点严格镜像):
  **第一类卖点 S1**: 上涨趋势中出现背驰(最后一个中枢之后, 离开段力度小于进入段),
      该上涨段结束处即一卖。它是"逃顶"性质的卖点。
  **第二类卖点 S2**: 第一类卖点之后的**第一次次级别反抽**, 其高点**不创一卖新高**,
      反抽结束处即二卖。它是对一卖有效性的第一次确认。
  **第三类卖点 S3**: 一个次级别走势**向下离开中枢**, 随后以一个次级别走势回抽,
      其高点**不升回中枢区间**( < ZD), 回抽结束处即三卖。它是下跌确认后的卖点。

镜像对照表(右边是本模块, 左边是 points.py):

    | 项 | 买点 | 卖点 |
    |---|---|---|
    | 触发事件 | 向下笔刚冻结 | 向上笔刚冻结 |
    | 背驰方向 | `leg_beichi(..., -1)` 比低点 | `leg_beichi(..., +1)` 比高点 |
    | 突破判据 | `c.low  <  B.dd` (跌破最低触及) | `c.high >  B.gg` (涨破最高触及) |
    | 趋势判据 | `B.zg < A.zd` (中枢依次下移) | `B.zd > A.zg` (中枢依次上移) |
    | 构造确认 | 回调低点不破参照低点 | 反抽高点不破参照高点 |
    | 三类 | 回试低点 > ZG | 回抽高点 < ZD |

**为什么单独开一个模块, 而不是往 `PointScanner` 里加**:
  1. 触发事件相反 —— 买点挂在"向下笔刚冻结", 卖点挂在"向上笔刚冻结", 塞进同一个
     `scan()` 会让它每根 bar 都被调用两次、且内部 `last_b1` 状态机语义混乱;
  2. 本模块是**纯新增**: 不修改 `points.py` / `engine.py` / `strategy.py` 的任何
     既有代码路径 ⇒ 现有 A/B 结果逐位不变(2026-09-28 新增, 用于 4h 级别 B1 离场实验)。
  3. 卖点只在"已持仓"时才有意义, 它的调用方是离场管理器, 不是入场选股器 ——
     职责边界本来就不同。

无未来函数: 与买点同源 —— 所有判定只用到"已冻结"的笔; 卖点的右端必须是**刚冻结的
向上笔**(即一个上涨段已经走完并定型的那个事件)。
"""

from __future__ import annotations

from .beichi import leg_beichi
from .points import Point, _enter_leg, _exit_leg

__all__ = ["S1", "S2", "S3", "SellScanner", "S_LABEL", "is_exit"]

S1, S2, S3 = "S1", "S2", "S3"
S_LABEL = {S1: "卖①背驰", S2: "卖②反抽", S3: "卖③三卖"}


def is_exit(p) -> bool:
    return getattr(p, "kind", "") in (S1, S2, S3)


# ------------------------------------------------------------------ 内部工具
def _s1_feats(r, path: str, lag: int, amp_a: float, amp_c: float,
              zs_break: float, trend_beichi: bool) -> dict:
    """S1 的可量化信息拍平成特征字典(**纯观测, 不参与判定**)。

    字段名与买点的 `_b1_feats` 刻意保持一致(只是前缀换成语义中性的名字),
    这样入场/离场两侧的特征可以直接进同一张建模表。
    """
    rt = r.ratios or {}

    def g(k: str, d: float = 0.0) -> float:
        v = rt.get(k, d)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return d
        return d if v > 1e8 else v

    br = rt.get("bars") or [0, 0]
    return {
        "beichi_score": int(r.score),
        "beichi_new_extreme": int(bool(r.new_extreme)),
        "ratio_area": g("area", 1.0), "ratio_per": g("per", 1.0),
        "ratio_peak": g("peak", 1.0), "ratio_hist": g("hist_peak", 1.0),
        "bars_a": int(br[0]) if len(br) > 1 else 0,
        "bars_c": int(br[1]) if len(br) > 1 else 0,
        "s1_path": path, "s1_lag": int(lag),
        "amp_a": amp_a, "amp_c": amp_c,
        "amp_ratio": (amp_c / amp_a) if amp_a else 0.0,
        "zs_break": float(zs_break),
        "trend_beichi": int(bool(trend_beichi)),
    }


def _rally(bis: list, start_i: int, ref: float, min_bi: int, n: int,
           tol: float = 0.0) -> float | None:
    """自 start_i 起的反抽段是否成立, 成立则返回其最高点。

    要求: (1) 至少 min_bi 根冻结笔; (2) 最后一根是向上笔(反抽已结束);
          (3) 反抽全程最高点不越过参考价(含容差)。

    `_pullback` 的严格镜像(那边查最低点不破参考低点)。
    """
    if start_i < 0 or start_i >= n:
        return None
    after = bis[start_i:n]
    if len(after) < min_bi:
        return None
    if after[-1].direction < 0:          # 反抽尚未结束
        return None
    hi = max(b.high for b in after)
    if hi > ref * (1.0 + tol):
        return None
    return float(hi)


def _last_up_leg_frozen(bis: list, n: int) -> bool:
    """触发条件: 最后一根冻结笔是向上笔(即一个上涨段刚定型) —— 买点判据的镜像。"""
    return n > 0 and bis[n - 1].direction > 0


# ------------------------------------------------------------------ 扫描器
class SellScanner:
    """三类卖点扫描器。状态化: 记住最近一次 S1, 用于 S2 判定。"""

    def __init__(self, pull_bi: int = 1, tol: float = 0.0,
                 beichi_tol: float = 0.02, beichi_strict: bool = False,
                 s2_max_bi: int = 20, s1_max_lag: int = 12,
                 s1_zs_window: int = 1, s3_zs_window: int = 2,
                 s1_allow_no_zs: bool = True,
                 enable=(S1, S2, S3)) -> None:
        self.pull_bi = int(pull_bi)
        self.tol = float(tol)
        self.beichi_tol = float(beichi_tol)
        self.beichi_strict = bool(beichi_strict)
        self.s2_max_bi = int(s2_max_bi)
        self.s1_max_lag = int(s1_max_lag)
        self.s1_zs_window = max(1, int(s1_zs_window))
        self.s3_zs_window = max(1, int(s3_zs_window))
        self.s1_allow_no_zs = bool(s1_allow_no_zs)
        self.enable = set(enable)
        self.last_s1: dict | None = None
        self.n_hit = {S1: 0, S2: 0, S3: 0}
        # ⚠ 踩过的坑(2026-09-28): 一个冻结的向上笔会**持续存在很多根K线**, 而
        # `scan()` 的守卫只看"最后一根冻结笔是否向上" ⇒ 调用方若每根 bar 都调一次,
        # 同一个卖点会被重复报出(实测 spot BTC 4h: 真实 64 个 S1 被报成 490 个, 7.7×)。
        # 买点侧没有这个问题, 是因为去重锁放在调用方 `Level.scan_points()` 里。
        # 卖点的调用方是离场管理器, 不该指望它记得这件事 ⇒ 锁放这里, 与 scan_points 同义。
        self.n_final_seen: int = 0

    # ------------------------------------------------------------------
    def scan(self, bis: list, zss: list, macd, n_frozen: int | None = None
             ) -> list[Point]:
        """在"向上笔刚冻结"的事件上扫描全部卖点。返回命中列表(通常 0~1 个)。

        ⚠ 与买点同款的坑: `enable` 是**输出过滤器**, 不是"关闭检测"。S2 的判定
        依赖 S1 维护的 `last_s1` 状态机, 所以 `_s1`/`_s2` 必须**始终**运行;
        只有 S3 无状态依赖, 才可以真正跳过。

        ⚠ 去重锁: 同一个冻结笔只扫一次(与 `Level.scan_points` 同义)。语义上等价于
        调用方"只在笔变化时扫", 但把责任放在本类内部, 避免重复报信号。
        """
        n = (len(bis) - 1) if n_frozen is None else int(n_frozen)
        if n <= 0 or n == self.n_final_seen:
            return []
        self.n_final_seen = n
        if not _last_up_leg_frozen(bis, n):
            return []
        p2 = self._s2(bis, n)
        p3 = self._s3(bis, n, zss) if S3 in self.enable else None
        p1 = self._s1(bis, n, zss, macd)
        out: list[Point] = []
        if p2 is not None and S2 in self.enable:
            out.append(p2)
        if p3 is not None:
            out.append(p3)
        if p1 is not None and S1 in self.enable:
            out.append(p1)
        for p in out:
            self.n_hit[p.kind] += 1
        return out

    # ------------------------------------------------------------------
    def _s1(self, bis: list, n: int, zss: list, macd) -> Point | None:
        """第一类卖点: 上涨背驰。

        有中枢语境时用原文口径: 比较"进入中枢的上涨段"与"离开中枢的上涨段"
        (盘整背驰); 若前一个中枢整体更低, 则是**趋势背驰**。
        无可用中枢语境时退化为经典的 MACD 顶背驰。
        """
        last = n - 1
        c = bis[last]
        if c.direction < 0:
            return None
        # ---- 路径 1: 有中枢语境 ----
        zi_lo = max(0, len(zss) - self.s1_zs_window)
        for zi in range(len(zss) - 1, zi_lo - 1, -1):
            B = zss[zi]
            if last - int(B.i1) > self.s1_max_lag:
                continue
            # 镜像判据: 必须涨破中枢"触及过的最高点"(GG), 而不是只涨到中枢区间上沿之上。
            # 否则中枢内部的正常震荡都会被判成背驰 —— 参照段(进入中枢的大涨)天然比
            # 中枢内的小波动力度大, 背驰判据会退化成恒真。
            if c.high <= B.gg:
                continue
            a_i = _enter_leg(bis, n, B, +1)
            if a_i is None or a_i >= last:
                continue
            a = bis[a_i]
            if a.amp <= c.amp * 0.15:
                continue
            r = leg_beichi(macd, a, c, +1, self.beichi_tol, self.beichi_strict)
            if not r.ok:
                continue
            trend = any(int(A.leave_i) >= 0 and B.zd > A.zg
                        for A in zss[:zi] if A.i != B.i)
            self._remember_s1(float(c.high), last, B.i)
            return Point(S1, "exit", float(c.high), last, B.i, B.level, r,
                         float(c.high), [
                             f"涨破中枢#{B.i}[{B.zd:.6g},{B.zg:.6g}]触及最高{B.gg:.6g}"
                             f" → 离开段高点{c.high:.6g}",
                             f"参照段=进入中枢的上涨段(高点{a.high:.6g}), {r.why}",
                             ("上涨趋势中的趋势背驰" if trend else "盘整背驰"),
                             r.short() + " —— 一卖, 持有中的多头应在此离场",
                         ], feats=_s1_feats(r, path="zs", lag=last - a_i,
                                            amp_a=float(a.amp), amp_c=float(c.amp),
                                            zs_break=(float(c.high) - float(B.gg))
                                            / float(B.gg) if B.gg else 0.0,
                                            trend_beichi=bool(trend)))
        # ---- 路径 2: 无中枢语境的 MACD 顶背驰 ----
        if self.s1_allow_no_zs and last >= 4:
            a = bis[last - 2]
            amp_ok = bool(a.amp > 0 and 0.25 <= c.amp / a.amp <= 4.0)
            if a.direction > 0 and amp_ok and c.high > a.high:
                r = leg_beichi(macd, a, c, +1, self.beichi_tol, self.beichi_strict)
                if r.ok:
                    self._remember_s1(float(c.high), last, -1)
                    return Point(S1, "exit", float(c.high), last, -1, 1, r,
                                 float(c.high), [
                                     f"与前一同向上涨段比较(高点{a.high:.6g} → {c.high:.6g}, 创新高)",
                                     f"两段幅度相当({a.amp:.6g} → {c.amp:.6g})",
                                     r.why,
                                     "MACD 顶背驰(无中枢语境) —— 一卖",
                                 ], feats=_s1_feats(r, path="macd", lag=2,
                                                    amp_a=float(a.amp),
                                                    amp_c=float(c.amp),
                                                    zs_break=0.0,
                                                    trend_beichi=False))
        return None

    # ------------------------------------------------------------------
    def _remember_s1(self, price: float, leg_i: int, zs_i: int) -> None:
        """记录"当前上涨浪的一卖基准"。

        只在**创新高**时刷新: 同一波上涨里后来的浅一卖不能覆盖前面更高的那个,
        否则二卖判定的参考高点会被压低, 把"其实已经创了一卖新高"误判成二卖。
        """
        if self.last_s1 is None or price > float(self.last_s1["price"]):
            self.last_s1 = {"price": price, "leg_i": leg_i,
                            "zs_i": zs_i, "kind": S1}

    # ------------------------------------------------------------------
    def _s2(self, bis: list, n: int) -> Point | None:
        """第二类卖点: 一卖之后第一次反抽不创新高。"""
        s1 = self.last_s1
        if not s1:
            return None
        lv = int(s1["leg_i"])
        if n - 1 - lv > self.s2_max_bi:
            self.last_s1 = None               # 机会过期
            return None
        start = lv + 2                        # lv+1 是下跌的笔, lv+2 起才是反抽
        hi = _rally(bis, start, float(s1["price"]), self.pull_bi, n, self.tol)
        if hi is None:
            return None
        if start + self.pull_bi - 1 != n - 1:
            return None                       # 必须是第一次反抽, 不能是第二次
        self.last_s1 = None                   # 二卖机会已用掉
        return Point(S2, "exit", hi, n - 1, int(s1["zs_i"]), 1, None, hi, [
            f"第一类卖点@{s1['price']:.6g} 已成立",
            f"其后第一次反抽高点{hi:.6g} 未创一卖新高",
            "反抽结束 => 第二类卖点",
        ], feats={
            "s1_lag_bi": int(n - 1 - lv),
            "retest_pct": (((float(hi) / float(s1["price"])) - 1.0)
                           if s1["price"] else 0.0),
            "pull_bi": int(self.pull_bi),
            "s1_had_zs": int(int(s1["zs_i"]) >= 0),
        })

    # ------------------------------------------------------------------
    def _s3(self, bis: list, n: int, zss: list) -> Point | None:
        """第三类卖点: 向下离开中枢后, 回抽不升回中枢区间。

        关键: 离开段是**取得最低点的那根向下笔**(见 `_exit_leg` 的说明),
        回抽段自中枢最后一根内部笔之后起算。
        """
        last = n - 1
        lo_z = max(0, len(zss) - self.s3_zs_window)
        for zi in range(len(zss) - 1, lo_z - 1, -1):
            z = zss[zi]
            if int(z.leave_i) < 0:
                continue
            eb = _exit_leg(bis, n, z, -1)
            if eb is None or eb > last:
                continue
            ex = bis[eb]
            if ex.low >= z.zd:
                continue                      # 向下离开必须真的跌破 ZD
            hi = _rally(bis, eb + 1, z.zd, self.pull_bi, n, self.tol)
            if hi is None:
                continue
            if eb + self.pull_bi != last:
                continue                      # 回抽必须是紧跟离开段的那一段
            return Point(S3, "exit", hi, last, z.i, z.level, None, hi, [
                f"向下离开中枢#{z.i}[{z.zd:.6g},{z.zg:.6g}]: 离开段低点{ex.low:.6g} < ZD",
                f"回抽{self.pull_bi}笔, 最高{hi:.6g} 未升回中枢下沿{z.zd:.6g}",
                "回抽结束 => 第三类卖点",
            ], feats={
                "zs_height_pct": ((float(z.zg) - float(z.zd)) / float(z.zd)
                                  if z.zd else 0.0),
                "below_zd_pct": (((float(z.zd) - float(hi)) / float(z.zd))
                                 if z.zd else 0.0),
                "exit_break_pct": (((float(z.zd) - float(ex.low)) / float(z.zd))
                                   if z.zd else 0.0),
                "exit_leg_amp": float(ex.amp),
                "pull_bi": int(self.pull_bi),
                "s3_zs_age": int(n - 1 - int(z.i1)),
            })
        return None
