"""三类买卖点与区间套 —— 把走势结构翻译成"可以下单的位置"。

原文定义(做多侧; 做空严格镜像):
  **第一类买点 B1**: 下跌趋势中出现背驰(最后一个中枢之后, 离开段力度小于进入段),
      该下跌段结束处即一买。它天然逆势, 是"抄底"性质的买点。
  **第二类买点 B2**: 第一类买点之后的**第一次次级别回调**, 其低点不跌破一买低点,
      回调结束处即二买。它是对一买有效性的第一次确认。
  **第三类买点 B3**: 一个次级别走势向上离开中枢, 随后以一个次级别走势回试,
      其低点**不跌回中枢区间**( > ZG), 回试结束处即三买。它是趋势确认后的买点。

本模块与旧引擎的关键差别(逐个对应旧实现的缺陷):
  1. B1 是**新增能力**: 旧内核没有背驰, 因此完全没有一买 —— 而"接刀"风险恰恰只能
     由一买来承担(小仓 + 宽止损), 而不是靠"均线多头排列"把超跌区一刀切掉。
  2. B3 不再是一条不等式: 旧实现只要求"回踩低点 > 前中枢上沿", 等于给追高发通行证;
     新实现要求 (a) 确实存在**向上离开中枢的走势**, (b) 回试段本身是**完整走势**
     (笔数下限可配 `pull_bi`, 默认 1 = 忠于原文的"一笔即次级别走势",
      设为 3 则要求回试构成线段级走势), (c) 回试低点不跌回 [ZD, ZG]。
  3. B2 是**独立定义**: 旧实现没有这个概念, 二买特征被混进 `ma_support`。

触发时机(与无未来函数直接相关):
  所有买点的"结束端"都是**向下笔**, 因此扫描统一挂在"向下笔刚刚被冻结"这个事件上
  (即 `bis[-1]` 是向上笔、`bis[:-1]` 以向下笔结尾)。确认代价是真实的:
  一个底分型要等到右侧合并K线出现、且反向分型间隔达标才能定型, 通常滞后 4~5 根
  合并K线 —— 这段滞后是缠论体系的固有成本, 不是实现缺陷。

区间套: `converge()` 把大级别的结构点位(如中枢上沿、前低)**投影到小级别的实际K线
区间**上, 供多级别联立时判定"小级别是否已经给出同向确认"。完整的递归背驰定位
(`nested_beichi`) 需要至少三个可用级别, 本模块提供接口与实现, 由策略层决定是否启用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .beichi import BeichiResult, leg_beichi

__all__ = ["Point", "PointScanner", "converge", "nested_beichi"]

B1, B2, B3 = "B1", "B2", "B3"


@dataclass
class Point:
    """一个可下单的结构点位。"""

    kind: str                 # B1 / B2 / B3
    side: str                 # long
    price: float              # 结构点位(极值) —— 止损的基准
    leg_i: int                # 触发的向下笔下标(在 bis 中)
    zs_i: int = -1            # 相关中枢下标(-1 表示不依赖中枢, 如 B2)
    level: int = 1
    beichi: BeichiResult | None = None
    pull_low: float = 0.0     # 回试/下跌段的最低点(与 price 同义, 保留语义)
    reasons: list[str] = field(default_factory=list)
    # 结构化特征(供事后建模/归因, **不参与任何判定**)。
    # 各 `_bX` 把自己的可量化信息塞进来, 经 Signal.feats 透传到 Trade.feats。
    # 用 dict 而不是加一长串具名字段: 三类买点的特征集不同(B1 有背驰四判据、
    # B2 有回调滞后笔数、B3 有中枢突破幅度), 且将来加特征零成本。
    feats: dict = field(default_factory=dict)

    def label(self) -> str:
        return {"B1": "买①背驰", "B2": "买②确认", "B3": "买③三买"}[self.kind]

    def __repr__(self) -> str:      # pragma: no cover
        return f"Point({self.kind}@{self.price:.6g} leg{self.leg_i} zs{self.zs_i})"


# --------------------------------------------------------------- 内部工具
def _b1_feats(r, path: str, lag: int, amp_a: float, amp_c: float,
              zs_break: float, trend_beichi: bool,
              vol_ratio: float = -1.0) -> dict:
    """把一买的可量化信息拍平成特征字典(**纯观测, 不参与判定**)。

    背驰四判据的力度比值是本轮最有信息量的候选特征: 它们是连续量,
    直接刻画"力度衰减了多少", 而不是"是否越过阈值"。
    `vol_ratio` 是后段均量/前段均量(见 `filters.VolCtx`); -1 = 未启用或数据缺失。
    """
    rt = r.ratios or {}
    def g(k: str, d: float = 0.0) -> float:
        v = rt.get(k, d)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return d
        return d if v > 1e8 else v          # 1e9 是"参照段面积≈0"的哨兵值
    br = rt.get("bars") or [0, 0]
    return {
        "beichi_score": int(r.score),
        "beichi_new_extreme": int(bool(r.new_extreme)),
        "ratio_area": g("area", 1.0), "ratio_per": g("per", 1.0),
        "ratio_peak": g("peak", 1.0), "ratio_hist": g("hist_peak", 1.0),
        "bars_a": int(br[0]) if len(br) > 1 else 0,
        "bars_c": int(br[1]) if len(br) > 1 else 0,
        "b1_path": path, "b1_lag": int(lag),
        "amp_a": amp_a, "amp_c": amp_c,
        "amp_ratio": (amp_c / amp_a) if amp_a else 0.0,
        "zs_break": float(zs_break),
        "trend_beichi": int(bool(trend_beichi)),
        "vol_ratio": float(vol_ratio),
    }


def _pullback(bis: list, start_i: int, ref: float, min_bi: int, n: int,
              tol: float = 0.0) -> float | None:
    """自 start_i 起的回试段是否成立, 成立则返回其最低点。

    要求: (1) 至少 min_bi 根冻结笔; (2) 最后一根是向下笔(回试已结束);
          (3) 回试全程最低点不破参考价(含容差)。

    n: 冻结笔数(bis[:n] 才是可用的)。显式传 n 而不是用 `bis[:-1]` 切片 ——
    长回测里每根K线都切一次会退化成 O(n²)。
    """
    if start_i < 0 or start_i >= n:
        return None
    after = bis[start_i:n]
    if len(after) < min_bi:
        return None
    if after[-1].direction > 0:          # 回试尚未结束
        return None
    lo = min(b.low for b in after)
    if lo < ref * (1.0 - tol):
        return None
    return float(lo)


def _last_down_leg_frozen(bis: list, n: int) -> bool:
    """触发条件: 最后一根冻结笔是向下笔(即一个下跌/回试段刚定型)。"""
    return n > 0 and bis[n - 1].direction < 0


def _exit_leg(bis: list, n: int, z, d: int) -> int | None:
    """中枢 z 的"离开段": 在 [z.i0, z.i1+1] 内, 沿方向 d 取得最极端值的那一笔。

    为什么不能直接用"第一个不重叠的笔": 价格离开中枢时, 通常先是**运动笔**从区间
    内部冲到区间外(它仍然与区间有重叠, 按定义算延伸), 紧接着的**回抽笔**才完全不重叠。
    若拿回抽笔当离开段, 方向就反了(下跌离开会拿到一根向上笔)。真正代表"离开力量"的,
    是按方向取得最极端值的那一笔 —— 这也是"进入段 vs 离开段"比较的语义所在。
    """
    lo = max(0, int(z.i0))
    hi = min(n - 1, int(z.i1) + 1)
    best, bestv = None, None
    for j in range(lo, hi + 1):
        b = bis[j]
        if b.direction != d:
            continue
        v = float(b.low) if d < 0 else float(b.high)
        if bestv is None or (v < bestv if d < 0 else v > bestv):
            best, bestv = j, v
    return best


def _enter_leg(bis: list, n: int, z, d: int) -> int | None:
    """中枢 z 的"进入段": 优先取中枢前一笔; 不存在(或方向不符)时取中枢内的首根同向笔。"""
    ent = int(z.enter_i)
    if 0 <= ent < n and bis[ent].direction == d:
        return ent
    lo = max(0, int(z.i0))
    hi = min(n - 1, int(z.i1))
    for j in range(lo, hi + 1):
        if bis[j].direction == d:
            return j
    return None


# ------------------------------------------------------------------ 扫描器
class PointScanner:
    """三类买点扫描器。状态化: 记住最近一次 B1, 用于 B2 判定。"""

    def __init__(self, pull_bi: int = 1, tol: float = 0.0,
                 beichi_tol: float = 0.02, beichi_strict: bool = False,
                 b2_max_bi: int = 20, b1_max_lag: int = 12,
                 b1_zs_window: int = 1, b3_zs_window: int = 2,
                 b1_allow_no_zs: bool = True,
                 b1_path: str = "both", b1_trend_adjacent: bool = False,
                 vol_gate: float = 0.0,
                 enable=(B1, B2, B3)) -> None:
        self.pull_bi = int(pull_bi)
        self.tol = float(tol)
        self.beichi_tol = float(beichi_tol)
        self.beichi_strict = bool(beichi_strict)
        self.b2_max_bi = int(b2_max_bi)     # B1 之后超过这么多笔则二买机会作废
        self.b1_max_lag = int(b1_max_lag)   # 中枢距当前超过这么多笔不再作为一买语境
        self.b1_zs_window = max(1, int(b1_zs_window))
        # 三买只看最近这么多个中枢。**性能与语义都要求有界**: 无界回看会让
        # 每次扫描都遍历全部历史中枢, 长回测直接退化成 O(n²)。
        self.b3_zs_window = max(1, int(b3_zs_window))
        self.b1_allow_no_zs = bool(b1_allow_no_zs)
        # ---- 2026-09-29 新增开关(默认值 = 改动前行为, 逐位不变) ----
        # b1_path: "both"(默认) / "zs_only" / "trend_only" / "macd_only"
        #   —— 一买两条路径的取舍。审计 A1 实测: 默认口径下 80.6% 的"一买"走的是
        #   **无中枢语境的 MACD 路径**, 原文口径的一买(中枢语境)不到两成。
        #   ⚠ 它是**判定**开关不是输出过滤: 被砍掉的路径既不出信号, 也不更新
        #   `last_b1` 状态机(B2 的判定基准)。这与 `enable` 的语义不同, 别混。
        self.b1_path = str(b1_path)
        # b1_trend_adjacent: True 时把"趋势背驰"从"是否走了中枢路径"改成
        #   原文口径 —— 只回看**紧邻的前一个**中枢, 要求它已结束且中枢下移。
        #   (默认 False = 保留审计 A1 指出的退化写法, 以便 A/B 可比。)
        self.b1_trend_adjacent = bool(b1_trend_adjacent)
        # vol_gate: >0 时启用"量能配合"判据 —— 要求 后段均量/前段均量 < vol_gate。
        #   0 = 关闭(默认)。量能视图由 `Level` 挂载(`scanner.vol = VolCtx(kl)`)。
        self.vol_gate = float(vol_gate)
        self.vol = None                     # filters.VolCtx | None
        self.enable = set(enable)
        self.last_b1: dict | None = None
        self.n_hit = {B1: 0, B2: 0, B3: 0}

    # ------------------------------------------------------------------
    def scan(self, bis: list, zss: list, macd, n_frozen: int | None = None
             ) -> list[Point]:
        """在"向下笔刚冻结"的事件上扫描全部买点。返回命中列表(通常 0~1 个)。

        bis: 完整笔列表(最后一根可为可延伸笔);
        zss: 全部已知中枢(含进行中的快照);
        macd: 已更新到当前bar的 MACD 实例;
        n_frozen: 冻结笔数(默认 len(bis)-1)。显式传 n 而不是切片 `bis[:-1]`,
                  否则长回测里每次扫描都复制一遍全部历史笔 ⇒ O(n²)。

        ⚠ `enable` 是**输出过滤器**, 不是"关闭检测": B2 的判定依赖 B1 维护的
        `last_b1` 状态机, 所以 `_b1`/`_b2` 必须**始终**运行, 只是不把结果放进
        返回值。否则"只做二买"会因为一买检测被整段跳过而永远拿不到 `last_b1`,
        退化成零信号(2026-09-27 实测: um 前 400 个标的 0 笔成交)。
        B3 无状态依赖, 才可以真正跳过。
        """
        n = (len(bis) - 1) if n_frozen is None else int(n_frozen)
        if not _last_down_leg_frozen(bis, n):
            return []
        # 顺序: 先 B2(要用到本次扫描之前的 last_b1), 再 B3, 最后 B1(可能更新 last_b1)
        p2 = self._b2(bis, n)
        p3 = self._b3(bis, n, zss) if B3 in self.enable else None
        p1 = self._b1(bis, n, zss, macd)
        out: list[Point] = []
        if p2 is not None and B2 in self.enable:
            out.append(p2)
        if p3 is not None:
            out.append(p3)
        if p1 is not None and B1 in self.enable:
            out.append(p1)
        for p in out:
            self.n_hit[p.kind] += 1
        return out

    # ------------------------------------------------------------------
    def _b1(self, bis: list, n: int, zss: list, macd) -> Point | None:
        """第一类买点: 下跌背驰。

        有中枢语境时用**原文口径**: 比较"进入中枢的下跌段"与"离开中枢的下跌段"
        (即 盘整背驰); 若前一个中枢整体更高, 则这是**趋势背驰**。
        没有可用中枢语境时退化为经典的 **MACD 底背驰**(与上一段同向下跌比力度),
        这是实务中最常用的背驰读法, 用 `b1_allow_no_zs` 控制开关。

        为什么不再要求"离开段恰是刚冻结的那一笔": 笔中枢非常密集(连续三笔必然重叠),
        中枢往往把"冲出区间的那根运动笔"也算作延伸, 真正不重叠的是紧随其后的回抽笔,
        拿它当离开段方向就反了。改为"当前下跌笔是否已跌到某中枢下沿之下"这个
        方向性更强的判据, 语义等价且稳定得多。
        """
        last = n - 1
        c = bis[last]
        if c.direction > 0:
            return None
        # ---- 路径 1: 有中枢语境 ----
        # 只看**最近的** 1 个中枢: 一买的语义是"从最近的中枢向下离开"。若允许回看更早
        # 的中枢, 会把"跨越了好几个中枢、价位完全不同的两段"拿来比力度 —— 那种比较
        # 没有结构意义(而且归一化也救不回参照段选错的问题)。
        zi_lo = max(0, len(zss) - self.b1_zs_window)
        for zi in (range(len(zss) - 1, zi_lo - 1, -1)
                   if self.b1_path in ("both", "zs_only", "trend_only") else ()):
            B = zss[zi]
            if last - int(B.i1) > self.b1_max_lag:
                continue
            # 必须跌破中枢"触及过的最低点"(DD), 而不是只跌到中枢区间下沿之下。
            # 否则中枢内部的正常震荡都会被判成背驰: 参照段(进入中枢的大跌)天然比
            # 中枢内的小波动力度大, 背驰判据会退化成恒真。
            if c.low >= B.dd:
                continue
            a_i = _enter_leg(bis, n, B, -1)
            if a_i is None or a_i >= last:
                continue
            a = bis[a_i]
            if a.amp <= c.amp * 0.15:
                continue                      # 参照段过小, 比较无意义
            r = leg_beichi(macd, a, c, -1, self.beichi_tol, self.beichi_strict)
            if not r.ok:
                continue
            trend = self._is_trend_zs(zss, zi, B)
            if self.b1_path == "trend_only" and not trend:
                continue                      # 只要原文口径的趋势背驰
            vr = self._vol_ratio(a, c)
            if not self._vol_ok(vr):
                continue                      # 量能未配合
            self._remember_b1(float(c.low), last, B.i)
            return Point(B1, "long", float(c.low), last, B.i, B.level, r,
                         float(c.low), [
                             f"跌破中枢#{B.i}[{B.zd:.6g},{B.zg:.6g}]触及最低{B.dd:.6g}"
                             f" → 离开段低点{c.low:.6g}",
                             f"参照段=进入中枢的下跌段(低点{a.low:.6g}), {r.why}",
                             ("下跌趋势中的趋势背驰(相邻中枢下移)" if trend else "盘整背驰"),
                             r.short() + " —— 一买天然逆势, 按小仓宽止损处理",
                         ], feats=_b1_feats(r, path="zs", lag=last - a_i,
                                           amp_a=float(a.amp), amp_c=float(c.amp),
                                           zs_break=(float(c.low) - float(B.dd))
                                           / float(B.dd) if B.dd else 0.0,
                                           trend_beichi=bool(trend),
                                           vol_ratio=(vr if vr is not None else -1.0)))
        # ---- 路径 2: 无中枢语境的 MACD 底背驰 ----
        if (self.b1_allow_no_zs and self.b1_path in ("both", "macd_only")
                and last >= 4):
            a = bis[last - 2]
            # 两段幅度必须可比: 否则会拿"一路大跌"与"中枢内的小抖动"比力度,
            # 面积比必然是小数, 背驰判据退化成恒真 —— 这是最容易埋进去的假信号。
            amp_ok = bool(a.amp > 0 and 0.25 <= c.amp / a.amp <= 4.0)
            if a.direction < 0 and amp_ok and c.low < a.low:
                r = leg_beichi(macd, a, c, -1, self.beichi_tol, self.beichi_strict)
                if r.ok:
                    vr = self._vol_ratio(a, c)
                    if not self._vol_ok(vr):
                        return None
                    self._remember_b1(float(c.low), last, -1)
                    return Point(B1, "long", float(c.low), last, -1, 1, r,
                                 float(c.low), [
                                     f"与前一同向下跌段比较(低点{a.low:.6g} → {c.low:.6g}, 创新低)",
                                     f"两段幅度相当({a.amp:.6g} → {c.amp:.6g})",
                                     r.why,
                                     "MACD 底背驰(无中枢语境) —— 小仓宽止损",
                                 ], feats=_b1_feats(r, path="macd", lag=2,
                                                    amp_a=float(a.amp),
                                                    amp_c=float(c.amp),
                                                    zs_break=0.0,
                                                    trend_beichi=False,
                                                    vol_ratio=(vr if vr is not None
                                                               else -1.0)))
        return None

    # ------------------------------------------------------------------
    def _is_trend_zs(self, zss: list, zi: int, B) -> bool:
        """中枢 B 是否处于"趋势背驰"语境。

        原文: 趋势背驰的结构是 `a + A + b + B + c`, A 与 B 必须**相邻**、同向、
        且**不重叠**(第二个中枢整体更低/更高)。盘整背驰则是同一中枢内前后两段相比。

        默认(False)保留改动前的写法 —— `any(...)` 扫**全部历史中枢**。审计 A1 实测
        它与 `path=='zs'` 的重合度 99.94%(差 8 笔 / 64,639 笔), 即 `trend` 已经退化
        成"是否走了中枢路径"这个同义反复。

        `b1_trend_adjacent=True` 时改回原文口径: 只回看**紧邻的前一个**中枢, 并要求
        它已结束(`leave_i >= 0`)、在时间上不重叠(`A.i1 < B.i0`)且中枢下移(`B.zg < A.zd`)。
        """
        if not self.b1_trend_adjacent:
            return any(int(A.leave_i) >= 0 and B.zg < A.zd
                       for A in zss[:zi] if A.i != B.i)
        if zi - 1 < 0:
            return False
        A = zss[zi - 1]
        return bool(int(A.leave_i) >= 0 and int(A.i1) < int(B.i0) and B.zg < A.zd)

    # ------------------------------------------------------------------
    def _vol_ratio(self, a, c) -> float | None:
        """后段均量 / 前段均量(`filters.VolCtx`)。未启用或区间无根 ⇒ None。"""
        v = self.vol
        if v is None or a is None or c is None:
            return None
        return v.ratio(int(a.start_ts), int(a.end_ts),
                       int(c.start_ts), int(c.end_ts))

    def _vol_ok(self, vr: float | None) -> bool:
        """量能判据。关闭(vol_gate<=0)或量能不可得 ⇒ 放行(不因缺数据拒单)。"""
        if self.vol_gate <= 0 or vr is None:
            return True
        return bool(vr < self.vol_gate)

    def _remember_b1(self, price: float, leg_i: int, zs_i: int) -> None:
        """记录"当前下跌浪的一买基准"。

        只在**创新低**时刷新: 同一波下跌里后来的浅一买不能覆盖前面更深的那个,
        否则二买判定的参考低点会被抬高, 把"其实已经破了一买低点"误判成二买。
        """
        if self.last_b1 is None or price < float(self.last_b1["price"]):
            self.last_b1 = {"price": price, "leg_i": leg_i,
                            "zs_i": zs_i, "kind": B1}

    # ------------------------------------------------------------------
    def _b3(self, bis: list, n: int, zss: list) -> Point | None:
        """第三类买点: 向上离开中枢后, 回试不跌回中枢区间。

        关键: 离开段是**取得最高点的那根向上笔**(见 `_exit_leg` 的说明),
        回试段自中枢最后一根内部笔之后起算。

        只回看最近 `b3_zs_window` 个中枢: 语义上三买说的就是"刚被向上离开的那个
        中枢", 回看几个月前的中枢没有意义; 实现上这也把每次扫描的代价从
        O(全部历史中枢) 压到 O(窗口) —— 无界回看会让长回测退化成 O(n²)。
        """
        last = n - 1
        lo_z = max(0, len(zss) - self.b3_zs_window)
        for zi in range(len(zss) - 1, lo_z - 1, -1):
            z = zss[zi]
            if int(z.leave_i) < 0:
                continue                      # 中枢尚未结束 => 还没形成"离开"
            eb = _exit_leg(bis, n, z, +1)
            if eb is None or eb > last:
                continue
            ex = bis[eb]
            if ex.high <= z.zg:
                continue                      # 向上离开必须真的越过 ZG
            lo = _pullback(bis, eb + 1, z.zg, self.pull_bi, n, self.tol)
            if lo is None:
                continue
            if eb + self.pull_bi != last:
                continue                      # 回试必须是紧跟离开段的那一段
            return Point(B3, "long", lo, last, z.i, z.level, None, lo, [
                f"向上离开中枢#{z.i}[{z.zd:.6g},{z.zg:.6g}]: 离开段高点{ex.high:.6g} > ZG",
                f"回试{self.pull_bi}笔, 最低{lo:.6g} 未跌回中枢上沿{z.zg:.6g}",
                "回试结束 => 第三类买点(旧实现只看'低点>ZG'这一个不等式)",
            ], feats={
                "zs_height_pct": ((float(z.zg) - float(z.zd)) / float(z.zd)
                                  if z.zd else 0.0),
                "above_zg_pct": (((float(lo) - float(z.zg)) / float(z.zg))
                                 if z.zg else 0.0),
                "exit_break_pct": (((float(ex.high) - float(z.zg)) / float(z.zg))
                                   if z.zg else 0.0),
                "exit_leg_amp": float(ex.amp),
                "pull_bi": int(self.pull_bi),
                "b3_zs_age": int(n - 1 - int(z.i1)),
            })
        return None

    # ------------------------------------------------------------------
    def _b2(self, bis: list, n: int) -> Point | None:
        """第二类买点: 一买之后第一次回调不破一买低点。"""
        b1 = self.last_b1
        if not b1:
            return None
        lv = int(b1["leg_i"])
        if n - 1 - lv > self.b2_max_bi:
            self.last_b1 = None               # 机会过期
            return None
        start = lv + 2                        # lv+1 是反弹向上的笔, lv+2 起才是回调
        lo = _pullback(bis, start, float(b1["price"]), self.pull_bi, n, self.tol)
        if lo is None:
            return None
        if start + self.pull_bi - 1 != n - 1:
            return None                       # 必须是第一次回调, 不能是第二次
        self.last_b1 = None                   # 二买机会已用掉
        return Point(B2, "long", lo, n - 1, int(b1["zs_i"]), 1, None, lo, [
            f"第一类买点@{b1['price']:.6g} 已成立",
            f"其后第一次回调低点{lo:.6g} 未跌破一买低点",
            "回调结束 => 第二类买点",
        ], feats={
            "b1_lag_bi": int(n - 1 - lv),
            "retest_up_pct": (((float(lo) / float(b1["price"])) - 1.0)
                              if b1["price"] else 0.0),
            "pull_bi": int(self.pull_bi),
            "b1_had_zs": int(int(b1["zs_i"]) >= 0),
        })


# ------------------------------------------------------------------ 区间套
def converge(klines: list[dict], ts0: int, ts1: int) -> list[dict]:
    """区间套的"投影"步骤: 取出小级别在 [ts0, ts1] 内的K线。

    大级别给出候选区间(如背驰段覆盖的时间范围), 小级别在该区间内递归寻找同向确认。
    """
    return [k for k in klines if ts0 <= int(k["ts"]) <= ts1]


def _lower_bound_attr(seq, n: int, attr: str, x: int, right: bool = False) -> int:
    """在 `seq[:n]` 上按属性做二分: 返回第一个 `attr > x`(right) / `attr >= x` 的下标。

    笔序列的 `start_ts` 与 `end_ts` 都单调递增, 所以可以二分。手写而不物化属性数组,
    是为了避免每次调用都产生一次 O(n) 拷贝 —— 那正是本函数要消除的代价。
    """
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        v = int(getattr(seq[mid], attr))
        if v < x or (right and v == x):
            lo = mid + 1
        else:
            hi = mid
    return lo


def nested_beichi(macd_hi, lo_ctx: dict, ts0: int, ts1: int, direction: int,
                  tol: float = 0.02) -> dict:
    """区间套的"递归确认"步骤(两级简化版)。

    lo_ctx 需提供: bis(小级别笔序列), macd(小级别 MACD)。
    在 [ts0, ts1] 内找出同向的两段小级别走势并做背驰比较;
    成立则返回 {'ok': True, 'low': ..., 'why': ...}, 否则 ok=False。

    选取口径: 与区间**时间重叠**的笔即可(`end_ts > ts0 且 start_ts < ts1`),
    不要求完全落在区间内部 —— 高一级走势的端点就是那个极值点, 对应的小级别笔
    往往**恰好结束在区间右端之外一点点**, 用"完全包含"会把最关键的那一笔
    (真正的背驰段)排除掉, 于是闸门退化成只在极少数情况下放行。

    性能: 老实现每次调用都 `bis[:-1]` 整表切片 + 整表扫描来找重叠窗口。5m 确认
    级别 6.6 年约 4 万根笔、一买候选 11 万个 ⇒ O(n²)(实测 um 单变体 443s,
    而其余变体 140s)。改为二分定位窗口, 单次 3.142s → 0.019s(166x), 结果逐字不变。

    说明: 原文的区间套是无限递归; 工程上递归深度受数据最小周期限制,
    这里实现"高级别候选 -> 次级别背驰"一次收敛, 是最常用也最稳的一级。
    """
    bis = lo_ctx["bis"]
    n = len(bis) - 1                      # 与老实现一致: 丢掉可延伸的最后一笔
    if n <= 0:
        return {"ok": False, "why": "次级别同向段不足两段(0)"}
    hi = _lower_bound_attr(bis, n, "start_ts", ts1)              # start_ts <  ts1
    lo = _lower_bound_attr(bis, n, "end_ts", ts0, right=True)    # end_ts   >  ts0
    downs = [b for b in bis[lo:hi] if b.direction < 0] if hi > lo else []
    if len(downs) < 2:
        return {"ok": False, "why": f"次级别同向段不足两段({len(downs)})"}
    a, c = downs[-2], downs[-1]
    r = leg_beichi(lo_ctx["macd"], a, c, direction, tol)
    if not r.ok:
        return {"ok": False, "why": r.why}
    lo_px = float(c.low) if direction < 0 else float(c.high)
    return {"ok": True, "low": lo_px, "range": [float(c.low), float(c.high)],
            "why": f"次级别背驰确认: {r.why}"}
