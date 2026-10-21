"""策略 —— 把"级别 + 结构 + 买卖点"组装成可下单的信号。

多级别联立的职责划分(与 DESIGN §3 的级别体系对应):
  大级别(level.trend)  : 定方向。走势类型 = 上涨趋势 / 盘整 / 下跌趋势 / 中阴。
  操作级别(level.op)   : 出买卖点。三类买点都产生在这里。
  确认级别(level.confirm): 做区间套。大级别给出候选区间, 小级别要求同向背驰
                         —— 把"抄底"的介入点从大级别的一段压到小级别的一个点。

**方向闸门(本策略的核心取舍, 直接针对旧引擎已证病根)**
旧引擎 `strategy.py` 的第 1 步是"均线多头排列"前置过滤:
    bull = ma5>ma10>ma20>ma60 and last_close>ma20
不通过直接 `return None`。这在结构上把**一买彻底排除**了 —— 一买必然发生在
均线空头排列的下跌末端。于是旧引擎只剩"追高型"的买点, 实测盈亏比 0.97~1.25
(设计 RR2)、止损被吃满 47~55%。

新策略按买卖点的**性质**分别设闸门, 而不是一刀切:
  * B1(逆势) —— **不限方向**。它本来就是下跌趋势末端的反转点; 真正的风控手段是
    "小仓 + 宽止损"(见 `risk.plan` 的 B1 特例)与可选的区间套确认, 而不是禁掉它。
  * B2/B3(顺势) —— 要求大级别不是下跌趋势。它们在下跌趋势里没有结构意义。
"中阴阶段"与"走势未完成"都视为**方向不明**: 顺势类买点放行(它们自带结构确认),
但会在信号里标注出来, 便于事后归因。

⚠ 本模块只做"筛选", 不做资金/名额/熔断 —— 那是组合层的事。A/B 的信号层口径
(候选池笔数加权毛期望)要求每个信号都被独立模拟到离场, 不受资金约束。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import risk
from .filters import ma_size_scale, ma_spread, ma_veto_ok, pos_scale
from .points import B1, B2, B3, Point
from .sell import S1
from .trend import DOWN, PENDING, RANGE, UP

__all__ = ["BookCfg", "Signal", "Strategy", "B1_DIR_PRESETS", "resolve_dir_gate",
           "EXIT_MODES", "PRODUCTION", "LEGACY", "ENHANCED", "recommended_spot"]

# 默认优先级: 同一根K线上同时命中多个买点时取哪一个。
# B2 是一买之后的第一次确认, 结构信息量最大; B3 是顺势确认; B1 逆势风险最高。
PRIORITY = (B2, B3, B1)

# 方向闸门: 买点类型 -> 允许的大级别走势类型(None = 不限)
DIR_GATE = {
    B1: None,                      # 一买天然逆势, 不设方向闸门
    B2: frozenset({UP, RANGE, PENDING}),
    B3: frozenset({UP, RANGE, PENDING}),
}

# B1 方向闸门预设(理论缺口审计 A2)。
#   原文: 一买 = **下跌趋势**中的背驰(趋势的最后一个中枢之后)。
#   默认实现完全不检查, 实测 33.0% 的"一买"发生在**上涨趋势**背景下 ——
#   成因是占 80.6% 的 MACD 路径只要求"与前一同向下跌段比、创新低、背驰",
#   上涨趋势里一次正常回调就满足全部三条。
#   `off`   = 不限(改动前行为, 也是全项目既有结论的口径)
#   `no_up` = 只排除上涨趋势(弱闸门, 保留 down/range/pending)
#   `down`  = 只允许下跌趋势(原文最严口径)
B1_DIR_PRESETS = {
    "off": None,
    "no_up": frozenset({DOWN, RANGE, PENDING}),
    "down": frozenset({DOWN}),
}

# 均线闸门模式(见 filters.ma_veto_ok): off / ma / close / both / size
MA_VETO_MODES = ("ma", "close", "both")

# 离场路径模式(2026-09-29 第二轮新增; 同日升级为默认 "stop")
#   "old"  = 追踪 3.0×risk + RR2 目标 + (struct_on 时)结构破位 —— 改动前行为
#   "stop" = 只有入场时算出的结构止损("灾难点"), 不追踪 / 不设目标 / 不跑结构破位
# 为什么需要它: 主引擎此前**无条件**跑 `manage()`, 于是 `sell_exit` 只能"叠加"在
# 机械离场之上。实测叠加后单笔净期望 ≈ 0 —— 追踪止损在浮盈 1R 后立即用
# `trail_mult × risk` 机械回撤, 中位持仓被压到 0.75 天, 结构判据(中位 3.8 天)
# 没有机会表态。卖点要生效必须能**替换**默认离场, 所以需要一个显式的路径开关。
EXIT_MODES = ("old", "stop")

# --------------------------------------------------------------- 策略档案
# 2026-09-29 **升级**: 把三轮回测里唯一有硬证据的一项固化为生产默认 ——
# 「结构止损 + 一卖」离场。证据见 `ENGINE_VARIANTS_2026-09-29.md` /
# `CHANGE_PLAN_2026-09-29.md`:
#   * `old`(改动前默认) 单笔净期望在 1h/2h/4h 上**全为负**
#     (−0.162% / −0.230% / −0.440%);
#   * `stop + S1` 是唯一单笔净 > 0.4% 的口径(1h +0.493% / 2h +0.426% /
#     4h +1.766%), 折算年化 +0.8% → +11.1%, vs 同池指数 0.11× → 1.50×;
#   * 其余 6 项改造(b1_dir / pending_out / vol_gate / pos_layer / zs_merge /
#     b1_path) 实测为负或零触发, 保持关闭; ma_mode="ma" 唯一略正但幅度落在
#     池基准 tf 不确定度带(±1.2pp)内 ⇒ 只作可选增强, 不设默认。
#
# ⚠⚠ 这是本项目**第一次改变默认行为**。此前铁律是"默认 = 改动前行为", 为的是
#    保住全部历史 A/B 的可比性。默认一变, 那些结论就再也不能靠裸 `BookCfg()`
#    复现, 所以改动前口径被**显式固化**成 `LEGACY`: 任何复现历史结论的代码
#    (老实验脚本 / 判决报告 / 回归测试)都必须写 `BookCfg.legacy(...)`。
PRODUCTION: dict = {"exit_mode": "stop", "sell_exit": (S1,)}
LEGACY: dict = {"exit_mode": "old", "sell_exit": ()}
# 可选增强: 再加两条均线(否决型 5/10)。**不设默认** —— 它只赢 0.7pp, 而同池
# 基准自身带 tf 依赖(同池同再平衡, 1h 读 +7.7% / 4h 读 +10.0%)⇒ 该幅度不可辨。
ENHANCED: dict = {**PRODUCTION, "ma_mode": "ma"}


def resolve_dir_gate(b1_dir: str = "off", pending_out: bool = False) -> dict:
    """由两个开关导出方向闸门字典。

    `pending_out=True` 把中阴(PENDING)移出 B2/B3 的放行集合 —— 对应审计 A3:
    `trend.TrendState.incomplete/zhongyin` 算出来了却从未被任何闸门读取, 而
    `strategy.DIR_GATE` 反过来把 PENDING 放行(实测 B2 有 24.7% 在中阴状态成交),
    与文档里"未完成走势不得作为操作依据"自相矛盾。
    """
    allow23 = (frozenset({UP, RANGE}) if pending_out
               else frozenset({UP, RANGE, PENDING}))
    return {B1: B1_DIR_PRESETS.get(b1_dir, None), B2: allow23, B3: allow23}


@dataclass
class BookCfg:
    """全书版策略配置。**默认值 = 生产口径**(2026-09-29 升级后)。

    默认离场 = 「结构止损 + 一卖 S1」(`exit_mode="stop"`, `sell_exit=(S1,)`)。
    这是三轮回测 74 个变体里唯一有硬证据的改造项, 也是本轮升级的全部内容。

    要复现**改动前的历史结论**(2026-09-29 之前的所有 A/B 报告), 必须显式使用
    `BookCfg.legacy(...)`; 裸 `BookCfg(...)` 从今天起不再等于历史口径。
    要打开唯一略正的可选增强(两条均线 5/10), 用 `BookCfg.enhanced(...)`。

    ⚠ 除离场口径外, 其余全部开关(方向闸门 / 一买路径 / 量能 / 仓位分层 /
      中枢扩展)**默认仍等于改动前行为** —— 它们实测为负或零触发, 一律关闭。
    """

    market: str = "um"
    kinds: tuple = (B1, B2, B3)
    priority: tuple = PRIORITY
    dir_gate: dict | None = None          # None = 由 b1_dir/pending_out 导出

    # 买卖点扫描器参数(透传 PointScanner)
    pull_bi: int = 1
    beichi_tol: float = 0.02
    beichi_strict: bool = False
    b2_max_bi: int = 20
    b1_max_lag: int = 12
    b1_zs_window: int = 1
    b3_zs_window: int = 2
    b1_allow_no_zs: bool = True

    # 区间套(确认级别)
    confirm_gate: bool = False     # 仅 B1 生效: 要求确认级别亦给出同向背驰
    confirm_tol: float = 0.02
    # 计算区间套结论但**不拦截** —— 用于采集"通过/未通过"两侧的样本。
    # ⚠ 只有 confirm_gate=False 时才有意义: 若闸门开着, 所有进场的 B1 都恒为"通过",
    #   特征就没有方差, 无法用来学"区间套是否预示更少止损"。
    # 代价: 需要加载并推进确认级别 ⇒ 单标的耗时约 2.4× (um all 132s → 324s)。
    record_confirm: bool = False

    # 风控覆盖(None = 用 risk.CALIB)
    atr_mult: float | None = None
    max_loss_pct: float | None = None
    rr: float | None = None
    notional: float | None = None
    b1_wide: bool = True
    stop_buf: str = "old"          # "old" 复刻旧引擎(被钳位) / "min" 原注释本意

    # 离场
    struct_on: bool = True
    trail_mult: float | None = None

    # ---- 2026-09-29 新增: 审计缺口开关(除最末两项外, 默认 = 改动前行为) ----
    # 方向闸门(审计 A2/A3)
    b1_dir: str = "off"            # off(不限) / no_up(排除上涨) / down(只要下跌)
    pending_out: bool = False      # True => B2/B3 闸门排除中阴(PENDING)
    # 一买路径(审计 A1): both / zs_only(要中枢语境) / trend_only(只要趋势背驰)
    #                  / macd_only(只要无中枢的 MACD 底背驰)
    b1_path: str = "both"
    b1_trend_adjacent: bool = False  # True => 趋势背驰改回"紧邻前一中枢下移"的原文口径
    # 量能(审计 B2): >0 时要求 后段均量/前段均量 < vol_gate
    vol_gate: float = 0.0
    # ---- 波动率上限闸门(2026-09-29 第三轮新增, 移植 chan-quant `app/vol.py`) ----
    # >0 时要求 **日线 ATR14% <= atr1d_cap**; 0 = 关闭(默认, 逐位等价)。
    # ⚠ 与上面的 `vol_gate` 是**两个完全不同的维度**: `vol_gate` 是**成交量比**
    #   (量能配合), `atr1d_cap` 是**日线波动率的绝对水平**。chan-quant 把后者
    #   也叫 `vol_gate` —— 名字撞车, 别混用(见 GAP_FROM_CHANQUANT_2026-09-29.md)。
    atr1d_cap: float = 0.0
    atr1d_n: int = 14
    # 两条均线(审计 B6): off / ma / close / both(否决型) / size(仓位型)
    ma_mode: str = "off"
    ma_fast: int = 5
    ma_slow: int = 10
    # 仓位分层(审计 B8): off / amp / ratio / kind
    pos_layer: str = "off"
    # 中枢扩展(审计 B3): True => 相邻同向重叠中枢合并为高一级中枢
    zs_merge: bool = False
    # 卖点离场接入主引擎(审计 B1 类): () / ("S1",) / ("S1","S2","S3")
    # ★ 升级默认 = ("S1",): 一卖。卖点**替换**默认离场(见 exit_mode), 不叠加。
    sell_exit: tuple = (S1,)
    # 离场路径: old(追踪+RR+结构破位) / stop(只认灾难点) —— 见 EXIT_MODES
    # ★ 升级默认 = "stop": `sell_exit` 要生效必须先关掉机械离场。
    exit_mode: str = "stop"

    # ---- 2026-09-29 第四轮: chan-quant 能力补全(全部默认关闭 = 逐位等价) ----
    # 见 `market_filters.py` / `guards.py`。这些是 chan-quant 有、chanbook 缺的
    # 研究维度, 移植口径已逐位自检(见 PARITY_2026-09-29.md)。
    # ⚠ 全部是"信号后置过滤", 默认 off; 开启才建视图对象 ⇒ 默认路径零开销。
    # 大盘状态闸门(chan-quant app/regime.py): bull/range 只做多, bear 停机
    regime_gate: bool = False
    # 过热带(app/overheat.py): 4h/1d 过热则拦截
    overheat_on: bool = False
    # 前高压制(app/resistance.py): 贴着前高/前低正下方则拦截
    resistance_on: bool = False
    # 斐波那契回撤深度(app/fib.py): 太浅/太深则拦截
    fib_on: bool = False
    # 联动度(app/correl.py): 高联动币方向须与大盘一致
    correl_on: bool = False
    # 多周期分形离场(app/nesting.py exit_fenxing): 入场后新形成的反向分型离场。
    # 这是 chan-quant 独有的**离场侧**候选(chanbook 之前只有 S1/S2/S3 三个)。
    # 默认关闭; 开启后作为独立离场分支(见 engine.run_symbol)。
    exit_fenxing: bool = False
    fx_min_bars_after: int = 1       # 分型右侧至少几根合并K线才算确认
    fx_require_bi: bool = False      # True 时要求反向笔已成立(更强确认)

    def __post_init__(self) -> None:
        if self.dir_gate is None:
            self.dir_gate = resolve_dir_gate(self.b1_dir, self.pending_out)
        self.sell_exit = tuple(self.sell_exit or ())
        if self.exit_mode not in EXIT_MODES:
            raise ValueError(f"exit_mode 必须属于 {EXIT_MODES}, 收到 {self.exit_mode!r}")
        if float(self.atr1d_cap) < 0.0:
            raise ValueError(f"atr1d_cap 不得为负, 收到 {self.atr1d_cap!r}")

    # ---- 档案构造器(2026-09-29 升级新增) ----
    @classmethod
    def legacy(cls, **over) -> "BookCfg":
        """**改动前口径**(2026-09-29 之前): 追踪 + RR 目标 + 结构破位, 不接卖点。

        复现历史 A/B 报告、判决文档、老实验脚本时**必须显式**用它 ——
        裸 `BookCfg(...)` 自升级之日起已不等于历史口径。
        """
        return cls(**{**LEGACY, **over})

    @classmethod
    def enhanced(cls, **over) -> "BookCfg":
        """生产口径 + 两条均线(否决型 5/10)。唯一略正的可选增强, 不设默认。"""
        return cls(**{**ENHANCED, **over})

    # ------------------------------------------------------------------
    def scanner_kw(self) -> dict:
        return {
            "pull_bi": self.pull_bi, "beichi_tol": self.beichi_tol,
            "beichi_strict": self.beichi_strict, "b2_max_bi": self.b2_max_bi,
            "b1_max_lag": self.b1_max_lag, "b1_zs_window": self.b1_zs_window,
            "b3_zs_window": self.b3_zs_window,
            "b1_allow_no_zs": self.b1_allow_no_zs, "enable": tuple(self.kinds),
            "b1_path": self.b1_path,
            "b1_trend_adjacent": self.b1_trend_adjacent,
            "vol_gate": self.vol_gate,
        }

    def ma_periods(self) -> tuple | None:
        """需要建均线时返回周期元组, 否则 None(不建对象)。"""
        if self.ma_mode in MA_VETO_MODES or self.ma_mode == "size":
            return (int(self.ma_fast), int(self.ma_slow))
        return None

    def vol_on(self) -> bool:
        return float(self.vol_gate) > 0.0

    def op_extra(self) -> dict:
        """传给操作级别 `Level` 的新增开关。"""
        return {"ma_periods": self.ma_periods(), "vol_on": self.vol_on(),
                "zs_merge": bool(self.zs_merge)}

    def market_filter_on(self) -> bool:
        """是否开启了任一"市场状态/位置/联动"过滤器(第四轮补全)。"""
        return bool(self.regime_gate or self.overheat_on or self.resistance_on
                    or self.fib_on or self.correl_on)


@dataclass
class Signal:
    sym: str
    side: str
    kind: str
    ts: int
    i: int
    entry: float
    stop: float
    target: float
    atr: float
    risk_pct: float
    notional: float
    trend: str = ""
    trend_label: str = ""
    counter: bool = False
    zs_i: int = -1
    beichi: str = ""
    confirm: str = ""
    reasons: list = field(default_factory=list)
    feats: dict = field(default_factory=dict)   # 结构化特征(透传 Point.feats)

    def to_dict(self) -> dict:
        return {
            "sym": self.sym, "side": self.side, "kind": self.kind, "ts": self.ts,
            "entry": self.entry, "stop": self.stop, "target": self.target,
            "atr": self.atr, "risk_pct": self.risk_pct, "notional": self.notional,
            "trend": self.trend, "counter": int(self.counter), "zs_i": self.zs_i,
            "beichi": self.beichi, "confirm": self.confirm,
            "why": " | ".join(self.reasons),
        }


class Strategy:
    """无状态判定器(除了可配的开关)。每个 bar 由引擎驱动一次。"""

    def __init__(self, cfg: BookCfg) -> None:
        self.cfg = cfg
        self.n_seen = {B1: 0, B2: 0, B3: 0}       # 结构层面命中数
        self.n_blocked = {B1: 0, B2: 0, B3: 0}    # 被方向闸门拦掉的数量
        self.n_ma_block = 0                       # 被均线闸门拦掉的数量(2026-09-29 新增)
        self.n_volcap_block = 0                   # 被波动率上限闸门拦掉的数量(第三轮新增)
        self.n_regime_block = 0                   # 被大盘状态闸门拦掉的数量(第四轮新增)
        self.n_overheat_block = 0                 # 被过热带拦掉的数量(第四轮新增)
        self.n_resistance_block = 0               # 被前高压制拦掉的数量(第四轮新增)
        self.n_fib_block = 0                      # 被斐波那契拦掉的数量(第四轮新增)
        self.n_correl_block = 0                   # 被联动度拦掉的数量(第四轮新增)
        self.n_reject = 0                         # 止损距离过大被风控拒掉

    # ------------------------------------------------------------------
    def pick(self, ls, pts: list[Point]) -> Signal | None:
        """从本 bar 命中的买卖点里挑一个可下单的。"""
        if not pts:
            return None
        order = {k: n for n, k in enumerate(self.cfg.priority)}
        cands = sorted(pts, key=lambda p: order.get(p.kind, 99))
        for p in cands:
            self.n_seen[p.kind] += 1
            sig = self._make(ls, p)
            if sig is not None:
                return sig
            self.n_blocked[p.kind] += 1
        return None

    # ------------------------------------------------------------------
    def _make(self, ls, p: Point) -> Signal | None:
        cfg = self.cfg
        tstate = ls.trend.state
        tkind = tstate.kind
        # ---- 方向闸门 ----
        allow = cfg.dir_gate.get(p.kind, None)
        if allow is not None and tkind not in allow:
            return None
        # 入场时刻 = 当前操作级别最后一根已收盘K线
        _i = int(ls.op.cur) - 1
        entry = float(ls.op.kl.c[_i])
        _close_ts = int(ls.op.kl.ts[_i]) + int(ls.op.tf_ms)
        # ---- 第四轮补全: 市场状态/位置/联动过滤器(全部默认关闭) ----
        # 位置在方向闸门之后、波动率闸门之前: 它们只依赖已收盘的多周期序列,
        # 与买卖点结构无关, 越早拦越省算力。视图由 engine 注入到 `ls.mviews`。
        mviews = getattr(ls, "mviews", None)
        if cfg.regime_gate:
            from .engine import _regime_at
            reg = _regime_at(ls.market, _close_ts)
            if reg == "bear":
                self.n_regime_block += 1
                return None
        if mviews is not None:
            if cfg.overheat_on and mviews["overheat"] is not None:
                blocked, why = mviews["overheat"].verdict(_close_ts, p.side)
                if blocked:
                    self.n_overheat_block += 1
                    return None
            if cfg.resistance_on and mviews["resistance"] is not None:
                blocked, why = mviews["resistance"].verdict(_close_ts, p.side, entry)
                if blocked:
                    self.n_resistance_block += 1
                    return None
            if cfg.fib_on and mviews["fib"] is not None:
                blocked, why, d = mviews["fib"].verdict(ls.op.kl, _i, p.side, entry)
                if blocked:
                    self.n_fib_block += 1
                    return None
            if cfg.correl_on and mviews["correl"] is not None:
                blocked, why = mviews["correl"].verdict(_close_ts, p.side)
                if blocked:
                    self.n_correl_block += 1
                    return None
        # ---- 波动率上限闸门(chan-quant `app/vol.py` 移植; 默认关闭) ----
        # 位置: 方向闸门之后、均线之前 —— 它只依赖"日线已收盘序列", 与买卖点结构
        # 无关, 越早拦越省算力。
        # 口径: 要求 **日线 ATR14% <= atr1d_cap**("平静优于狂躁")。
        # fail-open: 闸门关闭 / 日线不足 15 根 ⇒ 放行(与 chan-quant 一致)。
        if float(cfg.atr1d_cap) > 0.0:
            av = getattr(ls, "atr1d", None)
            if av is not None:
                _j = int(ls.op.cur) - 1
                _cts = int(ls.op.kl.ts[_j]) + int(ls.op.tf_ms)
                if av.ok(_cts, float(cfg.atr1d_cap)) is False:
                    self.n_volcap_block += 1
                    return None
        # ---- 两条均线(审计 B6): 否决型 / 仓位型 ----
        # ⚠ 与旧引擎被证伪的"四条均线多头排列闸门"完全不同 —— 这里只用两条均线,
        # 而且默认关闭。旧闸门 `ma5>ma10>ma20>ma60 and close>ma20` 会把**整个一买类别**
        # 排除掉(一买必然发生在均线空头排列的下跌末端), 那是"把入场整类删掉";
        # 两条均线只是"短期不再被压制"的弱确认。
        scale = 1.0
        if cfg.ma_mode != "off":
            ma = getattr(ls.op, "ma", None)
            if ma is not None:
                if cfg.ma_mode == "size":
                    scale *= ma_size_scale(ma_spread(ma, cfg.ma_fast, cfg.ma_slow))
                else:
                    ok = ma_veto_ok(ma, entry, cfg.ma_mode, cfg.ma_fast,
                                    cfg.ma_slow)
                    if ok is False:            # False 才拦; None(未就绪)放行
                        self.n_ma_block += 1
                        return None
        # ---- 区间套确认(仅 B1) ----
        # `confirm_gate` 决定是否**拦截**; `record_confirm` 只要求**算出结论**,
        # 两者分开是为了让数据集能采到"未通过区间套"的那一半样本(否则特征无方差)。
        conf_why = ""
        conf_ok = -1                      # -1 = 该买点不适用区间套(B2/B3)
        if (cfg.confirm_gate or cfg.record_confirm) and p.kind == B1:
            ok, conf_why = confirm_b1(ls, p, cfg.confirm_tol)
            conf_ok = int(bool(ok))
            if cfg.confirm_gate and not ok:
                return None
        # ---- 仓位分层(审计 B8) ----
        # ⚠ 用**连续比值**(amp_ratio / ratio_area)分档, 不能用 beichi_score ——
        # 审计 A4 已证明 score 在判据结构上退化为 {3,4}(4 分占 90.1%), 无分档能力。
        if cfg.pos_layer != "off":
            scale *= pos_scale(cfg.pos_layer, p.kind, p.feats)
        # ---- 风控: 止损/目标/仓位 ----
        a = ls.op.atr()
        pl = risk.plan(p.side, entry, float(p.price), a, cfg.market, p.kind,
                       atr_mult=cfg.atr_mult, max_loss_pct=cfg.max_loss_pct,
                       rr=cfg.rr, notional=cfg.notional, b1_wide=cfg.b1_wide,
                       stop_buf=cfg.stop_buf, nm_scale=scale)
        if not pl.ok:
            self.n_reject += 1
            return None
        feats = dict(p.feats)
        # ⚠ 价格量纲的特征必须归一到入场价: 否则 BTC 的 amp=1e4 与某小币的 amp=1e-3
        # 在模型眼里是两回事, 而它们表达的是同一件事(波动幅度)。归一后是"占入场价
        # 的比例", 跨标的可比。原字段删除以免误用。
        if entry:
            for k in ("amp_a", "amp_c", "exit_leg_amp"):
                v = feats.pop(k, None)
                if v is not None:
                    feats[k + "_pct"] = round(float(v) / entry, 6)
        feats["confirm_ok"] = conf_ok
        feats["confirm_why"] = conf_why
        feats["atr_pct"] = (float(a) / entry) if (a and entry) else 0.0
        feats["gap_entry_struct"] = ((entry - float(p.price)) / entry
                                     if entry else 0.0)
        return Signal(
            sym=ls.sym, side=p.side, kind=p.kind, ts=int(ls.op.kl.ts[ls.op.cur - 1]),
            i=ls.op.cur - 1, entry=entry, stop=pl.stop, target=pl.target, atr=pl.atr,
            risk_pct=pl.risk_pct, notional=pl.notional,
            trend=tkind, trend_label=tstate.label(),
            counter=bool(p.kind == B1),
            zs_i=int(p.zs_i), beichi=(p.beichi.short() if p.beichi else ""),
            confirm=conf_why, reasons=list(p.reasons), feats=feats,
        )


# --------------------------------------------------------------- 区间套
def confirm_b1(ls, p: Point, tol: float = 0.02) -> tuple[bool, str]:
    """区间套确认: 大级别(操作级别)给出 B1 候选区间, 要求确认级别同向背驰。

    候选区间 = 触发 B1 的那根向下笔覆盖的时间范围。确认级别在该区间内必须
    也能找到"两段同向下跌且力度衰减"的结构 —— 这就是原文区间套的工程落地:
    把介入点从大级别的一段压到小级别的一个点, 代价是再等一个小级别的确认。
    """
    from .points import nested_beichi
    cf = getattr(ls, "confirm", None)
    if cf is None:
        return True, "未加载确认级别"
    fb = ls.op.bib.bis
    if p.leg_i < 0 or p.leg_i >= len(fb):
        return True, "触发笔下标越界"
    leg = fb[p.leg_i]
    ctx = {"bis": cf.bib.bis, "macd": cf.macd}
    r = nested_beichi(cf.macd, ctx, int(leg.start_ts), int(leg.end_ts), -1, tol)
    if r.get("ok"):
        return True, str(r.get("why", "ok"))
    return False, str(r.get("why", "未确认"))


# --------------------------------------------------------------- 推荐配置
def recommended_spot(**over) -> BookCfg:
    """现货生产配置 —— 2026-09-29 升级后的**系统默认**口径。

    ⚠ 升级后本函数与裸 `BookCfg(market="spot", kinds=(B1, B2))` **逐位等价**
      (默认值已经是生产口径)。保留它有两个理由:
        ① 生产/实盘侧显式声明"我在用生产口径", 读代码时不必反查默认值;
        ② 历史文档与脚本已在引用它, 改名只会产生无谓 diff。

    升级内容(全部证据见 `CHANGE_PLAN_2026-09-29.md`):
      * `exit_mode="stop"` —— **必须**。主引擎原本无条件跑 `manage()`(追踪 +
        RR 目标) + 结构破位, 机械离场把中位持仓压到 0.75 天, 结构判据(中位
        3.83 天)没有机会表态。只开 `sell_exit` 而不动离场路径 = 空操作
        (实测: 单笔净 −0.156% vs 现行 −0.162%, 几乎不变)。
      * `sell_exit=(S1,)` —— 一卖。**卖点必须替换默认离场, 不能叠加。**

    ⚠ 不要顺手打开这几项(均实测为负或零): `b1_dir` / `pending_out` /
      `vol_gate` / `pos_layer` / `zs_merge`。`ma_mode="ma"` 可作可选增强 ——
      用 `BookCfg.enhanced()`(1h 折算年化 +11.1% → +11.8%, 但幅度在池基准
      tf 不确定度 ±1.2pp 内)。

    要复现改动前的历史结论, 用 `BookCfg.legacy(...)`。
    """
    kw = dict(market="spot", kinds=(B1, B2), **PRODUCTION)
    kw.update(over)
    return BookCfg(**kw)
