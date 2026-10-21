"""级别 —— 结构级别(递归) + 时间周期映射(工程近似)。

原文的级别是**递归定义**的, 与时间周期没有固定对应:
    L0 笔 → L1 线段 → L2 本级别走势(线段→中枢→走势) → L3 高一级走势 → ...
工程上做两件事:
  1. `Level` 把某一周期上的完整流水线封装成一个**流式对象**
     (包含处理 → 分型 → 笔 → 线段 → 中枢 → 走势类型 → MACD → 买卖点),
     一根一根喂K线, 状态原地更新, 无未来函数由构造保证;
  2. `LEVEL_TF` 给出"大级别定方向 / 操作级别出买卖点 / 确认级别做区间套"的
     时间周期三元组, 与旧实盘一致以保证 A/B 可比。

**级别映射的取舍(显式声明, 见 DESIGN §6.2)**:
  操作级别(um 15m / spot 1h)用**笔**作为中枢的次级别构件 —— 与旧引擎同口径,
  信号频率落在可交易区间; 大级别(um 1h / spot 4h)用**线段** —— 线段比笔稳得多,
  用它得到的"走势类型"才配得上"定方向"这个职责。二者不等价于原文的严格递归,
  但比旧引擎"完全不做走势类型判断"要接近原文一个数量级。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import data as D
from .bi import BiBuilder
from .ckline import KLineMerger
from .fenxing import scan_fenxing_at
from .filters import MA, VolCtx
from .macd import MACD
from .points import Point, PointScanner
from .risk import atr_at
from .seg import SegBuilder
from .trend import PENDING, TrendBuilder, TrendState
from .zhongshu import ZSBuilder, merge_extensions

__all__ = ["LEVEL_TF", "Level", "LevelSet"]

# market -> (大级别/定方向, 操作级别/出买卖点, 确认级别/区间套)
LEVEL_TF = {
    "um": ("1h", "15m", "5m"),
    "spot": ("4h", "1h", "15m"),
}

MIN_KLINES = {"op": 500, "trend": 200}

# 走势判定只看"最后一个已结束中枢之后"最近这么多根走势的价格极值(防 O(n²))
TAIL_MAX = 48


@dataclass
class Snapshot:
    """某一时刻的冻结视图(调试与报告用, 不参与判定)。"""

    tf: str = ""
    i: int = 0
    ts: int = 0
    close: float = 0.0
    n_bi: int = 0
    n_final: int = 0
    n_seg: int = 0
    n_zs: int = 0
    trend: str = PENDING
    trend_why: str = ""
    zss: list = field(default_factory=list)


class Level:
    """单级别的流式流水线。"""

    __slots__ = ("name", "tf", "tf_ms", "mode", "merger", "bib", "segb", "zb",
                 "tb", "macd", "scanner", "kl", "cur", "n_final_seen", "state",
                 "_n_bi", "ma", "vol", "vol_on", "zs_merge", "_zs_cache", "_zs_key")

    def __init__(self, name: str, tf: str, mode: str = "bi",
                 min_gap: int = 4, scanner_kw: dict | None = None,
                 macd_kw: dict | None = None,
                 ma_periods=None, vol_on: bool = False,
                 zs_merge: bool = False) -> None:
        self.name = name
        self.tf = tf
        self.tf_ms = D.TF_MS[tf]
        self.mode = mode                     # "bi" | "seg": 中枢的次级别构件
        self.merger = KLineMerger()
        self.bib = BiBuilder(min_gap)
        self.segb = SegBuilder()
        self.zb = ZSBuilder(1)
        self.tb = TrendBuilder()
        # macd_kw 缺省 = MACD(fast=12, slow=26, signal=9)(原文默认)。仅为
        # tools/macd_sweep.py 的**敏感性测试**开的口子; 默认路径与改动前逐位一致。
        self.macd = MACD(**(macd_kw or {}))
        self.scanner = PointScanner(**(scanner_kw or {}))
        self.kl = None
        self.cur = 0                         # 已喂入的根数
        self.n_final_seen = 0
        self._n_bi = 0                       # 上次见到的笔数(用于变更检测)
        self.state = TrendState()
        # ---- 2026-09-29 新增(全部默认关闭, 关闭时对象不创建、开销为零) ----
        # ma_periods: 收盘价均线周期, 如 (5, 10)。None = 不建均线(默认)。
        self.ma = MA(ma_periods) if ma_periods else None
        # vol_on: 挂载成交量前缀和视图, 供 `scanner.vol_gate` 使用。
        self.vol_on = bool(vol_on)
        self.vol = None
        # zs_merge: 开启后 `zss()` / 走势判定使用 `zhongshu.merge_extensions`
        # 的结果(相邻同级别同向且区间重叠的中枢 => 合并为高一级中枢)。
        # ⚠ 会改变中枢数 ⇒ 买卖点全变 ⇒ 与所有历史结论不可比, 只作独立实验。
        self.zs_merge = bool(zs_merge)
        self._zs_cache: list | None = None
        self._zs_key = -1

    # ------------------------------------------------------------------ 驱动
    def attach(self, kl) -> "Level":
        self.kl = kl
        if self.vol_on and kl is not None:
            self.vol = VolCtx(kl)
            self.scanner.vol = self.vol
        return self

    def advance(self, close_ts: int) -> int:
        """喂入所有"在 close_ts 之前已收盘"的K线。返回新增根数。

        `end_at` 已扣除本周期长度 ⇒ 未收盘的下一根不会被喂进来。
        """
        if self.kl is None:
            return 0
        upto = self.kl.end_at(close_ts, self.tf_ms)
        n0 = self.cur
        if upto > n0:
            for i in range(n0, upto):
                self._push(i)
            self.cur = upto
        return self.cur - n0

    def _push(self, i: int) -> None:
        hi, lo, cl, ts = self.kl.h[i], self.kl.l[i], self.kl.c[i], self.kl.ts[i]
        self.merger.push(i, ts, hi, lo)
        self.macd.update(ts, cl)
        if self.ma is not None:
            self.ma.push(cl)
        j = len(self.merger.cks) - 2
        if j >= 1:
            fx = scan_fenxing_at(self.merger.cks, j)
            if fx is not None:
                self.bib.push(fx)            # 重复推送同一分型是幂等的
        nb = len(self.bib.bis)
        if nb != self._n_bi:
            self._n_bi = nb
            nf = nb - 1                      # 冻结笔数(最后一笔仍可延伸)
            self.segb.update(self.bib.bis, nf)
            if self.mode == "bi":
                self.zb.update(self.bib.bis, nf)
                self._trend_push(self.bib.bis, nf)
            else:
                ns = len(self.segb.segs)
                self.zb.update(self.segb.segs, ns)
                self._trend_push(self.segb.segs, ns)

    # ------------------------------------------------------------------ 视图
    def legs(self) -> list:
        """中枢的次级别构件(已冻结)。mode='bi' → 笔; mode='seg' → 线段。"""
        return self.bib.final_bis if self.mode == "bi" else self.segb.segs

    def zss(self) -> list:
        return self.zb.all_zss()

    def zss_done(self) -> list:
        return self.zb.zss

    def _merged_done(self) -> list:
        """已结束中枢 —— 开启 `zs_merge` 时返回中枢扩展(级别上移)后的结果。

        缓存键是 `len(zb.zss)`: 已结束中枢一旦 append 就不再修改(见 `zhongshu.ZS`
        的构造), 所以长度不变 ⇒ 合并结果不变。没有这层缓存, `_trend_push` 会在
        **每次笔变化**时做一次 O(中枢数) 合并, 长回测直接退化成 O(n²)。
        """
        done = self.zb.zss
        if not self.zs_merge:
            return done
        if self._zs_key == len(done) and self._zs_cache is not None:
            return self._zs_cache
        self._zs_cache = merge_extensions(done)
        self._zs_key = len(done)
        return self._zs_cache

    def zss_view(self) -> list:
        """扫描/判定用的中枢列表(已结束 + 进行中快照的拼接, 见 `all_zss`)。"""
        if not self.zs_merge:
            return self.zb.all_zss()
        cur = self.zb.cur_snapshot()
        return self._merged_done() + ([cur] if cur is not None else [])

    def _trend_push(self, legs: list, n: int) -> None:
        """把"最后一个已结束中枢之后的价格区极值"交给走势判定。

        ⚠ 性能陷阱(踩过): 直觉写法是 `tail = legs[z1+1:n]`。但**中枢延伸时
        `zss[-1].i1` 固定在很早的位置**, tail 会随走势数无限增长 ⇒ 每次笔变化都做
        一次 O(n) 切片与求极值, 整轮退化成 O(n²)(实测单标的 >120s)。这里只取
        最近 `TAIL_MAX` 个走势 —— 判"是否已离开中枢"只需要看最近的动作, 语义不变。
        """
        zss = self._merged_done()             # 只把已结束中枢交给走势判定
        cur = self.zb.cur_snapshot()
        lo_i = 0
        if zss:
            lo_i = int(zss[-1].i1) + 1
        if n - lo_i > TAIL_MAX:
            lo_i = n - TAIL_MAX
        lh = ll = None
        if n > lo_i:
            lh = legs[lo_i].high
            ll = legs[lo_i].low
            for k in range(lo_i + 1, n):
                x = legs[k]
                if x.high > lh:
                    lh = x.high
                if x.low < ll:
                    ll = x.low
        self.state = self.tb.update(zss, cur, lh, ll)

    # ------------------------------------------------------------------ 判定
    def scan_points(self) -> list[Point]:
        """在"向下笔刚冻结"的事件上扫买卖点。同一个冻结笔只扫一次。

        前置判断"冻结的最后一笔是否为向下笔"必须放在 `all_zss()` 之前:
        `all_zss()` 是 O(中枢数) 的列表拼接, 而这里有一半的冻结笔是向上笔,
        提前返回能省掉一半的构造。
        """
        nf = self.bib.n_final
        if nf == self.n_final_seen or nf <= 0:
            return []
        self.n_final_seen = nf
        bis = self.bib.bis
        if bis[nf - 1].direction >= 0:
            return []
        return self.scanner.scan(bis, self.zss_view(), self.macd, nf)

    def atr(self, n: int = 14, win: int = 300):
        """操作级别最近 win 根(含当前根)的 Wilder ATR。"""
        if self.kl is None or self.cur <= 0:
            return None
        return atr_at(self.kl, self.cur - 1, n, win)

    def snapshot(self) -> Snapshot:
        i = max(0, self.cur - 1)
        # `n_zs` 报告**判定看到的中枢数**: 开了 `zs_merge` 就报合并后的。
        # 否则 `zs_merge` 即使生效也会在诊断里显示为"什么都没发生"。
        return Snapshot(
            tf=self.tf, i=i, ts=self.kl.ts[i] if self.kl and self.cur else 0,
            close=self.kl.c[i] if self.kl and self.cur else 0.0,
            n_bi=len(self.bib.bis), n_final=self.bib.n_final,
            n_seg=len(self.segb.segs), n_zs=len(self._merged_done()),
            trend=self.state.kind, trend_why=self.state.why, zss=self.zb.zss,
        )

    def ready(self, min_bars: int) -> bool:
        return self.kl is not None and self.kl.n >= min_bars


class LevelSet:
    """三级联立: 大级别(定方向) / 操作级别(出买卖点) / 确认级别(区间套)。"""

    def __init__(self, market: str, sym: str = "", spec=None,
                 scanner_kw: dict | None = None, trend_mode: str = "seg",
                 op_mode: str = "bi", use_confirm: bool = True,
                 op_extra: dict | None = None) -> None:
        t_tf, o_tf, c_tf = spec or LEVEL_TF[market]
        self.market = market
        self.sym = sym
        # op_extra: 操作级别的新增开关(ma_periods / vol_on / zs_merge),
        # 缺省空字典 ⇒ 与改动前逐位一致。
        extra = dict(op_extra or {})
        self.trend = Level("trend", t_tf, trend_mode)
        self.op = Level("op", o_tf, op_mode, scanner_kw=scanner_kw, **extra)
        self.confirm = Level("confirm", c_tf, "bi") if use_confirm else None
        self.use_confirm = bool(use_confirm)
        # 日线波动率视图(`filters.ATR1D`), 由 `engine.run_symbol` 在开启闸门时注入;
        # 默认 None ⇒ `Strategy` 侧零开销、逐位等价。
        self.atr1d = None
        # 市场状态/位置/联动过滤器视图(`market_filters`), 第四轮补全时注入;
        # 默认 None ⇒ 零开销、逐位等价。
        self.mviews = None
        self.loaded = False

    def load(self) -> "LevelSet":
        m, s = self.market, self.sym
        self.trend.attach(D.load(m, s, self.trend.tf))
        self.op.attach(D.load(m, s, self.op.tf))
        if self.confirm is not None:
            self.confirm.attach(D.load(m, s, self.confirm.tf))
        self.loaded = True
        return self

    def ok(self) -> bool:
        if not self.loaded:
            return False
        return (self.op.ready(MIN_KLINES["op"])
                and self.trend.ready(MIN_KLINES["trend"]))

    def advance(self, close_ts: int) -> None:
        self.trend.advance(close_ts)
        if self.confirm is not None:
            self.confirm.advance(close_ts)
        self.op.advance(close_ts)
