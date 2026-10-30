"""多级别共振买点 —— 长周期与短周期**同时**给出买点。

使用者口径(2026-09-29):

> 对现货所有品种, **长周期与短周期出现共振**即为信号: 长周期或短周期上**必有一个
> 出现买①**, 另一个可以是买①, 也可以是买②或买③。

## 落地口径(三条, 显式声明)

1. **长周期 = 4h, 短周期 = 1h**。与 `level.LEVEL_TF["spot"] = ("4h","1h","15m")`
   的"大级别 / 操作级别"一致 —— 现货本来就是 4h 定方向、1h 出买卖点, 这里只是把
   15m 的"区间套确认"换成"1h 也出一个买点"的**共振**读法。两个级别都用
   `mode="bi"`(与 `tools/backtest_b12_tf.py` 的 4h 同口径, 可跨报告对照)。
   合约备用对 `("1h","15m")`, 本轮不跑。
2. **长周期给窗口、短周期给触发**。两个周期的 K 线边界不同(4h 落在 00/04/08/…,
   1h 每小时), "同一根 bar 上同时命中"没有定义。级别联立的原文读法本来就是
   "大级别给出候选区间, 小级别在该区间内给精确点"(见 `strategy.confirm_b1`)。
   所以: 长周期买点在 `l_win` 根(**长周期**)K线内保持有效, 短周期在有效期内
   给出买点 ⇒ 共振。
   **计时基准 = 长周期买点"被知道"的时刻**(它那根长周期K线的**收盘**), 不是
   笔端点、也不是那根K线的开盘。窗口因此从被知道起覆盖后面 `l_win` 根长周期K线。
   (计时用检测时刻而非笔端点, 是为了避免偷看未来 —— 笔端点在更早的时间点, 但
   那时 B1 还没被确认。)
3. **窗口可被价格提前终结**。短周期收盘价跌破长周期买点的结构低点
   (`Point.price` = 触发它的那根向下笔低点)即视为该长周期买点已被破坏。
   `invalidate=False` 可关掉做敏感性对照。

## 组合规则

`l_kind == B1 or s_kind == B1`(至少一侧是买①)。合法组合共 5 种:
`B1+B1` · `B1+B2` · `B1+B3` · `B2+B1` · `B3+B1`。
`B2+B2` / `B2+B3` / `B3+B3` **不是**信号(两侧都是顺势类, 没有一个逆势的买①)。

⚠ 本模块**不做方向闸门**。`strategy.DIR_GATE` 要求 B2/B3 的"高一级别不是下跌趋势",
在两级共振里没有自然的高一级别可用(短周期的高一级别是长周期, 长周期的高一级别
是日线)。与其硬塞一个, 不如把两个级别的走势状态原样记进 `Hit`, 事后按趋势分层看。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import data as D
from .level import MIN_KLINES, Level
from .points import B1, B2, B3, Point
from .strategy import PRIORITY
from .trend import PENDING

__all__ = ["RESO_TF", "DEFAULT_L_WIN", "Hit", "ResoScanner",
           "combo_ok", "load_pair", "scanner_kw"]

# market -> (长周期, 短周期)
RESO_TF = {
    "spot": ("4h", "1h"),
    "um": ("1h", "15m"),
}

# 长周期买点在这么多根(**长周期**)K线内保持有效
DEFAULT_L_WIN = 3

_PRIO = {k: n for n, k in enumerate(PRIORITY)}
_KIND_ALL = (B1, B2, B3)


def scanner_kw() -> dict:
    """两级都检测三类买点。

    ⚠ `enable` 是输出过滤器: B2 判定依赖 B1 维护的 `last_b1` 状态机, 所以
    B1/B2 必须**始终**运行(见 `PointScanner.scan` 的说明)。这里全开, 无此问题。
    """
    return {"pull_bi": 1, "beichi_tol": 0.02, "beichi_strict": False,
            "b2_max_bi": 20, "b1_max_lag": 12, "b1_zs_window": 1,
            "b3_zs_window": 2, "b1_allow_no_zs": True, "enable": _KIND_ALL}


def combo_ok(l_kind: str, s_kind: str) -> bool:
    """共振规则: 至少一侧是买①(第一类买点)。"""
    return l_kind == B1 or s_kind == B1


@dataclass
class Hit:
    """一次信号(共振, 或单级别对照口径下的一次买点)。"""

    sym: str = ""
    ts: int = 0                 # 触发 bar(短周期)的时间戳
    mode: str = "reso"          # reso | s_only | l_only
    l_tf: str = ""
    s_tf: str = ""
    l_kind: str = ""
    s_kind: str = ""
    l_ts: int = 0               # 长周期买点的**检出**时间(用检出而非笔终点, 无未来)
    l_price: float = 0.0        # 长周期买点的结构低点
    s_price: float = 0.0        # 短周期买点的结构低点(止损基准)
    entry: float = 0.0          # 触发 bar 的收盘价
    l_trend: str = PENDING
    s_trend: str = PENDING
    l_beichi: str = ""
    s_beichi: str = ""
    l_zsi: int = -1
    s_zsi: int = -1
    l_reasons: list = field(default_factory=list)
    s_reasons: list = field(default_factory=list)

    @property
    def combo(self) -> str:
        if self.mode == "s_only":
            return f"s:{self.s_kind}"
        if self.mode == "l_only":
            return f"l:{self.l_kind}"
        return f"{self.l_kind}_{self.s_kind}"

    @property
    def struct_price(self) -> float:
        """结构止损基准: 优先用短周期买点的低点(小级别定精确点)。"""
        return self.s_price if self.s_kind else self.l_price

    @property
    def risk_kind(self) -> str:
        """喂给 `risk.plan` 的买点类型(决定是否走 `b1_wide` 宽止损/减半名义额)。"""
        return self.s_kind or self.l_kind

    def label(self) -> str:
        if self.mode == "s_only":
            return f"{self.s_kind}@{self.s_tf} 单级别"
        if self.mode == "l_only":
            return f"{self.l_kind}@{self.l_tf} 单级别"
        return f"{self.l_kind}@{self.l_tf} + {self.s_kind}@{self.s_tf}"

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["combo"] = self.combo
        d["label"] = self.label()
        d["struct_price"] = self.struct_price
        return d


class ResoScanner:
    """长/短两级流式联立的共振检测器。

    用法(与 `Level` 同款, 一根一根喂)::

        rs = ResoScanner("spot", "BTCUSDT").load()
        for i in range(start, rs.skl.n):
            ts = int(rs.skl.ts[i])
            rs.advance_to(ts)
            for h in rs.poll(ts, rs.skl.c[i]):
                ...                        # 该 bar 收盘价成交
    """

    __slots__ = ("market", "sym", "l_tf", "s_tf", "l_win", "invalidate",
                 "long", "short", "lkl", "skl", "_active", "_fed_long", "loaded")

    def __init__(self, market: str = "spot", sym: str = "", tf_pair=None,
                 l_win: int = DEFAULT_L_WIN, invalidate: bool = True,
                 l_mode: str = "bi", s_mode: str = "bi") -> None:
        l_tf, s_tf = tf_pair or RESO_TF[market]
        self.market = market
        self.sym = sym
        self.l_tf, self.s_tf = l_tf, s_tf
        self.l_win = max(1, int(l_win))
        self.invalidate = bool(invalidate)
        kw = scanner_kw()
        self.long = Level("reso_long", l_tf, l_mode, scanner_kw=dict(kw))
        self.short = Level("reso_short", s_tf, s_mode, scanner_kw=dict(kw))
        self.lkl = None
        self.skl = None
        self._active: list[dict] = []
        self._fed_long = False
        self.loaded = False

    # ------------------------------------------------------------------ 装配
    def load(self) -> "ResoScanner":
        self.lkl = D.load(self.market, self.sym, self.l_tf)
        self.skl = D.load(self.market, self.sym, self.s_tf)
        self.long.attach(self.lkl)
        self.short.attach(self.skl)
        self.loaded = True
        return self

    def ok(self, l_min: int | None = None, s_min: int | None = None) -> bool:
        if not self.loaded or self.lkl is None or self.skl is None:
            return False
        return (self.lkl.n >= (l_min or MIN_KLINES["trend"])
                and self.skl.n >= (s_min or MIN_KLINES["op"]))

    # ------------------------------------------------------------------ 驱动
    def advance_to(self, bar_ts: int) -> None:
        """把两级都推进到「`bar_ts` 这根(短周期)K线**已收盘**」的**同一时刻**。

        ⚠️ 坑(2026-09-29 修): 长周期的判定时刻必须与短周期取**同一时刻**
        (`bar_ts + 短周期tf`), 不能写 `bar_ts + 长周期tf`。`Level.advance(close_ts)`
        喂的是「收盘 ≤ close_ts」的根, 所以后者等于宣称「开盘于 bar_ts 的那根长周期
        K线已经收盘」—— 而那根要到 `bar_ts + 4h` 才收盘。后果是长周期买点会**提前
        最多 1 整根长周期K线「被知道」**(4h/1h 实测领先 3 小时) ⇒ 偷看未来, 且让
        入场价系统性偏早(在下跌末端通常偏低)。自检见 `tools/reso_trace.py`。
        """
        now = bar_ts + self.short.tf_ms
        self._fed_long = self.long.advance(now) > 0
        self.short.advance(now)

    def _absorb_long(self) -> None:
        """把本步新检出的长周期买点收进「有效窗口」。

        `ts`    = 触发它的那根**长周期K线开盘**(用于展示 / 与长周期报告对齐)
        `known` = 那根长周期K线的**收盘时刻**, 即该买点真正「被知道」的时刻
                 —— 窗口从这一刻起算(见 `_prune`), 这才是可交易信息的时间戳。
        """
        for p in self.long.scan_points():
            if self.long.cur <= 0:
                break
            i = self.long.cur - 1
            self._active.append({
                "ts": int(self.lkl.ts[i]),
                "known": int(self.lkl.ts[i]) + self.long.tf_ms,
                "kind": p.kind, "price": float(p.price), "zsi": int(p.zs_i),
                "beichi": (p.beichi.short() if p.beichi else ""),
                "reasons": list(p.reasons),
                "trend": self.long.state.kind,
            })

    def _prune(self, bar_ts: int, bar_close: float) -> None:
        win = self.l_win * self.long.tf_ms
        now = bar_ts + self.short.tf_ms            # 本步的"当前时刻"
        keep = []
        for q in self._active:
            if now - q["known"] > win:
                continue                                  # 窗口到期(自"被知道"起算)
            if self.invalidate and bar_close < q["price"]:
                continue                                  # 结构已被破坏
            keep.append(q)
        self._active = keep

    def _pick_long(self, s_kind: str):
        """在有效窗口里挑配合的长周期买点。

        优先买①(逆势、更稀、是共振规则里的必需项), 同优先度取最新的。
        """
        cand = [q for q in self._active if combo_ok(q["kind"], s_kind)]
        if not cand:
            return None
        cand.sort(key=lambda q: (0 if q["kind"] == B1 else 1, -q["ts"]))
        return cand[0]

    # ------------------------------------------------------------------ 判定
    def step(self, bar_ts: int, bar_close: float) -> dict:
        """推进到该短周期 bar, 返回本 bar 的**原始事件**(不做判定)。

        为什么把"推进"和"判定"拆开: `Level.scan_points()` 是边沿触发 + 去重的
        (`n_final_seen`), **同一根 bar 第二次调用必然返回空**。所以"同一根 bar 上
        比较共振 / 单级别三种口径"必须共用一次扫描结果 —— 否则后两种口径会恒为
        零信号。返回的原料可以在 `decide()` 里反复使用。
        """
        n0 = len(self._active)
        self._absorb_long()
        new = self._active[n0:]
        self._prune(bar_ts, bar_close)
        alive = {id(q) for q in self._active}
        fresh_long = [q for q in new if id(q) in alive]
        return {"fresh_long": fresh_long, "short": self.short.scan_points()}

    def decide(self, bar_ts: int, bar_close: float, ev: dict,
               mode: str = "reso") -> list[Hit]:
        """**纯函数**: 由 `step()` 的原料判定信号, 返回 0~1 条。

        mode:
          * `reso`   —— 长/短两级共振(默认, 使用者口径);
          * `s_only` —— 只做短周期买点(**对照**: 共振相对单级别有没有增量);
          * `l_only` —— 只做长周期买点(同上, 另一侧)。
        后两种是**消融对照**, 用来回答"共振这个条件本身值不值钱"。
        """
        if mode == "l_only":
            fl = ev.get("fresh_long") or []
            if not fl:
                return []
            q = sorted(fl, key=lambda x: (0 if x["kind"] == B1 else 1))[0]
            return [self._mk_hit(bar_ts, bar_close, q, None, "l_only")]
        sps = ev.get("short") or []
        if not sps:
            return []
        sps = sorted(sps, key=lambda p: _PRIO.get(p.kind, 9))
        if mode == "s_only":
            return [self._mk_hit(bar_ts, bar_close, None, sps[0], "s_only")]
        for p in sps:
            q = self._pick_long(p.kind)
            if q is None:
                continue
            return [self._mk_hit(bar_ts, bar_close, q, p)]
        return []

    def poll(self, bar_ts: int, bar_close: float, mode: str = "reso") -> list[Hit]:
        """`step()` + `decide()` 的便捷封装(单口径用)。"""
        return self.decide(bar_ts, bar_close, self.step(bar_ts, bar_close), mode)

    def _mk_hit(self, bar_ts: int, bar_close: float, q: dict | None,
                p: Point | None, mode: str = "reso") -> Hit:
        return Hit(
            sym=self.sym, ts=int(bar_ts), mode=mode,
            l_tf=self.l_tf, s_tf=self.s_tf,
            l_kind=(q["kind"] if q else ""), s_kind=(p.kind if p else ""),
            l_ts=int(q["ts"]) if q else 0,
            l_price=float(q["price"]) if q else 0.0,
            s_price=float(p.price) if p else 0.0,
            entry=float(bar_close),
            l_trend=(q["trend"] if q else self.long.state.kind),
            s_trend=self.short.state.kind,
            l_beichi=(q["beichi"] if q else ""),
            s_beichi=((p.beichi.short() if p.beichi else "") if p else ""),
            l_zsi=(int(q["zsi"]) if q else -1),
            s_zsi=(int(p.zs_i) if p else -1),
            l_reasons=(list(q["reasons"]) if q else []),
            s_reasons=(list(p.reasons) if p else []),
        )

    # ------------------------------------------------------------------ 视图
    def pending(self) -> list[dict]:
        """长周期买点已成立、短周期还没给出买点的**观察名单**(当前有效窗口内)。"""
        return list(self._active)

    def short_atr(self):
        return self.short.atr()
