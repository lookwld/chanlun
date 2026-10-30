"""市场状态 / 位置 / 联动过滤 —— 移植 `chan-quant` 的 4 个研究维度。

2026-09-29 第四轮: 按"根据 chan-quant 的内容补全 chanbook"的要求, 把 chan-quant
`app/` 下 4 个**纯本地确定性**过滤器移植进来, 逐个与源实现做口径一致性自检:

  * **overheat**   (app/overheat.py)   多周期追高过滤: 4h/1d 已过热则拦截
  * **resistance** (app/resistance.py) 前高压制: 贴着 1h 前高/前低正下方则拦截
  * **fib**        (app/fib.py)        斐波那契回撤深度: 回调吃掉整段推进的多少
  * **correl**     (app/correl.py)     联动度: 高联动币方向须与 BTC/ETH 大盘一致

与 chan-quant 的差异(刻意为之, 全部是"适配本工程"而非"改口径"):
  1. **数据结构**: chan-quant 用 `list[dict]`(每根 bar 一个 dict), 本工程用
     `data.Kl` 列式数组。判据的**数值口径逐位一致**, 只是读取方式不同。
  2. **无未来函数由"查表"保证**: 这些过滤器依赖 1h/4h/1d **多周期**序列, 无法用
     操作级别的 `Level._push` 流式喂入。改为预计算 O(n) 后 `verdict(close_ts, ...)`
     二分查表, 天然只碰"已收盘"的根(与 `filters.ATR1D` 同一套路)。
  3. **chan-quant 的实盘包装**(`*_verdict(fetch, ...)` 带缓存/拉取/异常)全部**不移植**:
     那是执行层的事; 这里只移植**纯函数判据**, 回测与未来实盘共用同一份。

⚠ 全部**默认关闭**, 关闭时不创建对象、主流程零开销、逐位等价。
   fail-open 语义: 数据不足一律放行(宁可放过, 不可误杀)。
"""

from __future__ import annotations

from array import array
from bisect import bisect_right
from math import sqrt

from . import data as D

__all__ = ["Overheat", "Resistance", "Fib", "Correl", "fenxing_exit"]

MS_1H = D.MS_HOUR
MS_4H = 4 * D.MS_HOUR
MS_1D = D.MS_DAY


def _rsi_val(ag: float, al: float) -> float:
    if al == 0.0:
        return 100.0 if ag > 0 else 50.0
    return 100.0 - 100.0 / (1.0 + ag / al)


# ================================================================== overheat
class Overheat:
    """多周期追高过滤(移植 app/overheat.py)。查表模式, 无未来函数。

    四因子(默认阈值与 chan-quant config 一致):
      * 日线收盘偏离 MA20  > max_bias_1d (0.15)   —— 大级别乖离, 追高主判据
      * 日线最近 3 根累计涨幅 > max_gain_3d (0.18) —— 短线透支
      * 4h RSI(14)          > max_rsi_4h  (72)    —— 中周期超买
      * 4h 收盘偏离 MA20    > max_bias_4h (0.10)  —— 中周期乖离
    做多查超买, 做空查超卖(严格镜像)。任一触发 => 拦截。
    """

    __slots__ = ("ts1d", "ts4h", "bias_1d", "gain_3d", "rsi_4h", "bias_4h", "cfg")

    def __init__(self, kl_1d: "D.Kl", kl_4h: "D.Kl", cfg: dict | None = None) -> None:
        c = dict(cfg or {})
        self.cfg = {
            "max_bias_1d": float(c.get("max_bias_1d", 0.15)),
            "max_gain_3d": float(c.get("max_gain_3d", 0.18)),
            "max_rsi_4h": float(c.get("max_rsi_4h", 72.0)),
            "max_bias_4h": float(c.get("max_bias_4h", 0.10)),
        }
        self.ts1d = kl_1d.ts
        self.ts4h = kl_4h.ts
        # ---- 日线: 偏离 MA20 与 3 根累计涨幅 ----
        n = kl_1d.n
        bias = array("d", [0.0]) * n
        gain3 = array("d", [0.0]) * n
        c1 = kl_1d.c
        s = 0.0
        for i in range(n):
            s += c1[i]
            if i >= 20:
                s -= c1[i - 20]
            if i >= 19:
                ma20 = s / 20.0
                bias[i] = (c1[i] / ma20 - 1.0) if ma20 > 0 else 0.0
            if i >= 3 and c1[i - 3] > 0:
                gain3[i] = c1[i] / c1[i - 3] - 1.0
        self.bias_1d = bias
        self.gain_3d = gain3
        # ---- 4h: RSI(14) 与偏离 MA20 ----
        n = kl_4h.n
        rsi = array("d", [50.0]) * n
        bias4 = array("d", [0.0]) * n
        c4 = kl_4h.c
        if n >= 15:
            ag = al = 0.0
            for i in range(1, 15):
                chg = c4[i] - c4[i - 1]
                ag += max(chg, 0.0)
                al += max(-chg, 0.0)
            ag, al = ag / 14.0, al / 14.0
            rsi[14] = _rsi_val(ag, al)
            for i in range(15, n):
                chg = c4[i] - c4[i - 1]
                ag = (ag * 13.0 + max(chg, 0.0)) / 14.0
                al = (al * 13.0 + max(-chg, 0.0)) / 14.0
                rsi[i] = _rsi_val(ag, al)
        s = 0.0
        for i in range(n):
            s += c4[i]
            if i >= 20:
                s -= c4[i - 20]
            if i >= 19:
                ma20 = s / 20.0
                bias4[i] = (c4[i] / ma20 - 1.0) if ma20 > 0 else 0.0
        self.rsi_4h = rsi
        self.bias_4h = bias4

    def verdict(self, close_ts: int, side: str) -> tuple[bool, str]:
        """返回 (blocked, reason)。fail-open: 数据不足 => 放行。"""
        up = (side == "long")
        g = self.cfg
        hits: list[str] = []
        jd = bisect_right(self.ts1d, int(close_ts) - MS_1D) - 1
        jh = bisect_right(self.ts4h, int(close_ts) - MS_4H) - 1
        if jd >= 23:                       # chan-quant 判 len(d_cl) >= 24
            b = self.bias_1d[jd]
            lim = g["max_bias_1d"]
            if (up and b > lim) or (not up and b < -lim):
                hits.append(f"日线偏离MA20 {b:+.1%}(阈值{lim:+.0%})")
            if jd >= 6:
                g3 = self.gain_3d[jd]
                lim = g["max_gain_3d"]
                if (up and g3 > lim) or (not up and g3 < -lim):
                    hits.append(f"3日累计涨幅 {g3:+.1%}(阈值{lim:+.0%})")
        if jh >= 24:                       # chan-quant 判 len(h_cl) >= 25
            r = self.rsi_4h[jh]
            lim = g["max_rsi_4h"]
            if (up and r > lim) or (not up and r < 100.0 - lim):
                hits.append(f"4h RSI(14) {r:.0f}(阈值{lim:.0f}/{100-lim:.0f})")
            b = self.bias_4h[jh]
            lim = g["max_bias_4h"]
            if (up and b > lim) or (not up and b < -lim):
                hits.append(f"4h偏离MA20 {b:+.1%}(阈值{lim:+.0%})")
        return (True, "; ".join(hits)) if hits else (False, "")


# ================================================================== resistance
class Resistance:
    """前高/前低压制(移植 app/resistance.py)。只依赖 1h 已收盘序列。

    做多: 前高 = 最近 lookback 根已收盘 1h 最高价(剔除最近 exclude_last 根);
      现价 < 前高 且 现价 >= 前高*(1-proximity_pct) => 拦截(贴着前高正下方);
      现价 >= 前高 => 放行(已突破)。
    """

    __slots__ = ("ts1h", "lookback", "exclude_last", "prox", "h_roll", "l_roll")

    def __init__(self, kl_1h: "D.Kl", cfg: dict | None = None) -> None:
        c = dict(cfg or {})
        self.lookback = int(c.get("lookback", 24))
        self.exclude_last = int(c.get("exclude_last", 3))
        self.prox = float(c.get("proximity_pct", 0.005))
        self.ts1h = kl_1h.ts
        n = kl_1h.n
        h, l = kl_1h.h, kl_1h.l
        self.h_roll = array("d", [0.0]) * n
        self.l_roll = array("d", [0.0]) * n
        for i in range(n):
            # chan-quant 窗口 = kl[len-exclude_last-lookback : len-exclude_last],
            # 其中 len = 已收盘根数 = i+1(第 i 根是"最后一根已收盘")。
            a = (i + 1) - self.exclude_last - self.lookback
            b = (i + 1) - self.exclude_last
            if b <= 0:
                continue
            a = max(0, a)
            self.h_roll[i] = max(h[a:b])
            self.l_roll[i] = min(l[a:b])

    def verdict(self, close_ts: int, side: str, entry: float) -> tuple[bool, str]:
        """返回 (blocked, reason)。fail-open: 数据不足 => 放行。"""
        if entry <= 0:
            return False, ""
        j = bisect_right(self.ts1h, int(close_ts) - MS_1H) - 1
        if j < 0 or self.h_roll[j] <= 0:
            return False, ""
        if side == "long":
            prev = self.h_roll[j]
            if entry >= prev:
                return False, ""
            gap = 1.0 - entry / prev
            if gap <= self.prox:
                return True, (f"1h前高 {prev:.6g} 正上方压制, 距其仅 {gap:.2%}"
                              f"(阈值{self.prox:.1%})")
        else:
            prev = self.l_roll[j]
            if entry <= prev:
                return False, ""
            gap = entry / prev - 1.0
            if gap <= self.prox:
                return True, (f"1h前低 {prev:.6g} 正下方支撑, 距其仅 {gap:.2%}"
                              f"(阈值{self.prox:.1%})")
        return False, ""


# ================================================================== fib
class Fib:
    """斐波那契回撤深度(移植 app/fib.py)。只依赖操作级别(1h)已收盘序列。

    腿: swing(窗口最高价 + 它之前的最低价) / bi(缠论笔, 复用 chanbook.bi)。
    depth = (H - entry)/(H - L)  做多;  镜像 (entry - L)/(H - L) 做空。
    只放行 [min_depth, max_depth]; 太浅/太深均拦截。
    """

    __slots__ = ("leg", "lookback", "min_leg_pct", "min_depth", "max_depth")

    def __init__(self, cfg: dict | None = None) -> None:
        c = dict(cfg or {})
        self.leg = str(c.get("leg", "swing"))
        self.lookback = int(c.get("lookback", 60))
        self.min_leg_pct = float(c.get("min_leg_pct", 0.01))
        self.min_depth = float(c.get("min_depth", 0.0))
        self.max_depth = float(c.get("max_depth", 1.0))

    def depth(self, kl: "D.Kl", n: int, side: str, entry: float) -> float | None:
        """在操作级别 K 线(已收盘 n 根)上算回撤深度。无有效腿 => None。"""
        if entry <= 0 or n < 5:
            return None
        w = max(0, n - self.lookback)
        h, l = kl.h, kl.l
        if self.leg == "bi":
            from .bi import build_bi
            from .ckline import merge_klines
            from .fenxing import find_fenxing
            cks = merge_klines([kl.bar(i) for i in range(w, n)])
            bis = build_bi(find_fenxing(cks))
            if not bis:
                return None
            want = 1 if side == "long" else -1
            if bis[-1].direction == want:
                b = bis[-1]
            elif len(bis) >= 2 and bis[-2].direction == want:
                b = bis[-2]
            else:
                return None
            hi, lo = b.high, b.low
        else:
            if n - w < 5:
                return None
            hs = h[w:n]
            ls = l[w:n]
            if side == "long":
                k = max(range(len(hs)), key=lambda i: hs[i])
                if k == 0:
                    return None
                hi = hs[k]
                lo = min(ls[:k + 1])
            else:
                k = min(range(len(ls)), key=lambda i: ls[i])
                if k == 0:
                    return None
                hi = max(hs[:k + 1])
                lo = ls[k]
        if hi <= lo or (hi - lo) / hi < self.min_leg_pct:
            return None
        if side == "long":
            return (hi - entry) / (hi - lo)
        return (entry - lo) / (hi - lo)

    def verdict(self, kl: "D.Kl", n: int, side: str, entry: float
                ) -> tuple[bool, str, float | None]:
        d = self.depth(kl, n, side, entry)
        if d is None:
            return False, "", None
        if d < self.min_depth:
            return True, f"回撤太浅 {d:.1%} < {self.min_depth:.1%}", d
        if d > self.max_depth:
            return True, f"回撤过深 {d:.1%} > {self.max_depth:.1%}", d
        return False, "", d


# ================================================================== correl
class Correl:
    """BTC/ETH 联动度(移植 app/correl.py)。查表: 用 1h 收益率序列。

    rho(皮尔逊) >= min_rho 视为"高联动币": 开仓方向必须与联动基准(BTC/ETH
    取 rho 高者)的 MA20 趋势一致。
    """

    __slots__ = ("window", "min_rho", "sym_kl", "btc_kl", "eth_kl")

    def __init__(self, sym_kl: "D.Kl", btc_kl: "D.Kl", eth_kl: "D.Kl",
                 cfg: dict | None = None) -> None:
        c = dict(cfg or {})
        self.window = int(c.get("window_hours", 168))
        self.min_rho = float(c.get("min_rho", 0.6))
        self.sym_kl, self.btc_kl, self.eth_kl = sym_kl, btc_kl, eth_kl

    def _idx(self, kl: "D.Kl", close_ts: int) -> int:
        return bisect_right(kl.ts, int(close_ts) - MS_1H) - 1

    def _returns(self, kl: "D.Kl", j: int) -> list[float]:
        out = []
        lo = max(0, j - self.window)
        for i in range(lo + 1, j + 1):
            if kl.c[i - 1]:
                out.append((kl.c[i] - kl.c[i - 1]) / kl.c[i - 1])
        return out

    def verdict(self, close_ts: int, side: str) -> tuple[bool, str]:
        """返回 (blocked, reason)。fail-open: 数据不足 => 放行。"""
        j = self._idx(self.sym_kl, close_ts)
        jb = self._idx(self.btc_kl, close_ts)
        je = self._idx(self.eth_kl, close_ts)
        if j < 20 or jb < 20 or je < 20:
            return False, ""
        rs = self._returns(self.sym_kl, j)
        rb = self._returns(self.btc_kl, jb)
        re_ = self._returns(self.eth_kl, je)
        rho_b = _pearson(rs, rb)
        rho_e = _pearson(rs, re_)
        if rho_b is None or rho_e is None:
            return False, ""
        ref, rho = ("BTC", rho_b) if rho_b >= rho_e else ("ETH", rho_e)
        if rho < self.min_rho:
            return False, ""          # 低联动 => 不受大盘约束
        ref_kl = self.btc_kl if ref == "BTC" else self.eth_kl
        ref_j = jb if ref == "BTC" else je
        m = _sma_last(ref_kl.c, ref_j, 20)
        if m is None:
            return False, ""
        trend = "up" if ref_kl.c[ref_j] > m else "down"
        if (side == "long" and trend == "up") or (side == "short" and trend == "down"):
            return False, ""          # 与大盘一致 => 放行
        return True, f"{ref}联动{rho:.2f}, 大盘{trend}与{side}相悖"


# ------------------------------------------------------------------ 公共数学
def _pearson(a: list[float], b: list[float]) -> float | None:
    n = min(len(a), len(b))
    if n < 20:
        return None
    a, b = a[-n:], b[-n:]
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    num = sum(x * y for x, y in zip(da, db))
    den = sqrt(sum(x * x for x in da) * sum(y * y for y in db))
    return num / den if den else None


def _sma_last(c: array, j: int, n: int) -> float | None:
    if j < n - 1:
        return None
    return sum(c[j - n + 1:j + 1]) / n


# ================================================================== 分形离场
def fenxing_exit(cks, entry_ts: int, side: str, min_bars_after: int = 1,
                 require_bi: bool = False, bis=None, ref_price: float | None = None,
                 min_profit: float = 0.0):
    """多周期分形离场(移植 app/nesting.py `exit_fenxing`)。

    做多找**入场后新形成**的顶分型, 做空找底分型 —— 原趋势被反向分型否证即离场。
    复用 chanbook 的 `fenxing`(合并K线 + 分型) 与 `bi`(笔), 数值口径与 chan-quant
    一致(那边用 `app.chan.analyze` 得到同样的合并K线/分型/笔)。

    返回 (hit, detail, extreme)。无入场后新分型 => (False, "", None)。

    `cks` 是**已收盘**的合并K线(调用方保证不含未收盘bar, 防盘中插针)。
    为 O(n) 效率: 用二分定位 `entry_ts` 之后的起始分型, 只扫尾部。
    """
    from .fenxing import BOTTOM, TOP, scan_fenxing_at
    if not cks or len(cks) < 8:
        return False, "K线不足", None
    kind = TOP if side == "long" else BOTTOM
    n = len(cks)
    # 二分: 第一个 ts0 >= entry_ts 的合并K线下标
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if cks[mid].ts0 < entry_ts:
            lo = mid + 1
        else:
            hi = mid
    start = lo
    tail = n - 1
    best = None
    # 只扫 [max(1, start-1), n-1) —— 保留 entry_ts 前一根作左边界
    for j in range(max(1, start - 1), n - 1):
        f = scan_fenxing_at(cks, j)
        if f is None or f.kind != kind:
            continue
        if tail - f.idx < min_bars_after:
            continue
        if int(f.ts) < int(entry_ts):
            continue
        if best is None or f.idx > best.idx:
            best = f
    if best is None:
        return False, "无入场后新分型", None
    ext = float(best.extreme)
    if ref_price is not None:
        if kind == TOP and ext < float(ref_price) * (1.0 + min_profit):
            return False, f"顶分型高点{ext:.6g}未达成本要求", None
        if kind == BOTTOM and ext > float(ref_price) * (1.0 - min_profit):
            return False, f"底分型低点{ext:.6g}未达成本要求", None
    if require_bi:
        if not bis:
            return False, "反向笔尚未成立", None
        need_end = BOTTOM if kind == TOP else TOP
        b = bis[-1]
        started_after = (b is not None
                         and int(b.start.ts) >= int(entry_ts))
        if b is None or b.end.kind != need_end or not started_after:
            return False, "反向笔尚未成立", None
    name = "顶分型" if kind == TOP else "底分型"
    return True, f"{name}确认@{ext:.6g}", ext
