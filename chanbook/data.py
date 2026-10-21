"""行情读取 —— **只读**复用 `../chan-quant/data_archive/` 的币安归档K线。

严格边界(DESIGN §0 硬约束):
  * 本模块只 `open()` 归档 csv.gz, 绝不写入、绝不 import `chan-quant` 的任何代码;
  * 口径必须与旧引擎一致, A/B 才有意义 —— 因此列式存储、缺根不补、按实际 bar
    推进、`end_at()` 的"已收盘"判据都刻意与旧引擎保持同构。

归档布局:
    data_archive/
      um/k/<SYM>_{5m,15m,1h}.csv.gz        # 合约: 无 4h/1d, 现场聚合
      um/f/<SYM>_funding.csv.gz            # 合约资金费率
      spot/k/<SYM>_{15m,1h,4h,1d}.csv.gz   # 现货: 四档齐全
      all_symbols.txt / spot_symbols.txt
      universe_um.json / universe_spot.json   # {"YYYY-MM": [symbol, ...]}
列: ts,open,high,low,close,volume,quote_volume,trades
"""

from __future__ import annotations

import csv
import gzip
import json
import os
from array import array
from bisect import bisect_right
from datetime import datetime, timedelta, timezone

__all__ = [
    "MS_MIN", "MS_HOUR", "MS_DAY", "TF_MS", "Kl",
    "ARCH", "resample", "load", "load_funding", "symbols", "universe",
    "month_key", "ym_range", "fmt", "TZ",
]

TZ = timezone(timedelta(hours=8))
MS_MIN, MS_HOUR, MS_DAY = 60_000, 3_600_000, 86_400_000
# ⚠ `1M`(月线) 用 **31 天** 而不是"真实月长": 月长在 28~31 天之间浮动, 用 30 天会
# 在 2 月/31 天的月份上"提前宣布收盘" ⇒ 偷看未来。31 天是上界, 偏保守:
# 代价是月初那根要晚 0~3 天才被认作已收盘(对月线级别的窗口来说可忽略)。
# ⚠ `1w`/`3d`/`1M` 都是**非均匀周期**, `resample()` 的 `ts // tf_ms` 对齐对它们
# 不成立 —— 这三档必须直接读交易所的原始 K 线, 不能由更细周期聚合。`load()` 里
# 只有 `("4h","1d")` 会走 resample 兜底, 所以不会误触发。
TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000, "1h": MS_HOUR,
         "4h": 14_400_000, "1d": MS_DAY, "3d": 3 * MS_DAY,
         "1w": 7 * MS_DAY, "1M": 31 * MS_DAY}

_HERE = os.path.dirname(os.path.abspath(__file__))
ARCH = os.environ.get(
    "CHANBOOK_ARCH",
    os.path.join(os.path.dirname(os.path.dirname(_HERE)), "chan-quant", "data_archive"),
)


def fmt(ms: int, f: str = "%Y-%m-%d %H:%M") -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime(f)


def month_key(ms: int) -> str:
    return fmt(ms, "%Y-%m")


# ------------------------------------------------------------------ K线
class Kl:
    """连续K线(列式存储, 省内存)。ts 毫秒; 其余 float。

    与旧引擎同构的理由: A/B 要同口径。`bars()` 惰性构造整段 dict 列表并复用切片
    —— 长回测里每根 bar 都取窗口, 否则会产生千万级 dict 构造开销。
    """

    __slots__ = ("ts", "o", "h", "l", "c", "v", "qv", "n", "_dl")

    def __init__(self, path: str = "", rows=None):
        ts, o, h, l, c, v, qv = (array("q"), array("d"), array("d"),
                                 array("d"), array("d"), array("d"), array("d"))
        if rows is not None:
            for r in rows:
                ts.append(int(r[0]))
                o.append(float(r[1]))
                h.append(float(r[2]))
                l.append(float(r[3]))
                c.append(float(r[4]))
                v.append(float(r[5]) if len(r) > 5 else 0.0)
                qv.append(float(r[6]) if len(r) > 6 else 0.0)
        elif path and os.path.exists(path):
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                rdr = csv.reader(fh)
                next(rdr, None)
                for r in rdr:
                    if len(r) < 5:
                        continue
                    ts.append(int(r[0]))
                    o.append(float(r[1]))
                    h.append(float(r[2]))
                    l.append(float(r[3]))
                    c.append(float(r[4]))
                    v.append(float(r[5]) if len(r) > 5 else 0.0)
                    qv.append(float(r[6]) if len(r) > 6 else 0.0)
        self.ts, self.o, self.h, self.l, self.c, self.v, self.qv = ts, o, h, l, c, v, qv
        self.n = len(ts)
        self._dl = None

    def __len__(self) -> int:
        return self.n

    def bar(self, i: int) -> dict:
        return {"ts": self.ts[i], "open": self.o[i], "high": self.h[i],
                "low": self.l[i], "close": self.c[i], "volume": self.v[i],
                "quote_volume": self.qv[i]}

    def bars(self, a: int, b: int) -> list[dict]:
        a = max(0, a)
        if self._dl is None:
            self._dl = [self.bar(i) for i in range(self.n)]
        return self._dl[a:min(b, self.n)]

    def end_at(self, close_ts: int, tf_ms: int) -> int:
        """已收盘根数: 只含"开盘时刻 + 周期 <= 判定时刻"的根。

        不减去 tf_ms 会把尚未收盘的下一根也算进来 ⇒ 偷看未来。与旧引擎同义。
        """
        return bisect_right(self.ts, close_ts - tf_ms)


# ------------------------------------------------------------------ 聚合
def resample(base: Kl, tf_ms: int) -> Kl:
    """把更细周期聚合成 tf_ms(按 UTC 向下取整对齐)。末根可能不完整, 由 end_at 排除。"""
    out = Kl.__new__(Kl)
    ts, o, h, l, c, v, qv = (array("q"), array("d"), array("d"), array("d"),
                             array("d"), array("d"), array("d"))
    cur = -1
    for i in range(base.n):
        b = base.ts[i] // tf_ms
        if b != cur:
            ts.append(b * tf_ms)
            o.append(base.o[i])
            h.append(base.h[i])
            l.append(base.l[i])
            c.append(base.c[i])
            v.append(base.v[i])
            qv.append(base.qv[i])
            cur = b
        else:
            if base.h[i] > h[-1]:
                h[-1] = base.h[i]
            if base.l[i] < l[-1]:
                l[-1] = base.l[i]
            c[-1] = base.c[i]
            v[-1] += base.v[i]
            qv[-1] += base.qv[i]
    out.ts, out.o, out.h, out.l, out.c, out.v, out.qv = ts, o, h, l, c, v, qv
    out.n = len(ts)
    out._dl = None
    return out


_RS_CACHE: dict = {}


def load(market: str, sym: str, tf: str) -> Kl:
    """读某标的某周期; 归档缺档时用 1h 现场聚合(um 无 4h/1d)。"""
    p = os.path.join(ARCH, market, "k", f"{sym}_{tf}.csv.gz")
    if os.path.exists(p) or tf not in ("4h", "1d"):
        return Kl(p)
    key = (market, sym, tf)
    hit = _RS_CACHE.get(key)
    if hit is None:
        base = Kl(os.path.join(ARCH, market, "k", f"{sym}_1h.csv.gz"))
        hit = resample(base, TF_MS[tf])
        if len(_RS_CACHE) > 32:
            _RS_CACHE.clear()
        _RS_CACHE[key] = hit
    return hit


def load_funding(sym: str):
    """合约资金费率序列(ts, rate)。"""
    ts, rate = array("q"), array("d")
    p = os.path.join(ARCH, "um", "f", f"{sym}_funding.csv.gz")
    if os.path.exists(p):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            r = csv.reader(fh)
            next(r, None)
            for row in r:
                if len(row) < 2:
                    continue
                ts.append(int(row[0]))
                rate.append(float(row[1]))
    return ts, rate


# ------------------------------------------------------------------ 标的池
def symbols(market: str) -> list[str]:
    p = os.path.join(ARCH, "spot_symbols.txt" if market == "spot" else "all_symbols.txt")
    if not os.path.exists(p):
        p = os.path.join(ARCH, "all_symbols.txt")
    with open(p, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def universe(market: str, y2020: bool = False) -> dict:
    """时点标的池 {"YYYY-MM": [...]} —— 消除"用今天成交额选币"的存活偏差。"""
    name = ("universe_um_2020.json" if y2020 else
            "universe_um.json" if market == "um" else "universe_spot.json")
    p = os.path.join(ARCH, name)
    if not os.path.exists(p):
        return {}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def ym_range(start: str, end: str) -> list[str]:
    """['2020-01', ..., '2026-08']"""
    y0, m0 = (int(x) for x in start.split("-"))
    y1, m1 = (int(x) for x in end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out
