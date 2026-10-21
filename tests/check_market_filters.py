"""市场过滤器 + 熔断护栏自检 —— 2026-09-29 第四轮补全。

运行: python tests/check_market_filters.py

覆盖四类:
  A. **口径一致性**: chanbook.market_filters 与 chan-quant 源实现(app/*.py)在
     同一份真实归档数据上**逐位对拍**(这是"逐条对比"的证据核心);
  B. **无未来函数**: 截断一致性 —— 把数据截到某时刻, 之前的判定值不变;
  C. **fail-open**: 数据不足一律放行, 不误杀;
  D. **guards 熔断规则**: 连亏/日亏/标的冷却/方向保险丝, 统一 1h 计时。
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                     # noqa: E402
from chanbook import market_filters as MF                          # noqa: E402
from chanbook.guards import Guards                                 # noqa: E402

_PASS = 0
_FAIL = 0


def ck(cond: bool, msg: str) -> None:
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        print(f"  ✗ {msg}")


def _dicts(kl, upto):
    return [{"ts": kl.ts[i], "open": kl.o[i], "high": kl.h[i], "low": kl.l[i],
             "close": kl.c[i], "volume": kl.v[i]} for i in range(upto)]


# ================================================================ A. 口径一致性
def test_parity_overheat() -> None:
    """overheat 与 chan-quant app/overheat.overheat_from_klines 逐位一致。"""
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "chan-quant")))
    from app import overheat as OH
    kl1h, kl1d, kl4h = (D.load("spot", "BTCUSDT", t) for t in ("1h", "1d", "4h"))
    v = MF.Overheat(kl1d, kl4h)
    ok = 0
    for p in range(100, kl1h.n, max(1, kl1h.n // 25))[:25]:
        cts = kl1h.ts[p] + 3600000
        mb, _ = v.verdict(cts, "long")
        jd = D.bisect_right(kl1d.ts, cts - 86400000)
        jh = D.bisect_right(kl4h.ts, cts - 14400000)
        ob, _ = OH.overheat_from_klines(_dicts(kl1d, jd), _dicts(kl4h, jh),
                                        "long", {"enabled": True})
        ok += (mb == ob)
    ck(ok == 25, f"overheat 口径 {ok}/25 逐位一致")


def test_parity_resistance() -> None:
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "chan-quant")))
    from app import resistance as RS
    kl1h = D.load("spot", "BTCUSDT", "1h")
    v = MF.Resistance(kl1h)
    ok = 0
    for p in range(100, kl1h.n, max(1, kl1h.n // 25))[:25]:
        cts = kl1h.ts[p] + 3600000
        e = kl1h.c[p]
        mb, _ = v.verdict(cts, "long", e)
        j = D.bisect_right(kl1h.ts, cts - 3600000)
        ob, _ = RS.resistance_from_klines(_dicts(kl1h, j), "long", e,
                                          {"enabled": True})
        ok += (mb == ob)
    ck(ok == 25, f"resistance 口径 {ok}/25 逐位一致")


def test_parity_fib() -> None:
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "chan-quant")))
    from app import fib as FIB
    kl1h = D.load("spot", "BTCUSDT", "1h")
    v = MF.Fib()
    ok = 0
    for p in range(100, kl1h.n, max(1, kl1h.n // 25))[:25]:
        e = kl1h.c[p]
        md = v.depth(kl1h, p, "long", e)
        od = FIB.retrace_depth(_dicts(kl1h, p), "long", e, lookback=60,
                               leg="swing", min_leg_pct=0.01)
        same = (md is None) == (od is None) and (md is None or abs(md - od) < 1e-9)
        ok += same
    ck(ok == 25, f"fib 口径 {ok}/25 逐位一致")


def test_parity_correl() -> None:
    sys.path.insert(0, os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "chan-quant")))
    from app import correl as CO
    kl1h = D.load("spot", "BTCUSDT", "1h")
    v = MF.Correl(kl1h, kl1h, kl1h)
    ok = 0
    for p in range(100, kl1h.n, max(1, kl1h.n // 25))[:25]:
        cts = kl1h.ts[p] + 3600000
        blocked, _ = v.verdict(cts, "long")
        j = D.bisect_right(kl1h.ts, cts - 3600000)
        c1 = [kl1h.c[i] for i in range(j)]
        params = CO.compute_corr(c1, c1, c1, window=168)
        trend = CO.ref_trend(c1, ma=20)
        allowed, _ = CO.allow_open(params, "long", trend, min_rho=0.6)
        ok += (blocked == (not allowed))
    ck(ok == 25, f"correl 口径 {ok}/25 逐位一致")


# ================================================================ B. 无未来函数
def test_no_lookahead() -> None:
    """截断一致性: 把数据截到某时刻, 之前的判定值不变。"""
    kl1h, kl1d, kl4h = (D.load("spot", "BTCUSDT", t) for t in ("1h", "1d", "4h"))
    # 用前 2/3 数据建视图, 与全量数据在同一时刻的判定对比
    cut1h = D.Kl(rows=[(kl1h.ts[i], kl1h.o[i], kl1h.h[i], kl1h.l[i], kl1h.c[i],
                        kl1h.v[i], kl1h.qv[i]) for i in range(kl1h.n * 2 // 3)])
    cut1d = D.Kl(rows=[(kl1d.ts[i], kl1d.o[i], kl1d.h[i], kl1d.l[i], kl1d.c[i],
                        kl1d.v[i], kl1d.qv[i]) for i in range(kl1d.n * 2 // 3)])
    cut4h = D.Kl(rows=[(kl4h.ts[i], kl4h.o[i], kl4h.h[i], kl4h.l[i], kl4h.c[i],
                        kl4h.v[i], kl4h.qv[i]) for i in range(kl4h.n * 2 // 3)])
    full = MF.Overheat(kl1d, kl4h)
    part = MF.Overheat(cut1d, cut4h)
    # 取一个"截断点之前"的时刻
    j = cut1h.n // 2
    cts = cut1h.ts[j] + 3600000
    f = full.verdict(cts, "long")
    q = part.verdict(cts, "long")
    ck(f == q, f"overheat 截断一致: 全量{f} vs 截断{q}")
    fr = MF.Resistance(kl1h)
    pr = MF.Resistance(cut1h)
    e = cut1h.c[j]
    ck(fr.verdict(cts, "long", e) == pr.verdict(cts, "long", e),
       "resistance 截断一致")


# ================================================================ C. fail-open
def test_fail_open() -> None:
    """数据不足一律放行(不误杀)。"""
    from array import array
    tiny = D.Kl(rows=[(i * 3600000, 100.0, 101.0, 99.0, 100.5, 1.0, 100.0)
                      for i in range(5)])
    v = MF.Overheat(tiny, tiny)
    b, _ = v.verdict(5 * 3600000, "long")
    ck(not b, "overheat 数据不足放行")
    r = MF.Resistance(tiny)
    b, _ = r.verdict(5 * 3600000, "long", 100.0)
    ck(not b, "resistance 数据不足放行")
    f = MF.Fib()
    d = f.depth(tiny, 5, "long", 100.0)
    ck(d is None, "fib 数据不足返回 None(放行)")


# ================================================================ D. guards 熔断
def test_guards_consec_loss() -> None:
    g = Guards({"capital": {"trading_capital": 10.0}})
    ok, _ = g.can_open("X", 1000.0)
    ck(ok, "初始可开仓")
    for k in range(3):
        g.on_close({"qty": 1.0, "entry": 5.0, "pnl_pct": -0.05, "symbol": "X"},
                   1000.0 + k)
    ok, why = g.can_open("X", 1001.0)
    ck(not ok and "连续止损" in why, f"3 连亏触发全局冷却: {why}")


def test_guards_fee_level_neutral() -> None:
    g = Guards({"capital": {"trading_capital": 10.0}})
    # 手续费级小亏: 不计连亏, 但给同标的短冷却
    g.on_close({"qty": 1.0, "entry": 5.0, "pnl_pct": -0.0005, "symbol": "Y"},
               1000.0)
    ck(g.s["consec_losses"] == 0, "手续费级小亏不计连亏")
    ok, _ = g.can_open("Y", 1000.5)
    ck(not ok, "手续费级小亏触发同标的短冷却")


def test_guards_daily_loss() -> None:
    g = Guards({"capital": {"trading_capital": 10.0}})
    # 一笔 -20% 亏损(10U * 20% = 2U), 超过 daily_max_loss 15%
    g.on_close({"qty": 10.0, "entry": 1.0, "pnl_pct": -0.20, "symbol": "Z"},
               1000.0)
    ck(g.s["pause_until"] > 1000.0, "当日亏损触线写 pause_until")


def test_guards_direction() -> None:
    g = Guards({"capital": {"trading_capital": 10.0}})
    msg = None
    for k in range(8):
        msg = g.on_close_dir("long", -0.01, 2000.0 + k)
    ck(msg is not None, "8 笔连亏触发方向保险丝")
    ok, _ = g.direction_ok("long", 2000.5)
    ck(not ok, "方向保险丝期内禁止该方向开仓")


def main() -> int:
    for f in (test_parity_overheat, test_parity_resistance, test_parity_fib,
              test_parity_correl, test_no_lookahead, test_fail_open,
              test_guards_consec_loss, test_guards_fee_level_neutral,
              test_guards_daily_loss, test_guards_direction):
        f()
    print(f"check_market_filters: 通过 {_PASS} 项 / 失败 {_FAIL} 项")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
