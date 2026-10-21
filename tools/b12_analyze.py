"""把 1h / 2h / 4h 三轮回测放到**同一口径**下对照。

## 为什么需要这个脚本

`backtest_b12_tf.py` 每个 tf 的"可用标的集"不同 —— 因为 `MIN_BARS=500` 是**按各自周期**
计的: 4h 要 500 根 = 83 天(只有 82/93 个现货标的够), 1h 要 500 根 = 21 天(91/93 个够)。
而池基准与策略统计都依赖这个集合 ⇒ 直接跨 tf 比会把"标的池不同"混进结论里
(实测: 现货逐日再平衡池指数在 82 标的的子集上是 +10.0%/年, 在 91 标的的子集上只有 +7.7%/年,
光这一项就足以让"1h 跑赢池子"变成"1h 跑平池子")。

本脚本把每个市场统一到 **所有 tf 都可用**的标交集上, 重算:
  * 池基准(逐日再平衡等权指数 + 买入持有每日收益)
  * 每个 (tf, 离场口径) 的策略口径(单笔毛/净、PF、每持仓日、在仓占比、折算年化)
输入是 `out/cache_b12/{market}_{tf}_{mode}.json`(逐笔成交 + 诊断), 只读。

运行: python tools/b12_analyze.py [--out out/b12_tf_compare.txt]
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import backtest_b12_tf as T                                       # noqa: E402
from chanbook import data as D                                    # noqa: E402
from chanbook import report as R                                  # noqa: E402

CACHE = "out/cache_b12"
MARKETS = ("spot", "um")
TFS = ("4h", "2h", "1h")
MODES = ("chan1_stop", "chan123_stop", "old")


def cache_path(market: str, tf: str, mode: str) -> str:
    return os.path.join(CACHE, f"{market}_{tf}_{mode}.json")


def load(market: str, tf: str, mode: str):
    p = cache_path(market, tf, mode)
    if not os.path.exists(p):
        return None, None
    tr, dgl = R.load_trades(p)
    return tr, {d.sym: d for d in dgl}


def per_day_of(s: dict) -> float:
    d = s["hold_min"] / 1440.0
    return (s["exp_pct"] / d) if d > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out/b12_tf_compare.txt")
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-08")
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    L: list[str] = []

    def emit(s: str = "") -> None:
        L.append(str(s))
        print(s, flush=True)

    t0 = T.ms_of(args.start) if args.start else 0
    t1 = T.ms_of(args.end, True) if args.end else 0

    for market in MARKETS:
        emit("=" * 104)
        emit(f"### {market}  —— 统一到「所有 tf 都可用」的标的交集")
        emit("=" * 104)
        # 1) 每个 tf 的可用集
        usable: dict[str, set] = {}
        for tf in TFS:
            us = set()
            for mode in MODES:
                tr, dg = load(market, tf, mode)
                if dg is None:
                    continue
                us |= {s for s, d in dg.items() if not getattr(d, "err", "")}
            if us:
                usable[tf] = us
        if not usable:
            emit("(无缓存)")
            emit("")
            continue
        for tf in TFS:
            if tf in usable:
                emit(f"  {tf:>3} 可用标的 {len(usable[tf]):>3} 个")
        common = set.intersection(*usable.values()) if usable else set()
        syms = [s for s in D.symbols(market) if s in common]
        emit(f"  ⇒ 交集 **{len(common)}** 个标的(以下所有数字都只用这 {len(common)} 个)")
        emit("")

        # 2) 池基准(交集上) —— 逐日再平衡 ⇒ **tf 无关**, 只算一次即可。
        # (在 4h 网格上再算一遍做自校验: 两条曲线应当几乎重合。)
        emit("池基准(交集标的集, 逐日再平衡 —— tf 无关):")
        pi_ref = None
        for btf in (("1h", "4h") if market == "spot" else ("1h",)):
            pi = T.pool_index(syms, btf, market, t0, t1,
                              rebal_bars=int(round(24.0 / T.TF_HOURS[btf])))
            if not pi:
                continue
            if pi_ref is None:
                pi_ref = pi
            emit(f"  在 {btf} 网格上算: 全程 {pi['total'] * 100:>+8.1f}%  "
                 f"每日 {pi['per_day'] * 100:>+8.4f}%  "
                 f"年化 {pi['ann'] * 100:>+7.1f}%")
        pool_ann = pi_ref.get("ann", 0.0) if pi_ref else 0.0
        emit(f"  ⇒ 用于对照的市场基准年化 = **{pool_ann * 100:+.1f}%**")
        emit("")

        # 3) 策略口径(交集上)
        hdr = (f"{'tf':>3} {'离场':>13} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} "
               f"{'单笔净':>9} {'PF':>6} {'均持仓':>8} {'每持仓日':>9} "
               f"{'在仓':>7} {'折算年化':>9} {'vs池(pp)':>9}")
        emit("策略口径(只统计交集标的的成交):")
        emit(hdr)
        emit("-" * len(hdr))
        for tf in TFS:
            for mode in MODES:
                tr, dg = load(market, tf, mode)
                if not tr:
                    continue
                sub = [t for t in tr if t.sym in common]
                if not sub:
                    continue
                s = R.summarize(sub, market)
                pdv = per_day_of(s)
                avail = (sum(int(getattr(d, "n_op_bars", 0))
                             for sym, d in dg.items() if sym in common
                             and not getattr(d, "err", ""))
                         * T.BAR_MIN.get(tf, 60))
                used = sum(t.hold_min for t in sub)
                share = (used / avail) if avail else 0.0
                ann = ((1.0 + pdv) ** (365.25 * share) - 1.0) if pdv > -1 else -1.0
                emit(f"{tf:>3} {mode:>13} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
                     f"{s['exp_pct'] * 100:>+8.3f}% {s['net_exp_pct'] * 100:>+8.3f}% "
                     f"{s['pf']:>6.2f} {s['hold_min'] / 1440:>6.2f}天 "
                     f"{pdv * 100:>+8.3f}% {share * 100:>6.2f}% "
                     f"{ann * 100:>+8.2f}% {(ann - pool_ann) * 100:>+8.2f}")
            emit("")

        # 4) B1 / B2 拆分(chan1_stop)
        emit("B1 / B2 拆分(chan1_stop, 交集标的):")
        emit(f"{'tf':>3} {'买点':>5} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} {'PF':>6} "
             f"{'均持仓':>8} {'每持仓日':>9}")
        for tf in TFS:
            tr, dg = load(market, tf, "chan1_stop")
            if not tr:
                continue
            for kd in ("B1", "B2"):
                sub = [t for t in tr if t.sym in common and t.kind == kd]
                s = R.summarize(sub, market)
                if not s["n"]:
                    emit(f"{tf:>3} {kd:>5} {0:>6}   (无成交)")
                    continue
                emit(f"{tf:>3} {kd:>5} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
                     f"{s['exp_pct'] * 100:>+8.3f}% {s['pf']:>6.2f} "
                     f"{s['hold_min'] / 1440:>6.2f}天 {per_day_of(s) * 100:>+8.3f}%")
        emit("")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(L) + "\n")
        print(f"\n已写入 {args.out}")
    except Exception as e:                                        # noqa: BLE001
        print(f"写盘失败: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
