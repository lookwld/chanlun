# -*- coding: utf-8 -*-
"""十字星策略 · 止盈离场后的价格后视分析(事后描述, 不构成新回测)。

问题: 止盈(zs1_half 减半 / zs2_exit 清仓)之后, 标的继续涨的多, 还是马上回落的多?

口径(core 变体, 与 DOJI_2026-10-01.md 判决同源):
- 离场成交所在 bar 记为第 0 根, 观察窗从**下一根**开始(离场当根已触碰目标价)。
- race(7 天窗): 先涨到 exit+2% 记 "up", 先跌到 exit-2% 记 "down",
  同一根同时触发记 "same"(保守并入 down 口径展示), 7 天内都没触发记 "neither"。
- mfe7/mae7: 7 天窗内相对离场价的最大上冲 / 最大下探(中位数汇报)。
- fwd1/fwd7: 离场后 1 天 / 7 天的收盘价相对离场价的收益。
- 数据不足 7 天(尾部截断)的样本 race 记 "short", fwd 不统计。

用法:
  python tools/analyze_doji_postexit.py --jobs 10
"""
from __future__ import annotations

import argparse
import bisect
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from chanbook import data as D                                    # noqa: E402
import tools.backtest_doji as BD                                  # noqa: E402

REASONS = ("zs1_half", "zs2_exit")
COMBOS = [("spot", "15m"), ("spot", "1h"), ("spot", "4h"), ("spot", "1d"),
          ("um", "5m"), ("um", "15m"), ("um", "1h")]
THR = 0.02          # race 阈值 ±2%


def _work(task):
    market, sym, tf = task
    tr, _dg = BD.run_symbol(market, sym, tf, False)
    if not tr:
        return []
    kl = D.load(market, sym, tf)
    bpd = 86_400_000 // D.TF_MS[tf]
    rows = []
    for t in tr:
        if t.reason not in REASONS:
            continue
        i0 = bisect.bisect_left(kl.ts, t.exit_ts)
        j0 = i0 + 1
        n = kl.n
        row = {"sym": sym, "reason": t.reason, "exit": t.exit,
               "gain": t.pnl_pct * 100, "hold_d": t.hold_min / 1440.0,
               "race": "short", "mfe7": 0.0, "mae7": 0.0,
               "fwd1": None, "fwd7": None}
        w7 = min(j0 + 7 * bpd, n)
        if w7 - j0 >= bpd:                      # 至少 1 天数据才算完整 7 天窗
            race, mfe, mae = "neither", 0.0, 0.0
            for j in range(j0, w7):
                mfe = max(mfe, kl.h[j] / t.exit - 1.0)
                mae = min(mae, kl.l[j] / t.exit - 1.0)
                if race == "neither":
                    up = kl.h[j] >= t.exit * (1 + THR)
                    dn = kl.l[j] <= t.exit * (1 - THR)
                    if up and dn:
                        race = "same"
                    elif up:
                        race = "up"
                    elif dn:
                        race = "down"
            row.update(race=race, mfe7=mfe * 100, mae7=mae * 100)
        j1, j7 = i0 + bpd, i0 + 7 * bpd
        if j1 < n:
            row["fwd1"] = (kl.c[j1] / t.exit - 1.0) * 100
        if j7 < n:
            row["fwd7"] = (kl.c[j7] / t.exit - 1.0) * 100
        rows.append(row)
    return rows


def _med(xs: list) -> float:
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def _pct(a: int, b: int) -> float:
    return a / b * 100 if b else float("nan")


def summarize(rows: list, label: str) -> str:
    if not rows:
        return f"{label:<22} (无样本)"
    races = [r["race"] for r in rows]
    n = len(rows)
    up = races.count("up")
    dn = races.count("down") + races.count("same")
    ne = races.count("neither")
    sh = races.count("short")
    f7 = [r["fwd7"] for r in rows if r["fwd7"] is not None]
    f1 = [r["fwd1"] for r in rows if r["fwd1"] is not None]
    mf = [r["mfe7"] for r in rows if r["race"] != "short"]
    ma = [r["mae7"] for r in rows if r["race"] != "short"]
    return (f"{label:<22} n={n:>5} | 先涨2%: {up:>4}({_pct(up, n):4.1f}%) "
            f"先跌2%: {dn:>4}({_pct(dn, n):4.1f}%) "
            f"未触发: {ne:>4}({_pct(ne, n):4.1f}%) 截断: {sh} | "
            f"7天中位: 上冲{_med(mf):+6.2f}% 下探{_med(ma):+6.2f}% "
            f"收盘1天{_med(f1):+6.2f}% 7天{_med(f7):+6.2f}% "
            f"(7天上涨占比 {_pct(sum(1 for x in f7 if x > 0), len(f7)):.0f}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    a = ap.parse_args()

    print(f"# 止盈离场后视分析 · 阈值±{THR*100:.0f}% · 窗口7天 · core 口径")
    for market, tf in COMBOS:
        syms = D.symbols(market)
        if market == "um":                     # 与回测同池: spot 93 ∩ um 有档
            syms = [s for s in D.symbols("spot")
                    if os.path.exists(os.path.join(
                        D.ARCH, "um", "k", f"{s}_{tf}.csv.gz"))]
        t0 = time.time()
        if a.jobs > 1:
            import multiprocessing as mp
            with mp.Pool(a.jobs) as pool:
                rows = [r for part in pool.imap_unordered(
                    _work, [(market, s, tf) for s in syms], chunksize=1)
                    for r in part]
        else:
            rows = [r for s in syms for r in _work((market, s, tf))]
        print(f"\n== {market} {tf} · {len(syms)} 标的 · {time.time()-t0:.0f}s "
              f"· 止盈成交 {len(rows)} 笔 ==")
        for rs in REASONS:
            print(summarize([r for r in rows if r["reason"] == rs],
                            f"  {rs}"))
        print(summarize(rows, "  合计"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
