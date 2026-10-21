"""从 `out/cache/<market>_<variant>.json` 的成交明细算**统计显著性**与集中度。

为什么需要: 现货"只做一买 + 区间套"只有 923 笔成交(9 年), 毛期望 +0.197%。
单看均值不够 —— 必须给出标准误与置信区间, 并做**按标的聚类的稳健估计**, 因为
同一标的在不同月份的成交显然不独立(同一波行情/同一个币种特性)。

口径:
  * 朴素 t   = mean / (sd/sqrt(n))
  * 聚类稳健 = sqrt(sum_g S_g^2) / n,  S_g = 组内 pnl 之和; 再乘 sqrt(G/(G-1))
  * 聚类自助 = 按组重抽样 B 次, 取 2.5%/97.5% 分位

用法: python tools/ab_stats.py um all,b1confirm  [spot b1confirm ...]
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FEE = {"um": 0.0005, "spot": 0.001}
THR = {"um": 0.0012, "spot": 0.0020}
TZ = timezone.utc


def _month(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y-%m")


def load(market: str, variant: str, cache: str = "out/cache"):
    p = os.path.join(cache, f"{market}_{variant}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        d = json.load(fh)
    return d.get("tr", [])


def stats(rows: list[dict], market: str, key=("sym",)) -> dict:
    xs = [float(r["pnl_pct"]) for r in rows]
    n = len(xs)
    if n < 2:
        return {"n": n}
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1))
    t_naive = mean / (sd / math.sqrt(n)) if sd else 0.0

    groups: dict = {}
    for r, x in zip(rows, xs):
        g = tuple(str(r.get(k, "")) if k != "ym" else _month(int(r["entry_ts"]))
                  for k in key)
        groups.setdefault(g, 0.0)
        groups[g] += x
    G = len(groups)
    ss = math.sqrt(sum(v * v for v in groups.values()))
    se_cr = ss / n * math.sqrt(G / (G - 1)) if G > 1 else float("inf")
    t_cr = mean / se_cr if se_cr not in (0.0, float("inf")) else 0.0

    rnd = random.Random(20260927)
    sums = list(groups.values())
    boots = []
    for _ in range(2000):
        tot = 0.0
        for _ in range(G):
            tot += sums[rnd.randrange(G)]
        boots.append(tot / n)
    boots.sort()
    lo, hi = boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]

    by_sym = {}
    for r, x in zip(rows, xs):
        by_sym[r["sym"]] = by_sym.get(r["sym"], 0.0) + x
    pos = sum(1 for v in by_sym.values() if v > 0)
    return {"n": n, "mean": mean, "sd": sd, "t_naive": t_naive,
            "t_cr": t_cr, "G": G, "ci": (lo, hi),
            "pos_sym": pos, "tot_sym": len(by_sym),
            "fee": FEE[market], "thr": THR[market]}


def main(argv: list[str]) -> int:
    pairs = [(argv[1], argv[2])] if len(argv) > 2 else [
        ("um", "all,b1,b2,b3,b1confirm,nogate,nostruct,b1wideoff,strict,pull3,"
               "stopmin,combo,b1x"),
        ("spot", "all,b1,b2,b3,b1confirm,nogate,nostruct,b1wideoff,strict,pull3,"
                 "stopmin,combo,b1x")]
    for market, names in pairs:
        print(f"\n## {market}（门槛: 毛期望 > {THR[market] * 100:.2f}%）\n")
        print("| 变体 | 笔数 | 毛期望 | 净期望 | 朴素t | 聚类t | 聚类自助95%CI(毛) "
              "| 正净标的 |")
        print("|---|---:|---:|---:|---:|---:|:--|:--|")
        for v in names.split(","):
            rows = load(market, v.strip())
            if not rows:
                continue
            s = stats(rows, market)
            if s["n"] < 2:
                print(f"| `{v}` | {s['n']} | — | — | — | — | — | — |")
                continue
            w_exp = sum(float(r["pnl_pct"]) * float(r["notional"]) for r in rows) \
                / sum(float(r["notional"]) for r in rows)
            print(f"| `{v}` | {s['n']:,} | {s['mean'] * 100:+.3f}% | "
                  f"{(w_exp - 2 * s['fee']) * 100:+.3f}% | {s['t_naive']:.2f} | "
                  f"{s['t_cr']:.2f} | [{s['ci'][0] * 100:+.3f}%, "
                  f"{s['ci'][1] * 100:+.3f}%] "
                  f"| {s['pos_sym']}/{s['tot_sym']} |")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
