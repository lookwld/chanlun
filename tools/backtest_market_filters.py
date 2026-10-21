"""第四轮补全 · 全池对比 —— 7 个 chan-quant 能力逐个在主引擎路径上验证。

对比基准 = 生产默认(stop + S1), 逐个打开:
  regime_gate / overheat_on / resistance_on / fib_on / correl_on / exit_fenxing

口径与 `backtest_engine_path.py` 一致(主引擎 `engine.run_all` / `LevelSet`)。

  python tools/backtest_market_filters.py --jobs 10 --out out/mf_spot.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import engine as E                                     # noqa: E402
from chanbook import data as D                                       # noqa: E402
from chanbook.points import B1, B2, B3                               # noqa: E402
from chanbook.strategy import BookCfg                                # noqa: E402

import tools.backtest_engine as BT                                   # noqa: E402

TF = "1h"
MARKET = "spot"
# 基准 + 6 个过滤器逐个开启(其余保持生产默认 stop+S1)
CONFIGS = [
    ("基准 生产默认(stop+S1)",      dict()),
    ("+regime_gate 大盘状态闸门",    dict(regime_gate=True)),
    ("+overheat 过热带",             dict(overheat_on=True)),
    ("+resistance 前高压制",         dict(resistance_on=True)),
    ("+fib 斐波那契回撤",            dict(fib_on=True)),
    ("+correl 联动度",               dict(correl_on=True)),
    ("+exit_fenxing 分形离场",       dict(exit_fenxing=True)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="out/mf_spot.txt")
    ap.add_argument("--kinds", default="b1,b2")
    a = ap.parse_args()
    kinds = tuple(BT.KIND_MAP[k] for k in a.kinds.split(","))

    syms = D.symbols(MARKET)
    lines: list[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(f"# 第四轮补全 · 主引擎路径全池 · {TF} · 构造 {a.kinds} · {len(syms)} 标的")
    emit(f"# 生成于 {stamp} · 多进程 {a.jobs} · 基准 = 生产默认(stop + S1)")
    emit("")
    emit(BT.HDR)

    base_r = None
    out_rows = []
    for name, kw in CONFIGS:
        cfg = BookCfg(market=MARKET, kinds=kinds, **kw)
        t0 = time.time()
        tr, dg = E.run_all(MARKET, cfg, jobs=a.jobs)
        r = BT.stats_row(tr, MARKET, TF, dg, name)
        out_rows.append(r)
        blk = (f"  拦[regime {r.get('n_regime_block', 0)} "
               f"overheat {r.get('n_overheat_block', 0)} "
               f"resistance {r.get('n_resistance_block', 0)} "
               f"fib {r.get('n_fib_block', 0)} "
               f"correl {r.get('n_correl_block', 0)}]")
        emit(BT.fmt_row(r) + blk + f"   [{time.time() - t0:.0f}s]")
        if base_r is None:
            base_r = r

    emit("")
    emit("## 分标的配对(变好比例; 基准 = 生产默认)")
    for r in out_rows[1:]:
        n, b, p = BT.signed_pairs(r["by_sym"], base_r["by_sym"])
        emit(f"  {r['label']:<34} {b}/{n} = {p:.0%}")

    emit("")
    emit("## 同池等权指数(逐日再平衡) 对照")
    pi = BT.pool_index(syms, TF, MARKET, 0, 0, rebal_bars=24)
    if pi:
        emit(f"  池基准  总 {pi['total']:+.1%}  每日 {pi['per_day']:+.4%}  "
             f"年化 {pi['ann']:+.1%}  ({pi['days']:.0f} 天)")
        emit("")
        emit(f"  {'配置':<34} {'每自然日':>10} {'池每日':>10} {'倍数':>7}")
        for r in out_rows:
            if not r.get("n"):
                continue
            pdn = r["per_day"] * r["share"]
            emit(f"  {r['label']:<34} {pdn:+.4%} {pi['per_day']:+.4%} "
                 f"{pdn / pi['per_day']:>6.2f}×")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
