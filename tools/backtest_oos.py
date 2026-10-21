"""闸① 样本外分段验证 —— 2017-2021 训练 / 2022-2026 验证。

用**主引擎路径**(engine.run_all / LevelSet / 原生 4h)在现货 1h 上, 把生产默认
(stop + S1)与升级前 legacy(old)分开读两个时间段:

  * 训练段 2017-08 ~ 2021-12
  * 验证段 2022-01 ~ 2026-08

判据(沿用 MIGRATION §2 的四条验收线, 但这里只做第 ① 条"分段符号"):
  * 生产默认在**验证段**的"单笔净"符号是否仍为正;
  * 逐年(2022~2026)符号一致性;
  * vs 同池等权指数是否仍跑赢。

⚠ 分段 = 数据时间截断, 不改任何引擎/策略逻辑。t_start/t_end 直接透传给
   `engine.run_all`, 由 `run_symbol` 的 bisect 定位 + break 截断实现, 无未来函数。

  python tools/backtest_oos.py --jobs 10 --out out/oos.txt
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
from chanbook.points import B1, B2                                   # noqa: E402
from chanbook.strategy import BookCfg                                # noqa: E402

import tools.backtest_engine as BT                                   # noqa: E402

TF = "1h"
MARKET = "spot"
KINDS = (B1, B2)

# 分段边界(毫秒, 用 BT.ms_of)
TRAIN = ("2017-08", "2021-12")
TEST = ("2022-01", "2026-08")


def _seg(label: str, seg: tuple[str, str]) -> tuple[int, int]:
    return BT.ms_of(seg[0]), BT.ms_of(seg[1], end=True)


def _run(cfg: BookCfg, t0: int, t1: int, jobs: int, syms: list):
    return E.run_all(MARKET, cfg, symbols=syms, jobs=jobs, t_start=t0, t_end=t1)


def _by_year(tr: list) -> dict[int, dict]:
    """逐年单笔净符号(只统计有成交的年份)。"""
    from collections import defaultdict
    agg = defaultdict(list)
    for t in tr:
        y = datetime.fromtimestamp(t.exit_ts / 1000, tz=__import__("datetime").timezone.utc).year
        agg[y].append(t.pnl_pct)
    out = {}
    for y in sorted(agg):
        pnls = agg[y]
        out[y] = {"n": len(pnls), "mean": sum(pnls) / len(pnls)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="out/oos.txt")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    syms = D.symbols(MARKET)
    if a.limit:
        syms = syms[:a.limit]

    lines: list[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def emit(s: str):
        print(s, flush=True)
        lines.append(s)

    emit(f"# 闸① 样本外分段验证 (主引擎路径 engine.run_all / LevelSet / 原生4h)")
    emit(f"# 现货 {TF} · 构造 b1,b2 · {len(syms)} 标的 · {a.jobs} 进程")
    emit(f"# 生成于 {stamp}")
    emit(f"# 训练段 {TRAIN[0]}~{TRAIN[1]}  验证段 {TEST[0]}~{TEST[1]}")
    emit("")

    segs = [("训练", TRAIN), ("验证", TEST)]
    configs = [
        ("生产默认 stop+S1", dict(exit_mode="stop", sell_exit=("S1",))),
        ("升级前 legacy old", dict(exit_mode="old", sell_exit=())),
    ]

    results = {}   # (seg, cfg_label) -> stats dict

    for seg_label, seg in segs:
        t0, t1 = _seg(seg_label, seg)
        # 池基准(分段)
        pi = BT.pool_index(syms, TF, MARKET, t0, t1, rebal_bars=24)
        emit(f"## {seg_label}段 {seg[0]}~{seg[1]}")
        if pi:
            emit(f"  池基准(逐日再平衡)  总 {pi['total']:+.1%}  每日 {pi['per_day']:+.4%}  "
                 f"年化 {pi['ann']:+.1%}  ({pi['days']:.0f} 天)")
        emit("")
        emit(BT.HDR)
        for label, kw in configs:
            cfg = BookCfg(market=MARKET, kinds=KINDS, **kw)
            t0c = time.time()
            tr, dg = _run(cfg, t0, t1, a.jobs, syms)
            r = BT.stats_row(tr, MARKET, TF, dg, label)
            results[(seg_label, label)] = r
            emit(BT.fmt_row(r) + f"   [{time.time() - t0c:.0f}s]")
            # 逐年
            if r.get("n"):
                by = _by_year(tr)
                pos = sum(1 for v in by.values() if v["mean"] > 0)
                emit(f"    逐年单笔净: " + "  ".join(
                    f"{y}:{v['mean']:+.2%}({v['n']})" for y, v in by.items()))
                emit(f"    → 为正年份 {pos}/{len(by)}")
        emit("")

    # ---- 判决 ----
    emit("=" * 72)
    emit("## 判决: 生产默认在验证段是否仍为正")
    for label, _kw in configs:
        tr_r = results.get(("训练", label))
        te_r = results.get(("验证", label))
        if not tr_r or not te_r:
            continue
        emit(f"  {label}:")
        emit(f"    训练段 单笔净 {tr_r['net']:+.3%}  PF {tr_r['pf']:.2f}  "
             f"在仓 {tr_r['share']:.1%}  折算年化 {tr_r['fold']:+.1%}")
        emit(f"    验证段 单笔净 {te_r['net']:+.3%}  PF {te_r['pf']:.2f}  "
             f"在仓 {te_r['share']:.1%}  折算年化 {te_r['fold']:+.1%}")
    # 核心结论: 生产默认验证段符号
    prod_test = results.get(("验证", "生产默认 stop+S1"))
    if prod_test:
        pnl = prod_test["net"]
        emit("")
        if pnl > 0:
            emit(f"  ✅ 生产默认验证段单笔净 {pnl:+.3%} > 0 —— 通过闸①符号门槛")
        else:
            emit(f"  ❌ 生产默认验证段单笔净 {pnl:+.3%} ≤ 0 —— 未通过闸①符号门槛")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
