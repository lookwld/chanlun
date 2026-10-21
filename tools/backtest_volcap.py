"""波动率上限闸门回测 —— chan-quant 的 `vol_gate` 移植到 chanbook 主引擎的效果。

**来源与证据**
`chan-quant/app/vol.py` + `chan-quant/config.example.yaml::vol_gate`(**已 `enabled: true`**)。
chan-quant 自己的实测结论（原文照抄该配置的注释）:

    现货 9 年全历史(2017-08 起) 组合层 -25.15U(-83.8%) -> +20.43U(+68.1%),
    本项目首个组合层为正的配置; 3%~5% 整个阈值区间都过成本门槛(非精调尖峰)。
    诚实项: 2021-22 两年轻微反向; 5% 为扫描后选定。
    * 仅现货循环接线; 合约回测无差别(5% 上限拦 89% 候选仍 -25.12U), 合约侧不部署。

**必须先读清的一个名称冲突（否则结论会读错）**

    * chan-quant 的 `vol_gate`  == **波动率上限**（日线 ATR14% <= 阈值）
    * chanbook   的 `vol_gate`  == **成交量比**（后段均量 / 前段均量）

同名，完全不同的两个维度。本轮测的是**前者**，开关名 `atr1d_cap`。
（chanbook 的成交量版已在 `ENGINE_VARIANTS_2026-09-29.md` 的 E5 测过，结论：负。）

**口径**：与 `tools/backtest_engine.py` 完全一致（同一套 `stats_row` /
`pool_index` / `signed_pairs`），但走**主引擎自己的调用路径**
（`engine.run_all` / `LevelSet`）—— 闸门是改在主引擎里的，结论要能落到生产。

⚠ 本工具只跑 `spot` 的默认操作级别（`LEVEL_TF["spot"] = 4h / 1h / 15m`）。
多周期（2h/4h）在 `tools/backtest_engine.py` 的旁路链上跑（它能自由指定操作级别）。

跑法:
  python tools/backtest_volcap.py --jobs 10 --out out/volcap_spot.txt
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

TF = "1h"                 # spot 的操作级别
MARKET = "spot"
CAPS = (0.0, 0.03, 0.05, 0.08)
BASES = (("legacy", "legacy  old"), ("prod", "prod    stop+S1"))


def build(base: str, kinds: tuple, cap: float) -> BookCfg:
    """`base` 决定离场口径; `cap` 是波动率上限(0 = 关闭)。"""
    over = dict(market=MARKET, kinds=kinds, atr1d_cap=float(cap))
    return BookCfg.legacy(**over) if base == "legacy" else BookCfg(**over)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="out/volcap_spot.txt")
    ap.add_argument("--kinds", default="b1,b2")
    a = ap.parse_args()
    kinds = tuple(BT.KIND_MAP[k] for k in a.kinds.split(","))

    syms = D.symbols(MARKET)
    lines: list[str] = []

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    emit(f"# 波动率上限闸门(chan-quant `vol_gate` 移植) · 主引擎路径 · "
         f"{MARKET} {TF} · 构造 {a.kinds}")
    emit(f"# 生成于 {stamp}   {len(syms)} 个标的   多进程 {a.jobs}")
    emit("# 口径: 日线 ATR14% <= atr1d_cap 才放行 (与 chan-quant app/vol.py 同源, fail-open)")
    emit("")
    emit(BT.HDR)

    rows: dict[tuple, dict] = {}
    for base, blabel in BASES:
        for cap in CAPS:
            label = f"{blabel} " + ("无闸门" if cap <= 0 else f"<={cap:.0%}")
            cfg = build(base, kinds, cap)
            t0 = time.time()
            tr, dg = E.run_all(MARKET, cfg, jobs=a.jobs)
            r = BT.stats_row(tr, MARKET, TF, dg, label)
            r["cap"] = cap
            rows[(base, cap)] = r
            emit(BT.fmt_row(r) + f"   [{time.time() - t0:.0f}s]")
        emit("")

    # ---- 配对检验: 同离场口径下, 加闸门 vs 不加闸门 ----
    emit("## 分标的配对(变好比例; 基准 = 同离场口径的无闸门)")
    for base, blabel in BASES:
        base_r = rows.get((base, 0.0)) or {}
        if not base_r.get("n"):
            continue
        for cap in CAPS[1:]:
            r = rows.get((base, cap)) or {}
            if not r.get("n"):
                continue
            n, b, p = BT.signed_pairs(r["by_sym"], base_r["by_sym"])
            emit(f"  {blabel:<16} <={cap:.0%}  {b}/{n} = {p:.0%}")
    emit("")

    # ---- 闸门诊断: 拦了多少、成交少了多少 ----
    emit("## 闸门诊断(被拦的结构命中数 vs 成交笔数变化)")
    emit(f"  {'配置':<26} {'拦(命中)':>10} {'笔数':>7} {'笔数变化':>10} {'单笔净':>10}")
    for base, blabel in BASES:
        base_r = rows.get((base, 0.0)) or {}
        if not base_r.get("n"):
            continue
        for cap in CAPS:
            r = rows.get((base, cap)) or {}
            if not r.get("n"):
                continue
            label = f"{blabel} " + ("无闸门" if cap <= 0 else f"<={cap:.0%}")
            dn = r["n"] - base_r["n"]
            emit(f"  {label:<26} {r.get('n_volcap_block', 0):>10} {r['n']:>7} "
                 f"{dn:>+10} {r['net'] * 100:>+9.3f}%")
    emit("")

    # ---- 同池等权指数对照 ----
    emit("## 同池等权指数(逐日再平衡) 对照")
    pi = BT.pool_index(syms, TF, MARKET, 0, 0, rebal_bars=24)
    if pi:
        emit(f"  池基准  总 {pi['total']:+.1%}  每日 {pi['per_day']:+.4%}  "
             f"年化 {pi['ann']:+.1%}  ({pi['days']:.0f} 天)")
        emit("")
        emit(f"  {'配置':<26} {'每自然日':>10} {'池每日':>10} {'倍数':>7}")
        for base, blabel in BASES:
            for cap in CAPS:
                r = rows.get((base, cap)) or {}
                if not r.get("n"):
                    continue
                label = f"{blabel} " + ("无闸门" if cap <= 0 else f"<={cap:.0%}")
                pdn = r["per_day"] * r["share"]
                emit(f"  {label:<26} {pdn:+.4%} {pi['per_day']:+.4%} "
                     f"{pdn / pi['per_day']:>6.2f}x")

    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n已写 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
