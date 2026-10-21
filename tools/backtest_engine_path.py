"""主引擎路径全池回测 —— 用 `engine.run_all`(而不是旁路脚本)验证"改完能到什么效果"。

为什么还要一个: `tools/backtest_engine.py` 走的是自己搭的 `Level`/`_LS`, 与主引擎的
`LevelSet` 有两处已知差异(大级别用**现场聚合的 4h** vs `LevelSet` 的**原生 4h`;
预热/最少根数判据)。结论要落到生产, 就必须在**主引擎自己的调用路径**上再读一遍数。

口径与 `backtest_engine.py` 完全一致(同一套 `stats_row` / `pool_index` / `signed_pairs`)。

  python tools/backtest_engine_path.py --jobs 10 --out out/eng_path.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import engine as E                                     # noqa: E402
from chanbook import data as D                                       # noqa: E402
from chanbook.points import B1, B2, B3                               # noqa: E402
from chanbook.strategy import BookCfg                                # noqa: E402

import tools.backtest_engine as BT                                   # noqa: E402

TF = "1h"
MARKET = "spot"
# 7 个候选配置(A = 升级前, G = 裸默认, 其余 = 中间态与变体)
# ⚠ 2026-09-29 升级后**每一项都必须显式写全** exit_mode/sell_exit ——
#   "不写 = 取默认值" 这个习惯已经失效(默认值从 old 变成了 stop)。
CONFIGS = [
    ("A 升级前默认 legacy (exit=old)",         dict(exit_mode="old", sell_exit=())),
    ("B exit=stop (停掉追踪/RR/结构破位)",     dict(exit_mode="stop", sell_exit=())),
    ("C exit=stop + S1  ← 系统默认",          dict(exit_mode="stop", sell_exit=("S1",))),
    ("D exit=stop + S1 + ma 5/10",            dict(exit_mode="stop", sell_exit=("S1",),
                                                   ma_mode="ma", ma_fast=5, ma_slow=10)),
    ("E exit=stop + S123",                    dict(exit_mode="stop", sell_exit=("S1", "S2", "S3"))),
    ("F exit=old + S1 (旧的'叠加'写法)",       dict(exit_mode="old", sell_exit=("S1",))),
    ("G 裸 BookCfg() = 生产默认",              dict()),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--out", default="out/eng_path.txt")
    ap.add_argument("--kinds", default="b1,b2")
    a = ap.parse_args()
    kinds = tuple(BT.KIND_MAP[k] for k in a.kinds.split(","))

    syms = D.symbols(MARKET)
    lines: list[str] = []
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def emit(s: str) -> None:
        print(s, flush=True)
        lines.append(s)

    emit(f"# 主引擎路径 (engine.run_all / LevelSet) 现货全池 · {TF} · 构造 {a.kinds}")
    emit(f"# 生成于 {stamp}   {len(syms)} 个标的   多进程 {a.jobs}")
    emit(f"# 大级别 = {BT.TREND_TF} 原生(LevelSet); 统计口径同 tools/backtest_engine.py")
    emit("")
    emit(BT.HDR)

    base_r = None
    out_rows = []
    trs: dict[str, list] = {}
    for name, kw in CONFIGS:
        cfg = BookCfg(market=MARKET, kinds=kinds, **kw)
        t0 = time.time()
        tr, dg = E.run_all(MARKET, cfg, jobs=a.jobs)
        trs[name[:1]] = tr                       # 供"升级自证"取指纹
        r = BT.stats_row(tr, MARKET, TF, dg, name)
        out_rows.append(r)
        emit(BT.fmt_row(r) + f"   [{time.time() - t0:.0f}s]")
        if base_r is None:
            base_r = r

    # ---- 升级自证: 裸 `BookCfg()`(= 生产默认) 必须与显式的 C 口径逐位一致 ----
    # 这是"默认值已升级"这件事唯一的机器可验证证据 —— 光看代码容易漏掉某个
    # 被 `__post_init__` 二次加工过的字段。
    def _fp(trades) -> tuple:
        rows = sorted((t.sym, t.entry_ts, t.exit_ts, round(t.entry, 8),
                       round(t.exit, 8), round(t.pnl_pct, 12), t.reason, t.kind)
                      for t in trades)
        return hashlib.sha1(json.dumps(rows).encode()).hexdigest()[:16], len(rows)

    emit("")
    emit("## 升级自证 (2026-09-29: 默认离场 := stop + 一卖 S1)")
    for k in ("C", "G"):
        if k in trs:
            h, n = _fp(trs[k])
            emit(f"  {k}: {n} 笔   指纹 {h}")
    if "C" in trs and "G" in trs:
        same = _fp(trs["C"]) == _fp(trs["G"])
        emit(f"  → {'逐位相同 ✓' if same else '有差异 ✗'}   "
             f"(裸 BookCfg() ≡ exit_mode='stop' + sell_exit=('S1',))")

    emit("")
    emit("## 分标的配对(变好比例; 基准 = A 升级前默认)")
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
