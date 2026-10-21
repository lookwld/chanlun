"""离场 alpha 对照 —— 被动持有 + 缠论离场 vs 被动持有 + 朴素离场。

## 背景（上一轮随机入场对照的结论）

`backtest_random.py` 证明"进场择时"是负 alpha（训练段 p=92%~100%，真实显著跑输随机）。
但发现了一个关键现象：缠论 B1/B2 买点出现在底部区域，**往往不触发 S1 卖点**（底部
无上涨段形成一卖背驰），导致持仓被锁在底部、均持仓拖到 25 天、错过牛市最快 beta。

这暗示：**缠论的价值可能不在"何时进"，而在"何时出"（离场）**。本工具验证这一点。

## 实验设计：把进场完全固定，只让离场变化

要回答"离场是否有 alpha"，必须让**进场侧零择时**，两组唯一差异是离场规则：

  * **进场（两组完全相同）**：每 bar 以固定概率随机入场（与上一轮随机对照完全同款，
    止损位取最近冻结向下笔低点，用 risk.plan 构造）。进场侧不含任何缠论买点信息。
  * **A 组 · 缠论离场**：入场后走"结构止损 + 一卖 S1"（生产默认 stop+S1 的离场链）。
  * **B 组 · 朴素离场**：入场后**固定持有 N 根 bar**后离场（无任何缠论信息），
    或固定持有 N 天后离场。止损仍保留（否则变成"扛单"，不公平）。

两组进场笔数、进场时刻、止损位**逐笔对齐**（同一随机种子、同一 rate），
唯一差异是 A 多了"S1 卖点提前离场"这条通道。差值 = 离场 alpha。

## 判决口径

  * 若 A（缠论离场）的每持仓日 / 单笔净显著优于 B（朴素离场），且持仓时长对齐，
    ⇒ 离场有正 alpha。
  * 若 A ≈ B 或 A 更差 ⇒ 缠论离场无 alpha，甚至"过早离场砍掉了右尾"（负 alpha）。

运行示例：
  python tools/backtest_exit_alpha.py --jobs 10 --out out/exit_alpha.txt
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402

D.TF_MS.setdefault("2h", 2 * D.MS_HOUR)
D.TF_MS.setdefault("3h", 3 * D.MS_HOUR)
D.TF_MS.setdefault("8h", 8 * D.MS_HOUR)

import tools.backtest_engine as BT                                # noqa: E402
import tools.backtest_random as BR                                # noqa: E402

MK = BR.MK                       # "spot"
TF = BR.TF                       # "1h"
SEGMENTS = BR.SEGMENTS

SEEDS = 40
CALIB_ITER = BR.CALIB_ITER
CALIB_TOL = BR.CALIB_TOL

# 朴素离场的固定持有期（bar 数，1h ⇒ N 小时）
NAIVE_HOLD_BARS = [24, 48, 168, 336]     # 1 天 / 2 天 / 1 周 / 2 周


def _run_exit_symbol(sym: str, rows: list, n_real: int, hold_bars: int):
    """单标的：进场固定（随机），离场分两组（缠论 vs 朴素固定持有）。

    返回 (缠论离场成交列表, 朴素离场成交列表)。进场逐笔对齐。
    `rows` / `n_real` 由调用方预先录好（跨持有期档位共享，避免重复推进级别）。
    """
    seed0 = BR.zlib_crc(sym)

    # 进场：与上一轮随机对照完全同款（rate 校准对齐笔数）
    rate = BR.rate0_guess(n_real, len(rows))
    chan_outs = []
    naive_outs = []
    for _ in range(max(1, CALIB_ITER)):
        rngs = [random.Random(seed0 * 1000003 + s * 2654435761)
                for s in range(SEEDS)]
        chan_pos = [None] * SEEDS
        naive_pos = [None] * SEEDS      # 存 (pos, 剩余持有bar数)
        chan_outs = [[] for _ in range(SEEDS)]
        naive_outs = [[] for _ in range(SEEDS)]
        for (ts, c, lo, struct, atr, sell_kind) in rows:
            for s in range(SEEDS):
                # ---- 缠论离场组 ----
                pos = chan_pos[s]
                if pos is not None:
                    if sell_kind:
                        chan_outs[s].append(BT._close(pos, c, ts, "sell_" + sell_kind))
                        chan_pos[s] = None
                    elif lo <= pos["stop"]:
                        chan_outs[s].append(BT._close(pos, pos["stop"], ts, "stop_loss"))
                        chan_pos[s] = None
                # ---- 朴素离场组（固定持有）----
                np_ = naive_pos[s]
                if np_ is not None:
                    pos2, remain = np_
                    remain -= 1
                    if remain <= 0:
                        naive_outs[s].append(BT._close(pos2, c, ts, "naive_hold"))
                        naive_pos[s] = None
                    elif lo <= pos2["stop"]:
                        naive_outs[s].append(BT._close(pos2, pos2["stop"], ts, "stop_loss"))
                        naive_pos[s] = None
                    else:
                        naive_pos[s] = (pos2, remain)
                # ---- 进场（两组同款，仅当该种子两组都空仓才进场，保证逐笔对齐）----
                if chan_pos[s] is None and naive_pos[s] is None:
                    if rngs[s].random() >= rate:
                        continue
                    if struct is None:
                        continue
                    pl = BT.risk.plan("long", c, struct, atr, MK, BR.B1)
                    if not pl.ok:
                        continue
                    sig = SimpleNamespace(sym=sym, side="long", kind=BR.B1,
                                          stop=pl.stop, target=pl.target, trend="",
                                          counter=1, confirm="", risk_pct=pl.risk_pct,
                                          notional=pl.notional, feats={})
                    chan_pos[s] = BT.open_pos(sig, "passive", ts, c)
                    naive_pos[s] = (BT.open_pos(sig, "passive", ts, c), hold_bars)
        got = sorted(len(o) for o in chan_outs)[SEEDS // 2]
        if got <= 0 or abs(got - n_real) / n_real <= CALIB_TOL:
            break
        rate = min(0.9, rate * n_real / got)

    return chan_outs, naive_outs


def _agg(outs: list) -> dict:
    """把多种子的成交列表聚合成每持仓日/单笔净的中位。"""
    from chanbook.engine import Diag
    nets, pds, holds, exps = [], [], [], []
    for tr in outs:
        dg = [Diag(sym="", n_op_bars=1) for _ in range(max(1, len(tr)))]
        r = BT.stats_row(tr, MK, TF, dg, "x")
        nets.append(r.get("net", 0.0))
        pds.append(r.get("per_day", 0.0))
        holds.append(r.get("hold_d", 0.0))
        exps.append(r.get("exp", 0.0))
    return {
        "net": statistics.median(nets), "per_day": statistics.median(pds),
        "hold": statistics.median(holds), "exp": statistics.median(exps),
        "net_list": nets, "pd_list": pds,
    }


def main() -> int:
    global TF
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--tf", default=TF, help="操作级别时间周期: 1h/3h/4h/8h/1d")
    ap.add_argument("--out", default="out/exit_alpha.txt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hold-bars", type=int, default=0,
                    help="固定持有期(bar)，0=测全部默认档位")
    args = ap.parse_args()

    TF = args.tf
    BR.TF = args.tf          # 让 backtest_random._record_schedule 用同周期
    if TF not in D.TF_MS:
        raise ValueError(f"未支持周期 {TF}")

    syms = D.symbols(MK)
    if args.limit:
        syms = syms[: args.limit]
    holds = [args.hold_bars] if args.hold_bars else NAIVE_HOLD_BARS

    lines: list[str] = []
    emit = lines.append
    emit("=" * 78)
    emit("离场 alpha 对照 —— 被动持有 + 缠论离场 vs 被动持有 + 朴素(固定持有)离场")
    emit(f"标的 {len(syms)} · 种子 {args.seeds} · 进场=随机(两组逐笔对齐) · 只变离场规则")
    emit("=" * 78)

    for seg_name, (ym0, ym1) in SEGMENTS.items():
        t0 = BT.ms_of(ym0)
        t1 = BT.ms_of(ym1, end=True)
        emit("")
        emit(f"### {seg_name} {ym0} ~ {ym1}")
        # 先录好每个标的的 schedule（最昂贵，跨持有期档位共享一次）
        sched: dict = {}
        t_seg = time.time()
        for k, sym in enumerate(syms):
            rows = BR._record_schedule(sym, t0, t1)
            if rows is None:
                continue
            tr_real, _ = BT.run_symbol(sym, {"exit": "s1"}, TF, MK,
                                       (BR.B1, BR.B2), False, t0, t1)
            n_real = len(tr_real)
            if n_real == 0:
                continue
            sched[sym] = (rows, n_real)
            if (k + 1) % 30 == 0:
                print(f"    {seg_name} 录 schedule {k+1}/{len(syms)} "
                      f"{time.time()-t_seg:.0f}s", file=sys.stderr, flush=True)

        for hb in holds:
            t_seg = time.time()
            chan_all = [[] for _ in range(args.seeds)]
            naive_all = [[] for _ in range(args.seeds)]
            n_used = 0
            for sym, (rows, n_real) in sched.items():
                chan_outs, naive_outs = _run_exit_symbol(sym, rows, n_real, hb)
                if chan_outs is None:
                    continue
                n_used += 1
                for s in range(args.seeds):
                    chan_all[s].extend(chan_outs[s])
                    naive_all[s].extend(naive_outs[s])
            print(f"    {seg_name} hold={hb} {n_used} 标的 "
                  f"{time.time()-t_seg:.0f}s", file=sys.stderr, flush=True)

            a = _agg(chan_all)
            b = _agg(naive_all)
            emit(f"  [朴素持有 {hb}bar={hb/24:.1f}天] 参与标的 {n_used}")
            emit(f"    缠论离场: 单笔净 {a['net']:+.3%}  每持仓日 {a['per_day']:+.4%}  "
                 f"均持仓 {a['hold']:.2f}天")
            emit(f"    朴素离场: 单笔净 {b['net']:+.3%}  每持仓日 {b['per_day']:+.4%}  "
                 f"均持仓 {b['hold']:.2f}天")
            # p 值：缠论离场 优于 朴素离场 的比例（每持仓日）
            diff = [x - y for x, y in zip(a["pd_list"], b["pd_list"])]
            p = sum(1 for d in diff if d > 0) / len(diff)
            emit(f"    离场 alpha(缠论-朴素 每持仓日) 中位 "
                 f"{statistics.median(diff):+.4%}  优于比例 {p:.0%}")

    emit("")
    emit("=" * 78)
    emit("读法：离场 alpha 中位 > 0 且优于比例高 ⇒ 缠论离场有正 alpha；")
    emit("      ≈ 0 或为负 ⇒ 缠论离场无 alpha（甚至砍掉右尾）。")
    emit("=" * 78)

    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
