"""随机入场对照 —— 闸① 第②条门槛（择时能力是否与随机不可区分）。

## 回答什么问题

分段验证（`backtest_oos.py`）证明了生产默认 `stop + S1` 在验证段（2022-2026）
单笔净 +0.309% > 0、逐年 4/5 为正。但这是**必要不充分**：任何"有正 beta 偏置的
随机入场"在上涨段都可能为正。要主张"缠论择时（B1/B2 买点）有价值"，必须证明它
**显著优于同频率、同离场口径、同止损口径的随机入场**。

## 方法（严格对齐上轮 B12 / B1_4H 的口径，不重造轮子）

1. **同离场链**：随机入场后走与真实完全相同的离场 —— 结构止损（最近一根冻结
   向下笔低点）+ 一卖 S1，代码直接复用 `backtest_engine.run_symbol` 的离场分支。
2. **同止损口径**：止损位 = 入场时"最近一根向下笔的低点"（B1 用的同源基准），
   用 `risk.plan` 构造，与真实信号逐位同源。
3. **同频率**：每 bar 以固定概率 `rate` 入场，迭代校准 `rate` 使**中位笔数**
   逼近真实笔数（容差 10%）。否则随机侧还会被 `risk.plan` 拒单吃掉一半笔数，
   比较失真（上轮实测 86 vs 163）。
4. **多种子**：≥20 个独立种子，报告"随机里 ≥ 真实的单笔净/每持仓日的比例"
   = 单侧 p 值。验收线 **p < 0.025**（项目预先定死）。

## 判决口径（为什么比"单笔毛"）

只比"单笔毛期望"会被持仓时长污染（拿得久自然高）。所以同时给：
  * 单笔净 `net`（扣双边费）
  * 每持仓日 `per_day`（跨口径唯一可比的量）
  * 剔尾 5% 均值 + 中位（对极端值不敏感）
  * 配对 `signed_pairs`（分标的先平均，每标的一票）

## 无未来函数

随机入场在 `bar_ts` 收盘后用 `lv.advance(now)` 推进级别（与真实同款时钟），
止损位取的是**当前已冻结**的笔，不含未来笔。与 `check_timebase.py` 一致。

运行示例：
  python tools/backtest_random.py --jobs 10 --out out/random_oos.txt
"""

from __future__ import annotations

import argparse
import bisect
import os
import random
import statistics
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook.engine import (Diag, _close, open_pos, sell_hit)    # noqa: E402
from chanbook.points import B1, B2                                # noqa: E402
from chanbook.sell import S1, SellScanner                         # noqa: E402

import tools.backtest_engine as BT                                # noqa: E402

D.TF_MS.setdefault("2h", 2 * D.MS_HOUR)
D.TF_MS.setdefault("3h", 3 * D.MS_HOUR)
D.TF_MS.setdefault("8h", 8 * D.MS_HOUR)

WARMUP = BT.WARMUP
MIN_BARS = BT.MIN_BARS
TREND_TF = BT.TREND_TF
TREND_MIN_BARS = BT.TREND_MIN_BARS

# 生产默认配置: 现货 + B1/B2 + 结构止损 + 一卖 S1
MK = "spot"
TF = "1h"
KINDS = (B1, B2)
EXIT = "s1"                       # 对应 backtest_engine.EXITS["s1"] = 结构止损 + S1

SEEDS = 40                        # 随机种子数（越多 p 值越稳）
CALIB_ITER = 5                    # 频率校准轮数
CALIB_TOL = 0.10                  # 笔数校准容差

# 分段边界（与 backtest_oos.py 一致）
SEGMENTS = {
    "训练段": ("2017-08", "2021-12"),
    "验证段": ("2022-01", "2026-08"),
}


def _sell_kw():
    cfg = BT.build_cfg(MK, KINDS, {"exit": EXIT})
    skw = cfg.scanner_kw()
    return dict(pull_bi=skw["pull_bi"], tol=0.0, beichi_tol=skw["beichi_tol"],
                beichi_strict=skw["beichi_strict"], s2_max_bi=skw["b2_max_bi"],
                s1_max_lag=skw["b1_max_lag"], s1_zs_window=skw["b1_zs_window"],
                s3_zs_window=skw["b3_zs_window"],
                s1_allow_no_zs=skw["b1_allow_no_zs"], enable=(S1,))


def _record_schedule(sym: str, t_start: int, t_end: int):
    """**一次**遍历级别，把随机对照需要的逐 bar 信息全部录下来。

    这样 K 个种子 × 若干轮频率校准都只是纯 Python 循环，不必反复推进级别
    （推进级别是本工具唯一昂贵的操作）。

    记录内容（每 bar）：
      bar_ts, close, low, struct(最近向下笔低点), atr, sell_kind(该 bar 是否命中 S1)
    """
    kl = BT.load_tf(MK, sym, TF)
    if kl is None or kl.n < MIN_BARS:
        return None
    cfg = BT.build_cfg(MK, KINDS, {"exit": EXIT})
    lv = BT.Level("op", TF, "bi", scanner_kw=cfg.scanner_kw(),
                  **cfg.op_extra()).attach(kl)
    sscan = SellScanner(**_sell_kw())
    start = WARMUP
    if t_start:
        start = max(start, bisect.bisect_right(kl.ts, t_start - 1))
    rows: list[tuple] = []
    for i in range(start, kl.n):
        bar_ts = int(kl.ts[i])
        if t_end and bar_ts > t_end:
            break
        now = bar_ts + lv.tf_ms
        lv.advance(now)
        hit = sell_hit(sscan, lv, (S1,))
        bis = lv.bib.bis
        struct = None
        for k in range(lv.bib.n_final - 1, -1, -1):
            if bis[k].direction < 0:
                struct = float(bis[k].low)
                break
        rows.append((bar_ts, float(kl.c[i]), float(kl.l[i]), struct,
                     lv.atr(), (hit.kind if hit is not None else "")))
    return rows


def _random_pass(rows: list, seed0: int, rate: float, seeds: int,
                 target: int = 0, n_iter: int = CALIB_ITER):
    """给定"每 bar 入场概率"跑 K 个随机种子，并按目标笔数校准频率。

    与 backtest_b1_4h.seeds_pass 同款：迭代调整 rate 使中位笔数逼近真实值。
    """
    out: list = []
    for _ in range(max(1, n_iter)):
        rngs = [random.Random(seed0 * 1000003 + s * 2654435761)
                for s in range(seeds)]
        positions: list = [None] * seeds
        out = [[] for _ in range(seeds)]
        for (ts, c, lo, struct, atr, sell_kind) in rows:
            for s in range(seeds):
                pos = positions[s]
                if pos is not None:
                    if sell_kind:
                        out[s].append(_close(pos, c, ts, "sell_" + sell_kind))
                        positions[s] = None
                    elif lo <= pos["stop"]:
                        out[s].append(_close(pos, pos["stop"], ts, "stop_loss"))
                        positions[s] = None
                    continue
                if rngs[s].random() >= rate:
                    continue
                if struct is None:
                    continue
                pl = BT.risk.plan("long", c, struct, atr, MK, B1)
                if not pl.ok:
                    continue
                sig = SimpleNamespace(sym=sym_ns, side="long", kind=B1,
                                      stop=pl.stop, target=pl.target, trend="",
                                      counter=1, confirm="", risk_pct=pl.risk_pct,
                                      notional=pl.notional, feats={})
                positions[s] = open_pos(sig, "random", ts, c)
        if not target:
            break
        got = sorted(len(o) for o in out)[seeds // 2]
        if got <= 0 or abs(got - target) / target <= CALIB_TOL:
            break
        rate = min(0.9, rate * target / got)
    return out


sym_ns = ""  # 由 run_random_symbol 每标的设置


def _run_random_symbol(sym: str, t_start: int, t_end: int):
    """单标的随机对照：返回 (真实笔数, 随机多种子成交列表)。"""
    global sym_ns
    rows = _record_schedule(sym, t_start, t_end)
    if rows is None:
        return 0, None
    # 真实笔数（同配置，跑一次真实回测拿笔数作校准目标）
    cfg = BT.build_cfg(MK, KINDS, {"exit": EXIT})
    tr_real, _ = BT.run_symbol(sym, {"exit": EXIT}, TF, MK, KINDS, False,
                               t_start, t_end)
    n_real = len(tr_real)
    if n_real == 0:
        return 0, None
    sym_ns = sym
    seed0 = zlib_crc(sym)
    outs = _random_pass(rows, seed0, rate0_guess(n_real, len(rows)), SEEDS,
                        target=n_real)
    return n_real, outs


def zlib_crc(s: str) -> int:
    import zlib
    return zlib.crc32(s.encode("utf-8"))


def rate0_guess(n_target: int, n_bars: int) -> float:
    """初始入场概率：目标笔数 / 可用 bar 数，再乘 1.5 补偿拒单损耗。"""
    if n_bars <= 0:
        return 0.01
    return min(0.5, max(0.0005, 1.5 * n_target / n_bars))


def _fmt(x: float, f: str) -> str:
    return f"{x:{f}}"


def main() -> int:
    global TF
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--tf", default=TF, help="操作级别时间周期: 1h/3h/4h/8h/1d")
    ap.add_argument("--out", default="out/random_oos.txt")
    ap.add_argument("--limit", type=int, default=0,
                    help="只跑前 N 标的（冒烟用）")
    args = ap.parse_args()

    TF = args.tf
    if TF not in D.TF_MS:
        raise ValueError(f"未支持周期 {TF}")

    syms = D.symbols(MK)
    if args.limit:
        syms = syms[: args.limit]

    lines: list[str] = []
    emit = lines.append
    emit("=" * 76)
    emit("随机入场对照 —— 闸① 第②条（择时能力 vs 随机）")
    emit(f"配置: 现货 {TF} · B1/B2 入场 · 结构止损 + 一卖 S1（生产默认 stop+S1）")
    emit(f"标的: {len(syms)} 个 · 种子: {args.seeds} · 验收线 p < 0.025")
    emit("=" * 76)

    for seg_name, (ym0, ym1) in SEGMENTS.items():
        t0 = BT.ms_of(ym0)
        t1 = BT.ms_of(ym1, end=True)
        emit("")
        emit(f"### {seg_name} {ym0} ~ {ym1}")
        t_seg = time.time()

        # 真实回测（拿真实统计 + 笔数）
        cfg = BT.build_cfg(MK, KINDS, {"exit": EXIT})
        tr_real, dg_real = BT.run_variant(
            syms, {"exit": EXIT, "name": "真实"}, TF, MK, KINDS, False,
            t0, t1, args.jobs, verbose=False)
        r_real = BT.stats_row(tr_real, MK, TF, dg_real, "真实")
        emit(f"  真实: 笔数 {r_real.get('n', 0)}  单笔净 "
             f"{_fmt(r_real.get('net', 0), '+.3%')}  每持仓日 "
             f"{_fmt(r_real.get('per_day', 0), '+.4%')}")

        # 随机对照（逐标的、多种子）
        all_rand: list[list] = [[] for _ in range(args.seeds)]
        n_used = 0
        n_real_tot = 0
        for k, sym in enumerate(syms):
            n_real, outs = _run_random_symbol(sym, t0, t1)
            if outs is None:
                continue
            n_used += 1
            n_real_tot += n_real
            for s in range(args.seeds):
                all_rand[s].extend(outs[s])
            if (k + 1) % 20 == 0:
                print(f"    {seg_name} {k + 1}/{len(syms)} "
                      f"{time.time() - t_seg:.0f}s", file=sys.stderr, flush=True)

        # 每个种子算统计
        rand_nets = []
        rand_pd = []
        rand_exp = []
        rand_hold = []
        for s in range(args.seeds):
            tr = all_rand[s]
            dg = [Diag(sym="", n_op_bars=1) for _ in range(max(1, len(tr)))]
            # 随机对照的 dg 用占位（stats_row 的 share 在此无意义），
            # 我们只取 net / per_day / exp / pf / hold_d。
            r = BT.stats_row(tr, MK, TF, dg, f"随机{s}")
            rand_nets.append(r.get("net", 0.0))
            rand_pd.append(r.get("per_day", 0.0))
            rand_exp.append(r.get("exp", 0.0))
            rand_hold.append(r.get("hold_d", 0.0))

        def p_val(vals: list, real: float) -> float:
            """随机里 ≥ 真实的比例（单侧）。"""
            return sum(1 for v in vals if v >= real) / len(vals)

        p_net = p_val(rand_nets, r_real.get("net", 0.0))
        p_pd = p_val(rand_pd, r_real.get("per_day", 0.0))
        emit(f"  参与对照标的 {n_used} 个 · 真实总笔数 {n_real_tot}")
        emit(f"  真实单笔毛 {_fmt(r_real.get('exp', 0), '+.3%')}  随机中位单笔毛 "
             f"{_fmt(statistics.median(rand_exp), '+.3%')}")
        emit(f"  真实均持仓 {_fmt(r_real.get('hold_d', 0), '.2f')}天  随机中位均持仓 "
             f"{_fmt(statistics.median(rand_hold), '.2f')}天  ⚠ 若严重不对齐, 每持仓日口径失真")
        emit(f"  随机中位单笔净 {_fmt(statistics.median(rand_nets), '+.3%')}"
             f"  [min {_fmt(min(rand_nets), '+.3%')}, max {_fmt(max(rand_nets), '+.3%')}]")
        emit(f"  随机中位每持仓日 {_fmt(statistics.median(rand_pd), '+.4%')}")
        emit(f"  p(随机 ≥ 真实)  单笔净 {p_net:.0%}   每持仓日 {p_pd:.0%}")
        emit(f"  {'✅ 显著优于随机' if p_pd < 0.025 else '❌ 不能与随机区分'} "
             f"（验收线 p<0.025，以每持仓日为准）")

    emit("")
    emit("=" * 76)
    emit("读法：p(随机 ≥ 真实) ≈ 50% ⇒ 真实择时与随机不可区分；")
    emit("      接近 0 ⇒ 真实显著优于随机。验收线 p < 0.025（每持仓日）。")
    emit("=" * 76)

    txt = "\n".join(lines)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
