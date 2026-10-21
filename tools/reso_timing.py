"""共振的**时间对齐**测量 —— 回答「长周期和短周期到底对得上表吗」。

背景: 共振口径是「长周期开窗口, 短周期给触发」——窗口只向**未来**开
(`[L.ts, L.ts + l_win × 长周期tf]`)。但长周期买点是**滞后确认**的
(B1 要等后面一笔往上走才认), 而短周期更快, 常常在长周期被确认**之前**就已经给点了。
若真是这样, 那共振就是**系统性漏点**, 与参数无关 —— 这也解释了为什么共振信号
稀到 9 年 91 个标的只有 140 笔。

测量(全池、全历史、一次遍历):
  1. 逐 bar 记下所有长周期买点事件 (`fresh_long`) 与短周期买点事件 (`short`);
  2. 对每个短周期买点, 找**最近**的一个满足 `combo_ok` 的长周期买点, 记
     Δ = 短周期时间 − 长周期时间;
  3. 按 Δ 分桶: Δ<0 = 短周期**先到**(落在窗口之前 ⇒ 共振抓不到),
     0 ≤ Δ ≤ win = **落在窗内**(共振能抓), Δ>win = 太晚;
  4. 与「随机对齐」的期望比较: 若两级完全独立, 落在窗内的期望条数
     ≈ n_短 × n_长 × win / 总时长。

用法:
  CHANBOOK_ARCH=$PWD/out/arch python tools/reso_timing.py --jobs 8 --out out/reso_timing.txt
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from chanbook import data as D                                # noqa: E402
from chanbook.resonance import DEFAULT_L_WIN, ResoScanner, combo_ok   # noqa: E402

MS_HOUR = D.MS_HOUR
L_MIN_BARS = 200


def one(task: tuple) -> tuple:
    sym, market, l_tf, s_tf, l_win, invalidate = task
    rs = ResoScanner(market, sym, tf_pair=(l_tf, s_tf), l_win=l_win, invalidate=invalidate)
    try:
        rs.load()
    except Exception as e:                                    # noqa: BLE001
        return sym, [], [], 0, 0, f"{e}"
    if rs.lkl.n < L_MIN_BARS or not rs.skl.n:
        return sym, [], [], 0, 0, f"数据不足(1h {rs.skl.n} / {l_tf} {rs.lkl.n})"
    lg: list[tuple] = []
    sh: list[tuple] = []
    for i in range(rs.skl.n):
        ts = int(rs.skl.ts[i])
        rs.advance_to(ts)
        ev = rs.step(ts, float(rs.skl.c[i]))
        for q in ev.get("fresh_long") or []:
            lg.append((int(q["ts"]), q["kind"]))
        for p in ev.get("short") or []:
            sh.append((ts, p.kind))
    return sym, lg, sh, int(rs.skl.ts[0]), int(rs.skl.ts[-1]), ""


def _mp(task: tuple) -> tuple:
    return one(task)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=["spot", "um"])
    ap.add_argument("--l-tf", default="")
    ap.add_argument("--s-tf", default="")
    ap.add_argument("--l-win", type=int, default=DEFAULT_L_WIN)
    ap.add_argument("--no-invalidate", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--null", type=int, default=200,
                    help="循环平移零假设的重复次数(0=跳过)")
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "reso_timing.txt"))
    args = ap.parse_args()

    from chanbook.resonance import RESO_TF
    l_tf, s_tf = (args.l_tf, args.s_tf) if (args.l_tf and args.s_tf) else RESO_TF[args.market]
    inv = not args.no_invalidate
    syms = D.symbols(args.market)
    if args.limit:
        syms = syms[: args.limit]
    tasks = [(s, args.market, l_tf, s_tf, args.l_win, inv) for s in syms]

    t0 = time.time()
    if args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(args.jobs) as pool:
            res = list(pool.imap_unordered(_mp, tasks))
    else:
        res = [_mp(t) for t in tasks]

    span_ms = 0
    n_lg = n_sh = 0
    buckets: Counter = Counter()
    near: list[float] = []
    in_obs = in_pre = 0
    n_long_b1 = n_short_b1 = 0
    n_pairs = n_pairs_pre = 0
    e_pairs = 0.0
    bad: list[str] = []
    win_h = args.l_win * D.TF_MS[l_tf] / MS_HOUR
    for sym, lg, sh, t_a, t_b, err in res:
        if err:
            bad.append(f"{sym}({err})")
            continue
        span_ms += (t_b - t_a)
        hours_sym = (t_b - t_a) / MS_HOUR
        n_lg += len(lg)
        n_sh += len(sh)
        n_long_b1 += sum(1 for _, k in lg if k == "B1")
        n_short_b1 += sum(1 for _, k in sh if k == "B1")
        # 随机对齐的期望(必须**按标的**算: A 的短周期不可能配到 B 的长周期)
        e_pairs += (len(sh) * len(lg) * 2 * win_h) / hours_sym if hours_sym > 0 else 0.0
        elig = list(lg)
        for ts, sk in sh:
            cand = [t for t, lk in elig if combo_ok(lk, sk)]
            if not cand:
                continue
            for t in cand:                       # 视图B: 数「合法配对」
                d = (ts - t) / MS_HOUR
                if -win_h <= d <= win_h:
                    n_pairs += 1
                    if d < 0:
                        n_pairs_pre += 1
            t = min(cand, key=lambda x: abs(x - ts))
            d_h = (ts - t) / MS_HOUR
            near.append(d_h)
            if 0 <= d_h <= win_h:
                buckets["窗内(能抓)"] += 1
                in_obs += 1
            elif -win_h <= d_h < 0:
                buckets["窗内反向(短先到)"] += 1
                in_pre += 1
            elif d_h < 0:
                buckets["窗前更远"] += 1
            else:
                buckets["窗后太晚"] += 1

    hours = span_ms / MS_HOUR
    ends = [t_b for _s, _l, _h, _a, t_b, err in res if not err]
    out = ["=" * 100,
           f"共振时间对齐测量   {args.market} {l_tf}(长) + {s_tf}(短)   "
           f"l_win={args.l_win} ({win_h:.0f}h)",
           f"生成 {time.strftime('%Y-%m-%d %H:%M')}   "
           f"标的 {len(res) - len(bad)} 个   "
           f"数据末端 {D.fmt(max(ends)) if ends else '—'}(北京)",
           "=" * 100, ""]
    out.append(f"长周期买点事件 {n_lg} 个(其中 B1 {n_long_b1}) · "
               f"短周期买点事件 {n_sh} 个(其中 B1 {n_short_b1})")
    out.append(f"总标的×时长 {hours / 24:,.0f} 天 ({hours / 24 / 365:,.2f} 年)")
    out.append("")
    tot = sum(buckets.values()) or 1
    out.append("【视图A · 每个短周期买点归类到「最近的一个合法长周期买点」】")
    out.append(f"{'分类':<20}{'条数':>9}{'占比':>9}")
    out.append("-" * 100)
    for k in ("窗内(能抓)", "窗内反向(短先到)", "窗前更远", "窗后太晚"):
        out.append(f"{k:<20}{buckets[k]:>9}{buckets[k] / tot:>8.1%}")
    out.append("")
    out.append("【视图B · 合法配对数(每个短周期对所有合法长周期) vs 随机期望】")
    out.append(f"  解析期望 ≈ Σ标的 n短 × n长 × (2×win) / 该标的时长 = {e_pairs:,.0f} 对"
               "   ⚠ 这个基准**偏乐观**: 它假设长周期事件在时间上均匀分布,")
    out.append("     而实际上长周期买点是**成簇**出现的(一段下跌末端会连着出 B1/B2/B3),")
    out.append("     成簇会让 ±窗口 的并集覆盖变短 ⇒ 随机命中数应低于解析值。")
    out.append(f"  实测落进 [-win, +win] 的: {n_pairs} 对 (反向 {n_pairs_pre} 对)")
    if e_pairs:
        out.append(f"  ⇒ 实测 / 解析期望 = {n_pairs / e_pairs:.2f}×  "
                   f"({'**低于**随机对齐' if n_pairs < e_pairs else '高于随机对齐'})")

    # ---- 循环平移零假设(比解析公式更可靠) ----
    if args.null > 0:
        import random
        from bisect import bisect_left, bisect_right
        rng = random.Random(args.seed)
        win_ms = args.l_win * D.TF_MS[l_tf]
        pack = []
        for _sym, lg, sh, t_a, t_b, err in res:
            if err:
                continue
            pack.append(([t for t, _ in lg], [k for _, k in lg], sh, t_b - t_a, t_a))
        nulls = []
        for _ in range(args.null):
            tot_n = 0
            for lts, lks, sh, span, t_a in pack:
                if span <= 0 or not lts or not sh:
                    continue
                off = rng.random() * span
                for ts, sk in sh:
                    s = t_a + ((ts - t_a + off) % span)
                    lo = bisect_left(lts, s - win_ms)
                    hi = bisect_right(lts, s + win_ms)
                    for k in range(lo, hi):
                        if combo_ok(lks[k], sk):
                            tot_n += 1
            nulls.append(tot_n)
        nulls.sort()
        mean = sum(nulls) / len(nulls)
        pct = sum(1 for v in nulls if v <= n_pairs) / len(nulls)
        out.append("")
        out.append(f"【零假设 · 把每个标的的短周期序列在**自身时长内循环平移** "
                   f"(n={args.null}, seed={args.seed})】")
        out.append(f"  平移后合法配对数: 均值 {mean:.0f} · p5 {nulls[int(len(nulls) * .05)]} · "
                   f"p50 {nulls[len(nulls) // 2]} · p95 {nulls[int(len(nulls) * .95)]}")
        out.append(f"  实测 {n_pairs} 对 ⇒ 分位 {pct:.1%} (越低 ⇒ 两级越是**互相回避**)")
        if pct < 0.05:
            out.append("  ⇒ 显著低于零假设: 长/短两级的买点在时间上**互相回避**, "
                       "不是「稀有但精准」")
        elif pct > 0.95:
            out.append("  ⇒ 显著高于零假设: 两级确实倾向于同时出现")
        else:
            out.append("  ⇒ 与零假设无法区分")
    out.append("")
    if near:
        near.sort()
        q = lambda p: near[min(len(near) - 1, int(len(near) * p))]   # noqa: E731
        out.append("【Δ 分位(小时, 短 − 长; 负=长周期更晚才确认)】")
        out.append("  " + "  ".join(f"p{int(p * 100)}={q(p):+.0f}h"
                                    for p in (0.05, 0.25, 0.5, 0.75, 0.95)))
        neg = sum(1 for x in near if x < 0)
        out.append(f"  短周期买点里, 长周期「更晚才确认」的占 {neg / len(near):.1%} "
                   f"—— 这些天然不可能被正向窗口抓到")
    out.append("")
    out.append("【细粒度分布(小时)】")
    edges = [(-1e9, -24 * 7, "≤ -7天"), (-24 * 7, -48, "-7~-2天"),
             (-48, -24, "-48~-24h"), (-24, -12, "-24~-12h"),
             (-12, 0, "-12~0h(窗内反向)"), (0, 12, "0~12h(窗内 ✓)"),
             (12, 24, "12~24h"), (24, 48, "24~48h"),
             (48, 24 * 7, "2~7天"), (24 * 7, 1e9, "≥ 7天")]
    for lo, hi, lab in edges:
        cnt = sum(1 for x in near if lo <= x < hi)
        bar = "█" * int(round(cnt / max(1, tot) * 60))
        out.append(f"  {lab:<18}{cnt:>7}  {cnt / tot:>6.1%}  {bar}")
    if bad:
        out.append("")
        out.append(f"跳过 {len(bad)}: " + ", ".join(bad[:10]))

    txt = "\n".join(out)
    print(txt)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(txt + "\n")
    print(f"\n耗时 {time.time() - t0:.0f}s  → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
