"""结构强度选币对照实验 —— 缠论从「触发器」转「过滤器」的第一个实证检验。

背景(2026-09-30):
  三轮对照已证伪「缠论择时」在现货 1h 无 alpha(进场 p=92%~100% 跑输随机、离场 alpha→0)。
  但「标的选择」维度从未被证伪。本实验回答一个问题:
    缠论买卖点的「结构质量」能否区分「后续会大涨的标的」vs「平庸标的」?
    若能, 则「只持有结构强度最高的少数标的」应跑赢「全池等权」。

方法:
  1. 用 engine.run_all 跑全池(现货 1h, 生产默认 stop+S1), 收集每笔 Trade 的 feats + pnl。
  2. 定义结构强度评分 strength_score(feats, kind), 从背驰力度/中枢突破/回调质量提炼。
  3. 标的层聚合: 每个标的 = 其所有 trade 的评分均值(可加样本数惩罚)。
  4. 三路对照:
       A. 全池等权(基准, 用 pool_index 的等权收益率 + 全池 trades 统计)
       B. 结构强度前 N% 标的(只保留这些标的的 trades)
       C. 随机选 N% 标的(多种子, 看「结构强度选币」是否显著优于随机)
  5. 口径复用 backtest_engine: signed_pairs / trimmed / per_day / stats_row。

诚实前提:
  本实验只测「选币」这一个维度, 不改任何入场/离场逻辑, 复用生产默认口径。
  若结构强度选币仍无 alpha, 则缠论在现货 1h 的最后一条「未证伪路径」也被关闭。
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook.data import symbols as D_symbols
from tools import backtest_engine as BE

# ------------------------------------------------------------------ 评分函数


def strength_score(feats: dict, kind: str) -> float:
    """从一笔 Trade 的结构化特征提炼「结构强度」分(0~1 量级, 越高越强)。

    原则:
      - 只用「结构质量」特征, 不用价格量纲(已在 strategy 归一为 _pct)。
      - B1/B2/B3 三类特征不同, 分别打分, 缺失字段给中性 0。
      - 方向: 背驰越强(ratio 越小=力度衰减越多)、突破中枢越有力、回调越浅 → 分越高。
    """
    if not feats:
        return 0.0
    s = 0.0

    if kind == "B1":
        # 背驰力度: ratio_area/per/peak/hist 越小 = 后段力度越弱 = 背驰越强
        ratio = feats.get("ratio_area", 1.0)
        if ratio and ratio < 1e8:
            s += max(0.0, 1.0 - min(float(ratio), 2.0)) * 0.5  # ratio<1 给正分
        # 背驰评分(4 分最典型, 但作弱信号)
        sc = feats.get("beichi_score", 0)
        s += (float(sc) - 3.0) * 0.05
        # 区间套确认通过(+)
        if feats.get("confirm_ok", 0) > 0:
            s += 0.15
        # 逆势接刀: 幅度比越小越接近极端(宽止损代价), 弱化
        amp_ratio = feats.get("amp_ratio", 1.0)
        if amp_ratio and amp_ratio < 1e8:
            s += max(0.0, 1.0 - min(float(amp_ratio), 2.0)) * 0.2

    elif kind == "B2":
        # 回调越浅(retest_up_pct 越大 = 越接近不破前低)越好
        ret = feats.get("retest_up_pct", 0.0)
        s += min(max(float(ret) * 10.0, -0.5), 0.5) * 0.6
        # 一买有中枢依托更好
        if feats.get("b1_had_zs", 0) > 0:
            s += 0.2
        # 回调笔数越少越干净
        pull = feats.get("pull_bi", 1)
        s += max(0.0, 0.2 - float(pull) * 0.05)

    elif kind == "B3":
        # 突破中枢越有力(exit_break_pct 越大)越好
        brk = feats.get("exit_break_pct", 0.0)
        s += min(max(float(brk) * 5.0, 0.0), 0.5) * 0.6
        # 回踩不跌回中枢(above_zg_pct 越大越好)
        abv = feats.get("above_zg_pct", 0.0)
        s += min(max(float(abv) * 10.0, -0.3), 0.3) * 0.4
        # 中枢高度(波动空间)适中为正
        zh = feats.get("zs_height_pct", 0.0)
        s += min(max(float(zh), 0.0), 0.1) * 0.5

    return s


def sym_strength(trades: list) -> dict:
    """标的层聚合: {sym: (平均强度分, 笔数)}。只统计有 feats 的成交。"""
    acc: dict[str, list] = defaultdict(list)
    for t in trades:
        if t.feats:
            acc[t.sym].append(strength_score(t.feats, t.kind))
    return {s: (sum(v) / len(v), len(v)) for s, v in acc.items()}


# ------------------------------------------------------------------ 统计口径


def stats(tr: list) -> dict:
    """单笔口径的净期望 + 样本数 + 剔尾/中位。"""
    xs = [t.pnl_pct for t in tr]
    n = len(xs)
    if not n:
        return {"n": 0, "exp": 0.0, "trim": 0.0, "med": 0.0}
    mean = sum(xs) / n
    k = int(n * 0.05)
    xs2 = sorted(xs)
    trim = sum(xs2[k:n - k]) / (n - 2 * k) if n - 2 * k >= 5 else mean
    return {"n": n, "exp": mean, "trim": trim, "med": statistics.median(xs)}


def signed_pairs(a: dict, b: dict, min_n: int = 3) -> tuple[int, int, float]:
    common = [s for s in a if s in b]
    if not common:
        return 0, 0, 0.0
    better = sum(1 for s in common if a[s] > b[s])
    return len(common), better, better / len(common)


def sym_means(tr: list) -> dict:
    acc: dict[str, list] = defaultdict(list)
    for t in tr:
        acc[t.sym].append(t.pnl_pct)
    return {s: sum(v) / len(v) for s, v in acc.items()}


# ------------------------------------------------------------------ 主流程


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--market", default="spot")
    ap.add_argument("--top", type=float, default=0.30, help="选币比例(前 N% 标的)")
    ap.add_argument("--seeds", type=int, default=30, help="随机对照种子数")
    ap.add_argument("--limit", type=int, default=0, help="限制标的数(0=全池)")
    ap.add_argument("--min-trades", type=int, default=3,
                    help="标的至少多少笔才参与排序")
    ap.add_argument("--jobs", type=int, default=0)
    args = ap.parse_args()

    t0 = time.time()
    syms = D_symbols(args.market)
    if args.limit:
        syms = syms[: args.limit]
    print(f"[数据] 标的 {len(syms)} 个  tf={args.tf}", file=sys.stderr)

    # 生产默认口径 = kinds (B1,B2) + 离场 "s1"(结构止损 + 一卖 S1), 方向闸门开。
    # ⚠ kinds 必须大写: build_cfg → scanner.enable 用 B1/B2/B3 常量判定。
    v = {"name": f"strength {args.tf}", "tf": args.tf, "kinds": "b1,b2",
         "exit": "s1"}
    print("[回测] 全池跑生产默认口径...", file=sys.stderr)
    trades, diags = BE.run_variant(syms, v, args.tf, args.market, ("B1", "B2"),
                                   gate=True, t0=0, t1=0, jobs=args.jobs,
                                   verbose=True)
    print(f"[回测] 全池成交 {len(trades)} 笔, 用时 {time.time()-t0:.0f}s",
          file=sys.stderr)

    if not trades:
        print("无成交, 退出", file=sys.stderr)
        return 1

    # ---------- A. 全池等权(基准) ----------
    A = stats(trades)
    A_sym = sym_means(trades)

    # ---------- B. 结构强度前 N% 标的 ----------
    ss = sym_strength(trades)
    # 只保留交易笔数达标的标的
    qual = {s: v for s, v in ss.items() if v[1] >= args.min_trades}
    ranked = sorted(qual.items(), key=lambda kv: kv[1][0], reverse=True)
    k = max(1, int(len(ranked) * args.top))
    top_syms = {s for s, _ in ranked[:k]}
    B_tr = [t for t in trades if t.sym in top_syms]
    B = stats(B_tr)
    B_sym = sym_means(B_tr)

    # ---------- C. 随机选 N% 标的(多种子) ----------
    pool_syms = sorted({t.sym for t in trades})
    n_pick = max(1, int(len(pool_syms) * args.top))
    C_exp = []
    C_sym_means = {}
    for seed in range(args.seeds):
        rng = random.Random(seed)
        pick = set(rng.sample(pool_syms, n_pick))
        C_tr = [t for t in trades if t.sym in pick]
        C_exp.append(stats(C_tr)["exp"])
        C_sym_means[seed] = sym_means(C_tr)
    C_mean = sum(C_exp) / len(C_exp)
    # 结构强度选币 vs 随机: 用配对(公共标的上 B 均值 > 随机均值 的比例)的均值
    pair_ratios = []
    for seed in range(args.seeds):
        _, _, ratio = signed_pairs(B_sym, C_sym_means[seed], min_n=1)
        pair_ratios.append(ratio)
    avg_pair = sum(pair_ratios) / len(pair_ratios)

    # ---------- D. 时间分段检验(消除前视: 前半算分 / 后半验证) ----------
    ts_mid = sorted(t.entry_ts for t in trades)[len(trades) // 2]
    tr_early = [t for t in trades if t.entry_ts <= ts_mid]
    tr_late = [t for t in trades if t.entry_ts > ts_mid]
    ss_early = sym_strength(tr_early)
    qual_e = {s: v for s, v in ss_early.items() if v[1] >= max(2, args.min_trades - 1)}
    ranked_e = sorted(qual_e.items(), key=lambda kv: kv[1][0], reverse=True)
    k_e = max(1, int(len(ranked_e) * args.top))
    top_syms_e = {s for s, _ in ranked_e[:k_e]}
    # 用前半段选出的标的, 验证后半段它们的表现
    D_tr = [t for t in tr_late if t.sym in top_syms_e]
    D = stats(D_tr)
    late_all = stats(tr_late)
    late_pool = sorted({t.sym for t in tr_late})
    n_pick_late = max(1, int(len(late_pool) * args.top))
    D_exp_rand = []
    for seed in range(args.seeds):
        rng = random.Random(seed + 1000)
        pick = set(rng.sample(late_pool, n_pick_late))
        D_exp_rand.append(stats([t for t in tr_late if t.sym in pick])["exp"])
    D_rand_mean = sum(D_exp_rand) / len(D_exp_rand)

    # ---------- E. 分位单调性(5 档) ----------
    qual_q = {s: v for s, v in ss.items() if v[1] >= args.min_trades}
    ranked_q = sorted(qual_q.items(), key=lambda kv: kv[1][0])
    nq = len(ranked_q)
    quint = []
    if nq >= 5:
        for q in range(5):
            lo = nq * q // 5
            hi = nq * (q + 1) // 5
            syms_q = {s for s, _ in ranked_q[lo:hi]}
            tr_q = [t for t in trades if t.sym in syms_q]
            quint.append(stats(tr_q)["exp"])
    else:
        quint = []

    # ---------- 输出 ----------
    print()
    print("=" * 72)
    print(f"结构强度选币对照  tf={args.tf}  market={args.market}  top={args.top}")
    print("=" * 72)
    print(f"{'口径':<22}{'笔数':>8}{'单笔期望%':>12}{'剔尾5%%':>10}{'中位%':>10}")
    print(f"{'A 全池等权(基准)':<22}{A['n']:>8}{A['exp']*100:>12.3f}"
          f"{A['trim']*100:>10.3f}{A['med']*100:>10.3f}")
    print(f"{'B 结构强度前N%':<22}{B['n']:>8}{B['exp']*100:>12.3f}"
          f"{B['trim']*100:>10.3f}{B['med']*100:>10.3f}")
    print(f"{'C 随机选N%(均值)':<22}{'-':>8}{C_mean*100:>12.3f}"
          f"{'-':>10}{'-':>10}")
    print("-" * 72)
    print(f"B vs A: 单笔期望差 {(B['exp']-A['exp'])*100:+.3f}%")
    print(f"B vs C: 单笔期望差 {(B['exp']-C_mean)*100:+.3f}%")
    print(f"B vs C 配对(标的层): B 均值更优比例 = {avg_pair:.3f}")
    print(f"结构强度选中的标的数: {len(top_syms)} / 有成交标的 {len(pool_syms)}")
    print()
    # ---------- G. 右尾依赖检验 ----------
    def tail_contrib(tr):
        xs = sorted(t.pnl_pct for t in tr)
        n = len(xs)
        if not n:
            return 0.0, 0.0
        top1 = xs[-1]
        # 去掉最大一笔后的均值
        rest = sum(xs[:-1]) / (n - 1) if n > 1 else 0.0
        return top1, rest
    a_top1, a_rest = tail_contrib(trades)
    b_top1, b_rest = tail_contrib(B_tr)
    print("G 右尾依赖(去掉最大单笔后):")
    print(f"  全池: 最大单笔={a_top1*100:.1f}%  去最大后均值={a_rest*100:.3f}%")
    print(f"  B选币: 最大单笔={b_top1*100:.1f}%  去最大后均值={b_rest*100:.3f}%")
    print(f"  => 去最大单笔后 B vs A 差额: {(b_rest-a_rest)*100:+.3f}%")
    print()
    print("D 时间分段(前半算分 → 后半验证, 消除前视):")
    print(f"  后半段全池: 笔数={late_all['n']}  期望={late_all['exp']*100:.3f}%")
    print(f"  前半段选中的标的在后半段: 笔数={D['n']}  期望={D['exp']*100:.3f}%")
    print(f"  后半段随机选币均值: 期望={D_rand_mean*100:.3f}%")
    print(f"  => 结构强度选币 vs 随机: {(D['exp']-D_rand_mean)*100:+.3f}%")
    print()
    if quint:
        print("E 分位单调性(强度从低到高 5 档的单笔期望%):")
        for q, e in enumerate(quint):
            bar = "█" * max(1, int(abs(e) * 30))
            print(f"  Q{q+1} [{e*100:+6.3f}%] {bar}")
        mono = all(quint[i] <= quint[i + 1] for i in range(4))
        print(f"  单调递增? {'是(有选币力)' if mono else '否(无选币力/噪声)'}")
    print()

    # ---------- F. 单特征扫描(排除"评分函数没设计好"的干扰) ----------
    print("F 单特征选币扫描(每个 feats 字段独立做 top30% 选币, 看有无自带选币力):")
    print(f"{'特征':<18}{'标的均值差%':>14}{'vs随机%':>12}")
    all_syms = sorted({t.sym for t in trades})
    # 每个标的每个特征取均值
    for fname in ("beichi_score", "ratio_area", "ratio_per", "ratio_peak",
                  "ratio_hist", "amp_ratio", "amp_a_pct", "amp_c_pct",
                  "atr_pct", "gap_entry_struct", "confirm_ok",
                  "retest_up_pct", "zs_height_pct", "above_zg_pct",
                  "exit_break_pct", "exit_leg_amp_pct"):
        acc: dict[str, list] = defaultdict(list)
        for t in trades:
            v = t.feats.get(fname)
            if v is None:
                continue
            try:
                acc[t.sym].append(float(v))
            except (TypeError, ValueError):
                continue
        if len(acc) < 10:
            continue
        sym_mean = {s: sum(v) / len(v) for s, v in acc.items() if len(v) >= 2}
        if len(sym_mean) < 10:
            continue
        ranked_f = sorted(sym_mean.items(), key=lambda kv: kv[1], reverse=True)
        kf = max(1, int(len(ranked_f) * args.top))
        top_f = {s for s, _ in ranked_f[:kf]}
        bot_f = {s for s, _ in ranked_f[-kf:]}
        top_tr = [t for t in trades if t.sym in top_f]
        bot_tr = [t for t in trades if t.sym in bot_f]
        e_top = stats(top_tr)["exp"] if top_tr else 0.0
        e_bot = stats(bot_tr)["exp"] if bot_tr else 0.0
        print(f"{fname:<18}{((e_top-e_bot)*100):>14.3f}{(e_top*100):>12.3f}")
    print()
    print("选中标的(按强度降序, 前 15):")
    for s, (sc, n) in ranked[:15]:
        print(f"  {s:<18} 强度={sc:+.4f}  笔数={n}")
    print()
    print(f"总用时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
