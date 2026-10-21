#!/usr/bin/env python
"""入场判别建模 —— **纯样本内多变量基线**（`DATASET.md` §5.4 第①步）。

本脚本只做一件事：回答"多变量模型能不能在 `pnl_pct` 上做出**样本外**可复现的判别"。
它**不是**策略、**不产生**任何可部署参数，也不会碰 `chan-quant/`。

口径（与 `DESIGN.md` §5 / `DATASET.md` §1.1 一致）：
  * 样本 = 信号层候选池（`all` 变体），不受资金/名额/熔断约束；
  * 标签 = `pnl_pct`（回归主标签）+ `y_win`（分类辅标签，看 AUC）。**不用 `R`**
    ——`R = pnl_pct / risk_pct` 有分母假象（`DATASET.md` §4.1）；
  * 特征只保留"入场时刻已确定"的列，泄漏列一律剔除（`DATASET.md` §2.1）。

五道纪律（本脚本的实现要点，缺一条结论就不算数）：
  1. **按标的 GroupKFold** —— 同一标的的成交高度相关（§4.6）。折内不共享标的。
  2. **时间 valid 段做最终留出** —— 数据集自带 `split`（按时间 70/30）；
     CV 只在 train 段内做，valid 段一次都不参与模型选择。
  3. **特征筛选守门人** —— `gated` 特征集只放 §5.1 那张 Δpnl 表里入榜的列
     （二分点用 **train 段中位数**，比 `dataset_report` 用全样本中位更严格）。
  4. **标签打乱对照** —— 把 `pnl_pct` 在 train 段内整体打乱重跑同一套 CV，
     得到"纯噪声下的 OOF 指标分布"。真实指标落在该分布里 = 无信号。
  5. **经济指标优先于统计指标** —— R²/IC 再好看，也要过两条线：
     ① 选出子集的毛期望是否过门槛（合约 0.12% / 现货 0.20%）；
     ② 按标的聚类的 t 值 / 自助 CI 是否显著（朴素 t 会高估好几倍）。
     **"事后从 7 条规则里挑最好那条"本身就是本项目翻过车的 `combo` 模式**，
     所以判据一律挂在**整段**与**十分位价差**上，次要规则表只作诊断。

用法:
  $PY tools/model_baseline.py out/dataset/um_all.jsonl --market um --out out/model_baseline_um.txt
  $PY tools/model_baseline.py out/dataset/spot_all.jsonl --market spot --out out/model_baseline_spot.txt
  $PY tools/model_baseline.py --aggregate out/model_baseline_um.json out/model_baseline_spot.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

FEE = {"um": 0.0005, "spot": 0.001}       # 单边费率
THR = {"um": 0.0012, "spot": 0.0020}      # 信号层毛期望门槛
N_SPLITS = 5
N_SHUFFLE = 3
RNG = 20260928

# ---------------------------------------------------------------- 列白/黑名单
# 泄漏（离场之后才知道）—— DATASET.md §2.1
LEAK = ("exit_ts", "pnl_pct", "reason", "hold_min", "R", "y_stop", "y_win",
        "censored")
# 元数据
META = ("sym", "market", "source", "entry_ts", "entry_time", "split", "side")

# 字段字典顺序（= dataset_report.NUMF 的顺序），用于**确定性地**从共线簇里取代表
NUM_ORDER = [
    "risk_pct", "notional", "atr_pct", "gap_entry_struct",
    "beichi_score", "ratio_area", "ratio_per", "ratio_peak", "ratio_hist",
    "bars_a", "bars_c", "b1_lag", "amp_a_pct", "amp_c_pct", "amp_ratio",
    "zs_break", "trend_beichi", "b1_lag_bi", "retest_up_pct", "b1_had_zs",
    "zs_height_pct", "above_zg_pct", "exit_break_pct", "exit_leg_amp_pct",
    "pull_bi", "b3_zs_age", "d_ma20_dev", "d_ma60_dev", "d_atr_pct",
    "d_atr_rank", "d_ret20", "d_ma20_slope",
]
# 零方差常量（DATASET.md §4.2）
DEAD = ("beichi_new_extreme", "b1_lag_bi", "b3_zs_age", "pull_bi")
# 污染列：d_close 是绝对价格（跨标的不可比）；notional ≡ kind（§4.5）
POLLUTE = ("d_close", "notional")
# kind 的同义列：counter / confirm_ok（含 -1 档）/ confirm（文本）都只是 kind 的函数
KIND_DUP = ("counter", "confirm_ok", "confirm")
# 共线簇（|corr|>0.80）—— 按 NUM_ORDER 取**第一个**（确定性规则，不按效果挑）
CLUSTERS = [
    ("risk_pct", "gap_entry_struct"),
    ("above_zg_pct", "exit_break_pct", "exit_leg_amp_pct"),
    ("d_ma20_dev", "d_ma60_dev", "d_ret20", "d_ma20_slope"),
]
# "是否适用"指示列（§4.4：缺失多为"不适用"，必须显式编码，不能一律 dropna）
INDIC = [
    ("has_b1feat", "beichi_score"),
    ("has_b1zs", "retest_up_pct"),
    ("has_b3feat", "zs_height_pct"),
    ("has_daily", "d_ma20_dev"),
]
CATS = ["kind", "trend", "regime", "b1_path"]


def group_keep() -> dict:
    """共线簇 → {成员: 代表}。"""
    m = {}
    for cl in CLUSTERS:
        rep = min(cl, key=lambda k: NUM_ORDER.index(k))
        for k in cl:
            m[k] = rep
    return m


def numeric_cols(mode: str) -> list:
    """mode: clean（剔冗余）| noredund（只剔泄漏/污染/常量）。"""
    gk = group_keep()
    out = []
    for k in NUM_ORDER:
        if k in DEAD or k in POLLUTE:
            continue
        if mode == "clean" and k in gk and gk[k] != k:
            continue
        out.append(k)
    return out


# ---------------------------------------------------------------- 统计工具
def clustered(x: np.ndarray, g: np.ndarray, boot: int = 2000) -> dict | None:
    """按标的聚类的稳健统计（口径同 tools/ab_stats.py，改 numpy 向量化自助）。"""
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return None
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    t_naive = mean / (sd / math.sqrt(n)) if sd > 0 else 0.0
    _, inv = np.unique(g, return_inverse=True)
    s = np.bincount(inv, weights=x)
    G = s.size
    se = (math.sqrt(float((s ** 2).sum())) / n * math.sqrt(G / (G - 1))
          if G > 1 else float("inf"))
    t_cr = mean / se if 0 < se < float("inf") else 0.0
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, G, size=(boot, G))
    bs = s[idx].sum(axis=1) / n
    lo, hi = (float(v) for v in np.percentile(bs, [2.5, 97.5]))
    return {"n": n, "mean": mean, "sd": sd, "t_naive": t_naive, "t_cr": t_cr,
            "G": G, "ci": (lo, hi), "pos_sym": int((s > 0).sum())}


def clustered_diff(a: np.ndarray, ga: np.ndarray, b: np.ndarray,
                   gb: np.ndarray, boot: int = 2000) -> dict | None:
    """两组的**聚类差**统计：mean(a) − mean(b)，**两组各自**按标的自助重抽。

    这是十分位价差的正确口径 —— 把两组拼起来再重抽会得到与实际价差
    对不上的点估计（两档笔数不等时更明显）。
    """
    if a.size < 2 or b.size < 2:
        return None
    d = float(a.mean() - b.mean())
    sa, sb = np.bincount(np.unique(ga, return_inverse=True)[1], weights=a), \
        np.bincount(np.unique(gb, return_inverse=True)[1], weights=b)
    Ga, Gb = sa.size, sb.size
    rng = np.random.default_rng(RNG)
    ba = sa[rng.integers(0, Ga, size=(boot, Ga))].sum(axis=1) / a.size
    bb = sb[rng.integers(0, Gb, size=(boot, Gb))].sum(axis=1) / b.size
    diff = ba - bb
    lo, hi = (float(v) for v in np.percentile(diff, [2.5, 97.5]))
    se = float(diff.std(ddof=1))
    # 该价差本身也是"复合统计量"：两组均值相等 ⇒ 差为 0。CI 是否覆盖 0 定论。
    return {"diff": d, "se": se, "t": (d / se) if se > 0 else 0.0,
            "ci": (lo, hi), "Ga": Ga, "Gb": Gb,
            "sig0": lo > 0 or hi < 0}


def wpnl(x: np.ndarray, w: np.ndarray) -> float:
    """笔数加权（名义额加权）期望 —— A/B 的官方口径。"""
    return float((x * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


def ic(pred: np.ndarray, y: np.ndarray) -> float:
    """Spearman 秩相关（预测 vs 真实 pnl）。"""
    if pred.size < 10:
        return float("nan")
    rp = np.argsort(np.argsort(pred)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rp -= rp.mean()
    ry -= ry.mean()
    d = math.sqrt(float((rp ** 2).sum()) * float((ry ** 2).sum()))
    return float((rp * ry).sum() / d) if d > 0 else 0.0


def r2(pred: np.ndarray, y: np.ndarray) -> float:
    sst = float(((y - y.mean()) ** 2).sum())
    return float(1 - ((y - pred) ** 2).sum() / sst) if sst > 0 else float("nan")


def auc(score: np.ndarray, y_win: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_win, score))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------- 矩阵构建
def load_rows(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def dt_y(ms: int) -> int:
    """UTC 年份（必须 UTC，见 DATASET.md §1.3 的 16 小时坑）。"""
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).year


def dt_w(ms: int) -> int:
    """UTC 周桶（用于市场中性化：同一周的横截面视为同一市场环境）。"""
    return int(int(ms) / 1000 // 604800)


def build(rows: list, ncols: list) -> tuple:
    n = len(rows)
    X = np.full((n, len(ncols)), np.nan, float)
    for j, k in enumerate(ncols):
        col = np.array([np.nan if r.get(k) is None else float(r[k]) for r in rows])
        X[:, j] = col
    I = np.zeros((n, len(INDIC)), float)
    for j, (_, src) in enumerate(INDIC):
        I[:, j] = [0.0 if r.get(src) is None else 1.0 for r in rows]
    names = list(ncols) + [a for a, _ in INDIC]
    for c in CATS:
        for v in sorted({str(r.get(c)) for r in rows}):
            I = np.column_stack(
                [I, np.array([1.0 if str(r.get(c)) == v else 0.0 for r in rows])])
            names.append(f"{c}={v}")
    return np.column_stack([X, I]), names


def impute(train: np.ndarray, others: list) -> tuple:
    """线性模型用：训练段中位数填充 + 标准化（只用 train 的统计量）。"""
    med = np.nanmedian(train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where((sd == 0) | np.isnan(sd), 1.0, sd)
    f = lambda A: np.where(np.isnan(A), med, A)  # noqa: E731
    return (f(train) - mu) / sd, [((f(o) - mu) / sd) for o in others]


# ---------------------------------------------------------------- 守门人
def dnpnl_gate(rows: list, ncols: list, thr: float = 0.002) -> tuple:
    """§5.1 的 Δpnl 检验当**特征筛选守门人**。

    低档 pnl 中位 − 高档 pnl 中位，要求 train/valid 两段同号且都 > thr。
    二分点用 **train 段中位数**（比 dataset_report 用全样本中位更严格，无 valid 泄漏）。
    """
    tr = [r for r in rows if r.get("split") == "train"]
    va = [r for r in rows if r.get("split") == "valid"]
    passed, table = [], []
    for k in ncols:
        vtr = np.array([np.nan if r.get(k) is None else float(r[k]) for r in tr])
        if np.isfinite(vtr).sum() < 0.2 * len(tr):
            continue
        med = float(np.nanmedian(vtr))
        ds = {}
        for tag, sub in (("tr", tr), ("va", va)):
            p = np.array([float(r["pnl_pct"]) for r in sub])
            x = np.array([np.nan if r.get(k) is None else float(r[k]) for r in sub])
            fin = np.isfinite(x)
            plo, phi = p[fin][x[fin] <= med], p[fin][x[fin] > med]
            ds[tag] = (float(np.median(plo) - np.median(phi))
                       if plo.size >= 50 and phi.size >= 50 else None)
        if ds["tr"] is None or ds["va"] is None:
            continue
        ok = ((ds["tr"] > 0) == (ds["va"] > 0)
              and min(abs(ds["tr"]), abs(ds["va"])) > thr)
        table.append((k, ds["tr"], ds["va"], ok))
        if ok:
            passed.append(k)
    return passed, table


# ---------------------------------------------------------------- 单市场主流程
def run(path: str, market: str, out) -> dict:
    fee, thr = FEE[market], THR[market]
    t0 = time.time()
    rows0 = load_rows(path)
    raw_n = len(rows0)
    drop_cens = [r for r in rows0 if r.get("censored")]
    rows = [r for r in rows0 if not r.get("censored")]

    def w(s=""):
        print(s)
        out.write(s + "\n")

    w("=" * 96)
    w(f"# {market}  多变量基线   {os.path.basename(path)}")
    w("=" * 96)
    w(f"样本 {raw_n} → 剔除期末强平 {len(drop_cens)} → 入模 {len(rows)}")
    tr = [r for r in rows if r.get("split") == "train"]
    va = [r for r in rows if r.get("split") == "valid"]
    w(f"train {len(tr)} / valid {len(va)}   标的 train "
      f"{len({r['sym'] for r in tr})} / valid {len({r['sym'] for r in va})}")
    ytr = np.array([float(r["pnl_pct"]) for r in tr])
    yva = np.array([float(r["pnl_pct"]) for r in va])
    wtr = np.array([float(r["notional"]) for r in tr])
    wva = np.array([float(r["notional"]) for r in va])
    wtr_win = np.array([float(r["y_win"]) for r in tr])
    wva_win = np.array([float(r["y_win"]) for r in va])
    sym_tr = np.array([r["sym"] for r in tr])
    sym_va = np.array([r["sym"] for r in va])
    st = clustered(ytr, sym_tr)
    w(f"train 基线: 毛期望 {st['mean']:+.4%}  加权 {wpnl(ytr, wtr):+.4%}  "
      f"聚类t {st['t_cr']:+.2f}  CI [{st['ci'][0]:+.4%},{st['ci'][1]:+.4%}]")
    w(f"valid 基线: 毛期望 {yva.mean():+.4%}  加权 {wpnl(yva, wva):+.4%}   "
      f"胜率 train {wtr_win.mean():.1%} / valid {wva_win.mean():.1%}")
    w(f"门槛: 毛期望 > {thr:.2%}   净 = 毛 − 2×{fee:.2%} = 毛 − {2 * fee:.2%}")

    # ---------------- 特征集定义 ----------------
    w("")
    w("-- 特征集定义 --")
    w(f"排除(泄漏): {', '.join(LEAK)}")
    w(f"排除(常量): {', '.join(DEAD)}")
    w(f"排除(污染): {', '.join(POLLUTE)}   ※ notional 仅当**权重**用，不入模")
    w(f"排除(kind 同义): {', '.join(KIND_DUP)}   ※ 含区间套文字结论")
    gk = group_keep()
    for cl in CLUSTERS:
        w(f"共线簇 {cl} → 保留 `{gk[cl[0]]}`（按字段字典顺序取第一个，确定性规则）")

    nclean = numeric_cols("clean")
    nnr = numeric_cols("noredund")
    gated, gtab = dnpnl_gate(rows, nnr)
    w("")
    w(f"数值特征: clean {len(nclean)} 列 / noredund {len(nnr)} 列")
    w(f"Δpnl 守门人(train 中位二分, 两段同号且 |Δ|>0.20pp): "
      f"{len(gated)} 个 → {gated if gated else '(空 —— 一个都没进榜)'}")
    if gtab:
        w(f"  {'特征':<18}{'Δpnl train':>12}{'Δpnl valid':>12}  入榜")
        for k, a, b, ok in sorted(gtab, key=lambda x: -min(abs(x[1]), abs(x[2]))):
            w(f"  {k:<18}{a:>+11.2%}{b:>+12.2%}  {'是' if ok else '否'}")

    SETS = [("clean", nclean),
            ("noredund", nnr),
            ("gated", gated if gated else []),
            ("catonly", [])]     # 零数值特征：只剩 kind/trend/regime/b1_path

    summary = {"market": market, "n": len(rows), "train": len(tr),
               "valid": len(va), "gate": gated, "thr": thr, "fee": fee,
               "base_valid": float(yva.mean()), "base_train": float(ytr.mean()),
               "sets": {}}
    keep_gbdt = {}

    for sname, ncols in SETS:
        w("")
        w("-" * 96)
        w(f"## 特征集 `{sname}`  （数值 {len(ncols)} + 指示 {len(INDIC)} + one-hot）")
        w("-" * 96)
        Xall, names = build(rows, ncols)
        itr = np.array([i for i, r in enumerate(rows) if r.get("split") == "train"])
        iva = np.array([i for i, r in enumerate(rows) if r.get("split") == "valid"])
        Xtr, Xva = Xall[itr], Xall[iva]

        nullc = float(ytr.mean())
        km = {}
        for r in tr:
            km.setdefault(r["kind"], []).append(float(r["pnl_pct"]))
        kmean = {k: float(np.mean(v)) for k, v in km.items()}
        pv_null = np.array([kmean.get(r["kind"], nullc) for r in va])
        w(f"null 基线: 常数 R²={r2(np.full_like(yva, nullc), yva):+.5f}  |  "
          f"kind 均值 R²={r2(pv_null, yva):+.5f}  (kind 均值 "
          f"{ {k: round(v, 5) for k, v in kmean.items()} })")

        # ---- GroupKFold by sym（只在 train 段内）----
        gkf = GroupKFold(n_splits=N_SPLITS)
        oof_gb = np.full(ytr.size, np.nan)
        oof_rd = np.full(ytr.size, np.nan)
        for f, (a, b) in enumerate(gkf.split(Xtr, ytr, sym_tr)):
            g = HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.06, max_depth=3,
                min_samples_leaf=200, l2_regularization=1.0,
                early_stopping=False, random_state=RNG)
            g.fit(Xtr[a], ytr[a])
            oof_gb[b] = g.predict(Xtr[b])
            _tr, [_te] = impute(Xtr[a], [Xtr[b]])
            rd = Ridge(alpha=1.0).fit(_tr, ytr[a])
            oof_rd[b] = rd.predict(_te)
            print(f"    [{sname}] fold {f + 1}/{N_SPLITS}  "
                  f"held-out 标的 {len(set(sym_tr[b]))}  笔 {b.size}"
                  f"  ({time.time() - t0:.0f}s)")

        # ---- 时间留出：train 全量重训 → valid ----
        gfull = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.06, max_depth=3, min_samples_leaf=200,
            l2_regularization=1.0, early_stopping=False, random_state=RNG)
        gfull.fit(Xtr, ytr)
        pv_gb = gfull.predict(Xva)
        _tr, [_te] = impute(Xtr, [Xva])
        pv_rd = Ridge(alpha=1.0).fit(_tr, ytr).predict(_te)

        w("")
        w(f"  {'':<14}{'R²':>10}{'IC':>10}{'AUC(y_win)':>12}")
        w(f"  {'OOF GBDT':<14}{r2(oof_gb, ytr):>+10.5f}{ic(oof_gb, ytr):>+10.5f}"
          f"{auc(oof_gb, wtr_win):>12.4f}")
        w(f"  {'OOF Ridge':<14}{r2(oof_rd, ytr):>+10.5f}{ic(oof_rd, ytr):>+10.5f}"
          f"{auc(oof_rd, wtr_win):>12.4f}")
        w(f"  {'valid GBDT':<14}{r2(pv_gb, yva):>+10.5f}{ic(pv_gb, yva):>+10.5f}"
          f"{auc(pv_gb, wva_win):>12.4f}")
        w(f"  {'valid Ridge':<14}{r2(pv_rd, yva):>+10.5f}{ic(pv_rd, yva):>+10.5f}"
          f"{auc(pv_rd, wva_win):>12.4f}")
        w(f"  {'valid 常数':<14}{r2(np.full_like(yva, nullc), yva):>+10.5f}"
          f"{'--':>10}{0.5:>12.4f}   ← 及格线")

        # ---- 分档（OOF，按预测十分位）----
        order = np.argsort(oof_gb)
        dec = np.array_split(order, 10)
        w("")
        w("  OOF 十分位(train 段 §按预测值低→高; 理想=单调且首尾拉开):")
        w(f"    {'档':<6}{'笔数':>8}{'预测均值':>12}{'实际毛期望':>12}"
          f"{'实际中位':>12}{'止损率':>9}{'标的数':>8}"
          f"{'均值atr%':>10}{'均值d_atr%':>11}")
        atr_tr = np.array([np.nan if r.get("atr_pct") is None
                           else float(r["atr_pct"]) for r in tr])
        datr_tr = np.array([np.nan if r.get("d_atr_pct") is None
                            else float(r["d_atr_pct"]) for r in tr])
        for d, idx in enumerate(dec):
            w(f"    D{d + 1:<5}{idx.size:>8}{oof_gb[idx].mean():>+12.4%}"
              f"{ytr[idx].mean():>+12.4%}{np.median(ytr[idx]):>+12.4%}"
              f"{np.mean([tr[i].get('y_stop', 0) for i in idx]):>9.1%}"
              f"{len(set(sym_tr[idx])):>8}"
              f"{np.nanmean(atr_tr[idx]):>10.4%}{np.nanmean(datr_tr[idx]):>11.4%}")
        lo_d, hi_d = dec[0], dec[-1]
        sd = clustered_diff(ytr[hi_d], sym_tr[hi_d], ytr[lo_d], sym_tr[lo_d])
        w(f"    D10−D1 = {sd['diff']:+.4%}   聚类 se {sd['se']:.4%}  "
          f"t {sd['t']:+.2f}  CI [{sd['ci'][0]:+.4%},{sd['ci'][1]:+.4%}]  "
          f"{'CI 不含 0 ✅' if sd['sig0'] else 'CI 含 0 ❌'}")
        w("")
        w("  OOF 十分位价差**按折分解**（检验信号是否只靠某一折）:")
        w(f"    {'折':<5}{'held-out 标的':>14}{'D10−D1':>11}{'聚类t':>9}")
        fold_spreads = []
        for f, (a, b) in enumerate(gkf.split(Xtr, ytr, sym_tr)):
            ob, yb, gb = oof_gb[b], ytr[b], sym_tr[b]
            dd = np.array_split(np.argsort(ob), 10)
            if dd[0].size >= 5:
                sf = clustered_diff(yb[dd[-1]], gb[dd[-1]], yb[dd[0]],
                                    gb[dd[0]], boot=400)
                fold_spreads.append(round(sf["diff"], 6))
                w(f"    {f + 1:<5}{len(set(gb)):>14}{sf['diff']:>+11.4%}"
                  f"{sf['t']:>+9.2f}")

        # ---- 按年分解（判"样本内信号"是不是某个牛熊段的产物）----
        yr_tr = np.array([dt_y(r["entry_ts"]) for r in tr])
        w("")
        w("  OOF 十分位价差**按自然年分解**（判它是不是某一段行情的产物）:")
        w(f"    {'年':<7}{'笔数':>9}{'该年整段毛期望':>16}{'D10−D1':>11}{'聚类t':>9}")
        year_spreads = {}
        for y in sorted(set(yr_tr.tolist())):
            m = np.where(yr_tr == y)[0]
            if m.size < 200:
                continue
            dd = np.array_split(m[np.argsort(oof_gb[m])], 10)
            sf = clustered_diff(ytr[dd[-1]], sym_tr[dd[-1]], ytr[dd[0]],
                                sym_tr[dd[0]], boot=400)
            year_spreads[int(y)] = round(sf["diff"], 6)
            w(f"    {y:<7}{m.size:>9}{ytr[m].mean():>+16.4%}"
              f"{sf['diff']:>+11.4%}{sf['t']:>+9.2f}")
        npos = sum(1 for v in year_spreads.values() if v > 0)
        w(f"    → {npos}/{len(year_spreads)} 年为正；"
          f"{'**逐年同号才叫信号**，否则就是某段行情' if npos not in (0, len(year_spreads)) else '逐年同号'}")

        # ---- valid 十分位（看符号是否翻转）----
        decv = np.array_split(np.argsort(pv_gb), 10)
        w("")
        w("  valid 十分位（模型未见过的时间段；与上面 OOF 表**对照看是否翻转**）:")
        w(f"    {'档':<6}{'笔数':>8}{'预测均值':>12}{'实际毛期望':>12}"
          f"{'实际中位':>12}{'止损率':>9}{'标的数':>8}")
        for d, idx in enumerate(decv):
            w(f"    D{d + 1:<5}{idx.size:>8}{pv_gb[idx].mean():>+12.4%}"
              f"{yva[idx].mean():>+12.4%}{np.median(yva[idx]):>+12.4%}"
              f"{np.mean([va[i].get('y_stop', 0) for i in idx]):>9.1%}"
              f"{len(set(sym_va[idx])):>8}")
        sdv = clustered_diff(yva[decv[-1]], sym_va[decv[-1]],
                             yva[decv[0]], sym_va[decv[0]])
        w(f"    D10−D1 = {sdv['diff']:+.4%}   聚类 se {sdv['se']:.4%}  "
          f"t {sdv['t']:+.2f}  CI [{sdv['ci'][0]:+.4%},{sdv['ci'][1]:+.4%}]  "
          f"{'CI 不含 0 ✅' if sdv['sig0'] else 'CI 含 0 ❌'}")
        w(f"    OOF 价差 {sd['diff']:+.4%} → valid 价差 {sdv['diff']:+.4%}  "
          f"⇒ {'**符号翻转**（样本内学到的东西出了时间就反了）' if sd['diff'] * sdv['diff'] < 0 else '同号'}")

        # ---- 波动中性化：价差是不是"波动率暴露"的伪装 ----
        atr_va = np.array([np.nan if r.get("atr_pct") is None
                           else float(r["atr_pct"]) for r in va])
        datr_va = np.array([np.nan if r.get("d_atr_pct") is None
                            else float(r["d_atr_pct"]) for r in va])
        # 分箱边界只用 train 段的分位（不引入 valid 信息）
        edges = {}
        for nm, a_tr in (("atr_pct", atr_tr), ("d_atr_pct", datr_tr)):
            f = a_tr[np.isfinite(a_tr)]
            edges[nm] = np.quantile(f, np.linspace(0, 1, 21))[1:-1] if f.size else np.array([])

        def resid(y, which):
            res = y.astype(float).copy()
            for nm, a in which:
                if edges[nm].size == 0:
                    continue
                b = np.digitize(a, edges[nm])
                for bi in np.unique(b):
                    m = b == bi
                    if m.sum() >= 30:
                        res[m] = res[m] - res[m].mean()
            return res

        vn = {}
        for tag, nm, a_tr, a_va in (("+atr_pct", "atr_pct", atr_tr, atr_va),
                                    ("+d_atr_pct", "d_atr_pct", datr_tr, datr_va),
                                    ("+both", "both", None, None)):
            which_tr = [("atr_pct", atr_tr), ("d_atr_pct", datr_tr)] \
                if tag == "+both" else [(nm, a_tr)]
            which_va = [("atr_pct", atr_va), ("d_atr_pct", datr_va)] \
                if tag == "+both" else [(nm, a_va)]
            ro = resid(ytr, which_tr)
            di = np.array_split(np.argsort(oof_gb), 10)
            so = clustered_diff(ro[di[-1]], sym_tr[di[-1]], ro[di[0]],
                                sym_tr[di[0]], boot=800)
            rv = resid(yva, which_va)
            dv2 = np.array_split(np.argsort(pv_gb), 10)
            sv = clustered_diff(rv[dv2[-1]], sym_va[dv2[-1]], rv[dv2[0]],
                                sym_va[dv2[0]], boot=800)
            vn[tag] = {"oof": so["diff"], "oof_t": so["t"],
                       "val": sv["diff"], "val_t": sv["t"],
                       "val_sig": bool(sv["sig0"])}
        w("")
        w("  **波动中性化后的价差**（把 pnl 减去同波动分箱均值；"
          "分箱边界只用 train 分位）:")
        w(f"    {'中性化对象':<16}{'OOF D10−D1':>13}{'t':>8}"
          f"{'valid D10−D1':>14}{'t':>8}  CI 不含 0")
        w(f"    {'(未中性化)':<16}{sd['diff']:>+13.4%}{sd['t']:>+8.2f}"
          f"{sdv['diff']:>+14.4%}{sdv['t']:>+8.2f}  "
          f"{'是' if sdv['sig0'] else '否'}")
        for tag, v in vn.items():
            w(f"    {tag:<16}{v['oof']:>+13.4%}{v['oof_t']:>+8.2f}"
              f"{v['val']:>+14.4%}{v['val_t']:>+8.2f}  "
              f"{'是' if v['val_sig'] else '否'}")
        col = vn["+both"]
        w(f"    → 波动中性化后 valid 价差 {col['val']:+.4%}（原 {sdv['diff']:+.4%}）；"
          f"{'**归零 ⇒ 模型学到的就是波动率暴露，不是入场判别**' if abs(col['val']) < abs(sdv['diff']) * 0.5 else '未被解释'}")

        # ---- 市场中性化 + 打分时间相关（判"横截面 alpha"还是"时间 beta"）----
        wk_tr = np.array([dt_w(r["entry_ts"]) for r in tr])
        wk_va = np.array([dt_w(r["entry_ts"]) for r in va])

        def mkt_neutral(y, wk, minn=20):
            res = y.astype(float).copy()
            for x in np.unique(wk):
                m = wk == x
                if m.sum() >= minn:
                    res[m] -= res[m].mean()
            return res

        ro_m = mkt_neutral(ytr, wk_tr)
        rv_m = mkt_neutral(yva, wk_va)
        so_m = clustered_diff(ro_m[hi_d], sym_tr[hi_d], ro_m[lo_d], sym_tr[lo_d],
                              boot=800)
        dv3 = np.array_split(np.argsort(pv_gb), 10)
        sv_m = clustered_diff(rv_m[dv3[-1]], sym_va[dv3[-1]], rv_m[dv3[0]],
                              sym_va[dv3[0]], boot=800)
        ic_t_tr = ic(oof_gb, np.array([float(r["entry_ts"]) for r in tr]))
        ic_t_va = ic(pv_gb, np.array([float(r["entry_ts"]) for r in va]))
        w("")
        w("  **市场中性化**（减去同 UTC 周的横截面均值）与**打分-时间相关**:")
        w(f"    原始     OOF {sd['diff']:+.4%} (t={sd['t']:+.2f})   "
          f"valid {sdv['diff']:+.4%} (t={sdv['t']:+.2f})")
        w(f"    周中性化 OOF {so_m['diff']:+.4%} (t={so_m['t']:+.2f})   "
          f"valid {sv_m['diff']:+.4%} (t={sv_m['t']:+.2f})")
        w(f"    打分 vs 时间 IC: OOF {ic_t_tr:+.4f} / valid {ic_t_va:+.4f}"
          f"   （|IC| 大 = 模型主要在排序'哪一段时间'，不是'哪一笔交易'）")
        w(f"    → {'价差主要是**时间(beta)成分**' if abs(ic_t_tr) > 0.10 else '价差不是单纯的时间成分，含横截面成分'}；"
          f"周中性化后 OOF 价差 {so_m['diff']:+.4%} "
          f"(t={so_m['t']:+.2f}, {'仍显著' if so_m['sig0'] else '不显著'})，"
          f"valid {sv_m['diff']:+.4%} (t={sv_m['t']:+.2f})")

        # ---- 选择规则（valid 段，仅作诊断）----
        w("")
        w(f"  valid 段选择规则(**事后挑选 = combo 模式, 仅供诊断**; "
          f"毛门槛 {thr:.2%}):")
        w(f"    {'规则':<16}{'笔数':>9}{'占比':>7}{'毛期望':>10}{'加权毛':>10}"
          f"{'净期望':>10}{'聚类t':>8}{'自助95%CI':>22}{'正净标的':>10}")
        rules = {}
        rlist = [(f"top{q:.0%}", np.argsort(pv_gb)[-max(1, int(round(q * yva.size))):])
                 for q in (0.05, 0.10, 0.20, 0.30, 0.50)]
        rlist.append(("pred>0", np.where(pv_gb > 0)[0]))
        rlist.append(("pred>基线均值", np.where(pv_gb > nullc)[0]))
        for rname, sel in rlist:
            if sel.size < 2:
                w(f"    {rname:<16}{sel.size:>9}  —— 样本不足")
                continue
            s = clustered(yva[sel], sym_va[sel])
            rules[rname] = {"n": s["n"], "gross": s["mean"],
                            "net": s["mean"] - 2 * fee, "t_cr": s["t_cr"],
                            "ci": list(s["ci"]), "pass": bool(s["mean"] > thr)}
            possym = f"{s['pos_sym']}/{s['G']}"
            w(f"    {rname:<16}{s['n']:>9}{s['n'] / yva.size:>7.1%}"
              f"{s['mean']:>+10.4%}{wpnl(yva[sel], wva[sel]):>+10.4%}"
              f"{s['mean'] - 2 * fee:>+10.4%}{s['t_cr']:>+8.2f}"
              f"   [{s['ci'][0]:+.4%},{s['ci'][1]:+.4%}]"
              f"{possym:>10}")
        summary["sets"][sname] = {
            "n_num": len(ncols),
            "oof": {"r2": r2(oof_gb, ytr), "ic": ic(oof_gb, ytr),
                    "auc": auc(oof_gb, wtr_win)},
            "oof_ridge": {"r2": r2(oof_rd, ytr), "ic": ic(oof_rd, ytr)},
            "val": {"r2": r2(pv_gb, yva), "ic": ic(pv_gb, yva),
                    "auc": auc(pv_gb, wva_win)},
            "val_ridge": {"r2": r2(pv_rd, yva), "ic": ic(pv_rd, yva)},
            "decile": {"diff": sd["diff"], "t": sd["t"], "ci": list(sd["ci"]),
                       "sig0": bool(sd["sig0"]), "folds": fold_spreads,
                       "years": year_spreads},
            "decile_valid": {"diff": sdv["diff"], "t": sdv["t"],
                             "ci": list(sdv["ci"]), "sig0": bool(sdv["sig0"])},
            "vol_neutral": vn,
            "mkt_neutral": {"oof": so_m["diff"], "oof_t": so_m["t"],
                            "val": sv_m["diff"], "val_t": sv_m["t"],
                            "ic_time_oof": ic_t_tr, "ic_time_val": ic_t_va},
            "rules": rules,
        }
        if sname == "clean":
            keep_gbdt = {"model": gfull, "X": Xall[iva], "names": names,
                         "pv": pv_gb}

    # ---------------- 标签打乱对照 ----------------
    w("")
    w("-" * 96)
    w(f"## 标签打乱对照（train 段内打乱 `pnl_pct`，clean 集同一套 CV 重跑 "
      f"{N_SHUFFLE} 次）")
    w("-" * 96)
    Xc, nmc = build(rows, nclean)
    itr = np.array([i for i, r in enumerate(rows) if r.get("split") == "train"])
    iva = np.array([i for i, r in enumerate(rows) if r.get("split") == "valid"])
    Xtr_c = Xc[itr]
    rng = np.random.default_rng(RNG)
    w(f"  {'#':<4}{'R²':>11}{'Spearman IC':>14}")
    r2s, ics = [], []
    for i in range(N_SHUFFLE):
        ysh = rng.permutation(ytr)
        gkf = GroupKFold(n_splits=N_SPLITS)
        oof = np.full(ytr.size, np.nan)
        for a, b in gkf.split(Xtr_c, ysh, sym_tr):
            g = HistGradientBoostingRegressor(
                max_iter=150, learning_rate=0.06, max_depth=3,
                min_samples_leaf=200, l2_regularization=1.0,
                early_stopping=False, random_state=RNG + i)
            g.fit(Xtr_c[a], ysh[a])
            oof[b] = g.predict(Xtr_c[b])
        r2s.append(r2(oof, ysh))
        ics.append(ic(oof, ysh))
        w(f"  {i + 1:<4}{r2s[-1]:>+11.5f}{ics[-1]:>+14.5f}")
        print(f"    [shuffle {i + 1}/{N_SHUFFLE}]  ({time.time() - t0:.0f}s)")
    w(f"  噪声带: R² ∈ [{min(r2s):+.5f},{max(r2s):+.5f}]   "
      f"IC ∈ [{min(ics):+.5f},{max(ics):+.5f}]")
    r2_real = summary["sets"]["clean"]["oof"]["r2"]
    ic_real = summary["sets"]["clean"]["oof"]["ic"]
    w(f"  clean 实际 OOF: R² {r2_real:+.5f}  IC {ic_real:+.5f}  →  "
      f"{'落在噪声带内 ❌' if min(r2s) <= r2_real <= max(r2s) else '明显高于噪声带 ✅'}"
      f"（R²）/ {'落在噪声带内 ❌' if min(ics) <= ic_real <= max(ics) else '明显高于噪声带 ✅'}"
      f"（IC）")
    summary["shuffle"] = {"r2": r2s, "ic": ics}

    # ---------------- 特征重要度 ----------------
    w("")
    w("-" * 96)
    w("## 特征重要度（clean 集 GBDT, valid 段置换重要性 n_repeats=2）")
    w("-" * 96)
    pi = permutation_importance(keep_gbdt["model"], keep_gbdt["X"], yva,
                               n_repeats=2, random_state=RNG, scoring="r2")
    w(f"  (基准 valid R² = {r2(keep_gbdt['pv'], yva):+.5f}; "
      f"ΔR² <= 0 = 打乱该列不损失/反而更好)")
    w(f"  {'特征':<24}{'ΔR² 均值':>12}{'标准差':>10}")
    top = np.argsort(-pi.importances_mean)[:15]
    for j in top:
        w(f"  {keep_gbdt['names'][j]:<24}{pi.importances_mean[j]:>+12.6f}"
          f"{pi.importances_std[j]:>10.6f}")
    summary["importance"] = [
        {"name": keep_gbdt["names"][j], "d_r2": float(pi.importances_mean[j])}
        for j in top]
    print(f"    [perm-importance] ({time.time() - t0:.0f}s)")

    summary["secs"] = round(time.time() - t0, 1)
    summary["verdict"] = verdict_market(summary)
    w("")
    w(f"## 本市场判决\n{summary['verdict']}")
    w(f"\n（{market} 用时 {time.time() - t0:.0f}s）")
    return summary


# ---------------------------------------------------------------- 判决
def verdict_market(s: dict) -> str:
    """§5.4 的三个接入条件，逐条给绿/红。"""
    thr, fee = s["thr"], s["fee"]
    out = []
    c = s["sets"]["clean"]
    sh = s["shuffle"]
    in_noise_r2 = min(sh["r2"]) <= c["oof"]["r2"] <= max(sh["r2"])
    in_noise_ic = min(sh["ic"]) <= c["oof"]["ic"] <= max(sh["ic"])
    out.append(
        f"① 样本内可学性：clean OOF R²={c['oof']['r2']:+.5f} "
        f"AUC={c['oof']['auc']:.4f} IC={c['oof']['ic']:+.5f}  "
        f"→ {'与标签打乱后的噪声带无法区分 ❌' if in_noise_r2 and in_noise_ic else '高于噪声带 ✅'}")
    out.append(
        f"② 十分位价差（聚类）：OOF D10−D1={c['decile']['diff']:+.4%} "
        f"t={c['decile']['t']:+.2f} CI[{c['decile']['ci'][0]:+.4%},"
        f"{c['decile']['ci'][1]:+.4%}] {'显著 ✅' if c['decile']['sig0'] else '不显著 ❌'}"
        f"  |  valid D10−D1={c['decile_valid']['diff']:+.4%} "
        f"t={c['decile_valid']['t']:+.2f} "
        f"{'显著 ✅' if c['decile_valid']['sig0'] else '不显著 ❌'}"
        f"  → {'**符号翻转** ❌' if c['decile']['diff'] * c['decile_valid']['diff'] < 0 else '同号'}")
    foldsp = c["decile"].get("folds") or []
    yrsp = c["decile"].get("years") or {}
    npos = sum(1 for v in yrsp.values() if v > 0)
    out.append(
        f"   （OOF 价差按折分解: "
        f"{['%+.3f%%' % (x * 100) for x in foldsp]}；只需有一折为负就说明不稳）")
    out.append(
        f"   （OOF 价差按年分解: "
        f"{ {k: '%+.3f%%' % (v * 100) for k, v in yrsp.items()} }"
        f" → {npos}/{len(yrsp)} 年为正）")
    out.append(
        f"③ 时间留出(valid)：整段毛期望 {s['base_valid']:+.4%}、"
        f"R²={c['val']['r2']:+.5f} AUC={c['val']['auc']:.4f} → "
        f"{'整段本身为负 ⇒ 选择规则须显著跑赢整段' if s['base_valid'] < 0 else '整段为正'}")
    sig = [(n, v) for n, v in c["rules"].items()
           if v["gross"] > thr and abs(v["t_cr"]) > 1.96]
    out.append(
        f"④ 过门槛且显著的选择规则：{['%s(%+.3f%%,t=%+.2f)' % (n, v['gross'] * 100, v['t_cr']) for n, v in sig] if sig else '无 ❌'}")
    vn = c.get("vol_neutral", {}).get("+both")
    if vn:
        out.append(
            f"⑤ 波动中性化（减去 atr_pct/d_atr_pct 同箱均值）后："
            f"OOF D10−D1 {vn['oof']:+.4%} (t={vn['oof_t']:+.2f}) / "
            f"valid {vn['val']:+.4%} (t={vn['val_t']:+.2f}) → "
            f"{'价差被波动率解释掉 ⇒ 模型学的是波动暴露，不是入场判别' if abs(vn['val']) < abs(c['decile_valid']['diff']) * 0.6 else '仍有未被波动率解释的残差'}")
    return "\n".join(out)


def aggregate(paths: list) -> int:
    ds = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            ds.append(json.load(fh))
    print("=" * 96)
    print("# 跨市场一致性判决（本项目铁律：**只在两个市场同向才算证据**）")
    print("=" * 96)
    print(f"{'市场':<8}{'样本':>9}{'门槛':>9}{'整段毛':>10}"
          f"{'OOF R²':>10}{'OOF AUC':>9}{'OOF IC':>9}"
          f"{'OOF价差':>10}{'价差t':>8}{'valid价差':>11}{'valid t':>9}"
          f"{'波动中性valid':>14}")
    by = {d["market"]: d for d in ds}
    for m in ("um", "spot"):
        if m not in by:
            continue
        d, c = by[m], by[m]["sets"]["clean"]
        vn = c.get("vol_neutral", {}).get("+both", {})
        print(f"{m:<8}{d['n']:>9,}{d['thr']:>9.2%}{d['base_valid']:>+10.4%}"
              f"{c['oof']['r2']:>+10.5f}{c['oof']['auc']:>9.4f}"
              f"{c['oof']['ic']:>+9.5f}{c['decile']['diff']:>+10.4%}"
              f"{c['decile']['t']:>+8.2f}{c['decile_valid']['diff']:>+11.4%}"
              f"{c['decile_valid']['t']:>+9.2f}"
              f"{vn.get('val', float('nan')):>+14.4%}")
    print()
    for m in ("um", "spot"):
        if m in by:
            print(f"--- {m} ---")
            print(by[m]["verdict"])
            print()
    if "um" in by and "spot" in by:
        du, dp = by["um"], by["spot"]
        cu, cp = du["sets"]["clean"], dp["sets"]["clean"]
        print("## 总判决（逐条对照 DATASET.md §5.4 的三个接入条件）")
        # 条件①：valid 段存在「过门槛且显著为正（聚类 t>1.96）」的选择规则
        hits = {}
        for m, d, c in (("um", du, cu), ("spot", dp, cp)):
            hits[m] = [(n, v) for n, v in c["rules"].items()
                       if v["gross"] > d["thr"] and v["t_cr"] > 1.96]
            best = max((v["gross"] for v in c["rules"].values()),
                       default=float("nan"))
            txt = " / ".join("%s(%+.3f%%, t=%+.2f)" % (n, v["gross"] * 100, v["t_cr"])
                             for n, v in hits[m]) or "无"
            print("  条件① %s 有[过门槛且显著为正]的规则？ %s"
                  "   （valid 最好子集毛期望 %+.4f%% / 门槛 %.2f%%）"
                  % (m, txt, best * 100, d["thr"] * 100))
        c1 = bool(hits["um"]) and bool(hits["spot"])
        print("  => 条件① %s" % ("满足 OK" if c1 else "**不满足 FAIL**"))
        # 条件②：两个市场在 valid 段同向（且必须为正）
        same_v = (cu["decile_valid"]["diff"] > 0) == (cp["decile_valid"]["diff"] > 0)
        pos_v = cu["decile_valid"]["diff"] > 0 and cp["decile_valid"]["diff"] > 0
        print("  条件② valid 段十分位价差两市场同向？ %s"
              "  （um %+.4f%% t=%+.2f / spot %+.4f%% t=%+.2f）"
              % ("是" if same_v else "否",
                 cu["decile_valid"]["diff"] * 100, cu["decile_valid"]["t"],
                 cp["decile_valid"]["diff"] * 100, cp["decile_valid"]["t"]))
        print("  => 条件② %s" % ("两市场一致为正 OK" if pos_v else
              "**两市场一致为负** —— 按本项目铁律，这本身就是一条跨市场一致的结论："
              "入场判别结构出时间即反转 FAIL"))
        # 条件③：整段净期望
        for m, d in (("um", du), ("spot", dp)):
            print("  条件③ %s valid 整段：毛 %+.4f%% / 净 %+.4f%% => %s"
                  % (m, d["base_valid"] * 100,
                     (d["base_valid"] - 2 * d["fee"]) * 100,
                     "过门槛" if d["base_valid"] > d["thr"] else "**为负**"))
        print()
        print("  => **总判决：%s**"
              % ("不满足 §5.4 接入条件，不部署" if not (c1 and pos_v)
                 else "满足，需人工复核"))
        print("     样本内对照：OOF 价差 um %+.4f%% (t=%+.2f) / spot %+.4f%% (t=%+.2f)，"
              "两市场都显著为正 => [入场判别可学] 在样本内是真的，问题出在时间外推。"
              % (cu["decile"]["diff"] * 100, cu["decile"]["t"],
                 cp["decile"]["diff"] * 100, cp["decile"]["t"]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--market", choices=["um", "spot"])
    ap.add_argument("--out")
    ap.add_argument("--aggregate", nargs="*")
    a = ap.parse_args()
    if a.aggregate:
        return aggregate(a.aggregate)
    if not (a.path and a.market and a.out):
        ap.error("需要 path / --market / --out，或 --aggregate")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        s = run(a.path, a.market, fh)
    with open(a.out.replace(".txt", ".json"), "w", encoding="utf-8") as fh:
        json.dump(s, fh, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
