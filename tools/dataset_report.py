#!/usr/bin/env python
"""数据集体检 —— 建模之前必须先看的东西: 字段/缺失/标签分布/初步关系。

不做任何建模, 只回答三个问题:
  1. **字段是否齐备** —— 每个特征的缺失率(缺失可能来自"不适用"而非"没算",
     比如 `confirm_ok=-1` 表示该买点不用区间套, 不能当缺失处理)。
  2. **标签是否平衡** —— `y_stop` / `y_win` 的基线比例就是模型的及格线。
  3. **特征是否有信号** —— 单特征分组的标签率。若所有分组都在基线附近,
     说明这个特征没有区分度, 不必建模。

用法:
  $PY tools/dataset_report.py out/dataset/um_all.jsonl [out/dataset/spot_all.jsonl ...]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

META = ("sym", "market", "source", "entry_ts", "entry_time", "split")
LABELS = ("y_stop", "y_win", "R", "censored", "pnl_pct", "reason", "hold_min",
          "exit_ts")
# 入场时已知的数值特征(入模候选)
NUMF = ("risk_pct", "notional", "atr_pct", "gap_entry_struct",
        "beichi_score", "ratio_area", "ratio_per", "ratio_peak", "ratio_hist",
        "bars_a", "bars_c", "b1_lag", "amp_a_pct", "amp_c_pct", "amp_ratio",
        "zs_break", "trend_beichi", "b1_lag_bi", "retest_up_pct", "b1_had_zs",
        "zs_height_pct", "above_zg_pct", "exit_break_pct", "exit_leg_amp_pct",
        "pull_bi", "b3_zs_age", "d_ma20_dev", "d_ma60_dev", "d_atr_pct",
        "d_atr_rank", "d_ret20", "d_ma20_slope")
CATF = ("kind", "trend", "regime", "counter", "confirm_ok", "b1_path")


def load(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]


def _num(rows: list, k: str):
    v = [r.get(k) for r in rows]
    return np.array([np.nan if x is None else float(x) for x in v], dtype=float)


def _share(rows: list, k: str, val=1) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r.get(k) == val) / len(rows)


def report(path: str, with_stability: bool = True) -> None:
    rows = load(path)
    n = len(rows)
    print(f"\n{'=' * 88}\n## {os.path.basename(path)}   样本 {n}\n{'=' * 88}")
    if not n:
        return
    ts = [r["entry_ts"] for r in rows]
    print(f"时间跨度: {min(ts)} ~ {max(ts)}   "
          f"标的数 {len({r['sym'] for r in rows})}   "
          f"来源 {sorted({r.get('source') for r in rows})}")
    sp = {s: sum(1 for r in rows if r.get("split") == s) for s in ("train", "valid")}
    print(f"切分: {sp}")

    # ---------------- 标签 ----------------
    print(f"\n-- 标签 --")
    print(f"  y_stop(被止损) = {_share(rows, 'y_stop'):.1%}   "
          f"y_win(盈利) = {_share(rows, 'y_win'):.1%}   "
          f"censored(期末强平) = {_share(rows, 'censored'):.1%}")
    rr = _num(rows, "R")
    rr = rr[~np.isnan(rr)]
    if rr.size:
        print(f"  R(初始风险单位): 均值 {rr.mean():+.4f}  中位 {np.median(rr):+.4f}  "
              f"p10 {np.percentile(rr, 10):+.2f}  p90 {np.percentile(rr, 90):+.2f}")
    cnt: dict = {}
    for r in rows:
        cnt[r["reason"]] = cnt.get(r["reason"], 0) + 1
    print("  离场方式: " + "  ".join(f"{k}={v / n:.1%}" for k, v in
                                     sorted(cnt.items(), key=lambda kv: -kv[1])))
    # 按离场结果分档的止损率(剔除期末强平, 它是截断不是真实离场)
    live = [r for r in rows if not r.get("censored")]
    if live:
        print(f"  剔除期末强平后: {len(live)} 笔, 其中止损 "
              f"{_share(live, 'y_stop'):.1%}, 盈利 {_share(live, 'y_win'):.1%}")

    # ---------------- 缺失率 ----------------
    print(f"\n-- 特征缺失率(仅列 >0 的) --")
    miss = []
    for k in NUMF + CATF:
        m = sum(1 for r in rows if r.get(k) is None)
        if m:
            miss.append(f"{k}={m / n:.1%}")
    print("  " + ("  ".join(miss) if miss else "无缺失"))
    print("  (注: 各买点特征集不同 —— B2/B3 天然没有 ratio_*/b1_*; "
          "'缺失'多为'不适用', 不能一律当 NaN 丢弃)")

    # ---------------- 数值分布 ----------------
    print(f"\n-- 数值特征分布(仅列有效值 >=5% 的) --")
    print(f"  {'特征':<18}{'有效':>7}{'均值':>12}{'p10':>12}{'中位':>12}{'p90':>12}")
    for k in NUMF:
        v = _num(rows, k)
        ok = v[~np.isnan(v)]
        if ok.size < n * 0.05:
            continue
        print(f"  {k:<18}{ok.size / n:>6.0%}{ok.mean():>12.4f}"
              f"{np.percentile(ok, 10):>12.4f}{np.median(ok):>12.4f}"
              f"{np.percentile(ok, 90):>12.4f}")

    # ---------------- 单特征区分度 ----------------
    print(f"\n-- 单特征分组止损率(基线 {_share(rows, 'y_stop'):.1%}; "
          f"差距越大越有信号) --")
    for k in ("kind", "trend", "regime", "counter", "confirm_ok", "b1_path"):
        vals = sorted({r.get(k) for r in rows}, key=lambda x: str(x))
        if len(vals) > 8:
            continue
        parts = []
        for v in vals:
            sub = [r for r in rows if r.get(k) == v]
            if len(sub) < 30:
                continue
            parts.append(f"{v}: {_share(sub, 'y_stop'):.1%}(n={len(sub)})")
        if parts:
            print(f"  {k:<12} " + "  ".join(parts))
    # 数值特征按中位数二分
    for k in ("risk_pct", "atr_pct", "d_ma20_dev", "d_atr_rank", "ratio_area"):
        v = _num(rows, k)
        ok = ~np.isnan(v)
        if ok.sum() < n * 0.2:
            continue
        med = np.nanmedian(v)
        lo = [r for r, x in zip(rows, v) if x == x and x <= med]
        hi = [r for r, x in zip(rows, v) if x == x and x > med]
        print(f"  {k:<12} <=中位 {_share(lo, 'y_stop'):.1%}(n={len(lo)})   "
              f">中位 {_share(hi, 'y_stop'):.1%}(n={len(hi)})")

    if with_stability:
        stability(rows)
        collinearity(rows)


def _med_of(sub: list, k: str, med: float, low: bool, fld: str) -> float | None:
    xs = []
    for r in sub:
        x = r.get(k)
        if x is None or r.get(fld) is None:
            continue
        if (float(x) <= med) == low:
            xs.append(float(r[fld]))
    return float(np.median(xs)) if len(xs) >= 50 else None


def stability(rows: list) -> None:
    """train/valid 稳定性 —— 只在一个时段成立的特征不是信号, 是时间漂移。

    **主判据用 `pnl_pct`, 不是 `R`。** 这是本轮踩到的一个大坑:
      `R = pnl_pct / risk_pct`, 分母就是 `risk_pct` 本身。于是"risk_pct 越大 ⇒
      R 越接近 0"是**除法的必然结果**, 哪怕 pnl 完全一样。用 R 当标签、又把
      `risk_pct`/`gap_entry_struct`(与 risk_pct 共线 0.96)当特征, 会凭空造出
      一个"强特征"。所以三个量一起看, 以 Δpnl 定论:
      * `Δpnl`  = 低档 pnl_pct 中位 − 高档 pnl_pct 中位 —— **无假象, 说了算**;
      * `Δ止损` = 低档止损率 − 高档止损率 —— 直观但**不等于赚钱**;
      * `ΔR`    —— 仅供参考, 受分母污染。
    """
    tr = [r for r in rows if r.get("split") == "train"]
    va = [r for r in rows if r.get("split") == "valid"]
    if len(tr) < 200 or len(va) < 200:
        return
    p1 = [r["pnl_pct"] for r in tr if r.get("pnl_pct") is not None]
    p2 = [r["pnl_pct"] for r in va if r.get("pnl_pct") is not None]
    print("\n-- train/valid 稳定性(全样本中位二分; 两段 Δpnl 同号且 |Δpnl|>0.2pp) --")
    print(f"  基线: train 止损 {_share(tr, 'y_stop'):.1%} pnl中位 "
          f"{np.median(p1):+.3%}   |   valid 止损 {_share(va, 'y_stop'):.1%} "
          f"pnl中位 {np.median(p2):+.3%}")
    print(f"  {'特征':<18}{'Δpnl tr':>9}{'Δpnl va':>9}"
          f"{'Δ止损tr':>9}{'Δ止损va':>9}{'同号':>6}{'最弱|Δpnl|':>11}")
    hit = 0
    for k in NUMF:
        v = _num(rows, k)
        if (~np.isnan(v)).sum() < len(rows) * 0.2:
            continue
        med = float(np.nanmedian(v))
        m: dict = {}
        for tag, sub in (("tr", tr), ("va", va)):
            lo = [r for r in sub if r.get(k) is not None and float(r[k]) <= med]
            hi = [r for r in sub if r.get(k) is not None and float(r[k]) > med]
            a = _med_of(sub, k, med, True, "pnl_pct")
            b = _med_of(sub, k, med, False, "pnl_pct")
            m[tag] = (None if len(lo) < 50 or len(hi) < 50 or a is None
                      or b is None else
                      (a - b, _share(lo, "y_stop") - _share(hi, "y_stop")))
        if m["tr"] is None or m["va"] is None:
            continue
        d1, d2 = m["tr"][0], m["va"][0]
        if (d1 > 0) == (d2 > 0) and min(abs(d1), abs(d2)) > 0.002:
            print(f"  {k:<18}{d1:>+8.2%}{d2:>+9.2%}"
                  f"{m['tr'][1]:>+9.1%}{m['va'][1]:>+9.1%}{'  是':>6}"
                  f"{min(abs(d1), abs(d2)):>11.2%}")
            hit += 1
    if not hit:
        print("  (无 —— 没有任何数值特征在两个时段上稳定地区分收益率)")


def collinearity(rows: list, thr: float = 0.80) -> None:
    """共线性: |corr|>thr 的特征对 —— 冗余列会让线性模型系数不稳、树模型互相抢功。

    已知的两组必然共线(不是 bug, 是设计):
      * `notional` 只取 2 个值 —— 它完全由 `kind` 决定(一买减半), 与 `kind` 等价;
      * `gap_entry_struct` 与 `risk_pct` 都度量"入场价距止损位多远"(差一个 buffer)。
    """
    cols = [k for k in NUMF if (~np.isnan(_num(rows, k))).sum() > len(rows) * 0.2]
    mats = {k: _num(rows, k) for k in cols}
    print(f"\n-- 共线性(|corr|>{thr:.2f}) --")
    hit = 0
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            x, y = mats[a], mats[b]
            ok = ~np.isnan(x) & ~np.isnan(y)
            if ok.sum() < 200:
                continue
            u, w = x[ok], y[ok]
            if u.std() == 0 or w.std() == 0:
                continue
            c = float(np.corrcoef(u, w)[0, 1])
            if abs(c) > thr:
                print(f"  {a:<20}{b:<20}corr {c:+.3f}  (n={ok.sum()})")
                hit += 1
    if not hit:
        print("  (无)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--no-stability", action="store_true")
    a = ap.parse_args()
    for p in a.paths:
        report(p, with_stability=not a.no_stability)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
