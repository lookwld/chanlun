#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
山寨币「底部」特征因子提取。

底部定义：价格创出 60 日新低，且未来 M 日反弹 >= REBOUND（默认 30%）。
正样本 = 新低日（底部日）的因子快照；负样本 = 随机抽样「未触底/未反弹」日。

在 pump 因子基础上，额外加入「底部专属」因子：
- 超跌深度：距60日高点回撤幅度
- 连跌天数
- 相对52周低点距离(是否已接近历史低)
- 量能萎缩度：近7日量 / 60日量(底部常缩量)
- 波动收敛：近7日振幅 / 30日振幅(底部常窄幅)
"""
import json
import os
import random
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KL_DIR = os.path.join(HERE, "..", "out", "scan", "klines_1d")
OUT = os.path.join(HERE, "..", "out", "scan")

M = 14         # 未来反弹窗口
REBOUND = 0.30 # 反弹阈值 30%
NEWLOW_WIN = 60  # 60日新低窗口
MIN_BARS = 120


def load_kl(sym):
    path = os.path.join(KL_DIR, f"{sym}_1d.json")
    if not os.path.exists(path):
        return None
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except Exception:
        return None
    if len(rows) < MIN_BARS:
        return None
    return rows


def pct_chg(c, i, n):
    if n == 1:
        if i - 1 < 0:
            return None
        base = c[i - 1]
    else:
        if i - n + 1 < 0:
            return None
        base = c[i - n + 1]
    if base <= 0:
        return None
    return c[i] / base - 1.0


def sma(vals, n):
    if len(vals) < n:
        return None
    return sum(vals[-n:]) / n


def build_bottom_factors(rows, i, btc_closes=None):
    """在 index=i（底部日）构建因子，含底部专属因子。"""
    o = [r[1] for r in rows]
    h = [r[2] for r in rows]
    l = [r[3] for r in rows]
    c = [r[4] for r in rows]
    v = [r[5] for r in rows]
    f = {}

    # 动量（底部通常为负）
    f['ret_3d'] = pct_chg(c, i, 3)
    f['ret_7d'] = pct_chg(c, i, 7)
    f['ret_14d'] = pct_chg(c, i, 14)
    f['ret_30d'] = pct_chg(c, i, 30)

    # 超跌深度：距60日高点回撤
    win = min(i + 1, 60)
    hi60 = max(h[i - win + 1:i + 1])
    f['drawdown_60d'] = c[i] / hi60 - 1.0 if hi60 > 0 else None

    # 距52周低点（是否已接近历史低）
    win52 = min(i + 1, 252)
    lo52 = min(l[i - win52 + 1:i + 1])
    f['dist_52w_low'] = c[i] / lo52 - 1.0 if lo52 > 0 else None

    # 距52周高点（超跌程度）
    hi52 = max(h[i - win52 + 1:i + 1])
    f['dist_52w_high'] = c[i] / hi52 - 1.0 if hi52 > 0 else None

    # 连跌天数
    down = 0
    j = i
    while j > 0 and c[j] < c[j - 1]:
        down += 1
        j -= 1
    f['consec_down'] = down

    # 量能萎缩度：近7日量 / 60日量
    if i >= 60:
        v7 = sma(v[i - 6:i + 1], 7)
        v60 = sma(v[i - 59:i + 1], 60)
        f['vol_ratio_7_60'] = (v7 / v60) if v60 else None
        v3 = sma(v[i - 2:i + 1], 3)
        f['vol_ratio_3_60'] = (v3 / v60) if v60 else None
    else:
        f['vol_ratio_7_60'] = None
        f['vol_ratio_3_60'] = None

    # 波动收敛：近7日振幅 / 30日振幅
    if i >= 30:
        amp7 = sum((h[j] - l[j]) / c[j] for j in range(i - 6, i + 1) if c[j] > 0) / 7
        amp30 = sum((h[j] - l[j]) / c[j] for j in range(i - 29, i + 1) if c[j] > 0) / 30
        f['amp_7_over_30'] = (amp7 / amp30) if amp30 > 0 else None
        f['amp_7d'] = amp7
    else:
        f['amp_7_over_30'] = None
        f['amp_7d'] = None

    # 波动率 30日
    if i >= 30:
        rets = [pct_chg(c, j, 1) for j in range(i - 29, i + 1)]
        rets = [r for r in rets if r is not None]
        f['vol_30d'] = st.pstdev(rets) if len(rets) > 2 else None
    else:
        f['vol_30d'] = None

    # RSI(14)
    if i >= 14:
        gains, losses = [], []
        for j in range(i - 13, i + 1):
            chg = c[j] - c[j - 1]
            gains.append(max(chg, 0))
            losses.append(max(-chg, 0))
        ag = sum(gains) / 14
        al = sum(losses) / 14
        f['rsi_14'] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
    else:
        f['rsi_14'] = None

    # 价格相对均线
    ma20 = sma(c[i - 19:i + 1], 20) if i >= 19 else None
    f['dist_ma20'] = (c[i] / ma20 - 1.0) if ma20 else None
    ma50 = sma(c[i - 49:i + 1], 50) if i >= 49 else None
    f['dist_ma50'] = (c[i] / ma50 - 1.0) if ma50 else None

    # 相对 BTC（近7日超额）
    f['excess_7d_vs_btc'] = None
    if btc_closes is not None:
        btc_ret7 = pct_chg(btc_closes, min(i, len(btc_closes) - 1), 7)
        if f['ret_7d'] is not None and btc_ret7 is not None:
            f['excess_7d_vs_btc'] = f['ret_7d'] - btc_ret7

    return f


def detect_bottoms(rows):
    """返回底部事件 index 列表：当日创60日新低，且未来M日反弹>=REBOUND。"""
    c = [r[4] for r in rows]
    l = [r[3] for r in rows]
    n = len(c)
    events = []
    for i in range(NEWLOW_WIN, n - M):
        # 创 60 日新低（含当日，用最低价）
        win_lo = min(l[i - NEWLOW_WIN + 1:i + 1])
        if l[i] > win_lo:
            continue
        # 未来 M 日反弹
        base = c[i]
        if base <= 0:
            continue
        peak = max(c[i + 1:i + M + 1])
        if peak / base - 1.0 >= REBOUND:
            events.append(i)
    return events


def merge_events(events, gap=5):
    if not events:
        return []
    merged = [events[0]]
    for e in events[1:]:
        if e - merged[-1] <= gap:
            continue
        merged.append(e)
    return merged


def auc(pos_vals, neg_vals):
    pos = sorted(pos_vals)
    neg = sorted(neg_vals)
    ni, nj = len(pos), len(neg)
    if ni == 0 or nj == 0:
        return None
    u = 0
    j = 0
    for p in pos:
        while j < nj and neg[j] < p:
            j += 1
        u += j
    ties = sum(1 for p in pos for g in neg if p == g)
    u += ties * 0.5
    return u / (ni * nj)


def main():
    syms = sorted(f.rsplit("_", 1)[0] for f in os.listdir(KL_DIR)
                  if f.endswith("_1d.json"))
    print(f"载入标的数: {len(syms)}")

    btc_rows = load_kl("BTCUSDT")
    btc_closes = [r[4] for r in btc_rows] if btc_rows else None

    all_bottoms = []
    for sym in syms:
        rows = load_kl(sym)
        if rows is None:
            continue
        for e in merge_events(detect_bottoms(rows)):
            all_bottoms.append((sym, rows, e))
    print(f"检测到底部事件(合并后): {len(all_bottoms)}")

    # 正样本
    pos_factors = defaultdict(list)
    for sym, rows, e in all_bottoms:
        f = build_bottom_factors(rows, e, btc_closes)
        for k, val in f.items():
            if val is not None:
                pos_factors[k].append(val)

    # 负样本：随机抽样「未触底」日
    random.seed(42)
    neg_factors = defaultdict(list)
    for sym in syms:
        rows = load_kl(sym)
        if rows is None:
            continue
        c = [r[4] for r in rows]
        l = [r[3] for r in rows]
        ev = set(merge_events(detect_bottoms(rows)))
        n = len(c)
        cand = []
        for i in range(NEWLOW_WIN, n - M):
            if i in ev:
                continue
            win_lo = min(l[i - NEWLOW_WIN + 1:i + 1])
            if l[i] <= win_lo:  # 排除所有"创60日新低"日
                continue
            if all(i - e > M for e in ev if e < i):
                cand.append(i)
        if not cand:
            continue
        for i in random.sample(cand, min(3, len(cand))):
            f = build_bottom_factors(rows, i, btc_closes)
            for k, val in f.items():
                if val is not None:
                    neg_factors[k].append(val)

    print(f"正样本: {len(all_bottoms)}, 负样本: {sum(len(v) for v in neg_factors.values()) // max(1,len(neg_factors))}")

    results = []
    for k in pos_factors:
        if k not in neg_factors:
            continue
        p = pos_factors[k]
        n = neg_factors[k]
        a = auc(p, n)
        pm = sum(p) / len(p)
        nm = sum(n) / len(n)
        results.append({
            'factor': k, 'auc': a, 'auc_abs': abs(a - 0.5),
            'pos_mean': pm, 'neg_mean': nm, 'diff': pm - nm,
            'direction': 'high→bottom' if pm > nm else 'low→bottom',
            'n_pos': len(p), 'n_neg': len(n),
        })
    results.sort(key=lambda r: -r['auc_abs'])

    print("\n" + "=" * 92)
    print(f"{'因子':<18} {'AUC':>7} {'区分度':>7} {'正样本均值':>12} {'负样本均值':>12} {'方向':>12}")
    print("=" * 92)
    for r in results:
        print(f"{r['factor']:<18} {r['auc']:>7.3f} {r['auc_abs']:>7.3f} "
              f"{r['pos_mean']:>12.4f} {r['neg_mean']:>12.4f} {r['direction']:>12}")

    json.dump(results, open(os.path.join(OUT, 'bottom_factors.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n结果已写入 out/scan/bottom_factors.json")


if __name__ == "__main__":
    main()
