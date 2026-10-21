#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
山寨币暴涨前特征因子提取。

方法论（事件研究 + 横截面因子对比）：
1. 暴涨事件定义：未来 M 个交易日内累计涨幅 >= THRESH（如 7 天 >= 50%）。
   在日线收盘价上滚动检测，事件日为「暴涨启动日」。
2. 正样本：事件日前一天(暴涨前)的因子快照。
3. 负样本（对照）：全市场所有「未发生暴涨」的交易日因子快照（随机抽样）。
4. 对比两组因子的分布，用 AUC / 均值差 / 信息系数(IC) 衡量因子区分度。
5. 输出按区分度排序的先导因子。

因子清单（全部可量化、仅用 OHLCV）：
- 动量类：近N日收益率、相对BTC超额收益、距52周高/低点距离
- 量价类：量比(近N日量/长期均量)、量价背离、换手率代理(量/流通)
- 波动类：近N日波动率、振幅、最大回撤
- 形态类：连涨天数、RSI、价格相对均线位置
"""
import json
import os
import sys
import statistics as st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
KL_DIR = os.path.join(HERE, "..", "out", "scan", "klines_1d")
OUT = os.path.join(HERE, "..", "out", "scan")

# ---- 参数 ----
M = 7          # 未来 M 日
THRESH = 0.50  # 涨幅阈值 50%
LOOKBACK = 30  # 因子回看窗口
MIN_BARS = 120 # 最少历史根数


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
    return rows  # [ts,o,h,l,c,v,quote_vol,trades]


def pct_chg(c, i, n):
    """近 n 日收益率（含当日）。n=1 时为单日涨跌 c[i]/c[i-1]-1。"""
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


def build_factors(rows, i, btc_closes=None):
    """在 index=i（暴涨前最后一天）构建因子。"""
    o = [r[1] for r in rows]
    h = [r[2] for r in rows]
    l = [r[3] for r in rows]
    c = [r[4] for r in rows]
    v = [r[5] for r in rows]
    f = {}

    # 动量
    f['ret_3d'] = pct_chg(c, i, 3)
    f['ret_7d'] = pct_chg(c, i, 7)
    f['ret_14d'] = pct_chg(c, i, 14)
    f['ret_30d'] = pct_chg(c, i, 30)

    # 距52周高低点
    win = min(i + 1, 252)
    hi52 = max(h[i - win + 1:i + 1])
    lo52 = min(l[i - win + 1:i + 1])
    f['dist_52w_high'] = c[i] / hi52 - 1.0 if hi52 > 0 else None
    f['dist_52w_low'] = c[i] / lo52 - 1.0 if lo52 > 0 else None

    # 量能：近7日均量 / 近30日均量 (量比)
    if i >= 30:
        v7 = sma(v[i - 6:i + 1], 7)
        v30 = sma(v[i - 29:i + 1], 30)
        f['vol_ratio_7_30'] = (v7 / v30 - 1.0) if v30 else None
        # 近3日量比
        v3 = sma(v[i - 2:i + 1], 3)
        f['vol_ratio_3_30'] = (v3 / v30 - 1.0) if v30 else None
    else:
        f['vol_ratio_7_30'] = None
        f['vol_ratio_3_30'] = None

    # 波动率：近30日收益率标准差
    if i >= 30:
        rets = [pct_chg(c, j, 1) for j in range(i - 29, i + 1)]
        rets = [r for r in rets if r is not None]
        if len(rets) > 2:
            f['vol_30d'] = st.pstdev(rets)
        else:
            f['vol_30d'] = None
    else:
        f['vol_30d'] = None

    # 振幅：近7日平均 (h-l)/c
    if i >= 7:
        amps = [(h[j] - l[j]) / c[j] for j in range(i - 6, i + 1) if c[j] > 0]
        f['amp_7d'] = sum(amps) / len(amps) if amps else None
    else:
        f['amp_7d'] = None

    # 连涨/连跌天数
    up = 0
    j = i
    while j > 0 and c[j] > c[j - 1]:
        up += 1
        j -= 1
    f['consec_up'] = up

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

    # 相对 BTC 超额（近7日）
    f['excess_7d_vs_btc'] = None
    if btc_closes is not None:
        btc_ret7 = pct_chg(btc_closes, min(i, len(btc_closes) - 1), 7)
        if f['ret_7d'] is not None and btc_ret7 is not None:
            f['excess_7d_vs_btc'] = f['ret_7d'] - btc_ret7

    return f


def detect_pumps(rows):
    """返回暴涨事件启动日 index 列表（未来 M 日累计涨幅>=THRESH）。"""
    c = [r[4] for r in rows]
    events = []
    n = len(c)
    for i in range(n - M):
        base = c[i]
        if base <= 0:
            continue
        peak = max(c[i + 1:i + M + 1])
        if peak / base - 1.0 >= THRESH:
            events.append(i)
    return events


def merge_events(events, gap=3):
    """合并间隔过近的事件（同一波上涨只算一次）。"""
    if not events:
        return []
    merged = [events[0]]
    for e in events[1:]:
        if e - merged[-1] <= gap:
            continue
        merged.append(e)
    return merged


def auc(pos_vals, neg_vals):
    """AUC（曼-惠特尼 U 统计量），衡量因子区分正负样本的能力。0.5=无区分。"""
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
    # 处理并列
    ties = sum(1 for p in pos for g in neg if p == g)
    u += ties * 0.5
    return u / (ni * nj)


def main():
    syms = sorted(f.rsplit("_", 1)[0] for f in os.listdir(KL_DIR)
                  if f.endswith("_1d.json"))
    print(f"载入标的数: {len(syms)}")

    # 载入 BTC 收盘价作为市场基准
    btc_rows = load_kl("BTCUSDT")
    btc_closes = [r[4] for r in btc_rows] if btc_rows else None

    # 收集正样本(暴涨前)与全部候选因子
    pos_factors = defaultdict(list)   # 因子名 -> 正样本值列表
    neg_factors = defaultdict(list)   # 因子名 -> 负样本值列表
    pump_count = 0
    total_events = 0

    # 先扫一遍所有事件，收集正样本
    all_pumps = []  # (sym, event_idx)
    for sym in syms:
        rows = load_kl(sym)
        if rows is None:
            continue
        events = merge_events(detect_pumps(rows))
        total_events += len(events)
        for e in events:
            all_pumps.append((sym, rows, e))

    print(f"检测到暴涨事件(合并后): {len(all_pumps)}")

    # 正样本因子
    for sym, rows, e in all_pumps:
        if e < 1:
            continue
        f = build_factors(rows, e - 1, btc_closes)  # 暴涨前最后一天
        for k, val in f.items():
            if val is not None:
                pos_factors[k].append(val)
        pump_count += 1

    print(f"正样本(暴涨前快照): {pump_count}")

    # 负样本：全市场随机抽取「未暴涨」日的因子（每个标的抽几个平稳日）
    import random
    random.seed(42)
    neg_count = 0
    for sym in syms:
        rows = load_kl(sym)
        if rows is None:
            continue
        c = [r[4] for r in rows]
        events = set(merge_events(detect_pumps(rows)))
        n = len(c)
        # 抽取最多 5 个非暴涨日（且不在事件前 M 日窗口内，避免标签泄漏）
        candidates = [i for i in range(MIN_BARS, n - M)
                      if i not in events
                      and all(i - e > M for e in events if e < i)]
        if not candidates:
            continue
        picks = random.sample(candidates, min(5, len(candidates)))
        for i in picks:
            f = build_factors(rows, i, btc_closes)
            for k, val in f.items():
                if val is not None:
                    neg_factors[k].append(val)
            neg_count += 1

    print(f"负样本(未暴涨快照): {neg_count}")

    # 计算每个因子的区分度
    results = []
    for k in pos_factors:
        if k not in neg_factors:
            continue
        p = pos_factors[k]
        n = neg_factors[k]
        a = auc(p, n)
        pm = sum(p) / len(p)
        nm = sum(n) / len(n)
        # 使 AUC 方向统一为「因子越高越偏向暴涨」(取 |AUC-0.5| 并标记方向)
        direction = 1 if pm > nm else -1
        results.append({
            'factor': k,
            'auc': a,
            'auc_abs': abs(a - 0.5),
            'pos_mean': pm,
            'neg_mean': nm,
            'diff': pm - nm,
            'direction': 'high→pump' if pm > nm else 'low→pump',
            'n_pos': len(p),
            'n_neg': len(n),
        })

    results.sort(key=lambda r: -r['auc_abs'])

    # 输出
    print("\n" + "=" * 90)
    print(f"{'因子':<20} {'AUC':>7} {'区分度':>7} {'正样本均值':>12} {'负样本均值':>12} {'方向':>12}")
    print("=" * 90)
    for r in results:
        print(f"{r['factor']:<20} {r['auc']:>7.3f} {r['auc_abs']:>7.3f} "
              f"{r['pos_mean']:>12.4f} {r['neg_mean']:>12.4f} {r['direction']:>12}")

    json.dump(results, open(os.path.join(OUT, 'pump_factors.json'), 'w'),
              ensure_ascii=False, indent=2)
    print(f"\n结果已写入 out/scan/pump_factors.json")
    print(f"事件数 {pump_count}, 负样本 {neg_count}")


if __name__ == "__main__":
    main()
