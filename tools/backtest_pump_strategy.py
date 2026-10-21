#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
山寨币拐点策略完整回测（选币 + 持有 + 离场）—— 干净对照版。

核心问题：因子选币能否跑出相对「随机选币」和「等权持有全市场」的超额 alpha？

干净对照设计：
  - 三种模式用完全相同的持仓框架：每日选出 TopK 标的，等权买入，固定持有
    HOLD_MAX 个交易日后离场（无止损/回撤择时，只测「选币」本身）。
  - 唯一区别是「怎么选币」：
      factor : 量价因子 z-score 综合打分，取 TopK
      random : 随机选 K 个
      equal  : 全市场等权持有（beta 基准，等价于持有整个市场指数）
  - 组合收益用「每日持仓标的当日收益等权平均」，三模式完全同口径。

离场（本次聚焦"选币"，故简化为固定持有期；后续可叠加"回撤20%"离场再测）：
  固定持有 HOLD_MAX 天到期离场。
"""
import json
import os
import random
import statistics as st
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
KL_DIR = os.path.join(HERE, "..", "out", "scan", "klines_1d")
OUT = os.path.join(HERE, "..", "out", "scan")

TOP_K = 10
HOLD_MAX = 14          # 固定持有 14 个交易日
MIN_BARS = 120
LOOKBACK = 30
WARMUP = 120
FEE = 0.001
MC_LOW = 1e7
MC_HIGH = 1e10

FACTOR_WEIGHTS = {
    'vol_30d': 0.22,
    'amp_7d': 0.22,
    'vol_ratio_7_30': 0.14,
    'vol_ratio_3_30': 0.12,
    'ret_30d': 0.12,
    'dist_ma50': 0.10,
    'rsi_14': 0.08,
}


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


def load_mcap():
    rows = json.load(open(os.path.join(OUT, "market_cap_rows.json")))
    mc = {}
    for r in rows:
        if r['cat'] == 'crypto' and r.get('mc_circ'):
            mc[r['base']] = float(r['mc_circ'])
    return mc


def precompute_factors(rows):
    c = [r[4] for r in rows]
    h = [r[2] for r in rows]
    l = [r[3] for r in rows]
    v = [r[5] for r in rows]
    n = len(rows)

    def rolling_sum(vals, w):
        prefix = [0.0] * (len(vals) + 1)
        for i in range(len(vals)):
            prefix[i + 1] = prefix[i] + vals[i]
        out = [None] * len(vals)
        for i in range(len(vals)):
            if i >= w - 1:
                out[i] = prefix[i + 1] - prefix[i + 1 - w]
        return out

    v30 = rolling_sum(v, 30)
    v7 = rolling_sum(v, 7)
    v3 = rolling_sum(v, 3)
    ret1 = [None] * n
    for i in range(1, n):
        if c[i - 1] > 0:
            ret1[i] = c[i] / c[i - 1] - 1.0

    factors = [{} for _ in range(n)]
    for i in range(n):
        f = {}
        f['ret_30d'] = (c[i] / c[i - 30] - 1.0) if (i >= 30 and c[i - 30] > 0) else None
        if i >= 30 and v30[i] and v30[i] > 0:
            f['vol_ratio_7_30'] = (v7[i] / 7) / (v30[i] / 30) - 1.0 if v7[i] else None
            f['vol_ratio_3_30'] = (v3[i] / 3) / (v30[i] / 30) - 1.0 if v3[i] else None
        else:
            f['vol_ratio_7_30'] = f['vol_ratio_3_30'] = None
        if i >= 30:
            win = [x for x in ret1[i - 29:i + 1] if x is not None]
            f['vol_30d'] = st.pstdev(win) if len(win) > 2 else None
        else:
            f['vol_30d'] = None
        if i >= 6:
            amps = [(h[j] - l[j]) / c[j] for j in range(i - 6, i + 1) if c[j] > 0]
            f['amp_7d'] = sum(amps) / len(amps) if amps else None
        else:
            f['amp_7d'] = None
        if i >= 14:
            g = lo = 0.0
            for j in range(i - 13, i + 1):
                chg = c[j] - c[j - 1]
                if chg > 0:
                    g += chg
                else:
                    lo += -chg
            ag, al = g / 14, lo / 14
            f['rsi_14'] = 100 - 100 / (1 + ag / al) if al > 0 else 100.0
        else:
            f['rsi_14'] = None
        if i >= 49:
            ma50 = sum(c[i - 49:i + 1]) / 50
            f['dist_ma50'] = c[i] / ma50 - 1.0 if ma50 > 0 else None
        else:
            f['dist_ma50'] = None
        factors[i] = f
    return factors


def main():
    print("加载数据...", flush=True)
    syms = sorted(f.rsplit('_', 1)[0] for f in os.listdir(KL_DIR)
                  if f.endswith('_1d.json'))
    mc = load_mcap()

    btc_rows = load_kl('BTCUSDT')
    cal = [r[0] for r in btc_rows]
    n_days = len(cal)
    print(f"交易日历 {n_days} 天", flush=True)

    pool = [s for s in syms
            if MC_LOW <= mc.get(s.replace('USDT', ''), 0) <= MC_HIGH]

    prepared = {}
    for s in pool:
        rows = load_kl(s)
        if rows is None:
            continue
        fac = precompute_factors(rows)
        ts_list = [r[0] for r in rows]
        o_list = [r[1] for r in rows]
        c_list = [r[4] for r in rows]
        ts2i = {t: i for i, t in enumerate(ts_list)}
        prepared[s] = (ts_list, o_list, c_list, fac, ts2i)
    print(f"标的池 {len(prepared)} 个", flush=True)

    def run(mode, seed=42):
        rng = random.Random(seed)
        # holdings[sym] = (entry_day, entry_px)
        holdings = {}
        trades = []
        daily_ret = []

        for t in range(n_days):
            today_ts = cal[t]

            # 1) 固定持有期离场
            to_close = []
            for sym, (ed, epx) in holdings.items():
                sdata = prepared[sym]
                i = sdata[4].get(today_ts)
                if i is None:
                    continue
                px = sdata[2][i]
                if t - ed >= HOLD_MAX:
                    ret = px / epx - 1.0
                    trades.append({'sym': sym, 'days': t - ed,
                                   'ret': ret, 'net_ret': ret - 2 * FEE,
                                   'reason': 'time'})
                    to_close.append(sym)
            for sym in to_close:
                del holdings[sym]

            # 2) 组合当日收益
            if holdings:
                s = 0.0
                cnt = 0
                for sym, (ed, epx) in holdings.items():
                    sdata = prepared[sym]
                    i = sdata[4].get(today_ts)
                    pi = sdata[4].get(cal[t - 1]) if t > 0 else None
                    if i is not None and pi is not None and sdata[2][pi] > 0:
                        s += sdata[2][i] / sdata[2][pi] - 1.0
                        cnt += 1
                daily_ret.append(s / cnt if cnt else 0.0)
            else:
                daily_ret.append(0.0)

            # 3) 选币（T 日信号，T+1 入场）
            if t < WARMUP or t >= n_days - 1:
                continue

            if mode == 'equal':
                for sym, sdata in prepared.items():
                    i = sdata[4].get(today_ts)
                    if i is None:
                        continue
                    if sym not in holdings:
                        holdings[sym] = (t, sdata[2][i])
                continue

            keys = list(FACTOR_WEIGHTS.keys())
            fac_today = {}
            for sym, sdata in prepared.items():
                i = sdata[4].get(today_ts)
                if i is None or i < LOOKBACK:
                    continue
                fac_today[sym] = sdata[3][i]

            if mode == 'factor':
                mu, sd = {}, {}
                for k in keys:
                    vals = [fac_today[s][k] for s in fac_today
                            if fac_today[s].get(k) is not None]
                    mu[k], sd[k] = (0.0, 0.0) if len(vals) < 5 else \
                        (sum(vals) / len(vals), st.pstdev(vals))
                scored = []
                for s, f in fac_today.items():
                    sc = sum(FACTOR_WEIGHTS[k] * (f[k] - mu[k]) / sd[k]
                             for k in keys if f.get(k) is not None and sd[k] > 0)
                    scored.append((sc, s))
                scored.sort(key=lambda x: -x[0])
                picked = [s for _, s in scored[:TOP_K]]
            else:  # random
                cands = list(fac_today.keys())
                picked = rng.sample(cands, min(TOP_K, len(cands)))

            for sym in picked:
                if sym in holdings:
                    continue
                sdata = prepared[sym]
                i = sdata[4].get(today_ts)
                holdings[sym] = (t + 1, sdata[2][i])  # T+1 收盘价近似入场

        return summarize(trades, daily_ret, n_days, mode)

    results = {}
    for mode in ['factor', 'random', 'equal']:
        print(f"回测 {mode} ...", flush=True)
        results[mode] = run(mode)
        print(fmt(results[mode]), flush=True)

    # 补充 BTC 持有基准（真 beta）
    btc_prepared = None
    if 'BTCUSDT' in prepared:
        btc_prepared = prepared['BTCUSDT']
    else:
        b = load_kl('BTCUSDT')
        if b is not None:
            bfac = precompute_factors(b)
            bt = [r[0] for r in b]
            bo = [r[1] for r in b]
            bc = [r[4] for r in b]
            btc_prepared = (bt, bo, bc, bfac, {t: i for i, t in enumerate(bt)})
    if btc_prepared:
        # BTC 买入持有：全程持有，逐日收益
        btc_daily = []
        for t in range(1, n_days):
            i = btc_prepared[4].get(cal[t])
            pi = btc_prepared[4].get(cal[t - 1])
            if i is not None and pi is not None and btc_prepared[2][pi] > 0:
                btc_daily.append(btc_prepared[2][i] / btc_prepared[2][pi] - 1.0)
            else:
                btc_daily.append(0.0)
        cum = 1.0
        for r in btc_daily:
            cum *= (1 + r)
        years = n_days / 365.0
        results['btc_hold'] = {
            'mode': 'btc_hold', 'n_trades': 1,
            'total_ret': cum - 1.0,
            'ann_ret': cum ** (1 / years) - 1.0 if years > 0 and cum > 0 else None,
        }
        print(f"\n  {'btc_hold':<8} 买入持有 BTC：累计={ (cum-1)*100:+.2f}%  年化={(cum**(1/years)-1)*100:+.2f}%")

    json.dump(results, open(os.path.join(OUT, "strategy_results.json"), "w"),
              ensure_ascii=False, indent=2, default=str)
    print("\n已保存 out/scan/strategy_results.json", flush=True)


def summarize(trades, daily_ret, n, mode):
    res = {'mode': mode, 'n_trades': len(trades)}
    if not trades:
        return res
    nets = [t['net_ret'] for t in trades]
    days = [t['days'] for t in trades]
    res['win_rate'] = sum(1 for x in nets if x > 0) / len(nets)
    res['avg_net'] = sum(nets) / len(nets)
    res['median_net'] = sorted(nets)[len(nets) // 2]
    res['avg_days'] = sum(days) / len(days)
    cum = 1.0
    for r in daily_ret:
        cum *= (1 + r)
    res['total_ret'] = cum - 1.0
    years = n / 365.0
    res['ann_ret'] = cum ** (1 / years) - 1.0 if years > 0 and cum > 0 else None
    res['reasons'] = dict(Counter(t['reason'] for t in trades))
    return res


def fmt(res):
    def pct(x):
        return f"{x*100:+.2f}%" if x is not None else "N/A"
    wr = pct(res['win_rate']) if res.get('win_rate') is not None else 'N/A'
    return (f"  {res['mode']:<8} 笔数={res['n_trades']}  胜率={wr}  "
            f"单笔净={pct(res.get('avg_net'))}  中位净={pct(res.get('median_net'))}  "
            f"均持有={res.get('avg_days',0):.1f}天\n"
            f"    累计={pct(res.get('total_ret'))}  年化={pct(res.get('ann_ret'))}")


if __name__ == "__main__":
    main()
