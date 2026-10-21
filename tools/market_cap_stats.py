#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
币安全市场现货(USDT)市值与因子统计。

数据源:
  - 币安 exchangeInfo + ticker/24hr (data-api.binance.vision, 直连可用)
  - coinlore.net 市值数据 (含 circulating/total/max supply)

产出:
  1. 流动市值(circulating market cap) 与 完全稀释市值(FDV) 的分位数分布
  2. 每个可量化因子的分布统计(价格/成交量/成交额/换手/流通率/稀释率/涨跌幅/排名)
"""
import json
import os
import time
import urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "scan")
BINANCE_BASE = "https://data-api.binance.vision/api/v3"

STABLES = {'USDT', 'USDC', 'FDUSD', 'TUSD', 'DAI', 'BUSD', 'EUR', 'USDP',
           'AEUR', 'EURI', 'PYUSD', 'USD1', 'XAUT', 'PAXG', 'USDE', 'SUSD',
           'USDX', 'USTC', 'FEI', 'LUSD', 'USDY', 'USDF'}

STOCK_TICKERS = {'AAPL', 'AMZN', 'MSFT', 'GOOGL', 'META', 'TSLA', 'NVDA',
                 'AMD', 'INTC', 'IBM', 'COIN', 'HOOD', 'GME', 'AMC', 'BB',
                 'NFLX', 'DELL', 'ARM', 'ASML', 'AVGO', 'AXTI', 'BABA',
                 'COHR', 'CRWD', 'CRWV', 'CYPH', 'DJT', 'EWY', 'FLNC', 'GLW',
                 'GPRO', 'GS', 'HIMS', 'MU', 'ORCL', 'PLTR', 'PFE', 'QCOM',
                 'SMCI', 'SNPS', 'TSM', 'WBD', 'BA', 'DIS', 'JPM', 'V',
                 'MSTR', 'WMT', 'XOM'}


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode('utf-8'))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def classify(base):
    if base in STABLES:
        return 'stable'
    if base in STOCK_TICKERS:
        return 'stock'
    if base.endswith('B') and base[:-1] in STOCK_TICKERS:
        return 'stock'
    return 'crypto'


def resolve_coinlore(base, cl_by_sym):
    """返回 (coinlore记录, 匹配模式)。"""
    if base in cl_by_sym:
        return cl_by_sym[base], 'direct'
    if base.startswith('1000') and base[4:] in cl_by_sym:
        return cl_by_sym[base[4:]], '1000x'
    if base.endswith('X') and base[:-1] in cl_by_sym:
        return cl_by_sym[base[:-1]], 'x-suffix'
    return None, None


def load_coinlore(cache_path):
    if os.path.exists(cache_path):
        return json.load(open(cache_path, encoding='utf-8'))
    coins = []
    start = 0
    while True:
        d = _get(f"https://api.coinlore.net/api/tickers/?start={start}&limit=100")
        batch = d.get('data', [])
        if not batch:
            break
        coins.extend(batch)
        start += 100
        time.sleep(0.3)
    json.dump(coins, open(cache_path, 'w', encoding='utf-8'), ensure_ascii=False)
    return coins


def main():
    os.makedirs(OUT, exist_ok=True)

    # 1. 币安全市场 USDT 现货
    exch = _get(f"{BINANCE_BASE}/exchangeInfo")
    usdt_spot = [s for s in exch['symbols']
                 if s.get('quoteAsset') == 'USDT'
                 and s.get('status') == 'TRADING'
                 and s.get('isSpotTradingAllowed', True)]

    # 2. 全市场 24hr ticker (价格/量/额/涨跌幅)
    tickers = _get(f"{BINANCE_BASE}/ticker/24hr")
    tk_by_sym = {t['symbol']: t for t in tickers}

    # 3. coinlore 市值
    cl = load_coinlore(os.path.join(OUT, 'coinlore_all.json'))
    cl_by_sym = {}
    for c in cl:
        sym = c['symbol'].upper()
        mc = fnum(c.get('market_cap_usd')) or 0
        if sym not in cl_by_sym or mc > (fnum(cl_by_sym[sym].get('market_cap_usd')) or 0):
            cl_by_sym[sym] = c

    rows = []
    for s in usdt_spot:
        sym = s['symbol']
        base = s['baseAsset']
        cat = classify(base)
        tk = tk_by_sym.get(sym, {})

        price = fnum(tk.get('lastPrice'))
        vol24 = fnum(tk.get('volume'))          # base 数量
        quote_vol = fnum(tk.get('quoteVolume')) # USDT 成交额
        pct1h = None
        pct24 = fnum(tk.get('priceChangePercent'))

        row = {
            'symbol': sym, 'base': base, 'cat': cat,
            'price': price, 'volume24': vol24, 'quote_volume24': quote_vol,
            'pct24': pct24,
            'mc_circ': None, 'mc_fdv': None,
            'csupply': None, 'tsupply': None, 'msupply': None,
            'rank': None, 'match': None,
        }

        if cat == 'crypto':
            rec, mode = resolve_coinlore(base, cl_by_sym)
            if rec is not None:
                row['match'] = mode
                row['rank'] = fnum(rec.get('rank'))
                row['csupply'] = fnum(rec.get('csupply'))
                row['tsupply'] = fnum(rec.get('tsupply'))
                row['msupply'] = fnum(rec.get('msupply'))
                mc_circ = fnum(rec.get('market_cap_usd'))
                row['mc_circ'] = mc_circ
                # FDV = 价格 * 最大供应量(优先 msupply，其次 tsupply)
                sup = row['msupply'] or row['tsupply'] or row['csupply']
                if sup and price:
                    row['mc_fdv'] = price * sup
                else:
                    row['mc_fdv'] = mc_circ

        rows.append(row)

    # 输出原始数据
    json.dump(rows, open(os.path.join(OUT, 'market_cap_rows.json'), 'w'),
              ensure_ascii=False)
    print(f"总标的: {len(rows)}")
    print(f"分类: {dict(Counter(r['cat'] for r in rows))}")

    # 只对 crypto 且有市值的做统计
    crypto = [r for r in rows if r['cat'] == 'crypto' and r['mc_circ']]
    print(f"有市值的加密货币标的: {len(crypto)}")

    return rows, crypto


if __name__ == '__main__':
    rows, crypto = main()
