#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""下载币安全市场 USDT 现货的日线(1d)长历史 K 线，用于暴涨事件研究。"""
import json
import os
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "out", "scan", "klines_1d")
BASE = "https://data-api.binance.vision/api/v3"
TF = "1d"
LIMIT = 1000

os.makedirs(OUT, exist_ok=True)


def load_symbols():
    exch = os.path.join(HERE, "..", "out", "scan", "exchangeInfo.json")
    d = json.load(open(exch, encoding="utf-8"))
    syms = [s["symbol"] for s in d["symbols"]
            if s.get("quoteAsset") == "USDT"
            and s.get("status") == "TRADING"
            and s.get("isSpotTradingAllowed", True)]
    return sorted(syms)


def fetch_one(sym):
    path = os.path.join(OUT, f"{sym}_{TF}.json")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return sym, True, "cached"
    url = f"{BASE}/klines?symbol={urllib.parse.quote(sym)}&interval={TF}&limit={LIMIT}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            rows = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return sym, False, str(e)
    # 提取 ts,o,h,l,c,v,quote_vol,trades
    data = [[int(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]),
             float(x[5]), float(x[7]), int(x[8])] for x in rows]
    json.dump(data, open(path, "w", encoding="utf-8"))
    return sym, True, f"{len(data)} bars"


def main():
    syms = load_symbols()
    print(f"标的数: {len(syms)}")
    jobs = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    done = 0
    fail = []
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for sym, ok, msg in ex.map(fetch_one, syms):
            done += 1
            if not ok:
                fail.append((sym, msg))
            if done % 50 == 0:
                print(f"进度 {done}/{len(syms)}", flush=True)
    print(f"完成 {done}, 失败 {len(fail)}")
    for s, m in fail:
        print("  FAIL", s, m)


if __name__ == "__main__":
    main()
