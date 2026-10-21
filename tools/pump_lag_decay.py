#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""稳健性检验：因子区分度随「暴涨前提前量(lag)」的衰减。

若因子是「真先导信号」，越靠前(T-20)仍应有区分度；
若因子是「同步/已启动信号」，区分度会随 lag 增大迅速衰减到 0.5。
"""
import sys, os, random
from collections import defaultdict
import importlib.util

spec = importlib.util.spec_from_file_location("pfa", "tools/pump_factor_analysis.py")
pfa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pfa)

syms = sorted(f.rsplit("_", 1)[0] for f in os.listdir(pfa.KL_DIR)
              if f.endswith("_1d.json"))
btc_rows = pfa.load_kl("BTCUSDT")
btc_closes = [r[4] for r in btc_rows] if btc_rows else None

events = []
for sym in syms:
    rows = pfa.load_kl(sym)
    if rows is None:
        continue
    for e in pfa.merge_events(pfa.detect_pumps(rows)):
        events.append((sym, rows, e))

lags = [1, 3, 5, 10, 15, 20]
factors = ["ret_30d", "ret_14d", "ret_7d", "ret_3d", "amp_7d", "vol_30d",
           "vol_ratio_7_30", "vol_ratio_3_30", "rsi_14", "dist_ma20", "dist_ma50"]

random.seed(42)
neg = defaultdict(list)
for sym in syms:
    rows = pfa.load_kl(sym)
    if rows is None:
        continue
    c = [r[4] for r in rows]
    ev = set(pfa.merge_events(pfa.detect_pumps(rows)))
    n = len(c)
    cand = [i for i in range(pfa.MIN_BARS, n - pfa.M)
            if i not in ev and all(i - e > pfa.M for e in ev if e < i)]
    if not cand:
        continue
    for i in random.sample(cand, min(3, len(cand))):
        f = pfa.build_factors(rows, i, btc_closes)
        for k, v in f.items():
            if v is not None:
                neg[k].append(v)

header = "因子".ljust(16) + "".join(("T-%d" % l).rjust(9) for l in lags)
print(header)
print("-" * len(header))
for fk in factors:
    line = fk.ljust(16)
    for lag in lags:
        pos = []
        for sym, rows, e in events:
            if e - lag < 1:
                continue
            f = pfa.build_factors(rows, e - lag, btc_closes)
            if f.get(fk) is not None:
                pos.append(f[fk])
        if pos and neg.get(fk):
            a = pfa.auc(pos, neg[fk])
            line += ("%.3f" % a).rjust(9)
        else:
            line += "-".rjust(9)
    print(line)
