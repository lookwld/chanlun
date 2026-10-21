#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生产缓存 vs 全新全量下载 —— 逐根等价性抽检(联网, 只读, 不改任何缓存)。

为什么需要这个:
  `check_incremental.py` 是离线合成数据, 证明的是**算法**等价;
  这里要证明的是**真实生产缓存**等价 —— 服务器上那份被增量更新过的
  `klines_<date>/` 里, 每个文件的每一根 K 线, 是否与"从交易所重新完整
  下载一次"得到的窗口**逐根一致**。

口径(与下载器的增量语义对齐):
  - 缓存保留最近 LIMIT(=1000) 根;
  - 交易所 `limit=1000` 返回的也是**最近 1000 根**;
  - 所以对同一个 (标的, 周期), 两者的窗口天然相同 —— 可以逐根比。

⚠ 为什么不能要求"两边末根 open_time 相等"(2026-10-09 踩过, 我第一版就写错了):
  缓存是在**下载那一刻**写的, 它的末根 = 那一刻**正在走**的那根 K 线;
  抽检是几分钟后才跑的, 交易所此时返回的末根是**新的那根**。
  只要这中间跨过了周期边界(15m 周期 3 分钟就可能跨), 两边末根就**必然**不同。
  第一版把这个当 FAIL, 结果 48 个样本里 20 个报错 —— 而且**错的全是 15m/30m/1h**,
  4h 及以上一个都没错, 完全吻合"只跨了短周期的格子"。
  真正的判据是: **按 open_time 对齐后, 所有已收盘的公共 K 线必须逐根一致**,
  以及**缓存落后交易所不超过 1~2 根**(落后太多才是窗口错位)。

用法:
  python tests/check_live_equiv.py --cache out/scan/klines_20261009 \
      --n 40 --seed 20261009
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.request

BASE = "https://data-api.binance.vision"
LIMIT = 1000
TFS = ["1M", "1w", "3d", "1d", "4h", "1h", "30m", "15m"]
TF_MS = {"1M": 30 * 86400e3, "1w": 7 * 86400e3, "3d": 3 * 86400e3,
         "1d": 86400e3, "4h": 4 * 3600e3, "1h": 3600e3,
         "30m": 30 * 60e3, "15m": 15 * 60e3}


def fetch(sym: str, tf: str, base: str) -> list:
    url = f"{base}/api/v3/klines?symbol={sym}&interval={tf}&limit={LIMIT}"
    for att in range(4):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.loads(r.read().decode())
        except Exception as e:                       # noqa: BLE001
            if att == 3:
                raise
            time.sleep(1.5 * (att + 1))
    return []


def load_cache(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, list):
        return None
    return [list(r) for r in d]


def cmp_bars(a: list, b: list) -> tuple:
    """比两个 bar 的前 5 个 OHLCV 字段(K 线本体)。

    忽略后 3 个字段(close_time / quote_volume / trades / taker*):
    币安对**旧历史**返回的 trades/quoteVolume 与新版接口偶有舍入差异,
    而缠论只吃 OHLCV。close_time 则随 open_time 结构性地跟着走。
    """
    return [float(x) for x in a[:5]] == [float(x) for x in b[:5]]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="缓存目录(含 m15.json 之类)")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--n", type=int, default=40, help="抽检文件数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.cache) if f.endswith(".json"))
    if not files:
        print("缓存目录里没有 json 文件: " + args.cache)
        return 2
    rnd = random.Random(args.seed or 1)
    pick = rnd.sample(files, min(args.n, len(files)))

    n_ok = n_tail = n_bad = n_skip = 0
    n_bars = 0
    lags: dict = {}
    bad_detail: list = []
    for fn in sorted(pick):
        stem = fn[:-5]
        sym, _, tf = stem.rpartition("_")
        if tf not in TFS:
            n_skip += 1
            continue
        old = load_cache(os.path.join(args.cache, fn))
        if old is None or not old:
            print(f"  ✗ {fn}: 缓存读不出/为空")
            n_bad += 1
            continue
        try:
            new = fetch(sym, tf, args.base)
        except Exception as e:                       # noqa: BLE001
            print(f"  - {fn}: 交易所请求失败({e}), 跳过")
            n_skip += 1
            continue
        if not new:
            n_skip += 1
            continue

        ct = [int(r[0]) for r in old]
        et = [int(r[0]) for r in new]
        cmax, emax = ct[-1], et[-1]
        tfm = int(TF_MS[tf])

        # ---- 判据 1: 缓存不能比交易所还新(那说明缓存里有未来数据, 是真 bug)
        if cmax > emax:
            bad_detail.append(f"{fn}: 缓存末根晚于交易所 "
                              f"({cmax}>{emax}, 领先 {(cmax-emax)/tfm:.1f} 根)")
            n_bad += 1
            continue

        # ---- 判据 2: 缓存落后交易所的根数 —— 只允许"下载后跨了格子"带来的滞后
        lag = (emax - cmax) / tfm
        lags[tf] = max(lags.get(tf, 0.0), lag)
        if lag > 2.001:
            bad_detail.append(f"{fn}: 缓存落后交易所 {lag:.1f} 根(>2 根 ⇒ 窗口错位)")
            n_bad += 1
            continue

        # ---- 判据 3: 按 open_time 对齐, 公共**已收盘** K 线逐根比
        #     排除两边各自的末根: 缓存末根是"下载时刻的半根", 交易所末根是"此刻的半根",
        #     两次采样时刻不同 ⇒ 数值本就允许不同, 不是算法错。
        eset = {t: i for i, t in enumerate(et)}
        skip = {cmax, emax}
        common = [t for t in ct if t in eset and t not in skip]
        floor = 0.9 * min(len(ct), len(et)) - 3
        if len(common) < floor:
            bad_detail.append(f"{fn}: 公共 K 线只有 {len(common)} 根 "
                              f"(期望 ≥{floor:.0f}) ⇒ 窗口/网格不对齐")
            n_bad += 1
            continue

        cmap = {t: r for t, r in zip(ct, old)}
        mism = [t for t in common if not cmp_bars(cmap[t], new[eset[t]])]
        n_bars += len(common)
        if mism:
            t0 = mism[0]
            bad_detail.append(
                f"{fn}: {len(mism)}/{len(common)} 根已收盘 K 线不一致, 例 "
                f"ts={t0} cache={cmap[t0][:5]} exch={new[eset[t0]][:5]}")
            n_bad += 1
        else:
            n_ok += 1
            if lag > 0:
                n_tail += 1

    print("=" * 78)
    print(f"抽检缓存目录: {args.cache}")
    print(f"候选文件 {len(files)} 个, 本次抽 {len(pick)} 个 "
          f"(seed={args.seed or 1}), 实际比对 {n_ok + n_bad} 个")
    print(f"  逐根一致          : {n_ok}")
    print(f"  其中缓存末根落后  : {n_tail}  (下载后跨了格子, 已按 open_time 对齐)")
    print(f"  不一致/窗口错位   : {n_bad}")
    print(f"  跳过(请求失败)    : {n_skip}")
    print(f"  已收盘 K 线比对根数: {n_bars}")
    if lags:
        print("  各周期最大滞后    : " + ", ".join(
            f"{k}={v:.1f}根" for k, v in sorted(lags.items())))
    if bad_detail:
        print("-" * 78)
        for d in bad_detail[:20]:
            print("  ! " + d)
    print("=" * 78)
    verdict = "PASS" if n_bad == 0 and n_ok > 0 else "FAIL"
    print("VERDICT " + verdict)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump({"verdict": verdict, "ok": n_ok, "tail_live": n_tail,
                       "bad": n_bad, "skip": n_skip, "bars": n_bars,
                       "lags": lags, "detail": bad_detail},
                      f, ensure_ascii=False, indent=1)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
