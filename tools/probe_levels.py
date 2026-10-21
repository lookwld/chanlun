"""级别可行性探针: 4h / 日线 / 3日线 / 周线 能否跑完整缠论流水线并给出第一类买点。

只读。用 1h 归档现场聚合(4h/1d 在归档里基本缺档), 并在**本进程内**临时把
"3d"/"1w" 补进 TF_MS —— 内核源码不改(那属于另一条线的工作)。

运行: python tools/probe_levels.py [SYM ...]
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                              # noqa: E402
from chanbook.level import Level                            # noqa: E402
from chanbook.points import B1, B2, B3                      # noqa: E402

EXTRA = {"3d": 3 * D.MS_DAY, "1w": 7 * D.MS_DAY}
TFS = ["1h", "4h", "1d", "3d", "1w"]


def tf_ms_of(tf: str) -> int:
    return D.TF_MS[tf]


def build(market: str, sym: str, tf: str):
    base = D.load(market, sym, "1h")
    if base.n == 0:
        return None, None
    ms = tf_ms_of(tf)
    kl = base if tf == "1h" else D.resample(base, ms)
    lv = Level("op", tf, "bi")      # 与项目 op 级别同口径: 笔作中枢的次级别构件
    lv.attach(kl)
    return kl, lv


def run(market: str, sym: str, tf: str) -> dict:
    kl, lv = build(market, sym, tf)
    if kl is None:
        return {"tf": tf, "n": 0, "err": "无 1h 底档"}
    t0 = time.time()
    cnt = {B1: 0, B2: 0, B3: 0}
    b1 = []
    for i in range(kl.n):
        lv.advance(kl.ts[i] + lv.tf_ms)
        for p in lv.scan_points():
            cnt[p.kind] = cnt.get(p.kind, 0) + 1
            if p.kind == B1:
                b1.append(p)
    dt = time.time() - t0
    sn = lv.snapshot()
    # 一买里"有中枢语境"的比例 —— 才算得上原文的"趋势背驰"
    zs_ctx = sum(1 for p in b1 if p.zs_i >= 0)
    tb = sum(1 for p in b1 if p.feats.get("trend_beichi"))
    return {
        "tf": tf, "n": kl.n, "bi": sn.n_bi, "final": sn.n_final,
        "seg": sn.n_seg, "zs": sn.n_zs, "trend": lv.state.kind,
        "B1": cnt[B1], "B2": cnt[B2], "B3": cnt[B3],
        "b1_zs": zs_ctx, "b1_tb": tb, "sec": dt,
        "first_ts": kl.ts[0], "last_ts": kl.ts[-1],
        "b1_feats": (b1[0].feats if b1 else None),
    }


def main() -> int:
    syms = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]
    D.TF_MS.update(EXTRA)          # 仅本进程内补齐 3d/1w, 内核源码不动
    print("=" * 108)
    print("级别可行性探针 —— 口径: 笔作中枢次级别构件(mode='bi'), 与项目 op 级别一致")
    print("=" * 108)
    for market in ("um", "spot"):
        for sym in syms:
            for tf in TFS:
                r = run(market, sym, tf)
                if r.get("err") or r["n"] == 0:
                    print(f"{market:5}{sym:10}{tf:4}  —— 无法构建({r.get('err', '空)')})")
                    continue
                print(f"{market:5}{sym:10}{tf:4}  根数 {r['n']:7d}  "
                      f"{D.fmt(r['first_ts'], '%Y-%m-%d')}→{D.fmt(r['last_ts'], '%Y-%m-%d')}  "
                      f"({r['n'] * tf_ms_of(tf) / D.MS_DAY / 365.25:5.2f} 年)  "
                      f"笔 {r['bi']:5d}(冻结{r['final']:5d})  线段 {r['seg']:4d}  "
                      f"中枢 {r['zs']:3d}  {r['trend']}\n"
                      f"{'':26}买点 B1 {r['B1']:4d}(有中枢语境 {r['b1_zs']:3d} / "
                      f"真趋势背驰 {r['b1_tb']:3d})  "
                      f"B2 {r['B2']:4d}  B3 {r['B3']:4d}   "
                      f"用时 {r['sec']:.1f}s")
                if r["b1_feats"]:
                    print(f"{'':26}B1 首例特征: {r['b1_feats']}")
            print("-" * 108)
    return 0


if __name__ == "__main__":
    sys.exit(main())
