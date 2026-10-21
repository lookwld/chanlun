"""级别流水线冒烟: 在真实归档数据上跑一遍, 检查性能/内存/结构自洽。

运行: python tools/smoke_level.py [SYM] [MAX_BARS]
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook.data import fmt                      # noqa: E402
from chanbook.level import LevelSet                # noqa: E402


def main() -> int:
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    max_bars = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    for market in ("um", "spot"):
        ls = LevelSet(market, sym, use_confirm=True).load()
        if not ls.ok():
            print(f"[{market}] {sym} 数据不足 "
                  f"(op={ls.op.kl.n} trend={ls.trend.kl.n})")
            continue
        op = ls.op.kl
        n = op.n if not max_bars else min(op.n, max_bars)
        t0 = time.time()
        n_ev = 0
        kinds = {"B1": 0, "B2": 0, "B3": 0}
        first_ts = last_ts = 0
        for i in range(n):
            close_ts = op.ts[i] + ls.op.tf_ms
            ls.advance(close_ts)
            for p in ls.op.scan_points():
                n_ev += 1
                kinds[p.kind] += 1
                if not first_ts:
                    first_ts = op.ts[i]
                last_ts = op.ts[i]
        dt = time.time() - t0
        st = ls.trend.state
        sn = ls.op.snapshot()
        print(f"[{market}] {sym}  op={ls.op.tf} {n} 根  trend={ls.trend.tf} "
              f"{ls.trend.kl.n} 根  confirm={ls.confirm.kl.n if ls.confirm else '-'} 根")
        print(f"   耗时 {dt:.1f}s ({dt / max(1, n) * 1e6:.1f} µs/根)  "
              f"笔 {sn.n_bi}(冻结{sn.n_final}) 线段 {sn.n_seg} 中枢 {sn.n_zs}")
        print(f"   大级别走势: {st.label()} | {st.why}")
        print(f"   买卖点 {n_ev} 个 {kinds}  "
              f"首个 {fmt(first_ts) if first_ts else '-'}"
              f" 末个 {fmt(last_ts) if last_ts else '-'}")
        print(f"   自检: 笔顶底交替={bi_alternate(ls.op.bib.bis)} "
              f"线段≥3笔={all(s.n_bi >= 3 for s in ls.op.segb.segs)} "
              f"中枢ZG>ZD={all(z.zg > z.zd for z in ls.op.zb.zss)}")
    return 0


def bi_alternate(bis) -> bool:
    for k in range(1, len(bis)):
        if bis[k].direction == bis[k - 1].direction:
            return False
    return True


if __name__ == "__main__":
    sys.exit(main())
