"""合成行情下的三类买点诊断(开发用, 也可作为端到端冒烟)。

用法: python tools/explore_points.py [scenario]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook.bi import BiBuilder                                # noqa: E402
from chanbook.ckline import KLineMerger                          # noqa: E402
from chanbook.fenxing import scan_fenxing_at                     # noqa: E402
from chanbook.macd import MACD                                   # noqa: E402
from chanbook.points import PointScanner                         # noqa: E402
from chanbook.zhongshu import ZSBuilder                           # noqa: E402

TF = 900_000


def path(segments, noise=0.0, seed=7):
    """segments: [(根数, 终点价)] 线性插值成价格路径。"""
    import random
    rnd = random.Random(seed)
    px, cur = [], segments[0][1]
    for n, end in segments:
        for k in range(1, n + 1):
            v = cur + (end - cur) * k / n
            if noise:
                v *= 1 + rnd.uniform(-noise, noise)
            px.append(round(v, 4))
        cur = end
    return px


def run(px, pull_bi=1, tag=""):
    m = KLineMerger()
    bb = BiBuilder()
    zb = ZSBuilder(1)
    mac = MACD()
    sc = PointScanner(pull_bi=pull_bi)
    events = []
    seen_bi = 0
    for i, p in enumerate(px):
        ts = i * TF
        m.push(i, ts, p * 1.003, p * 0.997)
        j = len(m.cks) - 2
        if j >= 1:
            f = scan_fenxing_at(m.cks, j)
            if f is not None:
                bb.push(f)
        mac.update(ts, p)
        zb.update(bb.bis[:-1])
        if len(bb.bis) != seen_bi:
            seen_bi = len(bb.bis)
            for pt in sc.scan(bb.bis, zb.all_zss(), mac):
                events.append((i, pt))
    print(f"===== {tag} (pull_bi={pull_bi}) 根数={len(px)} "
          f"合并K线={len(m.cks)} 笔={len(bb.bis)} 中枢={len(zb.all_zss())} "
          f"线段破坏 first/second/force={zb.n_new} =====")
    print("  笔: ", [(b.i, b.direction, round(b.start_price, 2), round(b.end_price, 2))
                    for b in bb.bis])
    for z in zb.all_zss():
        print(f"  中枢#{z.i} [{z.zd:.2f},{z.zg:.2f}] i0={z.i0} i1={z.i1} "
              f"enter={z.enter_i} leave={z.leave_i} {z.kind}")
    print(f"  命中: " + (", ".join(f"bar{e[0]}:{e[1].kind}@{e[1].price:.2f}"
                                  for e in events) or "无"))
    return events


def sc_b3():
    return path([(30, 100.0), (40, 130.0), (60, 128.0), (30, 175.0),
                 (18, 152.0), (35, 210.0)], noise=0.004)


def sc_b1():
    """两段下跌 + 两段盘整(后者更低) + 第三段更弱的下跌 + 抬升 => 期待一买/二买。"""
    return path([(30, 100.0), (40, 160.0), (70, 157.0), (25, 112.0),
                 (70, 109.0), (45, 95.0), (20, 108.0), (30, 104.0),
                 (40, 130.0)], noise=0.004)


def sc_rally_range():
    return path([(30, 100.0), (50, 140.0), (80, 138.0), (40, 180.0)], noise=0.003)


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "b3"):
        run(sc_b3(), 1, "做多离开中枢后回试")
    if which in ("all", "b1"):
        run(sc_b1(), 1, "两中枢下跌后二次背驰")
        run(sc_b1(), 3, "同上前提(pull_bi=3)")
    if which in ("all", "rally"):
        run(sc_rally_range(), 1, "上涨后盘整再上")
        run(sc_rally_range(), 3, "同上前提(pull_bi=3)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
