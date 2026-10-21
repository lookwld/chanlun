"""线段自检: 特征序列分型 / 第一与第二种破坏 / 强制兜底 / 流式一致性。

运行: python tests/check_seg.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook.bi import BiBuilder                              # noqa: E402
from chanbook.fenxing import BOTTOM, TOP, FX                    # noqa: E402
from chanbook.seg import SegBuilder, build_segs, validate       # noqa: E402

N_PASS = 0
FAILS: list[str] = []


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


def make_bis(prices: list[float]):
    """把转折点价格序列(偶数下标=底, 奇数=顶)转成笔序列。"""
    fxs = []
    for i, p in enumerate(prices):
        kind = BOTTOM if i % 2 == 0 else TOP
        fxs.append(FX(kind, i * 4, p, p, i * 4000, p))
    b = BiBuilder()
    for f in fxs:
        b.push(f)
    return b.bis


# 一段上涨(顶 30→40→45 依次抬高, 回踩低点 25→34→36 依次抬高) 后
# 出现 (42→30) 的回踩: 45 之后的回踩低点 30 低于前一个 34 与 36?
# 分型中心落在 bi5, 且第1/2特征元素间无缺口 => 第一种破坏。
UP_FIRST = [10, 30, 25, 40, 34, 45, 36, 42, 30, 38, 22, 35, 15, 28, 18, 30, 20]

# 上升线段被一根直接砸穿线段起点的向下笔强制破坏
FORCE = [10, 30, 25, 40, 34, 45, 5, 20, 15, 32, 12, 26, 20, 34, 22, 40, 25]


def test_seg_first_break() -> None:
    bis = make_bis(UP_FIRST)
    ck(len(bis) == len(UP_FIRST) - 1,
       f"应生成 {len(UP_FIRST)-1} 笔, 实际 {len(bis)}")
    segs = build_segs(bis)
    ck(len(segs) == 2, f"应确认 2 条线段, 实际 {len(segs)}: {segs}")
    if len(segs) >= 2:
        s0, s1 = segs[0], segs[1]
        ck(s0.direction == 1, "第1条应为向上线段")
        ck((s0.b0, s0.b1) == (0, 4), f"第1条笔区间应为 0-4, 实际 {s0.b0}-{s0.b1}")
        ck((s0.start_price, s0.end_price) == (10, 45),
           f"第1条应为 10→45, 实际 {s0.start_price}→{s0.end_price}")
        ck(s0.how == "first", f"第1条应为第一种破坏, 实际 {s0.how}")
        ck(s0.n_bi == 5, f"第1条应含 5 笔, 实际 {s0.n_bi}")
        ck(s1.direction == -1, "第2条应为向下线段")
        ck((s1.b0, s1.b1) == (5, 11), f"第2条笔区间应为 5-11, 实际 {s1.b0}-{s1.b1}")
        ck((s1.start_price, s1.end_price) == (45, 15),
           f"第2条应为 45→15, 实际 {s1.start_price}→{s1.end_price}")
        ck(s1.how == "first", f"第2条应为第一种破坏, 实际 {s1.how}")
    ck(not validate(segs), f"线段结构自检应通过: {validate(segs)}")


def test_seg_force_break() -> None:
    bis = make_bis(FORCE)
    b = SegBuilder()
    b.update(bis)
    ck(len(b.segs) >= 1, f"应至少确认 1 条线段, 实际 {len(b.segs)}")
    if b.segs:
        s0 = b.segs[0]
        ck(s0.direction == 1, "第1条应为向上线段")
        ck(s0.how == "force", f"应为强制破坏, 实际 {s0.how}")
        ck(s0.end_price == 45, f"顶应落在 45, 实际 {s0.end_price}")
        ck(s0.n_bi >= 3, f"线段至少 3 笔, 实际 {s0.n_bi}")
    ck(b.n_force >= 1, "计数器应记录到强制破坏")


def test_seg_streaming_equivalent() -> None:
    """逐笔增量 update 与一次性 update 必须一致。"""
    bis = make_bis(UP_FIRST)
    one = SegBuilder()
    one.update(bis)
    inc = SegBuilder()
    got = []
    for k in range(1, len(bis) + 1):
        got += inc.update(bis[:k])
    a = [(s.b0, s.b1, s.direction, s.end_price, s.how) for s in one.segs]
    b = [(s.b0, s.b1, s.direction, s.end_price, s.how) for s in got]
    ck(a == b, f"增量与批量结果必须一致:\n  批量={a}\n  增量={b}")


def test_seg_no_overlap_and_float() -> None:
    """线段之间笔区间必须连续不重叠, 且当前线段(provisional)不重复计入。"""
    bis = make_bis(UP_FIRST)
    b = SegBuilder()
    b.update(bis)
    cur = b.cur_snapshot(bis)
    for k in range(1, len(b.segs)):
        ck(b.segs[k].b0 == b.segs[k - 1].b1 + 1,
           f"线段{k}应与前一条笔区间连续")
    if cur is not None:
        ck(cur.b0 == b.segs[-1].b1 + 1,
           f"当前线段应从 {b.segs[-1].b1+1} 起, 实际 {cur.b0}")
        ck(cur.i == len(b.segs), "当前线段序号应等于已确认线段数")


def test_seg_counter_isolation() -> None:
    """两种破坏类型必须分别计数, 便于后续归因。"""
    b = SegBuilder()
    b.update(make_bis(UP_FIRST))
    ck(b.n_first + b.n_second + b.n_force == len(b.segs),
       "破坏类型计数之和应等于线段总数")


def test_seg_uses_frozen_bi_only() -> None:
    """最后一笔(可延伸)不得参与线段结论: 追加新笔不应改写已确认线段。"""
    bis = make_bis(UP_FIRST)
    b1 = SegBuilder()
    b1.update(bis)
    before = [(s.b0, s.b1, s.end_price) for s in b1.segs]
    # 追加一笔(靠延长序列), 已确认线段必须不变
    bis2 = make_bis(UP_FIRST + [44, 50])
    b2 = SegBuilder()
    b2.update(bis2)
    after = [(s.b0, s.b1, s.end_price) for s in b2.segs]
    ck(after[:len(before)] == before,
       f"追加数据不得改写已确认线段:\n  前={before}\n  后={after[:len(before)]}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:                 # noqa: BLE001
            FAILS.append(f"{t.__name__} 抛异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n== 线段自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
