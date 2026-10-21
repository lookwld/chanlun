"""结构内核自检: 包含处理 / 分型 / 笔。

运行: python tests/check_core.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook.bi import BiBuilder, build_bi, validate          # noqa: E402
from chanbook.ckline import KLineMerger, merge_klines          # noqa: E402
from chanbook.fenxing import BOTTOM, TOP, find_fenxing         # noqa: E402

N_PASS = 0
FAILS: list[str] = []


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


def feeder(pairs):
    m = KLineMerger()
    for i, (h, l) in enumerate(pairs):
        m.push(i, 1_700_000_000_000 + i * 3_600_000, h, l)
    return m


# ====================================================================== 包含处理
def test_ckline_up() -> None:
    """上升途中的包含: 高高。"""
    m = feeder([(10, 8), (12, 9), (11, 9.5), (14, 10)])
    cks = m.cks
    ck(len(cks) == 3, f"上升包含: 合并后应为3根, 实际{len(cks)}")
    ck((cks[1].high, cks[1].low) == (12, 9.5),
       f"上升包含: 第2根应为 h12/l9.5, 实际 h{cks[1].high}/l{cks[1].low}")
    ck(cks[1].n == 2, "上升包含: 第2根应合并了2根原始K线")
    ck((cks[2].high, cks[2].low) == (14, 10), "上升包含: 第3根应保持原值")
    ck(m.direction == 1, "上升包含: 方向应为向上")


def test_ckline_down() -> None:
    """下降途中的包含: 低低。"""
    m = feeder([(14, 12), (13, 10), (12.5, 10.2), (9, 7)])
    cks = m.cks
    ck(len(cks) == 3, f"下降包含: 合并后应为3根, 实际{len(cks)}")
    ck((cks[1].high, cks[1].low) == (12.5, 10),
       f"下降包含: 第2根应为 h12.5/l10, 实际 h{cks[1].high}/l{cks[1].low}")
    ck(m.direction == -1, "下降包含: 方向应为向下")


def test_ckline_equal_endpoints() -> None:
    """端点相等也算包含。"""
    m = feeder([(10, 8), (12, 9), (12, 9.5)])
    ck(len(m.cks) == 2, "相等端点应判为包含并合并")


def test_ckline_index_tracking() -> None:
    """合并K线必须记录原始下标/时间区间。"""
    ck(merge_klines([]) == [], "空输入应返回空")
    m = KLineMerger()
    for i, (h, l) in enumerate([(10, 8), (12, 9), (11, 9.5), (14, 10)]):
        m.push(i, i * 1000 + 7, h, l)
    c = m.cks[1]
    ck((c.i0, c.i1) == (1, 2), f"合并区间应为[1,2], 实际[{c.i0},{c.i1}]")
    ck((c.ts0, c.ts1) == (1007, 2007), f"时间区间应为(1007,2007), 实际({c.ts0},{c.ts1})")


# ======================================================================== 分型
ZIG = [(10, 8), (11, 9), (12, 10), (11, 9), (10, 8), (9, 7), (8, 6), (9, 7),
       (10, 8), (11, 9), (12, 10), (11, 9), (10, 8), (9, 7), (8, 6), (9, 7)]


def test_fenxing() -> None:
    m = feeder(ZIG)
    cks = m.cks
    ck(len(cks) == len(ZIG), f"锯齿序列不应有包含关系(应{len(ZIG)}根, 实际{len(cks)})")
    fx = find_fenxing(cks)
    kinds = [(f.idx, f.kind, f.extreme) for f in fx]
    ck(kinds == [(2, TOP, 12), (6, BOTTOM, 6), (10, TOP, 12), (14, BOTTOM, 6)],
       f"锯齿序列分型应为 顶@2/底@6/顶@10/底@14, 实际 {kinds}")


def test_fenxing_strength() -> None:
    """力度必须用"回撤深度"度量, 不能用"第3根是否反破" (那恒为真)。"""
    m_strong = feeder([(10, 8), (12, 10), (11, 7)])       # 第3根深跌
    m_weak = feeder([(10, 8), (12, 10), (11, 9.5)])       # 第3根浅跌
    f1 = find_fenxing(m_strong.cks)
    f2 = find_fenxing(m_weak.cks)
    ck(len(f1) == 1 and f1[0].kind == TOP, "强顶分型应被识别")
    ck(len(f2) == 1 and f2[0].kind == TOP, "弱顶分型应被识别")
    ck(abs(f1[0].prominence - f2[0].prominence) < 1e-9,
       "两例突出度应相同(左右高点一致)")
    ck(f1[0].reversal > f2[0].reversal,
       f"深跌例回撤深度应更大({f1[0].reversal} vs {f2[0].reversal})")
    ck(f1[0].strength > f2[0].strength,
       f"强分型力度应更大({f1[0].strength} vs {f2[0].strength})")
    # 任一合法顶分型的回撤深度必然为正 —— 这条正是"恒真判据无区分度"的证据
    ck(f1[0].reversal > 0 and f2[0].reversal > 0, "合法顶分型回撤深度恒为正")


def test_fenxing_confirmation_lag() -> None:
    """分型确认需要右侧一根: 判定边界不得越过 len-2。"""
    m = feeder(ZIG[:4])
    fx = find_fenxing(m.cks)
    ck(all(f.idx <= len(m.cks) - 2 for f in fx), "分型不得使用未确认的右侧K线")
    ck(any(f.idx == 2 for f in fx), "idx2 顶分型在 4 根K线下应已确认")


# ========================================================================== 笔
def test_bi_basic() -> None:
    m = feeder(ZIG)
    fx = find_fenxing(m.cks)
    bis = build_bi(fx)
    errs = validate(bis)
    ck(not errs, f"笔结构自检应通过, 实际问题: {errs}")
    ck(len(bis) == 3, f"锯齿序列应成 3 笔, 实际 {len(bis)}")
    ck([b.direction for b in bis] == [-1, 1, -1],
       f"三笔方向应为 下/上/下, 实际 {[b.direction for b in bis]}")
    ck((bis[0].start.extreme, bis[0].end.extreme) == (12, 6),
       f"第1笔应为 12→6, 实际 {bis[0].start.extreme}→{bis[0].end.extreme}")
    ck((bis[1].start.extreme, bis[1].end.extreme) == (6, 12),
       f"第2笔应为 6→12, 实际 {bis[1].start.extreme}→{bis[1].end.extreme}")


def test_bi_min_gap() -> None:
    """间隔不足不得成笔; 放宽 min_gap 后应成笔。"""
    m = feeder([(8, 6), (9, 7), (10, 8), (9, 7), (8, 6), (9, 7), (10, 8)])
    fx = find_fenxing(m.cks)
    bis = build_bi(fx, min_gap=4)
    ck(len(bis) == 0, f"间隔不足应成 0 笔, 实际 {len(bis)}")
    b2 = BiBuilder(min_gap=4)
    for f in fx:
        b2.push(f)
    ck(b2.n_dropped > 0, "应有因间隔不足被丢弃的分型")
    bis_loose = build_bi(fx, min_gap=2)
    ck(len(bis_loose) == 1, f"min_gap=2 应成 1 笔, 实际 {len(bis_loose)}")


def test_bi_extension_and_final() -> None:
    """同类更极端分型到来时最后一笔延伸; final 始终排除最后一笔。

    必须"增量喂K线 + 只推新出现的分型", 否则重复推送旧分型会打乱序列
    —— 这本身就是流式语义的一部分。
    """
    pairs = ZIG + [(8, 6), (7, 5), (6, 4), (7, 5)]      # 追加一段下跌, 造更低的底(4)
    m = KLineMerger()
    b = BiBuilder()
    seen = 0
    snap = {}
    for i, (h, l) in enumerate(pairs):
        m.push(i, 1_700_000_000_000 + i * 3_600_000, h, l)
        fxs = find_fenxing(m.cks)
        for f in fxs[seen:]:
            b.push(f)
        seen = len(fxs)
        if i == len(ZIG) - 1:
            snap["n_bis"] = len(b.bis)
            snap["n_final"] = b.n_final
            snap["end"] = b.bis[-1].end.extreme
    ck(snap["n_bis"] == 3, f"16根锯齿应成 3 笔, 实际 {snap['n_bis']}")
    ck(snap["n_final"] == 2, f"可冻结笔应为 2, 实际 {snap['n_final']}")
    ck(snap["end"] == 6, f"延伸前最后一笔末端应为 6, 实际 {snap['end']}")
    ck(len(b.bis) == 3, f"延伸不应新增笔, 实际 {len(b.bis)}")
    ck(b.bis[-1].end.extreme == 4,
       f"最后一笔应延伸至新低 4, 实际 {b.bis[-1].end.extreme}")
    ck(b.bis[-1].end.idx == 18, f"端点应后移到 idx18, 实际 {b.bis[-1].end.idx}")
    ck(not validate(b.bis), f"延伸后结构自检仍应通过: {validate(b.bis)}")


def test_bi_streaming_matches_offline() -> None:
    """流式推送与离线批量必须给出一致的冻结笔(无未来函数的表现形式)。"""
    m = feeder(ZIG)
    fx = find_fenxing(m.cks)
    off = build_bi(fx)
    b = BiBuilder()
    for f in fx:
        b.push(f)
    ck(len(b.final_bis) == len(off) - 1,
       f"final 应为 全部-1 ({len(off)-1}), 实际 {len(b.final_bis)}")
    for x, y in zip(b.final_bis, off):
        ck((x.start.extreme, x.end.extreme) == (y.start.extreme, y.end.extreme),
           "冻结笔与离线结果不一致")


# ======================================================================== main
def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:                 # noqa: BLE001
            FAILS.append(f"{t.__name__} 抛异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n== 结构内核自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
