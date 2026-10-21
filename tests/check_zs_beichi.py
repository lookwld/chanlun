"""中枢 / 走势类型 / MACD / 背驰 自检。

运行: python tests/check_zs_beichi.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook.beichi import leg_beichi, verdict                   # noqa: E402
from chanbook.macd import MACD, ema_series                        # noqa: E402
from chanbook.trend import TrendBuilder, decompose, UP, DOWN       # noqa: E402
from chanbook.zhongshu import ZSBuilder, build_zhongshu, merge_extensions  # noqa: E402

N_PASS = 0
FAILS: list[str] = []


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


class Leg:
    """测试用的次级别走势替身(与 Bi/Seg 同名接口)。"""

    def __init__(self, high, low, direction, i=0, ts0=0, ts1=0):
        self.high, self.low, self.direction, self.i = high, low, direction, i
        self.start_ts, self.end_ts = ts0, ts1


# ====================================================================== 中枢
def test_zs_basic() -> None:
    """三个连续走势的重叠: ZG=min(high), ZD=max(low)。"""
    legs = [Leg(12, 10, 1), Leg(11, 9, -1), Leg(13, 10.5, 1)]
    zs = build_zhongshu(legs)
    ck(len(zs) == 1, f"应形成 1 个中枢, 实际 {len(zs)}")
    if zs:
        z = zs[0]
        ck(z.zg == 11, f"ZG 应为 min(12,11,13)=11, 实际 {z.zg}")
        ck(z.zd == 10.5, f"ZD 应为 max(10,9,10.5)=10.5, 实际 {z.zd}")
        ck(z.gg == 13 and z.dd == 9, f"GG/DD 应为 13/9, 实际 {z.gg}/{z.dd}")
        ck(z.level == 1, "笔中枢级别应为 1")
        ck(z.n_legs == 3, f"应由 3 段构成, 实际 {z.n_legs}")


def test_zs_no_overlap() -> None:
    """无重叠不得成中枢。"""
    legs = [Leg(12, 11, 1), Leg(15, 14, -1), Leg(18, 17, 1)]
    zs = build_zhongshu(legs)
    ck(len(zs) == 0, f"无重叠应成 0 个中枢, 实际 {len(zs)}")


def test_zs_extension_and_rebirth() -> None:
    """延伸: 与区间有重叠 => 延续; 无重叠 => 结束并新生。"""
    legs = [Leg(12, 10, 1), Leg(11, 9, -1), Leg(13, 10.5, 1),   # 中枢 A
            Leg(12.5, 10.2, -1),                                 # 延伸
            Leg(20, 19, 1), Leg(22, 20, -1), Leg(24, 22, 1)]     # 完全在上方 => 结束A
    b = ZSBuilder(1)
    done = b.update(legs)
    ck(len(done) == 1, f"应结束 1 个中枢, 实际 {len(done)}")
    if done:
        z = done[0]
        ck(z.n_legs == 4, f"中枢 A 应含 4 段(含延伸), 实际 {z.n_legs}")
        ck(z.leave_i == 4, f"离开段应为第 4 段, 实际 {z.leave_i}")
        ck(z.enter_i == -1, f"起点即中枢, enter_i 应为 -1, 实际 {z.enter_i}")
    ck(len(b.zss) == 1, f"已结束中枢应只有 1 个, 实际 {len(b.zss)}")
    # 上面第 4 段之后的三段(20/22/24)仍未重叠 => 不构成新中枢
    cur = b.cur_snapshot()
    ck(cur is None, "20-19/22-20/24-22 无重叠, 不应形成中枢")


def test_zs_expansion_flag() -> None:
    """延伸达 9 段 => 标记为扩张(级别上移)。"""
    legs = [Leg(12, 10, 1), Leg(11, 9, -1), Leg(13, 10.5, 1)]
    for k in range(6):
        legs.append(Leg(12.6 - k * 0.05, 9.4 + k * 0.05, 1 if k % 2 else -1))
    b = ZSBuilder(1)
    b.update(legs)
    legs2 = legs + [Leg(30, 29, 1)]           # 一支"离开段"让中枢收尾
    b2 = ZSBuilder(1)
    done = b2.update(legs2)
    ck(len(done) == 1, f"应结束 1 个中枢, 实际 {len(done)}")
    if done:
        ck(done[0].n_legs >= 9, f"应延伸至 9 段以上, 实际 {done[0].n_legs}")
        ck(done[0].expanded, "9 段以上应标记 expanded")


def test_zs_merge_extension() -> None:
    """中枢扩展: 相邻同向且重叠的两个中枢合并、级别 +1。"""
    a = Leg(12, 10, 1); b_ = Leg(11, 9, -1); c = Leg(13, 10.5, 1)
    d = Leg(12.5, 10.2, -1); e = Leg(20, 12, 1)     # 与 A 重叠的离开段
    zss = build_zhongshu([a, b_, c, d, e])
    m = merge_extensions(zss)
    ck(len(m) <= len(zss), "合并后数量不得增加")


# ================================================================== 走势类型
def test_trend_up_down_range() -> None:
    # 两个同向不重叠中枢 => 上涨趋势
    zs_up = build_zhongshu([Leg(12, 10, 1), Leg(11, 9, -1), Leg(13, 10.5, 1)])
    # 手工造第二个更高的中枢
    from chanbook.zhongshu import ZS
    z2 = ZS(i=1, level=1, i0=4, i1=6, zg=20, zd=18, gg=22, dd=17,
            start_ts=100, end_ts=200, enter_i=3, leave_i=-1, enter_dir=1)
    st = TrendBuilder().update([zs_up[0], z2])
    ck(st.kind == UP, f"应为上涨趋势, 实际 {st.kind}")
    ck(st.n_zs == 2, f"应含 2 个中枢, 实际 {st.n_zs}")
    ck(not st.zhongyin, "尚未离开最后中枢, 不应是中阴")

    z3 = ZS(i=2, level=1, i0=7, i1=9, zg=9, zd=7, gg=10, dd=6,
            start_ts=300, end_ts=400, enter_i=6, leave_i=-1, enter_dir=-1)
    st2 = TrendBuilder().update([zs_up[0], z3])
    ck(st2.kind == DOWN, f"应为下跌趋势, 实际 {st2.kind}")

    z4 = ZS(i=3, level=1, i0=10, i1=12, zg=12, zd=10.5, gg=13, dd=10,
            start_ts=500, end_ts=600, enter_i=9, leave_i=-1, enter_dir=1)
    st3 = TrendBuilder().update([zs_up[0], z4])
    ck(st3.kind == "range", f"区间重叠应判为盘整, 实际 {st3.kind}")


def test_trend_zhongyin_and_incomplete() -> None:
    ck(TrendBuilder().update([]).incomplete is True, "无中枢 => 走势未完成")
    zs = build_zhongshu([Leg(12, 10, 1), Leg(11, 9, -1), Leg(13, 10.5, 1)])
    st = TrendBuilder().update(zs, leg_high=20.0)
    ck(st.zhongyin, "已明显向上离开中枢 => 应标记中阴阶段")
    ck(st.label() == "中阴阶段", f"label 应为中阴阶段, 实际 {st.label()}")


def test_trend_decompose() -> None:
    from chanbook.zhongshu import ZS
    a = ZS(0, 1, 0, 2, 11, 10.5, 12, 10, 0, 10, -1, 3, enter_dir=1)
    b = ZS(1, 1, 3, 5, 20, 18, 22, 17, 20, 30, -1, 6, enter_dir=1)
    c = ZS(2, 1, 6, 8, 30, 28, 32, 27, 40, 50, -1, 9, enter_dir=1)
    ts = decompose([a, b, c])
    ck(len(ts) == 1 and ts[0].kind == UP,
       f"3 个同向不重叠中枢应分解为 1 段上涨趋势, 实际 {[(t.kind, t.n_zs) for t in ts]}")
    ck(ts[0].n_zs == 3, f"应含 3 个中枢, 实际 {ts[0].n_zs}")
    d = ZS(3, 1, 9, 11, 20, 18, 22, 17, 60, 70, -1, 12, enter_dir=-1)
    ts2 = decompose([a, b, d])
    ck(len(ts2) == 2, f"方向切换应分解为 2 段, 实际 {len(ts2)}")


# ====================================================================== MACD
def test_macd_streaming_matches_offline() -> None:
    import random
    random.seed(3)
    px = [100.0]
    for _ in range(300):
        px.append(max(1.0, px[-1] * (1 + random.uniform(-0.02, 0.02))))
    m = MACD()
    for i, p in enumerate(px):
        m.update(i * 1000, p)
    ef = ema_series(px, 12)
    es = ema_series(px, 26)
    diff = [a - b for a, b in zip(ef, es)]
    ck(abs(m.d[-1] - diff[-1]) < 1e-9,
       f"增量 diff 应与离线一致: {m.d[-1]} vs {diff[-1]}")
    ck(m.n == len(px), "根数应一致")


def test_macd_area_and_peak() -> None:
    px = [100.0]
    for _ in range(40):
        px.append(px[-1] - 1.0)          # 单边下跌 => HIST 持续为负
    m = MACD()
    for i, p in enumerate(px):
        m.update(i * 1000, p)
    a_all = m.area(0, 40 * 1000)
    a_neg = m.area(0, 40 * 1000, -1)
    ck(abs(a_all - a_neg) < 1e-9, "单边下跌时 |HIST| 面积应全在负侧")
    ck(a_all > 0, "面积应为正")
    ck(m.peak(0, 40 * 1000, -1) > 0, "DIFF 峰值应为正")
    ck(m.n_bars(0, 30 * 1000) == 31, f"根数应为 31, 实际 {m.n_bars(0, 30*1000)}")


def _feed(prices: list[float]) -> MACD:
    m = MACD()
    for i, p in enumerate(prices):
        m.update(i * 3_600_000, p)
    return m


def test_beichi_positive() -> None:
    """后段下跌更慢 => 面积/峰值双衰减 => 背驰成立, 且创出新低。"""
    px = [100.0]
    while px[-1] > 50:
        px.append(px[-1] - 2.5)                 # 快跌 100→50
    a0, a1 = len(px) - 1, 0
    while px[-1] < 90:
        px.append(px[-1] + 2.5)                 # 反弹到 90
    c_start = len(px) - 1
    while px[-1] > 45:
        px.append(px[-1] - 0.6)                 # 慢跌 90→45(创新低)
    c_end = len(px) - 1
    m = _feed(px)
    r = verdict(m, 0, a0 * 3_600_000, c_start * 3_600_000, c_end * 3_600_000,
                direction=-1, px_a=50, px_c=45)
    ck(r.ok, f"应判定为背驰: {r.why}")
    ck(r.new_extreme, "后段创出新低 => new_extreme 应为 True")
    ck(r.ratios["area"] < 0.9, f"面积比应明显小于1, 实际 {r.ratios['area']}")
    ck(r.short().startswith("趋势背驰"), f"应标注趋势背驰, 实际 {r.short()}")


def test_beichi_negative() -> None:
    """后段下跌更快 => 不背驰。"""
    px = [100.0]
    while px[-1] > 70:
        px.append(px[-1] - 0.6)                 # 慢跌
    a0 = len(px) - 1
    while px[-1] < 95:
        px.append(px[-1] + 1.2)
    c_start = len(px) - 1
    while px[-1] > 60:
        px.append(px[-1] - 3.0)                 # 加速下跌
    c_end = len(px) - 1
    m = _feed(px)
    r = verdict(m, 0, a0 * 3_600_000, c_start * 3_600_000, c_end * 3_600_000,
                direction=-1, px_a=70, px_c=60)
    ck(not r.ok, f"加速下跌不应判为背驰, 实际 {r.why}")


def test_beichi_strict_and_guards() -> None:
    """严格模式更难过; 区间重叠 / 数据不足必须安全失败。"""
    px = [100.0]
    for _ in range(30):
        px.append(px[-1] - 1.0)
    m = _feed(px)
    r = verdict(m, 0, 0, 0, 0, -1)
    ck(not r.ok, "数据不足或区间非法时必须 ok=False")
    r2 = verdict(m, 0, 10 * 3_600_000, 5 * 3_600_000, 20 * 3_600_000, -1)
    ck(not r2.ok and "重叠" in r2.why, f"重叠区间应被拒: {r2.why}")


def test_leg_beichi_wrapper() -> None:
    a = Leg(100, 50, -1, ts0=0, ts1=20 * 3_600_000)
    c = Leg(90, 40, -1, ts0=32 * 3_600_000, ts1=60 * 3_600_000)
    px = [100.0]
    while px[-1] > 50:
        px.append(px[-1] - 2.5)
    while px[-1] < 90:
        px.append(px[-1] + 2.5)
    while px[-1] > 40:
        px.append(px[-1] - 0.6)
    m = _feed(px)
    r = leg_beichi(m, a, c, -1)
    ck(isinstance(r.ok, bool), "leg_beichi 应返回结果对象")
    ck(r.a.get("area", 0) > r.c.get("area", 0),
       "前段面积应大于后段(该构造下)")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:                 # noqa: BLE001
            FAILS.append(f"{t.__name__} 抛异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n== 中枢/走势/MACD/背驰 自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
