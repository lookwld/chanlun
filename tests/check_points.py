"""三类买点与无未来函数自检。

运行: python tests/check_points.py
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook.bi import BiBuilder                                 # noqa: E402
from chanbook.ckline import KLineMerger                           # noqa: E402
from chanbook.fenxing import scan_fenxing_at                      # noqa: E402
from chanbook.macd import MACD                                    # noqa: E402
from chanbook.points import PointScanner                          # noqa: E402
from chanbook.zhongshu import ZSBuilder                            # noqa: E402

N_PASS = 0
FAILS: list[str] = []
TF = 900_000


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


def path(segments, noise=0.0, seed=7):
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


def run(px, pull_bi=1, upto=None, enable=("B1", "B2", "B3")):
    """逐bar跑完整流水线, 返回 (events, scanner, bis, zss)。

    events: [(bar_index, Point)] —— 只在"新笔产生"时扫描, 与引擎里的触发方式一致。
    """
    m = KLineMerger()
    bb = BiBuilder()
    zb = ZSBuilder(1)
    mac = MACD()
    sc = PointScanner(pull_bi=pull_bi, enable=enable)
    events = []
    seen = 0
    n = len(px) if upto is None else min(upto, len(px))
    for i in range(n):
        p = px[i]
        ts = i * TF
        m.push(i, ts, p * 1.003, p * 0.997)
        j = len(m.cks) - 2
        if j >= 1:
            f = scan_fenxing_at(m.cks, j)
            if f is not None:
                bb.push(f)
        mac.update(ts, p)
        zb.update(bb.bis[:-1])
        if len(bb.bis) != seen:
            seen = len(bb.bis)
            for pt in sc.scan(bb.bis, zb.all_zss(), mac):
                events.append((i, pt))
    return events, sc, bb.bis, zb.all_zss()


SC_B3 = [(30, 100.0), (40, 130.0), (60, 128.0), (30, 175.0), (18, 152.0), (35, 210.0)]
SC_B1 = [(30, 100.0), (40, 160.0), (70, 157.0), (25, 112.0), (70, 109.0),
         (45, 95.0), (20, 108.0), (30, 104.0), (40, 130.0)]


def kinds(ev):
    return [p.kind for _, p in ev]


def test_b3() -> None:
    ev, sc, bis, zss = run(path(SC_B3, 0.004), 1)
    ks = kinds(ev)
    ck("B3" in ks, f"应识别出第三类买点, 实际命中 {ks}")
    p = next(p for _, p in ev if p.kind == "B3")
    ck(p.zs_i >= 0, "三买必须挂在某个中枢上")
    ck(len(p.reasons) >= 3, "三买应给出完整的中文依据")
    z = [z for z in zss if z.i == p.zs_i]
    ck(z and p.price >= z[0].zg - 1e-9,
       f"三买的回试低点({p.price})必须不低于中枢上沿({z[0].zg if z else '?'})")


def test_b1_trend_beichi() -> None:
    ev, sc, bis, zss = run(path(SC_B1, 0.004), 1)
    ks = kinds(ev)
    ck("B1" in ks, f"应识别出第一类买点, 实际命中 {ks}")
    b1 = [p for _, p in ev if p.kind == "B1"]
    ck(all(p.beichi is not None for p in b1), "一买必须带背驰依据")
    ck(all(p.beichi.ok for p in b1), "一买携带的背驰结论必须成立")
    deep = min(b1, key=lambda p: p.price)
    ck(abs(deep.price - 94.86) < 2.0,
       f"最深的一买应落在末段下跌低点附近(≈94.86), 实际 {deep.price}")
    ck(any("跌破中枢" in r or "创新低" in r for r in deep.reasons),
       "一买依据应说明是跌破中枢/创新低造成的背驰")


def test_b2_after_b1() -> None:
    ev, sc, bis, zss = run(path(SC_B1, 0.004), 1)
    ks = kinds(ev)
    ck("B2" in ks, f"一买之后应出现二买, 实际命中 {ks}")
    p = next(p for _, p in ev if p.kind == "B2")
    ck(any("第一类买点" in r for r in p.reasons), "二买依据应引用一买")
    ck("未跌破" in "".join(p.reasons), "二买依据应说明未跌破一买低点")


def test_pull_bi_suppresses_b2() -> None:
    """要求回试构成 3 笔时, 单笔回试不得成为二买。"""
    ev, sc, bis, zss = run(path(SC_B1, 0.004), 3)
    ck("B2" not in kinds(ev), "pull_bi=3 时单笔回试不应触发二买")


def test_no_lookahead() -> None:
    """截断数据不得改变已发生的命中 —— 无未来函数的直接体现。"""
    px = path(SC_B1, 0.004)
    full, _, _, _ = run(px, 1)
    for cut in (150, 250, 320):
        part, _, _, _ = run(px, 1, upto=cut)
        a = [(i, p.kind, round(p.price, 8)) for i, p in full if i < cut]
        b = [(i, p.kind, round(p.price, 8)) for i, p in part]
        ck(a[:len(b)] == b, f"截断到 {cut} 根时命中不一致:\n  完整={a}\n  截断={b}")


def test_enable_switch() -> None:
    ev, _, _, _ = run(path(SC_B1, 0.004), 1, enable=("B3",))
    ck("B1" not in kinds(ev) and "B2" not in kinds(ev),
       f"关闭 B1/B2 后不应命中, 实际 {kinds(ev)}")
    ev2, _, _, _ = run(path(SC_B3, 0.004), 1, enable=("B1",))
    ck("B3" not in kinds(ev2), f"关闭 B3 后不应命中, 实际 {kinds(ev2)}")


def test_scanner_counters() -> None:
    ev, sc, _, _ = run(path(SC_B1, 0.004), 1)
    ck(sum(sc.n_hit.values()) == len(ev),
       f"计数器总数({sum(sc.n_hit.values())})应等于命中数({len(ev)})")


def test_beichi_normalization() -> None:
    """归一化算术: 面积比必须等于 (C/norm_c) / (A/norm_a)。

    语义背景(为什么必须有这个参数): MACD 是价格之差, 量纲就是价格。下跌趋势里
    后一段的价位天然更低, |HIST| 面积、|DIFF| 峰值都会系统性变小, 于是"价格越低
    越像背驰"。按各段价位归一之后, 四个判据都变成无量纲比值, 不同价位的两段才可比。
    这里直接验证算术关系, 避免用合成行情去间接推测。
    """
    from chanbook.beichi import verdict
    import random
    rnd = random.Random(11)
    px = [100.0]
    for _ in range(80):
        px.append(max(1.0, px[-1] * (1 + rnd.uniform(-0.01, 0.012))))
    m = MACD()
    for i, p in enumerate(px):
        m.update(i * TF, p)
    raw = verdict(m, 0, 30 * TF, 45 * TF, 75 * TF, 1)
    ka, kc = 100.0, 10.0
    nrm = verdict(m, 0, 30 * TF, 45 * TF, 75 * TF, 1, norm_a=ka, norm_c=kc)
    ck(raw.ok is not None, "基准判定应可执行")

    # 期望值必须用**未舍入**的 r.a / r.c 反推: r.ratios 在 verdict 内部已 round(...,4),
    # 直接乘 ka/kc(=10) 会把舍入误差放大一个数量级, 用绝对容差必然误报。
    exp_area = raw.c["area"] / raw.a["area"] * ka / kc
    exp_peak = raw.c["peak"] / raw.a["peak"] * ka / kc
    ck(abs(nrm.ratios["area"] / exp_area - 1.0) < 1e-3,
       f"面积比应满足 (C/nc)/(A/na): 期望 {exp_area}, 实际 {nrm.ratios['area']}")
    ck(abs(nrm.ratios["peak"] / exp_peak - 1.0) < 1e-3,
       f"DIFF 峰值比应满足同样的归一关系: 期望 {exp_peak}, "
       f"实际 {nrm.ratios['peak']}")
    same = verdict(m, 0, 30 * TF, 45 * TF, 75 * TF, 1, norm_a=7.0, norm_c=7.0)
    ck(abs(same.ratios["area"] - raw.ratios["area"]) < 1e-9,
       "两段归一因子相同时结果必须与不归一化完全一致")


def test_nested_beichi() -> None:
    """区间套的"次级别确认"步骤: 只认时间**重叠**的笔, 不要求完全落在区间内。

    为什么这个口径重要: 高一级走势的端点就是那个极值点, 对应的小级别笔往往恰好
    结束在区间右端之外一点点。若用"完全包含"选取, 真正的那一笔背驰段会被排除,
    闸门退化成只在极少数情况下放行(实测通过率从 ~10% 掉到 ~6%)。
    """
    from chanbook.points import nested_beichi
    import random
    rnd = random.Random(23)
    px = [100.0]
    for _ in range(160):
        px.append(max(1.0, px[-1] * (1 + rnd.uniform(-0.012, 0.008))))
    m = MACD()
    for i, p in enumerate(px):
        m.update(i * TF, p)
    mm = KLineMerger()
    bb = BiBuilder()
    for i, p in enumerate(px):
        mm.push(i, i * TF, p * 1.004, p * 0.996)
        j = len(mm.cks) - 2
        if j >= 1:
            f = scan_fenxing_at(mm.cks, j)
            if f is not None:
                bb.push(f)
    ck(len(bb.final_bis) >= 4, f"合成序列应产生足够笔, 实际 {len(bb.final_bis)}")
    ctx = {"bis": bb.bis, "macd": m}
    downs = [b for b in bb.final_bis if b.direction < 0]
    a, c = downs[-2], downs[-1]
    # 区间取"恰好包含最新一笔的大部分" —— 右端刻意收在两个端点之间
    ts0 = (int(a.start_ts) + int(c.start_ts)) // 2
    ts1 = int(c.end_ts) - 1
    r = nested_beichi(m, ctx, ts0, ts1, -1)
    ck(r["ok"] or "why" in r, "应返回 ok 或说明原因")
    ck(isinstance(r["why"], str) and r["why"], "必须给出原因(无论成立与否)")
    # 区间完全错开 => 无论如何都不该成立
    bad = nested_beichi(m, ctx, int(c.end_ts) + TF * 100, int(c.end_ts) + TF * 200, -1)
    ck(not bad["ok"] and "不足两段" in bad["why"],
       f"区间内无同向段时应不成立, 实际 {bad}")
    # 只有一个向下笔的窄区间 => 也不成立
    narrow = nested_beichi(m, ctx, int(c.start_ts), int(c.start_ts) + 2 * TF, -1)
    ck(not narrow["ok"], "窄区间内同向段不足时不应成立")


def test_confirm_b1() -> None:
    """确认级别的加载与降级: 未加载确认级别时必须 fail-open(放行)。"""
    from chanbook.points import Point
    from chanbook.strategy import confirm_b1

    class _LS:
        confirm = None
        class op:  # noqa: N801
            class bib:  # noqa: N801
                bis: list = []

    p = Point("B1", "long", 100.0, 0, -1, 1, None, 100.0, [])
    ok, why = confirm_b1(_LS(), p, 0.02)
    ck(ok, "没有确认级别时应 fail-open 放行")
    ck("未加载" in why, f"应说明原因, 实际 {why}")


def test_nested_beichi_window() -> None:
    """`nested_beichi` 的窗口 = {笔 | end_ts > ts0 且 start_ts < ts1}(时间重叠)。

    改为二分定位后语义必须逐字不变, 尤其边界: 端点**恰好**等于 ts0 / ts1 的笔
    必须被排除(严格不等式)。同时验证"完全落在右侧"的笔不会混进窗口。
    """
    from chanbook.points import nested_beichi

    class _Bi:
        def __init__(self, s, e, d, lo, hi):
            self.start_ts, self.end_ts = s, e
            self.direction, self.low, self.high = d, lo, hi
            self.amp = abs(hi - lo)

    class _M:
        n = 100000

        @staticmethod
        def n_bars(a0, a1):
            return 5

        @staticmethod
        def _k(a0):
            return 1000.0 / (1.0 + a0)

        def summary(self, a0, a1, sign):
            k = self._k(a0)
            return {"area": k, "per": k, "peak": k}

        def hist_peak(self, a0, a1, sign):
            return self._k(a0)

    # 笔必须方向交替(与真实笔序列一致), 否则 verdict 会判"两段区间重叠"
    # ts 取 100 的整数倍, 端点可精确落在窗口边界上。
    #   下标: 0=上 1=下 2=上 3=下 4=上 5=下
    bis = [_Bi(0, 100, +1, 100.0, 110.0),
           _Bi(100, 200, -1, 90.0, 110.0),      # end_ts == ts0 时须被排除
           _Bi(200, 300, +1, 90.0, 105.0),
           _Bi(300, 400, -1, 80.0, 105.0),      # start_ts == ts1 时须被排除
           _Bi(400, 500, +1, 80.0, 100.0),
           _Bi(500, 600, -1, 70.0, 95.0)]       # 注意: 末笔是可延伸笔, 恒被丢弃

    # 窗口恰好只含 bis[1] (bis[0] 的 end==ts0、bis[2] 的 start==ts1 都被排除)
    r = nested_beichi(None, {"bis": bis, "macd": _M()}, 100, 200, -1, 0.02)
    ck(not r["ok"], "窗口内同向笔不足两段时必须 ok=False")
    ck("不足两段" in r["why"], f"未命中专用原因, 实际 {r['why']}")

    # 放宽到 [50, 400) => 窗口含 bis[0..3] => 两段同向(bis[1] vs bis[3]) => 走背驰判定
    r2 = nested_beichi(None, {"bis": bis, "macd": _M()}, 50, 400, -1, 0.02)
    ck(r2["ok"], f"两段同向重叠且力度衰减时应 ok=True, 实际 {r2}")
    ck(abs(r2["low"] - 80.0) < 1e-12,
       f"确认段低点应取窗口内最后一根向下笔的低点 80.0, 实际 {r2['low']}")

    # 空/退化输入不得抛异常
    ck(not nested_beichi(None, {"bis": [], "macd": _M()}, 0, 10, -1)["ok"],
       "空笔序列应安全返回 ok=False")
    ck(not nested_beichi(None, {"bis": [_Bi(0, 10, -1, 1.0, 2.0)], "macd": _M()},
                         0, 10, -1)["ok"], "单笔应安全返回 ok=False")
    # 末笔恒被丢弃: 只剩末笔重叠时必须同样返回 ok=False
    ck(not nested_beichi(None, {"bis": bis, "macd": _M()}, 500, 700, -1)["ok"],
       "窗口只覆盖可延伸的末笔时应返回 ok=False")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:                     # noqa: BLE001
            FAILS.append(f"{t.__name__} 抛异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n== 买卖点自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
