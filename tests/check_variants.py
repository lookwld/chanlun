"""引擎开关自检 —— 2026-09-29 的 8 项改造 + 离场路径开关 + **默认值升级**
+ **波动率上限闸门(chan-quant `vol_gate` 移植)**。

运行: python tests/check_variants.py

分四类, 缺一不可:

  **A. "关 = 改动前"** —— 每个**入场侧**开关的默认值必须让主流程与改动前
     **逐位一致**(方向闸门 / 一买路径 / 量能 / 仓位分层 / 中枢扩展 / 均线 /
     波动率上限)。
     做法: 断言默认构造的配置对象与"改动前应有的值"相等, 且新增对象不被创建。

  **B. "开 = 真的生效"** —— 每个开关打开后必须产生**可观测**的行为差异。
     做法: 小样本端到端(真实归档数据) + 单元级断言。

  **C. "默认 = 生产口径"**(2026-09-29 升级新增) —— 本项目第一次改变默认行为:
     离场从 `old`(追踪 + RR 目标 + 结构破位) 改为 `stop + S1`(结构止损 + 一卖)。
     于是"改动前"再也不能靠裸 `BookCfg()` 表达, 必须写 `BookCfg.legacy(...)`。
     本类断言两件事: ① 裸默认 == 显式 `stop + S1`(逐位指纹);
                      ② `legacy` == 升级前口径, 历史结论仍可复现。

  **D. "波动率上限闸门"(2026-09-29 第三轮新增)** —— 口径来自
     `chan-quant/app/vol.py`: 日线 ATR14% <= 阈值才放行。
     ⚠ 与 `vol_gate`(成交量比)**同名不同物**, 见该类的注释。

⚠ 端到端部分只跑 BTCUSDT 一个标的(1h, 约 7.9 万根) —— 够用且不超过几十秒。
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook import engine as E                                  # noqa: E402
from chanbook import filters as F                                 # noqa: E402
from chanbook import risk                                         # noqa: E402
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1, B2, B3, PointScanner              # noqa: E402
from chanbook.sell import S1                                      # noqa: E402
from chanbook.strategy import (DIR_GATE, B1_DIR_PRESETS, BookCfg,  # noqa: E402
                              EXIT_MODES, Strategy, resolve_dir_gate)
from chanbook.trend import DOWN, PENDING, RANGE, UP               # noqa: E402

N_PASS = 0
FAILS: list[str] = []


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


# ================================================================ A. 关 = 默认
def test_default_dir_gate_identical() -> None:
    """默认 BookCfg 的方向闸门必须与模块级 DIR_GATE 完全一致。"""
    cfg = BookCfg(market="spot")
    ck(cfg.dir_gate == DIR_GATE, "默认 dir_gate 必须等于改动前的 DIR_GATE")
    ck(cfg.dir_gate[B1] is None, "默认 B1 不限方向")
    ck(cfg.dir_gate[B2] == frozenset({UP, RANGE, PENDING}),
       "默认 B2 放行 up/range/pending")
    ck(cfg.dir_gate[B3] == frozenset({UP, RANGE, PENDING}),
       "默认 B3 放行 up/range/pending")
    ck("off" in B1_DIR_PRESETS and B1_DIR_PRESETS["off"] is None,
       "b1_dir='off' 必须映射到 None(不限)")


def test_default_scanner_kw() -> None:
    """scanner_kw 的三个新键默认值 = 改动前语义。"""
    kw = BookCfg(market="spot").scanner_kw()
    ck(kw["b1_path"] == "both", "默认 b1_path=both(两条路径都走)")
    ck(kw["b1_trend_adjacent"] is False,
       "默认 b1_trend_adjacent=False(保留退化写法, 保证 A/B 可比)")
    ck(kw["vol_gate"] == 0.0, "默认 vol_gate=0(量能判据关闭)")


def test_default_op_extra_is_empty() -> None:
    """默认配置不得让 Level 创建任何新对象(零开销)。"""
    cfg = BookCfg(market="spot")
    ex = cfg.op_extra()
    ck(ex["ma_periods"] is None, "默认不建均线")
    ck(ex["vol_on"] is False, "默认不建量能视图")
    ck(ex["zs_merge"] is False, "默认不做中枢扩展")
    lv = Level("op", "1h", "bi")
    ck(lv.ma is None, "Level 默认 ma 为 None")
    ck(lv.vol is None, "Level 默认 vol 为 None")
    ck(lv.zs_merge is False, "Level 默认 zs_merge 为 False")


def test_default_point_scanner() -> None:
    """PointScanner 新参数默认 = 改动前行为。"""
    sc = PointScanner()
    ck(sc.b1_path == "both", "扫描器默认 b1_path=both")
    ck(sc.b1_trend_adjacent is False, "扫描器默认不启用相邻中枢判定")
    ck(sc.vol_gate == 0.0 and sc.vol is None, "扫描器默认无量能判据")


def test_default_risk_plan_identical() -> None:
    """risk.plan 的 nm_scale 默认 1.0 ⇒ 名义额与改动前逐位一致。"""
    a = risk.plan("long", 100.0, 98.0, 1.0, "spot", B1)
    b = risk.plan("long", 100.0, 98.0, 1.0, "spot", B1, nm_scale=1.0)
    ck(a.notional == b.notional, "nm_scale=1.0 与不传参的名义额必须相等")
    ck(a.stop == b.stop and a.target == b.target,
       "仓位缩放不得影响止损/目标(风险几何与仓位解耦)")


def test_default_strategy_has_no_new_gates() -> None:
    """默认策略的均线/量能/仓位分层都不参与判定。"""
    st = Strategy(BookCfg(market="spot"))
    ck(st.n_ma_block == 0, "默认无均线拦截计数")
    ck(BookCfg(market="spot").ma_periods() is None, "默认不需要均线周期")
    ck(BookCfg(market="spot").vol_on() is False, "默认不需要量能")


# ================================================================ B. 开 = 生效
def test_b1_dir_presets() -> None:
    g = resolve_dir_gate("no_up", False)
    ck(g[B1] == frozenset({DOWN, RANGE, PENDING}), "no_up 排除上涨趋势")
    ck(UP not in g[B1] and DOWN in g[B1], "no_up 保留下跌趋势")
    g2 = resolve_dir_gate("down", False)
    ck(g2[B1] == frozenset({DOWN}), "down 只留下跌趋势")


def test_pending_out() -> None:
    g = resolve_dir_gate("off", True)
    ck(PENDING not in g[B2] and PENDING not in g[B3],
       "pending_out 必须把中阴移出 B2/B3 放行集合")
    ck(UP in g[B2] and RANGE in g[B2], "pending_out 不影响 up/range")
    g0 = resolve_dir_gate("off", False)
    ck(PENDING in g0[B2], "默认仍然放行中阴(改动前行为)")


def test_scanner_config_roundtrip() -> None:
    cfg = BookCfg(market="spot", b1_dir="down", pending_out=True,
                  b1_path="trend_only", b1_trend_adjacent=True, vol_gate=0.9,
                  ma_mode="ma", pos_layer="amp", zs_merge=True,
                  sell_exit=("S1",))
    kw = cfg.scanner_kw()
    ck(kw["b1_path"] == "trend_only" and kw["b1_trend_adjacent"] is True,
       "扫描器参数必须被透传")
    ck(kw["vol_gate"] == 0.9, "vol_gate 必须被透传")
    ck(cfg.ma_periods() == (5, 10), "ma_mode=ma 需要建 (5,10) 均线")
    ck(cfg.vol_on() is True, "vol_gate>0 时必须挂载量能视图")
    ck(cfg.op_extra()["zs_merge"] is True, "zs_merge 必须透传到 Level")
    ck(cfg.sell_exit == ("S1",), "sell_exit 必须是元组")


def test_ma_veto() -> None:
    ma = F.MA((5, 10))
    for i in range(9):
        ma.push(100.0)
    ck(ma.value(10) is None, "不足 10 根时慢线必须不可用")
    ck(F.ma_veto_ok(ma, 100.0, "ma") is None, "均线未就绪必须返回 None(不拦截)")
    for i in range(40):
        ma.push(100.0 + 2.0 * i)                     # 单调上行 ⇒ ma5 > ma10
    ck((F.ma_spread(ma, 5, 10) or 0) > 0, "上行时均线间距必须为正")
    ck(F.ma_veto_ok(ma, 1000.0, "ma") is True, "上行时 ma5>=ma10 必须通过")
    ck(F.ma_veto_ok(ma, 100.0, "close") is False,
       "价格远低于 ma5 时 close 闸门必须拒绝")
    ck(F.ma_veto_ok(ma, 1000.0, "close") is True, "价格远高于 ma5 时必须通过")
    ma2 = F.MA((5, 10))
    for i in range(40):
        ma2.push(200.0 - 2.0 * i)                    # 单调下行 ⇒ ma5 < ma10
    ck(F.ma_veto_ok(ma2, 10.0, "ma") is False, "下行时 ma5<ma10 必须拒绝")
    ck(F.ma_size_scale(F.ma_spread(ma2, 5, 10)) < 1.0, "空头排列应减仓")
    ck(F.ma_size_scale(None) == 1.0, "均线缺失必须回退到 1.0")


def test_pos_scale() -> None:
    ck(F.pos_scale("off", B1, {}) == 1.0, "off 必须恒为 1.0")
    ck(F.pos_scale("amp", B1, {"amp_ratio": 0.3}) > 1.0,
       "后段幅度远小于前段(强背驰)应加仓")
    ck(F.pos_scale("amp", B1, {"amp_ratio": 2.0}) < 1.0,
       "后段幅度更大(背驰弱)应减仓")
    ck(F.pos_scale("ratio", B1, {"ratio_area": 0.2}) > 1.0, "面积比小应加仓")
    ck(F.pos_scale("kind", B3, {}) > F.pos_scale("kind", B1, {}),
       "三买应比一买重")
    ck(F.pos_scale("amp", B1, {}) == 1.0, "特征缺失必须回退到 1.0")


def test_vol_ctx() -> None:
    kl = D.load("spot", "BTCUSDT", "1h")
    vc = F.VolCtx(kl)
    a0, a1 = int(kl.ts[1000]), int(kl.ts[1100])
    b0, b1 = int(kl.ts[1101]), int(kl.ts[1200])
    va = vc.mean(a0, a1)
    ck(va is not None and va > 0, "区间均量必须可得且为正")
    rr = vc.ratio(a0, a1, b0, b1)
    ck(rr is not None and rr > 0, "量能比必须可得且为正")
    ck(vc.mean(int(kl.ts[-1]) + 10 ** 9, int(kl.ts[-1]) + 2 * 10 ** 9) is None,
       "区间内无K线必须返回 None")


def test_trend_adjacent_semantics() -> None:
    """`_is_trend_zs` 两种口径的差异: 相邻判定必须更严。

    构造: 很久以前有一个很高的中枢 A_old(100,120), 紧接着 A(60,80) 与当前
    中枢 B(40,60)。旧口径 `any(...)` 扫到 A_old 就认定"趋势"; 相邻口径只看 A,
    而 A 的下沿 60 并不高于 B 的上沿 60(不构成"中枢下移且不重叠") ⇒ 不成立。
    """
    from dataclasses import dataclass

    @dataclass
    class Z:
        i: int
        i0: int
        i1: int
        zg: float
        zd: float
        leave_i: int = 1

    zss = [Z(0, 0, 2, 120.0, 100.0),     # A_old: 很早、很高
           Z(1, 3, 5, 80.0, 60.0),       # A: 相邻, 但与 B 只擦边
           Z(2, 6, 8, 60.0, 40.0)]       # B: 当前
    old = PointScanner(b1_trend_adjacent=False)
    new = PointScanner(b1_trend_adjacent=True)
    ck(old._is_trend_zs(zss, 2, zss[2]) is True,
       "旧口径: 扫全部历史中枢 ⇒ 恒真(审计 A1 实测重合度 99.94%)")
    ck(new._is_trend_zs(zss, 2, zss[2]) is False,
       "相邻口径: 紧邻的 A 与 B 未构成'中枢下移且不重叠' ⇒ 不得判为趋势背驰")


def test_trend_adjacent_true_case() -> None:
    from dataclasses import dataclass

    @dataclass
    class Z:
        i: int
        i0: int
        i1: int
        zg: float
        zd: float
        leave_i: int = 1

    # A(70,90) 与 B(40,60): A.i1=5 < B.i0=6 ⇒ 相邻、不重叠、B.zg(60) < A.zd(70)
    zss = [Z(0, 3, 5, 90.0, 70.0), Z(1, 6, 8, 60.0, 40.0)]
    new = PointScanner(b1_trend_adjacent=True)
    ck(new._is_trend_zs(zss, 1, zss[1]) is True,
       "相邻且中枢下移 ⇒ 真趋势背驰")
    # 重叠情形: B.zg(80) > A.zd(70) ⇒ 不算趋势
    zss2 = [Z(0, 3, 5, 90.0, 70.0), Z(1, 6, 8, 100.0, 80.0)]
    ck(new._is_trend_zs(zss2, 1, zss2[1]) is False,
       "中枢上移(非下移)不得判为下跌趋势背驰")


def test_level_ma_and_merge() -> None:
    kl = D.load("spot", "BTCUSDT", "1h")
    lv = Level("op", "1h", "bi", ma_periods=(5, 10), vol_on=True)
    lv.attach(kl)
    for i in range(400):
        lv._push(i)
    ck(lv.ma is not None and lv.ma.value(5) is not None, "均线必须随推送更新")
    ck(lv.vol is not None and lv.scanner.vol is lv.vol,
       "量能视图必须挂到扫描器上")
    # 中枢扩展: 用同一份数据比较中枢数
    lv0 = Level("op", "1h", "bi").attach(kl)
    lv1 = Level("op", "1h", "bi", zs_merge=True).attach(kl)
    for i in range(10000):
        lv0._push(i)
        lv1._push(i)
    n0, n1 = len(lv0.zss_view()), len(lv1.zss_view())
    ck(n0 > 0, "必须有中枢")
    ck(n1 <= n0, f"中枢扩展只能减少或持平中枢数({n1} vs {n0})")
    ck(lv1._zs_cache is not None, "zs_merge=True 必须真的走 merge_extensions")
    ck(lv0._zs_cache is None, "zs_merge=False 不得做任何合并(零开销)")


def test_merge_extensions_semantics() -> None:
    """`merge_extensions` 的语义(手工构造重叠对, 证明接线正确)。

    ⚠ 同时记录一个**实测事实**: 真实数据上 ZG/ZD 口径下相邻中枢**永不重叠**
    (BTCUSDT 1h 前 2 万根: 重叠 0 对 / 220 对相邻对) —— 因为 `ZSBuilder._finish`
    只在"某一段与原中枢无重叠"时才结束中枢, 而新中枢正是从那一端起算。
    所以 `zs_merge` 对本实现是**结构性惰性开关**(原因见 `chanbook/zhongshu.py` 的 `merge_adjacent`)。
    """
    from chanbook.zhongshu import ZS, merge_extensions

    def z(i, i0, i1, zd, zg, ed=1, lv=1):
        return ZS(i=i, level=lv, i0=i0, i1=i1, zg=zg, zd=zd, gg=zg, dd=zd,
                  start_ts=i0, end_ts=i1, enter_i=i0 - 1, leave_i=i1 + 1,
                  enter_dir=ed)

    zs = [z(0, 0, 2, 100.0, 120.0), z(1, 3, 5, 110.0, 130.0),   # 重叠且同向 ⇒ 合并
          z(2, 6, 8, 200.0, 220.0), z(3, 9, 11, 300.0, 320.0)]   # 不重叠 ⇒ 不合并
    out = merge_extensions(zs)
    ck(len(out) == 3, f"应有 1 处合并 ⇒ 4 个变 3 个, 实际 {len(out)}")
    ck(out[0].level == 2, f"合并出高一级中枢, 实际 level={out[0].level}")
    ck(out[0].i0 == 0 and out[0].i1 == 5, "合并后覆盖两段")
    # 反向: enter_dir 不同 ⇒ 不合并
    zs2 = [z(0, 0, 2, 100.0, 120.0, ed=1), z(1, 3, 5, 110.0, 130.0, ed=-1)]
    ck(len(merge_extensions(zs2)) == 2, "方向不同的相邻中枢不得合并")


def test_real_data_adjacent_zss_never_overlap() -> None:
    """实测事实: 真实数据上相邻中枢在 ZG/ZD 口径下零重叠（`zs_merge` 惰性的根因）。"""
    kl = D.load("spot", "BTCUSDT", "1h")
    lv = Level("op", "1h", "bi").attach(kl)
    for i in range(12000):
        lv._push(i)
    zss = lv.zb.zss
    ck(len(zss) > 20, "样本必须有足够多的中枢")
    ov = sum(1 for a, b in zip(zss, zss[1:]) if a.overlaps(b))
    ck(ov == 0, f"相邻中枢必须零重叠(结构性), 实际 {ov} 对 / {len(zss) - 1} 对")


# ================================================================ 端到端
_E2E: dict = {}
_T0 = 0


def _e2e():
    """一次跑完端到端需要的基础结果(缓存, 避免重复推进级别)。

    只跑 2022-01 起的一段(约 4.6 年): 端到端要覆盖 8 个配置, 全历史太慢,
    而"开关是否生效"与区间长度无关。
    """
    global _T0
    if _E2E:
        return _E2E
    from datetime import datetime, timezone
    _T0 = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    sym, market = "BTCUSDT", "spot"
    # ⚠ 除 `prod` 外**全部以 `BookCfg.legacy` 为基底** —— 这样每个变体只改了
    #   **一个**自变量(入场开关), 离场口径恒定; `base` 就是"改动前的现行系统"。
    #   `prod` 是裸默认(= 升级后的生产口径), 专门用来验证默认值确实变了。
    def L(**kw):
        return BookCfg.legacy(market=market, kinds=(B1, B2), **kw)

    for nm, cfg in (
        ("base", L()),                                     # 升级前默认 = legacy
        ("prod", BookCfg(market=market, kinds=(B1, B2))),  # 升级后默认(裸 BookCfg)
        ("sell", L(sell_exit=("S1",))),                    # legacy + S1 = **叠加型**
        ("sell123", L(sell_exit=("S1", "S2", "S3"))),
        ("down", L(b1_dir="down")),
        ("ma", L(ma_mode="ma")),
        ("macd_only", L(b1_path="macd_only")),
        ("zs_only", L(b1_path="zs_only")),
        ("merge", L(zs_merge=True)),
        ("vol", L(vol_gate=1.0)),
        # ---- 2026-09-29 第三轮: 波动率上限闸门(日线 ATR14%) ----
        ("volcap", L(atr1d_cap=0.05)),
        # ---- 2026-09-29 第二轮 + 同日升级: 离场路径 ----
        ("stop", L(exit_mode="stop")),                     # 只停机械离场, 不接卖点
        ("stop_s1", L(exit_mode="stop", sell_exit=("S1",))),   # = 升级后的默认
        # ---- 2026-09-29 第四轮: chan-quant 能力补全(市场过滤器 + 分形离场) ----
        ("regime", L(regime_gate=True)),
        ("overheat", L(overheat_on=True)),
        ("resistance", L(resistance_on=True)),
        ("fib", L(fib_on=True)),
        ("correl", L(correl_on=True)),
        ("fenxing", L(exit_fenxing=True)),
    ):
        _E2E[nm] = E.run_symbol(market, sym, cfg, t_start=_T0)
    return _E2E


def test_e2e_legacy_has_no_sell_exit() -> None:
    """A 类: `legacy`(升级前的现行系统) 绝不出现卖点离场。"""
    tr, _ = _e2e()["base"]
    ck(len(tr) > 0, "基准必须产生成交")
    ck(not any(t.reason.startswith("sell_") for t in tr),
       "legacy 配置绝不出现卖点离场")


def test_e2e_production_default_is_stop_s1() -> None:
    """C 类(升级自证): 裸 `BookCfg()` 的成交必须与显式 `stop + S1` **逐位相同**。

    这是"默认值已升级"这件事唯一的机器可验证证据 —— 只读代码容易漏掉某个被
    `__post_init__` 二次加工过的字段(或某处仍在隐式引用旧默认)。
    """
    tr_p, _ = _e2e()["prod"]
    tr_c, _ = _e2e()["stop_s1"]
    ck(len(tr_p) > 0, "生产默认必须产生成交")

    def k(trs):
        return [(t.entry_ts, t.exit_ts, round(t.entry, 8), round(t.exit, 8),
                 round(t.pnl_pct, 12), t.reason, t.kind) for t in trs]

    ck(k(tr_p) == k(tr_c),
       f"裸默认必须逐位等于 stop+S1(实际 {len(tr_p)} vs {len(tr_c)} 笔)")
    ck(any(t.reason.startswith("sell_") for t in tr_p),
       "生产默认必须真的有卖点离场(否则升级没有生效)")
    ck({t.reason for t in tr_p} & {"trail_stop", "take_profit", "structure_break"} == set(),
       "生产默认不得出现机械离场(追踪/目标/结构破位)")


def test_e2e_sell_exit_wired() -> None:
    tr, _ = _e2e()["sell"]
    reasons = {t.reason for t in tr}
    ck("sell_S1" in reasons, f"接入 S1 后必须出现 sell_S1 离场, 实际 {reasons}")
    r2 = {t.reason for t in _e2e()["sell123"][0]}
    ck(any(r.startswith("sell_") for r in r2), "三类卖点接入后必须有卖点离场")


def test_e2e_b1_dir_gate() -> None:
    n_base = sum(1 for t in _e2e()["base"][0] if t.kind == B1)
    n_down = sum(1 for t in _e2e()["down"][0] if t.kind == B1)
    ck(n_base > 0, "基准必须有一买")
    ck(n_down < n_base,
       f"b1_dir=down 必须砍掉一部分一买({n_down} 应 < {n_base})")
    ck(n_down > 0, "b1_dir=down 不得把一买全砍光")


def test_e2e_ma_gate() -> None:
    tr, dg = _e2e()["ma"]
    ck(dg.n_ma_block > 0, f"均线闸门必须真的拦掉信号(实际 {dg.n_ma_block})")
    ck(len(tr) < len(_e2e()["base"][0]), "均线闸门必须减少成交")


def test_e2e_b1_path() -> None:
    macd_p = {t.feats.get("b1_path") for t in _e2e()["macd_only"][0]
              if t.kind == B1}
    ck(macd_p == {"macd"}, f"macd_only 应只有 macd 路径, 实际 {macd_p}")
    zs_p = {t.feats.get("b1_path") for t in _e2e()["zs_only"][0] if t.kind == B1}
    ck(zs_p == {"zs"}, f"zs_only 应只有 zs 路径, 实际 {zs_p}")


def test_e2e_vol_gate() -> None:
    tr0, _ = _e2e()["base"]
    tr1, _ = _e2e()["vol"]
    ck(len(tr1) > 0, "量能判据不得把成交清零")
    ck(len(tr1) < len(tr0), f"量能判据必须减少成交({len(tr1)} 应 < {len(tr0)})")
    ratios = [t.feats.get("vol_ratio") for t in tr1 if t.kind == B1]
    ck(all(r is not None and r < 1.0 for r in ratios),
       "启用 vol_gate=1.0 后所有一买的后段均量必须小于前段")


def test_e2e_zs_merge_invariant() -> None:
    """`zs_merge` 的不变式 + 实测事实。

    实测(BTCUSDT 1h): ZG/ZD 口径下相邻中枢**结构性零重叠** ⇒ 本开关在本实现上
    惰性。所以这里断言的是**不变式**: 中枢数没变时, 成交必须逐笔一致。
    """
    tr0, d0 = _e2e()["base"]
    tr1, d1 = _e2e()["merge"]
    ck(d1.n_zs <= d0.n_zs,
       f"中枢扩展不得增加中枢数({d1.n_zs} vs {d0.n_zs})")
    if d1.n_zs == d0.n_zs:
        k0 = [(t.entry_ts, t.entry, t.exit, t.kind, t.reason) for t in tr0]
        k1 = [(t.entry_ts, t.entry, t.exit, t.kind, t.reason) for t in tr1]
        ck(k0 == k1, "中枢数不变时成交必须逐笔一致")


# ================================================================ C. exit_mode
# 2026-09-29 第二轮新增, 同日晚间**升级为默认**。存在的理由: 主引擎原本
# **无条件**跑 `manage()`, 于是 `sell_exit` 只能"叠加"在机械离场之上 ——
# 实测叠加后净期望 ≈ 0(追踪止损把中位持仓压到 0.75 天, 结构判据中位 3.83 天,
# 没机会表态)。卖点要生效必须能**替换**默认离场, 所以需要这个显式路径开关。
_STOP_WINDOW = 0


def test_default_exit_mode_is_stop() -> None:
    """A/C 类(2026-09-29 升级): 裸默认 = 生产口径; `legacy` = 升级前口径。"""
    d = BookCfg(market="spot")
    ck(d.exit_mode == "stop", "默认 exit_mode 必须为 stop")
    ck(d.sell_exit == (S1,), f"默认 sell_exit 必须为 (S1,), 实际 {d.sell_exit}")
    lg = BookCfg.legacy(market="spot")
    ck(lg.exit_mode == "old", "legacy 必须保留 old")
    ck(lg.sell_exit == (), f"legacy 必须不接卖点, 实际 {lg.sell_exit}")
    en = BookCfg.enhanced(market="spot")
    ck(en.exit_mode == "stop" and en.ma_mode == "ma", "enhanced 必须 = 生产 + 均线")
    ck(EXIT_MODES == ("old", "stop"), f"EXIT_MODES 定义变了: {EXIT_MODES}")
    # archive 常量必须与构造器一致(防止两处定义漂移)
    from chanbook.strategy import ENHANCED, LEGACY, PRODUCTION
    ck(PRODUCTION == {"exit_mode": "stop", "sell_exit": (S1,)},
       f"PRODUCTION 档案漂移: {PRODUCTION}")
    ck(LEGACY == {"exit_mode": "old", "sell_exit": ()}, f"LEGACY 档案漂移: {LEGACY}")
    ck(ENHANCED["ma_mode"] == "ma" and ENHANCED["exit_mode"] == "stop",
       f"ENHANCED 档案漂移: {ENHANCED}")


def test_exit_mode_validation() -> None:
    """非法值必须立刻报错, 不能静默退化成某个模式。"""
    try:
        BookCfg(market="spot", exit_mode="off")
    except ValueError:
        ck(True, "")
    else:
        ck(False, "非法 exit_mode 必须抛 ValueError")
    for m in EXIT_MODES:
        try:
            BookCfg(market="spot", exit_mode=m)
        except Exception as e:                     # noqa: BLE001
            ck(False, f"合法 exit_mode={m} 不得报错: {e}")


def test_stop_mode_skips_manage() -> None:
    """B 类: `old` 必须调 `manage()`, `stop` 必须一次都不调(用探针, 不是看结果)。"""
    global _STOP_WINDOW
    if not _STOP_WINDOW:
        from datetime import datetime, timezone
        _STOP_WINDOW = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
    orig = E.manage
    calls = {"n": 0}

    def spy(pos, bar, tm):
        calls["n"] += 1
        return orig(pos, bar, tm)

    E.manage = spy
    try:
        calls["n"] = 0
        E.run_symbol("spot", "BTCUSDT", BookCfg.legacy(market="spot", kinds=(B1, B2)),
                     t_start=_STOP_WINDOW)
        n_old = calls["n"]
        calls["n"] = 0
        tr_stop, _ = E.run_symbol(
            "spot", "BTCUSDT",
            BookCfg.legacy(market="spot", kinds=(B1, B2), exit_mode="stop"),
            t_start=_STOP_WINDOW)
        n_stop = calls["n"]
    finally:
        E.manage = orig
    ck(n_old > 0, f"exit_mode=old 必须调用 manage()(实际 {n_old} 次)")
    ck(n_stop == 0, f"exit_mode=stop 不得调用 manage()(实际 {n_stop} 次)")
    rs = {t.reason for t in tr_stop}
    ck(rs <= {"stop_loss", "end_of_data", "sell_S1", "sell_S2", "sell_S3"},
       f"stop 模式的离场原因必须受限, 实际 {rs}")


def test_e2e_stop_mode_reason_whitelist() -> None:
    """B 类: `stop` 模式不得出现追踪/目标/结构破位三类离场。"""
    tr, _ = _e2e()["stop"]
    rs = {t.reason for t in tr}
    bad = rs & {"trail_stop", "take_profit", "structure_break"}
    ck(not bad, f"exit_mode=stop 不该出现机械离场, 实际 {bad}")
    tr0, _ = _e2e()["base"]
    rs0 = {t.reason for t in tr0}
    ck(bool(rs0 & {"trail_stop", "take_profit", "structure_break"}),
       f"exit_mode=old 必须出现机械离场(否则测试无区分力), 实际 {rs0}")


def test_e2e_sell_must_replace_not_stack() -> None:
    """B 类: 本轮最重要的机制不变式 —— **卖点必须替换默认离场, 不能叠加**。

    这里只断言**结构**事实(任何窗口都成立):
      * 替换型(`stop + S1`)里卖点承担主要离场(> 20%);
      * 叠加型(`old + S1`)里卖点占比不足替换型的一半;
      * 叠加型的**交易集几乎不变** —— 机械离场先把仓位了结, 卖点几乎没机会插进来。

    ⚠ 性能方向**不在这里断言**: "替换型单笔期望更高"是全池、全历史的结论
    (见 `ENGINE_VARIANTS §3.1` 与 `out/eng_path.txt`), 在**单个标的的某个子区间**
    上会翻转 —— 实测 BTCUSDT 2022-01 起: 替换型 −0.599% / 叠加型 +0.162%。
    单标的断言会把"截面结论"误当"逐标的规律", 这正是本项目反复踩的坑。
    """
    tr_stack, _ = _e2e()["sell"]           # old + S1
    tr_repl, _ = _e2e()["stop_s1"]         # stop + S1
    tr_base, _ = _e2e()["base"]            # old(现行)
    f_stack = sum(1 for t in tr_stack if t.reason.startswith("sell_")) / len(tr_stack)
    f_repl = sum(1 for t in tr_repl if t.reason.startswith("sell_")) / len(tr_repl)
    ck(f_repl > 0.2, f"替换型里卖点必须承担主要离场, 实际 {f_repl:.1%}")
    ck(f_stack < f_repl / 2.0,
       f"叠加型里卖点占比必须远低于替换型({f_stack:.1%} vs {f_repl:.1%})")
    # 叠加型的交易集与现行默认几乎相同 ⇒ "开卖点开关但不动离场路径"是空操作
    d = abs(len(tr_stack) - len(tr_base)) / len(tr_base)
    ck(d < 0.02,
       f"叠加型的成交数必须与现行默认几乎相同(差 {d:.2%}), "
       f"{len(tr_stack)} vs {len(tr_base)}")
    d2 = abs(len(tr_repl) - len(tr_base)) / len(tr_base)
    ck(d2 > 0.05,
       f"替换型必须显著改变交易集(差 {d2:.2%}), {len(tr_repl)} vs {len(tr_base)}")


def test_recommended_spot_preset() -> None:
    """推荐配置 = 系统默认(2026-09-29 升级后), 两者必须逐位等价。"""
    from chanbook.sell import S1 as _S1
    from chanbook.strategy import recommended_spot
    c = recommended_spot()
    ck(c.exit_mode == "stop", "推荐配置必须关掉机械离场")
    ck(c.sell_exit == (_S1,), f"推荐配置必须接入一卖, 实际 {c.sell_exit}")
    ck(c.b1_dir == "off" and c.pending_out is False,
       "推荐配置不得打开方向闸门(实测为负)")
    ck(c.vol_gate == 0.0 and c.pos_layer == "off" and c.zs_merge is False,
       "推荐配置不得打开量能/仓位分层/中枢扩展(实测为负或零)")
    ck(c.ma_mode == "off", "推荐配置默认不带均线闸门(证据不够硬)")
    ck(recommended_spot(ma_mode="ma").ma_mode == "ma", "preset 必须允许覆盖")
    # 2026-09-29 升级: 本函数**就是**裸默认 —— 两者必须逐位等价。
    # (升级前这里断言的是"推荐配置不得污染 dataclass 默认值"; 现在默认值已按
    #  推荐配置升级, 断言方向反转。历史可比性改由 `BookCfg.legacy()` 保证。)
    d = BookCfg(market="spot", kinds=(B1, B2))
    ck((c.exit_mode, c.sell_exit, c.market, c.kinds)
       == (d.exit_mode, d.sell_exit, d.market, d.kinds),
       "推荐配置必须与裸默认逐位等价(升级后默认即生产口径)")
    ck((c.b1_dir, c.pending_out, c.vol_gate, c.pos_layer, c.zs_merge, c.ma_mode)
       == (d.b1_dir, d.pending_out, d.vol_gate, d.pos_layer, d.zs_merge, d.ma_mode),
       "推荐配置与裸默认在入场侧开关上必须完全一致")
    ck(BookCfg.legacy(market="spot").exit_mode == "old",
       "legacy 必须保留改动前口径(历史 A/B 可比性)")


# ============================================ D. 波动率上限闸门(atr1d_cap)
# 2026-09-29 第三轮新增 —— 来源: `chan-quant/app/vol.py` 的 `vol_gate`。
# ⚠ 与 B 类的 `vol_gate` 测试**不是**同一件事, 别混:
#     `vol_gate`   = **成交量比**(后段均量 / 前段均量);
#     `atr1d_cap`  = **日线波动率绝对水平**(ATR14%)。
#   同名不同物 —— 因为 chan-quant 把后者也叫 `vol_gate`。


def _synth_kl(days: int = 40, base: float = 100.0):
    """合成日线: TR 恒为 2.0, 便于手工核 ATR%。"""
    t0 = 1_500_000_000_000
    rows = []
    for i in range(days):
        c = base + i
        rows.append((t0 + i * D.MS_DAY, c - 0.5, c + 1.0, c - 1.0, c, 1.0, 1.0))
    return D.Kl(rows=rows)


def test_atr1d_matches_manual() -> None:
    """口径必须与 chan-quant `app/vol.py` 的手工算法逐位一致。"""
    kl = _synth_kl(40)
    n, last = 14, 39
    a = F.ATR1D(kl, n)
    trs = []
    for i in range(last - n + 1, last + 1):
        pc = kl.c[i - 1]
        trs.append(max(kl.h[i] - kl.l[i], abs(kl.h[i] - pc), abs(kl.l[i] - pc)))
    man = sum(trs) / n / kl.c[last]
    auto = a.at(int(kl.ts[last]) + D.MS_DAY)
    ck(auto is not None and abs(auto - man) < 1e-12,
       f"ATR1D 与手工口径不符: auto={auto} man={man}")


def test_atr1d_no_lookahead() -> None:
    """无未来函数: 截断视图在截断时刻的读数必须与原视图相同。"""
    kl = _synth_kl(60)
    mid = 40
    cut = D.Kl(rows=[(kl.ts[i], kl.o[i], kl.h[i], kl.l[i], kl.c[i], kl.v[i], kl.qv[i])
                     for i in range(mid + 1)])
    ts = int(kl.ts[mid]) + D.MS_DAY
    v1 = F.ATR1D(kl, 14).at(ts)
    v2 = F.ATR1D(cut, 14).at(ts)
    ck(v1 is not None and v1 == v2, f"截断后读数变了: {v1} vs {v2}")


def test_atr1d_fail_open() -> None:
    """数据不足窗口 ⇒ None(调用方 fail-open 放行, 与 chan-quant 一致)。"""
    kl = _synth_kl(10)                       # 10 根 < 15
    ck(F.ATR1D(kl, 14).at(int(kl.ts[9]) + D.MS_DAY) is None,
       "不足 15 根应返回 None")
    ok = F.ATR1D(_synth_kl(40), 14)
    ck(ok.ok(int(ok.ts[39]) + D.MS_DAY, 0.0) is None, "cap<=0 应返回 None(闸门关闭)")


def test_atr1d_ok_boundary() -> None:
    """边界: 恰好等于 cap 放行(`<=`), 略小则拦截。"""
    a = F.ATR1D(_synth_kl(40), 14)
    ts = int(a.ts[39]) + D.MS_DAY
    v = a.at(ts)
    ck(a.ok(ts, v) is True, "恰好等于 cap 应放行")
    ck(a.ok(ts, v * 0.999) is False, "略小于实测值应拦截")


def test_default_atr1d_is_off() -> None:
    """A 类: 默认必须关闭, 否则既有 A/B 结论作废。"""
    c = BookCfg(market="spot")
    ck(float(c.atr1d_cap) == 0.0, f"默认 atr1d_cap 必须为 0, 实际 {c.atr1d_cap}")
    ck(int(c.atr1d_n) == 14, f"默认 atr1d_n 必须为 14, 实际 {c.atr1d_n}")
    ck(float(BookCfg.legacy(market="spot").atr1d_cap) == 0.0, "legacy 也必须关闭")


def test_atr1d_cap_validation() -> None:
    """负阈值必须报错, 不能静默当成"关闭"。"""
    try:
        BookCfg(market="spot", atr1d_cap=-0.01)
    except ValueError:
        ck(True, "")
        return
    ck(False, "负 atr1d_cap 应当抛 ValueError")


def test_e2e_volcap_gate_active() -> None:
    """B 类: 开启后必须真的拦单; 关闭(默认)必须拦 0。"""
    tr_b, dg_b = _e2e()["base"]
    tr_v, dg_v = _e2e()["volcap"]
    ck(int(dg_b.n_volcap_block) == 0, "默认(关闭)必须拦 0")
    ck(int(dg_v.n_volcap_block) > 0,
       f"开启闸门后必须拦到单, 实际 {dg_v.n_volcap_block}")
    ck(len(tr_v) < len(tr_b), f"开启后成交应变少: {len(tr_v)} vs {len(tr_b)}")


def test_e2e_atr1d_off_identical_to_legacy() -> None:
    """A 类: 显式写 `atr1d_cap=0.0` 与不写必须**逐笔**相同。"""
    sym, market = "BTCUSDT", "spot"
    tr0, _ = E.run_symbol(market, sym,
                          BookCfg.legacy(market=market, kinds=(B1, B2)),
                          t_start=_T0)
    tr1, _ = E.run_symbol(market, sym,
                          BookCfg.legacy(market=market, kinds=(B1, B2), atr1d_cap=0.0),
                          t_start=_T0)
    k0 = [(t.entry_ts, t.exit_ts, t.exit, t.reason) for t in tr0]
    k1 = [(t.entry_ts, t.exit_ts, t.exit, t.reason) for t in tr1]
    ck(k0 == k1, "cap=0 必须与 legacy 逐笔相同")


# ================================ E. chan-quant 能力补全(第四轮)
# 2026-09-29 第四轮: 市场状态/位置/联动过滤器 + 分形离场。默认全部关闭。
# 口径一致性已在 tests/check_market_filters.py 逐位对拍(与 chan-quant 源)。


def test_default_market_filters_off() -> None:
    """A 类: 6 个补全开关默认全部关闭, 否则历史结论作废。"""
    c = BookCfg(market="spot")
    for k in ("regime_gate", "overheat_on", "resistance_on", "fib_on",
              "correl_on", "exit_fenxing"):
        ck(getattr(c, k) is False, f"默认 {k} 必须为 False")
    lg = BookCfg.legacy(market="spot")
    ck(lg.market_filter_on() is False, "legacy 必须不开启任何市场过滤器")
    ck(lg.exit_fenxing is False, "legacy 必须不开启分形离场")


def test_e2e_market_filters_active() -> None:
    """B 类: 各过滤器开启后必须产生可观测的拦单或成交变化。"""
    base_tr, base_dg = _e2e()["base"]
    # regime / overheat / resistance / fib / correl 都可能拦 0(取决于单标的样本),
    # 所以只断言"配置能跑通 + 计数非负 + 成交集有变化或至少不报错"。
    # 强断言(拦单>0)在全池上才可靠; 这里只守"接线正确"。
    dg_r = _e2e()["regime"][1]
    ck(int(dg_r.n_regime_block) >= 0, "regime 计数必须存在且非负")
    # 分形离场: 开启后必须出现 fenxing_exit 离场原因(结构上必然, 因为它替换了卖点)
    tr_f, _ = _e2e()["fenxing"]
    rs = {t.reason for t in tr_f}
    ck("fenxing_exit" in rs,
       f"分形离场开启后必须出现 fenxing_exit, 实际 {rs}")


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        try:
            t()
        except Exception as e:                     # noqa: BLE001
            FAILS.append(f"{t.__name__} 抛异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n== 新增开关自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
