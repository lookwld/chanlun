"""共振模块自检 (`chanbook/resonance.py`)。

运行: python tests/check_resonance.py

为什么单独补这一份: 共振模块 2026-09-29 上线时**没有任何自检覆盖**, 结果带了一个
**偷看未来**的 bug —— `advance_to()` 里长周期用了 `bar_ts + 长周期tf`, 等于宣称
"开盘于 bar_ts 的那根长周期K线已经收盘"(4h/1h 上实测领先 3 小时)。它在回测里
表现为入场价系统性偏早, 而且不会报错、不会崩, 只能靠不变量检查抓出来。
所以下面第 2 项是**硬不变量**: 任何时候, 长周期已喂入的数据不得超出短周期当前时刻。

重点四件事:
  1. 组合规则 `combo_ok` 只放行 5 种(至少一侧买①);
  2. **没有未来函数**: 长周期只看到与短周期同一时刻已收盘的根; fresh_long 的
     "被知道"时刻 == 那根长周期K线收盘;
  3. 窗口不变量: 有效窗口内的买点必然 `now - known <= l_win × 长周期tf`, 且
     未被价格破坏;
  4. `scan_points()` 是边沿触发: 同一根 bar 第二次 `step()` 必须返回空 —— 这是
     消融对照能共用一次遍历的前提。
"""

from __future__ import annotations

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                            # noqa: E402
from chanbook.points import B1, B2, B3                     # noqa: E402
from chanbook.resonance import ResoScanner, combo_ok       # noqa: E402

N_PASS = 0
FAILS: list[str] = []
SKIP: list[str] = []

SYM = "BTCUSDT"
TAIL = 400                      # 只遍历末段, 自检要快


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


def _scanner() -> ResoScanner | None:
    try:
        rs = ResoScanner("spot", SYM).load()
    except Exception as e:                                 # noqa: BLE001
        SKIP.append(f"{SYM} 装载失败: {e}")
        return None
    if not rs.skl.n or not rs.lkl.n:
        SKIP.append(f"{SYM} 数据不足 (1h {rs.skl.n} / 4h {rs.lkl.n})")
        return None
    return rs


# ------------------------------------------------------------------ 1 组合规则
def t_combo() -> None:
    legal = [(B1, B1), (B1, B2), (B1, B3), (B2, B1), (B3, B1)]
    illegal = [(B2, B2), (B2, B3), (B3, B2), (B3, B3)]
    ck(all(combo_ok(a, b) for a, b in legal), "combo_ok: 5 种合法组合被误拒")
    ck(not any(combo_ok(a, b) for a, b in illegal),
       "combo_ok: 两侧都顺势的组合被误放行")
    ck(combo_ok(B1, "") is False or combo_ok(B1, B2),
       "combo_ok: 空 kind 行为未定义")


# ------------------------------------------------------------------ 2 未来函数
def t_no_lookahead() -> None:
    """长周期喂入的信息不得超过短周期当前时刻。"""
    rs = _scanner()
    if rs is None:
        return
    n = rs.skl.n
    bad_lead = bad_known = 0
    for i in range(max(0, n - TAIL), n):
        ts = int(rs.skl.ts[i])
        rs.advance_to(ts)
        now = ts + rs.short.tf_ms                       # 本步的"当前时刻"
        j = rs.long.cur - 1
        if j >= 0 and int(rs.lkl.ts[j]) + rs.long.tf_ms > now:
            bad_lead += 1
        k = rs.short.cur - 1
        if k >= 0 and int(rs.skl.ts[k]) + rs.short.tf_ms > now:
            bad_lead += 1
        for q in rs.pending():
            if q["known"] != q["ts"] + rs.long.tf_ms:
                bad_known += 1
    ck(bad_lead == 0, f"偷看未来: {bad_lead} 根 bar 上长/短周期信息超出当前时刻 "
                      f"(advance_to 必须用 bar_ts + **短周期**tf)")
    ck(bad_known == 0, f"known != ts + 长周期tf: {bad_known} 处")


# ------------------------------------------------------------------ 3 窗口不变量
def t_window_invariants() -> None:
    rs = _scanner()
    if rs is None:
        return
    n = rs.skl.n
    win = rs.l_win * rs.long.tf_ms
    bad_age = bad_inv = bad_fresh = 0
    n_pend = 0
    for i in range(max(0, n - TAIL), n):
        ts = int(rs.skl.ts[i])
        c = float(rs.skl.c[i])
        rs.advance_to(ts)
        ev = rs.step(ts, c)
        now = ts + rs.short.tf_ms
        for q in rs.pending():
            n_pend += 1
            if now - q["known"] > win:
                bad_age += 1
            if rs.invalidate and c < q["price"]:
                bad_inv += 1
        for q in ev["fresh_long"]:
            if q["known"] != now:
                bad_fresh += 1
    ck(bad_age == 0, f"窗口到期未剔除: {bad_age} 处")
    ck(bad_inv == 0, f"已跌破结构低点的买点仍在窗口内: {bad_inv} 处")
    ck(bad_fresh == 0, f"fresh_long 的被知道时刻 != 本步当前时刻: {bad_fresh} 处")
    ck(n_pend > 0, "末段 400 根内窗口始终为空 —— 自检失去意义(数据或口径异常)")


# ------------------------------------------------------------------ 4 边沿触发
def t_edge_trigger() -> None:
    rs = _scanner()
    if rs is None:
        return
    n = rs.skl.n
    dup_short = dup_long = n_hit = 0
    for i in range(max(0, n - TAIL), n):
        ts = int(rs.skl.ts[i])
        c = float(rs.skl.c[i])
        rs.advance_to(ts)
        ev1 = rs.step(ts, c)
        ev2 = rs.step(ts, c)                            # 同一根 bar 再问一次
        if ev1["short"]:
            n_hit += 1
        if ev2["short"]:
            dup_short += 1
        if ev2["fresh_long"]:
            dup_long += 1
    ck(dup_short == 0, f"同 bar 第二次 step() 重复报出短周期买点: {dup_short} 次"
                       f" —— 消融对照会因此塌成零信号")
    ck(dup_long == 0, f"同 bar 第二次 step() 重复报出长周期买点: {dup_long} 次")
    ck(n_hit > 0, "末段 400 根内一次短周期买点都没有 —— 自检失去意义")


# ------------------------------------------------------------------ 5 挑配合的买点
def t_pick_long_prefers_b1() -> None:
    rs = _scanner()
    if rs is None:
        return
    rs._active = [
        {"ts": 100, "known": 200, "kind": B2, "price": 1.0},
        {"ts": 300, "known": 400, "kind": B1, "price": 1.0},
        {"ts": 500, "known": 600, "kind": B3, "price": 1.0},
    ]
    q = rs._pick_long(B2)
    ck(q is not None and q["kind"] == B1, "_pick_long 应优先买①(逆势、必需项)")
    q = rs._pick_long(B1)
    ck(q is not None and q["kind"] == B1, "_pick_long(B1) 应仍取买①")
    # 只有两侧都顺势时无解
    rs._active = [{"ts": 100, "known": 200, "kind": B2, "price": 1.0}]
    ck(rs._pick_long(B2) is None, "B2+B2 不该配对")
    ck(rs._pick_long(B1) is not None, "B2+B1 应当配对")
    rs._active = []


# ------------------------------------------------------------------ 6 口径常量
def t_defaults() -> None:
    from chanbook.resonance import DEFAULT_L_WIN, RESO_TF
    ck(RESO_TF["spot"] == ("4h", "1h"), "现货共振对应为 4h+1h")
    ck(DEFAULT_L_WIN == 3, "默认窗口应为 3 根长周期K线")
    ck(D.TF_MS["4h"] == 4 * D.MS_HOUR, "4h 的毫秒长度不对")


def main() -> int:
    for t in (t_combo, t_no_lookahead, t_window_invariants, t_edge_trigger,
              t_pick_long_prefers_b1, t_defaults):
        try:
            t()
        except Exception as e:                             # noqa: BLE001
            FAILS.append(f"{t.__name__} 抛异常: {type(e).__name__}: {e}")
            traceback.print_exc()
    if SKIP:
        for s in SKIP:
            print("  ~ 跳过:", s)
    print(f"\n== 共振 自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
