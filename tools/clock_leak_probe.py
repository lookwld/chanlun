"""量化「多级别各加自己的 tf_ms」这个时钟泄漏到底影响什么。

运行: python tools/clock_leak_probe.py [--market spot] [--l-tf 4h] [--s-tf 1h] [--limit N]

## 背景

`Level.advance(close_ts)` 的契约是「传当前时刻」, 每一级自己减去本周期长度。
`tools/backtest_b12_tf.py` 曾经写成:

    lv.advance(bar_ts + lv.tf_ms)       # 正确(操作级别就是它自己)
    tlv.advance(bar_ts + tlv.tf_ms)     # ❌ 趋势级别被当成"开盘于 bar_ts 的 4h 已收盘"

后者等于让趋势级别**多看一根自己的K线**(4h/1h 上 = 3 小时, 4h/2h 上 = 2 小时)。
修掉之后回测数字几乎不动 —— 本工具就是回答"**为什么几乎不动**"。

## 它测什么

沿**短周期**时钟逐根推进, 同时维护两个趋势级别:

  A(正确)  `trend_a.advance(bar_ts + 短周期tf)`
  B(泄漏)  `trend_b.advance(bar_ts + 长周期tf)`

然后只在**真的会用到趋势状态的那些 bar**上比较 —— 即操作级别当根扫出了 B2
候选(`strategy.DIR_GATE` 只对非 B1 的买点设闸门, B1 是 `None` 不受影响)。
输出:

  * 趋势状态在**全部 bar** 上的不一致率(泄漏的"名义"影响面);
  * 在 **B2 候选 bar** 上的不一致率(泄漏的"实际"影响面);
  * 如果按两种时钟各自过一遍闸门, **入场决定翻转**的次数。

三者从大到小, 说明为什么"名义上偷看 3 小时"最终只改掉了个位数成交。

⚠ 本文件是**唯一**带 `# noqa: TIMEBASE` 豁免的地方: 它必须在同一个作用域里同时写
"正确写法"与"错误写法"才能做对照。`tests/check_timebase.py` 会扫出这一处,
是有意保留的例外(不是漏网)。
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from chanbook import data as D                                  # noqa: E402
from chanbook.level import Level                                # noqa: E402
from chanbook.points import B1, B2, B3                          # noqa: E402
from chanbook.strategy import DIR_GATE, BookCfg                 # noqa: E402

GATED = [k for k, v in DIR_GATE.items() if v is not None]       # 受闸门约束的买点


def probe_one(sym: str, market: str, l_tf: str, s_tf: str) -> dict:
    skl = D.load(market, sym, s_tf)
    lkl = D.load(market, sym, l_tf)
    if skl is None or lkl is None:
        return {"sym": sym, "err": "无数据"}
    if skl.n < 500 or lkl.n < 200:
        return {"sym": sym, "err": f"根数不足 s={skl.n} l={lkl.n}"}
    cfg = BookCfg.legacy(market=market, kinds=(B1, B2, B3))
    op = Level("op", s_tf, "bi", scanner_kw=cfg.scanner_kw()).attach(skl)
    ta = Level("ta", l_tf, "seg").attach(lkl)       # 正确
    tb = Level("tb", l_tf, "seg").attach(lkl)       # 泄漏

    n_bar = n_diff = n_gated_bar = n_gated_diff = n_flip = 0
    for i in range(skl.n):
        bar_ts = int(skl.ts[i])
        now = bar_ts + op.tf_ms                     # 唯一正确的时刻
        op.advance(now)
        ta.advance(now)
        # ⚠ 这里**故意**复刻修复前的错误写法当对照 —— 本工具的全部意义就是量出它的影响面。
        # 所以要对 `tests/check_timebase.py` 的静态闸门显式豁免(见下方 noqa)。
        tb.advance(bar_ts + tb.tf_ms)               # noqa: TIMEBASE
        n_bar += 1
        if ta.state.kind != tb.state.kind:
            n_diff += 1
        pts = [p for p in op.scan_points() if p.kind in GATED]
        if not pts:
            continue
        n_gated_bar += 1
        ka, kb = ta.state.kind, tb.state.kind
        if ka != kb:
            n_gated_diff += 1
        for p in pts:
            allow = DIR_GATE.get(p.kind)
            if (ka in allow) != (kb in allow):
                n_flip += 1
                break
    return {"sym": sym, "err": "", "n_bar": n_bar, "n_diff": n_diff,
            "n_gated_bar": n_gated_bar, "n_gated_diff": n_gated_diff,
            "n_flip": n_flip}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot")
    ap.add_argument("--l-tf", default="4h")
    ap.add_argument("--s-tf", default="1h")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "out",
                                                  "clock_leak_probe.txt"))
    args = ap.parse_args()

    syms = D.symbols(args.market)
    if args.limit:
        syms = syms[:args.limit]
    rows, bad = [], []
    for s in syms:
        r = probe_one(s, args.market, args.l_tf, args.s_tf)
        (bad if r["err"] else rows).append(r)

    L: list[str] = []

    def emit(s: str = "") -> None:
        L.append(str(s))
        print(s, flush=True)

    tot_bar = sum(r["n_bar"] for r in rows)
    tot_diff = sum(r["n_diff"] for r in rows)
    tot_gb = sum(r["n_gated_bar"] for r in rows)
    tot_gd = sum(r["n_gated_diff"] for r in rows)
    tot_fl = sum(r["n_flip"] for r in rows)
    emit("=" * 88)
    emit(f"时钟泄漏影响面探针   {args.market}   长周期 {args.l_tf} / 短周期 {args.s_tf}")
    emit(f"标的 {len(rows)} 个(跳过 {len(bad)})   推进 {tot_bar:,} 根短周期K线")
    emit(f"受闸门约束的买点: {[k for k in GATED]}   (B1 的 DIR_GATE=None, 天然不受影响)")
    emit("=" * 88)
    emit("")
    emit("【三层影响面】")
    emit(f"  ① 名义  趋势状态在**全部 bar** 上不同:  {tot_diff:,} / {tot_bar:,}"
         f"  = {tot_diff / max(1, tot_bar):.3%}")
    emit(f"  ② 相关  在**受闸门买点出现的 bar** 上不同: {tot_gd:,} / {tot_gb:,}"
         f"  = {tot_gd / max(1, tot_gb):.4%}")
    emit(f"  ③ 实际  **入场决定翻转**: {tot_fl:,} 次"
         f"  = 每 {tot_bar / max(1, tot_fl):,.0f} 根短周期K线一次" if tot_fl else
         f"  ③ 实际  **入场决定翻转**: 0 次")
    emit("")
    emit("读法: ① 是\"偷看 3 小时的窗口有多宽\", ③ 才是\"真的改变了多少成交\"。")
    emit("      ② 与 ③ 之间的落差说明: 趋势状态只在**少数 bar** 上刚好处于切换边缘,")
    emit("      而买点落在那几个 bar 上的概率又很低。")
    emit("")
    if rows:
        emit("逐标的最差 10 个(按翻转次数, 再按 ① 排序):")
        emit(f"  {'标的':<20}{'① 全部bar不同':>15}{'② 闸门bar不同':>15}{'③ 翻转':>9}")
        emit("  " + "-" * 74)
        for r in sorted(rows, key=lambda x: (-x["n_flip"], -x["n_diff"]))[:10]:
            emit(f"  {r['sym']:<20}{r['n_diff'] / max(1, r['n_bar']):>14.2%}"
                 f"{r['n_gated_diff']:>15,}{r['n_flip']:>9,}")
    if bad:
        emit("")
        emit(f"跳过 {len(bad)}: " + ", ".join(f"{r['sym']}({r['err']})" for r in bad[:8]))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\n已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
