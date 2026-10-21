"""B1 参数敏感性测试 —— MACD 快慢线参数 与 背驰容差 `beichi_tol`。

## 为什么要有这个工具

`chanbook` 的 B1 **完全不含均线** —— 判据只有 MACD 背驰(见 `points._b1`)。而
`DESIGN.md` §6.3 明确写了「MACD 参数固定 12/26/9(原文默认), **未做参数寻优**(避免
过拟合)」。于是"有没有更适合的参数"这个问题在项目里一直是**悬空**的: 没人测过。

本工具就是把这个"未测"变成"测过"。**它不负责找出最优参数**, 只负责回答:
**换成别的参数, 结论会不会翻?** 这两个问题是完全不同的。

## 防过拟合的三条纪律(先写死在代码里, 再看数据)

1. **只扫两个参数, 每参数只取离散几个点**, 绝不做网格寻优 —— 网格寻优必然能挑出
   一个"最好"的点, 那只是噪声的极大值。
2. **判定标准是平坦度与单调性, 不是最大值**。参数曲线若是"中间平、两端掉"的
   平台, 说明结论对参数不敏感(这是好消息: 说明 12/26/9 落在平台中段, 不是运气);
   若是孤峰(只有一个点好、两侧都掉), 一律判为噪声, **不采信**。
3. **三个口径一起看**: 单笔毛期望 / 每持仓日 / 盈亏比。只挑其中涨的那一个就是
   在挑选噪声。本项目已经吃过一次这个教训(`B3` 在 `l_win=3` 下 PF 2.69, 放宽到
   12 就塌到 1.11)。

## 与既有报告的关系

测试台 = `tools/backtest_b1_4h.py` 的 **`chan1_stop`** 口径(现货 4h B1, 只做多,
S1 一卖 + 灾难止损), 基准数字 **926 笔 / 胜率 18.8% / 单笔毛 +1.956% / PF 1.41**
(见 `out/b1_4h_spot.txt`)。本工具跑 `12/26/9` 那一行**必须复现**这组数字 ——
这是唯一能证明"参数注入路径没写错"的自检(复现不上就是代码有病, 不是参数有用)。

运行:
    python tools/macd_sweep.py                       # 两臂全跑, 约 3~4 分钟
    python tools/macd_sweep.py --arm macd            # 只跑 MACD 臂
    python tools/macd_sweep.py --limit 20            # 冒烟(⚠ 幸存者偏差, 结论不可用)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from backtest_b1_4h import (MARKET, MIN_BARS, TF, WARMUP,  # noqa: E402
                            ms_of, per_day, run_symbol)
from chanbook import data as D                            # noqa: E402
from chanbook.report import summarize                     # noqa: E402

EXIT_MODE = "chan1_stop"
FEE_ROUND = 0.0020            # 现货双边成本(手续费), 门槛同 backtest_b1_4h

# 两臂。每臂内第一项之后**参数单调变化**, 便于看"曲线形状"。
MACD_ARM: list[tuple[str, dict]] = [
    ("6/19/6   快", {"fast": 6, "slow": 19, "signal": 6}),
    ("8/17/9", {"fast": 8, "slow": 17, "signal": 9}),
    ("12/26/9  基准", {"fast": 12, "slow": 26, "signal": 9}),
    ("16/34/9  慢", {"fast": 16, "slow": 34, "signal": 9}),
    ("20/50/10 更慢", {"fast": 20, "slow": 50, "signal": 10}),
]

TOL_ARM: list[tuple[str, dict]] = [
    ("0.00  任何衰减都算", {"beichi_tol": 0.0}),
    ("0.02  基准", {"beichi_tol": 0.02}),
    ("0.05  要求更明显", {"beichi_tol": 0.05}),
    ("0.10  很严", {"beichi_tol": 0.10}),
]

BASE_MACD = {"fast": 12, "slow": 26, "signal": 9}
BASE_TOL = 0.02


def run_one(label: str, macd_kw: dict | None, scan_over: dict | None,
            syms: list[str], t0: int, t1: int, verbose: bool = True) -> dict:
    """跑一遍全池, 返回汇总 + 结构命中数。"""
    tr_all: list = []
    n_seen = 0
    n_reject = 0
    n_bad = 0
    t_a = time.time()
    for k, sym in enumerate(syms):
        tr, dg = run_symbol(sym, EXIT_MODE, t0, t1,
                            macd_kw=macd_kw, scan_over=scan_over)
        tr_all.extend(tr)
        if dg.err:
            n_bad += 1
        n_seen += int(dg.n_seen.get("B1", 0))
        n_reject += int(dg.n_blocked.get("B1", 0))
        if verbose and (k + 1) % 30 == 0:
            print(f"    [{label}] {k + 1}/{len(syms)} 结构B1={n_seen}",
                  file=sys.stderr, flush=True)
    s = summarize(tr_all, MARKET)
    s["_label"] = label
    s["_secs"] = time.time() - t_a
    s["_n_seen"] = n_seen
    s["_n_reject"] = n_reject
    s["_n_bad"] = n_bad
    # 尾部明细: 单笔毛期望是**均值**, 而均值可以被极少数大赢家单独抬起来。
    # ⚠ 记录 (entry_ts, pnl, hold_min, reason) 四元组而不是只记 pnl —— 因为
    # "剔掉最大 N 笔之后还剩多少"必须能重算持仓时长(每持仓日不能对子集做加性分解)。
    s["_rows"] = [(int(t.entry_ts), float(t.pnl_pct), int(t.hold_min), t.reason)
                  for t in tr_all]
    s["_pnls"] = sorted(r[1] for r in s["_rows"])
    s["_eod"] = [r[1] for r in s["_rows"] if r[3] == "end_of_data"]
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="macd,tol",
                    help="macd / tol / 两者都要(逗号分隔)")
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/macd_tol_sweep_spot.txt")
    args = ap.parse_args()

    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    arms = [a.strip() for a in args.arm.split(",") if a.strip()]
    t0, t1 = ms_of(args.start), ms_of(args.end, end=True)
    syms = D.symbols(MARKET)
    if args.limit:
        syms = syms[:args.limit]

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(str(s))
        print(s, flush=True)

    emit("=" * 108)
    emit(f"B1 参数敏感性  市场={MARKET}  操作级别={TF}  出口={EXIT_MODE}  "
         f"{args.start} ~ {args.end}  ({len(syms)} 个标的)")
    emit(f"生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}   "
         f"预热 {WARMUP} 根 / 最少 {MIN_BARS} 根 / 只做多 / 4h 由 1h 现场聚合")
    emit("=" * 108)
    emit("判据(先写死, 再看数): 看**曲线是否平坦/单调**, 不看峰值。")
    emit("  平坦平台 => 结论对参数不敏感(说明 12/26/9 落在平台中段, 不是运气好);")
    emit("  孤峰     => 判为噪声, 不采信。")
    emit("自检: 「12/26/9 基准」+「tol 0.02 基准」两行必须复现既有报告 "
         "out/b1_4h_spot.txt 的 chan1_stop 数字。")
    emit("")

    results: dict[str, list[dict]] = {}

    if "macd" in arms:
        emit(f"[{time.strftime('%H:%M:%S')}] MACD 臂 —— {len(MACD_ARM)} 个配置 ...")
        rows = []
        for label, kw in MACD_ARM:
            same = (kw == BASE_MACD)
            s = run_one(label, kw, {"beichi_tol": BASE_TOL}, syms, t0, t1)
            s["_base"] = same
            rows.append(s)
            print(f"    [{label}] 笔={s['n']} 单笔毛={s['exp_pct']:+.3%} "
                  f"PF={s['pf']:.2f}  ({s['_secs']:.0f}s)", flush=True)
        results["macd"] = rows
        emit("")
        emit(_table("MACD 快慢线参数", rows))
        emit(_tail_table(rows))
        emit(_year_table(rows))
        emit(_shape(rows, "MACD"))

    if "tol" in arms:
        emit("")
        emit(f"[{time.strftime('%H:%M:%S')}] 背驰容差臂 —— {len(TOL_ARM)} 个配置 ...")
        rows = []
        for label, kw in TOL_ARM:
            same = (kw.get("beichi_tol") == BASE_TOL)
            s = run_one(label, BASE_MACD, kw, syms, t0, t1)
            s["_base"] = same
            rows.append(s)
            print(f"    [tol {label}] 笔={s['n']} 单笔毛={s['exp_pct']:+.3%} "
                  f"PF={s['pf']:.2f}  ({s['_secs']:.0f}s)", flush=True)
        results["tol"] = rows
        emit("")
        emit(_table("背驰容差 beichi_tol(四判据的衰减门槛)", rows))
        emit(_tail_table(rows))
        emit(_shape(rows, "beichi_tol"))

    emit("")
    emit("=" * 108)
    emit("附: 均线口径说明")
    emit("  本项目的 B1 **不含任何均线**。旧引擎(chan-quant/app/strategy.py)用的是")
    emit("  **四条**均线 ma5>ma10>ma20>ma60 且 close>ma20 作前置闸门, 那会把一买"
         "(必然发生在")
    emit("  均线空头排列的下跌末端)整段排除 ⇒ 旧引擎没有一买。新引擎只保留"
         "方向闸门 DIR_GATE,")
    emit("  且 B1 的方向闸门 = None(不限)。均线在本项目里只剩一个用途: "
         "engine.btc_regime()")
    emit("  的 BTC-1h MA20/MA60 三段式行情分桶 —— **仅事后归因, 不参与判定**。")
    emit("=" * 108)

    path = args.out
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n报告已写入 {path}")
    return 0


def _tail_table(rows: list[dict]) -> str:
    """剔最大 N 笔 —— 直接问「这个结论是不是靠几笔撑起来的」。

    ⚠ 为什么**不用中位数**当判据: 本口径胜率只有 ~19%、盈亏比 ~6, 中位数**天然**是
    负的(它就是一笔典型止损), 所有配置都一样 ⇒ 没有区分力。真正有区分力的问题是:
    **把最大的 N 笔拿掉之后还能不能站住**。

    所以这里给的是逐级剔除曲线: 剔 0 / 1 / 3 / 10 笔后的「单笔毛」与「每持仓日」。
    "每持仓日"必须在子集上**重算**(Σ收益 / Σ持仓分钟), 不能对全样本的值做加减。
    """
    out = ["【逐级剔尾】把最赚的 N 笔拿掉后还剩多少 —— 判定「结论是否靠几笔撑起」",
           f"{'配置':<18}{'笔数':>7}{'剔0笔':>10}{'剔1笔':>10}{'剔3笔':>10}"
           f"{'剔10笔':>10}{'剔0每持日':>11}{'剔3每持日':>11}{'最大1笔':>10}",
           "-" * 108]
    for s in rows:
        rs = s.get("_rows") or []
        n = len(rs)
        if not n:
            out.append(f"{s['_label']:<18}{0:>7}{'  无成交':>10}")
            continue
        cells = []
        for k in (0, 1, 3, 10):
            sub = rs if k == 0 else _drop_top(rs, k)
            if not sub:
                cells.append((0.0, 0.0))
                continue
            m = sum(r[1] for r in sub) / len(sub)
            d = sum(r[2] for r in sub) / 1440.0 / len(sub)
            cells.append((m, (m / d) if d > 0 else 0.0))
        e0 = cells[0][0]
        out.append(
            f"{s['_label']:<18}{n:>7}{e0:>9.3%}{cells[1][0]:>9.3%}"
            f"{cells[2][0]:>9.3%}{cells[3][0]:>9.3%}"
            f"{cells[0][1]:>10.3%}{cells[2][1]:>10.3%}{max(r[1] for r in rs):>9.1%}")
    out.append("-" * 108)
    out.append("  读法: 「剔0笔」正、「剔3笔」就转负 ⇒ 该配置的全部期望来自 3 笔以内,")
    out.append("        无论均值多高都**不构成可复制的优势**(样本外几乎不可能重现)。")
    return "\n".join(out)


def _drop_top(rs: list, k: int) -> list:
    idx = sorted(range(len(rs)), key=lambda i: rs[i][1])[:-k] if k else range(len(rs))
    return [rs[i] for i in idx]


def _year_table(rows: list[dict]) -> str:
    """按年(按入场年份, UTC)拆单笔毛 —— 看「好」是稳定的还是某一年的。"""
    def yr(ts: int) -> int:
        return datetime.fromtimestamp(ts / 1000, timezone.utc).year
    years = sorted({yr(r[0]) for s in rows for r in (s.get("_rows") or [])})
    if not years:
        return ""
    out = ["【按年 · 单笔毛】括号内为笔数 —— 判定「好」是稳定的还是靠某一年",
           f"{'配置':<18}" + "".join(f"{y:>11}" for y in years), "-" * 108]
    for s in rows:
        rs = s.get("_rows") or []
        if not rs:
            continue
        cells = []
        for y in years:
            sub = [r[1] for r in rs if yr(r[0]) == y]
            cells.append("—" if not sub else f"{sum(sub) / len(sub):+.2%}({len(sub)})")
        out.append(f"{s['_label']:<18}" + "".join(f"{c:>11}" for c in cells))
    out.append("-" * 108)
    out.append("  读法: 若「更慢的 MACD」只在 2020/2024 这类单边大牛市里赢、其余年份")
    out.append("        都输, 那它抓的不是「信号更准」, 而是「长仓吃到了大牛市」。")
    return "\n".join(out)


def _table(title: str, rows: list[dict]) -> str:
    head = (f"【{title}】候选池口径(单笔毛 = 笔数加权; 门槛 = 双边 {FEE_ROUND:.2%})")
    col = (f"{'配置':<18}{'结构B1':>8}{'笔数':>7}{'胜率':>8}{'单笔毛':>10}"
           f"{'单笔净':>10}{'PF':>7}{'均持仓':>9}{'每持仓日':>10}{'盈亏比':>8}")
    sep = "-" * 108
    out = [head, col, sep]
    for s in rows:
        if not s.get("n"):
            out.append(f"{s['_label']:<18}{s.get('_n_seen', 0):>8}{0:>7}"
                       f"{'  无成交':>8}")
            continue
        mark = " *" if s.get("_base") else "  "
        pd_ = per_day(s)
        flag = "!" if s["exp_pct"] < FEE_ROUND else " "
        out.append(
            f"{s['_label']:<18}{s.get('_n_seen', 0):>8}{s['n']:>7}"
            f"{s['win_rate']:>7.1%}{s['exp_pct']:>9.3%}{flag}"
            f"{s['net_exp_pct']:>9.3%} {s['pf']:>6.2f}"
            f"{s['hold_min'] / 1440:>8.2f}天{pd_:>9.3%}{s['rr']:>8.2f}{mark}")
    out.append(sep)
    out.append("  * = 基准(12/26/9 与 tol 0.02)   ! = 单笔毛低于双边成本门槛")
    return "\n".join(out)


def _shape(rows: list[dict], name: str) -> str:
    """曲线形状判定 —— 本工具的真正产出。"""
    use = [s for s in rows if s.get("n")]
    if len(use) < 3:
        return f"【{name} 曲线形状】样本点不足, 不做判定。"
    e = [s["exp_pct"] for s in use]
    pf = [s["pf"] for s in use]
    pd_ = [per_day(s) for s in use]
    best = max(range(len(use)), key=lambda i: e[i])
    worst = min(range(len(use)), key=lambda i: e[i])
    base_i = next((i for i, s in enumerate(use) if s.get("_base")), 0)
    out = [f"【{name} 曲线形状】",
           f"  单笔毛极差 {max(e) - min(e):.3%}  "
           f"(最好 {use[best]['_label'].strip()} {e[best]:+.3%} / "
           f"最差 {use[worst]['_label'].strip()} {e[worst]:+.3%})",
           f"  PF 极差 {max(pf) - min(pf):.2f}   "
           f"每持仓日极差 {max(pd_) - min(pd_):.3%}",
           f"  基准点 = {use[base_i]['_label'].strip()} "
           f"(单笔毛 {e[base_i]:+.3%} / PF {pf[base_i]:.2f} / "
           f"每持仓日 {pd_[base_i]:+.3%})"]
    # 单调性: 相邻点同向变动的比例
    de = [e[i + 1] - e[i] for i in range(len(e) - 1)]
    same = sum(1 for d in de if d > 0)
    out.append(f"  相邻点单笔毛变动: " +
               " ".join(f"{d:+.3%}" for d in de) +
               f"   (同向上升 {same}/{len(de)})")
    if best in (0, len(use) - 1) and abs(e[best] - e[base_i]) > 0.005:
        out.append("  ⚠ 峰值落在**端点** —— 端点更可能是「这条边界外还有空间」的信号,")
        out.append("    也可能只是单调漂移(参数越极端、信号越少、尾部越肥)。"
                   "不要据此就改默认值。")
    else:
        out.append("  峰值不在端点 ⇒ 形状像平台/内部峰; 是否采信仍要看幅度"
                   "是否超过\"一个身位\"。")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
