"""现货**多级别共振**入场 + 「结构止损 + 一卖 S1」离场 回测。

用户口径(2026-09-29):
> 对现货所有品种, 长周期与短周期共振即信号 —— 必有一侧是买①, 另一侧可以是 ①②③。

共振定义见 `chanbook/resonance.py`。本工具回答三件事:

1. **共振这套信号在历史上值多少钱**(单笔毛期望 / 每持仓日 / 折算年化 / 门槛);
2. **"共振"这个条件本身有没有增量** —— 同一根 bar 上并行跑三种入场口径:
   `reso`(共振) / `s_only`(只做短周期) / `l_only`(只做长周期), 其余完全一致。
   这是本轮的核心对照: 如果共振不如单边, 那"共振"就只是把信号变少而已。
3. **暴露够不够** —— 项目迄今唯一没被解决的瓶颈(在仓占比)。

为什么三种入场口径共用一次遍历: `Level.scan_points()` 是边沿触发 + 去重的,
同一根 bar 第二次调用必然返回空 ⇒ 必须共用 `ResoScanner.step()` 的原料,
否则后两种口径恒为零信号(见 `resonance.step` 的说明)。顺带把耗时压到 1/3。

复用: 统计口径 / 池基准 / 随机入场对照全部直接 import `tools/backtest_b12_tf.py`
的函数(同一份实现, 避免两处漂移); `chanbook/*.py` 既有路径一行未改。

运行::

  python tools/backtest_resonance.py --market spot --jobs 10 \\
      --out out/resonance_bt_spot.txt
  python tools/backtest_resonance.py --limit 8 --l-win 3 --out /tmp/smoke.txt   # 冒烟
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from chanbook import data as D                                    # noqa: E402
from chanbook import report as R                                  # noqa: E402
from chanbook import risk                                         # noqa: E402
from chanbook.engine import (Diag, Trade, _close, _regime_at,     # noqa: E402
                             manage, open_pos, struct_line)
from chanbook.points import B1, B2, B3                            # noqa: E402
from chanbook.resonance import RESO_TF, ResoScanner               # noqa: E402
from chanbook.sell import S1, S2, S3, SellScanner                 # noqa: E402

from backtest_b12_tf import (EXIT_MODES, MIN_BARS, SELL_KINDS,    # noqa: E402
                             WARMUP, _flush, _mp_ctrl, _sell_hit,
                             _sell_kw, benchmark, level_schedule,
                             ms_of, per_day, pool_index, risk_block,
                             seeds_pass)

ENTRY_MODES = {
    "reso": "长/短两级共振(用户口径)",
    "s_only": "只做短周期买点(**消融对照**)",
    "l_only": "只做长周期买点(**消融对照**)",
}
L_MIN_BARS = 200            # 长周期最少根数(与 level.MIN_KLINES["trend"] 一致)


# ------------------------------------------------------------------ 单标的
def run_symbol(sym: str, market: str, l_tf: str, s_tf: str, l_win: int,
               invalidate: bool, entry_modes: tuple, exit_mode: str,
               risk_kind: str, t_start: int = 0, t_end: int = 0):
    """三种入场口径共用一次级别遍历, 返回 (trades, diag, struct_counts)。"""
    rs = ResoScanner(market, sym, (l_tf, s_tf), l_win, invalidate).load()
    trades = {m: [] for m in entry_modes}
    diags = {m: Diag(sym=sym) for m in entry_modes}
    counts: dict = {"L": {}, "S": {}}
    if rs.skl is None or rs.lkl is None:
        for m in entry_modes:
            diags[m].err = "无数据"
        return trades, diags, counts
    if not rs.ok():
        e = f"样本不足 1h={rs.skl.n} {l_tf}={rs.lkl.n}"
        for m in entry_modes:
            diags[m].err = e
        return trades, diags, counts

    skl = rs.skl                     # 短周期 K线(数据)
    lv = rs.short                    # 短周期级别(Level, 用于结构/出口)
    sscan = SellScanner(**_sell_kw(exit_mode))
    ks = SELL_KINDS[exit_mode]
    tm = risk.trail_mult_of(market)
    use_stop = exit_mode.endswith("_stop")
    ratio = max(1, rs.long.tf_ms // rs.short.tf_ms)
    start = max(WARMUP, L_MIN_BARS * ratio, MIN_BARS)
    if t_start:
        start = max(start, bisect.bisect_right(skl.ts, t_start - 1))
    pos = {m: None for m in entry_modes}
    for m in entry_modes:
        diags[m].n_op_bars = max(0, skl.n - start)

    for i in range(start, skl.n):
        bar_ts = int(skl.ts[i])
        if t_end and bar_ts > t_end:
            break
        rs.advance_to(bar_ts)
        c = float(skl.c[i])
        bar = {"ts": bar_ts, "open": skl.o[i], "high": skl.h[i],
               "low": skl.l[i], "close": c}
        # ---- 1) 管理已有持仓(每根 bar 先离场, 平掉当根不再开新仓, 与引擎一致) ----
        closed_now = set()
        for m in entry_modes:
            if pos[m] is None:
                continue
            p0 = pos[m]
            closed = None
            if exit_mode == "old":
                p0, closed = manage(p0, bar, tm)
                pos[m] = p0
                if closed is None:
                    line = struct_line(lv.bib.bis, p0["entry_ts"], 1)
                    if line is not None and c < line:
                        closed = _close(p0, c, bar_ts, "structure_break")
                        pos[m] = None
            else:
                h = _sell_hit(sscan, lv, ks)
                if h is not None:
                    closed = _close(p0, c, bar_ts, "sell_" + h.kind)
                    pos[m] = None
                elif use_stop and bar["low"] <= p0["stop"]:
                    closed = _close(p0, p0["stop"], bar_ts, "stop_loss")
                    pos[m] = None
            if closed is not None:
                trades[m].append(closed)
                closed_now.add(m)

        # ---- 2) 本 bar 的原始事件(三种口径共用) ----
        ev = rs.step(bar_ts, c)
        for q in ev["fresh_long"]:
            counts["L"][q["kind"]] = counts["L"].get(q["kind"], 0) + 1
        for p in ev["short"]:
            counts["S"][p.kind] = counts["S"].get(p.kind, 0) + 1

        # ---- 3) 找新信号 ----
        a = lv.atr()
        for m in entry_modes:
            if pos[m] is not None or m in closed_now:
                continue
            hits = rs.decide(bar_ts, c, ev, m)
            if not hits:
                continue
            h = hits[0]
            diags[m].n_seen[h.combo] = diags[m].n_seen.get(h.combo, 0) + 1
            rk = {"auto": h.risk_kind, "b1": B1, "short": h.risk_kind,
                  "long": (h.l_kind or h.s_kind)}.get(risk_kind, h.risk_kind)
            pl = risk.plan("long", c, h.struct_price, a, market, rk)
            if not pl.ok:
                diags[m].n_reject += 1
                diags[m].n_blocked[h.combo] = diags[m].n_blocked.get(h.combo, 0) + 1
                continue
            feats = {"l_kind": h.l_kind, "s_kind": h.s_kind, "mode": m,
                     "l_tf": h.l_tf, "s_tf": h.s_tf, "combo": h.combo,
                     "l_trend": h.l_trend, "s_trend": h.s_trend,
                     "l_lag_h": (bar_ts - h.l_ts) / D.MS_HOUR if h.l_ts else 0.0,
                     "l_beichi": h.l_beichi, "s_beichi": h.s_beichi,
                     "struct_pct": ((c - h.struct_price) / c
                                    if h.struct_price and c else 0.0)}
            sig = SimpleNamespace(
                sym=sym, side="long", kind=h.combo, stop=pl.stop,
                target=pl.target, atr=pl.atr, risk_pct=pl.risk_pct,
                notional=pl.notional, trend=f"{h.l_trend}/{h.s_trend}",
                counter=int(h.l_kind == B1 or h.s_kind == B1),
                confirm="", feats=feats)
            diags[m].n_signals += 1
            pos[m] = open_pos(sig, _regime_at(market, bar_ts + lv.tf_ms),
                              bar_ts, c)

    last = skl.n - 1
    for m in entry_modes:
        if pos[m] is not None:
            trades[m].append(_close(pos[m], skl.c[last], int(skl.ts[last]),
                                    "end_of_data"))
        sn = lv.snapshot()
        diags[m].n_bi, diags[m].n_seg, diags[m].n_zs = sn.n_bi, sn.n_seg, sn.n_zs
        diags[m].trend_at_end = lv.state.kind
    return trades, diags, counts


def _mp_run(task):
    trades, diags, counts = run_symbol(*task)
    return task[0], trades, diags, counts


def run_all(syms, market, l_tf, s_tf, l_win, invalidate, entry_modes,
            exit_mode, risk_kind, t0, t1, jobs, label=""):
    tasks = [(s, market, l_tf, s_tf, l_win, invalidate, entry_modes,
              exit_mode, risk_kind, t0, t1) for s in syms]
    tr: dict = {m: [] for m in entry_modes}
    dg: dict = {m: [] for m in entry_modes}
    ct: dict = {"L": {}, "S": {}}

    def bump(dst, src):
        for k, v in src.items():
            dst[k] = dst.get(k, 0) + v

    def take(sym, t_, d_, c_):
        for m in entry_modes:
            tr[m].extend(t_[m])
            dg[m].append(d_[m])
        bump(ct["L"], c_["L"])
        bump(ct["S"], c_["S"])

    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(int(jobs)) as pool:
            for k, (sym, t_, d_, c_) in enumerate(
                    pool.imap_unordered(_mp_run, tasks, chunksize=1)):
                take(sym, t_, d_, c_)
                if (k + 1) % 25 == 0:
                    print(f"  {label}{k + 1}/{len(syms)} 笔={len(tr[entry_modes[0]])}",
                          file=sys.stderr, flush=True)
    else:
        for k, t in enumerate(tasks):
            sym, t_, d_, c_ = _mp_run(t)
            take(sym, t_, d_, c_)
            if (k + 1) % 25 == 0:
                print(f"  {label}{k + 1}/{len(syms)}", file=sys.stderr, flush=True)
    for m in entry_modes:
        dg[m].sort(key=lambda d: d.sym)
    return tr, dg, ct


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=["spot", "um"])
    ap.add_argument("--l-tf", default="")
    ap.add_argument("--s-tf", default="")
    ap.add_argument("--l-win", type=int, default=3)
    ap.add_argument("--no-invalidate", action="store_true")
    ap.add_argument("--entry", default="reso,s_only,l_only")
    ap.add_argument("--modes", default="chan1_stop")
    ap.add_argument("--risk-kind", default="auto",
                    choices=["auto", "b1", "short", "long"])
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--control", default="reso",
                    help="随机入场对照挂在哪个入场口径上('' = 关闭)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="out/resonance_bt_spot.txt")
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    market = args.market
    l_tf, s_tf = RESO_TF[market]
    if args.l_tf:
        l_tf = args.l_tf
    if args.s_tf:
        s_tf = args.s_tf
    entries = tuple(m.strip() for m in args.entry.split(",") if m.strip())
    bad = [m for m in entries if m not in ENTRY_MODES]
    if bad:
        print(f"未知入场口径 {bad}; 可用 {list(ENTRY_MODES)}")
        return 2
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in EXIT_MODES]
    if bad:
        print(f"未知离场口径 {bad}; 可用 {list(EXIT_MODES)}")
        return 2
    t0, t1 = ms_of(args.start), ms_of(args.end, end=True)
    syms = D.symbols(market)
    if args.limit:
        syms = syms[:args.limit]

    thr = 0.0012 if market == "um" else 0.0020
    bmin = int(round(D.TF_MS[s_tf] / D.MS_MIN))
    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(str(s))
        print(s, flush=True)

    emit(f"# chanbook 多级别共振回测  {l_tf}(长) + {s_tf}(短)  "
         f"{market}  {args.start} ~ {args.end}  ({len(syms)} 个标的)")
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"# 口径: 共振=两级同时出买点且至少一侧买① · 长周期买点在 {args.l_win} 根 "
         f"{l_tf}内有效({'价格破位即作废' if not args.no_invalidate else '只看时间'}) · "
         f"只做多 · 出口在 {s_tf} · 预热 {WARMUP} 根 · 最少 {MIN_BARS} 根 · "
         f"多进程 {args.jobs}")
    emit("")

    res: dict = {}
    diags: dict = {}
    counts: dict = {}
    for mode in modes:
        emit(f"[{time.strftime('%H:%M:%S')}] 遍历级别并跑 "
             f"{'/'.join(entries)} × {mode} ...")
        ta = time.time()
        tr, dg, ct = run_all(syms, market, l_tf, s_tf, args.l_win,
                             not args.no_invalidate, entries, mode,
                             args.risk_kind, t0, t1, args.jobs)
        for m in entries:
            res[(m, mode)] = tr[m]
            diags[(m, mode)] = dg[m]
        counts[mode] = ct
        emit(f"# 耗时 {time.time() - ta:.0f}s")
        emit("")
        if args.cache_dir:
            for m in entries:
                try:
                    R.save_trades(os.path.join(
                        args.cache_dir, f"{market}_{m}_{mode}.json"),
                        tr[m], dg[m])
                except Exception as e:                            # noqa: BLE001
                    print(f"缓存失败: {e}")

    # ---- 主表: 入场口径 × 离场口径 ----
    emit("=" * 108)
    emit("主表(候选池口径; 单笔期望 = 笔数加权毛期望; 「每持仓日」= 单笔毛期望/均持仓天数)")
    emit(f"{'入场口径':>8} {'离场':>13} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} "
         f"{'单笔净':>9} {'盈亏比':>7} {'PF':>6} {'均持仓':>9} {'每持仓日':>10}")
    for m in entries:
        for mode in modes:
            s = R.summarize(res[(m, mode)], market)
            if not s["n"]:
                emit(f"{m:>8} {mode:>13} {0:>6}   (无成交)")
                continue
            emit(f"{m:>8} {mode:>13} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
                 f"{s['exp_pct'] * 100:>+8.3f}% {s['net_exp_pct'] * 100:>+8.3f}% "
                 f"{s['rr']:>7.2f} {s['pf']:>6.2f} "
                 f"{s['hold_min'] / 1440:>7.2f}天 "
                 f"{per_day(s) * 100:>+9.3f}%")
    emit("")
    emit(f"门槛: 单笔**毛**期望 > {thr * 100:.2f}%"
         f"({'合约双边 0.10%' if market == 'um' else '现货双边 0.20%'} + 滑点余量)")
    emit("⚠ 「共振」是否有增量, 看 reso 那一行有没有跑赢 s_only / l_only 两行; "
         "只看 reso 自己好不好是回答不了这个问题的。")
    emit("")

    # ---- 结构层: 两侧买点的原始命中量 ----
    ct = counts[modes[0]]
    emit("结构层(全池原始命中, 与入场口径无关):")
    emit(f"  * 长周期({l_tf}) 买点: "
         + "  ".join(f"{k}={v:,}" for k, v in sorted(ct["L"].items()))
         + f"   合计 {sum(ct['L'].values()):,}")
    emit(f"  * 短周期({s_tf}) 买点: "
         + "  ".join(f"{k}={v:,}" for k, v in sorted(ct["S"].items()))
         + f"   合计 {sum(ct['S'].values()):,}")
    emit("")

    # ---- 共振组合分布 + 消融对照 ----
    for mode in modes:
        tr = res[("reso", mode)]
        if not tr:
            continue
        emit(f"【reso × {mode}】按组合拆分(共振的 5 种合法组合)")
        emit(f"{'组合':>8} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} {'PF':>6} "
             f"{'均持仓':>9} {'每持仓日':>10}")
        for combo in ("B1_B1", "B1_B2", "B1_B3", "B2_B1", "B3_B1"):
            sub = [t for t in tr if t.kind == combo]
            s = R.summarize(sub, market)
            if not s["n"]:
                emit(f"{combo:>8} {0:>6}   (无成交)")
                continue
            emit(f"{combo:>8} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
                 f"{s['exp_pct'] * 100:>+8.3f}% {s['pf']:>6.2f} "
                 f"{s['hold_min'] / 1440:>7.2f}天 {per_day(s) * 100:>+9.3f}%")
        emit("")

    # ---- 消融汇总 ----
    emit("=" * 108)
    emit("共振的**增量**(同一根 bar、同一套出口, 只换入场条件)")
    emit(f"{'入场口径':>8} {'离场':>13} {'笔数':>6} {'单笔毛':>9} {'每持仓日':>10} "
         f"{'在仓':>8} {'折算年化(线性)':>16}")
    for m in entries:
        for mode in modes:
            tr = res[(m, mode)]
            s = R.summarize(tr, market)
            used = sum(t.hold_min for t in tr)
            avail = sum(d.n_op_bars for d in diags[(m, mode)] if not d.err) * bmin
            share = (used / avail) if avail else 0.0
            fold = per_day(s) * 365 * share if s["n"] else 0.0
            emit(f"{m:>8} {mode:>13} {s['n']:>6} {s['exp_pct'] * 100:>+8.3f}% "
                 f"{per_day(s) * 100:>+9.3f}% {share * 100:>7.2f}% "
                 f"{fold * 100:>+15.1f}%")
    emit("")
    emit(f"仓位占用: 在仓占比 = Σ持仓分钟 / ({s_tf} 可用根数 × {bmin} 分钟), "
         f"「同标的同时至多一仓」口径。")
    emit("")

    # ---- 按年 / 离场归因 / 风险画像(主打 reso) ----
    pm = modes[0]
    tr = res[("reso", pm)]
    if tr:
        emit(f"【reso × {pm}】离场原因分布")
        for ln in R.by_reason(tr, market):
            emit("  " + ln)
        emit("")
        emit(f"【reso × {pm}】按年(单笔毛期望)")
        for ln in R.by_year(tr, market):
            emit("  " + ln)
        emit("")
        emit("风险画像(每笔等权 1 单位、按离场时间累加; **非**真实账户回撤)")
        emit(f"{'入场口径':>8} {'最好一笔':>10} {'最差一笔':>10} "
             f"{'累加最大回撤':>14} {'最长连亏':>9} {'累加总和':>10}")
        for m in entries:
            rk = risk_block(res[(m, pm)])
            emit(f"{m:>8} {rk['best'] * 100:>+9.2f}% {rk['worst'] * 100:>+9.2f}% "
                 f"{rk['mdd'] * 100:>+13.2f}% {rk['max_lose']:>7}笔 "
                 f"{rk['final']:>+9.1f}")
        emit("")

    # ---- 敏感性: 剔除 end_of_data ----
    emit("敏感性: 剔除 end_of_data(数据末尾强平)")
    emit(f"{'入场口径':>8} {'笔数':>6} {'单笔毛':>9} {'中位':>9} {'被剔':>6}")
    for m in entries:
        tr2 = res[(m, pm)]
        keep = [t for t in tr2 if t.reason != "end_of_data"]
        if not keep:
            continue
        exp = sum(t.pnl_pct for t in keep) / len(keep)
        srt = sorted(t.pnl_pct for t in keep)
        emit(f"{m:>8} {len(keep):>6} {exp * 100:>+8.3f}% "
             f"{srt[len(srt) // 2] * 100:>+8.3f}% {len(tr2) - len(keep):>6}")
    emit("")

    # ---- 基准 ----
    bm = benchmark(syms, s_tf, market, t0, t1)
    if bm:
        bm.sort(key=lambda x: x["total"])
        n = len(bm)
        med = bm[n // 2]
        mean_tot = sum(x["total"] for x in bm) / n
        pd_all = sorted(x["total"] / x["days"] for x in bm if x["days"] > 0)
        med_pd = pd_all[len(pd_all) // 2] if pd_all else 0.0
        btc = next((x for x in bm if x["sym"] == "BTCUSDT"), None)
        emit("=" * 108)
        emit("基准(同池、同区间)")
        bpd = int(round(24.0 * 3600_000 / D.TF_MS[s_tf]))
        pi_d = pool_index(syms, s_tf, market, t0, t1, rebal_bars=bpd)
        pi_t = pool_index(syms, s_tf, market, t0, t1, rebal_bars=1)
        if pi_d:
            emit(f"  ① 同池等权指数 · **逐日再平衡**(tf 无关, 主基准): "
                 f"全程 {pi_d['total'] * 100:+.1f}%  每日 {pi_d['per_day'] * 100:+.4f}%  "
                 f"年化 {pi_d['ann'] * 100:+.1f}%")
        if pi_t:
            emit(f"  ①' 逐 {s_tf} 再平衡(**上界**, 含波动率收割溢价): "
                 f"全程 {pi_t['total'] * 100:+.1f}%  年化 {pi_t['ann'] * 100:+.1f}%")
        emit(f"  ② 买入持有(单标的, 预热后首根 → 区间末): 全池 {n} 个, "
             f"全程 中位 {med['total'] * 100:+.0f}% / 均值 {mean_tot * 100:+.0f}%; "
             f"每日(线性)中位 {med_pd * 100:+.3f}%")
        if btc:
            emit(f"  ③ BTCUSDT: 全程 {btc['total'] * 100:+.0f}% "
                 f"(年化 {btc['ann'] * 100:+.0f}%, 每日 "
                 f"{btc['total'] / btc['days'] * 100:+.3f}%)")
        emit("")

    # ---- 随机入场对照 ----
    ctrl = args.control.strip()
    key = (ctrl, pm)
    if ctrl in entries and res.get(key):
        seeds = int(args.seeds)
        tr_real = res[key]
        real = R.summarize(tr_real, market)
        n1 = sum(1 for t in tr_real
                 if (t.feats or {}).get("l_kind") == B1
                 or (t.feats or {}).get("s_kind") == B1)
        kind_mix = [B1] * max(1, n1) + [B2] * max(1, len(tr_real) - n1)
        emit("=" * 108)
        emit(f"随机入场对照 —— 同离场口径({pm})、同止损口径、同入场频率, 只把"
             f"「共振择时」换成「随机择时」({seeds} 个种子)")
        dl = diags[key]
        tasks = []
        for d in dl:
            if d.err or not d.n_op_bars or not d.n_signals:
                continue
            tasks.append((d.sym, pm, s_tf, market, t0, t1, d.n_signals,
                          kind_mix, seeds))
        pool_tr: list[list] = [[] for _ in range(seeds)]
        used = 0
        if args.jobs and args.jobs > 1:
            import multiprocessing as mp
            with mp.Pool(int(args.jobs)) as pool:
                for k, r in enumerate(pool.imap_unordered(_mp_ctrl, tasks,
                                                          chunksize=1)):
                    if r is not None:
                        used += 1
                        for s in range(seeds):
                            pool_tr[s].extend(r[1][s])
                    if (k + 1) % 25 == 0:
                        print(f"  对照 {k + 1}/{len(tasks)}", file=sys.stderr,
                              flush=True)
        else:
            for t in tasks:
                r = _mp_ctrl(t)
                if r is not None:
                    used += 1
                    for s in range(seeds):
                        pool_tr[s].extend(r[1][s])
        summ = [R.summarize(x, market) for x in pool_tr]
        summ = [s for s in summ if s["n"]]
        if summ:
            def col(f):
                return sorted(f(r) for r in summ)

            def show(label, vals, real_v, pct=True):
                k = len(vals)
                lo, hi = vals[k // 20], vals[-1 - k // 20]
                better = sum(1 for v in vals if v >= real_v) / k
                f = ((lambda v: f"{v * 100:+.3f}%") if pct
                     else (lambda v: f"{v:.2f}"))
                emit(f"  {label:<10} 随机中位 {f(vals[k // 2])}  "
                     f"[P5 {f(lo)} , P95 {f(hi)}]   真实 {f(real_v)}   "
                     f"随机里 ≥ 真实的比例 {better:>5.0%}")
            emit(f"  * 参与对照标的 {used} 个;  随机池笔数中位 {summ[0]['n']} "
                 f"(真实 {real['n']})")
            show("单笔毛期望", col(lambda r: r["exp_pct"]), real["exp_pct"])
            show("每持仓日", col(lambda r: per_day(r)), per_day(real))
            show("胜率", col(lambda r: r["win_rate"]), real["win_rate"])
            show("PF", col(lambda r: r["pf"]), real["pf"], pct=False)
            emit("")
            emit("  读法: 「随机里 ≥ 真实的比例」≈0.5 ⇒ 与随机不可区分; "
                 "≈0 ⇒ 显著优于随机。以「每持仓日」为主(单笔期望被持仓时长污染)。")
            emit("")

    # ---- 池子诊断 ----
    emit("池子诊断:")
    dl = diags[key] if res.get(key) else diags[(entries[0], pm)]
    ok = [d for d in dl if not d.err]
    emit(f"  * 可用标的 {len(ok)}/{len(dl)};  {s_tf} 笔合计 "
         f"{sum(d.n_bi for d in ok):,};  中枢合计 {sum(d.n_zs for d in ok):,}")
    emit(f"  * 入场成交 {sum(d.n_signals for d in ok):,};  "
         f"被 risk.plan 拒单 {sum(d.n_reject for d in ok):,}")
    emit("")

    _flush(args.out, out)
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
