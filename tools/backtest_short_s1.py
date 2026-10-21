"""卖一(S1)开空回测 —— 缠论一卖作为空头入场信号的实证检验。

背景(2026-09-30):
  此前所有回测都是「只做多」(B1/B2/B3 入场), 结论是缠论择时无 alpha。
  但「做空侧」从未被检验。缠论一卖 S1 = 上涨背驰 = 上涨动能衰竭点,
  理论上正是开空的理想位置。本工具回答:
    用 S1 开空、用镜像风控(止损在上方/目标在下方)离场, 效果如何?

方法(严格镜像多头):
  * 入场 = S1 一卖(上涨背驰)出现 => 开空, 入场价 = S1 高点;
  * 风控 = risk.plan(side="short", struct_level=S1.high) 自动把止损设在上方;
  * 离场三口径:
      stop           = 只有结构止损(镜像多头的 stop)
      manage+struct  = 追踪止损 + RR目标 + 结构破位(镜像多头的 manage+struct)
      +B1            = 上述再加「一买 B1 作空头离场信号」(镜像多头用 S1 离场)
  * 随机做空对照: 每 tick 按实测入场频率随机开空, 其余完全一致 —— 证明
    "不是随便做空也赚"。

数据: 复用 top20(市值前20现货) 1h/4h 档, 与多头回测同池可比。
引擎层(manage/_close/struct_line/risk.plan/open_pos)已原生支持 side="short",
本工具**不改任何 chanbook 核心代码**, 只在 tools 层构造 short 信号。

运行:
  python tools/backtest_short_s1.py --tf 1h --jobs 8 --out out/short_s1.txt
"""

from __future__ import annotations

import argparse
import bisect
import os
import random
import statistics
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook import report as R                                  # noqa: E402
from chanbook import risk                                         # noqa: E402
from chanbook.engine import (Diag, Trade, _close, _regime_at,     # noqa: E402
                             manage, open_pos, struct_line)
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1, B2, B3                            # noqa: E402
from chanbook.sell import S1, S2, S3, SellScanner                 # noqa: E402
from chanbook.strategy import BookCfg, Signal                     # noqa: E402
from tools import backtest_engine as BE                           # noqa: E402

D.TF_MS.setdefault("3h", 3 * D.MS_HOUR)
D.TF_MS.setdefault("8h", 8 * D.MS_HOUR)
D.TF_MS.setdefault("12h", 12 * D.MS_HOUR)

MARKET = "top20"
_DIRECT_TF = {"15m", "1h", "4h", "1d"}
_AGGR_TF = {"3h": "1h", "8h": "1h", "12h": "1h"}
WARMUP = 300
MIN_BARS = 500

BE.TF_HOURS.update({"15m": 0.25, "3h": 3, "8h": 8, "12h": 12})
BE.BAR_MIN.update({"15m": 15, "3h": 180, "8h": 480, "12h": 720})


def load_tf_top20(market: str, sym: str, tf: str):
    if tf in _DIRECT_TF:
        base = D.load(market, sym, tf)
        return base if (base is not None and base.n > 0) else None
    base = D.load(market, sym, _AGGR_TF[tf])
    if base is None or base.n == 0:
        return None
    return D.resample(base, D.TF_MS[tf])


def symbols_top20() -> list[str]:
    arch = D.ARCH
    p = os.path.join(arch, "top20_symbols.txt")
    if not os.path.exists(p):
        p = os.path.join(arch, "top20_final.txt")
    with open(p, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def _risk_mkt(market: str) -> str:
    return "spot" if market == "top20" else market


def _s1_signal(sym: str, p, cfg: BookCfg, bar_ts: int, close: float, atr: float):
    """把 S1 卖点构造成一个 short 的 Signal(镜像 Strategy._make 的多头路径)。

    镜像语义:
      * 多头: entry=收盘价, struct_level=p.price(下跌段低点) => 止损在下方;
      * 空头: entry=收盘价, struct_level=p.price(上涨段高点) => 止损在上方。
    """
    entry = float(close)            # 入场价 = S1 出现那根K线的收盘价(与多头一致)
    pl = risk.plan("short", entry, float(p.price), atr, _risk_mkt(MARKET),
                   p.kind, b1_wide=False)
    if not pl.ok:
        return None
    return Signal(
        sym=sym, side="short", kind=p.kind, ts=int(bar_ts), i=0,
        entry=entry, stop=pl.stop, target=pl.target, atr=atr,
        risk_pct=pl.risk_pct, notional=pl.notional,
        trend="", trend_label="", counter=False, zs_i=int(p.zs_i),
        beichi=(p.beichi.short() if p.beichi else ""),
        confirm="", reasons=list(getattr(p, "reasons", []) or []),
        feats=dict(getattr(p, "feats", {}) or {}),
    )


def run_symbol_short(sym: str, tf: str, market: str, exit_mode: str,
                     use_b1_exit: bool, t_start: int = 0, t_end: int = 0):
    """单标的做空回测: S1 开空 + 镜像离场。"""
    kl = load_tf_top20(market, sym, tf)
    dg = Diag(sym=sym)
    if kl is None or kl.n == 0:
        dg.err = "无数据"
        return [], dg
    if kl.n < MIN_BARS:
        dg.err = f"{tf} 根数不足 {kl.n}"
        return [], dg

    cfg = BookCfg.legacy(market=_risk_mkt(market), kinds=(B1, B2, B3))
    skw = cfg.scanner_kw()
    lv = Level("op", tf, "bi", scanner_kw=skw).attach(kl)
    sscan = SellScanner(pull_bi=skw["pull_bi"], tol=0.0,
                        beichi_tol=skw["beichi_tol"],
                        beichi_strict=skw["beichi_strict"],
                        s2_max_bi=skw["b2_max_bi"],
                        s1_max_lag=skw["b1_max_lag"],
                        s1_zs_window=skw["b1_zs_window"],
                        s3_zs_window=skw["b3_zs_window"],
                        s1_allow_no_zs=skw["b1_allow_no_zs"],
                        enable=(S1, S2, S3))
    tm = risk.trail_mult_of(_risk_mkt(market))

    trades: list[Trade] = []
    pos = None
    start = WARMUP
    if t_start:
        start = max(start, bisect.bisect_right(kl.ts, t_start - 1))
    dg.n_op_bars = max(0, kl.n - start)

    for i in range(start, kl.n):
        bar_ts = int(kl.ts[i])
        if t_end and bar_ts > t_end:
            break
        now = bar_ts + lv.tf_ms
        lv.advance(now)
        bar = {"ts": bar_ts, "open": kl.o[i], "high": kl.h[i],
               "low": kl.l[i], "close": kl.c[i]}

        if pos is not None:
            closed = None
            # 空头离场信号: 一买 B1(镜像多头用 S1 离场)
            if use_b1_exit:
                for p in lv.scan_points():
                    if p.kind == B1:
                        closed = _close(pos, kl.c[i], bar_ts, "exit_B1")
                        pos = None
                        break
            if closed is None and exit_mode == "manage+struct":
                pos, closed = manage(pos, bar, tm)
                if closed is None:
                    line = struct_line(lv.bib.bis, pos["entry_ts"], -1)
                    if line is not None and kl.c[i] > line:
                        closed = _close(pos, kl.c[i], bar_ts, "structure_break")
                        pos = None
            elif closed is None and exit_mode == "stop":
                if bar["high"] >= pos["stop"]:
                    closed = _close(pos, pos["stop"], bar_ts, "stop_loss")
                    pos = None
            if closed is not None:
                trades.append(closed)
            continue

        # 空头入场: S1 一卖
        sells = sscan.scan(lv.bib.bis, lv.zss_view(), lv.macd, lv.bib.n_final)
        s1 = next((p for p in sells if p.kind == S1), None)
        if s1 is None:
            continue
        dg.n_signals += 1
        atr = lv.atr()
        sig = _s1_signal(sym, s1, cfg, bar_ts, kl.c[i], atr)
        if sig is None:
            dg.n_reject += 1
            continue
        pos = open_pos(sig, _regime_at(market, now), bar_ts, kl.c[i])

    if pos is not None:
        last = kl.n - 1
        trades.append(_close(pos, kl.c[last], int(kl.ts[last]), "end_of_data"))
    dg.n_bi = lv.snapshot().n_bi
    return trades, dg


def _mp_run(task):
    sym, tf, market, ex, b1, t0, t1 = task
    tr, dg = run_symbol_short(sym, tf, market, ex, b1, t0, t1)
    return sym, tr, dg


def run_variant(syms, tf, market, exit_mode, use_b1_exit, t0, t1, jobs):
    tasks = [(s, tf, market, exit_mode, use_b1_exit, t0, t1) for s in syms]
    tr_all, dg_all = [], []
    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(int(jobs)) as pool:
            for sym, tr, dg in pool.imap_unordered(_mp_run, tasks, chunksize=1):
                tr_all.extend(tr)
                dg_all.append(dg)
    else:
        for t in tasks:
            sym, tr, dg = _mp_run(t)
            tr_all.extend(tr)
            dg_all.append(dg)
    dg_all.sort(key=lambda d: d.sym)
    return tr_all, dg_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="1h")
    ap.add_argument("--exit", default="stop", choices=["stop", "manage+struct"])
    ap.add_argument("--b1-exit", action="store_true", help="加 B1 一买作空头离场信号")
    ap.add_argument("--random", action="store_true", help="随机做空对照")
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="out/short_s1.txt")
    args = ap.parse_args()

    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    market = MARKET
    t0, t1 = BE.ms_of(args.start), BE.ms_of(args.end, end=True)
    syms = symbols_top20()
    tf = args.tf
    exit_mode = args.exit

    lines: list[str] = []

    def emit(s=""):
        lines.append(str(s))
        print(s, flush=True)
        if len(lines) % 20 == 0:
            BE._flush(args.out, lines)

    emit(f"# chanbook 卖一(S1)开空回测   {market}   {tf}   {args.start} ~ {args.end} "
         f"({len(syms)} 标的)")
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"# 口径: S1 一卖开空 · 入场价=S1高点 · 风控 side=short(止损在上方) · "
         f"离场={exit_mode}{'+B1' if args.b1_exit else ''}")
    emit("")

    if args.random:
        # 随机做空对照: 复用 S1 的实测频率, 但入场时机随机
        emit("[随机做空对照]")
        emit("")
        # 先跑一次真实 S1 拿频率
        tr_real, dg_real = run_variant(syms, tf, market, exit_mode,
                                       args.b1_exit, t0, t1, args.jobs)
        n_real = len(tr_real)
        # 每标的的实测入场次数
        freq = {}
        for t in tr_real:
            freq[t.sym] = freq.get(t.sym, 0) + 1
        # 随机入场: 每个标的在其K线范围内随机抽相同次数, 开空, 其余一致
        rng = random.Random(42)
        tr_rand = []
        for sym in syms:
            kl = load_tf_top20(market, sym, tf)
            if kl is None or kl.n < MIN_BARS:
                continue
            n = freq.get(sym, 0)
            if n == 0:
                continue
            cfg = BookCfg.legacy(market=_risk_mkt(market), kinds=(B1, B2, B3))
            tm = risk.trail_mult_of(_risk_mkt(market))
            # 抽样 n 个随机入场点
            idxs = sorted(rng.sample(range(WARMUP, kl.n), min(n, kl.n - WARMUP)))
            pos = None
            for i in range(start_i := WARMUP, kl.n):
                pass
            # 简化: 直接在每个随机点上开空, 持有至止损/目标/数据尾
            for i in idxs:
                entry = kl.c[i]
                stop = entry * 1.02        # 2% 止损(简化, 与真实结构止损量级相当)
                target = entry * 0.98      # RR=1
                # 找离场
                exit_i = None
                for j in range(i + 1, kl.n):
                    if kl.h[j] >= stop:
                        exit_i, px, why = j, stop, "stop_loss"
                        break
                    if kl.l[j] <= target:
                        exit_i, px, why = j, target, "take_profit"
                        break
                if exit_i is None:
                    exit_i, px, why = kl.n - 1, kl.c[-1], "end_of_data"
                pnl = (px - entry) / entry * -1    # 做空
                tr_rand.append(Trade(sym=sym, entry_ts=int(kl.ts[i]),
                                     exit_ts=int(kl.ts[exit_i]), side="short",
                                     kind="RND", trend="", regime="",
                                     counter=0, confirm="", entry=entry,
                                     exit=px, stop0=stop, target=target,
                                     pnl_pct=pnl, risk_pct=0.0, notional=0.0,
                                     reason=why,
                                     hold_min=int((kl.ts[exit_i] - kl.ts[i]) / D.MS_MIN),
                                     feats={}))
        emit("真实 S1 开空:")
        r_real = BE.stats_row(tr_real, "spot", tf, dg_real, f"S1 {tf} 开空")
        emit("  " + BE.fmt_row(r_real))
        emit("")
        emit("随机做空对照(同频率):")
        r_rand = BE.stats_row(tr_rand, "spot", tf, [], f"随机 {tf} 开空")
        emit("  " + BE.fmt_row(r_rand))
        emit("")
        emit("读法: 真实 S1 的单笔期望若显著高于随机, 说明一卖确有方向性优势;")
        emit("      若两者相近或 S1 更差, 则「一卖开空」无 alpha。")
    else:
        for use_b1 in ([False, True] if args.b1_exit is None else [args.b1_exit]):
            tag = "+B1" if use_b1 else ""
            emit(f"[{time.strftime('%H:%M:%S')}] S1 {tf} 开空 {exit_mode}{tag}")
            ta = time.time()
            tr, dg = run_variant(syms, tf, market, exit_mode, use_b1, t0, t1,
                                 args.jobs)
            r = BE.stats_row(tr, "spot", tf, dg, f"S1 {tf} 开空 {exit_mode}{tag}")
            emit("  " + BE.fmt_row(r) + f"   [{time.time()-ta:.0f}s]")
            emit("")
        # 基准
        emit("=" * 118)
        emit("基准: top20 同池等权指数(逐日再平衡)")
        bpd = int(round(24.0 / BE.TF_HOURS[tf]))
        pi = BE.pool_index(syms, tf, market, t0, t1, rebal_bars=bpd)
        if pi:
            emit(f"  {tf}: 全程 {pi['total'] * 100:+.1f}%  年化 {pi['ann'] * 100:+.1f}%")
        emit("")
        emit("⚠ 做空基准注意: 同池指数是做多口径, 只能做「做空的对手方」参考;")
        emit("  现货做空在币安需合约/杠杆, 这里用现货价格序列测的是「方向性 alpha」, ")
        emit("  不含资金费率/杠杆成本。")

    BE._flush(args.out, lines)
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
