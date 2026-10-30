"""4h 级别第一类买点入场 + 缠论卖点离场的现货回测(9 年)。

使用者口径:
  * 操作级别 = **4 小时**; 入场 = **只做 B1(第一类买点/背驰)**;
  * 离场 = **也按缠论**(S1/S2/S3 三类卖点);
  * **只做多**, 不做空。

## 与项目既有 A/B 的关系

既有 `tools/run_ab.py` 的 spot 口径是 `LEVEL_TF["spot"] = ("4h","1h","15m")` ——
**4h 是大级别(定方向)、1h 才是操作级别**, 且离场走 `engine.manage`
(追踪止损 + RR 目标 + 结构破位), **完全没有卖点**。本轮是:
操作级别换成 4h, 且离场换成缠论卖点。所以需要新的 `chanbook/sell.py`(纯新增模块)。

## 四个离场口径

    old           旧离场: 追踪止损(3.0×当前risk) + RR=2 目标 + 结构破位  ← 对照
    chan123       纯缠论: 只在 S1/S2/S3 离场, **不设任何止损**
    chan123_stop  S1/S2/S3 优先, 另设**灾难止损**(结构止损, 无 RR 目标/无追踪)
    chan1_stop    仅 S1(一卖) + 灾难止损

⚠ **入场判据相同, 但不是同一批成交**: `risk.plan` 只依赖 (side, entry, 结构点位,
ATR, market, kind), 与离场口径无关 ⇒ 四个变体的入场**判据**逐笔一致; 但本项目
"同标的同时至多一仓", 离场越晚 ⇒ 占仓越久 ⇒ 后续候选池越小(实测 3 标的冒烟:
old 69 笔 vs chan123 64 笔)。这是**路径依赖**, 与 `rr`/`trail_mult` 扫描同源,
不可用"静态推断"跨变体比较, 只能各自完整重跑。

⚠ **只报"单笔毛期望"会误导**: 纯缠论无止损 ⇒ 持仓从 2.9 天拉到 17 天, 单笔期望
自然变大。因此汇总表**必须**同时给出"每持仓日收益"与"仅在仓年化", 并与同期
买入持有的年化对照。另单列"剔除 end_of_data"的敏感性 —— 数据末尾强平的那一笔
若是大赢家(实测 1 笔 +78.8% 就能把 64 笔的均值抬 1.2pp), 会把结论带偏。

## 数据口径(两个显式决定)

1. **强制用 1h 现场聚合出 4h**, 不走 `data.load`。原因: 归档里 spot 只有
   `BTCUSDT_4h` / `ETHUSDT_4h` 两个现成 4h 文件, 且起点是 **2021-01**(5.7 年),
   而 1h 底档从 **2017-08** 起(9 年)。若按 `data.load` 的"有文件就优先用文件",
   这两个标的会被静默截掉 3.4 年历史, 与其余 89 个标的(走聚合)口径不一致。
   统一走聚合 ⇒ 全池 9 年、同一套 4h 边界(UTC 00/04/08/12/16/20, 与币安一致)。
2. **预热 300 根**(与引擎 `WARMUP` 一致), 4h 即 50 天; 少于 500 根 4h 的标的跳过。

无未来函数: 每个 bar 先 `advance(bar_ts + tf_ms)` 把"本根已收盘"的事实喂进去,
再在该 bar 的**收盘价**成交; 卖点触发用的是已冻结的笔(与买点同源的确认代价)。

运行: python tools/backtest_b1_4h.py --start 2017-08 --end 2026-08 [--limit N]
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import random
import sys
import time
import zlib
from datetime import datetime, timezone
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook import report as R                                  # noqa: E402
from chanbook import risk                                         # noqa: E402
from chanbook.engine import (Diag, Trade, _close, _regime_at,     # noqa: E402
                             manage, open_pos, struct_line)
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1                                    # noqa: E402
from chanbook.sell import S1, S2, S3, SellScanner                 # noqa: E402
from chanbook.strategy import BookCfg, Signal                     # noqa: E402

MARKET = "spot"
TF = "4h"
WARMUP = 300
MIN_BARS = 500                     # 与 level.MIN_KLINES["op"] 一致

EXIT_MODES = {
    "old": "旧离场(追踪3.0×risk + RR2目标 + 结构破位) —— 对照",
    "chan123": "纯缠论卖点 S1/S2/S3, 不设止损",
    "chan123_stop": "缠论卖点 S1/S2/S3 + 灾难止损(无RR目标/无追踪)",
    "chan1": "仅一卖 S1, 不设止损(分离'卖点选择'与'有没有止损'两个因素)",
    "chan1_stop": "仅一卖 S1 + 灾难止损",
}
SELL_KINDS = {
    "old": (),
    "chan123": (S1, S2, S3),
    "chan123_stop": (S1, S2, S3),
    "chan1": (S1,),
    "chan1_stop": (S1,),
}
PRIO = {S1: 0, S2: 1, S3: 2}       # 同一 bar 命中多个卖点时取最强的那个


# ------------------------------------------------------------------ 单标的
def load_4h(sym: str):
    """1h 现场聚合出 4h(不用归档现成 4h 文件, 见模块 docstring 数据口径 1)。"""
    base = D.load(MARKET, sym, "1h")
    if base.n == 0:
        return None
    return D.resample(base, D.TF_MS[TF])


def run_symbol(sym: str, mode: str, t_start: int = 0, t_end: int = 0,
               macd_kw: dict | None = None, scan_over: dict | None = None):
    """macd_kw / scan_over: 仅供 tools/macd_sweep.py 的敏感性测试覆盖参数。

    两者缺省为 None ⇒ 与参数寻优前**逐位相同**的既有路径(默认 MACD 12/26/9、
    `BookCfg.scanner_kw()` 的全部默认值)。
    """
    kl = load_4h(sym)
    dg = Diag(sym=sym)
    if kl is None:
        dg.err = "无 1h 底档"
        return [], dg
    if kl.n < MIN_BARS:
        dg.err = f"4h 根数不足 {kl.n}"
        return [], dg
    cfg = BookCfg.legacy(market=MARKET, kinds=(B1,))        # 只做 B1
    skw = cfg.scanner_kw()
    if scan_over:
        skw.update(scan_over)
    lv = Level("op", TF, "bi", scanner_kw=skw, macd_kw=macd_kw).attach(kl)
    kinds = SELL_KINDS[mode]
    sscan = SellScanner(**{k: v for k, v in _sell_kw(mode).items()}) if kinds else None
    tm = risk.trail_mult_of(MARKET)
    use_stop = mode.endswith("_stop")
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
        lv.advance(bar_ts + lv.tf_ms)
        bar = {"ts": bar_ts, "open": kl.o[i], "high": kl.h[i],
               "low": kl.l[i], "close": kl.c[i]}
        if pos is not None:
            closed = None
            if mode == "old":
                pos, closed = manage(pos, bar, tm)
                if closed is None:
                    line = struct_line(lv.bib.bis, pos["entry_ts"], 1)
                    if line is not None and kl.c[i] < line:
                        closed = _close(pos, kl.c[i], bar_ts, "structure_break")
                        pos = None
            else:
                h = _sell_hit(sscan, lv, kinds)
                if h is not None:
                    closed = _close(pos, kl.c[i], bar_ts, "sell_" + h.kind)
                    pos = None
                elif use_stop and bar["low"] <= pos["stop"]:
                    closed = _close(pos, pos["stop"], bar_ts, "stop_loss")
                    pos = None
            if closed is not None:
                trades.append(closed)
            continue
        pts = lv.scan_points()                       # enable=(B1,) ⇒ 只返回 B1
        if not pts:
            continue
        p = pts[0]
        entry = float(kl.c[i])
        pl = risk.plan("long", entry, float(p.price), lv.atr(), MARKET, B1)
        if not pl.ok:
            dg.n_reject += 1
            continue
        sig = Signal(
            sym=sym, side="long", kind=B1, ts=bar_ts, i=i, entry=entry,
            stop=pl.stop, target=pl.target, atr=pl.atr, risk_pct=pl.risk_pct,
            notional=pl.notional, counter=True, zs_i=int(p.zs_i),
            beichi=(p.beichi.short() if p.beichi else ""),
            reasons=list(p.reasons), feats=dict(p.feats),
        )
        dg.n_signals += 1
        pos = open_pos(sig, _regime_at(MARKET, bar_ts + lv.tf_ms), bar_ts, entry)
    if pos is not None:
        last = kl.n - 1
        trades.append(_close(pos, kl.c[last], int(kl.ts[last]), "end_of_data"))
    dg.n_seen = {"B1": dg.n_signals}
    dg.n_blocked = {"B1": dg.n_reject}
    sn = lv.snapshot()
    dg.n_bi, dg.n_seg, dg.n_zs = sn.n_bi, sn.n_seg, sn.n_zs
    return trades, dg


def _sell_kw(mode: str) -> dict:
    """卖点参数 = 买点参数的镜像默认值(不调参, 与买点共用同一套阈值)。"""
    return {"pull_bi": 1, "beichi_tol": 0.02, "beichi_strict": False,
            "s2_max_bi": 20, "s1_max_lag": 12, "s1_zs_window": 1,
            "s3_zs_window": 2, "s1_allow_no_zs": True,
            "enable": SELL_KINDS[mode] or (S1, S2, S3)}


def _sell_hit(sscan, lv, kinds):
    pts = sscan.scan(lv.bib.bis, lv.zb.all_zss(), lv.macd, lv.bib.n_final)
    if not pts:
        return None
    pts = [p for p in pts if p.kind in kinds]
    if not pts:
        return None
    pts.sort(key=lambda p: PRIO.get(p.kind, 9))
    return pts[0]


def per_day(s: dict) -> float:
    """每持仓日收益(线性, 非复利) = 单笔毛期望 / 平均持仓天数。

    跨变体的唯一可比量: 离场口径一变, 持仓时长就变, 单笔期望本身不可比。
    与买入持有的 `total / days` 同口径, 可直接对照。
    """
    d = s["hold_min"] / 1440.0
    return (s["exp_pct"] / d) if d > 0 else 0.0


def risk_block(tr: list) -> dict:
    """候选池口径的风险画像(等权、按离场时间顺序累加)。

    ⚠ 口径说明: 这里的"权益曲线"是**每笔等权 1 单位**的累加和(不是复利资金曲线,
    也没有仓位限制)。它回答的是"平均一笔的序列风险" —— 本项目是候选池口径,
    组合层会被资金约束抹平, 所以只能用这个代理。它**不**代表实际账户回撤。
    """
    srt = sorted(tr, key=lambda t: t.exit_ts)
    eq = peak = mdd = 0.0
    cur_lose = max_lose = 0
    n_win = 0
    rets = []
    for t in srt:
        eq += t.pnl_pct
        rets.append(t.pnl_pct)
        peak = max(peak, eq)
        mdd = min(mdd, eq - peak)
        if t.pnl_pct <= 0:
            cur_lose += 1
            max_lose = max(max_lose, cur_lose)
        else:
            cur_lose = 0
            n_win += 1
    return {
        "best": max(rets) if rets else 0.0,
        "worst": min(rets) if rets else 0.0,
        "mdd": mdd,                       # 累加和的最大回撤(单位: 每笔等权 1)
        "max_lose": max_lose,             # 最长连亏笔数
        "final": eq,                      # 累加总和 = n × 单笔期望
    }


def _bar_rets(sym: str, t_start: int, t_end: int) -> dict:
    """某标的的 4h 逐根收益 {ts: ret}(用于构造同池等权指数)。"""
    kl = load_4h(sym)
    out = {}
    if kl is None or kl.n < MIN_BARS:
        return out
    for i in range(1, kl.n):
        ts = int(kl.ts[i])
        if t_start and ts < t_start:
            continue
        if t_end and ts > t_end:
            break
        c0, c1 = kl.c[i - 1], kl.c[i]
        if c0 > 0:
            out[ts] = c1 / c0 - 1.0
    return out


# ------------------------------------------------------------------ 随机入场对照
def level_schedule(sym: str, exit_mode: str, t_start: int = 0, t_end: int = 0):
    """**一次**遍历级别, 把随机对照需要的逐 bar 信息全部录下来。

    这样 K 个种子 × 若干轮频率校准都只是纯 Python 循环, 不必反复推进级别
    (推进级别是本工具里唯一昂贵的操作)。

    卖点日程与持仓无关(只依赖级别结构) ⇒ 录一次全种子共享。
    结构止损位取"最近一根冻结向下笔的低点" —— 正是 B1 用的那个基准
    (`Point.price` 就是触发它的那根向下笔低点), 保证止损口径同源。
    """
    kl = load_4h(sym)
    if kl is None or kl.n < MIN_BARS:
        return None
    cfg = BookCfg.legacy(market=MARKET, kinds=(B1,))
    lv = Level("op", TF, "bi", scanner_kw=cfg.scanner_kw()).attach(kl)
    kinds = SELL_KINDS[exit_mode]
    sscan = SellScanner(**_sell_kw(exit_mode))
    start = WARMUP
    if t_start:
        start = max(start, bisect.bisect_right(kl.ts, t_start - 1))
    rows: list[tuple] = []
    for i in range(start, kl.n):
        bar_ts = int(kl.ts[i])
        if t_end and bar_ts > t_end:
            break
        lv.advance(bar_ts + lv.tf_ms)
        hit = _sell_hit(sscan, lv, kinds)
        bis = lv.bib.bis
        struct = None
        for k in range(lv.bib.n_final - 1, -1, -1):   # 笔交替 ⇒ 至多两步
            if bis[k].direction < 0:
                struct = float(bis[k].low)
                break
        rows.append((bar_ts, float(kl.c[i]), float(kl.l[i]), struct, lv.atr(),
                     (hit.kind if hit is not None else "")))
    return rows


def seeds_pass(rows: list, seed0: int, rate: float, seeds: int,
               n_iter: int = 4, target: int = 0) -> list[list]:
    """给定"每 bar 入场概率"跑 K 个随机种子, 并按目标笔数校准频率。

    为什么要校准: 若把 rate 直接取成"真实 B1 的**已成交**频率", 随机侧还要再被
    `risk.plan` 拒单 + 占仓阻塞吃掉一半 ⇒ 笔数对不齐(实测 86 vs 163), 比较失真。
    这里迭代调整 rate 使**中位笔数**逼近真实值(最多 n_iter 轮, 容差 10%)。
    """
    reg = ""                                     # 对照不需要行情分桶
    out: list[list] = []
    for _ in range(max(1, n_iter)):
        rngs = [random.Random(seed0 * 1000003 + s * 2654435761) for s in range(seeds)]
        positions: list = [None] * seeds
        out = [[] for _ in range(seeds)]
        for (ts, c, lo, struct, atr, sell_kind) in rows:
            for s in range(seeds):
                pos = positions[s]
                if pos is not None:
                    if sell_kind:
                        out[s].append(_close(pos, c, ts, "sell_" + sell_kind))
                        positions[s] = None
                    elif lo <= pos["stop"]:
                        out[s].append(_close(pos, pos["stop"], ts, "stop_loss"))
                        positions[s] = None
                    continue
                if rngs[s].random() >= rate:
                    continue
                pl = risk.plan("long", c, struct, atr, MARKET, B1)
                if not pl.ok:
                    continue
                sig = SimpleNamespace(sym="", side="long", kind=B1, stop=pl.stop,
                                      target=pl.target, trend="", counter=1,
                                      confirm="", risk_pct=pl.risk_pct,
                                      notional=pl.notional, feats={})
                positions[s] = open_pos(sig, reg, ts, c)
        if not target:
            break
        got = sorted(len(o) for o in out)[seeds // 2]
        if got <= 0 or abs(got - target) / target <= 0.10:
            break
        rate = min(0.9, rate * target / got)
    return out


def pool_index(syms: list, t_start: int, t_end: int, min_n: int = 5) -> dict:
    """同池**等权、逐 4h 再平衡**的指数 —— 策略的公平市场基准。

    为什么不用 BTC: BTC 是这个池子里表现最好的资产(9 年 +1684%), 而池内**中位**标的
    是亏钱的。拿 BTC 当基准会把"跑输"说轻。等权池指数才对应"在这池子里随机持有一篮子"。
    """
    by_ts: dict[int, list[float]] = {}
    for sym in syms:
        for ts, r in _bar_rets(sym, t_start, t_end).items():
            by_ts.setdefault(ts, []).append(r)
    eq = 1.0
    n_bar = 0
    for ts in sorted(by_ts):
        rs = by_ts[ts]
        if len(rs) < min_n:
            continue
        eq *= 1.0 + sum(rs) / len(rs)
        n_bar += 1
    if n_bar == 0:
        return {}
    per_bar = eq ** (1.0 / n_bar) - 1.0
    return {"total": eq - 1.0, "n_bar": n_bar, "per_bar": per_bar,
            "per_day": (1.0 + per_bar) ** 6 - 1.0,
            "ann": (1.0 + per_bar) ** (365.25 * 6) - 1.0}


# ------------------------------------------------------------------ 买入持有基准
def benchmark(syms: list, t_start: int, t_end: int) -> list[dict]:
    out = []
    for sym in syms:
        kl = load_4h(sym)
        if kl is None or kl.n < MIN_BARS:
            continue
        a = WARMUP
        if t_start:
            a = max(a, bisect.bisect_right(kl.ts, t_start - 1))
        b = kl.n - 1
        if t_end:
            b = bisect.bisect_right(kl.ts, t_end) - 1
        if b <= a or not kl.c[a]:
            continue
        days = (int(kl.ts[b]) - int(kl.ts[a])) / D.MS_DAY
        tot = kl.c[b] / kl.c[a] - 1.0
        out.append({"sym": sym, "days": days, "total": tot,
                    "ann": ((1 + tot) ** (365.0 / days) - 1) if days > 0 else 0.0})
    return out


# ------------------------------------------------------------------ 主流程
def ms_of(ym: str, end: bool = False) -> int:
    y, m = (int(x) for x in ym.split("-"))
    if end:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--modes", default="old,chan123,chan123_stop,chan1,chan1_stop")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/b1_4h_spot.txt")
    ap.add_argument("--cache-dir", default="out/cache_b1_4h")
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t0, t1 = ms_of(args.start), ms_of(args.end, end=True)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    bad = [m for m in modes if m not in EXIT_MODES]
    if bad:
        print(f"未知离场口径 {bad}; 可用 {list(EXIT_MODES)}")
        return 2

    syms = D.symbols(MARKET)
    if args.limit:
        syms = syms[:args.limit]
    out_lines: list[str] = []

    def emit(s: str = "") -> None:
        out_lines.append(str(s))
        print(s, flush=True)

    thr = 0.0020                       # 现货门槛(= 双边手续费 0.20%)
    emit(f"# chanbook 4h-B1 现货回测  {MARKET}  {args.start} ~ {args.end}  "
         f"({len(syms)} 个标的)")
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"# 口径: 操作级别={TF} · 入场=只做 B1(第一类买点) · 只做多 · "
         f"4h 由 1h 现场聚合 · 预热 {WARMUP} 根 · 最少 {MIN_BARS} 根")
    emit("")

    res: dict[str, list] = {}
    diags: dict[str, list] = {}
    for mode in modes:
        emit(f"[{time.strftime('%H:%M:%S')}] 开始 {mode}: {EXIT_MODES[mode]} ...")
        ta = time.time()
        tr_all: list[Trade] = []
        dg_all: list[Diag] = []
        for k, sym in enumerate(syms):
            tr, dg = run_symbol(sym, mode, t0, t1)
            tr_all.extend(tr)
            dg_all.append(dg)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(syms)} 笔={len(tr_all)}", file=sys.stderr,
                      flush=True)
        res[mode] = tr_all
        diags[mode] = dg_all
        emit(R.full_report(MARKET, f"{mode} — {EXIT_MODES[mode]}", tr_all, dg_all))
        emit(f"# {mode} 耗时 {time.time() - ta:.0f}s")
        emit("")
        try:
            os.makedirs(args.cache_dir, exist_ok=True)
            R.save_trades(os.path.join(args.cache_dir, f"spot_{mode}.json"),
                          tr_all, dg_all)
        except Exception as e:                                # noqa: BLE001
            print(f"缓存失败: {e}")

    # ---- 汇总 ----
    emit("=" * 92)
    emit("汇总(候选池口径; 单笔期望 = 笔数加权毛期望, 唯一可跨变体比较的口径)")
    emit(f"{'离场口径':>14} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} {'单笔净':>9} "
         f"{'盈亏比':>7} {'PF':>6} {'均持仓':>9} {'每持仓日':>9} {'仅在仓年化':>10}")
    for mode in modes:
        s = R.summarize(res[mode], MARKET)
        if not s["n"]:
            emit(f"{mode:>14} {0:>6}   (无成交)")
            continue
        pd_ = per_day(s)
        emit(f"{mode:>14} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
             f"{s['exp_pct'] * 100:>+8.3f}% {s['net_exp_pct'] * 100:>+8.3f}% "
             f"{s['rr']:>7.2f} {s['pf']:>6.2f} "
             f"{s['hold_min'] / 1440:>7.2f}天 {pd_ * 100:>+8.3f}% "
             f"{pd_ * 365 * 100:>+9.1f}%")
    emit("")
    emit(f"门槛: 单笔**毛**期望需 > {thr * 100:.2f}%(双边手续费 0.20% + 滑点余量); "
         f"净期望 = 名义额加权毛期望 − 双边费")
    emit("⚠ 「每持仓日」= 单笔毛期望 / 平均持仓天数(**非复利**); 它是跨变体唯一"
         "可与买入持有年化对照的量。单笔毛期望高 ≠ 更优 —— 拿得久自然高。")
    emit("")

    # ---- 敏感性: 剔除数据末尾强平 ----
    emit("敏感性检验: 剔除 end_of_data(数据末尾强制平仓)后的单笔毛期望与中位")
    emit(f"{'离场口径':>14} {'笔数':>6} {'单笔毛':>9} {'中位':>9} {'被剔笔数':>9} "
         f"{'被剔那笔的收益':>14}")
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        eod = [t for t in tr if t.reason == "end_of_data"]
        keep = [t for t in tr if t.reason != "end_of_data"]
        if not keep:
            emit(f"{mode:>14} {0:>6}   (全被剔除)")
            continue
        exp = sum(t.pnl_pct for t in keep) / len(keep)
        srt = sorted(t.pnl_pct for t in keep)
        med = srt[len(srt) // 2]
        eod_s = "、".join(f"{t.pnl_pct * 100:+.1f}%" for t in eod) or "—"
        emit(f"{mode:>14} {len(keep):>6} {exp * 100:>+8.3f}% {med * 100:>+8.3f}% "
             f"{len(eod):>9} {eod_s:>14}")
    emit("")

    # ---- 结构诊断 ----
    emit("结构诊断(全池):")
    d0 = diags[modes[0]]
    ok = [d for d in d0 if not d.err]
    emit(f"  * 可用标的 {len(ok)}/{len(d0)}"
         + (f"; 跳过 {len(d0) - len(ok)} 个(样本不足)" if len(ok) != len(d0) else ""))
    emit(f"  * 4h 笔合计 {sum(d.n_bi for d in ok):,}  "
         f"中枢合计 {sum(d.n_zs for d in ok):,}")
    emit(f"  * 入场信号(过 risk.plan)合计 {sum(d.n_signals for d in ok):,}  "
         f"被风控拒单 {sum(d.n_reject for d in ok):,}")
    emit("")

    # ---- 离场归因 ----
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        emit(f"【{mode}】离场原因分布")
        for ln in R.by_reason(tr, MARKET):
            emit("  " + ln)
        emit("")
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        emit(f"【{mode}】按年")
        for ln in R.by_year(tr, MARKET):
            emit("  " + ln)
        emit("")

    # ---- 风险画像 ----
    emit("风险画像(每笔等权 1 单位、按离场时间累加; **非**真实账户回撤, 仅候选池口径)")
    emit(f"{'离场口径':>14} {'最好一笔':>9} {'最差一笔':>9} {'累加最大回撤':>13} "
         f"{'最长连亏':>9} {'累加总和':>10}")
    for mode in modes:
        tr = res[mode]
        if not tr:
            continue
        rk = risk_block(tr)
        emit(f"{mode:>14} {rk['best'] * 100:>+8.2f}% {rk['worst'] * 100:>+8.2f}% "
             f"{rk['mdd'] * 100:>+12.2f}% {rk['max_lose']:>7}笔 "
             f"{rk['final']:>+9.1f}")
    emit("")

    # ---- 仓位占用 ----
    emit("仓位占用(「同标的同时至多一仓」下的实际在仓时间占比; 4h 一根 = 240 分钟)")
    emit(f"{'离场口径':>14} {'在仓分钟':>12} {'可用分钟':>12} {'在仓占比':>9} "
         f"{'折算全年暴露':>12}")
    for mode in modes:
        used = sum(t.hold_min for t in res[mode])
        avail = sum(d.n_op_bars for d in diags[mode] if not d.err) * 240
        share = (used / avail) if avail else 0.0
        emit(f"{mode:>14} {used:>12,} {avail:>12,} {share * 100:>8.2f}% "
             f"{share * 365:>10.1f}天")
    emit("")

    # ---- 基准: 同池等权指数 + 买入持有 ----
    bm = benchmark(syms, t0, t1)
    if bm:
        bm.sort(key=lambda x: x["total"])
        n = len(bm)
        med = bm[n // 2]
        mean_tot = sum(x["total"] for x in bm) / n
        mean_ann = sum(x["ann"] for x in bm) / n
        pd_all = sorted(x["total"] / x["days"] for x in bm if x["days"] > 0)
        med_pd = pd_all[len(pd_all) // 2] if pd_all else 0.0
        mean_pd = sum(pd_all) / len(pd_all) if pd_all else 0.0
        btc = next((x for x in bm if x["sym"] == "BTCUSDT"), None)
        emit("=" * 92)
        emit("基准(同池、同区间、同预热)")
        emit(f"① 同池**等权、逐 4h 再平衡**指数 ← 公平基准(对应「在这池子里随机持有一篮子」)")
        pi = pool_index(syms, t0, t1)
        if pi:
            emit(f"   * 全程 {pi['total'] * 100:+.1f}% ({pi['n_bar']:,} 根 4h)  "
                 f"每根 {pi['per_bar'] * 100:+.4f}%  每日 {pi['per_day'] * 100:+.4f}%  "
                 f"年化 {pi['ann'] * 100:+.1f}%")
        emit(f"② 买入持有(单标的, 预算后第一根 → 区间末)")
        emit(f"   * 全池 {n} 个标的: 全程 中位 {med['total'] * 100:+.0f}% / "
             f"均值 {mean_tot * 100:+.0f}%  年化(复利) 均值 {mean_ann * 100:+.1f}%")
        emit(f"   * **每日收益(线性)**: 中位 {med_pd * 100:+.3f}% / "
             f"均值 {mean_pd * 100:+.3f}%  ← 与策略「每持仓日」同口径")
        if btc:
            emit(f"   * BTCUSDT(池内最强): 全程 {btc['total'] * 100:+.0f}% "
                 f"({btc['days']:.0f} 天, 年化 {btc['ann'] * 100:+.0f}%, "
                 f"每日 {btc['total'] / btc['days'] * 100:+.3f}%)")
        emit(f"   * ⚠ 买入持有是**满仓全程持有**, 策略是**多笔短持仓 + 至多一仓**; "
             f"两者不可直接相减。可比的只有单位时间收益, 且策略还要乘在仓占比。")
        emit("")
    # ---- 随机入场对照 ----
    ctrl = "chan123_stop"
    if ctrl in res and res[ctrl]:
        seeds = 20
        emit("=" * 92)
        emit(f"随机入场对照 —— 同离场口径({ctrl})、同止损口径、同入场频率, 只把"
             f"「B1 择时」换成「随机择时」({seeds} 个种子)")
        ref_diag = {d.sym: d for d in diags[ctrl]}
        pool: list[list] = [[] for _ in range(seeds)]
        used = 0
        tc = time.time()
        for k, sym in enumerate(syms):
            d = ref_diag.get(sym)
            if d is None or d.err or not d.n_op_bars or not d.n_signals:
                continue
            rows = level_schedule(sym, ctrl, t0, t1)
            if not rows:
                continue
            rate0 = d.n_signals / max(1, d.n_op_bars)
            outs = seeds_pass(rows, zlib.crc32(sym.encode("utf-8")), rate0,
                              seeds, target=d.n_signals)
            used += 1
            for s in range(seeds):
                pool[s].extend(outs[s])
            if (k + 1) % 20 == 0:
                print(f"  对照 {k + 1}/{len(syms)}", file=sys.stderr, flush=True)
        rows = []
        for s in range(seeds):
            su = R.summarize(pool[s], MARKET)
            if su["n"]:
                rows.append(su)
        if rows:
            def col(f):
                return sorted(f(r) for r in rows)

            real = R.summarize(res[ctrl], MARKET)

            def show(label, vals, real_v, pct=True):
                n_r = len(vals)
                lo, hi = vals[n_r // 20], vals[-1 - n_r // 20]
                better = (sum(1 for v in vals if v >= real_v) / n_r) if pct else 0.0
                f = (lambda v: f"{v * 100:+.3f}%") if pct else (lambda v: f"{v:.2f}")
                emit(f"  {label:<10} 随机中位 {f(vals[n_r // 2])}  "
                     f"[P5 {f(lo)} , P95 {f(hi)}]   "
                     f"真实 B1 {f(real_v)}   "
                     f"随机里 ≥ 真实的比例 {better:>5.0%}")
            emit(f"  * 参与对照标的 {used} 个;  随机池笔数中位 {rows[0]['n']}  "
                 f"(真实 {real['n']})")
            show("单笔毛期望", col(lambda r: r["exp_pct"]), real["exp_pct"])
            show("每持仓日", col(lambda r: per_day(r)), per_day(real))
            show("胜率", col(lambda r: r["win_rate"]), real["win_rate"])
            show("盈亏比", col(lambda r: r["rr"]), real["rr"], pct=False)
            show("PF", col(lambda r: r["pf"]), real["pf"], pct=False)
            emit("")
            emit("  读法: 「随机里 ≥ 真实的比例」≈ 0.5 ⇒ 真实 B1 与随机不可区分; "
                 "≈ 0(接近 0%) ⇒ 真实 B1 显著优于随机。只比单笔毛期望会被持仓时长污染, "
                 "请以「每持仓日」为主。")
            emit("")
        emit(f"  # 对照耗时 {time.time() - tc:.0f}s")
        emit("")

    _flush(args.out, out_lines)
    print(f"\n报告已写入 {args.out}")
    return 0


def _flush(path: str, lines: list) -> None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as e:                                        # noqa: BLE001
        print(f"写盘失败 {path}: {e}")


if __name__ == "__main__":
    sys.exit(main())
