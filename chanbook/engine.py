"""流式回测引擎 —— 逐 bar 驱动 `LevelSet`, 产出**信号层口径**的成交记录。

为什么要走信号层(候选池)口径:
  组合层(资金池/名额/熔断)会被"资金不足冻停"抹平 —— 一旦某个参数把资金打光,
  后续所有信号都开不出来, 于是"组合层结果"变成对资金曲线的描述, 而不是对策略
  优劣的描述。本项目四次过滤类实验(行情/波动/乖离/斐波)都在信号层口径下得出
  "天花板只有 0.01~0.05pp"这个结论, 所以 A/B 必须继续用它。

本引擎与旧引擎(`chan-quant/backtest_long.py`)的口径对齐清单:
  * 数据: 同一份归档, 缺根不补, 按实际 bar 推进;
  * 预热: 操作级别跳过前 300 根(旧引擎 start=300);
  * 持仓: 同标的同时至多一仓; 每根 bar 先管理既有持仓再找新信号;
  * 离开管理 —— **两条互斥路径**(`BookCfg.exit_mode`, 见 `strategy.EXIT_MODES`):
      `old`  : 追踪止损(浮盈达 1R 激活, 回撤 trail_mult × 当前 risk) → 止损 →
               RR 目标, **同根同时触发按止损优先**(保守), risk 用**当前 stop**
               (与实盘一致); 外加结构破位(入场后新形成的"抬高低点"被收盘价
               跌破, 基准取**操作级别的笔端点**而非单根K线分型 —— `struct_line`)。
               —— 2026-09-29 **之前**的默认, 现由 `BookCfg.legacy()` 给出。
      `stop` : 只认入场时算出的结构止损("灾难点"), 不追踪 / 不设目标 / 不跑
               结构破位。**这是 `sell_exit`(缠论卖点)能生效的前提** ——
               机械离场把中位持仓压到 0.75 天, 结构判据(中位 3.83 天)没机会
               表态。 —— 2026-09-29 起的**默认**, 依据见 DESIGN.md §9.7。
  * 卖点离场: `sell_exit` 非空时(默认 `(S1,)`), 每根 bar 先扫一次卖点,
    命中即在**当根收盘价**离场并标记 `sell_S1/S2/S3` —— 优先级**高于**上面
    两条路径。卖点自带"同一冻结笔只报一次"的去重锁, 可无条件逐 bar 调用。
  * 数据末尾强制平仓, 标记 `end_of_data`;
  * `pnl_pct` 为**毛**价格收益率(与旧引擎一致), 手续费在报告层单独扣。

与旧引擎的差异(有意为之, 也就是本次重写要检验的东西):
  * 入场判据从"均线排列 + 缠论择时 + 三买不等式"换成完整的三类买卖点;
  * 方向闸门按买卖点性质分别设置(见 strategy.DIR_GATE);
  * ATR 窗口、RR、ATR 倍数、名义额全部与旧引擎一致 ⇒ 差异只能来自入场结构;
  * **但 2026-09-29 起离场默认改成 `stop + 一卖`, 不再与旧引擎同源** ——
    旧引擎仍是"追踪 + RR 目标 + 结构破位"。要复现旧口径用 `BookCfg.legacy()`。
"""

from __future__ import annotations

import bisect
import os
import sys
from dataclasses import dataclass, field

from . import data as D
from . import risk
from .filters import ATR1D
from .level import LevelSet
from . import market_filters as MF
from .sell import S1, S2, S3, SellScanner
from .strategy import BookCfg, Strategy

__all__ = ["Trade", "run_symbol", "run_all", "btc_regime", "sell_hit"]

WARMUP = 300          # 操作级别预热根数(与旧引擎一致)

# 同一根 bar 同时命中多个卖点时取哪一个(与 `strategy.PRIORITY` 同义:
# 一卖信息量最大, 三卖是趋势确认后的最后机会)。
SELL_PRIO = {S1: 0, S2: 1, S3: 2}


@dataclass
class Trade:
    sym: str = ""
    entry_ts: int = 0
    exit_ts: int = 0
    side: str = "long"
    kind: str = ""
    trend: str = ""
    regime: str = ""
    counter: int = 0
    confirm: str = ""
    entry: float = 0.0
    exit: float = 0.0
    stop0: float = 0.0
    target: float = 0.0
    pnl_pct: float = 0.0
    risk_pct: float = 0.0
    notional: float = 0.0
    reason: str = ""
    hold_min: int = 0
    # 结构化特征(来自 Signal.feats)。**纯观测, 不参与任何判定** —— 加它不改变
    # 任何一笔成交的入场/离场, 只让成交记录携带更多可用于建模的信息。
    feats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


@dataclass
class Diag:
    sym: str = ""
    n_op_bars: int = 0
    n_signals: int = 0
    n_seen: dict = field(default_factory=dict)
    n_blocked: dict = field(default_factory=dict)
    n_reject: int = 0
    n_ma_block: int = 0        # 被两条均线闸门拦掉的数量(2026-09-29 新增)
    n_volcap_block: int = 0    # 被日线波动率上限闸门拦掉的数量(第三轮新增)
    n_regime_block: int = 0    # 被大盘状态闸门拦掉的数量(第四轮新增)
    n_overheat_block: int = 0  # 被过热带拦掉的数量(第四轮新增)
    n_resistance_block: int = 0  # 被前高压制拦掉的数量(第四轮新增)
    n_fib_block: int = 0       # 被斐波那契拦掉的数量(第四轮新增)
    n_correl_block: int = 0    # 被联动度拦掉的数量(第四轮新增)
    n_bi: int = 0
    n_seg: int = 0
    n_zs: int = 0
    trend_at_end: str = ""
    err: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


# ------------------------------------------------------------------ 行情分桶
_REG_CACHE: dict = {}


def btc_regime(market: str):
    """BTC 1h 的行情分桶(bull/range/bear), 口径与旧引擎 `app.regime` 完全一致。

    只用于**事后归因**(把成交按行情分类), 不参与任何判定 —— 因此放在引擎层,
    复刻 MA20/MA60 + 收盘价的三段式判据即可, 不需要引入旧代码。
    """
    hit = _REG_CACHE.get(market)
    if hit is not None:
        return hit
    kl = D.load(market, "BTCUSDT", "1h")
    FA, SL = 20, 60
    ts = kl.ts
    lab: list[str] = []
    sf = 0.0                       # 最近 FA 根收盘价之和(滚动)
    ss = 0.0                       # 最近 SL 根收盘价之和(滚动)
    for i in range(kl.n):
        c = kl.c[i]
        sf += c
        ss += c
        if i >= FA:
            sf -= kl.c[i - FA]
        if i >= SL:
            ss -= kl.c[i - SL]
        if i < SL:
            lab.append("range")    # 数据不足 => 最宽松(与旧引擎一致)
            continue
        cf, cs = sf / FA, ss / SL
        lab.append("bull" if (c > cf > cs) else
                   "bear" if (c < cf < cs) else "range")
    _REG_CACHE[market] = (ts, lab)
    return ts, lab


def _regime_at(market: str, close_ts: int) -> str:
    ts, lab = btc_regime(market)
    k = bisect.bisect_right(ts, close_ts - D.MS_HOUR)
    return lab[k - 1] if k > 0 else "range"


# ------------------------------------------------------------------ 持仓管理
def manage(pos: dict, bar: dict, trail_mult: float) -> tuple[dict | None, Trade | None]:
    """追踪止损 + 止损 + RR 目标(与旧引擎 `backtest_long.manage` 同款)。"""
    d = 1 if pos["side"] == "long" else -1
    key = "highest" if d == 1 else "lowest"
    pos[key] = (max if d == 1 else min)(pos[key], bar["high"] if d == 1 else bar["low"])
    r = abs(pos["entry"] - pos["stop"])
    if r > 0:
        if not pos["trail"] and (bar["close"] - pos["entry"]) * d >= r:
            pos["trail"] = True
        if pos["trail"]:
            cand = (pos["highest"] - trail_mult * r if d == 1
                    else pos["lowest"] + trail_mult * r)
            pos["stop"] = max(pos["stop"], cand) if d == 1 else min(pos["stop"], cand)
    hs = bar["low"] <= pos["stop"] if d == 1 else bar["high"] >= pos["stop"]
    ht = bar["high"] >= pos["target"] if d == 1 else bar["low"] <= pos["target"]
    if hs:                                  # 同根同时触发按止损优先(保守)
        px, why = pos["stop"], ("trail_stop" if pos["trail"] else "stop_loss")
    elif ht:
        px, why = pos["target"], "take_profit"
    else:
        return pos, None
    return None, _close(pos, px, bar["ts"], why)


def _close(pos: dict, px: float, exit_ts: int, why: str) -> Trade:
    d = 1 if pos["side"] == "long" else -1
    return Trade(
        sym=pos["sym"], entry_ts=pos["entry_ts"], exit_ts=int(exit_ts),
        side=pos["side"], kind=pos["kind"], trend=pos.get("trend", ""),
        regime=pos.get("regime", ""), counter=int(pos.get("counter", 0)),
        confirm=pos.get("confirm", ""),
        entry=pos["entry"], exit=float(px), stop0=pos["stop0"],
        target=pos["target"], pnl_pct=(float(px) - pos["entry"]) / pos["entry"] * d,
        risk_pct=pos.get("risk_pct", 0.0), notional=pos.get("notional", 0.0),
        reason=why, hold_min=int((int(exit_ts) - pos["entry_ts"]) / D.MS_MIN),
        feats=dict(pos.get("feats") or {}),
    )


def struct_line(bis: list, entry_ts: int, d: int) -> float | None:
    """入场后新形成的"抬高低点"(做多)/"下压高点"(做空)。

    与旧引擎的区别: 旧引擎在**入场周期的单根K线**上找新分型(更敏感、更易被噪声
    触发); 这里用**操作级别的笔端点** —— 笔要求的K线间隔更长, 所以结构线只在
    真正的次级别走势转折处抬升, 是"结构破位"更接近原文的读法。

    实现要点: 倒序遍历并**提前 break**(笔按时间有序), 不做任何切片 —— 否则每根
    bar 一次 O(n_笔) 复制, 长回测会退化成 O(n²)。
    """
    line = None
    for k in range(len(bis) - 2, -1, -1):
        b = bis[k]
        if int(b.end_ts) <= entry_ts:
            break
        if b.direction != d:
            continue
        v = float(b.low) if d < 0 else float(b.high)
        if line is None:
            line = v
        elif d < 0:
            line = max(line, v)
        else:
            line = min(line, v)
    return line


# ------------------------------------------------------------------ 单标的
def open_pos(sig, regime: str, bar_ts: int, entry: float) -> dict:
    return {
        "sym": sig.sym, "side": sig.side, "kind": sig.kind, "entry": entry,
        "stop": sig.stop, "stop0": sig.stop, "target": sig.target,
        "highest": entry, "lowest": entry, "trail": False, "entry_ts": int(bar_ts),
        "trend": sig.trend, "regime": regime, "counter": int(sig.counter),
        "confirm": sig.confirm, "risk_pct": sig.risk_pct,
        "notional": sig.notional, "feats": dict(sig.feats or {}),
    }


def _sell_scanner_of(cfg: BookCfg) -> SellScanner | None:
    """按买点扫描器的参数**镜像**构造卖点扫描器(不调参, 两边共用同一套阈值)。"""
    if not cfg.sell_exit:
        return None
    skw = cfg.scanner_kw()
    return SellScanner(
        pull_bi=skw["pull_bi"], tol=0.0, beichi_tol=skw["beichi_tol"],
        beichi_strict=skw["beichi_strict"], s2_max_bi=skw["b2_max_bi"],
        s1_max_lag=skw["b1_max_lag"], s1_zs_window=skw["b1_zs_window"],
        s3_zs_window=skw["b3_zs_window"], s1_allow_no_zs=skw["b1_allow_no_zs"],
        enable=tuple(cfg.sell_exit))


def sell_hit(sscan: SellScanner, op, kinds) -> object | None:
    """在当前操作级别状态下扫一次卖点, 返回优先级最高的一个(无则 None)。

    `SellScanner` 自带"同一个冻结笔只扫一次"的去重锁(`sell.py` 里那个坑的修复),
    所以调用方可以每根 bar 无条件调用, 不会重复报同一个卖点。

    ⚠ `kinds` 是**输出过滤**: `SellScanner.scan` 里 S1/S2 必须始终参与检测
    (S2 依赖 S1 维护的 `last_s1` 状态机), 只过滤返回值。
    """
    pts = sscan.scan(op.bib.bis, op.zss_view(), op.macd, op.bib.n_final)
    if not pts:
        return None
    pts = [p for p in pts if p.kind in kinds]
    if not pts:
        return None
    pts.sort(key=lambda p: SELL_PRIO.get(p.kind, 9))
    return pts[0]


def run_symbol(market: str, sym: str, cfg: BookCfg,
               t_start: int = 0, t_end: int = 0) -> tuple[list[Trade], Diag]:
    """单标的信号层回测。返回 (候选池成交, 诊断)。"""
    ls = LevelSet(market, sym, scanner_kw=cfg.scanner_kw(),
                  use_confirm=bool(cfg.confirm_gate or cfg.record_confirm),
                  op_extra=cfg.op_extra()).load()
    dg = Diag(sym=sym)
    if not ls.ok():
        dg.err = (f"数据不足 op={ls.op.kl.n} trend={ls.trend.kl.n}"
                  if ls.loaded else "加载失败")
        return [], dg
    strat = Strategy(cfg)
    # ---- 波动率上限闸门(默认关闭): 注入日线 ATR 视图 ----
    # 只在开启时加载日线 ⇒ 默认路径零 IO、零开销、逐位等价。
    if float(cfg.atr1d_cap) > 0.0:
        ls.atr1d = ATR1D(D.load(market, sym, "1d"), int(cfg.atr1d_n))
    # ---- 第四轮补全: 市场状态/位置/联动过滤器视图(默认关闭) ----
    # 只在开启时加载多周期数据 ⇒ 默认路径零 IO、逐位等价。
    if cfg.market_filter_on():
        mviews = {"overheat": None, "resistance": None, "fib": None, "correl": None}
        if cfg.overheat_on:
            mviews["overheat"] = MF.Overheat(D.load(market, sym, "1d"),
                                             D.load(market, sym, "4h"))
        if cfg.resistance_on:
            mviews["resistance"] = MF.Resistance(ls.op.kl)
        if cfg.fib_on:
            mviews["fib"] = MF.Fib()
        if cfg.correl_on:
            mviews["correl"] = MF.Correl(ls.op.kl, D.load(market, "BTCUSDT", "1h"),
                                         D.load(market, "ETHUSDT", "1h"))
        ls.mviews = mviews
    tm = (risk.trail_mult_of(market) if cfg.trail_mult is None
          else float(cfg.trail_mult))
    op = ls.op
    sscan = _sell_scanner_of(cfg)
    sell_kinds = tuple(cfg.sell_exit)
    trades: list[Trade] = []
    pos = None
    start = WARMUP
    if t_start:
        start = max(start, bisect.bisect_right(op.kl.ts, t_start - 1))
    dg.n_op_bars = max(0, op.kl.n - start)
    for i in range(start, op.kl.n):
        bar_ts = int(op.kl.ts[i])
        if t_end and bar_ts > t_end:
            # 分段截断: 边界即离场 —— 若有未平持仓, 用当前 bar 收盘价强制平仓,
            # 而不是持有到数据物理末尾(否则"训练段"会混入跨越数年的僵尸持仓)。
            if pos is not None:
                trades.append(_close(pos, op.kl.c[i], bar_ts, "segment_end"))
                pos = None
            break
        close_ts = bar_ts + op.tf_ms
        ls.advance(close_ts)
        bar = {"ts": bar_ts, "open": op.kl.o[i], "high": op.kl.h[i],
               "low": op.kl.l[i], "close": op.kl.c[i]}
        if pos is not None:
            closed = None
            # ---- 第 5 条离场分支: 缠论卖点(默认关闭; 见 THEORY_GAP 审计 B1 类) ----
            # 放在 `manage` **之前**: 一卖是结构判据, 中位持仓 3.8 天, 而追踪止损
            # 一旦激活就用 `trail_mult × risk` 的机械回撤离场 —— 让结构判据先表态,
            # 才测得出"卖点是否比追踪止损更早/更准".
            if sscan is not None:
                h = sell_hit(sscan, op, sell_kinds)
                if h is not None:
                    closed = _close(pos, op.kl.c[i], bar_ts, "sell_" + h.kind)
                    pos = None
            # ---- 第 6 条离场分支: 多周期分形离场(默认关闭; 第四轮补全) ----
            # 移植 chan-quant `app/nesting.exit_fenxing`: 入场后新形成的反向分型
            # (做多=顶分型) 否证原趋势即离场。放在 sell_hit 之后、stop 之前,
            # 与 sell_hit 同为"结构判据优先于机械止损"。
            if closed is None and cfg.exit_fenxing:
                hit, _why, _ext = MF.fenxing_exit(
                    op.merger.cks, pos["entry_ts"], pos["side"],
                    min_bars_after=cfg.fx_min_bars_after,
                    require_bi=cfg.fx_require_bi,
                    bis=op.bib.bis, ref_price=pos["entry"])
                if hit:
                    closed = _close(pos, op.kl.c[i], bar_ts, "fenxing_exit")
                    pos = None
            if closed is None and cfg.exit_mode == "stop":
                # ---- `exit_mode="stop"`: 只认入场时算出的结构止损("灾难点") ----
                # 不追踪 / 不设目标 / 不跑结构破位。这是 `sell_exit` 能**生效**的前提:
                # 机械离场会把中位持仓压到 0.75 天, 结构判据(中位 3.8 天)没有机会表态。
                d = 1 if pos["side"] == "long" else -1
                hit = (bar["low"] <= pos["stop"]) if d > 0 else (bar["high"] >= pos["stop"])
                if hit:
                    closed = _close(pos, pos["stop"], bar_ts, "stop_loss")
                    pos = None
            elif closed is None:
                pos, closed = manage(pos, bar, tm)
                if closed is None and cfg.struct_on:
                    d = 1 if pos["side"] == "long" else -1
                    line = struct_line(op.bib.bis, pos["entry_ts"], d)
                    if line is not None and ((op.kl.c[i] < line) if d > 0
                                             else (op.kl.c[i] > line)):
                        closed = _close(pos, op.kl.c[i], bar_ts, "structure_break")
                        pos = None
            if closed is not None:
                trades.append(closed)
            continue
        pts = op.scan_points()
        sig = strat.pick(ls, pts)
        if sig is None:
            continue
        dg.n_signals += 1
        pos = open_pos(sig, _regime_at(market, close_ts), bar_ts, op.kl.c[i])
    if pos is not None:
        last = op.kl.n - 1
        trades.append(_close(pos, op.kl.c[last], int(op.kl.ts[last]), "end_of_data"))
    dg.n_seen = dict(strat.n_seen)
    dg.n_blocked = dict(strat.n_blocked)
    dg.n_reject = int(strat.n_reject)
    dg.n_ma_block = int(strat.n_ma_block)
    dg.n_volcap_block = int(strat.n_volcap_block)
    dg.n_regime_block = int(strat.n_regime_block)
    dg.n_overheat_block = int(strat.n_overheat_block)
    dg.n_resistance_block = int(strat.n_resistance_block)
    dg.n_fib_block = int(strat.n_fib_block)
    dg.n_correl_block = int(strat.n_correl_block)
    sn = op.snapshot()
    dg.n_bi, dg.n_seg, dg.n_zs = sn.n_bi, sn.n_seg, sn.n_zs
    dg.trend_at_end = ls.trend.state.label()
    return trades, dg


# ------------------------------------------------------------------ 批量
def _pick_symbols(market: str, cfg_syms=None, universe_top: int = 0,
                  y2020: bool = False, limit: int = 0) -> list[str]:
    if cfg_syms:
        return list(cfg_syms)
    if universe_top:
        uni = D.universe(market, y2020)
        seen: list[str] = []
        for m in sorted(uni):
            for s in uni[m]:
                if s not in seen:
                    seen.append(s)
        return seen[:limit] if limit else seen
    syms = D.symbols(market)
    return syms[:limit] if limit else syms


def run_all(market: str, cfg: BookCfg, symbols=None, universe_top: int = 0,
            y2020: bool = False, limit: int = 0, jobs: int = 0,
            progress: int = 0, t_start: int = 0, t_end: int = 0):
    """并行跑一批标的, 返回 (trades, diags)。"""
    import multiprocessing as mp
    syms = _pick_symbols(market, symbols, universe_top, y2020, limit)
    if not syms:
        return [], []
    jobs = jobs or max(1, min(12, (os.cpu_count() or 4) - 1))
    payload = [(market, s, cfg, t_start, t_end) for s in syms]
    trades: list[Trade] = []
    diags: list[Diag] = []
    if jobs <= 1:
        for k, p in enumerate(payload):
            tr, dg = run_symbol(*p)
            trades.extend(tr)
            diags.append(dg)
            if progress and (k + 1) % progress == 0:
                print(f"  [{market}] {k + 1}/{len(payload)} 笔={len(trades)}",
                      file=sys.stderr, flush=True)
        return trades, diags
    with mp.Pool(jobs) as pool:
        for k, (tr, dg) in enumerate(pool.imap_unordered(_worker, payload,
                                                         chunksize=1)):
            trades.extend(tr)
            diags.append(dg)
            if progress and (k + 1) % progress == 0:
                print(f"  [{market}] {k + 1}/{len(payload)} 笔={len(trades)}",
                      file=sys.stderr, flush=True)
    return trades, diags


def _worker(payload):
    return run_symbol(*payload)
