"""六级别「任意两共振」回测 —— 1h/4h/1d/3d/周线/月线, 任意 2 个级别共振即信号。

使用者口径(2026-10-01):
> 在多级别共振中, 添加 1小时 4H 日线 3日线和周线月线级别, 任意 2 个共振即算有效信号。

## 落地口径(逐条显式声明)

1. **级别集** = {1h, 4h, 1d, 3d, 1w, 1M}。1h/4h/1d 用归档原始档; 3d/1w/1M 由 1d
   现场聚合(`D.resample`, UTC 对齐)。**月线 = 固定 30 天桶**(非日历月), 周线 = 固定
   7 天桶(自纪元对齐, 起始为周四)—— 都是"一致分桶"意义上的工程近似, 对结构判定
   无偏, 但与交易所原生周/月K不完全同桶。
2. **每级别独立跑完整流水线**(`Level(..., "bi")` + `resonance.scanner_kw()`,
   与既有两级共振同口径可对照), 检测 B1/B2/B3 三类买点, 边沿触发。
3. **时钟由 1h 驱动**(入场/离场/止损全在 1h 收盘)。每个 1h 收盘时刻把 6 个级别
   一起 advance 到同一时刻(与 `resonance.ResoScanner.advance_to` 的 2026-09-29
   修法同源: 用 `now = ts + 1h`, 不偷看)。
4. **窗口语义**(原两级共振的推广): 每个买点在**本级别** `--win`(默认 3) 根 K 线内
   有效; 计时从**被知道的时刻**(该级别检出买点那根K线的收盘)起算; 1h 收盘价跌破
   买点结构低点即作废(`--no-invalidate` 可关)。
5. **共振 = 事件制**: 某级别出现**新**买点(触发), 且此刻存在**不同级别**的买点仍在
   有效窗口内(配合) ⇒ 信号。两个级别同一 1h 时刻同时检出也 counts(与原实现
   `step()` 先吸收长级别再判定的行为一致)。
6. **配对选择**: 触发点优先 B1、其次细级别; 配合点同规则。**结构止损与 risk 种类
   取两级中较细级别那个点**(小级别给精确点, 与原 `Hit.struct_price` 的取向一致)。
7. **级别资格**: 各级别达到最小根数(1h 500 / 4h 200 / 1d 120 / 3d 72 / 1w 60 /
   1M 36)后才参与产信号; 数据不足的级别整个剔除(月线 9 年 ≈ 110 根, 只能放宽,
   已声明)。预热 = 1h 前 500 根(与 MIN_KLINES["op"] 一致)。
8. **入场口径四变体**(同一次遍历并行, 消融对照):
   * `any2`     —— 使用者口径: 任意 2 个级别共振(买点种类不限);
   * `any2_b1`  —— 老口径推广: 至少一侧 B1;
   * `any1`     —— 消融: 任意单级别买点即入场(不要求共振);
   * `s1h_only` —— 消融: 只做 1h 单级别买点(= 原两级实验的 s_only)。
9. 出口沿用生产口径 `chan1_stop`(仅一卖 S1 + 结构止损), 卖点扫描/止损都在 1h。

运行::

  python tools/backtest_reso6.py --market spot --jobs 10 --out out/reso6_spot.txt
  python tools/backtest_reso6.py --selfcheck
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys
import time
from datetime import datetime
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

from chanbook import data as D                                    # noqa: E402
from chanbook import risk                                         # noqa: E402
from chanbook.engine import (Diag, _close, _regime_at,            # noqa: E402
                             manage, open_pos, struct_line)
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1, B2, B3                            # noqa: E402
from chanbook.resonance import scanner_kw                         # noqa: E402
from chanbook.sell import SellScanner                             # noqa: E402

from backtest_b12_tf import (MIN_BARS, SELL_KINDS, WARMUP,         # noqa: E402
                             _flush, _sell_hit, _sell_kw, ms_of,
                             risk_block)
import tools.backtest_engine as BT                                # noqa: E402

# ---- 工具侧扩展周期映射 ----
# ⚠⚠ 本工具的 3d/1w/1M 桶是**自己从 1d 聚合**出来的(`load_tf`), 1M 桶宽 = **30 天**、
#   起点锚在 epoch —— **不是**交易所的日历月。而 `Level.__init__` 里的
#   `self.tf_ms = D.TF_MS[tf]` 会拿去判定"这根桶收盘了没有"(`Kl.end_at`)。
#   两者必须严格相等, 否则桶宽 30 天却按 31 天算收盘, 收盘时刻与自己的分桶错开一天。
#   主包 2026-10-02 给 `1M` 定的是 **31 天**(日历月的上界, 配合交易所原始月线 K 线),
#   与本工具口径不同 ⇒ **必须用 `update` 显式覆盖, 不能用 `setdefault`**
#   (setdefault 在主包已有该键时是哑操作, 会留下 30 天桶 + 31 天判定这种静默不一致)。
X_TF_MS = {"3d": 3 * D.MS_DAY, "1w": 7 * D.MS_DAY, "1M": 30 * D.MS_DAY}
D.TF_MS.update(X_TF_MS)

TF_ORDER = {"1h": 0, "4h": 1, "1d": 2, "3d": 3, "1w": 4, "1M": 5}
LEVEL_MIN = {"1h": 500, "4h": 200, "1d": 120, "3d": 72, "1w": 60, "1M": 36}
DEFAULT_TFS = "1h,4h,1d,3d,1w,1M"

ENTRY_MODES = {
    "any2": "任意 2 级别共振(使用者口径, 买点种类不限)",
    "any2_b1": "任意 2 级别共振且至少一侧 B1(老口径推广)",
    "any1": "消融: 任意单级别买点即入场",
    "s1h_only": "消融: 只做 1h 单级别买点",
}

_KL_CACHE: dict = {}


def load_tf(market: str, sym: str, tf: str):
    key = (market, sym, tf)
    if key in _KL_CACHE:
        return _KL_CACHE[key]
    if tf in ("1h", "4h", "1d"):
        kl = D.load(market, sym, tf)
    else:
        base = D.load(market, sym, "1d")
        kl = D.resample(base, X_TF_MS[tf]) if (base is not None and base.n) else None
    _KL_CACHE[key] = kl
    if len(_KL_CACHE) > 256:
        for k in list(_KL_CACHE)[:96]:
            _KL_CACHE.pop(k, None)
    return kl


# ------------------------------------------------------------------ 核心构件
class Multi6:
    """6 个级别的流式联立: 统一时钟(1h 收盘)、独立窗口、事件制共振。"""

    def __init__(self, market: str, sym: str, tfs, win: int,
                 invalidate: bool = True) -> None:
        self.market, self.sym = market, sym
        self.tfs = list(tfs)
        self.win = max(1, int(win))
        self.invalidate = bool(invalidate)
        kw = scanner_kw()
        self.kls: dict = {}
        self.lvls: dict = {}
        for tf in self.tfs:
            kl = load_tf(market, sym, tf)
            if kl is None or kl.n < LEVEL_MIN[tf]:
                continue
            lv = Level(f"r6_{tf}", tf, "bi", scanner_kw=dict(kw))
            lv.attach(kl)
            self.kls[tf] = kl
            self.lvls[tf] = lv
        self.active: list[dict] = []

    def ok(self) -> bool:
        return "1h" in self.lvls

    def advance_all(self, now: int) -> None:
        for lv in self.lvls.values():
            lv.advance(now)

    def step(self, now: int, close: float) -> list[dict]:
        """吸收本步新买点 -> 修剪窗口 -> 返回本步的新买点(fresh)。"""
        fresh: list[dict] = []
        for tf, lv in self.lvls.items():
            if lv.cur < LEVEL_MIN[tf]:          # 未达资格的级别不产信号
                continue
            kl = self.kls[tf]
            for p in lv.scan_points():
                i = lv.cur - 1
                q = {"tf": tf, "kind": p.kind, "price": float(p.price),
                     "known": int(kl.ts[i]) + lv.tf_ms, "ts": int(kl.ts[i])}
                self.active.append(q)
                fresh.append(q)
        self.active = prune(self.active, now, close, self.win, self.invalidate)
        return fresh


def prune(active: list[dict], now: int, close: float, win: int,
          invalidate: bool) -> list[dict]:
    """窗口修剪: 过期(now - known > win×本级别tf)或破位(close < price)即剔除。"""
    keep = []
    for q in active:
        if now - q["known"] > win * D.TF_MS[q["tf"]]:
            continue
        if invalidate and close < q["price"]:
            continue
        keep.append(q)
    return keep


def _rank(q: dict) -> tuple:
    return (0 if q["kind"] == B1 else 1, TF_ORDER.get(q["tf"], 9), -q["ts"])


def decide(fresh: list[dict], active: list[dict], mode: str):
    """**纯函数**: 由本步 fresh + 当前有效窗口判定信号。返回 (trig, partner|None)。

    * any2     : 需要一个不同级别的配合点(种类不限);
    * any2_b1  : 同上, 且触发与配合至少一侧 B1;
    * any1     : 只有触发点即可(消融);
    * s1h_only : 触发点必须是 1h 级别(消融)。
    """
    if not fresh:
        return None
    cand = [q for q in fresh if TF_ORDER.get(q["tf"], 9) < 9]
    if mode == "s1h_only":
        cand = [q for q in cand if q["tf"] == "1h"]
    if not cand:
        return None
    trig = sorted(cand, key=_rank)[0]
    if mode == "any1" or mode == "s1h_only":
        return (trig, None)
    others = [q for q in active if q["tf"] != trig["tf"]]
    if mode == "any2_b1" and trig["kind"] != B1:
        others = [q for q in others if q["kind"] == B1]
    if not others:
        return None
    partner = sorted(others, key=_rank)[0]
    return (trig, partner)


def pair_label(trig: dict, partner: dict | None) -> str:
    if partner is None:
        return f"{trig['kind']}@{trig['tf']} 单级"
    a, b = sorted((trig, partner), key=lambda q: TF_ORDER.get(q["tf"], 9))
    return f"{a['kind']}@{a['tf']}+{b['kind']}@{b['tf']}"


def struct_of(trig: dict, partner: dict | None) -> tuple[float, str]:
    """结构止损与 risk 种类取两级中**较细级别**的点。"""
    if partner is None or D.TF_MS[trig["tf"]] <= D.TF_MS[partner["tf"]]:
        return trig["price"], trig["kind"]
    return partner["price"], partner["kind"]


# ------------------------------------------------------------------ 单标的
def run_symbol(sym: str, market: str, tfs: tuple, win: int, invalidate: bool,
               entry_modes: tuple, exit_mode: str,
               t_start: int = 0, t_end: int = 0):
    m6 = Multi6(market, sym, tfs, win, invalidate)
    trades = {m: [] for m in entry_modes}
    diags = {m: Diag(sym=sym) for m in entry_modes}
    counts: dict = {"points": {tf: {} for tf in tfs}, "pairs": {}}
    if not m6.ok():
        for m in entry_modes:
            diags[m].err = "无 1h 数据或样本不足"
        return trades, diags, counts

    skl = m6.kls["1h"]
    lv = m6.lvls["1h"]
    # ⚠ 每个入场口径必须**各自**持有一个 SellScanner: 它是有状态的(`last_s1` 状态机
    # + `n_final_seen` 去重锁)。共用同一个实例时, 同 bar 内先被遍历到的口径会吃掉
    # 那个卖点, 排在后面的口径全部漏掉 S1 离场、跌落到止损 —— 实测 spot BTCUSDT
    # 4 口径共用时 s1h_only 从 361 笔掉到 225 笔(2026-10-02 发现并修复)。
    sscans = {m: SellScanner(**_sell_kw(exit_mode)) for m in entry_modes}
    ks = SELL_KINDS[exit_mode]
    tm = risk.trail_mult_of(market)
    use_stop = exit_mode.endswith("_stop")
    start = max(WARMUP, MIN_BARS)
    if t_start:
        start = max(start, bisect.bisect_right(skl.ts, t_start - 1))
    pos = {m: None for m in entry_modes}
    for m in entry_modes:
        diags[m].n_op_bars = max(0, skl.n - start)

    for i in range(start, skl.n):
        bar_ts = int(skl.ts[i])
        if t_end and bar_ts > t_end:
            break
        now = bar_ts + D.TF_MS["1h"]
        m6.advance_all(now)
        c = float(skl.c[i])
        bar = {"ts": bar_ts, "open": skl.o[i], "high": skl.h[i],
               "low": skl.l[i], "close": c}
        # ---- 1) 管理既有持仓(先离场, 平掉当根不再开新仓) ----
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
                h = _sell_hit(sscans[m], lv, ks)
                if h is not None:
                    closed = _close(p0, c, bar_ts, "sell_" + h.kind)
                    pos[m] = None
                elif use_stop and bar["low"] <= p0["stop"]:
                    closed = _close(p0, p0["stop"], bar_ts, "stop_loss")
                    pos[m] = None
            if closed is not None:
                trades[m].append(closed)
                closed_now.add(m)

        # ---- 2) 本步事件(四种口径共用) ----
        fresh = m6.step(now, c)
        for q in fresh:
            tf, kd = q["tf"], q["kind"]
            counts["points"][tf][kd] = counts["points"][tf].get(kd, 0) + 1

        # ---- 3) 找新信号 ----
        if not fresh:
            continue
        a = lv.atr()
        for m in entry_modes:
            if pos[m] is not None or m in closed_now:
                continue
            hit = decide(fresh, m6.active, m)
            if hit is None:
                continue
            trig, partner = hit
            combo = pair_label(trig, partner)
            diags[m].n_seen[combo] = diags[m].n_seen.get(combo, 0) + 1
            counts["pairs"][(m, combo)] = counts["pairs"].get((m, combo), 0) + 1
            spx, rk = struct_of(trig, partner)
            pl = risk.plan("long", c, spx, a, market, rk)
            if not pl.ok:
                diags[m].n_reject += 1
                diags[m].n_blocked[combo] = (diags[m].n_blocked.get(combo, 0) + 1)
                continue
            feats = {"trig_kind": trig["kind"], "trig_tf": trig["tf"],
                     "pt_kind": (partner["kind"] if partner else ""),
                     "pt_tf": (partner["tf"] if partner else ""),
                     "combo": combo, "mode": m,
                     "pt_lag_h": ((now - partner["known"]) / D.MS_HOUR
                                  if partner else 0.0)}
            sig = SimpleNamespace(
                sym=sym, side="long", kind=combo, stop=pl.stop,
                target=pl.target, atr=pl.atr, risk_pct=pl.risk_pct,
                notional=pl.notional, trend="",
                counter=int(trig["kind"] == B1
                            or (partner is not None and partner["kind"] == B1)),
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


def run_all(syms, market, tfs, win, invalidate, entry_modes, exit_mode,
            t0, t1, jobs, label=""):
    tasks = [(s, market, tfs, win, invalidate, entry_modes, exit_mode, t0, t1)
             for s in syms]
    tr = {m: [] for m in entry_modes}
    dg = {m: [] for m in entry_modes}
    ct = {"points": {}, "pairs": {}}

    def take(sym, t_, d_, c_):
        for m in entry_modes:
            tr[m].extend(t_[m])
            dg[m].append(d_[m])
        for tf, kd in (c_.get("points") or {}).items():
            ct["points"].setdefault(tf, {})
            for k, v in kd.items():
                ct["points"][tf][k] = ct["points"][tf].get(k, 0) + v
        for k, v in (c_.get("pairs") or {}).items():
            ct["pairs"][k] = ct["pairs"].get(k, 0) + v

    if jobs and jobs > 1:
        import multiprocessing as mp
        with mp.Pool(int(jobs)) as pool:
            for k, (sym, t_, d_, c_) in enumerate(
                    pool.imap_unordered(_mp_run, tasks, chunksize=1)):
                take(sym, t_, d_, c_)
                if (k + 1) % 25 == 0:
                    print(f"  {label}{k + 1}/{len(syms)} 笔="
                          f"{len(tr[entry_modes[0]])}", file=sys.stderr, flush=True)
    else:
        for k, t in enumerate(tasks):
            sym, t_, d_, c_ = _mp_run(t)
            take(sym, t_, d_, c_)
            if (k + 1) % 25 == 0:
                print(f"  {label}{k + 1}/{len(syms)}", file=sys.stderr, flush=True)
    for m in entry_modes:
        dg[m].sort(key=lambda d: d.sym)
    return tr, dg, ct


# ------------------------------------------------------------------ 自检
def _selfcheck() -> int:
    ok_all = [0]

    def ck(cond, msg):
        print(f"  [{'OK' if cond else 'FAIL'}] {msg}")
        if not cond:
            ok_all[0] += 1

    print("A) 周期扩展与聚合正确性")
    for tf, ms in (("3d", X_TF_MS["3d"]), ("1w", X_TF_MS["1w"]),
                   ("1M", X_TF_MS["1M"])):
        ck(D.TF_MS.get(tf) == ms, f"A1 TF_MS 含 {tf} = {ms//D.MS_DAY} 天")
    kl1d = D.load("spot", "BTCUSDT", "1d")
    ck(kl1d is not None and kl1d.n > 1000, f"A2 BTCUSDT 1d 有数据({kl1d.n if kl1d else 0} 根, 归档约 5.7 年)")
    for tf in ("3d", "1w", "1M"):
        kl = load_tf("spot", "BTCUSDT", tf)
        good = kl is not None and kl.n >= LEVEL_MIN[tf] and kl.n <= kl1d.n
        steps = ({int(b) - int(a) for a, b in zip(kl.ts, kl.ts[1:])}
                 if kl is not None and kl.n > 1 else set())
        mult = all(s % X_TF_MS[tf] == 0 for s in steps) if steps else True
        align = all(int(t) % X_TF_MS[tf] == 0 for t in kl.ts[:20]) if kl is not None else False
        ck(good and mult and align,
           f"A3 {tf}: n={0 if kl is None else kl.n} 根(≥{LEVEL_MIN[tf]}), "
           f"桶对齐且步长为 {tf} 的正整数倍")

    print("B) 窗口语义(prune)")
    now = 10 * D.MS_HOUR
    q1h = {"tf": "1h", "kind": B2, "price": 100.0, "known": now - 2 * D.MS_HOUR, "ts": 0}
    q4h = {"tf": "4h", "kind": B3, "price": 90.0, "known": now - 3 * D.MS_HOUR, "ts": 0}
    q1w = {"tf": "1w", "kind": B1, "price": 80.0,
           "known": now - 4 * X_TF_MS["1w"], "ts": 0}
    alive = prune([q1h, q4h, q1w], now, 100.0, 3, True)
    ck(len(alive) == 2 and q1h in alive and q4h in alive,
       "B1 1h(已过2根)与4h(3根, 窗口3×4h)存活, 周线点(4根, 窗口3×1w)过期")
    alive2 = prune([q1h, q4h], now, 95.0, 3, True)
    ck(q4h in alive2 and q1h not in alive2,
       "B2 破位: close 95 < 1h点 100 => 1h 点作废; 95 > 4h点 90 => 4h 点存活")
    alive2b = prune([q1h, q4h], now, 89.0, 3, True)
    ck(not alive2b, "B2' close 89 同时低于两点 => 全部作废")
    alive3 = prune([q4h], now, 89.0, 3, False)
    ck(q4h in alive3, "B3 --no-invalidate 时破位不剔除(只看时间窗)")

    print("C) 共振判定(decide)")
    f1h = {"tf": "1h", "kind": B2, "price": 100.0, "known": now, "ts": 5}
    ck(decide([f1h], [f1h], "any2") is None, "C1 单级别无配合 => any2 无信号")
    r = decide([f1h], [f1h, q4h], "any2")
    ck(r is not None and r[1] is q4h, "C2 1h 触发 + 4h 配合 => any2 信号")
    r = decide([f1h], [f1h, q4h], "any2_b1")
    ck(r is None, "C3 B2+B3 无 B1 => any2_b1 拒绝")
    fb4 = {"tf": "4h", "kind": B1, "price": 90.0, "known": now, "ts": 4}
    r = decide([f1h], [f1h, fb4], "any2_b1")
    ck(r is not None and r[1]["kind"] == B1, "C4 配合点是 B1 => any2_b1 通过")
    f1M = {"tf": "1M", "kind": B3, "price": 50.0, "known": now, "ts": 3}
    r = decide([f1M, f1h], [f1M, f1h], "any2")
    ck(r is not None and r[0]["tf"] == "1h", "C5 触发优先细级别(1h 先于 1M)")
    ck(decide([f1M], [f1M], "s1h_only") is None, "C6 s1h_only 拒绝非 1h 触发")
    ck(decide([f1h], [f1h], "s1h_only") is not None, "C7 s1h_only 接受 1h 触发")
    ck(pair_label(f1h, fb4) == "B2@1h+B1@4h", "C8 组合标签按级别排序")
    spx, rk = struct_of(f1M, f1h)
    ck(abs(spx - 100.0) < 1e-9 and rk == B2, "C9 止损取较细级别(1h)点的价格与种类")

    print("D) 截断一致性(无未来函数, 真数据端到端)")
    tfs = tuple(DEFAULT_TFS.split(","))
    tr_full, dg_full, _ = run_symbol("BTCUSDT", "spot", tfs, 3, True,
                                     ("any2",), "chan1_stop")
    klh = D.load("spot", "BTCUSDT", "1h")
    cut = int(klh.ts[int(klh.n * 0.6)])
    tr_cut, _, _ = run_symbol("BTCUSDT", "spot", tfs, 3, True,
                              ("any2",), "chan1_stop", 0, cut)
    # 截断跑会把"截断时仍持仓"的单子以 end_of_data 强平 —— 与全量跑不可比, 剔除。
    cut_clean = [t for t in tr_cut["any2"] if t.reason != "end_of_data"]
    n_cut = sum(1 for t in tr_full["any2"] if t.exit_ts <= cut)
    same = len(cut_clean) == n_cut
    for x, y in zip(tr_full["any2"], cut_clean):
        same = same and (x.entry_ts == y.entry_ts and x.exit_ts == y.exit_ts
                         and abs(x.entry - y.entry) < 1e-12
                         and abs(x.exit - y.exit) < 1e-12
                         and x.reason == y.reason and x.kind == y.kind)
    ck(same, f"D1 全量 {len(tr_full['any2'])} 笔, 截断点前 {n_cut} 笔逐字段一致"
             f"(截断跑出可比 {len(cut_clean)} 笔)")

    print("E) 级别资格")
    m6 = Multi6("spot", "BTCUSDT", tfs, 3, True)
    miss = [tf for tf in tfs if tf not in m6.lvls]
    ck(not miss, f"E1 BTCUSDT 六级别全部达资格(缺 {miss})")
    ck(all(m6.lvls[tf].cur == 0 for tf in m6.lvls), "E2 初始未推进(cur=0)")

    print("F) 多口径隔离性(回归: SellScanner 有状态, 不可跨口径共用)")
    # 2026-10-02 实测到的 bug: 四个口径共用同一个 SellScanner 时, 同 bar 内先被
    # 遍历到的口径会吃掉卖点, 后面的口径漏掉 S1 离场。此处用 s1h_only 做对照:
    # 单独跑 vs 与另外三个口径同时跑, 结果必须逐字段一致。
    iso_ok, iso_msg = True, []
    for sym in ("BTCUSDT", "ETHUSDT"):
        solo = run_symbol(sym, "spot", tfs, 3, True, ("s1h_only",),
                          "chan1_stop")[0]["s1h_only"]
        quad = run_symbol(sym, "spot", tfs, 3, True,
                          ("any2", "any2_b1", "any1", "s1h_only"),
                          "chan1_stop")[0]["s1h_only"]
        if len(solo) != len(quad):
            iso_ok = False
        else:
            for x, y in zip(solo, quad):
                if (x.entry_ts != y.entry_ts or x.exit_ts != y.exit_ts
                        or x.reason != y.reason):
                    iso_ok = False
                    break
        iso_msg.append(f"{sym} 单独{len(solo)}/共存{len(quad)}")
    ck(iso_ok, f"F1 单口径与四口径共存结果一致 ({'; '.join(iso_msg)})")
    print(f"\n自检结束: {'全部通过' if ok_all[0] == 0 else '存在失败'}")
    return 0 if ok_all[0] == 0 else 1


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=["spot", "um"])
    ap.add_argument("--tfs", default=DEFAULT_TFS)
    ap.add_argument("--win", type=int, default=3)
    ap.add_argument("--no-invalidate", action="store_true")
    ap.add_argument("--entry", default="any2,any2_b1,any1,s1h_only")
    ap.add_argument("--exit-mode", default="chan1_stop")
    ap.add_argument("--start", default="")
    ap.add_argument("--end", default="")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sym", default="")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--out", default="out/reso6_spot.txt")
    a = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    if a.selfcheck:
        return _selfcheck()

    tfs = tuple(t.strip() for t in a.tfs.split(",") if t.strip())
    bad = [t for t in tfs if t not in TF_ORDER]
    if bad:
        print(f"未知周期 {bad}; 可用 {list(TF_ORDER)}")
        return 2
    entries = tuple(m.strip() for m in a.entry.split(",") if m.strip())
    bad = [m for m in entries if m not in ENTRY_MODES]
    if bad:
        print(f"未知入场口径 {bad}; 可用 {list(ENTRY_MODES)}")
        return 2
    t0 = ms_of(a.start) if a.start else 0
    t1 = ms_of(a.end, end=True) if a.end else 0
    syms = [a.sym] if a.sym else D.symbols(a.market)
    if a.limit:
        syms = syms[:a.limit]

    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(str(s))
        print(s, flush=True)

    emit(f"# 六级别任意两共振  {a.market}  {'/'.join(tfs)}  "
         f"窗口 {a.win} 根本级别K线  出口 {a.exit_mode}  ({len(syms)} 个标的)")
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"# 口径: 信号=任一级别新买点出现时另有不同级别买点在窗内 · "
         f"止损/风险种类取较细级别 · 月线=固定30天桶 · 预热1h前{max(WARMUP, MIN_BARS)}根 · "
         f"多进程 {a.jobs}")
    emit("")
    emit(f"[{time.strftime('%H:%M:%S')}] 遍历 6 级别 × {len(syms)} 标的 × "
         f"{len(entries)} 入场口径 ...")
    ta = time.time()
    res, diags, ct = run_all(syms, a.market, tfs, a.win, not a.no_invalidate,
                             entries, a.exit_mode, t0, t1, a.jobs)
    emit(f"# 耗时 {time.time() - ta:.0f}s")
    emit("")

    # ---- 主表(五读数) ----
    emit("=" * 118)
    emit(f"主表(入口={a.exit_mode}; 「单笔净」= 名义额加权毛 − 双边费; "
         f"「折算年化」= 每持仓日×365×在仓占比)")
    emit(BT.HDR)
    for m in entries:
        r = BT.stats_row(res[m], a.market, "1h", diags[m], m)
        emit(BT.fmt_row(r) + ("" if r.get("n") else ""))
    emit("")

    # ---- 消融配对 ----
    emit("消融配对(分标的符号检验, 只数严格大于; min_n=3)")
    for x, y in (("any2", "s1h_only"), ("any2_b1", "s1h_only"),
                 ("any2", "any1"), ("any2_b1", "any2")):
        if x in entries and y in entries:
            ra = BT.stats_row(res[x], a.market, "1h", diags[x], x)
            rb = BT.stats_row(res[y], a.market, "1h", diags[y], y)
            if ra.get("n") and rb.get("n"):
                n, b, p = BT.signed_pairs(ra["by_sym"], rb["by_sym"])
                emit(f"  {x} vs {y}: 胜 {n} / 负 {b} / 配对率 {p:.0%}")
            else:
                emit(f"  {x} vs {y}: (有一侧无成交)")
    emit("")

    # ---- 结构层: 各级别原始买点量 ----
    emit("结构层(全池原始买点命中, 与入场口径无关):")
    for tf in tfs:
        kd = ct["points"].get(tf) or {}
        tot = sum(kd.values())
        emit(f"  * {tf:>3}: " + "  ".join(f"{k}={v:,}" for k, v in sorted(kd.items()))
             + f"   合计 {tot:,}")
    emit("")

    # ---- 共振组合分布 ----
    for m in ("any2", "any2_b1"):
        if m not in entries or not res[m]:
            continue
        emit(f"【{m}】按级别组合拆分(前 20 个, 按笔数)")
        pairs = {}
        for t in res[m]:
            k = (t.feats or {}).get("combo", t.kind)
            pairs.setdefault(k, []).append(t)
        rows = sorted(pairs.items(), key=lambda kv: -len(kv[1]))[:20]
        emit(f"{'组合':<28} {'笔数':>6} {'胜率':>7} {'单笔毛':>9} {'PF':>6} {'均持仓':>9}")
        for k, sub in rows:
            s = BT.R.summarize(sub, a.market)
            if not s["n"]:
                continue
            emit(f"{k:<28} {s['n']:>6} {s['win_rate'] * 100:>6.1f}% "
                 f"{s['exp_pct'] * 100:>+8.3f}% {s['pf']:>6.2f} "
                 f"{s['hold_min'] / 1440:>7.2f}天")
        emit("")

    # ---- 离场/分年/风险(主打 any2) ----
    pm = "any2" if "any2" in entries else entries[0]
    if res[pm]:
        emit(f"【{pm}】离场原因分布")
        for ln in BT.R.by_reason(res[pm], a.market):
            emit("  " + ln)
        emit("")
        emit(f"【{pm}】按年(单笔毛期望)")
        for ln in BT.R.by_year(res[pm], a.market):
            emit("  " + ln)
        emit("")
        emit(f"风险画像(每笔等权 1 单位; 非真实账户回撤)")
        for m in entries:
            rk = risk_block(res[m])
            emit(f"  {m:<10} 最好 {rk['best'] * 100:+.2f}%  最差 {rk['worst'] * 100:+.2f}%  "
                 f"累加最大回撤 {rk['mdd'] * 100:+.2f}%  最长连亏 {rk['max_lose']} 笔  "
                 f"累加总和 {rk['final']:+.1f}")
        emit("")

    # ---- 基准 ----
    pi = BT.pool_index(syms, "1h", a.market, 0, 0, rebal_bars=24)
    if pi:
        emit("基准(同池、逐日再平衡等权指数)")
        emit(f"  全程 {pi['total'] * 100:+.1f}%  每日 {pi['per_day'] * 100:+.4f}%  "
             f"年化 {pi['ann'] * 100:+.1f}%")
        emit("")

    # ---- 池子诊断 ----
    emit("池子诊断:")
    dl = diags[pm]
    okd = [d for d in dl if not d.err]
    emit(f"  * 可用标的 {len(okd)}/{len(dl)};  1h 笔合计 {sum(d.n_bi for d in okd):,}")
    emit(f"  * 入场成交 {sum(d.n_signals for d in okd):,};  "
         f"被 risk.plan 拒单 {sum(d.n_reject for d in okd):,}")
    emit("")

    _flush(a.out, out)
    print(f"\n报告已写入 {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
