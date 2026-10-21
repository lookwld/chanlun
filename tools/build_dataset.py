#!/usr/bin/env python
"""构建"入场判别建模"数据集 —— 把回测成交记录 + 日线上下文拍平成样本表。

样本口径 = **信号层候选池**: `Strategy` 判定可下单的每一次开仓全部纳入, 不受
资金/名额/熔断约束(与 A/B 同口径, 见 DESIGN §5)。理由: 组合层结果会被"资金不足
冻停"抹平, 那样学出来的是资金曲线而不是策略。

标签(三选一按需用, 本脚本一次全给):
  * `y_stop` —— 是否被止损打掉(`reason == "stop_loss"`)。**主标签**。
  * `y_win`  —— 是否盈利(`pnl_pct > 0`)。比"是否止损"更贴近实际盈亏, 因为
                结构破位离场也可能是亏损的。
  * `R`      —— 以初始风险为单位的收益 = `pnl_pct / risk_pct`。回归/排序用。

⚠ **防泄漏**: 以下字段是离场之后才知道的, 只能当标签, 绝不能当特征 ——
  `exit_ts` / `pnl_pct` / `reason` / `hold_min` / `R` / `y_stop` / `y_win` / `censored`。
  日线上下文一律只用"入场时刻**之前**已收盘"的日线, 由 `bisect_right(ts, entry-1d)` 保证。

用法:
  $PY tools/build_dataset.py --markets um,spot                     # 默认 all 变体 + 区间套旁路
  $PY tools/build_dataset.py --markets spot --no-confirm           # 快速版(不采区间套特征)
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from chanbook import data as D                      # noqa: E402
from chanbook.engine import run_all                 # noqa: E402
from chanbook.points import B1, B2, B3              # noqa: E402
from chanbook.strategy import DIR_GATE, BookCfg     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out", "dataset")

RANGE = {"um": ("2020-01", "2026-08"), "spot": ("2017-08", "2026-08")}

# 结果字段 —— 训练时必须排除
LABELS = ("y_stop", "y_win", "R", "censored")
RESULTS = ("exit_ts", "pnl_pct", "reason", "hold_min") + LABELS
META = ("sym", "market", "source", "entry_ts", "entry_time", "split")


# --------------------------------------------------------------- 日线上下文
_SC: dict = {}


def _sma(a: np.ndarray, n: int) -> np.ndarray:
    out = np.full(a.size, np.nan)
    if a.size < n:
        return out
    cs = np.cumsum(np.insert(a, 0, 0.0))
    out[n - 1:] = (cs[n:] - cs[:-n]) / n
    return out


def _wilder_atr(h, l, c, n: int = 14) -> np.ndarray:
    m = c.size
    out = np.full(m, np.nan)
    if m < n + 1:
        return out
    tr = np.empty(m)
    tr[0] = h[0] - l[0]
    pc = c[:-1]
    tr[1:] = np.maximum(h[1:] - l[1:],
                        np.maximum(np.abs(h[1:] - pc), np.abs(l[1:] - pc)))
    a = float(tr[1:n + 1].mean())
    out[n] = a
    for i in range(n + 1, m):
        a = (a * (n - 1) + tr[i]) / n
        out[i] = a
    return out


def _rank(a: np.ndarray, win: int = 250, min_n: int = 30) -> np.ndarray:
    """滚动分位: 当前值在**最近 win 个有效值**中的位置(0~1)。只用历史 ⇒ 无未来函数。"""
    m = a.size
    out = np.full(m, np.nan)
    for i in range(m):
        if np.isnan(a[i]):
            continue
        seg = a[max(0, i - win + 1):i + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size < min_n:
            continue
        out[i] = float((seg <= a[i]).mean())
    return out


def daily_table(market: str, sym: str):
    """返回 (ts_list, {列名: np.ndarray}) —— 日线上下文。um 无 1d 档, 由 1h 聚合。"""
    key = (market, sym)
    hit = _SC.get(key)
    if hit is not None:
        return hit
    kl = D.load(market, sym, "1d")
    c = np.asarray(kl.c, dtype=float)
    h = np.asarray(kl.h, dtype=float)
    lo = np.asarray(kl.l, dtype=float)
    ma20, ma60 = _sma(c, 20), _sma(c, 60)
    atr = _wilder_atr(h, lo, c, 14)
    with np.errstate(invalid="ignore", divide="ignore"):
        atrp = atr / c
        ret20 = np.full(c.size, np.nan)
        if c.size > 20:
            ret20[20:] = c[20:] / c[:-20] - 1.0
        slope = np.full(c.size, np.nan)
        if c.size > 10:
            slope[10:] = ma20[10:] / ma20[:-10] - 1.0
    tab = {
        "d_close": c,
        "d_ma20_dev": c / ma20 - 1.0,
        "d_ma60_dev": c / ma60 - 1.0,
        "d_atr_pct": atrp,
        "d_atr_rank": _rank(atrp),
        "d_ret20": ret20,
        "d_ma20_slope": slope,
    }
    hit = (list(kl.ts), tab)
    if len(_SC) > 64:
        _SC.clear()
    _SC[key] = hit
    return hit


def _at(ts: list, tab: dict, entry_ts: int) -> dict:
    """取入场时刻**之前最后一根已收盘日线**的上下文。"""
    k = bisect.bisect_right(ts, entry_ts - D.MS_DAY) - 1
    if k < 0:
        return {name: None for name in tab}
    out = {}
    for name, arr in tab.items():
        v = arr[k]
        out[name] = None if (v is None or (isinstance(v, float) and np.isnan(v))) \
            else round(float(v), 8)
    return out


# --------------------------------------------------------------- 样本构造
def to_row(t: dict, market: str, source: str, tab: dict, ts_all: list) -> dict:
    feat = dict(t.get("feats") or {})
    why = str(feat.pop("confirm_why", "") or "")
    row = {
        "sym": t.get("sym", ""), "market": market, "source": source,
        "entry_ts": int(t.get("entry_ts", 0)),
        "entry_time": D.fmt(int(t.get("entry_ts", 0))),
        # ---- 入场时已知的特征 ----
        "side": t.get("side", ""), "kind": t.get("kind", ""),
        "trend": t.get("trend", ""), "regime": t.get("regime", ""),
        "counter": int(t.get("counter", 0)),
        "risk_pct": round(float(t.get("risk_pct", 0.0)), 8),
        "notional": float(t.get("notional", 0.0)),
        "confirm": why[:120],
    }
    row.update(feat)
    # ---- 日线上下文 ----
    row.update(_at(ts_all, tab, row["entry_ts"]))
    # ---- 结果(标签) ----
    rp = float(t.get("risk_pct", 0.0)) or 0.0
    pnl = float(t.get("pnl_pct", 0.0))
    reason = str(t.get("reason", ""))
    row.update({
        "exit_ts": int(t.get("exit_ts", 0)),
        "pnl_pct": round(pnl, 8),
        "reason": reason,
        "hold_min": int(t.get("hold_min", 0)),
        "R": round(pnl / rp, 4) if rp > 0 else None,
        "y_stop": int(reason == "stop_loss"),
        "y_win": int(pnl > 0),
        "censored": int(reason == "end_of_data"),
    })
    return row


def ms_of(ym: str, end: bool = False) -> int:
    """'2020-01' -> 该月1日 00:00 的毫秒(**UTC**); end=True -> 该月末最后一毫秒。

    ⚠ 必须与 `tools/run_ab.py::ms_of` **完全同源**。曾用 UTC+8 算过一次, 区间两端
    各偏 8 小时, 样本数就和 A/B 基线差 3 笔 —— 数据集与基线不是同一批交易, 后续
    一切对照都失去意义。
    """
    y, m = (int(x) for x in ym.split("-"))
    if end:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1
    return int(datetime(y, m, 1, tzinfo=timezone.utc).timestamp() * 1000)


def build(market: str, source: str, cfg: BookCfg, start: str, end: str,
          jobs: int, limit: int, cache_dir: str = ""):
    ts_a, ts_b = ms_of(start), ms_of(end, end=True)
    trades, diags = run_all(market, cfg, jobs=jobs, limit=limit, progress=40,
                            t_start=ts_a, t_end=ts_b)
    if cache_dir:
        # 落一份与 run_ab 同格式的缓存 ⇒ 可用 tools/cmp_cache.py 核对
        # "record_confirm(加载确认级别)不改变任何信号"这条论断。
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{market}_allconf.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"tr": [t.to_dict() for t in trades], "ns": {}}, fh,
                      ensure_ascii=False)
    rows = []
    for tr in trades:
        d = tr.to_dict()
        ts_all, tab = daily_table(market, d["sym"])
        rows.append(to_row(d, market, source, tab, ts_all))
    return rows, diags


def _split(rows: list) -> None:
    """按**时间**切分(70/30)。绝不能随机切 —— 同一标的相邻交易高度相关, 随机切会泄漏。"""
    ts = sorted(r["entry_ts"] for r in rows)
    if not ts:
        return
    cut = ts[int(len(ts) * 0.7)]
    for r in rows:
        r["split"] = "train" if r["entry_ts"] < cut else "valid"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", default="um,spot")
    ap.add_argument("--source", default="all")
    ap.add_argument("--no-confirm", action="store_true",
                    help="不采集区间套特征(省掉加载确认级别, 快约 2.4 倍)")
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "out", "cache_verify"),
                    help="落一份 run_ab 同格式缓存供 cmp_cache 核对(空串=不落)")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    for market in [x.strip() for x in a.markets.split(",") if x.strip()]:
        # ⚠ 显式 legacy: 数据集要能与既有模型基线逐位对比, 离场口径不能漂移。
        cfg = BookCfg.legacy(market=market, kinds=(B1, B2, B3),
                             dir_gate=dict(DIR_GATE),
                             record_confirm=not a.no_confirm)
        start, end = RANGE[market]
        print(f"[{market}] source={a.source} confirm={cfg.record_confirm} "
              f"{start}~{end}", flush=True)
        rows, diags = build(market, a.source, cfg, start, end, a.jobs, a.limit,
                            a.cache_dir)
        _split(rows)
        path = os.path.join(a.out_dir, f"{market}_{a.source}.jsonl")
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        ok = sum(1 for d in diags if not d.err)
        print(f"[{market}] 标的 {len(diags)}(成功 {ok}) 样本 {len(rows)} → {path}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
