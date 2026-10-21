"""报告 —— 把成交记录聚合成可与旧引擎并排比较的指标。

指标口径与旧引擎 `backtest_long.seg_stats` / `portfolio` 对齐, 目的是让 A/B
两边**同一行数字含义相同**:

    笔数 / 胜率 / 单笔期望(pnl_pct 均值, 毛) / 均赢 / 均亏 / 均持仓
    名义净 U = Σ(notional × pnl_pct) − 名义 × 双边手续费 × 笔数
    单笔净期望 = 名义净 U / 笔数 / 名义

手续费: 合约 0.05%/边、现货 0.1%/边(与实盘 `config.yaml` 一致)。

**为什么单列"单笔期望"而不是只看净值**: 旧引擎的净值曲线是组合层结果, 会被
"资金不足冻停"抹平 —— 同一个策略在资金池打光前后的表现不可比。单笔期望是
信号层口径, 才是策略本身的度量。本项目四次过滤类实验都以此为判决依据。
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone

__all__ = ["summarize", "line", "by_kind", "by_year", "full_report",
           "save_trades", "load_trades", "TZ"]

TZ = timezone(timedelta(hours=8))
FEE = {"um": 0.0005, "spot": 0.001}


def _ym(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y-%m")


def _y(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, TZ).strftime("%Y")


def summarize(trades: list, market: str, notional: float | None = None) -> dict:
    n = len(trades)
    if not n:
        return {"n": 0}
    fee = FEE[market]
    tot_nm = sum(t.notional for t in trades) or 1.0
    nm = notional if notional is not None else tot_nm / n
    wins = [t for t in trades if t.pnl_pct > 0]
    loss = [t for t in trades if t.pnl_pct <= 0]
    sw = sum(t.pnl_pct for t in wins)
    sl = sum(t.pnl_pct for t in loss)
    aw = sw / len(wins) if wins else 0.0
    al = sl / len(loss) if loss else 0.0
    exp = sum(t.pnl_pct for t in trades) / n
    gross_u = sum(t.pnl_pct * t.notional for t in trades)
    fee_u = sum(t.notional * fee * 2 for t in trades)
    return {
        "n": n,
        "win_rate": len(wins) / n,
        "exp_pct": exp,
        # 名义额加权毛期望 —— 与净期望同基(单笔名义额不同时, 非加权期望不可与净期望相减)
        "w_exp_pct": gross_u / tot_nm,
        "avg_win": aw,
        "avg_loss": al,
        "rr": (aw / abs(al)) if al else 0.0,
        "pf": (sw / abs(sl)) if sl else 0.0,
        "hold_min": sum(t.hold_min for t in trades) / n,
        "gross_u": gross_u,
        "fee_u": fee_u,
        "net_u": gross_u - fee_u,
        "notional": nm,
        # 恒等式: 每笔成本 = 2 x fee x 名义额 => 净期望 = 加权毛期望 - 双边费
        "net_exp_pct": gross_u / tot_nm - fee * 2,
        "best": max(t.pnl_pct for t in trades),
        "worst": min(t.pnl_pct for t in trades),
        "be_win_rate": 1.0 / (1.0 + (aw / abs(al))) if al and aw else 0.0,
    }


def _notional(trades: list) -> float:
    """平均名义额(仅作缺省值)。

    不能用第一笔: 名义额按买点分档(一买走小仓特例 → 合约 7.5U / 现货 2.5U,
    二/三买 15U / 5U)。"三类全开"变体的第一笔若恰好是一买, 拿它当分母会把
    净期望的成本系统性放大(2026-09-27 实测 all: 报 -0.175%, 真值 -0.125%)。
    """
    tot = sum(t.notional for t in trades if t.notional)
    k = sum(1 for t in trades if t.notional)
    return (tot / k) if k else 1.0


def line(trades: list, market: str, label: str = "") -> str:
    """与旧 `seg_stats` 同格式的一行摘要。"""
    s = summarize(trades, market)
    if not s["n"]:
        return f"{label:>16} 无成交"
    return (f"{label:>16} {s['n']:>5}笔 胜率{s['win_rate'] * 100:5.1f}% "
            f"单笔期望{s['exp_pct'] * 100:+6.3f}% 净{s['net_u']:+8.2f}U "
            f"单笔净期望{s['net_exp_pct'] * 100:+6.3f}% "
            f"均赢{s['avg_win'] * 100:+5.2f}%/均亏{s['avg_loss'] * 100:+5.2f}% "
            f"盈亏比{s['rr']:4.2f} PF{s['pf']:4.2f} "
            f"均持{s['hold_min']:5.0f}min")


def by_kind(trades: list, market: str) -> list[str]:
    out = []
    for k in ("B1", "B2", "B3"):
        sub = [t for t in trades if t.kind == k]
        out.append(line(sub, market, k))
    return out


def by_year(trades: list, market: str) -> list[str]:
    ys = sorted({_y(t.entry_ts) for t in trades})
    out = []
    for y in ys:
        out.append(line([t for t in trades if _y(t.entry_ts) == y], market, y))
    return out


def by_reason(trades: list, market: str) -> list[str]:
    rs = sorted({t.reason for t in trades})
    return [line([t for t in trades if t.reason == r], market, r) for r in
            sorted(rs, key=lambda r: -len([t for t in trades if t.reason == r]))]


def by_regime(trades: list, market: str) -> list[str]:
    rs = sorted({t.regime for t in trades})
    return [line([t for t in trades if t.regime == r], market, r or "?")
            for r in rs]


def by_trend(trades: list, market: str) -> list[str]:
    rs = sorted({t.trend for t in trades})
    return [line([t for t in trades if t.trend == r], market, r or "?") for r in rs]


def full_report(market: str, label: str, trades: list, diags: list = ()) -> str:
    L = [f"===== {label} [{market}] ====="]
    L.append(line(trades, market, "合计"))
    s = summarize(trades, market)
    if s["n"]:
        L.append(f"{'':>16} 最好{s['best'] * 100:+6.2f}% 最差{s['worst'] * 100:+6.2f}% "
                 f"盈亏平衡胜率{s['be_win_rate'] * 100:5.1f}% "
                 f"均名义{s['notional']:.2f}U 手续费{s['fee_u']:.2f}U")
    L.append("-- 分买点 --")
    L.extend(by_kind(trades, market))
    L.append("-- 分行情(-- 归因用, 不参与判定) --")
    L.extend(by_regime(trades, market))
    L.append("-- 分大级别走势 --")
    L.extend(by_trend(trades, market))
    L.append("-- 分离场原因 --")
    L.extend(by_reason(trades, market))
    L.append("-- 分年 --")
    L.extend(by_year(trades, market))
    if diags:
        err = [d for d in diags if d.err]
        tot = sum(getattr(d, "n_op_bars", 0) for d in diags)
        L.append(f"-- 诊断 -- 标的{len(diags)}个(数据不足{len(err)}个) "
                 f"合计操作级别K线{tot}")
        seen: dict = {}
        blocked: dict = {}
        for d in diags:
            for k, v in (d.n_seen or {}).items():
                seen[k] = seen.get(k, 0) + v
            for k, v in (d.n_blocked or {}).items():
                blocked[k] = blocked.get(k, 0) + v
        if seen:
            L.append(f"   结构命中 {seen}  闸门拦掉 {blocked}  "
                     f"风控拒单 {sum(getattr(d, 'n_reject', 0) for d in diags)}")
        if err:
            L.append(f"   数据不足样例: {[d.sym for d in err[:8]]}")
    return "\n".join(L)


# ------------------------------------------------------------------ 落盘
def save_trades(path: str, trades: list, diags: list = ()) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tr": [t.to_dict() for t in trades],
                   "ns": {d.sym: d.to_dict() for d in diags}},
                  fh, ensure_ascii=False)


def load_trades(path: str) -> tuple[list, dict]:
    from .engine import Diag, Trade
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    tr = [Trade(**t) for t in d.get("tr", [])]
    ns = d.get("ns", {})
    diags = []
    for sym, v in ns.items():
        v = dict(v)
        v.pop("sym", None)
        diags.append(Diag(sym=sym, **v))
    return tr, diags
