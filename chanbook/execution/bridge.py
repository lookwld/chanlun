"""信号桥接 —— chanbook Signal/Trade <-> 执行层字段映射。

执行层(execution.*)不直接 import chanbook 的策略内部结构, 而是通过本模块的
纯函数把 chanbook 的 `Signal`(入场) 与 `Trade`(离场) 映射成执行层所需的 dict。
这样执行层与策略层解耦, 且保证"复刻执行层"不会反向牵动交易逻辑。

映射口径:
  * Signal.sym  -> symbol
  * Signal.kind -> grade(执行层沿用 chanbook 的 B1/B2/B3 语义)
  * Signal.side -> side(long/short)
  * Signal.entry/stop/target/notional -> entry/stop/target/notional
  * Trade.reason  -> exit_reason
  * Trade.pnl_pct -> pnl_pct
"""

from __future__ import annotations

__all__ = ["signal_to_order", "trade_to_close", "close_record"]


def signal_to_order(sig) -> dict:
    """chanbook Signal -> 执行层开仓指令 dict。"""
    return {
        "symbol": sig.sym,
        "side": sig.side,
        "grade": sig.kind,          # B1/B2/B3
        "kind": sig.kind,
        "entry": float(sig.entry),
        "stop": float(sig.stop),
        "target": float(sig.target),
        "notional": float(sig.notional),
        "atr": float(sig.atr) if sig.atr else 0.0,
        "trend": getattr(sig, "trend", ""),
        "ts": int(sig.ts),
        "reasons": list(getattr(sig, "reasons", []) or []),
    }


def trade_to_close(tr) -> dict:
    """chanbook Trade -> 执行层平仓记录 dict。"""
    return {
        "symbol": tr.sym,
        "side": tr.side,
        "kind": tr.kind,
        "entry": float(tr.entry),
        "exit": float(tr.exit),
        "pnl_pct": float(tr.pnl_pct),
        "exit_reason": tr.reason,
        "opened_at": float(tr.entry_ts),
        "closed_at": float(tr.exit_ts),
        "trend": tr.trend,
    }


def close_record(symbol: str, grade: str, side: str, entry: float, stop: float,
                 target: float, qty: float, opened_at: float, exit: float,
                 pnl_pct: float, exit_reason: str, closed_at: float,
                 reasons: list | None = None, kind: str = "", trend: str = "") -> dict:
    """构造 Analyzer.record_closed 所需的平仓记录 dict。"""
    return {
        "symbol": symbol, "grade": grade, "side": side,
        "entry": entry, "stop": stop, "target": target, "qty": qty,
        "opened_at": opened_at, "exit": exit, "pnl_pct": pnl_pct,
        "exit_reason": exit_reason, "closed_at": closed_at,
        "reasons": reasons or [], "kind": kind or grade, "trend": trend,
    }
