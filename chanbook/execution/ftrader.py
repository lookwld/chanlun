"""合约执行器 —— 复刻 `chan-quant/app/ftrader.py` 的执行闭环(不含入场逻辑)。

与 SpotTrader 的差异(与 chan-quant 对齐):
  * 独立保证金池(futures.trading_capital), 杠杆 leverage。
  * 支持双向(side = long/short)。
  * 方向保险丝(direction_ok / on_close_dir)接入开仓校验。

⚠ 边界: 同样**不做信号生成**; 离场仅"灾难止损 + 手动", 缠论结构离场由策略层
   口径决定, 不在本层重新实现。
"""

from __future__ import annotations

import time

from .account import Account, account_plan
from .analyzer import Analyzer
from .bridge import close_record
from .broker import Broker, get_broker
from .llm import LLMFilter
from ..guards import Guards

__all__ = ["FuturesTrader"]


class FuturesTrader:
    def __init__(self, cfg: dict, broker: Broker | None = None,
                 analyzer: Analyzer | None = None) -> None:
        self.cfg = cfg
        fcfg = cfg.get("futures", {}) or {}
        # 合约独立资金池
        _cfg = dict(cfg)
        _cfg["capital"] = {"trading_capital": float(fcfg.get("trading_capital", 10.0)),
                           "max_pairs": int(cfg.get("capital", {}).get("max_pairs", 10))}
        self.broker = broker or get_broker(_cfg)
        self.account = Account(_cfg, name="futures")
        self.analyzer = analyzer or Analyzer({"db_path": cfg.get("db_path", "data/trades_futures.db")})
        self.guards = Guards(cfg, name="futures", persist=bool(cfg.get("state_path")))
        self.llm = LLMFilter(cfg)
        self.leverage = int(fcfg.get("leverage", 3))
        self.positions: dict[str, dict] = {}

    def open_count(self) -> int:
        return len(self.positions)

    def try_open(self, order: dict) -> dict:
        sym = order["symbol"]
        side = order.get("side", "long")
        entry = order["entry"]
        stop = order["stop"]
        notional = order["notional"]

        ok, why = self.guards.can_open(sym)
        if not ok:
            return {"opened": False, "reason": why}
        ok, why = self.guards.direction_ok(side)
        if not ok:
            return {"opened": False, "reason": why}

        plan = account_plan(self.account.equity(), entry, stop, self.cfg,
                            self.open_count(), deployed=self.account.deployed(),
                            min_notional=self.broker.min_notional(sym),
                            pool=self.account.pool)
        if not plan.accepted:
            return {"opened": False, "reason": plan.reason}

        allow, _why = self.llm.filter(order)
        if not allow:
            return {"opened": False, "reason": "llm_reject"}

        qty = plan.qty
        price = self.broker.fetch_price(sym) if self.broker.live else entry
        if side == "short":
            # 简化: 合约做空同样走 broker.sell(空头开仓), 模拟层按持仓负数处理
            res = self.broker.sell(sym, qty, price)
        else:
            res = self.broker.buy(sym, qty, price)
        if not res.get("ok"):
            return {"opened": False, "reason": res.get("reason", "下单失败")}

        fill = res.get("price", price or entry)
        self.account.reserve(plan.notional)
        self.positions[sym] = {
            "symbol": sym, "side": side, "qty": qty, "entry": fill, "stop": stop,
            "target": order["target"], "grade": order.get("grade", ""),
            "kind": order.get("kind", ""), "trend": order.get("trend", ""),
            "notional": plan.notional, "opened_at": time.time(),
            "reasons": order.get("reasons", []),
        }
        return {"opened": True, "symbol": sym, "side": side, "qty": qty, "price": fill}

    def try_close(self, symbol: str, price: float, reason: str) -> dict:
        pos = self.positions.get(symbol)
        if not pos:
            return {"closed": False, "reason": "无此持仓"}
        qty = pos["qty"]
        side = pos.get("side", "long")
        if side == "short":
            res = self.broker.buy(symbol, qty, price)   # 平空
        else:
            res = self.broker.sell(symbol, qty, price)
        if not res.get("ok"):
            return {"closed": False, "reason": res.get("reason", "平仓失败")}
        fill = res.get("price", price)
        if side == "short":
            pnl_pct = (pos["entry"] - fill) / pos["entry"] if pos["entry"] else 0.0
        else:
            pnl_pct = (fill - pos["entry"]) / pos["entry"] if pos["entry"] else 0.0
        pnl_usd = pos["notional"] * pnl_pct
        self.account.release(pos["notional"], pnl_usd)

        rec = close_record(
            symbol, pos["grade"], side, pos["entry"], pos["stop"], pos["target"],
            qty, pos["opened_at"], fill, pnl_pct, reason, time.time(),
            reasons=pos.get("reasons"), kind=pos.get("kind", ""), trend=pos.get("trend", ""))
        self.analyzer.record_closed(rec)
        self.guards.on_close(rec)
        self.guards.on_close_dir(side, pnl_pct)
        self.positions.pop(symbol, None)
        return {"closed": True, "symbol": symbol, "side": side,
                "pnl_pct": pnl_pct, "pnl_usd": pnl_usd, "reason": reason}

    def check_stops(self, prices: dict[str, float]) -> list[dict]:
        out = []
        for sym, pos in list(self.positions.items()):
            px = prices.get(sym)
            if px is None or pos.get("stop", 0) <= 0:
                continue
            side = pos.get("side", "long")
            hit = px <= pos["stop"] if side == "long" else px >= pos["stop"]
            if hit:
                out.append(self.try_close(sym, px, "stop_loss"))
        return out

    def close_all(self, prices: dict[str, float]) -> list[dict]:
        out = []
        for sym in list(self.positions.keys()):
            px = prices.get(sym)
            if px is not None:
                out.append(self.try_close(sym, px, "manual_close"))
        return out

    def snapshot(self) -> dict:
        return {
            "account": self.account.snapshot(),
            "leverage": self.leverage,
            "open_count": self.open_count(),
            "positions": [dict(p) for p in self.positions.values()],
        }
