"""现货执行器 —— 复刻 `chan-quant/app/trader.py` 的执行闭环(不含入场逻辑)。

职责: 接收 chanbook 信号 -> 熔断/资金/LLM 校验 -> 下单(或模拟) -> 持仓管理 ->
离场判定(复用 chanbook 已有的 sell/guards 口径) -> 记账 -> 对账。

⚠ 边界(铁律): 本模块**不做信号生成**, 不做买卖点判定。入场信号由 chanbook
   策略层产出, 经 `bridge.signal_to_order` 传入。离场口径由调用方注入的
   `exit_fn` 决定(默认仅"灾难止损 + 手动平仓"), 不在此重新实现缠论离场。
"""

from __future__ import annotations

import time

from .account import Account, account_plan
from .analyzer import Analyzer
from .bridge import close_record
from .broker import Broker, get_broker
from .llm import LLMFilter
from ..guards import Guards

__all__ = ["SpotTrader"]


class SpotTrader:
    def __init__(self, cfg: dict, broker: Broker | None = None,
                 analyzer: Analyzer | None = None) -> None:
        self.cfg = cfg
        self.broker = broker or get_broker(cfg)
        self.account = Account(cfg, name="spot")
        self.analyzer = analyzer or Analyzer(cfg)
        self.guards = Guards(cfg, name="spot", persist=bool(cfg.get("state_path")))
        self.llm = LLMFilter(cfg)
        self.positions: dict[str, dict] = {}   # symbol -> 持仓 dict
        self._pending: list[dict] = []

    # ------------------------------------------------------------------ 状态
    def _load(self):
        """dry_run 下从 SimExchange 对账; live 下从交易所对账。返回空即无历史。"""
        holdings = self.broker.exchange_holdings()
        for sym, qty in holdings.items():
            if sym not in self.positions:
                self.positions[sym] = {"symbol": sym, "qty": qty, "entry": 0.0,
                                       "stop": 0.0, "target": 0.0, "grade": "",
                                       "opened_at": time.time()}

    def open_count(self) -> int:
        return len(self.positions)

    # ------------------------------------------------------------------ 开仓
    def try_open(self, order: dict) -> dict:
        """尝试按信号开仓。返回 {opened, reason, ...}。"""
        sym = order["symbol"]
        entry = order["entry"]
        stop = order["stop"]
        notional = order["notional"]

        # 1. 熔断
        ok, why = self.guards.can_open(sym)
        if not ok:
            return {"opened": False, "reason": why}

        # 2. 资金/槽位
        plan = account_plan(self.account.equity(), entry, stop, self.cfg,
                            self.open_count(), deployed=self.account.deployed(),
                            min_notional=self.broker.min_notional(sym),
                            pool=self.account.pool)
        if not plan.accepted:
            return {"opened": False, "reason": plan.reason}

        # 3. LLM 筛选(可选)
        allow, _why = self.llm.filter(order)
        if not allow:
            return {"opened": False, "reason": "llm_reject"}

        # 4. 下单
        qty = plan.qty
        price = self.broker.fetch_price(sym) if self.broker.live else entry
        res = self.broker.buy(sym, qty, price)
        if not res.get("ok"):
            return {"opened": False, "reason": res.get("reason", "下单失败")}

        fill = res.get("price", price or entry)
        self.account.reserve(plan.notional)
        self.positions[sym] = {
            "symbol": sym, "qty": qty, "entry": fill, "stop": stop,
            "target": order["target"], "grade": order.get("grade", ""),
            "kind": order.get("kind", ""), "trend": order.get("trend", ""),
            "notional": plan.notional, "opened_at": time.time(),
            "reasons": order.get("reasons", []),
        }
        return {"opened": True, "symbol": sym, "qty": qty, "price": fill}

    # ------------------------------------------------------------------ 离场
    def try_close(self, symbol: str, price: float, reason: str) -> dict:
        """按给定价格/理由平仓。"""
        pos = self.positions.get(symbol)
        if not pos:
            return {"closed": False, "reason": "无此持仓"}
        qty = pos["qty"]
        res = self.broker.sell(symbol, qty, price)
        if not res.get("ok"):
            return {"closed": False, "reason": res.get("reason", "平仓失败")}
        fill = res.get("price", price)
        pnl_pct = (fill - pos["entry"]) / pos["entry"] if pos["entry"] else 0.0
        pnl_usd = pos["notional"] * pnl_pct
        self.account.release(pos["notional"], pnl_usd)

        rec = close_record(
            symbol, pos["grade"], "long", pos["entry"], pos["stop"], pos["target"],
            qty, pos["opened_at"], fill, pnl_pct, reason, time.time(),
            reasons=pos.get("reasons"), kind=pos.get("kind", ""), trend=pos.get("trend", ""))
        self.analyzer.record_closed(rec)
        self.guards.on_close(rec)
        self.positions.pop(symbol, None)
        return {"closed": True, "symbol": symbol, "pnl_pct": pnl_pct,
                "pnl_usd": pnl_usd, "reason": reason}

    def check_stops(self, prices: dict[str, float]) -> list[dict]:
        """按止损线检查持仓(仅灾难止损, 不实现缠论结构离场 —— 那是策略层口径)。"""
        out = []
        for sym, pos in list(self.positions.items()):
            px = prices.get(sym)
            if px is None or pos.get("stop", 0) <= 0:
                continue
            if px <= pos["stop"]:
                out.append(self.try_close(sym, px, "stop_loss"))
        return out

    def close_all(self, prices: dict[str, float]) -> list[dict]:
        out = []
        for sym in list(self.positions.keys()):
            px = prices.get(sym)
            if px is not None:
                out.append(self.try_close(sym, px, "manual_close"))
        return out

    # ------------------------------------------------------------------ 快照
    def snapshot(self) -> dict:
        return {
            "account": self.account.snapshot(),
            "open_count": self.open_count(),
            "positions": [dict(p) for p in self.positions.values()],
        }
