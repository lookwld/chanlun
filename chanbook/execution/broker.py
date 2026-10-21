"""交易所适配层 —— 复刻 `chan-quant/app/trader.py`/`ftrader.py` 的交易所交互契约。

提供与 chan-quant 相同的方法语义(fetch_ohlcv / fetch_price / min_notional /
下单 / 对账 / 持仓), 但:
  * **不包含任何入场信号逻辑**(那是 chanbook 策略层的事)。
  * dry_run 模式走内存模拟撮合, live 模式走 ccxt 真实下单(需 API key + allow_live)。
  * 只做"下单执行"与"账户只读查询", 不做选币、不做信号生成。

⚠ 依赖 ccxt: `pip install ccxt`。dry_run 不需要 ccxt。
"""

from __future__ import annotations

__all__ = ["Broker", "SimExchange", "get_broker"]


class SimExchange:
    """dry_run 用的内存模拟交易所: 只记余额与持仓, 价格由调用方喂入。"""

    def __init__(self, initial_usdt: float = 10.0) -> None:
        self.usdt = float(initial_usdt)
        self.positions: dict[str, dict] = {}   # symbol -> {qty, entry}
        self._prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = float(price)

    def price(self, symbol: str, fallback: float | None = None) -> float | None:
        return self._prices.get(symbol, fallback)

    def buy(self, symbol: str, qty: float, price: float) -> dict:
        """买入。qty>0 开多/平空, 持仓 qty 增加(可为负->0->正)。"""
        cost = qty * price
        if cost > self.usdt + 1e-9:
            return {"ok": False, "reason": f"余额不足: 需{cost:.4f}U, 现有{self.usdt:.4f}U"}
        self.usdt -= cost
        p = self.positions.setdefault(symbol, {"qty": 0.0, "entry": 0.0})
        new_qty = p["qty"] + qty
        # 加权平均入场价(仅在同向加仓时更新; 平仓/反手时重设)
        if p["qty"] * new_qty >= 0 and new_qty != 0:
            p["entry"] = (p["entry"] * abs(p["qty"]) + price * qty) / abs(new_qty)
        else:
            p["entry"] = price
        p["qty"] = new_qty
        if abs(p["qty"]) <= 1e-12:
            self.positions.pop(symbol, None)
        self._prices[symbol] = price
        return {"ok": True, "symbol": symbol, "qty": qty, "price": price}

    def sell(self, symbol: str, qty: float, price: float) -> dict:
        """卖出。qty>0 平多/开空, 持仓 qty 减少(可为正->0->负)。"""
        self.usdt += qty * price
        p = self.positions.setdefault(symbol, {"qty": 0.0, "entry": 0.0})
        new_qty = p["qty"] - qty
        if p["qty"] * new_qty >= 0 and new_qty != 0:
            p["entry"] = (p["entry"] * abs(p["qty"]) + price * qty) / abs(new_qty)
        else:
            p["entry"] = price
        p["qty"] = new_qty
        if abs(p["qty"]) <= 1e-12:
            self.positions.pop(symbol, None)
        self._prices[symbol] = price
        return {"ok": True, "symbol": symbol, "qty": qty, "price": price}

    def balance_usdt(self) -> float:
        return self.usdt

    def holdings(self) -> dict[str, dict]:
        return {k: dict(v) for k, v in self.positions.items()}


class Broker:
    """交易所适配器。mode = dry_run | live。"""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.mode = cfg.get("mode", "dry_run")
        self.exchange_id = cfg.get("exchange", {}).get("id", "binance")
        self._ccxt = None
        if self.mode == "live":
            self._ccxt = self._make_ccxt(cfg)

    def _make_ccxt(self, cfg: dict):
        import ccxt  # 惰性导入, dry_run 不依赖
        ex = cfg.get("exchange", {})
        return ccxt.binance({
            "apiKey": ex.get("api_key") or __import__("os").environ.get("BQ_API_KEY", ""),
            "secret": ex.get("secret") or __import__("os").environ.get("BQ_SECRET", ""),
            "enableRateLimit": True,
        })

    @property
    def live(self) -> bool:
        return self.mode == "live"

    # ------------------------------------------------------------------ 行情
    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 300) -> list[dict]:
        """返回 [{"ts","open","high","low","close","volume"}...]。dry_run 返回空。"""
        if not self.live:
            return []
        data = self._ccxt.fetch_ohlcv(symbol, timeframe, limit=limit)
        return [{"ts": int(r[0]), "open": r[1], "high": r[2], "low": r[3],
                 "close": r[4], "volume": r[5]} for r in data]

    def fetch_price(self, symbol: str) -> float | None:
        if not self.live:
            return None
        t = self._ccxt.fetch_ticker(symbol)
        return float(t["last"]) if t.get("last") else None

    def min_notional(self, symbol: str) -> float:
        """交易所最小下单额。dry_run 硬编码 5.0(与 chan-quant 一致)。"""
        if not self.live:
            return 5.0
        try:
            m = self._ccxt.load_markets()
            limit = m.get(symbol, {}).get("limits", {}).get("cost", {}).get("min")
            return float(limit) if limit else 5.0
        except Exception:
            return 5.0

    # ------------------------------------------------------------------ 下单
    def buy(self, symbol: str, qty: float, price: float | None = None) -> dict:
        if not self.live:
            sim = getattr(self, "_sim", None)
            if sim is None:
                return {"ok": False, "reason": "dry_run 请通过 sim 属性注入 SimExchange"}
            return sim.buy(symbol, qty, price or sim.price(symbol, 0.0))
        order = self._ccxt.create_order(symbol, "market", "buy", qty)
        return {"ok": True, "order_id": order.get("id"), "qty": qty,
                "price": float(order.get("average") or order.get("price") or 0.0)}

    def sell(self, symbol: str, qty: float, price: float | None = None) -> dict:
        if not self.live:
            sim = getattr(self, "_sim", None)
            if sim is None:
                return {"ok": False, "reason": "dry_run 请通过 sim 属性注入 SimExchange"}
            return sim.sell(symbol, qty, price or sim.price(symbol, 0.0))
        order = self._ccxt.create_order(symbol, "market", "sell", qty)
        return {"ok": True, "order_id": order.get("id"), "qty": qty,
                "price": float(order.get("average") or order.get("price") or 0.0)}

    # ------------------------------------------------------------------ 查询
    def free_usdt(self) -> float | None:
        if not self.live:
            sim = getattr(self, "_sim", None)
            return sim.balance_usdt() if sim else None
        bal = self._ccxt.fetch_balance()
        return float(bal.get("USDT", {}).get("free", 0.0) or 0.0)

    def exchange_holdings(self) -> dict[str, float]:
        """symbol -> 持有数量。dry_run 读 SimExchange。"""
        if not self.live:
            sim = getattr(self, "_sim", None)
            return {k: v["qty"] for k, v in sim.holdings().items()} if sim else {}
        bal = self._ccxt.fetch_balance()
        out = {}
        for k, v in (bal.get("total") or {}).items():
            if float(v or 0) > 0 and k != "USDT":
                out[k] = float(v)
        return out


def get_broker(cfg: dict, sim: SimExchange | None = None) -> Broker:
    """构造 Broker。dry_run 时自动挂载 SimExchange(资金 = trading_capital)。"""
    b = Broker(cfg)
    if not b.live:
        capital = float(cfg.get("capital", {}).get("trading_capital", 10.0))
        b._sim = sim if sim is not None else SimExchange(capital)
    return b
