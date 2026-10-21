"""账户与资金管理 —— 复刻 `chan-quant/app/risk.py` + trader 的资金核算部分。

核心原则(与 chan-quant 一致): 规避风险优先于利润最大化。
  * equity 一律指"交易资金池"(trading_capital + 池内盈亏)。
  * 仓位规划复用 chanbook 的 `risk.plan`(结构止损/目标/名义额), 再做交易所
    最小下单额兜底与单笔最大亏损检查 —— 对应 chan-quant 的 `plan_position`。

⚠ 注意: chanbook 的 `risk.plan` 已内置 B1 名义额减半/止损加宽等规则, 此处
   只做"账户资金约束"这一层, 不重复实现止损几何(那是策略层的事, 不得改动)。
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Account", "account_plan"]


@dataclass
class AccountPlan:
    qty: float
    notional: float
    risk_amount: float      # 该笔的实际潜在亏损(USDT)
    accepted: bool
    reason: str
    fallback: bool = False  # 是否退化为最小下单额


def account_plan(equity: float, entry: float, stop: float, cfg: dict,
                 open_count: int, deployed: float = 0.0,
                 min_notional: float = 5.0, pool: float | None = None,
                 ignore_max_open: bool = False,
                 group_cap: float | None = None) -> AccountPlan:
    """账户层仓位规划(复刻 chan-quant plan_position 的口径)。

    pool: 资金池规模。默认取 capital.trading_capital(固定池)。
    group_cap: 单组金额上限(弹性模式), 传 None = 旧口径。
    """
    pos = cfg["position"]
    cap = float(pool) if pool is not None and pool > 0 else float(cfg["capital"]["trading_capital"])

    equity = min(equity, cap)
    remaining = cap - deployed
    if remaining < min_notional - 1e-9:
        return AccountPlan(0, 0, 0, False,
                           f"交易资金池剩余{remaining:.2f}U, 不足最小下单额{min_notional}U")

    if not ignore_max_open and open_count >= pos["max_open_positions"]:
        return AccountPlan(0, 0, 0, False, "持仓数量已达上限")

    per_unit_risk = max(entry - stop, 1e-12)
    notional = min(equity * pos["risk_per_trade"] / per_unit_risk * entry,
                   remaining,
                   equity * pos["max_per_symbol"])

    fallback = False
    grp = float(group_cap or 0.0)
    if grp > 0:
        want = max(grp, min_notional)
        if notional > want:
            notional = want
        if notional < want:
            fallback = True
            notional = min(want, remaining)
    elif notional < min_notional:
        fallback = True
        notional = min(min_notional, remaining)

    qty = notional / entry
    loss = notional * per_unit_risk / entry
    max_loss_usd = pos.get("max_loss_per_trade_usd", cap * 0.05)
    if loss > max_loss_usd:
        return AccountPlan(0, 0, round(loss, 2), False,
                           f"该信号单笔潜在亏损{loss:.2f}U超过上限{max_loss_usd}U, 放弃")
    return AccountPlan(round(qty, 6), round(notional, 2), round(loss, 2), True, "ok", fallback)


class Account:
    """资金池账户。持有敞口核算、槽位、可用资金。

    不直接下单; 下单由 trader/ftrader 调 broker 完成, 本类只记账资金占用。
    """

    def __init__(self, cfg: dict, name: str = "spot") -> None:
        self.cfg = cfg
        self.name = name
        self.pool = float(cfg["capital"]["trading_capital"])
        self._deployed = 0.0        # 已占用名义额(USDT)
        self._pnl = 0.0             # 已实现盈亏(USDT)
        self.max_pairs = int(cfg["capital"].get("max_pairs", 10))
        self.min_notional = 5.0

    # ------------------------------------------------------------------ 查询
    def equity(self) -> float:
        return self.pool + self._pnl

    def deployed(self) -> float:
        return self._deployed

    def free_cash(self) -> float:
        return max(0.0, self.equity() - self._deployed)

    def per_trade_usd(self) -> float:
        pt = self.cfg["position"].get("per_trade_usd", 5.0)
        return max(float(pt or 0.0), self.min_notional)

    def slots_left(self, open_count: int) -> int:
        """并发槽位 = min(硬上限, 资金能支撑的组数)。"""
        hard = max(0, self.max_pairs - open_count)
        cash = int(self.free_cash() / self.per_trade_usd()) if self.per_trade_usd() > 0 else 0
        return min(hard, cash)

    # ------------------------------------------------------------------ 记账
    def reserve(self, notional: float) -> None:
        self._deployed += notional

    def release(self, notional: float, pnl_usd: float) -> None:
        self._deployed = max(0.0, self._deployed - notional)
        self._pnl += pnl_usd

    def snapshot(self) -> dict:
        return {
            "pool": self.pool,
            "equity": round(self.equity(), 4),
            "deployed": round(self._deployed, 4),
            "free_cash": round(self.free_cash(), 4),
            "pnl": round(self._pnl, 4),
            "max_pairs": self.max_pairs,
        }
