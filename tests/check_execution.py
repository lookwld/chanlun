"""execution 执行层自检 —— 覆盖 config/account/analyzer/bridge/llm/trader/ftrader。

验证「复刻 chan-quant 执行层功能契约」且「不依赖策略层内部实现」。
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
from pathlib import Path

# 使 chanbook 包可导入(项目根加入 path, 通过 chanbook.execution 导入)
_ROOT = Path(__file__).resolve().parents[1]   # 项目根 .../chanbook/
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from chanbook.execution import config as _config
from chanbook.execution import account as _account
from chanbook.execution import analyzer as _analyzer
from chanbook.execution import bridge as _bridge
from chanbook.execution import broker as _broker
from chanbook.execution import llm as _llm
from chanbook.execution.trader import SpotTrader
from chanbook.execution.ftrader import FuturesTrader

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        print(f"  ✗ {name}  {detail}")


# ------------------------------------------------------------------ config
def test_config():
    # 缺文件回退默认 + dry_run 放行
    cfg = _config.load_config("/nonexistent/config.yaml")
    check("config 缺文件回退默认", cfg["mode"] == "dry_run")
    check("config 默认资金池", cfg["capital"]["trading_capital"] == 10.0)

    # live 未解锁 -> 拒绝
    c = copy.deepcopy(_config.DEFAULT_CONFIG)
    c["mode"] = "live"
    try:
        _config.load_config("/nonexistent/config.yaml")  # 默认文件
        # 用内存方式直接校验逻辑: 通过写临时文件
    except Exception:
        pass
    import io
    tmp = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    tmp.write("mode: live\ncapital:\n  trading_capital: 10.0\n  allow_live: false\nexchange:\n  id: binance\n")
    tmp.close()
    try:
        _config.load_config(tmp.name)
        check("config live 未解锁应拒绝", False, "未抛 PermissionError")
    except PermissionError:
        check("config live 未解锁拒绝", True)
    finally:
        os.unlink(tmp.name)


# ------------------------------------------------------------------ account
def test_account():
    cfg = copy.deepcopy(_config.DEFAULT_CONFIG)
    acct = _account.Account(cfg, name="spot")
    check("account 初始 equity=pool", acct.equity() == 10.0)
    acct.reserve(5.0)
    check("account reserve 占用", abs(acct.deployed() - 5.0) < 1e-9)
    acct.release(5.0, 0.5)
    check("account release 释放+盈亏", abs(acct.equity() - 10.5) < 1e-9)

    # account_plan 复刻 chan-quant 口径
    p = _account.account_plan(10.0, entry=100.0, stop=95.0, cfg=cfg, open_count=0,
                              min_notional=5.0)
    check("account_plan 接受", p.accepted and p.reason == "ok")
    # 单笔最大亏损超限 -> 放弃
    cfg2 = copy.deepcopy(_config.DEFAULT_CONFIG)
    cfg2["position"]["max_loss_per_trade_usd"] = 0.01
    p2 = _account.account_plan(10.0, entry=100.0, stop=90.0, cfg=cfg2, open_count=0,
                               min_notional=5.0)
    check("account_plan 超损放弃", not p2.accepted)


# ------------------------------------------------------------------ analyzer
def test_analyzer():
    with tempfile.TemporaryDirectory() as d:
        an = _analyzer.Analyzer({"db_path": str(Path(d) / "t.db")})
        an.record_closed({"symbol": "BTCUSDT", "grade": "B1", "entry": 100.0,
                          "stop": 95.0, "target": 110.0, "qty": 0.05,
                          "opened_at": 1000.0, "exit": 108.0, "pnl_pct": 0.08,
                          "exit_reason": "S1", "closed_at": 2000.0,
                          "reasons": [], "kind": "B1", "trend": "down"})
        rows = an.load()
        check("analyzer 记录并读取", len(rows) == 1)
        check("analyzer 字段对齐", rows[0]["pnl_pct"] == 0.08 and rows[0]["kind"] == "B1")
        rep = an.report()
        check("analyzer 报告生成", "平仓笔数" in rep)


# ------------------------------------------------------------------ bridge
def test_bridge():
    class S:
        sym = "BTCUSDT"; side = "long"; kind = "B1"; ts = 123
        entry = 100.0; stop = 95.0; target = 110.0; atr = 2.0
        notional = 5.0; trend = "down"; reasons = ["r1"]
    o = _bridge.signal_to_order(S())
    check("bridge signal->order 字段", o["symbol"] == "BTCUSDT" and o["grade"] == "B1")
    check("bridge 止损目标透传", o["stop"] == 95.0 and o["target"] == 110.0)
    rec = _bridge.close_record("BTCUSDT", "B1", "long", 100, 95, 110, 0.05,
                               1000, 108, 0.08, "S1", 2000)
    check("bridge close_record", rec["exit_reason"] == "S1" and rec["kind"] == "B1")


# ------------------------------------------------------------------ broker
def test_broker():
    cfg = copy.deepcopy(_config.DEFAULT_CONFIG)
    b = _broker.get_broker(cfg)
    check("broker dry_run 非 live", not b.live)
    check("broker min_notional dry_run=5", b.min_notional("BTCUSDT") == 5.0)
    b._sim.set_price("BTCUSDT", 100.0)
    r = b.buy("BTCUSDT", 0.05, 100.0)
    check("broker 模拟买入", r.get("ok"))
    check("broker 余额扣减", abs(b._sim.balance_usdt() - (10.0 - 5.0)) < 1e-9)
    r2 = b.sell("BTCUSDT", 0.05, 108.0)
    check("broker 模拟卖出", r2.get("ok"))
    check("broker 余额回补", abs(b._sim.balance_usdt() - (5.0 + 5.4)) < 1e-9)


# ------------------------------------------------------------------ llm
def test_llm():
    f = _llm.LLMFilter({})
    ok, why = f.filter({"symbol": "BTCUSDT"})
    check("llm 未启用放行", ok and why == "llm_disabled")


# ------------------------------------------------------------------ trader
def test_trader():
    cfg = copy.deepcopy(_config.DEFAULT_CONFIG)
    cfg["position"]["max_open_positions"] = 10
    with tempfile.TemporaryDirectory() as d:
        an = _analyzer.Analyzer({"db_path": str(Path(d) / "t.db")})
        t = SpotTrader(cfg, analyzer=an)
        o = {"symbol": "BTCUSDT", "side": "long", "entry": 100.0, "stop": 95.0,
             "target": 110.0, "notional": 5.0, "grade": "B1", "kind": "B1",
             "trend": "down", "reasons": []}
        r = t.try_open(o)
        check("trader 开仓", r.get("opened"))
        check("trader 持仓数", t.open_count() == 1)
        # 止损离场
        closed = t.check_stops({"BTCUSDT": 94.0})
        check("trader 止损离场", len(closed) == 1 and closed[0].get("closed"))
        check("trader 离场后清仓", t.open_count() == 0)
        check("trader 已记账", len(t.analyzer.load()) == 1)


# ------------------------------------------------------------------ ftrader
def test_ftrader():
    cfg = copy.deepcopy(_config.DEFAULT_CONFIG)
    with tempfile.TemporaryDirectory() as d:
        an = _analyzer.Analyzer({"db_path": str(Path(d) / "t.db")})
        f = FuturesTrader(cfg, analyzer=an)
        o = {"symbol": "BTCUSDT", "side": "short", "entry": 100.0, "stop": 105.0,
             "target": 90.0, "notional": 5.0, "grade": "S1", "kind": "S1",
             "trend": "up", "reasons": []}
        r = f.try_open(o)
        check("ftrader 空头开仓", r.get("opened") and r.get("side") == "short")
        closed = f.check_stops({"BTCUSDT": 106.0})
        check("ftrader 空头止损", len(closed) == 1 and closed[0].get("closed"))


def main():
    test_config()
    test_account()
    test_analyzer()
    test_bridge()
    test_broker()
    test_llm()
    test_trader()
    test_ftrader()
    print(f"\n执行层自检: {PASS} 通过 / {FAIL} 失败")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
