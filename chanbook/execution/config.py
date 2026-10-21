"""配置加载: config.yaml + .env 环境变量, 并执行硬性约束校验。

复刻 `chan-quant/app/config.py` 的契约(字段名、校验语义、dry_run/live 解锁
机制逐条一致)。差异: 增加了对 chanbook 执行层专有字段(execution)的默认填充,
使本包可脱离 chan-quant 独立运行。
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover - 依赖提示
    yaml = None

__all__ = ["load_config", "DEFAULT_CONFIG"]


# 本包运行所需的最小配置骨架(缺省值, 与 chan-quant config.example.yaml 对齐)
DEFAULT_CONFIG: dict = {
    "mode": "dry_run",            # dry_run=模拟盘(默认) | live=实盘
    "exchange": {
        "id": "binance",
        "api_key": "",
        "secret": "",
    },
    "capital": {
        "trading_capital": 10.0,
        "allow_live": False,      # live 模式硬性解锁开关
        "min_market_cap_usd": 1_000_000_000,
        "max_pairs": 10,
    },
    "position": {
        "risk_per_trade": 0.01,
        "max_loss_per_trade_usd": 0.5,
        "max_open_positions": 10,
        "max_per_symbol": 1.0,
        "elastic": False,
        "per_trade_usd": 5.0,
    },
    "protection": {
        "enabled": True,
        "daily_max_loss_pct": 0.15,
        "max_consecutive_losses": 3,
        "pause_hours": 1.0,
        "cooldown_hours": 1.0,
        "loss_eps_pct": 0.002,
        "fee_loss_cooldown_hours": 1.0,
    },
    "futures": {
        "trading_capital": 10.0,
        "leverage": 3,
        "allow_live": False,
        "dir_guard": {"window": 8, "pause_hours": 1.0},
    },
    "llm": {"enabled": False, "api_key_env": "DEEPSEEK_API_KEY",
            "base_url": "https://api.deepseek.com"},
}


def _load_dotenv(p: Path) -> None:
    """把 .env 文件(与 config.yaml 同目录)加载进环境变量。已存在的环境变量不覆盖。"""
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("'\"")
        if k:
            os.environ.setdefault(k, v)


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str = "config.yaml", *, require_file: bool = False) -> dict:
    """加载配置。require_file=False 时缺文件回退到 DEFAULT_CONFIG(便于研究/测试)。

    校验规则(与 chan-quant 一致):
      * mode=live 且 capital.allow_live != true -> PermissionError
      * mode=live 且无 API key -> ValueError
      * capital.trading_capital <= 0 -> ValueError
      * exchange.id != binance -> ValueError
    """
    p = Path(path)
    if not p.exists():
        if require_file:
            raise FileNotFoundError(
                f"缺少配置文件 {p.resolve()} —— 请先复制 config.example.yaml 为 config.yaml")
        cfg = copy.deepcopy(DEFAULT_CONFIG)
    else:
        if yaml is None:
            raise ImportError("需要 PyYAML: pip install pyyaml")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config.yaml 内容无效")
        cfg = _deep_merge(DEFAULT_CONFIG, raw)

    # .env 与 config.yaml 同目录; 兼容旧布局(项目根/.env)
    _load_dotenv(p.parent / ".env")
    _load_dotenv(Path(".env"))

    # ---- 硬性约束校验 (不可绕过) ----
    if cfg.get("mode") == "live" and not cfg.get("capital", {}).get("allow_live", False):
        raise PermissionError(
            "实盘未解锁: 请在 config.yaml 中将 capital.allow_live 设为 true 才能以 live 模式启动。"
            "解锁前程序只允许 dry_run 模拟盘。")
    if cfg.get("mode") == "live" and not (
            os.getenv("BQ_API_KEY") or cfg.get("exchange", {}).get("api_key")):
        raise ValueError(
            "实盘需要 API key: 请确认 .env 文件存在且包含 BQ_API_KEY / BQ_SECRET 两行。")
    if float(cfg.get("capital", {}).get("trading_capital", 0)) <= 0:
        raise ValueError("capital.trading_capital 必须为正数")
    ex = cfg.get("exchange", {})
    if ex.get("id") != "binance":
        raise ValueError("当前版本仅支持 binance")
    return cfg
