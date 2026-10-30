"""保护性熔断与资金护栏 —— 移植 `chan-quant/app/guards.py`(纯逻辑层)。

2026-09-29 第四轮补全。规则与 chan-quant 完全一致(2026-09-23 约定:
**所有熔断统一 1h, 发生即计时, 到点自动解除**):

  1. 连续真实亏损(幅度超过 loss_eps_pct)达 max_consecutive_losses 次
     -> 全局暂停开仓 pause_hours (1h)
  2. 当日累计亏损达资金池 daily_max_loss_pct -> 触发即计时 1h(不再整日封停)
  3. 某标的真实亏损后 cooldown_hours 内不再入场; 手续费级小亏给同标的短冷却
  4. 方向保险丝: 某方向最近 dir_window 笔净亏 -> 暂停该方向 dir_pause_hours

⚠ 与 chan-quant 的差异:
  * 状态文件持久化(`guards.json`)改为**可选**(`persist=False` 时纯内存态),
    便于信号层自检与单元测试; 生产若要跨进程持久化, 传 `state_path` 即可。
  * 时间用**秒级 epoch float**(与 chan-quant 的 `time.time()` 一致), 便于单测注入
    确定性时间, 不依赖真实时钟。

⚠ 这是**组合层**规则。上一轮能力补全审计已证明: 用
   信号层口径评估组合层规则必然得出"负增量", 因为信号层没有资金约束。本模块
   只补**实现 + 单元自检**, 不接入信号层回测口径 —— 接入要等组合层建好。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["Guards"]


class Guards:
    """保护性熔断(现货/合约各持一份, 共用 protection 段的时长配置)。"""

    def __init__(self, cfg: dict | None = None, name: str = "",
                 persist: bool = False) -> None:
        cfg = cfg or {}
        p = cfg.get("protection", {})
        self.enabled = bool(p.get("enabled", True))
        self.capital = float(cfg.get("capital", {}).get("trading_capital", 10.0))
        self.daily_max_loss = float(p.get("daily_max_loss_pct", 0.15))
        self.max_consec = int(p.get("max_consecutive_losses", 3))
        self.pause_hours = float(p.get("pause_hours", 1))
        self.cooldown_hours = float(p.get("cooldown_hours", 1))
        self.loss_eps = float(p.get("loss_eps_pct", 0.002))
        self.fee_cd_h = float(p.get("fee_loss_cooldown_hours", 1.0))
        dg = cfg.get("futures", {}).get("dir_guard", {})
        self.dir_window = int(dg.get("window", 8))
        self.dir_pause_hours = float(dg.get("pause_hours", 1))
        self.name = name
        self._persist = persist
        self._path: Path | None = None
        if persist:
            sp = Path(cfg.get("state_path", "data/state.json"))
            self._path = sp.parent / (f"guards_{name}.json" if name else "guards.json")
            self._path.parent.mkdir(parents=True, exist_ok=True)
        self.s = self._load()
        self.s.setdefault("dir_hist", {"long": [], "short": []})
        self.s.setdefault("dir_block", {"long": 0.0, "short": 0.0})
        self.s.setdefault("pause_reason", "")

    # ------------------------------------------------------------------ 状态
    def _load(self) -> dict:
        if self._path is not None and self._path.exists():
            return json.loads(self._path.read_text(encoding="utf-8"))
        return {"day": "", "day_pnl_usd": 0.0, "consec_losses": 0,
                "pause_until": 0.0, "pause_reason": "", "cooldowns": {}}

    def save(self) -> None:
        if self._path is not None:
            self._path.write_text(json.dumps(self.s, ensure_ascii=False, indent=2),
                                  encoding="utf-8")

    def _roll_day(self, now: float) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if self.s["day"] != day:
            self.s["day"] = day
            self.s["day_pnl_usd"] = 0.0
            self.save()

    # ------------------------------------------------------------------ 判定
    def can_open(self, symbol: str, now: float | None = None) -> tuple[bool, str]:
        if not self.enabled:
            return True, "ok"
        now = now if now is not None else time.time()
        self._roll_day(now)
        if now < self.s["pause_until"]:
            h = (self.s["pause_until"] - now) / 3600.0
            why = self.s.get("pause_reason") or "连续止损"
            return False, f"{why}触发全局冷却, 剩余{h:.1f}h禁止开仓"
        until = self.s["cooldowns"].get(symbol, 0.0)
        if now < until:
            h = (until - now) / 3600.0
            return False, f"{symbol}止损后冷却中, 剩余{h:.1f}h"
        return True, "ok"

    def on_close(self, rec: dict, now: float | None = None) -> None:
        """每笔平仓后更新熔断状态(收益分级, 与 chan-quant 逐条一致)。"""
        if not self.enabled or not rec:
            return
        now = now if now is not None else time.time()
        self._roll_day(now)
        pnl_usd = float(rec["qty"]) * float(rec["entry"]) * float(rec["pnl_pct"])
        self.s["day_pnl_usd"] += pnl_usd
        pnl_pct = float(rec["pnl_pct"])
        if pnl_pct < -self.loss_eps:
            self.s["consec_losses"] += 1
            self.s["cooldowns"][rec["symbol"]] = now + self.cooldown_hours * 3600.0
            if self.s["consec_losses"] >= self.max_consec:
                self.s["pause_until"] = now + self.pause_hours * 3600.0
                self.s["pause_reason"] = "连续止损"
                self.s["consec_losses"] = 0
        elif pnl_pct > 0:
            self.s["consec_losses"] = 0
        elif self.fee_cd_h > 0:
            self.s["cooldowns"][rec["symbol"]] = now + self.fee_cd_h * 3600.0
        if (self.s["day_pnl_usd"] <= -self.capital * self.daily_max_loss
                and now >= self.s["pause_until"]):
            self.s["pause_until"] = now + self.pause_hours * 3600.0
            self.s["pause_reason"] = "当日亏损触线"
        self.save()

    # ------------------------------------------------------------------ 方向
    def direction_ok(self, side: str, now: float | None = None) -> tuple[bool, str]:
        if not self.enabled or side not in ("long", "short"):
            return True, "ok"
        now = now if now is not None else time.time()
        until = self.s["dir_block"].get(side, 0.0)
        if now < until:
            h = (until - now) / 3600.0
            return False, f"{side}方向近期净亏, 自适应暂停剩余{h:.1f}h"
        return True, "ok"

    def on_close_dir(self, side: str, pnl_pct: float,
                     now: float | None = None) -> str | None:
        if not self.enabled or side not in ("long", "short"):
            return None
        now = now if now is not None else time.time()
        hist = self.s["dir_hist"].setdefault(side, [])
        hist.append({"pnl": pnl_pct, "ts": now})
        if len(hist) > 40:
            del hist[:-40]
        msg = None
        recent = hist[-self.dir_window:]
        if len(recent) >= self.dir_window and sum(x["pnl"] for x in recent) < 0:
            self.s["dir_block"][side] = now + self.dir_pause_hours * 3600.0
            self.s["dir_hist"][side] = []
            msg = (f"{side}方向最近{self.dir_window}笔净亏, "
                   f"自适应暂停{self.dir_pause_hours:.0f}h")
        self.save()
        return msg
