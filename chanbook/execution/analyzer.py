"""复盘与弊端自动诊断 —— 复刻 `chan-quant/app/analyzer.py`。

记录每笔交易到 sqlite, 生成胜率/盈亏比等统计, 并按规则自动指出策略弊端
(连续止损、追高入场、盈亏比失衡、过度频繁交易等)。

与 chan-quant 的差异:
  * `grade` 字段语义: chan-quant 存 "third_buy"/"ma_support"; 本包存 chanbook 的
    买卖点类型 "B1"/"B2"/"B3"(由 bridge 映射), 其余字段逐一对齐。
  * 新增 `kind`/`trend` 可选列, 便于复盘时按结构维度切片。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

__all__ = ["Analyzer"]


class Analyzer:
    def __init__(self, cfg: dict | None = None, db_path: str | None = None) -> None:
        cfg = cfg or {}
        self.db_path = Path(db_path or cfg.get("db_path", "data/trades.db"))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(str(self.db_path))

    def _init(self):
        conn = self._conn()
        try:
            conn.execute("""CREATE TABLE IF NOT EXISTS trades(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT, grade TEXT, entry REAL, stop REAL, target REAL,
                qty REAL, opened_at REAL, exit REAL, pnl_pct REAL,
                exit_reason TEXT, closed_at REAL, reasons TEXT,
                kind TEXT DEFAULT '', trend TEXT DEFAULT '')""")
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ 写入
    def record_closed(self, rec: dict):
        conn = self._conn()
        try:
            conn.execute(
                "INSERT INTO trades(symbol,grade,entry,stop,target,qty,opened_at,"
                "exit,pnl_pct,exit_reason,closed_at,reasons,kind,trend)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec["symbol"], rec.get("grade"), rec["entry"], rec["stop"],
                 rec["target"], rec["qty"], rec["opened_at"], rec["exit"],
                 rec["pnl_pct"], rec["exit_reason"], rec["closed_at"],
                 json.dumps(rec.get("reasons", []), ensure_ascii=False),
                 rec.get("kind", ""), rec.get("trend", "")))
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ 读取
    def load(self) -> list[dict]:
        conn = self._conn()
        try:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM trades ORDER BY closed_at").fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    # ------------------------------------------------------------------ 报告
    def report(self) -> str:
        ts = self.load()
        if not ts:
            return "暂无平仓记录。先 dry_run 跑一段时间再复盘。"
        n = len(ts)
        wins = [t for t in ts if t["pnl_pct"] > 0]
        losses = [t for t in ts if t["pnl_pct"] <= 0]
        wr = len(wins) / n
        avg_w = sum(t["pnl_pct"] for t in wins) / len(wins) if wins else 0
        avg_l = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
        pf = (sum(t["pnl_pct"] for t in wins) / abs(sum(t["pnl_pct"] for t in losses))
              if losses and sum(t["pnl_pct"] for t in losses) != 0 else float("inf"))
        eq = 1.0
        for t in ts:
            eq *= 1 + t["pnl_pct"]
        lines = [
            "========== 交易复盘报告 ==========",
            f"平仓笔数: {n}   胜率: {wr:.1%}",
            f"平均盈利单: {avg_w:+.2%}   平均亏损单: {avg_l:+.2%}",
            f"盈亏比(利润因子): {pf:.2f}   累计权益曲线: {eq:+.2%}",
            "",
            "---------- 弊端自动诊断 ----------",
        ]
        lines += self._critique(ts, wins, losses)
        return "\n".join(lines)

    def _critique(self, ts, wins, losses) -> list[str]:
        out, n = [], len(ts)
        # 1. 连续止损
        streak, best = 0, 0
        for t in ts:
            streak = streak + 1 if t["pnl_pct"] <= 0 else 0
            best = max(best, streak)
        if best >= 3:
            out.append(f"[趋势过滤] 最长连续止损{best}次 —— 结构/趋势过滤可能不够严, "
                       f"考虑提高操作级别或收紧买点确认条件。")
        # 2. 频繁交易
        span = (ts[-1]["closed_at"] - ts[0]["opened_at"]) / 86400 if n > 1 else 1
        if span > 0 and n / span > 3:
            out.append(f"[过度交易] 日均{n/span:.1f}笔 —— 买点确认条件偏松, "
                       f"建议提高结构确认严格度。")
        # 3. 止损被扫占比
        sl = [t for t in ts if t["exit_reason"] == "stop_loss"]
        if n and len(sl) / n > 0.6:
            out.append(f"[止损过近] {len(sl)}/{n}笔被止损 —— ATR缓冲可能不足, "
                       f"或止损应改放在结构低点而非仅分型低点。")
        # 4. 盈亏比失衡
        wins_p = [t for t in wins if t["pnl_pct"] > 0]
        if wins_p and losses:
            aw = sum(t["pnl_pct"] for t in wins_p) / len(wins_p)
            al = abs(sum(t["pnl_pct"] for t in losses) / len(losses))
            if aw / max(al, 1e-9) < 1.3:
                out.append("[目标价过近] 盈亏比<1.3 —— 提高RR或让离场判据接管, 让利润奔跑。")
        # 5. 追高检测 (入场价距止损过近)
        chase = [t for t in ts if t["entry"] > 0 and (t["entry"] - t["stop"]) / t["entry"] < 0.012]
        if n and len(chase) / n > 0.4:
            out.append("[入场过急] 大量信号止损距离<1.2%(结构低点贴得太近) —— "
                       "要求结构点与前低之间留出更多缓冲。")
        # 6. 买点质量(按 chanbook 的 B1/B2/B3 切片)
        for g in ("B1", "B2", "B3"):
            sub = [t for t in ts if t["kind"] == g or t["grade"] == g]
            if len(sub) >= 5:
                wr = len([t for t in sub if t["pnl_pct"] > 0]) / len(sub)
                out.append(f"[买点质量] {g}: {len(sub)}笔 胜率{wr:.0%}")
        if not out:
            out.append("未检出明显弊端, 继续积累样本(建议至少30笔后再下结论)。")
        return out
