"""把 out/ab_trail_*.txt 解析成 **参数 → 指标** 的曲线, 回答两个问题:

  1. 「追踪淹没 target」在全量上是否成立?  ← 看 rr 扫描里 target 与 trail 的换手
  2. trail_mult 的可提升区间在哪?          ← 看 trail_mult 扫描的峰值

为什么不能只看 `ab_summary` 的表:
  那张表按**变体名**排列(字母序), 而这一轮的自变量是**一个数值参数**。
  要回答「是否存在峰值、峰值在哪儿」, 必须按参数值排序, 并显式标出默认点。

用法:
    python tools/param_sweep.py out/ab_trail_um.txt out/ab_trail_spot.txt
"""

from __future__ import annotations

import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ab_summary import FEE, THR, corrected, parse  # noqa: E402

# 变体名 -> 该变体把哪个参数设成了什么
#   ("trail", 值) / ("rr", 值) / ("default", None)
SWEEP: dict[str, tuple[str, float | None]] = {
    "all": ("default", None),
    "trail15": ("trail", 1.5),
    "trail20": ("trail", 2.0),
    "trail30": ("trail", 3.0),
    "trail40": ("trail", 4.0),
    "rr10": ("rr", 1.0),
    "rr30": ("rr", 3.0),
}

# 报告里「-- 分离场原因 --」块的原文 reason -> 中文标签
REASON_CN = {
    "take_profit": "止盈(到目标)",
    "trail_stop": "追踪止损",
    "stop_loss": "止损",
    "structure_break": "结构破位",
    "end_of_data": "期末结算",
}
RE_REASON = re.compile(
    r"^\s*(take_profit|trail_stop|stop_loss|structure_break|end_of_data)\s+"
    r"(\d+)笔\s+胜率\s*([\d.]+)%.*?净\s*([+-][\d.]+)U")


def parse_reasons(path: str) -> dict[str, dict[str, dict]]:
    """变体 -> {reason: {n, win, net_u}}。"""
    out: dict[str, dict[str, dict]] = {}
    cur: str | None = None
    for raw in open(path, encoding="utf-8"):
        m = re.match(r"^=====\s+(\S+)\s+—", raw)
        if m:
            cur = m.group(1)
            out[cur] = {}
            continue
        if cur is None:
            continue
        m = RE_REASON.match(raw)
        if m:
            out[cur][m.group(1)] = {
                "n": int(m.group(2)), "win": float(m.group(3)) / 100,
                "net_u": float(m.group(4).replace("+", ""))}
    return out


def sweep_table(rows: list[dict], reasons: dict, market: str, key: str,
                title: str, default_val: float) -> None:
    """打一张『参数值 -> 指标』表。key = 'trail' | 'rr'。"""
    items: list[tuple[float, dict]] = []
    for c in rows:
        kind, val = SWEEP.get(c["name"], ("", None))
        # `all` 是 trail 与 rr **双默认**, 两张表里都要作为参照点出现
        if kind != key and kind != "default":
            continue
        items.append((default_val if val is None else float(val), c))
    if not items:
        return
    items.sort(key=lambda kv: kv[0])
    base = next((c for v, c in items if v == default_val), None)
    be = base["exp"] if base else 0.0

    print(f"\n### {title}（{market}，默认 = {default_val}）\n")
    print("| " + ("trail_mult" if key == "trail" else "rr") +
          " | 变体 | 笔数 | 胜率 | 毛期望 | 净期望 | 盈亏比 | PF | 均持 | "
          "止盈占比 | 追踪占比 | Δ毛期望 |")
    print("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for v, c in items:
        nexp, _ = corrected(c)
        rs = reasons.get(c["name"], {})
        n = c["n"] or 1
        tp = rs.get("take_profit", {}).get("n", 0) / n
        ts = rs.get("trail_stop", {}).get("n", 0) / n
        tag = " ←默认" if v == default_val else ""
        d = c["exp"] - be
        print(f"| {v:g}{tag} | `{c['name']}` | {c['n']:,} | {c['win']*100:.1f}% | "
              f"{c['exp']*100:+.3f}% | {nexp*100:+.3f}% | {c['rr']:.2f} | "
              f"{c['pf']:.2f} | {c['hold']:.0f}m | {tp:.1%} | {ts:.1%} | "
              f"{d*100:+.3f}pp |")

    best = max(items, key=lambda kv: kv[1]["exp"])
    print(f"\n峰值 = **{best[0]:g}**（毛期望 {best[1]['exp']*100:+.3f}%，"
          f"较默认 {be*100:+.3f}% 改善 {(best[1]['exp']-be)*100:+.3f}pp）；"
          f"门槛 {THR[market]*100:.2f}%，仍差 "
          f"{(THR[market]-best[1]['exp'])*100:.3f}pp。")


def main(argv: list[str]) -> int:
    for path in (argv[1:] or ["out/ab_trail_um.txt"]):
        if not os.path.exists(path):
            print(f"## {path} —— 尚未生成\n")
            continue
        rows = [c for c in parse(path) if c.get("n")]
        if not rows:
            continue
        mkt = rows[0]["market"]
        reasons = parse_reasons(path)
        defv = 2.0 if mkt == "um" else 3.0      # risk.CALIB[market]["trail_mult"]
        print(f"\n{'='*88}\n## {path}（{mkt}，门槛 = 毛期望 > {THR[mkt]*100:.2f}%）\n{'='*88}")
        sweep_table(rows, reasons, mkt, "trail", "trail_mult 扫描", defv)
        sweep_table(rows, reasons, mkt, "rr", "rr（目标距离）扫描", 2.0)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
