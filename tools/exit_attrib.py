"""离场归因诊断(只读) —— 用成交明细回答"设计 RR2.0 为什么没兑现"。

核心指标: **R 口径**
    R = (exit - entry) / (entry - stop0)     # 做多; 做空镜像
即"以该笔初始风险为单位"的收益。因 target = entry + 2*(entry-stop0),
到目标离场恒为 R = +2.0, 止损恒为 R = -1.0。

于是:
  - sum(R)        = 风险调整后总收益(与名义额无关, 可直接跨变体比)
  - 每个 reason 的笔均 R = 该离场方式"把交易截在几 R"
  - 若某离场方式的笔均 R 明显落在 (-1, +2) 开区间内, 它就是**提前离场**,
    正在把本可以走到 2R 的赢单截短(或把本该 -1R 的输单提前砍掉)。

用法:
    python tools/exit_attrib.py out/cache/spot_b1confirm.json [更多...]
    python tools/exit_attrib.py --all            # out/cache 全量
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(os.path.dirname(HERE), "out", "cache")

REASON_CN = {
    "stop_loss": "止损",
    "take_profit": "止盈(到目标)",
    "trail_stop": "追踪止损",
    "structure_break": "结构破位",
    "struct": "结构破位",
    "end_of_data": "期末结算",
    "time": "超时",
    "eod": "期末结算",
}


def load(path: str) -> tuple[list, dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    tr = d.get("tr", [])
    ns = d.get("ns", {})
    if not isinstance(ns, dict):
        ns = {}
    return tr, ns


def funnel(ns: dict) -> dict:
    """把所有标的的信号漏斗加总。"""
    seen = blk = rej = sig = 0
    for v in ns.values():
        if not isinstance(v, dict):
            continue
        seen += sum(int(x) for x in dict(v.get("n_seen", {})).values())
        blk += sum(int(x) for x in dict(v.get("n_blocked", {})).values())
        rej += int(v.get("n_reject", 0) or 0)
        sig += int(v.get("n_signals", 0) or 0)
    return {"seen": seen, "blocked": blk, "reject": rej, "signals": sig}


def r_of(t: dict) -> float | None:
    """以初始风险为单位的收益。"""
    e, x, s = t.get("entry"), t.get("exit"), t.get("stop0")
    if not e or x is None or not s:
        return None
    risk = e - s if t.get("side") == "long" else s - e
    if risk <= 0:
        return None
    move = (x - e) if t.get("side") == "long" else (e - x)
    return move / risk


def pf(vals: list[float]) -> float:
    g = sum(v for v in vals if v > 0)
    l = -sum(v for v in vals if v < 0)
    return (g / l) if l > 0 else float("inf")


def report(path: str) -> dict | None:
    tr, ns = load(path)
    if not tr:
        return None
    name = os.path.basename(path).replace(".json", "")
    rows: list[tuple[dict, float]] = []
    for t in tr:
        r = r_of(t)
        if r is not None:
            rows.append((t, r))
    if not rows:
        return None

    by_reason: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    for t, r in rows:
        by_reason[str(t.get("reason", "?"))].append((t, r))

    tot_r = sum(r for _, r in rows)
    n = len(rows)
    wins = sum(1 for _, r in rows if r > 0)
    fu = funnel(ns)

    print(f"\n{'='*84}")
    print(f"## {name}   笔数={n}   胜率={wins/n:.1%}   "
          f"sum(R)={tot_r:+.1f}   笔均R={tot_r/n:+.4f}")
    # 信号漏斗: 从"结构上出现买点"到"真正成交"的每一层流失
    if fu["seen"]:
        print(f"   信号漏斗: 结构触发 {fu['seen']} → 闸门拦 {fu['blocked']} "
              f"({fu['blocked']/fu['seen']:.1%}) → 风控拒 {fu['reject']} "
              f"→ 成交 {n} ({n/fu['seen']:.1%} 的原始触发)")
    print(f"{'='*84}")
    print(f"{'离场方式':<14}{'笔数':>8}{'占比':>8}{'笔均R':>10}{'中位R':>9}"
          f"{'笔均pnl%':>11}{'贡献(R)':>11}{'占比':>9}{'PF':>7}")

    for rs, items in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
        k = len(items)
        sr = sum(r for _, r in items)
        sr_list = sorted(r for _, r in items)
        med = sr_list[k // 2]
        sp = sum(t.get("pnl_pct", 0.0) for t, _ in items)
        cn = REASON_CN.get(rs, rs)
        print(f"{cn:<14}{k:>8}{k/n:>7.1%}{sr/k:>+10.3f}{med:>+9.3f}{sp/k:>+11.3%}"
              f"{sr:>+11.1f}{sr/tot_r if tot_r else 0:>8.1%}{pf([r for _, r in items]):>7.2f}")

    # R 落点分布: 判断"提前离场"的规模
    buckets = {"<-1": 0, "=-1(止损)": 0, "(-1,0)": 0, "0": 0, "(0,1)": 0,
               "[1,2)": 0, "=2(目标)": 0, ">2": 0}
    for _, r in rows:
        if r < -1.001:
            buckets["<-1"] += 1
        elif abs(r + 1) <= 0.001:
            buckets["=-1(止损)"] += 1
        elif r < 0:
            buckets["(-1,0)"] += 1
        elif r == 0:
            buckets["0"] += 1
        elif r < 1:
            buckets["(0,1)"] += 1
        elif r < 2:
            buckets["[1,2)"] += 1
        elif abs(r - 2) <= 0.001:
            buckets["=2(目标)"] += 1
        else:
            buckets[">2"] += 1
    mid = sum(v for k, v in buckets.items() if k in ("(-1,0)", "(0,1)", "[1,2)"))
    print(f"\nR 落点: " + " · ".join(f"{k} {v}" for k, v in buckets.items()))
    print(f"⇒ **截在开区间 (-1,+2) 内**的笔 = {mid} ({mid/n:.1%}) "
          f"—— 这些是「提前离场」, 既没走到目标、也没被止损打掉")

    # 理论: 若全部持有到目标/止损, sum(R) 应 ≈ 2*W - 1*L
    print(f"   纯 RR2 参考: 若胜负都走满, sum(R) 应≈{2*wins-(n-wins):+.1f}; "
          f"实际 {tot_r:+.1f} (差 {tot_r-(2*wins-(n-wins)):+.1f})")

    # 按止损距离四分位分组: 直接回答"缩小/放宽止损距离"的净效应
    srt = sorted(rows, key=lambda t: t[0].get("risk_pct", 0.0))
    m = len(srt) // 4
    if m >= 5:
        print("\n   按止损距离四分位(入场价 → 止损位):")
        print(f"   {'档位':<6}{'risk_pct 区间':>20}{'笔数':>8}{'止损率':>9}"
              f"{'笔均pnl%':>11}{'笔均R':>9}{'中位R':>9}")
        for qi in range(4):
            seg = srt[qi * m:(qi + 1) * m] if qi < 3 else srt[3 * m:]
            if not seg:
                continue
            k = len(seg)
            sl = sum(1 for t, _ in seg if t.get("reason") == "stop_loss")
            sp = sum(t.get("pnl_pct", 0.0) for t, _ in seg) / k
            sr = sum(r for _, r in seg)
            srl = sorted(r for _, r in seg)
            lo = seg[0][0].get("risk_pct", 0.0)
            hi = seg[-1][0].get("risk_pct", 0.0)
            print(f"   Q{qi+1:<5}{f'{lo:.2%}~{hi:.2%}':>20}{k:>8}{sl/k:>9.1%}"
                  f"{sp:>+11.3%}{sr/k:>+9.3f}{srl[k//2]:>+9.3f}")

    # 止损距离分布(入场价与止损位的几何)
    rp = sorted(t.get("risk_pct", 0.0) for t, _ in rows)
    if rp:
        q = lambda p: rp[min(len(rp) - 1, int(len(rp) * p))]  # noqa: E731
        print(f"   止损距离 risk_pct: 中位 {q(.5):.2%} · "
              f"25% {q(.25):.2%} · 75% {q(.75):.2%} · "
              f"90% {q(.9):.2%} · 最大 {rp[-1]:.2%}")
        # 目标位距离(应恒 = 2*risk_pct)
        tp = sorted((t.get("target", 0) - t.get("entry", 0)) / t.get("entry", 1)
                    * (1 if t.get("side") == "long" else -1) for t, _ in rows)
        print(f"   目标距离: 中位 {tp[len(tp)//2]:.2%}  (应 = 2×risk)")

    return {"name": name, "n": n, "sum_r": tot_r, "mid": mid,
            "wins": wins, "by_reason": {k: len(v) for k, v in by_reason.items()}}


def main() -> None:
    args = sys.argv[1:]
    paths: list[str] = []
    if not args or args[0] == "--all":
        paths = sorted(os.path.join(CACHE, f) for f in os.listdir(CACHE)
                       if f.endswith(".json"))
    else:
        paths = args
    out = []
    for p in paths:
        if os.path.exists(p):
            r = report(p)
            if r:
                out.append(r)
    if len(out) > 1:
        print(f"\n{'='*84}\n## 汇总: 截在开区间内的笔占比(越高 = 离场越「抢跑」)\n{'='*84}")
        print(f"{'变体':<22}{'笔数':>9}{'sum(R)':>10}{'笔均R':>10}{'提前离场占比':>14}")
        for r in sorted(out, key=lambda d: -d["sum_r"] / d["n"]):
            print(f"{r['name']:<22}{r['n']:>9}{r['sum_r']:>+10.1f}"
                  f"{r['sum_r']/r['n']:>+10.4f}{r['mid']/r['n']:>13.1%}")


if __name__ == "__main__":
    main()
