"""把 out/ab_*.txt 的变体报告解析成可比表格 + 门槛判决。

为什么需要它(而不是直接看报告里的"单笔净期望"列):
  `report.summarize` 用**第一笔成交的名义额**当分母算 `net_exp_pct`。
  而本项目的名义额是**按买点分档**的 —— 一买走小仓特例(合约 7.5U), 二/三买 15U。
  于是"三类全开"变体的第一笔若是 B1, 分母就变成 7.5 而分子里混着 15U 的笔,
  净期望被系统性放大成本(实测 all: -0.175%, 真值 -0.125%)。

  本工具改为: 先由每档的 `净<U>` 反推该档名义额
      名义额 = 净U / (笔数 x (单笔期望 - 双边费))
  再以 `笔数 x 名义额` 为权重对 (单笔期望 - 双边费) 加权 => 正确的单笔净期望。
  单档变体天然只有一种名义额, 于是 `毛期望 - 双边费` 即精确解, 可自校验。

用法:
    python tools/ab_summary.py out/ab_um.txt out/ab_spot.txt
"""

from __future__ import annotations

import re
import sys

FEE = {"um": 0.0005, "spot": 0.001}
THR = {"um": 0.0012, "spot": 0.0020}          # 毛期望门槛(双边费 + 资金费余量)

RE_HDR = re.compile(r"^=====\s+(\S+)\s+—\s+(.*?)\s+\[(\w+)\]\s+=====")
RE_SUM = re.compile(
    r"^\s+合计\s+(\d+)笔\s+胜率\s*([\d.]+)%\s+单笔期望\s*([+-][\d.]+)%\s+"
    r"净\s*([+-][\d.]+)U\s+单笔净期望\s*([+-][\d.]+)%\s+"
    r"均赢\s*([+-][\d.]+)%/均亏\s*([+-][\d.]+)%\s+盈亏比\s*([\d.]+)\s+"
    r"PF\s*([\d.]+)\s+均持\s*([\d.]+)min")
RE_AUX = re.compile(r"^\s*最好\s*([+-][\d.]+)%\s+最差\s*([+-][\d.]+)%\s+"
                    r"盈亏平衡胜率\s*([\d.]+)%\s+(?:均)?名义([\d.]+)U\s+"
                    r"手续费([\d.]+)U")
RE_KIND = re.compile(
    r"^\s+(B[123])\s+(?:(\d+)笔\s+胜率\s*([\d.]+)%\s+单笔期望\s*([+-][\d.]+)%\s+"
    r"净\s*([+-][\d.]+)U|无成交)")


def _f(x: str) -> float:
    return float(x.replace("+", ""))


def parse(path: str) -> list[dict]:
    out: list[dict] = []
    cur: dict | None = None
    for raw in open(path, encoding="utf-8"):
        line = raw.rstrip("\n")
        m = RE_HDR.match(line)
        if m:
            cur = {"name": m.group(1), "desc": m.group(2), "market": m.group(3),
                   "kinds": []}
            out.append(cur)
            continue
        if cur is None:
            continue
        m = RE_SUM.match(line)
        if m:
            cur.update(n=int(m.group(1)), win=_f(m.group(2)) / 100,
                       exp=_f(m.group(3)) / 100, net_u=_f(m.group(4)),
                       win_avg=_f(m.group(6)) / 100, loss_avg=_f(m.group(7)) / 100,
                       rr=_f(m.group(8)), pf=_f(m.group(9)), hold=_f(m.group(10)))
            continue
        m = RE_AUX.match(line)
        if m:
            cur.update(best=_f(m.group(1)) / 100, worst=_f(m.group(2)) / 100,
                       be_win=_f(m.group(3)) / 100, notional=_f(m.group(4)),
                       fee_u=_f(m.group(5)))
            continue
        m = RE_KIND.match(line)
        if m and m.group(1):
            if m.group(2) is None:
                cur["kinds"].append({"kind": m.group(1), "n": 0})
            else:
                cur["kinds"].append({
                    "kind": m.group(1), "n": int(m.group(2)),
                    "win": _f(m.group(3)) / 100, "exp": _f(m.group(4)) / 100,
                    "net_u": _f(m.group(5))})
    return out


def corrected(c: dict) -> tuple[float, float]:
    """返回 (校正后的单笔净期望, 加权口径下的平均名义额)。

    单档变体: 净期望 = 毛期望 - 双边费 (精确)。
    多档变体: 由每档净U 反推名义额, 再按 笔数x名义额 加权。
    """
    fee2 = FEE[c["market"]] * 2
    ks = [k for k in c["kinds"] if k.get("n")]
    nm0 = c.get("notional") or (c.get("fee_u", 0.0) / (c["n"] * fee2) if fee2 else 0.0)
    if len(ks) <= 1:
        return c["exp"] - fee2, nm0
    num = den = 0.0
    for k in ks:
        denom = k["n"] * (k["exp"] - fee2)
        nm = (k["net_u"] / denom) if denom else 0.0
        w = k["n"] * nm
        num += w * (k["exp"] - fee2)
        den += w
    return (num / den, den / sum(k["n"] for k in ks)) if den else (0.0, 0.0)


def main(argv: list[str]) -> int:
    paths = argv[1:] or ["out/ab_um.txt"]
    for path in paths:
        try:
            rows = parse(path)
        except FileNotFoundError:
            print(f"## {path} —— 尚未生成\n")
            continue
        if not rows:
            continue
        mkt = rows[0]["market"]
        thr = THR[mkt]
        zero = [c for c in rows if not c.get("n")]
        rows = [c for c in rows if c.get("n")]
        print(f"## {path}（{mkt}，门槛 = 毛期望 > {thr * 100:.2f}%）\n")
        if zero:
            print("零成交变体: " + ", ".join(f"`{c['name']}`" for c in zero) + "\n")
        print("| 变体 | 笔数 | 胜率 | 毛期望 | 净期望(校正) | 盈亏比 | PF | 均持 | 判决 |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|:--|")
        for c in rows:
            nexp, nm = corrected(c)
            verdict = ("**过门槛**" if c["exp"] > thr
                       else f"差 {(thr - c['exp']) * 100:.3f}pp")
            print(f"| `{c['name']}` | {c['n']:,} | {c['win'] * 100:.1f}% | "
                  f"{c['exp'] * 100:+.3f}% | {nexp * 100:+.3f}% | {c['rr']:.2f} | "
                  f"{c['pf']:.2f} | {c['hold']:.0f}m | {verdict} |")
        print()
        print("（均名义额: " + ", ".join(
            f"`{c['name']}` {corrected(c)[1]:.2f}U" for c in rows) + "）\n")
        print("分买点（毛期望 / 净期望）：\n")
        print("| 变体 | 档位 | 笔数 | 胜率 | 毛期望 | 净期望 |")
        print("|---|---|---:|---:|---:|---:|")
        for c in rows:
            for k in c["kinds"]:
                if not k.get("n"):
                    print(f"| `{c['name']}` | {k['kind']} | 0 | — | — | — |")
                else:
                    print(f"| `{c['name']}` | {k['kind']} | {k['n']:,} | "
                          f"{k['win'] * 100:.1f}% | {k['exp'] * 100:+.3f}% | "
                          f"{(k['exp'] - FEE[mkt] * 2) * 100:+.3f}% |")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
