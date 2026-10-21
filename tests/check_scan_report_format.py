"""扫描报告格式自检 —— 「标的共振总表」的口径不变量。

运行: python tests/check_scan_report_format.py [报告前缀]
     默认前缀 `out/scan/b123_spot`(即 .md + .csv + _bysym.csv 三份产物)

为什么需要它: 主表把「一行一个信号」重排成「一行一个标的 × 列=周期」, 重排会引入
一整类**静默错误** —— 顺序错了、格内的买点/卖点漏了、共振数算错了, 表面上都还是
能看的表格, 肉眼看不出。所以这里的核心手法是**从逐信号明细 CSV 独立重算一遍**,
再与渲染出来的 Markdown 逐行逐格比对:
  1. 行序 == 独立重算的排序键;
  2. 每行格内 == CSV 聚合结果, 口径 = **该方向窗口内最后一个信号的类型**
     (用户 2026-10-02 指定: 同方向只标最后一个 ⇒ 格内最多 2 个字母, 买点在前);
     同一根 K 线上多命中时按下单口径取一个(买点 `B2>B3>B1`, 卖点 `S3>S2>S1`);
  3. 买点/卖点共振数、买/卖信号数 == 独立重算;
  4. 排序键本身在全表上单调(无逆序), 且**同共振档内纯粹由周期优先决定**;
  5. `period_key` 语义自测: `月线B2 < 周线B1`, 且 `B1 < B2 < B3 < S3 < S2 < S1`;
  6. 分档标的数求和 == 主表行数;
  7. 宽表 `_bysym.csv` 与主表逐行逐格一致(列序: 标的名紧跟各周期格)。

失败即意味着报告与明细不一致 —— 先修 `tools/scan_spot_b1b2b3.py`, 别改这里。
"""

from __future__ import annotations

import collections
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 周期表(与 `tools/scan_spot_b1b2b3.py::LEVELS` 同序: 长周期在前) ----
TF = ["1M", "1w", "3d", "1d", "4h", "1h", "30m", "15m"]
LEVEL_CN = {"1M": "月线", "1w": "周线", "3d": "3日线", "1d": "日线", "4h": "4小时",
            "1h": "1小时", "30m": "30分钟", "15m": "15分钟"}

KIND_RANK = {"B1": 0, "B2": 1, "B3": 2, "S3": 3, "S2": 4, "S1": 5}
MISS = 9

# 同一根 K 线上多命中时取哪个 —— 必须与扫描器逐字一致:
# 买点用生产下单口径 `PRIO = {B2:0, B3:1, B1:2}`, 卖点用 `KIND_RANK` 的 S 侧相对次序。
PRIO_BUY = {"B2": 0, "B3": 1, "B1": 2}
PRIO_SELL = {"S3": 0, "S2": 1, "S1": 2}

N_PASS = 0
FAILS: list[str] = []


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)


def period_key(kinds_by_tf: dict, tfs: list) -> tuple:
    """与扫描器 `period_key` 逐字对齐: 按周期从长到短**逐格**取该格最优类型优先级。
    元组比较 = 逐格从左到右, 先分出大小者胜 ⇒ `月线B2` 赢 `周线B1`。"""
    return tuple(min((KIND_RANK[k] for k in (kinds_by_tf.get(t) or ())), default=MISS)
                 for t in tfs)


def cell_rank(cell: str) -> int:
    ks = [x for x in cell.split() if x in KIND_RANK]
    return min((KIND_RANK[k] for k in ks), default=MISS)


def last_kind(by_ts: dict, prio: dict) -> str | None:
    """该方向**最近一根** K 线上触发的类型; 同根多命中时按 `prio` 取一个。
    与扫描器 `_last_kind` 同口径 —— 这是主表格内容的唯一来源。"""
    if not by_ts:
        return None
    ts = max(by_ts)
    return min(by_ts[ts], key=lambda k: prio[k])


def aggregate(rows: list[dict]) -> dict:
    """从逐信号明细重算: 每标的每周期 各方向 {bar_ts: {类型}} / 条数 / 最近时间。"""
    per = collections.defaultdict(
        lambda: {"lv": collections.defaultdict(lambda: collections.defaultdict(set)),
                 "lvs": collections.defaultdict(lambda: collections.defaultdict(set)),
                 "n": collections.Counter(),
                 "sn": collections.Counter(),
                 "new": collections.defaultdict(int)})
    for r in rows:
        e = per[r["symbol"]]
        ts = int(r["trigger_ts"])
        if r["side"] == "buy":
            e["lv"][r["level"]][ts].add(r["kind"])
            e["n"][r["level"]] += 1
        else:
            e["lvs"][r["level"]][ts].add(r["kind"])
            e["sn"][r["level"]] += 1
        e["new"][r["level"]] = max(e["new"][r["level"]], ts)
    return per


def cell_kinds(e, t) -> list:
    """该格最终显示的 1~2 个字母: 买点在前、卖点在后。"""
    kb = last_kind(e["lv"][t], PRIO_BUY)
    ks = last_kind(e["lvs"][t], PRIO_SELL)
    return [k for k in (kb, ks) if k]


def cells_of(e) -> list:
    out = []
    for t in TF:
        ks = cell_kinds(e, t)
        out.append(" ".join(ks) if ks else "·")
    return out


def sym_key(kv) -> tuple:
    """① 买点共振数↓ ② 卖点共振数↑ ③ `period_key`(周期优先) ④ 信号数↓
    ⑤ 最近信号↓ ⑥ 名称。⚠ 第三项的输入必须是**格内实际显示的那 1~2 个字母**
    —— 否则会出现「两行格子一模一样、名次却不同」, 用户会当成 bug。"""
    _sym, e = kv
    nbuy = len([t for t in TF if e["lv"][t]])
    nsell = len([t for t in TF if e["lvs"][t]])
    kinds = {t: cell_kinds(e, t) for t in TF}
    return (-nbuy, nsell, period_key(kinds, TF),
            -sum(e["n"].values()), -max(e["new"].values()), _sym)


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "out/scan/b123_spot"
    csv_p = os.path.join(ROOT, prefix + ".csv")
    md_p = os.path.join(ROOT, prefix + ".md")
    bs_p = os.path.join(ROOT, prefix + "_bysym.csv")
    if not (os.path.exists(csv_p) and os.path.exists(md_p)):
        print(f"跳过: 找不到 {csv_p} / {md_p}(先跑 tools/scan_spot_b1b2b3.py)")
        return 0

    # ---- 测试自身的口径自测 ----
    ck(period_key({"1M": ["B2"]}, TF) < period_key({"1w": ["B1"]}, TF),
       "period_key 语义错: 月线B2 应排在 周线B1 之前(周期优先)")
    ck(period_key({"1M": ["B1"]}, TF) < period_key({"1M": ["B2"]}, TF) <
       period_key({"1M": ["B3"]}, TF) < period_key({"1M": ["S3"]}, TF) <
       period_key({"1M": ["S2"]}, TF) < period_key({"1M": ["S1"]}, TF),
       "period_key 语义错: 同周期内类型优先级应为 B1>B2>B3>S3>S2>S1")

    rows = list(csv.DictReader(open(csv_p, encoding="utf-8")))
    ck(bool(rows) and "side" in rows[0], "CSV 缺 `side` 列 —— 买卖点无法区分")
    exp = sorted(aggregate(rows).items(), key=sym_key)
    md = open(md_p, encoding="utf-8").read()

    # 抽取主表(第二章), 到「共振档位分布」为止
    if "## 二、标的共振总表" not in md:
        print("✗ 报告里找不到「## 二、标的共振总表」")
        return 1
    ck("B1 > B2 > B3 > S3 > S2 > S1" in md,
       "报告正文没有写明类型优先级 `B1 > B2 > B3 > S3 > S2 > S1`")
    seg = md.split("## 二、标的共振总表")[1].split("### 共振档位分布")[0]
    # 列序是用户点名的规格: 标的名**紧跟各周期信号**(长周期在前), 汇总列在后。
    # 一旦有人把「共振」挪回标的名后面, 这里必须报错 —— 那不是排版偏好问题。
    hdr = next((ln for ln in seg.splitlines() if ln.startswith("| # | 标的 |")), "")
    want_hdr = ("| # | 标的 | " + " | ".join(LEVEL_CN[t] for t in TF)
                + f" | **买点共振** | **卖点共振** | 信号数(买/卖) | 现价({TF[-1]}) | 最近信号 |")
    ck(hdr == want_hdr, f"主表列序不符规格:\n      实际 {hdr}\n      期望 {want_hdr}")

    pat = re.compile(r"^\| (\d+) \| `([^`]+)` \| (.*?) \| \*\*(\d)/8\*\* \| "
                     r"\*\*(\d)/8\*\* \| (\d+)/(\d+) \| (\S+) \| "
                     r"(\d\d-\d\d \d\d:\d\d) \|$")
    got = [m for m in (pat.match(ln) for ln in seg.splitlines()) if m]

    print(f"主表 {len(got)} 行 / 明细聚合 {len(exp)} 个标的 / "
          f"逐信号明细 {len(rows)} 条")
    ck(len(got) == len(exp), f"主表行数 {len(got)} != 明细聚合标的数 {len(exp)}")
    if len(got) != len(exp):
        print(f"\n== 扫描报告格式 自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
        for f in FAILS:
            print("  ✗", f)
        return 1

    bad_order = bad_cell = bad_num = bad_shape = 0
    for i, (m, (sym, e)) in enumerate(zip(got, exp), 1):
        n, gsym, cells, gres, gsres, gtot, gstot, _px, _tm = m.groups()
        if int(n) != i or gsym != sym:
            bad_order += 1
            if bad_order <= 3:
                print(f"  ✗ 第 {i} 行: 报告 `{gsym}` vs 期望 `{sym}`")
        nby = len([t for t in TF if e["lv"][t]])
        nse = len([t for t in TF if e["lvs"][t]])
        ntot, nstot = sum(e["n"].values()), sum(e["sn"].values())
        if (int(gres) != nby or int(gsres) != nse
                or int(gtot) != ntot or int(gstot) != nstot):
            bad_num += 1
            if bad_num <= 3:
                print(f"  ✗ {sym}: 共振买 {gres}/{nby} 卖 {gsres}/{nse}; "
                      f"信号买 {gtot}/{ntot} 卖 {gstot}/{nstot}")
        # 用户点名的形状规则: **每方向只标最后一个 ⇒ 格内 ≤2 个字母, 买点在前**。
        # `·` = 该方向无信号, 是合法值(这里踩过坑: 第一版把它当成非法记号)。
        for c in cells.split("|"):
            ks = c.split()
            if ks == ["·"]:
                continue
            if len(ks) > 2 or any(x not in KIND_RANK for x in ks) or (
                    len(ks) == 2 and not (ks[0].startswith("B") and ks[1].startswith("S"))):
                bad_shape += 1
                if bad_shape <= 3:
                    print(f"  ✗ {sym} 格 `{c}` 违反「每方向只标一个、买点在前」")
                break
        want = " | ".join(cells_of(e))
        if cells.strip() != want:
            bad_cell += 1
            if bad_cell <= 3:
                print(f"  ✗ {sym} 格内不符:\n      报告 {cells.strip()}\n      期望 {want}")

    ck(bad_order == 0, f"{bad_order} 行顺序与独立重算不一致")
    ck(bad_num == 0, f"{bad_num} 行的共振数/信号数不符")
    ck(bad_cell == 0, f"{bad_cell} 行格内类型或排序不符")
    ck(bad_shape == 0, f"{bad_shape} 行的格形状违反「每方向只标一个、买点在前」")

    # 排序键单调
    viol = [(a[0], b[0]) for a, b in zip(exp, exp[1:]) if sym_key(a) > sym_key(b)]
    ck(not viol, f"排序键出现 {len(viol)} 处逆序, 例: {viol[:3]}")

    # 同共振档内**纯粹**由 `period_key` 决定次序 —— 用报告自己渲染出的格子再算一遍
    tier_pk = collections.defaultdict(list)
    for m, (sym, e) in zip(got, exp):
        cells = m.group(3)
        nby = len([t for t in TF if e["lv"][t]])
        nse = len([t for t in TF if e["lvs"][t]])
        tier_pk[(nby, nse)].append(tuple(cell_rank(c) for c in cells.split("|")))
    bad = [k for k, v in tier_pk.items() if v != sorted(v)]
    ck(not bad, f"同共振档内的『周期优先』次序被破坏, 档位: {bad[:5]}")

    # 用户点名规则的正向见证: 必须真的存在「长周期·较差类型」压过「短周期·较好类型」的行
    witness = 0
    for m, (sym, e) in zip(got, exp):
        ranks = [cell_rank(c) for c in m.group(3).split("|")]
        for j in range(1, len(TF)):
            if ranks[j] < max(ranks[:j]):
                witness += 1
                break
    print(f"周期优先见证行数 = {witness}(存在『长周期较差类型』被排在『短周期较好类型』之前)")
    ck(witness > 0, "没有一行体现「周期优先」—— 排序可能退化成了类型优先")

    # 分档求和 == 主表行数
    tier = collections.Counter(len([t for t in TF if e["lv"][t]]) for _, e in exp)
    stier = collections.Counter(len([t for t in TF if e["lvs"][t]]) for _, e in exp)
    ck(sum(tier.values()) == len(got),
       f"分档求和 {sum(tier.values())} != 主表行数 {len(got)}")
    ck(sum(stier.values()) == len(got),
       f"卖点分档求和 {sum(stier.values())} != 主表行数 {len(got)}")

    # 宽表与主表逐行逐格一致
    ck(os.path.exists(bs_p), f"缺宽表产物: {bs_p}")
    if os.path.exists(bs_p):
        with open(bs_p, encoding="utf-8", newline="") as fh:
            bs = list(csv.reader(fh))
        head = (["rank", "symbol"] + [LEVEL_CN[t] for t in TF]
                + ["buy_resonance", "sell_resonance", "signals_buy", "signals_sell",
                   "last_close", "newest_ts", "newest_time"])
        ck(bs[0] == head, f"宽表列序不对:\n      实际 {bs[0]}\n      期望 {head}")
        ck(len(bs) - 1 == len(got), f"宽表行数 {len(bs)-1} != 主表行数 {len(got)}")
        ck([r[1] for r in bs[1:]] == [m.group(2) for m in got],
           "宽表行序与主表不一致(排序键没有同步)")
        for r, m in zip(bs[1:], got):
            got_cells = [x.strip() for x in m.group(3).split("|")]
            ck(r[2:2 + len(TF)] == got_cells,
               f"宽表 {r[1]} 各周期格与主表不符: {r[2:2+len(TF)]} vs {got_cells}")
            ck(r[2 + len(TF)] == f"{m.group(4)}/8"
               and r[3 + len(TF)] == f"{m.group(5)}/8",
               f"宽表 {r[1]} 双共振与主表不符: {r[2+len(TF)]}/{r[3+len(TF)]} "
               f"vs {m.group(4)}/8,{m.group(5)}/8")

    # 附录与主表同源
    dis = {r["symbol"] for r in rows}
    ck(len(dis) == len(exp), "明细 CSV 的标的去重数与主表不一致")

    print(f"\n== 扫描报告格式 自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
    print(f"   买点共振分档: " + " / ".join(f"{k}/8×{tier[k]}"
                                          for k in sorted(tier, reverse=True)))
    print(f"   卖点共振分档: " + " / ".join(f"{k}/8×{stier[k]}"
                                          for k in sorted(stier, reverse=True)))
    for f in FAILS:
        print("  ✗", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
