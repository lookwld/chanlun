"""「当前所处买点 / 卖点」扫描自检 —— 状态化读法的口径不变量。

运行: python tests/check_scan_state.py [产物前缀]
     默认前缀 `out/scan/b123_spot_state`(即 .md + .csv + _bysym.csv + .json 四份产物)

为什么需要它: 状态化读法在「历史信号表」之上加了一层**失效判定**, 而这层判定
引入了一整类**静默错误** —— 判据方向写反(买点用 `>` / 卖点用 `<`)、把触发根自己
算成"自己破自己"、止损用了收盘价而不是最低价/最高价、回退时取了最老的而不是最新的、
把买卖点混在同一个去重集合里……这些错误**都会让报告照常生成**, 表格看着完全正常。
所以核心手法是: **绕开扫描器, 直接从原始 K 线文件重算一遍失效判定**, 再与产物逐字段比对。

口径(使用者 2026-10-02 指定, 忘了就回来看这里):
  * 周期 8 级: 月线 / 周线 / 3日线 / 日线 / 4小时 / 1小时 / 30分钟 / 15分钟;
  * 每根 K 线冻结的笔上可能同时给买点和卖点, **同一格两者可以并存**;
  * 排序: ① 买点共振数↓ ② 卖点共振数↑ ③ **周期优先**(逐格从左到右比类型优先级)
    ④ 最近确认↓ ⑤ 名称; 类型优先级 = `B1 > B2 > B3 > S3 > S2 > S1`。

不变量:
  1. 每个 (标的, 周期, **方向**) 至多一个当前信号(买卖各一个, 互不挤占);
  2. `cur`/`scur` 必须等于「该方向存活集合里 `bar_ts` 最大」的那个(回退取最新);
  3. 存活判定 == 独立重算: 触发根**之后**是否出现过 最低价<结构位(买) / 最高价>结构位(卖);
  4. 同根 K 线多命中时: 买点按 `PRIO=(B2,B3,B1)`、卖点按 `S3>S2>S1` 取一个;
  5. `dist_pct` == 买 `(close/struct-1)*100` / 卖 `(struct/close-1)*100`;
  6. CSV 与 JSON 逐行一致; 报告总览 / 明细 / 主表三处口径互洽;
  7. 主表格内只可能是 `B1/B2/B3/S1/S2/S3/·`, **买卖并存时两个字母**(买点在前);
  8. 主表排序键在全表上单调(买点共振↓ → 卖点共振↑ → `period_key` → 最新↓);
  9. `period_key` 语义自测: `月线B2 < 周线B1`(周期优先)、`B1 < B2 < B3 < S3 < S2 < S1`;
 10. 宽表 `_bysym.csv` 与主表**逐行逐格一致**, 列序为 长周期在前, 行序同排序键。

失败即意味着产物与判据不一致 —— 先修 `tools/scan_spot_b1b2b3.py`, 别改这里。
"""

from __future__ import annotations

import collections
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 周期表(与 `tools/scan_spot_b1b2b3.py::LEVELS` 同序: 长周期在前) ----
TF = ["1M", "1w", "3d", "1d", "4h", "1h", "30m", "15m"]
LEVEL_CN = {"1M": "月线", "1w": "周线", "3d": "3日线", "1d": "日线", "4h": "4小时",
            "1h": "1小时", "30m": "30分钟", "15m": "15分钟"}
_D = 86_400_000
# ⚠ 月线用 **31 天** —— 月长在 28~31 天浮动, 用 30 天会在 2 月/31 天月的月初
# "提前宣布收盘"(偷看未来)。31 是上界, 偏保守。见 `chanbook/data.py::TF_MS`。
TF_MS = {"1M": 31 * _D, "1w": 7 * _D, "3d": 3 * _D, "1d": _D, "4h": 14_400_000,
         "1h": 3_600_000, "30m": 1_800_000, "15m": 900_000}

BUY_KINDS = ("B1", "B2", "B3")
SELL_KINDS = ("S1", "S2", "S3")
PRIO = {"B2": 0, "B3": 1, "B1": 2}            # 买点同根取舍(生产下单口径)
SELL_PRIO = {"S3": 0, "S2": 1, "S1": 2}       # 卖点同根取舍(展示口径)
KIND_RANK = {"B1": 0, "B2": 1, "B3": 2, "S3": 3, "S2": 4, "S1": 5}
MISS = 9                                      # 该周期没有任何当前信号

_fail = 0
_pass = 0


def ck(cond, msg: str) -> None:
    global _fail, _pass
    if cond:
        _pass += 1
    else:
        _fail += 1
        print(f"  [FAIL] {msg}")


def _read_json(p: str):
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def period_key(kinds_by_tf: dict, tfs: list) -> tuple:
    """与扫描器 `period_key` 逐字对齐: 按周期从长到短**逐格**取该格最优类型优先级。

    元组比较天然是「逐格从左到右, 先分出大小者胜」⇒ `月线B2` 赢 `周线B1`。
    """
    return tuple(min((KIND_RANK[k] for k in (kinds_by_tf.get(t) or ())), default=MISS)
                 for t in tfs)


def cell_rank(cell: str) -> int:
    """把渲染出来的格内容(`B2 S1` / `·`)折回一个优先级数字。"""
    ks = [x for x in cell.split() if x in KIND_RANK]
    return min((KIND_RANK[k] for k in ks), default=MISS)


def recompute_cur(hits: list, ext: list, ts: list, n_closed: int,
                  side: str) -> dict | None:
    """**独立重算**当前信号 —— 刻意不复用 `tools/` 里的任何函数。

    与实现唯一的共同约定是判据本身:
      买点: 触发根之后有根**最低价 < 结构位** ⇒ 失效;
      卖点: 触发根之后有根**最高价 > 结构位** ⇒ 失效。
    """
    idx = {int(t): i for i, t in enumerate(ts[:n_closed])}
    alive = []
    for h in hits:
        bi = idx.get(int(h["bar_ts"]))
        if bi is None:
            continue
        price = float(h["price"])
        if price <= 0:
            continue
        if side == "buy":
            dead = any(float(ext[j]) < price for j in range(bi + 1, n_closed))
        else:
            dead = any(float(ext[j]) > price for j in range(bi + 1, n_closed))
        if dead:
            continue
        alive.append(h)
    if not alive:
        return None
    newest = max(int(h["bar_ts"]) for h in alive)
    cand = [h for h in alive if int(h["bar_ts"]) == newest]
    cand.sort(key=lambda h: (PRIO if side == "buy" else SELL_PRIO)[h["kind"]])
    return cand[0]


AGE_RE = re.compile(r"^\d+(\.\d+)?(分钟|小时|天)$")


def age_hours(s: str):
    """把 `human_age` 渲染的年龄折回小时, 用于「最新 ≤ 最老」的单调性检查。"""
    m = re.match(r"^(\d+(?:\.\d+)?)(分钟|小时|天)$", s.strip())
    if not m:
        return None
    v, u = float(m.group(1)), m.group(2)
    return v / 60 if u == "分钟" else (v if u == "小时" else v * 24)


def main() -> int:
    prefix = sys.argv[1] if len(sys.argv) > 1 else "out/scan/b123_spot_state"
    md_p = os.path.join(ROOT, prefix + ".md")
    csv_p = os.path.join(ROOT, prefix + ".csv")
    js_p = os.path.join(ROOT, prefix + ".json")
    bs_p = os.path.join(ROOT, prefix + "_bysym.csv")
    for p in (md_p, csv_p, js_p):
        if not os.path.exists(p):
            print(f"缺产物: {p}")
            print("先跑: python tools/scan_spot_b1b2b3.py --state")
            return 2

    # ---------------- 0) 测试自身的口径自测 ----------------
    # 「长时间周期的 B2 > 短时间周期的 B1」以及 `B1>B2>B3>S3>S2>S1` —— 先证明
    # 本文件重算用的 `period_key` 就是这两句话, 否则后面所有排序断言都无意义。
    ck(period_key({"1M": ["B2"]}, TF) < period_key({"1w": ["B1"]}, TF),
       "period_key 语义错: 月线B2 应排在 周线B1 之前(周期优先)")
    ck(period_key({"1M": ["B1"]}, TF) < period_key({"1M": ["B2"]}, TF) <
       period_key({"1M": ["B3"]}, TF) < period_key({"1M": ["S3"]}, TF) <
       period_key({"1M": ["S2"]}, TF) < period_key({"1M": ["S1"]}, TF),
       "period_key 语义错: 同周期内类型优先级应为 B1>B2>B3>S3>S2>S1")
    ck(period_key({"1M": ["B3"]}, TF) < period_key({"15m": ["B1"]}, TF),
       "period_key 语义错: 任意长周期类型都应压过任意短周期类型")

    D = _read_json(js_p)
    kl_dir = os.path.join(ROOT, D["kl_dir"])
    md = open(md_p, encoding="utf-8").read()

    # ---------------- 1) (标的,周期,方向) 唯一 ----------------
    with open(csv_p, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    ck(rows and "side" in rows[0], "CSV 缺 `side` 列 —— 买卖点无法区分")
    keys = [(r["symbol"], r["level"], r["side"]) for r in rows]
    dup = [k for k, n in collections.Counter(keys).items() if n > 1]
    ck(not dup, f"CSV 存在同 (标的,周期,方向) 多行(违反『每方向至多一个』): {dup[:5]}")
    bad_kind = [r for r in rows
                if (r["side"] == "buy" and r["kind"] not in BUY_KINDS)
                or (r["side"] == "sell" and r["kind"] not in SELL_KINDS)
                or r["side"] not in ("buy", "sell")]
    ck(not bad_kind, f"CSV 出现与 side 不符的 kind: {[(r['symbol'], r['side'], r['kind']) for r in bad_kind[:5]]}")
    ck(all(r["level"] in TF for r in rows),
       f"CSV 出现未知周期: {sorted({r['level'] for r in rows} - set(TF))}")
    n_buy_rows = sum(1 for r in rows if r["side"] == "buy")
    n_sell_rows = sum(1 for r in rows if r["side"] == "sell")
    print(f"当前信号行数 = {len(rows)}(买点 {n_buy_rows} / 卖点 {n_sell_rows})")

    # ---------------- 2) CSV ↔ JSON 一致 ----------------
    jmap = {}
    for r in D["results"]:
        if r["err"]:
            continue
        for side, key in (("buy", "cur"), ("sell", "scur")):
            c = r.get(key)
            if c:
                jmap[(r["sym"], r["tf"], side)] = (c, r)
    ck({(r["symbol"], r["level"], r["side"]) for r in rows} == set(jmap),
       "CSV 的 (标的,周期,方向) 集合 != JSON 的 cur/scur 集合")
    for r in rows:
        k = (r["symbol"], r["level"], r["side"])
        if k not in jmap:
            continue
        cur, _res = jmap[k]
        ck(r["kind"] == cur["kind"], f"{k} kind 不一致")
        ck(abs(float(r["struct_price"]) - float(cur["price"])) <=
           1e-9 * max(1.0, abs(float(cur["price"]))), f"{k} 结构位不一致")
        ck(int(r["trigger_ts"]) == int(cur["bar_ts"]), f"{k} 确立时间不一致")
        ck(int(r["n_alive"]) == int(cur["n_alive"]), f"{k} n_alive 不一致")
        ck(r["dist_pct"] == f"{cur['dist_pct']:.2f}", f"{k} dist_pct 不一致")

    # ---------------- 3~5) 独立重算失效判定(买卖各一遍) ----------------
    kl_cache: dict = {}

    def _kl(sym: str, tf: str):
        key = (sym, tf)
        if key not in kl_cache:
            kp = os.path.join(kl_dir, f"{sym}_{tf}.json")
            if not os.path.exists(kp):
                kl_cache[key] = None
            else:
                rows_kl = _read_json(kp)
                kl_cache[key] = ([int(x[0]) for x in rows_kl],
                                 [float(x[3]) for x in rows_kl],   # low
                                 [float(x[2]) for x in rows_kl])   # high
        return kl_cache[key]

    n_checked = n_nojson = 0
    for (sym, tf, side), (cur, res) in jmap.items():
        kk = _kl(sym, tf)
        if kk is None:
            n_nojson += 1
            continue
        ts, lows, highs = kk
        asof = int(res.get("asof_ms") or 0)
        tf_ms = TF_MS[tf]
        n_closed = sum(1 for t in ts if t + tf_ms <= asof)
        hits = res["hits"] if side == "buy" else res["shits"]
        ext = lows if side == "buy" else highs
        exp = recompute_cur(hits, ext, ts, n_closed, side)
        n_checked += 1
        if exp is None:
            ck(False, f"{sym}/{tf}/{side} 独立重算为『无当前信号』, 报告却给了 {cur['kind']}")
            continue
        ck(exp["kind"] == cur["kind"],
           f"{sym}/{tf}/{side} 类型不一致: 重算={exp['kind']} 报告={cur['kind']}")
        ck(int(exp["bar_ts"]) == int(cur["bar_ts"]),
           f"{sym}/{tf}/{side} 回退目标不一致: 重算={exp['bar_ts']} 报告={cur['bar_ts']}"
           f"(回退必须取『最新的存活着』而不是最老的)")
        # 回退语义: cur 必须是存活集里 bar_ts 最大的那个
        newest_all = max(int(h["bar_ts"]) for h in hits)
        idx = {int(t): i for i, t in enumerate(ts[:n_closed])}
        alive_ts = set()
        for h in hits:
            bi = idx.get(int(h["bar_ts"]))
            if bi is None:
                continue
            p = float(h["price"])
            if p <= 0:
                continue
            if side == "buy":
                dead = any(float(lows[j]) < p for j in range(bi + 1, n_closed))
            else:
                dead = any(float(highs[j]) > p for j in range(bi + 1, n_closed))
            if not dead:
                alive_ts.add(int(h["bar_ts"]))
        if alive_ts:
            ck(int(cur["bar_ts"]) == max(alive_ts),
               f"{sym}/{tf}/{side} cur 不是存活集合里最新的")
            if int(cur["bar_ts"]) != newest_all:
                ck(int(cur["bar_ts"]) < newest_all,
                   f"{sym}/{tf}/{side} cur 晚于最新事件, 说明没走回退")
        # 距结构位(两个方向都是「还有多少安全边际」, >0 表示未被击穿)
        cl = float(res["close"])
        exp_dist = ((cl / float(cur["price"]) - 1.0) if side == "buy"
                    else (float(cur["price"]) / cl - 1.0)) * 100.0
        ck(abs(exp_dist - float(cur["dist_pct"])) < 1e-6,
           f"{sym}/{tf}/{side} dist_pct 不是 (close/struct-1)*100 方向的口径")
    print(f"独立重算覆盖 {n_checked} 个 (标的,周期,方向)"
          + (f", 跳过 {n_nojson} 个(缺 K 线文件)" if n_nojson else ""))
    ck(n_checked > 0, "没有任何组合被独立重算 —— 校验形同虚设")

    # ---------------- 6) 报告三处口径互洽 ----------------
    lines = md.splitlines()
    hdr = next((s for s in lines if s.startswith("| # | 标的 |")), "")
    msg = "找不到主表表头"
    ck(bool(hdr), msg)
    want_hdr = ("| # | 标的 | " + " | ".join(LEVEL_CN[t] for t in TF)
                + f" | **买点共振** | **卖点共振** | 现价({TF[-1]}) | 最新确认 | 最老确认 |")
    ck(hdr == want_hdr, f"主表列序不符规格:\n      实际 {hdr}\n      期望 {want_hdr}")

    i_main = lines.index(hdr) if hdr in lines else -1
    body = []
    if i_main >= 0:
        for s in lines[i_main + 2:]:
            if not s.startswith("|"):
                break
            body.append([x.strip() for x in s.strip("|").split("|")])
    ck(all(len(b) == 2 + len(TF) + 5 for b in body), "主表列数不对")

    # 用逐信号 CSV 独立聚合出「每标的每周期格内容」与双共振
    per = collections.defaultdict(lambda: {"k": {}, "sk": {}})
    for r in rows:
        (per[r["symbol"]]["k"] if r["side"] == "buy"
         else per[r["symbol"]]["sk"])[r["level"]] = r["kind"]

    def cell_of(e, t) -> str:
        ks = [x for x in (e["k"].get(t), e["sk"].get(t)) if x]
        return " ".join(ks) if ks else "·"

    ck(len(body) == len(per), f"主表行数 {len(body)} != 有当前信号的标的数 {len(per)}")
    for b in body:
        sym = b[1].strip("`")
        e = per[sym]
        got = b[2:2 + len(TF)]
        want = [cell_of(e, t) for t in TF]
        ck(got == want, f"主表 {sym} 格内容不符: 报告={got} 明细={want}")
        nby = sum(1 for t in TF if e["k"].get(t))
        nse = sum(1 for t in TF if e["sk"].get(t))
        ck(b[2 + len(TF)].strip("*") == f"{nby}/{len(TF)}",
           f"主表 {sym} 买点共振 {b[2+len(TF)]} != {nby}/{len(TF)}")
        ck(b[3 + len(TF)].strip("*") == f"{nse}/{len(TF)}",
           f"主表 {sym} 卖点共振 {b[3+len(TF)]} != {nse}/{len(TF)}")
        # 格内记号必须与两个方向的聚合一致(买卖并存 ⇒ 两个字母, 买点在前)
        for c, t in zip(got, TF):
            if c == "·":
                continue                       # `·` = 该周期买卖都没有, 是合法值
            ks = c.split()
            ck(all(x in KIND_RANK for x in ks),
               f"主表 {sym}/{t} 格内出现非买卖点记号: {c}")
            ck(len(ks) <= 2, f"主表 {sym}/{t} 格内超过 2 个记号: {c}")
            ck(len(set(ks)) == len(ks), f"主表 {sym}/{t} 格内有重复记号: {c}")
            if len(ks) == 2:
                ck(ks[0] in BUY_KINDS and ks[1] in SELL_KINDS,
                   f"主表 {sym}/{t} 买卖并存时买点未写在前: {c}")
        ck(AGE_RE.match(b[5 + len(TF)].replace(" ", "")),
           f"主表 {sym} 最新确认格式异常: {b[5+len(TF)]}")
        ck(AGE_RE.match(b[6 + len(TF)].replace(" ", "")),
           f"主表 {sym} 最老确认格式异常: {b[6+len(TF)]}")
        an, ao = age_hours(b[5 + len(TF)]), age_hours(b[6 + len(TF)])
        if an is not None and ao is not None:
            ck(an <= ao + 1e-9,
               f"主表 {sym} 「最新确认」应不早于「最老确认」: {b[5+len(TF)]} vs {b[6+len(TF)]}")

    # 排序键单调: 买点共振↓ → 卖点共振↑ → period_key(周期优先) → 最新确认↓ → 名称
    newest_of = collections.defaultdict(int)
    for (sym, tf, side), (cur, _r) in jmap.items():
        newest_of[sym] = max(newest_of[sym], int(cur["bar_ts"]))
    key = []
    for b in body:
        sym = b[1].strip("`")
        e = per[sym]
        nby = sum(1 for t in TF if e["k"].get(t))
        nse = sum(1 for t in TF if e["sk"].get(t))
        kinds = {t: [x for x in (e["k"].get(t), e["sk"].get(t)) if x] for t in TF}
        key.append((-nby, nse, period_key(kinds, TF), -newest_of[sym], sym))
    ck(key == sorted(key),
       "主表排序键不单调(买点共振↓ / 卖点共振↑ / 周期优先 / 最新确认↓ 被破坏)")
    # 同一(买点共振, 卖点共振)档内的次序必须**纯粹**由 `period_key` 决定 ——
    # 用**报告自己渲染出来的格子**再算一遍, 不走 CSV, 免得两边同时错。
    tier_pk = collections.defaultdict(list)
    for b in body:
        sym = b[1].strip("`")
        e = per[sym]
        nby = sum(1 for t in TF if e["k"].get(t))
        nse = sum(1 for t in TF if e["sk"].get(t))
        tier_pk[(nby, nse)].append(tuple(cell_rank(c) for c in b[2:2 + len(TF)]))
    bad = [k for k, v in tier_pk.items() if v != sorted(v)]
    ck(not bad, f"同共振档内的『周期优先』次序被破坏, 档位: {bad[:5]}")

    # 共振档位分布求和 == 主表行数
    m = re.search(r"### 共振档位分布(.*?)## 三、", md, re.S)
    ck(bool(m), "找不到共振档位分布")
    if m:
        pats = re.findall(r"\| \*\*(\d)/8\*\* \| (\d+) \| \*\*(\d)/8\*\* \| (\d+) \|",
                          m.group(1))
        tb = sum(int(x[1]) for x in pats)
        ts_ = sum(int(x[3]) for x in pats)
        ck(tb == len(body), f"买点档位分布求和 {tb} != 主表行数 {len(body)}")
        ck(ts_ == len(body), f"卖点档位分布求和 {ts_} != 主表行数 {len(body)}")

    # 明细行数 == CSV 行数(买点表 / 卖点表各一张)
    for tf in TF:
        for side, w in (("买点", "buy"), ("卖点", "sell")):
            mm = re.search(rf"### {LEVEL_CN[tf]}\({tf}\) —— (\d+) 个标的当前处于{side}", md)
            if mm:
                want = sum(1 for r in rows
                           if r["level"] == tf and r["side"] == w)
                ck(int(mm.group(1)) == want,
                   f"§三 {tf}/{side} 明细行数 {mm.group(1)} != CSV 中该级别该方向行数 {want}")
            else:
                ck(False, f"§三 缺 {tf} 的{side}小节标题")

    # ---------------- 7) 按标的聚合宽表 (_bysym.csv) ----------------
    # 这张表是「一标的一行」的展示层, 使用者直接拿它进 Excel。它最容易出的错是
    # **列序被改成短周期在前** 或 **与主表排序不一致** —— 两种都不会报错,
    # 表格照样能看。所以这里逐行逐格与主表比对。
    ck(os.path.exists(bs_p), f"缺宽表产物: {bs_p}")
    if os.path.exists(bs_p):
        with open(bs_p, encoding="utf-8", newline="") as fh:
            bs = list(csv.reader(fh))
        head = (["rank", "symbol"] + [LEVEL_CN[t] for t in TF]
                + ["buy_resonance", "sell_resonance", "last_close",
                   "newest_sig_time", "oldest_sig_time", "cells_long_to_short"])
        ck(bs[0] == head, f"宽表列序不对:\n      实际 {bs[0]}\n      期望 {head}")
        ck(len(bs) - 1 == len(body),
           f"宽表行数 {len(bs)-1} != 主表行数 {len(body)}")
        ck([r[1] for r in bs[1:]] == [b[1].strip("`") for b in body],
           "宽表行序与主表不一致(排序键没有同步)")
        for r, b in zip(bs[1:], body):
            sym = r[1]
            cells = r[2:2 + len(TF)]
            ck(cells == b[2:2 + len(TF)],
               f"宽表 {sym} 各周期信号与主表不符: {cells} vs {b[2:2+len(TF)]}")
            ck(r[2 + len(TF)] == b[2 + len(TF)].strip("*"),
               f"宽表 {sym} 买点共振 {r[2+len(TF)]} != 主表 {b[2+len(TF)]}")
            ck(r[3 + len(TF)] == b[3 + len(TF)].strip("*"),
               f"宽表 {sym} 卖点共振 {r[3+len(TF)]} != 主表 {b[3+len(TF)]}")
            # 「信号串」必须按 长→短 串, 且与格子一致
            want_str = " ".join(f"{c}({LEVEL_CN[t]})"
                                for c, t in zip(cells, TF) if c != "·") or "·"
            ck(r[-1] == want_str,
               f"宽表 {sym} 信号串不符: {r[-1]} 应为 {want_str}")
        # 宽表自身的排序键必须单调 —— 不能只依赖主表已排好
        k2 = []
        for r in bs[1:]:
            sym = r[1]
            e = per[sym]
            nby = sum(1 for t in TF if e["k"].get(t))
            nse = sum(1 for t in TF if e["sk"].get(t))
            kinds = {t: [x for x in (e["k"].get(t), e["sk"].get(t)) if x] for t in TF}
            k2.append((-nby, nse, period_key(kinds, TF), -newest_of[sym], sym))
        ck(k2 == sorted(k2), "宽表排序键不单调")
        print(f"宽表 {len(bs)-1} 行 / {len(head)} 列")

    # ---------------- 8) 总览表 ----------------
    tbl = md.split("## 一、总览")[1].split("## 二、")[0]
    for tf in TF:
        line = next((s for s in tbl.splitlines()
                     if s.startswith("| **") and f"| {tf} |" in s), None)
        if not line:
            ck(False, f"总览缺 {tf}")
            continue
        c = [x.strip() for x in line.strip("|").split("|")]
        ck(len(c) == 8, f"总览 {tf} 列数不对: {len(c)}")
        nbuy = sum(1 for r in rows if r["level"] == tf and r["side"] == "buy")
        nsell = sum(1 for r in rows if r["level"] == tf and r["side"] == "sell")
        nboth = sum(1 for s in per if per[s]["k"].get(tf) and per[s]["sk"].get(tf))
        ck(int(c[3].strip("*")) == nbuy, f"总览 {tf} 有当前买点数 {c[3]} != CSV {nbuy}")
        ck(int(c[5].strip("*")) == nsell, f"总览 {tf} 有当前卖点数 {c[5]} != CSV {nsell}")
        ck(int(c[7]) == nboth, f"总览 {tf} 买卖同时有效 {c[7]} != 独立重算 {nboth}")
        bk = collections.Counter(r["kind"] for r in rows
                                 if r["level"] == tf and r["side"] == "buy")
        sk = collections.Counter(r["kind"] for r in rows
                                 if r["level"] == tf and r["side"] == "sell")
        ck(c[4] == f"{bk['B1']}/{bk['B2']}/{bk['B3']}",
           f"总览 {tf} 买点构成 {c[4]} != {bk['B1']}/{bk['B2']}/{bk['B3']}")
        ck(c[6] == f"{sk['S1']}/{sk['S2']}/{sk['S3']}",
           f"总览 {tf} 卖点构成 {c[6]} != {sk['S1']}/{sk['S2']}/{sk['S3']}")

    print(f"\n通过 {_pass} 项, 失败 {_fail} 项")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
