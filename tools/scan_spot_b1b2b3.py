"""币安现货全市场 B1/B2/B3 扫描 —— 只扫**数字代币**, 按级别出表。

与 `tools/scan_all_market.py` 的区别:
  1. **资产分类**: 排除股票代币(bStocks) / 稳定币 / 实物锚定(黄金等), 只留数字代币;
  2. **窗口按级别自适应**: 原工具用固定 72 小时窗口, 对日线等于只看 3 天 —— 长周期
     几乎必然空表。本工具改成"最近 `--bars` 根 K 线 且 最近 `--legs` 根已冻结笔",
     同样的 bar 数在每个级别覆盖的日历时间天然不同(5m≈10h, 1d≈120d), 这正是
     "长周期看更长历史"的结构口径;
  3. **输出以「标的」为行**: 一行一个标的, 列 = 各时间周期(长周期在前), 格内 = 该
     周期**最后一个**(最近触发)的买点与卖点 —— **每个方向只标一个, 格内最多 2 个
     字母**(用户 2026-10-02 指定; 类型集合见附录与逐信号 CSV)。排序 = 共振级别数多的在前, 共振数相同时
     含更长周期的在前 —— 一眼看出"哪些标的在多个级别上同时出买点"。
     按级别分组的明细降为附录, 供逐级别排查。

────────────────────────────────────────────────────────────────────────
资产分类依据(全部可复核)
────────────────────────────────────────────────────────────────────────
* **股票代币 bStocks**: 币安 2026-06 起推出的代币化美股, 命名规则 = **股票代码 + "B"**
  (NVDAB=英伟达 / TSLAB=特斯拉 / AAPLB=苹果 / MSFTB=微软 / QQQB=纳指100ETF /
   SPYB=标普500ETF / MUB=美光 ...)。见币安官方教程与 ADGM FSRA 上市文件。
  ⇒ 规则: base asset 以 "B" 结尾 ⇒ 判为 bStock, **除非**在 `CRYPTO_END_B` 白名单里。
  白名单 8 个是真实加密货币(恰好以 B 结尾), 逐个核对过:
      ARB=Arbitrum  BB=BounceBit  BNB=BNB  CKB=Nervos CKB  DGB=DigiByte
      SHIB=Shiba Inu  TRB=Tellor  YB=Yield Basis
  ⚠ 别用"波动率/时段集中度"自动判别 —— 实测 bStocks 是 24/7 合成定价, 其
    美股时段波动集中度只有 0.89~2.84(COINB=1.00, BABAB=0.89 明明是股票),
    与真币(ARB=1.10, BNB=1.22)完全重叠, 分不开。名称规则才是可靠判据。

* **稳定币 / 法币锚定**: 与 USD/EUR/KGS 等 1:1 锚定。含本环境里的
      KGST = 吉尔吉斯斯坦国家稳定币(1:1 锚定索姆 KGS, 2025-12-24 上线币安) —— 
      实测波动率仅 0.0004/小时(与 USDC 同级), 与 PAXG/XAUT 相关性 ≈ 0(不是黄金)。

* **实物锚定**: PAXG(黄金) / XAUT(Tether Gold) / ...
────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook.level import Level                                  # noqa: E402
from chanbook.points import B1, B2, B3                            # noqa: E402
from chanbook.sell import S1, S2, S3, SellScanner                 # noqa: E402
from chanbook.strategy import BookCfg                             # noqa: E402

# ---- 扫描周期(用户 2026-10-02 指定; 长周期在前, 顺序即表头列序) ----
LEVELS = ["1M", "1w", "3d", "1d", "4h", "1h", "30m", "15m"]
LEVEL_CN = {"1M": "月线", "1w": "周线", "3d": "3日线", "1d": "日线", "4h": "4小时",
            "1h": "1小时", "30m": "30分钟", "15m": "15分钟"}

# 最低根数。500 是 `level.MIN_KLINES["op"]` —— 那是给**回测**定的样本量下限;
# 扫描只是"看现在处于什么结构", 门槛可以低得多。更关键的是粗周期受数据源物理
# 上限约束(币安 2017-08 上线 ⇒ 月线总共才 111 根), 用 500 会让月线/周线**整列全灭**。
# 实测(BTC): 月线 111 根已能出 8 笔 / 2 个中枢 / 1 个 B3 + 1 个 S1。
MIN_BARS = 500
MIN_BARS_BY_TF = {"1M": 48, "1w": 100, "3d": 200}


def min_bars_of(tf: str) -> int:
    return MIN_BARS_BY_TF.get(tf, MIN_BARS)


BUY = (B1, B2, B3)
SELL = (S1, S2, S3)
BUY_LABEL = {B1: "买①背驰", B2: "买②确认", B3: "买③三买"}
SELL_LABEL = {S1: "卖①背驰", S2: "卖②确认", S3: "卖③三卖"}

# 同一根 K 线同时命中多个**买点**时取哪一个 —— 与 `chanbook/strategy.py::PRIORITY` 同序
# (B2 是一买之后的第一次确认, 结构信息量最大; B3 顺势确认; B1 逆势风险最高)。
# 数字越小越优先。
PRIO = {B2: 0, B3: 1, B1: 2}

# 主表排序用的**展示优先级**(用户 2026-10-02 指定): `B1 > B2 > B3 > S3 > S2 > S1`。
#   * 买点全部排在卖点之前(看表先找买点);
#   * 买点内部 B1 优先(趋势末端的第一个买点, 赔率最高);
#   * 卖点内部 **S3 优先**(趋势确认后的卖点最可靠; S1 是逆势逃顶, 风险最高)。
# ⚠ 与 `PRIO` **不是一回事**: `PRIO` 是同一根 K 线多命中时的**下单**取舍(B2 优先),
#   这里是**看表找票**的排序。别"顺手统一"成一个常量。
KIND_RANK = {B1: 0, B2: 1, B3: 2, S3: 3, S2: 4, S1: 5}
MISS = 9                                  # 该周期没有当前信号


def sell_kw_from(scanner_kw: dict) -> dict:
    """把买点的 `scanner_kw` 映射成 `SellScanner` 的参数名(严格镜像口径)。

    ⚠ 两个扫描器的参数名不同(b 侧 `b2_max_bi` ↔ s 侧 `s2_max_bi`), 且买点独有
    `b1_path` / `b1_trend_adjacent` / `vol_gate` 三个键 —— 直接 `**scanner_kw`
    会 TypeError。S/ 侧没有这三项的对应物, 只能丢弃。
    """
    m = {"pull_bi": "pull_bi", "beichi_tol": "beichi_tol",
         "beichi_strict": "beichi_strict", "b2_max_bi": "s2_max_bi",
         "b1_max_lag": "s1_max_lag", "b1_zs_window": "s1_zs_window",
         "b3_zs_window": "s3_zs_window", "b1_allow_no_zs": "s1_allow_no_zs"}
    out = {m[k]: v for k, v in scanner_kw.items() if k in m}
    out["enable"] = SELL
    return out


def period_key(kinds_by_tf: dict, tfs: list) -> tuple:
    """主表排序的**第二级**键: 按周期从长到短**逐格**比类型优先级。

    「长时间周期的 B2 > 短时间周期的 B1」就是这条 —— 逐格从左向右比, 一旦分出
    大小就结束: 月线那格(B2=1)先在周线那格(B1=0)之前被比较, 所以月线 B2 赢。

    ⚠ 上一版(2026-10-02 上午)是「**类型优先**」: 把所有买点摊平成 `(类型, 周期)` 对
    再升序排, 于是 15分钟B1 会压过 日线B2。用户当天改回**周期优先** —— 别再改回去。

    每格可能同时有买点和卖点(实测 29.1% 的 (标的,周期) 两者都成立), 取该格
    优先级最高的那个作为代表值(`B1 > B2 > B3 > S3 > S2 > S1`)。
    """
    return tuple(
        min((KIND_RANK[k] for k in (kinds_by_tf.get(t) or ())), default=MISS)
        for t in tfs)

# ---------------------------------------------------------------- 排除清单
# 稳定币 / 法币锚定
EXCL_STABLE = {
    "USDC", "TUSD", "FDUSD", "USDP", "BUSD", "DAI", "USDE", "USDS", "USD1",
    "PYUSD", "BFUSD", "FRAX", "RLUSD", "XUSD", "AEUR", "EURI", "EUR", "GBPT",
    "USDY", "SUSD", "USTC", "USDR", "USDX", "KGST",
    # `U` 是靠"反向审计"抓出来的: 名字看不出是稳定币, 但实测 价=0.9997、
    # 1h 均|涨跌|=0.000063 —— 与 USDC(0.000037)/USDE(0.000055) 同级,
    # 数量级上比真币低两个数量级。见 `--audit` 的输出。
    "U",
}
# 实物锚定(黄金/白银/商品)
EXCL_PHYSICAL = {"PAXG", "XAUT", "KAU", "KAG", "GOLD", "SILVER", "XAU", "XAG"}

# 以 "B" 结尾但确实是加密货币的白名单(见模块 docstring 的逐个核对)
CRYPTO_END_B = {"ARB", "BB", "BNB", "CKB", "DGB", "SHIB", "TRB", "YB"}

# 人工覆盖(留出口, 改这里即可翻转单只标的的归属)
KEEP_EXTRA: set = set()
DROP_EXTRA: set = set()


def classify(base: str) -> tuple[bool, str]:
    """返回 (是否保留, 类别标签)。"""
    if base in KEEP_EXTRA:
        return True, "数字代币(人工保留)"
    if base in DROP_EXTRA:
        return False, "人工排除"
    if base in EXCL_STABLE:
        return False, "稳定币/法币锚定"
    if base in EXCL_PHYSICAL:
        return False, "实物锚定"
    if base.endswith("B") and base not in CRYPTO_END_B:
        return False, "股票代币(bStocks)"
    return True, "数字代币"


def _kl_from_json(rows: list) -> D.Kl:
    from array import array
    kl = D.Kl()
    ts, o, h, l, c, v, qv = [], [], [], [], [], [], []
    for r in rows:
        ts.append(int(r[0])); o.append(float(r[1])); h.append(float(r[2]))
        l.append(float(r[3])); c.append(float(r[4]))
        v.append(float(r[5])); qv.append(float(r[6]))
    kl.ts, kl.o, kl.h, kl.l, kl.c, kl.v, kl.qv = (
        array("q", ts), array("d", o), array("d", h), array("d", l),
        array("d", c), array("d", v), array("d", qv))
    kl.n = len(ts)
    kl._dl = None
    return kl


def scan_one(sym: str, tf: str, kl_dir: str, scanner_kw: dict,
             bars: int, legs: int, asof_ms: int = 0, state: bool = False) -> dict:
    """扫一个 (标的, 周期)。

    逐根 `advance` 并累积 B1/B2/B3 命中(带触发笔下标 leg_i 与 bar 时间戳),
    最后按「最近 bars 根 K 线」∩「最近 legs 根已冻结笔」筛窗口。

    `state=True` 时额外算「**当前**所处买点」(见 `_current_state`), 此时窗口
    参数应传 0(不限) —— 只回溯 120 根的话, "最近一个仍有效的买点"可能被截掉。

    ⚠ **只用真正已收盘的根**: `end_at(close_ts, tf)` 的判据是
    `bar.ts + tf <= close_ts`, 所以喂 `close_ts = bar.ts + tf` 就等于宣布
    "这根刚收盘"。但交易所返回的**最后一根是进行中的 K 线**, 直接喂进去
    等于拿半根 K 线当完整结构用(实时快照里的前视)。这里用 **文件的抓取时刻**
    (`mtime`) 当判定基准: 只有 `bar.ts + tf <= mtime` 的根才算收盘。
    """
    res = {"sym": sym, "tf": tf, "err": None, "hits": [], "shits": [],
           "n_all": 0, "n_sell": 0}
    path = os.path.join(kl_dir, f"{sym}_{tf}.json")
    if not os.path.exists(path):
        res["err"] = "无数据"; return res
    try:
        rows = json.load(open(path, encoding="utf-8"))
    except Exception:
        res["err"] = "读取失败"; return res
    if not rows:
        res["err"] = "空数据"; return res
    kl = _kl_from_json(rows)
    mb = min_bars_of(tf)
    if kl.n < mb:
        res["err"] = f"根数不足({kl.n}<{mb})"; return res

    tf_ms = D.TF_MS[tf]
    if asof_ms <= 0:
        try:
            asof_ms = int(os.path.getmtime(path) * 1000)
        except OSError:
            asof_ms = int(kl.ts[-1]) + tf_ms      # 退化: 当成末根已收盘
    lv = Level("op", tf, "bi", scanner_kw=scanner_kw).attach(kl)
    sell = SellScanner(**sell_kw_from(scanner_kw))

    seen: dict = {}                      # 买点 (kind, leg_i) -> (price, bar_ts, bar_i)
    sseen: dict = {}                     # 卖点, 同构
    last_open = None
    n_used = 0                           # 真正喂进去(已收盘)的根数
    nf_sell = -1                         # 卖点侧只在"冻结笔数变化"时才扫
    for i in range(kl.n):
        o = int(kl.ts[i])
        if o + tf_ms > asof_ms:
            break                        # 尚未收盘(进行中K线), 不喂
        last_open = o
        n_used = i + 1
        lv.advance(o + tf_ms)
        for p in lv.scan_points():
            k = (p.kind, int(p.leg_i))
            if k not in seen:
                seen[k] = (float(p.price), o, i)
        # 卖点挂在"**向上**笔刚冻结"这个反向事件上(见 `sell.py` 的镜像对照表)。
        # ⚠ `zss_view()` 是 O(中枢数) 的拼接, 必须在去重判断**之后**才求值,
        #   否则每根 bar 都白构一次 —— 买点侧的 `Level.scan_points()` 就是这么做的。
        nf_now = lv.bib.n_final
        if nf_now != nf_sell:
            nf_sell = nf_now
            for p in sell.scan(lv.bib.bis, lv.zss_view(), lv.macd, nf_now):
                k = (p.kind, int(p.leg_i))
                if k not in sseen:
                    sseen[k] = (float(p.price), o, i)
    if last_open is None:
        res["err"] = "无已收盘K线"; return res

    n_final = lv.bib.n_final
    leg_min = 0 if legs <= 0 else max(0, n_final - legs)
    ts_cut = 0 if bars <= 0 else last_open - bars * tf_ms

    def _pack(src: dict) -> list:
        out = []
        for (kind, leg_i), (price, bar_ts, _bi) in src.items():
            if leg_i >= leg_min and bar_ts >= ts_cut:
                out.append({"kind": kind, "price": price, "leg_i": leg_i,
                            "bar_ts": bar_ts,
                            "bars_ago": int(round((last_open - bar_ts) / tf_ms))})
        out.sort(key=lambda x: (x["bar_ts"], x["kind"]))
        return out

    hits, shits = _pack(seen), _pack(sseen)

    res.update(hits=hits, shits=shits, n_all=len(seen), n_sell=len(sseen),
               close=float(kl.c[-1]),
               last_ts=last_open, last_close_ts=last_open + tf_ms,
               asof_ms=asof_ms, n_bars=kl.n, n_final=n_final,
               n_closed=sum(1 for t in kl.ts if int(t) + tf_ms <= asof_ms))

    if state:
        # 价格基准取 `kl.c[-1]`(**最新一根, 可能是进行中的那根**) —— 与主表
        # 「现价」列同源。结构判定仍只用已收盘的根, 但「距结构位还有多少空间」
        # 是个展示量, 必须是此刻的价, 否则会和同表的「最新价」自相矛盾。
        px = float(kl.c[-1])
        res["cur"] = _current_state(seen, kl, n_used, tf_ms, last_open, px, "buy")
        res["scur"] = _current_state(sseen, kl, n_used, tf_ms, last_open, px, "sell")
    return res


def _current_state(seen: dict, kl, n_used: int, tf_ms: int,
                   last_ts: int, close: float, side: str = "buy") -> dict | None:
    """当前所处**买点**(`side="buy"`)或**卖点**(`side="sell"`)。

    买点 = 该级别上最近一个「结构止损仍未被跌破」的买点:
      B1 / B2 的结构位 = 背驰笔 / 首次回调笔的**低点**;
      B3      的结构位 = 回试段的**低点**(按定义它还必须高于中枢上沿 ZG)。
    买点**确立之后**只要有任意一根已收盘 K 线的最低价 < 结构位, 该买点即作废;
    作废了就向上回退到前一个仍有效的买点 —— 这与持仓管理里"止损被打掉后
    回到上一个有效结构"是同一件事。

    卖点是**严格镜像**: 结构位在高处, 失效判据 = 之后某根已收盘 K 线的
    **最高价 > 结构位**(见图 `sell.py` 的镜像对照表)。

    ⚠ 同级别的买卖点是**可以同时有效**的(实测全市场 29.1% 的 (标的,周期) 两者都成立):
    买点管的是"下方支撑还没破", 卖点管的是"上方压力还没破" —— 价格被夹在两者之间
    来回震荡时, 两个结构位都还活着。这不违反缠论(理论上的互斥说的是同一个走势段的
    位置), 而是"给信号加了有效期之后"的自然结果。

    ⚠ 判据从**触发的那根 K 线之后**开始查(`suf[bi+1]`), 不含触发根本身。
    触发根的极值就是被冻结的那根笔的极值, 把它算成"自己破自己"没有意义。
    ⚠ 用「最低价/最高价」而不是「收盘价」: 实盘止损单挂在结构位, 影线扫到就成交。
        用最保守(最容易失效)的口径, 免得把已经被打掉的信号留在表里。
    """
    n = n_used
    lo_side = (side == "buy")
    if lo_side:
        suf = [float("inf")] * (n + 1)    # suf[j] = min(low[j:]), O(1) 查"之后是否破位"
        for j in range(n - 1, -1, -1):
            v = float(kl.l[j])
            suf[j] = v if v < suf[j + 1] else suf[j + 1]
    else:
        suf = [float("-inf")] * (n + 1)   # suf[j] = max(high[j:])
        for j in range(n - 1, -1, -1):
            v = float(kl.h[j])
            suf[j] = v if v > suf[j + 1] else suf[j + 1]

    alive = []
    for (kind, leg_i), (price, bar_ts, bi) in seen.items():
        if not price or price <= 0 or bi + 1 > n:
            continue
        if lo_side:
            if suf[bi + 1] < price:       # 之后某根破过位 ⇒ 该买点已失效
                continue
        else:
            if suf[bi + 1] > price:       # 之后某根涨破位 ⇒ 该卖点已失效
                continue
        alive.append({"kind": kind, "price": price, "leg_i": leg_i,
                      "bar_ts": bar_ts, "side": side,
                      "bars_ago": int(round((last_ts - bar_ts) / tf_ms))})
    if not alive:
        return None

    newest = max(a["bar_ts"] for a in alive)
    cand = [a for a in alive if a["bar_ts"] == newest]
    # 同根多命中取一个。买点用生产下单口径 `PRIO`(B2 优先);
    # 卖点用展示口径 `KIND_RANK`(S3 优先 —— S3 是趋势确认后的卖点, 最可靠)。
    cand.sort(key=lambda a: (PRIO if lo_side else KIND_RANK)[a["kind"]])
    cur = dict(cand[0])
    cur["n_alive"] = len(alive)
    cur["n_alive_same"] = len(cand)
    # `dist_pct` = 现价距结构位的**安全边际**(两种方向都是 >0 表示还没被击穿)。
    cur["dist_pct"] = ((close / cur["price"] - 1.0) if lo_side
                       else (cur["price"] / close - 1.0)) * 100.0 \
        if (cur["price"] and close) else 0.0
    cur["close"] = close
    return cur


def load_universe(exch_path: str) -> list[tuple[str, str]]:
    """从 exchangeInfo 取 USDT 现货 TRADING 标的, 返回 [(base, symbol)]。"""
    d = json.load(open(exch_path, encoding="utf-8"))
    out = []
    for s in d.get("symbols", []):
        if (s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
                and s.get("isSpotTradingAllowed", True)):
            out.append((s["baseAsset"], s["symbol"]))
    return sorted(set(out))


def fmt_local(ms: int, f: str = "%m-%d %H:%M") -> str:
    return D.fmt(ms, f)


def human_age(ms: int) -> str:
    h = ms / 3_600_000
    if h < 1:
        return f"{ms/60000:.0f}分钟"
    if h < 48:
        return f"{h:.1f}小时"
    return f"{h/24:.1f}天"


def audit_pegged(kept: list[str], kl_dir: str, thr: float = 0.0008) -> list:
    """反向审计: 在**保留池**里找"看起来像稳定币/锚定物"的标的。

    名字规则会漏(实测 `U` 就是这么漏的)。判据 = 1h 均|对数涨跌| < thr。
    参考量级: USDC 0.00004 / USDE 0.00006 / FDUSD 0.00008 / KGST 0.00079;
    加密主流币普遍 0.002~0.012, 差两个数量级。
    """
    import math
    import statistics
    out = []
    for s in kept:
        p = os.path.join(kl_dir, f"{s}_1h.json")
        if not os.path.exists(p):
            continue
        try:
            rows = json.load(open(p, encoding="utf-8"))[-240:]
        except Exception:
            continue
        if len(rows) < 50:
            continue
        v, prev = [], None
        for r in rows:
            c = float(r[4])
            if prev and prev > 0:
                v.append(abs(math.log(c / prev)))
            prev = c
        if not v:
            continue
        m = statistics.mean(v)
        if m < thr:
            out.append((s, float(rows[-1][4]), m))
    return sorted(out, key=lambda x: x[2])


def build_state_report(args, tfs, keep, by_key, excluded, universe, syms,
                       span, NOW, ASOF, errs) -> tuple[str, list]:
    """生成「当前所处买点」报告。

    与历史信号表的关键差别: 每个 (标的, 周期) **只给一个**买点 —— 最近一个
    结构止损未被跌破的买点。所以格子里不会同时出现 B1/B2/B3。

    返回 `(markdown, bysym_rows)`。`bysym_rows` 是**一标的一行**的宽表:
    `标的 → 各周期买点(长周期在前) → 共振 → 现价 → 最新/最老买点`。
    """
    L = []
    A = L.append
    n_valid = {tf: sum(1 for k in keep
                       if (lambda r: r and not r["err"])(
                           by_key.get((k["symbol"], tf))))
               for tf in tfs}

    A("# 币安现货全市场 —— 当前所处买点 / 卖点(B1-B3 & S1-S3)")
    A("")
    A(f"- 生成时间: **{time.strftime('%Y-%m-%d %H:%M')}**(北京时间)")
    A("- 数据源: 币安现货(`data-api.binance.vision`), 每周期 **1000 根** K 线快照, "
      f"目录 `{args.kl_dir}`")
    A(f"- 标的范围: 币安 USDT 现货 **{len(universe)}** 个交易对 ⇒ "
      f"排除股票代币/稳定币/实物锚定 **{len(excluded)}** 个 ⇒ "
      f"实际扫描 **{len(syms)}** 个数字代币")
    A(f"- 级别: {' / '.join(LEVEL_CN[t] for t in tfs)}(**长周期在前**, 共 {len(tfs)} 级)")
    A("- 信号: 缠论买点 **B1 买①背驰 / B2 买②确认 / B3 买③三买** "
      "+ 卖点 **S1 卖①背驰 / S2 卖②确认 / S3 卖③三卖**(每个级别独立扫描; "
      "卖点走 `chanbook/sell.py::SellScanner`, 与买点严格镜像)")
    A("- 口径: `BookCfg.legacy(market='spot', kinds=(B1,B2,B3))` —— 生产默认口径, "
      "未开任何策略开关; 卖点侧用同名参数映射(`sell_kw_from`)保证两侧同参")
    A(f"- 已收盘判定基准: `{fmt_local(ASOF, '%Y-%m-%d %H:%M')}`(K线文件抓取时刻) —— "
      "交易所返回的最后一根是**进行中**的 K 线, 已剔除")
    A("")
    A("> ## 与「历史信号表」的区别(这一节决定了怎么读整张表)")
    A(">")
    A("> - **历史信号表**回答「窗口内**最近**出过哪些信号」—— 格内也是**每方向只标一个**"
      "(该方向窗口内最近触发的那一个), 所以格内最多 2 个字母; 窗口内出过的**类型集合**"
      "在那张表的附录与逐信号 CSV 里。")
    A("> - **本表**回答「**此刻处于**哪个信号」—— 每个 (标的, 周期, 方向) 只输出"
      "**一个**信号, 但同一格**买卖点可以并存**(见下)。")
    A(">")
    A("> **当前买点 = 该级别上最近一个「结构止损仍未被跌破」的买点。** 判据:")
    A(">")
    A("> 1. 结构位(`Point.price`): B1 = 背驰笔低点; B2 = 首次回调笔低点; "
      "B3 = 回试段低点(按定义它还必须高于中枢上沿 ZG)。")
    A("> 2. **失效**: 买点确立之后, 只要有任意一根**已收盘** K 线的**最低价 < 结构位**, "
      "该买点即作废 —— 与实盘挂在结构位上的止损单被打掉是同一件事。")
    A("> 3. **回退**: 最新的那个作废了, 就向上回退到前一个仍有效的买点; "
      "全部都作废 ⇒ 记 `·`(当前不在任何买点里)。")
    A("> 4. 同一根 K 线同时命中多个买点时, 按 `strategy.PRIORITY = (B2, B3, B1)` "
      "只取一个 —— 与生产下单口径一致。这种「同刻多命中」**只会是 `B2+B3` 同现**"
      "(二买与三买落在同一根回调笔上), 卖点侧同理只会有 `S2+S3`; "
      "三个类型同刻命中从未出现。")
    A(">")
    A("> **卖点是严格镜像**: 结构位在高处, 失效判据 = 之后某根已收盘 K 线的"
      "**最高价 > 结构位**; 同根多命中时按展示优先级取一个(`S3 > S2 > S1`)。")
    A(">")
    A("> ⚠ **同一方向恒为一个字母。** 本表每格最多两个字母, 而且那两个必然是"
      "「1 个买点 + 1 个卖点」—— 每个 (标的, 周期, 方向) 只输出一个信号。"
      "**同方向更早的信号并没有消失**, 它们只是不占格子: 条数记在第三章的 «同级存活» "
      "列(`2` = 这个方向还叠着 1 条更早的、尚未被击穿的结构)。")
    A(">")
    A("> ⚠ **同级别的买卖点可以同时有效**(全市场实测 **29.1%** 的 (标的,周期) 两者"
      "都成立): 买点管「下方支撑还没破」, 卖点管「上方压力还没破」, 价格夹在两者之间"
      "震荡时两个结构位都还活着。这不违反缠论 —— 理论上的互斥说的是同一走势段的"
      "位置, 而这里给每个信号加了**有效期**。格内渲染成 `B2 S1`(买点在前)。")
    A(">")
    A("> ⚠ 判据从**触发根之后**开始查, 不含触发根自身(那根的极值就是该笔的极值)。")
    A("> ⚠ 用「最低价 / 最高价」而非「收盘价」判定失效: 影线扫到即成交, 这是最保守的"
      "口径 —— 宁可把已经被打掉的信号删掉, 也不留在表里。")
    A("")
    A("> ## 历史深度 = 本表的硬约束")
    A(">")
    A("> " + " / ".join(
        f"**{LEVEL_CN[t]}**≈{human_age(span[t])}" for t in tfs if span[t]))
    A(">")
    A("> 数据源单次只给 1000 根。**「当前信号」只能追溯到这段历史里** —— 15分钟 只看得到 "
      "10 天, 日线看得到 2.7 年; 而月线/周线/3日线是**交易所历史长度的物理下限**"
      "(币安 2017-08 上线), 分别只有 111 / 477 / 1000 根, 不随日期增长。"
      "同一条判据在不同级别上的可回溯长度差两个数量级, "
      "**跨级别比较「距今」是没有意义的**。")
    A("")

    # ---- 每个 (标的, 周期) 的当前买点 / 卖点 ----
    per = defaultdict(lambda: {"nbuy": 0, "nsell": 0, "newest": 0,
                               "kind": {t: None for t in tfs},
                               "skind": {t: None for t in tfs},
                               "meta": {t: None for t in tfs},
                               "smeta": {t: None for t in tfs}})
    n_cur, n_scur = Counter(), Counter()
    for tf in tfs:
        for k in keep:
            r = by_key.get((k["symbol"], tf))
            if not r or r["err"] or not (r.get("cur") or r.get("scur")):
                continue
            e = per[k["symbol"]]
            if r.get("cur"):
                cur = r["cur"]
                e["nbuy"] += 1
                e["kind"][tf] = cur["kind"]
                e["meta"][tf] = cur
                e["newest"] = max(e["newest"], cur["bar_ts"])
                n_cur[tf] += 1
            if r.get("scur"):
                sc = r["scur"]
                e["nsell"] += 1
                e["skind"][tf] = sc["kind"]
                e["smeta"][tf] = sc
                # ⚠ `e["newest"]` 必须**买卖合并**取最大 —— 它同时喂给 `_rank` 的
                # 第 4 个判据(「最近确认的在前」)。曾经漏了这一行(只统计买点), 于是
                # 排序键的口径不同于渲染列(`tss` 是 meta+smeta 合并)与
                # `tests/check_scan_state.py` 的独立重算(遍历 cur + scur 两侧),
                # 导致「最新信号其实是卖点」的标的被排错位置 —— 2026-10-04 的
                # E 项自检报了 2 处「排序键不单调」正是这么来的。
                # 历史表侧 `per_sym` 一直是合并的(见 `lv_new` 的写点), 两边要一致。
                e["newest"] = max(e["newest"], sc["bar_ts"])
                n_scur[tf] += 1

    def cell(e, tf) -> str:
        """格内容。同一格**可能买卖点都有**(实测 29.1% 的 (标的,周期) 两者同时有效):
        买点写在前、卖点写在后, 如 `B2 S1`; 两者都没有写 `·`。"""
        ks = [x for x in (e["kind"][tf], e["skind"][tf]) if x]
        return " ".join(ks) if ks else "·"

    # ---- 一、总览 ----
    A("## 一、总览")
    A("")
    A("| 级别 | 周期 | 有数据 | **有当前买点** | 其中 B1/B2/B3 | "
      "**有当前卖点** | 其中 S1/S2/S3 | 买卖同时有效 |")
    A("|---|---|---:|---:|---|---:|---|---:|")
    gb, gs = Counter(), Counter()
    for tf in tfs:
        cb, cs = Counter(), Counter()
        nboth = 0
        for k in keep:
            r = by_key.get((k["symbol"], tf))
            if not r or r["err"]:
                continue
            if r.get("cur"):
                cb[r["cur"]["kind"]] += 1
            if r.get("scur"):
                cs[r["scur"]["kind"]] += 1
            if r.get("cur") and r.get("scur"):
                nboth += 1
        gb.update(cb); gs.update(cs)
        nv = n_valid[tf]
        A(f"| **{LEVEL_CN[tf]}** | {tf} | {nv} | **{sum(cb.values())}** | "
          f"{cb[B1]}/{cb[B2]}/{cb[B3]} | **{sum(cs.values())}** | "
          f"{cs[S1]}/{cs[S2]}/{cs[S3]} | {nboth} |")
    A(f"| **合计** | — | — | **{sum(gb.values())}** | "
      f"{gb[B1]}/{gb[B2]}/{gb[B3]} | **{sum(gs.values())}** | "
      f"{gs[S1]}/{gs[S2]}/{gs[S3]} | — |")
    A("")

    # ---- 二、主表 ----
    NT = len(tfs)

    def _rank(kv):
        """① 买点共振数多的在前 → ② 卖点共振数少的在前 → ③ `period_key`
        (按周期从长到短逐格比类型优先级) → ④ 最近确认的在前 → ⑤ 名称。

        ⚠ 喂给 `period_key` 的必须是**该格全部信号**(买卖都算) —— 只喂买点会让
        「仅卖点」的格子退化成 `MISS`, 于是「月线S1 vs 15分钟S1」这种本该由周期
        分出胜负的比较会双双落空、退到后面的判据上去。`MISS=9 > S1=5`, 所以
        加上卖点后"有信号的长周期格"能正确地压过"没信号的长周期格"。

        ⚠ 第 ④ 项 `e["newest"]` 同样是**买卖合并**的(在 per 构建时两侧都取 max)。
        只统计买点会让「最新信号是卖点」的标的被排错位 —— 见 per 构建处的注释。
        """
        sym, e = kv
        kinds = {t: [x for x in (e["kind"][t], e["skind"][t]) if x] for t in tfs}
        return (-e["nbuy"], e["nsell"], period_key(kinds, tfs),
                -e["newest"], sym)

    ranked = sorted(per.items(), key=_rank)
    A("## 二、当前买点/卖点总表(按买点共振级别数降序)")
    A("")
    A("**读法**: 一行 = 一个标的; 标的名**紧跟各周期信号**(一列一个周期, "
      "**长周期在前**); 格内 = 该周期**当前所处**的买点/卖点"
      "(`·` = 两者都不在)。**共振/现价/年龄是信号后面的汇总列**, "
      "不夹在标的名与信号之间。")
    A("")
    A("**排序**:")
    A("")
    A("1. **买点共振数多的在前**(几个周期同时处于买点);")
    A("2. 买点共振数相同时, **卖点共振数少的在前**(全是卖点的往后排);")
    A("3. 再相同时, **按周期从长到短逐格比类型** —— 这就是"
      "「**长时间周期的 B2 > 短时间周期的 B1**」: 逐格从左向右比, 一旦分出大小就结束"
      "(月线那格先于周线那格被比较, 所以月线 B3 也赢周线 B1);")
    A("4. 最近确认的在前; 5. 名称(保证确定性)。")
    A("")
    A("**类型优先级** = `B1 > B2 > B3 > S3 > S2 > S1` —— 买点全部排在卖点之前; "
      "买点内部 B1 优先(趋势末端的第一个买点, 赔率最高); 卖点内部 **S3 优先**"
      "(趋势确认后的卖点最可靠; S1 是逆势逃顶, 风险最高)。")
    A("")
    A("> ⚠ **同一格可能同时有买点和卖点**, 渲染成 `B2 S1` 这样的两个字母 —— "
      "这不是错。买点管的是「下方支撑还没破」, 卖点管的是「上方压力还没破」, "
      "价格被夹在两者之间来回震荡时, 两个结构位都还活着。全市场实测 **29.1%** 的 "
      "(标的, 周期) 组合两者同时有效 —— 是常态, 不是边界情况。"
      "排序时该格按优先级最高的那个(买点)参与比较。")
    A("")
    A("> ⚠ **「周期先于类型」**: 任意长周期的任意类型都排在短周期的任意类型之前 —— "
      "`月线B3 > 周线B1`。上一版口径正好相反(类型优先, `15分钟B1` 压过 `日线B2`), "
      "已于 2026-10-02 应用户要求改回周期优先。")
    A("")
    A("**共振** = 该标的在几个周期上**当前**处于买点 / 卖点(分母 = 周期总数)。")
    A("")
    A("| # | 标的 | " + " | ".join(LEVEL_CN[t] for t in tfs)
      + f" | **买点共振** | **卖点共振** | 现价({tfs[-1]}) | 最新确认 | 最老确认 |")
    A("|---:|---:|" + "---:|" * len(tfs) + "---:|---:|---:|---|---|")
    # 一标的一行的宽表: 标的 → 各周期信号(长→短) → 双共振 → 现价 → 最新/最老
    bysym = [["rank", "symbol"] + [LEVEL_CN[t] for t in tfs]
             + ["buy_resonance", "sell_resonance", "last_close",
                "newest_sig_time", "oldest_sig_time", "cells_long_to_short"]]
    for i, (sym, e) in enumerate(ranked, 1):
        cells = " | ".join(cell(e, t) for t in tfs)
        px = "—"
        for t in reversed(tfs):
            rr = by_key.get((sym, t))
            if rr and not rr["err"] and rr.get("close"):
                px = f"{rr['close']:.6g}"
                break
        # 年龄按**墙钟**算: 信号 K 线收盘时刻 = bar_ts + tf。
        # 两个数字必须同时给出, 否则会误读 —— 见下方说明。
        tss = [m["bar_ts"] + D.TF_MS[t] for t in tfs
               for m in (e["meta"][t], e["smeta"][t]) if m]
        A(f"| {i} | `{sym}` | {cells} | **{e['nbuy']}/{NT}** | "
          f"**{e['nsell']}/{NT}** | {px} | {human_age(NOW - max(tss))} | "
          f"{human_age(NOW - min(tss))} |")
        # 「信号串」: 把格子按 长周期→短周期 串成一行, 给 Excel / 纯文本阅读用
        bysym.append(
            [i, sym] + [cell(e, t) for t in tfs]
            + [f"{e['nbuy']}/{NT}", f"{e['nsell']}/{NT}", px,
               fmt_local(max(tss)), fmt_local(min(tss)),
               " ".join(f"{cell(e, t)}({LEVEL_CN[t]})"
                        for t in tfs if cell(e, t) != "·") or "·"])
    A("")
    A(f"（共 **{len(ranked)}** 个标的至少有 1 个周期处于买点或卖点; "
      f"`买点共振 5/{NT}` = {NT} 个周期里有 5 个处于买点。）")
    A("")
    A(f"> **机器可读产物**: `{args.out}_bysym.csv` —— **一行一个标的**, 列 = "
      f"各周期信号(**长周期在前**) + 买点/卖点共振 + 现价 + 最新/最老确认, "
      f"排序与上表完全一致; "
      f"`{args.out}.csv` —— 一行一个 (标的, 周期, 方向) 的当前信号"
      f"(含**结构位 / 距结构位 / 距今 / 同级存活数**), 供挂单与设止损直接取数。")
    A("")
    A("> **为什么「最新确认」和「最老确认」必须并排看**: 共振数会把「刚出现的 "
      "15分钟信号」和「半年前立下、至今没被击穿的月线信号」算成同样一票。"
      "两个数字差得越远, 说明这一串共振越依赖**旧结构托底**, 越不是"
      "「当下多级别同时给信号」—— 只看共振数会系统性高估新鲜度。")
    A("")
    A("> 补充: **「最老确认」是「仍在有效、但最陈旧」的那个信号的年龄**, "
      "不是「这个标的历史上最早出现过的信号」。它之所以还挂着, 唯一原因是结构还没被证伪。")
    A("")

    tier = Counter(e["nbuy"] for _, e in ranked)
    stier = Counter(e["nsell"] for _, e in ranked)
    A("### 共振档位分布")
    A("")
    _td = {NT: "**全周期共振** —— 大级别方向 + 小级别时机同时成立",
           NT - 1: f"{NT - 1} 周期共振", 3: "三周期共振", 2: "两周期共振",
           1: "仅单周期有信号(无共振)"}
    A("| 买点共振 | 标的数 | 卖点共振 | 标的数 | 含义(按买点侧读) |")
    A("|---:|---:|---:|---:|---|")
    for k in sorted(set(tier) | set(stier), reverse=True):
        A(f"| **{k}/{NT}** | {tier.get(k, 0)} | **{k}/{NT}** | {stier.get(k, 0)} | "
          f"{_td.get(k, '')} |")
    A("")
    A("> 买点共振 = 该标的在几个周期上当前处于**买点**; 卖点共振同理。"
      "两列**可以同时很高** —— 同一格可能买卖点都有(实测 29.1%)。")
    A("")

    # 「回退」占比: 当前信号不是该级别**最新**那次命中的比例。
    # 这个数字决定整张表的读法, 必须在报告里写明, 不能只让用户看日期自己发现。
    n_cur_total = n_fallback = 0
    n_s_total = n_s_fallback = 0
    n_neg = 0                            # 安全边际为负 = 现价已越过结构位, 但还没收盘确认
    n_neg_worst = 0.0
    # 「同级存活」统计: 同一 (标的,周期,**方向**) 上此刻仍有效的信号条数。
    # 用户会质疑「按缠论一个周期不可能同时有两个信号」—— 这里必须给出真实分布:
    # B2 成立的前提就是「回调不破 B1 低点」, 即 **B1 的结构位天然还没被击穿**,
    # 所以台阶式叠着 2 条以上是**结构的必然**, 不是重复计数。表里只渲染最近那一条。
    n_alive_sum = n_alive_multi = 0
    n_alive_max = 0
    for tf in tfs:
        for k in keep:
            r = by_key.get((k["symbol"], tf))
            if not r or r["err"]:
                continue
            if r.get("cur"):
                n_cur_total += 1
                if r["cur"]["bar_ts"] != max(h["bar_ts"] for h in r["hits"]):
                    n_fallback += 1
            if r.get("scur") and r["shits"]:
                n_s_total += 1
                if r["scur"]["bar_ts"] != max(h["bar_ts"] for h in r["shits"]):
                    n_s_fallback += 1
            for key in ("cur", "scur"):
                c = r.get(key)
                if c and c["dist_pct"] < 0:
                    n_neg += 1
                    n_neg_worst = min(n_neg_worst, c["dist_pct"])
                if c:
                    na = int(c.get("n_alive") or 1)
                    n_alive_sum += na
                    n_alive_multi += 1 if na >= 2 else 0
                    n_alive_max = max(n_alive_max, na)
    n_alive_n = n_cur_total + n_s_total
    n_alive_avg = (n_alive_sum / n_alive_n) if n_alive_n else 0.0

    # ---- 三、明细分级别 ----
    A("## 三、当前信号明细(长周期在前)")
    A("")
    A("> 主表回答「**哪些标的在共振**」; 本节给出每个信号的**结构位 / 距今 / "
      "距结构位的空间**, 供直接下单与挂止损参考。**买点、卖点各一张表**。")
    A(">")
    A("> - **结构位** = 该信号的失效基准(买点跌破 / 卖点涨破即失效)。")
    A("> - **距结构位** = 现价离结构位还有多少 % 的**安全边际** —— 数值越小越贴近失效线。"
      "买点侧 = 现价高于结构位; 卖点侧 = 现价低于结构位。")
    A("> - ⚠ **负的「距结构位」是合法的, 别当成算错**: 现价取最后一根 K 线(可能是"
      f"**进行中**的), 而失效判据只吃**已收盘**的根 ⇒ 会出现 {n_neg} "
      f"行现价已经越过结构位、但那一根还没收盘所以尚未判定失效(最深 {n_neg_worst:.2f}%)。"
      "在粗周期上尤其明显: 一根周线要等 7 天才收盘, 期间价格可以远远甩开压力位。"
      "这类要按「已经到线上/线上方」读, **不是「还有空间」**。")
    A("> - **同级存活** = 该级别、该方向上**同时**仍有效的信号个数, `1` 表示只有当前这一个。"
      "数值大说明还叠着若干更早的、尚未被证伪的结构。")
    A("> - ⚠ **同一方向同时存在多个信号是结构的必然, 不是重复计数。** 全市场实测: "
      f"平均 {n_alive_avg:.2f} 条, 叠着 ≥2 条的占 "
      f"{(n_alive_multi / n_alive_n * 100) if n_alive_n else 0:.1f}%(最多 {n_alive_max} 条)。"
      "原因是台阶式的: **B2 成立的前提就是「第一次回调不破 B1 低点」⇒ B1 的结构位"
      "天然还活着**; B3 又要求回试不跌回 ZG, 若它同时高于 B1/B2 的低点, 三个就一起挂着。"
      "主表格子只渲染**最近**那一条, 其余条数在这里。")
    A("")
    for tf in tfs:
        for side, skey in (("买点", "cur"), ("卖点", "scur")):
            rows = []
            for k in keep:
                r = by_key.get((k["symbol"], tf))
                if not r or r["err"] or not r.get(skey):
                    continue
                cur = r[skey]
                rows.append({"sym": k["symbol"], "kind": cur["kind"],
                             "price": cur["price"], "close": cur["close"],
                             "dist": cur["dist_pct"], "ts": cur["bar_ts"],
                             "age": NOW - (cur["bar_ts"] + D.TF_MS[tf]),
                             "n_alive": cur["n_alive"]})
            A(f"### {LEVEL_CN[tf]}({tf}) —— {len(rows)} 个标的当前处于{side}")
            A("")
            if not rows:
                A(f"_该级别当前没有任何有效{side}。_")
                A("")
                continue
            rows.sort(key=lambda x: (x["age"], -x["dist"], x["sym"]))
            A(f"| # | 标的 | 当前{side} | 结构位 | 现价 | 距结构位 | 确立时间 | "
              f"距今 | 同级存活 |")
            A("|---:|---|---|---:|---:|---:|---|---:|---:|")
            lab = BUY_LABEL if side == "买点" else SELL_LABEL
            for i, x in enumerate(rows, 1):
                A(f"| {i} | `{x['sym']}` | {x['kind']} {lab[x['kind']]} | "
                  f"{x['price']:.6g} | {x['close']:.6g} | {x['dist']:+.2f}% | "
                  f"{fmt_local(x['ts'])} | {human_age(x['age'])} | {x['n_alive']} |")
            A("")

    # ---- 四、排除清单 ----
    A("## 四、已排除标的(共 %d 个)" % len(excluded))
    A("")
    grp = defaultdict(list)
    for e in excluded:
        grp[e["why"]].append(e["base"])
    for why in ["股票代币(bStocks)", "稳定币/法币锚定", "实物锚定", "人工排除"]:
        if why not in grp:
            continue
        lst = sorted(grp[why])
        A(f"**{why}**({len(lst)} 个)")
        A("")
        A("`" + "` `".join(lst) + "`")
        A("")
    A("- **股票代币**: 币安 bStocks 命名规则 = 股票代码 + `B`(NVDAB=英伟达 / "
      "TSLAB=特斯拉 / AAPLB=苹果 / QQQB=纳指100ETF / MUB=美光 …), 由 BTECH "
      "Holdings(ADGM 持牌)1:1 托管真实股票。规则 = 以 `B` 结尾即排除, 白名单 "
      "`ARB/BB/BNB/CKB/DGB/SHIB/TRB/YB` 这 8 个真实加密货币除外。")
    A("- **稳定币**: 含 `KGST`(吉尔吉斯斯坦国家稳定币, 1:1 锚定索姆 KGS, "
      "2025-12-24 上线币安); 含反向审计抓出的 `U`(价 0.9997、1h 均|涨跌| 0.000063)。")
    A("- **实物锚定**: `PAXG`/`XAUT` 对标黄金。")
    A("")
    A("若某只标的的归属要翻, 直接改 `tools/scan_spot_b1b2b3.py` 顶部的 "
      "`KEEP_EXTRA` / `DROP_EXTRA` 集合。")
    A("")

    # ---- 五、原因分布 ----
    if errs:
        A("## 五、未纳入统计的原因分布")
        A("")
        A("| 原因 | 组合数 |")
        A("|---|---:|")
        for k, v in errs.most_common():
            A(f"| {k} | {v} |")
        A("")
        A("> `根数不足` = 该标的在该周期的可得 K 线少于**该周期的门槛**"
          "(月线 48 / 周线 100 / 3日线 200 / 其余 500 根)。粗周期受交易所历史长度的"
          "物理限制 —— 币安 2017-08 上线, 月线至今总共才 111 根, 沿用 500 的门槛"
          "会让月线整列全灭。门槛表见脚本顶部 `MIN_BARS_BY_TF`。")
        A("")

    # ---- 六、读数注意 ----
    A("## 六、读数注意")
    A("")
    A("1. **买卖点是事件, 不是状态。** 本表的「当前信号」是给事件加了一条**有效期**"
      "(结构位未被击穿)之后的状态化读法。它回答的是「这个标的自最近一次有效信号"
      "以来, 结构还没被证伪」, **不等于**「现在是最佳入场时刻」。")
    A(f"2. **买点侧 {n_fallback} / {n_cur_total} 个是「回退」得到的"
      f"({(n_fallback/n_cur_total*100 if n_cur_total else 0):.0f}%), "
      f"卖点侧 {n_s_fallback} / {n_s_total} "
      f"({(n_s_fallback/n_s_total*100 if n_s_total else 0):.0f}%)** —— "
      "即该级别最近的那次信号**已经被击穿**, 现在挂的是更早一个仍未失效的。"
      "这部分要按「旧结构托底」读, 不能当成新信号; 具体年龄见第三章的 `距今` 一列。")
    A("3. **`·` 是有信息的。** 它意味着该级别的**全部**历史信号都已被击穿 —— "
      "处于「无支撑 / 无压力」状态, 而不是「没扫到」。")
    A("4. **同一格可以同时有买卖点**(实测 29.1%)。支撑没破 + 压力没破 = 价格在区间里"
      "震荡。**卖点在这里的含义是「上方有压力」, 不是「立刻做空」** —— 缠论卖点"
      "原本的用途是做多持仓的**离场判据**(见 `sell.py`)与做空侧的入场信号。"
      "⚠ **同方向不会有第二个字母**: 格内出现两个字母时, 必然是「1 买 + 1 卖」。")
    A("4b. **同方向「同时有效」的多个信号是结构的必然, 不是重复计数。** 实测有 "
      f"{(n_alive_multi / n_alive_n * 100) if n_alive_n else 0:.0f}% 的 (标的,周期,方向) "
      f"此刻挂着 ≥2 条(平均 {n_alive_avg:.2f} 条)。因为 **B2 成立的前提就是「第一次回调"
      "不破 B1 低点」⇒ B1 的结构位天然还没被击穿**, 台阶式叠着; B3 再叠一层。"
      "主表只写最近的那一条, 条数在第三章 «同级存活»。")
    A("5. **不同级别的「距今」不可比。** 月线能回溯 9 年, 15分钟只能回溯 10 天。"
      "15分钟 的「距今 3 天」已接近该级别的历史极限, 而月线的「距今 3 天」是刚刚发生。")
    A("6. **共振数 ≠ 盈亏概率。** 共振只说明结构在更多尺度上被确认。"
      "实际盈亏由离场口径、费率、资金管理决定 —— 见 `bt_results/` 的多周期回测结论: "
      "**单笔毛期望 ≤ 双边手续费的组合长期必死**, 与胜率无关。")
    A("7. **不含仓位、不含执行成本。** 本表只是买/卖点的结构扫描, 不构成交易建议。")
    A("")
    return "\n".join(L) + "\n", bysym


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kl_dir", default="out/scan/klines_20261002")
    ap.add_argument("--exch", default="out/scan/exchangeInfo.json")
    ap.add_argument("--out", default="out/scan/b123_spot")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--tfs", default=",".join(LEVELS))
    ap.add_argument("--bars", type=int, default=120,
                    help="窗口: 最近 N 根 K 线(0=不限)")
    ap.add_argument("--legs", type=int, default=10,
                    help="窗口: 最近 N 根已冻结笔(0=不限)")
    ap.add_argument("--syms", default="", help="只扫这些标的(调试)")
    ap.add_argument("--asof", type=int, default=0,
                    help="已收盘判定基准(ms epoch); 0=用 K 线文件的抓取时刻(mtime)")
    ap.add_argument("--audit", action="store_true",
                    help="只做资产分类反向审计(在保留池里找疑似稳定币), 不扫描")
    ap.add_argument("--state", action="store_true",
                    help="只输出「**当前**所处买点」: 每个 (标的,周期) 只给一个买点 "
                         "= 最近一个结构止损未被跌破的买点(而不是窗口内出现过的类型集合)。"
                         "开启后自动不限窗口(--bars/--legs 置 0)")
    args = ap.parse_args()
    if args.state:
        # 必须全历史回溯: "最近一个仍有效的买点"可能发生在窗口之外, 截断会误判成 ·。
        args.bars, args.legs = 0, 0

    tfs = [x.strip() for x in args.tfs.split(",") if x.strip()]
    for t in tfs:
        assert t in D.TF_MS, f"未知周期 {t}"

    universe = load_universe(args.exch)
    keep, excluded = [], []
    for base, sym in universe:
        ok, why = classify(base)
        (keep if ok else excluded).append({"base": base, "symbol": sym, "why": why})
    if args.syms:
        want = {s.strip() for s in args.syms.split(",") if s.strip()}
        keep = [k for k in keep if k["symbol"] in want]

    syms = [k["symbol"] for k in keep]

    if args.audit:
        print(f"反向审计: 在 {len(syms)} 个保留标的中找 1h 均|涨跌| < 0.0008 的"
              f"(疑似稳定币/锚定物)")
        for s, price, v in audit_pegged(syms, args.kl_dir):
            print(f"  ⚠ {s:<14} 价={price:<12.6g} 1h均|涨跌|={v:.6f}")
        print("(以上应全部进 EXCL_STABLE —— 判据与 USDC/USDE/FDUSD/KGST 同级)")
        return 0

    tasks = [(s, tf) for s in syms for tf in tfs]
    print(f"标的 {len(syms)}(全部 {len(universe)}, 排除 {len(excluded)}) "
          f"× 周期 {len(tfs)} = {len(tasks)} 组合", flush=True)

    scanner_kw = BookCfg.legacy(market="spot", kinds=(B1, B2, B3)).scanner_kw()
    scanner_kw = dict(scanner_kw)

    results = []
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(scan_one, s, tf, args.kl_dir, scanner_kw,
                              args.bars, args.legs, args.asof, args.state)
                    for s, tf in tasks]
            for i, f in enumerate(as_completed(futs), 1):
                results.append(f.result())
                if i % 300 == 0:
                    print(f"  {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    else:
        for s, tf in tasks:
            results.append(scan_one(s, tf, args.kl_dir, scanner_kw,
                                    args.bars, args.legs, args.asof, args.state))

    print(f"扫描完成 {len(results)} 组合, 用时 {time.time()-t0:.0f}s", flush=True)

    # ---------------- 汇总 ----------------
    by_key = {(r["sym"], r["tf"]): r for r in results}
    errs = Counter()
    for r in results:
        if r["err"]:
            errs[r["err"].split("(")[0]] += 1

    # 长格式 CSV
    if args.state:
        # 一个 (标的, 周期, 方向) 一行 = 当前所处买点/卖点
        csv_rows = [["level", "symbol", "side", "kind", "label", "struct_price",
                     "close", "dist_pct", "trigger_ts", "trigger_time",
                     "bars_ago", "leg_i", "n_alive"]]
        for tf in tfs:
            for k in keep:
                r = by_key.get((k["symbol"], tf))
                if not r or r["err"]:
                    continue
                for side, key, lab in (("buy", "cur", BUY_LABEL),
                                       ("sell", "scur", SELL_LABEL)):
                    c = r.get(key)
                    if not c:
                        continue
                    csv_rows.append([
                        tf, k["symbol"], side, c["kind"], lab[c["kind"]],
                        f"{c['price']:.8g}", f"{c['close']:.8g}",
                        f"{c['dist_pct']:.2f}",
                        c["bar_ts"], fmt_local(c["bar_ts"], "%Y-%m-%d %H:%M"),
                        c["bars_ago"], c["leg_i"], c["n_alive"],
                    ])
    else:
        # 历史信号表: 每个信号一行(买点、卖点都列)
        csv_rows = [["level", "symbol", "side", "kind", "label", "sig_price",
                     "close", "trigger_ts", "trigger_time", "bars_ago",
                     "leg_i", "dist_pct"]]
        for tf in tfs:
            for k in keep:
                r = by_key.get((k["symbol"], tf))
                if not r or r["err"]:
                    continue
                cl = r["close"]
                for side, key, lab in (("buy", "hits", BUY_LABEL),
                                       ("sell", "shits", SELL_LABEL)):
                    for h in (r.get(key) or ()):
                        # `dist_pct` = 现价距结构位的**安全边际**(两种方向都 >0 表示未失效)
                        if side == "buy":
                            dist = f"{(cl/h['price']-1)*100:.2f}" if h["price"] else ""
                        else:
                            dist = f"{(h['price']/cl-1)*100:.2f}" if cl else ""
                        csv_rows.append([
                            tf, k["symbol"], side, h["kind"], lab[h["kind"]],
                            f"{h['price']:.8g}", f"{cl:.8g}",
                            h["bar_ts"], fmt_local(h["bar_ts"], "%Y-%m-%d %H:%M"),
                            h["bars_ago"], h["leg_i"], dist,
                        ])
    with open(args.out + ".csv", "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(csv_rows)

    # JSON(全量)
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M:%S"),
               "kl_dir": args.kl_dir, "bars": args.bars, "legs": args.legs,
               "levels": tfs, "kept": syms,
               "excluded": excluded, "results": results},
              open(args.out + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    # ---------------- 报告 ----------------
    L = []
    A = L.append
    # 各级别的实际历史深度: 取该级别**已收盘根数**的最大值(不取第一只标的 ——
    # 新上线币只有几十根, 会把跨度算成几天, 得出荒谬的结论)
    span = {}
    for tf in tfs:
        mx = max([(r.get("n_closed") or 0) for r in results
                  if r["tf"] == tf and not r["err"]] or [0])
        span[tf] = mx * D.TF_MS[tf] if mx else None
    NOW = int(time.time() * 1000)
    ASOF = max((r.get("asof_ms") or 0) for r in results) or NOW

    if args.state:
        md, bysym = build_state_report(args, tfs, keep, by_key, excluded,
                                       universe, syms, span, NOW, ASOF, errs)
        with open(args.out + ".md", "w", encoding="utf-8") as fh:
            fh.write(md)
        with open(args.out + "_bysym.csv", "w", encoding="utf-8",
                  newline="") as fh:
            csv.writer(fh).writerows(bysym)
        print(f"报告已写: {args.out}.md / .csv / _bysym.csv(一标的一行) / .json",
              flush=True)
        return 0

    A("# 币安现货全市场 B1/B2/B3 扫描")
    A("")
    A(f"- 生成时间: **{time.strftime('%Y-%m-%d %H:%M')}**(北京时间)")
    A(f"- 数据源: 币安现货(`data-api.binance.vision`), 每周期 **{1000} 根** K 线快照, "
      f"目录 `{args.kl_dir}`")
    A(f"- 标的范围: 币安 USDT 现货 **{len(universe)}** 个交易对 ⇒ "
      f"排除股票代币/稳定币/实物锚定 **{len(excluded)}** 个 ⇒ "
      f"实际扫描 **{len(syms)}** 个数字代币")
    A(f"- 级别: {' / '.join(LEVEL_CN[t] for t in tfs)}(**长周期在前**, 共 {len(tfs)} 级) —— "
      f"第二章主表**一行一个标的、一列一个周期**, 按买点共振级别数降序排列")
    A(f"- 信号: 缠论买点 **B1 买①背驰 / B2 买②确认 / B3 买③三买** + 卖点 "
      f"**S1 卖①背驰 / S2 卖②确认 / S3 卖③三卖**(每个级别独立扫描; "
      f"卖点走 `chanbook/sell.py::SellScanner`, 与买点严格镜像)")
    A(f"- 有效窗口: 最近 **{args.bars}** 根 K 线 且 最近 **{args.legs}** 根已冻结笔 "
      f"(两个条件取交集)")
    A("- 口径: `BookCfg.legacy(market='spot', kinds=(B1,B2,B3))` —— 生产默认口径, "
      f"未开任何策略开关; 卖点侧用同名参数映射(`sell_kw_from`)保证两侧同参")
    A(f"- 已收盘判定基准: `{fmt_local(ASOF, '%Y-%m-%d %H:%M')}`(K线文件抓取时刻) —— "
      f"交易所返回的最后一根是**进行中**的 K 线, 已剔除, 不用半根 K 线当完整结构")
    A("")
    A("> **窗口的级别含义**: 窗口按「根数」而非「小时数」计, 同样的 "
      f"{args.bars} 根在各级别覆盖的日历时间不同 —— 级别不同, 一根的分量不同。")
    A(">")
    A(f"> - **窗口**({args.bars} 根)覆盖: " + " / ".join(
        f"{LEVEL_CN[t]}≈{human_age(args.bars * D.TF_MS[t])}" for t in tfs))
    A("> - **可用历史**(数据源单次上限 1000 根)深度: " + " / ".join(
        f"{LEVEL_CN[t]}≈{human_age(span[t])}" for t in tfs if span[t]))
    A(">")
    A("> ⚠ 两者是不同概念: 窗口决定「哪些信号算数」, 历史深度决定「结构能回溯多远」。"
      "`历史上限内总命中` 一列受后者约束 —— 15分钟 只回溯 10 天, 日线回溯 2.7 年; "
      "月线 / 周线 / 3日线则受**交易所历史长度**的物理限制(币安 2017-08 上线), "
      "分别只有 111 / 477 / 1000 根。**不可跨级别比较**。")
    A("")

    # 总览
    A("## 一、总览")
    A("")
    A("| 级别 | 周期(K线) | 有效标的* | 买1 | 买2 | 买3 | **买点合计** | "
      "卖1 | 卖2 | 卖3 | **卖点合计** | 窗口内最大信号数 |")
    A("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    grand, gs = Counter(), Counter()
    for tf in tfs:
        cnt, cs = Counter(), Counter()
        nsig = 0
        mx = 0
        for k in keep:
            r = by_key.get((k["symbol"], tf))
            if not r or r["err"]:
                continue
            hits, shits = r["hits"], (r.get("shits") or ())
            mx = max(mx, len(hits) + len(shits))
            if not (hits or shits):
                continue
            nsig += 1
            for h in hits:
                cnt[h["kind"]] += 1
                grand[h["kind"]] += 1
            for h in shits:
                cs[h["kind"]] += 1
                gs[h["kind"]] += 1
        A(f"| **{LEVEL_CN[tf]}** | {tf} | {nsig} | {cnt[B1]} | {cnt[B2]} | "
          f"{cnt[B3]} | **{sum(cnt.values())}** | {cs[S1]} | {cs[S2]} | {cs[S3]} | "
          f"**{sum(cs.values())}** | {mx} |")
    A(f"| **合计** | — | — | {grand[B1]} | {grand[B2]} | {grand[B3]} | "
      f"**{sum(grand.values())}** | {gs[S1]} | {gs[S2]} | {gs[S3]} | "
      f"**{sum(gs.values())}** | — |")
    A("")
    A("\\* 有效标的 = 该级别有数据且窗口内有任意买点/卖点信号的币种数。")
    A("")

    # 「同刻多类型」统计: 同一根 K 线、同一方向命中 ≥2 个类型的组合数。
    # 用户会质疑「按缠论一个周期不可能同时有两个信号」—— 必须把真实数量写进报告,
    # 否则看到格内 `B2 B3` 会当成 bug。实测只可能是 B2+B3 / S2+S3(二买三买重合)。
    n_pair = 0
    pair_kinds: Counter = Counter()
    # 各周期**实测**回溯跨度(根): 窗口是「bars 根 ∩ legs 根笔」的交集, 粗周期上
    # 往往是 `legs` 更紧 —— 月线 120 根 = 3720 天的理论值根本用不到, 实测只回 41 个月。
    span_bars: dict = {t: 0 for t in tfs}
    for k in keep:
        for tf in tfs:
            r = by_key.get((k["symbol"], tf))
            if not r or r["err"]:
                continue
            for hkey in ("hits", "shits"):
                grp = defaultdict(set)
                for h in (r.get(hkey) or ()):
                    grp[h["bar_ts"]].add(h["kind"])
                    span_bars[tf] = max(span_bars[tf], int(h["bars_ago"]))
                for _ts, ks in grp.items():
                    if len(ks) >= 2:
                        n_pair += 1
                        pair_kinds[tuple(sorted(ks))] += 1

    # ---- 标的 × 级别 主表 ----
    # 一行 = 一个标的, 列 = 各时间周期(长→短), 格内 = 该周期**最后一个**(最近触发)
    # 的买点与卖点, 每个方向各一个(买点写在前)。
    # (按 `B1>B2>B3>S3>S2>S1` 排序, 买点写在前)。
    per_sym = defaultdict(lambda: {
        "tot": 0, "tot_s": 0, "nbuy": 0, "nsell": 0,
        "lv": {t: [] for t in tfs},        # 各级别命中的买点类型(去重 + 按展示优先级)
        "lv_s": {t: [] for t in tfs},      # 各级别命中的卖点类型
        "lv_n": {t: 0 for t in tfs},       # 各级别的买点信号条数
        "lv_new": {t: 0 for t in tfs},     # 各级别最近信号的 bar_ts
        "newest": 0,
    })
    # 主表格内 = 该方向**最后一个**(最近触发)信号的类型, 不是「窗口内出现过的类型集合」。
    # 用户 2026-10-02 指定: **同方向只标最后一个; 两个方向各标最后一个 ⇒ 格内最多 2 个字母。**
    # 同一根 K 线上多命中时按下单口径取一个(买点 `PRIO={B2,B3,B1}`、卖点 `S3>S2>S1`),
    # 与状态表 `_current_state` 完全同口径 —— 否则「同一根笔上 B2 与 B3 重合」两表会给出
    # 不同的字母。⚠ 类型集合仍完整保留在两处: 附录逐级别的「命中类型」列 + 逐信号 CSV。
    def _last_kind(hs: list, prio: dict) -> str | None:
        if not hs:
            return None
        last = max(h["bar_ts"] for h in hs)
        return min((h["kind"] for h in hs if h["bar_ts"] == last),
                   key=lambda k: prio[k])

    for tf in tfs:
        for k in keep:
            r = by_key.get((k["symbol"], tf))
            if not r or r["err"] or not (r["hits"] or r.get("shits")):
                continue
            e = per_sym[k["symbol"]]
            if r["hits"]:
                e["tot"] += len(r["hits"])
                e["nbuy"] += 1
                e["lv_n"][tf] = len(r["hits"])
                e["lv"][tf] = [_last_kind(r["hits"], PRIO)]
                e["lv_new"][tf] = max(h["bar_ts"] for h in r["hits"])
            if r.get("shits"):
                e["tot_s"] += len(r["shits"])
                e["nsell"] += 1
                e["lv_s"][tf] = [_last_kind(r["shits"], KIND_RANK)]
                e["lv_new"][tf] = max(e["lv_new"][tf],
                                      max(h["bar_ts"] for h in r["shits"]))
            e["newest"] = max(e["newest"], e["lv_new"][tf])

    def cell_of(e, t) -> str:
        """格内容: 买点在前、卖点在后(同一周期可能两者都有)。"""
        ks = list(e["lv"][t]) + list(e["lv_s"][t])
        return " ".join(ks) if ks else "·"

    def _sym_rank(kv):
        """① 买点共振数多的在前 → ② 卖点共振数少的在前 → ③ `period_key`
        (周期从长到短逐格比类型) → ④ 信号数多的在前 → ⑤ 最近信号新的在前 → ⑥ 名称。

        ⚠ `period_key` 喂**买卖合并**后的格内容 —— 理由同状态表 `_rank`。
        """
        sym, e = kv
        kinds = {t: list(e["lv"][t]) + list(e["lv_s"][t]) for t in tfs}
        return (-e["nbuy"], e["nsell"], period_key(kinds, tfs),
                -e["tot"], -e["newest"], sym)

    ranked = sorted(per_sym.items(), key=_sym_rank)

    A("## 二、标的共振总表(按买点共振级别数降序)")
    A("")
    A("**读法**: 一行 = 一个标的; 标的名**紧跟各周期信号**(一列一个周期, "
      "**长周期在前**); 格内 = 该周期**最后一个**(最近触发)的买点 / 卖点 —— "
      "**每个方向只标一个, 所以格内最多 2 个字母**, 买点在前、卖点在后; "
      "`·` = 该周期该方向没有信号。"
      "**共振/信号数是信号后面的汇总列**, 不夹在标的名与信号之间。")
    A("")
    A("**排序**:")
    A("")
    A("1. **买点共振数多的在前**(几个周期出买点);")
    A("2. 买点共振数相同时, **卖点共振数少的在前**;")
    A("3. 再相同时, **按周期从长到短逐格比类型** —— 即"
      "「**长时间周期的 B2 > 短时间周期的 B1**」: 逐格从左向右比, 一旦分出大小就结束"
      "(月线那格先于周线那格被比较); **比的就是格内那两个字母**, 与显示的完全一致;")
    A("4. 信号数多的在前; 5. 最近信号新的在前; 6. 名称(保证确定性)。")
    A("")
    A("**类型优先级** = `B1 > B2 > B3 > S3 > S2 > S1`(格内买点只有一个、卖点只有一个, "
      "所以这个优先级只在**跨周期比较**时起作用)。")
    A("")
    A("> ⚠ 格内**每方向只标最后一个**, 所以**字母数 ≠ 信号数** —— 一个周期可能出过 "
      "`B1 B2 B3` 三个买点, 格内只写最后触发的那个(例如 `B3`), 信号条数在汇总列。"
      "**「最后一个」= 窗口内该方向最近触发的那一个**; 同一根 K 线上多命中时按下单口径"
      "取一个(买点 `B2>B3>B1`, 卖点 `S3>S2>S1`), 与状态表同口径。")
    A(">")
    A("> ⚠ **想看「这个周期出过哪些类型」去附录**(第二章后每个级别的《逐级别明细》"
      "有 `命中类型` 列, 以及 `买1/买2/买3` 分开计数), 或者直接读逐信号 CSV"
      f"(`{os.path.basename(args.out)}.csv`, 一行一个信号)。主表只回答"
      "「最近一次是什么」。")
    A(">")
    A(f"> ⚠ **「同刻两个」是什么情况**: 全市场窗口内 **{n_pair}** 例「同一根 K 线、"
      f"同一方向命中 ≥2 个类型」, 且**全部**是 "
      + " / ".join(f"`{'+'.join(k)}` × {v}" for k, v in sorted(pair_kinds.items()))
      + " —— 这是缠论里 **二买与三买落在同一根回调笔上**的重合情形"
      "(一买之后的第一次回调恰好也没跌回中枢上沿), 不是重复计数, 也不是两个信号。"
      "**三个类型同刻命中从未出现**; 同一类型在同一根笔上更不会重复"
      "(触发事件是「笔冻结」, 按 `(类型, 笔号)` 去重)。这类重合格内只留一个字母。")
    A(">")
    A("> ⚠ **窗口跨度**: 本表实际回溯到的最大跨度(实测, 取该周期里最老的信号): "
      + " / ".join(f"{LEVEL_CN[t]}≈{span_bars[t] * D.TF_MS[t] / 86400000:.0f}天"
                   for t in tfs) + "。")
    A("")
    A("| # | 标的 | " + " | ".join(LEVEL_CN[t] for t in tfs)
      + f" | **买点共振** | **卖点共振** | 信号数(买/卖) | 现价({tfs[-1]}) | 最近信号 |")
    A("|---:|---:|" + "---:|" * len(tfs) + "---:|---:|---:|---:|---|")
    for i, (sym, e) in enumerate(ranked, 1):
        cells = " | ".join(cell_of(e, t) for t in tfs)
        px = "—"
        # 「最新价」取**最细周期**的收盘(级别越细, 最后一根已收盘 K 线离现在越近)
        for t in reversed(tfs):
            rr = by_key.get((sym, t))
            if rr and not rr["err"] and rr.get("close"):
                px = f"{rr['close']:.6g}"
                break
        A(f"| {i} | `{sym}` | {cells} | **{e['nbuy']}/{len(tfs)}** | "
          f"**{e['nsell']}/{len(tfs)}** | {e['tot']}/{e['tot_s']} | "
          f"{px} | {fmt_local(e['newest'])} |")
    A("")
    A(f"（共 **{len(ranked)}** 个标的在窗口内出现信号；"
      f"`买点共振 5/{len(tfs)}` = {len(tfs)} 个周期里 5 个出了买点。）")
    A("")

    # 共振档位分布
    tier = Counter(e["nbuy"] for _, e in ranked)
    stier = Counter(e["nsell"] for _, e in ranked)
    _NT = len(tfs)
    A("### 共振档位分布")
    A("")
    _tier_desc = {
        _NT: "**全周期共振** —— 大级别方向 + 小级别时机同时成立",
        _NT - 1: f"{_NT - 1} 周期共振",
        3: "三周期共振",
        2: "两周期共振",
        1: "仅单周期出信号(无共振)",
    }
    A("| 买点共振 | 标的数 | 卖点共振 | 标的数 | 含义(按买点侧读) |")
    A("|---:|---:|---:|---:|---|")
    for k in sorted(set(tier) | set(stier), reverse=True):
        A(f"| **{k}/{_NT}** | {tier.get(k, 0)} | **{k}/{_NT}** | {stier.get(k, 0)} | "
          f"{_tier_desc.get(k, '')} |")
    A("")
    A("> 共振是**级别间的相互印证**: 大周期出买点说明方向, 小周期出买点说明时机。"
      "共振级别数越多, 该标的的买点在越多的尺度上被确认 —— 但**不等于盈亏概率更高**, "
      "最终仍取决于离场口径与过滤层(见「读数注意」)。")
    A("")

    # 按标的聚合的宽表 CSV(便于直接进 Excel 透视)
    # ⚠ 列序与状态表宽表**必须一致**: 标的名紧跟各周期格(长周期在前), 汇总列在后 ——
    #   用户点名过这个规格, 两张表不一致会让人在 Excel 里对不上列。
    with open(args.out + "_bysym.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "symbol"] + [LEVEL_CN[t] for t in tfs]
                   + ["buy_resonance", "sell_resonance", "signals_buy",
                      "signals_sell", "last_close", "newest_ts", "newest_time"])
        for i, (sym, e) in enumerate(ranked, 1):
            px = ""
            for t in reversed(tfs):
                rr = by_key.get((sym, t))
                if rr and not rr["err"] and rr.get("close"):
                    px = f"{rr['close']:.10g}"
                    break
            w.writerow([i, sym] + [cell_of(e, t) for t in tfs]
                       + [f"{e['nbuy']}/{len(tfs)}", f"{e['nsell']}/{len(tfs)}",
                          e["tot"], e["tot_s"], px, e["newest"],
                          fmt_local(e["newest"], "%Y-%m-%d %H:%M")])

    # 排除清单
    A("## 三、已排除标的(共 %d 个)" % len(excluded))
    A("")
    grp = defaultdict(list)
    for e in excluded:
        grp[e["why"]].append(e["base"])
    for why in ["股票代币(bStocks)", "稳定币/法币锚定", "实物锚定", "人工排除"]:
        if why not in grp:
            continue
        lst = sorted(grp[why])
        A(f"**{why}**({len(lst)} 个)")
        A("")
        A("`" + "` `".join(lst) + "`")
        A("")
    A("判定依据:")
    A("")
    A("- **股票代币**: 币安 bStocks 命名规则 = 股票代码 + `B`(NVDAB=英伟达 / TSLAB=特斯拉 / "
      "AAPLB=苹果 / QQQB=纳指100ETF / MUB=美光 …), 由 BTECH Holdings(ADGM 持牌)托管 "
      "1:1 真实股票。规则=以 `B` 结尾即排除, 白名单 "
      "`ARB/BB/BNB/CKB/DGB/SHIB/TRB/YB` 这 8 个真实加密货币除外。")
    A("- **稳定币**: 含 `KGST`(吉尔吉斯斯坦国家稳定币, 1:1 锚定索姆 KGS, "
      "2025-12-24 上线币安; 实测 1h 波动率仅 0.0004, 与 USDC 同级)。")
    A("- **实物锚定**: `PAXG`/`XAUT` 对标黄金。")
    A("")
    A("若某只标的的归属要翻, 直接改 `tools/scan_spot_b1b2b3.py` 顶部的 "
      "`KEEP_EXTRA` / `DROP_EXTRA` 集合。")
    A("")

    # 未出信号的标的与错误
    if errs:
        A("## 四、未纳入统计的原因分布")
        A("")
        A("| 原因 | 组合数 |")
        A("|---|---:|")
        for k, v in errs.most_common():
            A(f"| {k} | {v} |")
        A("")
        A("> `根数不足` = 该标的在该周期的可得 K 线少于**该周期的门槛**"
          "(月线 48 / 周线 100 / 3日线 200 / 其余 500 根); 门槛表见脚本顶部 "
          "`MIN_BARS_BY_TF`。粗周期受交易所历史长度的物理限制 —— 币安 2017-08 上线, "
          "月线至今总共才 111 根, 沿用 500 的门槛会让月线整列全灭。")
        A("")

    # 分级别明细(附录: 供逐级别排查, 数据与主表同源)
    A("## 五、附录: 各级别信号明细(长周期在前, 买点/卖点分表)")
    A("")
    A("> 主表(第二章)回答「**哪些标的在共振**」; 本附录回答「**某个级别上都有谁**」。")
    A("")
    for tf in tfs:
        for side, key, cols in (("买点", "hits", (B1, B2, B3)),
                                ("卖点", "shits", (S1, S2, S3))):
            rows = []
            for k in keep:
                r = by_key.get((k["symbol"], tf))
                if not r or r["err"]:
                    continue
                hs = r.get(key) or ()
                if not hs:
                    continue
                cnt = Counter(h["kind"] for h in hs)
                newest = max(h["bar_ts"] for h in hs)
                rows.append({
                    "sym": k["symbol"], "c1": cnt[cols[0]], "c2": cnt[cols[1]],
                    "c3": cnt[cols[2]], "tot": len(hs), "newest": newest,
                    "kinds": ",".join(sorted({h["kind"] for h in hs})),
                    "close": r["close"], "n_all": r["n_all"] if side == "买点"
                    else r.get("n_sell", len(hs)),
                    # 「距今」以**墙钟**为准: 信号 K 线收盘时刻 = bar_ts + tf
                    "age": NOW - (newest + D.TF_MS[tf]),
                })
            A(f"### {LEVEL_CN[tf]}({tf}) —— {side}")
            A("")
            if not rows:
                A(f"_该级别在窗口内没有任何 {side} 信号。_")
                A("")
                continue
            rows.sort(key=lambda x: (-x["tot"], -x["newest"], x["sym"]))
            A(f"| # | 标的 | {cols[0]} | {cols[1]} | {cols[2]} | 合计 | 命中类型 | "
              f"最近信号 | 距今 | 最新价 | 历史上限内总命中 |")
            A("|---:|---|---:|---:|---:|---:|---|---|---:|---:|---:|")
            for i, x in enumerate(rows, 1):
                A(f"| {i} | `{x['sym']}` | {x['c1']} | {x['c2']} | {x['c3']} | "
                  f"**{x['tot']}** | {x['kinds']} | {fmt_local(x['newest'])} | "
                  f"{human_age(x['age'])} | {x['close']:.6g} | {x['n_all']} |")
            A("")
            A(f"（该级别共 **{len(rows)}** 个标的命中，**{sum(x['tot'] for x in rows)}** "
              f"个{side}信号）")
            A("")

    A("## 六、读数注意")
    A("")
    A("1. **买卖点是事件, 不是状态** —— 信号只在「一根笔冻结」的那一刻产生。"
      "窗口内有信号 ≠ 现在正处在买点上, 只表示「最近出现过」。")
    A("2. **共振 ≠ 更赚**。共振只说明该标的在多个尺度上同时出现同向结构, 是"
      "「值得进一步看」的筛选条件; 它的正期望仍取决于离场口径与过滤层(见第 7 条)。")
    A("3. **级别越低信号越多**, 这是结构密度决定的, 不是「低级别机会多」 —— "
      "15分钟 的一根笔和月线的一根笔都算「一笔」, 但前者只值几小时。"
      "所以「买点共振 8/8」里若含 15分钟, 含金量与「月线+周线」这种大级别共振"
      "不可同日而语 —— 主表的列序(长周期在前)就是为便于这样横向比较。")
    A("4. **卖点的含义**: 缠论卖点原本是**做多持仓的离场判据**(见 `sell.py`), "
      "同时也可作做空侧的入场信号。本表把买卖点并列, **不表示同时做多又做空** —— "
      "同一格两者都有, 只说「下方支撑和上方压力都还没被击穿」。")
    A(f"5. **格内每方向只标一个字母(= 窗口内该方向最近触发的那个), 不是「窗口内出过的"
      f"全部类型」。** 想知道某周期出过哪些类型, 看附录的 `命中类型` 列或逐信号 CSV"
      f"(真正「同刻」多命中的只有 {n_pair} 例, 且全是二买/三买、二卖/三卖落在同一根笔上)。"
      "主表回答「**最近一次是什么**」; 要问「**此刻**是否还活着」必须看 `--state` 那张表 —— "
      "两者不同: 状态表可能**回退**到更早一个仍有效的信号。")
    A("6. **不同级别的「距今」不可比。** 月线能回溯 9 年, 15分钟只能回溯 10 天。"
      "`历史上限内总命中` 一列受 1000 根上限约束, **不可跨级别直接比较**。")
    A("7. 本表**不含**仓位、止损、成本与收益 —— 它回答「哪里出现了什么结构」, "
      "不回答「该不该做」。缠论买卖点的正期望在 chanbook 的历史回测里依赖"
      "离场口径与过滤层, 单看信号密度排序会严重高估。")
    A("")

    txt = "\n".join(L) + "\n"
    with open(args.out + ".md", "w", encoding="utf-8") as fh:
        fh.write(txt)
    print(f"\n报告: {args.out}.md")
    print(f"按标的聚合: {args.out}_bysym.csv")
    print(f"逐信号明细: {args.out}.csv  全量: {args.out}.json")
    print("各类型命中:", dict(grand))
    return 0


if __name__ == "__main__":
    sys.exit(main())
