"""增量下载 + 磁盘护栏 自检 —— **离线、确定性、不联网**。

为什么要有它: `--incremental` 把"每次都从交易所重新取全量"换成了"在本地缓存上追加"。
这是拿**正确性**换时间的一步, 所以必须有东西盯着"它真的等价"。

覆盖三块:
  A 网格规则      `grid_of` 必须按 `min(周期, 1天)` —— 见 download_all_klines 的注释,
                  写成 `tf_ms` 会让 1343 个文件全部静默退回全量(提速归零还不报错)。
  B 缓存校验      7 类坏数据必须被拒; **真实缺口必须被放行**(FTT/LUNA 停牌是物理事实)。
  C 增量等价性    走 `update_one` 全流程(用伪造的交易所当数据源):
                  增量结果必须与"全新全量下载"逐根一致; 该回退的必须回退。
  D 磁盘护栏      `disk_plan` / `rotate_snapshots` / `seed_snapshot` 的边界:
                  永远至少留 1 份、只删自己创建的、硬链接是写时复制。

跑法: `PYTHONUTF8=1 <venv-python> tests/check_incremental.py`
"""
from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from chanbook import data as D                                    # noqa: E402
import tools.download_all_klines as DL                            # noqa: E402
import tools.rescan_market as RM                                  # noqa: E402

N_PASS = 0
FAILS: list[str] = []
TFS = ["1M", "1w", "3d", "1d", "4h", "1h", "30m", "15m"]
MS_DAY = 86_400_000
NOW = 1_790_000_000_000          # 固定"现在", 保证确定性


def ck(cond, msg: str) -> bool:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)
    return bool(cond)


# --------------------------------------------------------------- 造数据

def phase_anchor(tf: str) -> int:
    """该周期的**真实网格相位**在 2024-01-01 之后的第一个落点。

    这些相位是 2026-10-08 从 4024 个真实文件里实测出来的, 不是猜的:
      - `1M` 每月 **1 日 00:00 UTC**
      - `1w` 每周 **周一 00:00 UTC**（epoch 是周四 ⇒ `ts % 7天` 恒等于 4 天）
      - `3d` 相位是 `ts % 3天 == 1天`
      - `1d/4h/1h/30m/15m` 与 epoch 对齐, `ts % tf_ms == 0`
    """
    base = 1_704_067_200_000      # 2024-01-01 00:00:00 UTC(周一、亦是月初)
    tfm = D.TF_MS[tf]
    if tf == "1M":
        return base
    if tf == "1w":
        return base
    if tf == "3d":
        t = base
        while t % tfm != 86_400_000:
            t += MS_DAY
        return t
    return base


def mk_series(tf: str, n: int, end_ms: int, price0: float = 100.0) -> list:
    """造 n 根 ts 落在**真实网格相位**上的 K 线, 最后一根是不完整的"进行中"那根。"""
    tfm = D.TF_MS[tf]
    anchor = phase_anchor(tf)
    step_day = tf == "3d"
    # 往回推 n 根
    if tf == "1M":
        d = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        months = []
        while True:
            months.append(int(d.timestamp() * 1000))
            y, m = d.year, d.month + 1
            if m > 12:
                y, m = y + 1, 1
            d = d.replace(year=y, month=m)
            if d.timestamp() * 1000 > end_ms:
                break
        ts = months[-n:]
    else:
        step = MS_DAY if step_day else tfm
        k = int((end_ms - anchor) // step)
        ts = [anchor + i * step for i in range(max(0, k - n + 1), k + 1)]
    out = []
    for i, t in enumerate(ts):
        p = price0 + i
        out.append([t, p, p + 1.0, p - 1.0, p + 0.5, 10.0, 1000.0])
    return out


def grid_ok(tf: str, n: int) -> bool:
    ok, _ = DL.validate_cache(mk_series(tf, n, NOW), tf, NOW)
    return ok


# --------------------------------------------------------------- 伪造交易所

class FakePool:
    """只实现 `get(path)` —— `fetch_klines` 拿它当连接池用, 于是整条链路离线可跑。"""

    def __init__(self, series: dict, limit: int = 1000):
        self.series = series
        self.limit = limit

    def get(self, path: str) -> bytes:
        import urllib.parse as up
        d = dict(up.parse_qsl(up.urlsplit(path).query))
        rows = self.series[(d["symbol"], d["interval"])]
        lim = int(d.get("limit", 1000))
        st = d.get("startTime")
        sel = rows[-lim:] if st is None else [r for r in rows if r[0] >= int(st)][:lim]
        # 交易所原始格式是 12 列, 第 7 列(index 7)是 quote volume
        return json.dumps([[r[0], r[1], r[2], r[3], r[4], r[5], 0, r[6]]
                           for r in sel]).encode()


def run_update(tf: str, exchange: list, cache: list | None, prev_bars=None,
               keep: int = 1000, incremental: bool = True, sym: str = "TESTUSDT"):
    """把一份缓存放进临时目录, 用 FakePool 跑一次 `update_one`, 返回 (状态, 落盘的行)。"""
    d = tempfile.mkdtemp(prefix="cbincr_")
    try:
        path = os.path.join(d, f"{sym}_{tf}.json")
        if cache is not None:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
        pool = FakePool({(sym, tf): exchange})
        st = DL.update_one(pool, "/api/v3", sym, tf, path, keep,
                           incremental, NOW, prev_bars or {})
        rows = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else None
        return st, rows
    finally:
        shutil.rmtree(d, ignore_errors=True)


# =============================================================== A 网格规则

print("== A 网格规则 (grid_of) ==")
for tf in TFS:
    expect = min(D.TF_MS[tf], MS_DAY)
    ck(DL.grid_of(tf) == expect, f"A grid_of({tf}) 应为 {expect}, 实得 {DL.grid_of(tf)}")
# 这三档是"非均匀周期", 网格必然小于周期长度 —— 断言它们**不是** tf_ms
for tf in ("1M", "1w", "3d"):
    ck(DL.grid_of(tf) < D.TF_MS[tf],
       f"A {tf} 是非均匀周期, 网格必须小于 tf_ms(否则该列全部误判为坏缓存)")
    ck(grid_ok(tf, 60), f"A {tf} 的真实相位数据应通过校验")

# =============================================================== B 缓存校验

print("== B 缓存校验 (validate_cache) ==")
for tf in TFS:
    rows = mk_series(tf, 50, NOW)
    ok, why = DL.validate_cache(rows, tf, NOW)
    ck(ok, f"B {tf} 正常数据被误拒: {why}")

# 真实缺口(FTT/LUNA 那种停牌)必须放行 —— 实测 1w 5 个文件 / 3d 8 个 / 1d 1 个
gap = mk_series("1d", 30, NOW)
gap = gap[:10] + gap[20:]                       # 挖掉中间 10 天
ok, why = DL.validate_cache(gap, "1d", NOW)
ck(ok, f"B 真实缺口(停牌)被误拒: {why}")
# 但"间隔不是网格整数倍"必须拒
bad = mk_series("1d", 10, NOW)
bad[5][0] += 3600_000                           # 偏 1 小时 ⇒ 1 天的网格上非法
ck(not DL.validate_cache(bad, "1d", NOW)[0], "B 间隔偏离网格应被拒")

BAD_CASES = [
    ("空列表", lambda: []),
    ("空文件语义", lambda: None),
    ("行列数不足", lambda: [r[:5] for r in mk_series("1d", 10, NOW)]),
    ("价格非正", lambda: [[r[0], 0.0, r[2], r[3], r[4], r[5], r[6]]
                          for r in mk_series("1d", 10, NOW)]),
    ("非数值", lambda: [[r[0], "x", r[2], r[3], r[4], r[5], r[6]]
                        for r in mk_series("1d", 10, NOW)]),
    ("ts 未对齐网格", lambda: [[r[0] + 3600_000] + r[1:]
                               for r in mk_series("1d", 10, NOW)]),
    ("ts 重复", lambda: mk_series("1d", 10, NOW)[:5]
                        + [mk_series("1d", 10, NOW)[4]]
                        + mk_series("1d", 10, NOW)[6:]),
    ("ts 乱序", lambda: list(reversed(mk_series("1d", 10, NOW)))),
    ("末根在未来", lambda: [[r[0] + 400 * MS_DAY] + r[1:]
                            for r in mk_series("1d", 10, NOW)]),
]
for name, fn in BAD_CASES:
    rows = fn()
    if rows == [] and name == "空列表":
        pass
    ok, why = DL.validate_cache(rows, "1d", NOW)
    ck(not ok, f"B 坏数据未被拒: {name} (实得 ok={ok} {why})")

# =============================================================== C 增量等价性

print("== C 增量等价性 (update_one) ==")

def equiv_case(tf, keep=1000, n_exchange=1200, cache_len=1000, drag_new=True):
    """构造: 交易所全序列 + 一份"若干轮之前"的缓存, 断言增量结果 == 全新全量下载。"""
    ex = mk_series(tf, n_exchange, NOW)
    if cache_len >= n_exchange:
        cache = ex
    else:
        cache = ex[:cache_len]
    stt, rows = run_update(tf, ex, cache, keep=keep)
    expect = ex[-keep:]
    return stt, rows, expect, cache


for tf in TFS:
    if tf == "1M":
        continue                       # 月线真实历史上限 111 根, 这里用 1200 根不写实
    stt, rows, expect, _ = equiv_case(tf, cache_len=1000)
    ck(stt["mode"] == "incr", f"C {tf} 应走增量, 实得 {stt['mode']} ({stt['reason']})")
    ck(rows == expect, f"C {tf} 增量结果与全量下载不一致 "
                       f"({len(rows or [])} vs {len(expect)} 根)")

# ★ 末根必须被覆盖成终值: 缓存里那根是"进行中"的半截, 交易所那份是收盘终值
tf = "1h"
ex = mk_series(tf, 1100, NOW)
cache = [list(r) for r in ex[:1000]]
cache[-1][4] = cache[-1][4] + 999.0            # 缓存末根是"半截"的收盘价
stt, rows = run_update(tf, ex, cache)
ck(rows[-1][4] == ex[-1][4],
   "C 缓存末根没有被终值覆盖(会留下一根永远不更新的半截 K 线)")
ck(stt["mode"] == "incr", "C 末根覆盖场景应走增量")

# 缓存末根在交易所那边**对不上**(例如缓存来自更早、已被修剪的一段历史)
#   ⇒ 必须回退全量。构造: 缓存到 NOW-500h, 但该标的交易所只有最近 300 根(最早 NOW-299h),
#     于是 startTime=NOW-500h 返回的首根不是缓存末根。
#     注意 now-last=500h ≤ keep×tf=1000h, 所以要越过"缓存过旧"那道闸先撞上这条。
ex2 = mk_series("1h", 300, NOW)
cache2 = mk_series("1h", 1000, NOW - 500 * 3600_000)
stt, rows = run_update("1h", ex2, cache2)
ck(stt["mode"] == "full" and "首根" in stt["reason"],
   f"C 首根与缓存末根不匹配时应回退全量, 实得 {stt['mode']}/{stt['reason']}")

# 缓存过旧(缺口 > keep×tf) ⇒ 必须回退, 否则增量会被 limit 截断、**丢掉最新的**
old = mk_series("15m", 1000, NOW - 30 * MS_DAY)
stt, _ = run_update("15m", mk_series("15m", 1000, NOW), old, keep=1000)
ck(stt["mode"] == "full" and "过旧" in stt["reason"],
   f"C 缓存过旧必须回退全量(否则静默丢最新数据), 实得 {stt['mode']}/{stt['reason']}")

# 缓存被截短(比上轮少) ⇒ 必须回退 —— 增量只会追加, 靠它自己永远长不回来
ex3 = mk_series("1d", 800, NOW)
short = ex3[-2:]
stt, rows = run_update("1d", ex3, short, prev_bars={"TESTUSDT_1d": 800})
ck(stt["mode"] == "full" and "比上轮少" in stt["reason"],
   f"C 缓存被截短必须回退全量, 实得 {stt['mode']}/{stt['reason']}")
ck(len(rows) == 800, "C 截短回退后应恢复成完整的 800 根")

# 半数坏缓存: 只有坏的那个回退, 好的照走增量
d = tempfile.mkdtemp(prefix="cbincr_mix_")
try:
    good = mk_series("1d", 900, NOW)
    badrows = list(good)
    badrows.insert(7, good[6])
    for sym, rows_ in (("AAAUSDT", good), ("BBBUSDT", badrows)):
        with open(os.path.join(d, f"{sym}_1d.json"), "w", encoding="utf-8") as fh:
            json.dump(rows_, fh)
    pool = FakePool({("AAAUSDT", "1d"): good, ("BBBUSDT", "1d"): good})
    modes = {}
    for sym in ("AAAUSDT", "BBBUSDT"):
        s = DL.update_one(pool, "/api/v3", sym, "1d",
                          os.path.join(d, f"{sym}_1d.json"), 1000, True, NOW, {})
        modes[sym] = s["mode"]
    ck(modes["AAAUSDT"] == "incr", "C 好缓存应走增量")
    ck(modes["BBBUSDT"] == "full", "C 坏缓存应单独回退(不波及其它文件)")
finally:
    shutil.rmtree(d, ignore_errors=True)

# 全量模式的产物应与增量模式的产物一致(同一天内先全量后增量 = 幂等)
tf = "4h"
ex = mk_series(tf, 1050, NOW)
_, full_rows = run_update(tf, ex, None, incremental=False)
_, inc_rows = run_update(tf, ex, full_rows)
ck(inc_rows == full_rows, f"C 全量后再增量应幂等(逐根不变), {tf}")

# =============================================================== D 磁盘护栏

print("== D 磁盘护栏 ==")

with tempfile.TemporaryDirectory(prefix="cbdisk_") as d:
    plan = RM.disk_plan(d, 1500, 0)
    ck(plan["budget_mb"] > 0, "D 自动预算应为正")
    ck(abs(plan["budget_mb"] - max(0.0, plan["free_mb"] - 1500)) < 1.0,
       "D 自动预算应等于「剩余 − 地板」")
    plan2 = RM.disk_plan(d, 1500, 77)
    ck(plan2["budget_mb"] == 77 and not plan2["auto"], "D 显式预算应原样生效")

    # 造 3 份"自己的"快照 + 1 份"别人的", 每份塞一个 1KB 文件
    def mk(name, kb=1):
        p = os.path.join(d, name)
        os.makedirs(p, exist_ok=True)
        with open(os.path.join(p, "x.json"), "wb") as fh:
            fh.write(b"0" * (kb * 1024))
        return p

    class St:
        def __init__(self):
            self.d = {"snapshot": "klines_20260303"}
            self.msgs = []

        def log(self, m):
            self.msgs.append(m)

    for n in ("klines_20260101", "klines_20260102", "klines_20260303"):
        mk(n)
    mk("klines_20250505", 1)                     # 别人的(不在 owner 清单里)
    mk("klines_1d", 1)                           # 研究基线(名字不匹配 SNAP_RE)
    with open(os.path.join(d, RM.OWNER_FILE), "w", encoding="utf-8") as fh:
        json.dump(["klines_20260101", "klines_20260102", "klines_20260303"], fh)

    st = St()
    RM.rotate_snapshots(d, 2, 0, st)
    left = sorted(x for x in os.listdir(d) if x.startswith("klines_"))
    ck("klines_20260101" not in left, "D keep=2 时应删掉最旧的一份")
    ck("klines_20260303" in left and "klines_20260102" in left, "D 应保留最新两份")
    ck("klines_20250505" in left, "D **绝不能**删不是自己创建的快照")
    ck("klines_1d" in left, "D 研究基线必须原样保留")

    # 字节预算收紧 ⇒ 继续删, 但**永远至少留 1 份**
    st = St()
    RM.rotate_snapshots(d, 0, 0.0005, st)        # 预算小于一份 1KB ⇒ 只能留 1 份
    left = sorted(x for x in os.listdir(d) if x.startswith("klines_") and x in
                  ("klines_20260102", "klines_20260303"))
    ck(len(left) == 1, f"D 字节预算应把数量压到 1 份, 实得 {left}")
    ck("klines_20260303" in left, "D 字节预算下应保留**最新**那份")

    st = St()
    RM.rotate_snapshots(d, 0, 0.0000001, st)
    ck(os.path.isdir(os.path.join(d, "klines_20260303")),
       "D 预算极小也必须至少保留 1 份(全删的话下轮就没有基线可承接)")

# 硬链接 = 写时复制: 更新新快照不得改动旧快照
with tempfile.TemporaryDirectory(prefix="cbseed_") as d:
    src = os.path.join(d, "klines_20260101")
    os.makedirs(src, exist_ok=True)
    with open(os.path.join(src, "A_1d.json"), "w", encoding="utf-8") as fh:
        json.dump([[1, 1, 1, 1, 1, 1, 1]], fh)
    with open(os.path.join(d, RM.OWNER_FILE), "w", encoding="utf-8") as fh:
        json.dump(["klines_20260101"], fh)

    class St2:
        def __init__(self):
            self.d = {"snapshot": "klines_20260102"}
            self.msgs = []

        def log(self, m):
            self.msgs.append(m)

    dst = os.path.join(d, "klines_20260102")
    n = RM.seed_snapshot(d, dst, St2())
    ck(n == 1 and os.path.exists(os.path.join(dst, "A_1d.json")),
       "D 承接快照应把上一份的文件带过来")

    # 下载器用的是原子写 ⇒ os.replace 换掉目录项, 旧 inode 不受影响
    DL.write_atomic(os.path.join(dst, "A_1d.json"), [[2, 2, 2, 2, 2, 2, 2]])
    old = json.load(open(os.path.join(src, "A_1d.json"), encoding="utf-8"))
    new = json.load(open(os.path.join(dst, "A_1d.json"), encoding="utf-8"))
    ck(old == [[1, 1, 1, 1, 1, 1, 1]], "D 硬链接 + 原子写必须是写时复制: 旧快照不能被改")
    ck(new == [[2, 2, 2, 2, 2, 2, 2]], "D 新快照应写入新内容")

    # 承接后的目录应能被扫描器当正常缓存读(命名与格式都对)
    ck(os.path.exists(os.path.join(dst, "A_1d.json")), "D 承接产物文件名应保持 {SYM}_{tf}.json")

# _bars.json 不能被当成 K 线文件
sym_file = os.path.join(_ROOT, "out", "scan", "klines_20261004")
if os.path.isdir(sym_file):
    n_kline = len([f for f in os.listdir(sym_file) if f.endswith(".json")
                   and not f.startswith("_")])
    ck(n_kline == 4024, f"D 真实快照应是 4024 个 K 线文件, 实得 {n_kline}")

# ---- 被磁盘护栏拒绝时: 必须中止 + 必须把磁盘数字写进状态(看板要显示) ----
# 2026-10-09 实测发现过一次: 拒绝对了, 但 counts 在函数末尾才挂到 st.d 上,
# 于是 status.json 里 counts={} —— 看板磁盘卡片显示 "—"。偏偏这时候使用者最需要
# 看到"还剩多少 / 预算是多少"。这条断言就是钉住那个修复的。
with tempfile.TemporaryDirectory(prefix="cbrefuse_") as r:
    os.makedirs(os.path.join(r, "out", "scan"), exist_ok=True)
    called = {"dl": 0}
    _dp, _rc, _rx = RM.disk_plan, RM.run_child, RM.refresh_exchange_info
    _argv = sys.argv
    RM.disk_plan = lambda sd, res, bud: {
        "total_mb": 9283.0, "used_mb": 5975.0, "free_mb": 3292.0,
        "reserve_mb": 99999.0, "budget_mb": 0.0, "auto": True}
    RM.refresh_exchange_info = lambda root, base, st: (True, "3724 个交易对, 17.7MB")

    def _boom(*a, **k):
        called["dl"] += 1
        raise AssertionError("磁盘不足时**不该**启动下载子进程")

    RM.run_child = _boom
    sp = os.path.join(r, "out", "scan", "status.json")
    sys.argv = ["rescan_market.py", "--root", r, "--status", sp,
                "--log", os.path.join(r, "out", "scan", "scan.log"),
                "--reserve-mb", "99999"]
    try:
        rc = RM.main()
    finally:
        RM.disk_plan, RM.run_child, RM.refresh_exchange_info = _dp, _rc, _rx
        sys.argv = _argv
    stf = json.load(open(sp, encoding="utf-8"))
    ck(rc == 1, f"D 磁盘不足应返回 1, 实得 {rc}")
    ck(stf["state"] == "failed", "D 磁盘不足应记为 failed")
    ck("剩余空间不足" in (stf.get("error") or ""), "D 应给出人话错误")
    ck(called["dl"] == 0, "D 磁盘不足时绝不能启动下载")
    ck(stf["counts"].get("disk_free_mb") == 3292,
       f"D 被拒时也要把可用空间写进状态, 实得 {stf['counts'].get('disk_free_mb')}")
    ck(stf["counts"].get("disk_budget_mb") is not None,
       "D 被拒时也要把缓存预算写进状态(看板磁盘卡片靠它)")

# =============================================================== 汇总

print(f"\n== 增量下载自检: 通过 {N_PASS} 项, 失败 {len(FAILS)} 项 ==")
for f in FAILS:
    print("  ✗", f)
if FAILS:
    print("\n失败要改的是 tools/download_all_klines.py 或 tools/rescan_market.py, 不是本测试。")
return_code = 1 if FAILS else 0
sys.exit(return_code)
