"""全市场扫描编排 —— 一条命令跑完「刷 exchangeInfo → 下载 K 线 → 反向审计 →
当前状态表 → 历史信号表」。

**为什么要有这个脚本**: 给看板/定时任务提供一个**可观测、可复用**的单一入口。
每一步都把进度原子写进 `status.json`, 同时追加到日志文件 —— 外部(看板)只要读
这两个文件就能回答"现在在做什么/做到哪了/上次什么时候跑完/有没有失败",
不需要与子进程通信, 也不怕看板自身重启。

设计要点(每条都有理由, 改动前先读):

- **只用标准库**。与 `tools/*.py` 保持一致 —— 托管服务器上只装了 ccxt/requests,
  没有 pandas/numpy, 不能引入新依赖。
- **子进程一律被 `nice` 降权**(默认 15)。托管机上还有实盘 bot 在跑, 扫描是**二等公民**:
  1 核机器上"扫描把交易循环饿死"是真实风险, 不是理论问题。
- **并发默认 1**(`--jobs`)。ProcessPool 在 1 核机器上只会争抢 CPU 且成倍吃内存;
  实测 3192 组合单进程约 1 分钟, 完全可以接受。要快再手动调大。
- **K 线快照按日期建目录** `klines_YYYYMMDD`; 同日重跑**复用同一目录**(覆盖写,
  不浪费一次下载)。新的一天**不再从零下载**: 把上一份快照**硬链接**过来再增量更新
  (下载器写文件走「临时 + os.replace」⇒ 写时复制, **旧快照逐字节不变**)。
  轮转时**只删本脚本自己创建过的快照**(记在 `_rescan_snapshots.json`),
  且受**数量**(`--keep`)与**字节预算**(`--budget-mb`)双重约束, **永远至少留 1 份** ——
  别人的快照永不触碰, 靠命名约定推断"这也是我的"是错的(见 `rotate_snapshots`)。
- **下载前先查磁盘**。`--reserve-mb`(默认 1500)是留给别人的地板 —— 机器上还有实盘
  bot、日志、每日备份;**剩余空间不够就中止, 绝不会把磁盘占满**(见 `disk_plan`)。
- **下载走增量**(`--incremental`, 默认开)。实测下载步 241s → 约 45s:
  一半来自连接保活, 一半来自只拉增量。`--no-incremental` 可退回全量。
  等价性由 `tests/check_incremental.py` 看守。
- **下载失败率超阈值就中止**(`--max-fail-rate`, 默认 2%)。半截数据会静默地让一批
  标的变成"根数不足" —— 报告照样生成、表格照样能看, 这是最难发现的一类错误。
- **审计非空就中止**。`--audit` 的语义是"保留池里不该有稳定币", 输出非空 = 有漏网,
  继续扫出来的结果不可信(判据见 SKILL.md 铁律 3)。
- **`--dry` 只打印将执行的命令**, 不落盘、不发请求。
- **失败不破坏上一次的好结果**: 扫描器的产物是**全部跑完才落盘**的, 所以中途失败
  时旧的 `.csv/.json/.md` 原样保留; 本脚本只更新 `status.json`, 不动结果文件。

产物(默认在 `<root>/out/scan/` 下):
- `exchangeInfo.json`     交易所交易对清单(每次重扫刷新)
- `klines_YYYYMMDD/`      该日 K 线快照
- `b123_spot_state.*`     当前状态表(`--state`): 此刻处于哪个信号
- `b123_spot.*`           历史信号表: 窗口内最近出过哪些信号
- `status.json`           运行状态(看板读它)
- `scan.log`              日志
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)          # chanbook/ 根

STEPS = [
    ("exch", "刷新 exchangeInfo"),
    ("download", "下载全市场 K 线"),
    ("audit", "反向审计(保留池找稳定币)"),
    ("state", "当前状态表扫描"),
    ("history", "历史信号表扫描"),
]

DOWNLOAD_RE = re.compile(r"(\d+)/(\d+)\s*\(")
SCAN_RE = re.compile(r"^\s*(\d+)/(\d+)\s*\(")


# ---------------------------------------------------------------- 状态文件

def atomic_json(path: str, obj) -> None:
    """原子替换 —— 看板随时可能在读, 不能让它读到半截 JSON。"""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


class Status:
    """运行状态: 内存一份 + 落盘一份 + 追加日志。"""

    def __init__(self, path: str, log_path: str, args: dict):
        self.path = path
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.d = {
            "state": "running",
            "pid": os.getpid(),
            "started": time.time(),
            "started_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "finished": None,
            "duration_sec": None,
            "step_index": 0,
            "step_total": len(STEPS),
            "step": None,
            "step_name": "启动",
            "progress": "",
            "error": None,
            "snapshot": None,
            "counts": {},
            "steps": [],
            "args": args,
        }
        self.flush()

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        try:
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass          # 日志写不进去不能影响扫描本身

    def flush(self) -> None:
        try:
            atomic_json(self.path, self.d)
        except OSError:
            pass

    def begin(self, idx: int) -> None:
        key, name = STEPS[idx]
        self.d.update(step_index=idx + 1, step_total=len(STEPS), step=key,
                      step_name=name, progress="")
        self.flush()

    def set_progress(self, text: str) -> None:
        if text != self.d.get("progress"):
            self.d["progress"] = text
            self.flush()

    def end_step(self, key: str, name: str, state: str, secs: float,
                 detail: str = "") -> None:
        self.d["steps"].append({"key": key, "name": name, "state": state,
                                "secs": round(secs, 1), "detail": detail})
        self.d["progress"] = ""
        self.flush()

    def finish(self, state: str, error: str | None = None) -> None:
        self.d.update(state=state, error=error, finished=time.time(),
                      duration_sec=round(time.time() - self.d["started"], 1))
        self.flush()


# ---------------------------------------------------------------- 子进程

def run_child(cmd: list[str], st: Status, nice: int, timeout: int,
              on_line=None) -> tuple[int, list[str]]:
    """跑一个子命令, 逐行转发到日志, 返回 (rc, 输出行)。

    看门狗线程在 timeout 秒后 kill —— 用线程而不是"读循环里判超时", 因为子进程
    可能长时间不输出, 读循环会一直阻塞在 readline 上, 超时判据形同虚设。
    """
    if nice:
        cmd = ["nice", "-n", str(nice)] + cmd          # Windows 上没有 nice, 跳过
        if os.name != "posix":
            cmd = cmd[3:]
    st.log("  $ " + " ".join(cmd))
    p = subprocess.Popen(cmd, cwd=st.d["args"].get("root") or None,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace",
                         bufsize=1)
    killed = {"v": False}

    def _kill():
        killed["v"] = True
        try:
            p.kill()
        except OSError:
            pass

    timer = threading.Timer(timeout, _kill) if timeout > 0 else None
    if timer:
        timer.start()
    lines: list[str] = []
    try:
        for line in p.stdout:                      # type: ignore[union-attr]
            line = line.rstrip("\n")
            lines.append(line)
            st.log("  | " + line)
            if on_line:
                on_line(line)
        rc = p.wait()
    finally:
        if timer:
            timer.cancel()
    if killed["v"]:
        st.log(f"  ! 超时 {timeout}s 被终止")
        return 124, lines
    return rc, lines


def refresh_exchange_info(root: str, base: str, st: Status) -> tuple[bool, str]:
    path = os.path.join(root, "out", "scan", "exchangeInfo.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"{base}/api/v3/exchangeInfo"
    req = urllib.request.Request(url, headers={"User-Agent": "chanbook-scan/1.0"})
    tmp = path + ".tmp"
    with urllib.request.urlopen(req, timeout=60) as r:      # noqa: S310 (固定 https)
        raw = r.read()
    d = json.loads(raw.decode("utf-8"))
    n = len(d.get("symbols", []))
    if n < 50:
        return False, f"exchangeInfo 只有 {n} 个交易对, 拒绝覆盖旧文件"
    with open(tmp, "wb") as fh:
        fh.write(raw)
    os.replace(tmp, path)
    return True, f"{n} 个交易对, {len(raw) / 1e6:.1f}MB"


def head_symbols(root: str, limit: int) -> list[str]:
    """取前 N 个 USDT 现货标的 —— 与 `download_all_klines.load_symbols` 同规则同序,
    这样 `--limit` 下"下载的"与"扫描的"是同一批(调试/冒烟测试用)。"""
    if limit <= 0:
        return []
    p = os.path.join(root, "out", "scan", "exchangeInfo.json")
    d = json.load(open(p, encoding="utf-8"))
    out = [s["symbol"] for s in d.get("symbols", [])
           if s.get("quoteAsset") == "USDT" and s.get("status") == "TRADING"
           and s.get("isSpotTradingAllowed", True)]
    return sorted(out)[:limit]


SNAP_RE = re.compile(r"^klines_\d{8}$")
OWNER_FILE = "_rescan_snapshots.json"          # 记录"本脚本自己建的"快照目录
SNAP_MB_FALLBACK = 220.0                       # 无历史快照时的单份估算(实测约 188MB)


def dir_mb(path: str) -> float:
    total = 0
    for root_, _dirs, files in os.walk(path):
        for fn in files:
            try:
                total += os.path.getsize(os.path.join(root_, fn))
            except OSError:
                pass
    return total / 1e6


def read_owner(scan_dir: str) -> list:
    try:
        owner = json.load(open(os.path.join(scan_dir, OWNER_FILE), encoding="utf-8"))
        return owner if isinstance(owner, list) else []
    except Exception:                                             # noqa: BLE001
        return []


def owned_snapshots(scan_dir: str, extra: str | None = None) -> list:
    """本脚本创建过、且此刻还在磁盘上的快照目录, **新的在前**。"""
    owner = read_owner(scan_dir)
    if extra and SNAP_RE.match(extra) and extra not in owner:
        owner = owner + [extra]
    return sorted([str(d) for d in owner
                   if SNAP_RE.match(str(d)) and os.path.isdir(os.path.join(scan_dir, str(d)))],
                  reverse=True)


def disk_plan(scan_dir: str, reserve_mb: int, budget_mb: int) -> dict:
    """按**服务器实际剩余空间**算出缓存允许占用的上限。

    - `reserve_mb`(**地板**): 留给别人的空间 —— 这台机器上还有实盘 bot、日志、
      每日备份。扫描是二等公民, 绝不能把剩余空间吃光。
    - `budget_mb`: 缓存(所有本脚本创建的快照之和)允许占用的上限。
      传 **0 = 自动**, 即「剩余空间 − 地板」: 能占多少占多少, 但**不碰地板**。

    这样"空间限制"不是写死的常数, 而是跟着服务器实际剩余空间走 —— 服务器变紧了
    它自动收紧, 换了更大的盘它自动放宽。
    """
    du = shutil.disk_usage(scan_dir)
    free_mb = du.free / 1e6
    budget = float(budget_mb) if budget_mb > 0 else max(0.0, free_mb - float(reserve_mb))
    return {"total_mb": du.total / 1e6, "used_mb": du.used / 1e6, "free_mb": free_mb,
            "reserve_mb": float(reserve_mb), "budget_mb": budget,
            "auto": budget_mb <= 0}


def estimate_snapshot_mb(scan_dir: str, cur: str | None = None) -> float:
    """本轮新快照大概要占多少 —— 取现有快照里**最大**的那份当估计。

    ⚠ 用 max 而不是"最新一份": 万一最新那份是个小样本残留(2.6MB), 估计值就会小两个
      数量级, 磁盘护栏等于失效。护栏宁可保守。
    """
    own = [d for d in owned_snapshots(scan_dir, cur) if d != cur]
    if own:
        return max(dir_mb(os.path.join(scan_dir, d)) for d in own)
    return SNAP_MB_FALLBACK


def seed_snapshot(scan_dir: str, kl_dir: str, st: Status) -> int:
    """新的一天不从零下载: 把上一份快照**硬链接**过来, 剩下的交给增量更新。

    用硬链接而不是复制的两个理由:
    - 几乎不占空间、瞬间完成(188MB 复制在这台 1 核机上要几秒, 还得先扣掉一份空间);
    - 下载器写文件走「临时文件 + `os.replace`」⇒ 写时复制。更新一份**不会**改动旧快照
      的 inode, 所以**旧快照逐字节不变**, 研究基线依然不可变。

    返回链接了几个文件。跨设备(同一 out/scan 下理论上不会)链接失败就退回普通复制。
    """
    own = [d for d in owned_snapshots(scan_dir) if d != os.path.basename(kl_dir)]
    if not own:
        st.log("  无历史快照可承接, 本轮从零全量下载")
        return 0
    src = os.path.join(scan_dir, own[0])
    os.makedirs(kl_dir, exist_ok=True)
    n = copied = 0
    for fn in os.listdir(src):
        s, d = os.path.join(src, fn), os.path.join(kl_dir, fn)
        if os.path.exists(d) or not os.path.isfile(s):
            continue
        try:
            os.link(s, d)
            n += 1
        except OSError:
            try:
                shutil.copy2(s, d)      # copy2 保留 mtime(扫描器用它判"已收盘")
                copied += 1
            except OSError:
                pass
    st.log(f"  承接上一份快照 {own[0]}: 硬链接 {n} 个"
           + (f", 复制 {copied} 个" if copied else ""))
    return n + copied


def rotate_snapshots(scan_dir: str, keep: int, budget_mb: float, st: Status) -> None:
    """只删**本脚本自己建的**快照目录, 同时受「数量」与「字节预算」两个约束。

    ⚠ 为什么要记 owner 清单(2026-10-05 踩过): 最初的实现是"按名字前缀 klines_ 排序,
    保留最近 N 个, 其余 rmtree" —— 结果一次本地冒烟测试就把研究用的基线快照
    `klines_20261002`(223MB) 和 `klines_1d`(28MB) 一起删了。它们名字长得一样,
    但**不是这个脚本建的**, 删掉是不可逆的。
    教训: **轮转只能删自己拥有的东西**; 靠命名约定去推断"这也是我的"是错的。

    ⚠ 为什么还要字节预算(2026-10-08 加): 数量约束挡不住"单份快照变大"(币种变多、
    周期变多、币安放宽 limit)。盘子只剩 3.1G 时 188MB × keep=2 看着没事, 但没有任何
    东西阻止它涨成 1GB。两张网一起收, 超出任一约束就从**最旧的**开始删。
    ⚠ **永远至少留 1 份** —— 全删的话下一轮既没有可承接的快照, 也没有基线可比。
    """
    cur = st.d.get("snapshot")
    own = owned_snapshots(scan_dir, cur)          # 新的在前
    sizes = {d: dir_mb(os.path.join(scan_dir, d)) for d in own}
    total = sum(sizes.values())
    st.log(f"  缓存: {len(own)} 份快照 {total:.0f}MB / 预算 {budget_mb:.0f}MB / "
           f"数量上限 {keep}")

    alive = list(own)
    doomed: list = []
    for d in reversed(own):                       # 从最旧的开始删
        if len(alive) <= 1:
            break
        over_count = keep > 0 and len(alive) > keep
        over_bytes = budget_mb > 0 and total > budget_mb
        if not (over_count or over_bytes):
            break
        alive.remove(d)
        doomed.append(d)
        total -= sizes[d]
    for d in doomed:
        shutil.rmtree(os.path.join(scan_dir, d), ignore_errors=True)
        st.log(f"  轮转: 删除旧快照 {d}(本脚本创建, {sizes[d]:.0f}MB)")

    try:
        atomic_json(os.path.join(scan_dir, OWNER_FILE), sorted(alive))
    except OSError:
        pass
    others = [d for d in os.listdir(scan_dir)
              if d.startswith("klines_") and d != cur and d not in own]
    if others:
        st.log(f"  保留: {', '.join(sorted(others))}(非本脚本创建, 不参与轮转)")


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="chanbook 全市场扫描编排")
    ap.add_argument("--root", default=DEFAULT_ROOT, help="chanbook 根目录")
    ap.add_argument("--python", default=sys.executable, help="子进程解释器")
    ap.add_argument("--status", default="", help="status.json 路径(默认 <root>/out/scan/status.json)")
    ap.add_argument("--log", default="", help="日志路径(默认 <root>/out/scan/scan.log)")
    ap.add_argument("--date", default="", help="快照日期 YYYYMMDD(默认今天)")
    ap.add_argument("--jobs", type=int, default=1, help="扫描并发(1 核机器保持 1)")
    ap.add_argument("--jobs-download", type=int, default=8, help="下载线程数")
    ap.add_argument("--nice", type=int, default=15, help="子进程 nice 值(0=不降权)")
    ap.add_argument("--keep", type=int, default=2, help="保留最近 N 个快照目录(0=不限)")
    ap.add_argument("--reserve-mb", type=int, default=1500,
                    help="给别的进程留的磁盘地板(MB); 剩余空间不够就中止下载")
    ap.add_argument("--budget-mb", type=int, default=0,
                    help="缓存目录允许占用的上限(MB); 0=自动(剩余空间−地板)")
    ap.add_argument("--incremental", dest="incremental", action="store_true",
                    default=True, help="下载走增量(默认开)")
    ap.add_argument("--no-incremental", dest="incremental", action="store_false",
                    help="下载走全量(排查数据问题时用)")
    ap.add_argument("--base", default="https://data-api.binance.vision",
                    help="币安现货端点")
    ap.add_argument("--max-fail-rate", type=float, default=0.02,
                    help="下载失败率上限, 超过则中止(默认 2%%)")
    ap.add_argument("--step-timeout", type=int, default=2400, help="单步超时秒")
    ap.add_argument("--limit", type=int, default=0,
                    help="只处理前 N 个标的(0=全市场); 供服务器冒烟测试用")
    ap.add_argument("--dry", action="store_true", help="只打印计划, 不执行")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    scan_dir = os.path.join(root, "out", "scan")
    site_base = args.base.rstrip("/")          # 站点根, 用于拼 /api/v3/exchangeInfo
    api_base = site_base + "/api/v3"           # 端点基址, 直接传给 download_all_klines
    day = args.date or time.strftime("%Y%m%d")
    # ⚠ 小样本(--limit)的 **K 线目录也必须独立**(2026-10-08 补): 原实现只把小样本的
    #   结果表隔离到 _smoke/, K 线仍写进真实的快照目录 —— 那 10 个标的被刷新、其余
    #   原样, 快照就成了"新旧混杂", 而报告照样生成、零报错。增量模式下更隐蔽。
    #   用**不带日期**的固定目录, 所以不会逐轮堆积。
    if args.limit:
        kl_dir = os.path.join(scan_dir, "_smoke", "klines")
        out_dir = os.path.join(scan_dir, "_smoke")
    else:
        kl_dir = os.path.join(scan_dir, f"klines_{day}")
        out_dir = scan_dir
    status_path = args.status or os.path.join(out_dir, "status.json")
    log_path = args.log or os.path.join(out_dir, "scan.log")
    py = args.python
    scan_py = os.path.join(root, "tools", "scan_spot_b1b2b3.py")
    dl_py = os.path.join(root, "tools", "download_all_klines.py")

    if args.dry:
        lm = f" --limit {args.limit}" if args.limit else ""
        print("DRY RUN")
        print(f"  root       = {root}")
        print(f"  snapshot   = {kl_dir}")
        print(f"  out_dir    = {out_dir}" + ("   ← 小样本, 不碰真实产物" if args.limit else ""))
        print(f"  status     = {status_path}")
        print(f"  log        = {log_path}")
        print(f"  1) GET {site_base}/api/v3/exchangeInfo -> {scan_dir}/exchangeInfo.json")
        plan = disk_plan(scan_dir, args.reserve_mb, args.budget_mb)
        print(f"  2) {py} {dl_py} --exch {scan_dir}/exchangeInfo.json "
              f"--outdir {kl_dir} --jobs {args.jobs_download} --base {api_base}"
              f"{' --incremental' if args.incremental else ''}{lm}")
        print(f"  磁盘: 总 {plan['total_mb']:.0f}MB / 已用 {plan['used_mb']:.0f}MB / "
              f"剩 {plan['free_mb']:.0f}MB; 地板 {plan['reserve_mb']:.0f}MB; "
              f"缓存预算 {plan['budget_mb']:.0f}MB"
              + ("(自动)" if plan["auto"] else ""))
        print(f"  本轮约需 {estimate_snapshot_mb(scan_dir, os.path.basename(kl_dir)):.0f}MB")
        print(f"  3) {py} {scan_py} --kl_dir {kl_dir} --audit{lm}")
        print(f"  4) {py} {scan_py} --kl_dir {kl_dir} --jobs {args.jobs} --state "
              f"--out {out_dir}/b123_spot_state{lm}")
        print(f"  5) {py} {scan_py} --kl_dir {kl_dir} --jobs {args.jobs} --bars 120 "
              f"--legs 10 --out {out_dir}/b123_spot{lm}")
        print(f"  keep={args.keep} nice={args.nice} max_fail_rate={args.max_fail_rate}")
        return 0

    arg_dump = {"root": root, "snapshot": os.path.basename(kl_dir), "jobs": args.jobs,
                "jobs_download": args.jobs_download, "nice": args.nice,
                "keep": args.keep, "base": site_base, "python": py,
                "max_fail_rate": args.max_fail_rate, "step_timeout": args.step_timeout,
                "limit": args.limit, "incremental": bool(args.incremental),
                "reserve_mb": args.reserve_mb, "budget_mb": args.budget_mb}
    st = Status(status_path, log_path, arg_dump)
    st.log(f"=== 重扫开始 (pid {os.getpid()}, 快照 {os.path.basename(kl_dir)}"
           + (f", 仅前 {args.limit} 标的" if args.limit else "") + ") ===")
    st.d["snapshot"] = os.path.basename(kl_dir)

    os.makedirs(kl_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    counts: dict = {}

    # 新的一天不从头下载: 承接上一份快照(硬链接, 写时复制 ⇒ 旧快照不变), 再增量更新。
    # 小样本(--limit)不承接 —— 它的目录是共用的固定目录, 承接没有意义。
    if args.incremental and not args.limit:
        try:
            seed_snapshot(scan_dir, kl_dir, st)
        except OSError as e:
            st.log(f"  ! 承接快照失败({e}), 本轮按全量下载")

    try:
        # ---- 1/5 刷新 exchangeInfo ----
        st.begin(0)
        t0 = time.time()
        try:
            ok, detail = refresh_exchange_info(root, site_base, st)
        except Exception as e:                                   # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        st.end_step("exch", STEPS[0][1], "ok" if ok else "failed",
                    time.time() - t0, detail)
        if not ok:
            st.log(f"  ! 刷新 exchangeInfo 失败: {detail}")
            st.finish("failed", f"exchangeInfo: {detail}")
            return 1
        counts["exch_pairs"] = int(detail.split()[0])
        scan_syms = head_symbols(root, args.limit)
        if scan_syms:
            st.log(f"  小样本模式: 只处理 {len(scan_syms)} 个标的 "
                   f"({scan_syms[0]} … {scan_syms[-1]})")

        # ---- 2/5 下载 K 线 ----
        st.begin(1)
        t0 = time.time()

        # ★ 下载前先查磁盘。绝不"先下载再看装不装得下" —— 这台机器上还有实盘 bot,
        #   把根分区写满的后果不是"扫描失败"而是实盘写不进状态。
        plan = disk_plan(scan_dir, args.reserve_mb, args.budget_mb)
        est = estimate_snapshot_mb(scan_dir, os.path.basename(kl_dir))
        st.log(f"  磁盘: 总 {plan['total_mb']:.0f}MB / 已用 {plan['used_mb']:.0f}MB / "
               f"剩 {plan['free_mb']:.0f}MB; 地板 {plan['reserve_mb']:.0f}MB; "
               f"缓存预算 {plan['budget_mb']:.0f}MB"
               + ("(自动=剩余−地板)" if plan["auto"] else ""))
        counts["disk_free_mb"] = round(plan["free_mb"])
        counts["disk_budget_mb"] = round(plan["budget_mb"])
        # ⚠ 立刻把 counts 挂到状态对象上(2026-10-09 补): 否则**磁盘不足被拒**时
        #   下面直接 finish+return, 而 `st.d["counts"] = counts` 在函数末尾才执行 ——
        #   看板拿到 counts={}, 磁盘卡片显示 "—"。偏偏这时候使用者最需要看到那两个数字
        #   (剩多少、预算是多少)。挂上后每次 flush 都会带上, 运行中也能实时看到。
        st.d["counts"] = counts
        if plan["free_mb"] - est < plan["reserve_mb"]:
            msg = (f"剩余空间不足: 只有 {plan['free_mb']:.0f}MB, 本轮约需 {est:.0f}MB, "
                   f"还要留 {plan['reserve_mb']:.0f}MB 地板 —— 拒绝下载, 免得把磁盘占满。"
                   f"可手动清理 {scan_dir}/klines_* 或调小 --reserve-mb")
            st.log(f"  ! {msg}")
            st.finish("failed", msg)
            return 1

        def on_dl(line: str) -> None:
            m = DOWNLOAD_RE.search(line)
            if m:
                st.set_progress(f"{m.group(1)}/{m.group(2)}")

        dl_cmd = [py, "-u", dl_py, "--exch", os.path.join(scan_dir, "exchangeInfo.json"),
                  "--outdir", kl_dir, "--jobs", str(args.jobs_download),
                  # ⚠ 下载器的 --base 要的是**含 /api/v3 的端点基址**, 而本脚本的 --base 是
                  #   站点根(还要拼 /api/v3/exchangeInfo) —— 两者差一个后缀, 传错会全量 404。
                  "--base", api_base]
        if args.incremental:
            dl_cmd += ["--incremental"]
        if args.limit:
            dl_cmd += ["--limit", str(args.limit)]
        rc, dl_lines = run_child(dl_cmd, st, args.nice, args.step_timeout, on_dl)
        tail = "\n".join(dl_lines[-8:])
        # 下载器末尾会打一行 `SUMMARY {...}`; 优先解析它, 解析不到才退回人读的那行
        summary = None
        m = re.search(r"^SUMMARY (\{.*\})\s*$", tail, re.M)
        if m:
            try:
                summary = json.loads(m.group(1))
            except ValueError:
                summary = None
        if summary:
            done = int(summary.get("files_ok", 0))
            fail = int(summary.get("files_fail", 0))
            secs = float(summary.get("secs", 0.0))
            counts.update(files_ok=done, files_fail=fail, download_secs=secs)
            detail = f"{done} 个文件, 失败 {fail}, 用时 {secs:.0f}s"
            if summary.get("mode") == "incremental":
                # 回退数**必须可见**: 它一涨就说明本地缓存在出问题(被截短/被换过)
                counts.update(incr_ok=int(summary.get("incr_ok", 0)),
                              full_fallback=int(summary.get("full_fallback", 0)),
                              bars_new=int(summary.get("bars_new", 0)))
                detail += (f", 增量 {summary.get('incr_ok', 0)}"
                           f"/回退 {summary.get('full_fallback', 0)}"
                           f", 新增 {summary.get('bars_new', 0)} 根")
        else:
            m = re.search(r"完成 (\d+) 个, 失败 (\d+) 个, 用时 ([\d.]+)s", tail)
            if m:
                done, fail, secs = int(m.group(1)), int(m.group(2)), float(m.group(3))
                counts.update(files_ok=done, files_fail=fail, download_secs=secs)
                detail = f"{done} 个文件, 失败 {fail}, 用时 {secs:.0f}s"
            else:
                detail = tail.splitlines()[-1] if tail else f"rc={rc}"
        rate = (counts.get("files_fail", 0) / max(1, counts.get("files_ok", 0) + counts.get("files_fail", 0)))
        ok = rc == 0 and counts.get("files_ok", 0) > 0 and rate <= args.max_fail_rate
        st.end_step("download", STEPS[1][1], "ok" if ok else "failed",
                    time.time() - t0, detail)
        if not ok:
            msg = (f"下载失败率 {rate:.1%} 超上限 {args.max_fail_rate:.0%}"
                   if counts.get("files_ok") else f"下载失败 rc={rc}")
            st.log(f"  ! {msg} —— 中止, 不产出半截结果")
            st.finish("failed", msg)
            return 1

        # ---- 3/5 反向审计(非空即中止) ----
        st.begin(2)
        t0 = time.time()
        common = [py, "-u", scan_py, "--kl_dir", kl_dir, "--exch",
                  os.path.join(scan_dir, "exchangeInfo.json")]
        if scan_syms:
            common += ["--syms", ",".join(scan_syms)]
        rc, ad_lines = run_child(common + ["--audit"], st, args.nice, args.step_timeout)
        hits = [x for x in ad_lines if "⚠" in x]
        detail = f"保留池无漏网稳定币" if not hits else f"⚠ 发现 {len(hits)} 个疑似锚定物"
        st.end_step("audit", STEPS[2][1], "ok" if (rc == 0 and not hits) else "failed",
                    time.time() - t0, detail)
        if rc != 0 or hits:
            msg = f"反向审计不通过({len(hits)} 个疑似稳定币), 结果不可信, 中止"
            st.log(f"  ! {msg}")
            for h in hits:
                st.log("     " + h.strip())
            st.finish("failed", msg)
            return 1
        counts["audit_hits"] = 0

        # ---- 4/5 当前状态表 ----
        st.begin(3)
        t0 = time.time()

        def on_scan(line: str) -> None:
            m = SCAN_RE.match(line)
            if m:
                st.set_progress(f"{m.group(1)}/{m.group(2)}")

        rc, sc_lines = run_child(
            common + ["--jobs", str(args.jobs), "--state",
                      "--out", os.path.join(out_dir, "b123_spot_state")],
            st, args.nice, args.step_timeout, on_scan)
        detail = _scan_detail(sc_lines, rc)
        st.end_step("state", STEPS[3][1], "ok" if rc == 0 else "failed",
                    time.time() - t0, detail)
        if rc != 0:
            st.finish("failed", f"状态表扫描失败 rc={rc}")
            return 1

        # ---- 5/5 历史信号表 ----
        st.begin(4)
        t0 = time.time()
        rc, hi_lines = run_child(
            common + ["--jobs", str(args.jobs), "--bars", "120", "--legs", "10",
                      "--out", os.path.join(out_dir, "b123_spot")],
            st, args.nice, args.step_timeout, on_scan)
        detail = _scan_detail(hi_lines, rc)
        st.end_step("history", STEPS[4][1], "ok" if rc == 0 else "failed",
                    time.time() - t0, detail)
        if rc != 0:
            st.finish("failed", f"历史表扫描失败 rc={rc}")
            return 1

        # 结果文件的 mtime/大小写进状态, 供看板判断"展示的是哪一轮的结果"
        counts.update(_result_meta(out_dir, kl_dir))
        st.d["counts"] = counts
        # 小样本的轮转要跳过: 冒烟测试不该动真实缓存(上一版会拿"klines"当当前快照去轮转)
        if args.limit:
            st.log("  小样本模式: 跳过快照轮转(不碰真实缓存)")
        else:
            rotate_snapshots(scan_dir, args.keep, plan["budget_mb"], st)
        st.log(f"=== 重扫完成: {counts.get('symbols', '?')} 标的 / "
               f"{counts.get('cells', '?')} 组合 ===")
        st.finish("ok")
        return 0
    except Exception as e:                                        # noqa: BLE001
        st.log(f"  ! 未捕获异常: {type(e).__name__}: {e}")
        st.finish("failed", f"{type(e).__name__}: {e}")
        return 1


def _scan_detail(lines: list[str], rc: int) -> str:
    for ln in reversed(lines):
        m = re.search(r"标的 (\d+)\(全部 (\d+), 排除 (\d+)\) × 周期 (\d+) = (\d+) 组合", ln)
        if m:
            return (f"{m.group(1)} 标的(全部 {m.group(2)}, 排除 {m.group(3)}) "
                    f"× {m.group(4)} 周期 = {m.group(5)} 组合")
    return f"rc={rc}"


def _result_meta(scan_dir: str, kl_dir: str) -> dict:
    """把产物文件的时间戳/大小、以及组合数写进状态(供看板显示新鲜度)。"""
    meta: dict = {}
    try:
        d = json.load(open(os.path.join(scan_dir, "b123_spot_state.json"), encoding="utf-8"))
        meta["symbols"] = len(d.get("kept", []) or [])
        meta["excluded"] = len(d.get("excluded", []) or [])
        meta["levels"] = len(d.get("levels", []) or [])
        meta["cells"] = meta["symbols"] * meta["levels"]
        meta["generated"] = d.get("generated")
    except Exception:                                             # noqa: BLE001
        pass
    files = {}
    for name in ("b123_spot_state_bysym.csv", "b123_spot_state.csv",
                 "b123_spot_bysym.csv", "b123_spot.csv"):
        p = os.path.join(scan_dir, name)
        if os.path.exists(p):
            files[name] = {"mtime": os.path.getmtime(p),
                           "size": os.path.getsize(p)}
    meta["files"] = files
    meta["kl_dir"] = os.path.basename(kl_dir)
    return meta


if __name__ == "__main__":
    sys.exit(main())
