"""top20 现货 · 多周期回测 —— 用最新下载的币安市值前 20 现货数据, 横向对比周期效果。

背景(2026-09-30):
  币安市值前 20 现货 USDT 交易对数据(20 标的 × 5 档 15m/1h/4h/1d/1m)已下载到
  `../chan-quant/data_archive/top20/k/`。本工具回答:
    同一套生产默认口径(入场 B1+B2 · 离场 结构止损 + 一卖 S1)下,
    15m / 1h / 3h / 4h / 8h / 12h / 日线 各周期的效果如何?

方法:
  * 数据源 = top20(20 标的), 不复用 spot 全池;
  * 周期: 15m/1h/4h/1d 直接读归档文件(这四档已下载), 3h/8h/12h 由 1h 现场聚合;
  * 口径: 完全复用 `tools/backtest_engine.py` 的 `run_symbol` / `stats_row`,
    只替换其数据加载 `load_tf`(让它支持 15m 直读 + 多档聚合), 其余零改动。

诚实前提:
  本工具只测「周期」这一个维度。入场/离场/风控口径与生产默认一致, 不改任何策略开关。

无未来函数: 与 backtest_engine 同款(每根 bar 先 advance 已收盘, 再按收盘价成交)。

运行:
  python tools/backtest_top20_tf.py --tfs 15m,1h,3h,4h,8h,12h,1d --jobs 8 \\
      --out out/top20_tf.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                       # noqa: E402
from tools import backtest_engine as BE              # noqa: E402

# 补齐周期表(3h/8h/12h 不在 data.TF_MS, 与 2h 同款只在本进程内补齐, 不改 data.py)。
D.TF_MS.setdefault("3h", 3 * D.MS_HOUR)
D.TF_MS.setdefault("8h", 8 * D.MS_HOUR)
D.TF_MS.setdefault("12h", 12 * D.MS_HOUR)

# 直接归档可读的档位(top20 已下载 15m/1h/4h/1d)。
_DIRECT_TF = {"15m", "1h", "4h", "1d"}
# 需要从 1h 聚合的档位。
_AGGR_TF = {"3h": "1h", "8h": "1h", "12h": "1h"}

MARKET = "top20"

# 补齐 BE 的周期参数表(stats_row 里 BAR_MIN[tf] 用于"可用分钟数"折算)。
BE.TF_HOURS.update({"15m": 0.25, "3h": 3, "8h": 8, "12h": 12})
BE.BAR_MIN.update({"15m": 15, "3h": 180, "8h": 480, "12h": 720})


# ------------------------------------------------------------------ 数据加载(替换 BE.load_tf)
def load_tf_top20(market: str, sym: str, tf: str):
    """top20 多档加载: 15m/1h/4h/1d 直读, 3h/8h/12h 由 1h 聚合。"""
    if tf in _DIRECT_TF:
        base = D.load(market, sym, tf)
        return base if (base is not None and base.n > 0) else None
    base_tf = _AGGR_TF[tf]
    base = D.load(market, sym, base_tf)
    if base is None or base.n == 0:
        return None
    return D.resample(base, D.TF_MS[tf])


def symbols_top20() -> list[str]:
    """top20 标的池(直接读 top20_symbols.txt, 不用 data.symbols 的 all_symbols)。"""
    arch = D.ARCH
    p = os.path.join(arch, "top20_symbols.txt")
    if not os.path.exists(p):
        p = os.path.join(arch, "top20_final.txt")
    with open(p, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


# 替换 BE 的数据加载(只影响本进程; 不动 chanbook 核心代码)。
BE.load_tf = load_tf_top20


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", default="15m,1h,3h,4h,8h,12h,1d",
                    help="逗号分隔的周期列表")
    ap.add_argument("--kinds", default="b1,b2", help="入场构造")
    ap.add_argument("--exit", default="s1", help="离场口径(old/stop/s1/s123/s1_trail/s123_trail)")
    ap.add_argument("--gate", default="on", choices=["on", "off"])
    ap.add_argument("--start", default="2017-08")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", default="out/top20_tf.txt")
    ap.add_argument("--by-year", action="store_true")
    args = ap.parse_args()

    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    market = MARKET
    gate = args.gate == "on"
    t0, t1 = BE.ms_of(args.start), BE.ms_of(args.end, end=True)
    syms = symbols_top20()
    tfs = tuple(x.strip() for x in args.tfs.split(",") if x.strip())
    kinds = tuple(BE.KIND_MAP[k.strip()] for k in args.kinds.split(","))
    ex = BE.EXITS[args.exit]
    if args.exit not in BE.EXITS:
        print(f"未知离场口径 {args.exit}", file=sys.stderr)
        return 2

    # 变体 = 每个周期一个(固定口径)。
    variants = [{"name": f"top20 {tf} {args.kinds} {args.exit}",
                 "tf": tf, "kinds": args.kinds, "exit": args.exit} for tf in tfs]

    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(str(s))
        print(s, flush=True)
        if len(lines) % 20 == 0:
            BE._flush(args.out, lines)

    emit(f"# chanbook top20 现货多周期回测   {market}   {args.start} ~ {args.end}   "
         f"({len(syms)} 个标的, 门={args.gate})")
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit(f"# 口径: 入场 {args.kinds} · 离场 {args.exit}({ex['why']}) · 只做多 · "
         f"预热 {BE.WARMUP} 根 · 最少 {BE.MIN_BARS} 根 · 方向闸门高一级别 {BE.TREND_TF}")
    emit("# 周期: 15m/1h/4h/1d 直读归档, 3h/8h/12h 由 1h 现场聚合。")
    emit("")

    rows: list[dict] = []
    for v in variants:
        tf = v["tf"]
        emit(f"[{time.strftime('%H:%M:%S')}] {v['name']}  —  {ex['why']}")
        ta = time.time()
        tr, dg = BE.run_variant(syms, v, tf, market, kinds, gate, t0, t1, args.jobs)
        # top20 本质是现货, 统计口径(费率 FEE 等)复用 spot(market 只影响数据路径)。
        r = BE.stats_row(tr, "spot", tf, dg, v["name"])
        r.update(tf=tf, kinds=v["kinds"], exit=v["exit"], secs=time.time() - ta)
        rows.append(r)
        emit("  " + BE.fmt_row(r) + f"   [{r['secs']:.0f}s]")
        emit("")

    # ---- 汇总 ----
    emit("=" * 118)
    emit("周期横向对比(同一套入场/离场口径)")
    emit(BE.HDR)
    # 按周期从细到粗排序
    order = {"15m": 0, "1h": 1, "3h": 2, "4h": 3, "8h": 4, "12h": 5, "1d": 6}
    for r in sorted(rows, key=lambda x: order.get(x["tf"], 99)):
        emit(f"{r['tf']:>26} " + BE.fmt_row(r))
    emit("")
    emit("字段说明: 单笔毛 = 非加权毛价格期望 · 单笔净 = 名义额加权毛 − 双边费(现货0.20%)"
         " · 剔尾5% = 两侧各剔5%分位均值 · 中位 = 中位数 · 每持仓日 = 单笔毛/均持仓天数")
    emit("          在仓 = Σ持仓分钟/Σ可用分钟 · 折算年化 = 每持仓日×365×在仓占比(线性)")
    emit("          前3笔占比 = 最赚3笔占全部正收益比例(越高=越靠少数大赢家, 越不稳健)")

    # ---- 池基准 ----
    emit("")
    emit("=" * 118)
    emit("基准: top20 同池等权指数(逐日再平衡 —— tf 无关, 跨周期唯一可比基准)")
    for r in sorted(rows, key=lambda x: order.get(x["tf"], 99)):
        tf = r["tf"]
        bpd = int(round(24.0 / BE.TF_HOURS[tf])) if BE.TF_HOURS.get(tf) else 1
        pi = BE.pool_index(syms, tf, market, t0, t1, rebal_bars=bpd)
        if pi:
            emit(f"  {tf}: 全程 {pi['total'] * 100:+.1f}%  每日 {pi['per_day'] * 100:+.4f}%  "
                 f"年化 {pi['ann'] * 100:+.1f}%  ({pi['n_seg']:,} 段 / {pi['days']:.0f} 天)")

    # ---- 按年稳定性 ----
    if args.by_year:
        emit("")
        emit("=" * 118)
        emit("按年 · 单笔毛期望(入场年; — = 该年无成交)")
        yrs = sorted({datetime.fromtimestamp(t.entry_ts / 1000, __import__('datetime').timezone.utc).year
                      for r in rows if r.get("n") for t in r["trades"]})
        emit(f"{'周期':>8} {'笔数':>6} " + " ".join(f"{y:>7}" for y in yrs))
        for r in sorted(rows, key=lambda x: order.get(x["tf"], 99)):
            if not r.get("n"):
                continue
            acc: dict = {}
            for t in r["trades"]:
                y = datetime.fromtimestamp(t.entry_ts / 1000, __import__('datetime').timezone.utc).year
                acc.setdefault(y, []).append(t.pnl_pct)
            cells = [f"{sum(acc[y]) / len(acc[y]) * 100:>+6.2f}%" if y in acc else "   —  "
                     for y in yrs]
            emit(f"{r['tf']:>8} {r['n']:>6} " + " ".join(cells))
        emit("")

    BE._flush(args.out, lines)
    print(f"\n报告已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
