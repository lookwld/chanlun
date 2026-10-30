"""A/B 运行器 —— 同一份归档数据、同区间, 跑多组配置并输出可比报告。

用法:
    python tools/run_ab.py --market um --start 2020-01 --end 2026-08 \
        --variants all,b1,b2,b3,nostruct,b1confirm --jobs 12 --out out/ab_um.txt
    python tools/run_ab.py --market spot --start 2017-08 --end 2026-08 \
        --variants all,b1,b2,b3,nostruct,b1confirm --jobs 12 --out out/ab_spot.txt

标的池: --universe 0 = 归档里全部标的(默认); --top N = 时点标的池前 N 名的并集。
结果每 5 分钟分批落盘(--out), 进程被杀也不会丢已跑完的标的。

⚠ 长作业必须**脱离本会话**运行(建议交给系统计划任务托管),
否则作业会在会话结束时被回收。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                        # noqa: E402
from chanbook import report as R                      # noqa: E402
from chanbook.engine import run_all                   # noqa: E402
from chanbook.points import B1, B2, B3                # noqa: E402
from chanbook.strategy import DIR_GATE, BookCfg       # noqa: E402

# 变体表: name -> (说明, BookCfg 覆盖项)
VARIANTS: dict[str, tuple[str, dict]] = {
    "all":        ("三类买点全开 + 默认方向闸门", {}),
    "b1":         ("只做一买(逆势背驰)", {"kinds": (B1,)}),
    "b2":         ("只做二买(一买后确认)", {"kinds": (B2,)}),
    "b3":         ("只做三买(回抽不回中枢) —— 与旧引擎可比", {"kinds": (B3,)}),
    "b1confirm":  ("只做一买 + 区间套(确认级别同向背驰)", {"kinds": (B1,), "confirm_gate": True}),
    "nogate":     ("三类全开但去掉方向闸门", {"dir_gate": {B1: None, B2: None, B3: None}}),
    "nostruct":   ("三类全开 + 关掉结构破位离场", {"struct_on": False}),
    "b1wideoff":  ("三类全开 + 一买不做小仓宽止损特例", {"b1_wide": False}),
    "strict":     ("三类全开 + 背驰严格模式(四判据全成立)",
                   {"beichi_strict": True}),
    "pull3":      ("三类全开 + 回试须构成 ≥3 笔", {"pull_bi": 3}),
    "stopmin":    ("三类全开 + 止损缓冲按原注释本意(min ⇒ 结构-atr_mult×ATR)",
                   {"stop_buf": "min"}),
    # ---- 第二轮: 由第一轮实测结论(各自独立最优项)拼出的组合配置 ----
    "combo":      ("全开 + 回试≥3笔 + 无方向闸门 + 无一买小仓宽止损 + 一买区间套",
                   {"pull_bi": 3, "b1_wide": False, "confirm_gate": True,
                    "dir_gate": {B1: None, B2: None, B3: None}}),
    "b1x":        ("只做一买 + 区间套 + 无一买小仓宽止损",
                   {"kinds": (B1,), "b1_wide": False, "confirm_gate": True}),
    # ---- 第三轮: 目标距离(rr) 与 追踪倍数(trail_mult) ----
    # 动机: 2R 目标只有 4~12% 的笔能触发(rr 形同虚设),
    # 而 rr 是本项目 5 轮 20+ 变体里**唯一从未被检验过的维度**(此前全是入场过滤/离场开关)。
    # 注意: 改 rr 会改变路径(更多笔止盈离场 ⇒ 后续候选池不同) ⇒ 必须重跑, 不能静态推断。
    "rr10":       ("三类全开 + RR=1.0(目标=1×风险)", {"rr": 1.0}),
    "rr125":      ("三类全开 + RR=1.25", {"rr": 1.25}),
    "rr15":       ("三类全开 + RR=1.5", {"rr": 1.5}),
    "rr30":       ("三类全开 + RR=3.0(目标推远)", {"rr": 3.0}),
    # 追踪止损贡献了 um_all 的 +44,585R(笔均 +1.27R, 是真正的利润源), 其倍数同样从未被检验。
    # 默认值: um=2.0 / spot=3.0 ⇒ 两市场各能取到 {1.5, 2.0, 3.0(=all), 4.0} 四个点。
    "trail15":    ("三类全开 + 追踪倍数 1.5(更紧)", {"trail_mult": 1.5}),
    "trail20":    ("三类全开 + 追踪倍数 2.0(um 为默认值)", {"trail_mult": 2.0}),
    "trail30":    ("三类全开 + 追踪倍数 3.0(spot 为默认值)", {"trail_mult": 3.0}),
    "trail40":    ("三类全开 + 追踪倍数 4.0(更松)", {"trail_mult": 4.0}),
    # ---- 第四轮: 可加性检验 ----
    # spot 上 rr=3.0(+0.032pp) 与 trail_mult=2.0(+0.024pp) 各自改善, 且都表现为
    # "压低止盈占比、抬升追踪占比"。跑组合只为回答「0.024pp + 0.032pp 可加吗」,
    # 不是寻找可部署配置 —— 逐项拼最好正是 combo 过拟合的教训。
    "rr30t20":    ("三类全开 + RR=3.0 + 追踪倍数 2.0(可加性检验)",
                   {"rr": 3.0, "trail_mult": 2.0}),
}


def ms_of(ym: str, end: bool = False) -> int:
    """'2020-01' -> 该月1日 00:00 的毫秒(UTC); end=True -> 该月末 23:59。"""
    y, m = (int(x) for x in ym.split("-"))
    if end:
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        dt = datetime(y, m, 1, tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000) - 1
    dt = datetime(y, m, 1, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="um", choices=["um", "spot"])
    ap.add_argument("--start", default="2020-01")
    ap.add_argument("--end", default="2026-08")
    ap.add_argument("--variants", default="all,b1,b2,b3")
    ap.add_argument("--jobs", type=int, default=0)
    ap.add_argument("--top", type=int, default=0, help="时点标的池前 N 名")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 个标的(冒烟)")
    ap.add_argument("--out", default="")
    ap.add_argument("--cache-dir", default="")
    args = ap.parse_args()

    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    t0, t1 = ms_of(args.start), ms_of(args.end, end=True)
    names = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        print(f"未知变体: {unknown}; 可用: {list(VARIANTS)}")
        return 2
    out_path = args.out or f"out/ab_{args.market}_{args.start}_{args.end}.txt"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    lines: list[str] = []
    def emit(s: str = "") -> None:
        lines.append(str(s))
        try:
            print(s, flush=True)
        except Exception:
            pass

    emit(f"# chanbook A/B  {args.market}  {args.start} ~ {args.end}")
    emit(f"# 生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
         f"标的池={'全部' if not args.top else f'前{args.top}'}"
         f"{'  限' + str(args.limit) + '个' if args.limit else ''}")
    emit("")

    results: dict[str, list] = {}
    for name in names:
        desc, over = VARIANTS[name]
        # ⚠ 2026-09-29 升级后必须显式锚定 legacy: 本工具跑的是**历史 A/B**,
        #   变体之间的相对差异不能被离场口径的变化污染(否则老报告不可复现)。
        cfg = BookCfg.legacy(market=args.market, **over)
        emit(f"[{time.strftime('%H:%M:%S')}] 开始 {name}: {desc} ...")
        _flush(out_path, lines)
        ta = time.time()
        trades, diags = run_all(
            args.market, cfg, universe_top=args.top,
            limit=args.limit, jobs=args.jobs, progress=40,
            t_start=t0, t_end=t1)
        results[name] = trades
        emit(R.full_report(args.market, f"{name} — {desc}", trades, diags))
        emit(f"# {name} 耗时 {time.time() - ta:.0f}s")
        emit("")
        _flush(out_path, lines)
        if args.cache_dir:
            R.save_trades(os.path.join(args.cache_dir, f"{args.market}_{name}.json"),
                          trades, diags)

    emit("=" * 72)
    emit("汇总(单笔期望 = 候选池笔数加权毛期望, 唯一可跨变体比较的口径)")
    emit(f"{'变体':>12} {'笔数':>7} {'胜率':>7} {'单笔期望':>9} {'单笔净期望':>10} "
         f"{'盈亏比':>7} {'PF':>6} {'均持仓':>8}")
    for name in names:
        s = R.summarize(results[name], args.market)
        if not s["n"]:
            emit(f"{name:>12} {'0':>7}")
            continue
        emit(f"{name:>12} {s['n']:>7} {s['win_rate'] * 100:>6.1f}% "
             f"{s['exp_pct'] * 100:>+8.3f}% {s['net_exp_pct'] * 100:>+9.3f}% "
             f"{s['rr']:>7.2f} {s['pf']:>6.2f} {s['hold_min']:>7.0f}m")
    thr = 0.0012 if args.market == "um" else 0.0020
    emit("")
    emit(f"门槛: 单笔**毛**期望需 > {thr * 100:.2f}%"
         f"(双边手续费{'0.10%' if args.market == 'um' else '0.20%'}"
         f"+资金费余量) | 名义 合约15U / 现货5U, 一买走小仓特例减半")
    emit("说明: 净期望 = 名义额加权毛期望 - 双边费, 恒等式; 报告里的"
         "'单笔期望'是**非加权**均值(与旧引擎 seg_stats 同口径, 唯一可跨变体比较)")
    _flush(out_path, lines)
    print(f"\n报告已写入 {out_path}")
    return 0


def _flush(path: str, lines: list) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as e:                                # noqa: BLE001
        print(f"写盘失败 {path}: {e}")


if __name__ == "__main__":
    sys.exit(main())
