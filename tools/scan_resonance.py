"""现货全市场**多级别共振**扫描 —— 长周期与短周期同时给出买点。

用户口径(2026-09-29):
> 对现行现货市场中**所有品种**, 以**长周期与短周期共振**为信号 ——
> 长周期或短周期上**必有一个出现买①**, 另一个可以是买①/买②/买③。

共振的定义、窗口口径与合法性规则见 `chanbook/resonance.py` 的模块 docstring。
本工具只做一件事: **把当前(最新一根 1h K线)仍在有效期内的共振信号列出来**,
外加一份"长周期已成立、等短周期给点"的**观察名单**。

输出:
  * 文本报告(`--out`, 默认可不写): 当前信号 + 观察名单 + 数据新鲜度 + 池子诊断
  * **JSON**(`--json`, 默认 `out/resonance_scan.json`): 供 web 看板的"共振扫描"板块读取

为什么只回溯几十根:
  `Level` 是流式的, 第一根喂进去时会把之前全部历史一次性推进, 所以**结构状态与
  从第 0 根跑起完全一致**。但 `scan_points()` 是"边沿触发 + 去重"的 —— 循环起点那
  一次调用会把它**最近一次**检出的事件当成"当前事件"报出来(检出时间被错误地标成
  循环起点)。所以循环起点必须比"新鲜窗口 + 长周期有效窗口"更早, 让这类误标的
  事件自然过期(`SAFE_BARS` = 60 根 1h = 远大于 3 根 4h 的 12 小时窗口)。

运行::

  python tools/scan_resonance.py --jobs 10
  python tools/scan_resonance.py --limit 8 --fresh 5          # 冒烟
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                    # noqa: E402
from chanbook.resonance import RESO_TF, ResoScanner               # noqa: E402

SAFE_BARS = 60          # 循环起点至少提前这么多根短周期K线(见模块 docstring)
ALIVE_BARS = 6          # 最后一根距 NOW_REF 超过这么多根 ⇒ 判为"已停更"


# ------------------------------------------------------------------ 单标的
def scan_one(task):
    (sym, market, l_tf, s_tf, l_win, invalidate, lookback, t_end) = task
    rs = ResoScanner(market, sym, (l_tf, s_tf), l_win, invalidate).load()
    if rs.lkl is None or rs.skl is None or rs.skl.n == 0 or rs.lkl.n == 0:
        return {"sym": sym, "err": "无数据", "hits": [], "pending": []}
    if not rs.ok():
        return {"sym": sym, "err": f"样本不足 1h={rs.skl.n} {l_tf}={rs.lkl.n}",
                "hits": [], "pending": []}
    n = rs.skl.n
    start = max(0, n - (lookback + SAFE_BARS))
    hits: list[dict] = []
    for i in range(start, n):
        ts = int(rs.skl.ts[i])
        if t_end and ts > t_end:
            break
        rs.advance_to(ts)
        for h in rs.poll(ts, float(rs.skl.c[i])):
            hits.append(h.to_dict())
    last_ts = int(rs.skl.ts[n - 1])
    cur = float(rs.skl.c[n - 1])
    pend = []
    for q in rs.pending():
        px = float(q["price"])
        pend.append({
            "sym": sym, "l_kind": q["kind"], "l_ts": int(q["ts"]),
            "l_price": px, "l_trend": q["trend"], "l_beichi": q["beichi"],
            "age_h": (last_ts - int(q["ts"])) / D.MS_HOUR,
            "dist_pct": ((cur / px - 1.0) if px > 0 else 0.0),
        })
    return {"sym": sym, "err": "", "last_ts": last_ts, "last_close": cur,
            "n_1h": rs.skl.n, "n_l": rs.lkl.n, "hits": hits, "pending": pend}


def _mp(task):
    return scan_one(task)


# ------------------------------------------------------------------ 主流程
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="spot", choices=["spot"])
    ap.add_argument("--l-tf", default="")
    ap.add_argument("--s-tf", default="")
    ap.add_argument("--l-win", type=int, default=3,
                    help="长周期买点保持有效多少根长周期K线")
    ap.add_argument("--fresh", type=int, default=3,
                    help="只报最近这么多根短周期K线内的信号")
    ap.add_argument("--recent", type=int, default=30,
                    help="「近期信号」回溯天数(看板面板用)")
    ap.add_argument("--no-invalidate", action="store_true",
                    help="关掉「价格跌破长周期买点结构低点即作废」")
    ap.add_argument("--end", default="", help="截止月份 YYYY-MM(默认用全部数据)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="out/resonance_scan.txt")
    ap.add_argument("--json", default="out/resonance_scan.json")
    args = ap.parse_args()
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    market = args.market
    l_tf, s_tf = RESO_TF[market]
    if args.l_tf:
        l_tf = args.l_tf
    if args.s_tf:
        s_tf = args.s_tf
    t_end = 0
    if args.end:
        y, m = (int(x) for x in args.end.split("-"))
        m2, y2 = (1, y + 1) if m == 12 else (m + 1, y)
        from datetime import timezone
        t_end = int(datetime(y2, m2, 1, tzinfo=timezone.utc).timestamp() * 1000) - 1

    syms = D.symbols(market)
    if args.limit:
        syms = syms[:args.limit]
    # 回溯窗口: 覆盖"新鲜窗口"与"近期信号窗口"里更大的那个(单位=短周期根数)
    bars_per_hour = D.MS_HOUR / D.TF_MS[s_tf]
    lookback = max(int(args.fresh), int(args.recent * 24 * bars_per_hour))
    tasks = [(s, market, l_tf, s_tf, args.l_win, not args.no_invalidate,
              lookback, t_end) for s in syms]
    t0 = time.time()
    rows: list[dict] = []
    if args.jobs and args.jobs > 1:
        import multiprocessing as mp
        with mp.Pool(int(args.jobs)) as pool:
            for k, r in enumerate(pool.imap_unordered(_mp, tasks, chunksize=1)):
                rows.append(r)
                if (k + 1) % 20 == 0:
                    print(f"  {k + 1}/{len(tasks)}", file=sys.stderr, flush=True)
    else:
        for k, t in enumerate(tasks):
            rows.append(_mp(t))
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(tasks)}", file=sys.stderr, flush=True)
    rows.sort(key=lambda r: r["sym"])

    ok = [r for r in rows if not r["err"]]
    now_ref = max((r["last_ts"] for r in ok), default=0)
    alive_ms = ALIVE_BARS * D.TF_MS[s_tf]
    fresh_ms = args.fresh * D.TF_MS[s_tf]

    def is_alive(r):
        return bool(r["last_ts"]) and (now_ref - r["last_ts"] <= alive_ms)

    sig: list[dict] = []
    recent: list[dict] = []
    recent_ms = int(args.recent) * D.MS_DAY
    for r in ok:
        if not is_alive(r):
            continue
        for h in r["hits"]:
            age = now_ref - h["ts"]
            h["age_h"] = age / D.MS_HOUR
            h["last_close"] = r["last_close"]
            h["last_ts"] = r["last_ts"]
            if age <= recent_ms:
                recent.append(h)
            if age <= fresh_ms:
                sig.append(h)
    sig.sort(key=lambda h: (-h["ts"], h["sym"]))
    recent.sort(key=lambda h: (-h["ts"], h["sym"]))

    pend: list[dict] = []
    for r in ok:
        if not is_alive(r):
            continue
        for p in r["pending"]:
            p["last_close"] = r["last_close"]
            p["now_age_h"] = (now_ref - r["last_ts"]) / D.MS_HOUR
            pend.append(p)
    pend.sort(key=lambda p: (p["l_kind"] != "B1", -p["l_ts"], p["sym"]))

    n_alive = sum(1 for r in ok if is_alive(r))
    n_sig_sym = len({h["sym"] for h in sig})

    out: list[str] = []

    def emit(s: str = "") -> None:
        out.append(str(s))
        print(s, flush=True)

    emit("=" * 104)
    emit(f"现货多级别共振扫描   {l_tf}(长) + {s_tf}(短)   生成于 "
         f"{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    emit("=" * 104)
    emit(f"规则: 两级**同时**给出买点, 且**至少一侧是买①(B1)** ⇒ "
         f"B1+B1 / B1+B2 / B1+B3 / B2+B1 / B3+B1 五种组合")
    emit(f"口径: 长周期买点在 {args.l_win} 根 {l_tf}K线内有效"
         f"({'价格跌破结构低点即作废' if not args.no_invalidate else '只看时间窗口'})"
         f"; 只报最近 {args.fresh} 根 {s_tf} 内的信号")
    emit(f"池子: {len(syms)} 个归档标的 → 可用 {len(ok)} 个 → 仍在更新 {n_alive} 个")
    emit(f"参考时点(NOW_REF): {D.fmt(now_ref) if now_ref else '—'}")
    emit("")
    emit(f"■ 当前共振信号: {len(sig)} 条 / 覆盖 {n_sig_sym} 个品种")
    if not sig:
        emit("  (无)")
    else:
        emit(f"{'品种':<14}{'组合':<12}{'入场价':>13}{'结构低点':>13}"
             f"{'距低点':>9}{'长趋势':>8}{'短趋势':>8}{'年龄':>8}")
        for h in sig:
            lp = h["s_price"] or 0.0
            dist = (h["entry"] / lp - 1.0) if lp > 0 else 0.0
            emit(f"{h['sym']:<14}{h['label']:<12}{h['entry']:>13.6g}"
                 f"{lp:>13.6g}{dist * 100:>8.2f}%{h['l_trend']:>8}"
                 f"{h['s_trend']:>8}{h['age_h']:>7.1f}h")
    emit("")
    emit(f"■ 观察名单(长周期买点已成立、等短周期给点): {len(pend)} 个")
    if not pend:
        emit("  (无)")
    else:
        emit(f"{'品种':<14}{'长买点':<7}{'现价':>13}{'结构低点':>13}"
             f"{'距低点':>9}{'已过':>8}")
        for p in pend:
            emit(f"{p['sym']:<14}{p['l_kind']:<7}{p['last_close']:>13.6g}"
                 f"{p['l_price']:>13.6g}{p['dist_pct'] * 100:>8.2f}%"
                 f"{p['age_h']:>7.1f}h")
    emit("")
    emit(f"■ 近期信号(最近 {args.recent} 天): {len(recent)} 条 / "
         f"覆盖 {len({h['sym'] for h in recent})} 个品种")
    if not recent:
        emit("  (无)")
    else:
        from collections import Counter
        dist = Counter(h["combo"] for h in recent)
        emit("  组合分布: " + "  ".join(f"{k}={v}" for k, v in dist.most_common()))
        emit(f"{'品种':<14}{'组合':<12}{'触发时间':<18}{'入场价':>13}"
             f"{'结构低点':>13}{'距低点':>9}{'年龄':>9}")
        for h in recent[:40]:
            lp = h["s_price"] or h["l_price"] or 0.0
            distp = (h["entry"] / lp - 1.0) if lp > 0 else 0.0
            emit(f"{h['sym']:<14}{h['label']:<12}"
                 f"{D.fmt(h['ts']):<18}{h['entry']:>13.6g}{lp:>13.6g}"
                 f"{distp * 100:>8.2f}%{h['age_h'] / 24:>8.1f}d")
        if len(recent) > 40:
            emit(f"  … 另有 {len(recent) - 40} 条")
    emit("")
    skipped = [r for r in rows if r["err"]]
    if skipped:
        emit(f"■ 跳过 {len(skipped)} 个: "
             + "、".join(f"{r['sym']}({r['err']})" for r in skipped[:12])
             + (" …" if len(skipped) > 12 else ""))
        emit("")
    emit(f"# 耗时 {time.time() - t0:.0f}s; 多进程 {args.jobs}")

    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"\n文本报告 → {args.out}")
    except Exception as e:                                        # noqa: BLE001
        print(f"写盘失败 {args.out}: {e}")

    doc = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market": market, "l_tf": l_tf, "s_tf": s_tf,
        "l_win": args.l_win, "fresh_bars": args.fresh,
        "recent_days": args.recent,
        "invalidate": (not args.no_invalidate),
        "now_ref": now_ref, "now_ref_str": D.fmt(now_ref) if now_ref else "",
        "n_symbols": len(syms), "n_ok": len(ok), "n_alive": n_alive,
        "n_signals": len(sig), "n_signal_syms": n_sig_sym,
        "n_recent": len(recent),
        "signals": sig, "recent": recent, "pending": pend,
        "skipped": [{"sym": r["sym"], "err": r["err"]} for r in skipped],
    }
    try:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False, indent=1)
        print(f"JSON    → {args.json}")
    except Exception as e:                                        # noqa: BLE001
        print(f"写 JSON 失败 {args.json}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
