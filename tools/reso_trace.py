"""单品种共振溯源 —— 回答「这个品种为什么没出信号 / 那天到底发生了什么」。

扫描器(`scan_resonance.py`)只给**结果**(信号 / 观察名单), 但当你盯着某个品种问
「我看到的价和你说的不一样」「它到底触发了没」时, 需要的是**逐 bar 的经过**:
长周期买点哪天成立、锚在哪、窗口什么时候到期、窗内短周期有没有给点、
其间有没有跌破结构低点被作废。

用法(注意: 想扫实时数据要先刷快照):
  python tools/refresh_archive.py --market spot
  CHANBOOK_ARCH=$PWD/out/arch python tools/reso_trace.py --symbol NOMUSDT --since 2026-08-20

对齐口径见 `chanbook/resonance.py`: 长周期只负责**开窗口**(l_win 根长周期 K 线),
短周期负责**给触发**; 窗口用「检测时刻」(最后一根已收盘 bar) 计时, 不用笔端点,
避免偷看未来。
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from chanbook import data as D                     # noqa: E402
from chanbook.resonance import DEFAULT_L_WIN, ResoScanner   # noqa: E402

MS_HOUR = D.MS_HOUR
MODES = ("reso", "s_only", "l_only")


def _ts(s: str) -> int:
    """'2026-08-20' 或 '2026-08-20 14:00' -> 北京时间的毫秒戳。"""
    from datetime import datetime, timedelta, timezone
    tz = timezone(timedelta(hours=8))
    fmt = "%Y-%m-%d %H:%M" if len(s) > 10 else "%Y-%m-%d"
    return int(datetime.strptime(s, fmt).replace(tzinfo=tz).timestamp() * 1000)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--market", default="spot", choices=["spot", "um"])
    ap.add_argument("--l-tf", default="")
    ap.add_argument("--s-tf", default="")
    ap.add_argument("--l-win", type=int, default=DEFAULT_L_WIN)
    ap.add_argument("--no-invalidate", action="store_true")
    ap.add_argument("--since", default="", help="起看时间(北京时间), 默认数据末段 60 天")
    ap.add_argument("--all-events", action="store_true",
                    help="连没命中的短周期买点也逐条打印")
    args = ap.parse_args()

    pair = (args.l_tf, args.s_tf) if (args.l_tf and args.s_tf) else None
    rs = ResoScanner(args.market, args.symbol, tf_pair=pair, l_win=args.l_win,
                     invalidate=(not args.no_invalidate))
    try:
        rs.load()
    except Exception as e:                                  # noqa: BLE001
        print(f"装载失败 {args.symbol}: {e}", file=sys.stderr)
        return 1
    if not rs.skl.n or not rs.lkl.n:
        print(f"{args.symbol}: 数据不足 (1h {rs.skl.n} 根 / {rs.l_tf} {rs.lkl.n} 根)")
        return 1

    args.l_tf, args.s_tf = rs.l_tf, rs.s_tf            # 回填实际周期(默认取自 RESO_TF)

    win_h = args.l_win * rs.long.tf_ms / MS_HOUR
    end_ts = int(rs.skl.ts[-1])
    start_ts = _ts(args.since) if args.since else end_ts - 60 * 24 * MS_HOUR
    i0 = next((i for i in range(rs.skl.n) if int(rs.skl.ts[i]) >= start_ts), rs.skl.n)

    print("=" * 96)
    print(f"共振溯源  {args.symbol}  {args.l_tf}(长) + {args.s_tf}(短)   "
          f"l_win={args.l_win} ({win_h:.1f}h)   作废线={'开' if rs.invalidate else '关'}")
    print(f"数据: 1h {rs.skl.n} 根 / {args.l_tf} {rs.lkl.n} 根   末端 {D.fmt(end_ts)} (北京)")
    print(f"区间: {D.fmt(int(rs.skl.ts[i0]))} ~ {D.fmt(end_ts)}")
    print("=" * 96)
    print(f"{'时间(北京)':<18}{'1h收盘':>11}  事件")
    print("-" * 96)

    n_lg = n_sh = 0
    hits = {m: [] for m in MODES}
    life: list[dict] = []          # 长周期买点生命周期
    held: dict[int, dict] = {}
    lows: dict[int, float] = {}
    for i in range(i0, rs.skl.n):
        ts = int(rs.skl.ts[i])
        c = float(rs.skl.c[i])
        rs.advance_to(ts)
        ev = rs.step(ts, c)
        lg = ev.get("fresh_long") or []
        sh = ev.get("short") or []
        n_lg += len(lg)
        n_sh += len(sh)
        lines = []
        fired_this_bar = False
        for q in lg:
            d = {"ts": int(q["ts"]), "kind": q["kind"], "price": float(q["price"]),
                 "deadline": int(q["ts"]) + rs.l_win * rs.long.tf_ms,
                 "fired": "", "min_close": float("inf")}
            life.append(d)
            held[id(q)] = d
            lines.append(f"● 长周期 {q['kind']}@{args.l_tf} 成立   锚点 {q['price']:.6g}"
                         f"   窗口 {D.fmt(d['deadline'])} 到期")
        for p in sh:
            lines.append(f"○ 短周期 {p.kind}@{args.s_tf}   结构价 {p.price:.6g}")
        for m in MODES:
            for h in rs.decide(ts, c, ev, m):
                hits[m].append(h)
                if m == "reso":
                    fired_this_bar = True
                    lines.append(f"★ 共振命中  {h.combo}   入场 {h.entry:.6g} "
                                 f"(长 {h.l_kind}@{args.l_tf} / 短 {h.s_kind}@{args.s_tf})")
                    for d in life:
                        if d["ts"] == int(h.l_ts) and not d["fired"]:
                            d["fired"] = f"{h.combo} @ {D.fmt(ts)}"
        # 跟踪每个存活买点的窗内最低收盘 + 作废/到期
        for d in life:
            if d["deadline"] >= ts >= d["ts"]:
                d["min_close"] = min(d["min_close"], c)
        still = {id(q) for q in rs.pending()}
        for k in list(held):
            if k not in still:
                d = held.pop(k)
                if not d["fired"]:
                    why = "窗内跌破锚点" if (rs.invalidate and
                                             d["min_close"] < d["price"] - 1e-12) else "窗口到期"
                    lines.append(f"   └ 未触发: {why} (窗内最低收盘 "
                                 f"{d['min_close']:.6g} vs 锚点 {d['price']:.6g})")
        if lines and (lg or sh or fired_this_bar or args.all_events):
            for j, ln in enumerate(lines):
                pre = f"{D.fmt(ts):<18}{c:>11.6g}  " if j == 0 else " " * 31
                print(f"{pre}{ln}")

    print("-" * 96)
    pend = rs.pending()
    if pend:
        print(f"■ 收盘时仍在有效窗口内: {len(pend)} 个")
        for q in pend:
            print(f"   {q['kind']}@{args.l_tf} 锚点 {q['price']:.6g} "
                  f"({D.fmt(int(q['ts']))} 成立, 剩余 "
                  f"{(int(q['ts']) + rs.l_win * rs.long.tf_ms - end_ts) / MS_HOUR:.1f}h)")
    else:
        print("■ 收盘时无有效窗口内的长周期买点")
    print(f"■ 区间内: 长周期买点 {n_lg} 个 · 短周期买点 {n_sh} 个 · "
          + " · ".join(f"{m} 命中 {len(hits[m])}" for m in MODES))
    tail = [float(rs.skl.c[i]) for i in range(i0, rs.skl.n)]
    if tail:
        print(f"■ 区间价格: 收盘 最低 {min(tail):.6g} / 最高 {max(tail):.6g} / "
              f"末根 {tail[-1]:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
