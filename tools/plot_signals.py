#!/usr/bin/env python3
"""在 K 线图上可视化 chanbook 的缠论结构与买卖点。

输出一张 PNG：蜡烛图 + 笔连线 + 中枢矩形 + 三类买点(B1/B2/B3) + 三类卖点(S1/S2/S3)。

用法:
    python tools/plot_signals.py --symbol BTCUSDT --tf 1h --bars 500 --out out/btc_1h.png
    python tools/plot_signals.py --symbol ETHUSDT --tf 4h --bars 300

说明:
    - 逐 bar 复刻 `engine.run_symbol` 的推进方式(`LevelSet.advance`), 但**不模拟
      持仓/成交** —— 展示的是结构本身: 每 bar 无条件扫描全部买卖点。
    - 买点: `op.scan_points()` 事件驱动(向下笔冻结时触发, 内部有去重锁)。
    - 卖点: `SellScanner.scan` 自带"同一冻结笔只扫一次"的去重锁(见 engine.sell_hit
      注释), 每 bar 无条件调用不会重复报点。
    - 标记放在**结构位置**(触发笔端点分型的极值处), 而不是检测发生的 bar,
      这样直观对应缠论原文"底分型买、顶分型卖"。
"""

from __future__ import annotations

import argparse
import bisect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chanbook import data as D                                  # noqa: E402
from chanbook.engine import _sell_scanner_of, WARMUP            # noqa: E402
from chanbook.level import LevelSet                            # noqa: E402
from chanbook.strategy import BookCfg                          # noqa: E402
from chanbook.points import B1, B2, B3                         # noqa: E402
from chanbook.sell import S1, S2, S3                           # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                 # noqa: E402
from matplotlib.patches import Rectangle                        # noqa: E402

# 中文字体(避免买卖点/中枢等中文标注显示为方框)
for _f in ("Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"):
    try:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        continue

# 买卖点颜色(遵循 A 股惯例: 红=买/涨, 绿=卖/跌)
BUY_COLORS = {"B1": "#d32f2f", "B2": "#ff7043", "B3": "#e53935"}
SELL_COLORS = {"S1": "#2e7d32", "S2": "#66bb6a", "S3": "#1b5e20"}

# 操作级别 → (趋势级别, 操作级别, 确认级别) 规格。
# 默认 spot = ("4h", "1h", "15m"); 4h 操作级别上移一档: 趋势 1d / 确认 1h。
TF_SPEC = {
    "1h": None,                      # 用 LEVEL_TF 默认
    "4h": ("1d", "4h", "1h"),
}


def _collect(market: str, sym: str, cfg: BookCfg, spec=None):
    """逐 bar 推进, 收集**全部**买卖点 + 最终结构快照(不模拟持仓/成交)。

    - 买点: `op.scan_points()` 事件驱动(向下笔冻结时触发, 内部有去重锁),
      每 bar 无条件调用即可拿到全部买点。
    - 卖点: `SellScanner.scan` 自带"同一冻结笔只扫一次"的去重锁, 每 bar
      无条件调用不会重复报点(与 `engine.sell_hit` 的注释一致)。
    - 标记位置: 用触发笔(`p.leg_i`)端点分型的时间戳 + 结构价格 —— 即缠论
      原文里的"底分型/顶分型极值处"。
    """
    ls = LevelSet(market, sym, spec=spec, scanner_kw=cfg.scanner_kw(),
                  use_confirm=bool(cfg.confirm_gate or cfg.record_confirm),
                  op_extra=cfg.op_extra()).load()
    if not ls.ok():
        return None
    op = ls.op
    sscan = _sell_scanner_of(cfg)

    buys: list[tuple[int, object]] = []   # (笔端点ts, Point)
    sells: list[tuple[int, object]] = []  # (笔端点ts, Point)

    for i in range(WARMUP, op.kl.n):
        bar_ts = int(op.kl.ts[i])
        close_ts = bar_ts + op.tf_ms
        ls.advance(close_ts)

        def _xy(p):
            # 触发笔的端点分型: x=端点时间, y=结构极值(Point.price 即该极值)
            bi = op.bib.bis[p.leg_i]
            return int(bi.end_ts), p

        for p in op.scan_points():
            buys.append(_xy(p))
        if sscan is not None:
            for p in sscan.scan(op.bib.bis, op.zss_view(), op.macd,
                                op.bib.n_final):
                sells.append(_xy(p))

    return {"kl": op.kl, "buys": buys, "sells": sells,
            "bis": list(op.bib.bis), "zss": list(op.zss_view())}


def _ts_to_i(kl, ts: int) -> int:
    """把毫秒时间戳对齐到 kl 的 bar 下标(右边界内)。"""
    return bisect.bisect_right(kl.ts, ts) - 1


def plot(sym: str, tf: str, data: dict, bars: int, out: str) -> None:
    kl = data["kl"]
    n = kl.n
    # 画最近 bars 根
    i0 = max(0, n - bars)
    ts = list(kl.ts[i0:])
    o = list(kl.o[i0:])
    h = list(kl.h[i0:])
    l = list(kl.l[i0:])
    c = list(kl.c[i0:])
    m = len(ts)

    # x 轴用整数下标(避免日期定位的时区/稀疏问题), 刻度再转日期
    x = list(range(m))

    fig, ax = plt.subplots(figsize=(20, 10), dpi=110)
    fig.suptitle(f"{sym}  {tf}   缠论结构 + 买卖点  (最近 {m} 根)",
                 fontsize=16, fontweight="bold")

    # ---- 蜡烛 ----
    up = [c[k] >= o[k] for k in range(m)]
    ax.vlines(x, l, h, color="#888888", linewidth=0.6, zorder=1)
    from matplotlib.patches import Rectangle as R
    for k in range(m):
        col = "#d32f2f" if up[k] else "#2e7d32"
        body_lo = min(o[k], c[k])
        body_hi = max(o[k], c[k])
        hh = body_hi - body_lo
        if hh == 0:
            hh = max(h[k] - l[k], 1e-9) * 0.01
        ax.add_patch(R((k - 0.3, body_lo), 0.6, hh, facecolor=col,
                       edgecolor=col, zorder=3))

    # ---- 笔连线 ----
    for b in data["bis"]:
        i_a = _ts_to_i(kl, b.start_ts) - i0
        i_b = _ts_to_i(kl, b.end_ts) - i0
        if i_a < 0 or i_b < 0 or i_a >= m or i_b >= m:
            continue
        ax.plot([i_a, i_b], [b.start_price, b.end_price],
                color="#3f51b5", linewidth=1.2, alpha=0.75, zorder=2)
        ax.plot(i_a, b.start_price, marker="o", ms=3, color="#1a237e", zorder=4)
        ax.plot(i_b, b.end_price, marker="o", ms=3, color="#1a237e", zorder=4)

    # ---- 中枢矩形 ----
    for z in data["zss"]:
        i_a = max(0, _ts_to_i(kl, z.start_ts) - i0)
        i_b = min(m - 1, _ts_to_i(kl, z.end_ts) - i0)
        if i_b < 0 or i_a >= m or i_b <= i_a:
            continue
        ax.add_patch(Rectangle((i_a - 0.4, z.zd), i_b - i_a + 0.8, z.zg - z.zd,
                               facecolor="#ffb300", alpha=0.18,
                               edgecolor="#ff8f00", linewidth=1.0,
                               linestyle="--", zorder=1))
        ax.text(i_a, z.zg, f"ZS{z.i}", fontsize=7, color="#e65100",
                va="bottom", ha="left", zorder=5)

    # ---- 买点(标记在底分型极值下方) ----
    for ts_pt, p in data["buys"]:
        i = _ts_to_i(kl, ts_pt) - i0
        if i < 0 or i >= m:
            continue
        col = BUY_COLORS.get(p.kind, "#d32f2f")
        px = p.price
        ax.scatter(i, px, marker="^", s=130, color=col, edgecolor="white",
                   linewidth=1.0, zorder=6)
        ax.annotate(p.kind, (i, px), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=8,
                    fontweight="bold", color=col, zorder=7)

    # ---- 卖点(标记在顶分型极值上方) ----
    for ts_pt, p in data["sells"]:
        i = _ts_to_i(kl, ts_pt) - i0
        if i < 0 or i >= m:
            continue
        col = SELL_COLORS.get(p.kind, "#2e7d32")
        px = p.price
        ax.scatter(i, px, marker="v", s=130, color=col, edgecolor="white",
                   linewidth=1.0, zorder=6)
        ax.annotate(p.kind, (i, px), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8,
                    fontweight="bold", color=col, zorder=7)

    # ---- 刻度转日期 ----
    import datetime as dt
    step = max(1, m // 12)
    tick_pos = list(range(0, m, step))
    tick_lbl = [dt.datetime.fromtimestamp(ts[p] / 1000, dt.timezone.utc)
                .strftime("%Y-%m-%d") for p in tick_pos]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_lbl, rotation=45, fontsize=8)
    ax.set_ylabel("价格", fontsize=11)
    ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.5)
    ax.set_xlim(-1, m)

    # ---- 图例 ----
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="^", color="w", markerfacecolor="#d32f2f",
               ms=12, label="B1 买①背驰"),
        Line2D([], [], marker="^", color="w", markerfacecolor="#ff7043",
               ms=12, label="B2 买②确认"),
        Line2D([], [], marker="^", color="w", markerfacecolor="#e53935",
               ms=12, label="B3 买③三买"),
        Line2D([], [], marker="v", color="w", markerfacecolor="#2e7d32",
               ms=12, label="S1 卖①背驰"),
        Line2D([], [], marker="v", color="w", markerfacecolor="#66bb6a",
               ms=12, label="S2 卖②反抽"),
        Line2D([], [], marker="v", color="w", markerfacecolor="#1b5e20",
               ms=12, label="S3 卖③三卖"),
        Line2D([], [], color="#3f51b5", lw=1.5, label="笔"),
        Rectangle((0, 0), 1, 1, facecolor="#ffb300", alpha=0.3,
                  edgecolor="#ff8f00", label="中枢"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, ncol=2,
              framealpha=0.9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存: {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--tf", default="1h", choices=["1h", "4h"],
                    help="操作级别: 1h(默认, 趋势4h) 或 4h(趋势1d)")
    ap.add_argument("--bars", type=int, default=500, help="画最近 N 根")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    spec = TF_SPEC.get(args.tf)   # None → 用 LEVEL_TF 默认("4h","1h","15m")

    # 展示配置: 三类买卖点全开(生产默认 stop + S1 离场口径不影响结构点扫描)
    cfg = BookCfg(market="spot", kinds=(B1, B2, B3), sell_exit=(S1, S2, S3))
    data = _collect("spot", args.symbol, cfg, spec=spec)
    if data is None:
        print(f"错误: {args.symbol} 数据不足或加载失败", file=sys.stderr)
        return 1

    from collections import Counter
    cb = Counter(p.kind for _, p in data["buys"])
    cs = Counter(p.kind for _, p in data["sells"])
    print(f"{args.symbol} {args.tf} 全历史: "
          f"买点 {dict(sorted(cb.items()))} 共 {sum(cb.values())} 个, "
          f"卖点 {dict(sorted(cs.items()))} 共 {sum(cs.values())} 个, "
          f"笔 {len(data['bis'])} 根, 中枢 {len(data['zss'])} 个")

    out = args.out or f"out/{args.symbol}_{args.tf}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    plot(args.symbol, args.tf, data, args.bars, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
