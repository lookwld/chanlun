#!/usr/bin/env python
"""字段目录 —— 数据集有哪些列、各覆盖多少、什么量纲、属于哪一组。

与 `dataset_report.py` 的分工:
  * `dataset_report.py` 看**内容**(标签分布 / 单特征区分度);
  * `field_catalog.py`    看**目录**(字段是否齐备 / 覆盖率 / 量纲 / 分组)。

为什么要单独做: 手工维护字段清单必然与实际输出漂移(本轮就踩过一次 ——
`amp_a` 归一化后改名 `amp_a_pct`, 而体检脚本还在找旧名, 报"缺失 100%",
看起来像 bug 其实只是名字没对上)。这里**从数据本身反推字段全集**, 不依赖任何清单。

用法:
  $PY tools/field_catalog.py out/dataset/um_all.jsonl out/dataset/spot_all.jsonl
  $PY tools/field_catalog.py out/dataset/um_all.jsonl --json out/dataset/fields.json
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

# 字段 → (分组, 说明)。**未知字段会落在 "未归类" 并高亮**, 提示清单需要更新。
CATALOG: dict[str, tuple[str, str]] = {
    # --- 元数据 ---
    "sym": ("元数据", "标的"),
    "market": ("元数据", "um / spot"),
    "source": ("元数据", "变体名(all)"),
    "entry_ts": ("元数据", "入场毫秒(UTC)"),
    "entry_time": ("元数据", "入场时间可读"),
    "split": ("元数据", "train/valid(按时间切, 非随机)"),
    "side": ("元数据", "恒为 long"),
    # --- 标签(离场后才知道, 绝不能入模) ---
    "exit_ts": ("标签", "离场毫秒"),
    "pnl_pct": ("标签", "收益率"),
    "reason": ("标签", "离场方式"),
    "hold_min": ("标签", "持仓分钟"),
    "R": ("标签", "pnl_pct / risk_pct"),
    "y_stop": ("标签", "是否被止损(主标签)"),
    "y_win": ("标签", "是否盈利"),
    "censored": ("标签", "期末强平(截断)"),
    # --- 通用特征(入场时的风控量) ---
    "risk_pct": ("通用", "|entry-stop0|/entry —— 止损距离"),
    "notional": ("通用", "名义额U(按买点分档: 一买减半)"),
    "atr_pct": ("通用", "ATR/entry"),
    "gap_entry_struct": ("通用", "(entry-结构位)/entry —— 入场价比结构位高多少"),
    # --- 类别特征 ---
    "kind": ("类别", "B1/B2/B3"),
    "trend": ("类别", "操作级别走势: up/down/range/pending"),
    "regime": ("类别", "BTC 大级别: bull/bear/range"),
    "counter": ("类别", "是否逆势(一买=1)"),
    "confirm_ok": ("类别", "区间套结论: 1成立/0不成立/-1不适用"),
    "confirm": ("类别", "区间套文字结论(前120字)"),
    "b1_path": ("类别", "一买来源: zs(中枢路径)/macd(无中枢)/None"),
    # --- B1 专属: 背驰四判据 + 段几何 ---
    "beichi_score": ("B1背驰", "四判据命中数 0~4 —— 实测只取 3/4, 近乎无方差"),
    "beichi_new_extreme": ("B1背驰", "是否创新低 —— **恒为 1**(一买定义要求创新低)"),
    "ratio_area": ("B1背驰", "柱面积比 a/c"),
    "ratio_per": ("B1背驰", "单位力度比"),
    "ratio_peak": ("B1背驰", "DIFF 峰值比"),
    "ratio_hist": ("B1背驰", "HIST 峰值比"),
    "bars_a": ("B1背驰", "参照段 MACD 柱数"),
    "bars_c": ("B1背驰", "离开段 MACD 柱数"),
    "b1_lag": ("B1背驰", "离开段最低点距入场几根K"),
    "amp_a_pct": ("B1背驰", "参照段幅度/entry"),
    "amp_c_pct": ("B1背驰", "离开段幅度/entry"),
    "amp_ratio": ("B1背驰", "amp_c/amp_a"),
    "zs_break": ("B1背驰", "(离开段低点-ZD)/ZD —— 破中枢多深"),
    "trend_beichi": ("B1背驰", "是否趋势背驰(否则盘整背驰)"),
    # --- B1 无中枢路径专属 ---
    "b1_lag_bi": ("B1无中枢", "距上一笔末端几笔 —— **恒为 2**(只回看最近一笔)"),
    "retest_up_pct": ("B1无中枢", "回抽幅度/entry"),
    "b1_had_zs": ("B1无中枢", "当时是否已有中枢"),
    # --- B3 专属 ---
    "zs_height_pct": ("B3", "中枢高度/entry"),
    "above_zg_pct": ("B3", "(回试低点-ZG)/entry —— 离上沿多远"),
    "exit_break_pct": ("B3", "(离开段高点-ZG)/entry"),
    "exit_leg_amp_pct": ("B3", "离开段幅度/entry"),
    "pull_bi": ("B3", "回试笔数 —— **恒为 1**(配置 pull_bi=1)"),
    "b3_zs_age": ("B3", "回试完成距中枢形成几根K —— **恒为 1**"),
    # --- 日线上下文(只用入场前已收盘) ---
    "d_close": ("日线", "前一日收盘 —— 绝对价格, 跨标的不可比, 勿直接入模"),
    "d_ma20_dev": ("日线", "收盘/MA20-1 —— 乖离"),
    "d_ma60_dev": ("日线", "收盘/MA60-1"),
    "d_atr_pct": ("日线", "日线 ATR14/收盘"),
    "d_atr_rank": ("日线", "ATR% 的历史分位 0~1"),
    "d_ret20": ("日线", "20 日收益"),
    "d_ma20_slope": ("日线", "MA20 10 日斜率"),
}

LEAK = {"exit_ts", "pnl_pct", "reason", "hold_min", "R", "y_stop", "y_win",
        "censored"}


def scan(path: str) -> tuple[list, dict]:
    keys: set[str] = set()
    cnt: collections.Counter = collections.Counter()
    vals: dict[str, list] = collections.defaultdict(list)
    n = 0
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            if not ln.strip():
                continue
            d = json.loads(ln)
            n += 1
            keys |= set(d)
            for k, v in d.items():
                if v is not None:
                    cnt[k] += 1
                    if len(vals[k]) < 4000 and isinstance(v, (int, float)) \
                            and not isinstance(v, bool):
                        vals[k].append(float(v))
    return sorted(keys), {"n": n, "cnt": cnt, "vals": vals}


def _fmt(ok: list) -> str:
    if not ok:
        return "-"
    uniq = sorted(set(ok))
    if len(uniq) == 1:                       # 常量: 零方差, 入模只会添噪声
        return f"**常量! 恒为 {uniq[0]:.6g} —— 无区分度, 不可入模**"
    if len(uniq) <= 3 and all(float(x).is_integer() for x in uniq):
        return "枚举 " + ",".join(str(int(x)) for x in uniq) + " (弱)"
    if len(uniq) == len(ok) and len(uniq) <= 8:            # 每样本都不同 => 类别型
        return "枚举 " + ",".join(repr(x) for x in uniq)[:46]
    a = np.asarray(ok, dtype=float)
    return (f"均值{a.mean():.4g} 中位{np.median(a):.4g} "
            f"[{a.min():.4g},{a.max():.4g}] 唯一{len(uniq)}")


def catalog(path: str) -> dict:
    keys, st = scan(path)
    n = st["n"]
    print(f"\n{'=' * 100}")
    print(f"## {os.path.basename(path)}   样本 {n}   字段 {len(keys)}")
    print("=" * 100)
    print(f"{'字段':<22}{'覆盖率':>8}  {'分组':<10}{'泄漏':<6}{'说明 / 取值'}")
    print("-" * 100)
    out = {}
    unknown = []
    const: list[str] = []
    for grp in ("元数据", "通用", "类别", "B1背驰", "B1无中枢", "B3", "日线", "标签",
                "未归类"):
        hit = [k for k in keys if CATALOG.get(k, ("未归类", ""))[0] == grp]
        if not hit:
            continue
        print(f"[{grp}]")
        for k in sorted(hit):
            desc = CATALOG.get(k, ("未归类", "**清单未收录**"))[1]
            if k not in CATALOG:
                unknown.append(k)
            c = st["cnt"][k] / n
            v = st["vals"][k]
            if v:
                desc = f"{desc} | {_fmt(v)}"
                if len(set(v)) == 1:
                    const.append(k)
            print(f"  {k:<20}{c:>7.1%}  {grp:<10}"
                  f"{('是' if k in LEAK else ''):<6}{desc}")
            out[k] = {"cover": round(c, 4), "group": grp, "leak": k in LEAK,
                      "desc": CATALOG.get(k, ("未归类", ""))[1]}
    if unknown:
        print(f"\n⚠ 未归类字段 {unknown} —— 请补进 field_catalog.CATALOG, "
              f"否则字典会漏掉它们。")
    if const:
        print(f"\n⚠ 常量字段(零方差, 建模前必须剔除): {const}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--json", default="", help="把目录落成 JSON(供 DATASET.md 引用)")
    a = ap.parse_args()
    full = {}
    for p in a.paths:
        full[os.path.basename(p)] = catalog(p)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(full, fh, ensure_ascii=False, indent=1)
        print(f"\n目录已写入 {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
