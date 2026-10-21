#!/usr/bin/env python
"""成交明细缓存比对 —— 判定两轮回测是否产出**同一批交易**。

⚠ 为什么不能用 `md5sum` 直接比文件: `mp.Pool` 的完成顺序取决于调度, 成交记录的
  **输出顺序**会在两次运行之间变化(实测: 同一批标的、字节数完全相同的两个文件,
  md5 却不同)。必须先按 (sym, entry_ts, side, kind, reason) 排序再比集合。

用法:
  $PY tools/cmp_cache.py A.json B.json                 # 严格全字段比对
  $PY tools/cmp_cache.py A.json B.json --drop feats    # 忽略新增的观测字段
  $PY tools/cmp_cache.py A.json B.json --drop feats,confirm

退出码: 0 = 完全一致, 1 = 有差异。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

KEY = ("sym", "entry_ts", "side", "kind", "reason")


def load(path: str) -> list:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("tr", [])


def norm(rows: list, drop: set) -> list:
    out = []
    for r in rows:
        out.append({k: v for k, v in r.items() if k not in drop})
    return sorted(out, key=lambda r: tuple(str(r.get(k, "")) for k in KEY))


def digest(rows: list) -> str:
    blob = json.dumps(rows, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--drop", default="", help="逗号分隔的忽略字段")
    ap.add_argument("--show", type=int, default=2, help="打印前几条差异")
    a = ap.parse_args()
    drop = {x.strip() for x in a.drop.split(",") if x.strip()}

    ra, rb = load(a.a), load(a.b)
    na, nb = norm(ra, drop), norm(rb, drop)
    da, db = digest(na), digest(nb)
    print(f"A: {a.a}\n   笔数 {len(ra)}  排序后集合 md5 {da}")
    print(f"B: {a.b}\n   笔数 {len(rb)}  排序后集合 md5 {db}")
    if drop:
        print(f"忽略字段: {sorted(drop)}")
    if da == db:
        print("\n=> 一致: 两轮产出同一批交易(逐字段相同)。")
        return 0
    print("\n=> **不一致**")
    if len(na) != len(nb):
        print(f"   笔数不同: {len(na)} vs {len(nb)}")
    ka = {(r.get("sym"), r.get("entry_ts")) for r in na}
    kb = {(r.get("sym"), r.get("entry_ts")) for r in nb}
    only_a, only_b = sorted(ka - kb)[:8], sorted(kb - ka)[:8]
    if only_a:
        print(f"   仅 A 有 {len(ka - kb)} 条, 前几条: {only_a}")
    if only_b:
        print(f"   仅 B 有 {len(kb - ka)} 条, 前几条: {only_b}")
    shown = 0
    for x, y in zip(na, nb):
        if x != y:
            diff = {k: (x.get(k), y.get(k)) for k in set(x) | set(y)
                    if x.get(k) != y.get(k)}
            print(f"   首条差异: {json.dumps(diff, ensure_ascii=False)[:400]}")
            shown += 1
            if shown >= a.show:
                break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
