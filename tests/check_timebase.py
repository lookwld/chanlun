"""时间基准静态闸门 —— 防止"多级别各加自己的周期"这类**偷看未来**的 bug 再次发生。

运行: python tests/check_timebase.py

## 为什么要有这一份

同一个 bug 在本项目**已经犯过两次**:

  * 2026-09-29 共振模块 `resonance.py:advance_to()`:
    `long.advance(bar_ts + long.tf_ms)` —— 4h/1h 上实测**领先 3 小时**,
    直接把上一轮"共振是负增量"的结论污染成相反方向;
  * 同日复核发现 `tools/backtest_b12_tf.py:run_symbol()` 一模一样:
    `lv.advance(bar_ts + lv.tf_ms)` 与 `tlv.advance(bar_ts + tlv.tf_ms)` 并存
    —— 趋势级别(4h)被推进到"开盘于 bar_ts 的 4h 已收盘", 同样偷看未来。
    它污染的是 1h/2h B1+B2 那轮的**方向闸门**(B2 准入)。

两次都不是"写错一个数", 而是**没意识到 `advance()` 的契约**。

## 契约(唯一正确的写法)

`Level.advance(close_ts)` 的语义是「喂入所有**收盘时刻 ≤ close_ts** 的K线」
(`data.Kl.end_at` 会自己减去本周期长度)。因此:

    close_ts 必须是**当前墙钟时刻**, 由**最细的那个级别**给出;
    同一时刻在用的**所有级别, 必须传同一个 close_ts**。

  ✅ 对: `now = bar_ts + op.tf_ms; lv.advance(now); tlv.advance(now)`
  ✅ 对: `LevelSet.advance(close_ts)` —— 内部三级共用同一个入参
  ❌ 错: `lv.advance(bar_ts + lv.tf_ms); tlv.advance(bar_ts + tlv.tf_ms)`
         (tlv 比 op 粗时, 它看到了一根尚未收盘的自己的K线)

## 本自检做什么

1. **正控**: 先拿一段"故意写错"的源码喂给检查器, 必须被抓出来(否则闸门是空跑的);
2. **静态扫描**: 遍历 `chanbook/` + `tools/` + `tests/` 全部 `.py`, 用 AST 找出
   每个作用域(模块/函数)里所有 `.advance(...)` / `.advance_to(...)` 调用,
   **要求它们的实参表达式完全相同**。不同 ⇒ 报警。
   (需要豁免时在该函数任意一行写 `# noqa: TIMEBASE`。)
3. **运行时不变式**: 真跑一段 `LevelSet`, 逐根断言"每一级已喂入的最后一根K线
   的收盘时刻 ≤ 当前时刻"。这是上面那个 bug 的直接反命题。
"""

from __future__ import annotations

import ast
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from chanbook import data as D                                  # noqa: E402
from chanbook.level import LevelSet                             # noqa: E402

N_PASS = 0
FAILS: list[str] = []
SKIP: list[str] = []

WATCH = {"advance", "advance_to"}          # 会推进"当前时刻"的 API
SCAN_DIRS = ("chanbook", "tools", "tests")
SKIP_DIRS = {"__pycache__", "out", "arch", "data"}


def ck(cond: bool, msg: str) -> None:
    global N_PASS
    if cond:
        N_PASS += 1
    else:
        FAILS.append(msg)
        print(f"  ✗ {msg}")


def skip(msg: str) -> None:
    SKIP.append(msg)
    print(f"  - {msg}")


# --------------------------------------------------------------------- 检查器
def _clock_calls(node: ast.AST) -> list[ast.Call]:
    """收集本作用域内(不进入嵌套 def/lambda)的时钟调用。"""
    out: list[ast.Call] = []
    nested = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)

    def rec(nd: ast.AST) -> None:
        for ch in ast.iter_child_nodes(nd):
            if isinstance(ch, nested):
                continue
            if (isinstance(ch, ast.Call) and isinstance(ch.func, ast.Attribute)
                    and ch.func.attr in WATCH):
                out.append(ch)
            rec(ch)

    rec(node)
    return out


def _scopes(tree: ast.AST) -> list[ast.AST]:
    """模块本身 + 每个函数, 各算一个作用域(嵌套 def 由内层自己负责)。"""
    return [tree, *[n for n in ast.walk(tree)
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]]


def all_clock_calls(tree: ast.AST) -> list[ast.Call]:
    return [c for s in _scopes(tree) for c in _clock_calls(s)]


def scan_source(src: str, fname: str = "<src>") -> list[str]:
    """返回违规描述列表(空 = 干净)。"""
    tree = ast.parse(src)
    lines = src.splitlines()
    bad: list[str] = []
    for scope in _scopes(tree):
        calls = _clock_calls(scope)
        if len(calls) < 2:
            continue
        args = {ast.unparse(c.args[0]) if c.args else "<无参>" for c in calls}
        if len(args) < 2:
            continue
        lo = min(c.lineno for c in calls)
        hi = max(getattr(c, "end_lineno", c.lineno) for c in calls)
        body = "\n".join(lines[lo - 1:hi])
        if "noqa: TIMEBASE" in body:
            continue
        name = getattr(scope, "name", "<module>")
        bad.append(f"{os.path.basename(fname)}:{lo} 作用域 {name}() 内存在 "
                   f"{len(args)} 个不同的时钟实参: {sorted(args)}")
    return bad


# --------------------------------------------------------------------- 1. 正控
def t_positive_control() -> None:
    print("[1] 正控 —— 故意写错的源码必须被抓出来")
    wrong = (
        "def run(kl, lv, tlv):\n"
        "    for i in range(kl.n):\n"
        "        ts = kl.ts[i]\n"
        "        lv.advance(ts + lv.tf_ms)\n"
        "        tlv.advance(ts + tlv.tf_ms)\n"
    )
    ck(len(scan_source(wrong, "wrong.py")) == 1,
       "正控失败: 两级各加自己的 tf_ms 没被抓出来(闸门是空跑的)")
    # 反控: 正确的写法不得误报
    right = (
        "def run(kl, lv, tlv):\n"
        "    for i in range(kl.n):\n"
        "        now = kl.ts[i] + lv.tf_ms\n"
        "        lv.advance(now)\n"
        "        tlv.advance(now)\n"
    )
    ck(scan_source(right, "right.py") == [],
       "反控失败: 正确的写法(共用同一个 now)被误报")
    # 单级别: 一次调用不构成风险
    single = "def run(kl, lv):\n    lv.advance(kl.ts[0] + lv.tf_ms)\n"
    ck(scan_source(single, "single.py") == [], "单级别写法被误报")
    # 豁免注释生效
    esc = (
        "def run(kl, lv, tlv):\n"
        "    lv.advance(kl.ts[0] + lv.tf_ms)      # noqa: TIMEBASE\n"
        "    tlv.advance(kl.ts[0] + tlv.tf_ms)\n"
    )
    ck(scan_source(esc, "esc.py") == [], "noqa: TIMEBASE 豁免没生效")


# --------------------------------------------------------------------- 2. 静态
def t_static_scan() -> None:
    print("[2] 静态扫描 —— 全部 .py 里每个作用域只能有一个时钟实参")
    files: list[str] = []
    for d in SCAN_DIRS:
        base = os.path.join(ROOT, d)
        for dp, dns, fns in os.walk(base):
            dns[:] = [x for x in dns if x not in SKIP_DIRS]
            for fn in fns:
                if fn.endswith(".py"):
                    files.append(os.path.join(dp, fn))
    files.sort()
    bad: list[str] = []
    n_calls = 0
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fh:
                src = fh.read()
            bad += scan_source(src, f)
            n_calls += len(all_clock_calls(ast.parse(src)))
        except SyntaxError as e:                            # noqa: BLE001
            bad.append(f"{os.path.basename(f)}: 语法错误 {e}")
    print(f"     扫了 {len(files)} 个文件, {n_calls} 处时钟调用")
    ck(n_calls >= 10, f"时钟调用只找到 {n_calls} 处 —— 扫描面疑似失效")
    ck(not bad, "存在多时钟作用域:\n      " + "\n      ".join(bad))


# --------------------------------------------------------------------- 3. 运行时
TAIL = 400


def t_runtime_invariant() -> None:
    print("[3] 运行时不变式 —— 每一级都不得看到超出当前时刻的K线")
    sym = "BTCUSDT"
    try:
        ls = LevelSet("spot", sym).load()
    except Exception as e:                                   # noqa: BLE001
        skip(f"数据不足, 跳过: {e}")
        return
    lvs = [lv for lv in (ls.trend, ls.op, ls.confirm) if lv is not None
           and lv.kl is not None]
    if not ls.ok():
        skip(f"级别未就绪(trend={ls.trend.kl.n} op={ls.op.kl.n}), 跳过")
        return
    n = ls.op.kl.n
    start = max(300, n - TAIL)
    bad = 0
    worst = 9999.0
    for i in range(start, n):
        now = int(ls.op.kl.ts[i]) + ls.op.tf_ms
        ls.advance(now)
        for lv in lvs:
            j = lv.cur - 1
            if j < 0:
                continue
            seen = int(lv.kl.ts[j]) + lv.tf_ms          # 已喂入最后一根的收盘时刻
            if seen > now:                              # ⇒ 看到了未来
                bad += 1
            worst = min(worst, (now - seen) / 3.6e6)
    ck(bad == 0, f"有 {bad} 处级别看到了超出当前时刻的数据")
    print(f"     推进 {n - start} 根, 最紧的一级比当前时刻早 {worst:+.2f}h "
          f"(>=0 即无未来函数)")
    ck(worst >= 0, f"最紧的一级超前 {-worst:.2f}h —— 存在未来函数")


# --------------------------------------------------------------------- main
def main() -> int:
    print("=" * 78)
    print("时间基准静态闸门 (tests/check_timebase.py)")
    print("=" * 78)
    for fn in (t_positive_control, t_static_scan, t_runtime_invariant):
        try:
            fn()
        except Exception:                                    # noqa: BLE001
            FAILS.append(f"{fn.__name__} 抛异常")
            traceback.print_exc()
    print("-" * 78)
    if SKIP:
        print(f"跳过 {len(SKIP)} 项: " + "; ".join(SKIP))
    if FAILS:
        print(f"✗ {len(FAILS)} 项失败 (通过 {N_PASS} 项)")
        return 1
    print(f"✓ 全部通过: {N_PASS} 项")
    return 0


if __name__ == "__main__":
    sys.exit(main())
