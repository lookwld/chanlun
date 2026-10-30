# chanlun · 缠论（缠中说禅）全书版量化研究工程

从 K 线的包含处理一路推到三类买卖点与区间套，每个环节都是**流式**的（逐根喂已收盘 K 线、状态原地增量更新），因此同一份代码既能做历史回测，也能直接对最新行情做全市场扫描。

> 本项目为**研究与教学用途**，不构成任何投资建议。历史回测不代表未来收益。

**版本**：`chanbook 0.3.0`

---

## 目录

1. [能做什么](#1-能做什么)
2. [环境要求](#2-环境要求)
3. [安装](#3-安装)
4. [准备行情数据（必须先做）](#4-准备行情数据必须先做)
5. [五分钟上手](#5-五分钟上手)
6. [全市场扫描](#6-全市场扫描)
7. [工具索引（49 个）](#7-工具索引49-个)
8. [在 Python 里调用](#8-在-python-里调用)
9. [配置项 BookCfg](#9-配置项-bookcfg)
10. [自检](#10-自检)
11. [常见问题](#11-常见问题)
12. [目录结构](#12-目录结构)
13. [文档导航](#13-文档导航)
14. [已知限制](#14-已知限制)
15. [免责与许可](#15-免责与许可)

---

## 1. 能做什么

| 能力 | 说明 |
|---|---|
| 完整信号链 | 包含处理 → 分型（带强度）→ 笔 → 线段（特征序列 + 两种破坏）→ 中枢 → 走势类型/中阴 → 级别递归 → MACD → 背驰 → 三类买卖点（B1/B2/B3、S1/S2/S3）→ 区间套 |
| 流式增量 | 每个模块逐根推进，构造上无未来函数；`tests/check_timebase.py` 专门看守「多级别各加自己的周期」这类偷看未来的 bug |
| 回测/扫描同构 | 入场判别是纯函数，回测与实盘扫描走同一条代码路径，等价性由 `tests/check_live_equiv.py` 抽检 |
| A/B 实验 | `BookCfg` 开关 + 变体矩阵 + 排序后集合比对（`tools/cmp_cache.py`），保证实验可复现 |
| 全市场扫描 | 币安现货约 400 个标的 × 8 个周期，一条命令出「当前状态表」与「历史信号表」 |
| 自检 | 14 组 `tests/check_*.py`，约 4,000 条断言，覆盖每个模块与关键开关 |

---

## 2. 环境要求

| 项 | 要求 | 说明 |
|---|---|---|
| Python | **≥ 3.10** | 代码使用 `X \| None` 联合类型与 `list[str]` 内置泛型 |
| 依赖 | `numpy`、`matplotlib`、`scikit-learn`、`PyYAML` | 见 `requirements.txt`；用于回测统计、画图、建模、读配置 |
| 磁盘 | 建议 ≥ 10 GB | 全市场 8 周期 K 线归档约 1～3 GB，扫描快照按日期各存一份 |
| 扫描链 | **无需 numpy/pandas** | `rescan_market.py` / `scan_spot_b1b2b3.py` / `download_all_klines.py` 只用标准库，可在只有 `ccxt` 的小机器上跑 |

> 「扫描链」指第 6 节那条从下载到出表的流程，它与回测/建模用的依赖是分开的。这一点是刻意设计：扫描常部署在低配服务器上，装不下科学计算栈。

---

## 3. 安装

```bash
git clone git@github.com:lookwld/chanlun.git
cd chanlun

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

验证安装：

```bash
python -c "import chanbook; print(chanbook.__version__)"   # 期望输出 0.3.0
python tests/check_core.py                                  # 期望「通过 40 项, 失败 0 项」
```

---

## 4. 准备行情数据（必须先做）

仓库**不含任何行情数据**（`.gitignore` 排除了 `data/`、`out/`、`logs/`）。没有数据，除自检里少数合成行情用例外，所有工具都会报「加载失败」。

### 4.1 归档布局

数据层（`chanbook/data.py`）读的是按市场、按周期分文件的 CSV 压缩包：

```
data_archive/
  um/k/<SYM>_{5m,15m,1h}.csv.gz        # 合约（U 本位）：只有这三档
  um/f/<SYM>_funding.csv.gz            # 合约资金费率
  spot/k/<SYM>_{15m,1h,4h,1d}.csv.gz   # 现货：这四档
  all_symbols.txt                      # 合约标的清单
  spot_symbols.txt                     # 现货标的清单
  universe_um.json                     # 时点标的池 {"YYYY-MM": [标的, ...]}
  universe_spot.json
```

每行一列 K 线，列序固定为 `ts,open,high,low,close,volume,quote_volume,trades`，`ts` 是**毫秒时间戳**，按 UTC 对齐。

### 4.2 各周期的来源

| 周期 | 现货 `spot` | 合约 `um` |
|---|---|---|
| `15m` / `1h` | 归档原始文件 | 归档原始文件 |
| `5m` | 需自行准备 | 归档原始文件 |
| `4h` / `1d` | 归档原始文件 | **由 `1h` 现场聚合**（`resample()`，UTC 向下取整） |
| `30m` / `3d` / `1w` / `1M` | 需自行准备原始文件 | 需自行准备原始文件 |

有两条硬规则写在 `data.py` 里，改数据前务必知道：

- `1w` / `3d` / `1M` 是**非均匀周期**（周不等于 7×24 小时、月长度在 28～31 天浮动），不能用「时间戳整除」聚合，必须直接读交易所的原始 K 线；
- `1M` 的「已收盘」判据用 **31 天**（上界），而不是按真实月长算 —— 后者会在 2 月提前宣布收盘，等于偷看未来。

`download_all_klines.py` 默认拉的周期与扫描器 `LEVELS` 一致，共 8 档：`1M, 1w, 3d, 1d, 4h, 1h, 30m, 15m`。

### 4.3 数据放在哪

默认路径由 `chanbook/data.py` 的 `ARCH` 决定：**项目目录的上一级**下的 `chan-quant/data_archive`。

举例：项目在 `/data/chanlun`，则默认读 `/data/chan-quant/data_archive`。

这个默认值来自历史项目结构，不改也能用；更推荐显式指定环境变量：

```bash
# Linux / macOS
export CHANBOOK_ARCH=/srv/market/data_archive

# Windows PowerShell
$env:CHANBOOK_ARCH = "D:\market\data_archive"

# Windows CMD
set CHANBOOK_ARCH=D:\market\data_archive
```

所有工具都认这个变量，不需要逐个加参数。

### 4.4 下载数据

**方式一（推荐）**：交给编排脚本，它第一步就会去拉 `exchangeInfo`，然后下载 K 线：

```bash
python tools/rescan_market.py --dry      # 先看五步计划，无副作用
python tools/rescan_market.py            # 正式跑，见第 6 节
```

**方式二**：单独用下载脚本。注意它**不会自己拉交易规则**——标的清单是从 `exchangeInfo.json` 里筛的（USDT 计价 + 状态 `TRADING` + 允许现货交易），该文件不存在会直接报错，所以要先手工准备：

```bash
mkdir -p out/scan
curl -sS https://data-api.binance.vision/api/v3/exchangeInfo -o out/scan/exchangeInfo.json

python tools/download_all_klines.py --outdir out/scan/klines --limit 20   # 先试 20 个
python tools/download_all_klines.py --outdir out/scan/klines              # 全市场 8 个周期
python tools/download_all_klines.py --outdir out/scan/klines --incremental  # 已有缓存时只补增量
```

| 参数 | 默认 | 含义 |
|---|---|---|
| `--exch` | `out/scan/exchangeInfo.json` | **交易规则文件的路径**（不是交易所名）。标的清单从这里筛 |
| `--outdir` | `out/scan/klines` | 输出目录 |
| `--tfs` | 8 档全量 | 要拉的周期，逗号分隔：`1M,1w,3d,1d,4h,1h,30m,15m` |
| `--jobs` | `16` | 并发线程数（每线程一条长连接，实测这比什么都影响速度） |
| `--base` | `https://data-api.binance.vision/api/v3` | 端点**基址，必须含 `/api/v3`**。传成站点根会让每个请求都 404，而症状是「全部失败」不像报错 |
| `--incremental` | 关 | 校验本地缓存后只补新增；等价性由 `tests/check_incremental.py` 看守 |
| `--keep-bars` | `1000` | 滚动缓存保留根数（也是币安单次请求上限） |
| `--limit` | `0` | 只拉前 N 个标的（调试） |

下载完成后，`out/scan/klines` 就是一份可用的归档，指向它即可：

```bash
export CHANBOOK_ARCH=$PWD/out/scan/klines
```

日线长历史另有一个单点脚本：`python tools/download_1d_history.py`（供暴涨事件研究，无参数）。

---

## 5. 五分钟上手

```bash
# ① 核心自检：不依赖行情数据，先确认代码本身没问题
python tests/check_core.py
python tests/check_points.py
python tests/check_zs_beichi.py

# ② 合成行情端到端冒烟：也不需要数据
python tools/explore_points.py

# ③ 有数据之后：单标的回测，打印完整报告
python - <<'PY'
from chanbook import engine
from chanbook.report import full_report
cfg = engine.BookCfg(market="spot")
trades, diag = engine.run_symbol("spot", "BTCUSDT", cfg)
print(full_report("spot", "BTCUSDT", trades, [diag]))
PY

# ④ 全市场扫描（见第 6 节）
python tools/rescan_market.py --dry      # 先看计划，不执行
```

---

## 6. 全市场扫描

### 6.1 一条命令

```bash
python tools/rescan_market.py
```

它会依次做五步，每步把进度**原子写入** `out/scan/status.json`，同时追加到 `out/scan/scan.log`：

| 步骤 | 动作 | 产物 |
|---|---|---|
| 1 | 拉 `exchangeInfo`（交易规则与交易对状态） | `out/scan/exchangeInfo.json` |
| 2 | 下载 K 线（增量，失败率超阈值即中止） | `out/scan/klines_YYYYMMDD/` |
| 3 | **反向审计**：在保留池里找疑似稳定币 | 非空则整轮中止 |
| 4 | 输出「当前所处买点」表（`--state`） | `<out>_state_bysym.csv` 等 |
| 5 | 输出「历史信号」表（窗口 120 根 K 线 / 10 笔） | `<out>_bysym.csv` 等 |

单核机器、约 400 个标的 × 8 个周期，实测 **2～3 分钟**跑完。

常用参数：

| 参数 | 默认 | 用途 |
|---|---|---|
| `--dry` | — | 只打印五步计划，不执行（改配置前先看一遍） |
| `--limit N` | `0` | 只处理前 N 个标的，服务器冒烟测试用 |
| `--jobs` | `1` | 扫描并发。1 核机器保持 1，加大会争抢 CPU 并成倍吃内存 |
| `--incremental` / `--no-incremental` | 开 | 增量下载；排查数据问题时用 `--no-incremental` 退回全量 |
| `--keep N` | `2` | 只保留最近 N 份快照目录（`0` = 不限，但永远至少留 1 份） |
| `--reserve-mb` | `1500` | 磁盘地板（MB）。剩余空间低于此值就中止下载，不会把盘占满 |
| `--budget-mb` | `0`（自动） | 缓存目录允许占用的上限 |
| `--max-fail-rate` | `0.02` | 下载失败率上限，超过即中止（半截数据会静默毁掉一批标的） |
| `--step-timeout` | `2400` | 单步超时（秒） |
| `--nice` | `15` | 子进程降权，避免和同机其它服务抢 CPU |
| `--base` | `https://data-api.binance.vision` | 站点根。**注意与 4.4 的区别**：这里传站点根，编排器内部自己拼 `/api/v3`；而 `download_all_klines.py --base` 要的是含 `/api/v3` 的基址 |

> 首次运行时 `out/scan` 目录还不存在，`--dry` 也能正常打印计划（取磁盘余量时会向上回溯到最近的已存在目录），不会创建任何文件。

### 6.2 扫描产物

编排器默认把结果写到 `out/scan/b123_spot`（状态表另存为 `b123_spot_state`）：

| 文件 | 内容 |
|---|---|
| `b123_spot.md` | 人读报告：标的共振总表 + 说明 |
| `b123_spot.csv` | 一行一个 **(标的, 周期, 方向)** 的当前信号 |
| `b123_spot_bysym.csv` | **一行一个标的**的宽表（8 个周期各一列，格内只写字母如 `B3 S1`） |
| `b123_spot_bysym_times.json` | 悬停时间旁挂表：`标的 → 周期 → [字母, 开盘ms, 收盘ms]` |
| `b123_spot.json` | 机器可读汇总 |

**为什么时间单独放一个文件**：时间写进格子会把 8 个周期列撑宽到无法阅读，所以 CSV 只放字母，时间走旁挂 JSON 给看板做悬停提示。

### 6.3 只看「此刻处于什么信号」

```bash
# --kl_dir 必须指向本次的快照目录。它的默认值是历史遗留的固定日期目录，
# 几乎肯定不存在，所以单独运行时一律显式给。
python tools/scan_spot_b1b2b3.py --kl_dir out/scan/klines_20261031 --state
```

`--state` 与默认模式的区别：

- 默认模式回答「窗口内出现过哪些信号类型」（一个格子可能有多个字母）；
- `--state` 回答「**现在**处于哪个买点」—— 每个（标的, 周期）只给一个，取「最近一个结构止损未被跌破的买点」。开启后自动不限窗口。

其他常用参数：`--tfs`（周期，逗号分隔）、`--syms`（只扫指定标的，调试用）、`--audit`（只做稳定币反向审计）、`--asof`（已收盘判定基准的毫秒时间戳，`0` = 用 K 线文件的抓取时刻）、`--jobs`（默认 10，扫描并发）。

### 6.4 反向审计是什么

扫描前会在「保留池」里检查是否存在**疑似稳定币**（判据：1 小时平均绝对涨跌幅 < 0.0008）。命中就整轮中止，因为稳定币会让「涨幅榜」「信号密度」这类统计全部失真。

名单 `EXCL_STABLE` 是人工维护的，历史上 `U`、`RWUSD` 都是这么补进去的。审计不通过时，先确认是真稳定币（补名单）还是数据异常（查 `--no-incremental` 重下）。

---

## 7. 工具索引（49 个）

所有脚本都在 `tools/` 下，用 `python tools/<名字>.py --help` 看完整参数。

### 7.1 扫描

| 脚本 | 用途 |
|---|---|
| `rescan_market.py` | **推荐入口**：刷 exchangeInfo → 下载 → 反向审计 → 状态表 → 历史表，一条命令 |
| `scan_spot_b1b2b3.py` | 现货全市场 B1/B2/B3 扫描（只扫数字代币，按级别出表），支持 `--state` / `--audit` |
| `scan_all_market.py` | 通用全市场扫描：输出每个币每个周期当前处于哪个买卖点 |
| `scan_resonance.py` | 多级别共振扫描：长周期与短周期同时给出买点 |
| `resonance_board.py` | 生成共振扫描的独立预览页 |

### 7.2 回测

| 脚本 | 用途 |
|---|---|
| `backtest_engine.py` | 现货主引擎多变体回测：周期 × 构造 × 离场 × 8 个开关 |
| `backtest_engine_path.py` | 走 `engine.run_all` 的全池回测，验证「改完能到什么效果」 |
| `backtest_b12_tf.py` | 1h / 2h 的 B1+B2 入场、结构止损 + 一卖离场 |
| `backtest_b1_4h.py` | 4h 第一类买点入场 + 缠论卖点离场（9 年现货） |
| `backtest_resonance.py` | 多级别共振入场 + 结构止损 + 一卖离场 |
| `backtest_reso6.py` | 六个级别任意两两共振即入场 |
| `backtest_top20_tf.py` | 市值前 20 现货的多周期横向对比 |
| `backtest_short_s1.py` | 一卖作为**空头**入场信号的实证检验 |
| `backtest_strength.py` | 结构强度选币：缠论从「触发器」转「过滤器」 |
| `backtest_exit_alpha.py` | 离场 alpha 对照：缠论离场 vs 朴素离场 |
| `backtest_volcap.py` | 波动率上限闸门的效果 |
| `backtest_market_filters.py` | 7 个市场过滤能力逐个在主引擎路径上验证 |
| `backtest_doji.py` / `backtest_doji2.py` | 十字星拐点策略（v1 / v2） |
| `backtest_pump_strategy.py` | 山寨币拐点策略完整回测（选币 + 持有 + 离场） |

### 7.3 A/B 实验与统计检验

| 脚本 | 用途 |
|---|---|
| `run_ab.py` | A/B 运行器：同数据同区间跑多组配置，输出可比报告 |
| `ab_summary.py` | 把 `out/ab_*.txt` 解析成可比表格 + 门槛判决（修正名义额分档导致的分母偏差） |
| `ab_stats.py` | 从成交明细算统计显著性与集中度 |
| `param_sweep.py` | 把参数扫描报告解析成「参数 → 指标」曲线，找峰值 |
| `macd_sweep.py` | B1 参数敏感性：MACD 快慢线与背驰容差 |
| `backtest_oos.py` | 样本外分段验证：2017–2021 训练 / 2022–2026 验证 |
| `backtest_random.py` | 随机入场对照（安慰剂）：检验择时是否可与随机区分 |
| `cmp_cache.py` | 成交明细缓存比对：判定两轮回测是否产出同一批交易 |
| `exit_attrib.py` | 离场归因：回答「设计的 RR2.0 为什么没兑现」 |

### 7.4 数据与归档

| 脚本 | 用途 |
|---|---|
| `download_all_klines.py` | 全市场现货 K 线拉取（8 个周期） |
| `download_1d_history.py` | 全市场现货日线长历史 |
| `refresh_archive.py` | 把行情快照刷到最新（补最近 N 天 1h，只读源归档） |
| `build_dataset.py` | 把成交记录 + 日线上下文拍平成建模样本表 |
| `dataset_report.py` | 数据集体检：字段、缺失、标签分布、初步关系 |
| `field_catalog.py` | 字段目录：有哪些列、覆盖多少、什么量纲、属于哪一组 |

### 7.5 建模与因子

| 脚本 | 用途 |
|---|---|
| `model_baseline.py` | 入场判别建模：纯样本内多变量基线 |
| `pump_factor_analysis.py` | 山寨币暴涨前特征因子提取 |
| `bottom_factor_analysis.py` | 山寨币「底部」特征因子提取 |
| `pump_lag_decay.py` | 因子区分度随「暴涨前提前量」的衰减（稳健性检验） |
| `market_cap_stats.py` | 全市场现货市值与因子统计 |

### 7.6 诊断与可视化

| 脚本 | 用途 |
|---|---|
| `plot_signals.py` | 在 K 线图上画出缠论结构与买卖点（`--symbol`、`--tf 1h/4h`、`--bars`） |
| `explore_points.py` | 合成行情下的三类买点诊断（也可作端到端冒烟） |
| `smoke_level.py` | 级别流水线冒烟：真实数据上跑一遍，看性能/内存/结构自洽 |
| `probe_levels.py` | 级别可行性探针：4h / 日线 / 3日线 / 周线 能否跑通并给出买点 |
| `reso_trace.py` | 单品种共振溯源：为什么没出信号 / 那天发生了什么 |
| `reso_timing.py` | 共振的时间对齐测量：长短周期到底对不对得上 |
| `clock_leak_probe.py` | 量化「多级别各加自己的周期」这个时钟泄漏的影响 |
| `analyze_doji_postexit.py` | 十字星策略止盈离场后的价格后视（事后描述，不是新回测） |
| `b12_analyze.py` | 多轮回测同口径对照 |

---

## 8. 在 Python 里调用

### 8.1 读行情

```python
from chanbook import data as D

kl = D.load("spot", "BTCUSDT", "1h")     # market: "spot" | "um"
print(kl.n, D.fmt(kl.ts[-1]))            # 根数、最后一根的开盘时间（按 UTC+8 显示）
```

`D.load` 在缺失 `4h`/`1d` 时会用 `1h` 现场聚合；其余周期缺档会直接报错 —— 先确认 `CHANBOOK_ARCH` 指向的目录里确实有对应文件。

### 8.2 单标的回测

```python
from chanbook import engine, data as D
from chanbook.report import full_report

cfg = engine.BookCfg(market="spot")                 # 默认 = 生产口径
trades, diag = engine.run_symbol("spot", "BTCUSDT", cfg)
# 注意: run_symbol 返回单个诊断对象, 而报告函数要的是「诊断列表」, 所以包一层
print(full_report("spot", "BTCUSDT", trades, [diag]))

for t in trades[:3]:
    print(t.kind, D.fmt(t.entry_ts), f"{t.pnl_pct:+.2%}", t.reason)
```

`Trade` 的主要字段：`sym`（标的）、`kind`（B1/B2/B3）、`entry_ts`（开仓时刻，毫秒）、`exit_ts`（平仓时刻）、`entry`/`exit`（价格）、`pnl_pct`（单笔净收益率）、`risk_pct`（止损距离占比）、`notional`（名义本金，U）、`reason`（离场原因）、`hold_min`（持仓分钟数）、`feats`（结构化特征，纯观测、不参与判定）。

### 8.3 批量跑

```python
from chanbook import engine

cfg = engine.BookCfg(market="spot", kinds=("B1", "B2"))
trades, diags = engine.run_all("spot", cfg, limit=50, jobs=4, progress=10)
print(len(trades), "笔成交")
```

`run_all` 的 `symbols` 不传时按 `_pick_symbols` 规则取池；`universe_top=N` 取时点池前 N 名；`limit` 只跑前 N 个（冒烟）；`jobs=0` 自动取 `min(12, CPU 核数-1)`。

### 8.4 换一套口径做 A/B

```python
from chanbook.engine import BookCfg

base = BookCfg(market="spot")                    # 生产口径
legacy = BookCfg.legacy(market="spot")           # 2026-09-29 之前的历史口径
enh = BookCfg.enhanced(market="spot")            # 唯一略正的可选增强（两条均线 5/10）
tuned = BookCfg(market="spot", beichi_tol=0.03, b3_zs_window=3)
```

三个构造器的语义必须分清：

- 裸 `BookCfg(...)` 自 2026-09-29 起等于**生产口径**（离场 = 结构止损 + 一卖 S1）；
- 复现**旧报告、旧判决**必须显式用 `BookCfg.legacy(...)`，否则数字对不上；
- `BookCfg.enhanced(...)` 打开 `ma_mode="ma"`（5/10 均线），是唯一实测略正的可选增强。

---

## 9. 配置项 BookCfg

`chanbook/strategy.py` 的 `BookCfg` 是全部实验的唯一开关集合。下表只列最常改的，完整字段看源码（每个字段都有口径注释）。

### 9.1 买卖点扫描

| 字段 | 默认 | 含义 |
|---|---|---|
| `kinds` | `(B1, B2, B3)` | 启用哪几类买点 |
| `pull_bi` | `1` | 回拉笔数（二买要求回拉的笔数） |
| `beichi_tol` | `0.02` | 背驰容差：后段力度需比前段弱多少才认定背驰 |
| `beichi_strict` | `False` | 严格背驰（要求同时满足多个条件） |
| `b2_max_bi` | `20` | 二买相对一买的最大间隔笔数 |
| `b1_max_lag` | `12` | 一买允许的最大滞后笔数 |
| `b1_zs_window` | `1` | 一买回看的中枢个数 |
| `b3_zs_window` | `2` | 三买回看的中枢个数 |
| `b1_allow_no_zs` | `True` | 允许无中枢语境的一买（纯 MACD 底背驰） |

### 9.2 区间套（确认级别）

| 字段 | 默认 | 含义 |
|---|---|---|
| `confirm_gate` | `False` | 仅对 B1 生效：要求确认级别也给出同向背驰才放行 |
| `confirm_tol` | `0.02` | 确认级别的背驰容差 |
| `record_confirm` | `False` | 只记录区间套结论、**不拦截**，用于采集「通过/未通过」两侧样本。开启后单标的耗时约 2.4 倍 |

### 9.3 离场

| 字段 | 默认 | 含义 |
|---|---|---|
| `exit_mode` | `"stop"` | `stop` = 只认灾难点（结构止损）；`old` = 追踪止盈 + RR 目标 + 结构破位 |
| `sell_exit` | `(S1,)` | 卖点离场。`()` 关闭、`(S1,)` 一卖、`(S1,S2,S3)` 三类全开。**卖点是替换默认离场，不是叠加** |
| `struct_on` | `True` | 启用结构止损 |
| `trail_mult` | `None` | 追踪倍数（`exit_mode="old"` 时才生效） |
| `exit_fenxing` | `False` | 多周期分形离场（入场后新形成的反向分型触发） |
| `fx_min_bars_after` | `1` | 分型右侧至少几根合并 K 线才算确认 |
| `fx_require_bi` | `False` | 要求反向笔已成立（更强确认） |

### 9.4 风控

| 字段 | 默认 | 含义 |
|---|---|---|
| `atr_mult` | `None` | 止损 = ATR × 倍数；`None` 用 `risk.CALIB` 的标定值 |
| `max_loss_pct` | `None` | 单笔最大亏损占本金比例 |
| `rr` | `None` | 目标盈亏比 |
| `notional` | `None` | 名义本金（U）；`None` 按买点分档 |
| `b1_wide` | `True` | 一买用更宽的止损 |

### 9.5 后置过滤器（默认全关，开启才建视图对象）

| 字段 | 默认 | 含义 |
|---|---|---|
| `vol_gate` | `0.0` | 量能闸门：后段均量 / 前段均量 需小于此值。`0` = 关闭 |
| `atr1d_cap` | `0.0` | 日线 ATR14% 上限（绝对波动率水平）。`0` = 关闭 |
| `regime_gate` | `False` | 大盘状态闸门：牛市/震荡只做多，熊市停机 |
| `overheat_on` | `False` | 过热带：4h/1d 过热则拦截 |
| `resistance_on` | `False` | 前高压制：贴着前高/前低正下方则拦截 |
| `fib_on` | `False` | 斐波那契回撤：太浅/太深则拦截 |
| `correl_on` | `False` | 联动度：高联动币方向须与大盘一致 |
| `ma_mode` | `"off"` | 两条均线：`off` / `ma` / `close` / `both`（否决型）/ `size`（仓位型） |
| `pos_layer` | `"off"` | 仓位分层：`off` / `amp` / `ratio` / `kind` |
| `zs_merge` | `False` | 相邻同向重叠中枢合并为高一级中枢 |

> `vol_gate`（成交量比）与 `atr1d_cap`（日线波动率绝对水平）是**两个完全不同的维度**，只是历史上撞了名字，别混用。

### 9.6 不要顺手打开的开关

`b1_dir` / `pending_out` / `vol_gate` / `pos_layer` / `zs_merge` 在当前数据上实测为负或零触发，默认关闭。要开请先跑 `tools/run_ab.py` 做对照，不要凭直觉改默认值。

---

## 10. 自检

```bash
# 逐组跑
python tests/check_core.py         # 包含处理 / 分型 / 笔
python tests/check_points.py       # 三类买卖点 + 无未来函数

# 全部跑一遍
for t in tests/check_*.py; do echo "== $t"; python "$t"; done
```

| 脚本 | 覆盖 |
|---|---|
| `check_core.py` | 包含处理 / 分型 / 笔（40 项） |
| `check_points.py` | 三类买卖点与无未来函数（37 项） |
| `check_zs_beichi.py` | 中枢 / 走势类型 / MACD / 背驰（43 项） |
| `check_seg.py` | 线段：特征序列分型、两种破坏、强制兜底、流式一致性（24 项） |
| `check_timebase.py` | 时间基准静态闸门，防「多级别各加自己的周期」类偷看未来（6 项） |
| `check_engine.py` | 引擎 / 风控 / 报告 |
| `check_variants.py` | 8 项开关 + 离场路径 + 默认值升级 |
| `check_execution.py` | 执行层：配置 / 账户 / 分析 / 桥接 / LLM / 交易器（28 项） |
| `check_market_filters.py` | 市场过滤器 + 熔断护栏 |
| `check_resonance.py` | 共振模块 |
| `check_incremental.py` | 增量下载 + 磁盘护栏（离线、确定性、不联网） |
| `check_scan_state.py` | 「当前所处买点」扫描的口径不变量 |
| `check_scan_report_format.py` | 扫描报告格式的口径不变量 |
| `check_live_equiv.py` | 生产缓存 vs 全新全量下载逐根等价（联网、只读） |

`check_engine.py` 与 `check_variants.py` 依赖 `data_archive` 里的真实归档，没准备数据时会各有 2 项失败；`check_live_equiv.py` 需要联网下载，离线环境请跳过。其余各组在纯代码状态下即可全过。

---

## 11. 常见问题

**Q1：报「加载失败」或 `FileNotFoundError`。**
先确认 `CHANBOOK_ARCH` 指对了目录，再看该目录下是否有 `spot/k/BTCUSDT_1h.csv.gz` 这类文件。注意默认路径是「项目上一级的 `chan-quant/data_archive`」，很容易指错。

**Q2：某个周期所有标的都「根数不足」。**
粗周期受数据源物理上限约束：币安 2017 年 8 月上线，月线到现在也就 110 根左右。扫描器对月线/周线/3 日线的最低根数要求分别是 48 / 100 / 200（`MIN_BARS_BY_TF`），不是回测用的 500。

**Q3：扫描结果显示 `--audit` 不通过、整轮中止。**
保留池里出现了疑似稳定币（1 小时平均绝对涨跌幅 < 0.0008）。确认是稳定币就把它加进 `scan_spot_b1b2b3.py` 的 `EXCL_STABLE`；怀疑是数据坏了就用 `--no-incremental` 重下一次。

**Q4：`rescan_market.py` 中途停在下载步。**
多半是磁盘地板（`--reserve-mb`，默认 1500 MB）或失败率阈值（`--max-fail-rate`，默认 2%）触发的保护性中止。看 `out/scan/status.json` 的 `error` 字段和 `out/scan/scan.log`。

**Q5：回测数字和历史报告对不上。**
默认口径在 2026-09-29 升级过（离场改为结构止损 + 一卖 S1）。复现旧结论必须用 `BookCfg.legacy(...)`。

**Q6：跑得很慢。**
`run_all` 的 `jobs=0` 会自动取 `min(12, 核数-1)`；`record_confirm=True` 会让单标的耗时约 2.4 倍；`confirm_gate=True` 同样要加载确认级别。三者叠加最容易慢到不可用。

**Q7：Windows 下路径报错。**
环境变量用 `set CHANBOOK_ARCH=D:\...`（CMD）或 `$env:CHANBOOK_ARCH="D:\..."`（PowerShell）。脚本内部一律用 `os.path` 拼接，不写死分隔符。

---

## 12. 目录结构

```
chanlun/
  chanbook/                    核心包（31 个文件）
    data.py                    行情读取（只读归档，缺 4h/1d 时由 1h 聚合）
    ckline.py                  K 线包含处理
    fenxing.py                 分型（含强度）
    bi.py                      笔
    seg.py                     线段（特征序列 + 两种破坏）
    zhongshu.py                中枢
    trend.py                   走势类型 / 分解 / 中阴
    level.py                   级别与递归
    macd.py                    MACD（增量）
    beichi.py                  背驰判定
    points.py                  三类买卖点 + 区间套
    strategy.py                策略与 BookCfg
    sell.py                    卖点扫描器（S1/S2/S3）
    risk.py                    止损 / 目标 / 仓位
    filters.py                 ATR 等基础过滤
    market_filters.py          过热带 / 前高 / 斐波那契 / 联动度
    guards.py                  熔断护栏
    resonance.py               多级别共振
    engine.py                  流式回测引擎（run_symbol / run_all）
    report.py                  报告与成交明细读写
    execution/                 执行层（配置 / 账户 / 分析 / 桥接 / LLM / 交易器）
  tests/                       14 组自检
  tools/                       49 个可运行脚本
  docs/                        模块逐文件中文详解
  DESIGN.md                    总体设计与「近似与未实现声明」
  DATASET.md                   建模数据集说明
```

---

## 13. 文档导航

| 文档 | 内容 |
|---|---|
| [DESIGN.md](DESIGN.md) | 总体设计、模块职责、9 条工程近似、**「近似与未实现声明」诚实清单** |
| [DATASET.md](DATASET.md) | 建模数据集的字段定义、构造流程、标签口径 |
| [docs/模块中文详解_核心与策略风控.md](docs/模块中文详解_核心与策略风控.md) | 核心模块逐文件中文详解 |

读代码的建议顺序：`ckline → fenxing → bi → seg → zhongshu → trend → points → strategy → engine`。每一层的 docstring 都写了「为什么这么定」，比函数签名更值得读。

---

## 14. 已知限制

为把复杂度控制在可验证的范围内，部分原文概念做了**显式声明的工程近似**（完整 9 条见 `DESIGN.md` §6），例如：

- 线段的「多次缺口叠加」按首次缺口处理；
- 中枢回看有界（一买回看 1 个中枢、三买 2 个）；
- 区间套只做两级一次收敛；
- 月线「已收盘」按 31 天判定，月初那根会晚 0～3 天才被认作收盘。

这些近似都写进了对应模块的 docstring，不是隐藏行为。做研究时如果结论依赖某条近似，请先确认它的影响方向。

---

## 15. 免责与许可

本项目仅供学习与量化研究。加密货币交易风险极高，历史回测不代表未来收益，使用本代码产生的任何损失由使用者自行承担。

[MIT License](LICENSE)
