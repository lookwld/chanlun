# chanlun

缠论（缠中说禅）全书版量化研究工程。从 K 线包含处理到分型、笔、线段、中枢、走势类型、背驰、三类买卖点与区间套，实现一条完整的、可回测、可自检的信号链，并配备流式回测引擎与 A/B 实验框架。

> 本项目为**研究用途**，不构成任何投资建议。

## 特性

- **全书版信号链**：包含处理 → 分型（含强度）→ 笔 → 线段（特征序列 + 第一/第二种破坏）→ 中枢 → 走势类型/中阴 → 级别 → MACD → 背驰 → 三类买卖点（B1/B2/B3、S1/S2/S3）+ 区间套；
- **流式增量引擎**：所有模块逐根喂入已收盘 K 线，状态原地增量更新，无未来函数，天然支持实时扫描；
- **回测与实盘同构**：入场判别为纯函数，回测与扫描共用同一条代码路径（`tests/check_live_equiv.py`）；
- **A/B 实验框架**：`BookCfg` 配置开关 + 变体矩阵回测 + 排序后集合比对，保证实验可复现；
- **完整自检**：14 组 `tests/check_*.py`，覆盖每个模块与关键开关行为（4000+ 断言）。

## 目录结构

```
chanlun/
  chanbook/            核心包
    data.py            行情读取（本地归档）
    ckline.py          K线包含处理
    fenxing.py         分型（含强度）
    bi.py              笔
    seg.py             线段（特征序列）
    zhongshu.py        中枢
    trend.py           走势类型 / 分解 / 中阴
    level.py           级别与递归
    macd.py            MACD（增量）
    beichi.py          背驰判定
    points.py          三类买卖点 + 区间套
    strategy.py        策略（多级别联立）
    risk.py            止损 / 目标 / 仓位
    engine.py          流式回测引擎
    execution/         执行层（扫描状态机、通知）
    report.py          报告
  tests/               自检（每模块一组断言）
  tools/               回测 / 扫描 / 数据 / 分析脚本
  docs/                设计与历史说明
  DESIGN.md            设计文档（含「近似与未实现声明」诚实清单）
```

## 快速开始

```bash
# 1. 安装依赖（Python >= 3.10）
pip install -r requirements.txt

# 2. 跑核心自检
python tests/check_core.py
python tests/check_points.py
python tests/check_zs_beichi.py

# 3. 一键全量自检（可选）
for t in tests/check_*.py; do python "$t"; done
```

## 数据说明

仓库**不包含行情数据**。数据读取层（`chanbook/data.py`）默认指向本地 K 线归档目录；
可用 `tools/` 下的下载脚本自行构建：

- `tools/download_all_klines.py` — 全量下载 K 线归档；
- `tools/download_1d_history.py` — 下载日线历史；
- `tools/refresh_archive.py` / `tools/build_dataset.py` — 归档刷新与数据集构建。

## 主要工具

| 脚本 | 用途 |
|---|---|
| `tools/backtest_engine.py` | 通用回测入口（多周期矩阵、变体开关） |
| `tools/scan_all_market.py` | 全市场扫描各周期买卖点（支持 `--state` 查询当前所处信号） |
| `tools/scan_spot_b1b2b3.py` | 现货 B1/B2/B3 扫描 |
| `tools/run_ab.py` | A/B 变体实验框架 |
| `tools/backtest_oos.py` | 样本外（OOS）验证 |
| `tools/backtest_random.py` | 随机对照（安慰剂）检验 |
| `tools/param_sweep.py` | 参数敏感性扫描 |

## 文档导航

- [DESIGN.md](DESIGN.md) — 总体设计、模块职责、**近似与未实现声明（诚实清单）**；
- [docs/模块中文详解_核心与策略风控.md](docs/模块中文详解_核心与策略风控.md) — 核心模块逐文件中文详解；
- `docs/` 与根目录下带日期的 `*_2026-*.md` — 各轮实验的方法与结论存档（回测判决、参数敏感性、离场归因等）。

## 工程妥协（节选）

为实现有界复杂度，部分原文概念做了显式声明的工程近似（完整 9 条见 `DESIGN.md` §6），
例如：线段"多次缺口叠加"按首次缺口处理；中枢回看有界（一买 1 个 / 三买 2 个）；
区间套仅做两级一次收敛。所有妥协均标注在对应模块的 docstring 中。

## 免责声明

本项目仅供学习与量化研究使用。加密货币交易风险极高，历史回测不代表未来收益，
使用本代码产生的任何损失由使用者自行承担。

## License

[MIT](LICENSE)
