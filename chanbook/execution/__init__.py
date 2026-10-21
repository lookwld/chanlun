"""chanbook.execution —— 执行 / 运维 / 第三方层。

本包复刻 `chan-quant` 的执行层**功能契约**(配置加载、账户资金、记账复盘、
交易所适配、现货/合约执行器、看板、DeepSeek 筛选、熔断护栏), 但**信号来源**
是 chanbook 自己的缠论信号层(`chanbook.strategy.Signal`), 而非 chan-quant 的
均线排列 + 底分型入场。

边界(铁律, 与上一轮 PARITY/MIGRATION 一致):
  * **绝不改动 `chanbook/` 下 20 个策略/信号模块** —— 交易逻辑零改动。
  * 本包只做「信号之后」的事: 收到 Signal -> 资金规划 -> 下单/模拟 -> 持仓 ->
    离场 -> 记账 -> 看板。离场口径复用 chanbook 已有的 `sell`/`guards`, 不在此
    重新发明。
  * dry_run 默认, live 需配置显式解锁(与 chan-quant 的 `capital.allow_live` 一致)。

子模块:
  config    —— yaml + .env 加载 + 硬约束校验
  account   —— 资金池 / 仓位规划 / 槽位(复用 chanbook.risk.plan)
  analyzer  —— sqlite 记账 + 复盘报告 + 弊端诊断
  broker    —— ccxt 交易所适配(现货/合约), dry_run 走内存模拟
  bridge    —— chanbook Signal/Trade <-> 执行层字段映射
  trader    —— 现货执行器
  ftrader   —— 合约执行器
  llm       —— DeepSeek 筛选(可选)
  web       —— HTTP 看板
  guards    —— 引用自 chanbook.guards(不重复实现)
"""

__all__ = [
    "config", "account", "analyzer", "broker", "bridge",
    "trader", "ftrader", "llm", "web",
]
