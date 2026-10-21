"""chanbook —— 缠论全书版交易程序。

模块地图见 DESIGN.md。所有模块均为**流式**设计: 一根一根喂已收盘K线,
状态原地增量更新, 由构造保证无未来函数, 回测与实盘共用同一份代码。

版本沿革:
  0.1.0 (2026-09-27)  首个完整实现。默认离场 = 追踪 3.0×risk + RR2 目标
                      + 结构破位(与旧引擎 `backtest_long.manage` 同源)。
  0.2.0 (2026-09-29)  **默认离场升级**为「结构止损 + 一卖 S1」。
                      依据: 74 变体三轮回测 —— 旧默认单笔净期望在
                      1h/2h/4h 上全为负, 新默认是唯一单笔净 > 0.4% 的口径。
                      历史口径被固化为 `BookCfg.legacy()`, 逐位可复现。
                      详见 UPGRADE_2026-09-29.md。
  0.3.0 (2026-09-29)  **chan-quant 能力补全**(默认关闭, 逐位等价):
                      `market_filters`(过热带/前高压制/斐波那契/联动度) +
                      大盘状态闸门(regime_gate) + 多周期分形离场(exit_fenxing)
                      + `guards`(熔断护栏)。逐条口径自检见
                      PARITY_2026-09-29.md。生产默认口径未变。
"""

__version__ = "0.3.0"
__all__ = [
    "ckline", "fenxing", "bi", "seg", "zhongshu", "trend", "level",
    "macd", "beichi", "points", "strategy", "risk", "engine", "data",
    "filters", "market_filters", "guards",
]
