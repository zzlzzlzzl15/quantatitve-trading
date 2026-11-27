"""低位回调上涨 — 形态引擎正向信号 + 负向形态回避 (矩阵原生策略)。

形态口径来自 K 线形态引擎 (app/indicators/patterns.py, 9 种入场形态),
本策略是它在矩阵原生回测里的首个落地用法: 形态列无法在矩阵上现算
(判定依赖滚动窗口与量比), 由引擎在面板上算出后随矩阵字段注入,
策略直接 matrix_feature(market, "pat_*") 直读。

选型依据 (真实 A 股全样本实证, 见 reports/pattern_hit_stats/REPORT.md):
  - 正向: pat_low_backtrace_increase (低位回调上涨) —— 9 种形态里唯一
    接近可用者 (20 日超额 +2.28%, 胜率 53.01%, 超额仅 -0.09%);
  - 负向回避: pat_volume_surge_up (-1.53%, t=-45.6)、
    pat_platform_breakout (-0.75%, t=-26.8)、
    pat_turtle_trade (-0.72%, t=-29.0) —— 形态成立后相对基准显著跑输,
    作为回避信号比作为入场信号更有价值。

形态是状态而非事件: 同一形态可连续多日成立, 本策略按"边沿触发"
(首次成立日) 建仓, 避免在形态持续期内重复买入。
"""

import numpy as np

from app.backtest.matrix import (
    MarketDataMatrix,
    SignalMatrix,
    make_signal_matrix,
    matrix_feature,
)
from app.backtest.matrix import (
    valid_shift as shift,
)

META = {
    "id": "pattern_low_backtrace",
    "name": "低位回调上涨(形态)",
    "description": "低位回调上涨形态成立且放量/突破类形态未成立时入场, 跌破MA20离场",
    "tags": ["形态", "低位", "回调"],
    "asset_types": ["stock"],
    "timeframes": ["1d"],
    "params": [
        {
            "id": "use_edge_trigger",
            "label": "仅形态首次成立日买入",
            "type": "bool",
            "default": True,
        },
        {
            "id": "avoid_bearish_patterns",
            "label": "回避放量/突破类形态",
            "type": "bool",
            "default": True,
        },
    ],
    "scoring": {"momentum_20d": 0.5, "change_pct": 0.3, "vol_ratio_5d": 0.2},
    "order_by": "score",
    "descending": True,
    "limit": 50,
}

EXECUTION_BACKEND = "matrix_native"
ENTRY_SIGNALS = []
EXIT_SIGNALS = ["signal_ma20_breakdown"]
STOP_LOSS = -0.05
MAX_HOLD_DAYS = 10


class PatternLowBacktraceMatrixStrategy:
    def required_fields(self) -> frozenset[str]:
        return frozenset({
            "close",
            "pat_low_backtrace_increase",
            "pat_volume_surge_up",
            "pat_platform_breakout",
            "pat_turtle_trade",
        })

    def required_warmup_bars(self, params: dict) -> int:
        # 低位回调形态依赖 60 日区间 (高低点), 突破/海龟形态依赖 20 日通道。
        del params
        return 60

    def compute_signals(self, market: MarketDataMatrix, params: dict) -> SignalMatrix:
        low_backtrace = matrix_feature(market, "pat_low_backtrace_increase")
        entry = low_backtrace > 0.5
        if params.get("use_edge_trigger", True):
            # 边沿触发: 形态昨日不成立、今日成立 —— 避免持续期内重复建仓。
            prev = shift(low_backtrace, 1)
            entry = entry & (prev < 0.5)
        if params.get("avoid_bearish_patterns", True):
            bearish = (
                (matrix_feature(market, "pat_volume_surge_up") > 0.5)
                | (matrix_feature(market, "pat_platform_breakout") > 0.5)
                | (matrix_feature(market, "pat_turtle_trade") > 0.5)
            )
            entry = entry & ~bearish
        ma20 = matrix_feature(market, "ma20")
        exit_ = (market.close < ma20) & (shift(market.close, 1) >= shift(ma20, 1))
        return make_signal_matrix(
            market.shape,
            entry=entry.astype(np.uint8),
            exit=exit_.astype(np.uint8),
            entry_signal_code=np.where(entry, 0, -1).astype(np.int16),
            exit_signal_code=np.where(exit_, 0, -1).astype(np.int16),
            exit_signal_ids=("signal_ma20_breakdown",),
        )


MATRIX_STRATEGY = PatternLowBacktraceMatrixStrategy()
