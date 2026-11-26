"""K 线形态识别模块（兼容层）。

形态引擎的实现已下沉到 :mod:`backend.shared.kline_patterns`，
以便市场分析 API 与 Strategy Lab 回测共用同一套判定逻辑（单一事实源）。

本模块仅做 re-export，保持既有 import 路径
``from backend.services.api.market_analysis import patterns as P`` 可用。
"""

from __future__ import annotations

from backend.shared.kline_patterns import (
    DEFAULT_LOOKBACK,
    DIRECTION_BEARISH,
    DIRECTION_BULLISH,
    DIRECTION_NEUTRAL,
    ENTRY_SIGNAL_PATTERNS,
    STANDARD_PATTERNS,
    TURNOVER_THRESHOLD,
    PatternHit,
    SymbolPatterns,
    aggregate_patterns,
    detect_patterns,
    detect_standard_patterns,
    next_trading_day,
    pattern_catalog,
    scan_symbol,
    scan_universe,
    talib_available,
    turnover_yuan,
)

__all__ = [
    "DEFAULT_LOOKBACK",
    "DIRECTION_BEARISH",
    "DIRECTION_BULLISH",
    "DIRECTION_NEUTRAL",
    "ENTRY_SIGNAL_PATTERNS",
    "STANDARD_PATTERNS",
    "TURNOVER_THRESHOLD",
    "PatternHit",
    "SymbolPatterns",
    "aggregate_patterns",
    "detect_patterns",
    "detect_standard_patterns",
    "next_trading_day",
    "pattern_catalog",
    "scan_symbol",
    "scan_universe",
    "talib_available",
    "turnover_yuan",
]
