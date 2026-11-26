"""端到端：形态信号 → 回测成交闭环。

验证三件事：
1. 形态策略脚本无需 import 任何 backend 模块即可通过 AST 静态检查
   （形态能力由 ``ctx.pattern_*`` 提供，不受沙箱 import 白名单限制）；
2. 时点口径：形态在 T 日收盘识别，订单在 **T+1 交易日**按收盘价成交；
3. 边沿触发：同一形态持续存在时不重复下单。
"""

from __future__ import annotations

import pandas as pd
import pytest

from backend.services.engine.strategy_lab.engine.data_provider import InMemoryProvider
from backend.services.engine.strategy_lab.engine.loop import run_backtest
from backend.services.engine.strategy_lab.runner.ast_checker import assert_safe
from backend.services.engine.strategy_lab.sdk.context import Context
from backend.shared import kline_patterns


SYMBOL = "A"
START = "2026-01-05"
DAYS = 40

# 形态策略：命中任意入场准备标准 → T+1 买入
STRATEGY = """
def setup(ctx):
    ctx.universe = ["A"]
    ctx.start = "2026-01-05"
    ctx.end = "2026-03-31"
    ctx.cash = 1_000_000
    ctx.commission = 0.0
    ctx.slippage = 0.0
    ctx.tax_sell = 0.0
    ctx.transfer_fee = 0.0

def on_bar(ctx, bar):
    ctx.pattern_entry(bar.symbol, weight=0.5, reason="pattern entry")
"""

# 横截面形态筛选：当日扫描 → 次日调仓（T+1 口径）
SCREEN_STRATEGY = """
_pending = None

def setup(ctx):
    ctx.universe = ["A"]
    ctx.start = "2026-01-05"
    ctx.end = "2026-03-31"
    ctx.cash = 1_000_000
    ctx.commission = 0.0
    ctx.slippage = 0.0
    ctx.tax_sell = 0.0
    ctx.transfer_fee = 0.0

def on_universe(ctx, date, snapshot):
    global _pending
    if _pending:
        ctx.set_target_holdings(_pending, reason="pattern screen")
        _pending = None
    hits = ctx.pattern_scan(symbols=ctx.universe)
    _pending = [s for s, p in hits.items() if p["entry_signals"]]
"""


def _bars() -> pd.DataFrame:
    """稳步上涨的日线（会命中「均线多头」入场信号）。"""
    idx = pd.bdate_range(START, periods=DAYS)
    closes = [10.0 + i * 0.1 for i in range(DAYS)]
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.05 for c in closes],
            "low": [c - 0.05 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * DAYS,
        },
        index=idx,
    )


def _provider() -> InMemoryProvider:
    return InMemoryProvider({SYMBOL: _bars()})


def _exec(code: str) -> dict:
    g: dict = {}
    exec(compile(code, "<test>", "exec"), g, g)
    return g


def _first_signal_index() -> int:
    """形态首次成立日在 K 线序列中的下标（直接用引擎判定）。"""
    bars = _bars()
    for i in range(len(bars)):
        hits = kline_patterns.detect_patterns(bars.iloc[: i + 1], include_standard=False)
        if any(h.key == "ma_bull_alignment" for h in hits):
            return i
    raise AssertionError("构造的序列应能命中「均线多头」")


# ---------------------------------------------------------------------------
# 静态检查
# ---------------------------------------------------------------------------

def test_pattern_strategy_passes_ast_gate():
    """形态策略不 import backend 模块即可通过沙箱静态检查。"""
    assert_safe(STRATEGY)
    assert_safe(SCREEN_STRATEGY)


# ---------------------------------------------------------------------------
# T+1 成交口径
# ---------------------------------------------------------------------------

def test_pattern_signal_trades_on_next_trading_day():
    """T 日识别 → T+1 交易日按收盘价成交。"""
    sig_idx = _first_signal_index()
    bars = _bars()
    expected_date = bars.index[sig_idx + 1].strftime("%Y-%m-%d")
    expected_price = float(bars["close"].iloc[sig_idx + 1])

    ctx = Context()
    result = run_backtest(
        ctx=ctx, provider=_provider(), user_globals=_exec(STRATEGY)
    )

    assert result.status == "success"
    buys = [t for t in result.trades if t.direction == "BUY"]
    assert len(buys) == 1, "边沿触发：同一形态持续存在时只应买入一次"

    trade = buys[0]
    assert trade.symbol == SYMBOL
    assert trade.date == expected_date, "成交日应为形态成立日的下一个交易日"
    assert trade.price == pytest.approx(expected_price), "应按 T+1 当日收盘价成交"
    assert trade.reason == "pattern entry"
    assert trade.detail.get("signal_date") == bars.index[sig_idx].strftime("%Y-%m-%d")
    assert "ma_bull_alignment" in trade.detail.get("patterns", [])


def test_no_trade_before_pattern_appears():
    """形态成立前不得有任何成交。"""
    sig_idx = _first_signal_index()
    bars = _bars()

    ctx = Context()
    result = run_backtest(
        ctx=ctx, provider=_provider(), user_globals=_exec(STRATEGY)
    )

    assert result.status == "success"
    for trade in result.trades:
        assert trade.date > bars.index[sig_idx].strftime("%Y-%m-%d")


def test_pattern_screen_rebalances_next_day():
    """横截面形态筛选：当日扫描结果在下一个交易日才调仓。"""
    sig_idx = _first_signal_index()
    bars = _bars()
    expected_date = bars.index[sig_idx + 1].strftime("%Y-%m-%d")

    ctx = Context()
    result = run_backtest(
        ctx=ctx, provider=_provider(), user_globals=_exec(SCREEN_STRATEGY)
    )

    assert result.status == "success"
    buys = [t for t in result.trades if t.direction == "BUY"]
    assert buys, "命中形态后应建仓"
    assert buys[0].date == expected_date
    assert buys[0].reason == "pattern screen"


def test_pattern_api_degrades_without_provider():
    """provider 未挂载时形态 API 安全降级，不影响回测主流程。"""
    ctx = Context()
    payload = ctx.pattern_hits(SYMBOL)
    assert payload["hits"] == []
    assert ctx.pattern_scan([SYMBOL])[SYMBOL]["hits"] == []
