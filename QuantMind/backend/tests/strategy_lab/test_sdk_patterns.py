"""Tests for Context pattern API — 形态识别与 T+1 成交口径。

覆盖：
1. ``ctx.pattern_hits()`` —— 识别结构、时点口径（只用截至 T 日收盘的数据）
2. ``ctx.pattern_scan()`` —— 批量识别
3. ``ctx.pattern_entry() / pattern_exit()`` —— T 日识别 → T+1 交易日下单
"""

import pandas as pd
import pytest

from backend.services.engine.strategy_lab.engine.data_provider import InMemoryProvider
from backend.services.engine.strategy_lab.sdk.context import Context
from backend.services.engine.strategy_lab.sdk.position import Position


SYMBOL = "SH600036"


def _rising_bars(n: int = 60, with_amount: bool = False) -> pd.DataFrame:
    """构造一段稳步上涨的日线（会命中「均线多头」等入场信号）。"""
    dates = pd.bdate_range("2026-01-05", periods=n)
    closes = [10.0 + i * 0.1 for i in range(n)]
    data = {
        "open": [c - 0.05 for c in closes],
        "high": [c + 0.10 for c in closes],
        "low": [c - 0.10 for c in closes],
        "close": closes,
        "volume": [1_000_000.0 + i * 1000 for i in range(n)],
    }
    if with_amount:
        # amount 单位为「万元」（QuantDB 口径），成交额 = close*volume/1e4
        data["amount"] = [c * v / 1e4 for c, v in zip(closes, data["volume"])]
    return pd.DataFrame(data, index=dates)


def _ctx(data: pd.DataFrame) -> Context:
    c = Context()
    c._attach(InMemoryProvider({SYMBOL: data}), broker=None, cash=1_000_000)
    return c


# ---------------------------------------------------------------------------
# pattern_hits
# ---------------------------------------------------------------------------

def test_pattern_hits_returns_expected_shape():
    """返回结构与 SymbolPatterns.to_dict() 一致。"""
    data = _rising_bars()
    c = _ctx(data)
    c._set_today(data.index[-1])

    payload = c.pattern_hits(SYMBOL)
    for key in (
        "symbol",
        "signal_date",
        "hits",
        "bullish",
        "bearish",
        "neutral",
        "entry_signals",
    ):
        assert key in payload
    assert payload["symbol"] == SYMBOL
    assert payload["signal_date"] == data.index[-1].strftime("%Y-%m-%d")
    assert isinstance(payload["hits"], list)
    assert payload["hits"], "稳步上涨应命中至少一种形态"


def test_pattern_hits_detects_entry_signals():
    """稳步上涨应命中「均线多头」入场信号。"""
    data = _rising_bars()
    c = _ctx(data)
    c._set_today(data.index[-1])

    payload = c.pattern_hits(SYMBOL, include_standard=False)
    assert "ma_bull_alignment" in payload["entry_signals"]


def test_pattern_hits_uses_only_data_up_to_today():
    """时点口径：T 日判定不得使用 T 之后的数据。

    用「完整数据」和「只到 T 日的数据」分别构造 provider，判定结果必须一致。
    """
    data = _rising_bars(80)
    cut = 50
    today = data.index[cut]

    c_full = _ctx(data)
    c_full._set_today(today)
    c_cut = _ctx(data.iloc[: cut + 1])
    c_cut._set_today(today)

    assert (
        c_full.pattern_hits(SYMBOL)["hits"]
        == c_cut.pattern_hits(SYMBOL)["hits"]
    )


def test_pattern_hits_keys_filter():
    """keys 参数可过滤形态。"""
    data = _rising_bars()
    c = _ctx(data)
    c._set_today(data.index[-1])

    payload = c.pattern_hits(SYMBOL, keys=["ma_bull_alignment"])
    assert payload["hits"]
    assert all(h["key"] == "ma_bull_alignment" for h in payload["hits"])


def test_pattern_hits_without_provider_returns_empty_shell():
    """provider 未挂载时安全降级，不抛异常。"""
    c = Context()
    payload = c.pattern_hits(SYMBOL)
    assert payload["hits"] == []
    assert payload["signal_date"] is None


def test_pattern_hits_unknown_symbol_returns_empty_shell():
    """标的无数据时返回空壳。"""
    data = _rising_bars()
    c = _ctx(data)
    c._set_today(data.index[-1])

    payload = c.pattern_hits("SH999999")
    assert payload["hits"] == []


def test_pattern_hits_supports_amount_column():
    """provider 带 amount 列时走成交额口径，不应崩溃。"""
    data = _rising_bars(with_amount=True)
    c = _ctx(data)
    c._set_today(data.index[-1])

    payload = c.pattern_hits(SYMBOL)
    assert payload["symbol"] == SYMBOL


# ---------------------------------------------------------------------------
# pattern_scan
# ---------------------------------------------------------------------------

def test_pattern_scan_batch():
    """批量识别返回 {symbol: 结果}。"""
    data = _rising_bars()
    other = _rising_bars()
    c = Context()
    c._attach(
        InMemoryProvider({SYMBOL: data, "SH600519": other}),
        broker=None,
        cash=1_000_000,
    )
    c._set_today(data.index[-1])

    out = c.pattern_scan([SYMBOL, "SH600519"])
    assert set(out) == {SYMBOL, "SH600519"}
    assert out[SYMBOL]["hits"]


def test_pattern_scan_uses_universe_when_symbols_omitted():
    """symbols 省略时回落到 ctx.universe。"""
    data = _rising_bars()
    c = _ctx(data)
    c.universe = [SYMBOL]
    c._set_today(data.index[-1])

    out = c.pattern_scan()
    assert set(out) == {SYMBOL}


# ---------------------------------------------------------------------------
# pattern_entry / pattern_exit —— T+1 成交口径
# ---------------------------------------------------------------------------

def test_pattern_entry_defers_order_to_next_trading_day():
    """T 日命中只登记信号；T+1 交易日才下单。"""
    data = _rising_bars(60)
    c = _ctx(data)

    # T 日
    c._set_today(data.index[40])
    assert c.pattern_entry(SYMBOL, weight=0.4) is False
    assert c._drain_orders() == [], "T 日不应下单"

    # T+1 交易日
    c._set_today(data.index[41])
    assert c.pattern_entry(SYMBOL, weight=0.4) is True
    orders = c._drain_orders()
    assert len(orders) == 1
    o = orders[0]
    assert o.side == "buy"
    assert o.symbol == SYMBOL
    assert o.weight == 0.4
    assert o.detail["signal_date"] == data.index[40].strftime("%Y-%m-%d")
    assert o.detail["patterns"], "detail 应记录触发形态"


def test_pattern_entry_no_signal_never_orders():
    """无形态命中时不应产生任何订单。"""
    n = 60
    closes = [10.0] * n
    dates = pd.bdate_range("2026-01-05", periods=n)
    flat = pd.DataFrame(
        {
            "open": closes,
            "high": [c + 0.01 for c in closes],
            "low": [c - 0.01 for c in closes],
            "close": closes,
            "volume": [1_000_000.0] * n,
        },
        index=dates,
    )
    c = _ctx(flat)
    c._set_today(dates[40])
    assert c.pattern_entry(SYMBOL) is False
    c._set_today(dates[41])
    assert c.pattern_entry(SYMBOL) is False
    assert c._drain_orders() == []


def test_pattern_entry_with_explicit_keys():
    """指定 keys 时只认这些形态。"""
    data = _rising_bars(60)
    c = _ctx(data)

    c._set_today(data.index[40])
    assert c.pattern_entry(SYMBOL, keys=["ma_bull_alignment"]) is False
    c._set_today(data.index[41])
    assert c.pattern_entry(SYMBOL, keys=["ma_bull_alignment"]) is True
    o = c._drain_orders()[0]
    assert o.detail["patterns"] == ["ma_bull_alignment"]


def test_pattern_exit_sells_on_next_trading_day():
    """出场同样遵循 T+1：T 日登记、T+1 清仓。

    这里直接桩掉 ``pattern_hits``，只验证 T+1 延迟机制本身，
    不依赖具体 TA-Lib 形态的构造。
    """
    data = _rising_bars(60)
    c = _ctx(data)

    bearish = {
        "symbol": SYMBOL,
        "signal_date": None,
        "hits": [
            {
                "key": "three_black_crows",
                "name": "三只乌鸦",
                "direction": "bearish",
                "category": "standard",
                "strength": 100,
                "description": "stub",
            }
        ],
        "bullish": [],
        "bearish": ["three_black_crows"],
        "neutral": [],
        "entry_signals": [],
    }
    c.pattern_hits = lambda *a, **k: bearish  # type: ignore[method-assign]

    c._set_today(data.index[40])
    assert c.pattern_exit(SYMBOL) is False
    assert c._drain_orders() == [], "T 日不应卖出"

    c._set_today(data.index[41])
    assert c.pattern_exit(SYMBOL) is True
    orders = c._drain_orders()
    assert len(orders) == 1
    o = orders[0]
    assert o.side == "sell"
    assert o.all is True
    assert o.symbol == SYMBOL


def test_pattern_entry_edge_triggered_no_repeat_orders():
    """形态持续存在时不应反复下单（边沿触发）。"""
    data = _rising_bars(60)
    c = _ctx(data)

    c._set_today(data.index[40])
    c.pattern_entry(SYMBOL)
    c._set_today(data.index[41])
    c.pattern_entry(SYMBOL)
    assert len(c._drain_orders()) == 1

    # 形态仍在 → 不再产生新订单
    c._set_today(data.index[42])
    c.pattern_entry(SYMBOL)
    c._set_today(data.index[43])
    c.pattern_entry(SYMBOL)
    assert c._drain_orders() == []


def test_pattern_entry_skipped_when_holding_position():
    """已有持仓时不再登记新的入场信号。"""
    data = _rising_bars(60)
    c = _ctx(data)
    c._set_position(Position(symbol=SYMBOL, qty=100, cost=1000, market_value=1000))

    c._set_today(data.index[40])
    assert c.pattern_entry(SYMBOL) is False
    c._set_today(data.index[41])
    assert c.pattern_entry(SYMBOL) is False
    assert c._drain_orders() == []


def test_pattern_pending_is_reset_per_symbol():
    """挂起信号按标的隔离，互不干扰。"""
    data = _rising_bars(60)
    other = _rising_bars(60)
    c = Context()
    c._attach(
        InMemoryProvider({SYMBOL: data, "SH600519": other}),
        broker=None,
        cash=1_000_000,
    )

    c._set_today(data.index[40])
    c.pattern_entry(SYMBOL)
    c._drain_orders()

    c._set_today(data.index[41])
    c.pattern_entry(SYMBOL)
    orders = c._drain_orders()
    assert len(orders) == 1
    assert orders[0].symbol == SYMBOL
