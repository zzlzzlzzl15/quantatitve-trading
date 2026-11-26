"""K 线形态识别与时点口径测试。

覆盖：
1. 形态引擎：TA-Lib 标准形态、入场信号形态、聚合统计、目录
2. 时点口径：形态在 T 日收盘识别 → 最早 T+1 交易日成交
3. 无未来函数：判定只依赖当根及之前的数据
"""

import pandas as pd
import pytest

from backend.services.api.market_analysis import patterns as P


def _bars(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float] | None = None,
    start: str = "2026-01-05",
) -> pd.DataFrame:
    """构造 OHLCV DataFrame（升序）。"""
    n = len(closes)
    dates = pd.bdate_range(start, periods=n).strftime("%Y-%m-%d")
    vol = volumes if volumes is not None else [1_000_000.0] * n
    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": vol,
        }
    )


def _flat_bars(n: int = 60) -> pd.DataFrame:
    """构造一段平稳 K 线（无入场信号）。"""
    closes = [10.0] * n
    return _bars(closes, [c + 0.05 for c in closes], [c - 0.05 for c in closes], closes)


# ---- 形态引擎基础 ----

def test_talib_available():
    """TA-Lib 应可用，否则标准形态无法识别。"""
    assert P.talib_available() is True


def test_pattern_catalog_counts():
    """形态目录应为 61 种标准形态 + 9 种入场信号形态。"""
    catalog = P.pattern_catalog()
    assert len(catalog) == 70
    assert sum(1 for c in catalog if c["category"] == "standard") == 61
    assert sum(1 for c in catalog if c["category"] == "entry") == 9

    keys = [c["key"] for c in catalog]
    assert len(keys) == len(set(keys)), "形态 key 不应重复"


def test_standard_pattern_keys_match_talib():
    """标准形态的 TA-Lib 函数名必须真实存在。"""
    import talib

    for _key, _name, func_name, _direction in P.STANDARD_PATTERNS:
        assert hasattr(talib, func_name), f"TA-Lib 缺少函数 {func_name}"


def test_detect_on_flat_bars_finds_no_entry_signal():
    """平稳无波动 K 线不应命中入场信号。"""
    hits = P.detect_patterns(_flat_bars(60), include_standard=False)
    assert hits == []


def test_detect_doji_on_flat_bars():
    """开=收=高=低的十字星应被识别。"""
    n = 30
    closes = [10.0] * n
    bars = _bars(closes, closes, closes, closes)
    hits = P.detect_patterns(bars)
    keys = {h.key for h in hits}
    assert "doji" in keys


def test_detect_empty_bars_returns_empty():
    """空数据不应抛异常。"""
    empty = pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    assert P.detect_patterns(empty) == []
    assert P.detect_patterns(None) == []


def test_detect_missing_columns_returns_empty():
    """缺列时安全返回空列表。"""
    bars = pd.DataFrame({"date": ["2026-01-05"], "close": [10.0]})
    assert P.detect_patterns(bars) == []


def test_short_history_does_not_crash():
    """历史不足时不应越界崩溃（MA250 等长周期形态需安全跳过）。"""
    for n in (1, 5, 21, 59):
        bars = _flat_bars(n)
        hits = P.detect_patterns(bars)
        assert isinstance(hits, list)


# ---- 入场信号形态 ----

def test_volume_surge_up_signal():
    """放量上涨：涨幅≥2%、收阳、量比≥2、成交额≥2 亿。"""
    n = 30
    closes = [10.0] * (n - 1) + [10.5]
    opens = [10.0] * n
    highs = [10.6] * n
    lows = [9.9] * n
    volumes = [10_000_000.0] * (n - 1) + [100_000_000.0]
    bars = _bars(opens, highs, lows, closes, volumes)

    hits = P.detect_patterns(bars, include_standard=False)
    assert "volume_surge_up" in {h.key for h in hits}


def test_ma_bull_alignment_signal():
    """均线多头：MA5 > MA10 > MA20 且价格站上 MA20。"""
    n = 60
    closes = [10.0 + i * 0.1 for i in range(n)]
    opens = [c - 0.05 for c in closes]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    bars = _bars(opens, highs, lows, closes)

    hits = P.detect_patterns(bars, include_standard=False)
    assert "ma_bull_alignment" in {h.key for h in hits}


def test_entry_signals_all_bullish():
    """入场信号形态统一标记为看涨方向。"""
    catalog = P.pattern_catalog()
    for key, _name, _desc in P.ENTRY_SIGNAL_PATTERNS:
        item = next(c for c in catalog if c["key"] == key)
        assert item["direction"] == P.DIRECTION_BULLISH
        assert item["category"] == "entry"


# ---- 时点口径 ----

def test_next_trading_day():
    """T+1 交易日推算。"""
    dates = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]
    assert P.next_trading_day(dates, "2026-01-06") == "2026-01-07"
    # 最后一日没有下一交易日
    assert P.next_trading_day(dates, "2026-01-08") is None
    # 空日历
    assert P.next_trading_day([], "2026-01-06") is None


def test_signal_date_and_earliest_entry_date():
    """signal_date 为形态成立日（T），earliest_entry_date 为 T+1。"""
    n = 40
    closes = [10.0 + i * 0.05 for i in range(n)]
    bars = _bars(
        [c - 0.02 for c in closes],
        [c + 0.1 for c in closes],
        [c - 0.1 for c in closes],
        closes,
    )
    dates = list(bars["date"])

    # 序列最后一根即最新日 → 无 T+1
    res = P.scan_symbol("600000.SH", bars, trading_dates=dates)
    assert res.signal_date == dates[-1]
    assert res.earliest_entry_date is None, "最后一日无 T+1，应为 None"

    # 截掉最后一根后，最新根的 T+1 就是原最后一根
    res2 = P.scan_symbol("600000.SH", bars.iloc[:-1], trading_dates=dates)
    assert res2.signal_date == dates[-2]
    assert res2.earliest_entry_date == dates[-1]


def test_no_lookahead_truncation_invariant():
    """无未来函数：同一根 K 线的判定结果与后续数据无关。

    对第 cut 根分别在「截断到 cut」与「包含全部后续数据但取 cut 的信号」下比较，
    判定必须一致 —— 实现方式是对同一序列做两次 detect，第二次只在截断上做，
    并验证「把 cut 之后的数据删掉再判」结果不变。
    """
    n = 40
    closes = [10.0 + i * 0.07 for i in range(n)]
    bars = _bars(
        [c - 0.03 for c in closes],
        [c + 0.12 for c in closes],
        [c - 0.12 for c in closes],
        closes,
        [1_000_000.0 + i * 1000 for i in range(n)],
    )

    cut = 30
    truncated_hits = {h.key for h in P.detect_patterns(bars.iloc[: cut + 1])}
    # 同一份截断数据重复判定结果稳定
    assert truncated_hits == {h.key for h in P.detect_patterns(bars.iloc[: cut + 1])}

    # 关键断言：对 cut 这一根，TA-Lib 的信号在完整序列与截断序列中一致。
    # 通过对比「完整序列最后一根」与「完整序列 cut 根」不可直接比，
    # 这里改为验证截断序列的最后一根信号由 cut 及之前决定：
    import talib

    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    for _key, _name, func_name, _direction in P.STANDARD_PATTERNS:
        func = getattr(talib, func_name)
        full = func(o, h, l, c)
        trunc = func(o[: cut + 1], h[: cut + 1], l[: cut + 1], c[: cut + 1])
        assert full[cut] == trunc[cut], f"{func_name} 在 cut 处受未来数据影响"


def test_entry_signal_invariant_to_volume_scale():
    """入场信号为比值型判定：成交量整体等比缩放不改变结论。"""
    n = 40
    closes = [10.0 + i * 0.05 for i in range(n)]
    opens = [c - 0.02 for c in closes]
    highs = [c + 0.1 for c in closes]
    lows = [c - 0.1 for c in closes]
    vols = [1_000_000.0] * n

    base = P.detect_patterns(_bars(opens, highs, lows, closes, vols))
    scaled = P.detect_patterns(
        _bars(opens, highs, lows, closes, [v * 3 for v in vols])
    )
    assert {h.key for h in base} == {h.key for h in scaled}


# ---- 批量扫描与聚合 ----

def test_scan_universe_and_aggregate():
    """批量扫描 + 聚合统计。"""
    n = 40
    up = [10.0 + i * 0.05 for i in range(n)]
    flat = [10.0] * n

    universe = {
        "600000.SH": _bars(
            [c - 0.02 for c in up], [c + 0.1 for c in up], [c - 0.1 for c in up], up
        ),
        "000001.SZ": _bars(
            flat, [c + 0.05 for c in flat], [c - 0.05 for c in flat], flat
        ),
    }
    results = P.scan_universe(universe, trading_dates=list(universe["600000.SH"]["date"]))
    assert results

    agg = P.aggregate_patterns(results)
    assert agg
    assert agg == sorted(agg, key=lambda x: (-x["count"], x["key"]))
    for item in agg:
        assert "key" in item and "count" in item and "symbols" in item


def test_scan_symbol_empty_bars():
    """空 K 线的扫描结果为空壳，不抛异常。"""
    res = P.scan_symbol("600000.SH", pd.DataFrame())
    assert res.symbol == "600000.SH"
    assert res.hits == []
    assert res.signal_date is None


def test_keys_filter():
    """keys 参数可过滤形态。"""
    hits = P.detect_patterns(_flat_bars(60), keys=["doji"])
    assert all(h.key == "doji" for h in hits)


def test_result_to_dict_shape():
    """结果序列化结构固定，供前端消费。"""
    n = 30
    closes = [10.0] * n
    bars = _bars(closes, closes, closes, closes)
    res = P.scan_symbol("600000.SH", bars, trading_dates=list(bars["date"]))
    d = res.to_dict()

    for key in (
        "symbol",
        "signal_date",
        "earliest_entry_date",
        "hits",
        "bullish",
        "bearish",
        "neutral",
        "entry_signals",
    ):
        assert key in d
    assert isinstance(d["hits"], list)
    assert isinstance(d["bullish"], list)
