"""K 线形态列 (app/indicators/patterns.py) 的数值断言、边界与降级路径。

验证矩阵口径 (CONTRIBUTING §9): 指标类改动需
  - 固定样本上的数值断言 (每个形态各一条命中/未命中样本);
  - 历史边界 (窗口未满 ⇒ 不成立);
  - 降级路径 (amount 缺失 ⇒ fail-closed);
  - 无未来函数 (T 日判定不受 T 之后数据影响)。

本模块只测 patterns.py 自身的判定逻辑 (纯 polars 依赖), 不引入 pipeline,
以便在缺少数据源依赖的环境里也能运行。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from app.factors import registry
from app.indicators import patterns

_START = date(2024, 1, 1)


def _bar(
    o: float,
    h: float,
    low: float,
    c: float,
    v: float,
    amount: float | None = None,
) -> dict:
    """单根 K 线。amount 缺省取 close*volume*100 (足够大的成交额)。"""
    return {
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": v,
        "amount": c * v * 100 if amount is None else amount,
    }


def _flat(n: int, price: float = 10.0, volume: float = 1000.0) -> list[dict]:
    """n 根无实体 K 线 (横盘)。"""
    return [_bar(price, price, price, price, volume) for _ in range(n)]


def _frame(rows: list[dict], symbol: str = "600000") -> pl.DataFrame:
    """按时间升序构造单标的日线帧。"""
    return pl.DataFrame(
        [
            {"symbol": symbol, "date": _START + timedelta(days=i), **row}
            for i, row in enumerate(rows)
        ]
    )


def _with_base_columns(df: pl.DataFrame) -> pl.DataFrame:
    """补上形态依赖的基础指标列 (与 pipeline Pass 1/4 同口径)。"""
    return df.with_columns(
        [
            pl.col("close").shift(1).over("symbol").alias("prev_close"),
            pl.col("close").rolling_mean(5).over("symbol").alias("ma5"),
            pl.col("close").rolling_mean(10).over("symbol").alias("ma10"),
            pl.col("close").rolling_mean(20).over("symbol").alias("ma20"),
            pl.col("volume").rolling_mean(5).over("symbol").alias("vol_ma5"),
        ]
    )


def _compute(df: pl.DataFrame) -> pl.DataFrame:
    """在裸 OHLCV 帧上计算全部形态列 (复刻 pipeline 的形态 Pass)。"""
    df = _with_base_columns(df)
    df = df.with_columns(
        [expr.alias(name) for name, expr in patterns.build_mid_exprs().items()]
    )
    exprs = patterns.build_pattern_exprs(amount_available="amount" in df.columns)
    return df.with_columns([expr.alias(name) for name, expr in exprs.items()])


def _hits(df: pl.DataFrame, column: str) -> list[float]:
    """形态列取值列表 (1.0 = 形态成立, 0.0 = 不成立)。"""
    return df[column].to_list()


# ── 逐形态命中判定 (固定样本数值断言) ────────────────────────────


def test_volume_surge_up_requires_change_volume_ratio_and_turnover():
    """放量上涨: 涨跌幅≥2% + 收阳 + 量比≥2 + 成交额≥2亿, 四者同时成立。"""
    rows = _flat(20)
    rows.append(_bar(10.0, 10.6, 9.9, 10.5, 3000.0, amount=3e8))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_volume_surge_up") == [0.0] * 20 + [1.0]


def test_volume_surge_up_false_when_turnover_below_threshold():
    """成交额不足 2 亿 ⇒ 不成立 (其余条件均满足)。"""
    rows = _flat(20)
    rows.append(_bar(10.0, 10.6, 9.9, 10.5, 3000.0, amount=1.9e8))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_volume_surge_up") == [0.0] * 21


def test_volume_surge_up_false_when_change_below_two_percent():
    """涨幅不足 2% ⇒ 不成立 (其余条件均满足)。"""
    rows = _flat(20)
    rows.append(_bar(10.0, 10.1, 9.9, 10.1, 3000.0, amount=3e8))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_volume_surge_up") == [0.0] * 21


def test_volume_surge_up_false_when_base_volume_is_zero():
    """前 5 日均量为 0 (长期停牌) ⇒ 量比不可判定 ⇒ 不成立 (不得因 inf 误判)。"""
    rows = [_bar(10.0, 10.0, 10.0, 10.0, 0.0) for _ in range(20)]
    rows.append(_bar(10.0, 10.6, 9.9, 10.5, 3000.0, amount=3e8))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_volume_surge_up") == [0.0] * 21


def test_ma_bull_alignment_hit_on_rising_series():
    """均线多头: MA5 > MA10 > MA20 且收盘站上 MA20。"""
    rows = [_bar(p, p, p, p, 1000.0) for p in (10.0 + i for i in range(21))]
    out = _compute(_frame(rows))

    assert _hits(out, "pat_ma_bull_alignment") == [0.0] * 20 + [1.0]


def test_ma_bull_alignment_false_on_falling_series():
    """均线空头排列 ⇒ 不成立。"""
    rows = [_bar(p, p, p, p, 1000.0) for p in (30.0 - i for i in range(21))]
    out = _compute(_frame(rows))

    assert _hits(out, "pat_ma_bull_alignment") == [0.0] * 21


def test_breakout_ma_hit_on_cross_up_with_volume():
    """突破均线: 收盘上穿 MA20 (前一日在 MA20 之下) 且量比≥1.5。"""
    rows = _flat(20)
    rows.append(_bar(10.5, 11.0, 10.4, 11.0, 2000.0))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_breakout_ma") == [0.0] * 20 + [1.0]


def test_platform_breakout_hit_on_range_break():
    """突破平台: 20 日横盘后放量突破区间上沿。"""
    rows = [_bar(9.8, 10.0, 9.7, 9.8, 1000.0) for _ in range(21)]
    rows.append(_bar(10.1, 11.0, 10.0, 10.5, 3000.0))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_platform_breakout") == [0.0] * 21 + [1.0]


def test_backtrace_ma250_hit_on_pullback_to_annual_line():
    """回踩年线: 回踩 MA250 (2% 容差) 后收阳并站上 MA250。"""
    rows = _flat(250)
    rows.append(_bar(10.0, 10.6, 10.1, 10.5, 1000.0))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_backtrace_ma250") == [0.0] * 250 + [1.0]


def test_low_backtrace_increase_hit_at_range_bottom():
    """低位回调上涨: 处于 60 日区间低位 (≤35%) 后放量收阳。"""
    rows = [_bar(20.0, 20.1, 19.9, 20.0, 1000.0) for _ in range(30)]
    rows += [_bar(10.0, 10.1, 9.9, 10.0, 1000.0) for _ in range(29)]
    rows.append(_bar(9.9, 10.2, 9.8, 10.1, 3000.0))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_low_backtrace_increase") == [0.0] * 59 + [1.0]


def test_turtle_trade_hit_on_donchian_breakout():
    """海龟交易法则: 突破 20 日唐奇安上轨。"""
    rows = [_bar(9.5, 10.0, 9.4, 9.5, 1000.0) for _ in range(21)]
    rows.append(_bar(9.9, 11.0, 9.8, 10.8, 2000.0))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_turtle_trade") == [0.0] * 21 + [1.0]


def test_high_tight_flag_hit_on_three_limit_ups_then_shrink():
    """高位紧旗: 连续 3 日涨停后缩量整理。"""
    rows = _flat(22)
    rows += [_bar(10.0, 11.0, 10.0, 11.0, 5000.0) for _ in range(2)]
    rows.append(_bar(10.0, 11.0, 10.0, 11.0, 100.0))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_high_tight_flag") == [0.0] * 24 + [1.0]


def test_climax_limitdown_hit_on_volume_expansion():
    """放量跌停: 跌停且成交量较 5 日均量放大 2 倍以上。"""
    rows = _flat(20)
    rows.append(_bar(10.0, 10.0, 9.0, 9.0, 3000.0))
    out = _compute(_frame(rows))

    assert _hits(out, "pat_climax_limitdown") == [0.0] * 20 + [1.0]


# ── 历史边界 ───────────────────────────────────────────────────


def test_all_patterns_false_when_history_insufficient():
    """窗口未满时全部形态不成立 (不因 null 误判为命中)。"""
    rows = _flat(3)
    out = _compute(_frame(rows))

    for column in patterns.PATTERN_COLUMNS:
        assert _hits(out, column) == [0.0, 0.0, 0.0], column


def test_patterns_suppressed_before_min_history_bars():
    """历史不足 21 根时形态不判定 (对齐 QuantMind 入口早退条件)。"""
    rows = _flat(19)
    rows.append(_bar(10.0, 10.6, 9.9, 10.5, 3000.0, amount=3e8))
    out = _compute(_frame(rows))

    # 第 20 根: 涨幅/量比/成交额均满足, 但历史仅 20 根 ⇒ 抑制
    assert _hits(out, "pat_volume_surge_up") == [0.0] * 20


def test_patterns_need_enough_history_for_long_window():
    """回踩年线需要 250 根历史: 窗口未满时不成立, 刚好填满时可判定。"""
    insufficient = _flat(248)
    insufficient.append(_bar(10.0, 10.6, 10.1, 10.5, 1000.0))
    out_short = _compute(_frame(insufficient))

    assert _hits(out_short, "pat_backtrace_ma250") == [0.0] * 249

    sufficient = _flat(249)
    sufficient.append(_bar(10.0, 10.6, 10.1, 10.5, 1000.0))
    out_long = _compute(_frame(sufficient))

    assert _hits(out_long, "pat_backtrace_ma250") == [0.0] * 249 + [1.0]


# ── 降级路径 ───────────────────────────────────────────────────


def test_volume_surge_up_false_when_amount_column_missing():
    """amount 缺失 ⇒ 成交额门槛无法判定 ⇒ fail-closed 恒 0, 不静默放宽。"""
    rows = _flat(20)
    rows.append(_bar(10.0, 10.6, 9.9, 10.5, 3000.0))
    df = _frame(rows).drop("amount")
    out = _compute(df)

    assert _hits(out, "pat_volume_surge_up") == [0.0] * 21


def test_other_patterns_unaffected_when_amount_column_missing():
    """amount 缺失只影响依赖成交额门槛的形态, 其余形态判定不变。"""
    rows = [_bar(9.5, 10.0, 9.4, 9.5, 1000.0) for _ in range(21)]
    rows.append(_bar(9.9, 11.0, 9.8, 10.8, 2000.0))
    df = _frame(rows).drop("amount")
    out = _compute(df)

    assert _hits(out, "pat_turtle_trade") == [0.0] * 21 + [1.0]


# ── 无未来函数 ─────────────────────────────────────────────────


def test_pattern_result_ignores_bars_after_signal_date():
    """T 日的形态判定不受 T 之后数据影响 (只使用截至 T 日收盘的信息)。"""
    base_rows = [_flat(30)[0] for _ in range(30)]
    base_rows = [_bar(10.0, 10.1, 9.9, 10.0, 1000.0) for _ in range(30)]
    future_rows = [
        _bar(100.0, 120.0, 80.0, 110.0, 999999.0) for _ in range(5)
    ]
    short = _compute(_frame(base_rows))
    long = _compute(_frame(base_rows + future_rows))

    for column in patterns.PATTERN_COLUMNS:
        assert _hits(short, column) == _hits(long, column)[: len(base_rows)], column


# ── 契约一致性 (与因子注册表 / 依赖闭包) ────────────────────────


def test_registered_pattern_factors_match_module_columns():
    """注册表里的形态因子 id 与 patterns.PATTERN_COLUMNS 完全一致 (防漂移)。"""
    registered = {
        spec.id for spec in registry.all_factors() if spec.id.startswith("pat_")
    }
    assert registered == set(patterns.PATTERN_COLUMNS)


def test_registered_pattern_factors_declare_warmup():
    """每个形态因子都声明了足够的预热窗口。"""
    for column in patterns.PATTERN_COLUMNS:
        spec = registry.get_factor(column)
        assert spec is not None, column
        assert spec.kind == "base", column
        assert spec.warmup_bars >= 1, column
        assert "pattern" in spec.tags, column


def test_registered_warmup_matches_pattern_module():
    """注册表的 warmup_bars 与 patterns.PATTERN_WARMUP_BARS 一致 (防漂移)。"""
    assert set(patterns.PATTERN_WARMUP_BARS) == set(patterns.PATTERN_COLUMNS)
    for column, warmup in patterns.PATTERN_WARMUP_BARS.items():
        spec = registry.get_factor(column)
        assert spec is not None, column
        assert spec.warmup_bars == warmup, column


def test_pattern_dependencies_only_reference_known_columns():
    """形态依赖只引用已声明的中间列或基础指标列。"""
    base = {"prev_close", "ma5", "ma10", "ma20", "vol_ma5"}
    for column, deps in patterns.PATTERN_DEPENDENCIES.items():
        assert column in patterns.PATTERN_COLUMNS
        assert deps <= (base | patterns.MID_COLUMNS), column


def test_mid_dependencies_are_subset_of_mid_columns():
    """中间列依赖只引用中间列。"""
    for name, deps in patterns.MID_DEPENDENCIES.items():
        assert name in patterns.MID_COLUMNS
        assert deps <= patterns.MID_COLUMNS


def test_registry_labels_match_pattern_module():
    """注册表镜像 patterns.ENTRY_SIGNAL_PATTERNS 的中文名与判定说明 (防漂移)。"""
    for key, name, desc in patterns.ENTRY_SIGNAL_PATTERNS:
        spec = registry.get_factor(patterns.pattern_column(key))
        assert spec is not None, key
        assert spec.label == name, key
        assert spec.formula_text == desc, key
        assert spec.group == "形态", key


@pytest.mark.parametrize("column", patterns.PATTERN_COLUMNS)
def test_pattern_column_is_float_dummy(column: str):
    """形态列类型为 Float64 (0/1 哑变量, 供信号表达式与矩阵直接使用)。"""
    rows = _flat(30)
    out = _compute(_frame(rows))
    assert out[column].dtype == pl.Float64
    assert set(out[column].to_list()) <= {0.0, 1.0}
