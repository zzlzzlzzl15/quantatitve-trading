"""K 线形态列在 compute_indicators 流水线里的集成验证。

与 tests/test_indicators_patterns.py 的分工:
  - 前者只测 patterns.py 的判定逻辑 (纯 polars 依赖, 无需数据源);
  - 本文件测**接线**: 形态列是否真的被流水线算出来、临时列是否清理、
    多标的分组是否正确 (QuantMind 是逐标的循环, 这里一次向量化算整池)。

需要完整后端依赖 (tivelo-* 等); 缺失时整体跳过, 保证纯形态单测仍可独立运行。
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

pipeline = pytest.importorskip(
    "app.indicators.pipeline",
    reason="需要完整后端依赖 (tivelo-*)",
)

from app.indicators import patterns  # noqa: E402

_START = date(2024, 1, 1)


def _bar(o: float, h: float, low: float, c: float, v: float, amount: float) -> dict:
    return {
        "open": o,
        "high": h,
        "low": low,
        "close": c,
        "volume": v,
        "amount": amount,
    }


def _rows(symbol: str, bars: list[dict]) -> list[dict]:
    return [
        {"symbol": symbol, "date": _START + timedelta(days=i), **row}
        for i, row in enumerate(bars)
    ]


def _panel() -> pl.DataFrame:
    """两个标的: A 在最后一根放量上涨, B 全程横盘。"""
    flat = [_bar(10.0, 10.0, 10.0, 10.0, 1000.0, 1e6) for _ in range(20)]
    surge = _bar(10.0, 10.6, 9.9, 10.5, 3000.0, 3e8)
    flat_only = [_bar(10.0, 10.0, 10.0, 10.0, 1000.0, 1e6) for _ in range(21)]
    return pl.DataFrame(_rows("A", [*flat, surge]) + _rows("B", flat_only))


def test_compute_indicators_produces_pattern_columns():
    """流水线产出全部形态列, 且中间列用完即丢。"""
    out = pipeline.compute_indicators(_panel(), needed=set(patterns.PATTERN_COLUMNS))

    for column in patterns.PATTERN_COLUMNS:
        assert column in out.columns, column
        assert out[column].dtype == pl.Float64, column
    assert not [c for c in out.columns if c.startswith("_pat_")], "中间列未清理"


def test_pattern_columns_are_grouped_by_symbol():
    """多标的分组正确: 只有 A 的最后一根命中放量上涨。"""
    out = pipeline.compute_indicators(_panel(), needed={"pat_volume_surge_up"})

    a = out.filter(pl.col("symbol") == "A")["pat_volume_surge_up"].to_list()
    b = out.filter(pl.col("symbol") == "B")["pat_volume_surge_up"].to_list()

    assert a == [0.0] * 20 + [1.0]
    assert b == [0.0] * 21


def test_needed_subset_matches_full_computation():
    """按需计算与全量计算的形态列数值逐位一致。"""
    df = _panel()
    full = pipeline.compute_indicators(df)
    subset = pipeline.compute_indicators(df, needed=set(patterns.PATTERN_COLUMNS))

    for column in patterns.PATTERN_COLUMNS:
        assert full[column].to_list() == subset[column].to_list(), column


# ── 自定义信号接线 (形态列 → 信号定义 → 表达式 → 注入) ───────────


def test_pattern_columns_are_in_signal_allowlist():
    """形态列注册为因子后自动进入自定义信号字段白名单 (无需改白名单)。"""
    from app.strategy import custom_signals

    allowed = custom_signals.allowed_fields()
    for column in patterns.PATTERN_COLUMNS:
        assert column in allowed, column


def test_pattern_signal_seeds_are_valid():
    """9 个形态信号种子全部通过信号定义校验, 且依赖声明指向形态列。"""
    from app.strategy import custom_signals
    from app.strategy.signal_seeds import SEED_SIGNALS

    seeds = [s for s in SEED_SIGNALS if s["id"].startswith("pat_")]
    assert len(seeds) == len(patterns.PATTERN_COLUMNS)

    for seed in seeds:
        custom_signals.validate(seed)
        deps = custom_signals._definition_dependencies(seed)
        assert seed["id"] in deps, seed["id"]


def test_pattern_signal_fires_only_on_first_day_of_pattern():
    """形态是状态: 连续多日成立时, 信号只在首次成立日触发 (边沿触发)。"""
    from app.strategy import custom_signals
    from app.strategy.signal_seeds import SEED_SIGNALS

    rows = []
    for i in range(25):
        price = 10.0 + i * 0.1
        rows.append(
            {
                "symbol": "A",
                "date": _START + timedelta(days=i),
                "open": price,
                "high": price,
                "low": price,
                "close": price,
                "volume": 1000.0,
                "amount": 1e6,
            }
        )
    out = pipeline.compute_indicators(pl.DataFrame(rows))

    seed = next(s for s in SEED_SIGNALS if s["id"] == "pat_ma_bull_alignment")
    exprs = custom_signals.build_expressions([seed])
    injected = custom_signals.inject(out, exprs)

    pattern_hits = [i for i, v in enumerate(out["pat_ma_bull_alignment"].to_list()) if v]
    signal_hits = [i for i, v in enumerate(injected[seed["column"]].to_list()) if v]

    assert len(pattern_hits) > 1, "样本应让形态连续成立多日"
    assert signal_hits == [pattern_hits[0]], "信号只在形态首次成立日触发"
