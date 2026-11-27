"""matrix_native 策略直读形态列 (pat_*) —— 依赖解析 / 面板计算 / 矩阵注入。

形态列 (pat_*) 判定依赖滚动窗口与成交额量比, 无法在矩阵上现算
(matrix_feature 只现算 ma/ema/rsi/momentum 一族), 因此必须走三段:

  1. 依赖解析 (_resolve_base_columns / _resolve_matrix_native):
     形态列 ⇒ 面板加载 amount (成交额门槛) + 预热根数 (PATTERN_WARMUP_BARS)
     + 形态列本身进矩阵字段。
  2. 面板计算 (_prepare_panel_features): matrix_native 路径同样走
     compute_indicators —— 旧实现只在 polars 路径算指标, 形态列在矩阵
     原生路径恒缺失。
  3. 矩阵注入 (load_market_data_matrix_for_backtest): 面板算出的形态列
     随矩阵字段载入, 供 matrix_feature() 直读。
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from app.backtest.engine import BacktestEngine
from app.backtest.matrix import matrix_feature
from app.backtest.strategy import (
    StrategyDependencyResolver,
    _resolve_base_columns,
)
from app.indicators.pipeline import (
    PATTERN_BASE_COLUMNS,
    PATTERN_WARMUP_BARS,
    compute_indicators,
)
from app.strategy.engine import StrategyDef
from app.tickflow.repository import DataStore, KlineRepository

SYMS = ["600001.SH", "000002.SZ"]
DAYS = 120
START = date(2025, 1, 2)
PATTERN = "pat_ma_bull_alignment"


@pytest.fixture(autouse=True)
def _migrated_signal_definitions(monkeypatch):
    """注入迁移自内置的信号定义 (信号列定义驱动, 不依赖运行目录)。"""
    from tests.signal_seeds import install_pipeline_caches

    install_pipeline_caches(monkeypatch)


class _MatrixStrategy:
    """最小 matrix_strategy 替身: 只声明依赖字段与预热根数。"""

    def __init__(self, fields: frozenset[str], warmup: int = 21):
        self._fields = fields
        self._warmup = warmup

    def required_fields(self) -> frozenset[str]:
        return self._fields

    def required_warmup_bars(self, params: dict) -> int:
        return self._warmup


def _strategy(
    *,
    backend: str = "matrix_native",
    required_features: frozenset[str] = frozenset(),
    matrix_fields: frozenset[str] = frozenset(),
    warmup: int = 21,
) -> StrategyDef:
    return StrategyDef(
        meta={"id": "pattern_probe", "label": "形态探针"},
        basic_filter={},
        entry_signals=[],
        exit_signals=[],
        stop_loss=None,
        trailing_stop=None,
        trailing_take_profit_activate=None,
        trailing_take_profit_drawdown=None,
        max_hold_days=None,
        filter_fn=None,
        filter_history_fn=None,
        lookback_days=21,
        source="custom",
        required_features=required_features,
        execution_backend=backend,
        matrix_strategy=(
            _MatrixStrategy(matrix_fields, warmup) if backend == "matrix_native" else None
        ),
    )


def _price_rows() -> list[dict]:
    """持续上行序列: MA5 > MA10 > MA20 且收盘站上 MA20 ⇒ 均线多头形态成立。"""
    rows: list[dict] = []
    for i in range(DAYS):
        day = START + timedelta(days=i)
        for j, sym in enumerate(SYMS):
            close = 10.0 + 0.05 * i + j * 3.0
            volume = 1_000_000.0 + j * 1000.0
            rows.append({
                "symbol": sym,
                "date": day,
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.12,
                "close": close,
                "volume": volume,
                "amount": close * volume,
                "raw_close": close,
                "raw_high": close + 0.10,
                "raw_low": close - 0.12,
                "turnover_rate": 1.0 + j * 0.1,
                "consecutive_limit_ups": 0,
                "consecutive_limit_downs": 0,
                "quote_ts": 0,
            })
    return rows


@pytest.fixture()
def data_dir(tmp_path: Path) -> Path:
    """按生产布局写 enriched: kline_daily_enriched/date=YYYY-MM-DD/part.parquet。"""
    enriched = tmp_path / "kline_daily_enriched"
    frame = pl.DataFrame(_price_rows())
    for day_frame in frame.partition_by("date", maintain_order=True):
        part = enriched / f"date={day_frame['date'][0].isoformat()}"
        part.mkdir(parents=True)
        day_frame.write_parquet(part / "part.parquet")
    inst = tmp_path / "instruments"
    inst.mkdir(parents=True)
    pl.DataFrame({
        "symbol": SYMS,
        "name": ["甲", "乙"],
        "limit_up": [13.0, 13.0],
        "limit_down": [9.0, 9.0],
        "listing_date": ["2000-01-04", "2000-01-04"],
        "total_shares": [1e9, 1e9],
        "float_shares": [1e9, 1e9],
    }).write_parquet(inst / "part.parquet")
    return tmp_path


def _engine(data_dir: Path) -> BacktestEngine:
    return BacktestEngine(KlineRepository(DataStore(data_dir)))


# ── 1. 依赖解析 ───────────────────────────────────────────────


def test_base_columns_load_amount_for_patterns():
    """形态列需要成交额门槛 ⇒ 面板必须带出 amount, 否则 pat_volume_surge_up 恒 0。"""
    plan_columns = _resolve_base_columns({"pat_volume_surge_up"})
    assert PATTERN_BASE_COLUMNS <= plan_columns
    assert "amount" in plan_columns


def test_base_columns_without_patterns_stay_lean():
    """不引用形态列时不额外加载 amount (避免无谓的列投影)。"""
    assert "amount" not in _resolve_base_columns({"close", "ma20"})


def test_matrix_native_plan_carries_pattern_columns():
    """形态列 ⇒ 进矩阵字段 + 进 indicator_columns (面板上算) + 预热根数足够。"""
    plan = StrategyDependencyResolver()._resolve_matrix_native(
        _strategy(matrix_fields=frozenset({PATTERN})),
        params={},
        basic_filter={},
        overrides={},
    )
    assert PATTERN in plan.matrix_columns
    assert PATTERN in plan.indicator_columns
    assert plan.warmup_bars >= PATTERN_WARMUP_BARS[PATTERN]
    assert "amount" in plan.base_columns


def test_matrix_native_warmup_covers_long_window_pattern():
    """MA250 回踩形态需要 250 根历史 ⇒ 预热不足时形态在区间起始处恒不成立。"""
    plan = StrategyDependencyResolver()._resolve_matrix_native(
        _strategy(matrix_fields=frozenset({"pat_backtrace_ma250"})),
        params={},
        basic_filter={},
        overrides={},
    )
    assert plan.warmup_bars >= PATTERN_WARMUP_BARS["pat_backtrace_ma250"] >= 250


def test_polars_path_resolves_pattern_columns():
    """polars_expr 路径: 形态列进 indicator_columns, 且面板带出 amount。"""
    plan = StrategyDependencyResolver().resolve(
        _strategy(
            backend="polars_expr",
            required_features=frozenset({PATTERN}),
        ),
        params={},
        basic_filter={},
        entry_signals=[],
        exit_signals=[],
    )
    assert PATTERN in plan.indicator_columns
    assert "amount" in plan.base_columns


# ── 2. 面板计算 ───────────────────────────────────────────────


def test_prepare_panel_features_computes_pattern_columns_for_matrix_native():
    """matrix_native 面板同样算指标列 (旧实现只在 polars 路径算)。"""
    engine = BacktestEngine(repo=None)  # type: ignore[arg-type]
    panel = pl.DataFrame(_price_rows())
    plan = StrategyDependencyResolver()._resolve_matrix_native(
        _strategy(matrix_fields=frozenset({PATTERN})),
        params={},
        basic_filter={},
        overrides={},
    )
    prepared = engine._prepare_panel_features(
        panel, plan, pl.DataFrame(), None, None,
    )
    assert PATTERN in prepared.columns
    assert prepared[PATTERN].max() == 1.0


# ── 3. 矩阵注入 (端到端) ──────────────────────────────────────


def test_matrix_load_injects_pattern_columns(data_dir: Path):
    """矩阵里能直读形态列, 且数值与 compute_indicators 逐位一致。"""
    engine = _engine(data_dir)
    plan = StrategyDependencyResolver()._resolve_matrix_native(
        _strategy(matrix_fields=frozenset({PATTERN})),
        params={},
        basic_filter={},
        overrides={},
    )
    market = engine.load_market_data_matrix_for_backtest(
        None, START, START + timedelta(days=DAYS - 1), plan,
    )
    assert PATTERN in market.fields

    expected_frame = compute_indicators(
        pl.DataFrame(_price_rows()), needed={PATTERN},
    )
    actual = matrix_feature(market, PATTERN)
    assert actual.shape == (DAYS, len(SYMS))
    for col, sym in enumerate(market.symbols):
        series = (
            expected_frame.filter(pl.col("symbol") == sym)
            .sort("date")[PATTERN]
            .to_numpy()
        )
        np.testing.assert_allclose(actual[:, col], series, rtol=1e-6, atol=1e-6)
    # 上行序列 ⇒ 均线多头形态在后半段成立 (验证不是恒 0 的空转)
    assert actual.max() == 1.0


def test_matrix_load_without_pattern_columns_keeps_parquet_route(data_dir: Path):
    """不引用形态列时仍走 parquet 直读 (保留流式与磁盘缓存)。"""
    engine = _engine(data_dir)
    plan = StrategyDependencyResolver()._resolve_matrix_native(
        _strategy(matrix_fields=frozenset({"ma20"})),
        params={},
        basic_filter={},
        overrides={},
    )
    market = engine.load_market_data_matrix_for_backtest(
        None, START, START + timedelta(days=DAYS - 1), plan,
    )
    assert PATTERN not in market.fields
    assert market.cache_status != "none"


def test_pattern_seed_strategy_reads_pattern_columns(data_dir: Path):
    """内置形态策略 (矩阵原生) 能在矩阵上直读形态列并产出掩码。"""
    from app.strategy.seeds import pattern_low_backtrace as seed

    engine = _engine(data_dir)
    plan = StrategyDependencyResolver()._resolve_matrix_native(
        _strategy(matrix_fields=seed.MATRIX_STRATEGY.required_fields()),
        params={},
        basic_filter={},
        overrides={},
    )
    market = engine.load_market_data_matrix_for_backtest(
        None, START, START + timedelta(days=DAYS - 1), plan,
    )
    for column in seed.MATRIX_STRATEGY.required_fields():
        if column.startswith("pat_"):
            assert column in market.fields
    signals = seed.MATRIX_STRATEGY.compute_signals(market, {
        "use_edge_trigger": True,
        "avoid_bearish_patterns": True,
    })
    assert signals.entry.shape == market.shape
    assert signals.exit.shape == market.shape
