"""板块级形态分析 (services/pattern_sector_analysis) 的单元测试。

口径依据 ``reports/pattern_hit_stats/REPORT.md``: 形态的超额收益全为负 ⇒ 板块
面板只给**相对分布**; 9 个形态按共现结构归并成 4 个信息簇。
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import polars as pl
import pytest

from app.services import pattern_sector_analysis as psa
from app.services.ext_data import ExtConfig, ExtConfigStore, ExtField


# ================================================================
# 工具
# ================================================================

def _row(
    symbol: str,
    *,
    hits: tuple[str, ...] = (),
    change_pct: float | None = 0.0,
    amount: float = 1000.0,
    name: str = "",
) -> psa.PatternRow:
    hit_set = frozenset(hits)
    clusters = frozenset(
        cluster.key
        for cluster in psa.PATTERN_CLUSTERS
        if any(p in hit_set for p in cluster.patterns)
    )
    return psa.PatternRow(
        symbol=symbol,
        name=name or symbol,
        close=10.0,
        change_pct=change_pct,
        amount=amount,
        turnover_rate=1.0,
        hits=hit_set,
        clusters=clusters,
    )


def _sector(key: str, name: str, codes: tuple[str, ...], kind: str = "concept") -> psa.SectorDef:
    return psa.SectorDef(
        key=key,
        name=name,
        kind=kind,
        level=None,
        source_field="test.概念",
        member_codes=codes,
    )


# ================================================================
# 形态 → 信息簇
# ================================================================

def test_cluster_definitions_match_report_cooccurrence():
    """突破簇 = 突破平台 / 海龟突破 / 均线多头 (互共现 75%~92%, 同一现象)。"""
    clusters = {c.key: c for c in psa.PATTERN_CLUSTERS}
    assert set(clusters) == {"breakout", "chase", "pullback", "standalone"}
    assert set(clusters["breakout"].patterns) == {
        "pat_platform_breakout",
        "pat_turtle_trade",
        "pat_ma_bull_alignment",
    }
    assert set(clusters["chase"].patterns) == {"pat_volume_surge_up"}
    assert set(clusters["pullback"].patterns) == {"pat_low_backtrace_increase"}
    assert set(clusters["standalone"].patterns) == {
        "pat_breakout_ma",
        "pat_backtrace_ma250",
        "pat_climax_limitdown",
    }
    # 只有低位簇是正向候选; 突破簇/追涨簇按实证结论作为回避信号
    assert clusters["pullback"].stance == "setup"
    assert clusters["breakout"].stance == "avoid"
    assert clusters["chase"].stance == "avoid"


def test_high_tight_flag_excluded_from_statistics():
    """高位紧旗定义互斥 (涨停 + 缩量), 全库仅 9 个样本 —— 不进统计口径。"""
    assert psa.EXPERIMENTAL_PATTERNS == ("pat_high_tight_flag",)
    stats = [c for c in psa.PATTERN_CLUSTERS if "pat_high_tight_flag" in c.patterns]
    assert stats == []
    assert "pat_high_tight_flag" not in psa.STATS_PATTERNS


def test_experimental_pattern_absent_from_ratio_stats():
    rows = [_row("A", hits=("pat_high_tight_flag", "pat_low_backtrace_increase"))]
    report = psa.analyze_sector_patterns(rows, [_sector("k", "板块", ("A",))], min_members=1)
    assert "pat_high_tight_flag" not in report["market"]["patterns"]
    assert "pat_high_tight_flag" not in report["sectors"][0]["patterns"]
    # 但仍在元数据里并标记为实验形态 (供前端提示)
    meta = {p["column"]: p for p in report["patterns"]}
    assert meta["pat_high_tight_flag"]["experimental"] is True
    assert meta["pat_low_backtrace_increase"]["experimental"] is False


def test_extract_rows_maps_pattern_hits_to_clusters():
    df = pl.DataFrame({
        "symbol": ["600000.SH", "000001.SZ"],
        "name": ["甲", "乙"],
        "close": [10.0, 20.0],
        "change_pct": [1.5, -0.5],
        "amount": [1e9, 2e8],
        "turnover_rate": [3.0, 1.5],
        "pat_platform_breakout": [1.0, 0.0],
        "pat_turtle_trade": [1.0, 0.0],
        "pat_low_backtrace_increase": [0.0, 1.0],
    })
    rows = psa.extract_rows(df)
    assert len(rows) == 2
    first, second = rows
    assert first.clusters == frozenset({"breakout"})
    assert second.clusters == frozenset({"pullback"})
    assert second.is_setup is True


def test_extract_rows_missing_pattern_columns_mean_not_hit():
    df = pl.DataFrame({"symbol": ["600000.SH"], "close": [10.0]})
    rows = psa.extract_rows(df)
    assert rows[0].hits == frozenset()
    assert rows[0].clusters == frozenset()
    assert rows[0].is_setup is False


def test_extract_rows_ignores_empty_frame():
    assert psa.extract_rows(pl.DataFrame()) == []


def test_normalize_code_handles_common_formats():
    assert psa.normalize_code("600000.SH") == "600000"
    assert psa.normalize_code("600000") == "600000"
    assert psa.normalize_code("sh600000") == "600000"
    assert psa.normalize_code(" 000001.sz ") == "000001"
    assert psa.normalize_code(None) == ""


# ================================================================
# 入场准备标准
# ================================================================

def test_setup_requires_positive_pattern_and_no_avoid_pattern():
    assert _row("A", hits=("pat_low_backtrace_increase",)).is_setup is True
    # 追涨形态同时成立 ⇒ 不算候选
    assert _row("B", hits=("pat_low_backtrace_increase", "pat_volume_surge_up")).is_setup is False
    assert _row("C", hits=("pat_platform_breakout",)).is_setup is False
    # 只有回避形态, 没有正向形态 ⇒ 不算候选
    assert _row("D", hits=("pat_turtle_trade",)).is_setup is False
    assert _row("E", hits=()).is_setup is False


# ================================================================
# 相对分布
# ================================================================

def test_sector_ratio_is_relative_to_market_not_absolute():
    """板块突破簇占比 60% > 全市场 25% ⇒ relative = 2.4, 绝对占比单独给 ratio。"""
    rows = [_row(f"S{i}", hits=("pat_platform_breakout",)) for i in range(3)]
    rows += [_row(f"M{i}") for i in range(9)]
    sectors = [_sector("k", "强板块", ("S0", "S1", "S2"))]

    report = psa.analyze_sector_patterns(rows, sectors, min_members=1)
    market = report["market"]
    assert market["total"] == 12
    assert market["clusters"]["breakout"]["ratio"] == pytest.approx(25.0)

    sector = report["sectors"][0]
    assert sector["clusters"]["breakout"]["ratio"] == pytest.approx(100.0)
    assert sector["clusters"]["breakout"]["market_ratio"] == pytest.approx(25.0)
    assert sector["clusters"]["breakout"]["relative"] == pytest.approx(4.0)


def test_relative_is_none_when_market_baseline_is_zero():
    rows = [_row("S0", hits=("pat_platform_breakout",))]
    sectors = [_sector("k", "板块", ("S0",))]
    report = psa.analyze_sector_patterns(rows, sectors, min_members=1)
    assert report["market"]["clusters"]["breakout"]["ratio"] == pytest.approx(100.0)
    assert report["sectors"][0]["clusters"]["breakout"]["relative"] == pytest.approx(1.0)


def test_sectors_below_min_members_are_skipped():
    rows = [_row("S0", hits=("pat_platform_breakout",)), _row("S1")]
    sectors = [_sector("small", "小板块", ("S0", "S1"))]
    report = psa.analyze_sector_patterns(rows, sectors, min_members=5)
    assert report["sectors"] == []


def test_sector_coverage_reports_missing_members():
    rows = [_row(f"S{i}") for i in range(4)]
    sectors = [_sector("k", "板块", ("S0", "S1", "S2", "S3", "X9"))]
    report = psa.analyze_sector_patterns(rows, sectors, min_members=1)
    sector = report["sectors"][0]
    assert sector["member_count"] == 5
    assert sector["valid_count"] == 4
    assert sector["coverage_ratio"] == pytest.approx(0.8)


def test_setup_score_prefers_pullback_over_breakout():
    """低位相对强、突破相对弱 ⇒ setup_score 为正。"""
    rows = [_row(f"P{i}", hits=("pat_low_backtrace_increase",)) for i in range(3)]
    rows += [_row(f"B{i}", hits=("pat_platform_breakout",)) for i in range(3)]
    rows += [_row(f"N{i}") for i in range(6)]
    sectors = [
        _sector("pull", "低位强", ("P0", "P1", "P2")),
        _sector("break", "突破强", ("B0", "B1", "B2")),
    ]
    report = psa.analyze_sector_patterns(rows, sectors, min_members=1)
    by_name = {s["name"]: s for s in report["sectors"]}
    assert by_name["低位强"]["setup_score"] > 0
    assert by_name["突破强"]["setup_score"] < 0
    # 排序按 setup_score 降序
    assert report["sectors"][0]["name"] == "低位强"


def test_sector_sort_by_relative_breakout():
    rows = [_row(f"B{i}", hits=("pat_platform_breakout",)) for i in range(4)]
    rows += [_row(f"N{i}") for i in range(6)]
    sectors = [
        _sector("break", "突破强", ("B0", "B1", "B2", "B3")),
        _sector("flat", "平淡", ("N0", "N1", "N2", "N3")),
    ]
    report = psa.analyze_sector_patterns(rows, sectors, min_members=1, sort_by="relative_breakout")
    assert report["sectors"][0]["name"] == "突破强"


def test_candidates_listed_per_sector_and_globally():
    rows = [
        _row("P0", hits=("pat_low_backtrace_increase",), change_pct=3.0, amount=5e9),
        _row("P1", hits=("pat_low_backtrace_increase",), change_pct=1.0, amount=2e9),
        _row("B0", hits=("pat_platform_breakout",), change_pct=5.0),
    ]
    sectors = [_sector("k", "板块", ("P0", "P1", "B0"))]
    report = psa.analyze_sector_patterns(rows, sectors, min_members=1)
    sector = report["sectors"][0]
    assert sector["setup_count"] == 2
    assert [c["symbol"] for c in sector["candidates"]] == ["P0", "P1"]
    # 全市场候选同样按涨幅排序, 且带出所属板块
    assert [c["symbol"] for c in report["candidates"]] == ["P0", "P1"]
    assert report["candidates"][0]["sectors"] == ["板块"]


def test_candidate_payload_carries_pattern_labels():
    rows = [_row("P0", hits=("pat_low_backtrace_increase",))]
    report = psa.analyze_sector_patterns(rows, [_sector("k", "板块", ("P0",))], min_members=1)
    candidate = report["candidates"][0]
    assert candidate["patterns"] == ["pat_low_backtrace_increase"]
    assert candidate["pattern_labels"] == ["低位回调上涨"]
    assert candidate["clusters"] == ["pullback"]


def test_report_exposes_cluster_and_pattern_metadata():
    rows = [_row("P0", hits=("pat_low_backtrace_increase",))]
    report = psa.analyze_sector_patterns(rows, [_sector("k", "板块", ("P0",))], min_members=1)
    labels = {c["key"] for c in report["clusters"]}
    assert labels == {"breakout", "chase", "pullback", "standalone"}
    pattern_labels = {p["label"] for p in report["patterns"]}
    assert "低位回调上涨" in pattern_labels
    assert report["setup_criteria"]["setup_patterns"] == ["pat_low_backtrace_increase"]


# ================================================================
# 板块目录 (ext_data)
# ================================================================

def _write_concept_ext(data_dir, concepts: dict[str, list[str]]) -> None:
    config = ExtConfig(
        id="concept_test",
        label="概念测试",
        mode="snapshot",
        fields=[
            ExtField("symbol", "string", "标的代码"),
            ExtField("concept", "string", "所属概念"),
        ],
    )
    ExtConfigStore(data_dir).upsert(config)
    ext_dir = data_dir / "ext_data" / config.id
    rows = []
    for name, codes in concepts.items():
        for code in codes:
            rows.append({"symbol": code, "concept": name})
    pl.DataFrame(rows).write_parquet(ext_dir / "part.parquet")


def test_build_sector_defs_reads_ext_data(tmp_path):
    _write_concept_ext(tmp_path, {"人工智能": ["600000", "000001"]})
    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    sectors = psa.build_sector_defs(repo, "concept")
    assert len(sectors) == 1
    assert sectors[0].name == "人工智能"
    assert set(sectors[0].member_codes) == {"600000", "000001"}


def test_build_sector_defs_filters_industry_level(tmp_path):
    config = ExtConfig(
        id="industry_test",
        label="行业测试",
        mode="snapshot",
        fields=[
            ExtField("symbol", "string", "标的代码"),
            ExtField("industry", "string", "所属行业"),
        ],
    )
    ExtConfigStore(tmp_path).upsert(config)
    ext_dir = tmp_path / "ext_data" / config.id
    pl.DataFrame({
        "symbol": ["600000"],
        "industry": ["电子-半导体-芯片"],
    }).write_parquet(ext_dir / "part.parquet")

    repo = SimpleNamespace(store=SimpleNamespace(data_dir=tmp_path))
    level1 = psa.build_sector_defs(repo, "industry", level=1)
    assert [s.name for s in level1] == ["电子"]
    level3 = psa.build_sector_defs(repo, "industry", level=3)
    assert [s.name for s in level3] == ["电子 / 半导体 / 芯片"]
    # level=None → 不按层级过滤, 三级全在
    assert len(psa.build_sector_defs(repo, "industry")) == 3


# ================================================================
# 报告入口
# ================================================================

class _Repo:
    def __init__(self, latest: date | None):
        self._latest = latest

    def enriched_latest_date(self):
        return self._latest

    def execute_one(self, *_args, **_kwargs):
        return None


def test_build_report_end_to_end(monkeypatch):
    rows = [
        _row("600000.SH", hits=("pat_low_backtrace_increase",), change_pct=2.0),
        _row("000001.SZ", hits=("pat_low_backtrace_increase",), change_pct=1.0),
        _row("600519.SH", hits=("pat_platform_breakout",), change_pct=5.0),
    ]
    monkeypatch.setattr(
        psa, "load_pattern_cross_section", lambda repo, as_of: rows,
    )
    monkeypatch.setattr(
        psa, "build_sector_defs",
        lambda repo, kind, level=None: [_sector("k", "人工智能", ("600000", "000001", "600519"))],
    )
    report = psa.build_pattern_sector_report(
        _Repo(date(2026, 11, 18)), kind="concept", min_members=1,
    )
    assert report["as_of"] == "2026-11-18"
    assert report["kind"] == "concept"
    assert report["market"]["total"] == 3
    assert report["sectors"][0]["name"] == "人工智能"
    assert report["sectors"][0]["setup_count"] == 2
    assert any("高位紧旗" in w for w in report["warnings"])


def test_build_report_without_data_returns_empty(monkeypatch):
    monkeypatch.setattr(psa, "load_pattern_cross_section", lambda repo, as_of: [])
    report = psa.build_pattern_sector_report(_Repo(date(2026, 11, 18)))
    assert report["sectors"] == []
    assert report["candidates"] == []
    assert report["warnings"]


def test_build_report_rejects_unknown_kind():
    with pytest.raises(ValueError):
        psa.build_pattern_sector_report(_Repo(date(2026, 11, 18)), kind="theme")
