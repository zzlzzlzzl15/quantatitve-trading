"""板块级 K 线形态分析 —— 形态信息簇在板块间的**相对分布**。

为什么是"相对分布"而不是绝对命中率
------------------------------------
``reports/pattern_hit_stats/REPORT.md`` 对真实 A 股全库 (5,873 标的 / 1991-2026)
做过形态命中与前瞻收益统计: 9 种形态的超额收益**全部为负** (t = -4.2 ~ -45.6,
全部远离随机基准 ±0.03%)。机制上说得通 —— 形态是**状态**而非事件, 成立即意味着
"该股已经涨过 / 放量过 / 突破过", T+1 开盘追进去是在为已发生的价格变动付溢价。

因此绝对命中率在板块之间没有可比性: 全市场突破簇占比就是那么高, 某个板块
"突破形态多"不代表该板块差, 只代表该板块当前处于强势状态。面板必须给出
**相对全市场基准的倍数**, 才回答得了"这个板块的形态结构与全市场有什么不同"。

为什么按信息簇归并
------------------
同报告的共现矩阵: 突破平台 / 海龟突破 / 均线多头 互共现 75%~92%, 本质是同一
现象的三种表述; 9 个形态实际只有约 5 个独立信息簇。若按单个形态铺开, 用户会在
同一批强势股上看到大量重复信号并重复建仓。面板因此按
「突破簇 / 追涨簇 / 低位簇 / 独立事件」四组归并。

「符合入场准备标准」的判定
--------------------------
同报告结论: 正向侧只保留「低位回调上涨」(20 日绝对 +2.28%, 胜率 53.0%, 超额
-0.09%); 「放量上涨」「突破平台」「海龟突破」应作为**回避信号**使用 (5 日超额
-0.38% ~ -0.44%, 20 日超额 -0.72% ~ -1.53%)。本模块据此定义
:data:`SETUP_PATTERNS` / :data:`AVOID_PATTERNS`。

口径边界
--------
- 形态是**状态**而非事件: 同一形态可连续多日成立, 本面板统计的是"当日成立"
  (状态口径), 不做边沿触发 —— 边沿触发是回测建仓口径, 由回测侧负责。
- 板块归属来自扩展数据 (ext_data, 内置同花顺概念 / 行业预设), 与看板
  「概念热度 / 行业热度」同源 (复用 SectorMonitorService 的板块目录)。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable, Sequence

import polars as pl

from app.indicators.patterns import (
    ENTRY_SIGNAL_PATTERNS,
    PATTERN_COLUMNS,
    pattern_column,
)

logger = logging.getLogger(__name__)

# 形态列是 0/1 数值哑变量 (compute_indicators 产出 1.0 / 0.0), >=0.5 视为成立
_PATTERN_HIT = 0.5

# 板块维度
SECTOR_KINDS: tuple[str, ...] = ("concept", "industry")


@dataclass(frozen=True)
class PatternCluster:
    """形态信息簇 —— 按共现结构归并, 避免同一现象被拆成多个重复信号。"""

    key: str
    label: str
    summary: str
    patterns: tuple[str, ...]
    stance: str  # avoid(回避) / setup(正向候选) / neutral(中性)


PATTERN_CLUSTERS: tuple[PatternCluster, ...] = (
    PatternCluster(
        key="breakout",
        label="突破簇",
        summary="突破平台 / 海龟突破 / 均线多头 —— 互共现 75%~92%, 同一现象的三种表述",
        patterns=(
            pattern_column("platform_breakout"),
            pattern_column("turtle_trade"),
            pattern_column("ma_bull_alignment"),
        ),
        stance="avoid",
    ),
    PatternCluster(
        key="chase",
        label="追涨簇",
        summary="放量上涨 —— 追涨型负超额最重的一组 (20 日超额 -1.53%)",
        patterns=(pattern_column("volume_surge_up"),),
        stance="avoid",
    ),
    PatternCluster(
        key="pullback",
        label="低位簇",
        summary="低位回调上涨 —— 唯一接近可用的正向形态 (20 日绝对 +2.28%, 胜率 53.0%)",
        patterns=(pattern_column("low_backtrace_increase"),),
        stance="setup",
    ),
    PatternCluster(
        key="standalone",
        label="独立事件",
        summary="突破均线 / 回踩年线 / 放量跌停 —— 与其余形态共现率低",
        patterns=(
            pattern_column("breakout_ma"),
            pattern_column("backtrace_ma250"),
            pattern_column("climax_limitdown"),
        ),
        stance="neutral",
    ),
)

# 高位紧旗: 定义自身互斥 (要求"当日仍涨停"同时"缩量"), 35 年全库仅 9 个样本,
# 不纳入统计口径 —— 见 REPORT.md §4.5。样本量不足时任何占比都没有意义。
EXPERIMENTAL_PATTERNS: tuple[str, ...] = (pattern_column("high_tight_flag"),)

# 参与统计口径的形态列 (排除样本量不足的实验形态)
STATS_PATTERNS: tuple[str, ...] = tuple(
    column for column in PATTERN_COLUMNS if column not in EXPERIMENTAL_PATTERNS
)

# 「符合入场准备标准」: 正向形态成立 且 未触发任一回避形态
SETUP_PATTERNS: tuple[str, ...] = (pattern_column("low_backtrace_increase"),)
AVOID_PATTERNS: tuple[str, ...] = (
    pattern_column("volume_surge_up"),
    pattern_column("platform_breakout"),
    pattern_column("turtle_trade"),
    pattern_column("climax_limitdown"),
)

# 形态中文名 / 判定说明 (取自 indicators.patterns 的单一事实源)
PATTERN_LABELS: dict[str, str] = {
    pattern_column(key): label for key, label, _ in ENTRY_SIGNAL_PATTERNS
}
PATTERN_DESCRIPTIONS: dict[str, str] = {
    pattern_column(key): desc for key, _, desc in ENTRY_SIGNAL_PATTERNS
}


@dataclass(frozen=True)
class SectorDef:
    """板块定义 (成员为扩展数据里的原始股票代码)。"""

    key: str
    name: str
    kind: str
    level: int | None
    source_field: str
    member_codes: tuple[str, ...]


@dataclass(frozen=True)
class PatternRow:
    """单只标的在当日横截面上的形态快照。"""

    symbol: str
    name: str
    close: float | None
    change_pct: float | None
    amount: float
    turnover_rate: float | None
    hits: frozenset[str]
    clusters: frozenset[str]

    @property
    def is_setup(self) -> bool:
        """是否符合入场准备标准 (正向形态成立且无回避形态)。"""
        if not any(p in self.hits for p in SETUP_PATTERNS):
            return False
        return not any(p in self.hits for p in AVOID_PATTERNS)


# ================================================================
# 基础工具
# ================================================================

def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_hit(value: Any) -> bool:
    number = _finite(value)
    return number is not None and number >= _PATTERN_HIT


def normalize_code(raw: Any) -> str:
    """股票代码归一化 —— 去掉交易所后缀 / 前缀, 只留 6 位数字主体。

    扩展数据里的成分股代码 ("600000" / "600000.SH" / "SH600000") 与行情侧的
    symbol ("600000.SH") 写法不同, 必须归一到同一把尺子才能对上。
    """
    text = str(raw or "").strip().upper()
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    if len(text) > 6 and text[:2] in {"SH", "SZ", "BJ"}:
        text = text[2:]
    return text


def _ratio(count: int, total: int) -> float:
    return round(count / total * 100, 2) if total else 0.0


def _relative(sector_ratio: float, market_ratio: float) -> float | None:
    """板块占比相对全市场基准的倍数 (None = 全市场为 0, 倍数无意义)。"""
    if market_ratio <= 0:
        return None
    return round(sector_ratio / market_ratio, 2)


# ================================================================
# 横截面加载
# ================================================================

def extract_rows(df: pl.DataFrame) -> list[PatternRow]:
    """从 enriched 截面抽出形态行 (缺列的形态视为不成立)。"""
    if df.is_empty() or "symbol" not in df.columns:
        return []
    keep = ["symbol", "name", "close", "change_pct", "amount", "turnover_rate"]
    cols = [c for c in keep if c in df.columns]
    cols.extend(c for c in PATTERN_COLUMNS if c in df.columns)
    rows: list[PatternRow] = []
    for raw in df.select(cols).to_dicts():
        symbol = str(raw.get("symbol") or "").strip()
        if not symbol:
            continue
        hits = frozenset(c for c in PATTERN_COLUMNS if _is_hit(raw.get(c)))
        clusters = frozenset(
            cluster.key
            for cluster in PATTERN_CLUSTERS
            if any(p in hits for p in cluster.patterns)
        )
        rows.append(
            PatternRow(
                symbol=symbol,
                name=str(raw.get("name") or ""),
                close=_finite(raw.get("close")),
                change_pct=_finite(raw.get("change_pct")),
                amount=_finite(raw.get("amount")) or 0.0,
                turnover_rate=_finite(raw.get("turnover_rate")),
                hits=hits,
                clusters=clusters,
            )
        )
    return rows


def _recompute_pattern_frame(repo, as_of: date, enriched_dirname: str) -> pl.DataFrame:
    """形态列缺失时的兜底: 从 parquet 重算当日形态 (指标缓存可能是旧版本构建的)。"""
    from app.indicators.pipeline import compute_indicators
    from app.parquet import parquet_glob, scan_enriched_parquet

    enriched_dir = repo.store.data_dir / enriched_dirname
    # 回踩年线需要 250 根 → 420 自然日足够覆盖 (A 股 ~290 交易日)
    start = as_of - timedelta(days=420)
    try:
        lf = (
            scan_enriched_parquet(parquet_glob(enriched_dir))
            .filter((pl.col("date") >= start) & (pl.col("date") <= as_of))
            .sort(["symbol", "date"])
        )
        cols = [
            c
            for c in ("symbol", "date", "open", "high", "low", "close", "volume", "amount")
            if c in lf.schema
        ]
        hist = lf.select(cols).collect()
    except Exception as exc:  # noqa: BLE001
        logger.warning("形态列重算失败 (%s): %s", as_of, exc)
        return pl.DataFrame()
    if hist.is_empty():
        return pl.DataFrame()
    full = compute_indicators(hist, needed=set(PATTERN_COLUMNS))
    return full.filter(pl.col("date") == as_of)


def load_pattern_cross_section(repo, as_of: date) -> list[PatternRow]:
    """取指定交易日的全市场形态横截面。

    复用 ScreenerService 的当日 enriched (含全量指标列), 因此形态列与看板其它
    卡片同源同口径; 仅在缓存里缺形态列时 (旧缓存) 才走 parquet 重算兜底。
    """
    from app.services.screener import ScreenerService

    svc = ScreenerService(repo)
    df = svc._load_enriched_for_date(as_of)
    if df.is_empty():
        return []
    missing = [c for c in PATTERN_COLUMNS if c not in df.columns]
    if missing:
        logger.info("形态列缺失 (%s), 按 parquet 重算: %s", ",".join(missing), as_of)
        df = _recompute_pattern_frame(repo, as_of, svc._enriched_dirname)
        if df.is_empty():
            return []
        if "name" not in df.columns:
            instruments = repo.get_instruments_asset("stock")
            if instruments is not None and not instruments.is_empty() and "name" in instruments.columns:
                df = df.join(
                    instruments.select("symbol", "name").unique(subset=["symbol"]),
                    on="symbol",
                    how="left",
                )
    return extract_rows(df)


# ================================================================
# 板块目录
# ================================================================

def build_sector_defs(
    repo,
    kind: str,
    *,
    level: int | None = None,
    min_members: int = 1,
) -> list[SectorDef]:
    """从扩展数据解析板块成分。

    复用 SectorMonitorService 的板块目录 —— 与看板「概念热度 / 行业热度」同源,
    避免同一份 ext_data 出现两套解析口径。``level`` 只对行业有意义 (同花顺行业
    是 "一级-二级-三级" 层级结构), None = 不按层级过滤。
    """
    from app.services.sector_monitor import SectorMonitorService

    monitor = SectorMonitorService(repo)
    targets = monitor.list_targets().get(kind, [])
    members = monitor.member_map()
    out: list[SectorDef] = []
    for target in targets:
        key = str(target.get("key") or "")
        codes = members.get(key) or set()
        if len(codes) < min_members:
            continue
        target_level = target.get("level")
        level_value = int(target_level) if target_level is not None else None
        if level is not None and level_value is not None and level_value != level:
            continue
        out.append(
            SectorDef(
                key=key,
                name=str(target.get("name") or ""),
                kind=kind,
                level=level_value,
                source_field=str(target.get("source_field") or ""),
                member_codes=tuple(sorted(codes)),
            )
        )
    return out


# ================================================================
# 聚合
# ================================================================

def _market_baseline(rows: Sequence[PatternRow]) -> dict:
    """全市场基准 —— 板块相对占比的分母。"""
    total = len(rows)
    patterns = {
        column: {
            "count": sum(1 for r in rows if column in r.hits),
        }
        for column in STATS_PATTERNS
    }
    for column, item in patterns.items():
        item["ratio"] = _ratio(item["count"], total)
    clusters = {
        cluster.key: {
            "count": sum(1 for r in rows if cluster.key in r.clusters),
            "label": cluster.label,
        }
        for cluster in PATTERN_CLUSTERS
    }
    for item in clusters.values():
        item["ratio"] = _ratio(item["count"], total)
    return {
        "total": total,
        "patterns": patterns,
        "clusters": clusters,
        "setup_count": sum(1 for r in rows if r.is_setup),
    }


def _row_payload(row: PatternRow, sectors: Sequence[str] | None = None) -> dict:
    payload = {
        "symbol": row.symbol,
        "name": row.name,
        "close": row.close,
        "change_pct": row.change_pct,
        "amount": row.amount,
        "turnover_rate": row.turnover_rate,
        "patterns": sorted(row.hits),
        "pattern_labels": [PATTERN_LABELS.get(p, p) for p in sorted(row.hits)],
        "clusters": sorted(row.clusters),
    }
    if sectors is not None:
        payload["sectors"] = list(sectors)
    return payload


def _sector_payload(
    sector: SectorDef,
    member_rows: Sequence[PatternRow],
    member_count: int,
    market: dict,
    candidates: Sequence[PatternRow],
    candidate_limit: int,
) -> dict:
    valid = len(member_rows)
    clusters: dict[str, dict] = {}
    for cluster in PATTERN_CLUSTERS:
        count = sum(1 for r in member_rows if cluster.key in r.clusters)
        ratio = _ratio(count, valid)
        market_ratio = market["clusters"][cluster.key]["ratio"]
        clusters[cluster.key] = {
            "label": cluster.label,
            "stance": cluster.stance,
            "count": count,
            "ratio": ratio,
            "market_ratio": market_ratio,
            "relative": _relative(ratio, market_ratio),
        }
    patterns: dict[str, dict] = {}
    for column in STATS_PATTERNS:
        count = sum(1 for r in member_rows if column in r.hits)
        ratio = _ratio(count, valid)
        market_ratio = market["patterns"][column]["ratio"]
        patterns[column] = {
            "label": PATTERN_LABELS.get(column, column),
            "count": count,
            "ratio": ratio,
            "market_ratio": market_ratio,
            "relative": _relative(ratio, market_ratio),
        }
    changes = [r.change_pct for r in member_rows if r.change_pct is not None]
    avg_change = round(sum(changes) / len(changes), 2) if changes else None
    pullback_rel = clusters["pullback"]["relative"]
    breakout_rel = clusters["breakout"]["relative"]
    setup_score = None
    if pullback_rel is not None and breakout_rel is not None:
        setup_score = round(pullback_rel - breakout_rel, 2)
    return {
        "key": sector.key,
        "name": sector.name,
        "kind": sector.kind,
        "level": sector.level,
        "source_field": sector.source_field,
        "member_count": member_count,
        "valid_count": valid,
        "coverage_ratio": round(valid / member_count, 4) if member_count else 0.0,
        "avg_change_pct": avg_change,
        "up_count": sum(1 for v in changes if v > 0),
        "down_count": sum(1 for v in changes if v < 0),
        "amount": round(sum(r.amount for r in member_rows), 2),
        "clusters": clusters,
        "patterns": patterns,
        "setup_count": len(candidates),
        "setup_score": setup_score,
        "candidates": [_row_payload(r) for r in candidates[:candidate_limit]],
    }


def analyze_sector_patterns(
    rows: Sequence[PatternRow],
    sectors: Sequence[SectorDef],
    *,
    min_members: int = 5,
    candidate_limit: int = 20,
    sector_limit: int = 40,
    sort_by: str = "setup_score",
) -> dict:
    """核心聚合 —— 纯函数 (输入横截面 + 板块定义, 输出报告)。

    板块筛选口径: 成分数不足 ``min_members`` 的板块直接剔除 —— 成分太少的板块
    占比波动极大, 相对倍数没有统计意义。
    """
    market = _market_baseline(rows)
    by_code: dict[str, list[PatternRow]] = {}
    for row in rows:
        by_code.setdefault(normalize_code(row.symbol), []).append(row)

    sector_payloads: list[dict] = []
    sector_names_by_symbol: dict[str, list[str]] = {}
    for sector in sectors:
        member_rows: list[PatternRow] = []
        for code in sector.member_codes:
            member_rows.extend(by_code.get(normalize_code(code), []))
        if len(member_rows) < min_members:
            continue
        candidates = [r for r in member_rows if r.is_setup]
        candidates.sort(key=lambda r: (r.change_pct is None, -(r.change_pct or 0.0), -r.amount))
        for row in candidates:
            sector_names_by_symbol.setdefault(row.symbol, []).append(sector.name)
        sector_payloads.append(
            _sector_payload(
                sector,
                member_rows,
                len(sector.member_codes),
                market,
                candidates,
                candidate_limit,
            )
        )

    def sort_key(item: dict) -> tuple:
        if sort_by == "relative_breakout":
            value = item["clusters"]["breakout"]["relative"]
            return (value is None, -(value or 0.0))
        if sort_by == "setup_count":
            return (-item["setup_count"], item["name"])
        value = item["setup_score"]
        return (value is None, -(value or 0.0), item["name"])

    sector_payloads.sort(key=sort_key)
    truncated = sector_payloads[:sector_limit]

    setup_rows = [r for r in rows if r.is_setup]
    setup_rows.sort(key=lambda r: (r.change_pct is None, -(r.change_pct or 0.0), -r.amount))
    candidates_payload = [
        _row_payload(r, sector_names_by_symbol.get(r.symbol, []))
        for r in setup_rows[: max(candidate_limit, 50)]
    ]

    return {
        "market": market,
        "sectors": truncated,
        "candidates": candidates_payload,
        "clusters": [
            {
                "key": c.key,
                "label": c.label,
                "summary": c.summary,
                "stance": c.stance,
                "patterns": list(c.patterns),
            }
            for c in PATTERN_CLUSTERS
        ],
        "patterns": [
            {
                "column": column,
                "label": PATTERN_LABELS.get(column, column),
                "description": PATTERN_DESCRIPTIONS.get(column, ""),
                "experimental": column in EXPERIMENTAL_PATTERNS,
            }
            for column in PATTERN_COLUMNS
        ],
        "setup_criteria": {
            "setup_patterns": list(SETUP_PATTERNS),
            "avoid_patterns": list(AVOID_PATTERNS),
        },
    }


def build_pattern_sector_report(
    repo,
    as_of: date | None = None,
    *,
    kind: str = "concept",
    level: int | None = None,
    limit: int = 20,
    min_members: int = 5,
    candidate_limit: int = 20,
    sort_by: str = "setup_score",
) -> dict:
    """板块级形态分析报告 (入口)。

    返回结构::

        {
          "as_of": "2026-11-18",
          "kind": "concept",
          "market": {"total": 5873, "clusters": {...}, "patterns": {...}, "setup_count": 123},
          "sectors": [{"name": "...", "clusters": {...}, "setup_score": 0.8, ...}],
          "candidates": [...],
        }
    """
    if kind not in SECTOR_KINDS:
        raise ValueError(f"kind 必须是 {SECTOR_KINDS} 之一, 收到 {kind!r}")
    from app.services.screener import ScreenerService

    svc = ScreenerService(repo)
    resolved = as_of or svc.latest_date()
    if resolved is None:
        return {
            "as_of": None,
            "kind": kind,
            "market": {"total": 0, "patterns": {}, "clusters": {}, "setup_count": 0},
            "sectors": [],
            "candidates": [],
            "clusters": [],
            "patterns": [],
            "setup_criteria": {
                "setup_patterns": list(SETUP_PATTERNS),
                "avoid_patterns": list(AVOID_PATTERNS),
            },
            "warnings": ["本地无可用行情数据"],
        }

    rows = load_pattern_cross_section(repo, resolved)
    if not rows:
        return {
            "as_of": str(resolved),
            "kind": kind,
            "market": {"total": 0, "patterns": {}, "clusters": {}, "setup_count": 0},
            "sectors": [],
            "candidates": [],
            "clusters": [],
            "patterns": [],
            "setup_criteria": {
                "setup_patterns": list(SETUP_PATTERNS),
                "avoid_patterns": list(AVOID_PATTERNS),
            },
            "warnings": [f"{resolved} 无可用行情数据"],
        }

    sectors = build_sector_defs(repo, kind, level=level)
    report = analyze_sector_patterns(
        rows,
        sectors,
        min_members=min_members,
        candidate_limit=candidate_limit,
        sector_limit=limit,
        sort_by=sort_by,
    )
    report["as_of"] = str(resolved)
    report["kind"] = kind
    report["level"] = level
    warnings: list[str] = []
    if not sectors:
        warnings.append("未找到扩展数据中的板块成分 — 请先在「数据」页拉取概念 / 行业扩展表")
    warnings.append(
        "「高位紧旗」定义互斥 (要求当日仍涨停同时缩量), 全库样本极少, 已排除出统计口径"
    )
    report["warnings"] = warnings
    return report


__all__ = [
    "AVOID_PATTERNS",
    "EXPERIMENTAL_PATTERNS",
    "PATTERN_CLUSTERS",
    "PATTERN_COLUMNS",
    "PatternCluster",
    "PatternRow",
    "SECTOR_KINDS",
    "SETUP_PATTERNS",
    "STATS_PATTERNS",
    "SectorDef",
    "analyze_sector_patterns",
    "build_pattern_sector_report",
    "build_sector_defs",
    "extract_rows",
    "load_pattern_cross_section",
    "normalize_code",
]
