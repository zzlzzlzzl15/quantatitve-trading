"""K 线形态扫描数据源 — 市场分析模块的形态识别入口。

从本地 QuantDB parquet（经 :class:`QuantDBDataHub`）读取个股日线，
批量识别指定交易日（或最近交易日）的 K 线形态与入场信号。

数据口径：
- K 线取 **前复权**（``1_kline_data/daily_forward``），列 ``symbol/time/open/high/low/close/volume/amount``；
- ``time`` 为日期列，转成 ``YYYY-MM-DD``；
- 板块成分取 ``2_base_sector/sector_concept/sector_members.parquet``。

时点口径（全项目统一）：
    形态在 T 日收盘后识别 → 最早 T+1 交易日成交。
    本模块只输出识别结果与 ``earliest_entry_date``，不含成交逻辑。
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Sequence

import pandas as pd

from backend.services.engine.data_platform.quantdb_hub import QuantDBDataHub
from backend.shared import kline_patterns as pattern_engine

logger = logging.getLogger(__name__)

# 板块分类 -> sector_concept 中的 SectorType（与 quantdb_feed.CATEGORY_TYPE 一致）
CATEGORY_TYPE = {"shenwan": "行业板块(二级)", "concept": "概念板块"}

# 形态识别所需的最少交易日（入场信号形态最长用到 MA250）
MIN_BARS = 60

# 默认回看窗口（交易日）
DEFAULT_LOOKBACK = 120


def _hub() -> QuantDBDataHub:
    """全局 QuantDB 数据中枢单例。"""
    return QuantDBDataHub.get_instance()


def _partition_dates(rel_path: str) -> list[str]:
    """从分区目录名提取日期列表（降序）。"""
    try:
        dd = _hub().data_dir / rel_path
        if not dd.exists():
            return []
        dates = []
        for entry in dd.iterdir():
            if entry.is_dir() and entry.name.startswith("dt="):
                val = entry.name.split("=", 1)[1]
                if val.isdigit():
                    dates.append(val)
        return sorted(dates, reverse=True)
    except Exception as exc:  # pragma: no cover
        logger.warning("读取分区日期列表失败 %s: %s", rel_path, exc)
        return []


def _trading_days(rel_path: str, end: str | None, n: int) -> list[str]:
    """截至 end 的最近 n 个交易日（降序，[0] 为最新）。"""
    dates = _partition_dates(rel_path)
    if dates:
        if end:
            dates = [d for d in dates if d <= end]
        return dates[:n]
    return []


def _read_klines(
    symbols: Sequence[str],
    end_date: str | None,
    lookback: int = DEFAULT_LOOKBACK,
) -> dict[str, pd.DataFrame]:
    """读取指定股票的日线（升序 DataFrame）。

    Args:
        symbols: 股票代码列表（后缀格式如 ``600036.SH`` 或前缀 ``SH600036``）。
        end_date: 截止交易日 YYYYMMDD；None 表示最新。
        lookback: 回看交易日数。

    Returns:
        {symbol: DataFrame[date, open, high, low, close, volume, amount]}，按时间升序。
    """
    if not symbols:
        return {}

    rel = "1_kline_data/daily_forward"
    days = _trading_days(rel, end_date, lookback)
    if len(days) < MIN_BARS:
        logger.warning("可用交易日不足（%s < %s），K 线数据可能无法支撑形态识别", len(days), MIN_BARS)
    if not days:
        return {}

    try:
        df = _hub()._read_partitioned(
            rel,
            days,
            cols="symbol, time, open, high, low, close, volume, amount",
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("读取 K 线失败: %s", exc)
        return {}

    if df is None or df.empty:
        return {}

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str)
    if "time" in df.columns:
        df["date"] = pd.to_datetime(df["time"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    else:
        return {}

    wanted = {str(s) for s in symbols}
    df = df[df["symbol"].isin(wanted)]
    if df.empty:
        return {}

    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    keep = ["date", "open", "high", "low", "close", "volume", "amount"]
    keep = [c for c in keep if c in df.columns]
    df = df[["symbol"] + keep].dropna(subset=["date"])

    out: dict[str, pd.DataFrame] = {}
    for symbol, grp in df.groupby("symbol"):
        grp = grp.sort_values("date", kind="stable")
        out[symbol] = grp.reset_index(drop=True)
    return out


def _sector_members(category: str = "shenwan") -> dict[str, list[str]]:
    """板块名 -> 成分股 symbol 列表。"""
    try:
        members = _hub().fetch_sector_members()
    except Exception as exc:  # pragma: no cover
        logger.warning("读取板块成分失败: %s", exc)
        return {}
    if members is None or members.empty or "symbol" not in members.columns:
        return {}

    stype = CATEGORY_TYPE.get(category)
    if stype and "sector_type" in members.columns:
        members = members[members["sector_type"] == stype]

    groups: dict[str, list[str]] = {}
    for row in members.itertuples(index=False):
        name = str(getattr(row, "sector_name", "") or "").strip()
        sym = str(getattr(row, "symbol", "") or "").strip()
        if name and sym:
            groups.setdefault(name, []).append(sym)
    return groups


def resolve_symbols(
    sector: str | None = None,
    category: str = "shenwan",
    symbols: Sequence[str] | None = None,
    limit: int = 200,
) -> list[str]:
    """解析待扫描的股票池。

    Args:
        sector: 板块名；None 表示全市场。
        category: 板块分类（shenwan / concept）。
        symbols: 显式股票列表；优先于 sector。
        limit: 最多扫描数量。

    Returns:
        symbol 列表。
    """
    if symbols:
        return [str(s) for s in symbols][:limit]

    if sector:
        groups = _sector_members(category)
        members = groups.get(sector, [])
        if members:
            return members[:limit]
        logger.warning("板块 %s（%s）无成分股", sector, category)
        return []

    # 全市场：取最新交易日全量
    rel = "1_kline_data/daily_forward"
    days = _partition_dates(rel)
    if not days:
        return []
    try:
        df = _hub()._read_partitioned(
            rel, [days[0]], cols="symbol"
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("读取全市场股票列表失败: %s", exc)
        return []
    if df is None or df.empty:
        return []
    return sorted(df["symbol"].astype(str).unique().tolist())[:limit]


def scan_patterns(
    sector: str | None = None,
    category: str = "shenwan",
    symbols: Sequence[str] | None = None,
    date: str | None = None,
    lookback: int = DEFAULT_LOOKBACK,
    limit: int = 200,
    include_entry: bool = True,
    include_standard: bool = True,
    keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """扫描板块（或全市场）在指定交易日的形态命中情况。

    Args:
        sector: 板块名；None 表示全市场。
        category: 板块分类（shenwan / concept）。
        symbols: 显式股票列表；优先于 sector。
        date: 判定交易日 YYYYMMDD；None 表示最新交易日。
        lookback: 回看交易日数。
        limit: 最多扫描股票数。
        include_entry: 是否识别入场信号形态。
        include_standard: 是否识别标准形态。
        keys: 仅保留指定形态 key。

    Returns:
        形如 ``{trade_date, sector, scanned, results, aggregate}`` 的字典。
    """
    rel = "1_kline_data/daily_forward"
    days = _partition_dates(rel)
    trade_date = date or (days[0] if days else None)
    if not trade_date:
        return {
            "trade_date": None,
            "sector": sector,
            "scanned": 0,
            "results": [],
            "aggregate": [],
            "available": False,
        }

    universe = resolve_symbols(sector=sector, category=category, symbols=symbols, limit=limit)
    if not universe:
        return {
            "trade_date": trade_date,
            "sector": sector,
            "scanned": 0,
            "results": [],
            "aggregate": [],
            "available": False,
        }

    bars_by_symbol = _read_klines(universe, end_date=trade_date, lookback=lookback)
    if not bars_by_symbol:
        return {
            "trade_date": trade_date,
            "sector": sector,
            "scanned": 0,
            "results": [],
            "aggregate": [],
            "available": False,
        }

    # 交易日历（升序 YYYY-MM-DD），用于推算 T+1 最早可成交日
    calendar = [
        f"{d[:4]}-{d[4:6]}-{d[6:]}"
        for d in sorted(days)
    ]

    scanned = pattern_engine.scan_universe(
        bars_by_symbol,
        trading_dates=calendar,
        include_entry=include_entry,
        include_standard=include_standard,
        keys=keys,
    )

    # 只保留判定日恰好等于 trade_date 的结果（避免用旧数据冒充当日）
    results = []
    for res in scanned:
        if res.signal_date and res.signal_date.replace("-", "") != trade_date:
            continue
        results.append(res)

    return {
        "trade_date": trade_date,
        "sector": sector,
        "category": category,
        "scanned": len(bars_by_symbol),
        "hit_count": len(results),
        "results": [r.to_dict() for r in results],
        "aggregate": pattern_engine.aggregate_patterns(results),
        "available": True,
    }


def get_pattern_catalog() -> dict[str, Any]:
    """返回形态目录，供前端下拉框使用。"""
    catalog = pattern_engine.pattern_catalog()
    return {
        "total": len(catalog),
        "standard_count": sum(1 for c in catalog if c["category"] == "standard"),
        "entry_count": sum(1 for c in catalog if c["category"] == "entry"),
        "talib_available": pattern_engine.talib_available(),
        "items": catalog,
    }
