"""个股 K 线形态识别引擎（跨服务单一事实源）。

被两处复用：
- ``backend/services/api/market_analysis`` —— 板块/个股形态扫描 API；
- ``backend/services/engine/strategy_lab`` —— 策略脚本 ``ctx.pattern_hits()``。

两类形态：

1. **标准形态（61 种）**：基于 TA-Lib 的 ``CDL*`` 函数（BSD 许可，可商用闭源）。
   只需 OHLCV，纯向量化计算。
2. **入场信号形态（9 种）**：均线/量能/波动率维度的入场准备标准判定，
   供筛选与回测验证使用。

时点口径（全项目统一）：
    因子在 T 日**收盘后**算出 → T+1 收盘成交 → 持仓吃到 bar [T+1, T+2]。
    因此形态识别只能使用**截至 T 日收盘**的数据；识别结果标记的 ``signal_date``
    为 T 日（形态成立的日期），而**最早可成交日**为 T+1 交易日。

本模块只做"识别"与"标注"，不含任何成交或持仓逻辑。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - 环境缺失时降级为不可用，不阻断整个 API
    import talib as tl

    _TALIB_AVAILABLE = True
except ImportError:  # pragma: no cover
    tl = None  # type: ignore[assignment]
    _TALIB_AVAILABLE = False


# 形态方向标签
DIRECTION_BULLISH = "bullish"
DIRECTION_BEARISH = "bearish"
DIRECTION_NEUTRAL = "neutral"

# 标准形态定义：(key, 中文名, TA-Lib 函数名, 方向)
#
# 命名沿用 myhhub/stock（Apache-2.0）的 key 约定，便于与既有生态对齐；
# 判定逻辑本身属于 TA-Lib（BSD），无许可障碍。
STANDARD_PATTERNS: tuple[tuple[str, str, str, str], ...] = (
    ("two_crows", "两只乌鸦", "CDL2CROWS", DIRECTION_BEARISH),
    ("upside_gap_two_crows", "向上跳空的两只乌鸦", "CDLUPSIDEGAP2CROWS", DIRECTION_BEARISH),
    ("three_black_crows", "三只乌鸦", "CDL3BLACKCROWS", DIRECTION_BEARISH),
    ("identical_three_crows", "三胞胎乌鸦", "CDLIDENTICAL3CROWS", DIRECTION_BEARISH),
    ("three_line_strike", "三线打击", "CDL3LINESTRIKE", DIRECTION_BEARISH),
    ("dark_cloud_cover", "乌云压顶", "CDLDARKCLOUDCOVER", DIRECTION_BEARISH),
    ("evening_doji_star", "十字暮星", "CDLEVENINGDOJISTAR", DIRECTION_BEARISH),
    ("doji_star", "十字星", "CDLDOJISTAR", DIRECTION_NEUTRAL),
    ("hanging_man", "上吊线", "CDLHANGINGMAN", DIRECTION_BEARISH),
    ("hikkake_pattern", "陷阱", "CDLHIKKAKE", DIRECTION_BEARISH),
    ("modified_hikkake_pattern", "修正陷阱", "CDLHIKKAKEMOD", DIRECTION_BEARISH),
    ("in_neck_pattern", "颈内线", "CDLINNECK", DIRECTION_BEARISH),
    ("on_neck_pattern", "颈上线", "CDLONNECK", DIRECTION_BEARISH),
    ("thrusting_pattern", "插入", "CDLTHRUSTING", DIRECTION_BULLISH),
    ("shooting_star", "射击之星", "CDLSHOOTINGSTAR", DIRECTION_BEARISH),
    ("stalled_pattern", "停顿形态", "CDLSTALLEDPATTERN", DIRECTION_NEUTRAL),
    ("advance_block", "大敌当前", "CDLADVANCEBLOCK", DIRECTION_BEARISH),
    ("high_wave_candle", "风高浪大线", "CDLHIGHWAVE", DIRECTION_NEUTRAL),
    ("engulfing_pattern", "吞噬模式", "CDLENGULFING", DIRECTION_NEUTRAL),
    ("abandoned_baby", "弃婴", "CDLABANDONEDBABY", DIRECTION_BEARISH),
    ("closing_marubozu", "收盘缺影线", "CDLCLOSINGMARUBOZU", DIRECTION_NEUTRAL),
    ("doji", "十字", "CDLDOJI", DIRECTION_NEUTRAL),
    ("up_down_gap", "向上/下跳空并列阳线", "CDLGAPSIDESIDEWHITE", DIRECTION_NEUTRAL),
    ("long_legged_doji", "长脚十字", "CDLLONGLEGGEDDOJI", DIRECTION_NEUTRAL),
    ("rickshaw_man", "黄包车夫", "CDLRICKSHAWMAN", DIRECTION_NEUTRAL),
    ("marubozu", "光头光脚", "CDLMARUBOZU", DIRECTION_NEUTRAL),
    ("three_inside_up_down", "三内部上涨和下跌", "CDL3INSIDE", DIRECTION_NEUTRAL),
    ("three_outside_up_down", "三外部上涨和下跌", "CDL3OUTSIDE", DIRECTION_NEUTRAL),
    ("three_stars_in_the_south", "南方三星", "CDL3STARSINSOUTH", DIRECTION_BULLISH),
    ("three_white_soldiers", "三个白兵", "CDL3WHITESOLDIERS", DIRECTION_BULLISH),
    ("belt_hold", "捉腰带线", "CDLBELTHOLD", DIRECTION_BULLISH),
    ("breakaway", "脱离", "CDLBREAKAWAY", DIRECTION_BEARISH),
    ("concealing_baby_swallow", "藏婴吞没", "CDLCONCEALBABYSWALL", DIRECTION_BULLISH),
    ("counterattack", "反击线", "CDLCOUNTERATTACK", DIRECTION_BULLISH),
    ("dragonfly_doji", "蜻蜓十字", "CDLDRAGONFLYDOJI", DIRECTION_NEUTRAL),
    ("evening_star", "暮星", "CDLEVENINGSTAR", DIRECTION_BEARISH),
    ("gravestone_doji", "墓碑十字", "CDLGRAVESTONEDOJI", DIRECTION_BEARISH),
    ("hammer", "锤头", "CDLHAMMER", DIRECTION_BULLISH),
    ("harami_pattern", "母子线", "CDLHARAMI", DIRECTION_NEUTRAL),
    ("harami_cross_pattern", "十字孕线", "CDLHARAMICROSS", DIRECTION_NEUTRAL),
    ("homing_pigeon", "家鸽", "CDLHOMINGPIGEON", DIRECTION_BULLISH),
    ("inverted_hammer", "倒锤头", "CDLINVERTEDHAMMER", DIRECTION_BULLISH),
    ("kicking", "反冲形态", "CDLKICKING", DIRECTION_NEUTRAL),
    ("kicking_bull_bear", "由较长缺影线决定的反冲形态", "CDLKICKINGBYLENGTH", DIRECTION_NEUTRAL),
    ("ladder_bottom", "梯底", "CDLLADDERBOTTOM", DIRECTION_BULLISH),
    ("long_line_candle", "长蜡烛", "CDLLONGLINE", DIRECTION_NEUTRAL),
    ("matching_low", "相同低价", "CDLMATCHINGLOW", DIRECTION_BEARISH),
    ("mat_hold", "铺垫", "CDLMATHOLD", DIRECTION_BULLISH),
    ("morning_doji_star", "十字晨星", "CDLMORNINGDOJISTAR", DIRECTION_BULLISH),
    ("morning_star", "晨星", "CDLMORNINGSTAR", DIRECTION_BULLISH),
    ("piercing_pattern", "刺透形态", "CDLPIERCING", DIRECTION_BULLISH),
    ("rising_falling_three", "上升/下降三法", "CDLRISEFALL3METHODS", DIRECTION_NEUTRAL),
    ("separating_lines", "分离线", "CDLSEPARATINGLINES", DIRECTION_BULLISH),
    ("short_line_candle", "短蜡烛", "CDLSHORTLINE", DIRECTION_NEUTRAL),
    ("spinning_top", "纺锤", "CDLSPINNINGTOP", DIRECTION_NEUTRAL),
    ("stick_sandwich", "条形三明治", "CDLSTICKSANDWICH", DIRECTION_BEARISH),
    ("takuri", "探水竿", "CDLTAKURI", DIRECTION_BEARISH),
    ("tasuki_gap", "跳空并列阴阳线", "CDLTASUKIGAP", DIRECTION_NEUTRAL),
    ("tristar_pattern", "三星", "CDLTRISTAR", DIRECTION_NEUTRAL),
    ("unique_3_river", "奇特三河床", "CDLUNIQUE3RIVER", DIRECTION_BULLISH),
    ("upside_downside_gap", "上升/下降跳空三法", "CDLXSIDEGAP3METHODS", DIRECTION_NEUTRAL),
)

# 入场信号形态定义：(key, 中文名, 说明)
#
# 这类是"符合入场准备标准"的判定，直接对应筛选与回测验证的需求。
ENTRY_SIGNAL_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("volume_surge_up", "放量上涨", "涨跌幅≥2% 且 收盘>开盘 且 量比≥2 且 成交额≥2亿"),
    ("ma_bull_alignment", "均线多头", "MA5 > MA10 > MA20 且 价格站上 MA20"),
    ("breakout_ma", "突破均线", "收盘价上穿 MA20 且量能同步放大"),
    ("platform_breakout", "突破平台", "N 日横盘后放量突破区间上沿"),
    ("backtrace_ma250", "回踩年线", "回踩 MA250 获得支撑且当日收阳"),
    ("low_backtrace_increase", "低位回调上涨", "处于低位区间后放量收阳"),
    ("turtle_trade", "海龟交易法则", "突破唐奇安通道上轨（20 日）"),
    ("high_tight_flag", "高位紧旗", "连续涨停后缩量整理"),
    ("climax_limitdown", "放量跌停", "跌停且成交量显著放大"),
)

# 默认观察窗口（交易日）
DEFAULT_LOOKBACK = 120

# 放量上涨的成交额门槛（元）
TURNOVER_THRESHOLD = 2e8


@dataclass
class PatternHit:
    """单只个股在某一交易日命中的形态。

    Attributes:
        key: 形态标识
        name: 中文名
        direction: bullish / bearish / neutral
        category: standard / entry
        strength: 强度（TA-Lib 原始值绝对值；自定义形态固定 100）
        description: 判定说明
    """

    key: str
    name: str
    direction: str
    category: str
    strength: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "direction": self.direction,
            "category": self.category,
            "strength": self.strength,
            "description": self.description,
        }


@dataclass
class SymbolPatterns:
    """单只个股的形态识别结果。

    Attributes:
        symbol: 股票代码
        signal_date: 形态成立日（T 日收盘）
        earliest_entry_date: 最早可成交日（T+1 交易日）；无法确定时为 None
        hits: 命中的形态列表
    """

    symbol: str
    signal_date: str | None = None
    earliest_entry_date: str | None = None
    hits: list[PatternHit] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal_date": self.signal_date,
            "earliest_entry_date": self.earliest_entry_date,
            "hits": [h.to_dict() for h in self.hits],
            "bullish": [h.key for h in self.hits if h.direction == DIRECTION_BULLISH],
            "bearish": [h.key for h in self.hits if h.direction == DIRECTION_BEARISH],
            "neutral": [h.key for h in self.hits if h.direction == DIRECTION_NEUTRAL],
            "entry_signals": [h.key for h in self.hits if h.category == "entry"],
        }


def talib_available() -> bool:
    """TA-Lib 是否可用。不可用时标准形态识别会被跳过。"""
    return _TALIB_AVAILABLE


def _sma(values: Sequence[float], period: int) -> np.ndarray:
    """简单移动平均；不足周期处为 NaN。

    返回长度与输入一致（不足周期的位置为 NaN），
    这样调用方可以安全地取 ``[-1]`` 而不必先判断长度。
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.full(0, np.nan)
    out = np.full(arr.size, np.nan)
    if arr.size >= period:
        cumsum = np.cumsum(np.insert(arr, 0, 0.0))
        out[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """平均真实波幅。"""
    n = close.size
    if n == 0:
        return np.full(0, np.nan)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)),
    )
    tr[0] = high[0] - low[0]
    return _sma(tr, period)


def turnover_yuan(bars: pd.DataFrame) -> np.ndarray:
    """成交额（元）序列。

    优先使用 ``amount`` 列，并按 ``close*volume/amount`` 的中位数自动判定量纲
    （>1 视为"万元"，否则视为"元"）——与
    ``backend/services/simulation/services/local_market_data.py`` 的
    ``_detect_amount_scale`` 同一口径，避免成交额偏 1e4 倍。

    无 ``amount`` 列时退化为 ``close*volume``（前复权价下为近似值）。
    """
    if bars is None or bars.empty:
        return np.full(0, np.nan)
    close = bars["close"].to_numpy(dtype=float)
    if "volume" in bars.columns:
        vol = bars["volume"].to_numpy(dtype=float)
    else:
        vol = np.zeros_like(close)
    estimate = close * vol

    if "amount" not in bars.columns:
        return estimate

    amount = bars["amount"].to_numpy(dtype=float)
    mask = (
        np.isfinite(estimate)
        & np.isfinite(amount)
        & (estimate > 0)
        & (amount > 0)
    )
    if not mask.any():
        return estimate
    ratio = float(np.median(estimate[mask] / amount[mask]))
    if ratio > 1.0:
        # amount 单位为万元 → 换算为元
        return amount * 1e4
    return amount


def detect_standard_patterns(
    bars: pd.DataFrame,
    keys: Iterable[str] | None = None,
) -> list[PatternHit]:
    """识别 TA-Lib 标准 K 线形态。

    Args:
        bars: 至少包含 open/high/low/close 的 DataFrame，按时间升序。
        keys: 仅识别指定形态 key；None 表示全部。

    Returns:
        命中的形态列表（按强度排序）。
    """
    if not _TALIB_AVAILABLE or bars is None or bars.empty:
        return []
    if not {"open", "high", "low", "close"}.issubset(bars.columns):
        return []

    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)

    wanted = set(keys) if keys is not None else None
    hits: list[PatternHit] = []

    for key, name, func_name, direction in STANDARD_PATTERNS:
        if wanted is not None and key not in wanted:
            continue
        func = getattr(tl, func_name, None)
        if func is None:
            continue
        try:
            raw = func(o, h, l, c)
        except Exception as exc:  # pragma: no cover - 单个形态异常不影响整体
            logger.debug("形态 %s 计算失败: %s", func_name, exc)
            continue
        if raw is None or len(raw) == 0:
            continue
        last = int(raw[-1])
        if last == 0:
            continue
        hits.append(
            PatternHit(
                key=key,
                name=name,
                direction=direction,
                category="standard",
                strength=int(abs(last)),
                description=f"TA-Lib {func_name} 于最后一根 K 线命中（原始值 {last}）",
            )
        )

    hits.sort(key=lambda x: (-abs(x.strength), x.key))
    return hits


def _entry_signal_hits(bars: pd.DataFrame) -> list[PatternHit]:
    """识别 9 类入场信号形态。

    判定全部使用截至最后一根 K 线的数据，不含未来函数。
    """
    hits: list[PatternHit] = []
    if bars is None or bars.empty or len(bars) < 21:
        return hits

    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    l = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    v = bars["volume"].to_numpy(dtype=float) if "volume" in bars.columns else np.zeros_like(c)
    turnover = turnover_yuan(bars)

    last = -1
    close = c[last]
    open_ = o[last]
    high_last = h[last]
    low_last = l[last]
    vol = v[last]

    def add(key: str, desc: str) -> None:
        for item in ENTRY_SIGNAL_PATTERNS:
            if item[0] == key:
                hits.append(
                    PatternHit(
                        key=key,
                        name=item[1],
                        direction=DIRECTION_BULLISH,
                        category="entry",
                        strength=100,
                        description=desc,
                    )
                )
                return

    # --- 放量上涨 ---
    prev_close = c[-2] if len(c) >= 2 else close
    pct_change = ((close - prev_close) / prev_close * 100.0) if prev_close else 0.0
    vol_ma5 = _sma(v, 5)
    mean_vol5 = vol_ma5[-2] if len(vol_ma5) >= 2 else np.nan
    amount = turnover[last] if turnover.size else close * vol  # 成交额（元）
    vol_ratio = (
        vol / mean_vol5
        if mean_vol5 and not np.isnan(mean_vol5) and mean_vol5 > 0
        else np.nan
    )
    if (
        pct_change >= 2.0
        and close >= open_
        and not np.isnan(vol_ratio)
        and vol_ratio >= 2.0
        and amount >= TURNOVER_THRESHOLD
    ):
        add(
            "volume_surge_up",
            f"涨幅 {pct_change:.2f}%，量比 {vol_ratio:.2f}，成交额 {amount/1e8:.2f} 亿",
        )

    # --- 均线多头 ---
    ma5, ma10, ma20 = _sma(c, 5), _sma(c, 10), _sma(c, 20)
    if not any(np.isnan(x[-1]) for x in (ma5, ma10, ma20)):
        if ma5[-1] > ma10[-1] > ma20[-1] and close > ma20[-1]:
            add(
                "ma_bull_alignment",
                f"MA5 {ma5[-1]:.2f} > MA10 {ma10[-1]:.2f} > MA20 {ma20[-1]:.2f}，收盘站上 MA20",
            )

    # --- 突破均线（MA20 上穿 + 量能放大）---
    if len(c) >= 21 and not np.isnan(ma20[-1]) and not np.isnan(ma20[-2]):
        if close > ma20[-1] and c[-2] <= ma20[-2]:
            if not np.isnan(vol_ratio) and vol_ratio >= 1.5:
                add(
                    "breakout_ma",
                    f"收盘 {close:.2f} 上穿 MA20 {ma20[-1]:.2f}，量比 {vol_ratio:.2f}",
                )

    # --- 突破平台（N 日横盘后放量突破上沿）---
    lookback = 20
    if len(c) >= lookback + 2:
        window = h[-lookback - 1:-1]
        upper = float(np.nanmax(window))
        flat_enough = float(np.nanmax(c[-lookback:-1])) <= upper * 1.02
        if (
            close > upper
            and not np.isnan(vol_ratio)
            and vol_ratio >= 1.5
            and flat_enough
        ):
            add(
                "platform_breakout",
                f"收盘 {close:.2f} 突破 {lookback} 日平台上沿 {upper:.2f}，量比 {vol_ratio:.2f}",
            )

    # --- 回踩年线 ---
    ma250 = _sma(c, 250)
    if len(c) >= 250 and not np.isnan(ma250[-1]):
        if low_last <= ma250[-1] * 1.02 and close > open_ and close > ma250[-1]:
            add("backtrace_ma250", f"最低 {low_last:.2f} 回踩 MA250 {ma250[-1]:.2f} 后收阳")

    # --- 低位回调上涨 ---
    if len(c) >= 60:
        window60 = c[-60:]
        span = window60.max() - window60.min()
        pos = (close - window60.min()) / span if span > 0 else 0.5
        if pos <= 0.35 and close > open_ and not np.isnan(vol_ratio) and vol_ratio >= 1.5:
            add("low_backtrace_increase", f"处于 60 日区间 {pos*100:.0f}% 低位后放量收阳")

    # --- 海龟交易法则（唐奇安上轨突破）---
    if len(c) >= 21:
        upper20 = float(np.nanmax(h[-21:-1]))
        if close > upper20 and close > open_ and not np.isnan(vol_ratio) and vol_ratio >= 1.2:
            add("turtle_trade", f"收盘 {close:.2f} 突破 20 日唐奇安上轨 {upper20:.2f}")

    # --- 高位紧旗（连续涨停后缩量）---
    if len(c) >= 10:
        tail = slice(-5, None)
        o_tail, c_tail = o[tail], c[tail]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct_tail = np.where(o_tail != 0, (c_tail - o_tail) / o_tail * 100.0, 0.0)
        limit_up = (pct_tail >= 9.8).tolist()
        if len(limit_up) >= 3 and all(limit_up[-3:]):
            vol_ma5_now = _sma(v, 5)
            if (
                not np.isnan(vol_ma5_now[-1])
                and vol_ma5_now[-1] > 0
                and vol < vol_ma5_now[-1] * 0.6
            ):
                add("high_tight_flag", "近 3 日连续涨停后缩量整理")

    # --- 放量跌停 ---
    vol_ma5_prev = _sma(v, 5)
    base = vol_ma5_prev[-2] if len(vol_ma5_prev) >= 2 else np.nan
    if prev_close and (prev_close - close) / prev_close * 100 >= 9.8:
        if close <= low_last * 1.001 and not np.isnan(base) and base > 0 and vol >= base * 2:
            add("climax_limitdown", "跌停且成交量较 5 日均量放大 2 倍以上")

    return hits


def detect_patterns(
    bars: pd.DataFrame,
    include_entry: bool = True,
    include_standard: bool = True,
    keys: Iterable[str] | None = None,
) -> list[PatternHit]:
    """识别全部形态（标准 + 入场信号）。

    Args:
        bars: OHLCV DataFrame，按时间升序（最后一根为判定日 T）。
        include_entry: 是否识别入场信号形态。
        include_standard: 是否识别 TA-Lib 标准形态。
        keys: 仅保留指定 key；None 则不过滤。

    Returns:
        命中的形态列表（入场信号优先）。
    """
    hits: list[PatternHit] = []
    if bars is None or bars.empty:
        return hits

    if include_standard:
        hits.extend(detect_standard_patterns(bars))
    if include_entry:
        hits.extend(_entry_signal_hits(bars))

    if keys is not None:
        wanted = set(keys)
        hits = [x for x in hits if x.key in wanted]

    hits.sort(key=lambda x: (0 if x.category == "entry" else 1, -abs(x.strength), x.key))
    return hits


def next_trading_day(dates: Sequence[str], signal_date: str) -> str | None:
    """求 signal_date 之后的第一个交易日（T+1）。

    Args:
        dates: 升序排列的交易日列表（YYYY-MM-DD）。
        signal_date: 形态成立日。

    Returns:
        下一个交易日；已是最后一天时返回 None。
    """
    if not dates:
        return None
    for d in dates:
        if d > signal_date:
            return d
    return None


def scan_symbol(
    symbol: str,
    bars: pd.DataFrame,
    trading_dates: Sequence[str] | None = None,
    include_entry: bool = True,
    include_standard: bool = True,
    keys: Iterable[str] | None = None,
) -> SymbolPatterns:
    """扫描单只个股在最后一根 K 线上的形态。

    Args:
        symbol: 股票代码。
        bars: OHLCV DataFrame，按时间升序。
        trading_dates: 升序交易日列表，用于推算 T+1；None 则不推算。
        include_entry: 是否识别入场信号形态。
        include_standard: 是否识别标准形态。
        keys: 仅保留指定 key。

    Returns:
        SymbolPatterns 结果。
    """
    result = SymbolPatterns(symbol=symbol)
    if bars is None or bars.empty:
        return result

    if "date" in bars.columns:
        result.signal_date = str(bars["date"].iloc[-1])[:10]

    result.hits = detect_patterns(
        bars,
        include_entry=include_entry,
        include_standard=include_standard,
        keys=keys,
    )

    if trading_dates and result.signal_date:
        result.earliest_entry_date = next_trading_day(trading_dates, result.signal_date)
    return result


def scan_universe(
    bars_by_symbol: dict[str, pd.DataFrame],
    trading_dates: Sequence[str] | None = None,
    include_entry: bool = True,
    include_standard: bool = True,
    keys: Iterable[str] | None = None,
) -> list[SymbolPatterns]:
    """批量扫描股票池。

    Args:
        bars_by_symbol: {symbol: OHLCV DataFrame}。
        trading_dates: 升序交易日列表。
        include_entry: 是否识别入场信号形态。
        include_standard: 是否识别标准形态。
        keys: 仅保留指定 key。

    Returns:
        每个有命中的个股一条结果。
    """
    out: list[SymbolPatterns] = []
    for symbol, bars in bars_by_symbol.items():
        res = scan_symbol(
            symbol,
            bars,
            trading_dates=trading_dates,
            include_entry=include_entry,
            include_standard=include_standard,
            keys=keys,
        )
        if res.hits:
            out.append(res)
    return out


def aggregate_patterns(results: Sequence[SymbolPatterns]) -> list[dict[str, Any]]:
    """按形态聚合统计命中数量。

    Returns:
        [{key, name, direction, category, count, symbols}]，按 count 降序。
    """
    agg: dict[str, dict[str, Any]] = {}
    for res in results:
        for hit in res.hits:
            item = agg.setdefault(
                hit.key,
                {
                    "key": hit.key,
                    "name": hit.name,
                    "direction": hit.direction,
                    "category": hit.category,
                    "count": 0,
                    "symbols": [],
                },
            )
            item["count"] += 1
            if len(item["symbols"]) < 200:
                item["symbols"].append(res.symbol)
    return sorted(agg.values(), key=lambda x: (-x["count"], x["key"]))


def pattern_catalog() -> list[dict[str, Any]]:
    """返回全部形态的目录（前端下拉框用）。"""
    catalog: list[dict[str, Any]] = []
    for key, name, func_name, direction in STANDARD_PATTERNS:
        catalog.append(
            {
                "key": key,
                "name": name,
                "direction": direction,
                "category": "standard",
                "source": func_name,
                "description": f"TA-Lib {func_name}",
            }
        )
    for key, name, desc in ENTRY_SIGNAL_PATTERNS:
        catalog.append(
            {
                "key": key,
                "name": name,
                "direction": DIRECTION_BULLISH,
                "category": "entry",
                "source": "custom",
                "description": desc,
            }
        )
    return catalog
