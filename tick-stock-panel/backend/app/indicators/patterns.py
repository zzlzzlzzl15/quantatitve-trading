"""K 线形态识别 —— 入场信号形态 (纯 Polars 向量化)。

判定口径与 QuantMind ``backend/shared/kline_patterns.py`` 的
``_entry_signal_hits`` 一致 (只用截至 T 日收盘的数据, 不含未来函数),
但改写为全 Polars 表达式: 按 ``symbol`` 分组一次算完整股票池,
不做逐标的 Python 循环, 与本仓其它指标同一套 ``.over("symbol")`` 口径。

形态列命名 ``pat_<key>`` (数值哑变量: 1.0 = 形态成立, 0.0 = 不成立),
由 ``compute_indicators`` 产出 —— 因此可被自定义信号 (字段白名单含注册表
因子)、评分因子、回测 ``REQUIRED_FEATURES`` 直接引用, 无需额外接线。

为何用数值而非布尔:
  - 本仓因子口径为数值 (布尔不参与因子 IC/排序, 见 factors/ext_factors.py);
  - 自定义信号条件的右值只能是数值或字段引用, 布尔列无法与常量比较;
  - 回测矩阵只收数值列 (``build_market_data_matrix`` 的 is_numeric 门控)。

时点口径:
  - 形态成立日 = T 日收盘; 最早可成交日 = T+1 交易日。
  - 成交价由回测策略的 ``matching`` 配置决定 (默认 ``open_t+1``);
    本模块只做识别, 不含成交与持仓逻辑。

形态是**状态**而非**事件**: 同一形态可连续多日成立。回测取信号应按
"边沿触发" (False -> True) 使用, 避免在形态持续期内重复建仓。

降级路径:
  - ``amount`` 列缺失时 ``pat_volume_surge_up`` 全为 False (无法判定成交额
    门槛 ⇒ 不成立) —— fail-closed, 不静默放宽过滤。
  - 历史不足 (窗口未满) 时对应形态为 False, 与指标列的 null 语义一致。
  - 历史长度不足 21 根时所有形态均不判定 (对齐 QuantMind 入口早退条件);
    突破平台另需 22 根。

与 QuantMind 的口径差异 (已知且可接受):
  - QuantMind 逐标的用 pandas/numpy 逐根判定, 本模块按 ``symbol`` 分组一次
    向量算出整池结果; 二者在同一根 K 线的判定结果一致, 但当单根 K 线的
    历史长度 < 21 时, QuantMind 直接返回空 (不判定), 本模块对**每一行**
    独立套用 21 根门槛 —— 等价且更通用。
  - QuantMind 的 ``turnover_yuan`` 会自动判定成交额量纲 (万/元); 本仓
    ``amount`` 统一为元, 故不需量纲换算。
"""
from __future__ import annotations

import polars as pl

# 形态列名前缀
PATTERN_PREFIX = "pat_"

# 入场信号形态定义: (key, 中文名, 判定说明)
#
# 这类形态对应"符合入场准备标准"的判定, 是选股筛选与回测验证的直接抓手。
# 本表是形态元数据的单一事实源; 因子注册表 (factors/registry.py) 镜像其中的
# 中文名与判定说明, 由 tests/test_indicators_patterns.py 兜底防漂移。
ENTRY_SIGNAL_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("volume_surge_up", "放量上涨", "涨跌幅≥2% 且 收盘>开盘 且 量比≥2 且 成交额≥2亿"),
    ("ma_bull_alignment", "均线多头", "MA5 > MA10 > MA20 且 价格站上 MA20"),
    ("breakout_ma", "突破均线", "收盘价上穿 MA20 且量能同步放大"),
    ("platform_breakout", "突破平台", "20 日横盘后放量突破区间上沿"),
    ("backtrace_ma250", "回踩年线", "回踩 MA250 获得支撑且当日收阳"),
    ("low_backtrace_increase", "低位回调上涨", "处于低位区间后放量收阳"),
    ("turtle_trade", "海龟交易法则", "突破唐奇安通道上轨（20 日）"),
    ("high_tight_flag", "高位紧旗", "连续涨停后缩量整理"),
    ("climax_limitdown", "放量跌停", "跌停且成交量显著放大"),
)

# 放量上涨的成交额门槛 (元) —— 与 QuantMind 同值
TURNOVER_THRESHOLD = 2e8

# 涨停 / 跌停判定阈值 (%, 相对前收盘或当日开盘)
LIMIT_UP_PCT = 9.8
LIMIT_DOWN_PCT = 9.8

# 量比分母: 前 5 个交易日均量 (不含当日)。与 QuantMind ``vol_ma5[-2]`` 同口径,
# 亦与本仓 ``vol_ratio_5d`` 的 ``_vol_ma5_prev`` 同口径。
_VOL_BASE_WINDOW = 5

# 历史长度门槛 —— 对齐 QuantMind ``_entry_signal_hits`` 的入口早退条件
# (len(bars) < 21 ⇒ 直接返回, 一类形态都不判定)。向量化的等价形式是
# "被判定的那一根 K 线, 其自身已收盘的历史长度 >= 21"。
_MIN_BARS = 21

# 突破平台的额外门槛 (QuantMind: len(c) >= lookback + 2 = 22)
_PLATFORM_MIN_BARS = 22

# 平台突破的横盘观察窗口 (交易日)
_PLATFORM_WINDOW = 20

# 低位回调的区间观察窗口 (交易日)
_LOW_WINDOW = 60

# 年线周期
_MA_LONG_WINDOW = 250


# 形态判定所需的基础行情列 (不含 OHLCV)。
# 面板加载时必须把这些列一并读出: 否则 amount 缺失会让 pat_volume_surge_up
# 退化为恒 0 (fail-closed), 表现为"放量上涨"信号永远不触发。
PATTERN_BASE_COLUMNS: frozenset[str] = frozenset({"amount"})


def pattern_column(key: str) -> str:
    """形态 key → 输出列名。"""
    return f"{PATTERN_PREFIX}{key}"


# 形态列名 (与 ENTRY_SIGNAL_PATTERNS 同序)
PATTERN_COLUMNS: tuple[str, ...] = tuple(
    pattern_column(key) for key, _, _ in ENTRY_SIGNAL_PATTERNS
)

# 中间列 (形态计算工件, 计算完成后丢弃)
MID_COLUMNS: frozenset[str] = frozenset({
    "_pat_vol_ma5_base",
    "_pat_vol_ratio",
    "_pat_ma250",
    "_pat_low60",
    "_pat_high60",
    "_pat_upper20",
    "_pat_flat_max19",
    "_pat_bar_count",
})

# 中间列 → 中间列依赖 (供 compute_indicators 的 needed 闭包展开)
MID_DEPENDENCIES: dict[str, frozenset[str]] = {
    "_pat_vol_ratio": frozenset({"_pat_vol_ma5_base"}),
}

# 形态列 → 依赖列 (指标列 + 中间列)。缺列即无法判定 ⇒ 形态不成立。
# 每个形态都依赖 ``_pat_bar_count`` —— 历史长度门槛 (QuantMind 的 len(bars) 检查)。
PATTERN_DEPENDENCIES: dict[str, frozenset[str]] = {
    "pat_volume_surge_up": frozenset({"prev_close", "_pat_vol_ratio", "_pat_bar_count"}),
    "pat_ma_bull_alignment": frozenset({"ma5", "ma10", "ma20", "_pat_bar_count"}),
    "pat_breakout_ma": frozenset({"ma20", "_pat_vol_ratio", "_pat_bar_count"}),
    "pat_platform_breakout": frozenset({
        "_pat_vol_ratio", "_pat_upper20", "_pat_flat_max19", "_pat_bar_count",
    }),
    "pat_backtrace_ma250": frozenset({"_pat_ma250", "_pat_bar_count"}),
    "pat_low_backtrace_increase": frozenset({
        "_pat_low60", "_pat_high60", "_pat_vol_ratio", "_pat_bar_count",
    }),
    "pat_turtle_trade": frozenset({"_pat_vol_ratio", "_pat_upper20", "_pat_bar_count"}),
    "pat_high_tight_flag": frozenset({"vol_ma5", "_pat_bar_count"}),
    "pat_climax_limitdown": frozenset({"prev_close", "_pat_vol_ma5_base", "_pat_bar_count"}),
}

# 各形态的最小预热根数 (= 历史长度门槛与依赖窗口的较大者)。
# 注册表的 warmup_bars 必须与此一致 (由 tests/test_indicators_patterns.py 兜底防漂移)。
PATTERN_WARMUP_BARS: dict[str, int] = {
    "pat_volume_surge_up": _MIN_BARS,
    "pat_ma_bull_alignment": _MIN_BARS,
    "pat_breakout_ma": _MIN_BARS,
    "pat_platform_breakout": _PLATFORM_MIN_BARS,
    "pat_backtrace_ma250": _MA_LONG_WINDOW,
    "pat_low_backtrace_increase": _LOW_WINDOW,
    "pat_turtle_trade": _MIN_BARS,
    "pat_high_tight_flag": _MIN_BARS,
    "pat_climax_limitdown": _MIN_BARS,
}


def build_mid_exprs() -> dict[str, pl.Expr]:
    """中间列表达式 (形态共用的滚动窗口工件)。

    窗口未满的位置为 null —— 与 QuantMind 的 ``_sma`` / ``nanmax`` 语义一致,
    形态判定因此在历史不足时自然不成立。
    """
    volume = pl.col("volume")
    close = pl.col("close")
    high = pl.col("high")
    # 前 N 日均量 (不含当日): 作为量比分母与放量基准
    base = volume.shift(1).rolling_mean(_VOL_BASE_WINDOW).over("symbol")
    return {
        "_pat_vol_ma5_base": base,
        # 量比: 分母非正 (窗口未满, 或均量因停牌而为 0) 时置 null ——
        # 对齐 QuantMind 的 np.nan 语义 (mean_vol5 <= 0 ⇒ 量比不可判定 ⇒ 不成立)。
        # 不能直接写 volume / base: 分母为 0 时会得到 inf 而误判为“放量”。
        "_pat_vol_ratio": pl.when(base > 0).then(volume / base),
        "_pat_ma250": close.rolling_mean(_MA_LONG_WINDOW).over("symbol"),
        "_pat_low60": close.rolling_min(_LOW_WINDOW).over("symbol"),
        "_pat_high60": close.rolling_max(_LOW_WINDOW).over("symbol"),
        # 唐奇安上轨: 前 20 日最高价 (不含当日)
        "_pat_upper20": high.shift(1).rolling_max(_PLATFORM_WINDOW).over("symbol"),
        # 平台横盘判定: 前 19 日收盘最高价 (不含当日, 与 QuantMind c[-20:-1] 同长)
        "_pat_flat_max19": close.shift(1).rolling_max(_PLATFORM_WINDOW - 1).over("symbol"),
        # 已收盘 K 线根数 (每标的各自计数) —— 历史长度门槛的判定依据
        "_pat_bar_count": close.cum_count().over("symbol"),
    }


def build_pattern_exprs(amount_available: bool = True) -> dict[str, pl.Expr]:
    """返回 {形态列名: 数值表达式} (1.0 = 成立, 0.0 = 不成立)。

    Args:
        amount_available: 帧内是否有 ``amount`` 列。缺失时 ``pat_volume_surge_up``
            退化为恒 0 (成交额门槛无法判定 ⇒ 不成立)。

    依赖的中间列必须已存在于帧中 (见 ``build_mid_exprs``)。
    """
    o = pl.col("open")
    low = pl.col("low")
    c = pl.col("close")
    v = pl.col("volume")
    prev_close = pl.col("prev_close")

    vol_ratio = pl.col("_pat_vol_ratio")
    vol_base = pl.col("_pat_vol_ma5_base")
    ma5, ma10, ma20 = pl.col("ma5"), pl.col("ma10"), pl.col("ma20")
    vol_ma5 = pl.col("vol_ma5")
    bar_count = pl.col("_pat_bar_count")

    # 前收盘缺失 (首行 / 上一日停牌) 时按 0 处理 ⇒ 涨跌幅为 0 ⇒ 不成立。
    prev_close_safe = prev_close.fill_null(0.0)

    # 相对前收盘涨跌幅 (%)
    pct_change = (
        pl.when(prev_close_safe > 0)
        .then((c - prev_close_safe) / prev_close_safe * 100.0)
        .otherwise(0.0)
    )
    # 当日实体涨跌幅 (%)
    intraday_pct = pl.when(o != 0).then((c - o) / o * 100.0).otherwise(0.0)
    # 相对前收盘跌幅 (%)
    drop_pct = (
        pl.when(prev_close_safe > 0)
        .then((prev_close_safe - c) / prev_close_safe * 100.0)
        .otherwise(0.0)
    )

    exprs: dict[str, pl.Expr] = {}

    # 放量上涨: 涨跌幅≥2% 且 收阳 且 量比≥2 且 成交额≥2亿
    if amount_available:
        exprs[pattern_column("volume_surge_up")] = (
            (pct_change >= 2.0)
            & (c >= o)
            & (vol_ratio >= 2.0)
            & (pl.col("amount") >= TURNOVER_THRESHOLD)
        )
    else:
        exprs[pattern_column("volume_surge_up")] = pl.lit(False)
    # 均线多头: MA5 > MA10 > MA20 且 收盘站上 MA20
    exprs[pattern_column("ma_bull_alignment")] = (ma5 > ma10) & (ma10 > ma20) & (c > ma20)

    # 突破均线: 收盘上穿 MA20 (前一日收盘在 MA20 之下) 且量能放大
    exprs[pattern_column("breakout_ma")] = (
        (c > ma20)
        & (prev_close <= ma20.shift(1).over("symbol"))
        & (vol_ratio >= 1.5)
    )

    # 突破平台: 20 日横盘后放量突破区间上沿
    upper20 = pl.col("_pat_upper20")
    exprs[pattern_column("platform_breakout")] = (
        (c > upper20)
        & (vol_ratio >= 1.5)
        & (pl.col("_pat_flat_max19") <= upper20 * 1.02)
        & (bar_count >= _PLATFORM_MIN_BARS)
    )

    # 回踩年线: 最低价回踩 MA250 (2% 容差) 后收阳并站上 MA250
    ma250 = pl.col("_pat_ma250")
    exprs[pattern_column("backtrace_ma250")] = (low <= ma250 * 1.02) & (c > o) & (c > ma250)

    # 低位回调上涨: 处于 60 日区间低位 (≤35%) 后放量收阳
    low60 = pl.col("_pat_low60")
    high60 = pl.col("_pat_high60")
    span = high60 - low60
    position = pl.when(span > 0).then((c - low60) / span).otherwise(0.5)
    exprs[pattern_column("low_backtrace_increase")] = (
        (position <= 0.35) & (c > o) & (vol_ratio >= 1.5)
    )

    # 海龟交易法则: 突破 20 日唐奇安上轨
    exprs[pattern_column("turtle_trade")] = (c > upper20) & (c > o) & (vol_ratio >= 1.2)

    # 高位紧旗: 连续 3 日涨停后缩量整理
    limit_up = intraday_pct >= LIMIT_UP_PCT
    exprs[pattern_column("high_tight_flag")] = (
        limit_up
        & limit_up.shift(1).over("symbol")
        & limit_up.shift(2).over("symbol")
        & (v < vol_ma5 * 0.6)
    )

    # 放量跌停: 跌停且成交量较 5 日均量放大 2 倍以上
    exprs[pattern_column("climax_limitdown")] = (
        (drop_pct >= LIMIT_DOWN_PCT)
        & (c <= low * 1.001)
        & (vol_base > 0)
        & (v >= vol_base * 2)
    )

    # 历史长度门槛: 不足 21 根历史的一律不判定 (对齐 QuantMind 入口早退条件)。
    history_ok = bar_count >= _MIN_BARS
    exprs = {name: (expr & history_ok) for name, expr in exprs.items()}

    # 统一转成 0/1 数值哑变量: null (窗口未满/依赖缺失) 一律算作不成立。
    return {
        name: expr.fill_null(False).cast(pl.Float64)
        for name, expr in exprs.items()
    }
