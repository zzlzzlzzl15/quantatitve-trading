"""板块级 K 线形态分析 API —— 形态信息簇在板块间的相对分布。

薄层: 计算在 ``app.services.pattern_sector_analysis``。
"""
from __future__ import annotations

from datetime import date
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request

from app.services import pattern_sector_analysis

router = APIRouter(prefix="/api/pattern-sector", tags=["pattern-sector"])


@router.get("/report")
def get_pattern_sector_report(
    request: Request,
    as_of: date | None = Query(None, description="分析日期 (YYYY-MM-DD); 缺省取最新有数据日"),
    kind: Literal["concept", "industry"] = Query("concept", description="板块维度: 概念/行业 二选一"),
    level: int | None = Query(
        None, ge=1, le=5, description="行业层级 (同花顺 一级/二级/三级); 仅 kind=industry 有效, 缺省不过滤",
    ),
    limit: int = Query(20, ge=1, le=100, description="返回板块数上限"),
    min_members: int = Query(5, ge=1, le=200, description="成分数门槛: 低于此值的板块不参与统计"),
    candidate_limit: int = Query(20, ge=1, le=200, description="每板块返回的入场候选数上限"),
    sort_by: Literal["setup_score", "relative_breakout", "setup_count"] = Query(
        "setup_score",
        description="板块排序维度: setup_score=低位相对强且突破相对弱 / relative_breakout=突破簇相对倍数 / setup_count=候选数",
    ),
):
    """指定交易日的板块级形态结构 (相对全市场基准)。

    返回三部分:
      - ``market``: 全市场形态基准 (板块相对占比的分母);
      - ``sectors``: 各板块的信息簇占比 + 相对全市场的倍数 + 入场候选;
      - ``candidates``: 全市场「符合入场准备标准」的个股 (低位回调上涨成立且无回避形态)。

    绝对命中率不具板块可比性 (9 种形态的全市场超额全为负) —— 这里只给相对分布。
    """
    try:
        return pattern_sector_analysis.build_pattern_sector_report(
            request.app.state.repo,
            as_of,
            kind=kind,
            level=level,
            limit=limit,
            min_members=min_members,
            candidate_limit=candidate_limit,
            sort_by=sort_by,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
