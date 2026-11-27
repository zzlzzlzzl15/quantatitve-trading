/**
 * 板块形态结构卡片 —— 形态信息簇在板块间的**相对分布**。
 *
 * 口径 (后端 app/services/pattern_sector_analysis.py, 依据
 * reports/pattern_hit_stats/REPORT.md): 9 种 K 线形态的超额收益全为负
 * (t = -4.2 ~ -45.6), 因此绝对命中率在板块之间**不可比** —— 全市场突破簇占比
 * 就是那么高。这里只展示「板块占比 ÷ 全市场占比」的倍数, 回答"这个板块的形态
 * 结构与全市场有什么不同"。
 *
 * 形态按共现结构归并成 4 个信息簇 (突破/追涨/低位/独立): 突破平台 / 海龟突破 /
 * 均线多头 互共现 75%~92%, 本质同一现象, 拆开看只会看到重复信号。
 */
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ChevronDown, ChevronRight, Layers } from 'lucide-react'
import { api, type PatternSectorRow } from '@/lib/api'
import { QK } from '@/lib/queryKeys'
import { cn } from '@/lib/cn'
import { fmtPrice, fmtStockPct, pctClass, SectionTitle, StockNameButton } from './shared'

type ClusterKey = 'breakout' | 'chase' | 'pullback' | 'standalone'

/** 信息簇展示顺序 (按对交易的含义: 回避 → 正向 → 中性) */
const CLUSTER_ORDER: ClusterKey[] = ['breakout', 'chase', 'pullback', 'standalone']

/** 相对倍数配色: 突破/追涨偏高偏红 (追高风险), 低位偏高偏绿 (回调到位) */
function relativeClass(key: ClusterKey, value: number | null): string {
  if (value == null) return 'text-muted'
  if (key === 'pullback') return value >= 1 ? 'text-bull' : 'text-muted'
  if (key === 'breakout' || key === 'chase') return value >= 1.5 ? 'text-bear' : 'text-muted'
  return 'text-muted'
}

function fmtRelative(value: number | null): string {
  if (value == null) return '—'
  return `${value.toFixed(2)}×`
}

function SectorRow({
  row,
  expanded,
  onToggle,
  onStockClick,
}: {
  row: PatternSectorRow
  expanded: boolean
  onToggle: () => void
  onStockClick: (symbol: string, name?: string) => void
}) {
  const breakout = row.clusters.breakout
  const pullback = row.clusters.pullback
  return (
    <div className="rounded-md border border-border/40 bg-surface/60">
      <button
        type="button"
        onClick={onToggle}
        aria-expanded={expanded}
        className="flex w-full items-center gap-1.5 px-1.5 py-1 text-left hover:bg-elevated/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-accent"
      >
        {expanded
          ? <ChevronDown className="h-3 w-3 shrink-0 text-muted" aria-hidden="true" />
          : <ChevronRight className="h-3 w-3 shrink-0 text-muted" aria-hidden="true" />}
        <span className="min-w-0 flex-1 truncate text-xs text-foreground" title={row.name}>{row.name}</span>
        <span className="shrink-0 font-mono text-[10px] text-muted">{row.valid_count}</span>
        <span className={cn('w-12 shrink-0 text-right font-mono text-[10px]', relativeClass('breakout', breakout?.relative ?? null))}>
          {fmtRelative(breakout?.relative ?? null)}
        </span>
        <span className={cn('w-12 shrink-0 text-right font-mono text-[10px]', relativeClass('pullback', pullback?.relative ?? null))}>
          {fmtRelative(pullback?.relative ?? null)}
        </span>
        <span className="w-8 shrink-0 text-right font-mono text-[10px] text-accent">{row.setup_count}</span>
      </button>
      {expanded && (
        <div className="border-t border-border/40 px-1.5 py-1">
          {row.candidates.length === 0 ? (
            <div className="py-1 text-center text-[10px] text-muted">该板块当日无符合入场准备标准的个股</div>
          ) : (
            <div className="space-y-0.5">
              {row.candidates.map((c) => (
                <div key={c.symbol} className="flex items-center gap-1.5 text-[10px]">
                  <StockNameButton
                    symbol={c.symbol}
                    name={c.name}
                    onClick={onStockClick}
                    className="min-w-0 flex-1 text-left text-foreground"
                  />
                  <span className={cn('shrink-0 font-mono', pctClass(c.change_pct))}>{fmtStockPct(c.change_pct)}</span>
                  <span className="shrink-0 font-mono text-muted">{fmtPrice(c.close)}</span>
                  <span className="hidden shrink-0 text-muted sm:inline">{c.pattern_labels.join('/')}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export function PatternSectorWidget({
  onStockClick,
  asOf,
}: {
  onStockClick: (symbol: string, name?: string) => void
  asOf?: string
}) {
  const [kind, setKind] = useState<'concept' | 'industry'>('concept')
  const [sortBy, setSortBy] = useState<'setup_score' | 'relative_breakout'>('setup_score')
  const [expanded, setExpanded] = useState<string | null>(null)

  const query = useQuery({
    queryKey: QK.patternSectorReport(kind, asOf, sortBy),
    queryFn: () => api.patternSectorReport({ kind, asOf, limit: 20, minMembers: 5, sortBy }),
    staleTime: 5 * 60 * 1000,
  })
  const report = query.data

  const market = report?.market
  const sectors = report?.sectors ?? []

  return (
    <div className="flex h-full min-h-0 flex-col">
      <SectionTitle
        icon={Layers}
        title="板块形态结构"
        hint={report?.as_of ?? undefined}
      />

      <div className="mb-1.5 flex shrink-0 items-center gap-1">
        {(['concept', 'industry'] as const).map((k) => (
          <button
            key={k}
            type="button"
            onClick={() => setKind(k)}
            className={cn(
              'rounded px-1.5 py-0.5 text-[10px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60',
              kind === k ? 'bg-accent/15 text-accent' : 'text-muted hover:bg-elevated hover:text-secondary',
            )}
          >
            {k === 'concept' ? '概念' : '行业'}
          </button>
        ))}
        <span className="ml-auto flex items-center gap-1">
          <button
            type="button"
            onClick={() => setSortBy('setup_score')}
            title="按低位相对强 / 突破相对弱排序"
            className={cn(
              'rounded px-1.5 py-0.5 text-[10px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60',
              sortBy === 'setup_score' ? 'bg-elevated text-secondary' : 'text-muted hover:text-secondary',
            )}
          >
            入场环境
          </button>
          <button
            type="button"
            onClick={() => setSortBy('relative_breakout')}
            title="按突破簇相对倍数排序"
            className={cn(
              'rounded px-1.5 py-0.5 text-[10px] transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/60',
              sortBy === 'relative_breakout' ? 'bg-elevated text-secondary' : 'text-muted hover:text-secondary',
            )}
          >
            突破强度
          </button>
        </span>
      </div>

      {query.isLoading && <div className="py-4 text-center text-[11px] text-muted">加载中…</div>}
      {query.isError && <div className="py-4 text-center text-[11px] text-bear">形态数据加载失败</div>}

      {!query.isLoading && !query.isError && (
        <>
          {/* 全市场基准 —— 板块相对占比的分母 */}
          {market && market.total > 0 && (
            <div className="mb-1.5 shrink-0 rounded-md border border-border/40 bg-elevated/35 px-1.5 py-1">
              <div className="mb-0.5 text-[10px] text-muted">
                全市场基准 · {market.total} 只 · 入场准备 {market.setup_count} 只
              </div>
              <div className="grid grid-cols-4 gap-1">
                {CLUSTER_ORDER.map((key) => {
                  const item = market.clusters?.[key]
                  return (
                    <div key={key} className="min-w-0">
                      <div className="truncate text-[9px] text-muted">{item?.label ?? key}</div>
                      <div className="font-mono text-[11px] text-foreground">
                        {(item?.ratio ?? 0).toFixed(1)}%
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {sectors.length === 0 ? (
            <div className="py-4 text-center text-[11px] text-muted">
              {market?.total ? '暂无满足成分数门槛的板块' : '暂无形态数据'}
            </div>
          ) : (
            <div className="min-h-0 flex-1 overflow-y-auto pr-0.5">
              <div className="mb-1 flex items-center gap-1.5 px-1.5 text-[9px] text-muted">
                <span className="min-w-0 flex-1">板块</span>
                <span className="w-12 shrink-0 text-right">成分</span>
                <span className="w-12 shrink-0 text-right">突破×</span>
                <span className="w-12 shrink-0 text-right">低位×</span>
                <span className="w-8 shrink-0 text-right">候选</span>
              </div>
              <div className="space-y-1">
                {sectors.map((row) => (
                  <SectorRow
                    key={row.key}
                    row={row}
                    expanded={expanded === row.key}
                    onToggle={() => setExpanded(expanded === row.key ? null : row.key)}
                    onStockClick={onStockClick}
                  />
                ))}
              </div>
              <div className="mt-1.5 px-1.5 text-[9px] leading-relaxed text-muted">
                × = 该板块占比 ÷ 全市场占比。突破/追涨为回避信号 (实证负超额), 低位回调上涨为正向候选。
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
