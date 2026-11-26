import React, { useEffect, useMemo, useState } from 'react';
import { CandlestickChart, Loader2, RefreshCw, Search, ShieldCheck, Target } from 'lucide-react';
import { Input, Spin, message } from 'antd';
import dayjs from 'dayjs';
import { SERVICE_ENDPOINTS } from '../../../config/services';

const MARKET_ANALYSIS_API = `${SERVICE_ENDPOINTS.USER_SERVICE}/market-analysis`;

export interface PatternHit {
  key: string;
  name: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  category: 'standard' | 'entry';
  strength: number;
  description: string;
}

export interface SymbolPatternResult {
  symbol: string;
  signal_date: string | null;
  earliest_entry_date: string | null;
  hits: PatternHit[];
  bullish: string[];
  bearish: string[];
  neutral: string[];
  entry_signals: string[];
}

export interface PatternAggregate {
  key: string;
  name: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  category: 'standard' | 'entry';
  count: number;
  symbols: string[];
}

export interface PatternCatalogItem {
  key: string;
  name: string;
  direction: 'bullish' | 'bearish' | 'neutral';
  category: 'standard' | 'entry';
  source: string;
  description: string;
}

interface PatternScanResponse {
  trade_date: string | null;
  sector: string | null;
  category: string;
  scanned: number;
  hit_count: number;
  results: SymbolPatternResult[];
  aggregate: PatternAggregate[];
  available: boolean;
}

const DIRECTION_STYLE: Record<PatternHit['direction'], string> = {
  bullish: 'bg-rose-50 text-rose-700 border-rose-200',
  bearish: 'bg-emerald-50 text-emerald-700 border-emerald-200',
  neutral: 'bg-slate-100 text-slate-600 border-slate-200',
};

const DIRECTION_LABEL: Record<PatternHit['direction'], string> = {
  bullish: '看涨',
  bearish: '看跌',
  neutral: '中性',
};

/** 形态扫描面板：按板块/个股筛选 K 线形态命中情况。

时点口径：形态在 T 日收盘后识别，最早 T+1 交易日成交（earliest_entry_date）。
*/
export const PatternScanPanel: React.FC<{ date?: string }> = ({ date }) => {
  const [loading, setLoading] = useState(false);
  const [sector, setSector] = useState('');
  const [category, setCategory] = useState<'shenwan' | 'concept'>('shenwan');
  const [symbolsInput, setSymbolsInput] = useState('');
  const [data, setData] = useState<PatternScanResponse | null>(null);
  const [catalog, setCatalog] = useState<PatternCatalogItem[]>([]);
  const [keyword, setKeyword] = useState('');

  // 形态目录（下拉筛选用）
  useEffect(() => {
    let cancelled = false;
    const token = localStorage.getItem('access_token') || '';
    fetch(`${MARKET_ANALYSIS_API}/patterns/catalog`, {
      headers: token ? { Authorization: `Bearer ${token}` } : {},
    })
      .then((res) => (res.ok ? res.json() : null))
      .then((d) => {
        if (!cancelled && d && Array.isArray(d.items)) setCatalog(d.items);
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  const nameByKey = useMemo(() => {
    const map = new Map<string, string>();
    catalog.forEach((c) => map.set(c.key, c.name));
    return map;
  }, [catalog]);

  const runScan = async () => {
    setLoading(true);
    const token = localStorage.getItem('access_token') || '';
    const params = new URLSearchParams();
    if (sector.trim()) params.set('sector', sector.trim());
    if (category) params.set('category', category);
    // 后端 date 口径为 YYYYMMDD
    if (date) params.set('date', date.replace(/-/g, ''));
    const symbolList = symbolsInput
      .split(/[,，\s]+/)
      .map((s) => s.trim())
      .filter(Boolean);
    symbolList.forEach((s) => params.append('symbols', s));

    try {
      const res = await fetch(`${MARKET_ANALYSIS_API}/patterns?${params.toString()}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
      });
      if (!res.ok) throw new Error(`请求失败: ${res.status}`);
      const json = await res.json();
      setData(json);
      if (!json?.available) {
        message.warning('形态数据不可用：后端未同步 K 线数据');
      }
    } catch (e) {
      console.error('形态扫描失败:', e);
      message.error('形态扫描失败，请检查后端服务');
    } finally {
      setLoading(false);
    }
  };

  // 板块名 → 名称映射（用 aggregate 里的 name 反查）
  const filteredResults = useMemo(() => {
    if (!data?.results) return [];
    const kw = keyword.trim().toLowerCase();
    if (!kw) return data.results;
    return data.results.filter((r) => {
      if (r.symbol.toLowerCase().includes(kw)) return true;
      return r.hits.some((h) => h.name.toLowerCase().includes(kw) || h.key.includes(kw));
    });
  }, [data, keyword]);

  const entryOnly = useMemo(
    () => filteredResults.filter((r) => r.entry_signals.length > 0),
    [filteredResults],
  );

  return (
    <div className="flex flex-col gap-3.5">
      {/* 🛠️ 控制条 */}
      <div className="bg-white/95 backdrop-blur-md rounded-2xl p-3.5 border border-purple-100/80 shadow-xs flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-extrabold text-slate-700 flex items-center gap-1">
            <Target className="w-3.5 h-3.5 text-purple-600" />
            <span>板块:</span>
          </span>
          <Input
            placeholder="如 半导体 / 银行（留空扫全市场）"
            value={sector}
            onChange={(e) => setSector(e.target.value)}
            className="!w-44 text-xs"
            size="small"
            allowClear
          />
        </div>

        <div className="flex items-center bg-slate-100/90 border border-slate-200/60 p-1 rounded-full gap-1 shadow-2xs">
          {([['shenwan', '通达信二级'], ['concept', '热门概念']] as const).map(([id, label]) => (
            <button
              key={id}
              onClick={() => setCategory(id)}
              className={`px-3.5 py-1 rounded-full text-xs font-extrabold transition-all duration-200 ${
                category === id
                  ? 'bg-purple-600 text-white shadow-md shadow-purple-600/20'
                  : 'text-slate-600 hover:text-slate-900 hover:bg-slate-200/60'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <div className="flex items-center gap-2">
          <span className="text-xs font-extrabold text-slate-700 flex items-center gap-1">
            <Search className="w-3.5 h-3.5 text-purple-600" />
            <span>个股:</span>
          </span>
          <Input
            placeholder="代码，逗号分隔（优先于板块）"
            value={symbolsInput}
            onChange={(e) => setSymbolsInput(e.target.value)}
            className="!w-56 text-xs"
            size="small"
            allowClear
          />
        </div>

        <button
          onClick={runScan}
          disabled={loading}
          className={`flex items-center gap-1.5 px-4 py-1.5 rounded-full text-xs font-extrabold shadow-md transition-all duration-200 cursor-pointer ${
            loading
              ? 'bg-purple-400 text-white cursor-wait opacity-80'
              : 'bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-700 hover:from-purple-500 text-white'
          }`}
        >
          {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
          <span>{loading ? '扫描中…' : '开始扫描'}</span>
        </button>

        <div className="flex items-center gap-1.5 text-[11px] text-slate-400 font-mono ml-auto">
          <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
          <span>T 日收盘识别 → 最早 T+1 交易日成交</span>
        </div>
      </div>

      {loading && (
        <div className="bg-white/90 rounded-2xl p-10 border border-purple-100/80 flex items-center justify-center gap-2 text-sm text-slate-500">
          <Spin size="small" />
          <span>正在扫描 K 线形态…</span>
        </div>
      )}

      {!loading && data && (
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-3.5 items-start">
          {/* 📊 形态命中排行 */}
          <div className="xl:col-span-4 bg-white/95 backdrop-blur-md rounded-2xl p-4 border border-purple-100/80 shadow-xs flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <h3 className="text-xs font-extrabold text-slate-800 flex items-center gap-1.5">
                <CandlestickChart className="w-3.5 h-3.5 text-purple-600" />
                <span>形态命中排行</span>
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">
                扫描 {data.scanned} · 命中 {data.hit_count}
              </span>
            </div>

            {data.aggregate.length === 0 ? (
              <div className="text-xs text-slate-400 py-8 text-center">暂无形态命中数据</div>
            ) : (
              <div className="flex flex-col gap-1.5 max-h-[520px] overflow-y-auto custom-scrollbar">
                {data.aggregate.map((a) => {
                  const max = data.aggregate[0]?.count || 1;
                  const pct = Math.max(4, (a.count / max) * 100);
                  const dir = a.direction as PatternHit['direction'];
                  const barColor = dir === 'bullish' ? '#e11d48' : dir === 'bearish' ? '#059669' : '#94a3b8';
                  return (
                    <div key={a.key} className="flex items-center gap-2">
                      <span className="w-28 text-[11px] font-bold text-slate-700 truncate flex-shrink-0" title={a.name}>
                        {a.name}
                      </span>
                      <div className="flex-1 h-4 bg-slate-100 rounded-full overflow-hidden">
                        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, backgroundColor: barColor }} />
                      </div>
                      <span className="w-9 text-right text-[11px] font-mono font-extrabold text-slate-600 flex-shrink-0">
                        {a.count}
                      </span>
                      {a.category === 'entry' && (
                        <span className="text-[10px] font-bold text-purple-600 bg-purple-50 px-1.5 py-0.5 rounded border border-purple-200 flex-shrink-0">
                          入场
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* 📋 个股命中明细 */}
          <div className="xl:col-span-8 bg-white/95 backdrop-blur-md rounded-2xl p-4 border border-purple-100/80 shadow-xs flex flex-col gap-3">
            <div className="flex items-center justify-between border-b border-slate-100 pb-2">
              <h3 className="text-xs font-extrabold text-slate-800 flex items-center gap-1.5">
                <Target className="w-3.5 h-3.5 text-purple-600" />
                <span>个股形态明细</span>
                <span className="text-[11px] font-normal text-slate-400 font-mono">
                  入场信号 {entryOnly.length} 只
                </span>
              </h3>
              <Input
                placeholder="筛选代码或形态名..."
                prefix={<Search className="w-3 h-3 text-purple-400" />}
                value={keyword}
                onChange={(e) => setKeyword(e.target.value)}
                className="!w-48 text-xs"
                size="small"
                allowClear
              />
            </div>

            {filteredResults.length === 0 ? (
              <div className="text-xs text-slate-400 py-8 text-center">无匹配结果</div>
            ) : (
              <div className="flex flex-col gap-2 max-h-[520px] overflow-y-auto custom-scrollbar">
                {filteredResults.map((r) => (
                  <div
                    key={r.symbol}
                    className="border border-slate-100 rounded-xl p-2.5 hover:border-purple-200 hover:bg-purple-50/30 transition-colors"
                  >
                    <div className="flex items-center justify-between gap-2 mb-1.5">
                      <div className="flex items-center gap-2 min-w-0">
                        <span className="text-xs font-extrabold font-mono text-slate-800">
                          {r.symbol}
                        </span>
                        {r.signal_date && (
                          <span className="text-[10px] font-mono text-slate-400">
                            T={r.signal_date}
                          </span>
                        )}
                        {r.earliest_entry_date && (
                          <span className="text-[10px] font-mono text-emerald-600 bg-emerald-50 px-1.5 py-0.5 rounded border border-emerald-200">
                            最早 T+1={r.earliest_entry_date}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-wrap gap-1.5">
                      {r.hits.map((h) => (
                        <span
                          key={h.key}
                          title={h.description}
                          className={`text-[11px] font-bold px-2 py-0.5 rounded-md border ${DIRECTION_STYLE[h.direction]}`}
                        >
                          {h.name}
                          {h.category === 'entry' && <span className="ml-1 opacity-70">★</span>}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
