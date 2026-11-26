# Strategy Lab 用户使用手册

> 把 Python 量化代码粘进编辑器，浏览器里直接看 K 线买卖点 + 净值曲线 + 4 关卡过拟合体检 + 一键转可上线策略模板。

## 1. 进入入口

登录后侧边栏 → **策略实验室**。三栏布局：左 = 示例代码，中 = Monaco 编辑器，右 = 回测结果。

## 2. 5 分钟跑通一个策略

1. 左栏选「双均线策略」或「买入持有」。
2. 顶栏点 **运行** —— 提交后右下角显示进度条（队列 → AST 检查 → 加载脚本 → 回测中 → 完成）。
3. 完成后右侧面板会显示：
   - **策略评分卡**（基础 0-100 分，运行 4 关卡后会与过拟合得分混合）
   - **KPI 卡片**：年化、Sharpe、最大回撤、Alpha 等
   - **6 个 Tab**：净值曲线 / K 线 ▲▼ / 月度热力 / 年度统计 / 成交明细 / 末日持仓 / 日志

## 3. 编辑器约定

```python
def setup(ctx):
    ctx.universe = ["sh600036", "sh000001"]
    ctx.start = "2024-01-02"
    ctx.end   = "2024-06-30"
    ctx.cash  = 1_000_000

def on_bar(ctx, bar):
    if bar.symbol == "sh600036" and ctx.position(bar.symbol).qty == 0:
        ctx.buy(bar.symbol, weight=0.5, reason="init")
```

**沙盒规则（违反会在 AST 检查阶段被拒）**：
- 禁止 `import os / sys / subprocess / socket / requests`
- 禁止 `open / eval / exec / __import__`
- 禁止赋值给以 `__` 开头的属性

**常用 SDK API**：
| API | 用途 |
|---|---|
| `ctx.universe / start / end / cash` | 声明回测窗口 |
| `ctx.buy(symbol, weight, reason)` | 按目标权重买入 |
| `ctx.sell(symbol, all=True, reason)` | 按持仓卖出 |
| `ctx.history(symbol, n=20, field='close')` | 历史数据 |
| `ctx.indicator('sma', symbol, n=20)` | 内置指标（sma / ema / rsi / macd） |
| `ctx.plot_line(name, value, symbol)` | K 线上画水平线 |
| `ctx.plot_marker(symbol, type, text, price)` | K 线上画自定义箭头 |
| `ctx.param('name', range(...), default=...)` | 声明可调参数（4 关卡参敏检测会自动扫描） |
| `ctx.pattern_hits(symbol)` | 识别当前交易日的 K 线形态（61 种标准形态 + 9 种入场信号形态） |
| `ctx.pattern_scan(symbols)` | 批量识别形态，返回 `{symbol: 结果}` |
| `ctx.pattern_entry(symbol, weight, keys)` | 形态入场：T 日识别 → T+1 交易日收盘成交 |
| `ctx.pattern_exit(symbol, keys)` | 形态离场：T 日识别 → T+1 交易日收盘清仓 |
| `ctx.log(msg)` | 写日志（在「日志」Tab 查看） |

## 3.1 形态策略：筛选 → 回测验证

左栏示例新增「形态」分类，4 个模板可直接跑：

| 示例 | 说明 |
|---|---|
| 入场信号形态 → T+1 买入 | 命中 9 种「入场准备标准」即登记信号，下一交易日收盘成交 |
| 看涨标准形态（晨星 / 锤头） | 用 `keys` 指定关注的 CDL* 形态 |
| 板块形态筛选 → T+1 建仓 | `ctx.pattern_scan()` 逐只扫描，当日候选次日调仓 |
| 形态入场 + 看跌形态离场 | 入场信号建仓，看跌形态清仓 |

**时点口径**（与平台其他模块一致）：因子在 **T 日收盘后**算出 → **T+1 收盘成交** →
持仓吃到 bar `[T+1, T+2]`。`ctx.pattern_entry()` / `ctx.pattern_exit()` 已内置该延迟，
你不需要自己数一天再下单。

**形态目录**：市场分析页新增「板块形态扫描」Tab，可按交易日 + 板块查看命中情况，
并导出所有形态的 key（标准形态 61 种 + 入场信号 9 种），用于填 `keys` 参数。

> 沙箱规则：形态能力通过 `ctx.*` 提供，脚本里**不要** import 形态引擎，
> AST 静态检查会直接拒绝 backend 内部模块的 import。

## 4. AI 助手

右上角 **AI 助手** 抽屉 — 自动带入：
- 当前编辑器代码
- 最近一次回测的报错（如有）

可以说「加 RSI 超卖反弹」「把 universe 换成 csi500」「修复刚才的报错」。AI 会在回复里给一段 ```python ...``` 代码块，点 **应用到编辑器** 即可一键替换原代码。

## 5. 4 关卡过拟合检测

回测完成后点 **4 关卡检测**（耗时 30-90s）：
1. **训测分割**：70/30 时间切分，看 OOS sharpe 是否塌陷
2. **3 段走查**：50→70%、70→85%、85→100% 三个 OOS 窗
3. **参敏扫描**：枚举 `ctx.param()` 取值，看 sharpe 标准差
4. **蒙卡 100 次**：随机洗牌日收益率，看你的累计收益排名

总分 = `0.3·关卡1 + 0.3·关卡2 + 0.2·关卡3 + 0.2·关卡4`，会与基础启发评分按 4:6 混合后显示。

## 6. 回测结果可视化

### K 线 ▲▼
- 每笔成交在对应 K 线上画箭头：红 ▲ = 买入（位于 K 线下方），绿 ▼ = 卖出
- 点 ▲▼ 弹「Why」原因弹窗（包含调用 `ctx.buy` 时传入的 reason 字段 + 任意 detail dict）
- `ctx.plot_line()` 画的水平线、`ctx.plot_marker()` 画的圆点也会渲染

### 月度热力 / 年度统计
按月汇总收益形成红绿热力；按年统计收益、基准、超额、最大回撤、交易日数。

### 成交明细
分页表格，「Why」按钮一键打开原因弹窗。

## 7. 一键上线：转模板 + 加入扫描

- **转模板**：把 SDK 脚本翻译成 Qlib 风格策略模板，自动保存到「我的策略」。可在「策略向导」继续编辑或上线。需要回测时间窗 / cash / universe 已声明。
- **加入每日扫描**：把脚本注册到 `qm:lab:watch`，工作日 18:30 由 Celery beat 自动跑一次（`STRATEGY_LAB_SCAN_ENABLED=true`），新触发的当日交易会出现在 Dashboard 的「策略实验室扫描」卡片。

手动触发：Dashboard 信号卡上 **立即扫描** 按钮。

## 8. 错误处理

回测失败时，会先用中文摘要错误类型（沙盒拦截、缩进错误、变量未定义、行情数据缺失等），下方仍可展开原始 traceback 复制。

数据缺失类错误会附 **去数据平台同步** 按钮。

## 9. 复现性

每次回测产出 `script_sha = sha256(code + sorted(params))`。
同 `script_sha` + 同 `data_snapshot_at` 重跑结果差 < `1e-6`。
两个字段在 KPI 卡顶部 / 结果 Tab 标题栏可见。

## 10. 性能基线

| 场景 | 目标耗时 |
|---|---|
| csi300, 1 年, 同步回测 | < 30s |
| csi300, 5 年, 异步回测 | < 5 min |
| K 线渲染 1 年日 K | < 500ms |
| AI 流式响应首 token | < 2s |

超时（默认 60s）会被强制 kill，错误中提示「运行超时」。

## 11. 进一步阅读

- [Strategy_Lab规范.md](./Strategy_Lab规范.md) —— 完整 19 天 sprint 规划与产品愿景
- [QuantMind策略编写手册_v2.md](./QuantMind策略编写手册_v2.md) —— Qlib 模板原生开发风格
- 后端代码：`backend/services/engine/strategy_lab/`
- 前端代码：`electron/src/features/strategy-lab/`
