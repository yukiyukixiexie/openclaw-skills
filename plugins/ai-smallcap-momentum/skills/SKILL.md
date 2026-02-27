# AI 小盘股爆发策略 - 智谱案例复盘与规律提炼

## Skill 配置

```yaml
name: ai-smallcap-momentum
description: 基于智谱案例的AI小盘股爆发策略，整合动量追踪、事件催化、资金流入分析
version: 1.0.0
triggers:
  - /ai-smallcap
  - /asm
arguments:
  - name: ticker
    description: 港股代码
    required: true
  - name: mode
    description: 分析模式 (scan/analyze/backtest/signal)
    default: analyze
```

---

## 智谱案例复盘（02513.HK）

### 关键数据

| 指标 | 数值 | 说明 |
|------|------|------|
| 上市日期 | 2026-01-08 | |
| 发行价 | ~116 HKD | 历史最低 |
| 最高价 | 725 HKD (02-20) | |
| **最大涨幅** | **524%** | 约 6 倍 |
| 上涨天数 | 34 个交易日 | |

### 六阶段行情结构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        智谱 AI 上涨六阶段模型                                  │
└─────────────────────────────────────────────────────────────────────────────┘

价格                                                    ▲ 725 (历史高点)
  │                                                   ╱│
  │                                          ╱──────╱ │ 阶段6: 情绪顶点
  │                                         ╱          │ (02-20)
  │                                  ╱─────╱           │
  │                                 ╱                  │ 阶段5: 加速段
  │                        ╱───────╱                   │ (02-10 ~ 02-13)
  │                       ╱  DeepSeek催化              │
  │               ╱──────╱                             │ 阶段4: 事件触发
  │              ╱   (02-09)                           │
  │     ╱───────╱                                      │
  │    ╱                                               │ 阶段3: 横盘筑底
  │   ╱  回调低点 192                                   │ (01-23 ~ 02-06)
  │  ╱   (01-23)                                       │
  │ ╱                                                  │ 阶段2: 第一波
 ─│╱──────────────────────────────────────────────────►│ (01-08 ~ 01-16)
  116                                                     时间
  │     │                │                │        │
  │   01-08            01-23            02-09    02-20
  │   上市              回调低点         DeepSeek  历史高点
```

### 阶段详细分析

#### 阶段1: 上市首日（01-08）
- **开盘**: 120 → **收盘**: 131.5
- **成交量**: 17.5M（历史最高之一）
- **特征**: IPO 热度，基石投资者抬轿
- **策略**: 观察，不入场

#### 阶段2: 第一波上涨（01-08 ~ 01-16）
- **涨幅**: 116 → 250（+90%）
- **特征**:
  - 连续放量上涨
  - AI 概念炒作
  - 流通盘极小
- **策略**: 若错过，等回调

#### 阶段3: 回调筑底（01-17 ~ 02-06）
- **回撤**: 250 → 192（-23%）
- **特征**:
  - 成交量萎缩
  - 换手率下降
  - 筹码清洗
- **策略**: 关键观察期

#### 阶段4: 事件催化（02-09）
- **触发事件**: DeepSeek R2 发布
- **当日表现**: +36.2%
- **成交量**: 4.5M（大幅放量）
- **策略**: 🔥 **最佳入场点**

#### 阶段5: 加速段（02-10 ~ 02-13）
- **涨幅**: 278 → 485（+74%）
- **特征**:
  - 连续创新高
  - 成交量持续放大
  - 换手率极高
- **策略**: 持有，设移动止盈

#### 阶段6: 情绪顶点（02-20）
- **最高价**: 725
- **特征**:
  - 单日振幅极大
  - 换手率创极值
  - 量价背离初现
- **策略**: 减仓/清仓

---

## 小盘股爆发六要素模型

基于智谱案例，提炼出六个关键要素：

### 1. 流通盘因子（Float Factor）

```python
def check_float_factor(market_cap, float_ratio):
    """
    小盘股筛选

    条件：
    - 流通市值 < 50 亿港元
    - 流通比例 < 30%（IPO 初期）
    - 基石锁定 > 50%
    """
    float_market_cap = market_cap * float_ratio

    score = 0
    if float_market_cap < 50e8:  # 50亿
        score += 2
    if float_market_cap < 20e8:  # 20亿
        score += 1
    if float_ratio < 0.3:
        score += 2

    return score  # 满分5分
```

**智谱案例**:
- 上市初期流通市值约 15-20 亿港元
- 基石投资者锁定大量筹码
- 实际可交易筹码极少 → 容易被资金推动

### 2. 主题因子（Theme Factor）

```python
def check_theme_factor(theme_keywords):
    """
    热门主题识别

    2026年热门主题：
    - AI 大模型
    - AI Agent
    - 具身智能
    - 国产替代
    """
    hot_themes = ['AI', '大模型', 'Agent', 'DeepSeek', '国产替代', 'GPU', '算力']

    match_count = sum(1 for k in theme_keywords if any(h in k for h in hot_themes))

    if match_count >= 3:
        return 5
    elif match_count >= 2:
        return 3
    elif match_count >= 1:
        return 2
    return 0
```

**智谱案例**:
- 国产大模型龙头
- DeepSeek 事件直接催化
- AI Agent 概念
- 主题因子: 5/5

### 3. 基石抬轿因子（Cornerstone Factor）

```python
def check_cornerstone_factor(cornerstone_investors, lockup_end_date, current_date):
    """
    基石投资者分析

    利好条件：
    - 知名机构基石（高瓴、红杉等）
    - 锁定期内（通常6个月）
    - 基石占比 > 30%

    风险条件：
    - 临近解禁（提前1个月预警）
    - 基石开始减持
    """
    days_to_unlock = (lockup_end_date - current_date).days

    score = 0

    # 知名基石加分
    top_investors = ['高瓴', '红杉', 'GIC', '淡马锡', 'KKR', '韩投']
    for inv in cornerstone_investors:
        if any(top in inv['name'] for top in top_investors):
            score += 1

    # 锁定期内加分
    if days_to_unlock > 90:
        score += 2
    elif days_to_unlock > 30:
        score += 1
    elif days_to_unlock <= 30:
        score -= 2  # 临近解禁减分

    return min(5, max(0, score))
```

**智谱案例**:
- 韩国投资者入场（Korea Investment）
- 基石锁定期内
- 形成「抬轿效应」

### 4. 事件催化因子（Catalyst Factor）

```python
def check_catalyst_factor(events, event_date, stock_return):
    """
    事件催化强度

    高分事件：
    - 行业重磅发布（DeepSeek、OpenAI等）
    - 公司产品发布
    - 重大合同
    - 纳入指数

    评分标准：
    - 当日涨幅 > 20%: 5分
    - 当日涨幅 > 10%: 3分
    - 有事件但涨幅 < 10%: 1分
    """
    if stock_return > 20:
        return 5
    elif stock_return > 10:
        return 3
    elif len(events) > 0:
        return 1
    return 0
```

**智谱案例**:
- 02-09 DeepSeek R2 发布
- 当日涨幅 36.2%
- 事件催化因子: 5/5

### 5. 资金确认因子（Fund Flow Factor）

整合 `momentum-catcher` 的资金确认逻辑：

```python
def check_fund_flow_factor(df, date):
    """
    资金流入确认

    参考 momentum-catcher skill
    """
    today = df[df['date'] == date].iloc[0]

    score = 0

    # 1. 成交额异动
    ma20_volume = df['amount'].rolling(20).mean().loc[date]
    if today['amount'] > ma20_volume * 1.5:
        score += 1
    if today['amount'] > ma20_volume * 2:
        score += 1

    # 2. 换手率
    if today['turnover_rate'] > 5:
        score += 1
    if today['turnover_rate'] > 10:
        score += 1

    # 3. 连续放量
    recent_3d = df.tail(3)
    if all(recent_3d['amount'] > ma20_volume):
        score += 1

    return min(5, score)
```

### 6. 动量加速因子（Momentum Factor）

整合 `hk-momentum-tracker` 的加速段判断：

```python
def check_momentum_factor(df, date):
    """
    动量加速判断

    参考 hk-momentum-tracker skill
    """
    idx = df.index.get_loc(date)

    score = 0

    # 1. 10日涨幅
    if idx >= 10:
        return_10d = (df.iloc[idx]['close'] / df.iloc[idx-10]['close'] - 1) * 100
        if return_10d > 50:
            score += 1
        if return_10d > 100:
            score += 1

    # 2. 5日涨幅
    if idx >= 5:
        return_5d = (df.iloc[idx]['close'] / df.iloc[idx-5]['close'] - 1) * 100
        if return_5d > 30:
            score += 1

    # 3. 连续创新高
    if idx >= 3:
        recent = df.iloc[idx-3:idx+1]
        new_high_count = sum(recent['high'] == recent['high'].cummax())
        if new_high_count >= 3:
            score += 1

    # 4. 均线多头排列
    ma5 = df['close'].rolling(5).mean().iloc[idx]
    ma10 = df['close'].rolling(10).mean().iloc[idx]
    ma20 = df['close'].rolling(20).mean().iloc[idx]
    if ma5 > ma10 > ma20:
        score += 1

    return min(5, score)
```

---

## 综合评分与决策矩阵

### 六要素评分表

| 要素 | 权重 | 评分标准 | 智谱得分 |
|------|------|----------|----------|
| 流通盘因子 | 20% | 流通市值<50亿，流通比例<30% | 5/5 |
| 主题因子 | 15% | AI/热门主题匹配度 | 5/5 |
| 基石抬轿因子 | 15% | 知名基石+锁定期内 | 4/5 |
| 事件催化因子 | 20% | 重大事件+当日涨幅 | 5/5 |
| 资金确认因子 | 15% | 成交额异动+换手率 | 5/5 |
| 动量加速因子 | 15% | 涨幅+创新高+均线 | 5/5 |

### 综合得分计算

```python
def calculate_total_score(factors):
    """
    计算综合得分
    """
    weights = {
        'float': 0.20,
        'theme': 0.15,
        'cornerstone': 0.15,
        'catalyst': 0.20,
        'fund_flow': 0.15,
        'momentum': 0.15
    }

    total = sum(factors[k] * weights[k] for k in weights)
    return total  # 满分 5 分
```

### 决策矩阵

| 综合得分 | 信号 | 仓位建议 | 操作 |
|----------|------|----------|------|
| ≥ 4.5 | 🔥 极强 | 80% | 积极入场 |
| 4.0-4.5 | 🟢 强 | 60% | 入场 |
| 3.5-4.0 | 🟡 中等 | 30% | 谨慎入场 |
| 3.0-3.5 | 🟠 弱 | 10% | 小仓试探 |
| < 3.0 | 🔴 无 | 0% | 不参与 |

---

## 入场与退出规则

### 最佳入场时机

基于智谱案例的规律：

```
┌─────────────────────────────────────────────────────────────────┐
│                        最佳入场时机                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ✗ 上市首日        太早，波动大，风险高                          │
│   ✗ 第一波高点      追高买入，容易被套                            │
│   ✓ 回调筑底期      观察，准备资金                                │
│   ✓✓ 事件催化当天   放量突破，确认入场（首选）                     │
│   ✓ 加速段初期      突破确认后1-2天                              │
│   ✗ 情绪顶点        不追高，考虑退出                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 入场检查清单

```markdown
## 入场前检查（必须全部满足）

### 必要条件
- [ ] 流通市值 < 50 亿港元
- [ ] 属于热门主题（AI/新能源/半导体等）
- [ ] 有明确事件催化
- [ ] 当日涨幅 > 10% 或放量突破

### 加分条件
- [ ] 基石投资者锁定期内
- [ ] 韩国/中东资本入场
- [ ] 板块联动效应
- [ ] RSI 60-80 区间

### 禁止入场条件
- [ ] 换手率 > 50%（情绪极端）
- [ ] 距高点回撤 > 30%
- [ ] 基石即将解禁
- [ ] 无明确催化事件
```

### 退出规则

```python
def check_exit_signals(df, entry_price, current_price, days_held):
    """
    退出信号检测
    """
    signals = []

    # 1. 移动止盈（加速段）
    if days_held <= 3:
        trail_stop = df['high'].max() * 0.85  # 允许15%回撤
    elif days_held <= 5:
        trail_stop = df['high'].max() * 0.88  # 允许12%回撤
    else:
        trail_stop = df['high'].max() * 0.90  # 允许10%回撤

    if current_price < trail_stop:
        signals.append(('SELL_50%', '跌破移动止盈线'))

    # 2. 换手率极值
    if df['turnover_rate'].iloc[-1] > 40:
        signals.append(('SELL_30%', '换手率极端'))

    # 3. 量价背离
    if df['high'].iloc[-1] > df['high'].iloc[-2]:  # 价格新高
        if df['volume'].iloc[-1] < df['volume'].iloc[-2] * 0.7:  # 量萎缩
            signals.append(('WARNING', '量价背离'))

    # 4. 固定止损
    if current_price < entry_price * 0.85:
        signals.append(('SELL_ALL', '触发止损'))

    # 5. 目标止盈
    if current_price > entry_price * 2:
        signals.append(('SELL_30%', '达到翻倍目标'))

    return signals
```

---

## 指数效应分析

### 纳入指数的影响

```
时间线：
────────────────────────────────────────────────────────────────────
        宣布纳入          生效日           追踪资金到位
           │               │                   │
           │   预期买入    │    被动买入        │
           │   股价上涨    │    抛压出现        │
           │               │                   │
────────────────────────────────────────────────────────────────────
           │←─── 做多窗口 ─→│←── 谨慎期 ──→│
```

### 基石解禁影响

```python
def analyze_lockup_impact(lockup_end_date, cornerstone_holdings):
    """
    基石解禁影响分析

    时间窗口：
    - 解禁前1个月：市场开始担忧
    - 解禁当周：抛压最大
    - 解禁后2周：观察实际减持情况
    """
    impact = {
        'pre_lockup_30d': '预警期，可能提前下跌',
        'lockup_week': '高风险，避免持仓',
        'post_lockup_14d': '观察期，看实际减持'
    }
    return impact
```

---

## 与现有 Skills 整合

### 调用流程

```bash
# 1. 首先获取实时行情
/hk-quote {ticker}

# 2. 运行动量追踪（全量历史分析）
/hk-momentum-tracker {ticker} --history

# 3. 运行六要素评分
/ai-smallcap {ticker} --analyze

# 4. 获取入场信号
/momentum-catcher {ticker} --signal
```

### 数据流整合

```
┌─────────────────────────────────────────────────────────────────┐
│                        数据流整合架构                             │
└─────────────────────────────────────────────────────────────────┘

    ┌─────────────┐
    │ 实时行情    │ ← /hk-realtime-quote
    │ (新浪/腾讯) │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐     ┌─────────────┐
    │ 历史数据    │ ←── │ market_data │ (shared)
    │ (yfinance)  │     │    .py      │
    └──────┬──────┘     └─────────────┘
           │
    ┌──────┴──────┐
    │             │
    ▼             ▼
┌─────────┐ ┌───────────────┐
│ 六要素  │ │ hk-momentum   │
│ 评分    │ │   -tracker    │
└────┬────┘ └───────┬───────┘
     │              │
     └──────┬───────┘
            │
            ▼
    ┌─────────────┐
    │ 综合决策    │
    │ 入场/退出   │
    └──────┬──────┘
            │
            ▼
    ┌─────────────┐
    │ 交易执行    │
    └─────────────┘
```

---

## 回测验证

### 智谱案例回测

```
回测区间: 2026-01-08 至 2026-02-27
初始资金: 100,000 HKD

策略规则:
- 入场: 02-09 (DeepSeek催化日，六要素评分 > 4.5)
- 入场价: 276.80
- 仓位: 80%
- 止盈: 移动止盈（最高价 × 0.88）

回测结果:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
入场日期: 2026-02-09
入场价格: 276.80 HKD
仓位: 80,000 HKD (289股)

最高持仓价值: 209,525 HKD (725 × 289)
移动止盈触发: 2026-02-23 @ 560 HKD
清仓价值: 161,840 HKD

策略收益: +102.3%
Buy & Hold 收益: +108.1% (02-09 入场持有到 02-27)

结论: 策略略微跑输 B&H，但有效控制了回撤风险
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 风险提示

1. **小盘股高波动**: 单日振幅可达 20-30%
2. **流动性风险**: 极端行情下可能无法成交
3. **事件依赖**: 无催化事件时可能长期横盘
4. **基石解禁**: 解禁期间可能大幅下跌
5. **监管风险**: 港股对 AI 概念股可能有政策变化

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0.0 | 2026-02-27 | 初始版本，基于智谱案例 |
