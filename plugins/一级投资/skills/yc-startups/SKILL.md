---
name: yc-startups
description: YC创业公司追踪 - 一级市场创意来源，重点关注金融/投资相关领域
argument-hint: [batch|fintech|all]
allowed-tools: WebFetch
model: sonnet
user-invocable: true
---

# YC Startups Tracker - YC创业公司追踪

**模式判断：**
- `batch` → 指定批次追踪 (如 W2026, S2025)
- `fintech` → 仅金融科技相关公司
- `all` → 全部公司概览（默认）

---

## 核心定位

为VC从业者提供YC创业公司动态，作为一级市场投资创意的重要来源：
- **Demo Day追踪**：每年2次Demo Day（冬季3月/夏季8月）
- **金融重点**：Fintech、支付、保险、加密货币、DeFi、财富管理
- **趋势信号**：从YC选择的方向识别市场共识

---

## 数据源

### 一、官方来源

| 来源 | URL | 用途 |
|------|-----|------|
| YC官网 | `https://www.ycombinator.com/companies` | 公司目录 |
| YC博客 | `https://www.ycombinator.com/blog` | 官方公告 |
| Hacker News | `https://news.ycombinator.com/` | 社区讨论 |

### 二、媒体来源

| 来源 | RSS/URL | 覆盖范围 |
|------|---------|----------|
| TechCrunch | `https://techcrunch.com/tag/y-combinator/` | Demo Day报道 |
| Crunchbase | `https://news.crunchbase.com/tag/y-combinator/` | 融资数据 |

---

## 金融相关领域分类

### 重点追踪赛道

**1. Fintech Infrastructure**
- 支付基础设施
- 银行即服务 (BaaS)
- 嵌入式金融

**2. Insurance Tech**
- AI保险经纪
- 保险科技平台
- 再保险科技

**3. Crypto/DeFi**
- 稳定币应用
- 加密支付
- DeFi协议
- 合规/监管科技

**4. Wealth Management**
- 智能投顾
- 另类资产平台
- 财富规划工具

**5. Lending/Credit**
- BNPL
- 信贷科技
- 供应链金融

**6. B2B Finance**
- 企业支出管理
- AP/AR自动化
- 财务运营平台

---

## 执行流程

### Step 1: 数据收集

**按顺序执行以下WebFetch调用：**

```
1. WebFetch: https://techcrunch.com/tag/y-combinator/
   Prompt: 提取最近的YC公司报道，包括公司名、融资金额、领域

2. WebFetch: https://www.ycombinator.com/blog
   Prompt: 提取最新的YC公告、Demo Day信息

3. WebFetch: https://news.crunchbase.com/tag/y-combinator/
   Prompt: 提取YC公司融资数据
```

### Step 2: 筛选金融相关

**关键词筛选：**
- fintech, finance, payment, banking, insurance
- crypto, defi, stablecoin, blockchain
- lending, credit, wealth, investment
- b2b payments, treasury, accounting

### Step 3: 分析输出

---

## 输出格式

### 批次追踪格式

```
YC [批次] 金融科技公司追踪

生成时间: [YYYY-MM-DD]

————————————————

[批次概览]

总公司数: [X]家
金融相关: [X]家
Demo Day: [日期]

————————————————

[金融科技公司]

1. [公司名]
   领域: [Fintech/Insurance/Crypto等]
   描述: [一句话描述]
   融资: [金额/轮次] (如有)
   创始人: [背景] (如有)
   网站: [URL]

2. [公司名]
   ...

————————————————

[赛道分布]

Fintech基础设施: [X]家
保险科技: [X]家
Crypto/DeFi: [X]家
财富管理: [X]家
借贷/信贷: [X]家
B2B金融: [X]家

————————————————

[投资信号]

YC在以下方向押注:
1. [方向1]: [判断依据]
2. [方向2]: [判断依据]

值得关注的趋势:
[从YC选择中提炼的市场信号]

————————————————

数据来源: YC官网, TechCrunch, Crunchbase
```

---

## YC批次时间线

| 批次 | Demo Day | 申请截止 |
|------|----------|----------|
| W2026 | 2026年3月 | 2025年10月 |
| S2025 | 2025年8月 | 2025年4月 |
| W2025 | 2025年3月 | 2024年10月 |

---

## 历史金融科技YC公司参考

### 知名YC Fintech毕业生

| 公司 | 批次 | 领域 | 估值/状态 |
|------|------|------|----------|
| Stripe | S2009 | 支付 | $50B+ |
| Coinbase | S2012 | 加密交易 | 上市 |
| Brex | W2017 | 企业信用卡 | $12B |
| Plaid | W2013 | 金融数据 | 被Visa收购 |
| Gusto | W2012 | 薪资/HR | $10B+ |
| Deel | W2019 | 全球薪资 | $12B |
| Mercury | S2019 | 创业银行 | $1.6B |
| Ramp | S2019 | 企业支出 | $8B |
| Zip | S2018 | 采购 | $2.1B (IPO) |

### YC Fintech成功模式

1. **API优先**: Stripe, Plaid - 开发者友好的金融API
2. **垂直聚焦**: Brex(创业公司), Mercury(初创银行)
3. **全球化**: Deel(全球薪资), Wise(跨境)
4. **合规自动化**: 减少人工审核，提升效率

---

## 投资视角

### YC选择的信号意义

YC的选择反映了:
1. **技术可行性**: 证明技术方案已成熟
2. **市场时机**: 时机合适，需求明确
3. **团队背书**: 创始人经过筛选
4. **生态资源**: 获得YC网络支持

### 关注维度

| 维度 | 问题 |
|------|------|
| 赛道集中度 | 哪个金融子领域公司最多？ |
| 地域分布 | 美国本土 vs 全球化 |
| 技术栈 | AI原生 vs 传统SaaS |
| 商业模式 | 交易费 vs SaaS订阅 |
| 监管风险 | 牌照要求、合规复杂度 |

---

## 与ai-funding-tracker联动

**建议使用流程：**

1. 使用 `yc-startups` 追踪YC批次公司
2. 使用 `ai-funding-tracker` 追踪后续融资动态
3. 两者结合形成完整一级市场视图

**联动示例：**
- YC Demo Day后 → 用本skill记录金融公司
- 3-6个月后 → 用ai-funding-tracker追踪其A轮融资
- 形成从孵化到成长的完整追踪

---

## 质量标准

### 必须包含
- 公司名称和一句话描述
- 金融子领域分类
- Demo Day批次
- 创始人背景（如有）

### 禁止
- 编造公司信息
- 虚构融资数据
- 过于笼统的分析

### 价值导向
- 发现早期投资机会
- 识别YC押注的方向
- 为VC提供deal flow来源
