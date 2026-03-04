# 港股财报分析数据源指南

本文档详细列出港股财报分析所需的各类数据源，包括官方渠道、第三方平台和搜索技巧。

---

## 一、官方数据源

### 1.1 港交所披露易 (HKEX News)

**网址**：https://www.hkexnews.hk

**功能**：
- 上市公司公告（财报、权益披露、回购记录等）
- 招股章程
- CCASS 持仓查询

**文档类型代码**：
| 类型 | 英文 | 说明 |
|------|------|------|
| 年报 | Annual Report | 年度财务报告 |
| 中期报告 | Interim Report | 半年度报告 |
| 季报 | Quarterly Report | 部分公司有 |
| 业绩公告 | Results Announcement | 财报摘要 |
| 权益披露 | Disclosure of Interests | 大股东增减持 |
| 股份回购 | Share Buy-back | 公司回购记录 |

**搜索技巧**：
```
site:hkexnews.hk "[股票代码]" "年報"
site:hkexnews.hk "[股票代码]" "業績公告"
site:hkexnews.hk "[股票代码]" "權益披露"
```

### 1.2 CCASS 持仓查询

**网址**：https://www.hkexnews.hk/sdw/search/searchsdw_c.aspx

**说明**：
- 可查询任一股票在各券商/托管行的持仓数量
- 支持历史数据查询（可追溯数年）
- 可追踪机构持仓变化

**使用方法**：
1. 输入股票代码
2. 选择日期
3. 查看各参与者持仓

**分析要点**：
- 汇丰银行：通常代表外资机构
- 中国银行：通常代表南向资金
- 花旗、摩根士丹利：外资投行持仓

---

## 二、分析师预期数据源

### 2.1 yfinance（推荐，最便捷）

**使用方法**：
```python
import yfinance as yf
ticker = yf.Ticker("0700.HK")
info = ticker.info

# 可获取数据
info["targetMeanPrice"]      # 目标价均值
info["targetHighPrice"]      # 目标价高值
info["targetLowPrice"]       # 目标价低值
info["numberOfAnalystOpinions"]  # 分析师数量
info["recommendationKey"]    # 推荐评级
```

**优点**：数据结构化、API 稳定
**缺点**：部分小市值股票覆盖不足

### 2.2 东方财富 (EastMoney)

**研报中心**：https://data.eastmoney.com/report/

**搜索方式**：
```
site:eastmoney.com "[股票代码]" "研报"
site:eastmoney.com "[公司名]" "盈利预测"
```

**可获取数据**：
- 各大券商研报
- 盈利预测（EPS、收入）
- 目标价
- 评级

### 2.3 智通财经 (Zhitong Caijing)

**网址**：https://www.zhitongcaijing.com

**特点**：
- 专注港美股
- 实时新闻
- 投行评级汇总
- 业绩会直播

**搜索方式**：
```
site:zhitongcaijing.com "[股票代码]" "目标价"
site:zhitongcaijing.com "[公司名]" "评级"
site:zhitongcaijing.com "[公司名]" "业绩会"
```

### 2.4 格隆汇 (Gelonghui)

**网址**：https://www.gelonghui.com

**特点**：
- 港股深度研究
- 业绩会实录
- 行业分析

**搜索方式**：
```
site:gelonghui.com "[公司名]" "业绩"
site:gelonghui.com "[公司名]" "分析"
```

### 2.5 AAStocks

**网址**：https://www.aastocks.com

**特点**：
- 港股行情
- 基本面数据
- 卖空数据

---

## 三、筹码数据源（港股特有）

### 3.1 南向资金（陆港通）

**数据源**：
- 东方财富陆港通：https://data.eastmoney.com/hsgt/
- 同花顺：https://data.10jqka.com.cn/hgt/

**可获取数据**：
- 每日南向资金流入/流出
- 个股南向持股数量
- 南向持股占比变化

**搜索方式**：
```
"[股票代码]" "南向资金" "持股变化"
"[公司名]" "陆股通" "持仓"
```

### 3.2 卖空数据

**数据源**：
- 港交所每日卖空统计
- AAStocks 卖空数据

**搜索方式**：
```
site:aastocks.com "[股票代码]" "卖空"
"[股票代码]" "short interest" "香港"
```

### 3.3 大股东权益披露

**官方来源**：港交所披露易

**搜索方式**：
```
site:hkexnews.hk "[股票代码]" "權益披露"
site:hkexnews.hk "[股票代码]" "增持"
site:hkexnews.hk "[股票代码]" "減持"
```

---

## 四、业绩会/分析师会议

### 4.1 获取途径

| 来源 | 说明 |
|------|------|
| 公司 IR 官网 | 最权威，通常有 webcast/文字稿 |
| 智通财经 | 业绩会直播和纪要 |
| 格隆汇 | 业绩会实录 |
| 新浪财经 | 部分业绩会直播 |

### 4.2 搜索技巧

```
"[公司名]" "业绩发布会" "纪要"
"[公司名]" "分析师会议" "问答"
"[公司名]" "investor presentation"
"[公司名]" "earnings call" "transcript"
```

### 4.3 关注要点

**Q&A 部分最重要**：
- 分析师问什么 = 市场核心关注点
- 管理层如何回答 = 对关键问题的态度
- 语气变化 = 信心指标

---

## 五、估值数据

### 5.1 实时估值

| 数据源 | 说明 |
|--------|------|
| 东方财富 | PE、PB、PS 等 |
| AAStocks | 港股估值数据 |
| yfinance | API 获取 |

### 5.2 历史估值

| 数据源 | 说明 |
|--------|------|
| 同花顺 | 历史 PE 分位 |
| 理杏仁 | 估值历史分析 |

### 5.3 同业对比

**搜索方式**：
```
"[行业名]" "港股" "估值对比"
"[公司名]" "同业" "PE" "对比"
```

---

## 六、主要投行/券商

### 外资投行
- 高盛 (Goldman Sachs)
- 摩根士丹利 (Morgan Stanley)
- 花旗 (Citi)
- 瑞银 (UBS)
- 摩根大通 (J.P. Morgan)
- 大和 (Daiwa)
- 野村 (Nomura)
- 美银美林 (BofA)

### 中资投行
- 中金 (CICC)
- 中信 (CITIC)
- 华泰 (Huatai)
- 招银国际 (CMBI)
- 交银国际 (BOCOM International)
- 国泰君安国际 (Guotai Junan International)

**搜索方式**：
```
"[股票代码]" "[投行名]" "目标价"
"[公司名]" "[投行名]" "评级"
```

---

## 七、Python 工具使用

### 7.1 分析师数据获取

```bash
# 完整分析师数据
python3 hk_analyst_data.py 00700

# 目标价
python3 hk_analyst_data.py 00700 --type targets

# 评级
python3 hk_analyst_data.py 00700 --type ratings

# EPS 预期
python3 hk_analyst_data.py 00700 --type eps
```

### 7.2 年报下载

```bash
# 搜索年报
python3 hk_annual_report.py 00700 --type annual

# 搜索中期报告
python3 hk_annual_report.py 00700 --type interim

# 下载最新报告
python3 hk_annual_report.py 00700 --download
```

### 7.3 股票代码搜索

```bash
# 按公司名搜索
python3 ../../../shared/search_ticker.py 腾讯
python3 ../../../shared/search_ticker.py 美团
python3 ../../../shared/search_ticker.py 智谱
```

---

## 八、常用搜索模板

### 财报相关
```
"[股票代码]" "业绩" "[季度/年度]"
"[公司名]" "财报" "解读"
"[股票代码]" "业绩" "超预期/不及预期"
```

### 分析师预期
```
"[股票代码]" "目标价" "[投行名]"
"[公司名]" "评级" "上调/下调"
"[股票代码]" "盈利预测" "分析师"
```

### 筹码结构
```
"[股票代码]" "南向资金" "增持/减持"
"[股票代码]" "CCASS" "持仓变化"
"[股票代码]" "大股东" "增持/减持"
```

### 业绩会
```
"[公司名]" "业绩会" "纪要"
"[公司名]" "业绩发布会" "管理层"
"[公司名]" "分析师会议" "问答"
```

---

## 九、数据质量评估

### 高质量数据
- 覆盖分析师 >= 10 人
- yfinance 数据完整
- 有多个大行研报支持

### 中等质量数据
- 覆盖分析师 5-10 人
- yfinance 数据部分缺失
- 仅有中资券商覆盖

### 低质量数据
- 覆盖分析师 < 5 人
- 依赖搜索引擎结果
- 数据可能过时

**处理原则**：
- 高质量：可直接使用
- 中等质量：需交叉验证
- 低质量：需明确标注数据局限性
