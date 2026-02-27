# 数据源清单

## 概述

本文档列出动量捕捉框架所需的所有数据源，优先使用免费方案。

---

## 推荐方案总结

| 数据类型 | 推荐方案 | 成本 | 说明 |
|----------|----------|------|------|
| **实时行情** | 新浪/腾讯财经 | 免费 | 近实时，延迟<1分钟 |
| **日线数据** | yfinance | 免费 | 延迟15-20分钟 |
| **技术指标** | 本地计算 | 免费 | 内置MACD/RSI/MA |
| **南向资金** | AKShare | 免费 | 每日更新 |
| **工具API** | QVeris MCP | 免费1000额度 | 10000+ API |

---

## 1. 新浪/腾讯财经 API（推荐，免费实时）

### 新浪财经 API

**URL格式**：`https://hq.sinajs.cn/list=rt_hk{code}`

```python
from indicators import fetch_realtime_quote

# 获取实时行情
quote = fetch_realtime_quote('02513')
print(f"现价: {quote['price']}, 涨跌: {quote['change_pct']}%")
```

**返回字段**：
| 字段 | 说明 |
|------|------|
| price | 现价 |
| open | 开盘价 |
| high | 最高价 |
| low | 最低价 |
| prev_close | 昨收价 |
| change | 涨跌额 |
| change_pct | 涨跌幅% |
| volume | 成交量 |
| amount | 成交额 |
| high_52w | 52周高 |
| low_52w | 52周低 |
| datetime | 行情时间 |

### 腾讯财经 API

**URL格式**：`https://qt.gtimg.cn/q=r_hk{code}`

```python
quote = fetch_realtime_quote('02513', source='tencent')
```

### 命令行使用

```bash
# 获取实时行情
python indicators.py --ticker 02513 --realtime

# 指定腾讯数据源
python indicators.py --ticker 02513 --realtime --realtime-source tencent
```

### 优点
- ✅ 完全免费
- ✅ 近实时（延迟<1分钟）
- ✅ 无需注册
- ✅ 支持所有港股

### 缺点
- ❌ 无分钟K线
- ❌ 无历史数据
- ❌ 可能被限流（频繁请求）

---

## 2. QVeris MCP（已配置）

**GitHub**: https://github.com/guangxiangdebizi/FinanceMCP

基于 Tushare API 的 MCP 服务，提供 43+ 金融数据接口，支持港股数据。

### 核心功能

| 工具 | 功能 | 参数示例 |
|------|------|----------|
| `stockData` | 行情+技术指标 | `code="00700.HK", market_type="hk", indicators="macd(12,26,9) rsi(14)"` |
| `moneyFlow` | 资金流向 | `ts_code="000001.SZ", start_date="20260101"` |
| `dragonTigerInst` | 龙虎榜 | 机构买卖动向 |
| `marginTrade` | 融资融券 | 杠杆资金情绪 |
| `financeNews` | 财经新闻 | 事件扫描 |

### 安装配置

```bash
# 克隆仓库
git clone https://github.com/guangxiangdebizi/FinanceMCP.git
cd FinanceMCP

# 安装依赖
npm install

# 配置 Tushare Token（需要注册 tushare.pro）
export TUSHARE_TOKEN=your_token_here

# 启动服务
npm run start
```

### Claude Code 配置

在 `~/.claude.json` 中添加：

```json
{
  "mcpServers": {
    "finance": {
      "command": "node",
      "args": ["/path/to/FinanceMCP/dist/index.js"],
      "env": {
        "TUSHARE_TOKEN": "your_token_here"
      }
    }
  }
}
```

### 使用示例

```
# 获取港股行情和技术指标
调用 stockData 工具:
- code: 00700.HK
- market_type: hk
- start_date: 20260101
- end_date: 20260227
- indicators: macd(12,26,9) rsi(14) ma(20)

# 获取资金流向
调用 moneyFlow 工具:
- query_type: stock
- ts_code: 000001.SZ
- start_date: 20260201
- end_date: 20260227
```

### 数据字段

**stockData 返回：**
- 日期、开盘、最高、最低、收盘、成交量
- MACD (DIF, DEA, MACD柱)
- RSI (14日)
- 均线 (MA5, MA10, MA20等)

**moneyFlow 返回：**
- 主力净流入金额和比例
- 超大单、大单、中单、小单流入
- 净流入/流出统计

### 注意事项

- 需要 Tushare Pro 账户（基础功能免费）
- 港股数据可能需要更高权限
- A股数据最完整

---

## 1. 行情数据

### 方案A：Yahoo Finance（推荐，免费）

```python
import yfinance as yf

# 获取港股数据（代码需加.HK后缀）
ticker = yf.Ticker("02513.HK")

# 获取历史K线
df = ticker.history(period="6mo", interval="1d")
# 返回: Open, High, Low, Close, Volume, Dividends, Stock Splits

# 获取基本信息
info = ticker.info
# 返回: marketCap, sharesOutstanding, floatShares, etc.
```

**优点**：
- 完全免费，无需注册
- 支持港股（.HK后缀）
- 数据延迟约15-20分钟

**缺点**：
- 无盘中实时数据
- 部分港股可能数据不全
- 无南向资金数据

**安装**：
```bash
pip install yfinance
```

---

### 方案B：富途OpenAPI（需开户）

```python
from futu import OpenQuoteContext, Market, SubType, KLType

# 连接富途牛牛客户端（需先启动）
quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)

# 获取股票基本信息
ret, data = quote_ctx.get_stock_basicinfo(Market.HK, SecurityType.STOCK, ['02513'])

# 订阅实时行情
quote_ctx.subscribe(['HK.02513'], [SubType.QUOTE, SubType.K_DAY])

# 获取K线数据
ret, data, page_req_key = quote_ctx.request_history_kline(
    'HK.02513',
    start='2026-01-01',
    end='2026-02-27',
    ktype=KLType.K_DAY
)

quote_ctx.close()
```

**优点**：
- 实时行情
- 数据准确完整
- 支持Level 2行情（付费）

**缺点**：
- 需要富途证券账户
- 需要运行富途牛牛客户端
- 免费额度有限

**配置步骤**：
1. 下载安装富途牛牛客户端
2. 登录账户
3. 启用OpenAPI：设置 → 接口设置 → 启用OpenD
4. 安装SDK：`pip install futu-api`

---

### 方案C：AKShare（免费，国内源）

```python
import akshare as ak

# 获取港股历史行情
df = ak.stock_hk_daily(symbol="02513", adjust="qfq")

# 获取港股实时行情
df = ak.stock_hk_spot()
```

**优点**：
- 免费
- 数据源来自东财/新浪
- 支持复权数据

**缺点**：
- 可能需要科学上网
- 数据稳定性一般

**安装**：
```bash
pip install akshare
```

---

## 2. 南向资金数据

### 方案A：港交所官网爬取（推荐）

**数据URL**：
- 每日数据：`https://www.hkex.com.hk/chi/stat/smstat/smstat_d_c.htm`
- 历史数据：`https://www.hkexnews.hk/ncms/data/hsstock/southbound_data.json`

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_southbound_daily():
    """获取南向资金每日数据"""
    url = "https://www.hkex.com.hk/chi/stat/smstat/smstat_d_c.htm"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.content, 'html.parser')

    # 解析表格数据
    tables = soup.find_all('table')
    # ... 解析逻辑

    return data
```

**数据字段**：
- 沪股通净买入
- 深股通净买入
- 港股通（沪）净买入
- 港股通（深）净买入
- 每日额度使用情况

---

### 方案B：AKShare南向资金接口

```python
import akshare as ak

# 获取南向资金历史数据
df = ak.stock_hsgt_south_money_em()
# 返回: 日期, 沪市港股通净流入, 深市港股通净流入, 合计

# 获取南向资金持股明细
df = ak.stock_hsgt_hold_stock_em(market="HK", indicator="今日排行")
```

---

### 方案C：东方财富网页爬取

**数据URL**：
`https://data.eastmoney.com/hsgt/index.html`

```python
import requests
import json

def fetch_southbound_em():
    """从东方财富获取南向资金数据"""
    url = "https://push2.eastmoney.com/api/qt/kamt.rtmin/get"
    params = {
        'fields1': 'f1,f2,f3,f4',
        'fields2': 'f51,f52,f53,f54,f55,f56'
    }

    resp = requests.get(url, params=params)
    data = resp.json()

    return data
```

---

## 3. 个股持股/资金流向

### 港交所披露易（官方数据）

**数据URL**：
`https://www.hkexnews.hk/sdw/search/searchsdw_c.aspx`

用于查询：
- 港股通持股数据
- 个股持股变动
- 机构持仓披露

```python
from playwright.sync_api import sync_playwright

def fetch_stock_holding(stock_code, date):
    """获取个股港股通持股数据"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        url = f"https://www.hkexnews.hk/sdw/search/searchsdw_c.aspx"
        page.goto(url)

        # 填写查询表单
        page.fill('#txtStockCode', stock_code)
        page.fill('#txtShareholdingDate', date)
        page.click('#btnSearch')

        # 等待结果
        page.wait_for_selector('.ccass-table')

        # 提取数据
        content = page.content()
        browser.close()

        return parse_holding_data(content)
```

---

## 4. 卖空数据

### 港交所卖空数据

**每日卖空报告**：
`https://www.hkex.com.hk/chi/stat/smstat/ssturnover/ssturnover_c.htm`

**数据字段**：
- 股票代码
- 卖空股数
- 卖空金额
- 卖空占成交比例

```python
def fetch_short_selling(date):
    """获取每日卖空数据"""
    # 格式: YYYYMMDD
    url = f"https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ssturnover_{date}.htm"

    resp = requests.get(url)
    # 解析HTML表格
    df = pd.read_html(resp.content)[0]

    return df
```

---

## 5. 技术指标计算

### pandas-ta（推荐，纯Python）

```python
import pandas as pd
import pandas_ta as ta

# 加载数据
df = pd.read_csv('stock_data.csv')

# 计算MACD
df.ta.macd(fast=12, slow=26, signal=9, append=True)
# 新增列: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9

# 计算RSI
df.ta.rsi(length=14, append=True)
# 新增列: RSI_14

# 计算移动平均
df.ta.sma(length=5, append=True)
df.ta.sma(length=10, append=True)
df.ta.sma(length=20, append=True)

# 计算布林带
df.ta.bbands(length=20, std=2, append=True)
# 新增列: BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, BBB_20_2.0, BBP_20_2.0

# 计算ATR（用于止损）
df.ta.atr(length=14, append=True)
```

**安装**：
```bash
pip install pandas-ta
```

---

### TA-Lib（C库，更快）

```python
import talib
import numpy as np

# 计算MACD
macd, signal, hist = talib.MACD(df['close'].values)

# 计算RSI
rsi = talib.RSI(df['close'].values, timeperiod=14)

# 计算移动平均
ma5 = talib.SMA(df['close'].values, timeperiod=5)
ma10 = talib.SMA(df['close'].values, timeperiod=10)
ma20 = talib.SMA(df['close'].values, timeperiod=20)
```

**安装（Mac）**：
```bash
brew install ta-lib
pip install TA-Lib
```

**安装（Windows）**：
```bash
# 下载预编译wheel
pip install TA_Lib-0.4.24-cp39-cp39-win_amd64.whl
```

---

## 6. 数据获取优先级

| 数据类型 | 首选方案 | 备选方案 | 说明 |
|----------|----------|----------|------|
| 日K线 | yfinance | AKShare | 免费稳定 |
| 实时行情 | 富途API | Yahoo Finance | 富途需开户 |
| 南向资金 | AKShare | 港交所爬取 | AKShare更方便 |
| 个股持仓 | 港交所披露易 | - | 官方数据 |
| 卖空数据 | 港交所 | - | 官方数据 |
| 技术指标 | pandas-ta | TA-Lib | pandas-ta无需编译 |

---

## 7. 环境配置

### 依赖安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate  # Windows

# 安装依赖
pip install yfinance pandas pandas-ta requests beautifulsoup4 akshare

# 可选：安装playwright（用于网页爬取）
pip install playwright
playwright install chromium
```

### requirements.txt

```
yfinance>=0.2.0
pandas>=2.0.0
pandas-ta>=0.3.0
requests>=2.28.0
beautifulsoup4>=4.12.0
akshare>=1.10.0
playwright>=1.40.0
```

---

## 8. API限制说明

| 数据源 | 请求限制 | 说明 |
|--------|----------|------|
| Yahoo Finance | 2000次/小时 | IP限制，建议缓存 |
| AKShare | 无明确限制 | 建议间隔1秒 |
| 港交所网站 | 无明确限制 | 建议间隔2秒 |
| 富途API | 根据账户等级 | 免费账户有限制 |

---

## 9. 数据缓存建议

```python
import os
import pickle
from datetime import datetime, timedelta

CACHE_DIR = ".cache"

def get_cached_data(key, max_age_hours=1):
    """获取缓存数据"""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")

    if os.path.exists(cache_file):
        mtime = datetime.fromtimestamp(os.path.getmtime(cache_file))
        if datetime.now() - mtime < timedelta(hours=max_age_hours):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

    return None

def save_cached_data(key, data):
    """保存缓存数据"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")

    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
```

---

## 10. 故障处理

### 常见问题

1. **Yahoo Finance无数据**
   - 检查股票代码格式（需要.HK后缀）
   - 部分新股可能需要等待几天才有数据

2. **AKShare连接失败**
   - 可能需要科学上网
   - 尝试更换数据源

3. **港交所网页爬取失败**
   - 检查网页结构是否变化
   - 使用playwright处理JavaScript渲染页面

4. **富途API连接失败**
   - 确保富途牛牛客户端已启动
   - 检查OpenD设置是否正确
