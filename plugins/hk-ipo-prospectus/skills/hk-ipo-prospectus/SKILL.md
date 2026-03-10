---
name: hk-ipo-prospectus
description: 港交所IPO招股书自动下载工具 - 支持公司名自动查询ticker，从披露易搜索并下载招股书PDF
argument-hint: <股票代码或公司名> [--ticker] [--search] [--visible]
allowed-tools: Bash, Read
model: sonnet
user-invocable: true
---

# 港交所IPO招股书下载工具

从港交所披露易 (HKEXnews) 自动搜索并下载上市公司招股书及相关IPO文件。

**核心功能**: 支持公司名称（中文/英文）自动查询股票代码，无需记住ticker。

## 快速开始

```bash
# 用公司名称（中文，自动查询ticker）
python3 {SKILL_DIR}/fetch_prospectus.py 智谱
python3 {SKILL_DIR}/fetch_prospectus.py 美团
python3 {SKILL_DIR}/fetch_prospectus.py 小米

# 用英文名称
python3 {SKILL_DIR}/fetch_prospectus.py meituan
python3 {SKILL_DIR}/fetch_prospectus.py xiaomi

# 用股票代码
python3 {SKILL_DIR}/fetch_prospectus.py 02513
```

## Ticker查询机制

脚本会按以下优先级自动查询股票代码：

1. **Yahoo Finance API** - 用于英文名查询，数据最全
2. **Brave Search** - 用于中文名查询，通过搜索引擎提取代码
3. **DuckDuckGo** - 备用搜索引擎
4. **Playwright** - 最后备用，通过披露易autocomplete获取

## 使用方法

### 1. 自动识别模式（推荐）

脚本会自动判断输入是股票代码还是公司名称：

```bash
# 输入公司名 → 自动查ticker → 搜索下载
python3 {SKILL_DIR}/fetch_prospectus.py 智谱

# 输入股票代码 → 直接搜索下载
python3 {SKILL_DIR}/fetch_prospectus.py 02513
```

### 2. 仅查询股票代码

```bash
# 只查ticker，不下载
python3 {SKILL_DIR}/fetch_prospectus.py 美团 --ticker

# 输出示例:
# {"query": "美团", "stock_code": "03690", "name_cn": "美团", "name_en": ""}
```

### 3. 搜索文件列表

```bash
# 查ticker后搜索文件（不下载）
python3 {SKILL_DIR}/fetch_prospectus.py 智谱 --search
```

### 4. 下载招股书

```bash
# 搜索并下载（默认模式）
python3 {SKILL_DIR}/fetch_prospectus.py 02513

# 显示浏览器窗口（调试）
python3 {SKILL_DIR}/fetch_prospectus.py 02513 --visible
```

### 5. 直接下载指定URL

```bash
python3 {SKILL_DIR}/fetch_prospectus.py --url "https://www1.hkexnews.hk/listedco/listconews/sehk/2025/xxx.pdf"
```

## 输出说明

### Ticker查询输出 (--ticker)

```json
{
  "query": "智谱",
  "stock_code": "02513",
  "name_cn": "智譜人工智能",
  "name_en": "ZHIPU AI"
}
```

### 多个匹配结果

如果公司名匹配到多个结果，会返回列表供确认：

```json
{
  "query": "中国",
  "matches": [
    {"code": "00941", "name_cn": "中國移動", "name_en": "CHINA MOBILE"},
    {"code": "00883", "name_cn": "中國海洋石油", "name_en": "CNOOC"}
  ],
  "message": "多个匹配结果，请指定更精确的名称或直接使用股票代码"
}
```

### 搜索结果输出 (--search)

```json
{
  "query": "智谱",
  "stock_code": "02513",
  "count": 14,
  "pdf_urls": ["https://..."]
}
```

### 下载结果输出

```json
{
  "query": "智谱",
  "stock_code": "02513",
  "downloaded": ["/Users/xxx/Downloads/hk_ipo_prospectus/02513/xxx.pdf"]
}
```

## 下载文件位置

默认保存到：`~/Downloads/hk_ipo_prospectus/<股票代码>/`

## 常见IPO文件类型

| 文件类型 | 说明 |
|---------|------|
| 全球發售 | 完整招股书（最重要） |
| 聆訊後資料集 | 聆讯后补充资料 |
| 發售價及配發結果公告 | 定价和配售结果 |
| 公司章程 | 公司章程全文 |
| 董事名單與其角色及職能 | 董事信息 |

## 招股书分析要点

下载招股书后，重点关注以下信息：

### 1. 股本结构
- 发行前总股本
- 本次发行股数
- 发行后总股本
- 公众持股比例

### 2. 募资详情
- 发行价区间
- 募资总额
- 超额配股权
- 稳定价格机制

### 3. 基石投资者
- 基石投资者名单
- 认购金额
- 锁定期安排

### 4. 股东结构
- 大股东持股比例
- 高管持股
- 锁定期安排

### 5. 资金用途
- 募资用途分配
- 研发投入计划
- 业务扩展计划

## 依赖安装

```bash
pip install playwright requests
playwright install chromium
```

## 数据源

- 港交所披露易 (HKEXnews): https://www.hkexnews.hk
- Ticker查询: Brave Search API（需设置 `BRAVE_API_KEY` 环境变量）
- Yahoo Finance API（英文名查询备用）

## 环境变量

```bash
# Brave Search API（推荐，用于中文公司名查询ticker）
export BRAVE_API_KEY="your_api_key"
```

## 注意事项

1. **英文名查询**：使用Yahoo Finance API，速度快且准确
2. **中文名查询**：使用搜索引擎，结果可能不稳定
3. 文件搜索需要Playwright浏览器，首次使用需安装
4. 搜索默认获取最近180天的上市文件
5. 下载默认限制最多10个文件

## 常见问题

### Q: 中文公司名称查询失败？
A: 搜索引擎结果有时不稳定，可尝试：
1. 使用**英文名称**（推荐，走Yahoo Finance，更可靠）
2. 直接使用**股票代码**
3. 稍后重试

### Q: 搜索返回0结果？
A: 检查该股票是否为近期IPO（默认搜索180天内）

### Q: 下载失败？
A:
1. 检查网络连接
2. 使用 `--visible` 模式调试
3. 确认 Playwright Chromium 已安装
