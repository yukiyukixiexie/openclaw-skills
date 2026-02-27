# HK Realtime Quote - 港股实时行情

免费获取港股实时行情数据，支持新浪财经和腾讯财经两个数据源。

## Skill 配置

```yaml
name: hk-realtime-quote
description: 免费港股实时行情查询，支持新浪/腾讯数据源
version: 1.0.0
author: OpenClaw
triggers:
  - /hk-quote
  - /hkq
  - /港股行情
arguments:
  - name: ticker
    description: 港股代码（如 02513, 0700, 9988）
    required: true
  - name: source
    description: 数据源 (sina/tencent)
    required: false
    default: sina
```

---

## 功能特点

- **完全免费**：无需注册，无需API Key
- **近实时**：数据延迟 < 1分钟
- **双数据源**：新浪财经 + 腾讯财经，互为备份
- **全港股覆盖**：支持所有港股代码

---

## 使用方式

### 命令行

```bash
# 查询单只股票
python hk_quote.py 02513

# 指定数据源
python hk_quote.py 02513 --source tencent

# 批量查询
python hk_quote.py 02513 0700 9988

# 输出JSON格式
python hk_quote.py 02513 --json
```

### Python 调用

```python
from hk_quote import get_realtime_quote, get_batch_quotes

# 单只股票
quote = get_realtime_quote('02513')
print(f"{quote['name']} 现价: {quote['price']} 涨跌: {quote['change_pct']}%")

# 批量查询
quotes = get_batch_quotes(['02513', '0700', '9988'])
for q in quotes:
    print(f"{q['code']}: {q['price']}")
```

---

## 返回数据

| 字段 | 类型 | 说明 |
|------|------|------|
| code | str | 股票代码 |
| name | str | 股票名称 |
| price | float | 现价 |
| open | float | 开盘价 |
| high | float | 最高价 |
| low | float | 最低价 |
| prev_close | float | 昨收价 |
| change | float | 涨跌额 |
| change_pct | float | 涨跌幅(%) |
| volume | int | 成交量 |
| amount | float | 成交额 |
| high_52w | float | 52周最高 |
| low_52w | float | 52周最低 |
| datetime | str | 行情时间 |
| source | str | 数据源 |

---

## API 说明

### 新浪财经 API

**URL**: `https://hq.sinajs.cn/list=rt_hk{code}`

**特点**:
- 需要 Referer 头
- 返回 GBK 编码
- 数据最全面

### 腾讯财经 API

**URL**: `https://qt.gtimg.cn/q=r_hk{code}`

**特点**:
- 无需特殊头
- 返回 GBK 编码
- 响应较快

---

## 示例输出

```
==================================================
[02513] 智谱 实时行情
==================================================
数据源: sina
时间: 2026/02/27 15:04:46

现价: 576.000 HKD
涨跌: +19.500 (+3.50%)

开盘: 540.000
最高: 592.000
最低: 482.000
昨收: 556.500

成交量: 3,543,079
成交额: 18.91亿

52周高: 725.000
52周低: 116.100
==================================================
```

---

## 注意事项

1. **请求频率**：建议间隔 1 秒以上，避免被限流
2. **交易时间**：港股交易时间为 9:30-12:00, 13:00-16:00 (北京时间)
3. **数据准确性**：非官方数据源，仅供参考
4. **网络要求**：需要能访问新浪/腾讯服务器

---

## 依赖

```
requests>=2.25.0
```

---

## License

MIT
