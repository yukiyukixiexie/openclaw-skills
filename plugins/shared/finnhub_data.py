#!/usr/bin/env python3
"""
Finnhub 金融数据接口

功能:
- 美股实时行情
- Earnings Calendar
- 分析师评级
- 公司新闻
- 财务数据

配置:
    export FINNHUB_API_KEY="your_key"

注册: https://finnhub.io/ (免费 60 calls/min)
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Union

FINNHUB_API_KEY = os.environ.get("FINNHUB_API_KEY")
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def _get(endpoint: str, params: dict = None) -> Union[Dict, List]:
    """发送 GET 请求"""
    if not FINNHUB_API_KEY:
        return {"error": "请设置 FINNHUB_API_KEY 环境变量 (https://finnhub.io/)"}

    if params is None:
        params = {}
    params["token"] = FINNHUB_API_KEY

    try:
        resp = requests.get(f"{FINNHUB_BASE_URL}/{endpoint}", params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": f"Finnhub 请求失败: {str(e)}"}


# ============== 行情数据 ==============

def get_quote(symbol: str) -> Dict:
    """
    获取实时报价

    Returns:
        dict: c(现价), h(最高), l(最低), o(开盘), pc(昨收), t(时间戳)
    """
    data = _get("quote", {"symbol": symbol})

    if isinstance(data, dict) and "error" not in data:
        return {
            "symbol": symbol,
            "price": data.get("c"),
            "high": data.get("h"),
            "low": data.get("l"),
            "open": data.get("o"),
            "prev_close": data.get("pc"),
            "change": data.get("d"),
            "change_pct": data.get("dp"),
            "timestamp": datetime.fromtimestamp(data.get("t", 0)).isoformat() if data.get("t") else None,
        }

    return data


def get_candles(
    symbol: str,
    resolution: str = "D",
    from_date: str = None,
    to_date: str = None
) -> Union[Dict, "pd.DataFrame"]:
    """
    获取K线数据

    Args:
        symbol: 股票代码
        resolution: K线周期
            - '1', '5', '15', '30', '60': 分钟
            - 'D': 日线
            - 'W': 周线
            - 'M': 月线
        from_date: 开始日期 'YYYY-MM-DD'
        to_date: 结束日期

    Returns:
        DataFrame: timestamp, open, high, low, close, volume
    """
    try:
        import pandas as pd
    except ImportError:
        return {"error": "需要安装 pandas: pip install pandas"}

    # 转换日期为时间戳
    if from_date is None:
        from_ts = int((datetime.now() - timedelta(days=365)).timestamp())
    else:
        from_ts = int(datetime.strptime(from_date, '%Y-%m-%d').timestamp())

    if to_date is None:
        to_ts = int(datetime.now().timestamp())
    else:
        to_ts = int(datetime.strptime(to_date, '%Y-%m-%d').timestamp())

    data = _get("stock/candle", {
        "symbol": symbol,
        "resolution": resolution,
        "from": from_ts,
        "to": to_ts
    })

    if isinstance(data, dict) and "error" in data:
        return data

    if data.get("s") != "ok":
        return {"error": f"Finnhub 无数据: {data.get('s')}"}

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(data["t"], unit='s'),
        "open": data["o"],
        "high": data["h"],
        "low": data["l"],
        "close": data["c"],
        "volume": data["v"],
    })
    df.set_index("timestamp", inplace=True)

    return df


# ============== Earnings ==============

def get_earnings_calendar(
    from_date: str = None,
    to_date: str = None,
    symbol: str = None
) -> List[Dict]:
    """
    获取财报日历

    Args:
        from_date: 开始日期
        to_date: 结束日期
        symbol: 股票代码（可选，筛选单个股票）

    Returns:
        list: 财报发布计划
    """
    if from_date is None:
        from_date = datetime.now().strftime('%Y-%m-%d')
    if to_date is None:
        to_date = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')

    params = {"from": from_date, "to": to_date}
    if symbol:
        params["symbol"] = symbol

    data = _get("calendar/earnings", params)

    if isinstance(data, dict) and "error" in data:
        return data

    earnings = data.get("earningsCalendar", [])

    return [
        {
            "symbol": e.get("symbol"),
            "date": e.get("date"),
            "hour": e.get("hour"),  # 'bmo' (before market open), 'amc' (after market close)
            "eps_estimate": e.get("epsEstimate"),
            "eps_actual": e.get("epsActual"),
            "revenue_estimate": e.get("revenueEstimate"),
            "revenue_actual": e.get("revenueActual"),
            "year": e.get("year"),
            "quarter": e.get("quarter"),
        }
        for e in earnings
    ]


def get_earnings_surprises(symbol: str, limit: int = 4) -> List[Dict]:
    """
    获取历史 Earnings Surprise

    Returns:
        list: 过去几个季度的 EPS 预期 vs 实际
    """
    data = _get("stock/earnings", {"symbol": symbol, "limit": limit})

    if isinstance(data, list):
        return [
            {
                "period": e.get("period"),
                "actual": e.get("actual"),
                "estimate": e.get("estimate"),
                "surprise": e.get("surprise"),
                "surprise_pct": e.get("surprisePercent"),
            }
            for e in data
        ]

    return data


# ============== 分析师评级 ==============

def get_analyst_ratings(symbol: str) -> Dict:
    """
    获取分析师评级汇总

    Returns:
        dict: buy, hold, sell, strongBuy, strongSell 数量
    """
    data = _get("stock/recommendation", {"symbol": symbol})

    if isinstance(data, list) and len(data) > 0:
        latest = data[0]
        return {
            "symbol": symbol,
            "period": latest.get("period"),
            "strong_buy": latest.get("strongBuy"),
            "buy": latest.get("buy"),
            "hold": latest.get("hold"),
            "sell": latest.get("sell"),
            "strong_sell": latest.get("strongSell"),
        }

    return data


def get_price_target(symbol: str) -> Dict:
    """
    获取分析师目标价

    Returns:
        dict: 目标价高/低/平均/中位数
    """
    data = _get("stock/price-target", {"symbol": symbol})

    if isinstance(data, dict) and "error" not in data:
        return {
            "symbol": symbol,
            "target_high": data.get("targetHigh"),
            "target_low": data.get("targetLow"),
            "target_mean": data.get("targetMean"),
            "target_median": data.get("targetMedian"),
            "last_updated": data.get("lastUpdated"),
        }

    return data


# ============== 公司信息 ==============

def get_company_profile(symbol: str) -> Dict:
    """获取公司基本信息"""
    data = _get("stock/profile2", {"symbol": symbol})

    if isinstance(data, dict) and "error" not in data:
        return {
            "symbol": symbol,
            "name": data.get("name"),
            "country": data.get("country"),
            "currency": data.get("currency"),
            "exchange": data.get("exchange"),
            "industry": data.get("finnhubIndustry"),
            "ipo_date": data.get("ipo"),
            "logo": data.get("logo"),
            "market_cap": data.get("marketCapitalization"),
            "shares_outstanding": data.get("shareOutstanding"),
            "website": data.get("weburl"),
        }

    return data


def get_company_news(symbol: str, days: int = 7) -> List[Dict]:
    """
    获取公司新闻

    Args:
        symbol: 股票代码
        days: 获取最近几天的新闻

    Returns:
        list: 新闻列表
    """
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')

    data = _get("company-news", {
        "symbol": symbol,
        "from": from_date,
        "to": to_date
    })

    if isinstance(data, list):
        return [
            {
                "datetime": datetime.fromtimestamp(n.get("datetime", 0)).isoformat(),
                "headline": n.get("headline"),
                "summary": n.get("summary"),
                "source": n.get("source"),
                "url": n.get("url"),
                "category": n.get("category"),
            }
            for n in data[:20]  # 限制数量
        ]

    return data


# ============== 财务数据 ==============

def get_basic_financials(symbol: str) -> Dict:
    """
    获取基本财务指标

    Returns:
        dict: PE, PB, PS, 股息率, ROE 等
    """
    data = _get("stock/metric", {"symbol": symbol, "metric": "all"})

    if isinstance(data, dict) and "metric" in data:
        m = data["metric"]
        return {
            "symbol": symbol,
            "pe_ttm": m.get("peTTM"),
            "pe_annual": m.get("peAnnual"),
            "pb_quarterly": m.get("pbQuarterly"),
            "ps_ttm": m.get("psTTM"),
            "dividend_yield": m.get("dividendYieldIndicatedAnnual"),
            "roe_ttm": m.get("roeTTM"),
            "roa_ttm": m.get("roaTTM"),
            "current_ratio": m.get("currentRatioQuarterly"),
            "debt_equity": m.get("totalDebtToEquityQuarterly"),
            "eps_ttm": m.get("epsTTM"),
            "revenue_per_share": m.get("revenuePerShareTTM"),
            "gross_margin": m.get("grossMarginTTM"),
            "net_margin": m.get("netProfitMarginTTM"),
            "52w_high": m.get("52WeekHigh"),
            "52w_low": m.get("52WeekLow"),
            "beta": m.get("beta"),
        }

    return data


# ============== CLI ==============

def main():
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
Finnhub 金融数据接口

用法:
    python finnhub_data.py <命令> [参数]

命令:
    quote <symbol>           实时报价
    candles <symbol> [from]  K线数据
    earnings [from] [to]     财报日历
    surprise <symbol>        Earnings Surprise
    ratings <symbol>         分析师评级
    target <symbol>          目标价
    profile <symbol>         公司信息
    news <symbol>            公司新闻
    financials <symbol>      财务指标

示例:
    python finnhub_data.py quote AAPL
    python finnhub_data.py earnings 2024-01-01 2024-01-31
    python finnhub_data.py ratings NVDA

环境变量:
    FINNHUB_API_KEY    API Key (https://finnhub.io/)
        """)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    result = None

    if cmd == "quote" and len(sys.argv) >= 3:
        result = get_quote(sys.argv[2])

    elif cmd == "candles" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        from_date = sys.argv[3] if len(sys.argv) > 3 else None
        result = get_candles(symbol, from_date=from_date)
        if hasattr(result, 'tail'):
            print(result.tail(10).to_string())
            sys.exit(0)

    elif cmd == "earnings":
        from_date = sys.argv[2] if len(sys.argv) > 2 else None
        to_date = sys.argv[3] if len(sys.argv) > 3 else None
        result = get_earnings_calendar(from_date, to_date)

    elif cmd == "surprise" and len(sys.argv) >= 3:
        result = get_earnings_surprises(sys.argv[2])

    elif cmd == "ratings" and len(sys.argv) >= 3:
        result = get_analyst_ratings(sys.argv[2])

    elif cmd == "target" and len(sys.argv) >= 3:
        result = get_price_target(sys.argv[2])

    elif cmd == "profile" and len(sys.argv) >= 3:
        result = get_company_profile(sys.argv[2])

    elif cmd == "news" and len(sys.argv) >= 3:
        result = get_company_news(sys.argv[2])

    elif cmd == "financials" and len(sys.argv) >= 3:
        result = get_basic_financials(sys.argv[2])

    else:
        print(f"❌ 未知命令或参数不足: {cmd}")
        sys.exit(1)

    if result:
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
