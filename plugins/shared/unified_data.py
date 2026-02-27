#!/usr/bin/env python3
"""
统一数据接口 - 自动选择最佳数据源

优先级:
- 美股实时: Alpaca > Finnhub > yfinance
- 美股历史: Alpaca > yfinance
- 港股实时: 新浪 > 腾讯
- 港股历史: 腾讯 > AKShare > yfinance
- A股: AKShare
- 加密货币: CCXT

使用方法:
    from unified_data import get_realtime, get_history, get_info

    # 自动识别市场
    price = get_realtime("AAPL")      # 美股
    price = get_realtime("0700.HK")   # 港股
    price = get_realtime("600519")    # A股
    price = get_realtime("BTC/USDT")  # 加密货币

    # 历史数据
    df = get_history("AAPL", "2024-01-01")
"""

from typing import Dict, Union, List
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None


def _detect_market(symbol: str) -> str:
    """自动检测市场类型"""
    symbol_upper = symbol.upper()

    if '/' in symbol:
        return "crypto"
    elif symbol_upper.endswith('.HK'):
        return "hk"
    elif symbol.isdigit():
        if len(symbol) == 5:
            return "hk"
        elif len(symbol) == 6:
            return "a"
    else:
        return "us"


# ============== 实时行情 ==============

def get_realtime(symbol: str, source: str = "auto") -> Dict:
    """
    获取实时行情（自动选择数据源）

    Args:
        symbol: 股票代码
            - 美股: AAPL, TSLA
            - 港股: 0700.HK 或 00700
            - A股: 600519, 000001
            - 加密货币: BTC/USDT
        source: 数据源 (auto/alpaca/finnhub/yahoo/sina/tencent)

    Returns:
        dict: 实时行情数据
    """
    market = _detect_market(symbol)

    if market == "us":
        return _get_us_realtime(symbol, source)
    elif market == "hk":
        return _get_hk_realtime(symbol, source)
    elif market == "a":
        return _get_a_realtime(symbol)
    elif market == "crypto":
        return _get_crypto_realtime(symbol)
    else:
        return {"error": f"无法识别市场: {symbol}"}


def _get_us_realtime(symbol: str, source: str = "auto") -> Dict:
    """美股实时行情"""
    errors = []

    # 尝试 Alpaca
    if source in ["auto", "alpaca"]:
        try:
            from alpaca_trading import get_alpaca_snapshot
            result = get_alpaca_snapshot([symbol])
            if isinstance(result, dict) and symbol in result:
                data = result[symbol]
                data["symbol"] = symbol
                data["source"] = "alpaca"
                return data
        except Exception as e:
            errors.append(f"alpaca: {e}")

    # 尝试 Finnhub
    if source in ["auto", "finnhub"]:
        try:
            from finnhub_data import get_quote
            result = get_quote(symbol)
            if isinstance(result, dict) and "error" not in result:
                result["source"] = "finnhub"
                return result
        except Exception as e:
            errors.append(f"finnhub: {e}")

    # 尝试 yfinance
    if source in ["auto", "yahoo"]:
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return {
                "symbol": symbol,
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "open": info.get("open") or info.get("regularMarketOpen"),
                "high": info.get("dayHigh") or info.get("regularMarketDayHigh"),
                "low": info.get("dayLow") or info.get("regularMarketDayLow"),
                "prev_close": info.get("previousClose") or info.get("regularMarketPreviousClose"),
                "volume": info.get("volume") or info.get("regularMarketVolume"),
                "market_cap": info.get("marketCap"),
                "source": "yahoo",
            }
        except Exception as e:
            errors.append(f"yahoo: {e}")

    return {"error": f"所有数据源失败: {'; '.join(errors)}"}


def _get_hk_realtime(symbol: str, source: str = "auto") -> Dict:
    """港股实时行情"""
    try:
        from market_data import get_hk_stock_realtime
        return get_hk_stock_realtime(symbol, source)
    except ImportError:
        return {"error": "market_data 模块不可用"}


def _get_a_realtime(symbol: str) -> Dict:
    """A股实时行情"""
    try:
        from market_data import get_a_stock_realtime
        return get_a_stock_realtime(symbol)
    except ImportError:
        return {"error": "market_data 模块不可用"}


def _get_crypto_realtime(symbol: str) -> Dict:
    """加密货币实时行情"""
    try:
        from market_data import get_crypto_ticker
        return get_crypto_ticker(symbol)
    except ImportError:
        return {"error": "market_data 模块不可用"}


# ============== 历史数据 ==============

def get_history(
    symbol: str,
    start_date: str,
    end_date: str = None,
    interval: str = "1d",
    source: str = "auto"
) -> Union[Dict, "pd.DataFrame"]:
    """
    获取历史K线数据（自动选择数据源）

    Args:
        symbol: 股票代码
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期
        interval: K线周期 ('1d', '1h', '1w', '1m')
        source: 数据源

    Returns:
        DataFrame: OHLCV 数据
    """
    market = _detect_market(symbol)

    if market == "us":
        return _get_us_history(symbol, start_date, end_date, interval, source)
    elif market == "hk":
        return _get_hk_history(symbol, start_date, end_date, source)
    elif market == "a":
        return _get_a_history(symbol, start_date, end_date)
    elif market == "crypto":
        return _get_crypto_history(symbol, interval)
    else:
        return {"error": f"无法识别市场: {symbol}"}


def _get_us_history(
    symbol: str,
    start_date: str,
    end_date: str = None,
    interval: str = "1d",
    source: str = "auto"
) -> Union[Dict, "pd.DataFrame"]:
    """美股历史数据"""
    errors = []

    # 映射 interval
    alpaca_interval_map = {
        "1d": "1Day", "1h": "1Hour", "1w": "1Week", "1m": "1Month",
        "1min": "1Min", "5min": "5Min", "15min": "15Min", "30min": "30Min"
    }

    # 尝试 Alpaca
    if source in ["auto", "alpaca"]:
        try:
            from alpaca_trading import get_alpaca_bars
            tf = alpaca_interval_map.get(interval.lower(), "1Day")
            result = get_alpaca_bars(symbol, start_date, end_date, tf)
            if not isinstance(result, dict) or "error" not in result:
                return result
            errors.append(f"alpaca: {result.get('error')}")
        except Exception as e:
            errors.append(f"alpaca: {e}")

    # 尝试 yfinance
    if source in ["auto", "yahoo"]:
        try:
            from market_data import get_us_stock
            result = get_us_stock(symbol, start_date, end_date, interval)
            if not isinstance(result, dict) or "error" not in result:
                return result
            errors.append(f"yahoo: {result.get('error')}")
        except Exception as e:
            errors.append(f"yahoo: {e}")

    return {"error": f"所有数据源失败: {'; '.join(errors)}"}


def _get_hk_history(
    symbol: str,
    start_date: str,
    end_date: str = None,
    source: str = "auto"
) -> Union[Dict, "pd.DataFrame"]:
    """港股历史数据"""
    try:
        from market_data import get_hk_stock
        return get_hk_stock(symbol, start_date, end_date, source)
    except ImportError:
        return {"error": "market_data 模块不可用"}


def _get_a_history(
    symbol: str,
    start_date: str,
    end_date: str = None
) -> Union[Dict, "pd.DataFrame"]:
    """A股历史数据"""
    try:
        from market_data import get_a_stock
        return get_a_stock(symbol, start_date, end_date)
    except ImportError:
        return {"error": "market_data 模块不可用"}


def _get_crypto_history(symbol: str, timeframe: str = "1d") -> Union[Dict, "pd.DataFrame"]:
    """加密货币历史数据"""
    try:
        from market_data import get_crypto
        return get_crypto(symbol, timeframe=timeframe)
    except ImportError:
        return {"error": "market_data 模块不可用"}


# ============== 基本面信息 ==============

def get_info(symbol: str, source: str = "auto") -> Dict:
    """
    获取股票基本面信息

    Returns:
        dict: 公司信息、估值指标等
    """
    market = _detect_market(symbol)

    if market == "us":
        return _get_us_info(symbol, source)
    elif market == "hk":
        # 港股暂用 yfinance
        return _get_us_info(symbol + ".HK" if not symbol.endswith(".HK") else symbol, "yahoo")
    else:
        return {"error": f"暂不支持 {market} 市场的基本面数据"}


def _get_us_info(symbol: str, source: str = "auto") -> Dict:
    """美股基本面信息"""
    errors = []

    # 尝试 Finnhub
    if source in ["auto", "finnhub"]:
        try:
            from finnhub_data import get_company_profile, get_basic_financials
            profile = get_company_profile(symbol)
            financials = get_basic_financials(symbol)

            if isinstance(profile, dict) and "error" not in profile:
                if isinstance(financials, dict) and "error" not in financials:
                    profile.update(financials)
                profile["source"] = "finnhub"
                return profile
        except Exception as e:
            errors.append(f"finnhub: {e}")

    # 尝试 yfinance
    if source in ["auto", "yahoo"]:
        try:
            from market_data import get_us_stock_info
            result = get_us_stock_info(symbol)
            if isinstance(result, dict) and "error" not in result:
                result["source"] = "yahoo"
                return result
        except Exception as e:
            errors.append(f"yahoo: {e}")

    return {"error": f"所有数据源失败: {'; '.join(errors)}"}


# ============== Earnings 数据 ==============

def get_earnings_calendar(
    from_date: str = None,
    to_date: str = None,
    symbol: str = None
) -> List[Dict]:
    """获取财报日历"""
    try:
        from finnhub_data import get_earnings_calendar as finnhub_earnings
        return finnhub_earnings(from_date, to_date, symbol)
    except ImportError:
        return {"error": "finnhub_data 模块不可用"}


def get_analyst_ratings(symbol: str) -> Dict:
    """获取分析师评级"""
    try:
        from finnhub_data import get_analyst_ratings as finnhub_ratings
        return finnhub_ratings(symbol)
    except ImportError:
        return {"error": "finnhub_data 模块不可用"}


# ============== 批量获取 ==============

def get_batch_realtime(symbols: List[str]) -> Dict[str, Dict]:
    """
    批量获取实时行情

    Args:
        symbols: 股票代码列表

    Returns:
        dict: {symbol: quote_data}
    """
    results = {}

    # 按市场分组
    us_symbols = []
    hk_symbols = []
    a_symbols = []
    crypto_symbols = []

    for s in symbols:
        market = _detect_market(s)
        if market == "us":
            us_symbols.append(s)
        elif market == "hk":
            hk_symbols.append(s)
        elif market == "a":
            a_symbols.append(s)
        elif market == "crypto":
            crypto_symbols.append(s)

    # 美股批量获取
    if us_symbols:
        try:
            from alpaca_trading import get_alpaca_snapshot
            us_data = get_alpaca_snapshot(us_symbols)
            if isinstance(us_data, dict) and "error" not in us_data:
                results.update(us_data)
        except:
            # 降级到逐个获取
            for s in us_symbols:
                results[s] = get_realtime(s)

    # 其他市场逐个获取
    for s in hk_symbols + a_symbols + crypto_symbols:
        results[s] = get_realtime(s)

    return results


# ============== CLI ==============

def main():
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
统一数据接口

用法:
    python unified_data.py <命令> [参数]

命令:
    quote <symbol>              实时行情
    history <symbol> [start]    历史数据
    info <symbol>               基本面信息
    batch <symbols>             批量行情 (逗号分隔)
    earnings [from] [to]        财报日历
    ratings <symbol>            分析师评级

示例:
    python unified_data.py quote AAPL
    python unified_data.py quote 0700.HK
    python unified_data.py history TSLA 2024-01-01
    python unified_data.py batch AAPL,TSLA,NVDA
        """)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "quote" and len(sys.argv) >= 3:
        result = get_realtime(sys.argv[2])
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "history" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        start = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
        result = get_history(symbol, start)
        if hasattr(result, 'tail'):
            print(result.tail(10).to_string())
        else:
            print(json.dumps(result, indent=2, default=str))

    elif cmd == "info" and len(sys.argv) >= 3:
        result = get_info(sys.argv[2])
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "batch" and len(sys.argv) >= 3:
        symbols = sys.argv[2].split(',')
        result = get_batch_realtime(symbols)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "earnings":
        from_date = sys.argv[2] if len(sys.argv) > 2 else None
        to_date = sys.argv[3] if len(sys.argv) > 3 else None
        result = get_earnings_calendar(from_date, to_date)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "ratings" and len(sys.argv) >= 3:
        result = get_analyst_ratings(sys.argv[2])
        print(json.dumps(result, indent=2, default=str))

    else:
        print(f"❌ 未知命令: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
