#!/usr/bin/env python3
"""
全球市场数据获取工具（免费数据源）
支持: 美股、A股、港股、加密货币
所有数据源均为免费，无需 API Key
"""

import sys
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union

try:
    import pandas as pd
except ImportError:
    pd = None


# ============== 美股数据 (yfinance) ==============

def get_us_stock(
    symbol: str,
    start_date: str,
    end_date: str = None,
    interval: str = "1d"
) -> Union[Dict, "pd.DataFrame"]:
    """
    获取美股/港股/ETF 日线数据 (免费)

    数据源: Yahoo Finance (yfinance)

    Args:
        symbol: 股票代码
            - 美股: AAPL, TSLA, NVDA
            - 港股: 0700.HK, 9988.HK
            - ETF: SPY, QQQ
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期，默认今天
        interval: K线周期 '1d', '1wk', '1mo'

    Returns:
        DataFrame: Open, High, Low, Close, Volume, Adj Close
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "需要安装 yfinance: pip install yfinance"}

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        return {"error": f"未找到 {symbol} 的数据"}

    return df


def get_us_stock_info(symbol: str) -> Dict:
    """
    获取股票基本面信息 (免费)

    Returns:
        dict: 公司名称、行业、市值、PE、EPS 等
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "需要安装 yfinance: pip install yfinance"}

    ticker = yf.Ticker(symbol)
    info = ticker.info

    return {
        "symbol": symbol,
        "name": info.get("longName"),
        "sector": info.get("sector"),
        "industry": info.get("industry"),
        "market_cap": info.get("marketCap"),
        "pe_ratio": info.get("trailingPE"),
        "forward_pe": info.get("forwardPE"),
        "pb_ratio": info.get("priceToBook"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "dividend_yield": info.get("dividendYield"),
        "eps": info.get("trailingEps"),
        "revenue": info.get("totalRevenue"),
        "profit_margin": info.get("profitMargins"),
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "current_price": info.get("currentPrice"),
        "target_price": info.get("targetMeanPrice"),
        "52w_high": info.get("fiftyTwoWeekHigh"),
        "52w_low": info.get("fiftyTwoWeekLow"),
    }


def get_us_stock_financials(symbol: str) -> Dict:
    """获取财务报表 (免费)"""
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "需要安装 yfinance: pip install yfinance"}

    ticker = yf.Ticker(symbol)

    result = {}

    if ticker.financials is not None and not ticker.financials.empty:
        result["income_statement"] = ticker.financials.to_dict()

    if ticker.balance_sheet is not None and not ticker.balance_sheet.empty:
        result["balance_sheet"] = ticker.balance_sheet.to_dict()

    if ticker.cashflow is not None and not ticker.cashflow.empty:
        result["cash_flow"] = ticker.cashflow.to_dict()

    return result


# ============== A股数据 (AKShare) ==============

def get_a_stock(
    symbol: str,
    start_date: str,
    end_date: str = None,
    adjust: str = "qfq"
) -> Union[Dict, "pd.DataFrame"]:
    """
    获取A股日线数据 (免费)

    数据源: AKShare (新浪财经)

    Args:
        symbol: 股票代码，如 '000001' (平安银行), '600519' (茅台)
        start_date: 开始日期 'YYYYMMDD' 或 'YYYY-MM-DD'
        end_date: 结束日期，默认今天
        adjust: 复权类型
            - 'qfq': 前复权 (推荐)
            - 'hfq': 后复权
            - '': 不复权

    Returns:
        DataFrame: 日期, 开盘, 收盘, 最高, 最低, 成交量, 成交额, 涨跌幅等
    """
    try:
        import akshare as ak
    except ImportError:
        return {"error": "需要安装 akshare: pip install akshare"}

    # 标准化日期格式
    start_date = start_date.replace('-', '')
    if end_date is None:
        end_date = datetime.now().strftime('%Y%m%d')
    else:
        end_date = end_date.replace('-', '')

    try:
        df = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust
        )

        if df.empty:
            return {"error": f"未找到 {symbol} 的数据"}

        # 标准化列名
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume',
                      'turnover', 'amplitude', 'change_pct', 'change', 'turnover_rate']
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        return df

    except Exception as e:
        return {"error": f"获取数据失败: {str(e)}"}


def get_a_stock_list() -> Union[Dict, "pd.DataFrame"]:
    """获取A股股票列表 (免费)"""
    try:
        import akshare as ak
        return ak.stock_zh_a_spot_em()
    except ImportError:
        return {"error": "需要安装 akshare: pip install akshare"}


def get_a_stock_realtime(symbol: str) -> Dict:
    """获取A股实时行情 (免费)"""
    try:
        import akshare as ak
    except ImportError:
        return {"error": "需要安装 akshare: pip install akshare"}

    df = ak.stock_zh_a_spot_em()
    stock = df[df['代码'] == symbol]

    if stock.empty:
        return {"error": f"未找到 {symbol}"}

    row = stock.iloc[0]
    return {
        "code": row['代码'],
        "name": row['名称'],
        "price": row['最新价'],
        "change_pct": row['涨跌幅'],
        "change": row['涨跌额'],
        "volume": row['成交量'],
        "turnover": row['成交额'],
        "high": row['最高'],
        "low": row['最低'],
        "open": row['今开'],
        "prev_close": row['昨收'],
        "pe": row.get('市盈率-动态'),
        "pb": row.get('市净率'),
    }


# ============== 加密货币数据 (CCXT) ==============

def get_crypto(
    symbol: str,
    exchange: str = "binance",
    timeframe: str = "1d",
    limit: int = 500
) -> Union[Dict, "pd.DataFrame"]:
    """
    获取加密货币K线数据 (免费)

    数据源: CCXT (交易所公开API)

    Args:
        symbol: 交易对，如 'BTC/USDT', 'ETH/USDT'
        exchange: 交易所
            - 'binance': 币安
            - 'okx': 欧易
            - 'huobi': 火币
            - 'coinbase': Coinbase
            - 'kucoin': KuCoin
        timeframe: K线周期 '1m', '5m', '15m', '1h', '4h', '1d', '1w'
        limit: 获取数量 (最大取决于交易所)

    Returns:
        DataFrame: timestamp, open, high, low, close, volume
    """
    try:
        import ccxt
    except ImportError:
        return {"error": "需要安装 ccxt: pip install ccxt"}

    try:
        exchange_class = getattr(ccxt, exchange)
        ex = exchange_class()

        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)

        if not ohlcv:
            return {"error": f"未找到 {symbol} 在 {exchange} 的数据"}

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        return df

    except Exception as e:
        return {"error": f"获取数据失败: {str(e)}"}


def get_crypto_ticker(symbol: str, exchange: str = "binance") -> Dict:
    """获取加密货币实时行情 (免费)"""
    try:
        import ccxt
    except ImportError:
        return {"error": "需要安装 ccxt: pip install ccxt"}

    try:
        exchange_class = getattr(ccxt, exchange)
        ex = exchange_class()

        ticker = ex.fetch_ticker(symbol)

        return {
            "symbol": ticker['symbol'],
            "price": ticker['last'],
            "bid": ticker['bid'],
            "ask": ticker['ask'],
            "high_24h": ticker['high'],
            "low_24h": ticker['low'],
            "volume_24h": ticker['baseVolume'],
            "quote_volume_24h": ticker['quoteVolume'],
            "change_24h": ticker['percentage'],
            "vwap": ticker.get('vwap'),
        }

    except Exception as e:
        return {"error": f"获取行情失败: {str(e)}"}


def get_crypto_symbols(exchange: str = "binance") -> Union[Dict, List[str]]:
    """获取交易所支持的交易对列表 (免费)"""
    try:
        import ccxt
    except ImportError:
        return {"error": "需要安装 ccxt: pip install ccxt"}

    try:
        exchange_class = getattr(ccxt, exchange)
        ex = exchange_class()
        ex.load_markets()
        return sorted(ex.symbols)

    except Exception as e:
        return {"error": f"获取交易对失败: {str(e)}"}


# ============== 港股数据 (yfinance) ==============

def get_hk_stock(
    symbol: str,
    start_date: str,
    end_date: str = None
) -> Union[Dict, "pd.DataFrame"]:
    """
    获取港股日线数据 (免费)

    Args:
        symbol: 港股代码，如 '0700' (腾讯), '9988' (阿里)
                会自动添加 .HK 后缀
    """
    # 标准化代码格式
    if not symbol.upper().endswith('.HK'):
        symbol = f"{symbol.zfill(4)}.HK"

    return get_us_stock(symbol, start_date, end_date)


# ============== 技术指标计算 ==============

def calculate_indicators(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    计算常用技术指标

    Args:
        df: 包含 open, high, low, close, volume 的 DataFrame
            列名不区分大小写

    Returns:
        DataFrame: 添加了技术指标的数据
    """
    if pd is None:
        return {"error": "需要安装 pandas: pip install pandas"}

    df = df.copy()

    # 标准化列名为小写
    df.columns = [c.lower() for c in df.columns]

    if 'close' not in df.columns:
        return {"error": "数据中缺少 close 列"}

    # 移动平均线
    df['sma_5'] = df['close'].rolling(window=5).mean()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_60'] = df['close'].rolling(window=60).mean()
    df['sma_120'] = df['close'].rolling(window=120).mean()
    df['sma_250'] = df['close'].rolling(window=250).mean()

    # EMA
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 布林带
    df['boll_mid'] = df['close'].rolling(window=20).mean()
    df['boll_std'] = df['close'].rolling(window=20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

    # ATR (需要 high, low)
    if 'high' in df.columns and 'low' in df.columns:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()

    # 成交量移动平均
    if 'volume' in df.columns:
        df['vol_ma_5'] = df['volume'].rolling(window=5).mean()
        df['vol_ma_20'] = df['volume'].rolling(window=20).mean()

    return df


# ============== 便捷函数 ==============

def get_stock(
    symbol: str,
    start_date: str,
    end_date: str = None,
    market: str = "auto"
) -> Union[Dict, "pd.DataFrame"]:
    """
    智能获取股票数据 (自动识别市场)

    Args:
        symbol: 股票代码
        market: 市场类型
            - 'auto': 自动识别
            - 'us': 美股
            - 'a': A股
            - 'hk': 港股
            - 'crypto': 加密货币
    """
    if market == "auto":
        # 自动识别市场
        if '/' in symbol:
            market = "crypto"
        elif symbol.upper().endswith('.HK'):
            market = "hk"
        elif symbol.isdigit() and len(symbol) == 6:
            market = "a"
        else:
            market = "us"

    if market == "us":
        return get_us_stock(symbol, start_date, end_date)
    elif market == "a":
        return get_a_stock(symbol, start_date, end_date)
    elif market == "hk":
        return get_hk_stock(symbol, start_date, end_date)
    elif market == "crypto":
        return get_crypto(symbol)
    else:
        return {"error": f"不支持的市场类型: {market}"}


# ============== CLI 接口 ==============

def main():
    if len(sys.argv) < 2:
        print("""
全球市场数据获取工具 (免费数据源)

用法:
    python market_data.py <命令> [参数]

命令:
    us <symbol> [start_date]     获取美股数据
    a <symbol> [start_date]      获取A股数据
    hk <symbol> [start_date]     获取港股数据
    crypto <symbol> [exchange]   获取加密货币数据
    info <symbol>                获取股票基本面

示例:
    python market_data.py us AAPL 2024-01-01
    python market_data.py a 600519 20240101
    python market_data.py hk 0700 2024-01-01
    python market_data.py crypto BTC/USDT binance
    python market_data.py info TSLA

依赖安装:
    pip install yfinance akshare ccxt pandas
        """)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "us" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        start_date = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
        result = get_us_stock(symbol, start_date)

    elif cmd == "a" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        start_date = sys.argv[3] if len(sys.argv) > 3 else "20240101"
        result = get_a_stock(symbol, start_date)

    elif cmd == "hk" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        start_date = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
        result = get_hk_stock(symbol, start_date)

    elif cmd == "crypto" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        exchange = sys.argv[3] if len(sys.argv) > 3 else "binance"
        result = get_crypto(symbol, exchange)

    elif cmd == "info" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        result = get_us_stock_info(symbol)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        sys.exit(0)

    else:
        print(f"❌ 未知命令或参数不足: {cmd}")
        sys.exit(1)

    # 输出结果
    if isinstance(result, dict):
        if "error" in result:
            print(f"❌ {result['error']}")
            sys.exit(1)
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n📊 {sys.argv[2]} 数据 (最近5条):")
        print(result.tail().to_string())
        print(f"\n共 {len(result)} 条记录")


if __name__ == "__main__":
    main()
