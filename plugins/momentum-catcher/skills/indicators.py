#!/usr/bin/env python3
"""
Momentum Catcher - 资金动量捕捉框架
技术指标计算和信号生成模块

Usage:
    python indicators.py --ticker 02513.HK
    python indicators.py --ticker 02513.HK --fetch-data
    python indicators.py --ticker 02513.HK --signal
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None
    print("提示: yfinance未安装，将使用Tushare (pip install yfinance)")

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    ts = None
    TUSHARE_AVAILABLE = False

# Technical indicators - use ta library or fallback to manual calculation
try:
    import ta as ta_lib
    USE_TA_LIB = True
except ImportError:
    USE_TA_LIB = False
    print("提示: 安装ta库可获得更多指标 (pip install ta)")

import requests
import re


# ============================================================
# 免费实时行情接口（新浪/腾讯）
# ============================================================

def fetch_realtime_quote_sina(ticker: str) -> Dict:
    """
    从新浪财经获取港股实时行情（免费）

    Args:
        ticker: 股票代码（如 02513, 2513, 02513.HK）

    Returns:
        Dict with realtime quote data
    """
    # 标准化代码
    code = ticker.upper().replace('.HK', '').zfill(5)
    url = f"https://hq.sinajs.cn/list=rt_hk{code}"

    headers = {
        'Referer': 'https://finance.sina.com.cn',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'gbk'

        match = re.search(r'"([^"]*)"', resp.text)
        if not match or not match.group(1):
            return {'error': f'无数据: {ticker}'}

        fields = match.group(1).split(',')
        if len(fields) < 18:
            return {'error': '数据格式错误'}

        return {
            'source': 'sina',
            'code': code,
            'name_en': fields[0],
            'name_cn': fields[1],
            'open': float(fields[2]) if fields[2] else 0,
            'prev_close': float(fields[3]) if fields[3] else 0,
            'high': float(fields[4]) if fields[4] else 0,
            'low': float(fields[5]) if fields[5] else 0,
            'price': float(fields[6]) if fields[6] else 0,
            'change': float(fields[7]) if fields[7] else 0,
            'change_pct': float(fields[8]) if fields[8] else 0,
            'bid': float(fields[9]) if fields[9] else 0,
            'ask': float(fields[10]) if fields[10] else 0,
            'amount': float(fields[11]) if fields[11] else 0,
            'volume': int(float(fields[12])) if fields[12] else 0,
            'high_52w': float(fields[15]) if len(fields) > 15 and fields[15] else 0,
            'low_52w': float(fields[16]) if len(fields) > 16 and fields[16] else 0,
            'datetime': f"{fields[17]} {fields[18]}" if len(fields) > 18 else '',
        }
    except Exception as e:
        return {'error': str(e)}


def fetch_realtime_quote_tencent(ticker: str) -> Dict:
    """
    从腾讯财经获取港股实时行情（免费）

    Args:
        ticker: 股票代码（如 02513, 2513, 02513.HK）

    Returns:
        Dict with realtime quote data
    """
    # 标准化代码
    code = ticker.upper().replace('.HK', '').zfill(5)
    url = f"https://qt.gtimg.cn/q=r_hk{code}"

    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = 'gbk'

        match = re.search(r'"([^"]*)"', resp.text)
        if not match or not match.group(1):
            return {'error': f'无数据: {ticker}'}

        fields = match.group(1).split('~')
        if len(fields) < 40:
            return {'error': '数据格式错误'}

        return {
            'source': 'tencent',
            'code': code,
            'name_cn': fields[1],
            'price': float(fields[3]) if fields[3] else 0,
            'prev_close': float(fields[4]) if fields[4] else 0,
            'open': float(fields[5]) if fields[5] else 0,
            'volume': int(float(fields[6])) if fields[6] else 0,
            'datetime': fields[30] if len(fields) > 30 else '',
            'change': float(fields[31]) if len(fields) > 31 and fields[31] else 0,
            'change_pct': float(fields[32]) if len(fields) > 32 and fields[32] else 0,
            'high': float(fields[33]) if len(fields) > 33 and fields[33] else 0,
            'low': float(fields[34]) if len(fields) > 34 and fields[34] else 0,
            'amount': float(fields[38]) if len(fields) > 38 and fields[38] else 0,
            'high_52w': float(fields[49]) if len(fields) > 49 and fields[49] else 0,
            'low_52w': float(fields[50]) if len(fields) > 50 and fields[50] else 0,
        }
    except Exception as e:
        return {'error': str(e)}


def fetch_realtime_quote(ticker: str, source: str = 'sina') -> Dict:
    """
    获取港股实时行情（免费）

    Args:
        ticker: 股票代码
        source: 数据源 ('sina' 或 'tencent')

    Returns:
        Dict with realtime quote
    """
    if source == 'tencent':
        return fetch_realtime_quote_tencent(ticker)
    else:
        return fetch_realtime_quote_sina(ticker)


def fetch_realtime_quotes_batch(tickers: List[str], source: str = 'sina') -> List[Dict]:
    """
    批量获取港股实时行情

    Args:
        tickers: 股票代码列表
        source: 数据源

    Returns:
        List of quote dicts
    """
    results = []
    for ticker in tickers:
        quote = fetch_realtime_quote(ticker, source)
        results.append(quote)
    return results


def print_realtime_quote(ticker: str):
    """打印实时行情"""
    quote = fetch_realtime_quote(ticker)

    if 'error' in quote:
        print(f"错误: {quote['error']}")
        return

    print(f"\n{'='*50}")
    print(f"[{quote['code']}] {quote.get('name_cn', '')} 实时行情")
    print(f"{'='*50}")
    print(f"数据源: {quote['source']}")
    print(f"时间: {quote.get('datetime', 'N/A')}")
    print(f"")
    print(f"现价: {quote['price']:.3f} HKD")
    print(f"涨跌: {quote['change']:+.3f} ({quote['change_pct']:+.2f}%)")
    print(f"")
    print(f"开盘: {quote['open']:.3f}")
    print(f"最高: {quote['high']:.3f}")
    print(f"最低: {quote['low']:.3f}")
    print(f"昨收: {quote['prev_close']:.3f}")
    print(f"")
    print(f"成交量: {quote['volume']:,}")
    print(f"成交额: {quote['amount']/100000000:.2f}亿")
    if quote.get('high_52w'):
        print(f"")
        print(f"52周高: {quote['high_52w']:.3f}")
        print(f"52周低: {quote['low_52w']:.3f}")
    print(f"{'='*50}")


# ============================================================
# 手动实现技术指标（备用方案）
# ============================================================

def _sma(series: pd.Series, length: int) -> pd.Series:
    """简单移动平均"""
    return series.rolling(window=length).mean()


def _ema(series: pd.Series, length: int) -> pd.Series:
    """指数移动平均"""
    return series.ewm(span=length, adjust=False).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """相对强弱指标"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=length).mean()
    avg_loss = loss.rolling(window=length).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """MACD指标"""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """平均真实范围"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()


def _bbands(series: pd.Series, length: int = 20, std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """布林带"""
    middle = _sma(series, length)
    std_dev = series.rolling(window=length).std()
    upper = middle + std * std_dev
    lower = middle - std * std_dev
    return upper, middle, lower


# ============================================================
# 数据获取模块
# ============================================================

def normalize_ticker(ticker: str) -> str:
    """
    标准化港股代码
    02513 -> 2513.HK
    2513 -> 2513.HK
    02513.HK -> 2513.HK
    0700.HK -> 0700.HK
    """
    ticker = ticker.upper().strip()
    if ticker.endswith('.HK'):
        code = ticker[:-3]
    else:
        code = ticker

    # 移除前导零（但保留至少4位）
    code = code.lstrip('0') or '0'
    # 港股代码通常是4位数字
    if len(code) < 4:
        code = code.zfill(4)

    return f"{code}.HK"


def normalize_ticker_tushare(ticker: str) -> str:
    """
    标准化港股代码为Tushare格式
    2513 -> 02513.HK
    02513.HK -> 02513.HK
    """
    ticker = ticker.upper().strip()
    if ticker.endswith('.HK'):
        code = ticker[:-3]
    else:
        code = ticker

    # Tushare港股需要5位数字
    code = code.zfill(5)
    return f"{code}.HK"


def fetch_hk_stock_data_tushare(
    ticker: str,
    token: str = None,
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    使用Tushare获取港股历史行情数据

    Args:
        ticker: 股票代码（如 02513.HK）
        token: Tushare API token（可通过环境变量 TUSHARE_TOKEN 设置）
        start_date: 开始日期 YYYYMMDD
        end_date: 结束日期 YYYYMMDD

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    if not TUSHARE_AVAILABLE:
        print("错误: tushare未安装，请运行: pip install tushare")
        return pd.DataFrame()

    # 获取token
    token = token or os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("错误: 未设置Tushare Token")
        print("请设置环境变量: export TUSHARE_TOKEN=你的token")
        print("或传入参数: --tushare-token xxx")
        return pd.DataFrame()

    ts_code = normalize_ticker_tushare(ticker)
    print(f"[Tushare] 正在获取 {ts_code} 行情数据...")

    try:
        pro = ts.pro_api(token)

        # 默认获取最近6个月数据
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=180)).strftime('%Y%m%d')

        # 获取港股日线数据
        df = pro.hk_daily(
            ts_code=ts_code,
            start_date=start_date,
            end_date=end_date
        )

        if df is None or df.empty:
            print(f"警告: {ts_code} 无数据")
            return pd.DataFrame()

        # 标准化列名
        df = df.rename(columns={
            'trade_date': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'vol': 'volume'
        })

        # 转换日期为索引
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        df = df.sort_index()

        # 只保留需要的列
        df = df[['open', 'high', 'low', 'close', 'volume']]

        print(f"获取到 {len(df)} 条数据，日期范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f"Tushare获取数据失败: {e}")
        return pd.DataFrame()


def fetch_hk_mins_tushare(
    ticker: str,
    token: str = None,
    freq: str = '1min',
    start_time: str = None,
    end_time: str = None
) -> pd.DataFrame:
    """
    使用Tushare获取港股分钟级行情数据

    Args:
        ticker: 股票代码
        token: Tushare API token
        freq: 频率 (1min, 5min, 15min, 30min, 60min)
        start_time: 开始时间 (YYYY-MM-DD HH:MM:SS)
        end_time: 结束时间

    Returns:
        DataFrame with minute-level OHLCV data
    """
    if not TUSHARE_AVAILABLE:
        print("错误: tushare未安装")
        return pd.DataFrame()

    token = token or os.environ.get('TUSHARE_TOKEN')
    if not token:
        print("错误: 未设置Tushare Token")
        return pd.DataFrame()

    ts_code = normalize_ticker_tushare(ticker)
    print(f"[Tushare] 正在获取 {ts_code} {freq}分钟数据...")

    try:
        pro = ts.pro_api(token)

        # 默认获取今天的数据
        if not end_time:
            end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        if not start_time:
            start_time = datetime.now().strftime('%Y-%m-%d') + ' 09:30:00'

        df = pro.hk_mins(
            ts_code=ts_code,
            freq=freq,
            start_date=start_time,
            end_date=end_time
        )

        if df is None or df.empty:
            print(f"警告: {ts_code} 无分钟数据")
            return pd.DataFrame()

        # 标准化
        df = df.rename(columns={
            'trade_time': 'datetime',
            'vol': 'volume',
            'amount': 'amount'
        })
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.set_index('datetime')
        df = df.sort_index()

        print(f"获取到 {len(df)} 条分钟数据")
        return df

    except Exception as e:
        print(f"获取分钟数据失败: {e}")
        return pd.DataFrame()


def fetch_hk_stock_data(ticker: str, period: str = "6mo", source: str = "auto", token: str = None) -> pd.DataFrame:
    """
    获取港股历史行情数据

    Args:
        ticker: 股票代码（如 02513.HK）
        period: 数据周期（1mo, 3mo, 6mo, 1y, 2y, 5y, max）
        source: 数据源 ("yfinance", "tushare", "auto")
        token: Tushare API token

    Returns:
        DataFrame with columns: open, high, low, close, volume
    """
    # 计算日期范围
    period_days = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}
    days = period_days.get(period, 180)
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y%m%d')

    # 自动选择数据源
    if source == "auto":
        # 优先尝试yfinance
        if yf is not None:
            df = _fetch_yfinance(ticker, period)
            if not df.empty:
                return df

        # yfinance失败则尝试tushare
        if TUSHARE_AVAILABLE and (token or os.environ.get('TUSHARE_TOKEN')):
            df = fetch_hk_stock_data_tushare(ticker, token, start_date, end_date)
            if not df.empty:
                return df

        print("所有数据源均失败")
        return pd.DataFrame()

    elif source == "tushare":
        return fetch_hk_stock_data_tushare(ticker, token, start_date, end_date)

    else:  # yfinance
        return _fetch_yfinance(ticker, period)


def _fetch_yfinance(ticker: str, period: str) -> pd.DataFrame:
    """使用yfinance获取数据"""
    if yf is None:
        print("yfinance未安装")
        return pd.DataFrame()

    ticker = normalize_ticker(ticker)
    print(f"[yfinance] 正在获取 {ticker} 行情数据...")

    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval="1d")

        if df.empty:
            print(f"警告: {ticker} 无数据，可能是新股或代码错误")
            return pd.DataFrame()

        # 标准化列名
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # 只保留需要的列
        df = df[['open', 'high', 'low', 'close', 'volume']]

        print(f"获取到 {len(df)} 条数据，日期范围: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f"yfinance获取数据失败: {e}")
        return pd.DataFrame()


def fetch_southbound_flow() -> pd.DataFrame:
    """
    获取南向资金流向数据

    Returns:
        DataFrame with columns: date, net_buy_sh, net_buy_sz, total
    """
    try:
        import akshare as ak
        print("正在获取南向资金数据...")

        df = ak.stock_hsgt_hist_em(symbol="南向")

        if df is not None and not df.empty:
            df = df.rename(columns={
                '日期': 'date',
                '当日资金流入': 'net_buy',
            })
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            print(f"获取到 {len(df)} 条南向资金数据")
            return df

    except ImportError:
        print("警告: akshare未安装，无法获取南向资金数据")
        print("请运行: pip install akshare")
    except Exception as e:
        print(f"获取南向资金数据失败: {e}")

    return pd.DataFrame()


# ============================================================
# 技术指标计算模块
# ============================================================

def calculate_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有动量相关技术指标

    Args:
        df: 包含 OHLCV 数据的 DataFrame

    Returns:
        添加了技术指标的 DataFrame
    """
    if df.empty:
        return df

    df = df.copy()

    # 1. 移动平均线
    df['ma5'] = _sma(df['close'], 5)
    df['ma10'] = _sma(df['close'], 10)
    df['ma20'] = _sma(df['close'], 20)
    df['ma60'] = _sma(df['close'], 60)

    # 2. MACD
    macd_line, signal_line, histogram = _macd(df['close'], fast=12, slow=26, signal=9)
    df['MACD_12_26_9'] = macd_line
    df['MACDs_12_26_9'] = signal_line
    df['MACDh_12_26_9'] = histogram

    # 3. RSI
    df['rsi_14'] = _rsi(df['close'], length=14)

    # 4. 布林带
    upper, middle, lower = _bbands(df['close'], length=20, std=2)
    df['BBU_20_2.0'] = upper
    df['BBM_20_2.0'] = middle
    df['BBL_20_2.0'] = lower

    # 5. ATR（用于止损计算）
    df['atr_14'] = _atr(df['high'], df['low'], df['close'], length=14)

    # 6. 20日最高价
    df['high_20d'] = df['high'].rolling(window=20).max()

    # 7. 成交量指标
    df['volume_ma20'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma20']

    # 8. 换手率（需要流通股本，这里用相对换手）
    df['turnover_ratio'] = df['volume'].pct_change().rolling(window=5).std()

    # 9. 量价关系
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()

    return df


def calculate_signal_scores(df: pd.DataFrame) -> Dict:
    """
    计算信号评分

    Returns:
        包含各项评分的字典
    """
    if df.empty or len(df) < 20:
        return {'error': '数据不足'}

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    scores = {
        'date': df.index[-1].strftime('%Y-%m-%d'),
        'close': round(latest['close'], 2),
    }

    # ============ 动量信号 ============
    momentum_signals = []

    # 1. 价格突破20日高点
    price_breakout = latest['close'] > latest['high_20d'] * 0.995  # 允许0.5%误差
    momentum_signals.append(('price_breakout', price_breakout))
    scores['price_breakout'] = {
        'value': f"{latest['close']:.2f} vs {latest['high_20d']:.2f}",
        'passed': price_breakout
    }

    # 2. MACD金叉且在零轴上方
    macd_col = [c for c in df.columns if c.startswith('MACD_') and not c.startswith('MACDh') and not c.startswith('MACDs')]
    signal_col = [c for c in df.columns if c.startswith('MACDs_')]

    if macd_col and signal_col:
        macd_val = latest[macd_col[0]]
        signal_val = latest[signal_col[0]]
        macd_golden = macd_val > signal_val and macd_val > 0
        momentum_signals.append(('macd_golden', macd_golden))
        scores['macd'] = {
            'value': f"DIF={macd_val:.3f}, DEA={signal_val:.3f}",
            'passed': macd_golden
        }
    else:
        scores['macd'] = {'value': 'N/A', 'passed': False}

    # 3. RSI在强势区（60-80）
    rsi_strong = 60 < latest['rsi_14'] < 80 if pd.notna(latest['rsi_14']) else False
    momentum_signals.append(('rsi_strong', rsi_strong))
    scores['rsi'] = {
        'value': f"{latest['rsi_14']:.1f}" if pd.notna(latest['rsi_14']) else 'N/A',
        'passed': rsi_strong
    }

    # 4. 均线多头排列
    ma_bullish = (
        pd.notna(latest['ma5']) and
        pd.notna(latest['ma10']) and
        pd.notna(latest['ma20']) and
        latest['ma5'] > latest['ma10'] > latest['ma20']
    )
    momentum_signals.append(('ma_bullish', ma_bullish))
    scores['ma_alignment'] = {
        'value': f"MA5={latest['ma5']:.2f}, MA10={latest['ma10']:.2f}, MA20={latest['ma20']:.2f}" if pd.notna(latest['ma5']) else 'N/A',
        'passed': ma_bullish
    }

    # 5. 成交量异动（>1.5倍20日均量）
    volume_surge = latest['volume_ratio'] > 1.5 if pd.notna(latest['volume_ratio']) else False
    momentum_signals.append(('volume_surge', volume_surge))
    scores['volume'] = {
        'value': f"{latest['volume_ratio']:.2f}x" if pd.notna(latest['volume_ratio']) else 'N/A',
        'passed': volume_surge
    }

    # 计算动量得分
    momentum_score = sum(1 for _, passed in momentum_signals if passed)
    scores['momentum_score'] = momentum_score
    scores['momentum_signals'] = momentum_signals

    # ============ 退出信号 ============
    exit_signals = []
    exit_weight = 0

    # 1. 量价背离（价格新高但成交量萎缩）
    if len(df) > 5:
        recent_high = df['close'].iloc[-5:].max()
        is_new_high = latest['close'] >= recent_high * 0.99

        if is_new_high:
            # 找到前高点的成交量
            high_idx = df['close'].iloc[-5:].idxmax()
            high_volume = df.loc[high_idx, 'volume']
            volume_divergence = latest['volume'] < high_volume * 0.7

            if volume_divergence:
                exit_signals.append(('volume_divergence', 0.30))
                exit_weight += 0.30

    # 2. RSI超买回落
    if pd.notna(latest['rsi_14']):
        recent_max_rsi = df['rsi_14'].iloc[-10:].max() if len(df) > 10 else 100
        rsi_overbought_pullback = recent_max_rsi > 80 and latest['rsi_14'] < 70
        if rsi_overbought_pullback:
            exit_signals.append(('rsi_overbought_pullback', 0.20))
            exit_weight += 0.20

    # 3. MACD死叉
    if macd_col and signal_col:
        macd_death = latest[macd_col[0]] < latest[signal_col[0]]
        if macd_death:
            exit_signals.append(('macd_death_cross', 0.15))
            exit_weight += 0.15

    # 4. 跌破10日均线
    if pd.notna(latest['ma10']):
        below_ma10 = latest['close'] < latest['ma10']
        if below_ma10:
            exit_signals.append(('below_ma10', 0.10))
            exit_weight += 0.10

    # 5. 跌破20日均线
    if pd.notna(latest['ma20']):
        below_ma20 = latest['close'] < latest['ma20']
        if below_ma20:
            exit_signals.append(('below_ma20', 0.25))
            exit_weight += 0.25

    scores['exit_signals'] = exit_signals
    scores['exit_weight'] = round(exit_weight, 2)

    return scores


# ============================================================
# 信号生成模块
# ============================================================

def generate_entry_signal(
    momentum_score: int,
    event_score: float = 4.0,
    capital_score: int = 3
) -> Tuple[str, str, int]:
    """
    生成入场信号

    Args:
        momentum_score: 动量信号得分 (0-5)
        event_score: 事件强度得分 (1-5)
        capital_score: 资金确认得分 (0-5)

    Returns:
        (signal, description, position_pct)
    """
    # 动量必须达到3分以上才考虑入场
    if momentum_score < 3:
        return ('HOLD', '动量不足，观望', 0)

    # 入场决策矩阵
    if event_score >= 4 and capital_score >= 4:
        return ('STRONG_BUY', '事件强+资金极强确认', 100)
    elif event_score >= 4 and capital_score >= 3:
        return ('BUY', '事件强+资金强确认', 80)
    elif event_score >= 3 and capital_score >= 4:
        return ('BUY', '事件中+资金极强确认', 60)
    elif event_score >= 3 and capital_score >= 3:
        return ('BUY', '事件中+资金强确认', 40)
    else:
        return ('HOLD', '条件不足，继续观望', 0)


def generate_exit_signal(exit_weight: float, below_ma20: bool = False) -> Tuple[str, str]:
    """
    生成退出信号

    Args:
        exit_weight: 退出信号权重累计
        below_ma20: 是否跌破20日均线

    Returns:
        (signal, description)
    """
    if below_ma20:
        return ('EXIT_ALL', '跌破20日均线，无条件清仓')
    elif exit_weight >= 0.5:
        return ('EXIT_ALL', f'衰减信号权重{exit_weight:.0%}≥50%，清仓')
    elif exit_weight >= 0.2:
        return ('REDUCE', f'衰减信号权重{exit_weight:.0%}≥20%，减仓30%')
    else:
        return ('HOLD', '无明显退出信号，继续持有')


# ============================================================
# 报告生成模块
# ============================================================

def generate_report(ticker: str, df: pd.DataFrame, scores: Dict) -> str:
    """生成分析报告"""
    ticker = normalize_ticker(ticker)
    report = []

    report.append(f"\n{'='*60}")
    report.append(f"[{ticker}] 动量分析报告")
    report.append(f"{'='*60}")
    report.append(f"分析日期: {scores.get('date', 'N/A')}")
    report.append(f"当前价格: {scores.get('close', 'N/A')} HKD")
    report.append("")

    # 动量信号
    report.append("-" * 40)
    report.append("【动量信号】")
    report.append("-" * 40)

    signal_names = {
        'price_breakout': '价格突破20日高点',
        'macd': 'MACD金叉(零轴上)',
        'rsi': 'RSI强势区(60-80)',
        'ma_alignment': '均线多头排列',
        'volume': '成交量异动(>1.5x)'
    }

    for key in ['price_breakout', 'macd', 'rsi', 'ma_alignment', 'volume']:
        if key in scores:
            info = scores[key]
            status = "✅" if info['passed'] else "❌"
            report.append(f"  {signal_names[key]}: {info['value']} {status}")

    momentum_score = scores.get('momentum_score', 0)
    report.append(f"\n  动量得分: {momentum_score}/5")

    # 入场信号
    signal, desc, position = generate_entry_signal(momentum_score)
    report.append(f"\n  入场信号: {signal}")
    report.append(f"  建议说明: {desc}")
    if position > 0:
        report.append(f"  建议仓位: {position}%")

    # 退出信号
    report.append("")
    report.append("-" * 40)
    report.append("【退出信号】")
    report.append("-" * 40)

    exit_signals = scores.get('exit_signals', [])
    exit_weight = scores.get('exit_weight', 0)

    if exit_signals:
        for sig_name, weight in exit_signals:
            sig_names = {
                'volume_divergence': '量价背离',
                'rsi_overbought_pullback': 'RSI超买回落',
                'macd_death_cross': 'MACD死叉',
                'below_ma10': '跌破10日均线',
                'below_ma20': '跌破20日均线'
            }
            report.append(f"  ⚠️ {sig_names.get(sig_name, sig_name)}: 权重 {weight:.0%}")
    else:
        report.append("  无退出信号触发")

    report.append(f"\n  退出信号权重合计: {exit_weight:.0%}")

    # 检查是否跌破MA20
    latest = df.iloc[-1]
    below_ma20 = latest['close'] < latest['ma20'] if pd.notna(latest['ma20']) else False

    exit_signal, exit_desc = generate_exit_signal(exit_weight, below_ma20)
    report.append(f"  退出信号: {exit_signal}")
    report.append(f"  建议说明: {exit_desc}")

    # 关键价位
    report.append("")
    report.append("-" * 40)
    report.append("【关键价位】")
    report.append("-" * 40)

    if pd.notna(latest['ma10']):
        report.append(f"  MA10 (短期支撑): {latest['ma10']:.2f}")
    if pd.notna(latest['ma20']):
        report.append(f"  MA20 (中期支撑): {latest['ma20']:.2f}")
    if pd.notna(latest['high_20d']):
        report.append(f"  20日高点 (突破位): {latest['high_20d']:.2f}")

    # 止损建议
    if pd.notna(latest['atr_14']):
        stop_loss = latest['close'] - 2 * latest['atr_14']
        report.append(f"  建议止损位 (2ATR): {stop_loss:.2f}")

    report.append("")
    report.append("="*60)

    return "\n".join(report)


# ============================================================
# 检测成交量异动
# ============================================================

def check_volume_anomaly(df: pd.DataFrame, threshold: float = 1.5) -> List[Dict]:
    """
    检测成交量异动

    Args:
        df: 行情数据
        threshold: 异动阈值（相对于20日均量的倍数）

    Returns:
        异动记录列表
    """
    if df.empty or 'volume_ratio' not in df.columns:
        return []

    anomalies = []

    for idx, row in df.iterrows():
        if pd.notna(row['volume_ratio']) and row['volume_ratio'] > threshold:
            anomalies.append({
                'date': idx.strftime('%Y-%m-%d'),
                'volume_ratio': round(row['volume_ratio'], 2),
                'close': round(row['close'], 2),
                'change': round(row['price_change'] * 100, 2) if pd.notna(row['price_change']) else 0
            })

    return anomalies


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Momentum Catcher - 资金动量捕捉框架')
    parser.add_argument('--ticker', '-t', required=True, help='股票代码（如 02513.HK 或 02513）')
    parser.add_argument('--period', '-p', default='6mo', help='数据周期（1mo, 3mo, 6mo, 1y）')
    parser.add_argument('--source', default='auto', choices=['auto', 'yfinance', 'tushare'],
                        help='数据源 (auto=自动选择, yfinance, tushare)')
    parser.add_argument('--tushare-token', help='Tushare API Token (或设置环境变量 TUSHARE_TOKEN)')
    parser.add_argument('--realtime', '-r', action='store_true', help='获取实时行情（免费）')
    parser.add_argument('--realtime-source', default='sina', choices=['sina', 'tencent'],
                        help='实时行情数据源')
    parser.add_argument('--fetch-data', action='store_true', help='仅获取数据')
    parser.add_argument('--signal', '-s', action='store_true', help='输出信号详情')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--event-score', type=float, default=4.0, help='事件强度评分（1-5）')
    parser.add_argument('--capital-score', type=int, default=3, help='资金确认评分（0-5）')

    args = parser.parse_args()

    # 实时行情模式
    if args.realtime:
        print_realtime_quote(args.ticker)
        sys.exit(0)

    # 获取数据
    df = fetch_hk_stock_data(args.ticker, args.period, source=args.source, token=args.tushare_token)

    if df.empty:
        print("无法获取数据，请检查股票代码")
        sys.exit(1)

    if args.fetch_data:
        # 仅获取数据模式
        output_file = args.output or f"{normalize_ticker(args.ticker).replace('.', '_')}_data.csv"
        df.to_csv(output_file)
        print(f"数据已保存到: {output_file}")
        sys.exit(0)

    # 计算指标
    print("\n正在计算技术指标...")
    df = calculate_momentum_indicators(df)

    # 计算信号
    scores = calculate_signal_scores(df)

    if 'error' in scores:
        print(f"错误: {scores['error']}")
        sys.exit(1)

    # 生成报告
    report = generate_report(args.ticker, df, scores)
    print(report)

    # 输出详细信号（可选）
    if args.signal:
        print("\n【详细信号数据】")
        print("-" * 40)

        # 成交量异动检测
        anomalies = check_volume_anomaly(df)
        if anomalies:
            print(f"\n近期成交量异动 ({len(anomalies)}次):")
            for a in anomalies[-5:]:  # 最近5次
                print(f"  {a['date']}: {a['volume_ratio']}x, 涨跌{a['change']}%")

    # 保存结果（可选）
    if args.output:
        df.to_csv(args.output)
        print(f"\n数据已保存到: {args.output}")


if __name__ == '__main__':
    main()
