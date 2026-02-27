#!/usr/bin/env python3
"""
港股历史数据获取工具

支持三个免费数据源，自动切换：
1. 新浪财经 (推荐，最稳定)
2. AKShare (东方财富)
3. yfinance (Yahoo Finance)

使用方法:
    python fetch_history.py 02513 --start 2026-01-08
    python fetch_history.py 02513 --start 2026-01-08 --end 2026-02-27
    python fetch_history.py 02513 --source sina
"""

import sys
import json
import re
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

try:
    import requests
    import pandas as pd
except ImportError:
    print("需要安装依赖: pip install requests pandas")
    sys.exit(1)


# ============== 方法1: 腾讯财经 (推荐，最稳定) ==============

def get_hk_history_tencent(
    symbol: str,
    start_date: str,
    end_date: str = None,
    limit: int = 500
) -> Union[Dict, pd.DataFrame]:
    """
    从腾讯财经获取港股历史K线

    API: https://web.ifzq.gtimg.cn/appstock/app/fqkline/get

    特点:
    - 免费，无需注册
    - 数据稳定，响应快
    - 支持前复权
    """
    symbol_clean = symbol.replace('.HK', '').replace('.hk', '').zfill(5)

    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param=hk{symbol_clean},day,,,{limit},qfq"

    try:
        resp = requests.get(url, timeout=15)
        data = resp.json()

        if data.get('code') != 0:
            return {"error": f"腾讯财经返回错误: {data.get('msg', 'unknown')}"}

        klines = data.get('data', {}).get(f'hk{symbol_clean}', {}).get('day', [])

        if not klines:
            return {"error": f"腾讯财经无 {symbol_clean} 数据"}

        # 解析K线数据
        # 格式: [日期, 开盘, 收盘, 最高, 最低, 成交量]
        df = pd.DataFrame(klines, columns=['date', 'open', 'close', 'high', 'low', 'volume'])
        df['open'] = df['open'].astype(float)
        df['close'] = df['close'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['volume'] = df['volume'].astype(float)

        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        df.sort_index(inplace=True)

        # 计算涨跌幅
        df['change_pct'] = df['close'].pct_change() * 100

        # 按日期筛选
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date) if end_date else datetime.now()
        df = df[(df.index >= start) & (df.index <= end)]

        return df

    except Exception as e:
        return {"error": f"腾讯财经请求失败: {e}"}


# ============== 方法2: AKShare (东方财富) ==============

def get_hk_history_akshare(
    symbol: str,
    start_date: str,
    end_date: str = None
) -> Union[Dict, pd.DataFrame]:
    """
    从AKShare获取港股历史K线

    底层API: 东方财富

    特点:
    - 免费开源库
    - 数据全面，含换手率
    - 需要安装 akshare
    """
    try:
        import akshare as ak
    except ImportError:
        return {"error": "需要安装 akshare: pip install akshare"}

    symbol_clean = symbol.replace('.HK', '').replace('.hk', '').zfill(5)

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        df = ak.stock_hk_hist(
            symbol=symbol_clean,
            period="daily",
            start_date=start_date.replace('-', ''),
            end_date=end_date.replace('-', ''),
            adjust="qfq"  # 前复权
        )

        if df.empty:
            return {"error": f"AKShare 无 {symbol_clean} 数据"}

        # 标准化列名
        df.columns = ['date', 'open', 'close', 'high', 'low', 'volume',
                      'turnover', 'amplitude', 'change_pct', 'change', 'turnover_rate']
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

        return df

    except Exception as e:
        return {"error": f"AKShare 请求失败: {e}"}


# ============== 方法3: yfinance (Yahoo Finance) ==============

def get_hk_history_yfinance(
    symbol: str,
    start_date: str,
    end_date: str = None
) -> Union[Dict, pd.DataFrame]:
    """
    从Yahoo Finance获取港股历史K线

    特点:
    - 全球数据源
    - 港股代码需加 .HK 后缀
    - 新股可能数据不全
    """
    try:
        import yfinance as yf
    except ImportError:
        return {"error": "需要安装 yfinance: pip install yfinance"}

    symbol_clean = symbol.replace('.HK', '').replace('.hk', '').zfill(5)
    yf_symbol = f"{symbol_clean}.HK"

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            return {"error": f"yfinance 无 {yf_symbol} 数据"}

        # 标准化列名
        df.columns = [c.lower() for c in df.columns]

        return df

    except Exception as e:
        return {"error": f"yfinance 请求失败: {e}"}


# ============== 自动选择数据源 ==============

def get_hk_history(
    symbol: str,
    start_date: str,
    end_date: str = None,
    source: str = "auto"
) -> Union[Dict, pd.DataFrame]:
    """
    获取港股历史K线（自动选择数据源）

    Args:
        symbol: 港股代码 (如 02513, 0700)
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD (默认今天)
        source: 数据源 (auto/sina/akshare/yfinance)

    Returns:
        DataFrame 或 错误信息
    """
    sources = {
        'tencent': get_hk_history_tencent,
        'akshare': get_hk_history_akshare,
        'yfinance': get_hk_history_yfinance
    }

    if source != "auto":
        if source in sources:
            return sources[source](symbol, start_date, end_date)
        else:
            return {"error": f"未知数据源: {source}"}

    # 自动尝试所有数据源
    errors = []
    for name, func in sources.items():
        result = func(symbol, start_date, end_date)
        if isinstance(result, pd.DataFrame) and not result.empty:
            print(f"✓ 使用 {name} 数据源成功")
            return result
        if isinstance(result, dict) and "error" in result:
            errors.append(f"{name}: {result['error']}")

    return {"error": f"所有数据源失败: {'; '.join(errors)}"}


# ============== 输出格式化 ==============

def print_history(df: pd.DataFrame, ticker: str):
    """打印历史数据"""
    print(f"\n{'='*70}")
    print(f"  {ticker} 历史K线数据")
    print(f"{'='*70}")
    print(f"数据区间: {df.index[0].strftime('%Y-%m-%d')} 至 {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"交易天数: {len(df)}")
    print()

    # 统计信息
    print("📊 统计概览:")
    print(f"   起始价: {df.iloc[0]['close']:.2f}")
    print(f"   最新价: {df.iloc[-1]['close']:.2f}")
    print(f"   最高价: {df['high'].max():.2f} ({df['high'].idxmax().strftime('%Y-%m-%d')})")
    print(f"   最低价: {df['low'].min():.2f} ({df['low'].idxmin().strftime('%Y-%m-%d')})")
    print(f"   区间涨幅: {(df.iloc[-1]['close']/df.iloc[0]['close']-1)*100:+.1f}%")
    print()

    # 最近数据
    print("📈 最近10个交易日:")
    print("-" * 70)
    print(f"{'日期':<12} {'开盘':>10} {'最高':>10} {'最低':>10} {'收盘':>10} {'成交量':>12}")
    print("-" * 70)

    for idx, row in df.tail(10).iterrows():
        date_str = idx.strftime('%Y-%m-%d')
        print(f"{date_str:<12} {row['open']:>10.2f} {row['high']:>10.2f} {row['low']:>10.2f} {row['close']:>10.2f} {row['volume']:>12,.0f}")

    print("=" * 70)


def export_csv(df: pd.DataFrame, filename: str):
    """导出为CSV"""
    df.to_csv(filename)
    print(f"✓ 已导出到 {filename}")


# ============== CLI ==============

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='港股历史数据获取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据源说明:
  sina      新浪财经 (推荐，最稳定)
  akshare   东方财富 (数据全，含换手率)
  yfinance  Yahoo Finance (全球数据)

示例:
  python fetch_history.py 02513 --start 2026-01-08
  python fetch_history.py 0700 --start 2025-01-01 --end 2025-12-31
  python fetch_history.py 02513 --source akshare --csv
        """
    )

    parser.add_argument('ticker', help='港股代码 (如 02513, 0700)')
    parser.add_argument('--start', '-s', required=True, help='开始日期 YYYY-MM-DD')
    parser.add_argument('--end', '-e', help='结束日期 YYYY-MM-DD (默认今天)')
    parser.add_argument('--source', default='auto',
                        choices=['auto', 'sina', 'akshare', 'yfinance'],
                        help='数据源 (默认 auto)')
    parser.add_argument('--json', action='store_true', help='输出JSON格式')
    parser.add_argument('--csv', action='store_true', help='导出CSV文件')

    args = parser.parse_args()

    # 获取数据
    result = get_hk_history(args.ticker, args.start, args.end, args.source)

    if isinstance(result, dict) and "error" in result:
        print(f"❌ 错误: {result['error']}")
        sys.exit(1)

    df = result

    # 输出
    if args.json:
        # JSON格式
        output = df.reset_index().to_dict(orient='records')
        for row in output:
            row['date'] = row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date'])
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        # 表格格式
        print_history(df, args.ticker)

    # 导出CSV
    if args.csv:
        filename = f"{args.ticker}_{args.start}_{args.end or 'now'}.csv"
        export_csv(df, filename)


if __name__ == "__main__":
    main()
