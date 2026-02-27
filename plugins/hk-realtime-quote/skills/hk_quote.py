#!/usr/bin/env python3
"""
HK Realtime Quote - 港股实时行情查询

免费获取港股实时行情，支持新浪财经和腾讯财经数据源。

Usage:
    python hk_quote.py 02513
    python hk_quote.py 02513 --source tencent
    python hk_quote.py 02513 0700 9988
    python hk_quote.py 02513 --json
"""

import argparse
import json
import re
import sys
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("请安装 requests: pip install requests")
    sys.exit(1)


def normalize_code(ticker: str) -> str:
    """标准化港股代码为5位数字"""
    code = ticker.upper().strip()
    code = code.replace('.HK', '').replace('HK', '')
    return code.zfill(5)


def get_realtime_quote_sina(ticker: str) -> Dict:
    """
    从新浪财经获取港股实时行情

    Args:
        ticker: 股票代码（如 02513, 2513, 02513.HK）

    Returns:
        Dict with realtime quote data
    """
    code = normalize_code(ticker)
    url = f"https://hq.sinajs.cn/list=rt_hk{code}"

    headers = {
        'Referer': 'https://finance.sina.com.cn',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.encoding = 'gbk'

        match = re.search(r'"([^"]*)"', resp.text)
        if not match or not match.group(1):
            return {'error': f'无数据: {ticker}', 'code': code}

        fields = match.group(1).split(',')
        if len(fields) < 18:
            return {'error': '数据格式错误', 'code': code}

        return {
            'source': 'sina',
            'code': code,
            'name_en': fields[0],
            'name': fields[1],
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
    except requests.RequestException as e:
        return {'error': f'网络错误: {e}', 'code': code}
    except Exception as e:
        return {'error': str(e), 'code': code}


def get_realtime_quote_tencent(ticker: str) -> Dict:
    """
    从腾讯财经获取港股实时行情

    Args:
        ticker: 股票代码

    Returns:
        Dict with realtime quote data
    """
    code = normalize_code(ticker)
    url = f"https://qt.gtimg.cn/q=r_hk{code}"

    try:
        resp = requests.get(url, timeout=10)
        resp.encoding = 'gbk'

        match = re.search(r'"([^"]*)"', resp.text)
        if not match or not match.group(1):
            return {'error': f'无数据: {ticker}', 'code': code}

        fields = match.group(1).split('~')
        if len(fields) < 40:
            return {'error': '数据格式错误', 'code': code}

        return {
            'source': 'tencent',
            'code': code,
            'name': fields[1],
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
    except requests.RequestException as e:
        return {'error': f'网络错误: {e}', 'code': code}
    except Exception as e:
        return {'error': str(e), 'code': code}


def get_realtime_quote(ticker: str, source: str = 'sina') -> Dict:
    """
    获取港股实时行情

    Args:
        ticker: 股票代码
        source: 数据源 ('sina' 或 'tencent')

    Returns:
        Dict with realtime quote
    """
    if source == 'tencent':
        quote = get_realtime_quote_tencent(ticker)
    else:
        quote = get_realtime_quote_sina(ticker)

    # 如果主数据源失败，尝试备用数据源
    if 'error' in quote:
        backup_source = 'tencent' if source == 'sina' else 'sina'
        if backup_source == 'tencent':
            backup_quote = get_realtime_quote_tencent(ticker)
        else:
            backup_quote = get_realtime_quote_sina(ticker)

        if 'error' not in backup_quote:
            return backup_quote

    return quote


def get_batch_quotes(tickers: List[str], source: str = 'sina') -> List[Dict]:
    """
    批量获取港股实时行情

    Args:
        tickers: 股票代码列表
        source: 数据源

    Returns:
        List of quote dicts
    """
    return [get_realtime_quote(ticker, source) for ticker in tickers]


def print_quote(quote: Dict):
    """打印行情信息"""
    if 'error' in quote:
        print(f"错误 [{quote.get('code', '?')}]: {quote['error']}")
        return

    print(f"\n{'='*50}")
    print(f"[{quote['code']}] {quote.get('name', '')} 实时行情")
    print(f"{'='*50}")
    print(f"数据源: {quote['source']}")
    print(f"时间: {quote.get('datetime', 'N/A')}")
    print()

    # 价格信息
    change_symbol = '+' if quote['change'] >= 0 else ''
    print(f"现价: {quote['price']:.3f} HKD")
    print(f"涨跌: {change_symbol}{quote['change']:.3f} ({change_symbol}{quote['change_pct']:.2f}%)")
    print()

    # OHLC
    print(f"开盘: {quote['open']:.3f}")
    print(f"最高: {quote['high']:.3f}")
    print(f"最低: {quote['low']:.3f}")
    print(f"昨收: {quote['prev_close']:.3f}")
    print()

    # 成交信息
    print(f"成交量: {quote['volume']:,}")
    if quote['amount'] > 0:
        print(f"成交额: {quote['amount']/100000000:.2f}亿")

    # 52周高低
    if quote.get('high_52w') and quote['high_52w'] > 0:
        print()
        print(f"52周高: {quote['high_52w']:.3f}")
        print(f"52周低: {quote['low_52w']:.3f}")

    print(f"{'='*50}")


def print_quotes_table(quotes: List[Dict]):
    """以表格形式打印多只股票行情"""
    print(f"\n{'代码':<8} {'名称':<10} {'现价':>10} {'涨跌幅':>10} {'成交额(亿)':>12}")
    print("-" * 55)

    for q in quotes:
        if 'error' in q:
            print(f"{q.get('code', '?'):<8} {'错误':<10} {q['error']}")
        else:
            change_str = f"{'+' if q['change_pct'] >= 0 else ''}{q['change_pct']:.2f}%"
            amount_str = f"{q['amount']/100000000:.2f}" if q['amount'] > 0 else "N/A"
            print(f"{q['code']:<8} {q.get('name', '')[:10]:<10} {q['price']:>10.3f} {change_str:>10} {amount_str:>12}")


def main():
    parser = argparse.ArgumentParser(
        description='港股实时行情查询 - 免费数据源',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python hk_quote.py 02513              # 查询智谱AI
  python hk_quote.py 0700               # 查询腾讯
  python hk_quote.py 02513 0700 9988    # 批量查询
  python hk_quote.py 02513 --json       # JSON输出
  python hk_quote.py 02513 -s tencent   # 使用腾讯数据源
        """
    )
    parser.add_argument('tickers', nargs='+', help='股票代码（支持多个）')
    parser.add_argument('-s', '--source', default='sina',
                        choices=['sina', 'tencent'],
                        help='数据源 (默认: sina)')
    parser.add_argument('-j', '--json', action='store_true',
                        help='输出JSON格式')
    parser.add_argument('-t', '--table', action='store_true',
                        help='表格格式输出（多只股票时）')

    args = parser.parse_args()

    # 获取行情
    if len(args.tickers) == 1:
        quote = get_realtime_quote(args.tickers[0], args.source)

        if args.json:
            print(json.dumps(quote, ensure_ascii=False, indent=2))
        else:
            print_quote(quote)
    else:
        quotes = get_batch_quotes(args.tickers, args.source)

        if args.json:
            print(json.dumps(quotes, ensure_ascii=False, indent=2))
        elif args.table or len(quotes) > 2:
            print_quotes_table(quotes)
        else:
            for quote in quotes:
                print_quote(quote)


if __name__ == '__main__':
    main()
