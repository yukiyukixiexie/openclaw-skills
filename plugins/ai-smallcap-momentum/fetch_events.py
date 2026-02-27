#!/usr/bin/env python3
"""
事件/新闻数据获取工具

支持多个数据源：
1. 港交所披露易（公司公告）
2. 新浪财经（新闻）
3. 东方财富（新闻+研报）
4. Brave Search API（全网搜索）

使用方法:
    python fetch_events.py 02513 --start 2026-01-01
    python fetch_events.py 02513 --source hkex
    python fetch_events.py --search "智谱AI 领导人"
"""

import sys
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

try:
    import requests
except ImportError:
    print("需要安装依赖: pip install requests")
    sys.exit(1)


# ============== 1. 港交所披露易（公司公告）==============

def get_hkex_announcements(
    stock_code: str,
    start_date: str = None,
    end_date: str = None,
    lang: str = "ZH"
) -> List[Dict]:
    """
    从港交所披露易获取公司公告

    API: https://www1.hkexnews.hk/search/titlesearch.xhtml

    Args:
        stock_code: 股票代码（如 02513）
        start_date: 开始日期 YYYY-MM-DD
        end_date: 结束日期 YYYY-MM-DD
        lang: 语言 ZH/EN

    Returns:
        公告列表
    """
    stock_code = stock_code.replace('.HK', '').zfill(5)

    url = "https://www1.hkexnews.hk/search/titlesearch.xhtml"

    # 构建搜索参数
    params = {
        "lang": lang,
        "category": 0,  # 所有类别
        "market": "SEHK",
        "searchType": 0,
        "t1code": 40000,  # 上市公司公告
        "t2Gcode": -2,
        "t2code": -2,
        "stockId": stock_code,
        "from": start_date.replace('-', '') if start_date else "",
        "to": end_date.replace('-', '') if end_date else "",
        "MB-Ede": "",
        "mession": ""
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Referer": "https://www1.hkexnews.hk/search/titlesearch.xhtml"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)

        # 港交所返回HTML，需要解析
        # 这里简化处理，返回原始响应供后续解析
        announcements = []

        # 尝试提取公告链接
        pattern = r'<a[^>]*href="([^"]*)"[^>]*>([^<]*)</a>'
        matches = re.findall(pattern, resp.text)

        for href, title in matches:
            if 'listedco' in href or '.pdf' in href:
                announcements.append({
                    'title': title.strip(),
                    'url': href if href.startswith('http') else f"https://www1.hkexnews.hk{href}",
                    'source': 'hkex'
                })

        return announcements

    except Exception as e:
        return [{"error": f"港交所API请求失败: {e}"}]


# ============== 2. 东方财富新闻 ==============

def get_eastmoney_news(
    stock_code: str,
    page: int = 1,
    page_size: int = 20
) -> List[Dict]:
    """
    从东方财富获取股票新闻

    API: https://search-api-web.eastmoney.com/search/jsonp

    Args:
        stock_code: 股票代码
        page: 页码
        page_size: 每页数量

    Returns:
        新闻列表
    """
    stock_code = stock_code.replace('.HK', '').zfill(5)

    # 东方财富搜索API
    url = "https://search-api-web.eastmoney.com/search/jsonp"

    params = {
        "cb": "jQuery_callback",
        "param": json.dumps({
            "uid": "",
            "keyword": stock_code,
            "type": ["cmsArticleWebOld"],  # 新闻文章
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "time",  # 按时间排序
                    "pageIndex": page,
                    "pageSize": page_size,
                    "preTag": "",
                    "postTag": ""
                }
            }
        })
    }

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://so.eastmoney.com/"
    }

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=15)

        # 解析JSONP响应
        json_str = re.search(r'jQuery_callback\((.*)\)', resp.text)
        if not json_str:
            return [{"error": "无法解析东方财富响应"}]

        data = json.loads(json_str.group(1))
        articles = data.get('result', {}).get('cmsArticleWebOld', [])

        news_list = []
        for article in articles:
            news_list.append({
                'title': article.get('title', ''),
                'date': article.get('date', ''),
                'url': article.get('url', ''),
                'source': 'eastmoney',
                'summary': article.get('content', '')[:200] if article.get('content') else ''
            })

        return news_list

    except Exception as e:
        return [{"error": f"东方财富API请求失败: {e}"}]


# ============== 3. 新浪财经港股新闻 ==============

def get_sina_hk_news(
    stock_code: str,
    page: int = 1
) -> List[Dict]:
    """
    从新浪财经获取港股新闻

    Args:
        stock_code: 股票代码
        page: 页码

    Returns:
        新闻列表
    """
    stock_code = stock_code.replace('.HK', '').zfill(5)

    url = f"https://vip.stock.finance.sina.com.cn/corp/go.php/vCB_AllNewsStock/symbol/hk{stock_code}/type/news.phtml"

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://finance.sina.com.cn"
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.encoding = 'gbk'

        news_list = []

        # 简单解析HTML提取新闻链接
        # 格式: <a href="...">新闻标题</a> <span>日期</span>
        pattern = r'<a[^>]*href="([^"]*)"[^>]*target="_blank"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, resp.text)

        for href, title in matches[:20]:  # 限制20条
            if 'finance.sina' in href or 'stock.sina' in href:
                news_list.append({
                    'title': title.strip(),
                    'url': href,
                    'source': 'sina'
                })

        return news_list

    except Exception as e:
        return [{"error": f"新浪财经请求失败: {e}"}]


# ============== 4. Brave Search API ==============

def search_brave(
    query: str,
    count: int = 10,
    freshness: str = None
) -> List[Dict]:
    """
    使用Brave Search API搜索

    需要设置环境变量 BRAVE_API_KEY

    Args:
        query: 搜索关键词
        count: 结果数量
        freshness: 时间范围 (pd=past day, pw=past week, pm=past month)

    Returns:
        搜索结果列表
    """
    api_key = os.environ.get('BRAVE_API_KEY')

    if not api_key:
        return [{"error": "未设置 BRAVE_API_KEY 环境变量"}]

    url = "https://api.search.brave.com/res/v1/web/search"

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key
    }

    params = {
        "q": query,
        "count": count,
        "search_lang": "zh-hans",
        "country": "cn"
    }

    if freshness:
        params["freshness"] = freshness

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)

        if resp.status_code == 401:
            return [{"error": "Brave API Key 无效"}]

        data = resp.json()
        results = data.get('web', {}).get('results', [])

        search_results = []
        for r in results:
            search_results.append({
                'title': r.get('title', ''),
                'url': r.get('url', ''),
                'description': r.get('description', ''),
                'date': r.get('age', ''),
                'source': 'brave'
            })

        return search_results

    except Exception as e:
        return [{"error": f"Brave Search请求失败: {e}"}]


# ============== 5. 综合事件搜索 ==============

def search_stock_events(
    stock_code: str,
    keywords: List[str] = None,
    start_date: str = None,
    sources: List[str] = None
) -> Dict:
    """
    综合搜索股票相关事件

    Args:
        stock_code: 股票代码
        keywords: 额外关键词（如 "领导人", "模型发布"）
        start_date: 开始日期
        sources: 数据源列表

    Returns:
        综合搜索结果
    """
    if sources is None:
        sources = ['eastmoney', 'sina', 'brave']

    if keywords is None:
        keywords = []

    results = {
        'stock_code': stock_code,
        'search_time': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'announcements': [],
        'news': [],
        'search_results': []
    }

    stock_code_clean = stock_code.replace('.HK', '').zfill(5)

    # 1. 获取公司公告
    print(f"📢 获取港交所公告...")
    announcements = get_hkex_announcements(stock_code_clean, start_date)
    if announcements and 'error' not in announcements[0]:
        results['announcements'] = announcements[:10]

    # 2. 获取东方财富新闻
    if 'eastmoney' in sources:
        print(f"📰 获取东方财富新闻...")
        em_news = get_eastmoney_news(stock_code_clean)
        if em_news and 'error' not in em_news[0]:
            results['news'].extend(em_news)

    # 3. 获取新浪新闻
    if 'sina' in sources:
        print(f"📰 获取新浪财经新闻...")
        sina_news = get_sina_hk_news(stock_code_clean)
        if sina_news and 'error' not in sina_news[0]:
            results['news'].extend(sina_news)

    # 4. Brave搜索关键事件
    if 'brave' in sources and os.environ.get('BRAVE_API_KEY'):
        print(f"🔍 Brave搜索关键事件...")

        # 搜索股票名称+关键词
        base_keywords = ["智谱AI", "02513"]
        event_keywords = ["领导人", "模型发布", "政策", "合作", "融资"]

        for kw in event_keywords + keywords:
            query = f"{base_keywords[0]} {kw}"
            brave_results = search_brave(query, count=5, freshness="pm")
            if brave_results and 'error' not in brave_results[0]:
                for r in brave_results:
                    r['keyword'] = kw
                results['search_results'].extend(brave_results)

    # 去重
    seen_urls = set()
    unique_news = []
    for news in results['news']:
        if news.get('url') and news['url'] not in seen_urls:
            seen_urls.add(news['url'])
            unique_news.append(news)
    results['news'] = unique_news

    return results


# ============== 事件提取与分类 ==============

def extract_event_timeline(results: Dict) -> List[Dict]:
    """
    从搜索结果中提取事件时间线

    Args:
        results: search_stock_events 的返回结果

    Returns:
        事件时间线
    """
    events = []

    # 定义事件关键词分类
    event_categories = {
        '模型发布': ['发布', '模型', 'GLM', '版本', '升级', '开源'],
        '领导人会见': ['领导', '总理', '主席', '会见', '调研', '视察'],
        '政策利好': ['政策', '支持', '补贴', '战略', '规划', '指导意见'],
        '合作签约': ['合作', '签约', '战略', '协议', '携手'],
        '融资动态': ['融资', '投资', '估值', '入股'],
        '产品发布': ['产品', '应用', '上线', '发布会'],
        '业绩相关': ['业绩', '营收', '利润', '财报', '盈利']
    }

    # 处理新闻
    for news in results.get('news', []):
        title = news.get('title', '')
        category = '其他'

        for cat, keywords in event_categories.items():
            if any(kw in title for kw in keywords):
                category = cat
                break

        events.append({
            'title': title,
            'category': category,
            'date': news.get('date', ''),
            'url': news.get('url', ''),
            'source': news.get('source', '')
        })

    # 处理搜索结果
    for result in results.get('search_results', []):
        title = result.get('title', '')
        category = result.get('keyword', '其他')

        events.append({
            'title': title,
            'category': category,
            'date': result.get('date', ''),
            'url': result.get('url', ''),
            'source': 'brave'
        })

    return events


# ============== CLI ==============

def print_results(results: Dict):
    """打印搜索结果"""
    print("\n" + "="*70)
    print(f"  {results['stock_code']} 事件/新闻搜索结果")
    print("="*70)
    print(f"搜索时间: {results['search_time']}")

    # 公告
    if results['announcements']:
        print(f"\n📢 港交所公告 ({len(results['announcements'])}条):")
        print("-"*70)
        for ann in results['announcements'][:5]:
            print(f"  • {ann['title'][:50]}...")

    # 新闻
    if results['news']:
        print(f"\n📰 相关新闻 ({len(results['news'])}条):")
        print("-"*70)
        for news in results['news'][:10]:
            date_str = f"[{news['date']}]" if news.get('date') else ""
            print(f"  {date_str} {news['title'][:50]}...")
            print(f"    └─ {news['url']}")

    # 搜索结果
    if results['search_results']:
        print(f"\n🔍 关键事件搜索 ({len(results['search_results'])}条):")
        print("-"*70)
        for r in results['search_results'][:10]:
            kw = f"[{r.get('keyword', '')}]" if r.get('keyword') else ""
            print(f"  {kw} {r['title'][:45]}...")
            print(f"    └─ {r['url']}")

    print("\n" + "="*70)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='股票事件/新闻数据获取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
数据源说明:
  hkex       港交所披露易（公司公告）
  eastmoney  东方财富（新闻+研报）
  sina       新浪财经（港股新闻）
  brave      Brave Search API（全网搜索，需设置BRAVE_API_KEY）

示例:
  python fetch_events.py 02513
  python fetch_events.py 02513 --start 2026-01-01
  python fetch_events.py --search "智谱AI 领导人会见"

设置Brave API:
  export BRAVE_API_KEY=your_api_key_here
        """
    )

    parser.add_argument('ticker', nargs='?', help='股票代码 (如 02513)')
    parser.add_argument('--start', '-s', help='开始日期 YYYY-MM-DD')
    parser.add_argument('--search', help='直接搜索关键词')
    parser.add_argument('--source', nargs='+',
                        choices=['hkex', 'eastmoney', 'sina', 'brave'],
                        help='指定数据源')
    parser.add_argument('--json', action='store_true', help='输出JSON格式')

    args = parser.parse_args()

    # 直接搜索模式
    if args.search:
        if not os.environ.get('BRAVE_API_KEY'):
            print("❌ 直接搜索需要设置 BRAVE_API_KEY")
            print("   export BRAVE_API_KEY=your_api_key_here")
            sys.exit(1)

        print(f"🔍 搜索: {args.search}")
        results = search_brave(args.search, count=20)

        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for r in results:
                if 'error' in r:
                    print(f"❌ {r['error']}")
                else:
                    print(f"\n• {r['title']}")
                    print(f"  {r['description'][:100]}...")
                    print(f"  {r['url']}")
        return

    # 股票事件搜索
    if not args.ticker:
        parser.print_help()
        sys.exit(1)

    sources = args.source if args.source else ['eastmoney', 'sina']

    # 如果设置了Brave API Key，自动加入
    if os.environ.get('BRAVE_API_KEY') and 'brave' not in sources:
        sources.append('brave')

    results = search_stock_events(
        args.ticker,
        start_date=args.start,
        sources=sources
    )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
