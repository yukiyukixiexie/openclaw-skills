#!/usr/bin/env python3
"""
政策/领导人行程监控工具

前置分析核心：提前发现政策催化事件

功能:
1. 领导人行程追踪 (新华网、人民网)
2. 政策文件监控 (国务院、发改委等)
3. 座谈会参与者追踪
4. 创始人动态监控

使用方法:
    python policy_monitor.py --leader          # 领导人行程
    python policy_monitor.py --policy          # 政策文件
    python policy_monitor.py --founder 唐杰    # 创始人追踪
    python policy_monitor.py --meeting         # 座谈会监控
"""

import sys
import json
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    print("需要安装依赖: pip install requests")
    sys.exit(1)


# ============== 1. 领导人行程追踪 ==============

def get_leader_activities(days: int = 7) -> List[Dict]:
    """
    从新华网获取领导人近期活动

    监控关键词: 考察、视察、调研、座谈、会见
    """
    results = []

    # 新华网领导人活动页
    urls = [
        "https://www.news.cn/politics/leaders/xijinping/index.htm",
        "https://www.news.cn/politics/leaders/likeqiang/index.htm",
    ]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }

    keywords = ['考察', '视察', '调研', '座谈', '会见', '出席', '主持']
    tech_keywords = ['科技', '人工智能', 'AI', '创新', '信创', '企业', '园区']

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            resp.encoding = 'utf-8'

            # 简单提取新闻链接和标题
            pattern = r'<a[^>]*href="([^"]*)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, resp.text)

            for href, title in matches:
                # 过滤关键词
                if any(kw in title for kw in keywords):
                    is_tech_related = any(tk in title for tk in tech_keywords)
                    results.append({
                        'title': title.strip(),
                        'url': href if href.startswith('http') else f"https://www.news.cn{href}",
                        'is_tech_related': is_tech_related,
                        'priority': 'HIGH' if is_tech_related else 'NORMAL'
                    })

        except Exception as e:
            print(f"获取 {url} 失败: {e}")

    return results[:20]  # 限制数量


def search_leader_tech_visits(query: str = None) -> List[Dict]:
    """
    搜索领导人科技企业/园区视察记录

    用于前置分析：哪些企业曾被视察？
    """
    from fetch_events import search_brave

    if query is None:
        query = "习近平 考察 科技企业 2026"

    results = search_brave(query, count=20)

    # 过滤和标注
    filtered = []
    for r in results:
        if 'error' not in r:
            title = r.get('title', '')
            # 检测是否涉及具体企业
            companies = extract_company_names(title + ' ' + r.get('description', ''))
            r['mentioned_companies'] = companies
            r['has_company'] = len(companies) > 0
            filtered.append(r)

    return filtered


def extract_company_names(text: str) -> List[str]:
    """从文本中提取公司名称"""
    # AI大模型公司列表
    known_companies = [
        '智谱', '百度', '阿里', '腾讯', '华为', '字节', '商汤',
        'DeepSeek', '月之暗面', 'Moonshot', '零一万物', '百川',
        '科大讯飞', '旷视', '云从', '依图', '寒武纪', '地平线'
    ]

    found = []
    for company in known_companies:
        if company.lower() in text.lower():
            found.append(company)

    return found


# ============== 2. 政策文件监控 ==============

def get_policy_updates(keywords: List[str] = None) -> List[Dict]:
    """
    获取近期政策文件

    来源: 国务院、发改委、工信部、科技部
    """
    if keywords is None:
        keywords = ['人工智能', '大模型', '信创', '科技创新']

    results = []

    # 政策来源
    sources = [
        {
            'name': '国务院',
            'url': 'https://www.gov.cn/zhengce/zuixin.htm',
        },
        {
            'name': '工信部',
            'url': 'https://www.miit.gov.cn/gzcy/zcwj/index.html',
        },
    ]

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for source in sources:
        try:
            resp = requests.get(source['url'], headers=headers, timeout=15)
            resp.encoding = 'utf-8'

            # 提取政策链接
            pattern = r'<a[^>]*href="([^"]*)"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, resp.text)

            for href, title in matches[:30]:
                # 关键词过滤
                if any(kw in title for kw in keywords):
                    results.append({
                        'title': title.strip(),
                        'url': href,
                        'source': source['name'],
                        'keywords_matched': [kw for kw in keywords if kw in title]
                    })

        except Exception as e:
            print(f"获取 {source['name']} 失败: {e}")

    return results


# ============== 3. 座谈会参与者追踪 ==============

def search_meeting_participants(meeting_type: str = "总书记座谈会") -> List[Dict]:
    """
    搜索重要会议参与者

    用于追踪：哪些企业家参与过高层座谈？
    """
    from fetch_events import search_brave

    queries = [
        f"{meeting_type} 企业家 参与",
        f"{meeting_type} 科技企业 负责人",
        f"习近平 座谈 企业家 名单",
    ]

    all_results = []
    for query in queries:
        results = search_brave(query, count=10)
        for r in results:
            if 'error' not in r:
                r['query'] = query
                all_results.append(r)

    return all_results


def track_founder_meetings(founder_name: str) -> List[Dict]:
    """
    追踪特定创始人的高层会议参与记录

    示例: track_founder_meetings("唐杰")
    """
    from fetch_events import search_brave

    queries = [
        f"{founder_name} 总书记 座谈",
        f"{founder_name} 习近平",
        f"{founder_name} 领导人 会见",
        f"{founder_name} 政府 交流",
    ]

    all_results = []
    for query in queries:
        results = search_brave(query, count=5)
        for r in results:
            if 'error' not in r:
                r['founder'] = founder_name
                r['query'] = query
                all_results.append(r)

    # 去重
    seen = set()
    unique = []
    for r in all_results:
        if r['url'] not in seen:
            seen.add(r['url'])
            unique.append(r)

    return unique


# ============== 4. 创始人动态监控 ==============

def get_founder_news(founder_name: str, company_name: str = None) -> List[Dict]:
    """
    获取创始人近期动态

    监控: 演讲、采访、获奖、参会等
    """
    from fetch_events import search_brave

    queries = [
        f"{founder_name} 演讲",
        f"{founder_name} 采访",
        f"{founder_name} 获奖",
        f"{founder_name} 参会",
    ]

    if company_name:
        queries.append(f"{founder_name} {company_name}")

    all_results = []
    for query in queries:
        results = search_brave(query, count=5, freshness="pm")  # 最近一个月
        for r in results:
            if 'error' not in r:
                r['query'] = query
                all_results.append(r)

    return all_results


# ============== 5. 综合监控报告 ==============

def generate_daily_report(tickers: List[str] = None) -> Dict:
    """
    生成每日监控报告

    整合: 领导人行程 + 政策文件 + 创始人动态
    """
    report = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'leader_activities': [],
        'policy_updates': [],
        'founder_news': [],
        'signals': []
    }

    print("📡 获取领导人行程...")
    report['leader_activities'] = get_leader_activities()

    print("📜 获取政策更新...")
    report['policy_updates'] = get_policy_updates()

    # 生成信号
    for activity in report['leader_activities']:
        if activity.get('is_tech_related'):
            report['signals'].append({
                'type': 'LEADER_TECH_VISIT',
                'priority': 'HIGH',
                'title': activity['title'],
                'url': activity['url'],
                'action': '关注相关AI/科技股'
            })

    for policy in report['policy_updates']:
        if len(policy.get('keywords_matched', [])) >= 2:
            report['signals'].append({
                'type': 'POLICY_UPDATE',
                'priority': 'MEDIUM',
                'title': policy['title'],
                'source': policy['source'],
                'action': '评估政策影响'
            })

    return report


# ============== CLI ==============

def print_results(results: List[Dict], title: str):
    """打印结果"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

    if not results:
        print("  无结果")
        return

    for i, r in enumerate(results[:15], 1):
        priority = r.get('priority', '')
        priority_icon = '🔴' if priority == 'HIGH' else '🟡' if priority == 'MEDIUM' else ''

        print(f"\n{i}. {priority_icon} {r.get('title', r.get('name', 'N/A'))}")

        if r.get('url'):
            print(f"   {r['url']}")

        if r.get('mentioned_companies'):
            print(f"   提及公司: {', '.join(r['mentioned_companies'])}")

        if r.get('keywords_matched'):
            print(f"   匹配关键词: {', '.join(r['keywords_matched'])}")

    print('\n' + '='*60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='政策/领导人行程监控工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python policy_monitor.py --leader              # 领导人行程
  python policy_monitor.py --leader-search       # 搜索领导人科技视察
  python policy_monitor.py --policy              # 政策文件
  python policy_monitor.py --founder 唐杰        # 创始人追踪
  python policy_monitor.py --meeting             # 座谈会参与者
  python policy_monitor.py --daily               # 每日综合报告
        """
    )

    parser.add_argument('--leader', action='store_true', help='领导人行程')
    parser.add_argument('--leader-search', action='store_true', help='搜索领导人科技视察')
    parser.add_argument('--policy', action='store_true', help='政策文件')
    parser.add_argument('--founder', type=str, help='创始人动态追踪')
    parser.add_argument('--meeting', action='store_true', help='座谈会参与者')
    parser.add_argument('--daily', action='store_true', help='每日综合报告')
    parser.add_argument('--json', action='store_true', help='JSON输出')

    args = parser.parse_args()

    if args.leader:
        results = get_leader_activities()
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print_results(results, "领导人近期活动")

    elif args.leader_search:
        results = search_leader_tech_visits()
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print_results(results, "领导人科技视察记录")

    elif args.policy:
        results = get_policy_updates()
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print_results(results, "近期政策文件")

    elif args.founder:
        print(f"🔍 追踪创始人: {args.founder}")

        # 高层会议参与
        meetings = track_founder_meetings(args.founder)
        print_results(meetings, f"{args.founder} 高层会议参与记录")

        # 近期动态
        news = get_founder_news(args.founder)
        print_results(news, f"{args.founder} 近期动态")

    elif args.meeting:
        results = search_meeting_participants()
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            print_results(results, "座谈会参与者记录")

    elif args.daily:
        report = generate_daily_report()

        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
        else:
            print(f"\n📊 每日监控报告 ({report['date']})")
            print('='*60)

            if report['signals']:
                print("\n⚡ 信号提醒:")
                for signal in report['signals']:
                    print(f"  [{signal['priority']}] {signal['type']}")
                    print(f"    {signal['title']}")
                    print(f"    建议: {signal['action']}")

            print_results(report['leader_activities'], "领导人活动")
            print_results(report['policy_updates'], "政策更新")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
