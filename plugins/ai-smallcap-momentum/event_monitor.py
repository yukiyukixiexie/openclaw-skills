#!/usr/bin/env python3
"""
事件催化监控系统 (Event Catalyst Monitor)

统一框架：监控所有类型的催化事件，实现前置分析

事件分类:
1. 政策背书类 (POLICY)     - 领导人视察、政策文件、座谈会
2. 产品发布类 (PRODUCT)    - 模型发布、版本更新、新产品
3. 商业验证类 (BUSINESS)   - 提价、大客户、合同、营收
4. 资金确认类 (CAPITAL)    - 机构入场、增持、融资
5. 行业共振类 (INDUSTRY)   - 竞品动态、行业政策、板块联动
6. 舆情扩散类 (SENTIMENT)  - 媒体报道、社交热度、KOL背书

使用方法:
    python event_monitor.py --ticker 02513 --all        # 全量监控
    python event_monitor.py --ticker 02513 --policy     # 政策类
    python event_monitor.py --ticker 02513 --product    # 产品类
    python event_monitor.py --ticker 02513 --business   # 商业类
    python event_monitor.py --scan --theme AI           # 主题扫描
"""

import sys
import json
import re
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import requests
except ImportError:
    print("需要安装依赖: pip install requests")
    sys.exit(1)


# ============== 事件分类定义 ==============

class EventType(Enum):
    POLICY = "政策背书"      # 领导人、政策文件
    PRODUCT = "产品发布"     # 模型、版本、新产品
    BUSINESS = "商业验证"    # 提价、合同、客户
    CAPITAL = "资金确认"     # 机构、融资、增持
    INDUSTRY = "行业共振"    # 竞品、板块、政策
    SENTIMENT = "舆情扩散"   # 媒体、社交、KOL


@dataclass
class CatalystEvent:
    """催化事件数据结构"""
    event_type: str          # 事件类型
    title: str               # 事件标题
    date: str                # 日期
    source: str              # 来源
    url: str                 # 链接
    priority: str            # 优先级 HIGH/MEDIUM/LOW
    signal_strength: int     # 信号强度 1-5
    ticker: str = ""         # 关联股票
    keywords: List[str] = None  # 匹配关键词
    raw_data: Dict = None    # 原始数据

    def to_dict(self):
        return asdict(self)


# ============== 事件信号强度配置 ==============

EVENT_CONFIG = {
    EventType.POLICY: {
        'weight': 5,  # 最高权重
        'keywords': {
            'high': ['习近平', '总书记', '国务院', '座谈会', '视察', '调研'],
            'medium': ['发改委', '工信部', '科技部', '政策', '规划', '试点'],
            'low': ['补贴', '标准', '指导意见']
        },
        'sources': [
            {'name': '新华网', 'url': 'news.cn', 'priority': 'HIGH'},
            {'name': '人民网', 'url': 'people.com.cn', 'priority': 'HIGH'},
            {'name': '国务院', 'url': 'gov.cn', 'priority': 'HIGH'},
        ]
    },
    EventType.PRODUCT: {
        'weight': 4,
        'keywords': {
            'high': ['发布', '重磅', '首发', '全球首个', '突破'],
            'medium': ['升级', '更新', '版本', '开源', '论文'],
            'low': ['优化', '改进', '修复']
        },
        'sources': [
            {'name': '官方公告', 'priority': 'HIGH'},
            {'name': 'Hugging Face', 'url': 'huggingface.co', 'priority': 'HIGH'},
            {'name': 'arXiv', 'url': 'arxiv.org', 'priority': 'MEDIUM'},
        ]
    },
    EventType.BUSINESS: {
        'weight': 4,
        'keywords': {
            'high': ['提价', '涨价', '大单', '战略合作', '独家'],
            'medium': ['客户', '合同', '营收', '盈利', '订单'],
            'low': ['合作', '签约', '入围']
        },
        'sources': [
            {'name': '港交所公告', 'url': 'hkexnews.hk', 'priority': 'HIGH'},
            {'name': '财经新闻', 'priority': 'MEDIUM'},
        ]
    },
    EventType.CAPITAL: {
        'weight': 3,
        'keywords': {
            'high': ['增持', '举牌', '战略入股', '融资'],
            'medium': ['基石', '机构', '南向资金', '北向资金'],
            'low': ['持仓', '调仓']
        },
        'sources': [
            {'name': 'CCASS', 'priority': 'HIGH'},
            {'name': '港交所', 'priority': 'HIGH'},
        ]
    },
    EventType.INDUSTRY: {
        'weight': 3,
        'keywords': {
            'high': ['OpenAI', 'DeepSeek', '行业突破', '颠覆'],
            'medium': ['竞品', '对标', '板块', '概念'],
            'low': ['行业', '趋势']
        },
        'sources': [
            {'name': '36氪', 'url': '36kr.com', 'priority': 'MEDIUM'},
            {'name': '虎嗅', 'url': 'huxiu.com', 'priority': 'MEDIUM'},
        ]
    },
    EventType.SENTIMENT: {
        'weight': 2,
        'keywords': {
            'high': ['刷屏', '热搜', '爆火', '疯狂'],
            'medium': ['关注', '讨论', '报道'],
            'low': ['提及', '评论']
        },
        'sources': [
            {'name': '微博热搜', 'priority': 'MEDIUM'},
            {'name': '雪球', 'url': 'xueqiu.com', 'priority': 'MEDIUM'},
        ]
    }
}


# ============== 公司/主题配置 ==============

COMPANY_CONFIG = {
    '02513': {
        'name': '智谱AI',
        'aliases': ['智谱', 'Zhipu', 'GLM'],
        'founders': ['唐杰', '张鹏'],
        'theme': ['AI', '大模型', '国产替代'],
        'competitors': ['百度', 'DeepSeek', '阿里', '字节']
    },
    '09888': {
        'name': '百度',
        'aliases': ['百度', 'Baidu', '文心一言'],
        'founders': ['李彦宏'],
        'theme': ['AI', '大模型', '搜索'],
        'competitors': ['智谱', 'DeepSeek', '阿里']
    },
    # 可扩展更多公司
}

THEME_CONFIG = {
    'AI': {
        'keywords': ['人工智能', 'AI', '大模型', 'LLM', 'AGI', 'GPT', 'Claude'],
        'tickers': ['02513', '09888', '09988', '00020'],
        'policy_keywords': ['人工智能', '大模型', '算力', '智能计算'],
    },
    '信创': {
        'keywords': ['信创', '国产替代', '自主可控', '安可'],
        'tickers': ['02513'],
        'policy_keywords': ['信息技术应用创新', '国产化', '自主可控'],
    },
    '机器人': {
        'keywords': ['机器人', '具身智能', '人形机器人'],
        'tickers': [],
        'policy_keywords': ['机器人', '智能制造'],
    }
}


# ============== 通用搜索函数 ==============

# 可信财经来源白名单
TRUSTED_SOURCES = [
    'sina.com', 'qq.com', '36kr.com', 'huxiu.com', 'cls.cn', 'jin10.com',
    'eastmoney.com', 'xueqiu.com', 'caixin.com', 'yicai.com', 'thepaper.cn',
    'news.cn', 'people.com.cn', 'gov.cn', 'hkexnews.hk', 'bilibili.com',
    'guancha.cn', 'ifeng.com', 'jiemian.com', 'leiphone.com', 'geekpark.net'
]

# 排除的噪音来源
BLOCKED_SOURCES = [
    'youtube.com', 'bannedbook.org', 'epochtimes.com', 'ntdtv.com',
    'rfa.org', 'voachinese.com', 'facebook.com', 'twitter.com'
]

# 政策搜索时排除的关键词（与投资无关的政治新闻）
POLITICAL_NOISE_KEYWORDS = ['解放军', '张又侠', '清洗', '偏执狂', '腐败', '军队']


def brave_search(query: str, count: int = 10, freshness: str = None, filter_sources: bool = True) -> List[Dict]:
    """Brave Search API"""
    api_key = os.environ.get('BRAVE_API_KEY')
    if not api_key:
        return [{"error": "未设置 BRAVE_API_KEY"}]

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"Accept": "application/json", "X-Subscription-Token": api_key}
    params = {"q": query, "count": count * 2, "search_lang": "zh-hans"}  # 多取一些用于过滤
    if freshness:
        params["freshness"] = freshness

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        data = resp.json()
        results = data.get('web', {}).get('results', [])

        # 过滤噪音来源
        if filter_sources:
            filtered = []
            for r in results:
                url_str = r.get('url', '')
                title = r.get('title', '')
                # 排除黑名单域名
                if any(blocked in url_str for blocked in BLOCKED_SOURCES):
                    continue
                # 排除政治噪音关键词
                if any(noise in title for noise in POLITICAL_NOISE_KEYWORDS):
                    continue
                filtered.append(r)
            return filtered[:count]

        return results[:count]
    except Exception as e:
        return [{"error": str(e)}]


# ============== 各类事件监控器 ==============

class EventMonitor:
    """事件监控器基类"""

    def __init__(self, ticker: str = None, theme: str = None):
        self.ticker = ticker
        self.theme = theme
        self.company_info = COMPANY_CONFIG.get(ticker, {})
        self.theme_info = THEME_CONFIG.get(theme, {})

    def search(self, event_type: EventType) -> List[CatalystEvent]:
        """搜索特定类型事件"""
        raise NotImplementedError

    def calculate_signal_strength(self, text: str, event_type: EventType) -> int:
        """计算信号强度 1-5"""
        config = EVENT_CONFIG[event_type]
        keywords = config['keywords']

        score = 0
        matched = []

        for kw in keywords.get('high', []):
            if kw in text:
                score += 3
                matched.append(kw)

        for kw in keywords.get('medium', []):
            if kw in text:
                score += 2
                matched.append(kw)

        for kw in keywords.get('low', []):
            if kw in text:
                score += 1
                matched.append(kw)

        # 归一化到1-5
        return min(5, max(1, score // 2)), matched


class PolicyMonitor(EventMonitor):
    """政策背书类事件监控"""

    def search(self) -> List[CatalystEvent]:
        events = []

        # 1. 领导人视察相关
        if self.company_info:
            company_name = self.company_info.get('name', '')
            founders = self.company_info.get('founders', [])

            # 搜索领导人+公司+动作词
            queries = [
                f"习近平 {company_name} 考察",
                f"习近平 {company_name} 视察",
                f"总书记 {company_name}",
            ]
            for query in queries:
                results = brave_search(query, count=5, freshness="pm")
                events.extend(self._parse_results(results, 'leader_company'))

            # 搜索领导人+创始人
            for founder in founders:
                query = f"习近平 {founder} 座谈"
                results = brave_search(query, count=3, freshness="pm")
                events.extend(self._parse_results(results, 'leader_founder'))

        # 2. 主题政策相关
        if self.theme_info:
            policy_keywords = self.theme_info.get('policy_keywords', [])
            for kw in policy_keywords[:2]:
                query = f"国务院 {kw} 政策 2026"
                results = brave_search(query, count=5, freshness="pm")
                events.extend(self._parse_results(results, 'policy'))

        return events

    def _parse_results(self, results: List[Dict], sub_type: str) -> List[CatalystEvent]:
        events = []
        for r in results:
            if 'error' in r:
                continue

            text = r.get('title', '') + ' ' + r.get('description', '')
            strength, keywords = self.calculate_signal_strength(text, EventType.POLICY)

            # 判断优先级
            if '习近平' in text or '总书记' in text:
                priority = 'HIGH'
                strength = 5
            elif '国务院' in text or '发改委' in text:
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            events.append(CatalystEvent(
                event_type=EventType.POLICY.value,
                title=r.get('title', ''),
                date=r.get('age', ''),
                source=r.get('url', '').split('/')[2] if r.get('url') else '',
                url=r.get('url', ''),
                priority=priority,
                signal_strength=strength,
                ticker=self.ticker or '',
                keywords=keywords
            ))

        return events


class ProductMonitor(EventMonitor):
    """产品发布类事件监控"""

    def search(self) -> List[CatalystEvent]:
        events = []

        if self.company_info:
            company_name = self.company_info.get('name', '')
            aliases = self.company_info.get('aliases', [])

            # 搜索产品发布
            for name in [company_name] + aliases[:2]:
                query = f"{name} 发布 模型 2026"
                results = brave_search(query, count=5, freshness="pm")
                events.extend(self._parse_results(results))

                query = f"{name} 新产品 发布"
                results = brave_search(query, count=5, freshness="pm")
                events.extend(self._parse_results(results))

        # Hugging Face热门
        if self.theme == 'AI':
            query = "Hugging Face trending model China 2026"
            results = brave_search(query, count=5)
            events.extend(self._parse_results(results))

        return events

    def _parse_results(self, results: List[Dict]) -> List[CatalystEvent]:
        events = []
        for r in results:
            if 'error' in r:
                continue

            text = r.get('title', '') + ' ' + r.get('description', '')
            strength, keywords = self.calculate_signal_strength(text, EventType.PRODUCT)

            if any(kw in text for kw in ['重磅', '突破', '首发', '全球首个']):
                priority = 'HIGH'
            elif any(kw in text for kw in ['发布', '升级', '开源']):
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            events.append(CatalystEvent(
                event_type=EventType.PRODUCT.value,
                title=r.get('title', ''),
                date=r.get('age', ''),
                source=r.get('url', '').split('/')[2] if r.get('url') else '',
                url=r.get('url', ''),
                priority=priority,
                signal_strength=strength,
                ticker=self.ticker or '',
                keywords=keywords
            ))

        return events


class BusinessMonitor(EventMonitor):
    """商业验证类事件监控"""

    def search(self) -> List[CatalystEvent]:
        events = []

        if self.company_info:
            company_name = self.company_info.get('name', '')

            # 商业动态搜索
            queries = [
                f"{company_name} 提价",
                f"{company_name} 大客户 签约",
                f"{company_name} 战略合作",
                f"{company_name} 营收 增长",
                f"{company_name} 盈利",
            ]

            for query in queries:
                results = brave_search(query, count=3, freshness="pm")
                events.extend(self._parse_results(results))

        return events

    def _parse_results(self, results: List[Dict]) -> List[CatalystEvent]:
        events = []
        for r in results:
            if 'error' in r:
                continue

            text = r.get('title', '') + ' ' + r.get('description', '')
            strength, keywords = self.calculate_signal_strength(text, EventType.BUSINESS)

            if any(kw in text for kw in ['提价', '涨价', '大单', '独家']):
                priority = 'HIGH'
            elif any(kw in text for kw in ['合同', '客户', '营收']):
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            events.append(CatalystEvent(
                event_type=EventType.BUSINESS.value,
                title=r.get('title', ''),
                date=r.get('age', ''),
                source=r.get('url', '').split('/')[2] if r.get('url') else '',
                url=r.get('url', ''),
                priority=priority,
                signal_strength=strength,
                ticker=self.ticker or '',
                keywords=keywords
            ))

        return events


class CapitalMonitor(EventMonitor):
    """资金确认类事件监控"""

    def search(self) -> List[CatalystEvent]:
        events = []

        if self.company_info:
            company_name = self.company_info.get('name', '')

            queries = [
                f"{company_name} 增持",
                f"{company_name} 机构 买入",
                f"{company_name} 南向资金",
                f"{company_name} 融资",
                f"{company_name} 战略投资",
            ]

            for query in queries:
                results = brave_search(query, count=3, freshness="pm")
                events.extend(self._parse_results(results))

        return events

    def _parse_results(self, results: List[Dict]) -> List[CatalystEvent]:
        events = []
        for r in results:
            if 'error' in r:
                continue

            text = r.get('title', '') + ' ' + r.get('description', '')
            strength, keywords = self.calculate_signal_strength(text, EventType.CAPITAL)

            if any(kw in text for kw in ['增持', '举牌', '战略入股']):
                priority = 'HIGH'
            elif any(kw in text for kw in ['机构', '基石', '融资']):
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            events.append(CatalystEvent(
                event_type=EventType.CAPITAL.value,
                title=r.get('title', ''),
                date=r.get('age', ''),
                source=r.get('url', '').split('/')[2] if r.get('url') else '',
                url=r.get('url', ''),
                priority=priority,
                signal_strength=strength,
                ticker=self.ticker or '',
                keywords=keywords
            ))

        return events


class IndustryMonitor(EventMonitor):
    """行业共振类事件监控"""

    def search(self) -> List[CatalystEvent]:
        events = []

        # 竞品动态
        if self.company_info:
            competitors = self.company_info.get('competitors', [])
            for comp in competitors[:3]:
                query = f"{comp} 发布 重大 2026"
                results = brave_search(query, count=3, freshness="pw")
                events.extend(self._parse_results(results))

        # 行业政策
        if self.theme_info:
            theme_keywords = self.theme_info.get('keywords', [])
            for kw in theme_keywords[:2]:
                query = f"{kw} 行业 政策 利好 2026"
                results = brave_search(query, count=3, freshness="pm")
                events.extend(self._parse_results(results))

        return events

    def _parse_results(self, results: List[Dict]) -> List[CatalystEvent]:
        events = []
        for r in results:
            if 'error' in r:
                continue

            text = r.get('title', '') + ' ' + r.get('description', '')
            strength, keywords = self.calculate_signal_strength(text, EventType.INDUSTRY)

            if any(kw in text for kw in ['突破', '颠覆', 'OpenAI', 'DeepSeek']):
                priority = 'HIGH'
            elif any(kw in text for kw in ['竞品', '对标', '板块']):
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            events.append(CatalystEvent(
                event_type=EventType.INDUSTRY.value,
                title=r.get('title', ''),
                date=r.get('age', ''),
                source=r.get('url', '').split('/')[2] if r.get('url') else '',
                url=r.get('url', ''),
                priority=priority,
                signal_strength=strength,
                ticker=self.ticker or '',
                keywords=keywords
            ))

        return events


class SentimentMonitor(EventMonitor):
    """舆情扩散类事件监控"""

    def search(self) -> List[CatalystEvent]:
        events = []

        if self.company_info:
            company_name = self.company_info.get('name', '')

            queries = [
                f"{company_name} 热搜",
                f"{company_name} 刷屏",
                f"{company_name} 爆火",
                f"雪球 {company_name} 讨论",
            ]

            for query in queries:
                results = brave_search(query, count=3, freshness="pw")
                events.extend(self._parse_results(results))

        return events

    def _parse_results(self, results: List[Dict]) -> List[CatalystEvent]:
        events = []
        for r in results:
            if 'error' in r:
                continue

            text = r.get('title', '') + ' ' + r.get('description', '')
            strength, keywords = self.calculate_signal_strength(text, EventType.SENTIMENT)

            if any(kw in text for kw in ['刷屏', '热搜', '爆火']):
                priority = 'HIGH'
            elif any(kw in text for kw in ['关注', '讨论']):
                priority = 'MEDIUM'
            else:
                priority = 'LOW'

            events.append(CatalystEvent(
                event_type=EventType.SENTIMENT.value,
                title=r.get('title', ''),
                date=r.get('age', ''),
                source=r.get('url', '').split('/')[2] if r.get('url') else '',
                url=r.get('url', ''),
                priority=priority,
                signal_strength=strength,
                ticker=self.ticker or '',
                keywords=keywords
            ))

        return events


# ============== 综合监控 ==============

class CatalystMonitor:
    """综合催化事件监控器"""

    def __init__(self, ticker: str = None, theme: str = None):
        self.ticker = ticker
        self.theme = theme

        self.monitors = {
            EventType.POLICY: PolicyMonitor(ticker, theme),
            EventType.PRODUCT: ProductMonitor(ticker, theme),
            EventType.BUSINESS: BusinessMonitor(ticker, theme),
            EventType.CAPITAL: CapitalMonitor(ticker, theme),
            EventType.INDUSTRY: IndustryMonitor(ticker, theme),
            EventType.SENTIMENT: SentimentMonitor(ticker, theme),
        }

    def scan_all(self) -> Dict[str, List[CatalystEvent]]:
        """扫描所有类型事件"""
        results = {}

        for event_type, monitor in self.monitors.items():
            print(f"📡 扫描 {event_type.value}...")
            events = monitor.search()
            results[event_type.value] = events

        return results

    def scan_type(self, event_type: EventType) -> List[CatalystEvent]:
        """扫描特定类型事件"""
        monitor = self.monitors.get(event_type)
        if monitor:
            return monitor.search()
        return []

    def generate_signals(self, events: Dict[str, List[CatalystEvent]]) -> List[Dict]:
        """生成交易信号"""
        signals = []

        for event_type, event_list in events.items():
            for event in event_list:
                if event.priority == 'HIGH' or event.signal_strength >= 4:
                    signals.append({
                        'type': event_type,
                        'priority': event.priority,
                        'strength': event.signal_strength,
                        'title': event.title,
                        'url': event.url,
                        'action': self._suggest_action(event)
                    })

        # 按信号强度排序
        signals.sort(key=lambda x: x['strength'], reverse=True)
        return signals

    def _suggest_action(self, event: CatalystEvent) -> str:
        """根据事件生成操作建议"""
        if event.event_type == EventType.POLICY.value:
            if event.signal_strength >= 5:
                return "🔥 重大政策信号，评估立即入场"
            return "关注政策后续落地"

        elif event.event_type == EventType.PRODUCT.value:
            if event.signal_strength >= 4:
                return "产品重大突破，评估入场时机"
            return "跟踪产品反馈"

        elif event.event_type == EventType.BUSINESS.value:
            if '提价' in (event.keywords or []):
                return "🔥 提价信号，商业验证强"
            return "关注商业进展"

        elif event.event_type == EventType.CAPITAL.value:
            return "资金入场信号，跟踪持续性"

        elif event.event_type == EventType.INDUSTRY.value:
            return "行业催化，评估联动效应"

        elif event.event_type == EventType.SENTIMENT.value:
            if event.signal_strength >= 4:
                return "⚠️ 舆情过热，警惕见顶"
            return "舆情升温，保持关注"

        return "持续监控"


# ============== CLI ==============

def print_events(events: List[CatalystEvent], title: str):
    """打印事件列表"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

    if not events:
        print("  暂无相关事件")
        return

    # 按优先级排序
    events.sort(key=lambda x: (x.priority != 'HIGH', -x.signal_strength))

    for i, e in enumerate(events[:10], 1):
        priority_icon = '🔴' if e.priority == 'HIGH' else '🟡' if e.priority == 'MEDIUM' else '⚪'
        stars = '⭐' * e.signal_strength

        print(f"\n{i}. {priority_icon} [{e.event_type}] {e.title[:50]}...")
        print(f"   信号强度: {stars}")
        if e.keywords:
            print(f"   关键词: {', '.join(e.keywords[:5])}")
        print(f"   来源: {e.source}")
        print(f"   {e.url}")


def print_signals(signals: List[Dict]):
    """打印交易信号"""
    print(f"\n{'='*70}")
    print("  ⚡ 交易信号汇总")
    print('='*70)

    if not signals:
        print("  暂无高优先级信号")
        return

    for i, s in enumerate(signals[:10], 1):
        stars = '⭐' * s['strength']
        print(f"\n{i}. [{s['type']}] {s['title'][:45]}...")
        print(f"   信号强度: {stars}")
        print(f"   建议: {s['action']}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='事件催化监控系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
事件类型:
  --policy    政策背书 (领导人视察、政策文件)
  --product   产品发布 (模型、新产品)
  --business  商业验证 (提价、合同、客户)
  --capital   资金确认 (机构入场、增持)
  --industry  行业共振 (竞品、板块联动)
  --sentiment 舆情扩散 (媒体热度、社交)
  --all       全量扫描

示例:
  python event_monitor.py --ticker 02513 --all
  python event_monitor.py --ticker 02513 --policy
  python event_monitor.py --theme AI --industry
  python event_monitor.py --ticker 02513 --product --business
        """
    )

    parser.add_argument('--ticker', type=str, help='股票代码')
    parser.add_argument('--theme', type=str, help='主题 (AI/信创/机器人)')
    parser.add_argument('--policy', action='store_true', help='政策背书类')
    parser.add_argument('--product', action='store_true', help='产品发布类')
    parser.add_argument('--business', action='store_true', help='商业验证类')
    parser.add_argument('--capital', action='store_true', help='资金确认类')
    parser.add_argument('--industry', action='store_true', help='行业共振类')
    parser.add_argument('--sentiment', action='store_true', help='舆情扩散类')
    parser.add_argument('--all', action='store_true', help='全量扫描')
    parser.add_argument('--json', action='store_true', help='JSON输出')

    args = parser.parse_args()

    if not args.ticker and not args.theme:
        parser.print_help()
        sys.exit(1)

    monitor = CatalystMonitor(ticker=args.ticker, theme=args.theme)

    if args.all:
        # 全量扫描
        all_events = monitor.scan_all()
        signals = monitor.generate_signals(all_events)

        if args.json:
            output = {
                'ticker': args.ticker,
                'theme': args.theme,
                'scan_time': datetime.now().isoformat(),
                'events': {k: [e.to_dict() for e in v] for k, v in all_events.items()},
                'signals': signals
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            for event_type, events in all_events.items():
                if events:
                    print_events(events, f"{event_type}事件")
            print_signals(signals)

    else:
        # 指定类型扫描
        type_mapping = {
            'policy': EventType.POLICY,
            'product': EventType.PRODUCT,
            'business': EventType.BUSINESS,
            'capital': EventType.CAPITAL,
            'industry': EventType.INDUSTRY,
            'sentiment': EventType.SENTIMENT,
        }

        all_events = []
        for arg_name, event_type in type_mapping.items():
            if getattr(args, arg_name, False):
                print(f"📡 扫描 {event_type.value}...")
                events = monitor.scan_type(event_type)
                if args.json:
                    all_events.extend([e.to_dict() for e in events])
                else:
                    print_events(events, f"{event_type.value}事件")

        if args.json and all_events:
            print(json.dumps(all_events, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
