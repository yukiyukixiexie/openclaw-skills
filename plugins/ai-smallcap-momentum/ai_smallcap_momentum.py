#!/usr/bin/env python3
"""
AI 小盘股爆发策略 - 六要素评分系统

基于智谱(02513.HK)案例复盘，识别具有爆发潜力的小盘股

使用方法:
    python ai_smallcap_momentum.py 02513 --analyze
    python ai_smallcap_momentum.py 02513 --signal
    python ai_smallcap_momentum.py --scan
"""

import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("需要安装依赖: pip install pandas numpy")
    sys.exit(1)

# 导入共享的市场数据模块
sys.path.insert(0, '../shared')
try:
    from market_data import get_hk_stock, get_hk_stock_realtime, calculate_indicators
except ImportError:
    # 如果导入失败，定义本地版本
    def get_hk_stock(*args, **kwargs):
        return {"error": "请确保 market_data.py 在 ../shared/ 目录下"}
    def get_hk_stock_realtime(*args, **kwargs):
        return {"error": "请确保 market_data.py 在 ../shared/ 目录下"}
    def calculate_indicators(df):
        return df


# ============== 六要素评分函数 ==============

def check_float_factor(
    market_cap: float,
    float_ratio: float,
    cornerstone_lockup_ratio: float = 0.0
) -> Tuple[int, Dict]:
    """
    流通盘因子评分

    条件：
    - 流通市值 < 50 亿港元
    - 流通比例 < 30%（IPO 初期）
    - 基石锁定 > 50%

    Args:
        market_cap: 总市值（港元）
        float_ratio: 流通比例（0-1）
        cornerstone_lockup_ratio: 基石锁定比例（0-1）

    Returns:
        (score, details): 评分(0-5)和详情
    """
    float_market_cap = market_cap * float_ratio
    float_market_cap_billion = float_market_cap / 1e9

    score = 0
    details = {
        "total_market_cap": f"{market_cap/1e9:.2f}亿",
        "float_ratio": f"{float_ratio*100:.1f}%",
        "float_market_cap": f"{float_market_cap_billion:.2f}亿",
        "cornerstone_lockup": f"{cornerstone_lockup_ratio*100:.1f}%"
    }

    # 流通市值评分
    if float_market_cap < 20e9:  # 20亿
        score += 3
        details["float_cap_score"] = "极小(<20亿) +3"
    elif float_market_cap < 50e9:  # 50亿
        score += 2
        details["float_cap_score"] = "小(<50亿) +2"
    elif float_market_cap < 100e9:  # 100亿
        score += 1
        details["float_cap_score"] = "中等(<100亿) +1"
    else:
        details["float_cap_score"] = "大(>100亿) +0"

    # 流通比例评分
    if float_ratio < 0.2:
        score += 2
        details["float_ratio_score"] = "极低(<20%) +2"
    elif float_ratio < 0.3:
        score += 1
        details["float_ratio_score"] = "低(<30%) +1"
    else:
        details["float_ratio_score"] = "正常(>30%) +0"

    return min(5, score), details


def check_theme_factor(
    company_description: str,
    sector_keywords: List[str]
) -> Tuple[int, Dict]:
    """
    主题因子评分

    2026年热门主题：AI大模型、AI Agent、具身智能、国产替代

    Args:
        company_description: 公司描述
        sector_keywords: 行业关键词列表

    Returns:
        (score, details): 评分(0-5)和详情
    """
    hot_themes = {
        'AI': 2,
        '大模型': 2,
        'LLM': 2,
        'Agent': 1,
        'DeepSeek': 2,
        '国产替代': 1,
        'GPU': 1,
        '算力': 1,
        '芯片': 1,
        '半导体': 1,
        '新能源': 1,
        '机器人': 1,
        '具身智能': 2,
    }

    matched_themes = []
    score = 0
    text = company_description + ' ' + ' '.join(sector_keywords)

    for theme, weight in hot_themes.items():
        if theme.lower() in text.lower():
            matched_themes.append(theme)
            score += weight

    details = {
        "matched_themes": matched_themes,
        "theme_count": len(matched_themes),
        "raw_score": score
    }

    # 归一化到0-5
    final_score = min(5, score)
    details["final_score"] = final_score

    return final_score, details


def check_cornerstone_factor(
    cornerstone_investors: List[Dict],
    lockup_end_date: datetime,
    current_date: datetime = None
) -> Tuple[int, Dict]:
    """
    基石抬轿因子评分

    利好条件：
    - 知名机构基石（高瓴、红杉等）
    - 锁定期内（通常6个月）
    - 基石占比 > 30%

    Args:
        cornerstone_investors: 基石投资者列表 [{"name": "xxx", "shares": xxx, "ratio": 0.1}]
        lockup_end_date: 锁定期结束日期
        current_date: 当前日期，默认今天

    Returns:
        (score, details): 评分(0-5)和详情
    """
    if current_date is None:
        current_date = datetime.now()

    days_to_unlock = (lockup_end_date - current_date).days

    score = 0
    details = {
        "days_to_unlock": days_to_unlock,
        "investors": [],
        "lockup_status": ""
    }

    # 知名基石机构
    top_investors = ['高瓴', '红杉', 'GIC', '淡马锡', 'KKR', '韩投', 'Korea Investment',
                     '黑石', 'Blackstone', '贝莱德', 'BlackRock', '富达', 'Fidelity',
                     '阿布扎比', 'Abu Dhabi', '中东', '主权基金']

    for inv in cornerstone_investors:
        inv_name = inv.get('name', '')
        is_top = any(top in inv_name for top in top_investors)
        if is_top:
            score += 1
            details["investors"].append(f"{inv_name} (知名机构)")
        else:
            details["investors"].append(inv_name)

    # 锁定期状态评分
    if days_to_unlock > 90:
        score += 2
        details["lockup_status"] = f"安全期（{days_to_unlock}天后解禁）+2"
    elif days_to_unlock > 30:
        score += 1
        details["lockup_status"] = f"观察期（{days_to_unlock}天后解禁）+1"
    elif days_to_unlock > 0:
        details["lockup_status"] = f"警告期（{days_to_unlock}天后解禁）+0"
    else:
        score -= 2
        details["lockup_status"] = f"已解禁（{-days_to_unlock}天前）-2"

    return max(0, min(5, score)), details


def check_catalyst_factor(
    events: List[Dict],
    daily_return: float
) -> Tuple[int, Dict]:
    """
    事件催化因子评分

    高分事件：
    - 行业重磅发布（DeepSeek、OpenAI等）
    - 公司产品发布
    - 重大合同
    - 纳入指数

    Args:
        events: 事件列表 [{"type": "product_launch", "name": "xxx", "date": "xxx"}]
        daily_return: 当日涨幅（百分比）

    Returns:
        (score, details): 评分(0-5)和详情
    """
    score = 0
    details = {
        "events": events,
        "daily_return": f"{daily_return:.1f}%",
        "return_score": 0,
        "event_score": 0
    }

    # 涨幅评分
    if daily_return > 20:
        score += 3
        details["return_score"] = "暴涨(>20%) +3"
    elif daily_return > 10:
        score += 2
        details["return_score"] = "大涨(>10%) +2"
    elif daily_return > 5:
        score += 1
        details["return_score"] = "上涨(>5%) +1"
    else:
        details["return_score"] = "平稳 +0"

    # 事件评分
    high_impact_events = ['deepseek', 'openai', 'chatgpt', '发布', '合同', '指数', '战略']
    event_score = 0
    for event in events:
        event_name = event.get('name', '').lower()
        if any(keyword in event_name for keyword in high_impact_events):
            event_score += 2
        else:
            event_score += 1

    score += min(2, event_score)
    details["event_score"] = f"事件数:{len(events)} +{min(2, event_score)}"

    return min(5, score), details


def check_fund_flow_factor(
    df: pd.DataFrame,
    date_idx: int = -1
) -> Tuple[int, Dict]:
    """
    资金确认因子评分

    Args:
        df: 包含 amount, turnover_rate 等列的 DataFrame
        date_idx: 分析日期索引，默认最后一天

    Returns:
        (score, details): 评分(0-5)和详情
    """
    score = 0
    details = {}

    if len(df) < 20:
        return 0, {"error": "数据不足20天"}

    # 确保有必要的列
    if 'turnover' not in df.columns:
        if 'amount' in df.columns:
            df['turnover'] = df['amount']
        else:
            return 0, {"error": "缺少成交额数据"}

    today = df.iloc[date_idx]

    # 1. 成交额异动
    ma20_turnover = df['turnover'].rolling(20).mean().iloc[date_idx]
    turnover_ratio = today['turnover'] / ma20_turnover if ma20_turnover > 0 else 0

    details["turnover_ratio"] = f"{turnover_ratio:.2f}x"
    if turnover_ratio > 2:
        score += 2
        details["turnover_score"] = "大幅放量(>2x) +2"
    elif turnover_ratio > 1.5:
        score += 1
        details["turnover_score"] = "放量(>1.5x) +1"
    else:
        details["turnover_score"] = "正常 +0"

    # 2. 换手率
    if 'turnover_rate' in df.columns:
        turnover_rate = today['turnover_rate']
        details["turnover_rate"] = f"{turnover_rate:.1f}%"

        if turnover_rate > 10:
            score += 2
            details["turnover_rate_score"] = "极高(>10%) +2"
        elif turnover_rate > 5:
            score += 1
            details["turnover_rate_score"] = "高(>5%) +1"
        else:
            details["turnover_rate_score"] = "正常 +0"

    # 3. 连续放量
    if len(df) >= 3:
        recent_3d = df.iloc[date_idx-2:date_idx+1] if date_idx >= 2 else df.tail(3)
        ma20_recent = df['turnover'].rolling(20).mean().iloc[date_idx-2:date_idx+1] if date_idx >= 2 else df['turnover'].rolling(20).mean().tail(3)

        consecutive_high_volume = all(recent_3d['turnover'].values > ma20_recent.values * 1.2)
        if consecutive_high_volume:
            score += 1
            details["consecutive_volume"] = "连续3日放量 +1"
        else:
            details["consecutive_volume"] = "无连续放量 +0"

    return min(5, score), details


def check_momentum_factor(
    df: pd.DataFrame,
    date_idx: int = -1
) -> Tuple[int, Dict]:
    """
    动量加速因子评分

    Args:
        df: 包含 close, high, volume 等列的 DataFrame
        date_idx: 分析日期索引，默认最后一天

    Returns:
        (score, details): 评分(0-5)和详情
    """
    score = 0
    details = {}

    if len(df) < 20:
        return 0, {"error": "数据不足20天"}

    idx = len(df) + date_idx if date_idx < 0 else date_idx

    # 1. 10日涨幅
    if idx >= 10:
        return_10d = (df.iloc[idx]['close'] / df.iloc[idx-10]['close'] - 1) * 100
        details["return_10d"] = f"{return_10d:.1f}%"

        if return_10d > 100:
            score += 2
            details["return_10d_score"] = "翻倍(>100%) +2"
        elif return_10d > 50:
            score += 1
            details["return_10d_score"] = "大涨(>50%) +1"
        else:
            details["return_10d_score"] = "正常 +0"

    # 2. 5日涨幅
    if idx >= 5:
        return_5d = (df.iloc[idx]['close'] / df.iloc[idx-5]['close'] - 1) * 100
        details["return_5d"] = f"{return_5d:.1f}%"

        if return_5d > 30:
            score += 1
            details["return_5d_score"] = "快速上涨(>30%) +1"
        else:
            details["return_5d_score"] = "正常 +0"

    # 3. 连续创新高
    if idx >= 3:
        recent = df.iloc[idx-3:idx+1]
        new_high_count = sum(recent['high'].values == recent['high'].cummax().values)
        details["new_high_count"] = new_high_count

        if new_high_count >= 3:
            score += 1
            details["new_high_score"] = "连续创新高 +1"
        else:
            details["new_high_score"] = "未连续创新高 +0"

    # 4. 均线多头排列
    if idx >= 20:
        ma5 = df['close'].rolling(5).mean().iloc[idx]
        ma10 = df['close'].rolling(10).mean().iloc[idx]
        ma20 = df['close'].rolling(20).mean().iloc[idx]

        details["ma5"] = f"{ma5:.2f}"
        details["ma10"] = f"{ma10:.2f}"
        details["ma20"] = f"{ma20:.2f}"

        if ma5 > ma10 > ma20:
            score += 1
            details["ma_alignment"] = "多头排列 +1"
        else:
            details["ma_alignment"] = "非多头排列 +0"

    return min(5, score), details


# ============== 综合评分 ==============

def calculate_total_score(factors: Dict[str, int]) -> Tuple[float, str]:
    """
    计算综合加权得分

    Args:
        factors: 各因子得分 {"float": 5, "theme": 4, ...}

    Returns:
        (score, signal): 综合得分和信号
    """
    weights = {
        'float': 0.20,
        'theme': 0.15,
        'cornerstone': 0.15,
        'catalyst': 0.20,
        'fund_flow': 0.15,
        'momentum': 0.15
    }

    total = sum(factors.get(k, 0) * weights[k] for k in weights)

    # 信号判定
    if total >= 4.5:
        signal = "🔥 极强 - 积极入场"
    elif total >= 4.0:
        signal = "🟢 强 - 入场"
    elif total >= 3.5:
        signal = "🟡 中等 - 谨慎入场"
    elif total >= 3.0:
        signal = "🟠 弱 - 小仓试探"
    else:
        signal = "🔴 无 - 不参与"

    return round(total, 2), signal


# ============== 主分析函数 ==============

def analyze_stock(
    ticker: str,
    start_date: str = None,
    company_info: Dict = None
) -> Dict:
    """
    对股票进行完整的六要素分析

    Args:
        ticker: 港股代码
        start_date: 历史数据开始日期
        company_info: 公司信息（可选，手动提供）

    Returns:
        完整分析报告
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')

    # 获取历史数据
    df = get_hk_stock(ticker, start_date)

    if isinstance(df, dict) and "error" in df:
        return df

    # 计算技术指标
    df = calculate_indicators(df)

    # 获取实时行情
    realtime = get_hk_stock_realtime(ticker)

    # 默认公司信息（如果未提供）
    if company_info is None:
        company_info = {
            "market_cap": realtime.get("market_cap", 0),
            "float_ratio": 0.25,  # 默认假设25%流通
            "description": "",
            "sector_keywords": [],
            "cornerstone_investors": [],
            "lockup_end_date": datetime.now() + timedelta(days=180),
            "events": []
        }

    # 计算各因子得分
    float_score, float_details = check_float_factor(
        company_info.get("market_cap", realtime.get("market_cap", 50e9)),
        company_info.get("float_ratio", 0.25)
    )

    theme_score, theme_details = check_theme_factor(
        company_info.get("description", ""),
        company_info.get("sector_keywords", [])
    )

    cornerstone_score, cornerstone_details = check_cornerstone_factor(
        company_info.get("cornerstone_investors", []),
        company_info.get("lockup_end_date", datetime.now() + timedelta(days=180))
    )

    # 计算当日涨幅
    if len(df) >= 2:
        daily_return = (df.iloc[-1]['close'] / df.iloc[-2]['close'] - 1) * 100
    else:
        daily_return = realtime.get("change_pct", 0)

    catalyst_score, catalyst_details = check_catalyst_factor(
        company_info.get("events", []),
        daily_return
    )

    fund_flow_score, fund_flow_details = check_fund_flow_factor(df)

    momentum_score, momentum_details = check_momentum_factor(df)

    # 计算综合得分
    factors = {
        'float': float_score,
        'theme': theme_score,
        'cornerstone': cornerstone_score,
        'catalyst': catalyst_score,
        'fund_flow': fund_flow_score,
        'momentum': momentum_score
    }

    total_score, signal = calculate_total_score(factors)

    # 生成报告
    report = {
        "ticker": ticker,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "realtime_quote": realtime,
        "factors": {
            "float": {"score": float_score, "details": float_details},
            "theme": {"score": theme_score, "details": theme_details},
            "cornerstone": {"score": cornerstone_score, "details": cornerstone_details},
            "catalyst": {"score": catalyst_score, "details": catalyst_details},
            "fund_flow": {"score": fund_flow_score, "details": fund_flow_details},
            "momentum": {"score": momentum_score, "details": momentum_details}
        },
        "total_score": total_score,
        "signal": signal,
        "position_suggestion": get_position_suggestion(total_score)
    }

    return report


def get_position_suggestion(score: float) -> Dict:
    """根据综合得分给出仓位建议"""
    if score >= 4.5:
        return {"position": "80%", "action": "积极入场", "stop_loss": "-8%"}
    elif score >= 4.0:
        return {"position": "60%", "action": "入场", "stop_loss": "-8%"}
    elif score >= 3.5:
        return {"position": "30%", "action": "谨慎入场", "stop_loss": "-5%"}
    elif score >= 3.0:
        return {"position": "10%", "action": "小仓试探", "stop_loss": "-5%"}
    else:
        return {"position": "0%", "action": "不参与", "stop_loss": "N/A"}


def print_report(report: Dict):
    """打印格式化的分析报告"""
    print("\n" + "="*60)
    print(f"  {report['ticker']} AI小盘股爆发策略分析报告")
    print("="*60)
    print(f"分析时间: {report['analysis_date']}")

    # 实时行情
    quote = report.get('realtime_quote', {})
    if quote and 'error' not in quote:
        print(f"\n📈 实时行情:")
        print(f"   现价: {quote.get('price', 'N/A')} HKD")
        print(f"   涨跌: {quote.get('change_pct', 'N/A')}%")

    # 六要素得分
    print(f"\n📊 六要素评分:")
    factors = report.get('factors', {})

    factor_names = {
        'float': '流通盘因子',
        'theme': '主题因子',
        'cornerstone': '基石抬轿因子',
        'catalyst': '事件催化因子',
        'fund_flow': '资金确认因子',
        'momentum': '动量加速因子'
    }

    for key, name in factor_names.items():
        factor = factors.get(key, {})
        score = factor.get('score', 0)
        stars = '⭐' * score + '☆' * (5 - score)
        print(f"   {name}: {stars} ({score}/5)")

    # 综合得分
    print(f"\n🎯 综合得分: {report.get('total_score', 0)}/5")
    print(f"   信号: {report.get('signal', 'N/A')}")

    # 仓位建议
    suggestion = report.get('position_suggestion', {})
    print(f"\n💰 操作建议:")
    print(f"   建议仓位: {suggestion.get('position', 'N/A')}")
    print(f"   操作: {suggestion.get('action', 'N/A')}")
    print(f"   止损: {suggestion.get('stop_loss', 'N/A')}")

    print("\n" + "="*60)


# ============== CLI ==============

def main():
    if len(sys.argv) < 2:
        print("""
AI 小盘股爆发策略 - 六要素评分系统

用法:
    python ai_smallcap_momentum.py <ticker> [--analyze|--signal|--json]

参数:
    ticker      港股代码（如 02513, 0700）
    --analyze   完整分析（默认）
    --signal    仅输出信号
    --json      JSON格式输出

示例:
    python ai_smallcap_momentum.py 02513
    python ai_smallcap_momentum.py 02513 --json
        """)
        sys.exit(1)

    ticker = sys.argv[1].replace('.HK', '').replace('.hk', '').zfill(5)

    # 解析参数
    output_json = '--json' in sys.argv
    signal_only = '--signal' in sys.argv

    # 运行分析
    report = analyze_stock(ticker)

    if isinstance(report, dict) and 'error' in report:
        print(f"❌ 错误: {report['error']}")
        sys.exit(1)

    # 输出结果
    if output_json:
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    elif signal_only:
        print(f"{report['ticker']}: {report['signal']} (得分: {report['total_score']})")
    else:
        print_report(report)


if __name__ == "__main__":
    main()
