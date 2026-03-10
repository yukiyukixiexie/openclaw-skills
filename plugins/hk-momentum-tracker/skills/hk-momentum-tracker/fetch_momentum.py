#!/usr/bin/env python3
"""
HK Momentum Tracker - 优化版数据获取脚本
专注于动量追踪，使用最稳定的数据源组合

数据源策略：
- 实时数据：腾讯（最快，有换手率）+ 东方财富（有流通市值）
- 历史K线：腾讯（稳定可靠）
- 换手率：通过流通市值计算
"""

import json
import sys
import time
import requests
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# ============================================================
# 配置
# ============================================================

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "*/*",
}

# AI 概念股列表
AI_STOCKS = [
    ("02513", "智谱"),        # AI大模型龙头
    ("00020", "商汤-W"),
    ("02013", "微盟集团"),
    ("09888", "百度集团"),
    ("09618", "京东集团"),
    ("03690", "美团-W"),
    ("09988", "阿里巴巴"),
    ("00700", "腾讯控股"),
    ("01024", "快手-W"),
    ("09626", "哔哩哔哩"),
    ("06060", "众安在线"),
    ("09999", "网易-S"),
    ("00981", "中芯国际"),
    ("02382", "舜宇光学"),
    ("06682", "第四范式"),
    ("03896", "金山云"),
    ("00268", "金蝶国际"),
    ("09660", "地平线机器人"),  # 智能驾驶
    ("02506", "讯飞医疗"),     # AI医疗
]


# ============================================================
# 实时数据获取
# ============================================================

def fetch_realtime_tencent(symbol: str) -> Dict:
    """
    腾讯财经实时行情（最快，有换手率）
    响应时间：~50ms
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)
    url = f"https://qt.gtimg.cn/q=r_hk{symbol}"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=5)
        content = resp.text

        if '="' not in content:
            return {"error": f"无数据: {symbol}"}

        data_str = content.split('="')[1].rstrip('";')
        f = data_str.split('~')

        if len(f) < 50:
            return {"error": "数据不完整"}

        return {
            "source": "tencent",
            "symbol": f[2] if f[2] else symbol,
            "name": f[1],
            "price": float(f[3]) if f[3] else 0,
            "prev_close": float(f[4]) if f[4] else 0,
            "open": float(f[5]) if f[5] else 0,
            "high": float(f[33]) if f[33] else 0,
            "low": float(f[34]) if f[34] else 0,
            "volume": int(float(f[6])) if f[6] else 0,  # 成交量（股）
            "turnover": float(f[37]) if f[37] else 0,  # 成交额（港元）
            "turnover_billion": float(f[37]) / 1e8 if f[37] else 0,
            "change": float(f[31]) if f[31] else 0,
            "change_pct": float(f[32]) if f[32] else 0,
            "amplitude": float(f[43]) if f[43] else 0,
            "turnover_rate": float(f[38]) if f[38] else 0,  # 换手率
            "market_cap": float(f[45]) if f[45] else 0,  # 总市值（亿）
            "pe": float(f[39]) if f[39] else 0,
            "week52_high": float(f[48]) if f[48] else 0,
            "week52_low": float(f[49]) if f[49] else 0,
            "update_time": f[30] if len(f) > 30 else "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_realtime_eastmoney(symbol: str) -> Dict:
    """
    东方财富实时行情（有流通市值，用于计算换手率）
    响应时间：~150ms
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)
    url = "https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"116.{symbol}",
        "fields": "f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f116,f117,f169,f170,f171"
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=5)
        data = resp.json()

        if data.get("data") is None:
            return {"error": f"无数据: {symbol}"}

        d = data["data"]

        return {
            "source": "eastmoney",
            "symbol": d.get("f57", symbol),
            "name": d.get("f58", ""),
            "price": d.get("f43", 0) / 1000,
            "high": d.get("f44", 0) / 1000,
            "low": d.get("f45", 0) / 1000,
            "open": d.get("f46", 0) / 1000,
            "prev_close": d.get("f60", 0) / 1000,
            "volume": d.get("f47", 0),
            "turnover": d.get("f48", 0),
            "turnover_billion": d.get("f48", 0) / 1e8,
            "volume_ratio": d.get("f50", 0) / 100,
            "change": d.get("f169", 0) / 1000,
            "change_pct": d.get("f170", 0) / 100,
            "amplitude": d.get("f171", 0) / 100,
            "total_market_cap": d.get("f116", 0) / 1e8,  # 总市值（亿）
            "float_market_cap": d.get("f117", 0) / 1e8,  # 流通市值（亿）
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_realtime(symbol: str) -> Dict:
    """
    获取实时数据（合并两个数据源的优势）
    - 价格/换手率来自腾讯（更快）
    - 流通市值来自东方财富（更准）
    """
    # 并行获取
    tencent = fetch_realtime_tencent(symbol)
    eastmoney = fetch_realtime_eastmoney(symbol)

    if "error" in tencent:
        return tencent

    # 合并数据
    result = tencent.copy()

    # 从东方财富补充流通市值
    if "error" not in eastmoney:
        result["float_market_cap"] = eastmoney.get("float_market_cap", 0)
        result["volume_ratio"] = eastmoney.get("volume_ratio", 0)

        # 如果腾讯没有换手率，用流通市值计算
        if not result.get("turnover_rate") and result["float_market_cap"] > 0:
            result["turnover_rate"] = result["turnover_billion"] / result["float_market_cap"] * 100

    return result


# ============================================================
# 历史数据获取
# ============================================================

def fetch_history_tencent(symbol: str, days: int = 120) -> Dict:
    """
    腾讯历史K线（最稳定）
    响应时间：~100ms
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)
    url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {"param": f"hk{symbol},day,,,{days},qfq"}

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
        data = resp.json()

        if data.get("code") != 0:
            return {"error": f"API错误: {data.get('msg')}"}

        stock_data = data.get("data", {}).get(f"hk{symbol}", {})
        klines = stock_data.get("day", []) or stock_data.get("qfqday", [])

        if not klines:
            return {"error": f"无历史数据: {symbol}"}

        # 解析K线
        history = []
        for line in klines:
            if len(line) >= 6:
                o, c, h, l = float(line[1]), float(line[2]), float(line[3]), float(line[4])
                vol = float(line[5])
                avg_price = (h + l + c) / 3 if h and l else c
                turnover = vol * avg_price

                history.append({
                    "date": line[0],
                    "open": o,
                    "close": c,
                    "high": h,
                    "low": l,
                    "volume": vol,
                    "turnover": turnover,
                    "amplitude": round((h - l) / o * 100, 2) if o > 0 else 0,
                    "change_pct": round((c / o - 1) * 100, 2) if o > 0 else 0,
                })

        if not history:
            return {"error": "解析失败"}

        return {
            "source": "tencent",
            "symbol": symbol,
            "data_start": history[0]["date"],
            "data_end": history[-1]["date"],
            "trading_days": len(history),
            "history": history
        }

    except Exception as e:
        return {"error": str(e)}


def fetch_full_data(symbol: str, days: int = 120) -> Dict:
    """
    获取完整数据（实时 + 历史 + 计算指标）
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    # 获取实时数据
    realtime = fetch_realtime(symbol)
    if "error" in realtime:
        return realtime

    # 获取历史数据
    hist_data = fetch_history_tencent(symbol, days)
    if "error" in hist_data:
        return {**realtime, "history_error": hist_data["error"]}

    history = hist_data["history"]

    # 计算统计指标
    closes = [h["close"] for h in history]
    turnovers = [h["turnover"] for h in history]

    # 用流通市值计算历史换手率
    float_cap = realtime.get("float_market_cap", 0) * 1e8  # 转为港元
    if float_cap > 0:
        for h in history:
            h["turnover_rate"] = h["turnover"] / float_cap * 100

    result = {
        **realtime,
        "data_start": hist_data["data_start"],
        "data_end": hist_data["data_end"],
        "trading_days": hist_data["trading_days"],

        # 价格统计
        "history_high": max(h["high"] for h in history),
        "history_low": min(h["low"] for h in history),

        # 涨跌幅
        "return_total": (closes[-1] / closes[0] - 1) * 100 if closes[0] > 0 else 0,

        # 成交统计
        "total_turnover_billion": sum(turnovers) / 1e8,
        "avg_turnover_billion": np.mean(turnovers) / 1e8,
        "turnover_p50": np.percentile(turnovers, 50) / 1e8,
        "turnover_p90": np.percentile(turnovers, 90) / 1e8,
        "max_turnover_billion": max(turnovers) / 1e8,

        # 近期涨跌幅
        "return_5d": (closes[-1] / closes[-6] - 1) * 100 if len(closes) >= 6 else None,
        "return_10d": (closes[-1] / closes[-11] - 1) * 100 if len(closes) >= 11 else None,
        "return_20d": (closes[-1] / closes[-21] - 1) * 100 if len(closes) >= 21 else None,

        # MA
        "ma5": np.mean(closes[-5:]) if len(closes) >= 5 else None,
        "ma10": np.mean(closes[-10:]) if len(closes) >= 10 else None,
        "ma20": np.mean(closes[-20:]) if len(closes) >= 20 else None,

        # 成交额比率
        "volume_ratio_5d": turnovers[-1] / np.mean(turnovers[-5:]) if len(turnovers) >= 5 else None,
        "volume_ratio_20d": turnovers[-1] / np.mean(turnovers[-20:]) if len(turnovers) >= 20 else None,

        # 回撤
        "current_drawdown": (max(closes) - closes[-1]) / max(closes) * 100,

        # 最近数据
        "recent_data": history[-20:],

        # 成交额 Top 10
        "top_volume_days": sorted(history, key=lambda x: x["turnover"], reverse=True)[:10],
    }

    # 计算阶段信号
    result["signals"] = calculate_signals(history)
    result["phase"] = determine_phase(result["signals"])
    result["strength_score"] = calculate_strength_score(result)
    result["entry_eval"] = evaluate_entry_opportunity(result)

    return result


# ============================================================
# 信号计算
# ============================================================

def calculate_signals(history: List[Dict]) -> Dict:
    """计算阶段信号"""
    signals = {
        "acceleration": {"score": 0, "details": []},
        "distribution": {"score": 0, "details": []},
        "breakdown": {"score": 0, "details": []}
    }

    if len(history) < 20:
        return signals

    latest = history[-1]
    closes = [h["close"] for h in history]
    turnovers = [h["turnover"] for h in history]

    # ========== 加速段信号 ==========
    # 1. 10日涨幅 > 80%（降低阈值更敏感）
    if len(history) >= 11:
        r10 = (closes[-1] / closes[-11] - 1) * 100
        if r10 > 80:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"10日涨幅 {r10:.1f}%")

    # 2. 5日涨幅 > 30%
    if len(history) >= 6:
        r5 = (closes[-1] / closes[-6] - 1) * 100
        if r5 > 30:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"5日涨幅 {r5:.1f}%")

    # 3. 连续创新高 >= 3天
    recent = closes[-10:]
    streak = 0
    running_max = 0
    for p in recent:
        if p > running_max:
            streak += 1
            running_max = p
        else:
            streak = 0
    if streak >= 3:
        signals["acceleration"]["score"] += 1
        signals["acceleration"]["details"].append(f"连续 {streak} 天新高")

    # 4. 成交额创60日新高
    if turnovers[-1] >= max(turnovers[-60:]):
        signals["acceleration"]["score"] += 1
        signals["acceleration"]["details"].append("成交额60日新高")

    # 5. 20日回撤 < 15%
    if len(history) >= 20:
        peak20 = max(closes[-20:])
        dd20 = (peak20 - closes[-1]) / peak20 * 100
        if dd20 < 15:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"20日回撤 {dd20:.1f}%")

    # ========== 分歧段信号 ==========
    # 1. 振幅扩大
    amps = [h["amplitude"] for h in history[-20:]]
    if latest["amplitude"] > np.mean(amps) * 1.5:
        signals["distribution"]["score"] += 1
        signals["distribution"]["details"].append(f"振幅扩大 {latest['amplitude']:.1f}%")

    # 2. 长上影线
    if latest["high"] > latest["low"]:
        upper_shadow = (latest["high"] - latest["close"]) / (latest["high"] - latest["low"])
        if upper_shadow > 0.6:
            signals["distribution"]["score"] += 1
            signals["distribution"]["details"].append(f"上影线 {upper_shadow:.1%}")

    # 3. 量增价滞
    if turnovers[-1] >= max(turnovers) * 0.9 and abs(latest["change_pct"]) < 5:
        signals["distribution"]["score"] += 1
        signals["distribution"]["details"].append("量增价滞")

    # 4. 连续收盘弱势
    weak = sum(1 for h in history[-2:] if h["close"] < h["high"] * 0.95)
    if weak >= 2:
        signals["distribution"]["score"] += 1
        signals["distribution"]["details"].append("连续弱势收盘")

    # ========== 破位段信号 ==========
    # 1. 距高点回撤 > 25%
    peak = max(closes)
    dd = (peak - closes[-1]) / peak * 100
    if dd > 25:
        signals["breakdown"]["score"] += 1
        signals["breakdown"]["details"].append(f"回撤 {dd:.1f}%")

    # 2. 跌破20日均线
    if len(history) >= 20:
        ma20 = np.mean(closes[-20:])
        if closes[-1] < ma20:
            signals["breakdown"]["score"] += 1
            signals["breakdown"]["details"].append(f"跌破MA20")

    # 3. 放量下跌
    if len(history) >= 20:
        avg_vol = np.mean(turnovers[-20:])
        if turnovers[-1] > avg_vol and latest["change_pct"] < 0:
            signals["breakdown"]["score"] += 1
            signals["breakdown"]["details"].append("放量下跌")

    return signals


def determine_phase(signals: Dict) -> Dict:
    """判断当前阶段"""
    if signals["breakdown"]["score"] >= 2:
        return {
            "phase": "BREAKDOWN",
            "phase_cn": "破位段",
            "score": signals["breakdown"]["score"],
            "details": signals["breakdown"]["details"],
            "action": "清仓/观望",
            "risk": "极高",
            "color": "red"
        }
    elif signals["distribution"]["score"] >= 3:
        return {
            "phase": "DISTRIBUTION",
            "phase_cn": "分歧段",
            "score": signals["distribution"]["score"],
            "details": signals["distribution"]["details"],
            "action": "减仓 30-50%",
            "risk": "高",
            "color": "orange"
        }
    elif signals["acceleration"]["score"] >= 3:
        return {
            "phase": "ACCELERATION",
            "phase_cn": "加速段",
            "score": signals["acceleration"]["score"],
            "details": signals["acceleration"]["details"],
            "action": "停止加仓，设移动止盈",
            "risk": "高",
            "color": "yellow"
        }
    elif signals["acceleration"]["score"] >= 1:
        return {
            "phase": "MOMENTUM",
            "phase_cn": "主升段",
            "score": signals["acceleration"]["score"],
            "details": signals["acceleration"]["details"],
            "action": "持有/轻仓参与",
            "risk": "中",
            "color": "green"
        }
    else:
        return {
            "phase": "CONSOLIDATION",
            "phase_cn": "盘整期",
            "score": 0,
            "details": [],
            "action": "观望等待",
            "risk": "低",
            "color": "gray"
        }


def calculate_strength_score(data: Dict) -> Dict:
    """计算资金强度评分"""
    details = []
    total = 0

    # 1. 成交额趋势 (30%)
    vr = data.get("volume_ratio_20d")
    if vr:
        score = min(100, vr * 50)
        details.append({"dim": "成交额趋势", "weight": 0.30, "value": f"VR={vr:.2f}", "score": score})
        total += score * 0.30

    # 2. 换手率 (25%)
    tr = data.get("turnover_rate")
    if tr:
        score = min(100, tr * 5)  # 20%换手率得100分
        details.append({"dim": "换手率", "weight": 0.25, "value": f"{tr:.1f}%", "score": score})
        total += score * 0.25

    # 3. 价格斜率 (25%)
    r10 = data.get("return_10d")
    if r10:
        slope = r10 / 10
        score = min(100, max(0, slope * 10 + 50))
        details.append({"dim": "价格斜率", "weight": 0.25, "value": f"{slope:.2f}%/日", "score": score})
        total += score * 0.25

    # 4. 距新高 (20%)
    dd = data.get("current_drawdown", 0)
    score = max(0, 100 - dd * 4)
    details.append({"dim": "距新高", "weight": 0.20, "value": f"回撤{dd:.1f}%", "score": score})
    total += score * 0.20

    return {"total": round(total, 1), "details": details}


def evaluate_entry_opportunity(data: Dict) -> Dict:
    """
    评估入场机会（叙事驱动趋势交易策略）

    目标：赚"市场刚形成共识"到"情绪过热"之间的 30%-80%

    不参与：
    - 第一天爆发
    - 最后疯狂冲顶
    - 已经10天3倍的尾声
    - 明显分歧阶段
    """
    result = {
        "suitable": False,
        "score": 0,
        "position_suggestion": 0,
        "stage": "",
        "reasons": [],
        "warnings": [],
        "action": ""
    }

    r5 = data.get("return_5d", 0) or 0
    r10 = data.get("return_10d", 0) or 0
    r20 = data.get("return_20d", 0) or 0
    dd = data.get("current_drawdown", 0) or 0
    signals = data.get("signals", {})
    phase = data.get("phase", {}).get("phase", "")
    vr = data.get("volume_ratio_20d", 0) or 0
    tr = data.get("turnover_rate", 0) or 0

    acc_score = signals.get("acceleration", {}).get("score", 0)
    dist_score = signals.get("distribution", {}).get("score", 0)
    break_score = signals.get("breakdown", {}).get("score", 0)

    # ========== 排除条件（硬性） ==========

    # 1. 已经10天3倍（200%）= 尾声，不参与
    if r10 > 200:
        result["warnings"].append(f"10日涨幅 {r10:.0f}% 已超200%，属于尾声阶段")
        result["stage"] = "尾声"
        result["action"] = "不参与 - 已错过最佳入场点"
        return result

    # 2. 明显分歧阶段
    if dist_score >= 3:
        result["warnings"].append(f"分歧信号 {dist_score}/4，情绪开始分歧")
        result["stage"] = "分歧"
        result["action"] = "不加仓 - 等待分歧结束"
        return result

    # 3. 破位阶段
    if break_score >= 2:
        result["warnings"].append(f"破位信号 {break_score}/3，趋势已破坏")
        result["stage"] = "破位"
        result["action"] = "清仓/观望 - 等待右侧机会"
        return result

    # 4. 回撤超过25%
    if dd > 25:
        result["warnings"].append(f"回撤 {dd:.1f}% 超过25%，趋势可能结束")
        result["stage"] = "深度回调"
        result["action"] = "观望 - 不抄底"
        return result

    # ========== 阶段判断 ==========

    # 已涨较多但回撤较大 = 分歧回调中
    if r10 > 50 and dd > 15:
        result["stage"] = "分歧回调"
        result["suitable"] = False
        result["score"] = 30
        result["position_suggestion"] = 0
        result["warnings"].append(f"10日涨 {r10:.0f}% 但回撤 {dd:.1f}%，处于分歧回调")
        result["warnings"].append("需观察能否止跌企稳")
        result["action"] = "观望 - 等待止跌信号"
        return result

    # 晚期：接近过热（10日涨幅 >150%）
    if r10 > 150:
        result["stage"] = "接近过热"
        result["suitable"] = False
        result["score"] = 20
        result["position_suggestion"] = 0
        result["warnings"].append(f"10日涨 {r10:.0f}%，可能接近顶部区域")
        result["action"] = "不新增仓位 - 考虑减仓"
        return result

    # 中期：稳定加速段（10日涨幅 80-150%，回撤小）
    if 80 <= r10 <= 150 and dd < 15:
        result["stage"] = "稳定加速段"
        result["suitable"] = True
        result["score"] = 60
        result["position_suggestion"] = 15  # 控制仓位
        result["reasons"].append(f"10日涨 {r10:.0f}%，处于加速阶段")
        result["reasons"].append(f"回撤 {dd:.1f}%，仍在强势区间")
        result["warnings"].append("加速段波动大，需设移动止盈")
        result["action"] = "可轻仓参与 - 设好止盈"
        return result

    # 早期：市场刚形成共识（10日涨幅 20-80%）
    if 20 < r10 < 80 and dd < 15:
        result["stage"] = "共识形成期"
        result["suitable"] = True
        result["score"] = 80
        result["position_suggestion"] = 25  # 可加到25%
        result["reasons"].append(f"10日涨 {r10:.0f}%，趋势确认但未过热")
        result["reasons"].append(f"回撤 {dd:.1f}%，仍在强势区间")
        result["action"] = "可入场/加仓 - 趋势确认阶段"
        return result

    # 启动初期（10日涨幅 10-20%）
    if 10 < r10 <= 20 and dd < 10:
        result["stage"] = "启动初期"
        result["suitable"] = True
        result["score"] = 70
        result["position_suggestion"] = 15
        result["reasons"].append(f"10日涨 {r10:.0f}%，可能是趋势启动")
        result["reasons"].append("需确认放量突破")
        result["action"] = "可小仓位参与 - 等待确认"
        return result

    # 盘整期
    if abs(r10) < 10:
        result["stage"] = "盘整观望"
        result["suitable"] = False
        result["score"] = 40
        result["position_suggestion"] = 10  # 观察仓位
        result["reasons"].append(f"10日涨跌 {r10:.0f}%，趋势未确认")
        result["action"] = "观望/小仓位埋伏 - 等待突破"
        return result

    # 下跌中
    if r10 < -10:
        result["stage"] = "下跌趋势"
        result["suitable"] = False
        result["score"] = 20
        result["position_suggestion"] = 0
        result["warnings"].append(f"10日跌 {abs(r10):.0f}%，处于下跌趋势")
        result["action"] = "不参与 - 等待右侧"
        return result

    # 默认
    result["stage"] = "观望"
    result["action"] = "继续观察"
    return result


# ============================================================
# 批量扫描
# ============================================================

def scan_ai_stocks() -> Dict:
    """扫描 AI 概念股，找出动量最强的"""
    results = []

    print("扫描 AI 概念股...", file=sys.stderr)

    for symbol, name in AI_STOCKS:
        try:
            data = fetch_realtime(symbol)
            if "error" not in data:
                data["name"] = name
                results.append(data)
        except:
            continue
        time.sleep(0.05)  # 避免请求过快

    # 分类
    hot = []  # 今日热门：涨幅>5% 或 换手率>10%
    momentum = []  # 动量股：涨幅>3% 且 量比>1.5
    weak = []  # 弱势股：跌幅>3%

    for s in results:
        change = s.get("change_pct", 0)
        tr = s.get("turnover_rate", 0)
        vr = s.get("volume_ratio", 0)

        info = {
            "symbol": s.get("symbol"),
            "name": s.get("name"),
            "price": s.get("price"),
            "change_pct": change,
            "turnover_rate": tr,
            "turnover_billion": s.get("turnover_billion"),
            "volume_ratio": vr,
            "amplitude": s.get("amplitude"),
        }

        if change > 5 or tr > 10:
            hot.append(info)
        if change > 3 and vr > 1.5:
            momentum.append(info)
        if change < -3:
            weak.append(info)

    # 排序
    hot.sort(key=lambda x: x["change_pct"], reverse=True)
    momentum.sort(key=lambda x: x["change_pct"], reverse=True)
    weak.sort(key=lambda x: x["change_pct"])

    return {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "hot": hot,
        "momentum": momentum,
        "weak": weak,
        "all": sorted([{
            "symbol": s.get("symbol"),
            "name": s.get("name"),
            "price": s.get("price"),
            "change_pct": s.get("change_pct"),
            "turnover_billion": s.get("turnover_billion"),
        } for s in results], key=lambda x: x["change_pct"], reverse=True)
    }


def scan_entry_opportunities() -> Dict:
    """扫描入场机会（基于叙事驱动趋势策略）"""
    results = []

    print("扫描入场机会...", file=sys.stderr)

    for symbol, name in AI_STOCKS:
        try:
            data = fetch_full_data(symbol, days=60)
            if "error" not in data:
                e = data.get("entry_eval", {})
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "price": data.get("price", 0),
                    "change_pct": data.get("change_pct", 0),
                    "return_5d": data.get("return_5d", 0),
                    "return_10d": data.get("return_10d", 0),
                    "return_20d": data.get("return_20d", 0),
                    "drawdown": data.get("current_drawdown", 0),
                    "turnover_billion": data.get("turnover_billion", 0),
                    "stage": e.get("stage", ""),
                    "suitable": e.get("suitable", False),
                    "score": e.get("score", 0),
                    "action": e.get("action", ""),
                    "position": e.get("position_suggestion", 0),
                    "reasons": e.get("reasons", []),
                    "warnings": e.get("warnings", []),
                })
        except:
            continue

    # 按评分排序
    results.sort(key=lambda x: (-x["score"], -x.get("return_10d", 0)))

    # 分类
    suitable = [r for r in results if r["suitable"]]
    watching = [r for r in results if not r["suitable"] and r["score"] >= 30]
    avoid = [r for r in results if r["score"] < 30]

    return {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": len(results),
        "suitable": suitable,
        "watching": watching,
        "avoid": avoid,
        "all": results
    }


# ============================================================
# 主函数
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("""
HK Momentum Tracker - 港股动量追踪

用法:
  python fetch_momentum.py <股票代码>           # 完整分析
  python fetch_momentum.py <股票代码> --rt      # 仅实时数据
  python fetch_momentum.py --scan               # 快速扫描（仅实时）
  python fetch_momentum.py --opportunity        # 入场机会扫描（完整分析）

示例:
  python fetch_momentum.py 00020
  python fetch_momentum.py 02513 --rt
  python fetch_momentum.py --opportunity
""")
        sys.exit(1)

    arg = sys.argv[1]

    # 快速扫描模式（仅实时数据）
    if arg == "--scan":
        result = scan_ai_stocks()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 入场机会扫描（完整分析）
    if arg == "--opportunity" or arg == "-o":
        result = scan_entry_opportunities()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    symbol = arg

    # 仅实时
    if "--rt" in sys.argv:
        result = fetch_realtime(symbol)
    else:
        result = fetch_full_data(symbol)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
