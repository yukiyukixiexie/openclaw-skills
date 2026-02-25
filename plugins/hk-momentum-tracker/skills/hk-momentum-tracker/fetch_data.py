#!/usr/bin/env python3
"""
HK Momentum Tracker - 港股数据抓取脚本
使用 AKShare 获取港股历史行情数据
"""

import akshare as ak
import pandas as pd
import numpy as np
import json
import sys
from datetime import datetime, timedelta

def fetch_hk_stock_data(symbol: str) -> dict:
    """
    获取港股完整历史数据

    Args:
        symbol: 股票代码，如 "00020" 或 "00020.HK"

    Returns:
        dict: 包含行情数据和计算指标的字典
    """
    # 清理股票代码
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    print(f"正在获取 {symbol} 的历史数据...", file=sys.stderr)

    try:
        # 获取历史数据
        df = ak.stock_hk_hist(symbol=symbol, period="daily", adjust="qfq")

        if df.empty:
            return {"error": f"无法获取 {symbol} 的数据"}

        # 转换日期格式
        df['日期'] = pd.to_datetime(df['日期'])
        df = df.sort_values('日期').reset_index(drop=True)

        # 基础统计
        result = {
            "symbol": symbol,
            "data_start": df['日期'].min().strftime("%Y-%m-%d"),
            "data_end": df['日期'].max().strftime("%Y-%m-%d"),
            "trading_days": len(df),

            # 价格统计
            "current_price": float(df['收盘'].iloc[-1]),
            "history_high": float(df['最高'].max()),
            "history_high_date": df.loc[df['最高'].idxmax(), '日期'].strftime("%Y-%m-%d"),
            "history_low": float(df['最低'].min()),
            "history_low_date": df.loc[df['最低'].idxmin(), '日期'].strftime("%Y-%m-%d"),
            "total_return": float((df['收盘'].iloc[-1] / df['收盘'].iloc[0] - 1) * 100),

            # 成交统计
            "total_turnover": float(df['成交额'].sum() / 1e8),  # 亿
            "avg_daily_turnover": float(df['成交额'].mean() / 1e8),
            "turnover_p50": float(df['成交额'].quantile(0.50) / 1e8),
            "turnover_p75": float(df['成交额'].quantile(0.75) / 1e8),
            "turnover_p90": float(df['成交额'].quantile(0.90) / 1e8),
            "turnover_p99": float(df['成交额'].quantile(0.99) / 1e8),
            "max_daily_turnover": float(df['成交额'].max() / 1e8),
            "max_turnover_date": df.loc[df['成交额'].idxmax(), '日期'].strftime("%Y-%m-%d"),

            # 换手率统计（如果有）
            "cumulative_turnover": float(df['换手率'].sum()) if '换手率' in df.columns else None,
            "avg_turnover_rate": float(df['换手率'].mean()) if '换手率' in df.columns else None,
            "turnover_rate_p90": float(df['换手率'].quantile(0.90)) if '换手率' in df.columns else None,
            "max_turnover_rate": float(df['换手率'].max()) if '换手率' in df.columns else None,
            "max_turnover_rate_date": df.loc[df['换手率'].idxmax(), '日期'].strftime("%Y-%m-%d") if '换手率' in df.columns else None,
        }

        # 计算近期指标
        if len(df) >= 5:
            result["return_5d"] = float((df['收盘'].iloc[-1] / df['收盘'].iloc[-6] - 1) * 100)
        if len(df) >= 10:
            result["return_10d"] = float((df['收盘'].iloc[-1] / df['收盘'].iloc[-11] - 1) * 100)
        if len(df) >= 20:
            result["return_20d"] = float((df['收盘'].iloc[-1] / df['收盘'].iloc[-21] - 1) * 100)
            result["ma20"] = float(df['收盘'].tail(20).mean())
            result["volume_ratio_20d"] = float(df['成交额'].iloc[-1] / df['成交额'].tail(20).mean())
        if len(df) >= 60:
            result["return_60d"] = float((df['收盘'].iloc[-1] / df['收盘'].iloc[-61] - 1) * 100)

        # 计算最大回撤
        df['cummax'] = df['收盘'].cummax()
        df['drawdown'] = (df['cummax'] - df['收盘']) / df['cummax'] * 100
        result["max_drawdown"] = float(df['drawdown'].max())

        # 计算当前回撤
        peak_price = df['收盘'].max()
        current_price = df['收盘'].iloc[-1]
        result["current_drawdown"] = float((peak_price - current_price) / peak_price * 100)

        # 计算阶段信号
        result["signals"] = calculate_signals(df)

        # 最近 20 个交易日的详细数据
        recent_df = df.tail(20).copy()
        result["recent_data"] = []
        for _, row in recent_df.iterrows():
            result["recent_data"].append({
                "date": row['日期'].strftime("%Y-%m-%d"),
                "open": float(row['开盘']),
                "high": float(row['最高']),
                "low": float(row['最低']),
                "close": float(row['收盘']),
                "volume": float(row['成交额'] / 1e8),
                "change": float(row['涨跌幅']),
                "amplitude": float(row['振幅']),
                "turnover_rate": float(row['换手率']) if '换手率' in row else None
            })

        # 成交额 Top 10 交易日
        top_volume_df = df.nlargest(10, '成交额')
        result["top_volume_days"] = []
        for _, row in top_volume_df.iterrows():
            result["top_volume_days"].append({
                "date": row['日期'].strftime("%Y-%m-%d"),
                "volume": float(row['成交额'] / 1e8),
                "change": float(row['涨跌幅']),
                "turnover_rate": float(row['换手率']) if '换手率' in row else None
            })

        return result

    except Exception as e:
        return {"error": str(e)}


def calculate_signals(df: pd.DataFrame) -> dict:
    """
    计算阶段信号
    """
    signals = {
        "acceleration": {"score": 0, "details": []},
        "distribution": {"score": 0, "details": []},
        "breakdown": {"score": 0, "details": []}
    }

    if len(df) < 60:
        return signals

    latest = df.iloc[-1]

    # ========== 加速段信号 ==========
    # 1. 10日涨幅 > 120%
    if len(df) >= 11:
        return_10d = (df['收盘'].iloc[-1] / df['收盘'].iloc[-11] - 1) * 100
        if return_10d > 120:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"10日涨幅 {return_10d:.1f}% > 120%")

    # 2. 5日涨幅 > 50%
    if len(df) >= 6:
        return_5d = (df['收盘'].iloc[-1] / df['收盘'].iloc[-6] - 1) * 100
        if return_5d > 50:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"5日涨幅 {return_5d:.1f}% > 50%")

    # 3. 连续创新高 >= 3天
    recent_highs = df['收盘'].tail(10)
    consecutive_new_highs = 0
    running_max = 0
    for price in recent_highs:
        if price > running_max:
            consecutive_new_highs += 1
            running_max = price
        else:
            consecutive_new_highs = 0
    if consecutive_new_highs >= 3:
        signals["acceleration"]["score"] += 1
        signals["acceleration"]["details"].append(f"连续 {consecutive_new_highs} 天创新高")

    # 4. 成交额为60日最高
    if latest['成交额'] >= df['成交额'].tail(60).max():
        signals["acceleration"]["score"] += 1
        signals["acceleration"]["details"].append("成交额创 60 日新高")

    # 5. 20日回撤 < 15%
    if len(df) >= 20:
        recent_peak = df['收盘'].tail(20).max()
        drawdown_20d = (recent_peak - latest['收盘']) / recent_peak * 100
        if drawdown_20d < 15:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"20日回撤 {drawdown_20d:.1f}% < 15%")

    # ========== 分歧段信号 ==========
    # 1. 振幅扩大 > 均值 1.5倍
    if len(df) >= 20:
        avg_amplitude = df['振幅'].tail(20).mean()
        if latest['振幅'] > avg_amplitude * 1.5:
            signals["distribution"]["score"] += 1
            signals["distribution"]["details"].append(f"振幅 {latest['振幅']:.1f}% > 均值 {avg_amplitude:.1f}% × 1.5")

    # 2. 长上影线
    if latest['最高'] > latest['最低']:
        upper_shadow_ratio = (latest['最高'] - latest['收盘']) / (latest['最高'] - latest['最低'])
        if upper_shadow_ratio > 0.6:
            signals["distribution"]["score"] += 1
            signals["distribution"]["details"].append(f"上影线比率 {upper_shadow_ratio:.2f} > 0.6")

    # 3. 量增价滞
    if len(df) >= 2:
        if latest['成交额'] >= df['成交额'].max() and abs(latest['涨跌幅']) < 5:
            signals["distribution"]["score"] += 1
            signals["distribution"]["details"].append("成交额新高但涨幅 < 5%（量增价滞）")

    # 4. 收盘弱势（连续2日收盘 < 日内高点 95%）
    if len(df) >= 2:
        weak_close_count = 0
        for i in [-1, -2]:
            row = df.iloc[i]
            if row['收盘'] < row['最高'] * 0.95:
                weak_close_count += 1
        if weak_close_count >= 2:
            signals["distribution"]["score"] += 1
            signals["distribution"]["details"].append("连续 2 日收盘弱势")

    # ========== 破位段信号 ==========
    # 1. 距高点回撤 > 30%
    peak = df['收盘'].max()
    drawdown = (peak - latest['收盘']) / peak * 100
    if drawdown > 30:
        signals["breakdown"]["score"] += 1
        signals["breakdown"]["details"].append(f"距高点回撤 {drawdown:.1f}% > 30%")

    # 2. 跌破20日均线
    if len(df) >= 20:
        ma20 = df['收盘'].tail(20).mean()
        if latest['收盘'] < ma20:
            signals["breakdown"]["score"] += 1
            signals["breakdown"]["details"].append(f"收盘 {latest['收盘']:.2f} < MA20 {ma20:.2f}")

    # 3. 放量下跌
    if len(df) >= 20:
        avg_volume = df['成交额'].tail(20).mean()
        if latest['成交额'] > avg_volume and latest['涨跌幅'] < 0:
            signals["breakdown"]["score"] += 1
            signals["breakdown"]["details"].append("放量下跌")

    return signals


def determine_phase(signals: dict) -> dict:
    """
    根据信号判断当前阶段
    """
    # 优先级：破位 > 分歧 > 加速 > 主升
    if signals["breakdown"]["score"] >= 2:
        return {
            "phase": "BREAKDOWN",
            "phase_cn": "破位段",
            "score": signals["breakdown"]["score"],
            "details": signals["breakdown"]["details"],
            "action": "清仓",
            "risk": "极高"
        }
    elif signals["distribution"]["score"] >= 3:
        return {
            "phase": "DISTRIBUTION",
            "phase_cn": "分歧段",
            "score": signals["distribution"]["score"],
            "details": signals["distribution"]["details"],
            "action": "减仓 30-50%",
            "risk": "高"
        }
    elif signals["acceleration"]["score"] >= 3:
        return {
            "phase": "PARABOLIC_ACCELERATION",
            "phase_cn": "加速段",
            "score": signals["acceleration"]["score"],
            "details": signals["acceleration"]["details"],
            "action": "停止加仓，只管理",
            "risk": "高"
        }
    else:
        return {
            "phase": "MOMENTUM",
            "phase_cn": "主升段/观望",
            "score": signals["acceleration"]["score"],
            "details": signals["acceleration"]["details"],
            "action": "可轻仓参与",
            "risk": "中"
        }


def calculate_strength_score(data: dict) -> dict:
    """
    计算资金强度评分
    """
    score_details = []
    total_score = 0

    # 1. 成交额趋势 (30%)
    if "volume_ratio_20d" in data:
        vr = data["volume_ratio_20d"]
        volume_score = min(100, vr * 50)  # 2倍得100分
        score_details.append({
            "dimension": "成交额趋势",
            "weight": 0.30,
            "raw_value": f"VR={vr:.2f}",
            "score": volume_score
        })
        total_score += volume_score * 0.30

    # 2. 换手率 (25%)
    if data.get("avg_turnover_rate"):
        tr = data["avg_turnover_rate"]
        turnover_score = min(100, tr * 3.33)  # 30%得100分
        score_details.append({
            "dimension": "换手率",
            "weight": 0.25,
            "raw_value": f"{tr:.1f}%",
            "score": turnover_score
        })
        total_score += turnover_score * 0.25

    # 3. 价格斜率 (25%)
    if "return_10d" in data:
        slope = data["return_10d"] / 10  # 日均涨幅
        slope_score = min(100, max(0, slope * 10 + 50))
        score_details.append({
            "dimension": "价格斜率",
            "weight": 0.25,
            "raw_value": f"{slope:.2f}%/日",
            "score": slope_score
        })
        total_score += slope_score * 0.25

    # 4. 创新高频率 (20%) - 简化计算
    if "current_drawdown" in data:
        dd = data["current_drawdown"]
        high_score = max(0, 100 - dd * 5)  # 回撤越小分越高
        score_details.append({
            "dimension": "距新高",
            "weight": 0.20,
            "raw_value": f"回撤 {dd:.1f}%",
            "score": high_score
        })
        total_score += high_score * 0.20

    return {
        "total_score": round(total_score, 1),
        "details": score_details
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python fetch_data.py <股票代码>")
        print("示例: python fetch_data.py 00020")
        sys.exit(1)

    symbol = sys.argv[1]

    # 获取数据
    data = fetch_hk_stock_data(symbol)

    if "error" in data:
        print(json.dumps({"error": data["error"]}, ensure_ascii=False, indent=2))
        sys.exit(1)

    # 判断阶段
    phase = determine_phase(data["signals"])
    data["phase"] = phase

    # 计算资金强度
    strength = calculate_strength_score(data)
    data["strength_score"] = strength

    # 输出 JSON
    print(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
