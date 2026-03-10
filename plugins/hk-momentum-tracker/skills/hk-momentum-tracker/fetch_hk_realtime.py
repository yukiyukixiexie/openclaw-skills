#!/usr/bin/env python3
"""
HK Momentum Tracker - 港股数据抓取脚本
支持实时行情 + 历史数据
"""

import json
import sys
import time
import requests
from datetime import datetime

# 通用请求头
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://quote.eastmoney.com/"
}

# ============================================================
# 实时行情 API
# ============================================================

def fetch_realtime_eastmoney(symbol: str) -> dict:
    """
    东方财富实时行情 API（推荐，JSON格式）
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    url = f"https://push2.eastmoney.com/api/qt/stock/get"
    params = {
        "secid": f"116.{symbol}",
        "fields": "f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f116,f117,f162,f168,f169,f170,f171"
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        data = resp.json()

        if data.get("data") is None:
            return {"error": f"无法获取 {symbol} 的数据"}

        d = data["data"]

        return {
            "source": "eastmoney",
            "symbol": d.get("f57", symbol),
            "name": d.get("f58", ""),
            "current_price": d.get("f43", 0) / 1000,  # 价格单位是 0.001
            "high": d.get("f44", 0) / 1000,
            "low": d.get("f45", 0) / 1000,
            "open": d.get("f46", 0) / 1000,
            "prev_close": d.get("f60", 0) / 1000,
            "volume": d.get("f47", 0),
            "turnover": d.get("f48", 0),  # 成交额（元）
            "turnover_billion": d.get("f48", 0) / 1e8,  # 亿
            "volume_ratio": d.get("f50", 0) / 100,
            "change_amount": d.get("f169", 0) / 1000,
            "change_percent": d.get("f170", 0) / 100,
            "amplitude": d.get("f171", 0) / 100,
            "total_market_cap": d.get("f116", 0),
            "total_market_cap_billion": d.get("f116", 0) / 1e8,
            "float_market_cap": d.get("f117", 0),
            "float_market_cap_billion": d.get("f117", 0) / 1e8,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_realtime_tencent(symbol: str) -> dict:
    """
    腾讯财经实时行情 API
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    url = f"https://qt.gtimg.cn/q=r_hk{symbol}"

    try:
        resp = requests.get(url, timeout=10)
        content = resp.text

        # 解析格式: v_r_hkXXXXX="data~data~data~..."
        if '="' not in content:
            return {"error": f"无法获取 {symbol} 的数据"}

        data_str = content.split('="')[1].rstrip('";')
        fields = data_str.split('~')

        if len(fields) < 50:
            return {"error": "数据字段不完整"}

        return {
            "source": "tencent",
            "symbol": fields[2],
            "name": fields[1],
            "current_price": float(fields[3]) if fields[3] else 0,
            "prev_close": float(fields[4]) if fields[4] else 0,
            "open": float(fields[5]) if fields[5] else 0,
            "volume": float(fields[6]) if fields[6] else 0,
            "high": float(fields[33]) if fields[33] else 0,
            "low": float(fields[34]) if fields[34] else 0,
            "change_amount": float(fields[31]) if fields[31] else 0,
            "change_percent": float(fields[32]) if fields[32] else 0,
            "turnover": float(fields[37]) if fields[37] else 0,
            "turnover_billion": float(fields[37]) / 1e8 if fields[37] else 0,
            "amplitude": float(fields[43]) if fields[43] else 0,
            "total_market_cap_billion": float(fields[45]) if fields[45] else 0,
            "week_52_high": float(fields[48]) if fields[48] else 0,
            "week_52_low": float(fields[49]) if fields[49] else 0,
            "turnover_rate": float(fields[51]) if fields[51] else 0,
            "pe_ratio": float(fields[52]) if fields[52] else 0,
            "update_time": fields[30] if len(fields) > 30 else "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_realtime_sina(symbol: str) -> dict:
    """
    新浪财经实时行情 API
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    url = f"https://hq.sinajs.cn/list=rt_hk{symbol}"
    headers = {"Referer": "https://finance.sina.com.cn"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        content = resp.text

        # 解析格式: var hq_str_rt_hkXXXXX="data,data,data,..."
        if '="' not in content:
            return {"error": f"无法获取 {symbol} 的数据"}

        data_str = content.split('="')[1].rstrip('";')
        fields = data_str.split(',')

        if len(fields) < 15:
            return {"error": "数据字段不完整"}

        return {
            "source": "sina",
            "symbol": symbol,
            "name_en": fields[0],
            "name": fields[1],
            "open": float(fields[2]) if fields[2] else 0,
            "prev_close": float(fields[3]) if fields[3] else 0,
            "high": float(fields[4]) if fields[4] else 0,
            "low": float(fields[5]) if fields[5] else 0,
            "current_price": float(fields[6]) if fields[6] else 0,
            "change_amount": float(fields[7]) if fields[7] else 0,
            "change_percent": float(fields[8]) if fields[8] else 0,
            "turnover": float(fields[11]) if fields[11] else 0,
            "turnover_billion": float(fields[11]) / 1e8 if fields[11] else 0,
            "volume": float(fields[12]) if fields[12] else 0,
            "week_52_high": float(fields[15]) if fields[15] else 0,
            "week_52_low": float(fields[16]) if fields[16] else 0,
            "update_time": f"{fields[17]} {fields[18]}" if len(fields) > 18 else "",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_realtime(symbol: str, source: str = "eastmoney") -> dict:
    """
    获取实时行情（支持多数据源）

    Args:
        symbol: 股票代码
        source: 数据源 (eastmoney/tencent/sina)
    """
    if source == "eastmoney":
        return fetch_realtime_eastmoney(symbol)
    elif source == "tencent":
        return fetch_realtime_tencent(symbol)
    elif source == "sina":
        return fetch_realtime_sina(symbol)
    else:
        return {"error": f"不支持的数据源: {source}"}


# ============================================================
# 历史数据 API
# ============================================================

def fetch_history_yahoo(symbol: str, days: int = 250) -> dict:
    """
    Yahoo Finance 历史K线 API（最稳定，有成交量）
    """
    # Yahoo 使用不带前导零的格式，如 2513.HK
    symbol_clean = symbol.replace(".HK", "").replace(".hk", "").lstrip("0")
    symbol_padded = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    # 根据天数选择 range
    if days <= 30:
        range_str = "1mo"
    elif days <= 90:
        range_str = "3mo"
    elif days <= 180:
        range_str = "6mo"
    else:
        range_str = "1y"

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol_clean}.HK"
    params = {
        "interval": "1d",
        "range": range_str
    }

    # 带重试的请求（处理限速）
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 429:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                    continue
                return {"error": "Yahoo Finance 限速，请稍后重试"}
            if resp.status_code != 200:
                return {"error": f"Yahoo Finance HTTP {resp.status_code}"}
            data = resp.json()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
            return {"error": str(e)}

    try:
        result_data = data.get("chart", {}).get("result", [])
        if not result_data:
            return {"error": f"Yahoo Finance 无法获取 {symbol_padded} 的数据"}

        meta = result_data[0].get("meta", {})
        timestamps = result_data[0].get("timestamp", [])
        quote = result_data[0].get("indicators", {}).get("quote", [{}])[0]

        opens = quote.get("open", [])
        highs = quote.get("high", [])
        lows = quote.get("low", [])
        closes = quote.get("close", [])
        volumes = quote.get("volume", [])

        if not timestamps or not closes:
            return {"error": "历史数据为空"}

        from datetime import datetime as dt
        import numpy as np

        # 解析K线数据
        history = []
        for i, ts in enumerate(timestamps):
            if closes[i] is None:
                continue
            avg_price = (highs[i] + lows[i] + closes[i]) / 3 if highs[i] and lows[i] else closes[i]
            turnover = volumes[i] * avg_price if volumes[i] else 0

            history.append({
                "date": dt.fromtimestamp(ts).strftime("%Y-%m-%d"),
                "open": float(opens[i]) if opens[i] else 0,
                "close": float(closes[i]),
                "high": float(highs[i]) if highs[i] else 0,
                "low": float(lows[i]) if lows[i] else 0,
                "volume": float(volumes[i]) if volumes[i] else 0,
                "turnover": turnover,  # 估算成交额
                "amplitude": round((highs[i] - lows[i]) / opens[i] * 100, 2) if opens[i] and highs[i] and lows[i] else 0,
                "change_percent": round((closes[i] / opens[i] - 1) * 100, 2) if opens[i] else 0,
                "change_amount": round(closes[i] - opens[i], 2) if opens[i] else 0,
                "turnover_rate": 0  # Yahoo 不提供换手率
            })

        if not history:
            return {"error": "解析后历史数据为空"}

        closes_arr = [h["close"] for h in history]
        turnovers = [h["turnover"] for h in history]

        result = {
            "source": "yahoo",
            "symbol": symbol_padded,
            "name": meta.get("shortName", ""),
            "data_start": history[0]["date"],
            "data_end": history[-1]["date"],
            "trading_days": len(history),

            # 价格统计
            "current_price": history[-1]["close"],
            "history_high": max(h["high"] for h in history),
            "history_low": min(h["low"] for h in history),

            # 涨跌幅
            "return_total": (history[-1]["close"] / history[0]["close"] - 1) * 100,

            # 成交统计（估算）
            "total_turnover_billion": sum(turnovers) / 1e8,
            "avg_turnover_billion": np.mean(turnovers) / 1e8 if turnovers else 0,
            "max_turnover_billion": max(turnovers) / 1e8 if turnovers else 0,

            # 最近数据
            "recent_data": history[-20:] if len(history) >= 20 else history,

            # 成交额 Top 10
            "top_volume_days": sorted(history, key=lambda x: x["turnover"], reverse=True)[:10]
        }

        # 计算近期涨跌幅
        if len(history) >= 6:
            result["return_5d"] = (history[-1]["close"] / history[-6]["close"] - 1) * 100
        if len(history) >= 11:
            result["return_10d"] = (history[-1]["close"] / history[-11]["close"] - 1) * 100
        if len(history) >= 21:
            result["return_20d"] = (history[-1]["close"] / history[-21]["close"] - 1) * 100
            result["ma20"] = np.mean([h["close"] for h in history[-20:]])
            result["volume_ratio_20d"] = history[-1]["turnover"] / np.mean([h["turnover"] for h in history[-20:]]) if np.mean([h["turnover"] for h in history[-20:]]) > 0 else 0

        # 计算阶段信号
        result["signals"] = calculate_signals_from_history(history)

        return result

    except Exception as e:
        return {"error": str(e)}


def fetch_history_tencent(symbol: str, days: int = 250) -> dict:
    """
    腾讯历史K线 API（备选方案，更稳定）
    成交额通过 volume * avg_price 估算
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {
        "param": f"hk{symbol},day,,,{days},qfq"
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()

        if data.get("code") != 0:
            return {"error": f"腾讯 API 返回错误: {data.get('msg', '未知错误')}"}

        stock_data = data.get("data", {}).get(f"hk{symbol}", {})
        klines = stock_data.get("day", []) or stock_data.get("qfqday", [])

        if not klines:
            return {"error": f"无法获取 {symbol} 的历史数据"}

        # 解析K线数据
        # 格式: [日期, 开盘, 收盘, 最高, 最低, 成交量]
        history = []
        for line in klines:
            if len(line) >= 6:
                open_p = float(line[1])
                close_p = float(line[2])
                high_p = float(line[3])
                low_p = float(line[4])
                volume = float(line[5])

                # 估算成交额 = 成交量 × 均价
                avg_price = (high_p + low_p + close_p) / 3 if high_p and low_p else close_p
                turnover = volume * avg_price

                history.append({
                    "date": line[0],
                    "open": open_p,
                    "close": close_p,
                    "high": high_p,
                    "low": low_p,
                    "volume": volume,
                    "turnover": turnover,  # 估算成交额
                    "amplitude": round((high_p - low_p) / open_p * 100, 2) if open_p > 0 else 0,
                    "change_percent": round((close_p / open_p - 1) * 100, 2) if open_p > 0 else 0,
                    "change_amount": round(close_p - open_p, 2),
                    "turnover_rate": 0  # 腾讯 API 不提供换手率
                })

        if not history:
            return {"error": "历史数据为空"}

        import numpy as np

        closes = [h["close"] for h in history]
        turnovers = [h["turnover"] for h in history]

        result = {
            "source": "tencent",
            "symbol": symbol,
            "name": "",
            "data_start": history[0]["date"],
            "data_end": history[-1]["date"],
            "trading_days": len(history),

            # 价格统计
            "current_price": history[-1]["close"],
            "history_high": max(h["high"] for h in history),
            "history_low": min(h["low"] for h in history),

            # 涨跌幅
            "return_total": (history[-1]["close"] / history[0]["close"] - 1) * 100,

            # 成交统计（估算）
            "total_turnover_billion": sum(turnovers) / 1e8,
            "avg_turnover_billion": np.mean(turnovers) / 1e8 if turnovers else 0,
            "max_turnover_billion": max(turnovers) / 1e8 if turnovers else 0,

            # 最近数据
            "recent_data": history[-20:] if len(history) >= 20 else history,

            # 成交额 Top 10
            "top_volume_days": sorted(history, key=lambda x: x["turnover"], reverse=True)[:10]
        }

        # 计算近期涨跌幅
        if len(history) >= 6:
            result["return_5d"] = (history[-1]["close"] / history[-6]["close"] - 1) * 100
        if len(history) >= 11:
            result["return_10d"] = (history[-1]["close"] / history[-11]["close"] - 1) * 100
        if len(history) >= 21:
            result["return_20d"] = (history[-1]["close"] / history[-21]["close"] - 1) * 100
            result["ma20"] = np.mean([h["close"] for h in history[-20:]])
            avg_turnover_20d = np.mean([h["turnover"] for h in history[-20:]])
            result["volume_ratio_20d"] = history[-1]["turnover"] / avg_turnover_20d if avg_turnover_20d > 0 else 0

        # 计算阶段信号
        result["signals"] = calculate_signals_from_history(history)

        return result

    except Exception as e:
        return {"error": str(e)}


def fetch_history_eastmoney(symbol: str, days: int = 120, max_retries: int = 3) -> dict:
    """
    东方财富历史K线 API（带重试机制）
    """
    symbol = symbol.replace(".HK", "").replace(".hk", "").zfill(5)

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": f"116.{symbol}",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101",  # 日K
        "fqt": "1",    # 前复权
        "end": "20500101",
        "lmt": days
    }

    # 带重试的请求
    data = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=30)
            data = resp.json()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1 + attempt)  # 递增等待
                continue
            return {"error": f"请求失败（重试 {max_retries} 次）: {str(e)}"}

    if data is None:
        return {"error": "请求返回空数据"}

    try:

        if data.get("data") is None or data["data"].get("klines") is None:
            return {"error": f"无法获取 {symbol} 的历史数据"}

        klines = data["data"]["klines"]

        # 解析K线数据
        # 格式: 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率
        history = []
        for line in klines:
            fields = line.split(',')
            if len(fields) >= 11:
                history.append({
                    "date": fields[0],
                    "open": float(fields[1]),
                    "close": float(fields[2]),
                    "high": float(fields[3]),
                    "low": float(fields[4]),
                    "volume": float(fields[5]),
                    "turnover": float(fields[6]),
                    "amplitude": float(fields[7]),
                    "change_percent": float(fields[8]),
                    "change_amount": float(fields[9]),
                    "turnover_rate": float(fields[10])
                })

        if not history:
            return {"error": "历史数据为空"}

        # 计算统计指标
        closes = [h["close"] for h in history]
        turnovers = [h["turnover"] for h in history]
        turnover_rates = [h["turnover_rate"] for h in history]

        import numpy as np

        result = {
            "source": "eastmoney",
            "symbol": symbol,
            "name": data["data"].get("name", ""),
            "data_start": history[0]["date"],
            "data_end": history[-1]["date"],
            "trading_days": len(history),

            # 价格统计
            "current_price": history[-1]["close"],
            "history_high": max(h["high"] for h in history),
            "history_low": min(h["low"] for h in history),

            # 涨跌幅
            "return_total": (history[-1]["close"] / history[0]["close"] - 1) * 100,

            # 成交统计
            "total_turnover_billion": sum(turnovers) / 1e8,
            "avg_turnover_billion": np.mean(turnovers) / 1e8,
            "turnover_p50": np.percentile(turnovers, 50) / 1e8,
            "turnover_p75": np.percentile(turnovers, 75) / 1e8,
            "turnover_p90": np.percentile(turnovers, 90) / 1e8,
            "max_turnover_billion": max(turnovers) / 1e8,

            # 换手率统计
            "cumulative_turnover_rate": sum(turnover_rates),
            "avg_turnover_rate": np.mean(turnover_rates),
            "turnover_rate_p90": np.percentile(turnover_rates, 90),
            "max_turnover_rate": max(turnover_rates),

            # 最近数据
            "recent_data": history[-20:] if len(history) >= 20 else history,

            # 成交额 Top 10
            "top_volume_days": sorted(history, key=lambda x: x["turnover"], reverse=True)[:10]
        }

        # 计算近期涨跌幅
        if len(history) >= 6:
            result["return_5d"] = (history[-1]["close"] / history[-6]["close"] - 1) * 100
        if len(history) >= 11:
            result["return_10d"] = (history[-1]["close"] / history[-11]["close"] - 1) * 100
        if len(history) >= 21:
            result["return_20d"] = (history[-1]["close"] / history[-21]["close"] - 1) * 100
            result["ma20"] = np.mean([h["close"] for h in history[-20:]])
            result["volume_ratio_20d"] = history[-1]["turnover"] / np.mean([h["turnover"] for h in history[-20:]])

        # 计算阶段信号
        result["signals"] = calculate_signals_from_history(history)

        return result

    except Exception as e:
        return {"error": str(e)}


def calculate_signals_from_history(history: list) -> dict:
    """
    根据历史数据计算阶段信号
    """
    signals = {
        "acceleration": {"score": 0, "details": []},
        "distribution": {"score": 0, "details": []},
        "breakdown": {"score": 0, "details": []}
    }

    if len(history) < 20:
        return signals

    latest = history[-1]
    closes = [h["close"] for h in history]

    # ========== 加速段信号 ==========
    # 1. 10日涨幅 > 120%
    if len(history) >= 11:
        return_10d = (history[-1]["close"] / history[-11]["close"] - 1) * 100
        if return_10d > 120:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"10日涨幅 {return_10d:.1f}% > 120%")

    # 2. 5日涨幅 > 50%
    if len(history) >= 6:
        return_5d = (history[-1]["close"] / history[-6]["close"] - 1) * 100
        if return_5d > 50:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"5日涨幅 {return_5d:.1f}% > 50%")

    # 3. 连续创新高
    recent_closes = [h["close"] for h in history[-10:]]
    consecutive_highs = 0
    running_max = 0
    for price in recent_closes:
        if price > running_max:
            consecutive_highs += 1
            running_max = price
        else:
            consecutive_highs = 0
    if consecutive_highs >= 3:
        signals["acceleration"]["score"] += 1
        signals["acceleration"]["details"].append(f"连续 {consecutive_highs} 天创新高")

    # 4. 成交额创60日新高
    turnovers = [h["turnover"] for h in history[-60:]]
    if latest["turnover"] >= max(turnovers):
        signals["acceleration"]["score"] += 1
        signals["acceleration"]["details"].append("成交额创 60 日新高")

    # 5. 20日回撤 < 15%
    if len(history) >= 20:
        recent_peak = max(h["close"] for h in history[-20:])
        drawdown_20d = (recent_peak - latest["close"]) / recent_peak * 100
        if drawdown_20d < 15:
            signals["acceleration"]["score"] += 1
            signals["acceleration"]["details"].append(f"20日回撤 {drawdown_20d:.1f}% < 15%")

    # ========== 分歧段信号 ==========
    # 1. 振幅扩大
    import numpy as np
    avg_amplitude = np.mean([h["amplitude"] for h in history[-20:]])
    if latest["amplitude"] > avg_amplitude * 1.5:
        signals["distribution"]["score"] += 1
        signals["distribution"]["details"].append(f"振幅 {latest['amplitude']:.1f}% > 均值 {avg_amplitude:.1f}% × 1.5")

    # 2. 长上影线
    if latest["high"] > latest["low"]:
        upper_shadow = (latest["high"] - latest["close"]) / (latest["high"] - latest["low"])
        if upper_shadow > 0.6:
            signals["distribution"]["score"] += 1
            signals["distribution"]["details"].append(f"上影线比率 {upper_shadow:.2f} > 0.6")

    # 3. 量增价滞
    if latest["turnover"] >= max(h["turnover"] for h in history) and abs(latest["change_percent"]) < 5:
        signals["distribution"]["score"] += 1
        signals["distribution"]["details"].append("成交额新高但涨幅 < 5%（量增价滞）")

    # 4. 收盘弱势
    weak_count = sum(1 for h in history[-2:] if h["close"] < h["high"] * 0.95)
    if weak_count >= 2:
        signals["distribution"]["score"] += 1
        signals["distribution"]["details"].append("连续 2 日收盘弱势")

    # ========== 破位段信号 ==========
    # 1. 距高点回撤 > 30%
    peak = max(closes)
    drawdown = (peak - latest["close"]) / peak * 100
    if drawdown > 30:
        signals["breakdown"]["score"] += 1
        signals["breakdown"]["details"].append(f"距高点回撤 {drawdown:.1f}% > 30%")

    # 2. 跌破20日均线
    if len(history) >= 20:
        ma20 = np.mean([h["close"] for h in history[-20:]])
        if latest["close"] < ma20:
            signals["breakdown"]["score"] += 1
            signals["breakdown"]["details"].append(f"收盘 {latest['close']:.2f} < MA20 {ma20:.2f}")

    # 3. 放量下跌
    if len(history) >= 20:
        avg_volume = np.mean([h["turnover"] for h in history[-20:]])
        if latest["turnover"] > avg_volume and latest["change_percent"] < 0:
            signals["breakdown"]["score"] += 1
            signals["breakdown"]["details"].append("放量下跌")

    return signals


def determine_phase(signals: dict) -> dict:
    """
    根据信号判断阶段
    """
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


def fetch_history(symbol: str, days: int = 250, source: str = "auto") -> dict:
    """
    获取历史数据（自动选择数据源或指定）

    Args:
        symbol: 股票代码
        days: 获取天数
        source: 数据源 (auto/yahoo/eastmoney/tencent)
    """
    if source == "yahoo":
        return fetch_history_yahoo(symbol, days)
    elif source == "tencent":
        return fetch_history_tencent(symbol, days)
    elif source == "eastmoney":
        return fetch_history_eastmoney(symbol, days)
    else:
        # auto: 优先 Yahoo（最稳定），失败则用腾讯
        result = fetch_history_yahoo(symbol, days)
        if "error" in result:
            print(f"Yahoo Finance 失败，切换到腾讯...", file=sys.stderr)
            result = fetch_history_tencent(symbol, days)
        return result


# ============================================================
# 批量扫描功能（发现投资机会）
# ============================================================

def fetch_hk_ai_stocks_realtime() -> list:
    """
    批量获取港股 AI 概念股实时行情
    用于发现投资机会
    """
    # AI 概念股列表
    ai_stocks = [
        ("02513", "智谱"),
        ("00020", "商汤-W"),
        ("09888", "百度集团-SW"),
        ("09618", "京东集团-SW"),
        ("03690", "美团-W"),
        ("09988", "阿里巴巴-SW"),
        ("00700", "腾讯控股"),
        ("09961", "携程集团-S"),
        ("01024", "快手-W"),
        ("09626", "哔哩哔哩-SW"),
        ("02518", "汽车之家-S"),
        ("06060", "众安在线"),
        ("09999", "网易-S"),
        ("02382", "舜宇光学科技"),
        ("00981", "中芯国际"),
    ]

    results = []
    for symbol, name in ai_stocks:
        try:
            data = fetch_realtime_eastmoney(symbol)
            if "error" not in data:
                data["name"] = name  # 确保名称正确
                results.append(data)
        except:
            continue
        time.sleep(0.1)  # 避免请求过快

    # 按涨跌幅排序
    results.sort(key=lambda x: x.get("change_percent", 0), reverse=True)
    return results


def scan_momentum_stocks() -> dict:
    """
    扫描有动量特征的股票
    """
    stocks = fetch_hk_ai_stocks_realtime()

    # 筛选标准
    hot_stocks = []  # 今日活跃
    breakout_stocks = []  # 放量突破
    weak_stocks = []  # 弱势股票

    for stock in stocks:
        change = stock.get("change_percent", 0)
        amplitude = stock.get("amplitude", 0)
        volume_ratio = stock.get("volume_ratio", 0)
        turnover = stock.get("turnover_billion", 0)

        info = {
            "symbol": stock.get("symbol"),
            "name": stock.get("name"),
            "price": stock.get("current_price"),
            "change": change,
            "amplitude": amplitude,
            "volume_ratio": volume_ratio,
            "turnover": turnover
        }

        # 今日活跃：涨幅 > 5% 或 振幅 > 10%
        if change > 5 or amplitude > 10:
            hot_stocks.append(info)

        # 放量突破：量比 > 1.5 且涨幅 > 3%
        if volume_ratio > 1.5 and change > 3:
            breakout_stocks.append(info)

        # 弱势：跌幅 > 5%
        if change < -5:
            weak_stocks.append(info)

    return {
        "scan_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_stocks": len(stocks),
        "hot_stocks": hot_stocks,
        "breakout_stocks": breakout_stocks,
        "weak_stocks": weak_stocks,
        "all_stocks": [{
            "symbol": s.get("symbol"),
            "name": s.get("name"),
            "price": s.get("current_price"),
            "change": s.get("change_percent"),
            "turnover": s.get("turnover_billion")
        } for s in stocks]
    }


# ============================================================
# 主函数
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("  python fetch_hk_realtime.py <股票代码> [--realtime] [--history] [--source=eastmoney|tencent|sina]")
        print("  python fetch_hk_realtime.py --scan                  # 扫描 AI 概念股")
        print("")
        print("示例:")
        print("  python fetch_hk_realtime.py 02513               # 默认：实时+历史")
        print("  python fetch_hk_realtime.py 02513 --realtime    # 仅实时行情")
        print("  python fetch_hk_realtime.py 02513 --history     # 仅历史数据")
        print("  python fetch_hk_realtime.py 02513 --source=tencent  # 指定数据源")
        print("  python fetch_hk_realtime.py --scan              # 扫描发现投资机会")
        sys.exit(1)

    # 扫描模式
    if sys.argv[1] == "--scan":
        print("正在扫描港股 AI 概念股...", file=sys.stderr)
        result = scan_momentum_stocks()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    symbol = sys.argv[1]

    # 解析参数
    only_realtime = "--realtime" in sys.argv
    only_history = "--history" in sys.argv

    source = "eastmoney"
    for arg in sys.argv:
        if arg.startswith("--source="):
            source = arg.split("=")[1]

    result = {"symbol": symbol}

    # 获取实时数据
    if not only_history:
        print(f"正在获取 {symbol} 的实时行情 (来源: {source})...", file=sys.stderr)
        realtime = fetch_realtime(symbol, source)
        result["realtime"] = realtime

    # 获取历史数据
    if not only_realtime:
        print(f"正在获取 {symbol} 的历史数据...", file=sys.stderr)
        history = fetch_history(symbol, days=250, source="auto")
        result["history"] = history

        # 判断阶段
        if "signals" in history:
            result["phase"] = determine_phase(history["signals"])

    # 输出 JSON
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
