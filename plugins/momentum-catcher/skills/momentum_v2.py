#!/usr/bin/env python3
"""
Momentum Catcher v2 - 增强版资金动量捕捉框架

新增功能：
- 美股支持（Alpaca 实时数据 + 交易）
- 统一数据源自动切换
- Finnhub Earnings 数据
- quantstats 绩效分析

Usage:
    # 港股分析
    python momentum_v2.py --ticker 02513.HK

    # 美股分析
    python momentum_v2.py --ticker AAPL --market us

    # 带绩效回测
    python momentum_v2.py --ticker NVDA --market us --backtest
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

# 添加 shared 模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("错误: 请安装 pandas numpy: pip install pandas numpy")
    sys.exit(1)

# 导入统一数据接口
try:
    from unified_data import get_realtime, get_history, get_batch_realtime, get_info
    from unified_data import get_earnings_calendar, get_analyst_ratings
    UNIFIED_DATA_AVAILABLE = True
except ImportError:
    UNIFIED_DATA_AVAILABLE = False
    print("提示: unified_data 模块不可用，使用内置数据源")

# 导入绩效分析
try:
    from performance import analyze_performance, generate_html_report, generate_monthly_returns
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    print("提示: performance 模块不可用，回测功能受限")

# 导入 Alpaca 交易
try:
    from alpaca_trading import get_account, get_positions, submit_order
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

# 导入 Finnhub
try:
    from finnhub_data import get_earnings_surprises, get_price_target, get_company_news
    FINNHUB_AVAILABLE = True
except ImportError:
    FINNHUB_AVAILABLE = False

# 导入 VectorBT 增强分析
try:
    from vectorbt_enhanced import (
        get_turnover_signals,
        calculate_volume_profile,
        find_chip_concentration,
        get_support_resistance,
        get_squeeze_signals,
        detect_top_signals,
        run_vectorbt_backtest,
        VBT_AVAILABLE
    )
    VECTORBT_AVAILABLE = VBT_AVAILABLE
except ImportError:
    VECTORBT_AVAILABLE = False


# ============================================================
# 数据获取（使用统一接口）
# ============================================================

def fetch_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    market: str = "auto",
    days: int = 180
) -> pd.DataFrame:
    """
    获取历史数据（自动选择最佳数据源）

    Args:
        ticker: 股票代码
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期
        market: 市场 (auto/us/hk/a)
        days: 历史天数（当 start_date 为 None 时使用）

    Returns:
        DataFrame with OHLCV data
    """
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    if UNIFIED_DATA_AVAILABLE:
        result = get_history(ticker, start_date, end_date)
        if isinstance(result, pd.DataFrame) and not result.empty:
            return _normalize_columns(result)
        elif isinstance(result, dict) and "error" in result:
            print(f"统一接口失败: {result['error']}")

    # 降级到内置方法
    return _fetch_fallback(ticker, start_date, market)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """标准化列名"""
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # 确保有必要的列
    required = ['open', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            # 尝试找到类似的列
            for c in df.columns:
                if col in c.lower():
                    df[col] = df[c]
                    break

    return df


def _fetch_fallback(ticker: str, start_date: str, market: str) -> pd.DataFrame:
    """降级数据获取"""
    try:
        import yfinance as yf

        # 标准化 ticker
        if market == "hk" and not ticker.endswith(".HK"):
            ticker = ticker.zfill(4) + ".HK"

        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)

        if df.empty:
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        return df[['open', 'high', 'low', 'close', 'volume']]

    except Exception as e:
        print(f"yfinance 获取失败: {e}")
        return pd.DataFrame()


def fetch_realtime_data(ticker: str) -> Dict:
    """获取实时行情"""
    if UNIFIED_DATA_AVAILABLE:
        return get_realtime(ticker)

    # 降级
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "symbol": ticker,
            "price": info.get("currentPrice") or info.get("regularMarketPrice"),
            "change_pct": info.get("regularMarketChangePercent"),
            "volume": info.get("volume"),
        }
    except:
        return {"error": "无法获取实时数据"}


# ============================================================
# 技术指标计算
# ============================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算所有技术指标"""
    if df.empty:
        return df

    df = df.copy()

    # 移动平均线
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma10'] = df['close'].rolling(10).mean()
    df['ma20'] = df['close'].rolling(20).mean()
    df['ma60'] = df['close'].rolling(60).mean()

    # EMA
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['macd'] = df['ema12'] - df['ema26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 布林带
    df['boll_mid'] = df['close'].rolling(20).mean()
    df['boll_std'] = df['close'].rolling(20).std()
    df['boll_upper'] = df['boll_mid'] + 2 * df['boll_std']
    df['boll_lower'] = df['boll_mid'] - 2 * df['boll_std']

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

    # 20日高点
    df['high_20d'] = df['high'].rolling(20).max()

    # 成交量指标
    df['vol_ma20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_ma20']

    # 收益率
    df['returns'] = df['close'].pct_change()
    df['returns_5d'] = df['close'].pct_change(5)
    df['returns_20d'] = df['close'].pct_change(20)

    return df


# ============================================================
# 信号评分
# ============================================================

def calculate_momentum_score(df: pd.DataFrame) -> Dict:
    """计算动量评分"""
    if df.empty or len(df) < 20:
        return {"error": "数据不足"}

    latest = df.iloc[-1]
    scores = {
        "date": str(df.index[-1].date()),
        "close": round(latest['close'], 2),
        "signals": [],
    }

    # 1. 价格突破20日高点
    price_breakout = latest['close'] > latest['high_20d'] * 0.99
    scores["signals"].append({
        "name": "价格突破20日高点",
        "value": f"{latest['close']:.2f} vs {latest['high_20d']:.2f}",
        "passed": price_breakout
    })

    # 2. MACD 金叉且零轴上方
    macd_golden = latest['macd'] > latest['macd_signal'] and latest['macd'] > 0
    scores["signals"].append({
        "name": "MACD金叉(零轴上)",
        "value": f"MACD={latest['macd']:.3f}, Signal={latest['macd_signal']:.3f}",
        "passed": macd_golden
    })

    # 3. RSI 强势区
    rsi_strong = 60 < latest['rsi'] < 80 if pd.notna(latest['rsi']) else False
    scores["signals"].append({
        "name": "RSI强势区(60-80)",
        "value": f"{latest['rsi']:.1f}" if pd.notna(latest['rsi']) else "N/A",
        "passed": rsi_strong
    })

    # 4. 均线多头排列
    ma_bullish = (
        pd.notna(latest['ma5']) and
        pd.notna(latest['ma10']) and
        pd.notna(latest['ma20']) and
        latest['ma5'] > latest['ma10'] > latest['ma20']
    )
    scores["signals"].append({
        "name": "均线多头排列",
        "value": f"MA5={latest['ma5']:.2f}, MA10={latest['ma10']:.2f}, MA20={latest['ma20']:.2f}",
        "passed": ma_bullish
    })

    # 5. 成交量异动
    vol_surge = latest['vol_ratio'] > 1.5 if pd.notna(latest['vol_ratio']) else False
    scores["signals"].append({
        "name": "成交量异动(>1.5x)",
        "value": f"{latest['vol_ratio']:.2f}x" if pd.notna(latest['vol_ratio']) else "N/A",
        "passed": vol_surge
    })

    # 计算总分
    scores["momentum_score"] = sum(1 for s in scores["signals"] if s["passed"])

    # 退出信号
    exit_signals = []
    exit_weight = 0

    # RSI 超买回落
    if pd.notna(latest['rsi']):
        recent_max_rsi = df['rsi'].iloc[-10:].max() if len(df) > 10 else 100
        if recent_max_rsi > 80 and latest['rsi'] < 70:
            exit_signals.append({"name": "RSI超买回落", "weight": 0.20})
            exit_weight += 0.20

    # MACD 死叉
    if latest['macd'] < latest['macd_signal']:
        exit_signals.append({"name": "MACD死叉", "weight": 0.15})
        exit_weight += 0.15

    # 跌破 MA20
    if pd.notna(latest['ma20']) and latest['close'] < latest['ma20']:
        exit_signals.append({"name": "跌破20日均线", "weight": 0.25})
        exit_weight += 0.25

    scores["exit_signals"] = exit_signals
    scores["exit_weight"] = round(exit_weight, 2)

    return scores


def get_entry_signal(momentum_score: int, event_score: float = 4.0, capital_score: int = 3) -> Dict:
    """生成入场信号"""
    if momentum_score < 3:
        return {"signal": "HOLD", "description": "动量不足，观望", "position": 0}

    if event_score >= 4 and capital_score >= 4:
        return {"signal": "STRONG_BUY", "description": "事件强+资金极强", "position": 100}
    elif event_score >= 4 and capital_score >= 3:
        return {"signal": "BUY", "description": "事件强+资金强", "position": 80}
    elif event_score >= 3 and capital_score >= 4:
        return {"signal": "BUY", "description": "事件中+资金极强", "position": 60}
    elif event_score >= 3 and capital_score >= 3:
        return {"signal": "BUY", "description": "事件中+资金强", "position": 40}
    else:
        return {"signal": "HOLD", "description": "条件不足", "position": 0}


# ============================================================
# 美股增强功能
# ============================================================

def get_us_stock_context(ticker: str) -> Dict:
    """获取美股额外上下文（Earnings、分析师评级等）"""
    context = {}

    if FINNHUB_AVAILABLE:
        # Earnings Surprise
        try:
            surprises = get_earnings_surprises(ticker)
            if isinstance(surprises, list) and surprises:
                context["earnings_surprises"] = surprises[:4]
        except:
            pass

        # 分析师目标价
        try:
            target = get_price_target(ticker)
            if isinstance(target, dict) and "error" not in target:
                context["price_target"] = target
        except:
            pass

        # 最近新闻
        try:
            news = get_company_news(ticker, days=7)
            if isinstance(news, list):
                context["recent_news"] = news[:5]
        except:
            pass

    # 分析师评级
    if UNIFIED_DATA_AVAILABLE:
        try:
            ratings = get_analyst_ratings(ticker)
            if isinstance(ratings, dict) and "error" not in ratings:
                context["analyst_ratings"] = ratings
        except:
            pass

    return context


# ============================================================
# 绩效回测
# ============================================================

def run_backtest(df: pd.DataFrame, ticker: str) -> Dict:
    """运行简单回测"""
    if not PERFORMANCE_AVAILABLE:
        return {"error": "performance 模块不可用"}

    if 'returns' not in df.columns:
        df = calculate_indicators(df)

    returns = df['returns'].dropna()

    if len(returns) < 20:
        return {"error": "数据不足"}

    # 使用 quantstats 分析
    report = analyze_performance(returns)

    # 生成 HTML 报告
    output_file = f"{ticker.replace('.', '_')}_performance.html"
    try:
        generate_html_report(returns, benchmark="SPY", output_file=output_file)
        report["html_report"] = output_file
    except:
        pass

    return report


# ============================================================
# 报告生成
# ============================================================

def generate_report(ticker: str, df: pd.DataFrame, scores: Dict, context: Dict = None) -> str:
    """生成完整分析报告"""
    lines = [
        f"\n{'='*60}",
        f"[{ticker}] 动量分析报告 (v2)",
        f"{'='*60}",
        f"分析日期: {scores.get('date', 'N/A')}",
        f"当前价格: {scores.get('close', 'N/A')}",
        "",
    ]

    # 动量信号
    lines.append("-" * 40)
    lines.append("【动量信号】")
    lines.append("-" * 40)

    for sig in scores.get("signals", []):
        status = "✅" if sig["passed"] else "❌"
        lines.append(f"  {sig['name']}: {sig['value']} {status}")

    momentum_score = scores.get("momentum_score", 0)
    lines.append(f"\n  动量得分: {momentum_score}/5")

    # 入场信号
    entry = get_entry_signal(momentum_score)
    lines.append(f"\n  入场信号: {entry['signal']}")
    lines.append(f"  说明: {entry['description']}")
    if entry['position'] > 0:
        lines.append(f"  建议仓位: {entry['position']}%")

    # 退出信号
    lines.append("")
    lines.append("-" * 40)
    lines.append("【退出信号】")
    lines.append("-" * 40)

    exit_signals = scores.get("exit_signals", [])
    if exit_signals:
        for sig in exit_signals:
            lines.append(f"  ⚠️ {sig['name']}: 权重 {sig['weight']:.0%}")
    else:
        lines.append("  无退出信号触发")

    lines.append(f"\n  退出权重合计: {scores.get('exit_weight', 0):.0%}")

    # 美股额外信息
    if context:
        lines.append("")
        lines.append("-" * 40)
        lines.append("【美股额外信息】")
        lines.append("-" * 40)

        # 分析师评级
        if "analyst_ratings" in context:
            r = context["analyst_ratings"]
            lines.append(f"  分析师评级: Buy={r.get('buy', 0)}, Hold={r.get('hold', 0)}, Sell={r.get('sell', 0)}")

        # 目标价
        if "price_target" in context:
            t = context["price_target"]
            lines.append(f"  目标价: ${t.get('target_mean', 'N/A')} (区间: ${t.get('target_low', 'N/A')}-${t.get('target_high', 'N/A')})")

        # Earnings Surprise
        if "earnings_surprises" in context:
            lines.append("  最近 Earnings:")
            for e in context["earnings_surprises"][:2]:
                surprise_pct = e.get("surprise_pct", 0) or 0
                lines.append(f"    {e.get('period', 'N/A')}: EPS {e.get('actual', 'N/A')} vs {e.get('estimate', 'N/A')} ({surprise_pct:+.1f}%)")

    # 关键价位
    lines.append("")
    lines.append("-" * 40)
    lines.append("【关键价位】")
    lines.append("-" * 40)

    latest = df.iloc[-1]
    if pd.notna(latest.get('ma10')):
        lines.append(f"  MA10 (短期支撑): {latest['ma10']:.2f}")
    if pd.notna(latest.get('ma20')):
        lines.append(f"  MA20 (中期支撑): {latest['ma20']:.2f}")
    if pd.notna(latest.get('high_20d')):
        lines.append(f"  20日高点: {latest['high_20d']:.2f}")
    if pd.notna(latest.get('atr')):
        stop_loss = latest['close'] - 2 * latest['atr']
        lines.append(f"  建议止损 (2ATR): {stop_loss:.2f}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# VectorBT 增强分析
# ============================================================

def run_enhanced_analysis(df: pd.DataFrame, ticker: str, float_shares: float = None) -> Dict:
    """运行 VectorBT 增强分析"""
    if not VECTORBT_AVAILABLE:
        return {"error": "vectorbt 模块不可用"}

    results = {}

    # 1. 换手率信号
    try:
        turnover = get_turnover_signals(df)
        results["turnover"] = turnover
    except Exception as e:
        results["turnover"] = {"error": str(e)}

    # 2. 筹码分布
    try:
        vol_profile = calculate_volume_profile(df, lookback=60)
        chip_zones = find_chip_concentration(vol_profile)
        current_price = df['close'].iloc[-1]
        sr = get_support_resistance(vol_profile, current_price)
        results["chip_distribution"] = {
            "zones": chip_zones,
            "support": sr.get("support"),
            "resistance": sr.get("resistance"),
            "main_cost_range": chip_zones[0]['price_range'] if chip_zones else "N/A"
        }
    except Exception as e:
        results["chip_distribution"] = {"error": str(e)}

    # 3. 供给受限检测（小盘股）
    try:
        squeeze = get_squeeze_signals(df)
        results["supply_squeeze"] = squeeze
    except Exception as e:
        results["supply_squeeze"] = {"error": str(e)}

    # 4. 综合顶部信号
    try:
        top = detect_top_signals(df)
        results["top_signals"] = top
    except Exception as e:
        results["top_signals"] = {"error": str(e)}

    return results


def run_vbt_backtest(df: pd.DataFrame, strategy: str = "momentum") -> Dict:
    """运行 VectorBT 回测"""
    if not VECTORBT_AVAILABLE:
        return {"error": "vectorbt 模块不可用"}

    try:
        return run_vectorbt_backtest(df, strategy=strategy)
    except Exception as e:
        return {"error": str(e)}


def print_enhanced_report(results: Dict):
    """打印增强分析报告"""
    print("\n" + "=" * 60)
    print("【VectorBT 增强分析】")
    print("=" * 60)

    # 换手率分析
    if "turnover" in results and "error" not in results["turnover"]:
        t = results["turnover"]
        print("\n[换手率分析]")
        current = t.get('current_turnover_relative')
        if current and isinstance(current, (int, float)):
            print(f"  相对换手率: {current:.2f}x")
        rank = t.get('turnover_rank')
        if rank and isinstance(rank, (int, float)):
            print(f"  历史排名: {rank*100:.0f}%")
        print(f"  信号: {t.get('signal', 'N/A')}")
        if t.get("is_extreme_high"):
            print(f"  ⚠️ 换手率极值! 减仓信号")
        if t.get("is_extreme_low"):
            print(f"  📉 地量信号! 可能是建仓时机")

    # 筹码分布
    if "chip_distribution" in results and "error" not in results["chip_distribution"]:
        c = results["chip_distribution"]
        print("\n[筹码分布]")
        print(f"  主要成本区: {c.get('main_cost_range', 'N/A')}")
        support = c.get('support')
        resistance = c.get('resistance')
        if support and isinstance(support, (int, float)):
            print(f"  支撑位: {support:.2f}")
        if resistance and isinstance(resistance, (int, float)):
            print(f"  阻力位: {resistance:.2f}")
        zones = c.get('zones', [])
        if zones:
            print("  筹码密集区:")
            for i, zone in enumerate(zones[:3], 1):
                print(f"    {i}. {zone['price_range']} ({zone['volume_pct']:.1f}%)")

    # 供给受限
    if "supply_squeeze" in results and "error" not in results["supply_squeeze"]:
        s = results["supply_squeeze"]
        print("\n[供给受限检测]")
        print(f"  当日振幅: {s.get('current_amplitude', 0):.1f}%")
        print(f"  成交量倍数: {s.get('current_vol_ratio', 0):.1f}x")
        print(f"  挤压排名: {s.get('squeeze_rank', 0)*100:.0f}%")
        print(f"  近20日挤压天数: {s.get('squeeze_days_20d', 0)}")
        print(f"  信号: {s.get('signal', 'N/A')}")

    # 顶部信号
    if "top_signals" in results and "error" not in results["top_signals"]:
        top = results["top_signals"]
        if top.get("top_signals"):
            print("\n[顶部信号警告]")
            for sig in top["top_signals"]:
                print(f"  ⚠️ {sig['signal']}: {sig['description']}")
            print(f"  顶部信号权重: {top.get('top_weight', 0):.0%}")
        else:
            print("\n[顶部信号检测]")
            print(f"  无明显顶部信号")
            print(f"  建议操作: {top.get('recommendation', 'N/A')}")


# ============================================================
# 交易执行（美股 Alpaca）
# ============================================================

def execute_trade(ticker: str, signal: str, position_pct: int, capital: float = 10000) -> Dict:
    """
    执行交易（仅支持美股 Alpaca）

    Args:
        ticker: 股票代码
        signal: 信号类型 (BUY/SELL)
        position_pct: 仓位百分比
        capital: 可用资金

    Returns:
        dict: 订单结果
    """
    if not ALPACA_AVAILABLE:
        return {"error": "Alpaca 模块不可用"}

    # 获取当前价格
    quote = fetch_realtime_data(ticker)
    if "error" in quote:
        return quote

    price = quote.get("price")
    if not price:
        return {"error": "无法获取价格"}

    # 计算数量
    amount = capital * (position_pct / 100)
    qty = int(amount / price)

    if qty <= 0:
        return {"error": "数量不足"}

    # 执行订单
    side = "buy" if "BUY" in signal else "sell"
    result = submit_order(ticker, qty, side)

    return result


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Momentum Catcher v2 - 增强版')
    parser.add_argument('--ticker', '-t', required=True, help='股票代码')
    parser.add_argument('--market', '-m', default='auto', choices=['auto', 'us', 'hk', 'a'],
                        help='市场类型')
    parser.add_argument('--start', '-s', help='开始日期 YYYY-MM-DD')
    parser.add_argument('--backtest', '-b', action='store_true', help='运行绩效回测')
    parser.add_argument('--realtime', '-r', action='store_true', help='只获取实时行情')
    parser.add_argument('--trade', action='store_true', help='执行交易（美股 Alpaca）')
    parser.add_argument('--capital', type=float, default=10000, help='交易资金')
    parser.add_argument('--output', '-o', help='输出文件')
    # VectorBT 增强分析参数
    parser.add_argument('--enhanced', '-e', action='store_true', help='运行 VectorBT 增强分析')
    parser.add_argument('--vbt-backtest', action='store_true', help='运行 VectorBT 策略回测')
    parser.add_argument('--strategy', default='momentum', choices=['momentum', 'rsi', 'macd', 'ma_cross'],
                        help='回测策略类型')
    parser.add_argument('--days', type=int, default=180, help='历史数据天数')

    args = parser.parse_args()

    # 实时行情模式
    if args.realtime:
        quote = fetch_realtime_data(args.ticker)
        import json
        print(json.dumps(quote, indent=2, default=str))
        return

    # 获取历史数据
    print(f"获取 {args.ticker} 数据...")
    df = fetch_data(args.ticker, args.start, market=args.market, days=args.days)

    if df.empty:
        print("无法获取数据")
        sys.exit(1)

    print(f"获取到 {len(df)} 条数据")

    # 计算指标
    print("计算技术指标...")
    df = calculate_indicators(df)

    # 计算信号
    scores = calculate_momentum_score(df)

    if "error" in scores:
        print(f"错误: {scores['error']}")
        sys.exit(1)

    # 获取美股额外信息
    context = None
    if args.market == "us" or (args.market == "auto" and not args.ticker.endswith(".HK")):
        print("获取美股额外信息...")
        context = get_us_stock_context(args.ticker)

    # 生成报告
    report = generate_report(args.ticker, df, scores, context)
    print(report)

    # 绩效回测
    if args.backtest:
        print("\n运行绩效回测...")
        backtest_result = run_backtest(df, args.ticker)
        if "error" not in backtest_result:
            print("\n【绩效分析】")
            print(f"  年化收益: {backtest_result.get('returns', {}).get('annualized', 'N/A'):.2%}")
            print(f"  夏普比率: {backtest_result.get('risk_adjusted', {}).get('sharpe_ratio', 'N/A'):.2f}")
            print(f"  最大回撤: {backtest_result.get('risk', {}).get('max_drawdown', {}).get('max_drawdown', 'N/A'):.2%}")
            if "html_report" in backtest_result:
                print(f"  HTML报告: {backtest_result['html_report']}")

    # VectorBT 增强分析
    if args.enhanced:
        print("\n运行 VectorBT 增强分析...")
        enhanced_results = run_enhanced_analysis(df, args.ticker)
        if "error" not in enhanced_results:
            print_enhanced_report(enhanced_results)
        else:
            print(f"增强分析失败: {enhanced_results.get('error')}")

    # VectorBT 策略回测
    if args.vbt_backtest:
        print(f"\n运行 VectorBT {args.strategy} 策略回测...")
        vbt_result = run_vbt_backtest(df, strategy=args.strategy)
        if "error" not in vbt_result:
            print("\n【VectorBT 策略回测】")
            print(f"  策略收益: {vbt_result.get('total_return', 0):.2%}")
            print(f"  基准收益: {vbt_result.get('benchmark_return', 0):.2%}")
            print(f"  超额收益: {vbt_result.get('alpha', 0):.2%}")
            print(f"  夏普比率: {vbt_result.get('sharpe_ratio', 'N/A')}")
            print(f"  最大回撤: {vbt_result.get('max_drawdown', 0):.2%}")
            print(f"  交易次数: {vbt_result.get('total_trades', 0)}")
        else:
            print(f"VectorBT 回测失败: {vbt_result.get('error')}")

    # 交易执行
    if args.trade:
        entry = get_entry_signal(scores.get("momentum_score", 0))
        if entry["position"] > 0:
            print(f"\n执行交易: {entry['signal']} {args.ticker}...")
            result = execute_trade(args.ticker, entry["signal"], entry["position"], args.capital)
            import json
            print(json.dumps(result, indent=2, default=str))
        else:
            print("\n无入场信号，不执行交易")

    # 保存数据
    if args.output:
        df.to_csv(args.output)
        print(f"\n数据已保存: {args.output}")


if __name__ == "__main__":
    main()
