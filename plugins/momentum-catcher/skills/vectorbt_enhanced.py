#!/usr/bin/env python3
"""
VectorBT 增强分析模块

基于智谱复盘方法论，实现：
1. 换手率极值检测 - 顶部信号
2. 地量信号 - 建仓时机
3. 筹码分布 - 成本区间分析
4. 供给受限检测 - 小盘股特有信号
5. 板块扩散信号 - 龙头见顶
6. 快速回测 - 策略验证

Usage:
    python vectorbt_enhanced.py --ticker 02513.HK --analyze
    python vectorbt_enhanced.py --ticker 02513.HK --backtest
    python vectorbt_enhanced.py --ticker 02513.HK --chip-distribution
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except ImportError:
    VBT_AVAILABLE = False
    print("警告: vectorbt 未安装，请运行: pip install vectorbt")

# 添加 shared 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))

try:
    from unified_data import get_history
    UNIFIED_DATA = True
except ImportError:
    UNIFIED_DATA = False


# ============================================================
# 数据获取
# ============================================================

def fetch_data(ticker: str, start_date: str = None, days: int = 180) -> pd.DataFrame:
    """获取历史数据"""
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    if UNIFIED_DATA:
        df = get_history(ticker, start_date)
        if isinstance(df, pd.DataFrame) and not df.empty:
            df.columns = [c.lower() for c in df.columns]
            return df

    # 降级到 yfinance
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date)
        df.columns = [c.lower() for c in df.columns]
        return df[['open', 'high', 'low', 'close', 'volume']]
    except Exception as e:
        print(f"获取数据失败: {e}")
        return pd.DataFrame()


# ============================================================
# 换手率极值检测
# ============================================================

def calculate_turnover_rate(df: pd.DataFrame, float_shares: float = None) -> pd.Series:
    """
    计算换手率

    Args:
        df: OHLCV 数据
        float_shares: 流通股数（可选）

    Returns:
        换手率序列 (%)
    """
    if 'turnover_rate' in df.columns:
        return df['turnover_rate']

    if float_shares:
        return df['volume'] / float_shares * 100

    # 没有流通股数据时，用相对换手（标准化）
    vol_ma = df['volume'].rolling(60).mean()
    return df['volume'] / vol_ma * 100


def detect_turnover_extreme(
    df: pd.DataFrame,
    window: int = 60,
    threshold: float = 0.95
) -> pd.DataFrame:
    """
    检测换手率极值

    基于智谱复盘：
    - 换手率 < 5%：地量（建仓信号）
    - 换手率 5-15%：正常
    - 换手率 > 25%：极值（减仓信号）

    Args:
        df: 包含 volume 的 DataFrame
        window: 滚动窗口
        threshold: 极值阈值（分位数）

    Returns:
        添加了换手率信号的 DataFrame
    """
    df = df.copy()

    # 计算相对换手率
    df['vol_ma60'] = df['volume'].rolling(window).mean()
    df['turnover_relative'] = df['volume'] / df['vol_ma60']

    # 滚动分位数排名
    df['turnover_rank'] = df['turnover_relative'].rolling(window).rank(pct=True)

    # 极值信号
    df['turnover_extreme_high'] = df['turnover_rank'] > threshold  # 高换手极值
    df['turnover_extreme_low'] = df['turnover_rank'] < (1 - threshold)  # 地量

    # 绝对阈值（基于智谱经验）
    # 假设相对换手 > 2.5 约等于 25%+ 绝对换手
    df['turnover_warning'] = df['turnover_relative'] > 2.5
    df['turnover_low_volume'] = df['turnover_relative'] < 0.5

    return df


def get_turnover_signals(df: pd.DataFrame) -> Dict:
    """获取换手率信号摘要"""
    df = detect_turnover_extreme(df)
    latest = df.iloc[-1]

    signals = {
        "current_turnover_relative": round(latest['turnover_relative'], 2),
        "turnover_rank": round(latest['turnover_rank'], 2),
        "is_extreme_high": bool(latest['turnover_extreme_high']),
        "is_extreme_low": bool(latest['turnover_extreme_low']),
        "signal": "NORMAL"
    }

    if latest['turnover_extreme_high']:
        signals["signal"] = "⚠️ 换手率极值 - 减仓信号"
    elif latest['turnover_extreme_low']:
        signals["signal"] = "📍 地量 - 潜在建仓机会"

    # 最近极值日期
    extreme_high_dates = df[df['turnover_extreme_high']].index
    if len(extreme_high_dates) > 0:
        signals["last_extreme_high"] = str(extreme_high_dates[-1].date())

    extreme_low_dates = df[df['turnover_extreme_low']].index
    if len(extreme_low_dates) > 0:
        signals["last_extreme_low"] = str(extreme_low_dates[-1].date())

    return signals


# ============================================================
# 筹码分布 / Volume Profile
# ============================================================

def calculate_volume_profile(
    df: pd.DataFrame,
    bins: int = 50,
    lookback: int = None
) -> pd.DataFrame:
    """
    计算筹码分布 (Volume Profile)

    基于智谱复盘：
    > 2/9-2/20 期间进场的资金成本集中在 280-500 区间

    Args:
        df: OHLCV 数据
        bins: 价格区间数
        lookback: 回看天数（None=全部）

    Returns:
        DataFrame with price levels and volume
    """
    if lookback:
        df = df.tail(lookback)

    # 价格区间
    price_min = df['low'].min()
    price_max = df['high'].max()
    price_bins = np.linspace(price_min, price_max, bins + 1)

    # 计算每个价格区间的成交量
    volume_profile = np.zeros(bins)

    for _, row in df.iterrows():
        # 当日成交量均匀分布在高低点之间
        day_low, day_high = row['low'], row['high']
        day_volume = row['volume']

        for i in range(bins):
            bin_low, bin_high = price_bins[i], price_bins[i + 1]

            # 计算重叠区域
            overlap_low = max(day_low, bin_low)
            overlap_high = min(day_high, bin_high)

            if overlap_high > overlap_low:
                # 按重叠比例分配成交量
                day_range = day_high - day_low if day_high > day_low else 1
                overlap_ratio = (overlap_high - overlap_low) / day_range
                volume_profile[i] += day_volume * overlap_ratio

    # 创建结果
    result = pd.DataFrame({
        'price_low': price_bins[:-1],
        'price_high': price_bins[1:],
        'price_mid': (price_bins[:-1] + price_bins[1:]) / 2,
        'volume': volume_profile,
        'volume_pct': volume_profile / volume_profile.sum() * 100
    })

    return result


def find_chip_concentration(volume_profile: pd.DataFrame, top_n: int = 3) -> List[Dict]:
    """
    找到筹码密集区

    Returns:
        筹码密集区列表
    """
    vp = volume_profile.sort_values('volume', ascending=False)

    concentrations = []
    for _, row in vp.head(top_n).iterrows():
        concentrations.append({
            "price_range": f"{row['price_low']:.2f} - {row['price_high']:.2f}",
            "price_mid": round(row['price_mid'], 2),
            "volume_pct": round(row['volume_pct'], 1),
        })

    return concentrations


def get_support_resistance(volume_profile: pd.DataFrame, current_price: float) -> Dict:
    """
    基于筹码分布找支撑/阻力位
    """
    # 找到当前价格以下的最大成交量区 = 支撑
    below = volume_profile[volume_profile['price_mid'] < current_price]
    if not below.empty:
        support_row = below.loc[below['volume'].idxmax()]
        support = support_row['price_mid']
    else:
        support = None

    # 找到当前价格以上的最大成交量区 = 阻力
    above = volume_profile[volume_profile['price_mid'] > current_price]
    if not above.empty:
        resistance_row = above.loc[above['volume'].idxmax()]
        resistance = resistance_row['price_mid']
    else:
        resistance = None

    return {
        "current_price": current_price,
        "support": round(support, 2) if support else None,
        "resistance": round(resistance, 2) if resistance else None,
    }


# ============================================================
# 供给受限检测
# ============================================================

def detect_supply_squeeze(
    df: pd.DataFrame,
    float_ratio: float = 0.25,
    lockup_ratio: float = 0.30
) -> pd.DataFrame:
    """
    检测供给受限信号

    基于智谱复盘：
    > 基石投资人锁定 6 个月，流通股仅占总股本约 25%
    > 实际可交易筹码更少

    Args:
        df: OHLCV 数据
        float_ratio: 流通盘占比
        lockup_ratio: 基石锁定占比

    Returns:
        添加了供给受限信号的 DataFrame
    """
    df = df.copy()

    # 有效流通比例
    effective_float = float_ratio * (1 - lockup_ratio * 0.5)  # 假设基石部分不交易

    # 日振幅
    df['amplitude'] = (df['high'] - df['low']) / df['close'].shift(1)

    # 相对成交量
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

    # 供给受限指标
    # 高振幅 + 高成交量 = 供给受限导致的价格非线性
    df['squeeze_score'] = df['amplitude'] * df['vol_ratio']
    df['squeeze_rank'] = df['squeeze_score'].rolling(60).rank(pct=True)

    # 供给受限信号
    df['supply_squeeze'] = (
        (df['amplitude'] > df['amplitude'].rolling(20).quantile(0.9)) &
        (df['vol_ratio'] > 1.5)
    )

    return df


def get_squeeze_signals(df: pd.DataFrame) -> Dict:
    """获取供给受限信号摘要"""
    df = detect_supply_squeeze(df)
    latest = df.iloc[-1]

    # 最近 20 天的挤压信号
    recent = df.tail(20)
    squeeze_days = recent['supply_squeeze'].sum()

    return {
        "current_amplitude": round(latest['amplitude'] * 100, 2),
        "current_vol_ratio": round(latest['vol_ratio'], 2),
        "squeeze_rank": round(latest['squeeze_rank'], 2),
        "is_squeeze": bool(latest['supply_squeeze']),
        "squeeze_days_20d": int(squeeze_days),
        "signal": "⚡ 供给受限 - 价格可能非线性" if latest['supply_squeeze'] else "正常",
    }


# ============================================================
# 综合顶部信号
# ============================================================

def detect_top_signals(df: pd.DataFrame) -> Dict:
    """
    检测顶部信号组合

    基于智谱复盘：
    > 换手率极值 + 板块扩散 + 价格滞涨 三个条件同时出现 = 清仓信号
    """
    df = detect_turnover_extreme(df)
    df = detect_supply_squeeze(df)

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else latest

    signals = {
        "date": str(df.index[-1].date()),
        "price": round(latest['close'], 2),
        "signals": [],
        "weight": 0,
    }

    # 1. 换手率极值 (权重 30%)
    if latest['turnover_extreme_high']:
        signals["signals"].append({
            "name": "换手率极值",
            "weight": 0.30,
            "value": f"相对换手 {latest['turnover_relative']:.1f}x (排名 {latest['turnover_rank']:.0%})"
        })
        signals["weight"] += 0.30

    # 2. 单日涨幅极值 (权重 25%)
    daily_return = (latest['close'] / prev['close'] - 1) * 100
    if daily_return > 40:
        signals["signals"].append({
            "name": "单日涨幅极值",
            "weight": 0.25,
            "value": f"+{daily_return:.1f}%"
        })
        signals["weight"] += 0.25
    elif daily_return > 20:
        signals["signals"].append({
            "name": "单日涨幅较大",
            "weight": 0.15,
            "value": f"+{daily_return:.1f}%"
        })
        signals["weight"] += 0.15

    # 3. 振幅极值 (权重 20%)
    if latest['amplitude'] > df['amplitude'].rolling(20).quantile(0.95).iloc[-1]:
        signals["signals"].append({
            "name": "振幅极值",
            "weight": 0.20,
            "value": f"{latest['amplitude']*100:.1f}%"
        })
        signals["weight"] += 0.20

    # 4. 供给受限 (权重 15%)
    if latest['supply_squeeze']:
        signals["signals"].append({
            "name": "供给受限",
            "weight": 0.15,
            "value": f"挤压得分 {latest['squeeze_rank']:.0%}"
        })
        signals["weight"] += 0.15

    # 5. RSI 超买 (权重 10%)
    df['rsi'] = calculate_rsi(df['close'])
    if df['rsi'].iloc[-1] > 80:
        signals["signals"].append({
            "name": "RSI超买",
            "weight": 0.10,
            "value": f"RSI={df['rsi'].iloc[-1]:.1f}"
        })
        signals["weight"] += 0.10

    # 综合判断
    signals["weight"] = round(signals["weight"], 2)

    if signals["weight"] >= 0.5:
        signals["action"] = "🔴 清仓"
    elif signals["weight"] >= 0.3:
        signals["action"] = "🟡 减仓 30%"
    elif signals["weight"] >= 0.15:
        signals["action"] = "🟡 警惕"
    else:
        signals["action"] = "🟢 持有"

    return signals


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


# ============================================================
# VectorBT 快速回测
# ============================================================

def run_vectorbt_backtest(
    df: pd.DataFrame,
    strategy: str = "momentum",
    init_cash: float = 100000
) -> Dict:
    """
    使用 VectorBT 运行快速回测

    Args:
        df: OHLCV 数据
        strategy: 策略类型
            - "momentum": 动量策略（地量买入，极值卖出）
            - "mean_reversion": 均值回归
        init_cash: 初始资金

    Returns:
        回测结果
    """
    if not VBT_AVAILABLE:
        return {"error": "vectorbt 未安装"}

    df = df.copy()
    df = detect_turnover_extreme(df)

    price = df['close']

    if strategy == "momentum":
        # 地量买入，换手率极值卖出
        entries = df['turnover_extreme_low'].shift(1).fillna(False).astype(bool)
        exits = df['turnover_extreme_high'].shift(1).fillna(False).astype(bool)

    elif strategy == "mean_reversion":
        # RSI 超卖买入，超买卖出
        df['rsi'] = calculate_rsi(df['close'])
        entries = (df['rsi'] < 30).shift(1).fillna(False).astype(bool)
        exits = (df['rsi'] > 70).shift(1).fillna(False).astype(bool)

    else:
        return {"error": f"未知策略: {strategy}"}

    # 运行回测
    try:
        pf = vbt.Portfolio.from_signals(
            price,
            entries=entries,
            exits=exits,
            init_cash=init_cash,
            fees=0.001,  # 0.1% 手续费
            slippage=0.001,  # 0.1% 滑点
            freq='1D',  # 日线频率
        )

        stats = pf.stats()

        return {
            "strategy": strategy,
            "total_return": round(stats['Total Return [%]'], 2),
            "sharpe_ratio": round(stats['Sharpe Ratio'], 2),
            "max_drawdown": round(stats['Max Drawdown [%]'], 2),
            "win_rate": round(stats['Win Rate [%]'], 2),
            "total_trades": int(stats['Total Trades']),
            "profit_factor": round(stats['Profit Factor'], 2) if stats['Profit Factor'] != np.inf else None,
            "final_value": round(stats['End Value'], 2),
            "benchmark_return": round((price.iloc[-1] / price.iloc[0] - 1) * 100, 2),
        }

    except Exception as e:
        return {"error": str(e)}


def run_parameter_optimization(
    df: pd.DataFrame,
    init_cash: float = 100000
) -> Dict:
    """
    参数优化 - 找到最佳换手率阈值
    """
    if not VBT_AVAILABLE:
        return {"error": "vectorbt 未安装"}

    df = df.copy()
    price = df['close']

    # 计算相对换手率
    df['vol_ma60'] = df['volume'].rolling(60).mean()
    df['turnover_relative'] = df['volume'] / df['vol_ma60']

    # 参数范围
    entry_thresholds = np.arange(0.3, 0.7, 0.1)  # 地量阈值
    exit_thresholds = np.arange(2.0, 4.0, 0.5)   # 极值阈值

    best_result = {"sharpe": -np.inf}

    for entry_th in entry_thresholds:
        for exit_th in exit_thresholds:
            entries = (df['turnover_relative'] < entry_th).shift(1).fillna(False)
            exits = (df['turnover_relative'] > exit_th).shift(1).fillna(False)

            try:
                pf = vbt.Portfolio.from_signals(
                    price,
                    entries=entries,
                    exits=exits,
                    init_cash=init_cash,
                    fees=0.001,
                )

                stats = pf.stats()
                sharpe = stats['Sharpe Ratio']

                if sharpe > best_result["sharpe"] and stats['Total Trades'] >= 3:
                    best_result = {
                        "entry_threshold": round(entry_th, 1),
                        "exit_threshold": round(exit_th, 1),
                        "sharpe": round(sharpe, 2),
                        "total_return": round(stats['Total Return [%]'], 2),
                        "max_drawdown": round(stats['Max Drawdown [%]'], 2),
                        "total_trades": int(stats['Total Trades']),
                    }
            except:
                continue

    return best_result


# ============================================================
# 报告生成
# ============================================================

def generate_analysis_report(ticker: str, df: pd.DataFrame) -> str:
    """生成完整分析报告"""
    lines = [
        f"\n{'='*60}",
        f"[{ticker}] VectorBT 增强分析报告",
        f"{'='*60}",
        f"分析日期: {df.index[-1].strftime('%Y-%m-%d')}",
        f"数据区间: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')}",
        f"当前价格: {df['close'].iloc[-1]:.2f}",
        "",
    ]

    # 换手率信号
    turnover_signals = get_turnover_signals(df)
    lines.append("-" * 40)
    lines.append("【换手率分析】")
    lines.append("-" * 40)
    lines.append(f"  相对换手率: {turnover_signals['current_turnover_relative']}x")
    lines.append(f"  历史排名: {turnover_signals['turnover_rank']:.0%}")
    lines.append(f"  信号: {turnover_signals['signal']}")
    if 'last_extreme_high' in turnover_signals:
        lines.append(f"  最近极值: {turnover_signals['last_extreme_high']}")
    if 'last_extreme_low' in turnover_signals:
        lines.append(f"  最近地量: {turnover_signals['last_extreme_low']}")

    # 筹码分布
    lines.append("")
    lines.append("-" * 40)
    lines.append("【筹码分布】")
    lines.append("-" * 40)

    vp = calculate_volume_profile(df, bins=30, lookback=60)
    concentrations = find_chip_concentration(vp, top_n=3)

    lines.append("  筹码密集区 (近60日):")
    for i, c in enumerate(concentrations, 1):
        lines.append(f"    {i}. {c['price_range']} ({c['volume_pct']:.1f}% 筹码)")

    sr = get_support_resistance(vp, df['close'].iloc[-1])
    if sr['support']:
        lines.append(f"  支撑位: {sr['support']:.2f}")
    if sr['resistance']:
        lines.append(f"  阻力位: {sr['resistance']:.2f}")

    # 供给受限
    lines.append("")
    lines.append("-" * 40)
    lines.append("【供给受限检测】")
    lines.append("-" * 40)

    squeeze = get_squeeze_signals(df)
    lines.append(f"  当日振幅: {squeeze['current_amplitude']:.1f}%")
    lines.append(f"  成交量倍数: {squeeze['current_vol_ratio']:.1f}x")
    lines.append(f"  挤压排名: {squeeze['squeeze_rank']:.0%}")
    lines.append(f"  近20日挤压天数: {squeeze['squeeze_days_20d']}")
    lines.append(f"  信号: {squeeze['signal']}")

    # 顶部信号
    lines.append("")
    lines.append("-" * 40)
    lines.append("【顶部信号检测】")
    lines.append("-" * 40)

    top = detect_top_signals(df)
    if top['signals']:
        for sig in top['signals']:
            lines.append(f"  ⚠️ {sig['name']}: {sig['value']} (权重 {sig['weight']:.0%})")
        lines.append(f"\n  综合权重: {top['weight']:.0%}")
        lines.append(f"  建议操作: {top['action']}")
    else:
        lines.append("  无明显顶部信号")
        lines.append(f"  建议操作: {top['action']}")

    # 回测结果
    if VBT_AVAILABLE:
        lines.append("")
        lines.append("-" * 40)
        lines.append("【策略回测】")
        lines.append("-" * 40)

        bt = run_vectorbt_backtest(df, strategy="momentum")
        if "error" not in bt:
            lines.append(f"  策略收益: {bt['total_return']}%")
            lines.append(f"  基准收益: {bt['benchmark_return']}%")
            lines.append(f"  夏普比率: {bt['sharpe_ratio']}")
            lines.append(f"  最大回撤: {bt['max_drawdown']}%")
            lines.append(f"  胜率: {bt['win_rate']}%")
            lines.append(f"  交易次数: {bt['total_trades']}")

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='VectorBT 增强分析')
    parser.add_argument('--ticker', '-t', required=True, help='股票代码')
    parser.add_argument('--days', '-d', type=int, default=180, help='数据天数')
    parser.add_argument('--analyze', '-a', action='store_true', help='运行完整分析')
    parser.add_argument('--backtest', '-b', action='store_true', help='运行回测')
    parser.add_argument('--optimize', action='store_true', help='参数优化')
    parser.add_argument('--chip-distribution', action='store_true', help='显示筹码分布')
    parser.add_argument('--top-signals', action='store_true', help='显示顶部信号')

    args = parser.parse_args()

    # 获取数据
    print(f"获取 {args.ticker} 数据...")
    df = fetch_data(args.ticker, days=args.days)

    if df.empty:
        print("无法获取数据")
        sys.exit(1)

    print(f"获取到 {len(df)} 条数据\n")

    # 完整分析
    if args.analyze:
        report = generate_analysis_report(args.ticker, df)
        print(report)
        return

    # 回测
    if args.backtest:
        print("运行回测...")
        result = run_vectorbt_backtest(df, strategy="momentum")
        import json
        print(json.dumps(result, indent=2))
        return

    # 参数优化
    if args.optimize:
        print("运行参数优化...")
        result = run_parameter_optimization(df)
        import json
        print(json.dumps(result, indent=2))
        return

    # 筹码分布
    if args.chip_distribution:
        print("计算筹码分布...")
        vp = calculate_volume_profile(df, bins=20)
        print("\n价格区间           成交量占比")
        print("-" * 40)
        for _, row in vp.sort_values('volume', ascending=False).head(10).iterrows():
            bar = "█" * int(row['volume_pct'])
            print(f"{row['price_low']:8.2f} - {row['price_high']:8.2f}  {row['volume_pct']:5.1f}% {bar}")
        return

    # 顶部信号
    if args.top_signals:
        print("检测顶部信号...")
        result = detect_top_signals(df)
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    # 默认：显示帮助
    parser.print_help()


if __name__ == "__main__":
    main()
