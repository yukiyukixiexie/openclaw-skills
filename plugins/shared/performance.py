#!/usr/bin/env python3
"""
投资组合绩效分析工具

基于 quantstats 和自定义指标，生成专业的绩效分析报告

功能:
- 收益率分析（日/月/年）
- 风险指标（波动率、最大回撤、Sharpe、Sortino）
- 对比基准（SPY、QQQ、恒生指数等）
- 生成 HTML 报告

依赖:
    pip install quantstats pandas numpy
"""

import os
from datetime import datetime
from typing import Union, Dict, List, Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None


# ============== 核心绩效指标 ==============

def calculate_returns(prices: "pd.Series") -> "pd.Series":
    """计算日收益率"""
    return prices.pct_change().dropna()


def calculate_cumulative_returns(returns: "pd.Series") -> "pd.Series":
    """计算累计收益率"""
    return (1 + returns).cumprod() - 1


def calculate_annualized_return(returns: "pd.Series", periods_per_year: int = 252) -> float:
    """计算年化收益率"""
    total_return = (1 + returns).prod() - 1
    n_periods = len(returns)
    return (1 + total_return) ** (periods_per_year / n_periods) - 1


def calculate_volatility(returns: "pd.Series", periods_per_year: int = 252) -> float:
    """计算年化波动率"""
    return returns.std() * np.sqrt(periods_per_year)


def calculate_sharpe_ratio(
    returns: "pd.Series",
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    计算夏普比率

    Args:
        returns: 日收益率序列
        risk_free_rate: 年化无风险利率
        periods_per_year: 每年交易日数
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(
    returns: "pd.Series",
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    计算索提诺比率（只考虑下行波动）
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()

    if downside_std == 0:
        return np.inf

    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std


def calculate_max_drawdown(returns: "pd.Series") -> Dict:
    """
    计算最大回撤

    Returns:
        dict: 最大回撤、开始日期、结束日期、恢复日期
    """
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max

    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()

    # 找到回撤开始点
    start_idx = cumulative[:end_idx].idxmax()

    # 找到恢复点
    recovery_idx = None
    if end_idx != drawdown.index[-1]:
        recovery_mask = cumulative[end_idx:] >= cumulative[start_idx]
        if recovery_mask.any():
            recovery_idx = recovery_mask.idxmax()

    return {
        "max_drawdown": max_dd,
        "start_date": str(start_idx),
        "end_date": str(end_idx),
        "recovery_date": str(recovery_idx) if recovery_idx else None,
    }


def calculate_calmar_ratio(returns: "pd.Series", periods_per_year: int = 252) -> float:
    """计算卡尔玛比率（年化收益/最大回撤）"""
    ann_return = calculate_annualized_return(returns, periods_per_year)
    max_dd = abs(calculate_max_drawdown(returns)["max_drawdown"])
    return ann_return / max_dd if max_dd != 0 else np.inf


def calculate_win_rate(returns: "pd.Series") -> float:
    """计算胜率"""
    wins = (returns > 0).sum()
    total = len(returns)
    return wins / total if total > 0 else 0


def calculate_profit_factor(returns: "pd.Series") -> float:
    """计算盈亏比"""
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    return gains / losses if losses != 0 else np.inf


# ============== 完整绩效报告 ==============

def analyze_performance(
    returns: "pd.Series",
    benchmark_returns: "pd.Series" = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict:
    """
    生成完整绩效分析报告

    Args:
        returns: 策略日收益率
        benchmark_returns: 基准日收益率（可选）
        risk_free_rate: 年化无风险利率
        periods_per_year: 每年交易日数

    Returns:
        dict: 完整绩效指标
    """
    report = {
        "period": {
            "start": str(returns.index[0]),
            "end": str(returns.index[-1]),
            "trading_days": len(returns),
        },
        "returns": {
            "total": float((1 + returns).prod() - 1),
            "annualized": float(calculate_annualized_return(returns, periods_per_year)),
            "ytd": None,  # 可以后续计算
            "mtd": None,
        },
        "risk": {
            "volatility": float(calculate_volatility(returns, periods_per_year)),
            "max_drawdown": calculate_max_drawdown(returns),
            "var_95": float(returns.quantile(0.05)),
            "cvar_95": float(returns[returns <= returns.quantile(0.05)].mean()),
        },
        "risk_adjusted": {
            "sharpe_ratio": float(calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)),
            "sortino_ratio": float(calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)),
            "calmar_ratio": float(calculate_calmar_ratio(returns, periods_per_year)),
        },
        "trading": {
            "win_rate": float(calculate_win_rate(returns)),
            "profit_factor": float(calculate_profit_factor(returns)),
            "best_day": float(returns.max()),
            "worst_day": float(returns.min()),
            "avg_daily_return": float(returns.mean()),
        },
    }

    # 添加基准对比
    if benchmark_returns is not None:
        # 对齐日期
        aligned = pd.concat([returns, benchmark_returns], axis=1, join='inner')
        aligned.columns = ['strategy', 'benchmark']

        strategy_ret = aligned['strategy']
        benchmark_ret = aligned['benchmark']

        # 计算 Alpha 和 Beta
        covariance = strategy_ret.cov(benchmark_ret)
        benchmark_var = benchmark_ret.var()
        beta = covariance / benchmark_var if benchmark_var != 0 else 0

        strategy_ann = calculate_annualized_return(strategy_ret, periods_per_year)
        benchmark_ann = calculate_annualized_return(benchmark_ret, periods_per_year)
        alpha = strategy_ann - (risk_free_rate + beta * (benchmark_ann - risk_free_rate))

        # 信息比率
        active_returns = strategy_ret - benchmark_ret
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)
        information_ratio = (strategy_ann - benchmark_ann) / tracking_error if tracking_error != 0 else 0

        report["vs_benchmark"] = {
            "benchmark_return": float((1 + benchmark_ret).prod() - 1),
            "benchmark_annualized": float(benchmark_ann),
            "excess_return": float(strategy_ann - benchmark_ann),
            "alpha": float(alpha),
            "beta": float(beta),
            "correlation": float(strategy_ret.corr(benchmark_ret)),
            "tracking_error": float(tracking_error),
            "information_ratio": float(information_ratio),
        }

    return report


def generate_monthly_returns(returns: "pd.Series") -> "pd.DataFrame":
    """生成月度收益率表"""
    monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    monthly.index = monthly.index.to_period('M')

    # 转换为透视表（年份为行，月份为列）
    df = pd.DataFrame(monthly)
    df.columns = ['return']
    df['year'] = df.index.year
    df['month'] = df.index.month

    pivot = df.pivot(index='year', columns='month', values='return')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # 添加年度总收益
    yearly = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    pivot['Year'] = yearly.values[:len(pivot)]

    return pivot


# ============== QuantStats 集成 ==============

def generate_html_report(
    returns: "pd.Series",
    benchmark: str = "SPY",
    output_file: str = "performance_report.html",
    title: str = "Portfolio Performance"
) -> str:
    """
    使用 quantstats 生成 HTML 绩效报告

    Args:
        returns: 日收益率序列
        benchmark: 基准代码（如 'SPY', 'QQQ'）
        output_file: 输出文件路径
        title: 报告标题

    Returns:
        str: 报告文件路径
    """
    try:
        import quantstats as qs
    except ImportError:
        return {"error": "需要安装 quantstats: pip install quantstats"}

    try:
        qs.reports.html(
            returns,
            benchmark=benchmark,
            output=output_file,
            title=title,
            download_filename=output_file
        )
        return output_file
    except Exception as e:
        return {"error": f"生成报告失败: {str(e)}"}


def generate_tearsheet(
    returns: "pd.Series",
    benchmark: str = "SPY"
) -> None:
    """
    在终端显示绩效 tearsheet

    Args:
        returns: 日收益率序列
        benchmark: 基准代码
    """
    try:
        import quantstats as qs
        qs.reports.full(returns, benchmark=benchmark)
    except ImportError:
        print("需要安装 quantstats: pip install quantstats")


def quick_stats(returns: "pd.Series") -> Dict:
    """
    快速统计（使用 quantstats）
    """
    try:
        import quantstats as qs
    except ImportError:
        # 降级到自定义实现
        return analyze_performance(returns)

    try:
        stats = qs.stats.metrics(returns, mode='full')
        return stats.to_dict()
    except Exception as e:
        return {"error": str(e)}


# ============== 回测结果分析 ==============

def analyze_backtest(
    trades: List[Dict],
    initial_capital: float = 100000
) -> Dict:
    """
    分析回测交易记录

    Args:
        trades: 交易列表，每个交易包含:
            - date: 交易日期
            - symbol: 股票代码
            - side: 'buy' 或 'sell'
            - price: 成交价
            - qty: 数量
            - pnl: 该笔交易盈亏（可选）
        initial_capital: 初始资金

    Returns:
        dict: 回测分析结果
    """
    if not trades:
        return {"error": "没有交易记录"}

    df = pd.DataFrame(trades)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # 计算每笔交易的盈亏
    if 'pnl' not in df.columns:
        # 简化处理：假设已经有 pnl
        df['pnl'] = 0

    # 基础统计
    total_trades = len(df)
    winning_trades = len(df[df['pnl'] > 0])
    losing_trades = len(df[df['pnl'] < 0])
    total_pnl = df['pnl'].sum()

    gross_profit = df[df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())

    return {
        "summary": {
            "initial_capital": initial_capital,
            "final_capital": initial_capital + total_pnl,
            "total_return": total_pnl / initial_capital,
            "total_pnl": total_pnl,
        },
        "trades": {
            "total": total_trades,
            "winning": winning_trades,
            "losing": losing_trades,
            "win_rate": winning_trades / total_trades if total_trades > 0 else 0,
        },
        "pnl": {
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.inf,
            "avg_win": gross_profit / winning_trades if winning_trades > 0 else 0,
            "avg_loss": gross_loss / losing_trades if losing_trades > 0 else 0,
            "largest_win": df['pnl'].max(),
            "largest_loss": df['pnl'].min(),
        },
        "period": {
            "start": str(df['date'].min()),
            "end": str(df['date'].max()),
            "trading_days": (df['date'].max() - df['date'].min()).days,
        }
    }


# ============== CLI ==============

def main():
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
投资组合绩效分析工具

用法:
    python performance.py <命令> [参数]

命令:
    analyze <csv_file>           分析收益率CSV文件
    report <csv_file> [benchmark] 生成HTML报告
    monthly <csv_file>           显示月度收益表

CSV 文件格式:
    date,return
    2024-01-02,0.01
    2024-01-03,-0.005
    ...

示例:
    python performance.py analyze returns.csv
    python performance.py report returns.csv SPY
    python performance.py monthly returns.csv

依赖:
    pip install quantstats pandas numpy
        """)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "analyze" and len(sys.argv) >= 3:
        csv_file = sys.argv[2]
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        returns = df.iloc[:, 0]  # 第一列作为收益率

        result = analyze_performance(returns)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "report" and len(sys.argv) >= 3:
        csv_file = sys.argv[2]
        benchmark = sys.argv[3] if len(sys.argv) > 3 else "SPY"

        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        returns = df.iloc[:, 0]

        output = generate_html_report(returns, benchmark)
        print(f"报告已生成: {output}")

    elif cmd == "monthly" and len(sys.argv) >= 3:
        csv_file = sys.argv[2]
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        returns = df.iloc[:, 0]

        monthly = generate_monthly_returns(returns)
        print("\n月度收益率 (%):")
        print((monthly * 100).round(2).to_string())

    else:
        print(f"❌ 未知命令: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
