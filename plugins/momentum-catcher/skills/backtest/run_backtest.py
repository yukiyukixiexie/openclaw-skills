#!/usr/bin/env python3
"""
Momentum Catcher - 回测模块
对智谱AI (02513.HK) 进行历史信号回溯和策略验证

Usage:
    python run_backtest.py --ticker 02513.HK
    python run_backtest.py --ticker 02513.HK --start 2026-01-15 --end 2026-02-27
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import (
    fetch_hk_stock_data,
    calculate_momentum_indicators,
    calculate_signal_scores,
    generate_entry_signal,
    generate_exit_signal,
    normalize_ticker,
    check_volume_anomaly
)


# ============================================================
# 回测引擎
# ============================================================

class BacktestEngine:
    """回测引擎"""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission: float = 0.001,  # 佣金率 0.1%
        slippage: float = 0.001,    # 滑点 0.1%
        event_score: float = 4.0,   # 默认事件评分
        capital_score: int = 3      # 默认资金评分
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.event_score = event_score
        self.capital_score = capital_score

        # 状态变量
        self.cash = initial_capital
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.trades = []
        self.daily_equity = []

    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.position = 0
        self.position_value = 0
        self.entry_price = 0
        self.trades = []
        self.daily_equity = []

    def execute_buy(self, date: str, price: float, position_pct: float, reason: str):
        """执行买入"""
        # 计算实际买入金额
        available = self.cash * (position_pct / 100)
        actual_price = price * (1 + self.slippage)  # 考虑滑点
        shares = int(available / actual_price / 100) * 100  # 港股一手100股

        if shares <= 0:
            return

        cost = shares * actual_price
        commission = cost * self.commission

        self.cash -= (cost + commission)
        self.position += shares
        self.entry_price = actual_price

        self.trades.append({
            'date': date,
            'action': 'BUY',
            'price': round(actual_price, 3),
            'shares': shares,
            'value': round(cost, 2),
            'commission': round(commission, 2),
            'reason': reason
        })

    def execute_sell(self, date: str, price: float, sell_pct: float, reason: str):
        """执行卖出"""
        if self.position <= 0:
            return

        shares_to_sell = int(self.position * (sell_pct / 100) / 100) * 100
        if shares_to_sell <= 0:
            shares_to_sell = self.position  # 清仓时卖出全部

        actual_price = price * (1 - self.slippage)  # 考虑滑点
        revenue = shares_to_sell * actual_price
        commission = revenue * self.commission

        self.cash += (revenue - commission)
        self.position -= shares_to_sell

        # 计算本次交易盈亏
        pnl = (actual_price - self.entry_price) * shares_to_sell

        self.trades.append({
            'date': date,
            'action': 'SELL',
            'price': round(actual_price, 3),
            'shares': shares_to_sell,
            'value': round(revenue, 2),
            'commission': round(commission, 2),
            'pnl': round(pnl, 2),
            'reason': reason
        })

        if self.position == 0:
            self.entry_price = 0

    def update_equity(self, date: str, price: float):
        """更新每日权益"""
        position_value = self.position * price
        total_equity = self.cash + position_value

        self.daily_equity.append({
            'date': date,
            'cash': round(self.cash, 2),
            'position': self.position,
            'position_value': round(position_value, 2),
            'total_equity': round(total_equity, 2),
            'price': round(price, 2)
        })

    def get_metrics(self) -> Dict:
        """计算回测指标"""
        if not self.daily_equity:
            return {}

        equity_df = pd.DataFrame(self.daily_equity)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df = equity_df.set_index('date')

        # 计算收益率序列
        equity_df['returns'] = equity_df['total_equity'].pct_change()

        # 总收益率
        total_return = (equity_df['total_equity'].iloc[-1] / self.initial_capital - 1) * 100

        # 最大回撤
        equity_df['cummax'] = equity_df['total_equity'].cummax()
        equity_df['drawdown'] = (equity_df['total_equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min() * 100

        # 夏普比率（假设无风险利率为3%）
        risk_free_rate = 0.03 / 252  # 日化
        excess_returns = equity_df['returns'] - risk_free_rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0

        # 交易统计
        trades_df = pd.DataFrame(self.trades)
        sell_trades = trades_df[trades_df['action'] == 'SELL'] if not trades_df.empty else pd.DataFrame()

        win_trades = len(sell_trades[sell_trades['pnl'] > 0]) if 'pnl' in sell_trades.columns else 0
        total_trades = len(sell_trades)
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

        # Buy & Hold 对比
        first_price = equity_df['price'].iloc[0]
        last_price = equity_df['price'].iloc[-1]
        bh_return = (last_price / first_price - 1) * 100

        return {
            'total_return': round(total_return, 2),
            'max_drawdown': round(max_drawdown, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'total_trades': total_trades,
            'win_rate': round(win_rate, 2),
            'buy_hold_return': round(bh_return, 2),
            'alpha': round(total_return - bh_return, 2),
            'final_equity': round(equity_df['total_equity'].iloc[-1], 2),
            'start_date': equity_df.index[0].strftime('%Y-%m-%d'),
            'end_date': equity_df.index[-1].strftime('%Y-%m-%d')
        }


def run_backtest(
    df: pd.DataFrame,
    event_score: float = 4.0,
    capital_score: int = 3,
    initial_capital: float = 100000
) -> Tuple[BacktestEngine, pd.DataFrame]:
    """
    运行回测

    Args:
        df: 包含技术指标的行情数据
        event_score: 事件强度评分
        capital_score: 资金确认评分
        initial_capital: 初始资金

    Returns:
        (BacktestEngine, signals_df)
    """
    engine = BacktestEngine(
        initial_capital=initial_capital,
        event_score=event_score,
        capital_score=capital_score
    )

    signals = []

    # 需要至少20个交易日的数据
    if len(df) < 20:
        print("数据不足，需要至少20个交易日")
        return engine, pd.DataFrame()

    # 逐日回测
    for i in range(20, len(df)):
        # 使用截至当日的数据计算信号
        current_df = df.iloc[:i+1].copy()
        current_date = current_df.index[-1].strftime('%Y-%m-%d')
        current_price = current_df.iloc[-1]['close']

        # 计算信号
        scores = calculate_signal_scores(current_df)

        if 'error' in scores:
            continue

        momentum_score = scores.get('momentum_score', 0)
        exit_weight = scores.get('exit_weight', 0)

        # 检查是否跌破MA20
        latest = current_df.iloc[-1]
        below_ma20 = latest['close'] < latest['ma20'] if pd.notna(latest['ma20']) else False

        # 生成入场信号
        entry_signal, entry_desc, position_pct = generate_entry_signal(
            momentum_score, event_score, capital_score
        )

        # 生成退出信号
        exit_signal, exit_desc = generate_exit_signal(exit_weight, below_ma20)

        # 记录信号
        signal_record = {
            'date': current_date,
            'close': round(current_price, 2),
            'momentum_score': momentum_score,
            'exit_weight': exit_weight,
            'entry_signal': entry_signal,
            'exit_signal': exit_signal,
            'position': engine.position
        }
        signals.append(signal_record)

        # 执行交易逻辑
        if engine.position == 0:
            # 未持仓，检查入场信号
            if entry_signal in ['STRONG_BUY', 'BUY'] and momentum_score >= 3:
                engine.execute_buy(current_date, current_price, position_pct, entry_desc)
        else:
            # 已持仓，检查退出信号
            if exit_signal == 'EXIT_ALL':
                engine.execute_sell(current_date, current_price, 100, exit_desc)
            elif exit_signal == 'REDUCE':
                engine.execute_sell(current_date, current_price, 30, exit_desc)
            # 检查是否可以加仓
            elif entry_signal == 'STRONG_BUY' and momentum_score >= 4:
                # 如果信号更强，可以加仓
                current_position_value = engine.position * current_price
                max_position_value = engine.initial_capital * (position_pct / 100)
                if current_position_value < max_position_value * 0.8:
                    engine.execute_buy(current_date, current_price, 20, "动量增强加仓")

        # 更新每日权益
        engine.update_equity(current_date, current_price)

    return engine, pd.DataFrame(signals)


def generate_backtest_report(
    ticker: str,
    engine: BacktestEngine,
    signals_df: pd.DataFrame
) -> str:
    """生成回测报告"""
    ticker = normalize_ticker(ticker)
    metrics = engine.get_metrics()

    report = []
    report.append(f"\n{'='*60}")
    report.append(f"[{ticker}] 回测报告")
    report.append(f"{'='*60}")
    report.append(f"回测区间: {metrics.get('start_date', 'N/A')} 至 {metrics.get('end_date', 'N/A')}")
    report.append(f"初始资金: {engine.initial_capital:,.0f} HKD")
    report.append("")

    report.append("-" * 40)
    report.append("【策略表现】")
    report.append("-" * 40)
    report.append(f"  总收益率: {metrics.get('total_return', 0):+.2f}%")
    report.append(f"  最大回撤: {metrics.get('max_drawdown', 0):.2f}%")
    report.append(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    report.append(f"  最终权益: {metrics.get('final_equity', 0):,.2f} HKD")
    report.append("")

    report.append("-" * 40)
    report.append("【交易统计】")
    report.append("-" * 40)
    report.append(f"  交易次数: {metrics.get('total_trades', 0)}")
    report.append(f"  胜率: {metrics.get('win_rate', 0):.1f}%")
    report.append("")

    report.append("-" * 40)
    report.append("【Buy & Hold 对比】")
    report.append("-" * 40)
    report.append(f"  B&H收益率: {metrics.get('buy_hold_return', 0):+.2f}%")
    report.append(f"  超额收益(Alpha): {metrics.get('alpha', 0):+.2f}%")
    report.append("")

    # 交易记录
    if engine.trades:
        report.append("-" * 40)
        report.append("【交易记录】")
        report.append("-" * 40)
        for trade in engine.trades:
            action = trade['action']
            date = trade['date']
            price = trade['price']
            shares = trade['shares']
            reason = trade['reason']

            if action == 'BUY':
                report.append(f"  {date} 买入 {shares}股 @ {price:.2f} ({reason})")
            else:
                pnl = trade.get('pnl', 0)
                pnl_str = f"+{pnl:.0f}" if pnl > 0 else f"{pnl:.0f}"
                report.append(f"  {date} 卖出 {shares}股 @ {price:.2f} ({reason}) 盈亏:{pnl_str}")

    report.append("")
    report.append("="*60)

    return "\n".join(report)


def save_backtest_results(
    ticker: str,
    engine: BacktestEngine,
    signals_df: pd.DataFrame,
    output_dir: str = "."
):
    """保存回测结果"""
    ticker_clean = normalize_ticker(ticker).replace('.', '_')

    # 保存信号记录
    if not signals_df.empty:
        signals_file = os.path.join(output_dir, f"{ticker_clean}_signals.csv")
        signals_df.to_csv(signals_file, index=False)
        print(f"信号记录已保存: {signals_file}")

    # 保存交易记录
    if engine.trades:
        trades_df = pd.DataFrame(engine.trades)
        trades_file = os.path.join(output_dir, f"{ticker_clean}_trades.csv")
        trades_df.to_csv(trades_file, index=False)
        print(f"交易记录已保存: {trades_file}")

    # 保存权益曲线
    if engine.daily_equity:
        equity_df = pd.DataFrame(engine.daily_equity)
        equity_file = os.path.join(output_dir, f"{ticker_clean}_equity.csv")
        equity_df.to_csv(equity_file, index=False)
        print(f"权益曲线已保存: {equity_file}")


def plot_equity_curve(engine: BacktestEngine, ticker: str, output_dir: str = "."):
    """绘制权益曲线（如果matplotlib可用）"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if not engine.daily_equity:
            return

        equity_df = pd.DataFrame(engine.daily_equity)
        equity_df['date'] = pd.to_datetime(equity_df['date'])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # 权益曲线
        ax1.plot(equity_df['date'], equity_df['total_equity'], label='Strategy', color='blue')
        ax1.axhline(y=engine.initial_capital, color='gray', linestyle='--', label='Initial Capital')
        ax1.set_ylabel('Equity (HKD)')
        ax1.set_title(f'{normalize_ticker(ticker)} Backtest Equity Curve')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 价格曲线
        ax2.plot(equity_df['date'], equity_df['price'], label='Price', color='green')
        ax2.set_ylabel('Price (HKD)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 标记交易点
        trades_df = pd.DataFrame(engine.trades)
        if not trades_df.empty:
            trades_df['date'] = pd.to_datetime(trades_df['date'])

            buys = trades_df[trades_df['action'] == 'BUY']
            sells = trades_df[trades_df['action'] == 'SELL']

            ax2.scatter(buys['date'], buys['price'], marker='^', color='red', s=100, label='Buy', zorder=5)
            ax2.scatter(sells['date'], sells['price'], marker='v', color='blue', s=100, label='Sell', zorder=5)
            ax2.legend()

        plt.tight_layout()

        ticker_clean = normalize_ticker(ticker).replace('.', '_')
        output_file = os.path.join(output_dir, f"{ticker_clean}_equity_curve.png")
        plt.savefig(output_file, dpi=150)
        plt.close()

        print(f"权益曲线图已保存: {output_file}")

    except ImportError:
        print("提示: 安装matplotlib可生成权益曲线图 (pip install matplotlib)")


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Momentum Catcher - 回测模块')
    parser.add_argument('--ticker', '-t', required=True, help='股票代码（如 02513.HK）')
    parser.add_argument('--period', '-p', default='6mo', help='数据周期（1mo, 3mo, 6mo, 1y）')
    parser.add_argument('--capital', '-c', type=float, default=100000, help='初始资金')
    parser.add_argument('--event-score', type=float, default=4.0, help='事件强度评分（1-5）')
    parser.add_argument('--capital-score', type=int, default=3, help='资金确认评分（0-5）')
    parser.add_argument('--output-dir', '-o', default='.', help='输出目录')
    parser.add_argument('--plot', action='store_true', help='生成权益曲线图')

    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取数据
    print(f"\n开始回测 {args.ticker}...")
    df = fetch_hk_stock_data(args.ticker, args.period)

    if df.empty:
        print("无法获取数据，请检查股票代码")
        sys.exit(1)

    # 计算指标
    print("正在计算技术指标...")
    df = calculate_momentum_indicators(df)

    # 运行回测
    print("正在运行回测...")
    engine, signals_df = run_backtest(
        df,
        event_score=args.event_score,
        capital_score=args.capital_score,
        initial_capital=args.capital
    )

    # 生成报告
    report = generate_backtest_report(args.ticker, engine, signals_df)
    print(report)

    # 保存结果
    save_backtest_results(args.ticker, engine, signals_df, args.output_dir)

    # 保存报告
    ticker_clean = normalize_ticker(args.ticker).replace('.', '_')
    report_file = os.path.join(args.output_dir, f"{ticker_clean}_report.md")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"回测报告已保存: {report_file}")

    # 绘制图表（可选）
    if args.plot:
        plot_equity_curve(engine, args.ticker, args.output_dir)


if __name__ == '__main__':
    main()
