#!/usr/bin/env python3
"""
AI 小盘股爆发策略 - 回测模块

基于历史数据验证策略效果

使用方法:
    python backtest.py 02513 --start 2026-01-08 --end 2026-02-27
"""

import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except ImportError:
    print("需要安装依赖: pip install pandas numpy")
    sys.exit(1)

sys.path.insert(0, '../shared')
try:
    from market_data import get_hk_stock, calculate_indicators
except ImportError:
    print("请确保 market_data.py 在 ../shared/ 目录下")
    sys.exit(1)


class SmallCapBacktester:
    """小盘股爆发策略回测器"""

    def __init__(
        self,
        ticker: str,
        start_date: str,
        end_date: str = None,
        initial_capital: float = 100000
    ):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.initial_capital = initial_capital

        # 加载数据
        self.df = self._load_data()
        if self.df is None:
            raise ValueError(f"无法加载 {ticker} 的数据")

        # 交易记录
        self.trades: List[Dict] = []
        self.positions: List[Dict] = []
        self.equity_curve: List[float] = []

    def _load_data(self) -> Optional[pd.DataFrame]:
        """加载历史数据"""
        df = get_hk_stock(self.ticker, self.start_date, self.end_date)

        if isinstance(df, dict) and "error" in df:
            print(f"错误: {df['error']}")
            return None

        df = calculate_indicators(df)
        return df

    def generate_signals(self) -> pd.DataFrame:
        """
        生成交易信号

        信号规则:
        1. 入场信号：成交额放量(>1.5x MA20) + 涨幅>5% + MACD金叉
        2. 退出信号：跌破MA10 或 换手率>40% 或 量价背离
        """
        df = self.df.copy()

        # 计算必要指标
        df['ma20_turnover'] = df['turnover'].rolling(20).mean() if 'turnover' in df.columns else df['volume'].rolling(20).mean() * df['close']
        df['volume_ratio'] = df['turnover'] / df['ma20_turnover'] if 'turnover' in df.columns else df['volume'] / df['volume'].rolling(20).mean()

        df['daily_return'] = df['close'].pct_change() * 100
        df['ma10'] = df['close'].rolling(10).mean()
        df['ma20'] = df['close'].rolling(20).mean()

        # MACD
        df['ema12'] = df['close'].ewm(span=12).mean()
        df['ema26'] = df['close'].ewm(span=26).mean()
        df['macd'] = df['ema12'] - df['ema26']
        df['signal_line'] = df['macd'].ewm(span=9).mean()
        df['macd_cross'] = (df['macd'] > df['signal_line']) & (df['macd'].shift(1) <= df['signal_line'].shift(1))

        # 入场条件
        df['entry_signal'] = (
            (df['volume_ratio'] > 1.5) &  # 放量
            (df['daily_return'] > 5) &     # 涨幅>5%
            (df['macd_cross'] | (df['macd'] > df['signal_line']))  # MACD金叉或多头
        )

        # 退出条件
        df['exit_signal'] = (
            (df['close'] < df['ma10']) |  # 跌破MA10
            ((df['turnover_rate'] > 40) if 'turnover_rate' in df.columns else False)  # 换手率极端
        )

        # 量价背离检测
        df['price_new_high'] = df['high'] == df['high'].cummax()
        df['volume_divergence'] = df['price_new_high'] & (df['volume'] < df['volume'].shift(1) * 0.7)
        df['exit_signal'] = df['exit_signal'] | df['volume_divergence']

        return df

    def run_backtest(self) -> Dict:
        """运行回测"""
        df = self.generate_signals()

        capital = self.initial_capital
        position = 0
        shares = 0
        entry_price = 0
        entry_date = None

        for idx, row in df.iterrows():
            date = idx if isinstance(idx, str) else idx.strftime('%Y-%m-%d')
            price = row['close']

            # 更新净值
            if position > 0:
                current_value = capital + shares * price
            else:
                current_value = capital
            self.equity_curve.append(current_value)

            # 检查入场信号
            if position == 0 and row.get('entry_signal', False):
                # 入场：使用80%资金
                invest_amount = capital * 0.8
                shares = int(invest_amount / price / 100) * 100  # 整手
                if shares > 0:
                    entry_price = price
                    entry_date = date
                    capital -= shares * price
                    position = 1

                    self.trades.append({
                        'date': date,
                        'action': 'BUY',
                        'price': price,
                        'shares': shares,
                        'value': shares * price,
                        'signal': '入场信号'
                    })

            # 检查退出信号
            elif position > 0:
                # 移动止盈
                days_held = len([t for t in self.trades if t['action'] == 'BUY'])
                if days_held <= 3:
                    trail_stop = df['high'].max() * 0.85
                elif days_held <= 5:
                    trail_stop = df['high'].max() * 0.88
                else:
                    trail_stop = df['high'].max() * 0.90

                # 固定止损
                stop_loss = entry_price * 0.85

                exit_reason = None
                if row.get('exit_signal', False):
                    exit_reason = '技术信号退出'
                elif price < stop_loss:
                    exit_reason = '止损退出'
                elif price < trail_stop:
                    exit_reason = '移动止盈退出'

                if exit_reason:
                    # 清仓
                    capital += shares * price
                    self.trades.append({
                        'date': date,
                        'action': 'SELL',
                        'price': price,
                        'shares': shares,
                        'value': shares * price,
                        'signal': exit_reason,
                        'pnl': (price - entry_price) * shares,
                        'return': (price / entry_price - 1) * 100
                    })
                    shares = 0
                    position = 0
                    entry_price = 0

        # 计算最终结果
        if position > 0:
            final_value = capital + shares * df.iloc[-1]['close']
        else:
            final_value = capital

        # Buy & Hold 对比
        bh_shares = int(self.initial_capital * 0.8 / df.iloc[0]['close'] / 100) * 100
        bh_final = self.initial_capital * 0.2 + bh_shares * df.iloc[-1]['close']

        # 计算统计指标
        equity_series = pd.Series(self.equity_curve)
        returns = equity_series.pct_change().dropna()
        max_drawdown = self._calculate_max_drawdown(equity_series)

        results = {
            'ticker': self.ticker,
            'period': f"{self.start_date} 至 {self.end_date}",
            'initial_capital': self.initial_capital,
            'final_value': round(final_value, 2),
            'total_return': round((final_value / self.initial_capital - 1) * 100, 2),
            'buy_hold_return': round((bh_final / self.initial_capital - 1) * 100, 2),
            'excess_return': round((final_value / self.initial_capital - bh_final / self.initial_capital) * 100, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(self._calculate_sharpe(returns), 2),
            'total_trades': len([t for t in self.trades if t['action'] == 'BUY']),
            'win_rate': self._calculate_win_rate(),
            'trades': self.trades
        }

        return results

    def _calculate_max_drawdown(self, equity: pd.Series) -> float:
        """计算最大回撤"""
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        return abs(drawdown.min())

    def _calculate_sharpe(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) < 2 or returns.std() == 0:
            return 0
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_win_rate(self) -> float:
        """计算胜率"""
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        if not sell_trades:
            return 0
        wins = sum(1 for t in sell_trades if t.get('pnl', 0) > 0)
        return round(wins / len(sell_trades) * 100, 2)

    def print_report(self, results: Dict):
        """打印回测报告"""
        print("\n" + "="*60)
        print(f"  {results['ticker']} 策略回测报告")
        print("="*60)
        print(f"回测区间: {results['period']}")
        print(f"初始资金: {results['initial_capital']:,.0f} HKD")
        print()
        print("📈 策略表现:")
        print(f"   最终净值: {results['final_value']:,.0f} HKD")
        print(f"   总收益: {results['total_return']:+.1f}%")
        print(f"   最大回撤: {results['max_drawdown']:.1f}%")
        print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
        print()
        print("📊 Buy & Hold 对比:")
        print(f"   B&H 收益: {results['buy_hold_return']:+.1f}%")
        print(f"   超额收益: {results['excess_return']:+.1f}%")
        print()
        print("💼 交易统计:")
        print(f"   交易次数: {results['total_trades']}")
        print(f"   胜率: {results['win_rate']}%")
        print()

        if results['trades']:
            print("📝 交易记录:")
            for trade in results['trades']:
                action = "买入" if trade['action'] == 'BUY' else "卖出"
                print(f"   {trade['date']} {action} @ {trade['price']:.2f} ({trade['signal']})")
                if trade['action'] == 'SELL':
                    print(f"      盈亏: {trade.get('pnl', 0):+,.0f} HKD ({trade.get('return', 0):+.1f}%)")

        print("\n" + "="*60)


def run_zhipu_case_study():
    """运行智谱案例回测"""
    print("\n📌 智谱AI (02513) 案例回测")
    print("   IPO: 2026-01-08, 发行价 ~116 HKD")
    print("   高点: 2026-02-20, 最高 725 HKD (+524%)")
    print()

    backtester = SmallCapBacktester(
        ticker="02513",
        start_date="2026-01-08",
        end_date="2026-02-27",
        initial_capital=100000
    )

    results = backtester.run_backtest()
    backtester.print_report(results)

    return results


def main():
    if len(sys.argv) < 2:
        print("""
AI 小盘股爆发策略 - 回测模块

用法:
    python backtest.py <ticker> [--start DATE] [--end DATE] [--capital AMOUNT]

参数:
    ticker      港股代码
    --start     开始日期 (YYYY-MM-DD)
    --end       结束日期 (YYYY-MM-DD)
    --capital   初始资金 (默认 100000)
    --zhipu     运行智谱案例回测

示例:
    python backtest.py 02513 --start 2026-01-08 --end 2026-02-27
    python backtest.py --zhipu
        """)
        sys.exit(1)

    # 智谱案例
    if '--zhipu' in sys.argv:
        run_zhipu_case_study()
        return

    ticker = sys.argv[1]

    # 解析参数
    start_date = None
    end_date = None
    capital = 100000

    for i, arg in enumerate(sys.argv):
        if arg == '--start' and i + 1 < len(sys.argv):
            start_date = sys.argv[i + 1]
        elif arg == '--end' and i + 1 < len(sys.argv):
            end_date = sys.argv[i + 1]
        elif arg == '--capital' and i + 1 < len(sys.argv):
            capital = float(sys.argv[i + 1])

    if not start_date:
        start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')

    try:
        backtester = SmallCapBacktester(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital
        )

        results = backtester.run_backtest()
        backtester.print_report(results)

        # 输出JSON（可选）
        if '--json' in sys.argv:
            print("\n" + json.dumps(results, indent=2, ensure_ascii=False, default=str))

    except Exception as e:
        print(f"❌ 回测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
