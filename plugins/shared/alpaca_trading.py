#!/usr/bin/env python3
"""
Alpaca 美股交易与数据接口

功能:
- 实时行情数据（免费，无延迟）
- 历史K线数据
- 纸盘交易（模拟）
- 实盘交易（需要账户）

配置:
    export ALPACA_API_KEY="your_key"
    export ALPACA_SECRET_KEY="your_secret"

    # 纸盘（测试）
    export ALPACA_PAPER="true"

注册: https://alpaca.markets/ (免费)
"""

import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union

try:
    import pandas as pd
except ImportError:
    pd = None


# ============== 配置 ==============

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"


def _check_alpaca():
    """检查 Alpaca 配置"""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        return {"error": "请设置 ALPACA_API_KEY 和 ALPACA_SECRET_KEY 环境变量"}
    return None


# ============== 行情数据 ==============

def get_alpaca_bars(
    symbol: str,
    start_date: str,
    end_date: str = None,
    timeframe: str = "1Day"
) -> Union[Dict, "pd.DataFrame"]:
    """
    获取美股历史K线数据（Alpaca）

    Args:
        symbol: 股票代码，如 'AAPL', 'TSLA'
        start_date: 开始日期 'YYYY-MM-DD'
        end_date: 结束日期，默认今天
        timeframe: K线周期
            - '1Min', '5Min', '15Min', '30Min', '1Hour'
            - '1Day', '1Week', '1Month'

    Returns:
        DataFrame: open, high, low, close, volume, vwap, trade_count
    """
    try:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    except ImportError:
        return {"error": "需要安装 alpaca-py: pip install alpaca-py"}

    # 解析 timeframe
    timeframe_map = {
        "1Min": TimeFrame.Minute,
        "5Min": TimeFrame(5, TimeFrameUnit.Minute),
        "15Min": TimeFrame(15, TimeFrameUnit.Minute),
        "30Min": TimeFrame(30, TimeFrameUnit.Minute),
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
        "1Week": TimeFrame.Week,
        "1Month": TimeFrame.Month,
    }

    tf = timeframe_map.get(timeframe, TimeFrame.Day)

    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    try:
        # 可以不提供 key 获取免费数据
        client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            start=datetime.strptime(start_date, '%Y-%m-%d'),
            end=datetime.strptime(end_date, '%Y-%m-%d'),
            timeframe=tf
        )

        bars = client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            return {"error": f"未找到 {symbol} 的数据"}

        # 重置索引
        df = df.reset_index()
        if 'symbol' in df.columns:
            df = df.drop(columns=['symbol'])
        df = df.set_index('timestamp')

        return df

    except Exception as e:
        return {"error": f"Alpaca 获取数据失败: {str(e)}"}


def get_alpaca_quote(symbol: str) -> Dict:
    """
    获取美股实时报价（Alpaca）

    Returns:
        dict: bid, ask, bid_size, ask_size, timestamp
    """
    try:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
    except ImportError:
        return {"error": "需要安装 alpaca-py: pip install alpaca-py"}

    try:
        client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )

        request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        quote = client.get_stock_latest_quote(request)

        q = quote[symbol]
        return {
            "symbol": symbol,
            "bid": q.bid_price,
            "ask": q.ask_price,
            "bid_size": q.bid_size,
            "ask_size": q.ask_size,
            "timestamp": str(q.timestamp),
        }

    except Exception as e:
        return {"error": f"获取报价失败: {str(e)}"}


def get_alpaca_snapshot(symbols: List[str]) -> Dict:
    """
    批量获取美股快照（当前价格、成交量等）

    Args:
        symbols: 股票代码列表，如 ['AAPL', 'TSLA', 'NVDA']

    Returns:
        dict: 每个股票的快照数据
    """
    try:
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockSnapshotRequest
    except ImportError:
        return {"error": "需要安装 alpaca-py: pip install alpaca-py"}

    try:
        client = StockHistoricalDataClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_SECRET_KEY
        )

        request = StockSnapshotRequest(symbol_or_symbols=symbols)
        snapshots = client.get_stock_snapshot(request)

        result = {}
        for symbol, snap in snapshots.items():
            result[symbol] = {
                "price": snap.latest_trade.price if snap.latest_trade else None,
                "volume": snap.daily_bar.volume if snap.daily_bar else None,
                "open": snap.daily_bar.open if snap.daily_bar else None,
                "high": snap.daily_bar.high if snap.daily_bar else None,
                "low": snap.daily_bar.low if snap.daily_bar else None,
                "close": snap.daily_bar.close if snap.daily_bar else None,
                "vwap": snap.daily_bar.vwap if snap.daily_bar else None,
                "prev_close": snap.prev_daily_bar.close if snap.prev_daily_bar else None,
            }

            # 计算涨跌幅
            if result[symbol]["price"] and result[symbol]["prev_close"]:
                change = result[symbol]["price"] - result[symbol]["prev_close"]
                change_pct = (change / result[symbol]["prev_close"]) * 100
                result[symbol]["change"] = round(change, 2)
                result[symbol]["change_pct"] = round(change_pct, 2)

        return result

    except Exception as e:
        return {"error": f"获取快照失败: {str(e)}"}


# ============== 交易接口 ==============

def get_trading_client():
    """获取交易客户端"""
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        return None, {"error": "需要安装 alpaca-py: pip install alpaca-py"}

    err = _check_alpaca()
    if err:
        return None, err

    client = TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=ALPACA_PAPER
    )

    return client, None


def get_account() -> Dict:
    """
    获取账户信息

    Returns:
        dict: 账户余额、购买力、持仓市值等
    """
    client, err = get_trading_client()
    if err:
        return err

    try:
        account = client.get_account()
        return {
            "account_id": account.id,
            "status": account.status.value,
            "currency": account.currency,
            "cash": float(account.cash),
            "portfolio_value": float(account.portfolio_value),
            "buying_power": float(account.buying_power),
            "equity": float(account.equity),
            "last_equity": float(account.last_equity),
            "long_market_value": float(account.long_market_value),
            "short_market_value": float(account.short_market_value),
            "daytrade_count": account.daytrade_count,
            "pattern_day_trader": account.pattern_day_trader,
            "trading_blocked": account.trading_blocked,
            "paper": ALPACA_PAPER,
        }
    except Exception as e:
        return {"error": f"获取账户失败: {str(e)}"}


def get_positions() -> List[Dict]:
    """
    获取当前持仓

    Returns:
        list: 持仓列表
    """
    client, err = get_trading_client()
    if err:
        return err

    try:
        positions = client.get_all_positions()
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_entry_price": float(p.avg_entry_price),
                "current_price": float(p.current_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,  # 转为百分比
                "side": p.side.value,
            }
            for p in positions
        ]
    except Exception as e:
        return {"error": f"获取持仓失败: {str(e)}"}


def submit_order(
    symbol: str,
    qty: float,
    side: str,
    order_type: str = "market",
    limit_price: float = None,
    stop_price: float = None,
    time_in_force: str = "day"
) -> Dict:
    """
    提交订单

    Args:
        symbol: 股票代码
        qty: 数量（支持小数股）
        side: 方向 'buy' 或 'sell'
        order_type: 订单类型
            - 'market': 市价单
            - 'limit': 限价单
            - 'stop': 止损单
            - 'stop_limit': 止损限价单
        limit_price: 限价（limit/stop_limit 订单需要）
        stop_price: 止损价（stop/stop_limit 订单需要）
        time_in_force: 有效期
            - 'day': 当日有效
            - 'gtc': 取消前有效
            - 'ioc': 立即成交或取消
            - 'fok': 全部成交或取消

    Returns:
        dict: 订单信息
    """
    try:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest, StopLimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    except ImportError:
        return {"error": "需要安装 alpaca-py: pip install alpaca-py"}

    client, err = get_trading_client()
    if err:
        return err

    # 映射参数
    side_map = {"buy": OrderSide.BUY, "sell": OrderSide.SELL}
    tif_map = {
        "day": TimeInForce.DAY,
        "gtc": TimeInForce.GTC,
        "ioc": TimeInForce.IOC,
        "fok": TimeInForce.FOK,
    }

    try:
        if order_type == "market":
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_map[side],
                time_in_force=tif_map.get(time_in_force, TimeInForce.DAY)
            )
        elif order_type == "limit":
            if limit_price is None:
                return {"error": "限价单需要指定 limit_price"}
            request = LimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_map[side],
                limit_price=limit_price,
                time_in_force=tif_map.get(time_in_force, TimeInForce.DAY)
            )
        elif order_type == "stop":
            if stop_price is None:
                return {"error": "止损单需要指定 stop_price"}
            request = StopOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_map[side],
                stop_price=stop_price,
                time_in_force=tif_map.get(time_in_force, TimeInForce.DAY)
            )
        elif order_type == "stop_limit":
            if stop_price is None or limit_price is None:
                return {"error": "止损限价单需要指定 stop_price 和 limit_price"}
            request = StopLimitOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side_map[side],
                stop_price=stop_price,
                limit_price=limit_price,
                time_in_force=tif_map.get(time_in_force, TimeInForce.DAY)
            )
        else:
            return {"error": f"不支持的订单类型: {order_type}"}

        order = client.submit_order(request)

        return {
            "order_id": str(order.id),
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "qty": str(order.qty),
            "filled_qty": str(order.filled_qty),
            "side": order.side.value,
            "type": order.type.value,
            "status": order.status.value,
            "created_at": str(order.created_at),
            "paper": ALPACA_PAPER,
        }

    except Exception as e:
        return {"error": f"提交订单失败: {str(e)}"}


def cancel_order(order_id: str) -> Dict:
    """取消订单"""
    client, err = get_trading_client()
    if err:
        return err

    try:
        client.cancel_order_by_id(order_id)
        return {"success": True, "order_id": order_id}
    except Exception as e:
        return {"error": f"取消订单失败: {str(e)}"}


def get_orders(status: str = "open") -> List[Dict]:
    """
    获取订单列表

    Args:
        status: 订单状态
            - 'open': 未成交
            - 'closed': 已成交
            - 'all': 全部
    """
    try:
        from alpaca.trading.requests import GetOrdersRequest
        from alpaca.trading.enums import QueryOrderStatus
    except ImportError:
        return {"error": "需要安装 alpaca-py: pip install alpaca-py"}

    client, err = get_trading_client()
    if err:
        return err

    status_map = {
        "open": QueryOrderStatus.OPEN,
        "closed": QueryOrderStatus.CLOSED,
        "all": QueryOrderStatus.ALL,
    }

    try:
        request = GetOrdersRequest(status=status_map.get(status, QueryOrderStatus.ALL))
        orders = client.get_orders(request)

        return [
            {
                "order_id": str(o.id),
                "symbol": o.symbol,
                "qty": str(o.qty),
                "filled_qty": str(o.filled_qty),
                "side": o.side.value,
                "type": o.type.value,
                "status": o.status.value,
                "limit_price": str(o.limit_price) if o.limit_price else None,
                "stop_price": str(o.stop_price) if o.stop_price else None,
                "filled_avg_price": str(o.filled_avg_price) if o.filled_avg_price else None,
                "created_at": str(o.created_at),
            }
            for o in orders
        ]
    except Exception as e:
        return {"error": f"获取订单失败: {str(e)}"}


# ============== CLI ==============

def main():
    import sys
    import json

    if len(sys.argv) < 2:
        print("""
Alpaca 美股交易接口

用法:
    python alpaca_trading.py <命令> [参数]

数据命令:
    bars <symbol> [start_date]    获取K线数据
    quote <symbol>                获取实时报价
    snapshot <symbols>            批量获取快照 (逗号分隔)

交易命令:
    account                       查看账户
    positions                     查看持仓
    orders [status]               查看订单 (open/closed/all)
    buy <symbol> <qty>            市价买入
    sell <symbol> <qty>           市价卖出
    cancel <order_id>             取消订单

示例:
    python alpaca_trading.py bars AAPL 2024-01-01
    python alpaca_trading.py snapshot AAPL,TSLA,NVDA
    python alpaca_trading.py buy AAPL 1
    python alpaca_trading.py account

环境变量:
    ALPACA_API_KEY       API Key
    ALPACA_SECRET_KEY    Secret Key
    ALPACA_PAPER         是否纸盘 (true/false, 默认 true)
        """)
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "bars" and len(sys.argv) >= 3:
        symbol = sys.argv[2]
        start = sys.argv[3] if len(sys.argv) > 3 else "2024-01-01"
        result = get_alpaca_bars(symbol, start)
        if isinstance(result, dict) and "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(result.tail(10).to_string())

    elif cmd == "quote" and len(sys.argv) >= 3:
        result = get_alpaca_quote(sys.argv[2])
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "snapshot" and len(sys.argv) >= 3:
        symbols = sys.argv[2].split(',')
        result = get_alpaca_snapshot(symbols)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "account":
        result = get_account()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "positions":
        result = get_positions()
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "orders":
        status = sys.argv[2] if len(sys.argv) > 2 else "open"
        result = get_orders(status)
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "buy" and len(sys.argv) >= 4:
        symbol = sys.argv[2]
        qty = float(sys.argv[3])
        result = submit_order(symbol, qty, "buy")
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "sell" and len(sys.argv) >= 4:
        symbol = sys.argv[2]
        qty = float(sys.argv[3])
        result = submit_order(symbol, qty, "sell")
        print(json.dumps(result, indent=2, default=str))

    elif cmd == "cancel" and len(sys.argv) >= 3:
        result = cancel_order(sys.argv[2])
        print(json.dumps(result, indent=2, default=str))

    else:
        print(f"❌ 未知命令: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
