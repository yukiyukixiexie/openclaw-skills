#!/usr/bin/env python3
"""
港股分析师数据模块
获取港股的分析师预期、目标价、评级等数据
数据源优先级: yfinance > 东方财富 > 新浪财经
"""

import json
import sys
import requests
from typing import Optional, Dict, List
from datetime import datetime

# 尝试导入 yfinance
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

# 尝试导入 akshare (更可靠的港股数据源)
try:
    import akshare as ak
    import pandas as pd
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False


class HKAnalystDataFetcher:
    """港股分析师数据获取器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "application/json",
        })

    def get_all_analyst_data(self, stock_code: str) -> Dict:
        """
        获取完整的分析师数据

        Args:
            stock_code: 股票代码（如 "00700" 或 "0700" 或 "700"）

        Returns:
            dict: 包含 eps_forecast, revenue_forecast, price_targets, ratings 等
        """
        code = self._normalize_code(stock_code)
        result = {
            "stock_code": code,
            "symbol": f"{code}.HK",
            "fetch_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "data_sources": [],
        }

        # 1. 从 yfinance 获取数据（主要来源）
        yf_data = self._fetch_from_yfinance(code)
        if yf_data.get("success"):
            result["data_sources"].append("yfinance")
            result.update(yf_data["data"])

        # 2. 从东方财富获取补充数据
        em_data = self._fetch_from_eastmoney(code)
        if em_data.get("success"):
            result["data_sources"].append("eastmoney")
            # 合并数据（如果 yfinance 没有的字段）
            for key, value in em_data["data"].items():
                if key not in result or result[key] is None:
                    result[key] = value

        # 3. 从腾讯获取实时数据
        qq_data = self._fetch_from_tencent(code)
        if qq_data.get("success"):
            result["data_sources"].append("tencent")
            for key, value in qq_data["data"].items():
                if value is not None:
                    # 腾讯价格数据优先
                    if key in ["current_price", "change_pct"] or key not in result:
                        result[key] = value

        # 4. 从新浪获取补充数据
        sina_data = self._fetch_from_sina(code)
        if sina_data.get("success"):
            result["data_sources"].append("sina")
            for key, value in sina_data["data"].items():
                if value is not None and key not in result:
                    result[key] = value

        # 5. 从 AKShare 获取历史数据（如果可用）
        ak_data = self._fetch_from_akshare(code)
        if ak_data.get("success"):
            result["data_sources"].append("akshare")
            for key, value in ak_data["data"].items():
                if value is not None and key not in result:
                    result[key] = value

        # 6. 数据质量检查
        result["data_quality"] = self._assess_data_quality(result)

        return result

    def _normalize_code(self, stock_code: str) -> str:
        """标准化股票代码为5位"""
        code = stock_code.replace(".HK", "").replace(".hk", "").strip()
        return code.zfill(5)

    def _get_yfinance_symbol(self, code: str) -> str:
        """
        获取 yfinance 格式的港股代码
        yfinance 使用不带前导零的格式，如 "0700.HK" 而非 "00700.HK"
        """
        # 去掉前导零但保留至少4位
        code_int = int(code)
        if code_int < 10:
            return f"000{code_int}.HK"
        elif code_int < 100:
            return f"00{code_int}.HK"
        elif code_int < 1000:
            return f"0{code_int}.HK"
        else:
            return f"{code_int}.HK"

    def _fetch_from_yfinance(self, code: str) -> Dict:
        """从 yfinance 获取分析师数据"""
        if not YFINANCE_AVAILABLE:
            return {"success": False, "error": "yfinance not installed"}

        try:
            yf_symbol = self._get_yfinance_symbol(code)
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            # 基本信息
            data = {
                "company_name": info.get("shortName") or info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "currency": info.get("currency", "HKD"),
            }

            # 当前价格数据
            data["current_price"] = info.get("currentPrice") or info.get("regularMarketPrice")
            data["market_cap"] = info.get("marketCap")
            data["pe_ratio"] = info.get("trailingPE")
            data["forward_pe"] = info.get("forwardPE")
            data["pb_ratio"] = info.get("priceToBook")
            data["dividend_yield"] = info.get("dividendYield")

            # 分析师目标价
            data["price_targets"] = {
                "mean": info.get("targetMeanPrice"),
                "high": info.get("targetHighPrice"),
                "low": info.get("targetLowPrice"),
                "median": info.get("targetMedianPrice"),
                "num_analysts": info.get("numberOfAnalystOpinions"),
            }

            # 计算目标价上涨空间
            if data["current_price"] and data["price_targets"]["mean"]:
                upside = (data["price_targets"]["mean"] / data["current_price"] - 1) * 100
                data["price_targets"]["upside_pct"] = round(upside, 2)

            # 分析师推荐
            data["recommendation"] = {
                "key": info.get("recommendationKey"),
                "mean": info.get("recommendationMean"),  # 1=Strong Buy, 5=Strong Sell
            }

            # 盈利预期
            data["eps_forecast"] = {
                "trailing_eps": info.get("trailingEps"),
                "forward_eps": info.get("forwardEps"),
            }

            # 收入预期
            data["revenue_info"] = {
                "total_revenue": info.get("totalRevenue"),
                "revenue_growth": info.get("revenueGrowth"),
            }

            # 利润率
            data["margins"] = {
                "gross": info.get("grossMargins"),
                "operating": info.get("operatingMargins"),
                "net": info.get("profitMargins"),
            }

            # 获取分析师评级分布（如果有）
            try:
                recommendations = ticker.recommendations
                if recommendations is not None and not recommendations.empty:
                    # 获取最近一年的评级
                    recent = recommendations.tail(50)
                    data["ratings_history"] = []
                    for _, row in recent.iterrows():
                        data["ratings_history"].append({
                            "firm": row.get("Firm", "Unknown"),
                            "to_grade": row.get("To Grade", row.get("toGrade")),
                            "from_grade": row.get("From Grade", row.get("fromGrade")),
                            "action": row.get("Action", row.get("action")),
                        })
            except Exception:
                pass

            return {"success": True, "data": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fetch_from_eastmoney(self, code: str) -> Dict:
        """从东方财富获取分析师数据"""
        try:
            # 东方财富港股行情 API
            url = f"https://push2.eastmoney.com/api/qt/stock/get"
            params = {
                "secid": f"116.{code}",  # 港股市场代码
                "fields": "f43,f44,f45,f46,f47,f48,f49,f50,f51,f52,f57,f58,f59,f60,f116,f117,f162,f163,f164,f165,f166,f167,f168,f169,f170",
                "ut": "fa5fd1943c7b386f172d6893dbfba10b",
            }

            resp = self.session.get(url, params=params, timeout=10)
            json_data = resp.json()

            if not json_data.get("data"):
                return {"success": False, "error": "No data from eastmoney"}

            raw = json_data["data"]
            data = {}

            # 解析字段
            if "f57" in raw:
                data["em_code"] = raw["f57"]
            if "f58" in raw:
                data["em_name"] = raw["f58"]

            # 研报评级数据（如果有）
            # 东方财富的研报接口
            report_url = "https://datacenter.eastmoney.com/securities/api/data/get"
            report_params = {
                "type": "RPT_HK_HOLD_STOCK_ANALYREPORT",
                "sty": "ALL",
                "filter": f'(SECURITY_CODE="{code}")',
                "p": "1",
                "ps": "20",
                "st": "REPORT_DATE",
                "sr": "-1",
            }

            try:
                report_resp = self.session.get(report_url, params=report_params, timeout=10)
                report_json = report_resp.json()
                if report_json.get("result", {}).get("data"):
                    reports = report_json["result"]["data"]
                    data["em_reports"] = []
                    for r in reports[:10]:  # 最近10篇
                        data["em_reports"].append({
                            "institution": r.get("ORG_NAME"),
                            "rating": r.get("RATING"),
                            "target_price": r.get("TARGET_PRICE"),
                            "report_date": r.get("REPORT_DATE"),
                            "title": r.get("TITLE"),
                        })
            except Exception:
                pass

            return {"success": True, "data": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fetch_from_tencent(self, code: str) -> Dict:
        """从腾讯股票 API 获取实时数据"""
        try:
            url = f"https://qt.gtimg.cn/q=r_hk{code}"
            resp = self.session.get(url, timeout=10)

            if resp.status_code != 200:
                return {"success": False, "error": f"HTTP {resp.status_code}"}

            # 解析数据: v_r_hk00100="100~MINIMAX-WP~00100~764.500~..."
            text = resp.text
            if '="' not in text:
                return {"success": False, "error": "Invalid response format"}

            data_str = text.split('="')[1].rstrip('";')
            parts = data_str.split('~')

            if len(parts) < 50:
                return {"success": False, "error": "Insufficient data fields"}

            data = {
                "qq_name": parts[1],
                "current_price": float(parts[3]) if parts[3] else None,
                "prev_close": float(parts[4]) if parts[4] else None,
                "qq_open": float(parts[5]) if parts[5] else None,
                "qq_volume": float(parts[6]) if parts[6] else None,
                "qq_high": float(parts[33]) if len(parts) > 33 and parts[33] else None,
                "qq_low": float(parts[34]) if len(parts) > 34 and parts[34] else None,
                "qq_turnover": float(parts[37]) if len(parts) > 37 and parts[37] else None,
                "qq_amplitude": float(parts[43]) if len(parts) > 43 and parts[43] else None,
                "qq_pe": float(parts[52]) if len(parts) > 52 and parts[52] else None,
                "qq_52w_high": float(parts[47]) if len(parts) > 47 and parts[47] else None,
                "qq_52w_low": float(parts[48]) if len(parts) > 48 and parts[48] else None,
            }

            # 计算涨跌
            if data["current_price"] and data["prev_close"]:
                data["change"] = round(data["current_price"] - data["prev_close"], 3)
                data["change_pct"] = round((data["change"] / data["prev_close"]) * 100, 2)

            return {"success": True, "data": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fetch_from_sina(self, code: str) -> Dict:
        """从新浪股票 API 获取实时数据"""
        try:
            url = f"https://hq.sinajs.cn/list=rt_hk{code}"
            headers = {"Referer": "https://finance.sina.com.cn"}
            resp = self.session.get(url, headers=headers, timeout=10)

            if resp.status_code != 200:
                return {"success": False, "error": f"HTTP {resp.status_code}"}

            # 解析: var hq_str_rt_hk00100="MINIMAX-WP,MINIMAX-WP,703.500,..."
            text = resp.text
            if '="' not in text:
                return {"success": False, "error": "Invalid response format"}

            data_str = text.split('="')[1].rstrip('";')
            parts = data_str.split(',')

            if len(parts) < 15:
                return {"success": False, "error": "Insufficient data fields"}

            data = {
                "sina_name_cn": parts[0],
                "sina_name_en": parts[1],
                "sina_open": float(parts[2]) if parts[2] else None,
                "prev_close": float(parts[3]) if parts[3] else None,
                "sina_high": float(parts[4]) if parts[4] else None,
                "sina_low": float(parts[5]) if parts[5] else None,
                "current_price": float(parts[6]) if parts[6] else None,
                "sina_change": float(parts[7]) if parts[7] else None,
                "change_pct": float(parts[8]) if parts[8] else None,
                "sina_turnover": float(parts[11]) if len(parts) > 11 and parts[11] else None,
                "sina_volume": float(parts[12]) if len(parts) > 12 and parts[12] else None,
                "sina_52w_high": float(parts[15]) if len(parts) > 15 and parts[15] else None,
                "sina_52w_low": float(parts[16]) if len(parts) > 16 and parts[16] else None,
            }

            return {"success": True, "data": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _fetch_from_akshare(self, code: str) -> Dict:
        """从 AKShare 获取港股历史数据（更可靠的数据源）"""
        if not AKSHARE_AVAILABLE:
            return {"success": False, "error": "akshare not installed"}

        try:
            # 获取历史数据
            df = ak.stock_hk_hist(symbol=code, period="daily", adjust="qfq")

            if df.empty:
                return {"success": False, "error": "No data from akshare"}

            # 转换日期格式
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values('日期').reset_index(drop=True)

            data = {
                # 基本信息
                "ak_trading_days": len(df),
                "ak_data_start": df['日期'].min().strftime("%Y-%m-%d"),
                "ak_data_end": df['日期'].max().strftime("%Y-%m-%d"),

                # 价格数据（覆盖 yfinance）
                "current_price": float(df['收盘'].iloc[-1]),
                "ak_open": float(df['开盘'].iloc[-1]),
                "ak_high": float(df['最高'].iloc[-1]),
                "ak_low": float(df['最低'].iloc[-1]),

                # 历史统计
                "history_high": float(df['最高'].max()),
                "history_high_date": df.loc[df['最高'].idxmax(), '日期'].strftime("%Y-%m-%d"),
                "history_low": float(df['最低'].min()),

                # 涨跌幅
                "change_pct": float(df['涨跌幅'].iloc[-1]),
                "amplitude": float(df['振幅'].iloc[-1]),
            }

            # 计算回撤
            peak = df['收盘'].max()
            current = df['收盘'].iloc[-1]
            data["drawdown_from_high"] = round((peak - current) / peak * 100, 2)

            # 计算区间收益
            if len(df) >= 5:
                data["return_5d"] = round((df['收盘'].iloc[-1] / df['收盘'].iloc[-5] - 1) * 100, 2)
            if len(df) >= 10:
                data["return_10d"] = round((df['收盘'].iloc[-1] / df['收盘'].iloc[-10] - 1) * 100, 2)
            if len(df) >= 20:
                data["return_20d"] = round((df['收盘'].iloc[-1] / df['收盘'].iloc[-20] - 1) * 100, 2)

            # 成交量数据
            data["volume"] = float(df['成交量'].iloc[-1])
            data["turnover"] = float(df['成交额'].iloc[-1])
            if '换手率' in df.columns:
                data["turnover_rate"] = float(df['换手率'].iloc[-1])

            # 最近交易数据
            recent = df.tail(5)
            data["recent_prices"] = []
            for _, row in recent.iterrows():
                data["recent_prices"].append({
                    "date": row['日期'].strftime("%Y-%m-%d"),
                    "close": float(row['收盘']),
                    "change_pct": float(row['涨跌幅']),
                    "turnover": float(row['成交额'] / 1e8),  # 亿
                })

            return {"success": True, "data": data}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _assess_data_quality(self, data: Dict) -> Dict:
        """评估数据质量"""
        num_analysts = data.get("price_targets", {}).get("num_analysts")
        if num_analysts is None:
            num_analysts = 0

        quality = {
            "has_price_targets": bool(data.get("price_targets", {}).get("mean")),
            "has_ratings": bool(data.get("recommendation", {}).get("key")),
            "has_eps_forecast": bool(data.get("eps_forecast", {}).get("forward_eps")),
            "num_analysts": num_analysts,
            "data_completeness": 0,
        }

        # 计算数据完整度
        key_fields = [
            "current_price", "pe_ratio", "price_targets", "recommendation",
            "eps_forecast", "revenue_info", "margins"
        ]
        filled = sum(1 for k in key_fields if data.get(k))
        quality["data_completeness"] = round(filled / len(key_fields) * 100, 1)

        # 数据质量评级
        if num_analysts >= 10 and quality["data_completeness"] >= 80:
            quality["rating"] = "HIGH"
        elif num_analysts >= 5 and quality["data_completeness"] >= 50:
            quality["rating"] = "MEDIUM"
        else:
            quality["rating"] = "LOW"

        return quality


# 便捷函数
def get_hk_eps_forecast(stock_code: str) -> Dict:
    """获取港股 EPS 预期"""
    fetcher = HKAnalystDataFetcher()
    data = fetcher.get_all_analyst_data(stock_code)
    return {
        "stock_code": data.get("stock_code"),
        "company_name": data.get("company_name"),
        "trailing_eps": data.get("eps_forecast", {}).get("trailing_eps"),
        "forward_eps": data.get("eps_forecast", {}).get("forward_eps"),
        "pe_ratio": data.get("pe_ratio"),
        "forward_pe": data.get("forward_pe"),
        "data_sources": data.get("data_sources"),
    }


def get_hk_revenue_forecast(stock_code: str) -> Dict:
    """获取港股收入预期"""
    fetcher = HKAnalystDataFetcher()
    data = fetcher.get_all_analyst_data(stock_code)
    return {
        "stock_code": data.get("stock_code"),
        "company_name": data.get("company_name"),
        "total_revenue": data.get("revenue_info", {}).get("total_revenue"),
        "revenue_growth": data.get("revenue_info", {}).get("revenue_growth"),
        "gross_margin": data.get("margins", {}).get("gross"),
        "operating_margin": data.get("margins", {}).get("operating"),
        "data_sources": data.get("data_sources"),
    }


def get_hk_analyst_ratings(stock_code: str) -> Dict:
    """获取港股分析师评级"""
    fetcher = HKAnalystDataFetcher()
    data = fetcher.get_all_analyst_data(stock_code)
    return {
        "stock_code": data.get("stock_code"),
        "company_name": data.get("company_name"),
        "recommendation": data.get("recommendation"),
        "num_analysts": data.get("price_targets", {}).get("num_analysts"),
        "ratings_history": data.get("ratings_history", [])[:10],
        "data_sources": data.get("data_sources"),
    }


def get_hk_price_targets(stock_code: str) -> Dict:
    """获取港股目标价"""
    fetcher = HKAnalystDataFetcher()
    data = fetcher.get_all_analyst_data(stock_code)
    return {
        "stock_code": data.get("stock_code"),
        "company_name": data.get("company_name"),
        "current_price": data.get("current_price"),
        "price_targets": data.get("price_targets"),
        "em_reports": data.get("em_reports", [])[:5],  # 东方财富研报（含目标价）
        "data_sources": data.get("data_sources"),
    }


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="港股分析师数据获取工具")
    parser.add_argument("stock_code", help="股票代码（如 00700 或 700）")
    parser.add_argument(
        "--type",
        choices=["all", "eps", "revenue", "ratings", "targets"],
        default="all",
        help="数据类型（默认: all）"
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="输出格式（默认: json）"
    )

    args = parser.parse_args()

    # 获取数据
    if args.type == "eps":
        data = get_hk_eps_forecast(args.stock_code)
    elif args.type == "revenue":
        data = get_hk_revenue_forecast(args.stock_code)
    elif args.type == "ratings":
        data = get_hk_analyst_ratings(args.stock_code)
    elif args.type == "targets":
        data = get_hk_price_targets(args.stock_code)
    else:
        fetcher = HKAnalystDataFetcher()
        data = fetcher.get_all_analyst_data(args.stock_code)

    # 输出
    if args.format == "json":
        print(json.dumps(data, ensure_ascii=False, indent=2, default=str))
    else:
        # 文本格式输出
        print(f"\n{'='*60}")
        print(f"港股分析师数据: {data.get('stock_code', args.stock_code)}.HK")
        print(f"公司名称: {data.get('company_name', 'N/A')}")
        print(f"数据来源: {', '.join(data.get('data_sources', []))}")
        print(f"{'='*60}")

        if "current_price" in data:
            print(f"\n当前价格: ${data['current_price']}")

        if "price_targets" in data and data["price_targets"]:
            pt = data["price_targets"]
            print(f"\n目标价:")
            print(f"  均值: ${pt.get('mean', 'N/A')}")
            print(f"  高: ${pt.get('high', 'N/A')}")
            print(f"  低: ${pt.get('low', 'N/A')}")
            print(f"  分析师数: {pt.get('num_analysts', 'N/A')}")
            if pt.get('upside_pct'):
                print(f"  上涨空间: {pt['upside_pct']}%")

        if "recommendation" in data and data["recommendation"]:
            rec = data["recommendation"]
            print(f"\n分析师推荐:")
            print(f"  评级: {rec.get('key', 'N/A')}")
            print(f"  平均分: {rec.get('mean', 'N/A')} (1=强买, 5=强卖)")

        if "eps_forecast" in data and data["eps_forecast"]:
            eps = data["eps_forecast"]
            print(f"\nEPS:")
            print(f"  历史 EPS: {eps.get('trailing_eps', 'N/A')}")
            print(f"  预期 EPS: {eps.get('forward_eps', 'N/A')}")

        if "data_quality" in data:
            dq = data["data_quality"]
            print(f"\n数据质量: {dq.get('rating', 'N/A')} ({dq.get('data_completeness', 0)}% 完整)")

        print()


if __name__ == "__main__":
    main()
