#!/usr/bin/env python3
"""
港股年报/中期报告下载模块
从港交所披露易 (hkexnews.hk) 搜索并下载年报 PDF
"""

import requests
import json
import sys
import os
import re
from datetime import datetime
from typing import Optional, List, Dict

# 尝试导入 PDF 处理库
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class HKAnnualReportDownloader:
    """港股年报下载器"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        })
        # 披露易搜索 API
        self.search_url = "https://www1.hkexnews.hk/search/titlesearch.xhtml"

    def search_reports(
        self,
        stock_code: str,
        report_type: str = "annual",
        limit: int = 5
    ) -> List[Dict]:
        """
        搜索港股财报

        Args:
            stock_code: 股票代码（如 "00700" 或 "700"）
            report_type: 报告类型 - "annual"(年报) / "interim"(中期) / "quarterly"(季报)
            limit: 返回结果数量

        Returns:
            list: 报告列表，每项包含 title, date, url, doc_type
        """
        # 标准化股票代码为5位
        code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)

        # 文档类型关键词映射
        type_keywords = {
            "annual": ["Annual Report", "年報", "年度報告"],
            "interim": ["Interim Report", "中期報告", "中期業績"],
            "quarterly": ["Quarterly Report", "季報", "季度報告"],
        }

        keywords = type_keywords.get(report_type, type_keywords["annual"])
        results = []

        # 尝试直接获取公告列表
        try:
            # 使用 HKEX 公告搜索 API
            search_api_url = "https://www1.hkexnews.hk/search/titlesearch.xhtml"

            # 构建 POST 请求
            for keyword in keywords[:1]:  # 只用第一个关键词
                form_data = {
                    "lang": "ZH",
                    "category": "0",
                    "market": "SEHK",
                    "stockId": code,
                    "documentType": "-1",
                    "fromDate": "",
                    "toDate": "",
                    "title": keyword,
                    "sortDir": "desc",
                    "sortByDate": "desc",
                    "rowRange": "0-20",
                }

                resp = self.session.post(
                    search_api_url,
                    data=form_data,
                    timeout=15,
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )

                if resp.status_code == 200:
                    parsed = self._parse_search_results(resp.text, code)
                    results.extend(parsed)

        except Exception as e:
            print(f"搜索出错: {e}", file=sys.stderr)

        # 如果 API 没有结果，提供搜索指导
        if not results:
            results = self._generate_search_guidance(code, report_type)

        # 去重并按日期排序
        seen_urls = set()
        unique_results = []
        for r in results:
            url_key = r.get("url", r.get("search_url", ""))
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique_results.append(r)

        unique_results.sort(key=lambda x: x.get("date", ""), reverse=True)
        return unique_results[:limit]

    def _generate_search_guidance(self, code: str, report_type: str) -> List[Dict]:
        """生成搜索指导（当 API 无法获取数据时）"""
        type_names = {
            "annual": "年報",
            "interim": "中期報告",
            "quarterly": "季報",
        }
        type_name = type_names.get(report_type, "年報")

        return [{
            "type": "search_guidance",
            "stock_code": code,
            "doc_type": report_type,
            "search_url": f"https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=ZH&stockId={code}",
            "manual_search_steps": [
                f"1. 访问港交所披露易: https://www.hkexnews.hk",
                f"2. 在搜索框输入股票代码: {code}",
                f"3. 文档类型选择: {type_name}",
                f"4. 点击搜索查看结果",
            ],
            "web_search_keywords": [
                f'"{code}" "{type_name}" site:hkexnews.hk',
                f'"{code}" "Annual Report" site:hkexnews.hk' if report_type == "annual" else None,
            ],
        }]

    def _parse_search_results(self, html: str, stock_code: str) -> List[Dict]:
        """解析披露易搜索结果 HTML"""
        results = []

        # 简单的正则解析（避免依赖 BeautifulSoup）
        # 匹配 PDF 链接和标题
        pattern = r'href="([^"]*\.pdf)"[^>]*>([^<]+)</a>'
        matches = re.findall(pattern, html, re.IGNORECASE)

        # 匹配日期
        date_pattern = r'(\d{4}/\d{2}/\d{2})'
        dates = re.findall(date_pattern, html)

        for i, (url, title) in enumerate(matches):
            # 过滤只保留年报相关文档
            if not any(k in title for k in ["Annual", "年報", "年度", "Interim", "中期", "Quarterly", "季"]):
                continue

            # 构建完整 URL
            if not url.startswith("http"):
                url = "https://www1.hkexnews.hk" + url

            result = {
                "title": title.strip(),
                "url": url,
                "stock_code": stock_code,
                "date": dates[i] if i < len(dates) else "",
            }

            # 判断文档类型
            if any(k in title for k in ["Annual", "年報", "年度報告"]):
                result["doc_type"] = "annual"
            elif any(k in title for k in ["Interim", "中期"]):
                result["doc_type"] = "interim"
            elif any(k in title for k in ["Quarterly", "季"]):
                result["doc_type"] = "quarterly"
            else:
                result["doc_type"] = "other"

            results.append(result)

        return results

    def download_report(
        self,
        url: str,
        save_path: Optional[str] = None
    ) -> Dict:
        """
        下载年报 PDF

        Args:
            url: PDF 链接
            save_path: 保存路径（可选）

        Returns:
            dict: 包含 success, path, size 等信息
        """
        try:
            resp = self.session.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            # 自动生成文件名
            if not save_path:
                filename = url.split("/")[-1]
                save_path = os.path.join(os.getcwd(), filename)

            # 保存文件
            total_size = 0
            with open(save_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        total_size += len(chunk)

            return {
                "success": True,
                "path": save_path,
                "size_mb": round(total_size / (1024 * 1024), 2),
                "url": url,
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        max_pages: int = 50
    ) -> Dict:
        """
        从 PDF 提取文本

        Args:
            pdf_path: PDF 文件路径
            pages: 指定页码列表（从0开始），None 表示全部
            max_pages: 最大处理页数

        Returns:
            dict: 包含 text, pages, error 等信息
        """
        if not PDF_SUPPORT:
            return {
                "success": False,
                "error": "pdfplumber 未安装，请运行: pip install pdfplumber",
            }

        try:
            text_parts = []
            total_pages = 0

            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)

                if pages:
                    target_pages = [p for p in pages if p < total_pages]
                else:
                    target_pages = range(min(total_pages, max_pages))

                for page_num in target_pages:
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"--- 第 {page_num + 1} 页 ---\n{text}")

            return {
                "success": True,
                "text": "\n\n".join(text_parts),
                "total_pages": total_pages,
                "extracted_pages": len(target_pages),
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def extract_financial_summary(self, text: str) -> Dict:
        """
        从年报文本中提取财务摘要

        Args:
            text: 年报文本

        Returns:
            dict: 提取的财务数据
        """
        result = {
            "revenue": None,
            "net_profit": None,
            "gross_profit_margin": None,
            "eps": None,
            "total_assets": None,
            "total_equity": None,
        }

        # 收入匹配模式（支持多种格式）
        revenue_patterns = [
            r"收入[：:]\s*([\d,\.]+)\s*(?:百万|千萬|億)",
            r"Revenue[：:]\s*(?:HK\$|RMB)?\s*([\d,\.]+)\s*(?:million|billion)?",
            r"營業額[：:]\s*([\d,\.]+)",
        ]

        # 净利润匹配
        profit_patterns = [
            r"(?:股東應佔|歸屬於)?淨利潤[：:]\s*([\d,\.]+)",
            r"(?:Net )?Profit[：:]\s*(?:HK\$|RMB)?\s*([\d,\.]+)",
            r"純利[：:]\s*([\d,\.]+)",
        ]

        # 每股收益
        eps_patterns = [
            r"每股(?:基本)?(?:盈利|收益)[：:]\s*([\d,\.]+)",
            r"(?:Basic )?EPS[：:]\s*(?:HK\$|RMB)?\s*([\d,\.]+)",
        ]

        for pattern in revenue_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["revenue"] = match.group(1).replace(",", "")
                break

        for pattern in profit_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["net_profit"] = match.group(1).replace(",", "")
                break

        for pattern in eps_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["eps"] = match.group(1).replace(",", "")
                break

        return result


def search_hkex_reports(stock_code: str, report_type: str = "annual") -> List[Dict]:
    """
    搜索港交所年报的便捷函数

    Args:
        stock_code: 股票代码
        report_type: annual/interim/quarterly

    Returns:
        list: 报告列表
    """
    downloader = HKAnnualReportDownloader()
    return downloader.search_reports(stock_code, report_type)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="港股年报下载工具")
    parser.add_argument("stock_code", help="股票代码（如 00700 或 700）")
    parser.add_argument(
        "--type",
        choices=["annual", "interim", "quarterly"],
        default="annual",
        help="报告类型（默认: annual）"
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="下载最新的报告"
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="提取 PDF 文本（需要已下载的文件）"
    )
    parser.add_argument(
        "--file",
        help="指定 PDF 文件路径（用于 --extract）"
    )

    args = parser.parse_args()

    downloader = HKAnnualReportDownloader()

    # 搜索报告
    print(f"\n搜索 {args.stock_code} 的 {args.type} 报告...", file=sys.stderr)
    reports = downloader.search_reports(args.stock_code, args.type)

    if not reports:
        print(json.dumps({"error": "未找到相关报告"}, ensure_ascii=False, indent=2))
        sys.exit(1)

    print(f"\n找到 {len(reports)} 份报告:", file=sys.stderr)
    for i, r in enumerate(reports, 1):
        if r.get("type") == "search_guidance":
            print(f"  {i}. [搜索指引] 请访问: {r.get('search_url', 'N/A')}", file=sys.stderr)
        else:
            print(f"  {i}. [{r.get('date', 'N/A')}] {r.get('title', 'N/A')}", file=sys.stderr)

    # 下载报告
    if args.download and reports:
        # 检查是否有可下载的报告（非搜索指引）
        downloadable = [r for r in reports if r.get("type") != "search_guidance" and r.get("url")]
        if downloadable:
            print(f"\n下载最新报告...", file=sys.stderr)
            result = downloader.download_report(downloadable[0]["url"])
            if result["success"]:
                print(f"✓ 已下载: {result['path']} ({result['size_mb']} MB)", file=sys.stderr)
            else:
                print(f"✗ 下载失败: {result['error']}", file=sys.stderr)
        else:
            print(f"\n无法自动下载，请手动访问搜索页面", file=sys.stderr)

    # 提取文本
    if args.extract:
        pdf_path = args.file if args.file else None
        if not pdf_path and args.download and reports:
            pdf_path = reports[0]["url"].split("/")[-1]

        if pdf_path and os.path.exists(pdf_path):
            print(f"\n提取 PDF 文本...", file=sys.stderr)
            result = downloader.extract_text_from_pdf(pdf_path, max_pages=20)
            if result["success"]:
                print(f"✓ 已提取 {result['extracted_pages']}/{result['total_pages']} 页", file=sys.stderr)
                print("\n" + "="*50 + "\n")
                print(result["text"][:5000])  # 只打印前5000字符
            else:
                print(f"✗ 提取失败: {result['error']}", file=sys.stderr)

    # 输出 JSON 结果
    output = {
        "stock_code": args.stock_code.zfill(5),
        "report_type": args.type,
        "reports": reports,
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
