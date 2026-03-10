#!/usr/bin/env python3
"""
港交所披露易招股书自动抓取工具

使用 Playwright 自动化浏览器，从披露易网站搜索并下载招股书PDF

依赖安装:
    pip install playwright requests
    playwright install chromium

用法:
    python fetch_prospectus.py <股票代码或公司名>       # 自动识别，搜索并下载
    python fetch_prospectus.py <公司名> --search       # 仅搜索，显示PDF链接
    python fetch_prospectus.py <公司名> --ticker       # 仅查询股票代码
    python fetch_prospectus.py <股票代码> --visible    # 显示浏览器窗口（调试）
    python fetch_prospectus.py --url <PDF_URL>         # 直接下载指定URL

示例:
    python fetch_prospectus.py 02513
    python fetch_prospectus.py 智谱              # 自动查询ticker
    python fetch_prospectus.py "Zhipu AI"        # 英文名也支持
    python fetch_prospectus.py 智谱 --ticker     # 只查ticker不下载
"""

import json
import os
import re
import sys
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, quote

# ============================================================
# 配置
# ============================================================

TITLE_SEARCH_URL = "https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=zh"
BASE_URL = "https://www1.hkexnews.hk"
DOWNLOAD_DIR = os.path.expanduser("~/Downloads/hk_ipo_prospectus")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
}

# IPO相关关键词（用于识别招股书相关文件）
PROSPECTUS_KEYWORDS = [
    "招股", "Prospectus", "Global Offering", "全球發售",
    "聆訊", "配發結果", "發售價", "上市文件",
    "申請版本", "Post Hearing"
]

# 港交所股票搜索API (备用)
HKEX_STOCK_SEARCH_API = "https://www1.hkexnews.hk/ncms/json/eds/search_stockcode_c.json"
YAHOO_FINANCE_SEARCH_API = "https://query1.finance.yahoo.com/v1/finance/search"


# ============================================================
# Ticker查询
# ============================================================

def is_stock_code(query: str) -> bool:
    """判断输入是否为股票代码（纯数字或数字.HK格式）"""
    clean = query.replace(".HK", "").replace(".hk", "").strip()
    return clean.isdigit()


def lookup_ticker_yahoo(query: str, limit: int = 5) -> List[Dict]:
    """
    使用Yahoo Finance API查询港股代码 (最可靠)
    注意: Yahoo Finance只支持英文/拼音查询

    返回:
        [{"code": "03690", "name_cn": "", "name_en": "MEITUAN", "source": "yahoo"}, ...]
    """
    results = []

    # 检测是否包含中文，如果包含中文则跳过Yahoo（它不支持中文）
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
    if has_chinese:
        print(f"Yahoo Finance不支持中文查询，跳过...", file=sys.stderr)
        return []

    try:
        print(f"Yahoo Finance查询: {query}...", file=sys.stderr)
        params = {
            "q": query,
            "quotesCount": 10,
            "newsCount": 0
        }
        resp = requests.get(
            YAHOO_FINANCE_SEARCH_API,
            params=params,
            headers=HEADERS,
            timeout=10
        )
        resp.raise_for_status()

        data = resp.json()
        quotes = data.get("quotes", [])

        for quote in quotes:
            symbol = quote.get("symbol", "")
            # 只保留港股 (.HK后缀)
            if ".HK" in symbol:
                code = symbol.replace(".HK", "").zfill(5)
                name_en = quote.get("shortname", "") or quote.get("longname", "")
                results.append({
                    "code": code,
                    "name_cn": "",
                    "name_en": name_en,
                    "source": "yahoo"
                })

                if len(results) >= limit:
                    break

    except Exception as e:
        print(f"Yahoo Finance查询失败: {e}", file=sys.stderr)

    return results


def lookup_ticker_duckduckgo(query: str) -> List[Dict]:
    """
    使用搜索引擎查询港股代码
    优先使用 Brave Search API（需要设置 BRAVE_API_KEY 环境变量）
    失败则 fallback 到 HTML 爬取

    返回:
        [{"code": "02513", "name_cn": "智譜", "name_en": "", "source": "search"}, ...]
    """
    import subprocess
    from collections import Counter

    all_codes = []

    # 方法1: 使用 Brave Search API（最可靠）
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if brave_api_key:
        try:
            search_query = f"{query} 港股 ticker"
            print(f"Brave API搜索: {search_query}...", file=sys.stderr)

            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": search_query, "count": 20},
                headers={
                    "X-Subscription-Token": brave_api_key,
                    "Accept": "application/json"
                },
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()

            # 从搜索结果中提取股票代码
            results_text = ""
            for result in data.get("web", {}).get("results", []):
                results_text += result.get("title", "") + " "
                results_text += result.get("description", "") + " "

            # 提取股票代码
            patterns = [
                r'[Hh][Kk][:\s]?(\d{4,5})',  # HK2513
                r'(\d{4,5})\.HK',  # 02513.HK
            ]
            for pattern in patterns:
                matches = re.findall(pattern, results_text, re.IGNORECASE)
                for match in matches:
                    code = match.zfill(5)
                    if 100 <= int(code) <= 99999:
                        if code not in ['00000', '12345', '10000', '20000', '30000']:
                            all_codes.append(code)

            if all_codes:
                print(f"Brave API找到 {len(set(all_codes))} 个股票代码", file=sys.stderr)

        except Exception as e:
            print(f"Brave API失败: {e}，使用备用方法...", file=sys.stderr)

    # 方法2: Fallback 到 HTML 爬取
    if not all_codes:
        search_queries = [
            f"{query} 港股 ticker",
            f"{query} hkex stock code",
        ]

        for search_query in search_queries:
            # 尝试 Brave Search HTML
            try:
                print(f"搜索: {search_query}...", file=sys.stderr)
                cmd = [
                    "curl", "-s",
                    "-A", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                    f"https://search.brave.com/search?q={quote(search_query)}"
                ]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
                html = result.stdout

                if "captcha" not in html.lower() and len(html) > 1000:
                    patterns = [
                        r'[Hh][Kk][:\s]?(\d{4,5})',
                        r'(\d{4,5})\.HK',
                    ]
                    for pattern in patterns:
                        matches = re.findall(pattern, html, re.IGNORECASE)
                        for match in matches:
                            code = match.zfill(5)
                            if 100 <= int(code) <= 99999:
                                if code not in ['00000', '12345', '10000', '20000', '30000']:
                                    all_codes.append(code)

            except Exception as e:
                print(f"搜索失败: {e}", file=sys.stderr)

    if not all_codes:
        return []

    # 按出现频率排序
    code_counts = Counter(all_codes)
    results = []

    for code, count in code_counts.most_common(5):
        results.append({
            "code": code,
            "name_cn": query,
            "name_en": "",
            "source": "search",
            "confidence": count
        })

    return results


def lookup_ticker_playwright(query: str, limit: int = 10, headless: bool = True) -> List[Dict]:
    """
    使用Playwright获取披露易autocomplete结果

    参数:
        query: 公司名称或部分名称
        limit: 返回结果数量限制
        headless: 是否无头模式

    返回:
        [{"code": "02513", "name_cn": "智譜人工智能", "name_en": "ZHIPU AI"}, ...]
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("错误: 需要安装 playwright", file=sys.stderr)
        return []

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="zh-CN"
        )
        page = context.new_page()

        try:
            print(f"Playwright查询: {query}...", file=sys.stderr)
            page.goto(TITLE_SEARCH_URL, wait_until="networkidle", timeout=30000)
            page.wait_for_selector("#searchStockCode", timeout=10000)

            # 关闭Cookie弹窗
            try:
                accept_btn = page.locator("button:has-text('接受'), button:has-text('Accept')")
                if accept_btn.count() > 0 and accept_btn.first.is_visible():
                    accept_btn.first.click()
                    page.wait_for_timeout(500)
            except:
                pass

            # 输入查询词
            stock_input = page.locator("#searchStockCode")
            stock_input.fill(query)
            page.wait_for_timeout(2000)  # 等待autocomplete

            # 获取autocomplete结果
            autocomplete_rows = page.locator("table.stockcode-list tr, .autocomplete-suggestions tr, table tr").all()

            for row in autocomplete_rows[:limit * 2]:
                try:
                    text = row.inner_text().strip()
                    if not text or len(text) < 3:
                        continue

                    parts = text.split()
                    if len(parts) >= 2:
                        code = ""
                        name_cn = ""
                        name_en = ""

                        for part in parts:
                            clean_part = part.strip()
                            if clean_part.isdigit() and len(clean_part) <= 5:
                                code = clean_part.zfill(5)
                                break

                        if code:
                            remaining = text.replace(code.lstrip('0'), '').strip()
                            cn_chars = []
                            en_chars = []
                            for char in remaining:
                                if '\u4e00' <= char <= '\u9fff' or char in '－-()（）':
                                    cn_chars.append(char)
                                elif char.isalpha() or char.isspace():
                                    en_chars.append(char)

                            name_cn = ''.join(cn_chars).strip()
                            name_en = ''.join(en_chars).strip()

                            if code and (name_cn or name_en):
                                results.append({
                                    "code": code,
                                    "name_cn": name_cn,
                                    "name_en": name_en
                                })

                                if len(results) >= limit:
                                    break
                except:
                    continue

        except Exception as e:
            print(f"Playwright查询失败: {e}", file=sys.stderr)
        finally:
            browser.close()

    return results


def lookup_ticker(query: str, limit: int = 10) -> List[Dict]:
    """
    查询股票代码

    优先级:
    1. Yahoo Finance API (最可靠，数据全)
    2. DuckDuckGo搜索 (备用，适合新上市股票)
    3. Playwright获取autocomplete (最后备用)

    返回:
        [{"code": "02513", "name_cn": "智譜人工智能", "name_en": "ZHIPU AI"}, ...]
    """
    # 方法1: Yahoo Finance (最可靠)
    results = lookup_ticker_yahoo(query, limit)
    if results:
        print(f"找到股票代码: {results[0]['code']} ({results[0].get('name_en', '')})", file=sys.stderr)
        return results[:limit]

    # 方法2: DuckDuckGo搜索 (适合新上市股票，Yahoo可能还没收录)
    print("Yahoo无结果，尝试搜索引擎...", file=sys.stderr)
    results = lookup_ticker_duckduckgo(query)
    if results:
        print(f"找到股票代码: {results[0]['code']}", file=sys.stderr)
        return results[:limit]

    # 方法3: Playwright获取autocomplete (最后备用)
    print("搜索无结果，尝试Playwright...", file=sys.stderr)
    return lookup_ticker_playwright(query, limit)


def resolve_ticker(query: str, auto_select: bool = True) -> Tuple[Optional[str], List[Dict]]:
    """
    解析输入，返回股票代码

    参数:
        query: 用户输入（股票代码或公司名称）
        auto_select: 如果只有一个匹配结果或置信度高，自动选择

    返回:
        (stock_code, matches) - stock_code为None表示需要用户确认
    """
    # 如果已经是股票代码，直接返回
    if is_stock_code(query):
        code = query.replace(".HK", "").replace(".hk", "").zfill(5)
        return code, []

    # 查询ticker
    matches = lookup_ticker(query)

    if not matches:
        print(f"未找到匹配 '{query}' 的股票", file=sys.stderr)
        return None, []

    # 自动选择逻辑
    if auto_select:
        # 情况1: 只有一个结果
        if len(matches) == 1:
            selected = matches[0]
            print(f"自动选择: {selected['code']} - {selected['name_cn']}", file=sys.stderr)
            return selected["code"], matches

        # 情况2: 第一个结果置信度明显高于其他 (至少是第二名的2倍)
        if len(matches) >= 2:
            first_conf = matches[0].get("confidence", 1)
            second_conf = matches[1].get("confidence", 1)
            if first_conf >= second_conf * 2:
                selected = matches[0]
                print(f"自动选择(高置信度): {selected['code']} - {selected['name_cn']} (出现{first_conf}次)", file=sys.stderr)
                return selected["code"], matches

    # 多个结果，返回列表供选择
    return None, matches


def print_ticker_matches(matches: List[Dict]):
    """打印匹配的股票列表"""
    print("\n找到以下匹配的股票:", file=sys.stderr)
    print("-" * 60, file=sys.stderr)
    for i, m in enumerate(matches, 1):
        print(f"  {i}. {m['code']} - {m['name_cn']} ({m['name_en']})", file=sys.stderr)
    print("-" * 60, file=sys.stderr)


# ============================================================
# Playwright 自动化搜索
# ============================================================

def search_documents_playwright(
    stock_code: str = None,
    company_name: str = None,
    category: str = None,
    date_from: str = None,
    date_to: str = None,
    headless: bool = True,
    timeout: int = 30000
) -> List[Dict]:
    """
    使用 Playwright 从披露易搜索文件

    参数:
        stock_code: 股票代码 (如 02513)
        company_name: 公司名称 (如 智谱)
        category: 文件类别 (如 "上市文件")
        date_from: 开始日期 (YYYY-MM-DD)
        date_to: 结束日期 (YYYY-MM-DD)
        headless: 是否无头模式
        timeout: 超时时间(毫秒)

    返回:
        [{"title": ..., "url": ..., "file_type": ...}, ...]
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("错误: 需要安装 playwright", file=sys.stderr)
        print("运行: pip install playwright && playwright install chromium", file=sys.stderr)
        return []

    results = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            locale="zh-CN"
        )
        page = context.new_page()

        try:
            print(f"访问披露易搜索页面...", file=sys.stderr)
            page.goto(TITLE_SEARCH_URL, wait_until="networkidle", timeout=timeout)

            # 等待页面加载完成 - 等待搜索按钮出现 (蓝色按钮)
            page.wait_for_selector("a.btn-blue", timeout=timeout)

            # 关闭Cookie弹窗
            try:
                accept_btn = page.locator("button:has-text('接受'), button:has-text('Accept')")
                if accept_btn.count() > 0 and accept_btn.first.is_visible():
                    print("关闭Cookie弹窗...", file=sys.stderr)
                    accept_btn.first.click()
                    page.wait_for_timeout(500)
            except:
                pass

            # 输入股票代码
            if stock_code:
                stock_code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)
                print(f"输入股票代码: {stock_code}", file=sys.stderr)

                # 使用正确的选择器 - #searchStockCode
                stock_input = page.locator("#searchStockCode")
                if stock_input.count() > 0 and stock_input.is_visible():
                    stock_input.fill(stock_code)
                    page.wait_for_timeout(1500)  # 等待autocomplete出现

                    # 查找autocomplete表格行并点击第一个匹配的
                    autocomplete_rows = page.locator("table tr").all()
                    for row in autocomplete_rows:
                        try:
                            text = row.inner_text()
                            if stock_code in text:
                                row.click()
                                page.wait_for_timeout(500)
                                print(f"选择股票: {text.strip()[:30]}", file=sys.stderr)
                                break
                        except:
                            continue
                else:
                    # 备用选择器
                    page.fill("input[placeholder*='股份代號/股份名稱']", stock_code)

            # 输入公司名称
            if company_name:
                print(f"输入公司名称: {company_name}", file=sys.stderr)
                stock_input = page.locator("#searchStockCode")
                if stock_input.count() > 0 and stock_input.is_visible():
                    stock_input.fill(company_name)
                    page.wait_for_timeout(1500)
                    # 点击第一个匹配结果
                    autocomplete_rows = page.locator("table tr").all()
                    for row in autocomplete_rows:
                        try:
                            if company_name in row.inner_text():
                                row.click()
                                break
                        except:
                            continue

            # 选择文件类别（上市文件）- 披露易使用combobox下拉
            if category:
                print(f"选择文件类别: {category}", file=sys.stderr)
                try:
                    # 找到标题类别下拉框并点击
                    dropdowns = page.locator(".combobox-field, .dropdown-toggle")
                    for i in range(dropdowns.count()):
                        dropdown = dropdowns.nth(i)
                        if dropdown.is_visible():
                            dropdown.click()
                            page.wait_for_timeout(300)
                            # 选择上市文件
                            option = page.locator(f"text={category}")
                            if option.count() > 0:
                                option.first.click()
                                page.wait_for_timeout(300)
                                break
                except:
                    pass

            # 点击搜索按钮 (蓝色按钮)
            print("执行搜索...", file=sys.stderr)
            search_btn = page.locator("a.btn-blue, a.filter__btn-applyFilters-js")
            if search_btn.count() > 0:
                search_btn.first.click()
            else:
                page.locator("a:has-text('搜尋')").nth(2).click()

            # 等待结果加载
            page.wait_for_load_state("networkidle", timeout=timeout)
            page.wait_for_timeout(2000)  # 额外等待动态内容

            # 提取PDF链接
            print("提取搜索结果...", file=sys.stderr)

            # 只提取listedco目录下的PDF（公司文件）
            pdf_links = page.locator("a[href*='listedco'][href*='.pdf']").all()
            print(f"找到 {len(pdf_links)} 个公司PDF文件", file=sys.stderr)

            for link in pdf_links[:100]:  # 最多处理100个
                try:
                    href = link.get_attribute("href")
                    title = link.inner_text().strip()

                    if href:
                        full_url = href if href.startswith("http") else urljoin(BASE_URL, href)
                        results.append({
                            "title": title,
                            "url": full_url,
                            "file_type": "PDF"
                        })
                except Exception as e:
                    continue

            # 截图保存（调试用）
            if os.environ.get("DEBUG"):
                screenshot_path = f"/tmp/hkex_search_{stock_code or 'all'}.png"
                page.screenshot(path=screenshot_path, full_page=True)
                print(f"截图保存到: {screenshot_path}", file=sys.stderr)

        except Exception as e:
            print(f"搜索失败: {e}", file=sys.stderr)
            # 保存错误截图
            try:
                page.screenshot(path="/tmp/hkex_error.png")
            except:
                pass

        finally:
            browser.close()

    # 去重
    seen = set()
    unique_results = []
    for r in results:
        url = r.get("url", "")
        if url and url not in seen:
            seen.add(url)
            unique_results.append(r)

    return unique_results


def find_prospectus_pdfs(stock_code: str, headless: bool = True, days: int = 180) -> List[Dict]:
    """
    搜索股票的招股书PDF

    参数:
        stock_code: 股票代码
        headless: 是否无头模式
        days: 搜索最近多少天的文件

    返回:
        [{"title": ..., "url": ...}, ...]
    """
    # 计算日期范围
    date_to = datetime.now().strftime("%Y-%m-%d")
    date_from = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    results = search_documents_playwright(
        stock_code=stock_code,
        category="上市文件",
        date_from=date_from,
        date_to=date_to,
        headless=headless
    )

    return results


# ============================================================
# PDF下载
# ============================================================

def download_pdf(url: str, save_dir: str = None, filename: str = None) -> Optional[str]:
    """下载PDF文件"""
    if not url:
        return None

    if save_dir is None:
        save_dir = DOWNLOAD_DIR

    os.makedirs(save_dir, exist_ok=True)

    if filename is None:
        filename = os.path.basename(url.split("?")[0])
        if not filename.endswith(".pdf"):
            filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    filepath = os.path.join(save_dir, filename)

    try:
        print(f"下载: {url}", file=sys.stderr)
        resp = requests.get(url, headers=HEADERS, timeout=120, stream=True)
        resp.raise_for_status()

        # 检查是否是PDF
        content_type = resp.headers.get("Content-Type", "")
        if "pdf" not in content_type.lower() and not url.endswith(".pdf"):
            print(f"警告: 可能不是PDF文件 (Content-Type: {content_type})", file=sys.stderr)

        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(filepath)
        print(f"保存到: {filepath} ({file_size / 1024 / 1024:.2f} MB)", file=sys.stderr)
        return filepath

    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return None


def download_prospectus(stock_code: str, save_dir: str = None, headless: bool = True, max_files: int = 10) -> List[str]:
    """
    搜索并下载股票的招股书

    优先下载主招股书（全球發售），然后下载其他文件

    参数:
        stock_code: 股票代码
        save_dir: 保存目录
        headless: 是否无头模式
        max_files: 最多下载文件数

    返回:
        下载的文件路径列表
    """
    stock_code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)

    if save_dir is None:
        save_dir = os.path.join(DOWNLOAD_DIR, stock_code)

    print(f"搜索 {stock_code} 的招股书...", file=sys.stderr)
    # 搜索365天范围以确保找到主招股书
    results = find_prospectus_pdfs(stock_code, headless=headless, days=365)

    if not results:
        print(f"未找到 {stock_code} 的招股书", file=sys.stderr)
        return []

    print(f"找到 {len(results)} 个PDF文件", file=sys.stderr)

    # 主招股书关键词（必须包含）
    main_keywords = ['全球發售', 'Global Offering', '招股章程']
    # 排除的关键词（不是主招股书）
    exclude_in_main = ['公告', '協調人']

    # 分类：主招股书 vs 其他文件
    main_prospectus_list = []
    other_files = []

    for item in results:
        title = item.get("title", "")

        # 检查是否是主招股书
        is_main = any(kw in title for kw in main_keywords)
        is_excluded = any(kw in title for kw in exclude_in_main)

        if is_main and not is_excluded:
            main_prospectus_list.append(item)
            print(f"找到主招股书: {title}", file=sys.stderr)
        else:
            other_files.append(item)

    # 重新排序：主招股书优先（取第一个，通常是中文版）
    sorted_results = []
    if main_prospectus_list:
        sorted_results.append(main_prospectus_list[0])  # 只取第一个主招股书
    sorted_results.extend(other_files[:max_files - 1])

    if not main_prospectus_list:
        print("警告: 未找到主招股书（全球發售）", file=sys.stderr)
        sorted_results = results[:max_files]

    downloaded = []
    for i, item in enumerate(sorted_results[:max_files]):
        url = item.get("url")
        title = item.get("title", "")

        # 使用标题作为文件名（清理非法字符）
        if title:
            # 清理文件名
            safe_title = re.sub(r'[\\/:*?"<>|]', '_', title)
            safe_title = safe_title.strip()[:50]  # 限制长度
            filename = f"{safe_title}.pdf"
        else:
            filename = None

        filepath = download_pdf(url, save_dir, filename)
        if filepath:
            downloaded.append(filepath)
        time.sleep(1)  # 避免请求过快

    return downloaded


# ============================================================
# PDF文本提取
# ============================================================

def extract_pdf_text(pdf_path: str, output_path: str = None, max_pages: int = None) -> Optional[str]:
    """
    提取PDF文本内容

    参数:
        pdf_path: PDF文件路径
        output_path: 输出txt文件路径（可选，不指定则返回文本）
        max_pages: 最多提取页数（可选）

    返回:
        提取的文本内容，或输出文件路径
    """
    try:
        import pdfplumber
    except ImportError:
        print("错误: 需要安装 pdfplumber", file=sys.stderr)
        print("运行: pip install pdfplumber", file=sys.stderr)
        return None

    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 {pdf_path}", file=sys.stderr)
        return None

    print(f"提取PDF文本: {pdf_path}", file=sys.stderr)

    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            pages_to_extract = min(total_pages, max_pages) if max_pages else total_pages

            print(f"共 {total_pages} 页，提取 {pages_to_extract} 页...", file=sys.stderr)

            for i, page in enumerate(pdf.pages[:pages_to_extract]):
                text = page.extract_text()
                if text:
                    text_parts.append(f"\n{'='*60}\n第 {i+1} 页\n{'='*60}\n")
                    text_parts.append(text)

                # 进度显示
                if (i + 1) % 50 == 0:
                    print(f"  已提取 {i+1}/{pages_to_extract} 页...", file=sys.stderr)

    except Exception as e:
        print(f"提取失败: {e}", file=sys.stderr)
        return None

    full_text = '\n'.join(text_parts)
    print(f"提取完成，共 {len(full_text)} 字符", file=sys.stderr)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"保存到: {output_path}", file=sys.stderr)
        return output_path

    return full_text


def extract_key_sections(pdf_path: str, output_path: str = None) -> Optional[str]:
    """
    智能提取招股书关键章节

    关键章节:
    - 概要 / Summary
    - 风险因素 / Risk Factors
    - 业务 / Business
    - 财务资料 / Financial Information
    - 所得款项用途 / Use of Proceeds
    - 基石投资者 / Cornerstone Investors

    返回:
        提取的关键章节文本
    """
    try:
        import pdfplumber
    except ImportError:
        print("错误: 需要安装 pdfplumber", file=sys.stderr)
        print("运行: pip install pdfplumber", file=sys.stderr)
        return None

    if not os.path.exists(pdf_path):
        print(f"错误: 文件不存在 {pdf_path}", file=sys.stderr)
        return None

    # 关键章节标题（中英文）- 按重要性排序
    key_sections = [
        ("概要", "SUMMARY", 30),           # 最多提取30页
        ("風險因素", "RISK FACTORS", 20),
        ("業務", "BUSINESS", 40),
        ("財務資料", "FINANCIAL INFORMATION", 30),
        ("所得款項用途", "USE OF PROCEEDS", 10),
        ("基石投資者", "CORNERSTONE INVESTORS", 10),
        ("股本", "SHARE CAPITAL", 15),
        ("行業概覽", "INDUSTRY OVERVIEW", 20),
    ]

    print(f"智能提取关键章节: {pdf_path}", file=sys.stderr)

    # 第一遍：扫描所有页面，记录每个章节首次出现的位置
    section_start_pages = {}  # {章节名: 起始页码}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"共 {total_pages} 页，扫描章节位置...", file=sys.stderr)

            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text:
                    continue

                # 检查前200字符（章节标题通常在页面顶部）
                first_part = text[:200].upper()

                for cn_title, en_title, _ in key_sections:
                    # 只记录首次出现
                    if cn_title in section_start_pages:
                        continue

                    # 更严格的匹配：标题应该在页面开头
                    if cn_title in text[:150] or en_title in first_part:
                        section_start_pages[cn_title] = i
                        print(f"  找到章节: {cn_title} (第{i+1}页)", file=sys.stderr)

            # 第二遍：提取每个章节的内容
            extracted = {}

            for cn_title, en_title, max_pages in key_sections:
                if cn_title not in section_start_pages:
                    continue

                start_page = section_start_pages[cn_title]
                end_page = min(start_page + max_pages, total_pages)

                section_text = [f"\n{'#'*60}\n# {cn_title} / {en_title}\n# 第 {start_page+1} 页 - 第 {end_page} 页\n{'#'*60}\n"]

                for j in range(start_page, end_page):
                    page_text = pdf.pages[j].extract_text()
                    if page_text:
                        section_text.append(page_text)

                extracted[cn_title] = '\n'.join(section_text)

    except Exception as e:
        print(f"提取失败: {e}", file=sys.stderr)
        return None

    if not extracted:
        print("未找到关键章节，尝试提取前100页作为摘要...", file=sys.stderr)
        return extract_pdf_text(pdf_path, output_path, max_pages=100)

    # 组合所有章节（按预定顺序）
    result_parts = [
        "=" * 60,
        "港股招股书关键章节摘要",
        f"来源: {os.path.basename(pdf_path)}",
        f"提取章节数: {len(extracted)}",
        "=" * 60,
        ""
    ]

    for cn_title, _, _ in key_sections:
        if cn_title in extracted:
            result_parts.append(extracted[cn_title])
            result_parts.append("\n")

    full_text = '\n'.join(result_parts)
    print(f"提取完成，共 {len(extracted)} 个章节，{len(full_text)} 字符", file=sys.stderr)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(full_text)
        print(f"保存到: {output_path}", file=sys.stderr)
        return output_path

    return full_text


def find_main_prospectus(pdf_files: List[str]) -> Optional[str]:
    """
    从下载的PDF文件中找到主招股书（全球發售）

    识别规则：
    1. 文件名包含: 全球發售, Global Offering, Prospectus
    2. 文件大小通常 > 5MB（主招股书几百页）
    3. 排除: 公告, 月報表, 章程, 董事名單 等小文件
    """
    if not pdf_files:
        return None

    # 排除的关键词（这些通常是小文件）
    exclude_keywords = ['公告', '月報表', '章程', '董事', '翌日', '變動', '配發結果', '資料集']

    # 主招股书关键词（优先级从高到低）
    main_keywords = ['全球發售', 'Global', 'Prospectus', '招股章程', '招股書']

    candidates = []

    for pdf in pdf_files:
        if not os.path.exists(pdf):
            continue

        filename = os.path.basename(pdf)
        size = os.path.getsize(pdf)

        # 排除小文件和非招股书文件
        is_excluded = any(kw in filename for kw in exclude_keywords)
        if is_excluded:
            continue

        # 检查是否包含主招股书关键词
        priority = 0
        for i, kw in enumerate(main_keywords):
            if kw.lower() in filename.lower():
                priority = len(main_keywords) - i  # 优先级
                break

        candidates.append((pdf, size, priority))

    if not candidates:
        # 如果没有匹配关键词的，返回最大的文件（可能是主招股书）
        valid_files = [(f, os.path.getsize(f)) for f in pdf_files if os.path.exists(f)]
        if valid_files:
            # 选择大于2MB的最大文件
            large_files = [(f, s) for f, s in valid_files if s > 2 * 1024 * 1024]
            if large_files:
                return max(large_files, key=lambda x: x[1])[0]
            return max(valid_files, key=lambda x: x[1])[0]
        return None

    # 按优先级和大小排序（优先级高 > 文件大）
    candidates.sort(key=lambda x: (x[2], x[1]), reverse=True)

    selected = candidates[0][0]
    print(f"识别主招股书: {os.path.basename(selected)} ({candidates[0][1]/1024/1024:.1f}MB)", file=sys.stderr)
    return selected


# ============================================================
# 主函数
# ============================================================

def print_help():
    """打印帮助信息"""
    print("""
港交所招股书自动抓取工具

用法:
  python fetch_prospectus.py <股票代码或公司名>       # 自动识别，搜索并下载
  python fetch_prospectus.py <公司名> --ticker       # 仅查询股票代码
  python fetch_prospectus.py <公司名> --search       # 查ticker后搜索文件
  python fetch_prospectus.py <股票代码> --extract    # 下载并提取完整文本
  python fetch_prospectus.py <股票代码> --summary    # 下载并提取关键章节
  python fetch_prospectus.py <股票代码> --visible    # 显示浏览器窗口（调试）
  python fetch_prospectus.py --url <PDF_URL>         # 直接下载指定URL

示例:
  python fetch_prospectus.py 02513                   # 用股票代码下载
  python fetch_prospectus.py 智谱                    # 自动查询ticker
  python fetch_prospectus.py 智谱 --extract          # 下载并提取完整文本
  python fetch_prospectus.py 智谱 --summary          # 下载并提取关键章节
  python fetch_prospectus.py --url "https://www1.hkexnews.hk/xxx.pdf"

下载位置: ~/Downloads/hk_ipo_prospectus/<股票代码>/
提取文本: ~/Downloads/hk_ipo_prospectus/<股票代码>/<股票代码>_full.txt
关键章节: ~/Downloads/hk_ipo_prospectus/<股票代码>/<股票代码>_summary.txt

依赖安装:
  pip install playwright requests pdfplumber
  playwright install chromium
""")


def main():
    if len(sys.argv) < 2:
        print_help()
        sys.exit(1)

    arg = sys.argv[1]

    # 帮助
    if arg in ["-h", "--help"]:
        print_help()
        return

    headless = "--visible" not in sys.argv
    ticker_only = "--ticker" in sys.argv
    search_only = "--search" in sys.argv
    extract_full = "--extract" in sys.argv
    extract_summary = "--summary" in sys.argv

    # 直接URL下载模式
    if arg == "--url":
        if len(sys.argv) < 3:
            print("错误: 请提供PDF URL", file=sys.stderr)
            sys.exit(1)
        url = sys.argv[2]
        filepath = download_pdf(url)
        if filepath:
            print(json.dumps({"url": url, "downloaded": filepath}, ensure_ascii=False, indent=2))
        return

    query = arg

    # ========================================
    # 步骤1: 解析/查询股票代码
    # ========================================
    stock_code, matches = resolve_ticker(query, auto_select=True)

    # 如果有多个匹配，显示列表
    if not stock_code and matches:
        print_ticker_matches(matches)
        # 输出JSON供程序使用
        print(json.dumps({
            "query": query,
            "matches": matches,
            "message": "多个匹配结果，请指定更精确的名称或直接使用股票代码"
        }, ensure_ascii=False, indent=2))
        return

    # 未找到匹配
    if not stock_code:
        print(json.dumps({
            "query": query,
            "error": "未找到匹配的股票",
            "suggestion": "请检查公司名称或直接使用股票代码"
        }, ensure_ascii=False, indent=2))
        sys.exit(1)

    # ========================================
    # 步骤2: 仅查询ticker模式
    # ========================================
    if ticker_only:
        # 重新查询以获取完整信息
        if is_stock_code(query):
            matches = lookup_ticker(stock_code)
        output = {
            "query": query,
            "stock_code": stock_code,
        }
        if matches:
            output["name_cn"] = matches[0].get("name_cn", "")
            output["name_en"] = matches[0].get("name_en", "")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # ========================================
    # 步骤3: 搜索文件
    # ========================================
    if search_only:
        results = find_prospectus_pdfs(stock_code, headless=headless)
        pdf_urls = [r.get("url") for r in results]
        output = {
            "query": query,
            "stock_code": stock_code,
            "count": len(results),
            "files": results,
            "pdf_urls": pdf_urls
        }
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return

    # ========================================
    # 步骤4: 默认模式 - 搜索并下载
    # ========================================
    downloaded = download_prospectus(stock_code, headless=headless)

    output = {
        "query": query,
        "stock_code": stock_code,
        "downloaded": downloaded
    }

    # ========================================
    # 步骤5: 提取文本（如果指定）
    # ========================================
    if (extract_full or extract_summary) and downloaded:
        # 找到主招股书
        main_pdf = find_main_prospectus(downloaded)

        if main_pdf:
            save_dir = os.path.dirname(main_pdf)

            if extract_summary:
                # 提取关键章节
                summary_path = os.path.join(save_dir, f"{stock_code}_summary.txt")
                result = extract_key_sections(main_pdf, summary_path)
                if result:
                    output["summary_txt"] = summary_path
                    output["summary_size"] = os.path.getsize(summary_path)

            elif extract_full:
                # 提取完整文本
                full_path = os.path.join(save_dir, f"{stock_code}_full.txt")
                result = extract_pdf_text(main_pdf, full_path)
                if result:
                    output["full_txt"] = full_path
                    output["full_size"] = os.path.getsize(full_path)

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
