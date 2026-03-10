#!/usr/bin/env python3
"""
HK IPO Prospectus Downloader - 港交所招股书下载工具

功能：
1. 从披露易搜索IPO招股书
2. 下载招股书PDF文件
3. 提取关键IPO数据（股本结构、发行价、基石投资者等）

数据源：
- 披露易 (HKEXnews): https://www.hkexnews.hk
- 搜索API: titleSearchServlet.do
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

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Referer": "https://www.hkexnews.hk/",
}

BASE_URL = "https://www1.hkexnews.hk"
SEARCH_API = f"{BASE_URL}/search/titleSearchServlet.do"
STOCK_SEARCH_API = f"{BASE_URL}/search/prefix.do"

# 文件类别代码
CATEGORY_CODES = {
    "listing_docs": "30000",      # 上市文件
    "announcements": "10000",     # 公告及通告
    "circulars": "20000",         # 通函
    "financial": "40000",         # 财务报表
    "application": "91000",       # 申请版本
}

# 下载目录
DOWNLOAD_DIR = os.path.expanduser("~/Downloads/hk_ipo_prospectus")


# ============================================================
# 股票搜索
# ============================================================

def search_stock(keyword: str) -> List[Dict]:
    """
    搜索股票代码/名称，获取stockId

    参数:
        keyword: 股票代码（如 02513）或公司名称（如 智谱）

    返回:
        [{"stockId": xxx, "code": "02513", "name": "智譜"}, ...]
    """
    url = STOCK_SEARCH_API
    params = {
        "callback": "cb",
        "lang": "ZH",
        "type": "A",
        "name": keyword,
        "market": "SEHK"
    }

    try:
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        # 解析JSONP响应: cb({"stockInfo": [...]})
        content = resp.text
        json_str = content[content.find("(") + 1:content.rfind(")")]
        data = json.loads(json_str)

        stocks = data.get("stockInfo", [])
        return stocks
    except Exception as e:
        print(f"搜索股票失败: {e}", file=sys.stderr)
        return []


def get_stock_id(code: str) -> Optional[int]:
    """获取股票的stockId"""
    code = code.replace(".HK", "").replace(".hk", "").zfill(5)
    stocks = search_stock(code)

    for stock in stocks:
        if stock.get("code") == code:
            return stock.get("stockId")

    return None


# ============================================================
# 文档搜索
# ============================================================

def search_documents(
    stock_id: Optional[int] = None,
    stock_code: Optional[str] = None,
    title: Optional[str] = None,
    category: Optional[str] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    row_range: int = 100,
    lang: str = "ZH"
) -> Dict:
    """
    搜索披露易文档

    参数:
        stock_id: 股票ID（从search_stock获取）
        stock_code: 股票代码（备用搜索方式）
        title: 标题关键词
        category: 文件类别代码（见CATEGORY_CODES）
        from_date: 开始日期 (YYYY-MM-DD)
        to_date: 结束日期 (YYYY-MM-DD)
        row_range: 返回记录数
        lang: 语言 (ZH/EN)

    返回:
        {"recordCnt": n, "results": [...]}
    """
    params = {
        "sortDir": "0",
        "sortByOptions": "DateTime",
        "market": "SEHK",
        "searchType": "0",
        "rowRange": str(row_range),
        "lang": lang
    }

    if stock_id:
        params["stockId"] = str(stock_id)

    if title:
        params["title"] = title

    if category:
        params["t1code"] = category

    if from_date:
        params["fromDate"] = from_date.replace("-", "/")

    if to_date:
        params["toDate"] = to_date.replace("-", "/")

    try:
        resp = requests.get(SEARCH_API, params=params, headers=HEADERS, timeout=30)
        data = resp.json()

        result_str = data.get("result", "null")
        if result_str and result_str != "null":
            results = json.loads(result_str)
        else:
            results = []

        return {
            "recordCnt": data.get("recordCnt", 0),
            "hasNextRow": data.get("hasNextRow", False),
            "results": results
        }
    except Exception as e:
        print(f"搜索文档失败: {e}", file=sys.stderr)
        return {"recordCnt": 0, "results": []}


def search_prospectus(stock_code: str, months_back: int = 6) -> List[Dict]:
    """
    搜索股票的招股书/上市文件

    参数:
        stock_code: 股票代码
        months_back: 向前搜索的月数

    返回:
        招股书文档列表
    """
    stock_code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)

    # 获取stockId
    stock_id = get_stock_id(stock_code)

    # 计算日期范围
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")

    results = []

    # 1. 搜索上市文件类别
    print(f"搜索 {stock_code} 的上市文件...", file=sys.stderr)
    docs = search_documents(
        stock_id=stock_id,
        category=CATEGORY_CODES["listing_docs"],
        from_date=from_date,
        to_date=to_date
    )
    results.extend(docs.get("results", []))

    # 2. 搜索申请版本类别
    print(f"搜索 {stock_code} 的申请版本...", file=sys.stderr)
    apps = search_documents(
        stock_id=stock_id,
        category=CATEGORY_CODES["application"],
        from_date=from_date,
        to_date=to_date
    )
    results.extend(apps.get("results", []))

    # 3. 用标题关键词搜索（招股章程、Prospectus）
    keywords = ["招股章程", "招股書", "Prospectus", "Global Offering", "全球發售"]
    for kw in keywords:
        print(f"搜索关键词: {kw}...", file=sys.stderr)
        kw_docs = search_documents(
            stock_id=stock_id,
            title=kw,
            from_date=from_date,
            to_date=to_date
        )
        results.extend(kw_docs.get("results", []))

    # 去重
    seen = set()
    unique_results = []
    for doc in results:
        file_link = doc.get("FILE_LINK", "")
        if file_link and file_link not in seen:
            seen.add(file_link)
            unique_results.append(doc)

    # 按日期排序
    unique_results.sort(key=lambda x: x.get("DATE_TIME", ""), reverse=True)

    return unique_results


def search_recent_ipos(days_back: int = 90) -> List[Dict]:
    """
    搜索近期IPO的招股书

    参数:
        days_back: 向前搜索的天数

    返回:
        招股书文档列表
    """
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    results = []

    # 搜索包含招股章程关键词的文档
    keywords = ["招股章程", "Prospectus", "Global Offering"]

    for kw in keywords:
        print(f"搜索近期 {kw}...", file=sys.stderr)
        docs = search_documents(
            title=kw,
            from_date=from_date,
            to_date=to_date,
            row_range=200
        )
        results.extend(docs.get("results", []))

    # 去重
    seen = set()
    unique_results = []
    for doc in results:
        file_link = doc.get("FILE_LINK", "")
        if file_link and file_link not in seen:
            seen.add(file_link)
            unique_results.append(doc)

    return unique_results


# ============================================================
# PDF下载
# ============================================================

def download_pdf(file_link: str, save_dir: str = None, filename: str = None) -> Optional[str]:
    """
    下载PDF文件

    参数:
        file_link: PDF链接（相对或绝对路径）
        save_dir: 保存目录
        filename: 文件名（可选，默认从URL提取）

    返回:
        保存的文件路径，失败返回None
    """
    if not file_link:
        return None

    # 构建完整URL
    if file_link.startswith("/"):
        url = urljoin(BASE_URL, file_link)
    elif not file_link.startswith("http"):
        url = f"{BASE_URL}/{file_link}"
    else:
        url = file_link

    # 确定保存目录
    if save_dir is None:
        save_dir = DOWNLOAD_DIR

    os.makedirs(save_dir, exist_ok=True)

    # 确定文件名
    if filename is None:
        filename = os.path.basename(file_link)

    filepath = os.path.join(save_dir, filename)

    try:
        print(f"下载: {url}", file=sys.stderr)
        resp = requests.get(url, headers=HEADERS, timeout=60, stream=True)
        resp.raise_for_status()

        with open(filepath, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"保存到: {filepath}", file=sys.stderr)
        return filepath

    except Exception as e:
        print(f"下载失败: {e}", file=sys.stderr)
        return None


def download_prospectus(stock_code: str, save_dir: str = None) -> List[str]:
    """
    下载股票的所有招股书文档

    参数:
        stock_code: 股票代码
        save_dir: 保存目录

    返回:
        下载的文件路径列表
    """
    stock_code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)

    # 搜索招股书
    docs = search_prospectus(stock_code)

    if not docs:
        print(f"未找到 {stock_code} 的招股书", file=sys.stderr)
        return []

    print(f"找到 {len(docs)} 个文档", file=sys.stderr)

    # 设置保存目录
    if save_dir is None:
        save_dir = os.path.join(DOWNLOAD_DIR, stock_code)

    downloaded = []

    for doc in docs:
        file_link = doc.get("FILE_LINK", "")
        file_type = doc.get("FILE_TYPE", "")

        # 只下载PDF
        if file_type.upper() != "PDF":
            continue

        # 构建文件名
        date_str = doc.get("DATE_TIME", "").replace("/", "").replace(":", "").replace(" ", "_")
        title = doc.get("TITLE", "document")[:50].replace("/", "_").replace(" ", "_")
        filename = f"{date_str}_{title}.pdf"

        filepath = download_pdf(file_link, save_dir, filename)
        if filepath:
            downloaded.append(filepath)

        time.sleep(0.5)  # 避免请求过快

    return downloaded


# ============================================================
# 直接URL构建（备用方案）
# ============================================================

def build_prospectus_urls(stock_code: str, ipo_date: str) -> List[str]:
    """
    根据股票代码和IPO日期构建可能的招股书URL

    URL格式：
    https://www1.hkexnews.hk/listedco/listconews/sehk/YYYY/MMDD/YYYYMMDDXXXXX.pdf

    参数:
        stock_code: 股票代码
        ipo_date: IPO日期 (YYYY-MM-DD)

    返回:
        可能的URL列表
    """
    stock_code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)

    # 解析日期
    try:
        dt = datetime.strptime(ipo_date, "%Y-%m-%d")
    except:
        return []

    urls = []

    # 招股书通常在IPO前1-14天发布
    for days_before in range(0, 15):
        check_date = dt - timedelta(days=days_before)
        year = check_date.strftime("%Y")
        mmdd = check_date.strftime("%m%d")
        date_str = check_date.strftime("%Y%m%d")

        # 构建URL（尝试不同的文件编号）
        base_path = f"/listedco/listconews/sehk/{year}/{mmdd}/"
        urls.append(f"{BASE_URL}{base_path}")

    return urls


def try_direct_download(stock_code: str, ipo_date: str, save_dir: str = None) -> List[str]:
    """
    尝试直接从可能的URL下载招股书

    参数:
        stock_code: 股票代码
        ipo_date: IPO日期 (YYYY-MM-DD)
        save_dir: 保存目录

    返回:
        下载的文件路径列表
    """
    stock_code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)

    if save_dir is None:
        save_dir = os.path.join(DOWNLOAD_DIR, stock_code)

    os.makedirs(save_dir, exist_ok=True)

    # 获取可能的目录列表
    possible_dirs = build_prospectus_urls(stock_code, ipo_date)

    downloaded = []

    for dir_url in possible_dirs:
        try:
            # 尝试列出目录内容
            resp = requests.get(dir_url, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                continue

            # 查找PDF链接
            pdf_links = re.findall(r'href="([^"]+\.pdf)"', resp.text, re.IGNORECASE)

            for pdf_link in pdf_links:
                # 检查是否是招股书（通过文件名或大小判断）
                full_url = urljoin(dir_url, pdf_link)

                # 下载
                filename = os.path.basename(pdf_link)
                filepath = os.path.join(save_dir, filename)

                pdf_resp = requests.get(full_url, headers=HEADERS, timeout=60, stream=True)
                if pdf_resp.status_code == 200:
                    with open(filepath, "wb") as f:
                        for chunk in pdf_resp.iter_content(chunk_size=8192):
                            f.write(chunk)
                    downloaded.append(filepath)
                    print(f"下载: {filepath}", file=sys.stderr)

        except Exception as e:
            continue

    return downloaded


# ============================================================
# 手动URL下载
# ============================================================

def download_from_url(url: str, stock_code: str = None, save_dir: str = None) -> Optional[str]:
    """
    从指定URL下载招股书

    参数:
        url: PDF完整URL
        stock_code: 股票代码（用于命名）
        save_dir: 保存目录

    返回:
        保存的文件路径
    """
    if save_dir is None:
        if stock_code:
            save_dir = os.path.join(DOWNLOAD_DIR, stock_code)
        else:
            save_dir = DOWNLOAD_DIR

    os.makedirs(save_dir, exist_ok=True)

    # 从URL提取文件名
    filename = os.path.basename(url.split("?")[0])
    if not filename.endswith(".pdf"):
        filename = f"prospectus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    return download_pdf(url, save_dir, filename)


# ============================================================
# 已知IPO招股书URL库
# ============================================================

# 常用的IPO招股书URL格式
KNOWN_PROSPECTUS_PATTERNS = {
    # 格式: https://www1.hkexnews.hk/listedco/listconews/sehk/YYYY/MMDD/YYYYMMDDXXXXX.pdf
    # 或者: https://www1.hkexnews.hk/app/sehk/YYYY/XXXXXX/sehkYYMMDDXXXXX.pdf
}

# 已知的IPO招股书链接（手动添加）
# 使用方法：找到招股书后，将URL添加到这里
KNOWN_PROSPECTUS_URLS = {
    # 智谱 (2026-01-08 上市)
    # 招股书URL需要从披露易手动查找后填入
    # "02513": "https://www1.hkexnews.hk/listedco/listconews/sehk/2026/XXXX/XXXXXX.pdf",

    # 商汤 (2021-12-30 上市)
    # "00020": "https://www1.hkexnews.hk/listedco/listconews/sehk/2021/1217/2021121700888.pdf",
}


def get_known_prospectus_url(stock_code: str) -> Optional[str]:
    """获取已知的招股书URL"""
    stock_code = stock_code.replace(".HK", "").replace(".hk", "").zfill(5)
    return KNOWN_PROSPECTUS_URLS.get(stock_code)


# ============================================================
# 招股书内容分析
# ============================================================

def analyze_prospectus_structure(pdf_path: str) -> Dict:
    """
    分析招股书PDF，提取关键IPO信息

    提取内容:
    - 公司名称
    - 股票代码
    - 发行价区间
    - 发行股数
    - 募资规模
    - 基石投资者
    - 股本结构
    - 主要财务数据

    参数:
        pdf_path: PDF文件路径

    返回:
        分析结果字典
    """
    result = {
        "file_path": pdf_path,
        "file_size_mb": 0,
        "extracted_data": {},
        "error": None
    }

    try:
        # 获取文件大小
        result["file_size_mb"] = round(os.path.getsize(pdf_path) / (1024 * 1024), 2)

        # 尝试使用 PyPDF2 或 pdfplumber 提取文本
        try:
            import pdfplumber

            with pdfplumber.open(pdf_path) as pdf:
                # 提取前20页的文本（通常包含关键信息）
                text = ""
                for i, page in enumerate(pdf.pages[:20]):
                    text += page.extract_text() or ""

                result["extracted_data"] = extract_ipo_data_from_text(text)
                result["page_count"] = len(pdf.pages)

        except ImportError:
            try:
                import PyPDF2

                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for i, page in enumerate(reader.pages[:20]):
                        text += page.extract_text() or ""

                    result["extracted_data"] = extract_ipo_data_from_text(text)
                    result["page_count"] = len(reader.pages)

            except ImportError:
                result["error"] = "需要安装 pdfplumber 或 PyPDF2: pip install pdfplumber"
                result["extracted_data"] = {"note": "PDF已下载，需要安装PDF解析库才能提取内容"}

    except Exception as e:
        result["error"] = str(e)

    return result


def extract_ipo_data_from_text(text: str) -> Dict:
    """
    从招股书文本中提取关键IPO数据

    参数:
        text: 招股书文本内容

    返回:
        提取的数据字典
    """
    data = {}

    # 提取发行价区间
    price_patterns = [
        r'發售價\s*[:：]?\s*每股\s*(?:H股\s*)?(?:港幣|HK\$|HKD)?\s*([\d.]+)\s*(?:港幣|HK\$|HKD)?\s*[至到-]\s*(?:港幣|HK\$|HKD)?\s*([\d.]+)',
        r'Offer Price\s*[:：]?\s*HK\$([\d.]+)\s*to\s*HK\$([\d.]+)',
        r'每股\s*(?:港幣|HK\$)?\s*([\d.]+)\s*[至到]\s*(?:港幣|HK\$)?\s*([\d.]+)',
    ]
    for pattern in price_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["offer_price_range"] = f"HK${match.group(1)} - HK${match.group(2)}"
            break

    # 提取募资规模
    proceeds_patterns = [
        r'所得款項淨額\s*(?:約為|約|估計)?\s*(?:港幣|HK\$)?\s*([\d,.]+)\s*(?:百萬|億)',
        r'Net Proceeds\s*(?:approximately|about)?\s*HK\$([\d,.]+)\s*(?:million|billion)',
    ]
    for pattern in proceeds_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["net_proceeds"] = match.group(1)
            break

    # 提取发行股数
    shares_patterns = [
        r'發售股份\s*[:：]?\s*([\d,]+)\s*股',
        r'Offer Shares\s*[:：]?\s*([\d,]+)\s*Shares',
    ]
    for pattern in shares_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["offer_shares"] = match.group(1)
            break

    # 提取基石投资者
    cornerstone_section = re.search(
        r'基石投資者|Cornerstone Investors?',
        text, re.IGNORECASE
    )
    if cornerstone_section:
        data["has_cornerstone"] = True

    # 提取总股本
    total_shares_patterns = [
        r'股本總額\s*[:：]?\s*([\d,]+)\s*股',
        r'Total Share Capital\s*[:：]?\s*([\d,]+)\s*shares',
    ]
    for pattern in total_shares_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data["total_shares"] = match.group(1)
            break

    # 提取公司名称
    company_patterns = [
        r'^(.+?(?:有限公司|LIMITED|Ltd\.?))',
    ]
    for pattern in company_patterns:
        match = re.search(pattern, text[:500], re.MULTILINE | re.IGNORECASE)
        if match:
            data["company_name"] = match.group(1).strip()
            break

    return data


# ============================================================
# 主函数
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("""
港交所IPO招股书下载工具

用法:
  python fetch_ipo_prospectus.py <股票代码>              # 搜索并下载招股书
  python fetch_ipo_prospectus.py <股票代码> --search    # 仅搜索，不下载
  python fetch_ipo_prospectus.py --recent               # 搜索近期IPO
  python fetch_ipo_prospectus.py <股票代码> --ipo-date YYYY-MM-DD  # 指定IPO日期
  python fetch_ipo_prospectus.py --url <PDF_URL>        # 直接下载指定URL
  python fetch_ipo_prospectus.py --analyze <PDF文件>    # 分析已下载的招股书

示例:
  python fetch_ipo_prospectus.py 02513
  python fetch_ipo_prospectus.py 02513 --search
  python fetch_ipo_prospectus.py 02513 --ipo-date 2026-01-08
  python fetch_ipo_prospectus.py --recent
  python fetch_ipo_prospectus.py --url "https://www1.hkexnews.hk/xxx.pdf" --code 02513
  python fetch_ipo_prospectus.py --analyze ~/Downloads/prospectus.pdf
""")
        sys.exit(1)

    arg = sys.argv[1]

    # 直接URL下载模式
    if arg == "--url":
        if len(sys.argv) < 3:
            print("错误: 请提供PDF URL", file=sys.stderr)
            sys.exit(1)

        url = sys.argv[2]
        stock_code = None
        if "--code" in sys.argv:
            idx = sys.argv.index("--code")
            if idx + 1 < len(sys.argv):
                stock_code = sys.argv[idx + 1]

        filepath = download_from_url(url, stock_code)
        if filepath:
            print(json.dumps({
                "url": url,
                "downloaded": filepath
            }, ensure_ascii=False, indent=2))
        return

    # 分析模式
    if arg == "--analyze":
        if len(sys.argv) < 3:
            print("错误: 请提供PDF文件路径", file=sys.stderr)
            sys.exit(1)

        pdf_path = sys.argv[2]
        if not os.path.exists(pdf_path):
            print(f"错误: 文件不存在: {pdf_path}", file=sys.stderr)
            sys.exit(1)

        result = analyze_prospectus_structure(pdf_path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 搜索近期IPO
    if arg == "--recent":
        docs = search_recent_ipos(days_back=90)
        print(json.dumps({
            "count": len(docs),
            "documents": docs[:50]
        }, ensure_ascii=False, indent=2))
        return

    stock_code = arg

    # 仅搜索模式
    if "--search" in sys.argv:
        docs = search_prospectus(stock_code)
        print(json.dumps({
            "stock_code": stock_code,
            "count": len(docs),
            "documents": docs
        }, ensure_ascii=False, indent=2))
        return

    # 指定IPO日期模式
    if "--ipo-date" in sys.argv:
        idx = sys.argv.index("--ipo-date")
        if idx + 1 < len(sys.argv):
            ipo_date = sys.argv[idx + 1]
            downloaded = try_direct_download(stock_code, ipo_date)
            print(json.dumps({
                "stock_code": stock_code,
                "ipo_date": ipo_date,
                "downloaded": downloaded
            }, ensure_ascii=False, indent=2))
            return

    # 检查是否有已知URL
    known_url = get_known_prospectus_url(stock_code)
    if known_url:
        print(f"使用已知URL下载...", file=sys.stderr)
        filepath = download_from_url(known_url, stock_code)
        if filepath:
            print(json.dumps({
                "stock_code": stock_code,
                "source": "known_url",
                "downloaded": [filepath]
            }, ensure_ascii=False, indent=2))
            return

    # 默认：搜索并下载
    downloaded = download_prospectus(stock_code)
    print(json.dumps({
        "stock_code": stock_code,
        "downloaded": downloaded
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
