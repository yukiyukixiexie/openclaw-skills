#!/usr/bin/env python3
"""
港股 Ticker 搜索工具（带缓存）
按公司名称搜索港股代码，优先查本地缓存
"""

import requests
import sys
import json
import os
from datetime import datetime

# 缓存文件路径（与脚本同目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(SCRIPT_DIR, "ticker_cache.json")


def load_cache() -> dict:
    """加载缓存"""
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_cache(cache: dict):
    """保存缓存"""
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"保存缓存失败: {e}", file=sys.stderr)


def search_from_api(keyword: str) -> list:
    """
    从东方财富API搜索港股代码
    """
    try:
        url = "https://searchapi.eastmoney.com/api/suggest/get"
        params = {
            "input": keyword,
            "type": "14",
            "token": "D43BF722C8E33BDC906FB84D85E326E8",
            "count": "10"
        }
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Referer": "https://quote.eastmoney.com/"
        }

        resp = requests.get(url, params=params, headers=headers, timeout=10)
        data = resp.json()

        results = []
        if data.get("QuotationCodeTable", {}).get("Data"):
            for item in data["QuotationCodeTable"]["Data"]:
                code = item.get("Code", "")
                name = item.get("Name", "")

                # 只要港股正股（5位数字代码，排除窝轮/牛熊证）
                if len(code) == 5 and code.isdigit():
                    if any(x in name for x in ["购", "沽", "牛", "熊"]):
                        continue
                    if int(code) > 10000:
                        continue
                    results.append({"code": code, "name": name, "market": "HK"})

        return results
    except Exception as e:
        return [{"error": str(e)}]


def get_ticker(keyword: str, use_cache: bool = True) -> dict:
    """
    获取股票代码（主入口）

    逻辑：
    1. 如果输入已经是5位数字代码，直接返回
    2. 查缓存
    3. 缓存没有就搜索API
    4. 搜到后更新缓存

    Returns:
        dict: {"code": "00020", "name": "商汤-W", "source": "cache/api"}
    """
    # 如果已经是代码格式，直接返回
    clean_input = keyword.replace(".HK", "").replace(".hk", "").strip()
    if clean_input.isdigit() and len(clean_input) <= 5:
        return {
            "code": clean_input.zfill(5),
            "name": "",
            "source": "direct"
        }

    # 查缓存
    if use_cache:
        cache = load_cache()
        if keyword in cache:
            result = cache[keyword]
            return {
                "code": result["code"],
                "name": result["name"],
                "source": "cache"
            }

    # 搜索API
    results = search_from_api(keyword)

    if not results:
        return {"error": f"未找到 '{keyword}'"}

    if "error" in results[0]:
        return results[0]

    # 取第一个结果
    best_match = results[0]

    # 更新缓存
    cache = load_cache()
    cache[keyword] = {
        "code": best_match["code"],
        "name": best_match["name"],
        "updated": datetime.now().strftime("%Y-%m-%d")
    }
    save_cache(cache)

    return {
        "code": best_match["code"],
        "name": best_match["name"],
        "source": "api",
        "all_results": results  # 如果有多个匹配，也返回
    }


def main():
    if len(sys.argv) < 2:
        print("用法: python search_ticker.py <公司名称或代码>")
        print("示例: python search_ticker.py 商汤")
        print("      python search_ticker.py 00020")
        print("\n选项:")
        print("  --no-cache    不使用缓存，强制搜索")
        print("  --list        列出所有缓存的公司")
        sys.exit(1)

    # 处理选项
    if "--list" in sys.argv:
        cache = load_cache()
        print("\n已缓存的公司:")
        print("-" * 50)
        for name, info in sorted(cache.items()):
            print(f"{name:10} -> {info['code']}.HK  {info['name']}")
        print("-" * 50)
        print(f"共 {len(cache)} 条记录")
        sys.exit(0)

    keyword = sys.argv[1]
    use_cache = "--no-cache" not in sys.argv

    result = get_ticker(keyword, use_cache=use_cache)

    if "error" in result:
        print(f"❌ {result['error']}")
        sys.exit(1)

    # 输出结果
    source_label = {
        "cache": "📦 缓存",
        "api": "🔍 搜索",
        "direct": "✓ 代码"
    }.get(result["source"], result["source"])

    print(f"\n{source_label}: {result['code']}.HK", end="")
    if result["name"]:
        print(f"  ({result['name']})")
    else:
        print()

    # 如果是新搜索且有多个结果，显示所有
    if result.get("all_results") and len(result["all_results"]) > 1:
        print("\n其他匹配:")
        for i, r in enumerate(result["all_results"][1:], 2):
            print(f"  {i}. {r['code']}.HK  {r['name']}")
        print(f"\n已将 '{keyword}' -> {result['code']} 添加到缓存")

    # 输出JSON供程序调用
    print(json.dumps({"code": result["code"], "name": result["name"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
