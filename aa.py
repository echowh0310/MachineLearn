import time
import requests


def fetch_url(url):
    """模拟一个耗时的网络请求（同步版本）"""
    print(f"开始获取: {url}")
    time.sleep(2)  # 模拟 2 秒网络延迟
    print(f"完成获取: {url}")
    return f"来自 {url} 的数据"


def main_sync():
    urls = ['https://example.com/1', 'https://example.com/2', 'https://example.com/3']
    results = []
    start = time.time()

    for url in urls:
        result = fetch_url(url)  # 必须等上一个完成才能开始下一个
        results.append(result)

    end = time.time()
    print(f"同步版本总耗时: {end - start:.2f} 秒")
    print(f"结果: {results}")


if __name__ == "__main__":
    main_sync()