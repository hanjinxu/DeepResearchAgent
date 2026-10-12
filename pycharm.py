# 定义工具

from langchain_core.tools import tool


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """搜索网络获取信息

    Args:
        query: 搜索关键词
        max_results: 返回的最大结果数
    """
    # 实际调用搜索 API
    results = tavily_client.search(query, max_results=max_results)
    return format_results(results)


@tool
def calculate(expression: str) -> str:
    """执行数学计算

    Args:
        expression: 数学表达式，如 "1 + 2 * 3"
    """
    return str(eval(expression))

