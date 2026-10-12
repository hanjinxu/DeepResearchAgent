#***********************************************
#      Filename: custom_tools_example.py
#   Description: 自定义 Tool 的外部模块示例
#***********************************************

"""
展示如何在不修改 project-Deep_Research 核心代码的情况下，
通过 registry 注册自定义工具。

每个文件可以独立定义自己的工具集，import 即注册。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deep_research.tools.registry import register_tool
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# ============================================================
# 示例 1：自定义搜索类工具
# ============================================================

@register_tool(category="search", metadata={"provider": "serpapi", "version": "1.0"})
@tool
def custom_web_search(query: str, max_results: int = 5) -> str:
    """
    自定义网络搜索工具。
    通过 SerpAPI / Bing / 自定义引擎搜索互联网。

    Args:
        query: 搜索关键词
        max_results: 返回结果数量（1-10）
    """
    logger.info(f"[custom_web_search] query='{query}', max_results={max_results}")
    # 伪实现：实际使用时替换为真实的 API 调用
    return f"[自定义搜索] '{query}' 的 {max_results} 条结果：\n" + "\n".join([
        f"  {i}. 结果{i}：关于 {query} 的相关信息"
        for i in range(1, max_results + 1)
    ])


@register_tool(category="search", metadata={"requires": ["arxiv_api"]})
@tool
def search_arxiv(query: str, max_results: int = 3) -> str:
    """
    搜索 arXiv 学术论文。
    当需要查找最新学术研究、论文时使用。

    Args:
        query: 论文搜索关键词
        max_results: 返回论文数（默认3）
    """
    logger.info(f"[search_arxiv] query='{query}'")
    return (
        f"[arXiv搜索结果] '{query}'\n"
        f"  1. Title: Recent Advances in {query}\n"
        f"     Authors: A. Smith et al.\n"
        f"     URL: https://arxiv.org/abs/2401.xxxxx\n"
        f"     Summary: 本文讨论了{query}领域的最新进展...\n"
        f"  2. Title: A Survey of {query}\n"
        f"     Authors: B. Wang et al.\n"
        f"     URL: https://arxiv.org/abs/2402.xxxxx\n"
    )


# ============================================================
# 示例 2：自定义分析类工具
# ============================================================

@register_tool(category="analysis", metadata={"type": "pseudo"})
@tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """
    对给定文本进行智能摘要。

    Args:
        text: 需要摘要的原文
        max_length: 摘要的最大字数
    """
    # 伪实现：截取前 max_length 字
    summary = text[:max_length] + ("..." if len(text) > max_length else "")
    return f"[摘要] {summary}"


@register_tool(category="analysis", metadata={"type": "pseudo"})
@tool
def compare_documents(doc_a: str, doc_b: str, aspects: list[str]) -> str:
    """
    对比两篇文档在指定维度上的异同。

    Args:
        doc_a: 文档A的内容
        doc_b: 文档B的内容
        aspects: 对比维度列表
    """
    result = ["【文档对比分析】"]
    for a in aspects:
        result.append(f"\n[{a}]")
        result.append(f"  文档A：{doc_a[:50]}...")
        result.append(f"  文档B：{doc_b[:50]}...")
        result.append(f"  差异分析：基于{a}维度，两篇文档存在以下异同...")
    return "\n".join(result)


# ============================================================
# 示例 3：Schema 式（伪工具）— 用于决策流程
# ============================================================

from langchain_core.tools import tool as langchain_tool_schema
from pydantic import BaseModel, Field


@register_tool(category="reasoning", metadata={"type": "pseudo"})
@tool
class EvaluateConfidence(BaseModel):
    """
    评估当前研究结果的可信度。
    伪工具：不执行外部操作，只做自我评估记录。
    """
    conclusion: str = Field(description="当前得出的结论或观点")
    supporting_evidence: list[str] = Field(description="支持该结论的证据列表")
    confidence_level: float = Field(description="可信度评分 0.0-1.0", ge=0.0, le=1.0)
    remaining_risks: list[str] = Field(description="仍然存在的风险或不确定性")


@register_tool(category="reasoning", metadata={"type": "pseudo"})
@tool
class ProposeHypothesis(BaseModel):
    """
    基于已有信息提出一个研究假设。
    伪工具：仅记录假设，供后续搜索验证。
    """
    hypothesis: str = Field(description="提出的研究假设")
    rationale: str = Field(description="提出该假设的理由和依据")
    test_method: str = Field(description="如何验证这个假设（需要搜索什么信息）")


# ============================================================
# 示例 4：函数式工具 — 外部 API
# ============================================================

@register_tool(category="data", metadata={"requires": ["api_key"]})
@tool
def fetch_stock_data(ticker: str, days: int = 30) -> str:
    """
    获取指定股票的历史价格数据。

    Args:
        ticker: 股票代码，如 AAPL, TSLA, 0700.HK
        days: 获取最近多少天的数据
    """
    return (
        f"[股票数据] {ticker} 最近 {days} 天\n"
        f"  开盘范围：XXX - XXX\n"
        f"  收盘范围：XXX - XXX\n"
        f"  成交量：XXX\n"
        f"  （实际使用时替换为真实 API 调用）"
    )


@register_tool(category="data", metadata={"requires": []})
@tool
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """
    获取指定时区的当前时间。

    Args:
        timezone: 时区名称，如 Asia/Shanghai, US/Eastern
    """
    from datetime import datetime, timezone as tz
    import pytz
    try:
        tz_obj = pytz.timezone(timezone)
        now = datetime.now(tz_obj)
        return f"[{timezone}] 当前时间：{now.strftime('%Y-%m-%d %H:%M:%S %Z')}"
    except Exception as e:
        return f"[错误] 无效时区 '{timezone}': {e}"


# ============================================================
# 示例 5：用 category 做分组，方便按角色绑定不同工具集
# ============================================================

"""
推荐的工具分类体系：

  search     — 搜索类工具（web, arxiv, news, academic）
  analysis   — 分析类工具（摘要、对比、分类、提取）
  reasoning  — 推理类工具（思考、评估、假设、决策）
  data       — 数据类工具（股票、天气、时间、数据库）
  write      — 写作类工具（草稿、精修、格式化）
  delegate   — 委派类工具（子任务分派）
  system     — 系统类工具（完成、终止、状态检查）

在 agent 中按角色绑定不同分类：
  research_agent  → model.bind_tools(get_tools(categories=["search", "reasoning"]))
  supervisor      → model.bind_tools(get_tools(categories=["delegate", "write", "reasoning", "system"]))
  writer          → model.bind_tools(get_tools(categories=["write"]))
"""


def print_registration_summary():
    """打印当前注册的自定义工具总结"""
    from deep_research.tools.registry import list_tools
    print("自定义工具已全部注册。")
    print("=" * 60)
    list_tools()
