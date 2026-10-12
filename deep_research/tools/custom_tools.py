#***********************************************
#      Filename: custom_tools.py
#   Description: 自定义Tool — 通过Registry注册，不修改核心代码
#***********************************************

"""
自定义工具模块。

所有工具通过 @register 自动注册到全局注册表，
不需要修改 tools/tool.py 或任何核心文件。
"""

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from deep_research.tools.registry import register
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# ============================================================
# 模式一：函数式 Tool — 由 @tool 装饰的普通函数
# ============================================================

@register(roles=["supervisor", "researcher"])
@tool(parse_docstring=True)
def search_web(query: str) -> str:
    """搜索网络获取信息

    Args:
        query: 搜索关键词
    """
    logger.info(f"[search_web] {query}")
    return (
        f"【搜索结果】\n"
        f"查询：{query}\n"
        f"来源1：相关文章A — 摘要信息\n"
        f"来源2：相关文章B — 详细分析\n"
        f"来源3：相关文章C — 补充数据"
    )


@register(roles=["supervisor", "researcher"])
@tool(parse_docstring=True)
def get_page_content(url: str) -> str:
    """获取网页正文内容

    Args:
        url: 网页地址
    """
    return f"【页面内容】{url}\n正文摘要：...（约2000字）"


# ============================================================
# 模式二：Schema 式 Tool — 用 BaseModel 定义参数
# ============================================================

@register(roles=["supervisor"])
@tool(parse_docstring=True)
class DelegateResearch(BaseModel):
    """将研究子任务委派给专门的子Agent"""

    research_topic: str = Field(
        description="研究主题，必须详细描述至少一个段落"
    )
    focus_areas: list[str] = Field(
        description="重点关注的方面"
    )


@register(roles=["supervisor", "researcher"])
@tool(parse_docstring=True)
class TaskComplete(BaseModel):
    """标记研究任务已完成，准备输出最终答案"""

    summary: str = Field(description="对最终答案的简要总结")


# ============================================================
# 模式三：伪 Tool — 不执行外部操作，只记录状态
# ============================================================

@register(roles=["supervisor", "researcher"])
@tool(parse_docstring=True)
class Think(BaseModel):
    """反思当前进展，规划下一步行动"""

    reflection: str = Field(description="对当前进展的深度反思")
    next_plan: str = Field(description="下一步的具体行动计划")


@register(roles=["supervisor"])
@tool(parse_docstring=True)
class CheckQuality(BaseModel):
    """检查当前报告或信息的质量"""

    content: str = Field(description="待检查的内容")
    criteria: list[str] = Field(
        description="检查标准，如 ['完整性', '准确性', '可读性']"
    )


# ============================================================
# 增加新工具只需要在这里加，什么都不用改
# ============================================================

@register(roles=["researcher"])
@tool(parse_docstring=True)
def search_papers(topic: str, year: int = 2025) -> str:
    """搜索学术论文

    Args:
        topic: 论文主题
        year: 发表年份
    """
    return (
        f"【论文搜索结果】\n"
        f"主题：{topic} ({year})\n"
        f"1. Paper A — 作者, 期刊\n"
        f"2. Paper B — 作者, 期刊\n"
        f"3. Paper C — 作者, 期刊"
    )


@register(roles=["writer"])
@tool(parse_docstring=True)
def format_report(content: str, style: str = "markdown") -> str:
    """格式化报告输出

    Args:
        content: 原始内容
        style: 输出格式
    """
    return f"【格式化报告】\n{content}"
