#***********************************************
#      Filename: draft_agent.py
#   Description: 根据用户需求生成报告草稿  
#***********************************************


"""该文件的作用是澄清用户问题和生成研究简要报
主要包含以下3个模块：
1. 用户问题澄清：确认是否有必要反问
2. 确定提纲：根据问题生成一个研究的简报
3. 报告草稿：生成一个初步的报告草稿
"""

import os
import json
import re
from typing_extensions import Literal
from rich.markdown import Markdown
from rich.console import Console

from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research import logging as dr_logging
from deep_research.llm import get_chat_model
from deep_research.prompts import (
    RESEARCH_BRIEF_PROMPT,
    DRAFT_REPORT_PROMPT,
    LEGAL_AUDIT_DRAFT_REPORT_PROMPT,
    is_legal_audit_query,
)
from deep_research.states import AgentState, ResearchQuestion, AgentInputState, DraftReport
from deep_research.utils import get_today_str 

logger = dr_logging.get_logger(__name__)


# 初始化模型 
draft_model = get_chat_model("draft")


# ===== Langgraph的节点 =====

def write_research_brief(state: AgentState) -> Command[Literal["write_draft_report"]]:
    """根据用户的query生成一个研究提纲，内容包含需要调研哪些方面，注意事项等等"""

    logger.debug(
        "write research_brief invoked with %d messages", len(state.get("messages", []))
    )

    # 组装prompt
    prompt = RESEARCH_BRIEF_PROMPT.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str()
    ) + "\n\n请以JSON格式返回，格式为：{\"research_brief\": \"...\"}"
    logger.debug("write_research_brief invoking draft_model with prompt_length=%d", len(prompt))

    # 直接调用模型，手动解析JSON（DeepSeek不支持response_format）
    response = draft_model.invoke([HumanMessage(content=prompt)])
    text = response.content
    json_match = re.search(r'\{.*"research_brief".*\}', text, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        research_brief = parsed["research_brief"]
    else:
        # fallback: 去掉可能的markdown标记
        research_brief = text.strip().strip("`").strip()
    logger.debug("write_research_brief produced research_brief length=%d", len(research_brief))

    # 更新Messages并回传给主agent 
    return Command(
            goto="write_draft_report", 
            update={"research_brief": research_brief}
        )

def write_draft_report(state: AgentState) -> Command[Literal["__end__"]]:
    """生成提纲生成一个研究报告的草稿"""

    logger.debug(
        "write_draft_report invoked with research_brief present=%s",
        bool(state.get("research_brief")),
    )

    # 组装prompt
    research_brief = state.get("research_brief", "")

    # 提取用户原始 query（最后一条 HumanMessage），做舆情/法务关键词检测：
    # 命中则草稿阶段即套用 Layer 1 四段骨架（与最终报告模板结构一致）
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "HumanMessage":
            user_query = msg.content if isinstance(msg.content, str) else ""
            break

    if is_legal_audit_query(user_query):
        draft_tpl = LEGAL_AUDIT_DRAFT_REPORT_PROMPT
        logger.debug("write_draft_report using legal_audit draft template")
    else:
        draft_tpl = DRAFT_REPORT_PROMPT

    draft_report_prompt = draft_tpl.format(
        research_brief=research_brief,
        date=get_today_str()
    ) + "\n\n请以JSON格式返回，格式为：{\"draft_report\": \"...\"}"

    # 直接调用模型，手动解析JSON
    response = draft_model.invoke([HumanMessage(content=draft_report_prompt)])
    text = response.content
    json_match = re.search(r'\{.*"draft_report".*\}', text, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        draft_report = parsed["draft_report"]
    else:
        draft_report = text.strip().strip("`").strip()
    logger.debug("write_draft_report produced draft_report length=%d", len(draft_report))

    return {
        "research_brief": research_brief,
        "draft_report": draft_report, 
        "supervisor_messages": [
            HumanMessage(content="Here is the draft report: " + draft_report),
            HumanMessage(content=research_brief)
        ]
    }


if __name__ == "__main__":
    # 构建Graph
    deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

    # 增加节点 
    deep_researcher_builder.add_node("write_research_brief", write_research_brief)
    deep_researcher_builder.add_node("write_draft_report", write_draft_report)

    # 增加边
    deep_researcher_builder.add_edge(START, "write_research_brief")
    deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
    deep_researcher_builder.add_edge("write_draft_report", END)

    # 编译graph
    draft_agent = deep_researcher_builder.compile()

    # 打印graph
    print(draft_agent.get_graph().draw_ascii())

    # 测试问题
    thread = {"configurable": {"thread_id": "1", "recursion_limit": 50}}
    result = draft_agent.invoke({"messages": [HumanMessage(content="帮我写一个关于英伟达最新GPU的调研报告")]}, config=thread)

    # 输出
    console = Console()
    print("=====  Research Brief ====")
    console.print(Markdown(result["research_brief"]))
    print()

    print("=====  Draft Report ====")
    console.print(Markdown(result["draft_report"]))

