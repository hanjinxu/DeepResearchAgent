#***********************************************
#      Filename: agent_builder.py
#   Description: 多智能体深度研究Builder 
#***********************************************


from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from deep_research.utils import get_today_str
from deep_research.states import AgentState, AgentInputState
from deep_research.prompts import (
    FINAL_REPORT_PROMPT,
    LEGAL_AUDIT_FINAL_REPORT_PROMPT,
    is_legal_audit_query,
)
from deep_research.agents import write_research_brief, write_draft_report
from deep_research.agents import supervisor_agent
from deep_research.llm import get_chat_model


from langchain_core.runnables import RunnableConfig

# ===== Config =====

writer_model = get_chat_model("writer")


def _pick_final_report_prompt(query_text: str, force_template: str = "auto") -> tuple[str, str]:
    """
    根据 query 文本或显式配置选择 final_report 的 prompt。
    返回: (template_name, prompt_text)
    """
    if force_template == "legal_audit":
        return "legal_audit", LEGAL_AUDIT_FINAL_REPORT_PROMPT
    if force_template == "general":
        return "general", FINAL_REPORT_PROMPT
    # auto 模式：关键词检测
    if is_legal_audit_query(query_text or ""):
        return "legal_audit", LEGAL_AUDIT_FINAL_REPORT_PROMPT
    return "general", FINAL_REPORT_PROMPT


async def final_report_generation(state: AgentState, config: RunnableConfig | None = None):
    """最终报告的生成: 用户query，研究简报，findings, 报告初稿 => 报告
    根据 query 是否命中"舆情/法务"关键词，自动套用 Layer 1 硬性骨架模板。
    可通过 config={"configurable": {"report_template": "legal_audit"}} 强制模板。
    """

    # 取出所有的notes
    notes = state.get("notes", [])
    findings = "\n".join(notes)

    # 读取 force_template（来自 config.configurable，无则按 auto）
    cfg = (config or {}).get("configurable") or {}
    force_tpl = cfg.get("report_template", "auto")

    # 提取用户 query（最后一条 HumanMessage）
    user_query = ""
    for msg in reversed(state.get("messages", [])):
        if msg.__class__.__name__ == "HumanMessage":
            user_query = msg.content if isinstance(msg.content, str) else ""
            break

    # 选 prompt 模板
    tpl_name, prompt_tpl = _pick_final_report_prompt(user_query, force_template=force_tpl)

    # 组装prompt
    final_report_prompt = prompt_tpl.format(
        research_brief=state.get("research_brief", ""),
        findings=findings,
        date=get_today_str(),
        draft_report=state.get("draft_report", ""),
    )

    # 生成最后的报告
    final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])

    print(f"\n📐 [Final Report] 模板选择: {tpl_name} | query 命中关键词: {is_legal_audit_query(user_query)}")

    return {
        "final_report": final_report.content,
        "messages": ["最终的报告: " + final_report.content],
    }


# ===== 图的构建 =====

# 构建深度研究的workflow
deep_researcher_builder = StateGraph(AgentState, input_schema=AgentInputState)

# 添加节点
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("write_draft_report", write_draft_report)
deep_researcher_builder.add_node("supervisor_subgraph", supervisor_agent)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)

# 添加边
deep_researcher_builder.add_edge(START, "write_research_brief")
deep_researcher_builder.add_edge("write_research_brief", "write_draft_report")
deep_researcher_builder.add_edge("write_draft_report", "supervisor_subgraph")
deep_researcher_builder.add_edge("supervisor_subgraph", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)

# 编译graph
agent = deep_researcher_builder.compile()
