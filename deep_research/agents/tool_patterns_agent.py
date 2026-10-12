#***********************************************
#      Filename: tool_patterns_agent.py
#   Description: 展示LangGraph四种Tool模式的Agent
#***********************************************

"""
此文件展示在 LangGraph 中实现可调用 tool 的 agent 的四种模式：

模式一：函数式 Tool     — @tool def fn(args): 实际执行，如搜索
模式二：Schema 式 Tool   — @tool class X(BaseModel): 触发子图/子流程
模式三：伪 Tool          — 不执行外部操作，只记录到 state，如 Think
模式四：混合模式          — 以上三种组合使用

遵循本项目的风格：config.yml → get_chat_model → state graph
不修改任何现有文件，仅作为参考示例。
"""

from typing_extensions import Literal, TypedDict, Annotated, Sequence, List, Optional
from pydantic import BaseModel, Field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

from deep_research.llm import get_chat_model
from deep_research.utils import get_today_str
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)


# ============================================================
# 1. State 定义
# ============================================================

class ToolPatternsState(TypedDict):
    """展示四种 tool 模式的全局状态"""

    # LangGraph 标准消息历史（来自 MessagesState）
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 伪工具 Think 的记录
    reflection_history: Annotated[List[str], operator.add]

    # Schema 式工具触发的待处理子任务
    pending_delegations: List[dict]

    # 最终输出
    final_answer: str

    # 迭代计数器
    iteration: int


# ============================================================
# 2. 四种 Tool 模式的定义
# ============================================================

# -----------------------------------------------------------
# 模式一：函数式 Tool — 真实执行外部操作
# -----------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """
    搜索网络获取信息。当你需要查询实时信息、获取最新数据时使用此工具。

    Args:
        query: 搜索关键词，应尽可能具体
    """
    logger.info(f"[TOOL:search_web] query='{query}'")
    # 伪实现：返回模拟结果
    return (
        f"【搜索结果】\n"
        f"查询：{query}\n"
        f"结果1：[来源A] 关于'{query}'的核心信息概述...\n"
        f"结果2：[来源B] 相关详细分析和数据...\n"
        f"结果3：[来源C] 补充参考信息..."
    )


@tool
def fetch_page(url: str) -> str:
    """
    获取指定 URL 的网页内容。当搜索结果摘要不足时，需要获取原文时使用。

    Args:
        url: 需要获取内容的完整网页地址
    """
    logger.info(f"[TOOL:fetch_page] url='{url}'")
    return (
        f"【页面内容摘要】\n"
        f"URL：{url}\n"
        f"标题：（页面标题）\n"
        f"正文：（页面内容的摘要，约 2000 字）\n"
        f"--- 内容已获取 ---"
    )


# -----------------------------------------------------------
# 模式二：Schema 式 Tool — 定义参数 schema，触发子图
# -----------------------------------------------------------

@tool
class DelegateResearch(BaseModel):
    """
    将研究子任务委派给专门的子 Agent。
    当你发现需要独立调研一个子主题时调用此工具。

    调用后系统会自动为该主题启动一个子 Agent 进行独立研究，
    结果会在下一轮返回。
    """
    sub_topic: str = Field(
        description="子研究主题，必须详细描述（至少一段），包含研究目的和需要回答的具体问题"
    )
    focus_areas: List[str] = Field(
        description="需要重点关注的方面列表，例如 ['市场规模', '主要玩家', '技术路线']"
    )


@tool
class TaskComplete(BaseModel):
    """标记当前任务已完成，准备输出最终答案。调用此工具即表示研究结束。"""
    summary: str = Field(description="对最终答案的简要总结，用于验证研究是否完整")
    pass


# -----------------------------------------------------------
# 模式三：伪 Tool — 不执行外部操作，只记录思考过程到 state
# -----------------------------------------------------------

@tool
class Think(BaseModel):
    """
    停下来反思当前进展，规划下一步行动。
    在执行任何其他操作之前，应当先思考。
    这不会触发外部调用，只是记录你的思考过程供后续参考。
    """
    reflection: str = Field(description="对当前进展的深度反思：找到了什么、还缺什么、有什么疑虑")
    next_plan: str = Field(description="下一步的具体行动计划：搜索什么、委派什么、还是准备收尾")


@tool
class QualityCheck(BaseModel):
    """
    对当前收集的信息做自我质量检查。
    评估已有信息是否能支撑最终结论。
    这是一个伪工具，不会调用外部 API。
    """
    question: str = Field(description="当前的研究问题")
    info_collected: str = Field(description="已收集的核心信息摘要")
    gaps: List[str] = Field(description="仍然缺失的信息清单")


# -----------------------------------------------------------
# 工具注册池
# -----------------------------------------------------------

# 注册所有工具
all_tools = [
    search_web,        # 模式一：函数式
    fetch_page,        # 模式一：函数式
    DelegateResearch,  # 模式二：Schema 式
    TaskComplete,      # 模式二：Schema 式
    Think,             # 模式三：伪工具
    QualityCheck,      # 模式三：伪工具
]

# 按名称索引，用于 tool_node 路由
tools_by_name = {
    tool.name if hasattr(tool, "name") else tool.__name__: tool
    for tool in all_tools
}

logger.info(f"已注册工具：{list(tools_by_name.keys())}")

# 初始化模型
model = get_chat_model("supervisor")  # 复用 supervisor 角色的模型
model_with_tools = model.bind_tools(all_tools)

# 最大迭代次数
MAX_ITERATIONS = 15


# ============================================================
# 3. Node 定义
# ============================================================

def agent_node(state: ToolPatternsState) -> Command[Literal["tool_executor"]]:
    """
    Agent 节点：LLM 读取当前状态，决定调用哪个工具。
    这是 LangGraph 的"思考"节点——LLM 是编排者。
    """
    iteration = state.get("iteration", 0)
    messages = state.get("messages", [])

    system_message = SystemMessage(
        content=f"""你是一个研究助理。今天的日期是 {get_today_str()}。

你有以下工具可用：

【搜索类】search_web — 网络搜索；fetch_page — 获取网页正文
【委派类】DelegateResearch — 启动子 Agent 专项研究
【思考类】Think — 反思进展、规划步骤；QualityCheck — 质量自检
【完成类】TaskComplete — 研究结束，输出最终答案

规则：
1. 先 Think 再行动 —— 调用其他工具前先反思
2. 信息不足时用 search_web，需要专题研究时用 DelegateResearch
3. 质量不达标时用 QualityCheck 自查
4. 确认完成后用 TaskComplete 结束
5. 最多 {MAX_ITERATIONS} 次迭代
"""
    )

    response = model_with_tools.invoke([system_message] + list(messages))

    logger.info(
        "[AGENT] iteration=%d, tool_calls=%s",
        iteration,
        [tc["name"] for tc in (response.tool_calls or [])],
    )

    return Command(
        goto="tool_executor",
        update={
            "messages": [response],
            "iteration": iteration + 1,
        },
    )


def tool_executor(
    state: ToolPatternsState,
) -> Command[Literal["agent", "__end__"]]:
    """
    Tool 执行节点：根据 LLM 选的工具，按模式分流执行。
    这是四种模式实际被执行的地方。
    """
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls or []
    updates: dict = {}
    next_step = "agent"

    if not tool_calls:
        # LLM 没有调用任何工具 → 结束
        return Command(goto="__end__")

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        logger.warning("[TOOL] 达到最大迭代次数，强制结束")
        return Command(goto="__end__", update={"final_answer": "研究超时，未完成"})

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]

        # ── 模式一：函数式 Tool — 直接调用 ──
        if name == "search_web":
            logger.info(f"[EXEC:函数式] search_web({args})")
            result = search_web.invoke(args)
            updates.setdefault("messages", []).append(
                ToolMessage(content=result, name=name, tool_call_id=tc["id"])
            )

        elif name == "fetch_page":
            logger.info(f"[EXEC:函数式] fetch_page({args})")
            result = fetch_page.invoke(args)
            updates.setdefault("messages", []).append(
                ToolMessage(content=result, name=name, tool_call_id=tc["id"])
            )

        # ── 模式二：Schema 式 Tool — 路由到子图 / 记录待处理 ──
        elif name == "DelegateResearch":
            logger.info(f"[EXEC:Schema式] DelegateResearch({args})")
            # Schema 式工具不在这里执行真实逻辑，
            # 而是把任务记录到 state，路由到子图节点
            sub_topics = args.get("sub_topic", "")
            focus = args.get("focus_areas", [])
            updates["pending_delegations"] = [
                {"sub_topic": sub_topics, "focus_areas": focus}
            ]
            # 在当前实现中，直接返回模拟结果
            updates.setdefault("messages", []).append(
                ToolMessage(
                    content=(
                        f"【子Agent报告】\n"
                        f"主题：{sub_topics}\n"
                        f"重点：{', '.join(focus)}\n"
                        f"研究完成，返回 {len(focus)} 个方面的详细分析。"
                    ),
                    name=name,
                    tool_call_id=tc["id"],
                )
            )

        elif name == "TaskComplete":
            logger.info(f"[EXEC:Schema式] TaskComplete({args})")
            # 标记结束
            updates["final_answer"] = args.get("summary", "")
            next_step = "__end__"

        # ── 模式三：伪 Tool — 只写 state，不执行外部操作 ──
        elif name == "Think":
            logger.info(f"[EXEC:伪工具] Think({args})")
            reflection = args.get("reflection", "")
            plan = args.get("next_plan", "")
            # 记录到 state，不调用任何外部
            updates.setdefault("reflection_history", []).append(
                f"[iter {state.get('iteration', 0)}] {reflection} → 计划：{plan}"
            )
            updates.setdefault("messages", []).append(
                ToolMessage(
                    content=f"✓ 思考已记录。下一步计划：{plan}",
                    name=name,
                    tool_call_id=tc["id"],
                )
            )

        elif name == "QualityCheck":
            logger.info(f"[EXEC:伪工具] QualityCheck({args})")
            # 伪工具：只做自我评估，不调用外部
            gaps = args.get("gaps", [])
            gap_text = "\n".join([f"  - {g}" for g in gaps])
            updates.setdefault("messages", []).append(
                ToolMessage(
                    content=(
                        f"【质量检查结果】\n"
                        f"问题：{args.get('question', '')}\n"
                        f"已有信息：{args.get('info_collected', '')[:100]}...\n"
                        f"信息缺口：\n{gap_text if gap_text else '  无，信息已完整'}\n"
                        f"{'建议继续搜索补充缺失信息。' if gaps else '质量达标，可以准备结束。'}"
                    ),
                    name=name,
                    tool_call_id=tc["id"],
                )
            )

        else:
            logger.warning(f"[EXEC] 未知工具：{name}")
            updates.setdefault("messages", []).append(
                ToolMessage(
                    content=f"错误：未知工具 '{name}'",
                    name=name,
                    tool_call_id=tc["id"],
                )
            )

    logger.info("[TOOL_EXECUTOR] next_step=%s, updates_keys=%s", next_step, list(updates.keys()))
    return Command(goto=next_step, update=updates)


# ============================================================
# 4. Graph 构建
# ============================================================

def build_agent():
    """构建并编译展示四种 tool 模式的 LangGraph agent"""

    builder = StateGraph(ToolPatternsState)

    # 添加节点
    builder.add_node("agent", agent_node)
    builder.add_node("tool_executor", tool_executor)

    # 添加边
    builder.add_edge(START, "agent")
    builder.add_edge("agent", "tool_executor")
    builder.add_conditional_edges(
        "tool_executor",
        lambda state: "__end__" if state.get("final_answer") else "agent",
        {
            "agent": "agent",
            "__end__": "__end__",
        },
    )

    agent = builder.compile()
    return agent


# 编译 agent
tool_patterns_agent = build_agent()


# ============================================================
# 5. 流程图
# ============================================================

def print_graph():
    """打印 agent 的图结构"""
    print("=" * 60)
    print("Tool Patterns Agent 图结构")
    print("=" * 60)
    print(tool_patterns_agent.get_graph().draw_ascii())
    print()
    print("四种 Tool 模式的位置：")
    print("  ┌─ agent 节点        ← LLM 选择调哪个工具")
    print("  │")
    print("  └─ tool_executor 节点  ← 按模式分流执行")
    print("       ├─ 模式一（函数式）: search_web, fetch_page → 直接调用")
    print("       ├─ 模式二（Schema式）: DelegateResearch, TaskComplete → 路由/标记")
    print("       ├─ 模式三（伪工具）: Think, QualityCheck → 写 state，无外部调用")
    print("       └─ 模式四（混合）: 以上三类在同一个 tool_executor 中共存")


# ============================================================
# 6. 运行入口
# ============================================================

if __name__ == "__main__":
    print_graph()

    # 测试运行
    import asyncio

    async def test():
        print("\n" + "=" * 60)
        print("运行示例：研究 AI Agent 的最新发展")
        print("=" * 60)

        initial_state: ToolPatternsState = {
            "messages": [
                HumanMessage(content="帮我调研一下 2025 年 AI Agent 框架的最新发展趋势")
            ],
            "reflection_history": [],
            "pending_delegations": [],
            "final_answer": "",
            "iteration": 0,
        }

        result = await tool_patterns_agent.ainvoke(initial_state)

        print("\n" + "=" * 60)
        print("执行完成")
        print("=" * 60)
        print(f"迭代次数：{result.get('iteration', 0)}")
        print(f"消息数：{len(result.get('messages', []))}")
        print(f"思考记录：{len(result.get('reflection_history', []))} 条")
        print(f"最终答案：{result.get('final_answer', '未调用 TaskComplete')[:200]}...")

        print("\n思考历史：")
        for i, ref in enumerate(result.get("reflection_history", []), 1):
            print(f"  [{i}] {ref[:120]}...")

    asyncio.run(test())
