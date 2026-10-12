#***********************************************
#      Filename: agent_with_custom_tools.py
#   Description: 使用自定义 Tool 注册中心的 Agent
#***********************************************

"""
一个完整的 LangGraph Agent，展示如何用 registry 加载自定义工具。

关键设计点：
  1. 不修改 tool.py — 所有自定义工具在外部模块声明
  2. 按 category 分组 — 不同 agent 角色绑定不同工具集
  3. 在 tool_executor 里用 registry 路由
  4. import 即注册 — 导入 custom_tools_example 自动完成注册
"""

from typing_extensions import Literal, TypedDict, Annotated, Sequence, List, Optional

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import Command

from deep_research.llm import get_chat_model
from deep_research.utils import get_today_str
from deep_research import logging as dr_logging

# ── 核心：注册中心 ──
from deep_research.tools.registry import (
    register_tool,
    get_tools,
    get_tools_by_name,
    freeze,
)

# ── 外部自定义工具（import 即注册） ──
# 这些模块不修改任何核心文件，只是声名它们自己的工具
from deep_research.tools import custom_tools_example  # noqa: F401

# 也可以用自己的独立文件注册：
# from my_project.my_tools import my_tool  # @register_tool 装饰器自动注册

logger = dr_logging.get_logger(__name__)


# ============================================================
# 状态定义
# ============================================================

class CustomToolAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    iteration: int
    final_answer: str


# ============================================================
# Agent 构建
# ============================================================

MAX_ITERATIONS = 15

# 按角色选择工具分类
# supervisor 级别的 agent 需要：搜索 + 推理 + 分析 + 数据
TOOL_CATEGORIES = ["search", "reasoning", "analysis", "data"]

# 从注册中心获取工具
tools = get_tools(categories=TOOL_CATEGORIES)
tools_by_name = get_tools_by_name()

logger.info(
    f"加载了 {len(tools)} 个工具（categories={TOOL_CATEGORIES}）: "
    f"{[t.name for t in tools]}"
)

model = get_chat_model("supervisor")
model_with_tools = model.bind_tools(tools)


# ============================================================
# 节点
# ============================================================

def agent_node(state: CustomToolAgentState) -> Command[Literal["tool_executor"]]:
    """LLM 决策节点：从注册的工具中选择调用"""
    iteration = state.get("iteration", 0)
    messages = state.get("messages", [])

    # 自动生成工具列表（从 registry 中获取 doc）
    categories = get_tools_by_category()
    tool_desc_parts = []
    for cat, names in categories.items():
        if cat not in TOOL_CATEGORIES:
            continue
        tool_desc_parts.append(f"\n【{cat}】")
        for name in names:
            t = tools_by_name.get(name)
            if t and t.__doc__:
                desc = t.__doc__.strip().split("\n")[0]
                tool_desc_parts.append(f"  {name} - {desc}")

    system_text = f"""你是一个研究助手，今天的日期是 {get_today_str()}。

可用工具：{"".join(tool_desc_parts)}

规则：
- 需要信息时调用搜索类工具
- 需要梳理分析时调用分析类工具
- 每一步之前先反思
- 完成后不需要调用任何工具即可结束

最多 {MAX_ITERATIONS} 次迭代。"""

    response = model_with_tools.invoke([SystemMessage(content=system_text)] + list(messages))

    return Command(
        goto="tool_executor",
        update={"messages": [response], "iteration": iteration + 1},
    )


def tool_executor(
    state: CustomToolAgentState,
) -> Command[Literal["agent", "__end__"]]:
    """执行节点：通过 registry 按 name 路由到正确的工具"""
    last_msg = state["messages"][-1]
    tool_calls = last_msg.tool_calls or []
    updates: dict = {}

    if not tool_calls:
        return Command(goto="__end__")

    if state.get("iteration", 0) >= MAX_ITERATIONS:
        return Command(goto="__end__", update={"final_answer": "研究超时"})

    for tc in tool_calls:
        name = tc["name"]
        args = tc["args"]
        tool_obj = tools_by_name.get(name)

        if tool_obj is None:
            logger.warning(f"未知工具: {name}")
            updates.setdefault("messages", []).append(
                ToolMessage(content=f"错误：没有注册名为 '{name}' 的工具", tool_call_id=tc["id"])
            )
            continue

        try:
            # 函数式 tool：直接调用
            if hasattr(tool_obj, "invoke"):
                result = tool_obj.invoke(args)
            else:
                result = tool_obj(**args)

            logger.info(f"工具 '{name}' 执行成功")
            updates.setdefault("messages", []).append(
                ToolMessage(content=str(result), name=name, tool_call_id=tc["id"])
            )

        except Exception as e:
            logger.error(f"工具 '{name}' 执行失败: {e}")
            updates.setdefault("messages", []).append(
                ToolMessage(content=f"[错误] {e}", tool_call_id=tc["id"])
            )

    return Command(goto="agent", update=updates)


def get_tools_by_category():
    from deep_research.tools.registry import get_categories
    return get_categories()


# ============================================================
# Graph 构建
# ============================================================

def build_agent():
    builder = StateGraph(CustomToolAgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tool_executor", tool_executor)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "tool_executor",
        lambda s: "agent" if not s.get("final_answer") else "__end__",
        {"agent": "agent", "__end__": "__end__"},
    )

    return builder.compile()


custom_tool_agent = build_agent()


# ============================================================
# 测试运行（仅用于演示结构）
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("自定义 Tool 注册中心演示")
    print("=" * 60)

    from deep_research.tools.registry import list_tools
    list_tools()

    print()

    # 打印当前绑定的工具
    print(f"当前 agent 绑定的工具（categories={TOOL_CATEGORIES}）：")
    for t in tools:
        print(f"  - {t.name}")
    print()

    # 打印图结构
    print("Agent 图结构：")
    print(custom_tool_agent.get_graph().draw_ascii())
