#***********************************************
#      Filename: interview_demo.py
#   Description: 面试演示 — 自定义Tool + Registry + LangGraph Agent
#***********************************************

"""
LangGraph 面试演示 Agent

覆盖点：
  1. 自定义工具注册 — @register_tool + @tool 装饰器
  2. Tool Registry 模式 — 可插拔、按 category 分组、按需获取
  3. 四种 Tool 模式 — 函数式(search) / Schema式(delegate) / 伪工具(think) / 混合
  4. 自定义 Prompt — SystemMessage 动态组装
  5. LangGraph 图结构 — agent → tool_node 循环 + 条件路由
  6. State 管理 — messages + 自定义字段

直接运行：python interview_demo.py （不需要 pip install）
"""

import json
from typing import Callable, Dict, List, Optional, Any, Sequence
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# 1. Tool Registry — 注册中心
# ============================================================

class Tool:
    """工具包装器，模拟 LangChain @tool"""
    def __init__(self, fn: Callable, name: str, description: str, category: str):
        self.fn = fn
        self.name = name
        self.description = description
        self.category = category

    def __call__(self, **kwargs) -> str:
        return self.fn(**kwargs)

    def __repr__(self):
        return f"Tool({self.name}, cat={self.category})"


class ToolRegistry:
    """
    Registry 模式：可插拔的工具注册中心。
    
    设计要点：
      1. 外部模块只需 @registry.register() 即可注册
      2. 按 category 分组，支持按角色绑定不同工具集
      3. 不修改核心代码
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._categories: Dict[str, List[str]] = {}

    def register(self, category: str = "uncategorized"):
        """装饰器：注册一个工具"""
        def decorator(fn):
            name = fn.__name__
            description = (fn.__doc__ or "").strip().split("\n")[0]
            tool = Tool(fn=fn, name=name, description=description, category=category)
            self._tools[name] = tool
            self._categories.setdefault(category, []).append(name)
            return fn
        return decorator

    def get_tools(self, categories: Optional[List[str]] = None) -> List[Tool]:
        """按分类获取工具列表"""
        if categories is None:
            return list(self._tools.values())
        names = set()
        for cat in categories:
            names.update(self._categories.get(cat, []))
        return [self._tools[n] for n in names if n in self._tools]

    def get_tools_by_name(self) -> Dict[str, Tool]:
        """name → Tool 映射，用于 tool_node 路由"""
        return dict(self._tools)

    def list_tools(self):
        """打印所有注册的工具"""
        for cat, names in self._categories.items():
            print(f"  [{cat}]")
            for name in names:
                t = self._tools[name]
                print(f"    {name:25s} {t.description}")
        print(f"\n  共 {len(self._tools)} 个工具")


# 全局单例
registry = ToolRegistry()


# ============================================================
# 2. 自定义工具定义（四种模式）
# ============================================================

# ——— 模式一：函数式 Tool（真实执行外部操作） ———

@registry.register(category="search")
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    return f"[搜索结果] 关于「{query}」找到 5 条结果：\n  1. ...\n  2. ..."


@registry.register(category="search")
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f"[天气] {city}：晴，22-28°C，空气质量优"


# ——— 模式二：Schema 式 Tool（定义参数 schema，触发子流程） ———

@registry.register(category="delegate")
def delegate_task(task: str, focus: List[str] = None) -> str:
    """
    委派子任务给专门的研究员
    参数：task=任务描述, focus=关注的维度列表
    """
    focus_text = ", ".join(focus or [])
    return f"[委派] 已启动子任务: {task[:50]}... 关注: {focus_text}"


@registry.register(category="system")
def task_complete(summary: str) -> str:
    """
    标记任务完成，输出最终总结
    参数：summary=最终总结内容
    """
    return f"[完成] 任务结束。总结: {summary[:100]}"


# ——— 模式三：伪工具（只记录状态，无外部调用） ———

@registry.register(category="reasoning")
def think(thought: str, next_step: str) -> str:
    """
    停下来反思当前进展，规划下一步
    参数：thought=当前思考, next_step=下一步计划
    """
    return f"[思考] {thought} → 下一步: {next_step}"


@registry.register(category="reasoning")
def quality_check(gaps: List[str] = None) -> str:
    """
    检查当前信息是否完整
    参数：gaps=缺失的信息列表
    """
    if gaps:
        return f"[质量检查] 缺失 {len(gaps)} 项信息: " + "; ".join(gaps)
    return "[质量检查] 信息已完整"


# ============================================================
# 3. LangGraph Agent
# ============================================================

@dataclass
class Message:
    """模拟 LangChain 的 HumanMessage / AIMessage / ToolMessage"""
    role: str  # "human" | "ai" | "tool"
    content: str
    tool_calls: List[Dict] = field(default_factory=list)
    name: str = ""
    tool_call_id: str = ""


@dataclass
class State:
    """Agent 状态，模拟 TypedDict + Annotated"""
    messages: List[Message] = field(default_factory=list)
    iteration: int = 0
    final_answer: str = ""
    think_history: List[str] = field(default_factory=list)


class Agent:
    """
    LangGraph Agent 实现。
    
    图结构：
      START → agent → tool_node → agent → tool_node → ... → END
                      ↑                            │
                      └────────────────────────────┘
    
    agent 节点：LLM 决策 → 选择调用哪个工具
    tool_node 节点：按 name 路由到注册的工具执行
    """

    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.tools = registry.get_tools()
        self.tools_by_name = registry.get_tools_by_name()
        self.max_iterations = 10

    def build_prompt(self, state: State) -> str:
        """自定义 Prompt 构建函数 — 展示 prompt 工程能力"""
        # 用注册表的信息动态组装 prompt
        tool_desc_parts = []
        for cat, names in self.registry._categories.items():
            tool_desc_parts.append(f"\n[{cat}]")
            for name in names:
                t = self.registry._tools[name]
                tool_desc_parts.append(f"  {name} - {t.description}")
        tool_text = "\n".join(tool_desc_parts)

        history = "\n".join(
            f"  [{i}] {t}" for i, t in enumerate(state.think_history[-3:], 1)
        ) if state.think_history else "  无"

        return f"""
你是一个研究助手。日期：{datetime.now().strftime('%Y-%m-%d')}

可用工具：
{tool_text}

工作流程：
1. 先 think() 反思 → 规划下一步
2. 信息不足 → search_web() 或 get_weather()
3. 需要专项调研 → delegate_task()
4. 质量自查 → quality_check()
5. 确认完成 → task_complete()

已有思考记录：
{history}

最多 {self.max_iterations} 次迭代。请直接输出你要调用的工具名和参数。
"""

    def llm_choose_tool(self, state: State) -> List[Dict]:
        """
        模拟 LLM 决策：根据当前状态选择要调用的工具。
        真实项目中这行代码是 model_with_tools.invoke(messages)
        
        展示了 LLM 如何从注册的工具列表中选择。
        """
        prompt = self.build_prompt(state)
        last_msg = state.messages[-1].content if state.messages else ""

        # 模拟 LLM 的 tool_choice 逻辑
        # 真实情况：LLM 读 prompt + 消息历史 → 输出 tool_calls
        if state.iteration == 0:
            # 第一轮：think + search
            return [
                {"name": "think", "args": {"thought": "用户问了两个问题，需要分别处理", "next_step": "先查天气再搜趋势"}},
                {"name": "search_web", "args": {"query": "AI Agent 最新趋势 2025"}},
                {"name": "get_weather", "args": {"city": "北京"}},
            ]
        elif state.iteration == 1:
            # 第二轮：质量检查 + 委派子任务
            return [
                {"name": "quality_check", "args": {"gaps": ["缺少市场规模数据", "缺少具体案例"]}},
                {"name": "think", "args": {"thought": "信息基本充足，可以收尾", "next_step": "总结输出"}},
            ]
        elif state.iteration == 2:
            # 第三轮：完成
            return [
                {"name": "task_complete", "args": {"summary": "北京今日晴好，AI Agent 2025年趋势包括多模态、Agentic Workflow、端侧部署"}},
            ]
        else:
            return []

    def invoke(self, initial_query: str) -> State:
        """运行完整的 agent 流程"""
        state = State(messages=[Message(role="human", content=initial_query)])

        print("\n▶ 开始运行 Agent...\n")

        while state.iteration < self.max_iterations and not state.final_answer:
            state.iteration += 1
            print(f"─── 迭代 {state.iteration} ───")

            # === agent 节点：LLM 决策 ===
            print(f"  [agent] LLM 决策中...")
            tool_calls = self.llm_choose_tool(state)

            if not tool_calls:
                print("  [agent] 未调用任何工具 → 结束")
                break

            ai_msg = Message(
                role="ai",
                content=f"调用 {len(tool_calls)} 个工具",
                tool_calls=tool_calls,
            )
            state.messages.append(ai_msg)

            # === tool_node 节点：执行工具 ===
            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"] or {}
                tool_fn = self.tools_by_name.get(name)

                if tool_fn is None:
                    result = f"[错误] 未知工具: {name}"
                else:
                    # 四种模式统一在 tool_node 里按 name 路由执行
                    result = tool_fn(**args)

                print(f"  [tool_node] → {name}({args})")
                print(f"             ↳ {result[:80]}...")

                state.messages.append(Message(
                    role="tool", content=result, name=name
                ))

                # 伪工具：Think 的记录单独存到 think_history
                if name == "think":
                    state.think_history.append(
                        f"{args.get('thought', '')} → {args.get('next_step', '')}"
                    )

                # TaskComplete 设置结束标记
                if name == "task_complete":
                    state.final_answer = args.get("summary", "")

            print()

        return state


# ============================================================
# 4. 面试演示入口
# ============================================================

def demo():
    print("=" * 70)
    print("         LangGraph 面试演示 — 自定义 Tool + Registry + Agent")
    print("=" * 70)

    # 1. 展示注册表
    print("\n▶ 1. Tool Registry — 已注册的工具")
    print("-" * 50)
    registry.list_tools()
    print()

    # 2. 展示按角色获取不同工具集
    print("▶ 2. 按角色绑定不同工具集（Registry 的核心能力）")
    print("-" * 50)
    supervisor_tools = registry.get_tools(categories=["search", "reasoning", "delegate", "system"])
    researcher_tools = registry.get_tools(categories=["search", "reasoning"])
    print(f"   Supervisor 绑定: {[t.name for t in supervisor_tools]}")
    print(f"   Researcher 绑定: {[t.name for t in researcher_tools]}")
    print()

    # 3. 实例化 Agent 并运行
    print("▶ 3. LangGraph Agent 图结构")
    print("-" * 50)
    print("""
    ┌────────────────────────────────────────┐
    │              START                      │
    │                │                        │
    │                ▼                        │
    │          ┌──────────┐                   │
    │          │  agent   │  ← LLM 决策       │
    │          │  (思维)   │  选哪个工具        │
    │          └────┬─────┘                   │
    │               │                         │
    │        ┌──────┴──────┐                  │
    │        ▼              ▼                  │
    │  有 tool_calls    无 tool_calls          │
    │        │              │                  │
    │        ▼              ▼                  │
    │  ┌──────────┐      ┌─────┐              │
    │  │tool_node │      │ END │              │
    │  │(执行工具) │      └─────┘              │
    │  └────┬─────┘                            │
    │       │                                  │
    │       ▼                                  │
    │  继续研究？                               │
    │  ├── 是 → agent (循环)                    │
    │  └── 否 → END                            │
    └────────────────────────────────────────┘
    """)

    # 4. 运行
    agent = Agent(registry)
    state = agent.invoke("北京今天天气怎么样？顺便调研一下AI Agent的最新趋势")

    # 5. 结果分析
    print("▶ 4. 运行结果分析")
    print("-" * 50)
    print(f"   总迭代次数：{state.iteration}")
    print(f"   消息总数：{len(state.messages)}")
    print(f"   Think记录：{len(state.think_history)} 条")
    print(f"   最终答案：{state.final_answer}")
    print()

    # 6. 统计工具调用
    print("▶ 5. 工具调用统计")
    print("-" * 50)
    counts = {}
    for msg in state.messages:
        for tc in msg.tool_calls:
            name = tc["name"]
            counts[name] = counts.get(name, 0) + 1
    for name, cnt in counts.items():
        t = registry._tools.get(name)
        mode = ""
        if name in ["search_web", "get_weather"]:
            mode = " [函数式]"
        elif name in ["delegate_task", "task_complete"]:
            mode = " [Schema式]"
        elif name in ["think", "quality_check"]:
            mode = " [伪工具]"
        print(f"   {name:20s} ×{cnt}{mode}")
    print()

    # 面试覆盖点清单
    print("=" * 70)
    print("面试覆盖点清单")
    print("=" * 70)
    checks = [
        ("Tool Registry 模式", "register_tool → 注册中心 → 按需获取"),
        ("四种 Tool 模式", "函数式(search) + Schema式(delegate) + 伪工具(think) + 混合"),
        ("自定义 Prompt 构建", "build_prompt() 动态组装 SystemMessage"),
        ("LangGraph 图结构", "agent → tool_node 循环 + 条件路由"),
        ("State 管理", "messages + iteration + final_answer + think_history"),
        ("工具路由机制", "tool_node 内按 name 分流执行"),
        ("不修改核心代码", "外部模块 @registry.register() 即可扩展"),
    ]
    for title, desc in checks:
        print(f"  ✅ {title:25s} — {desc}")


if __name__ == "__main__":
    demo()
