#***********************************************
#      Filename: interview_agent.py
#   Description: 面试演示Agent — 自定义Tool + Registry + LangGraph
#***********************************************

"""
LangGraph 面试演示 Agent。

纯 Python 实现（不需要 pip install），展示与项目一致的设计风格。

覆盖面试点：
  1. Registry 模式 — @registry.register 注册工具
  2. 三种 Tool 模式 — 函数式 / Schema式 / 伪工具
  3. 按角色绑定 — supervisor 和 researcher 用不同工具集
  4. 自定义 Prompt — 动态组装 SystemMessage
  5. LangGraph 图 — agent → tool_node + 条件路由
  6. State 管理 — messages + 自定义字段
"""

from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime


# ============================================================
# 1. Tool Registry
# ============================================================

@dataclass
class Tool:
    """模拟 LangChain 的 @tool 对象"""
    name: str
    fn: Callable
    description: str
    roles: List[str]

    def invoke(self, args: dict) -> str:
        return self.fn(**args)


class Registry:
    """
    可插拔 Tool 注册中心。
    与项目 search_factory.py 的 _PROVIDER_REGISTRY 设计一致。
    """
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._role_map: Dict[str, List[str]] = {}

    def register(self, roles: Optional[List[str]] = None):
        """装饰器：注册工具"""
        def decorator(fn):
            name = fn.__name__
            desc = (fn.__doc__ or "").strip().split("\n")[0]
            self._tools[name] = Tool(name=name, fn=fn, description=desc, roles=roles or [])
            if roles:
                for role in roles:
                    self._role_map.setdefault(role, []).append(name)
            return fn
        return decorator

    def get_tools(self, role: Optional[str] = None) -> List[Tool]:
        """按角色获取工具列表"""
        if role is None:
            return list(self._tools.values())
        names = self._role_map.get(role, [])
        return [self._tools[n] for n in names if n in self._tools]

    def get_tools_by_name(self) -> Dict[str, Tool]:
        return dict(self._tools)

    def list_tools(self):
        print(f"共 {len(self._tools)} 个工具：\n")
        for role, names in self._role_map.items():
            print(f"  [{role}]")
            for name in names:
                t = self._tools[name]
                print(f"    {name:28s} {t.description[:60]}")
            print()


registry = Registry()


# ============================================================
# 2. 自定义工具（三种模式）
# ============================================================

# 模式一：函数式 Tool（真实执行外部操作）
@registry.register(roles=["supervisor", "researcher"])
def search_web(query: str) -> str:
    """搜索网络获取最新信息"""
    return f"[搜索结果] 关于「{query}」找到以下信息：\n  来源1：相关文章摘要\n  来源2：详细分析数据"


@registry.register(roles=["supervisor"])
def get_weather(city: str) -> str:
    """查询城市天气"""
    return f"[天气] {city}：晴 22°C"


# 模式二：Schema 式 Tool（参数多，触发子流程）
@registry.register(roles=["supervisor"])
def delegate_research(topic: str, focus: list) -> str:
    """委派子任务给专门的研究员"""
    return f"[子Agent报告] 主题：{topic[:50]}... 覆盖：{', '.join(focus)}"


@registry.register(roles=["supervisor", "researcher"])
def task_complete(summary: str) -> str:
    """标记任务完成"""
    return f"[完成] 研究结束"


# 模式三：伪 Tool（只记录状态，无外部调用）
@registry.register(roles=["supervisor", "researcher"])
def think(reflection: str, next_plan: str) -> str:
    """反思当前进展，规划下一步"""
    return f"[思考] {reflection[:50]}... → 计划: {next_plan[:50]}"


# ============================================================
# 3. State
# ============================================================

@dataclass
class Message:
    role: str
    content: str
    tool_calls: List[Dict] = field(default_factory=list)
    name: str = ""


@dataclass
class State:
    messages: List[Message] = field(default_factory=list)
    iteration: int = 0
    final_answer: str = ""
    reflection_history: List[str] = field(default_factory=list)
    draft_report: str = ""


# ============================================================
# 4. Prompt（动态组装）
# ============================================================

def build_prompt(state: State) -> str:
    """自定义 Prompt — 从 registry 获取工具列表 + 注入上下文"""
    tools_for_role = registry.get_tools(role="supervisor")
    tool_lines = [f"  {t.name:25s} {t.description}" for t in tools_for_role]
    tool_text = "\n".join(tool_lines)

    history = ""
    if state.reflection_history:
        history = "\n已有思考：\n" + "\n".join(
            f"  [{i}] {h[:70]}..."
            for i, h in enumerate(state.reflection_history[-2:], 1)
        )

    return f"""您是研究主管，日期：{datetime.now().strftime('%Y-%m-%d')}

<可用工具>
{tool_text}
</可用工具>

<工作流程>
1. think() — 反思当前进展，规划下一步
2. search_web() — 搜索信息  
3. delegate_research() — 委派专题研究
4. task_complete() — 完成后调用
</工作流程>

<提示>
先 think 再行动，信息足够就完成。最多 10 次迭代。
{history}
"""


# ============================================================
# 5. Agent — 模拟 LangGraph
# ============================================================

class Agent:
    """
    LangGraph Agent
    
    图结构：
      START → agent → tool_node → agent → ... → END
                      ↑ 条件路由 ──┘
    
    agent 节点：LLM 决策 → 选择工具
    tool_node 节点：从 registry 按 name 路由执行
    条件路由：tool_calls 为空 or final_answer 有值 → END
    """

    def __init__(self, registry: Registry):
        self.registry = registry
        self.tools_by_name = registry.get_tools_by_name()
        self.max_iter = 10

    def llm_decide(self, state: State) -> List[Dict]:
        """
        模拟 LLM 决策。
        
        真实项目中：
          response = model_with_tools.invoke([system_msg] + messages)
          return response.tool_calls
        """
        if state.iteration == 0:
            return [
                {"name": "think",
                 "args": {"reflection": "用户问了两个问题：天气 + AI趋势",
                          "next_plan": "分头搜索再汇总"}},
                {"name": "search_web",
                 "args": {"query": "AI Agent 2025 最新趋势"}},
                {"name": "get_weather",
                 "args": {"city": "北京"}},
            ]
        elif state.iteration == 1:
            return [
                {"name": "think",
                 "args": {"reflection": "信息已收集",
                          "next_plan": "收尾"}},
                {"name": "delegate_research",
                 "args": {"topic": "AI Agent企业落地案例",
                          "focus": ["落地场景", "效果评估"]}},
            ]
        elif state.iteration == 2:
            return [
                {"name": "task_complete",
                 "args": {"summary": "北京晴好22°C。AI趋势：多模态、Agentic Workflow、端侧部署"}},
            ]
        return []

    def invoke(self, query: str) -> State:
        """运行 Agent — 对应 LangGraph 的 graph.invoke()"""
        state = State(messages=[Message(role="human", content=query)])

        print("\n▸ Agent 开始运行...\n")

        while state.iteration < self.max_iter and not state.final_answer:
            state.iteration += 1
            print(f"  ─── 迭代 {state.iteration} ───")

            # agent 节点
            print(f"  [agent] LLM 选择工具...")
            tool_calls = self.llm_decide(state)
            if not tool_calls:
                print(f"  [agent] 无工具调用 → 结束\n")
                break

            state.messages.append(Message(
                role="ai", content=f"调用 {len(tool_calls)} 个工具",
                tool_calls=tool_calls
            ))

            # tool_node 节点
            for tc in tool_calls:
                name = tc["name"]
                args = tc["args"] or {}
                tool_fn = self.tools_by_name.get(name)

                if tool_fn is None:
                    result = f"[错误] 未知工具: {name}"
                else:
                    result = tool_fn.invoke(args)

                print(f"  [tool_node] → {name}(...)")
                print(f"              ↳ {result[:70]}...")

                state.messages.append(Message(
                    role="tool", content=result, name=name
                ))

                # 伪工具：think → reflection_history
                if name == "think":
                    state.reflection_history.append(
                        f"{args.get('reflection','')[:60]} → {args.get('next_plan','')[:60]}"
                    )

                # Schema式：delegate → draft
                if name == "delegate_research":
                    state.draft_report = f"子报告：{args.get('topic','')[:50]}..."

                # Schema式：task_complete → 结束
                if name == "task_complete":
                    state.final_answer = args.get("summary", "")
                    print(f"  [完成] 答案已生成")

            print()

        return state


# ============================================================
# 6. 演示入口
# ============================================================

def demo():
    print("=" * 66)
    print("  LangGraph Agent 面试演示")
    print("  (自定义 Tool + Registry + LangGraph)")
    print("=" * 66)

    # 1. 注册表
    print("\n▶ 1. Tool Registry — 已注册工具")
    print("-" * 60)
    registry.list_tools()

    # 2. 按角色分组
    print("▶ 2. 按角色绑定不同工具集")
    print("-" * 60)
    sup = [t.name for t in registry.get_tools(role="supervisor")]
    res = [t.name for t in registry.get_tools(role="researcher")]
    print(f"  supervisor：{sup}")
    print(f"  researcher：{res}")
    print()

    # 3. 图结构
    print("▶ 3. LangGraph 图结构")
    print("-" * 60)
    print("""
    START
      │
      ▼
  ┌─────────┐     ┌────────────┐
  │  agent   │────→│  tool_node │
  │ (LLM决策) │     │ (路由执行)  │
  └────┬────┘     └────┬───────┘
       │               │
       │         ┌─────┴──────┐
       │         ▼            ▼
       │     有tool_calls   无/final
       │         │            │
       └─────────┘            ▼
                            END
    """)

    # 4. Prompt
    print("▶ 4. 自定义 Prompt（动态组装）")
    print("-" * 60)
    init = State(messages=[Message(role="human", content="北京天气 + AI趋势")])
    print(build_prompt(init)[:500])
    print("...\n")

    # 5. 运行
    print("▶ 5. 运行 Agent")
    print("-" * 60)
    agent = Agent(registry)
    state = agent.invoke("北京今天天气怎么样？顺便调研一下AI Agent的最新趋势")

    # 6. 结果
    print("▸ 运行结果")
    print("-" * 60)
    print(f"  迭代次数：{state.iteration}")
    print(f"  消息总数：{len(state.messages)}")
    print(f"  反思记录：{len(state.reflection_history)} 条")
    print(f"  报告草稿：{'有' if state.draft_report else '无'}")
    print(f"  最终答案：{state.final_answer[:120] if state.final_answer else '无'}")
    print()

    # 7. 调用统计
    print("▸ 工具调用分析")
    print("-" * 60)
    counts = {}
    for msg in state.messages:
        for tc in msg.tool_calls:
            counts[tc["name"]] = counts.get(tc["name"], 0) + 1
    for name, cnt in counts.items():
        mode = " [函数式]" if name in ["search_web", "get_weather"] else (
            " [Schema式]" if name in ["delegate_research", "task_complete"] else " [伪工具]"
        )
        print(f"  {name:25s} ×{cnt}{mode}")
    print()

    # 8. 面试覆盖点
    print("=" * 66)
    print("  面试覆盖点")
    print("=" * 66)
    for p in [
        "Registry — @registry.register 注册外部工具，不修改核心代码",
        "三种 Tool 模式 — 函数式 / Schema式 / 伪工具 共存",
        "按角色分组 — supervisor 和 researcher 绑定不同工具集",
        "自定义 Prompt — build_prompt() 动态组装 + 上下文注入",
        "LangGraph 图 — agent → tool_node + 条件路由循环",
        "State 管理 — messages + iteration + final_answer + reflection_history",
        "条件路由 — tool_calls为空/final_answer有值 判断是否结束",
    ]:
        print(f"  ✅ {p}")


if __name__ == "__main__":
    demo()
