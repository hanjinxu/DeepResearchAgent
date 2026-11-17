# ***********************************************
#      Filename: supervisor_agent.py
#   Description: 监督智能体
# ***********************************************

"""用于协调多个Sub-Research-Agent的监督。该模块实现了一种监督者模式，其中：
1. Supervisor Agent协调研究活动并分配任务
2. 多个Sub-Research-Agent独立地处理特定的子主题
3. 结果汇总并压缩，用于最终报告
Supervisor Agent采用并行执行方式来提高效率，同时为每个研究主题保持独立的上下文窗口。
"""

import asyncio
import json
from typing_extensions import Literal
from pydantic import BaseModel, Field, ValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import (
    HumanMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
    filter_messages
)
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

from deep_research.llm import get_chat_model
from deep_research.prompts import CRITICAL_ADDRESS_PROMPT, MULTI_STEP_DENOISE_PROMPT
from deep_research.agents.research_agent import researcher_agent
from deep_research.agents.red_team_agent import red_team_node
from deep_research.agents.evaluator_agent import evaluate_draft_quality
from deep_research.states import (
    SupervisorState,
    ConductResearch,
    ResearchComplete,
    QualityMetric
)
from deep_research.utils import get_today_str
from deep_research.tools import _think_tool, _refine_draft_report_tool
from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)

# 确保async在jupyter环境的兼容性
try:
    import nest_asyncio

    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            nest_asyncio.apply()
    except ImportError:
        pass  # Not Jupyter
except ImportError:
    pass


async def _ainvoke_with_retry(coro_factory, attempts: int = 3):
    """对子 agent 调用做非法 tool call 自动重试。

    小模型（如 Qwen3-7B）偶发 tool call 的 arguments 不是合法 JSON dict，
    VLLM 原样透传导致 LangChain pydantic 校验抛 ValidationError。
    配合角色级 temperature>0（重采样）重试可大概率恢复。
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return await coro_factory()
        except ValidationError as e:
            last_error = e
            logger.warning(
                "[RETRY] invalid tool_call args on attempt %d/%d: %s",
                attempt,
                attempts,
                str(e)[:200],
            )
    raise last_error  # type: ignore[misc]


def get_notes_from_tool_calls(messages: list[BaseMessage]) -> list[str]:
    """从Supervisor agent消息历史记录中的 ToolMessage 对象提取Research Notes。
    当Supervisor通过 ConductResearch tools调用将研究任务委托给子代理时，
    每个Sub-Agent 都会返回其压缩的研究结果（以 ToolMessage 内容形式）。
    此函数提取所有此类 ToolMessage 内容，以得到合并后的最终的研究笔记。

    Args：
        messages：主管对话历史记录中的消息列表

    Return：
        从ToolMessage对象中提取的Research Notes字符串列表
    """
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]


# ===== CONFIGURATION =====

# Supervisor 的结构化决策输出（替代原生 tool calling，规避小模型非法 tool call JSON 问题）：
# vLLM 侧配合 JSON Schema 语法约束解码（xgrammar），token 层面保证输出合法。
class SupervisorDecision(BaseModel):
    """Supervisor 每轮的唯一决策：选择一个动作并给出对应参数。"""

    action: Literal["think_tool", "ConductResearch", "refine_draft_report", "ResearchComplete"] = Field(
        description="本轮要执行的动作，四选一"
    )
    reflection: str = Field(
        default="",
        description="当 action=think_tool 时，填写反思与规划内容；其他 action 时填空字符串",
    )
    research_topics: list[str] = Field(
        default_factory=list,
        description="当 action=ConductResearch 时，填写要委派的研究主题列表（可多个并行，最多 3 个）；其他 action 时填空列表",
    )


def _decision_to_message(decision: SupervisorDecision, iteration: int) -> AIMessage:
    """把结构化决策转换为带 tool_calls 的 AIMessage，供 supervisor_tools 节点按原逻辑执行。"""
    tool_calls = []
    if decision.action == "think_tool":
        tool_calls.append({
            "name": "think_tool",
            "args": {"reflection": decision.reflection},
            "id": f"call_think_{iteration}",
            "type": "tool_call",
        })
    elif decision.action == "ConductResearch":
        for idx, topic in enumerate(decision.research_topics):
            if not topic.strip():
                continue
            tool_calls.append({
                "name": "ConductResearch",
                "args": {"research_topic": topic},
                "id": f"call_research_{iteration}_{idx}",
                "type": "tool_call",
            })
    elif decision.action == "refine_draft_report":
        tool_calls.append({
            "name": "refine_draft_report",
            "args": {},
            "id": f"call_refine_{iteration}",
            "type": "tool_call",
        })
    elif decision.action == "ResearchComplete":
        tool_calls.append({
            "name": "ResearchComplete",
            "args": {},
            "id": f"call_complete_{iteration}",
            "type": "tool_call",
        })
    return AIMessage(content="", tool_calls=tool_calls)


supervisor_model = get_chat_model("supervisor")

# 用结构化输出替代 bind_tools，从根本上杜绝 "arguments 是字符串而非字典" 这类格式错误。
# 不同后端支持的约束方式不同，这里按能力从强到弱依次降级，运行时自动探测，切后端无需改代码：
#   1) json_schema      —— 本地 vLLM(xgrammar) 语法硬约束解码，7B 小模型格式最稳
#   2) function_calling —— DeepSeek 及绝大多数 OpenAI 兼容商业 API（DeepSeek 不支持 json_schema）
#   3) 纯文本 + 手动解析 —— 兜底：任何后端都能用（与 draft_agent 同一套做法）
_STRUCTURED_METHODS = ("json_schema", "function_calling")

_supervisor_decision_models = []
for _method in _STRUCTURED_METHODS:
    try:
        _supervisor_decision_models.append(
            (_method, supervisor_model.with_structured_output(SupervisorDecision, method=_method))
        )
    except Exception as _e:  # pragma: no cover - 个别后端连构造都不支持
        logger.warning("[SUPERVISOR] structured method '%s' unavailable: %s", _method, str(_e)[:200])

# 兼容历史引用（其它模块可能直接 import supervisor_decision_model）
supervisor_decision_model = _supervisor_decision_models[0][1] if _supervisor_decision_models else None

# 兜底用的 JSON Schema 说明，供纯文本模式拼进 prompt
_SUPERVISOR_JSON_INSTRUCTION = (
    "\n\n【输出格式要求】\n"
    "只输出一个 JSON 对象，不要输出任何解释、不要使用代码块标记。格式如下：\n"
    '{"action": "think_tool" | "ConductResearch" | "refine_draft_report" | "ResearchComplete", '
    '"reflection": "字符串，仅当 action=think_tool 时填写反思内容，否则为空字符串", '
    '"research_topics": ["字符串数组，仅当 action=ConductResearch 时填写研究主题，最多3个，否则为空数组"]}'
)


def _extract_json_object(text: str) -> dict:
    """从模型自由文本里抠出第一个合法 JSON 对象（兼容 ```json 代码块与前后废话）。"""
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        # 去掉语言标识，如 ```json
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    # strict=False：结构化输出里的多行字段会带裸控制字符，strict=True 会整体失败
    return json.loads(raw, strict=False)


async def _decide_with_fallback(messages) -> SupervisorDecision:
    """按后端能力依次尝试结构化输出；全失败则退化成「自由输出 + 手动解析 JSON」。"""
    last_error: Exception | None = None

    for method, model in _supervisor_decision_models:
        for attempt in range(1, 4):
            try:
                return await model.ainvoke(messages)
            except (ValidationError, OutputParserException) as e:
                # 输出不合法 → 同一方式重采样重试
                last_error = e
                logger.warning(
                    "[SUPERVISOR] invalid decision output (method=%s, attempt %d/3): %s",
                    method, attempt, str(e)[:300],
                )
            except Exception as e:
                # 后端不支持该约束方式（如 DeepSeek 拒绝 response_format=json_schema）→ 换下一种
                last_error = e
                logger.warning(
                    "[SUPERVISOR] backend rejected structured method=%s, fallback to next: %s",
                    method, str(e)[:300],
                )
                break

    # 兜底：纯文本输出 + 手动解析
    logger.warning("[SUPERVISOR] all structured methods failed, fallback to plain-text JSON parsing")
    fallback_messages = list(messages) + [SystemMessage(content=_SUPERVISOR_JSON_INSTRUCTION)]
    for attempt in range(1, 4):
        try:
            resp = await supervisor_model.ainvoke(fallback_messages)
            return SupervisorDecision.model_validate(_extract_json_object(resp.content))
        except Exception as e:
            last_error = e
            logger.warning("[SUPERVISOR] plain-text parse failed on attempt %d/3: %s", attempt, str(e)[:300])

    raise last_error  # type: ignore[misc]

# System constants (最大迭代次数/最大并行Sub-Agents)
max_researcher_iterations = 15  # Calls to think_tool + ConductResearch + refine_draft_report
max_concurrent_researchers = 3  # 最大并行子agent数
min_need_repair_score = 6.0  # 评估低于这个分数，就要出发agent修复提醒


# ===== SUPERVISOR NODES =====

async def supervisor(state: SupervisorState) -> Command[Literal["supervisor_tools"]]:
    """分析研究简报和当前进展
    功能：
        - 需要研究哪些主题
        - 是否开展并行研究
        - 研究何时完成

    Args：
        state：当前supervisor状态，包含messages和progress

    Returns：
        用于跳转到 supervisor_tools 节点并更新状态的命令
    """
    supervisor_messages = state.get("supervisor_messages", [])
    iteration = state.get("research_iterations", 0)
    logger.info("[SUPERVISOR] supervisor invoked (iteration=%d, messages=%d)", iteration, len(supervisor_messages))

    # 组装系统提示词
    system_message = MULTI_STEP_DENOISE_PROMPT.format(
        date=get_today_str(),
        max_concurrent_research_units=max_concurrent_researchers,
        max_researcher_iterations=max_researcher_iterations
    )

    messages = [SystemMessage(content=system_message)] + supervisor_messages

    # 动态上下文注入：检查并注入任何未处理的对抗性反馈，实现自我纠正机制。
    critiques = state.get("active_critiques", [])
    unaddressed = [c for c in critiques if not c.addressed]
    if unaddressed:
        critique_text = "\n".join([f"- {c.author} says: {c.concern}" for c in unaddressed])
        intervention = SystemMessage(content=CRITICAL_ADDRESS_PROMPT.format(critique_text=critique_text))
        messages.append(intervention)

    # 如果上一次迭代中质量得分较低，则会发出提醒
    if state.get("needs_quality_repair"):
        messages.append(SystemMessage(content="上一稿报告质量较低（得分低于7/10），请继续完善。"))

    print("=================supervisor messages====================", messages)

    # 决策本轮动作（结构化输出：自动探测后端能力 + 重试 + 手动解析兜底）
    decision = await _decide_with_fallback(messages)

    # 把结构化决策转换为带 tool_calls 的 AIMessage（supervisor_tools 节点按原逻辑执行）
    response = _decision_to_message(decision, iteration)

    print("=================supervisor decision====================", decision.model_dump())
    print("=================response===============:", response.model_dump())
    logger.info(
        "supervisor decision action=%s tool_calls=%d",
        decision.action,
        len(response.tool_calls or []),
    )

    # 跳转到supervisor_tools
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": iteration + 1,
            "needs_quality_repair": False  # 在向supervisor发出提醒后，重置修复标志
        }
    )


async def supervisor_tools(state: SupervisorState) -> Command[Literal["supervisor", "__end__"]]:
    """
    执行Supervisor决策——继续下一轮研究或者是结束流程。

    功能：
        - 执行 think_tool 调用以进行思考
        - 并行启动针对不同主题的research agent
        - 汇总研究结果
        - 确定研究何时完成

    参数：
        state：包含supervisor messages和迭代次数

    返回值：
        继续下一轮supervisor/结束流程
    """
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = supervisor_messages[-1]

    # 检查是否达到了最大迭代次数或者supervisor是否输出工具调用
    exceeded_iterations = research_iterations >= max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete"
        for tool_call in most_recent_message.tool_calls
    )

    # 如果超过则退出
    if exceeded_iterations or no_tool_calls or research_complete:
        # 如果满足退出条件，我们会准备最终的、经过整理的notes。
        # 优先使用结构化的知识库，但如果知识库为空，则使用raw notes。
        final_notes = get_notes_from_tool_calls(state.get("supervisor_messages", []))
        logger.info("[REPORT] The research is complete, writing the final report.")

        # 我们返回一个END来结束这个子图，并将最后的notes传递给supervisor。
        return Command(
            goto=END,
            update={
                "notes": final_notes,
                "research_brief": state.get("research_brief", "")
            })

    else:
        # 初始化变量
        tool_messages = []
        all_raw_notes = []
        draft_report = state.get("draft_report", "")
        updates = {}
        next_step = "supervisor"

        # 执行所有的工具调用
        try:
            think_tool_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "think_tool"
            ]

            conduct_research_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "ConductResearch"
            ]

            refine_report_calls = [
                tool_call for tool_call in most_recent_message.tool_calls
                if tool_call["name"] == "refine_draft_report"
            ]

            logger.info(
                "[SUPERVISOR] supervisor_tools executing think=%d conduct=%d refine=%d",
                len(think_tool_calls),
                len(conduct_research_calls),
                len(refine_report_calls),
            )

            # 调用 think 工具（在调用其他工具之前，必须拿到反思结果）(synchronous)
            for tool_call in think_tool_calls:
                observation = _think_tool.invoke(tool_call["args"])
                tool_messages.append(
                    ToolMessage(
                        content=observation,
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    )
                )

            # 调用 ConductResearch工具 (asynchronous)
            if conduct_research_calls:
                # 并行启动多个 research agents（带非法 tool call 自动重试）
                coros = [
                    _ainvoke_with_retry(
                        lambda tc=tool_call: researcher_agent.ainvoke({
                            "researcher_messages": [
                                HumanMessage(content=tc["args"]["research_topic"])
                            ],
                            "research_topic": tc["args"]["research_topic"],
                        })
                    )
                    for tool_call in conduct_research_calls
                ]

                # 等待所有research agents 返回研究结果
                tool_results = await asyncio.gather(*coros)


                # 将研究结果格式化为工具消息
                # 每个research agent都会在 result["compressed_research"] 中返回压缩后的研究结果
                # 我们将这些压缩后的研究结果写入 ToolMessage 的内容，以便
                # supervisor agent 可以通过 get_notes_from_tool_calls() 检索到这些结果
                research_tool_messages = [
                    ToolMessage(
                        content=result.get("compressed_research", "Error synthesizing research report"),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"]
                    ) for result, tool_call in zip(tool_results, conduct_research_calls)
                ]

                tool_messages.extend(research_tool_messages)

                # 聚合所有的raw notes
                all_raw_notes = [
                    "\n".join(result.get("raw_notes", []))
                    for result in tool_results
                ]

            # 开始调用大模型结合已有信息修正调研报告
            for tool_call in refine_report_calls:
                findings = "\n".join(get_notes_from_tool_calls(state.get("supervisor_messages", [])))

                new_draft = _refine_draft_report_tool.invoke({
                    "research_brief": state.get("research_brief", ""),
                    "findings": findings,
                    "draft_report": state.get("draft_report", "")
                })

                # 执行Critical Step：Self-Evolution的评估
                eval_result = evaluate_draft_quality(
                    research_brief=state.get("research_brief", ""),
                    draft_report=new_draft
                )
                logger.info(
                    "[EVALUATOR] comprehensive score=%f, accuracy score=%f, coherence score=%f",
                    eval_result.comprehensiveness_score,
                    eval_result.accuracy_score,
                    eval_result.coherence_score
                )
                logger.info(f"[EVALUATOR] scoing reason: {eval_result.reason}")

                # 评估报告质量得分：(综合得分+准确率得分+一致性得分) / 3
                avg_score = (
                                        eval_result.comprehensiveness_score + eval_result.accuracy_score + eval_result.coherence_score) / 3

                # 把质量得分追加到tool message, 供Supervisor Agent参考
                tool_messages.append(ToolMessage(
                    content=f"Draft Updated.\nQuality Score: {avg_score}/10.\nJudge Feedback: {eval_result.reason}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"]
                ))

                draft_report = new_draft
                updates["draft_report"] = draft_report

                # 记录报告质量评分的记录，如果分数低于 min_need_repair_score，把repaire标志位置位true
                updates["quality_history"] = [QualityMetric(
                    score=avg_score,
                    feedback=eval_result.reason,
                    iteration=state.get("research_iterations", 0))
                ]

                if avg_score < min_need_repair_score:
                    updates["needs_quality_repair"] = True

                # 跳转到self-correction节点 (Red Team)
                next_step = "red_team"

            # 更新本次迭代状态信息
            updates["supervisor_messages"] = tool_messages
            updates["raw_notes"] = all_raw_notes

            return Command(goto=next_step, update=updates)

        except Exception as e:
            return Command(
                goto=END,
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", "")
                }
            )


# ===== GRAPH CONSTRUCTION =====

supervisor_builder = StateGraph(SupervisorState)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_node("red_team", red_team_node)

supervisor_builder.add_edge(START, "supervisor")
supervisor_builder.add_edge("supervisor", "supervisor_tools")
supervisor_builder.add_edge("red_team", "supervisor")

supervisor_agent = supervisor_builder.compile()

if __name__ == "__main__":
    print(supervisor_agent.get_graph().draw_ascii())
