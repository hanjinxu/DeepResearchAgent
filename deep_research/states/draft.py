#***********************************************
#      Filename: state_draft.py
#   Description: 报告草稿的结构化字段定义 
#***********************************************

"""用于draft State格式定义。
这定义了用于State对象和结构化模式，包括状态管理和输入输出格式。
"""

import operator
from typing_extensions import Optional, Annotated, List, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import MessagesState
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


# ===== STATE DEFINITIONS =====

class AgentInputState(MessagesState):
    """Agent的输入状态 - 仅包含来自用户输入的消息"""
    pass

class AgentState(MessagesState):
    """多Agent深度研究系统的主状态。
    扩展 MessagesState，添加用于研究协调的附加字段。
    注意：为了正确定义状态，某些字段在不同的状态类中重复出现。子图与主工作流之间的管理。
    """

    research_brief: Optional[str]                                           # 根据用户对话历史生成的研究简报
    supervisor_messages: Annotated[Sequence[BaseMessage], add_messages]     # 与Supervisor Agent交换的协调消息
    raw_notes: Annotated[list[str], operator.add] = []                      # 研究阶段收集的原始未处理研究笔记
    notes: Annotated[list[str], operator.add] = []                          # 已处理和结构化的笔记，可用于生成报告
    draft_report: str                                                       # 研究报告草稿
    final_report: str                                                       # 最终格式化的研究报告

    # ===== 事件记忆层字段（由 event_recall / event_memory_commit 节点读写）=====
    event_id: str = ""                            # 事件唯一 ID（subject|issue），判重命中或新抽取
    is_repeat_event: bool = False                 # 是否为已建档事件的增量追踪（True 时不再重跑历史节点）
    event_increment_context: str = ""             # 注入 write_research_brief 的「增量追踪」指令块
    event_memory_block: str = ""                  # 注入 write_draft_report 的历史基线（1.2 时间线 / 2.2 传闻状态机）
    event_history: dict = {}                      # 命中历史事件时的归档快照（EventRecord.model_dump()）
    delta_updates: dict = {}                      # 本次增量进展（DeltaUpdate.model_dump()）
    skip_research_loop: bool = False              # True 时跳过 supervisor 主研究循环（增量 fast 模式）
    memory_meta: dict = {}                        # 记忆层运行元信息（命中层级/存储后端/delta 统计），供日志与看板使用


# ===== STRUCTURED OUTPUT SCHEMAS =====

class ResearchQuestion(BaseModel):
    """用于生成结构化研究简报的字段定义"""

    research_brief: str = Field(
        description="A research question that will be used to guide the research.",
    )

class DraftReport(BaseModel):
    """用于生成结构化草稿报告的字段定义"""

    draft_report: str = Field(
        description="A draft report that will be used to guide the research.",
    )
