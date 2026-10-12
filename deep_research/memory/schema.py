#***********************************************
#      Filename: schema.py
#   Description: 事件记忆层的数据结构定义
#                事件（Event）→ 时间线（Timeline）→ 传闻状态机（Rumor State Machine）
#***********************************************

"""
【设计说明】

记忆层以"事件"为最小单元，而不是以"报告"为单元。
event_id = f"{subject}|{issue}"，例如 "华为|内部推荐政策"。

一份报告 = 某个事件在某个 report_date 的快照。
同一事件的多次报告通过 event_id 归并成一条不断增长的记录：
  - content      : 最新一次报告全文（保留最近 N 份历史）
  - timeline     : 累积的事件时间线（去重后按时间排序）
  - rumors       : 传闻状态机（unconfirmed → confirmed / refuted）
  - delta_log    : 每次增量更新的审计日志（本次新增了什么）
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ===== 常量 =====

RumorStatus = Literal["unconfirmed", "confirmed", "refuted", "partially_confirmed", "unknown"]

RUMOR_STATUS_LABEL: Dict[str, str] = {
    "unconfirmed": "未确认",
    "confirmed": "已证实",
    "refuted": "已证伪",
    "partially_confirmed": "部分证实",
    "unknown": "状态不明",
}


# ===== 子结构 =====

class TimelineNode(BaseModel):
    """事件时间线上的一个节点。对应报告骨架的「1.2 关键时间线」表格行。"""

    time: str = Field(default="", description="时间点，尽量精确到天/小时，如 '2026-06-15' 或 '2026-06-15 14:00'")
    node: str = Field(default="", description="事件节点描述，一句话")
    source: str = Field(default="", description="来源 URL 或信源名称")
    is_new: bool = Field(default=False, description="是否为本次增量新增（新时间点）")
    first_seen: str = Field(default="", description="该节点首次被观测到的日期（用于追踪）")

    def dedup_key(self) -> str:
        """时间线去重键：时间 + 节点文本归一化后的指纹。"""
        return _norm_key(f"{self.time}::{self.node}")


class RumorState(BaseModel):
    """传闻状态机的一条记录。对应报告骨架的「2.2 未确认传闻 / 存疑信息」。"""

    rumor_id: str = Field(default="", description="传闻指纹，由文本归一化后哈希得到，跨报告稳定")
    text: str = Field(default="", description="传闻内容描述")
    status: RumorStatus = Field(default="unconfirmed", description="状态：未确认/已证实/已证伪/部分证实")
    confidence: str = Field(default="低", description="当前可信度：低 / 中 / 高")
    conflict_point: str = Field(default="", description="冲突点：与哪一方的说法矛盾 / 未找到原始出处")
    evidence: List[str] = Field(default_factory=list, description="支持该状态的证据（来源 URL 或引文）")
    first_seen: str = Field(default="", description="首次出现日期")
    updated_at: str = Field(default="", description="最近一次状态更新日期")
    history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="状态迁移历史，用于'辟谣追踪'：每项形如 {'date':..., 'from':..., 'to':..., 'reason':..., 'evidence':...}",
    )

    @staticmethod
    def make_id(text: str) -> str:
        return "rmr_" + hashlib.md5(_norm_key(text).encode("utf-8")).hexdigest()[:12]


class DeltaUpdate(BaseModel):
    """一次增量进展追踪的产出。"""

    generated_at: str = Field(default="", description="本次增量更新日期")
    queries: List[str] = Field(default_factory=list, description="本次实际执行的「时间+关键词」检索式")
    summary: str = Field(default="", description="本轮新进展的一句话摘要")
    timeline_updates: List[TimelineNode] = Field(default_factory=list, description="抽出的新时间线节点")
    rumor_updates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="传闻状态迁移：每项形如 {'rumor_id'|'text':..., 'new_status':..., 'evidence':[...], 'reason':...}",
    )
    raw_findings: str = Field(default="", description="供下游报告直接引用的增量检索原始摘要")


# ===== 主记录 =====

class EventRecord(BaseModel):
    """事件记忆层落库的完整文档。

    前 6 个字段是约定的最小 schema（与 ES mapping 一一对应）：
        {"event_id":"华为|内部推荐政策",
         "subject":"华为",
         "issue":"内部推荐政策",
         "report_date":"2024-06-15",
         "content":"……报告全文……",
         "embedding":[ ... ]}
    其余为事件追踪所需的扩展字段。
    """

    # ---- 约定最小 schema ----
    event_id: str = Field(description="事件唯一 ID，格式 subject|issue")
    subject: str = Field(default="", description="事件主体（公司/人名/机构）")
    issue: str = Field(default="", description="事件议题")
    report_date: str = Field(default="", description="最近一次报告日期 YYYY-MM-DD")
    content: str = Field(default="", description="报告全文")
    embedding: List[float] = Field(default_factory=list, description="content 的向量表示")

    # ---- 事件追踪扩展 ----
    timeline: List[TimelineNode] = Field(default_factory=list, description="累积时间线")
    rumors: List[RumorState] = Field(default_factory=list, description="传闻状态机")
    delta_log: List[Dict[str, Any]] = Field(default_factory=list, description="增量更新审计日志")
    aliases: List[str] = Field(default_factory=list, description="主体别名/简称，供 L1 精确匹配使用")
    first_seen: str = Field(default="", description="该事件首次进入记忆的日期")
    last_seen: str = Field(default="", description="最近一次被命中的日期")
    report_count: int = Field(default=0, description="累计报告次数（>1 说明做过增量追踪）")
    content_archive: List[Dict[str, str]] = Field(
        default_factory=list, description="历史报告归档，仅保留最近 N 份，每项 {'report_date':..., 'content':...}"
    )

    # ---- ES 元信息（仅落库时由 store 填充）----
    meta: Dict[str, Any] = Field(default_factory=dict, description="存储层元信息（如 embedding_provider/dim）")

    def history_digest(self, content_limit: int = 3000) -> str:
        """给 LLM 看的历史摘要：上次报告 + 时间线 + 传闻状态机。"""
        parts: List[str] = [
            f"【历史事件】{self.event_id}",
            f"【首次出现】{self.first_seen}　【上次报告】{self.report_date}　【累计报告次数】{self.report_count}",
        ]
        if self.timeline:
            parts.append("\n【已知时间线】")
            for n in sorted(self.timeline, key=lambda x: x.time or ""):
                parts.append(f"- {n.time}｜{n.node}（来源：{n.source or '未标注'}）")
        if self.rumors:
            parts.append("\n【已知传闻状态机】")
            for r in self.rumors:
                parts.append(
                    f"- [{RUMOR_STATUS_LABEL.get(r.status, r.status)}] {r.text}"
                    f"（可信度：{r.confidence}；冲突点：{r.conflict_point or '无'}；更新于 {r.updated_at}）"
                )
        if self.content:
            parts.append(f"\n【上次报告正文（截断 {content_limit} 字）】\n{self.content[:content_limit]}")
        return "\n".join(parts)


# ===== 工具函数 =====

_NOISE_TOKENS = (
    # 舆情体裁词
    "事件", "风波", "舆情", "争议", "热搜", "爆料", "传闻",
    # 增量/时效词（同一事件的不同阶段说法，不该影响 L1 精确匹配）
    "最新", "进展", "动态", "近况", "后续", "始末", "详情", "回应", "声明",
    # 提问/任务词
    "情况", "相关", "关于", "如何看待", "怎么样", "什么", "为何", "为什么",
    "请问", "一下", "帮我", "调研", "分析", "报告",
)

_PUNCT = "，。！？、；：""''（）【】《》…—～·,.!?;:\"'()[]{}<>/\\|@#$%^&*_+=~`-"

# 所有空白（含全角空格）统一去掉：中文场景下空格不承载语义，
# 但"华为 内推"和"华为内推"必须归一到同一个 key，否则指纹/匹配会漂移。
_WHITESPACE_RE = re.compile(r"[\s\u3000]+")


def _norm_key(text: str) -> str:
    """文本归一化：仅用于生成指纹/比较键，不用于展示。"""
    if not text:
        return ""
    s = _WHITESPACE_RE.sub("", str(text)).lower()
    for ch in _PUNCT:
        s = s.replace(ch, "")
    return s


def normalize_entity(text: str) -> str:
    """实体/议题归一化：去标点 + 去舆情噪词，供 L1 精确匹配使用。"""
    s = _norm_key(text)
    for tok in _NOISE_TOKENS:
        s = s.replace(_norm_key(tok), "")
    return s


def make_event_id(subject: str, issue: str) -> str:
    """event_id = subject|issue（保持原始书写，仅去首尾空白）。"""
    return f"{(subject or '').strip()}|{(issue or '').strip()}"


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")
