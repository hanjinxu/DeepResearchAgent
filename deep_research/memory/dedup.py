#***********************************************
#      Filename: dedup.py
#   Description: 热点事件三级判重（L1 精确匹配 → L2 向量召回 → L3 LLM 判定）
#***********************************************

"""
【为什么是三级，而不是直接上向量 / 直接上 LLM】

判重在这个场景里是一个"高精度优先"的问题：
错误合并（把两个不同事件当成一个）会把不相干的历史基线灌进当前报告上下文，
污染比"没合并"严重得多。所以设计成漏斗：

    L1 精确匹配（成本 0，精度最高）
      实体 + 议题归一化后完全一致 → 直接命中，不调模型。
      覆盖"每天重复触发同一个热点"这个主要场景（占绝大多数请求）。

    L2 向量召回 top5（成本 1 次向量计算，无需 LLM）
      字面不同但语义相近（换了说法/加了"最新回应"）。只负责**粗筛候选**，
      不做最终判定——召回错了由 L3 兜。

    L3 LLM 判定（成本 1 次 LLM 调用，只在前两级都没结论时才走）
      在候选里做最终的"同一事件"判定，输出 event_id + 置信度 + 理由。

三级都未命中 → 判定为新事件，走完整研究流程并新建档案。

⚠️ 设计取向：L3 在模棱两可时**倾向判否**（宁可新开一条，不要错误合并）。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deep_research import logging as dr_logging
from deep_research.memory.embedding import BaseEmbedder
from deep_research.memory.event_store import EventStore, recall_by_vector
from deep_research.memory.json_utils import safe_extract_json_object
from deep_research.memory.schema import (
    EventRecord,
    make_event_id,
    normalize_entity,
    today_str,
)
from deep_research.prompts.event_memory import (
    EVENT_DEDUP_JUDGE_PROMPT,
    EVENT_KEY_EXTRACT_PROMPT,
)

logger = dr_logging.get_logger(__name__)


# ===== 数据结构 =====

@dataclass
class EventKeys:
    """query 里抽出来的事件键。"""

    subject: str = ""
    issue: str = ""
    aliases: List[str] = field(default_factory=list)
    source: str = "llm"  # llm | heuristic

    @property
    def event_id(self) -> str:
        return make_event_id(self.subject, self.issue)


@dataclass
class DedupResult:
    """判重结论。"""

    is_repeat: bool = False
    level: str = "none"          # L1 | L2+L3 | L2 | none
    confidence: float = 0.0
    reason: str = ""
    keys: Optional[EventKeys] = None
    history: Optional[EventRecord] = None
    trace: List[str] = field(default_factory=list)

    def trace_text(self) -> str:
        return " → ".join(self.trace)


# ===== 工具 =====

def _query_text_of(state_or_query: Any) -> str:
    """从 state 或裸字符串中取出用户 query。"""
    if isinstance(state_or_query, str):
        return state_or_query
    for msg in reversed((state_or_query or {}).get("messages", []) or []):
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content if isinstance(msg.content, str) else ""
    return ""


def _heuristic_keys(query: str) -> EventKeys:
    """LLM 抽取失败时的兜底：用最朴素的切分给出 subject/issue。"""
    q = (query or "").strip()
    for sep in ("，", ",", "、", " ", "  "):
        if sep in q:
            head, _, tail = q.partition(sep)
            if head and tail:
                return EventKeys(subject=head.strip()[:20], issue=tail.strip()[:20], source="heuristic")
    return EventKeys(subject=q[:20], issue=q[:20], source="heuristic")


# ===== L0：事件键抽取 =====

async def extract_event_keys(query: str, model) -> EventKeys:
    """用一次 LLM 调用抽出 (subject, issue, aliases)。"""
    if not query:
        return EventKeys()
    prompt = EVENT_KEY_EXTRACT_PROMPT.format(query=query)
    try:
        from langchain_core.messages import HumanMessage

        resp = await model.ainvoke([HumanMessage(content=prompt)])
        data = safe_extract_json_object(getattr(resp, "content", "") or "")
        keys = EventKeys(
            subject=str(data.get("subject", "")).strip(),
            issue=str(data.get("issue", "")).strip(),
            aliases=[str(a).strip() for a in (data.get("aliases") or []) if str(a).strip()][:3],
            source="llm",
        )
        if keys.subject or keys.issue:
            logger.info("[DEDUP] 事件键抽取: subject=%s issue=%s aliases=%s", keys.subject, keys.issue, keys.aliases)
            return keys
        logger.warning("[DEDUP] 事件键抽取返回空，使用启发式兜底")
    except Exception as exc:
        logger.warning("[DEDUP] 事件键抽取失败（%s），使用启发式兜底", str(exc)[:200])
    return _heuristic_keys(query)


# ===== L1：实体 + 议题精确匹配 =====

def l1_exact_match(keys: EventKeys, records: Sequence[EventRecord]) -> Optional[EventRecord]:
    """归一化后比对 subject 与 issue，两者都一致才算命中。"""
    subj_n = normalize_entity(keys.subject)
    iss_n = normalize_entity(keys.issue)
    if not subj_n or not iss_n:
        return None

    alias_ns = {normalize_entity(a) for a in keys.aliases if normalize_entity(a)}

    for rec in records:
        rec_subj = normalize_entity(rec.subject)
        rec_iss = normalize_entity(rec.issue)
        if not rec_iss or not rec_subj:
            continue

        # 主体一致：完全相等 / 互为子串 / 命中别名
        subj_ok = (
            subj_n == rec_subj
            or subj_n in rec_subj
            or rec_subj in subj_n
            or rec_subj in alias_ns
            or bool(alias_ns & {rec_subj})
        )
        # 议题一致：完全相等，或双方都够长时互为子串（避免"政策"这种短词过度合并）
        iss_ok = iss_n == rec_iss or (
            len(iss_n) >= 4 and len(rec_iss) >= 4 and (iss_n in rec_iss or rec_iss in iss_n)
        )
        if subj_ok and iss_ok:
            return rec
    return None


# ===== L2：向量召回 =====

# ⚠️ 阈值标定（实测，勿凭直觉调大）
#
# hashing-ngram 与真 embedding 的余弦量纲完全不同，绝不能照搬"相似度 0.8 才算像"的经验：
#
#   关系                              实测余弦
#   ────────────────────────────────  ─────────
#   同事件 · 换说法                      0.35
#   同事件 · 长报告正文 / 摘要            0.18 ~ 0.27
#   同事件 · 短标题（字面重叠少）          0.10
#   相关但不同事件（同主体不同议题）        0.07
#   完全无关 / 议题不同                 -0.07 ~ -0.04
#
# 关键观察：**同事件的低分（0.10）与不同事件的高分（0.07）几乎贴在一起**。
# 结论 → L2 的分数只能用来「排序召回」，不能用来「判定」；判定必须交给 L3。
# 因此 floor 只承担"过滤明显无关/负相关"的职责，取值必须很低。
DEFAULT_FLOOR_HASHING = 0.05
DEFAULT_FLOOR_REAL_EMBEDDING = 0.55


def default_l2_floor(embedder: BaseEmbedder) -> float:
    """按 embedder 类型给出 L2 相似度下限的默认值。"""
    name = getattr(embedder, "name", "") or ""
    if name.startswith("openai-compatible") or "embedding" in name.lower():
        return DEFAULT_FLOOR_REAL_EMBEDDING
    return DEFAULT_FLOOR_HASHING


def l2_vector_recall(
    query: str,
    store: EventStore,
    embedder: BaseEmbedder,
    top_k: int = 5,
    floor: float = DEFAULT_FLOOR_HASHING,
    candidate_limit: int = 2000,
    since_days: int = 0,
) -> List[Tuple[EventRecord, float]]:
    """向量召回 top-k 候选（只粗筛，不做判定）。"""
    qvec = embedder.embed(query)
    since_date = None
    if since_days and since_days > 0:
        from datetime import datetime, timedelta

        since_date = (datetime.now() - timedelta(days=since_days)).strftime("%Y-%m-%d")

    hits = recall_by_vector(store, qvec, top_k=top_k, candidate_limit=candidate_limit, since_date=since_date)
    return [(rec, score) for rec, score in hits if score >= floor]


# ===== L3：LLM 判定 =====

async def l3_llm_judge(
    query: str,
    candidates: Sequence[Tuple[EventRecord, float]],
    model,
) -> Dict[str, Any]:
    """让 LLM 在候选里判定是否同一事件。返回 {is_same, matched_event_id, confidence, reason}。"""
    if not candidates:
        return {"is_same": False, "matched_event_id": "", "confidence": 0.0, "reason": "无候选"}

    blocks: List[str] = []
    for i, (rec, score) in enumerate(candidates, 1):
        excerpt = (rec.content or "").replace("\n", " ")[:400]
        blocks.append(
            f"[候选 {i}] event_id = {rec.event_id}\n"
            f"  subject = {rec.subject}｜issue = {rec.issue}\n"
            f"  上次报告日期 = {rec.report_date}｜累计报告次数 = {rec.report_count}\n"
            f"  最近报告摘要 = {excerpt}\n"
            f"  向量相似度 = {score:.3f}"
        )
    candidates_text = "\n\n".join(blocks)

    prompt = EVENT_DEDUP_JUDGE_PROMPT.format(candidates=candidates_text, query=query)
    try:
        from langchain_core.messages import HumanMessage

        resp = await model.ainvoke([HumanMessage(content=prompt)])
        data = safe_extract_json_object(getattr(resp, "content", "") or "")
        return {
            "is_same": bool(data.get("is_same", False)),
            "matched_event_id": str(data.get("matched_event_id", "") or "").strip(),
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "reason": str(data.get("reason", "") or "").strip(),
        }
    except Exception as exc:
        logger.warning("[DEDUP] L3 判定失败: %s", str(exc)[:200])
        return {"is_same": False, "matched_event_id": "", "confidence": 0.0, "reason": f"L3 失败: {str(exc)[:80]}"}


# ===== 编排 =====

async def resolve_event(
    query: str,
    store: EventStore,
    embedder: BaseEmbedder,
    model,
    *,
    l2_top_k: int = 5,
    l2_floor: Optional[float] = None,
    l3_enabled: bool = True,
    candidate_limit: int = 2000,
    since_days: int = 0,
) -> DedupResult:
    """三级判重主入口。

    Args:
        query: 当前热点事件 query
        store: 事件存储
        embedder: 向量化组件
        model: 用于抽取事件键与 L3 判定的 LLM
    Returns:
        DedupResult（is_repeat=True 时 history 为命中的历史记录）
    """
    result = DedupResult()

    # ---- L0：抽事件键 ----
    keys = await extract_event_keys(query, model)
    result.keys = keys
    result.trace.append(f"L0 抽取 subject='{keys.subject}' issue='{keys.issue}'（{keys.source}）")

    # ---- L1：精确匹配 ----
    try:
        records = store.load_recent(limit=candidate_limit)
    except Exception as exc:
        logger.warning("[DEDUP] 读取历史事件失败，按新事件处理: %s", str(exc)[:200])
        records = []

    if not records:
        result.level = "none"
        result.reason = "记忆库为空，判为新事件"
        result.trace.append("L1 跳过（记忆库为空）")
        return result

    hit = l1_exact_match(keys, records)
    if hit is not None:
        result.is_repeat = True
        result.level = "L1"
        result.confidence = 0.99
        result.history = hit
        result.reason = f"实体+议题归一化后精确命中 {hit.event_id}"
        result.trace.append(f"L1 ✅ 精确命中 {hit.event_id}")
        logger.info("[DEDUP] L1 命中: %s", hit.event_id)
        return result
    result.trace.append(f"L1 ✗ 未命中（比对 {len(records)} 条历史记录）")

    # ---- L2：向量召回 ----
    floor = l2_floor if l2_floor is not None else default_l2_floor(embedder)
    try:
        candidates = l2_vector_recall(
            query, store, embedder, top_k=l2_top_k, floor=floor,
            candidate_limit=candidate_limit, since_days=since_days,
        )
    except Exception as exc:
        logger.warning("[DEDUP] L2 向量召回失败: %s", str(exc)[:200])
        candidates = []

    if not candidates:
        result.level = "none"
        result.reason = "L2 无候选，判为新事件"
        result.trace.append("L2 ✗ 无候选（低于相似度下限）")
        return result

    top_score = candidates[0][1]
    result.trace.append(
        f"L2 ✅ 召回 {len(candidates)} 条候选（floor={floor:.3f}），"
        f"最高分 {top_score:.3f}（{candidates[0][0].event_id}）"
    )
    logger.info("[DEDUP] L2 召回 %d 条，top1=%.3f（floor=%.3f）", len(candidates), top_score, floor)

    # ---- L3：LLM 判定 ----
    if not l3_enabled:
        result.is_repeat = True
        result.level = "L2"
        result.confidence = top_score
        result.history = candidates[0][0]
        result.reason = "L3 关闭，直接采用 L2 top1"
        result.trace.append("L3 已关闭，直接采用 L2 top1")
        return result

    verdict = await l3_llm_judge(query, candidates, model)
    result.trace.append(
        f"L3 {'✅ 判定同一事件' if verdict['is_same'] else '✗ 判定不同事件'}"
        f"（confidence={verdict['confidence']:.2f}，{verdict['reason'][:60]}）"
    )

    if verdict["is_same"]:
        matched_id = verdict["matched_event_id"]
        history = next((rec for rec, _ in candidates if rec.event_id == matched_id), None) or candidates[0][0]
        result.is_repeat = True
        result.level = "L2+L3"
        result.confidence = verdict["confidence"]
        result.history = history
        result.reason = verdict["reason"]
        logger.info("[DEDUP] L2+L3 命中: %s", history.event_id)
    else:
        result.is_repeat = False
        result.level = "none"
        result.confidence = verdict["confidence"]
        result.reason = verdict["reason"] or "L3 判定为新事件"
        logger.info("[DEDUP] 判为新事件（L3 否决 L2 候选）")

    return result
