#***********************************************
#      Filename: delta_tracker.py
#   Description: 增量进展追踪（时间 + 关键词检索 → 传闻状态迁移 / 时间线新节点）
#***********************************************

"""
【这一层到底在省什么】

命中历史事件后，**不再重跑完整研究流程**（不重新拆子 Query、不重新做全量搜索、
不重新去噪与冲突检测），只做一件事：

    用「时间 + 关键词」检索"这件事最新发生了什么"，然后把结果归位到两个槽：

        → 能证实/证伪某条传闻     ⇒ 更新 2.2 事实状态机（rumor status transition）
        → 带新时间点的事件进展     ⇒ 抽成 1.2 时间线的新节点

这就是「省历史节点重搜」+「进展追踪 / 辟谣追踪」两个产品能力的实现机制。

【为什么关键词是这 11 个】

    回应 声明 辟谣 通报 处罚 整改 起诉 反转 否认 致歉 热搜

舆情事件的"实质进展"基本只发生在这几类动作上。用它们做检索式，相当于给
"最新进展"这个模糊概念装了一个**领域词典化的锚点**，比单搜"最新消息"命中率高得多。

【成本控制】

增量检索**不走** tools.tool 里那套「网页全文 LLM 摘要」管线（每条 URL 一次 LLM 调用），
而是直接取 Tavily 返回的标题 + 摘要片段（content，已截断）。
理由：增量追踪只需要判断"有没有新事实"，不需要长文摘要；这样 6 条检索式 ≈ 0 次摘要 LLM 调用，
只有最后 1 次 delta 抽取调用。这是本层相对全量研究真正的成本节省点。
"""

from __future__ import annotations

import difflib
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

from deep_research import logging as dr_logging
from deep_research.memory.json_utils import safe_extract_json_object
from deep_research.memory.schema import (
    DeltaUpdate,
    EventRecord,
    RumorState,
    TimelineNode,
    normalize_entity,
    today_str,
)
from deep_research.prompts.event_memory import DELTA_EXTRACT_PROMPT, DELTA_TRACK_KEYWORDS

logger = dr_logging.get_logger(__name__)

DEFAULT_LOOKBACK_DAYS = 7
DEFAULT_MAX_QUERIES = 6


# ===== 1. 检索式构造 =====

def build_delta_queries(
    subject: str,
    issue: str,
    keywords: Optional[Sequence[str]] = None,
    *,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_queries: int = DEFAULT_MAX_QUERIES,
    keywords_per_query: int = 3,
    now: Optional[datetime] = None,
) -> List[str]:
    """生成「时间 + 关键词」检索式。

    结构：
      1) 时间锚定式：`主体 议题 最新进展 2026年10月` / `主体 议题 2026年10月`
      2) 动作关键词式：把 11 个关键词分成若干组，每组一条检索式
         （分组而非逐词，是为了在有限 query 预算内覆盖更多动作类型）
    """
    now = now or datetime.now()
    kws = list(keywords or DELTA_TRACK_KEYWORDS)
    base = f"{subject} {issue}".strip()
    if not base:
        return []

    # 时间锚定：覆盖近 lookback_days 的月份窗口
    start = now - timedelta(days=max(lookback_days, 1))
    anchors = {f"{now.year}年{now.month}月", f"{start.year}年{start.month}月"}

    queries: List[str] = [f"{base} 最新进展 {now.year}年{now.month}月"]
    for a in sorted(anchors, reverse=True):
        q = f"{base} {a}"
        if q not in queries:
            queries.append(q)

    # 关键词分组：固定每组 keywords_per_query 个，保证 11 个关键词在 max_queries 预算内被完整覆盖
    group_size = max(1, int(keywords_per_query))
    for i in range(0, len(kws), group_size):
        group = kws[i:i + group_size]
        if group:
            queries.append(f"{base} " + " ".join(group))

    # 去重保序 + 截断
    seen, out = set(), []
    for q in queries:
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out[:max_queries]


# ===== 2. 增量检索 =====

def search_delta(
    queries: Sequence[str],
    *,
    max_results: int = 3,
    topic: str = "news",
) -> List[Dict[str, Any]]:
    """执行增量检索，返回扁平的 [{query, title, url, content, published_date}]，按 url 去重。

    单条检索式失败不影响整体（增量追踪是"锦上添花"，不该让整轮研究挂掉）。
    """
    if not queries:
        return []

    try:
        from deep_research.tools.tool import _resolve_search_runtime

        provider, client, defaults = _resolve_search_runtime()
    except Exception as exc:
        logger.warning("[DELTA] 搜索运行时初始化失败: %s", str(exc)[:200])
        return []

    if provider is None or client is None or defaults is None:
        logger.warning("[DELTA] 搜索运行时不可用，跳过增量检索")
        return []

    timeout_seconds = defaults.get("timeout_seconds")
    docs: List[Dict[str, Any]] = []
    for q in queries:
        try:
            resp = provider.search(
                client,
                q,
                max_results=max_results,
                include_raw_content=False,  # 增量追踪不需要全文，省流量也省后续摘要成本
                topic=topic,
                timeout_seconds=timeout_seconds,
            )
        except Exception as exc:
            logger.warning("[DELTA] 检索式失败（跳过）: query='%s' err=%s", q, str(exc)[:160])
            continue

        for item in (resp or {}).get("results", []) or []:
            url = item.get("url", "")
            if not url:
                continue
            docs.append(
                {
                    "query": q,
                    "title": (item.get("title") or "").strip(),
                    "url": url,
                    "content": (item.get("content") or "").strip()[:800],
                    "published_date": item.get("published_date") or "",
                }
            )

    # 按 url 去重，保留更长的 content（信息量更大）
    best: Dict[str, Dict[str, Any]] = {}
    for d in docs:
        cur = best.get(d["url"])
        if cur is None or len(d["content"]) > len(cur["content"]):
            best[d["url"]] = d

    result = list(best.values())
    logger.info("[DELTA] 增量检索完成: %d 条检索式 → %d 条去重结果", len(queries), len(result))
    return result


def format_delta_findings(docs: Sequence[Dict[str, Any]], per_doc_limit: int = 600) -> str:
    """把检索结果格式化成给 LLM 的 findings 块。"""
    if not docs:
        return "（本次增量检索未返回任何结果）"
    parts: List[str] = []
    for i, d in enumerate(docs, 1):
        date_hint = f"（发布时间：{d['published_date']}）" if d.get("published_date") else ""
        parts.append(
            f"--- 结果 {i} {date_hint}\n"
            f"标题：{d.get('title', '')}\n"
            f"URL：{d.get('url', '')}\n"
            f"摘要：{(d.get('content') or '')[:per_doc_limit]}\n"
            f"（检索式：{d.get('query', '')}）"
        )
    return "\n\n".join(parts)


# ===== 3. delta 抽取 =====

async def extract_delta(
    history_text: str,
    findings_text: str,
    model,
    *,
    queries: Optional[Sequence[str]] = None,
) -> DeltaUpdate:
    """调用 LLM 抽取增量：传闻状态迁移 + 新时间线节点。"""
    prompt = DELTA_EXTRACT_PROMPT.format(
        history=history_text or "（无历史基线，本次为首次建档，所有信息都算新增）",
        findings=findings_text or "（无检索结果）",
        date=today_str(),
    )

    update = DeltaUpdate(generated_at=today_str(), queries=list(queries or []))
    try:
        from langchain_core.messages import HumanMessage

        resp = await model.ainvoke([HumanMessage(content=prompt)])
        data = safe_extract_json_object(getattr(resp, "content", "") or "")
    except Exception as exc:
        logger.warning("[DELTA] delta 抽取失败: %s", str(exc)[:200])
        update.summary = f"delta 抽取失败：{str(exc)[:80]}"
        return update

    update.summary = str(data.get("summary", "") or "").strip()

    for node in data.get("timeline_updates") or []:
        if not isinstance(node, dict):
            continue
        t = str(node.get("time", "")).strip()
        n = str(node.get("node", "")).strip()
        if not t or not n:
            continue
        update.timeline_updates.append(
            TimelineNode(time=t, node=n, source=str(node.get("source", "")).strip(), is_new=True)
        )

    for u in data.get("rumor_updates") or []:
        if not isinstance(u, dict):
            continue
        target = str(u.get("target", "")).strip()
        if not target:
            continue
        status = str(u.get("new_status", "unconfirmed")).strip() or "unconfirmed"
        if status not in ("unconfirmed", "confirmed", "refuted", "partially_confirmed"):
            status = "unconfirmed"
        update.rumor_updates.append(
            {
                "target": target,
                "new_status": status,
                "confidence": str(u.get("confidence", "")).strip() or "低",
                "evidence": [str(e) for e in (u.get("evidence") or []) if str(e).strip()],
                "reason": str(u.get("reason", "")).strip(),
            }
        )

    logger.info(
        "[DELTA] 抽取完成: summary=%s | timeline+%d | rumor_updates=%d",
        update.summary[:40] or "(空)",
        len(update.timeline_updates),
        len(update.rumor_updates),
    )
    return update


# ===== 4. delta 合并回事件记录（状态机迁移 + 时间线去重） =====

def _match_rumor(target: str, rumors: Sequence[RumorState], threshold: float = 0.6) -> Optional[RumorState]:
    """把 LLM 给出的 target 文本定位到已有传闻上（归一化 + 模糊匹配）。"""
    t_norm = normalize_entity(target)
    if not t_norm:
        return None

    best: Optional[RumorState] = None
    best_score = 0.0
    for r in rumors:
        r_norm = normalize_entity(r.text)
        if not r_norm:
            continue
        if t_norm in r_norm or r_norm in t_norm:
            score = 0.95
        else:
            score = difflib.SequenceMatcher(None, t_norm, r_norm).ratio()
        if score > best_score:
            best, best_score = r, score
    return best if best_score >= threshold else None


def apply_delta_to_record(record: EventRecord, delta: DeltaUpdate) -> EventRecord:
    """把 delta 合并进事件记录：时间线去重追加 + 传闻状态机迁移。返回同一个（被就地修改的）record。"""
    today = today_str()

    # ---- 时间线：按 (time, node) 指纹去重后追加 ----
    existing_keys = {n.dedup_key() for n in record.timeline}
    added = 0
    for node in delta.timeline_updates:
        key = node.dedup_key()
        if key in existing_keys:
            continue
        node.is_new = True
        node.first_seen = node.first_seen or today
        record.timeline.append(node)
        existing_keys.add(key)
        added += 1
    record.timeline.sort(key=lambda n: n.time or "")

    # ---- 传闻状态机：定位 → 迁移 → 记录迁移历史 ----
    transitions = 0
    for u in delta.rumor_updates:
        target = u.get("target", "")
        matched = _match_rumor(target, record.rumors)
        if matched is None:
            matched = RumorState(
                rumor_id=RumorState.make_id(target),
                text=target,
                status="unconfirmed",
                first_seen=today,
            )
            record.rumors.append(matched)

        old_status = matched.status
        new_status = u.get("new_status", "unconfirmed")
        evidence = u.get("evidence") or []

        if old_status != new_status:
            matched.history.append(
                {
                    "date": today,
                    "from": old_status,
                    "to": new_status,
                    "reason": u.get("reason", ""),
                    "evidence": " ; ".join(evidence)[:500],
                }
            )
            matched.status = new_status
            transitions += 1

        if u.get("confidence"):
            matched.confidence = u["confidence"]
        if evidence:
            matched.evidence = list(dict.fromkeys([*matched.evidence, *evidence]))
        matched.updated_at = today

    record.delta_log.append(
        {
            "date": today,
            "summary": delta.summary,
            "queries": list(delta.queries),
            "timeline_added": added,
            "rumor_transitions": transitions,
            "source_count": len(delta.timeline_updates) + len(delta.rumor_updates),
        }
    )
    logger.info("[DELTA] 合并完成: timeline+%d, rumor_transitions=%d", added, transitions)
    return record
