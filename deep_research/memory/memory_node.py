#***********************************************
#      Filename: memory_node.py
#   Description: 事件记忆层的 LangGraph 节点
#                event_recall（判重 + 增量追踪） / event_memory_commit（归档回写）
#***********************************************

"""
【两个节点在图里的位置】

    START
      → event_recall            ← 判重；命中历史事件则做增量追踪、算出"本次只需补什么"
      → write_research_brief    ← 增量模式下改写简报：不再"从零调研"，而是"补最新进展"
      → write_draft_report      ← 草稿的 1.2 时间线 / 2.2 传闻状态机注入历史基线
      → supervisor_subgraph     ← 主研究循环（搜索/去噪/红队/评分）
      → final_report_generation
      → event_memory_commit     ← 抽 delta 归档 + 传闻状态机迁移 + 写入 ES
      → END

【容错原则】

记忆层是"增值能力"，不是"必要链路"。两个节点**全部**包在 try/except 里：
任何异常（记忆库读不到、embedding 服务挂了、LLM 抽取超时）都只降级为"无历史"，绝不中断研究。
"""

import os
from typing import Any, Dict, Tuple

from typing_extensions import Literal
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from deep_research import logging as dr_logging
from deep_research.llm import get_chat_model
from deep_research.memory.dedup import DedupResult, default_l2_floor, resolve_event
from deep_research.memory.delta_tracker import (
    DEFAULT_LOOKBACK_DAYS,
    DEFAULT_MAX_QUERIES,
    apply_delta_to_record,
    build_delta_queries,
    extract_delta,
    format_delta_findings,
    search_delta,
)
from deep_research.memory.embedding import get_embedder
from deep_research.memory.event_store import get_event_store
from deep_research.memory.schema import (
    EventRecord,
    RUMOR_STATUS_LABEL,
    DeltaUpdate,
    make_event_id,
    today_str,
)
from deep_research.utils import load_config

logger = dr_logging.get_logger(__name__)

DEFAULT_STAGE = "prod"
_MEMORY_CFG_CACHE: Dict[str, Dict[str, Any]] = {}
_RUNTIME_CACHE: Dict[str, Tuple[Any, Any, Any]] = {}


# ===== 配置与运行时 =====

def _load_memory_cfg() -> Dict[str, Any]:
    """读取 config.yml 中 stage 下的 event_memory 段。"""
    path = os.environ.get("CONFIG_PATH", "config.yml")
    stage = os.environ.get("STAGE") or DEFAULT_STAGE
    cache_key = f"{path}::{stage}"
    if cache_key in _MEMORY_CFG_CACHE:
        return _MEMORY_CFG_CACHE[cache_key]

    try:
        cfg = load_config(stage_name=stage, config_path=path) or {}
        mem_cfg = cfg.get("event_memory") or {}
    except Exception as exc:
        logger.warning("[MEMORY] 加载 event_memory 配置失败，按默认值运行: %s", str(exc)[:200])
        mem_cfg = {}

    _MEMORY_CFG_CACHE[cache_key] = mem_cfg
    return mem_cfg


def _get_judge_model(mem_cfg: Dict[str, Any]):
    """判重/抽取用的 LLM。优先用配置里的 judge_role，不存在则回落到 draft 角色。"""
    role = mem_cfg.get("judge_role") or "event_judge"
    try:
        return get_chat_model(role)
    except Exception:
        try:
            return get_chat_model("draft")
        except Exception:
            return get_chat_model("writer")


def _get_runtime(mem_cfg: Dict[str, Any]):
    """懒加载 (store, embedder, judge_model)，全局复用避免重复建连接。"""
    key = str(id(mem_cfg))
    if key in _RUNTIME_CACHE:
        return _RUNTIME_CACHE[key]

    embedder = get_embedder(mem_cfg.get("embedding") or {})
    store = get_event_store(mem_cfg.get("store") or {}, dim=getattr(embedder, "dim", 0))
    model = _get_judge_model(mem_cfg)
    _RUNTIME_CACHE[key] = (store, embedder, model)
    return _RUNTIME_CACHE[key]


def _last_user_query(state: Dict[str, Any]) -> str:
    for msg in reversed(state.get("messages", []) or []):
        if msg.__class__.__name__ == "HumanMessage":
            return msg.content if isinstance(msg.content, str) else ""
    return ""


# ===== 上下文渲染 =====

def _render_increment_context(record: EventRecord, delta: DeltaUpdate) -> str:
    """给 write_research_brief 用的「增量追踪」指令块。"""
    known_timeline = "\n".join(
        f"  - {n.time}｜{n.node}" for n in sorted(record.timeline, key=lambda x: x.time or "")
    ) or "  （暂无）"
    known_rumors = "\n".join(
        f"  - [{RUMOR_STATUS_LABEL.get(r.status, r.status)}] {r.text}（可信度：{r.confidence}）"
        for r in record.rumors
    ) or "  （暂无）"

    return f"""# ⚠️ 本次为「历史事件增量追踪」，不是从零调研

该事件此前已建档：**{record.event_id}**
- 首次出现：{record.first_seen}　最近一次报告：{record.report_date}　累计报告次数：{record.report_count}
- 本次增量检索到的进展摘要：{delta.summary or '（未发现新进展）'}

## 已知时间线（不要重复调研这些节点）
{known_timeline}

## 已知传闻状态机（只需判断其中哪条被证实/证伪，或是否有全新传闻）
{known_rumors}

## 本次研究的唯一目标
**只补充"相比上述基线新增了什么"**，具体聚焦三类：
1. 上述已知传闻中，哪些被官方/权威来源**证实或证伪**了（辟谣追踪）；
2. 出现了哪些**带明确时间点的新进展**（进展追踪）；
3. 事件的**当前状态与后续走向**是否发生变化（法律风险等级、处置动作是否需要调整）。

## 明确禁止
- ❌ 不要重新梳理已在「已知时间线」里的事实（那是历史节点重搜，浪费预算）；
- ❌ 不要重新解释事件背景，除非背景本身发生了新变化。"""


def _render_increment_brief(record: EventRecord, delta: DeltaUpdate) -> str:
    """fast 模式直接在 event_recall 里产出的研究简报（跳过 write_research_brief）。"""
    known_timeline = "；".join(
        f"{n.time} {n.node}" for n in sorted(record.timeline, key=lambda x: x.time or "")
    ) or "（暂无）"
    known_rumors = "；".join(
        f"[{RUMOR_STATUS_LABEL.get(r.status, r.status)}] {r.text}" for r in record.rumors
    ) or "（暂无）"

    return (
        f"【增量追踪任务】事件档案 {record.event_id}"
        f"（首次建档 {record.first_seen}，上次报告 {record.report_date}，累计报告 {record.report_count} 次）。\n\n"
        f"该事件此前已完成完整调研，本次**只做增量更新**，不要从零重写。\n\n"
        f"已知时间线：{known_timeline}\n\n"
        f"已知传闻状态：{known_rumors}\n\n"
        f"本次「时间+关键词」增量检索到的新进展：{delta.summary or '（未发现新进展）'}\n\n"
        f"请基于以上材料，输出该事件报告的**更新版**，重点更新：\n"
        f"1) ①1.2 关键时间线 —— 只补充带明确时间点的新节点，历史节点照旧保留；\n"
        f"2) ②2.2 事实状态机 —— 更新被证实/证伪的传闻，保留历史状态并追加「迁移记录」；\n"
        f"3) ③④ 两节 —— 若事件状态发生变化，相应调整法律风险等级与法务处置动作。\n"
        f"若本次未发现新进展，如实说明「本周期无新增实质进展」，不要编造。"
    )


def _render_memory_block(record: EventRecord, delta: DeltaUpdate) -> str:
    """给 write_draft_report 用的「历史基线预填」块，对应骨架 1.2 与 2.2 两节。"""
    lines = [
        "# 📌 历史基线（用于草稿的 1.2 关键时间线与 2.2 事实状态机）",
        f"当前事件档案：{record.event_id}（首次建档 {record.first_seen}，累计报告 {record.report_count} 次）",
    ]
    if record.timeline or delta.timeline_updates:
        lines.append("\n## 1.2 关键时间线（历史已知 + 本次新增，标 🆕 的是本次新增）")
        for n in sorted(record.timeline, key=lambda x: x.time or ""):
            flag = "🆕 " if n.is_new else ""
            lines.append(f"| {n.time} | {flag}{n.node} | {n.source or '（历史来源）'} |")
    if record.rumors:
        lines.append("\n## 2.2 传闻状态机（请在此基础上更新，不要丢弃历史状态）")
        for r in record.rumors:
            lines.append(
                f"- [{RUMOR_STATUS_LABEL.get(r.status, r.status)}] {r.text}"
                f"（可信度：{r.confidence}；冲突点：{r.conflict_point or '无'}；最后更新：{r.updated_at}）"
            )
            for h in r.history[-3:]:
                lines.append(
                    f"    · 迁移记录 {h.get('date')}：{RUMOR_STATUS_LABEL.get(h.get('from',''), h.get('from',''))}"
                    f" → {RUMOR_STATUS_LABEL.get(h.get('to',''), h.get('to',''))}"
                    f"（{h.get('reason','')[:60]}）"
                )
        lines.append(
            "\n> ⚠️ 传闻状态机是**累积**的：历史状态必须保留，只在有新证据时迁移状态，并在「迁移记录」里追加一行。"
        )
    lines.append(
        f"\n## 本次增量进展（{delta.generated_at}）\n"
        f"{delta.summary or '（未发现新进展）'}\n\n"
        f"### 增量检索原始资料（可直接引用，需带 [n] 编号）\n{delta.raw_findings[:4000] or '（无）'}"
    )
    return "\n".join(lines)


# ===== 节点 1：事件召回（判重 + 增量追踪） =====

async def event_recall_node(
    state: Dict[str, Any], config: RunnableConfig | None = None
) -> Command[Literal["write_research_brief", "write_draft_report"]]:
    """判重 → 命中则做增量追踪 → 产出注入下游的上下文。

    两种增量模式（config.yml → event_memory.increment_mode）：
      - "full"（默认）：仍然走完整主研究循环，但简报收敛到「只补最新进展」；
      - "fast"：跳过整个 supervisor 主研究循环，用历史基线 + 增量资料直接生成报告。
    """
    empty_update: Dict[str, Any] = {
        "event_id": "",
        "is_repeat_event": False,
        "event_increment_context": "",
        "event_memory_block": "",
        "event_history": {},
        "delta_updates": {},
        "skip_research_loop": False,
        "memory_meta": {"enabled": False},
    }

    mem_cfg = _load_memory_cfg()
    if not mem_cfg.get("enabled", False):
        logger.info("[MEMORY] event_memory 未启用，跳过判重")
        empty_update["memory_meta"] = {"enabled": False, "reason": "disabled"}
        return Command(goto="write_research_brief", update=empty_update)

    query = _last_user_query(state)
    if not query:
        return Command(goto="write_research_brief", update=empty_update)

    dedup_cfg = mem_cfg.get("dedup") or {}
    mode = str(mem_cfg.get("increment_mode", "full")).strip().lower()
    if mode not in ("full", "fast"):
        logger.warning("[MEMORY] 未知 increment_mode='%s'，回退为 full", mode)
        mode = "full"
    try:
        store, embedder, model = _get_runtime(mem_cfg)

        # l2_floor 未显式配置时按 embedder 类型取默认值
        # （hashing-ngram 与真 embedding 的余弦量纲不同，见 dedup.py 的标定注释）
        raw_floor = dedup_cfg.get("l2_floor")
        l2_floor = float(raw_floor) if raw_floor is not None else default_l2_floor(embedder)

        result: DedupResult = await resolve_event(
            query,
            store,
            embedder,
            model,
            l2_top_k=int(dedup_cfg.get("l2_top_k", 5)),
            l2_floor=l2_floor,
            l3_enabled=bool(dedup_cfg.get("l3_enabled", True)),
            candidate_limit=int(dedup_cfg.get("candidate_limit", 2000)),
            since_days=int(dedup_cfg.get("since_days", 0)),
        )

        keys = result.keys
        base_update: Dict[str, Any] = {
            "event_id": (result.history.event_id if result.history else (keys.event_id if keys else "")),
            "is_repeat_event": result.is_repeat,
            "event_increment_context": "",
            "event_memory_block": "",
            "event_history": {},
            "delta_updates": {},
            "memory_meta": {
                "enabled": True,
                "store": getattr(store, "backend_name", "unknown"),
                "embedding": getattr(embedder, "name", "unknown"),
                "increment_mode": mode,
                "is_repeat": result.is_repeat,
                "level": result.level,
                "confidence": result.confidence,
                "reason": result.reason,
                "trace": result.trace,
                "subject": getattr(keys, "subject", ""),
                "issue": getattr(keys, "issue", ""),
                "aliases": getattr(keys, "aliases", []),
            },
        }

        if not result.is_repeat or result.history is None:
            logger.info("[MEMORY] 判为新事件，走完整研究流程 | %s", result.trace_text())
            return Command(goto="write_research_brief", update=base_update)

        # ---- 命中历史事件：进入增量追踪模式 ----
        history_rec = result.history
        logger.info(
            "[MEMORY] 命中历史事件 %s（%s），启动增量进展追踪 | %s",
            history_rec.event_id, result.level, result.trace_text(),
        )

        delta = await _run_delta_tracking(history_rec, mem_cfg, model)

        # 把 delta 合并进记录副本，用于渲染上下文（真正的落库在 commit 节点）
        merged = apply_delta_to_record(history_rec.model_copy(deep=True), delta)

        base_update.update(
            {
                "event_increment_context": _render_increment_context(merged, delta),
                "event_memory_block": _render_memory_block(merged, delta),
                "event_history": history_rec.model_dump(),
                "delta_updates": delta.model_dump(),
                "memory_meta": {
                    **base_update["memory_meta"],
                    "increment_mode": mode,
                    "delta_summary": delta.summary,
                    "delta_timeline_added": len(delta.timeline_updates),
                    "delta_rumor_updates": len(delta.rumor_updates),
                    "delta_queries": delta.queries,
                },
            }
        )

        if mode == "fast":
            # fast 模式：跳过主研究循环，把增量资料直接塞进 findings（notes），
            # 由 write_draft_report 生成更新版草稿 → final_report_generation 出报告。
            base_update.update(
                {
                    "skip_research_loop": True,
                    "research_brief": _render_increment_brief(merged, delta),
                    "notes": [delta.raw_findings or "（未发现新进展）"],
                    "raw_notes": [delta.raw_findings or ""],
                }
            )
            logger.info("[MEMORY] fast 模式：跳过主研究循环，直达 write_draft_report")
            return Command(goto="write_draft_report", update=base_update)

        return Command(goto="write_research_brief", update=base_update)

    except Exception as exc:
        logger.error("[MEMORY] event_recall 失败，降级为无历史模式: %s", str(exc)[:300], exc_info=False)
        empty_update["memory_meta"] = {"enabled": True, "error": str(exc)[:300]}
        return Command(goto="write_research_brief", update=empty_update)


async def _run_delta_tracking(record: EventRecord, mem_cfg: Dict[str, Any], model) -> DeltaUpdate:
    """执行「时间 + 关键词」增量检索并抽取 delta。"""
    delta_cfg = mem_cfg.get("delta") or {}
    lookback_days = int(delta_cfg.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
    max_queries = int(delta_cfg.get("max_queries", DEFAULT_MAX_QUERIES))
    keywords = delta_cfg.get("keywords") or None

    queries = build_delta_queries(
        record.subject, record.issue,
        keywords=keywords,
        lookback_days=lookback_days,
        max_queries=max_queries,
        keywords_per_query=int(delta_cfg.get("keywords_per_query", 3)),
    )
    logger.info("[MEMORY] 增量检索式（%d 条）: %s", len(queries), queries)

    docs = search_delta(
        queries,
        max_results=int(delta_cfg.get("max_results", 3)),
        topic=str(delta_cfg.get("topic", "news")),
    )
    findings_text = format_delta_findings(docs)

    delta = await extract_delta(
        record.history_digest(),
        findings_text,
        model,
        queries=queries,
    )
    delta.raw_findings = findings_text
    return delta


# ===== 节点 2：归档回写 =====

async def event_memory_commit_node(
    state: Dict[str, Any], config: RunnableConfig | None = None
) -> Dict[str, Any]:
    """研究完成后：抽 delta → 状态机迁移 → 事件记录 upsert 到存储。"""
    mem_cfg = _load_memory_cfg()
    if not mem_cfg.get("enabled", False):
        return {"memory_meta": {"enabled": False}}

    final_report = state.get("final_report", "") or ""
    if not final_report.strip():
        logger.warning("[MEMORY] 无 final_report，跳过归档")
        return {"memory_meta": {"enabled": True, "committed": False, "reason": "empty_report"}}

    memory_meta = dict(state.get("memory_meta") or {})
    try:
        store, embedder, model = _get_runtime(mem_cfg)

        event_id = state.get("event_id") or make_event_id(
            memory_meta.get("subject", ""), memory_meta.get("issue", "")
        )
        if not event_id or event_id == "|":
            logger.warning("[MEMORY] 无法确定 event_id，跳过归档")
            return {"memory_meta": {**memory_meta, "committed": False, "reason": "no_event_id"}}

        history_dict = state.get("event_history") or {}
        history_rec = EventRecord.model_validate(history_dict) if history_dict else None

        # ---- 用最终报告作为"本次资料"再抽一次 delta（这样首次建档也能得到 timeline/rumors）----
        history_text = history_rec.history_digest() if history_rec else ""
        delta_cfg = mem_cfg.get("delta") or {}
        report_findings = f"【本次最终报告全文】\n{final_report[:12000]}"
        delta = await extract_delta(history_text, report_findings, model, queries=[])

        # ---- upsert ----
        today = today_str()
        if history_rec is not None:
            record = history_rec.model_copy(deep=True)
            record.content_archive.append({"report_date": record.report_date, "content": record.content})
            keep = int((mem_cfg.get("store") or {}).get("archive_keep", 5))
            record.content_archive = record.content_archive[-keep:]
            record.report_count += 1
        else:
            subject = memory_meta.get("subject", "") or (event_id.split("|") + [""])[0]
            issue = memory_meta.get("issue", "") or (event_id.split("|") + [""])[1]
            record = EventRecord(
                event_id=event_id,
                subject=subject,
                issue=issue,
                first_seen=today,
                report_count=1,
                aliases=list(memory_meta.get("aliases") or []),
            )
            if memory_meta.get("aliases"):
                record.aliases = list(dict.fromkeys([*record.aliases, *memory_meta["aliases"]]))

        apply_delta_to_record(record, delta)
        record.report_date = today
        record.last_seen = today
        record.content = final_report
        record.embedding = embedder.embed(final_report)
        record.meta = {
            "embedding_provider": getattr(embedder, "name", "unknown"),
            "embedding_dim": len(record.embedding),
            "store_backend": getattr(store, "backend_name", "unknown"),
        }

        store.upsert(record)
        logger.info(
            "[MEMORY] ✅ 已归档事件 %s（report_count=%d, timeline=%d, rumors=%d, dim=%d）",
            record.event_id, record.report_count, len(record.timeline), len(record.rumors), len(record.embedding),
        )

        return {
            "memory_meta": {
                **memory_meta,
                "committed": True,
                "report_count": record.report_count,
                "timeline_size": len(record.timeline),
                "rumor_size": len(record.rumors),
                "timeline_added": len(delta.timeline_updates),
                "rumor_updates": len(delta.rumor_updates),
                "commit_summary": delta.summary,
            }
        }

    except Exception as exc:
        logger.error("[MEMORY] event_memory_commit 失败（不影响报告产出）: %s", str(exc)[:300])
        return {"memory_meta": {**memory_meta, "committed": False, "error": str(exc)[:300]}}
