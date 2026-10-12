#***********************************************
#      Filename: __init__.py
#   Description: 事件记忆层（判重 / 增量进展追踪 / 传闻状态机 / 归档存储）
#***********************************************

"""
事件记忆层模块地图：

    schema.py        —— EventRecord / TimelineNode / RumorState / DeltaUpdate
    embedding.py     —— 可插拔向量化（hashing-ngram 零依赖兜底 / OpenAI 兼容真 embedding）
    event_store.py   —— EventStore 抽象 + JsonlEventStore + ElasticsearchEventStore
    dedup.py         —— L1 精确匹配 → L2 向量召回 → L3 LLM 判定
    delta_tracker.py —— 「时间+关键词」增量检索 → 状态机迁移 / 时间线新节点
    memory_node.py   —— 接入 LangGraph 的两个节点：event_recall / event_memory_commit
"""

from deep_research.memory.schema import (
    DeltaUpdate,
    EventRecord,
    RumorState,
    RUMOR_STATUS_LABEL,
    TimelineNode,
    make_event_id,
    normalize_entity,
)
from deep_research.memory.embedding import cosine, get_embedder
from deep_research.memory.event_store import (
    ElasticsearchEventStore,
    EventStore,
    JsonlEventStore,
    get_event_store,
    recall_by_vector,
)
from deep_research.memory.dedup import DedupResult, EventKeys, resolve_event
from deep_research.memory.delta_tracker import (
    apply_delta_to_record,
    build_delta_queries,
    extract_delta,
    format_delta_findings,
    search_delta,
)

__all__ = [
    # schema
    "EventRecord",
    "TimelineNode",
    "RumorState",
    "DeltaUpdate",
    "RUMOR_STATUS_LABEL",
    "make_event_id",
    "normalize_entity",
    # embedding
    "get_embedder",
    "cosine",
    # store
    "EventStore",
    "JsonlEventStore",
    "ElasticsearchEventStore",
    "get_event_store",
    "recall_by_vector",
    # dedup
    "resolve_event",
    "DedupResult",
    "EventKeys",
    # delta
    "build_delta_queries",
    "search_delta",
    "format_delta_findings",
    "extract_delta",
    "apply_delta_to_record",
]
