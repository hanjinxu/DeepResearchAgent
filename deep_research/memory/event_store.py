#***********************************************
#      Filename: event_store.py
#   Description: 事件记忆的持久化层（ES 为主，JSONL 兜底）
#***********************************************

"""
【存储层设计原则】

1. **ES 不是硬依赖**。config 里配 `backend: elasticsearch`，但连接/建索引失败会自动
   降级到本地 JSONL 并告警——绝不允许记忆层把已经跑通的主链路拖挂。
2. **写入语义 = 事件级 upsert**。以 event_id 为文档 _id，同一事件多次报告更新同一文档，
   历史正文进 content_archive（只留最近 N 份）。
3. **检索语义 = 最近窗口内的向量召回**。L2 只在最近 lookback 窗口内做内存余弦计算，
   量级可控（按 150 热点/天算，一年约 5.5 万条，窗口内一般几百条）。

ES 文档结构（用户约定的最小 schema + 扩展字段）：
    {"event_id":"华为|内部推荐政策",
     "subject":"华为",
     "issue":"内部推荐政策",
     "report_date":"2024-06-15",
     "content":"……报告全文……",
     "embedding":[ ... ],
     ... timeline / rumors / delta_log / aliases / first_seen / last_seen / report_count}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from deep_research import logging as dr_logging
from deep_research.memory.embedding import cosine
from deep_research.memory.schema import EventRecord

logger = dr_logging.get_logger(__name__)

DEFAULT_JSONL_PATH = "./data/event_memory/events.jsonl"
DEFAULT_ES_INDEX = "deepresearch_events"


# ===== 协议 =====

class EventStore(Protocol):
    """事件存储协议。"""

    backend_name: str

    def upsert(self, record: EventRecord) -> None:
        ...

    def get(self, event_id: str) -> Optional[EventRecord]:
        ...

    def load_recent(self, limit: int = 2000, since_date: Optional[str] = None) -> List[EventRecord]:
        ...

    def count(self) -> int:
        ...

    def close(self) -> None:
        ...


# ===== 实现 1：本地 JSONL（零依赖兜底） =====

class JsonlEventStore:
    """追加式 JSONL 存储：每行一个事件快照，读取时按 event_id 取最后一条（last-write-wins）。"""

    backend_name = "jsonl"

    def __init__(self, path: str = DEFAULT_JSONL_PATH) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[EVENT_STORE] 使用 JSONL 存储: %s", self.path)

    # -- 内部 --
    def _iter_raw(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        out: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("[EVENT_STORE] 跳过损坏的 JSONL 行: %s", line[:120])
        return out

    # -- 协议实现 --
    def upsert(self, record: EventRecord) -> None:
        with self.path.open("a", encoding="utf-8") as fd:
            fd.write(record.model_dump_json() + "\n")

    def get(self, event_id: str) -> Optional[EventRecord]:
        hit: Optional[EventRecord] = None
        for raw in self._iter_raw():
            if raw.get("event_id") == event_id:
                hit = EventRecord.model_validate(raw)
        return hit

    def load_recent(self, limit: int = 2000, since_date: Optional[str] = None) -> List[EventRecord]:
        latest: Dict[str, EventRecord] = {}
        for raw in self._iter_raw():
            try:
                rec = EventRecord.model_validate(raw)
            except Exception:
                continue
            if since_date and (rec.report_date or "") and rec.report_date < since_date:
                continue
            latest[rec.event_id] = rec
        records = sorted(latest.values(), key=lambda r: r.report_date or "", reverse=True)
        return records[:limit]

    def count(self) -> int:
        return len({r.get("event_id") for r in self._iter_raw()})

    def compact(self) -> int:
        """压缩：按 event_id 只保留最新快照。返回压缩后的记录数。"""
        records = self.load_recent(limit=10 ** 9)
        tmp = self.path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fd:
            for rec in records:
                fd.write(rec.model_dump_json() + "\n")
        os.replace(tmp, self.path)
        logger.info("[EVENT_STORE] JSONL 已压缩至 %d 条", len(records))
        return len(records)

    def close(self) -> None:
        return None


# ===== 实现 2：Elasticsearch =====

class ElasticsearchEventStore:
    """ES 存储。dense_vector 字段支持 kNN，同时保留 keyword/text 用于精确与全文检索。"""

    backend_name = "elasticsearch"

    def __init__(
        self,
        hosts: Optional[List[str]] = None,
        index: str = DEFAULT_ES_INDEX,
        username: str = "",
        password: str = "",
        dim: int = 0,
        timeout: int = 30,
    ) -> None:
        from elasticsearch import Elasticsearch  # 延迟导入，未装则调用方降级

        hosts = hosts or ["http://localhost:9200"]
        kwargs: Dict[str, Any] = {"hosts": hosts, "request_timeout": timeout}
        if username:
            kwargs["basic_auth"] = (username, password)

        self.client = Elasticsearch(**kwargs)
        self.index = index
        self.dim = int(dim or 0)

        if not self.client.ping():
            raise ConnectionError(f"Elasticsearch 无法连接: {hosts}")

        self._ensure_index()
        logger.info("[EVENT_STORE] 使用 Elasticsearch: %s / index=%s", hosts, index)

    # -- 内部 --
    def _ensure_index(self) -> None:
        if self.client.indices.exists(index=self.index):
            return
        if self.dim <= 0:
            # ES 的 dense_vector 必须声明 dims，维度未知时无法建索引 → 让工厂降级
            raise ValueError("创建 ES 索引需要已知的 embedding 维度（dims），请在 event_memory.embedding 中显式配置 dim")

        vec_field: Dict[str, Any] = {
            "type": "dense_vector",
            "dims": self.dim,
            "index": True,
            "similarity": "cosine",
        }

        mapping = {
            "mappings": {
                "properties": {
                    "event_id": {"type": "keyword"},
                    "subject": {"type": "keyword"},
                    "issue": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                    "report_date": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                    "content": {"type": "text"},
                    "embedding": vec_field,
                    "timeline": {"type": "object", "enabled": False},
                    "rumors": {"type": "object", "enabled": False},
                    "delta_log": {"type": "object", "enabled": False},
                    "content_archive": {"type": "object", "enabled": False},
                    "aliases": {"type": "keyword"},
                    "first_seen": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                    "last_seen": {"type": "date", "format": "yyyy-MM-dd||epoch_millis"},
                    "report_count": {"type": "integer"},
                    "meta": {"type": "object", "enabled": False},
                }
            }
        }
        self.client.indices.create(index=self.index, **mapping)
        logger.info("[EVENT_STORE] 已创建 ES 索引 %s（dense_vector dims=%d）", self.index, self.dim)

    # -- 协议实现 --
    def upsert(self, record: EventRecord) -> None:
        self.client.index(
            index=self.index,
            id=record.event_id,
            document=record.model_dump(),
            refresh=False,
        )

    def get(self, event_id: str) -> Optional[EventRecord]:
        try:
            hit = self.client.get(index=self.index, id=event_id)
        except Exception:
            return None
        return EventRecord.model_validate(hit["_source"])

    def load_recent(self, limit: int = 2000, since_date: Optional[str] = None) -> List[EventRecord]:
        query: Dict[str, Any] = {"match_all": {}}
        if since_date:
            query = {"range": {"report_date": {"gte": since_date}}}
        resp = self.client.search(
            index=self.index,
            query=query,
            sort=[{"report_date": {"order": "desc"}}],
            size=limit,
            track_total_hits=False,
        )
        out: List[EventRecord] = []
        for hit in resp.get("hits", {}).get("hits", []):
            try:
                out.append(EventRecord.model_validate(hit["_source"]))
            except Exception:
                continue
        return out

    def count(self) -> int:
        try:
            return int(self.client.count(index=self.index)["count"])
        except Exception:
            return 0

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


# ===== 工厂 =====

def get_event_store(cfg: Dict[str, Any] | None = None, dim: int = 0) -> EventStore:
    """按配置构造事件存储；ES 不可用时自动降级为 JSONL。

    cfg 形如：
        {"backend": "elasticsearch",
         "jsonl_path": "./data/event_memory/events.jsonl",
         "elasticsearch": {"hosts": ["http://localhost:9200"], "index": "deepresearch_events"}}
    """
    cfg = cfg or {}
    backend = str(cfg.get("backend") or "jsonl").strip().lower()
    jsonl_path = cfg.get("jsonl_path") or DEFAULT_JSONL_PATH

    if backend in ("elasticsearch", "es"):
        es_cfg = cfg.get("elasticsearch") or {}
        try:
            return ElasticsearchEventStore(
                hosts=es_cfg.get("hosts") or ["http://localhost:9200"],
                index=es_cfg.get("index") or DEFAULT_ES_INDEX,
                username=es_cfg.get("username", ""),
                password=es_cfg.get("password", ""),
                dim=dim,
                timeout=es_cfg.get("timeout", 30),
            )
        except Exception as exc:
            logger.warning(
                "[EVENT_STORE] Elasticsearch 不可用（%s），自动降级为 JSONL：%s",
                (es_cfg.get("hosts") or ["http://localhost:9200"]),
                str(exc)[:200],
            )

    return JsonlEventStore(jsonl_path)


# ===== 向量召回（放在 store 之上，两种后端共用同一套逻辑） =====

def recall_by_vector(
    store: EventStore,
    query_vec: List[float],
    top_k: int = 5,
    candidate_limit: int = 2000,
    since_date: Optional[str] = None,
) -> List[tuple[EventRecord, float]]:
    """在最近窗口内做内存余弦召回，返回 [(记录, 相似度)]，按相似度降序。

    对 JSONL / ES 通用：ES 上即便建了 dense_vector 索引，这里也走内存计算，
    好处是行为一致、无需依赖 ES 版本特性；数据量再上一个量级时可换成 ES kNN。
    """
    candidates = store.load_recent(limit=candidate_limit, since_date=since_date)
    scored = [(rec, cosine(query_vec, rec.embedding)) for rec in candidates if rec.embedding]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
