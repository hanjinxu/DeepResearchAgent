#***********************************************
#      Filename: embedding.py
#   Description: 可插拔的向量化组件（供 L2 向量召回使用）
#***********************************************

"""
【为什么需要可插拔】

DeepSeek 官方 API **不提供 embedding 接口**，所以不能想当然地沿用 LLM 那套 backend 配置。
本模块提供两种 embedder，按 config 切换：

1. HashingNgramEmbedder（默认，零依赖）
   字符 2-gram + 3-gram 做 hashing trick 映射到固定维度（默认 256）后 L2 归一化。
   - 优点：离线可跑、无外部调用、固定维度可直接落 ES 的 dense_vector
   - 代价：语义泛化能力弱于真 embedding（只对字面重叠敏感）
   - 定位：L2 只负责「粗筛 top5 候选」，最终判定由 L3 的 LLM 完成，
     所以字面级召回在这里是够用的工程取舍。

2. OpenAICompatEmbedder（可选，真向量）
   指向任意 OpenAI 兼容的 /v1/embeddings 服务，例如本地 vLLM 部署的
   Qwen3-Embedding-0.6B / BGE-M3。此时 ES 端可建 dense_vector + kNN 检索。
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from typing import Any, Dict, Iterable, List, Protocol

from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)

DEFAULT_NGRAM_DIM = 256


# ===== 相似度工具 =====

def cosine(a: List[float], b: List[float]) -> float:
    """余弦相似度。任一向量为空则返回 0.0。"""
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = math.sqrt(sum(x * x for x in a[:n]))
    nb = math.sqrt(sum(x * x for x in b[:n]))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


# ===== 协议 =====

class BaseEmbedder(Protocol):
    """embedder 协议。"""

    name: str
    dim: int

    def embed(self, text: str) -> List[float]:
        ...

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        ...


# ===== 实现 1：零依赖 hashing n-gram =====

class HashingNgramEmbedder:
    """字符 n-gram + hashing trick 的轻量向量化（零外部依赖）。"""

    name = "hashing-ngram"

    def __init__(self, dim: int = DEFAULT_NGRAM_DIM, ngrams: tuple[int, ...] = (2, 3)) -> None:
        self.dim = int(dim) if dim and dim > 0 else DEFAULT_NGRAM_DIM
        self.ngrams = ngrams

    # -- 内部 --
    def _tokens(self, text: str) -> List[str]:
        s = (text or "").strip().lower()
        if not s:
            return []
        toks: List[str] = []
        for n in self.ngrams:
            if len(s) < n:
                continue
            toks.extend(s[i:i + n] for i in range(len(s) - n + 1))
        return toks

    def _bucket(self, token: str) -> tuple[int, float]:
        """用 md5 做稳定哈希（Python 内置 hash 带随机盐，跨进程不稳定）。"""
        h = hashlib.md5(token.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "big") % self.dim
        sign = 1.0 if (h[4] & 1) else -1.0
        return idx, sign

    # -- 协议实现 --
    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = self._tokens(text)
        if not toks:
            return vec
        for t in toks:
            idx, sign = self._bucket(t)
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# ===== 实现 2：OpenAI 兼容 embedding 服务 =====

class OpenAICompatEmbedder:
    """调用任意 OpenAI 兼容的 /v1/embeddings 服务（如本地 vLLM 的 Qwen3-Embedding）。

    用 stdlib urllib 实现，避免额外依赖（requests/httpx 版本差异曾在本项目踩过坑）。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "local-vllm",
        dim: int = 0,
        timeout: int = 60,
    ) -> None:
        self.name = f"openai-compatible:{model}"
        self.base_url = (base_url or "").rstrip("/")
        self.model = model
        self.api_key = api_key or "local-vllm"
        self.timeout = timeout
        # 维度在首次调用时探测（配置里写 0 表示自动）
        self.dim = int(dim) if dim and dim > 0 else 0

    def _post(self, texts: List[str]) -> List[List[float]]:
        if not self.base_url:
            raise ValueError("openai_compatible embedder 缺少 base_url")
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        data = sorted(body.get("data", []), key=lambda d: d.get("index", 0))
        return [d.get("embedding", []) or [] for d in data]

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Iterable[str]) -> List[List[float]]:
        items = [t or "" for t in texts]
        if not items:
            return []
        vecs = self._post(items)
        if vecs and not self.dim:
            self.dim = len(vecs[0])
        return vecs


# ===== 工厂 =====

def get_embedder(cfg: Dict[str, Any] | None = None) -> BaseEmbedder:
    """按配置构造 embedder；未知 provider 一律降级为零依赖实现。

    cfg 形如：
        {"provider": "hashing_ngram", "dim": 256}
        {"provider": "openai_compatible",
         "openai_compatible": {"base_url": "...", "model": "...", "api_key": "...", "dim": 1024}}
    """
    cfg = cfg or {}
    provider = str(cfg.get("provider") or "hashing_ngram").strip().lower()

    if provider in ("openai_compatible", "openai", "vllm"):
        sub = cfg.get("openai_compatible") or cfg.get("openai") or {}
        try:
            emb = OpenAICompatEmbedder(
                base_url=sub.get("base_url", ""),
                model=sub.get("model", "Qwen3-Embedding-0.6B"),
                api_key=sub.get("api_key", "local-vllm"),
                dim=sub.get("dim", 0),
                timeout=sub.get("timeout", 60),
            )
            # 探活：失败则降级，绝不让记忆层拖垮主链路
            emb.embed("探活")
            logger.info("[EMBEDDING] 使用 openai_compatible embedder: %s (dim=%s)", emb.model, emb.dim)
            return emb
        except Exception as exc:
            logger.warning(
                "[EMBEDDING] openai_compatible embedder 不可用（%s），降级为 hashing_ngram：%s",
                sub.get("base_url"),
                str(exc)[:200],
            )

    dim = cfg.get("dim") or DEFAULT_NGRAM_DIM
    emb = HashingNgramEmbedder(dim=int(dim))
    logger.info("[EMBEDDING] 使用 %s（dim=%d）", emb.name, emb.dim)
    return emb
