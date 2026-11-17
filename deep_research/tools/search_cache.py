#***********************************************
#      Filename: search_cache.py
#   Description: 搜索结果磁盘缓存（支持录制/回放）
#***********************************************

"""搜索结果缓存层。

解决的问题：压测 / 量化对比 / 回归测试会反复打同一批 query，每次都真实调用
搜索 API（Tavily 约 $0.008/次）纯属浪费。本模块把搜索结果落盘，支持四种模式：

    off      直通，不读不写（关闭缓存）
    on       读 + 写。开发迭代默认模式，第二次跑同一 query 零成本
    record   只写不读。强制走真实 API，用于一次性录制基线语料
    replay   只读不写。**零 API 成本**，用于测速度 / 量化对比 / 回归测试

用法（由 search_factory 自动接入，业务代码无感知）：

    # config.yml
    search:
      backend: tavily
      cache:
        mode: replay
        dir: ./data/search_cache

典型压测流程：

    1) 先录制：mode=record 跑 30 条真实事件 → 缓存落盘约 1500 条搜索结果
    2) 换量化配置，mode=replay 反复跑 → 搜索成本恒为 0，只烧 GPU
    3) 想看命中率 / 省了多少调用：python -m deep_research.tools.search_cache --stats
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_research import logging as dr_logging

logger = dr_logging.get_logger(__name__)

DEFAULT_CACHE_DIR = "./data/search_cache"
VALID_MODES = ("off", "on", "record", "replay")


def normalize_mode(mode: Optional[Any]) -> str:
    """把配置里的 mode 规整成合法值，非法值一律回落 off（宁可不用，也不能静默出错）。

    ⚠️ 必须处理布尔：YAML 1.1 会把裸写的 on/off/yes/no 解析成 bool，
    即 `mode: on` 到 Python 里是 True、`mode: off` 是 False。
    不特判就会出现「用户配了 on，却被当成非法值回落 off」的静默失效。
    """
    if isinstance(mode, bool):
        return "on" if mode else "off"

    m = str(mode if mode is not None else "off").strip().lower()
    if m not in VALID_MODES:
        logger.warning("未知的 search cache mode=%r，回落到 'off'（合法值：%s）", mode, VALID_MODES)
        return "off"
    return m


def make_cache_key(
    backend: str,
    query: str,
    *,
    max_results: int,
    topic: str,
    include_raw_content: bool,
) -> str:
    """生成稳定缓存 key。

    只包含影响搜索结果本身的参数——不含 api_key / timeout 这类与结果无关的字段，
    否则换个 key 就会全部 miss。用 sha256 而非 Python hash()（后者带随机盐，跨进程不稳定）。
    """
    payload = json.dumps(
        [backend, query, int(max_results or 0), str(topic or ""), bool(include_raw_content)],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


@dataclass
class CacheStats:
    """缓存命中统计，压测报告里可直接引用（面试讲述用得上）。"""

    hits: int = 0
    misses: int = 0
    writes: int = 0
    bypass: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "writes": self.writes,
            "bypass": self.bypass,
            "total": self.total,
            "hit_rate": round(self.hit_rate, 4),
        }

    def summary(self) -> str:
        if self.total == 0:
            return "搜索缓存：本进程无查询记录"
        saved = self.hits * 0.008  # Tavily basic search $0.008/credit
        return (
            f"搜索缓存：命中 {self.hits} / 查询 {self.total}（命中率 {self.hit_rate:.1%}），"
            f"落盘 {self.writes}，直通 {self.bypass}，约省 ${saved:.2f}"
        )


class CacheMissError(RuntimeError):
    """replay 模式下缓存未命中，且 strict_miss=True。

    为什么必须能抛错：训练/评测时静默返回空结果会造出「搜了个寂寞」的假轨迹——
    agent 拿着空材料继续写报告，reward 偏低，模型学到的是"这条路别走"，
    而真相是缓存没命中。这种样本比随机噪声更坏，因为它带系统性偏差，
    而且不报错、跑完才发现。
    """


class SearchCache:
    """磁盘搜索缓存。一个 key 一个 json 文件，写入走临时文件 + os.replace 保证原子性。"""

    def __init__(self, cache_dir: str = DEFAULT_CACHE_DIR, mode: str = "on",
                 strict_miss: bool = False):
        self.mode = normalize_mode(mode)
        self.strict_miss = bool(strict_miss)
        self.dir = Path(cache_dir)
        self.stats = CacheStats()
        self._lock = threading.Lock()
        if self.mode != "off":
            try:
                self.dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error("无法创建搜索缓存目录 %s: %s，缓存自动关闭", self.dir, e)
                self.mode = "off"

    # ---- 读写 ----

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Any | None:
        """只在 on / replay 模式下命中缓存。"""
        if self.mode in ("off", "record"):
            return None
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("读取搜索缓存失败 key=%s: %s", key, e)
            return None

    def put(self, key: str, value: Any) -> None:
        """只在 on / record 模式下写缓存。"""
        if self.mode in ("off", "replay"):
            return
        path = self._path(key)
        tmp = path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, path)
        except (OSError, TypeError) as e:
            logger.warning("写入搜索缓存失败 key=%s: %s", key, e)

    # ---- 统计 ----

    def _bump(self, name: str) -> None:
        with self._lock:
            setattr(self.stats, name, getattr(self.stats, name) + 1)

    def hit(self) -> None:
        self._bump("hits")

    def miss(self) -> None:
        self._bump("misses")

    def wrote(self) -> None:
        self._bump("writes")

    def bypassed(self) -> None:
        self._bump("bypass")

    def count_files(self) -> int:
        if not self.dir.exists():
            return 0
        return sum(1 for _ in self.dir.glob("*.json"))


class CachedProvider:
    """包装任意 SearchProvider，在不改上层代码的前提下加上缓存。

    实现 SearchProvider 协议（build_client / search / defaults），
    因此 search_factory 拿到的仍然是一个普通 provider。
    """

    def __init__(self, base: Any, base_name: str, cache: SearchCache):
        self._base = base
        self._base_name = base_name
        self._cache = cache

    @property
    def cache(self) -> SearchCache:
        return self._cache

    def build_client(self, provider_cfg: Dict[str, Any]) -> Any:
        return self._base.build_client(provider_cfg)

    def search(
        self,
        client: Any,
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
        topic: str,
        timeout_seconds: int | None,
    ) -> Any:
        key = make_cache_key(
            self._base_name,
            query,
            max_results=max_results,
            topic=topic,
            include_raw_content=include_raw_content,
        )

        cached = self._cache.get(key)
        if cached is not None:
            self._cache.hit()
            logger.debug("[CACHE HIT] backend=%s query=%r", self._base_name, query[:60])
            return cached

        # replay 模式未命中：
        #   strict_miss=True  → 抛错（训练/评测必须这样，空结果会污染样本）
        #   strict_miss=False → 返回合法空结果让链路继续（压测可用，但会改变工作负载形态）
        if self._cache.mode == "replay":
            self._cache.miss()
            msg = (
                f"[CACHE MISS/REPLAY] 回放模式缓存未命中，query={query[:60]!r} —— "
                f"先跑一遍 mode=record 录制基线语料可避免"
            )
            if self._cache.strict_miss:
                raise CacheMissError(msg + "；strict_miss=True，已中止本次调用。")
            logger.warning(msg + "；返回空结果。")
            return {"query": query, "results": [], "response_time": 0.0, "_cache_miss": True}

        self._cache.miss()
        result = self._base.search(
            client,
            query,
            max_results=max_results,
            include_raw_content=include_raw_content,
            topic=topic,
            timeout_seconds=timeout_seconds,
        )
        self._cache.put(key, result)
        self._cache.wrote()
        return result

    def defaults(self, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return self._base.defaults(provider_cfg)

    def stats(self) -> Dict[str, Any]:
        return self._cache.stats.as_dict()


# ---------------------------------------------------------------------------
# CLI：查看缓存状态 / 清理
# ---------------------------------------------------------------------------

def _iter_cache_providers() -> List[CachedProvider]:
    """从 search_factory 的 provider 缓存里捞出所有 CachedProvider。

    故意延迟 import，避免模块级循环依赖（search_factory 会 import 本模块）。
    """
    try:
        from deep_research.tools import search_factory
    except Exception as e:  # pragma: no cover
        logger.warning("无法导入 search_factory: %s", e)
        return []
    found = []
    for provider in getattr(search_factory, "_PROVIDER_CACHE", {}).values():
        if isinstance(provider, CachedProvider):
            found.append(provider)
    return found


def _main(argv: List[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="搜索缓存的查看与清理")
    ap.add_argument("--stats", action="store_true", help="按 config.yml 构建 provider 后打印统计")
    ap.add_argument("--dir", default=DEFAULT_CACHE_DIR, help="缓存目录")
    ap.add_argument("--count", action="store_true", help="只数缓存文件条数")
    ap.add_argument("--clear", action="store_true", help="清空缓存目录中的所有条目")
    args = ap.parse_args(argv)

    cache_dir = Path(args.dir)

    if args.count:
        n = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0
        print(f"{cache_dir} 缓存条目：{n}")
        return 0

    if args.clear:
        if not cache_dir.exists():
            print(f"{cache_dir} 不存在，无需清理")
            return 0
        removed = 0
        for p in cache_dir.glob("*.json"):
            try:
                p.unlink()
                removed += 1
            except OSError as e:
                logger.warning("删除 %s 失败: %s", p, e)
        print(f"已删除 {removed} 条缓存")
        return 0

    if args.stats:
        # 触发一次 provider 构建，然后读统计
        try:
            from deep_research.tools.search_factory import get_search_provider
            get_search_provider()
        except Exception as e:
            print(f"构建 provider 失败（可能未启用缓存）：{e}")
        provs = _iter_cache_providers()
        if not provs:
            print("当前未启用搜索缓存（config.yml 的 search.cache.mode 需为 on/record/replay）")
            return 0
        for p in provs:
            s = p.stats()
            print(f"backend={p._base_name} mode={p._cache.mode} dir={p._cache.dir}")
            print(f"  {p._cache.stats.summary()}")
            print(f"  落盘条目：{p._cache.count_files()}")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main(sys.argv[1:]))
