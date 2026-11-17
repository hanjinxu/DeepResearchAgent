#***********************************************
#      Filename: test_search_cache.py
#   Description: 搜索缓存层离线测试
#***********************************************

"""搜索缓存层测试（全部离线，不发任何真实请求）。

跑法：python test_search_cache.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

sys.path.insert(0, str(Path(__file__).parent))

from deep_research.tools.search_cache import (  # noqa: E402
    CachedProvider,
    SearchCache,
    make_cache_key,
    normalize_mode,
)

PASSED = 0
FAILED = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print(f"  [OK]   {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name}  {detail}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


class FakeProvider:
    """记录调用次数的假 provider，用来证明缓存是否真的拦截了请求。"""

    def __init__(self, name: str = "fake"):
        self.name = name
        self.calls = 0
        self.last_query = None

    def build_client(self, provider_cfg: Dict[str, Any]) -> Any:
        return {"client_for": self.name}

    def search(self, client, query, *, max_results, include_raw_content, topic, timeout_seconds):
        self.calls += 1
        self.last_query = query
        return {
            "query": query,
            "results": [
                {"title": f"R{i}", "url": f"https://example.com/{query}/{i}", "content": f"body {i}"}
                for i in range(max_results)
            ],
            "raw_content_calls": max_results,
        }

    def defaults(self, provider_cfg: Dict[str, Any]) -> Dict[str, Any]:
        return {"max_results": 3, "topic": "general", "include_raw_content": True, "timeout_seconds": None}


def _tmpdir() -> str:
    return tempfile.mkdtemp(prefix="search_cache_test_")


def call(provider: CachedProvider, query: str, **kw):
    defaults = dict(max_results=3, include_raw_content=True, topic="general", timeout_seconds=None)
    defaults.update(kw)
    return provider.search(None, query, **defaults)


# ---------------------------------------------------------------------------


def t1_key_stability() -> None:
    section("t1 make_cache_key 稳定性")
    a = make_cache_key("tavily", "华为 回应", max_results=3, topic="news", include_raw_content=True)
    b = make_cache_key("tavily", "华为 回应", max_results=3, topic="news", include_raw_content=True)
    check("同参数 → 同 key", a == b, f"{a} != {b}")

    c = make_cache_key("tavily", "华为 回应", max_results=5, topic="news", include_raw_content=True)
    check("max_results 变了 → key 变", a != c)

    d = make_cache_key("tavily", "华为 回应", max_results=3, topic="general", include_raw_content=True)
    check("topic 变了 → key 变", a != d)

    e = make_cache_key("parallel", "华为 回应", max_results=3, topic="news", include_raw_content=True)
    check("backend 变了 → key 变", a != e)

    f = make_cache_key("tavily", "华为 回应 ", max_results=3, topic="news", include_raw_content=True)
    check("query 尾空格也算不同（不做隐式归一化，避免语义漂移）", a != f)

    check("key 长度固定 24", len(a) == 24, f"len={len(a)}")


def t2_mode_normalize() -> None:
    section("t2 normalize_mode 非法值回落 + YAML 布尔兼容")
    check("On → on", normalize_mode("On") == "on")
    check(" RECORD  → record", normalize_mode(" RECORD ") == "record")
    check("None → off", normalize_mode(None) == "off")
    check("乱写 → off", normalize_mode("yes-please") == "off")

    # YAML 1.1 会把裸写的 on/off 解析成 bool，必须还原（否则 mode: on 会静默失效）
    check("bool True（YAML 里的 `mode: on`）→ on", normalize_mode(True) == "on")
    check("bool False（YAML 里的 `mode: off`）→ off", normalize_mode(False) == "off")


def t3_off_mode() -> None:
    section("t3 mode=off 直通（行为与改造前一致）")
    d = _tmpdir()
    base = FakeProvider()
    cache = SearchCache(d, mode="off")
    p = CachedProvider(base, "fake", cache)

    call(p, "q1")
    call(p, "q1")
    check("每次都真的调 base", base.calls == 2, f"calls={base.calls}")
    check("不落盘", cache.count_files() == 0, f"files={cache.count_files()}")
    check("off 模式不建目录", not Path(d).exists() or cache.count_files() == 0)
    shutil.rmtree(d, ignore_errors=True)


def t4_on_mode() -> None:
    section("t4 mode=on 读+写")
    d = _tmpdir()
    base = FakeProvider()
    cache = SearchCache(d, mode="on")
    p = CachedProvider(base, "fake", cache)

    r1 = call(p, "q1")
    check("首次未命中 → 调 base", base.calls == 1, f"calls={base.calls}")
    check("首次结果正确", len(r1["results"]) == 3)

    r2 = call(p, "q1")
    check("二次命中 → 不再调 base", base.calls == 1, f"calls={base.calls}")
    check("命中返回内容一致", r1 == r2)

    call(p, "q2")
    check("新 query 仍走 base", base.calls == 2, f"calls={base.calls}")

    st = cache.stats.as_dict()
    check("统计 hits=1", st["hits"] == 1, str(st))
    check("统计 misses=2", st["misses"] == 2, str(st))
    check("统计 writes=2", st["writes"] == 2, str(st))
    check("命中率 1/3（as_dict 保留 4 位小数）", abs(st["hit_rate"] - round(1 / 3, 4)) < 1e-9, str(st))
    check("落盘 2 个文件", cache.count_files() == 2, f"files={cache.count_files()}")
    shutil.rmtree(d, ignore_errors=True)


def t5_record_mode() -> None:
    section("t5 mode=record 只写不读（强制走真实 API）")
    d = _tmpdir()
    base = FakeProvider()
    cache = SearchCache(d, mode="record")
    p = CachedProvider(base, "fake", cache)

    call(p, "q1")
    call(p, "q1")
    check("两次都调 base（即使已有缓存）", base.calls == 2, f"calls={base.calls}")
    check("但都落盘了", cache.count_files() == 1, f"files={cache.count_files()}")
    check("record 模式 hits=0", cache.stats.hits == 0, str(cache.stats.as_dict()))
    shutil.rmtree(d, ignore_errors=True)


def t6_replay_mode() -> None:
    section("t6 mode=replay 只读不写（零 API 成本，关键场景）")
    d = _tmpdir()

    # 先用 on 模式录制两条
    base0 = FakeProvider()
    SearchCache(d, mode="on")  # 建目录
    p0 = CachedProvider(base0, "fake", SearchCache(d, mode="record"))
    call(p0, "q1")
    call(p0, "q2")
    check("录制阶段调了 2 次 base", base0.calls == 2, f"calls={base0.calls}")

    # 切 replay
    base1 = FakeProvider()
    cache1 = SearchCache(d, mode="replay")
    p1 = CachedProvider(base1, "fake", cache1)

    r = call(p1, "q1")
    check("命中缓存：**零** base 调用", base1.calls == 0, f"calls={base1.calls}")
    check("回放内容正确", len(r["results"]) == 3)

    r_new = call(p1, "never_recorded_before")
    check("未命中：仍**不**调 base（这是零成本的代价）", base1.calls == 0, f"calls={base1.calls}")
    check("未命中返回空结果（链路可继续）", r_new["results"] == [], str(r_new)[:120])
    check("未命中带 _cache_miss 标记", r_new.get("_cache_miss") is True)

    check("replay 不写新文件", cache1.count_files() == 2, f"files={cache1.count_files()}")
    st = cache1.stats.as_dict()
    check("replay 统计 hits=1 misses=1", st["hits"] == 1 and st["misses"] == 1, str(st))
    check("replay 统计 writes=0", st["writes"] == 0, str(st))
    shutil.rmtree(d, ignore_errors=True)


def t7_delegation() -> None:
    section("t7 CachedProvider 透传 build_client / defaults")
    d = _tmpdir()
    base = FakeProvider("tavily")
    p = CachedProvider(base, "tavily", SearchCache(d, mode="on"))

    client = p.build_client({"api_key": "x"})
    check("build_client 透传", client == {"client_for": "tavily"}, str(client))

    dv = p.defaults({})
    check("defaults 透传", dv["max_results"] == 3, str(dv))
    shutil.rmtree(d, ignore_errors=True)


def t8_corrupt_file() -> None:
    section("t8 缓存文件损坏时降级为 miss，不炸链路")
    d = _tmpdir()
    base = FakeProvider()
    cache = SearchCache(d, mode="on")
    p = CachedProvider(base, "fake", cache)

    key = make_cache_key("fake", "q1", max_results=3, topic="general", include_raw_content=True)
    Path(d, f"{key}.json").write_text("{ this is not json", encoding="utf-8")

    r = call(p, "q1")
    check("损坏缓存不抛异常", isinstance(r, dict), str(type(r)))
    check("降级为真实调用", base.calls == 1, f"calls={base.calls}")
    check("并已覆盖修复", cache.count_files() == 1)
    shutil.rmtree(d, ignore_errors=True)


def t9_factory_wrap() -> None:
    section("t9 search_factory 接入（_maybe_wrap_with_cache）")
    from deep_research.tools.search_factory import _maybe_wrap_with_cache

    base = FakeProvider()

    p_off = _maybe_wrap_with_cache("fake", base, {"cache": {"mode": "off"}})
    check("mode=off 不包装", p_off is base)

    p_none = _maybe_wrap_with_cache("fake", base, {})
    check("无 cache 段不包装", p_none is base)

    p_bad = _maybe_wrap_with_cache("fake", base, {"cache": "not-a-dict"})
    check("cache 段类型错误不包装", p_bad is base)

    # YAML 把 `mode: on` 解析成 True —— 必须仍能正确包装
    d_bool = _tmpdir()
    p_bool_on = _maybe_wrap_with_cache("fake", base, {"cache": {"mode": True, "dir": d_bool}})
    check("mode 为 YAML 布尔 True 时仍包装", isinstance(p_bool_on, CachedProvider), str(type(p_bool_on)))
    check("且归一化为 on", getattr(p_bool_on, "cache", None) is not None
          and p_bool_on.cache.mode == "on", str(getattr(p_bool_on, "cache", None)))
    shutil.rmtree(d_bool, ignore_errors=True)

    p_bool_off = _maybe_wrap_with_cache("fake", base, {"cache": {"mode": False}})
    check("mode 为 YAML 布尔 False 时不包装", p_bool_off is base)

    d = _tmpdir()
    p_on = _maybe_wrap_with_cache("fake", base, {"cache": {"mode": "on", "dir": d}})
    check("mode=on 已包装", isinstance(p_on, CachedProvider), str(type(p_on)))
    check("包装后仍是合法 SearchProvider（有 search/defaults）",
          hasattr(p_on, "search") and hasattr(p_on, "defaults"))

    # 端到端：通过包装后的 provider 走一遍
    p_on.search(None, "zz", max_results=2, include_raw_content=True, topic="general", timeout_seconds=None)
    p_on.search(None, "zz", max_results=2, include_raw_content=True, topic="general", timeout_seconds=None)
    check("包装后第二次命中，base 只调 1 次", base.calls == 1, f"calls={base.calls}")
    shutil.rmtree(d, ignore_errors=True)


def t10_stats_summary() -> None:
    section("t10 统计摘要文案（面试报告用）")
    d = _tmpdir()
    base = FakeProvider()
    cache = SearchCache(d, mode="on")
    p = CachedProvider(base, "fake", cache)

    check("无查询时的文案", "无查询记录" in cache.stats.summary(), cache.stats.summary())

    call(p, "a")
    call(p, "a")
    call(p, "b")
    s = cache.stats.summary()
    check("有查询时含命中率", "命中率" in s, s)
    check("含省钱估算", "约省 $" in s, s)
    print(f"        → {s}")
    shutil.rmtree(d, ignore_errors=True)


def main() -> int:
    print("搜索缓存层离线测试")
    for fn in (t1_key_stability, t2_mode_normalize, t3_off_mode, t4_on_mode,
               t5_record_mode, t6_replay_mode, t7_delegation, t8_corrupt_file,
               t9_factory_wrap, t10_stats_summary):
        try:
            fn()
        except Exception as e:
            global FAILED
            FAILED += 1
            print(f"  [ERROR] {fn.__name__} 抛异常: {type(e).__name__}: {e}")

    print(f"\n{'=' * 56}")
    print(f"通过 {PASSED} / 失败 {FAILED}")
    print(f"{'=' * 56}")
    return 0 if FAILED == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
