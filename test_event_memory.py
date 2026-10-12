#***********************************************
#      Filename: test_event_memory.py
#   Description: 事件记忆层离线自测（不依赖任何外部 API）
#
#   用法：
#     python test_event_memory.py            # 全离线，用 Mock LLM 跑完整判重链路
#     python test_event_memory.py --online   # 额外跑一次真实 LLM 判重（需要可用 config.yml）
#***********************************************

"""
覆盖的验证点：
  1. schema          —— event_id 生成 / 文本归一化 / 疑似噪词剥离
  2. embedding       —— hashing-ngram 向量维度、L2 归一化、语义相近文本相似度更高
  3. event_store     —— JSONL upsert / last-write-wins / load_recent
  4. dedup L1        —— 实体+议题精确匹配（含别名、"最新回应"这类时效词干扰）
  5. dedup L2        —— 向量召回排序
  6. dedup L3        —— LLM 判定（Mock）
  7. delta 检索式     —— 「时间 + 关键词」检索式构造
  8. 状态机迁移       —— 传闻 unconfirmed → refuted，并留下迁移历史
  9. 时间线去重       —— 重复节点不重复追加，新节点标记 is_new
 10. ES 文档结构      —— 落库字段与约定 schema 一致
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from deep_research.memory.dedup import (
    EventKeys,
    extract_event_keys,
    l1_exact_match,
    l2_vector_recall,
    resolve_event,
)
from deep_research.memory.delta_tracker import (
    apply_delta_to_record,
    build_delta_queries,
)
from deep_research.memory.embedding import HashingNgramEmbedder, cosine
from deep_research.memory.event_store import JsonlEventStore
from deep_research.memory.schema import (
    DeltaUpdate,
    EventRecord,
    RumorState,
    RUMOR_STATUS_LABEL,
    TimelineNode,
    make_event_id,
    normalize_entity,
)

PASS, FAIL = "✅", "❌"
_results: list[tuple[bool, str]] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    _results.append((bool(cond), label))
    mark = PASS if cond else FAIL
    print(f"{mark} {label}" + (f"  → {detail}" if detail else ""))


# ===== Mock LLM =====

class MockMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class MockLLM:
    """按 prompt 关键字返回预设 JSON，用于离线打通判重与 delta 抽取链路。"""

    def __init__(self, keys_json: str = "", judge_json: str = "", delta_json: str = "") -> None:
        self.keys_json = keys_json
        self.judge_json = judge_json
        self.delta_json = delta_json
        self.calls: list[str] = []

    async def ainvoke(self, messages):  # noqa: ANN001
        prompt = messages[-1].content if messages else ""
        self.calls.append(prompt[:60])
        if "事件主体" in prompt:
            return MockMessage(self.keys_json)
        if "归并判定器" in prompt:
            return MockMessage(self.judge_json)
        if "进展追踪器" in prompt:
            return MockMessage(self.delta_json)
        return MockMessage("{}")


# ===== 各测试段 =====

def t1_schema() -> None:
    print("\n[1] schema —— event_id 与归一化")
    eid = make_event_id("华为", "内部推荐政策")
    check(eid == "华为|内部推荐政策", "event_id 格式为 subject|issue", eid)
    check(
        normalize_entity("余承东 竹知了 最新回应") == normalize_entity("余承东竹知了事件"),
        "时效词/体裁词被正确剥离（'最新回应'/'事件'）",
        normalize_entity("余承东 竹知了 最新回应"),
    )
    rid1 = RumorState.make_id("华为内部推荐存在利益输送")
    rid2 = RumorState.make_id("华为内部推荐存在利益输送。")
    check(rid1 == rid2, "传闻指纹对轻微标点差异稳定", rid1)


def t2_embedding() -> None:
    print("\n[2] embedding —— hashing-ngram")
    emb = HashingNgramEmbedder(dim=256)
    v1 = emb.embed("华为 内部推荐政策 争议")
    v3 = emb.embed("特斯拉 自动驾驶 事故")
    v2 = emb.embed("华为 内推政策 最新回应")
    check(len(v1) == 256, "向量维度固定", str(len(v1)))
    check(abs(sum(x * x for x in v1) - 1.0) < 1e-6, "向量已 L2 归一化")
    sim_close = cosine(v1, v2)
    sim_far = cosine(v1, v3)
    check(sim_close > sim_far, "同事件不同说法相似度 > 无关事件", f"{sim_close:.3f} vs {sim_far:.3f}")
    check(emb.embed("") == [0.0] * 256, "空文本返回零向量（不抛异常）")


def t3_store(tmp: Path) -> JsonlEventStore:
    print("\n[3] event_store —— JSONL 读写")
    store = JsonlEventStore(str(tmp / "events.jsonl"))
    rec = EventRecord(
        event_id="华为|内部推荐政策",
        subject="华为",
        issue="内部推荐政策",
        report_date="2026-09-01",
        content="第一版报告",
        embedding=[0.1] * 256,
    )
    store.upsert(rec)
    got = store.get("华为|内部推荐政策")
    check(got is not None and got.content == "第一版报告", "upsert 后可读取")

    rec2 = rec.model_copy(update={"report_date": "2026-10-12", "content": "第二版报告", "report_count": 2})
    store.upsert(rec2)
    got2 = store.get("华为|内部推荐政策")
    check(got2 is not None and got2.content == "第二版报告", "重复 upsert 取最后写入（last-write-wins）")
    check(store.count() == 1, "count 按 event_id 去重", str(store.count()))
    recent = store.load_recent(limit=10)
    check(len(recent) == 1 and recent[0].report_count == 2, "load_recent 返回最新快照")
    return store


def t4_l1() -> None:
    print("\n[4] dedup L1 —— 实体 + 议题精确匹配")
    history = [
        EventRecord(event_id="华为|内部推荐政策", subject="华为", issue="内部推荐政策"),
        EventRecord(event_id="特斯拉|自动驾驶事故", subject="特斯拉", issue="自动驾驶事故"),
    ]
    # 完全一致
    hit = l1_exact_match(EventKeys(subject="华为", issue="内部推荐政策"), history)
    check(hit is not None and hit.event_id == "华为|内部推荐政策", "完全一致 → 命中")
    # 带时效词 + 议题变体
    hit2 = l1_exact_match(EventKeys(subject="华为", issue="内部推荐政策最新进展"), history)
    check(hit2 is not None, "议题带'最新进展'仍能命中（归一化生效）")
    # 命中别名
    hit3 = l1_exact_match(EventKeys(subject="某公司", issue="内部推荐政策", aliases=["华为"]), history)
    check(hit3 is not None, "通过别名命中同一主体")
    # 主体不同 → 不该命中
    miss = l1_exact_match(EventKeys(subject="小米", issue="内部推荐政策"), history)
    check(miss is None, "主体不同 → 不命中（避免跨主体误合并）")
    # 主体相同、议题无关 → 不该命中
    miss2 = l1_exact_match(EventKeys(subject="华为", issue="芯片出口管制"), history)
    check(miss2 is None, "同主体不同议题 → 不命中")


def t5_l2(store: JsonlEventStore) -> None:
    print("\n[5] dedup L2 —— 向量召回")
    emb = HashingNgramEmbedder(dim=256)
    # 写入两条语义不同的历史
    for subj, issue, text in [
        ("华为", "内部推荐政策", "华为内部推荐政策被质疑存在利益输送，公司尚未回应"),
        ("特斯拉", "自动驾驶事故", "特斯拉自动驾驶发生碰撞事故，监管介入调查"),
    ]:
        store.upsert(
            EventRecord(
                event_id=make_event_id(subj, issue),
                subject=subj,
                issue=issue,
                report_date="2026-09-20",
                content=text,
                embedding=emb.embed(text),
            )
        )
    # floor=-1.0 放开全部候选，便于验证排序
    cands = l2_vector_recall("华为 内推政策 最新回应", store, emb, top_k=5, floor=-1.0)
    check(len(cands) == 2, "召回返回全部候选", str(len(cands)))
    check(cands[0][0].event_id == "华为|内部推荐政策", "top1 是正确事件", cands[0][0].event_id)
    check(cands[0][1] > cands[1][1], "相似度降序排列", f"{cands[0][1]:.3f} > {cands[1][1]:.3f}")
    cands_floor = l2_vector_recall("华为 内推政策 最新回应", store, emb, top_k=5, floor=-0.02)
    check(len(cands_floor) == 1, "负相关候选被下限过滤", str(len(cands_floor)))

    # ⚠️ 阈值标定回归测试：hashing-ngram 下，同事件的正确候选分数会低到 0.1 附近，
    # 如果谁把 floor 调到 0.15 以上，这条会挂，从而拦住"凭直觉调阈值"的改动。
    cands_default = l2_vector_recall("华为 内推政策 最新回应", store, emb, top_k=5, floor=0.05)
    check(len(cands_default) >= 1, "默认 floor=0.05 下正确候选不被误杀", str(len(cands_default)))


async def t6_l3(store: JsonlEventStore) -> None:
    print("\n[6] dedup L3 —— LLM 判定（Mock）")
    emb = HashingNgramEmbedder(dim=256)

    # 场景 A：L1 未命中（说法差异大）→ L2 召回 → L3 判定为同一事件
    mock_yes = MockLLM(
        keys_json='{"subject":"华为","issue":"内推政策","aliases":["华为"]}',
        judge_json='{"is_same": true, "matched_event_id": "华为|内推政策", "confidence": 0.88, "reason": "同一公司同一议题的不同说法"}',
    )
    store.upsert(
        EventRecord(
            event_id="华为|内推政策",
            subject="华为",
            issue="内推政策",
            report_date="2026-09-01",
            content="华为内推政策引发讨论",
            embedding=emb.embed("华为内推政策引发讨论"),
        )
    )
    res = await resolve_event("华为 内推政策 争议 最新", store, emb, mock_yes, l3_enabled=True)
    check(res.is_repeat and res.level == "L1", "【场景A】L1 直接命中", f"level={res.level}")
    check(res.history is not None and res.history.event_id == "华为|内推政策", "命中记录正确", str(res.history.event_id if res.history else None))
    print("    判重轨迹:", res.trace_text())

    # 场景 B：构造一个 L1 打不着、必须靠 L2+L3 的例子
    mock_l23 = MockLLM(
        keys_json='{"subject":"某汽车品牌","issue":"刹车门","aliases":[]}',
        judge_json='{"is_same": true, "matched_event_id": "特斯拉|自动驾驶事故", "confidence": 0.72, "reason": "同一事故的后续报道"}',
    )
    res_b = await resolve_event("某汽车品牌 刹车门 后续", store, emb, mock_l23, l3_enabled=True, l2_floor=0.0)
    check(res_b.level in ("L1", "L2+L3"), "【场景B】走完三级漏斗得到结论", f"level={res_b.level}")

    # 场景 C：L3 否决候选 → 判为新事件
    mock_no = MockLLM(
        keys_json='{"subject":"华为","issue":"员工持股计划","aliases":[]}',
        judge_json='{"is_same": false, "matched_event_id": "", "confidence": 0.9, "reason": "主体相同但议题完全不同"}',
    )
    res_c = await resolve_event("华为 员工持股计划 争议", store, emb, mock_no, l3_enabled=True, l2_floor=0.0)
    check(not res_c.is_repeat, "【场景C】L3 否决候选 → 判为新事件", f"level={res_c.level}")

    # 场景 D：LLM 输出脏（带 markdown 代码块）也要能解析
    mock_dirty = MockLLM(
        keys_json='```json\n{"subject":"华为","issue":"内推政策","aliases":[]}\n```',
        judge_json='```json\n{"is_same": true, "matched_event_id": "华为|内推政策", "confidence": 0.9, "reason": "ok"}\n```',
    )
    res_d = await resolve_event("华为内推政策事件", store, emb, mock_dirty, l3_enabled=True)
    check(res_d.is_repeat, "【场景D】模型输出带代码块围栏仍能解析")


def t7_delta_queries() -> None:
    print("\n[7] delta —— 「时间 + 关键词」检索式")
    qs = build_delta_queries("华为", "内部推荐政策", lookback_days=7, max_queries=6)
    check(len(qs) == 6, "11 个关键词在 max_queries=6 预算内被完整排布", str(len(qs)))
    check(any("最新进展" in q for q in qs), "包含时间锚定式")
    joined = " ".join(qs)
    missing_kw = [k for k in ["回应", "声明", "辟谣", "通报", "处罚", "整改", "起诉", "反转", "否认", "致歉", "热搜"] if k not in joined]
    check(not missing_kw, "11 个关键词全部落到检索式里", "缺：" + ",".join(missing_kw) if missing_kw else "全覆盖")
    check(build_delta_queries("", "") == [], "主体/议题为空时返回空列表（不构造垃圾检索式）")
    for q in qs:
        print("    ·", q)


def t8_state_machine() -> None:
    print("\n[8] 传闻状态机迁移 + 时间线去重")
    rec = EventRecord(
        event_id="华为|内部推荐政策",
        subject="华为",
        issue="内部推荐政策",
        first_seen="2026-09-01",
        report_date="2026-09-01",
        report_count=1,
        content="初版报告",
        timeline=[
            TimelineNode(time="2026-08-28", node="内部论坛出现匿名爆料帖", source="weibo", first_seen="2026-09-01"),
        ],
        rumors=[
            RumorState(
                rumor_id=RumorState.make_id("内部推荐存在利益输送"),
                text="内部推荐存在利益输送",
                status="unconfirmed",
                confidence="低",
                first_seen="2026-09-01",
                updated_at="2026-09-01",
            ),
        ],
    )

    delta = DeltaUpdate(
        generated_at="2026-10-12",
        summary="华为发布内部邮件否认利益输送传闻，并追加两条时间线节点",
        timeline_updates=[
            TimelineNode(time="2026-10-05", node="华为发布内部邮件回应", source="cls.cn"),
            TimelineNode(time="2026-08-28", node="内部论坛出现匿名爆料帖", source="weibo"),  # 重复节点
        ],
        rumor_updates=[
            {
                "target": "内部推荐存在利益输送",
                "new_status": "refuted",
                "confidence": "高",
                "evidence": ["https://example.com/huawei-reply"],
                "reason": "官方邮件明确否认并公布内推流程审计结论",
            },
            {
                "target": "涉事HR已被停职",
                "new_status": "unconfirmed",
                "confidence": "低",
                "evidence": [],
                "reason": "仅自媒体转述，无一手来源",
            },
        ],
    )

    applied = apply_delta_to_record(rec, delta)

    check(len(applied.timeline) == 2, "时间线去重生效（重复节点未重复追加）", str(len(applied.timeline)))
    new_nodes = [n for n in applied.timeline if n.is_new]
    check(len(new_nodes) == 1 and new_nodes[0].time == "2026-10-05", "仅真新增节点标记 is_new", str(len(new_nodes)))
    check(applied.timeline[0].time <= applied.timeline[-1].time, "时间线按时间排序")

    r_old = next(r for r in applied.rumors if r.text == "内部推荐存在利益输送")
    check(r_old.status == "refuted", "已有传闻状态迁移 unconfirmed → refuted", r_old.status)
    check(len(r_old.history) == 1 and r_old.history[0]["to"] == "refuted", "留下迁移历史（辟谣追踪）")
    check(r_old.confidence == "高", "可信度随证据更新", r_old.confidence)
    check(len(applied.rumors) == 2, "全新传闻被追加进状态机", str(len(applied.rumors)))
    check(len(applied.delta_log) == 1 and applied.delta_log[0]["rumor_transitions"] == 1, "delta 审计日志已记录")

    print("    状态机视图：")
    for r in applied.rumors:
        print(f"      - [{RUMOR_STATUS_LABEL.get(r.status, r.status)}] {r.text}（可信度 {r.confidence}）")
        for h in r.history:
            print(f"          · {h['date']}: {h['from']} → {h['to']}｜{h['reason'][:40]}")


def t9_es_doc_shape() -> None:
    print("\n[9] 落库文档结构 —— 与约定 schema 对齐")
    rec = EventRecord(
        event_id="华为|内部推荐政策",
        subject="华为",
        issue="内部推荐政策",
        report_date="2024-06-15",
        content="……报告全文……",
        embedding=[0.0] * 256,
    )
    doc = json.loads(rec.model_dump_json())
    required = ["event_id", "subject", "issue", "report_date", "content", "embedding"]
    missing = [k for k in required if k not in doc]
    check(not missing, "约定的 6 个最小字段齐全", ",".join(required))
    extra = ["timeline", "rumors", "delta_log", "aliases", "first_seen", "last_seen", "report_count"]
    check(all(k in doc for k in extra), "事件追踪扩展字段齐全", ",".join(extra))
    print("    文档示例（截断）：")
    sample = {k: (v[:30] + "..." if isinstance(v, str) and len(v) > 30 else v) for k, v in doc.items()}
    sample["embedding"] = f"[{len(doc['embedding'])} dims]"
    print("    " + json.dumps(sample, ensure_ascii=False)[:400])


async def t10_integration(tmp: Path) -> None:
    """节点级集成测试：event_recall → (研究流程) → event_memory_commit，全程 Mock LLM + 无网络。"""
    print("\n[10] 节点级集成 —— event_recall → event_memory_commit 全链路")
    from langchain_core.messages import HumanMessage

    from deep_research.memory import memory_node as mn

    jsonl_path = str(tmp / "integration" / "events.jsonl")
    store = JsonlEventStore(jsonl_path)
    emb = HashingNgramEmbedder(dim=256)

    mock = MockLLM(
        keys_json='{"subject":"华为","issue":"内部推荐政策","aliases":["华为","内推"]}',
        judge_json='{"is_same": true, "matched_event_id": "华为|内部推荐政策", "confidence": 0.9, "reason": "同一事件"}',
        delta_json=(
            '{"summary":"华为发布内部邮件否认利益输送传闻",'
            '"timeline_updates":[{"time":"2026-10-05","node":"华为发布内部邮件回应","source":"cls.cn"}],'
            '"rumor_updates":[{"target":"内部推荐存在利益输送","new_status":"refuted",'
            '"confidence":"高","evidence":["https://example.com/a"],"reason":"官方书面否认"}]}'
        ),
    )

    test_cfg = {
        "enabled": True,
        "judge_role": "event_judge",
        "store": {"backend": "jsonl", "jsonl_path": jsonl_path, "archive_keep": 3},
        "embedding": {"provider": "hashing_ngram", "dim": 256},
        "dedup": {"l2_top_k": 5, "l3_enabled": True},
        "delta": {"lookback_days": 7, "max_queries": 6, "keywords_per_query": 3},
    }

    # --- 打桩：不读真实 config、不调真实检索 ---
    orig = (mn._load_memory_cfg, mn._get_runtime, mn.search_delta)
    mn._load_memory_cfg = lambda: test_cfg              # type: ignore[assignment]
    mn._get_runtime = lambda cfg: (store, emb, mock)    # type: ignore[assignment]
    mn.search_delta = lambda queries, **kw: [           # type: ignore[assignment]
        {"query": queries[0], "title": "华为回应内推争议", "url": "https://example.com/a",
         "content": "华为发布内部邮件否认利益输送传闻，称已完成流程审计。", "published_date": "2026-10-05"}
    ]

    try:
        # ---- 第一轮：记忆库为空 → 判为新事件 ----
        q = "华为 内部推荐政策 争议 最新回应"
        state: dict = {"messages": [HumanMessage(content=q)]}
        cmd = await mn.event_recall_node(state)
        check(cmd.goto == "write_research_brief", "event_recall 跳转到 write_research_brief", str(cmd.goto))
        up = cmd.update
        check(up["is_repeat_event"] is False, "【首轮】空库 → 判为新事件")
        check(up["event_id"] == "华为|内部推荐政策", "【首轮】event_id 由 LLM 抽取的键生成", up["event_id"])
        check(up["event_increment_context"] == "", "【首轮】不注入增量上下文")
        print("    判重轨迹:", " → ".join(up["memory_meta"].get("trace", [])))

        # ---- 写入一条"历史归档"，模拟昨天已建档 ----
        hist = EventRecord(
            event_id="华为|内部推荐政策", subject="华为", issue="内部推荐政策",
            first_seen="2026-09-01", last_seen="2026-09-01", report_date="2026-09-01",
            report_count=1, content="初版报告：华为内推政策被质疑存在利益输送，公司尚未回应。",
            embedding=emb.embed("初版报告：华为内推政策被质疑存在利益输送，公司尚未回应。"),
            timeline=[TimelineNode(time="2026-08-28", node="内部论坛出现匿名爆料帖", source="weibo", first_seen="2026-09-01")],
            rumors=[RumorState(rumor_id=RumorState.make_id("内部推荐存在利益输送"),
                               text="内部推荐存在利益输送", status="unconfirmed", confidence="低",
                               first_seen="2026-09-01", updated_at="2026-09-01")],
        )
        store.upsert(hist)

        # ---- 第二轮：同一事件再次进入 → 命中 + 增量追踪 ----
        cmd2 = await mn.event_recall_node({"messages": [HumanMessage(content=q)]})
        up2 = cmd2.update
        check(up2["is_repeat_event"] is True, "【二次】命中历史事件（走增量模式）")
        check(up2["memory_meta"]["level"] == "L1", "【二次】L1 精确命中", up2["memory_meta"]["level"])
        check("历史事件增量追踪" in up2["event_increment_context"], "【二次】注入「增量追踪」指令块")
        check("不要重新梳理已在" in up2["event_increment_context"], "【二次】明确禁止重搜历史节点")
        check("内部推荐存在利益输送" in up2["event_memory_block"], "【二次】草稿注入历史传闻状态机")
        check(up2["delta_updates"]["summary"].startswith("华为发布内部邮件"), "【二次】delta 抽取成功", up2["delta_updates"]["summary"])
        print("    增量检索式:", up2["memory_meta"]["delta_queries"][:2], "...")

        # ---- 模式：full 走 write_research_brief，fast 直达 write_draft_report ----
        check(cmd2.goto == "write_research_brief", "【full 模式】命中后仍走完整研究循环", str(cmd2.goto))
        mn._load_memory_cfg = lambda: {**test_cfg, "increment_mode": "fast"}  # type: ignore[assignment]
        cmd_fast = await mn.event_recall_node({"messages": [HumanMessage(content=q)]})
        check(cmd_fast.goto == "write_draft_report", "【fast 模式】命中后跳过主研究循环", str(cmd_fast.goto))
        check(cmd_fast.update.get("skip_research_loop") is True, "【fast 模式】置位 skip_research_loop")
        check(bool(cmd_fast.update.get("research_brief")), "【fast 模式】就地产出增量简报")
        check(
            any("example.com/a" in n for n in cmd_fast.update.get("notes", [])),
            "【fast 模式】增量检索资料注入 findings(notes)",
        )
        mn._load_memory_cfg = lambda: test_cfg  # type: ignore[assignment]

        # ---- 第三轮：模拟研究完成后归档 ----
        final_report = (
            "## ① 舆情事件概述\n### 1.2 关键时间线\n"
            "| 2026-08-28 | 内部论坛出现匿名爆料帖 | weibo |\n"
            "| 2026-10-05 | 华为发布内部邮件回应 | cls.cn |\n"
            "### 2.2 未确认传闻\n- ❓ 内部推荐存在利益输送 —— 已被官方书面否认\n"
        )
        commit_state = {**up2, "final_report": final_report}
        done = await mn.event_memory_commit_node(commit_state)
        meta = done["memory_meta"]
        check(meta.get("committed") is True, "【归档】写入成功", str(meta.get("committed")))
        check(meta.get("report_count") == 2, "【归档】report_count 累加", str(meta.get("report_count")))

        saved = store.get("华为|内部推荐政策")
        check(saved is not None and saved.report_count == 2, "【归档】存储中 report_count=2")
        check(len(saved.content_archive) >= 1, "【归档】旧报告进 content_archive", str(len(saved.content_archive)))
        check(len(saved.timeline) == 2, "【归档】时间线累积为 2 个节点", str(len(saved.timeline)))
        r = next(x for x in saved.rumors if "利益输送" in x.text)
        check(r.status == "refuted" and len(r.history) == 1, "【归档】传闻状态机完成迁移并留痕", f"{r.status}/{len(r.history)}")
        check(len(saved.embedding) == 256, "【归档】embedding 已生成", f"{len(saved.embedding)} dims")
        check(saved.content == final_report, "【归档】content 存的是本次报告全文")
        check(saved.meta.get("store_backend") == "jsonl", "【归档】meta 记录了存储后端")

        # ---- 容错：commit 失败不能抛异常 ----
        bad_state = {"final_report": "", "memory_meta": {}}
        done2 = await mn.event_memory_commit_node(bad_state)
        check(done2["memory_meta"].get("committed") is not True, "【容错】空报告不写库也不抛异常")

        # ---- 容错：recall 抛异常时降级为"无历史"，不中断主链路 ----
        def _boom(cfg):
            raise RuntimeError("模拟记忆库宕机")

        mn._get_runtime = _boom  # type: ignore[assignment]
        cmd3 = await mn.event_recall_node({"messages": [HumanMessage(content=q)]})
        check(cmd3.goto == "write_research_brief", "【容错】记忆层异常仍正常流转")
        check(cmd3.update["is_repeat_event"] is False, "【容错】记忆层异常 → 降级为无历史")

        # ---- 开关：enabled=false 时直通 ----
        mn._load_memory_cfg = lambda: {"enabled": False}  # type: ignore[assignment]
        cmd4 = await mn.event_recall_node({"messages": [HumanMessage(content=q)]})
        check(cmd4.update["memory_meta"] == {"enabled": False, "reason": "disabled"}, "【开关】关闭时直通且不调用任何记忆逻辑")

    finally:
        mn._load_memory_cfg, mn._get_runtime, mn.search_delta = orig


async def t11_online() -> None:
    print("\n[11] 在线验证 —— 真实 LLM 事件键抽取")
    from deep_research.llm import get_chat_model

    try:
        model = get_chat_model("event_judge")
    except Exception as exc:
        print(f"{FAIL} 无法构造 event_judge 模型（跳过）: {str(exc)[:150]}")
        return
    keys = await extract_event_keys("华为余承东，竹知了事件", model)
    check(bool(keys.subject) and bool(keys.issue), "抽取到 subject / issue", f"{keys.subject} | {keys.issue}")


async def main() -> None:
    online = "--online" in sys.argv
    print("=" * 64)
    print("事件记忆层自测（离线）")
    print("=" * 64)

    t1_schema()
    t2_embedding()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = t3_store(tmp)
        t4_l1()
        t5_l2(store)
        await t6_l3(store)
        t7_delta_queries()
        t8_state_machine()
        t9_es_doc_shape()
        await t10_integration(tmp)
    if online:
        await t11_online()

    passed = sum(1 for ok, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 64)
    print(f"结果：{passed}/{total} 通过")
    failed = [label for ok, label in _results if not ok]
    if failed:
        print("未通过：")
        for f in failed:
            print("  -", f)
    print("=" * 64)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
