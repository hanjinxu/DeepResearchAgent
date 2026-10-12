"""
Baseline 测试脚本
舆情领域：测试当前 research_agent 的搜索轮数分布和报告质量
"""

import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage

from deep_research.agents.research_agent import researcher_agent
from deep_research.agents.evaluator_agent import evaluate_draft_quality


# =====================================================================
# 舆情领域 Query 列表 — 30条
# 按场景分类，覆盖简单/中等/复杂三种预期搜索深度
# =====================================================================
BASELINE_QUERIES = [
    # ========== 1. 品牌声誉监测（简单～中等，预期2-5轮） ==========
    {
        "id": "brand_001",
        "research_topic": "2025年比亚迪在海外市场的舆情口碑和主要争议事件",
        "expected_depth": "medium",
        "category": "brand_reputation",
    },
    {
        "id": "brand_002",
        "research_topic": "近期特斯拉在中国市场的消费者投诉热点及官方回应",
        "expected_depth": "medium",
        "category": "brand_reputation",
    },
    {
        "id": "brand_003",
        "research_topic": "苹果公司2025年产品发布后的社交媒体舆情分析",
        "expected_depth": "medium",
        "category": "brand_reputation",
    },
    {
        "id": "brand_004",
        "research_topic": "农夫山泉2024年舆情危机的时间线和处理措施",
        "expected_depth": "simple",
        "category": "brand_reputation",
    },
    {
        "id": "brand_005",
        "research_topic": "华为Mate系列新品发布后的舆论反馈综述",
        "expected_depth": "medium",
        "category": "brand_reputation",
    },
    # ========== 2. 舆情危机事件（中等～复杂，预期3-6轮） ==========
    {
        "id": "crisis_001",
        "research_topic": "2025年多家车企数据泄露事件的舆情发展脉络与影响评估",
        "expected_depth": "complex",
        "category": "crisis",
    },
    {
        "id": "crisis_002",
        "research_topic": "近期互联网大厂裁员舆情：涉及企业、裁员规模及舆论反应",
        "expected_depth": "complex",
        "category": "crisis",
    },
    {
        "id": "crisis_003",
        "research_topic": "直播带货头部主播舆情危机回顾：从李佳琦到小杨哥的翻车事件",
        "expected_depth": "complex",
        "category": "crisis",
    },
    {
        "id": "crisis_004",
        "research_topic": "2024-2025年教育培训行业监管舆情变化趋势",
        "expected_depth": "medium",
        "category": "crisis",
    },
    {
        "id": "crisis_005",
        "research_topic": "近期房地产行业债务违约事件的舆论关注度变化",
        "expected_depth": "complex",
        "category": "crisis",
    },
    # ========== 3. 行业竞品监测（简单～中等，预期2-4轮） ==========
    {
        "id": "competitive_001",
        "research_topic": "2025年国产大模型厂商的市场声量对比：DeepSeek、通义千问、文心一言",
        "expected_depth": "medium",
        "category": "competitive",
    },
    {
        "id": "competitive_002",
        "research_topic": "抖音与视频号2025年电商业务的口碑对比分析",
        "expected_depth": "medium",
        "category": "competitive",
    },
    {
        "id": "competitive_003",
        "research_topic": "拼多多与京东在低价策略上的舆论反馈差异",
        "expected_depth": "simple",
        "category": "competitive",
    },
    {
        "id": "competitive_004",
        "research_topic": "理想、蔚来、小鹏2025年产品口碑对比：用户满意度与槽点",
        "expected_depth": "complex",
        "category": "competitive",
    },
    {
        "id": "competitive_005",
        "research_topic": "SHEIN与Temu在海外市场的舆论口碑对比",
        "expected_depth": "medium",
        "category": "competitive",
    },
    # ========== 4. 政策与法规舆情（中等～复杂，预期3-5轮） ==========
    {
        "id": "policy_001",
        "research_topic": "2025年数据安全法实施后的企业合规舆情热点",
        "expected_depth": "complex",
        "category": "policy",
    },
    {
        "id": "policy_002",
        "research_topic": "AI生成内容标识政策落地后的公众反应与行业影响",
        "expected_depth": "medium",
        "category": "policy",
    },
    {
        "id": "policy_003",
        "research_topic": "近期游戏版号政策调整的舆论反应分析",
        "expected_depth": "simple",
        "category": "policy",
    },
    {
        "id": "policy_004",
        "research_topic": "跨境电商税收政策2025年变化的舆情关注度",
        "expected_depth": "medium",
        "category": "policy",
    },
    {
        "id": "policy_005",
        "research_topic": "新能源汽车购置税减免政策延续后的舆论反应",
        "expected_depth": "simple",
        "category": "policy",
    },
    # ========== 5. 热点事件追踪（复杂，预期4-6轮） ==========
    {
        "id": "hotspot_001",
        "research_topic": "2025年315晚会曝光企业名单及后续舆情处理效果评估",
        "expected_depth": "complex",
        "category": "hotspot",
    },
    {
        "id": "hotspot_002",
        "research_topic": "近期医疗反腐舆情的代表性案例和舆论关注点",
        "expected_depth": "complex",
        "category": "hotspot",
    },
    {
        "id": "hotspot_003",
        "research_topic": "2025年食品安全热点事件盘点及公众情绪分析",
        "expected_depth": "complex",
        "category": "hotspot",
    },
    {
        "id": "hotspot_004",
        "research_topic": "近期金融监管行政处罚典型案例的舆论传播路径",
        "expected_depth": "medium",
        "category": "hotspot",
    },
    {
        "id": "hotspot_005",
        "research_topic": "2025年娱乐圈塌房事件的舆情周期与发酵规律",
        "expected_depth": "complex",
        "category": "hotspot",
    },
    # ========== 6. 专项深度分析（复杂，预期4-6轮） ==========
    {
        "id": "deep_001",
        "research_topic": "美国对华芯片出口管制2025年最新政策的全球舆论反应",
        "expected_depth": "complex",
        "category": "deep_analysis",
    },
    {
        "id": "deep_002",
        "research_topic": "ESG评级在中国资本市场的影响力与争议舆情综述",
        "expected_depth": "complex",
        "category": "deep_analysis",
    },
    {
        "id": "deep_003",
        "research_topic": "AI替代就业焦虑的舆论演变：从ChatGPT到2025年的主流叙事",
        "expected_depth": "complex",
        "category": "deep_analysis",
    },
    {
        "id": "deep_004",
        "research_topic": "中概股2025年监管环境变化的舆情分析及市场情绪回顾",
        "expected_depth": "complex",
        "category": "deep_analysis",
    },
    {
        "id": "deep_005",
        "research_topic": "2025年全球供应链重构对中国制造业影响的舆论关注点",
        "expected_depth": "complex",
        "category": "deep_analysis",
    },
]

# ===== 配置 =====
RESULTS_DIR = Path("./baseline_results")


# =====================================================================
# 核心函数
# =====================================================================

async def run_single_query(query_item: dict) -> dict:
    """
    跑单条 query，返回搜索轮数 + 报告质量 + 其他指标
    """
    topic = query_item["research_topic"]
    query_id = query_item["id"]
    category = query_item["category"]

    print(f"\n{'=' * 60}")
    print(f"[{query_id}] ({category}) {topic[:60]}...")
    print(f"{'=' * 60}")

    start_time = time.time()

    try:
        result = await researcher_agent.ainvoke({
            "researcher_messages": [HumanMessage(content=topic)],
            "research_topic": topic,
        })
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  [ERROR] 搜索流程异常: {e}")
        return {
            "id": query_id,
            "category": category,
            "topic": topic[:80],
            "avg_score": 0.0,
            "comprehensiveness": 0,
            "accuracy": 0,
            "coherence": 0,
            "elapsed_seconds": round(elapsed, 1),
            "compressed_len": 0,
            "raw_notes_count": 0,
            "error": str(e)[:200],
        }

    elapsed = time.time() - start_time

    # ===== 提取关键指标 =====
    compressed = result.get("compressed_research", "")
    raw_notes = result.get("raw_notes", [])

    # ===== 调用 LLM 评估报告质量 =====
    try:
        eval_result = evaluate_draft_quality(
            research_brief=topic,
            draft_report=compressed if compressed else "（无报告生成）"
        )
        avg_score = (
            eval_result.comprehensiveness_score
            + eval_result.accuracy_score
            + eval_result.coherence_score
        ) / 3.0
    except Exception as e:
        print(f"  [WARN] 评估失败: {e}")
        avg_score = 0.0
        eval_result = type("obj", (object,), {
            "comprehensiveness_score": 0,
            "accuracy_score": 0,
            "coherence_score": 0,
        })()

    # ===== 行为检测 =====
    summary = {
        "id": query_id,
        "category": category,
        "topic": topic[:80],
        "avg_score": round(avg_score, 2),
        "comprehensiveness": eval_result.comprehensiveness_score,
        "accuracy": eval_result.accuracy_score,
        "coherence": eval_result.coherence_score,
        "elapsed_seconds": round(elapsed, 1),
        "compressed_len": len(compressed),
        "raw_notes_count": len(raw_notes),
        "error": "",
    }

    print(f"  → 质量评分: {avg_score:.1f} | 耗时: {elapsed:.0f}s")
    return summary


def print_statistics(all_results: list[dict]):
    """打印统计摘要"""
    valid = [r for r in all_results if not r.get("error")]

    print("\n\n" + "=" * 60)
    print("📊 Baseline 测试结果统计")
    print("=" * 60)
    print(f"✅ 成功: {len(valid)}/{len(all_results)}")

    if not valid:
        print("❌ 无有效结果")
        return

    scores = [r["avg_score"] for r in valid]

    print(f"\n📊 报告质量评分:")
    print(f"   平均分: {sum(scores) / len(scores):.2f}")
    print(f"   最高分: {max(scores):.2f}")
    print(f"   最低分: {min(scores):.2f}")

    # 按类别统计
    print(f"\n📂 按类别统计:")
    cats = {}
    for r in valid:
        c = r["category"]
        if c not in cats:
            cats[c] = {"scores": [], "count": 0}
        cats[c]["scores"].append(r["avg_score"])
        cats[c]["count"] += 1

    for c, d in sorted(cats.items()):
        avg_s = sum(d["scores"]) / len(d["scores"])
        print(f"   {c:20s} | 数量: {d['count']:2d} | 平均分: {avg_s:.2f}")

    # ===== 核心对比指标 =====
    print(f"\n🎯 核心对比指标:")
    print(f"   ① 平均报告质量:         {sum(scores) / len(scores):.2f}")
    print(f"   ② 最高报告质量:         {max(scores):.2f}")
    print(f"   ③ 最低报告质量:         {min(scores):.2f}")


async def main():
    print("=" * 60)
    print("🔬 Baseline 测试 — 舆情领域 Research Agent")
    print(f"   Query 数量: {len(BASELINE_QUERIES)}")
    print(f"   开始时间:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)

    all_results = []
    for i, query_item in enumerate(BASELINE_QUERIES):
        print(f"\n--- [{i + 1}/{len(BASELINE_QUERIES)}] ---")
        try:
            result = await run_single_query(query_item)
            all_results.append(result)
        except Exception as e:
            print(f"  [FATAL] {query_item['id']} 崩溃: {e}")
            all_results.append({
                "id": query_item["id"],
                "category": query_item["category"],
                "topic": query_item["research_topic"][:80],
                "avg_score": 0.0,
                "comprehensiveness": 0,
                "accuracy": 0,
                "coherence": 0,
                "elapsed_seconds": 0,
                "compressed_len": 0,
                "raw_notes_count": 0,
                "error": str(e)[:200],
            })

    # 打印统计
    print_statistics(all_results)

    # 导出 CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"baseline_{timestamp}.csv"
    fieldnames = [
        "id", "category", "topic",
        "avg_score", "comprehensiveness", "accuracy", "coherence",
        "elapsed_seconds", "compressed_len", "raw_notes_count", "error",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n💾 CSV 已保存: {csv_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
