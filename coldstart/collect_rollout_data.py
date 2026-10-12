#***********************************************
#      Filename: collect_rollout_data.py
#   Description: Critic 冷启动数据收集脚本
#                导入项目 LLM 和 prompt，自己实现红队循环
#                生成 (prompt, return) 训练对，供 value head 预训练用
#
#   用法（AutoDL 上）：
#     source /root/autodl-tmp/envs/research/bin/activate
#     cd /root/autodl-tmp/project-Deep_Research
#     python coldstart/collect_rollout_data.py
#***********************************************

import asyncio
import json
import os
import re
import sys
import time

# 把项目根目录加入 path，以便导入 deep_research 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from langchain_core.messages import HumanMessage
from deep_research.llm import get_chat_model
from deep_research.prompts import RED_TEAM_PROMPT, DRAFT_EVALUATOR_PROMPT

# ===== 配置 =====
MAX_ROUNDS = 10          # 红队最大轮次
GAMMA = 0.99             # return 折扣因子
LAMBDA_R = 0.02          # 轮次惩罚系数

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "rollout_data.json")

# 10 个舆情主题（覆盖不同领域，保证数据多样性）
TOPICS = [
    "余承东竹知了事件",
    "某新能源车企电池自燃舆情",
    "某互联网大厂暴力裁员舆情",
    "某直播带货主播偷税漏税事件",
    "某食品企业食品安全问题舆情",
    "某科技公司用户数据泄露事件",
    "某明星代言翻车舆情",
    "某 P2P 金融平台暴雷跑路事件",
    "某教育机构大规模退费难舆情",
    "某医疗美容机构手术事故舆情",
]

# ===== 模型初始化（走 config.yml 的角色配置）=====
print("初始化模型...")
draft_model = get_chat_model("draft")          # 通用实例 8000
red_team_model = get_chat_model("red_team")    # 红队实例 8001
evaluator_model = get_chat_model("evaluator")  # 通用实例 8000
print("模型初始化完成\n")


async def call_model(model, prompt: str) -> str:
    """调用 LLM，返回文本"""
    resp = await model.ainvoke([HumanMessage(content=prompt)])
    return resp.content


def parse_score(text: str) -> float:
    """从评估器输出解析分数，返回 0-100 分"""
    json_match = re.search(r'\{.*"comprehensiveness_score".*\}', text, re.DOTALL)
    if json_match:
        parsed = json.loads(json_match.group())
        avg = (
            parsed.get("comprehensiveness_score", 5)
            + parsed.get("accuracy_score", 5)
            + parsed.get("coherence_score", 5)
        ) / 3
        return avg * 10  # 0-10 → 0-100
    return 50.0


def compute_return(
    initial_score: float,
    final_score: float,
    score_history: list[tuple[float, float]],
) -> float:
    """计算 episode return R

    R = 2.0 * (final - initial) / max(100 - initial, 5)      # 终端利用率
      + 1.0 * sum(deltaScore_i / max(100 - score_before_i, 5)) # 过程利用率
      - lambda_r * sum(max(0, i - 5))                          # 轮次惩罚
    """
    terminal = 2.0 * (final_score - initial_score) / max(100 - initial_score, 5)

    process = 0.0
    for score_before, score_after in score_history:
        delta = score_after - score_before
        process += delta / max(100 - score_before, 5)

    penalty = LAMBDA_R * sum(max(0, i - 5) for i in range(1, len(score_history) + 1))

    return terminal + process - penalty


async def run_one_rollout(topic: str) -> list[dict]:
    """跑一条完整的红队 rollout，返回训练对列表"""

    brief = f"舆情事件：{topic}。请从法务审计角度进行深度分析，覆盖事实梳理、法律风险识别、法务处置建议。"

    # 1. 生成草稿
    draft_prompt = (
        f"请针对以下舆情事件写一份法务审计报告草稿，严格包含四部分：\n"
        f"① 舆情事件概述\n② 关键事实梳理\n③ 法律风险识别\n④ 法务处置建议\n\n"
        f"舆情事件：{topic}"
    )
    draft = await call_model(draft_model, draft_prompt)

    # 2. 初始打分
    eval_prompt = DRAFT_EVALUATOR_PROMPT.format(research_brief=brief, draft_report=draft)
    eval_prompt += '\n\n请以JSON格式返回：{"comprehensiveness_score": 0-10, "accuracy_score": 0-10, "coherence_score": 0-10, "reason": "..."}'
    eval_resp = await call_model(evaluator_model, eval_prompt)
    initial_score = parse_score(eval_resp)

    # 3. 红队循环
    history = []
    saved_prompts = []
    score_history = []
    current_score = initial_score

    for round_idx in range(MAX_ROUNDS):
        # 构造红队 prompt（draft + history）
        prompt = RED_TEAM_PROMPT.format(research_brief=brief, draft_report=draft)
        if history:
            hist_text = "\n".join([
                f"第{i+1}轮 | 分数：{h['score']:.1f}\n{h['critique']}"
                for i, h in enumerate(history)
            ])
            prompt += f"\n\n## 历史批评记录\n{hist_text}"

        saved_prompts.append(prompt)

        # 红队批评
        critique = await call_model(red_team_model, prompt)

        if "PASS" in critique or len(critique) < 20:
            print(f"  R{round_idx+1}: PASS，结束")
            break

        # 修复草稿
        refine_prompt = (
            f"请根据以下批评意见修复报告草稿，输出修复后的完整草稿。\n\n"
            f"<草稿>\n{draft}\n</草稿>\n\n"
            f"<批评>\n{critique}\n</批评>"
        )
        draft = await call_model(draft_model, refine_prompt)

        # 打分
        eval_prompt = DRAFT_EVALUATOR_PROMPT.format(research_brief=brief, draft_report=draft)
        eval_prompt += '\n\n请以JSON格式返回：{"comprehensiveness_score": 0-10, "accuracy_score": 0-10, "coherence_score": 0-10, "reason": "..."}'
        eval_resp = await call_model(evaluator_model, eval_prompt)
        new_score = parse_score(eval_resp)

        score_history.append((current_score, new_score))
        history.append({"critique": critique[:500], "score": new_score})
        current_score = new_score
        print(f"  R{round_idx+1}: {score_history[-1][0]:.1f} -> {new_score:.1f}")

    # 4. 计算 return
    R = compute_return(initial_score, current_score, score_history)

    # 5. 构造训练对：每个 prompt 配 gamma 折扣后的 R
    T = len(saved_prompts)
    train_pairs = []
    for t, p in enumerate(saved_prompts):
        label = (GAMMA ** (T - 1 - t)) * R
        train_pairs.append({
            "prompt": p,
            "return": round(label, 6),
            "round": t + 1,
            "topic": topic,
        })

    print(f"  => {topic}: {initial_score:.1f} -> {current_score:.1f} | R={R:.4f} | rounds={T} | pairs={len(train_pairs)}")
    return train_pairs


async def main():
    print("=" * 60)
    print("Critic 冷启动数据收集")
    print(f"主题数: {len(TOPICS)} | 最大轮次: {MAX_ROUNDS}")
    print("=" * 60)

    all_pairs = []
    t0 = time.time()

    for i, topic in enumerate(TOPICS):
        print(f"\n[{i+1}/{len(TOPICS)}] {topic}")
        try:
            pairs = await run_one_rollout(topic)
            all_pairs.extend(pairs)
        except Exception as e:
            print(f"  ERROR: {e}")

    elapsed = time.time() - t0

    # 保存
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_pairs, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"收集完成：{len(all_pairs)} 条训练对")
    print(f"耗时: {elapsed/60:.1f} 分钟")
    print(f"保存到: {OUTPUT_FILE}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
