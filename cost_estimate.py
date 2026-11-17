#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
压测成本估算：Deep Research Agent 单条舆情事件的 token 消耗与费用

所有结构参数都取自代码里的真实常量：
  - max_researcher_iterations = 15          (agents/supervisor.py:239)
  - max_concurrent_researchers = 3          (agents/supervisor.py:240)
  - MAX_CRITIC = 3                          (agents/red_team_agent.py:23)
  - max_results = 3                         (tools/tool.py，每次搜索返回条数)
  - include_raw_content = True              (tools/tool.py，每条结果都触发一次摘要 LLM 调用)
  - MAX_CONTEXT_LENGTH = 250000 字符        (tools/tool.py:29，单页 raw_content 上限)
  - prompt 体积由静态统计得到（见 PROMPT_TOKENS）

输入 token 按「可缓存性」分三档，因为 Agent 场景 prefix 复用差异极大：
  static  固定 prompt 模板 / system —— 跨事件高度重复，命中率默认 0.90
  hist    累积的对话历史（ReAct / supervisor 每轮都重发）—— 命中率默认 0.70
  fresh   新网页正文、新 findings —— 基本无复用，命中率 0.00

用法:
    python cost_estimate.py                      # 三档场景 + 1000 条总账
    python cost_estimate.py --runs 5000
    python cost_estimate.py --scenario 基准
    python cost_estimate.py --no-cache           # 关掉缓存折扣
"""

from __future__ import annotations

import argparse

# ---------------------------------------------------------------------------
# 一、静态 prompt 体积（实测统计，单位 token）
# ---------------------------------------------------------------------------
PROMPT_TOKENS = {
    "event_key_extract": 1769,
    "research_brief": 934,
    "supervisor_sys": 1985,
    "researcher_sys": 651,
    "summarize_sys": 1218,
    "compress_sys": 943,
    "draft_report": 1225,
    "red_team": 1968,
    "evaluator": 338,
    "final_report": 1595,
    "final_legal": 4227,      # 舆情/法务模板，本项目主要走这条
}

# ---------------------------------------------------------------------------
# 二、体积假设
# ---------------------------------------------------------------------------
PAGE_TOKENS = 4200          # 一次摘要调用的网页正文（实测 raw_content 5k~30k 字符，取中枢）
SUMMARY_OUT = 600
TOOL_MSG_PER_RESULT = 700   # 一条摘要结果落进 tool message 的体积
AI_TOOLCALL_OUT = 150
BRIEF_OUT = 700
DRAFT_OUT = 1600
COMPRESS_OUT = 900
REDTEAM_OUT = 350
EVAL_OUT = 180
FINAL_OUT = 3200
DELTA_OUT = 450
CLARIFY_TOKENS = 350

P_L3_JUDGE = 0.6

# 缓存命中率
HIT_STATIC = 0.90
HIT_HIST = 0.70
HIT_FRESH = 0.00

CACHE_READ_DISCOUNT = 0.10   # 缓存命中按标价的 10% 计（DeepSeek 0.30 vs 2.00 ≈ 15%，取保守）


def simulate(sup_rounds: int, researchers_per_round: float, react_turns: int,
             searches_per_turn: int = 1, results_per_search: int = 3,
             is_legal: bool = True) -> dict:
    """模拟单条事件，返回 in/out/calls/searches/breakdown，并标注输入的缓存分档。"""
    tin = tout = calls = searches = 0
    bd: dict[str, list] = {}

    def add(name: str, i: int, o: int, n: int = 1, cache: str = "static") -> None:
        nonlocal tin, tout, calls
        tin += i * n
        tout += o * n
        calls += n
        rec = bd.setdefault(name, [0, 0, 0, {"static": 0, "hist": 0, "fresh": 0}])
        rec[0] += i * n
        rec[1] += o * n
        rec[2] += n
        rec[3][cache] += i * n

    # --- 阶段 0：事件记忆层 ---
    add("记忆·事件键抽取", PROMPT_TOKENS["event_key_extract"], 120)
    if P_L3_JUDGE > 0:
        add("记忆·L3判重", 1400 + 5 * 220, 180, 1, cache="hist")
    add("记忆·delta抽取", 6800, DELTA_OUT, 1, cache="hist")

    # --- 阶段 1-2：简报 + 初稿 ---
    add("研究简报", PROMPT_TOKENS["research_brief"] + CLARIFY_TOKENS, BRIEF_OUT)
    add("报告初稿", PROMPT_TOKENS["draft_report"] + 400, DRAFT_OUT, 1, cache="hist")

    # --- 阶段 3：supervisor 主循环 ---
    n_researchers = int(round(sup_rounds * researchers_per_round))
    sup_accum = 0
    tool_msg_per_round = 200 + researchers_per_round * COMPRESS_OUT
    for _ in range(sup_rounds):
        add("主线·supervisor", PROMPT_TOKENS["supervisor_sys"] + sup_accum, 180, 1, cache="hist")
        sup_accum += tool_msg_per_round + 60

    # --- 阶段 4：researcher 子 agent ---
    for _ in range(n_researchers):
        accum = 0
        for _t in range(react_turns):
            add("子agent·ReAct", PROMPT_TOKENS["researcher_sys"] + accum, AI_TOOLCALL_OUT, 1, cache="hist")
            n_sum = searches_per_turn * results_per_search
            # 摘要：system 部分可缓存，网页正文是全新的
            add("子agent·网页摘要", PROMPT_TOKENS["summarize_sys"], SUMMARY_OUT, n_sum, cache="static")
            add("子agent·网页正文", PAGE_TOKENS, 0, n_sum, cache="fresh")
            searches += searches_per_turn
            accum += AI_TOOLCALL_OUT + searches_per_turn * results_per_search * TOOL_MSG_PER_RESULT
        add("子agent·压缩", PROMPT_TOKENS["compress_sys"] + accum, COMPRESS_OUT, 1, cache="hist")

    # --- 阶段 5-6：对抗 ---
    n_red = max(1, round(sup_rounds * 0.25))
    add("对抗·红队", PROMPT_TOKENS["red_team"] + BRIEF_OUT + DRAFT_OUT, REDTEAM_OUT, n_red, cache="hist")
    n_eval = max(1, round(sup_rounds * 0.3))
    add("对抗·评分", PROMPT_TOKENS["evaluator"] + BRIEF_OUT + DRAFT_OUT, EVAL_OUT, n_eval, cache="hist")

    # --- 阶段 7-8：终稿 + 归档 ---
    findings = n_researchers * COMPRESS_OUT + sup_rounds * 200
    tpl = PROMPT_TOKENS["final_legal"] if is_legal else PROMPT_TOKENS["final_report"]
    add("最终报告", tpl, FINAL_OUT)
    add("最终报告·findings", findings + DRAFT_OUT + BRIEF_OUT, 0, 1, cache="fresh")
    add("记忆·归档", 800 + FINAL_OUT, DELTA_OUT, 1, cache="hist")

    return {"in": tin, "out": tout, "calls": calls, "searches": searches, "breakdown": bd}


def simulate_fast(delta_queries: int = 6) -> dict:
    """fast 模式：命中历史事件，跳过 supervisor 主循环与全部 researcher 子 agent。

    链路 = event_recall（判重 + delta 检索）→ write_draft_report（就地产简）→
           final_report_generation → event_memory_commit
    """
    tin = tout = calls = 0
    searches = delta_queries  # 只做增量检索，不做全量调研
    bd: dict[str, list] = {}

    def add(name: str, i: int, o: int, n: int = 1, cache: str = "static") -> None:
        nonlocal tin, tout, calls
        tin += i * n
        tout += o * n
        calls += n
        rec = bd.setdefault(name, [0, 0, 0, {"static": 0, "hist": 0, "fresh": 0}])
        rec[0] += i * n
        rec[1] += o * n
        rec[2] += n
        rec[3][cache] += i * n

    add("记忆·事件键抽取", PROMPT_TOKENS["event_key_extract"], 120)
    add("记忆·L3判重", 1400 + 5 * 220, 180, 1, cache="hist")
    # delta 检索：delta_queries 条检索式，每条 3 个结果，只取标题+摘要片段 → 无摘要 LLM 调用
    delta_findings = delta_queries * 3 * 220
    add("记忆·delta抽取", 800 + 2000 + delta_findings, DELTA_OUT, 1, cache="hist")
    add("报告初稿(增量简)", PROMPT_TOKENS["draft_report"] + 2500, DRAFT_OUT, 1, cache="hist")
    add("最终报告", PROMPT_TOKENS["final_legal"], FINAL_OUT)
    add("记忆·归档", 800 + FINAL_OUT, DELTA_OUT, 1, cache="hist")

    return {"in": tin, "out": tout, "calls": calls, "searches": searches, "breakdown": bd}


def effective_input(res: dict, use_cache: bool) -> tuple[float, float]:
    """把输入拆成 (计费等价输入, 原始输入)。缓存命中部分按折扣计价。"""
    raw = 0
    billed = 0.0
    hit_map = {"static": HIT_STATIC, "hist": HIT_HIST, "fresh": HIT_FRESH}
    for _name, rec in res["breakdown"].items():
        for cls, tok in rec[3].items():
            raw += tok
            if use_cache and cls != "fresh":
                h = tok * hit_map[cls]
                billed += h * CACHE_READ_DISCOUNT + (tok - h)
            else:
                billed += tok
    return billed, raw


PRICES = [
    # (标签, 输入 元/M, 输出 元/M, 是否支持缓存)
    ("DeepSeek-Flash（空闲时段）", 1.0, 4.0, True),
    ("DeepSeek-Flash（高峰时段）", 2.0, 8.0, True),
    ("DeepSeek-V4-Pro（高峰时段）", 9.0, 27.0, True),
    ("百炼 qwen3.7-flash（控制台价）", 0.2, 0.8, True),
    ("百炼 qwen3.7-flash（保守价）", 1.0, 4.0, True),
    ("百炼 qwen3.7-plus", 2.0, 8.0, True),
]

SCENARIOS = [
    ("轻量", 4, 1.0, 3),
    ("基准", 6, 2.0, 4),
    ("重型", 10, 2.5, 6),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--scenario", default=None, help="只跑某一档：轻量/基准/重型")
    ap.add_argument("--no-cache", action="store_true", help="关闭缓存折扣")
    args = ap.parse_args()

    use_cache = not args.no_cache
    scens = [s for s in SCENARIOS if args.scenario is None or s[0] == args.scenario]
    if not scens:
        raise SystemExit(f"未知场景 {args.scenario}，可选：{[s[0] for s in SCENARIOS]}")

    for name, r, n, t in scens:
        res = simulate(r, n, t)
        billed, raw = effective_input(res, use_cache)
        tin_r, tout_r = res["in"], res["out"]
        print(f"\n{'=' * 84}")
        print(f"【{name}档】R={r}轮 × {n}researcher/轮 × T={t}轮ReAct      "
              f"LLM 调用 {res['calls']} 次 / 搜索 {res['searches']} 次")
        print(f"  单条：原始输入 {tin_r:,} tok → 计费等价输入 {billed:,.0f} tok，输出 {tout_r:,} tok"
              f"（缓存{'开' if use_cache else '关'}）")
        print(f"  {args.runs:,} 条：等效输入 {billed * args.runs / 1e6:,.1f}M tok，"
              f"输出 {tout_r * args.runs / 1e6:,.1f}M tok")
        print(f"{'=' * 84}")
        for label, pi, po, _cache_ok in PRICES:
            c = (billed * args.runs / 1e6) * pi + (tout_r * args.runs / 1e6) * po
            print(f"  {label:<32} ¥{pi:>5.2f}/¥{po:>5.2f}  →  ¥{c:>12,.0f}")
        tav = res["searches"] * args.runs
        print(f"  {'Tavily 搜索（另计）':<32} {tav:,} 次 × $0.008 = ${tav * 0.008:,.0f} "
              f"≈ ¥{tav * 0.008 * 7.1:,.0f}")

    # 基准档拆解
    if args.scenario in (None, "基准"):
        res = simulate(6, 2.0, 4)
        billed, raw = effective_input(res, use_cache)
        print(f"\n{'=' * 84}")
        print("【基准档逐环节拆解 · 单条事件】")
        print(f"{'=' * 84}")
        print(f"{'环节':<22}{'原始输入':>12}{'输出':>10}{'调用':>7}{'占输入比':>10}")
        print("-" * 84)
        for k, rec in sorted(res["breakdown"].items(), key=lambda x: -x[1][0]):
            pct = rec[0] / raw * 100
            print(f"{k:<22}{rec[0]:>12,}{rec[1]:>10,}{rec[2]:>7}{pct:>9.1f}%")
        print("-" * 84)
        print(f"{'合计':<22}{res['in']:>12,}{res['out']:>10,}{res['calls']:>7}{100.0:>9.1f}%")

    # 事件记忆层命中时的 fast 模式对比
    if args.scenario in (None, "基准"):
        full = simulate(6, 2.0, 4)
        fast = simulate_fast()
        fb, _ = effective_input(full, use_cache)
        kb, _ = effective_input(fast, use_cache)
        print(f"\n{'=' * 84}")
        print("【事件记忆层命中历史事件 → fast 模式 · 单条对比】")
        print(f"{'=' * 84}")
        print(f"{'':<16}{'等效输入tok':>14}{'输出tok':>12}{'LLM调用':>10}{'搜索':>8}")
        print("-" * 84)
        print(f"{'全量调研':<16}{fb:>14,.0f}{full['out']:>12,}{full['calls']:>10}{full['searches']:>8}")
        print(f"{'fast 增量':<16}{kb:>14,.0f}{fast['out']:>12,}{fast['calls']:>10}{fast['searches']:>8}")
        print(f"{'节省':<16}{(1 - kb / fb) * 100:>13.1f}%{(1 - fast['out'] / full['out']) * 100:>11.1f}%"
              f"{(1 - fast['calls'] / full['calls']) * 100:>9.1f}%{(1 - fast['searches'] / full['searches']) * 100:>7.1f}%")

        print(f"\n  {'命中率':<10}{'1000条 LLM 费(flash)':>24}{'搜索费':>14}{'合计':>14}")
        print("  " + "-" * 60)
        for hit in (0.0, 0.3, 0.5, 0.7):
            bl = fb * (1 - hit) + kb * hit
            ou = full["out"] * (1 - hit) + fast["out"] * hit
            se = full["searches"] * (1 - hit) + fast["searches"] * hit
            llm = (bl * args.runs / 1e6) * 0.2 + (ou * args.runs / 1e6) * 0.8
            srch = se * args.runs * 0.008 * 7.1
            print(f"  {hit * 100:>5.0f}%      ¥{llm:>16,.0f}{'¥' + format(srch, ',.0f'):>14}"
                  f"{'¥' + format(llm + srch, ',.0f'):>14}")


if __name__ == "__main__":
    main()
