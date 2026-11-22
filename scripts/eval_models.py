#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三组对比评估：base / LoRA / 通用大模型 API。

这是"领域专精能否超过通用大模型"那个主张的实验工具。
设计要点（改动前先读）：

1. **主指标是「有效检出/份」，不是 precision。**
   纯 precision 可以靠"只提一条最保险的"刷到很高，
   所以用"每份草稿平均成立的批评数"——它同时惩罚少提和提错，
   而且不需要人工标注"这份草稿总共有几个真问题"。召回率要 ground truth，
   我们拿不到；有效检出数是它的可行替代。

2. **base 与 LoRA 必须同服务、只换 model 名。**
   用 vLLM 的 --enable-lora 同时提供 base 和 adapter，
   这样两组之间唯一的差异就是权重，排除了服务、量化、参数差异。

3. **judge 不能和任何候选同源。**
   脚本会检查 judge 的 backend 与候选的 backend 是否相同并告警——
   自己判自己，结论不成立。

4. **过滤率要单独报。**
   某组过滤率高，说明它连"按格式输出"都做不到，
   那是能力差异的一部分，不能悄悄从分母里消失。

用法：
    # 先在 config.yml 的 roles 里定义三个评估角色（见 docs）
    python scripts/eval_models.py \
        --drafts "output_report_*.md" \
        --roles base,qwen_14b_lora,api_deepseek \
        --k 4 --out data/flywheel/eval_3way.json
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_drafts(pattern: str) -> list[Path]:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    return [p for p in paths if p.is_file()]


def main() -> int:
    ap = argparse.ArgumentParser(description="base / LoRA / API 三组对比")
    ap.add_argument("--drafts", default="output_report_*.md", help="测试草稿（glob）")
    ap.add_argument("--roles", default="base,qwen_14b_lora,api_deepseek",
                    help="config.yml 里 roles 下的角色名，逗号分隔；第一个是参照组")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=900)
    ap.add_argument("--max-draft-chars", type=int, default=16000)
    ap.add_argument("--out", default="data/flywheel/eval_3way.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from langchain_core.messages import HumanMessage

    from deep_research.flywheel import filter_critiques, parse_critiques
    from deep_research.flywheel.verifier import CombinedVerifier, JudgeVerifier
    from deep_research.llm import get_chat_model
    from deep_research.prompts import RED_TEAM_PROMPT

    drafts = _load_drafts(args.drafts)
    if not drafts:
        print(f"没有匹配到草稿：{args.drafts}", file=sys.stderr)
        return 2
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    print(f"草稿 {len(drafts)} 份 × 角色 {len(roles)} 个 × k={args.k}"
          f" = {len(drafts) * len(roles) * args.k} 次生成")

    if args.dry_run:
        for p in drafts:
            print(f"[dry] {p.name}（{len(p.read_text(encoding='utf-8'))} 字）")
        print(f"[dry] 将调用角色：{roles}｜每份草稿 k={args.k}")
        print("[dry] 未调用模型")
        return 0

    models = {name: get_chat_model(name) for name in roles}

    # judge 同源检查：backend 名相同就告警
    judge = JudgeVerifier()
    judge_backend = getattr(judge.model, "openai_api_base", "?")
    for name, m in models.items():
        if getattr(m, "openai_api_base", None) == judge_backend:
            print(f"⚠️  judge 与 {name} 指向同一个后端（{judge_backend}），"
                  f"同源判分会偏袒，结论不可用。", file=sys.stderr)

    # ---- 采样 + 核实 ----
    rows = []
    for draft_path in drafts:
        draft = draft_path.read_text(encoding="utf-8", errors="replace")[: args.max_draft_chars]
        brief = "；".join(ln.strip() for ln in draft.splitlines()[:4] if ln.strip())[:300]
        prompt = RED_TEAM_PROMPT.format(research_brief=brief, draft_report=draft)

        for role in roles:
            model = models[role]
            for i in range(args.k):
                try:
                    resp = model.invoke([HumanMessage(content=prompt)],
                                        max_tokens=args.max_new_tokens)
                    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [ERR] {draft_path.name} {role} #{i}: {str(exc)[:100]}",
                          file=sys.stderr)
                    continue

                crits = parse_critiques(raw)
                fr = filter_critiques(crits, draft, require_quote=True)
                vers = CombinedVerifier(draft, judge=judge).verify_all(fr.usable)

                n_valid = sum(1 for v in vers if v.verdict == "valid")
                n_invalid = sum(1 for v in vers if v.verdict == "invalid")
                rows.append({
                    "draft": draft_path.name, "role": role, "sample": i,
                    "parsed": len(crits), "usable": len(fr.usable),
                    "dropped": len(fr.dropped),
                    "valid": n_valid, "invalid": n_invalid,
                })
                print(f"  {draft_path.name[:24]:<26} {role:<18} #{i} "
                      f"解析{len(crits)} 可用{len(fr.usable)} 成立{n_valid} 误报{n_invalid}")

    # ---- 汇总 ----
    by_role = defaultdict(list)
    for r in rows:
        by_role[r["role"]].append(r)

    print("\n" + "=" * 78)
    print(f"{'角色':<20}{'有效检出/份':>12}{'precision':>11}{'过滤率':>9}"
          f"{'平均批评数':>11}{'生成次数':>9}")
    print("-" * 78)
    summary = {}
    for role in roles:
        rs = by_role.get(role, [])
        if not rs:
            continue
        n_samples = len(rs)
        valid_per = sum(r["valid"] for r in rs) / n_samples
        judged = sum(r["valid"] + r["invalid"] for r in rs)
        prec = (sum(r["valid"] for r in rs) / judged) if judged else 0.0
        parsed = sum(r["parsed"] for r in rs)
        dropped = sum(r["dropped"] for r in rs)
        drop_rate = dropped / parsed if parsed else 0.0
        avg_crit = sum(r["usable"] for r in rs) / n_samples
        print(f"{role:<20}{valid_per:>12.2f}{prec:>10.0%}{drop_rate:>9.0%}"
              f"{avg_crit:>11.1f}{n_samples:>9}")
        summary[role] = {
            "valid_per_draft": round(valid_per, 3),
            "precision": round(prec, 3),
            "drop_rate": round(drop_rate, 3),
            "avg_critiques": round(avg_crit, 2),
            "n_samples": n_samples,
        }
    print("=" * 78)

    # ---- 分草稿看稳定性（避免结论被某一份草稿带偏）----
    print("\n分草稿（有效检出数）：")
    header = f"{'草稿':<30}" + "".join(f"{r:>12}" for r in roles)
    print(header)
    print("-" * len(header))
    for d in drafts:
        cells = []
        for role in roles:
            vals = [x["valid"] for x in by_role.get(role, []) if x["draft"] == d.name]
            cells.append(f"{statistics.mean(vals):.1f}" if vals else "—")
        print(f"{d.name[:28]:<30}" + "".join(f"{c:>12}" for c in cells))

    out = {"drafts": [d.name for d in drafts], "k": args.k,
           "summary": summary, "rows": rows}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {args.out}")

    # ---- 判读 ----
    base = summary.get(roles[0], {})
    troupe = summary.get(roles[1], {}) if len(roles) > 1 else {}
    if base and troupe:
        delta = troupe["valid_per_draft"] - base["valid_per_draft"]
        rel = delta / base["valid_per_draft"] if base["valid_per_draft"] else 0
        print(f"\n主张 A（训练是否有效）：{roles[1]} vs {roles[0]} "
              f"{base['valid_per_draft']:.2f} → {troupe['valid_per_draft']:.2f}"
              f"（{rel:+.0%}）")
        print("  → 这一条同模型同数据，只有权重差，结论最干净。" if delta > 0
              else "  → 训练没带来提升，先查训练本身（奖励符号、优势计算、步数）。")
    if len(roles) > 2:
        api = summary.get(roles[2], {})
        if api and troupe:
            print(f"主张 B（能否超过通用大模型）：{roles[1]} vs {roles[2]} "
                  f"{troupe['valid_per_draft']:.2f} vs {api['valid_per_draft']:.2f}")
            if troupe["valid_per_draft"] >= api["valid_per_draft"]:
                print("  → 超过了，领域专精成立。")
            else:
                print("  → 还没追上。别硬讲，改讲「训练带来了 X% 提升」——"
                      "这条更稳，也不丢人。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
