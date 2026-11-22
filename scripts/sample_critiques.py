#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""采样红队批评：对同一份草稿跑 k 次，落成 jsonl。

为什么需要独立脚本：
红队现在是 LangGraph 里的一个节点，跑一次要整条链路（检索 + 写作 + 评分）。
但训 GRPO 只要"同一份草稿的 k 条批评"，不需要重跑上游。
所以把红队这一层单独拎出来采样，成本从"一次 20 分钟"降到"几秒钟"。

输出一行一条采样，字段对齐 flywheel 的输入：
    {"draft_path", "draft_fp", "sample_idx", "temperature", "raw", "usage"}

用法：
    # 先用 dry-run 确认 prompt 和温度（不花钱、不联网）
    python scripts/sample_critiques.py --draft output_report_x.md --k 4 --dry-run

    # 真采样（走 config 里 red_team 角色的 backend）
    DR_LLM_TEMPERATURE=0.9 python scripts/sample_critiques.py \
        --draft output_report_x.md --k 4 --out data/flywheel/raw.jsonl

    # 批量：目录下所有 report 都采
    DR_LLM_TEMPERATURE=0.9 python scripts/sample_critiques.py \
        --draft-dir . --pattern "output_report_*.md" --k 4 --out data/flywheel/raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _guess_brief(draft: str, max_chars: int = 400) -> str:
    """从报告里抠一段当研究简报。

    红队提示词需要 research_brief，但历史报告没有单独存简报。
    退化方案：取标题 + 第一段。够用，因为红队主要看草稿；
    简报只影响"审查方向"，不影响它能不能挑出具体问题。
    """
    lines = [ln.strip() for ln in draft.splitlines() if ln.strip()]
    head = []
    for ln in lines[:6]:
        head.append(ln.lstrip("#").strip())
    text = "；".join(x for x in head if x)
    return text[:max_chars] or "（无简报，按报告本身审查）"


def main() -> int:
    ap = argparse.ArgumentParser(description="采样红队批评（k 条/草稿）")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--draft", help="单份草稿文件")
    src.add_argument("--draft-dir", help="草稿目录（配合 --pattern）")
    ap.add_argument("--pattern", default="output_report_*.md", help="目录模式")
    ap.add_argument("--k", type=int, default=4, help="每份草稿采样条数（GRPO 建议 >=4）")
    ap.add_argument("--temperature", type=float, default=None,
                    help="不填则用 DR_LLM_TEMPERATURE；两者都没有就沿用 config（通常为 0，采不出多样性）")
    ap.add_argument("--brief", default=None, help="手动指定研究简报；不填则从草稿里抠")
    ap.add_argument("--out", default="data/flywheel/raw.jsonl")
    ap.add_argument("--prompt", choices=("checklist", "critique"), default="checklist",
                    help="checklist=按业务评分卡逐条核对（默认）；critique=旧版自由提批评")
    ap.add_argument("--dry-run", action="store_true", help="只打印 prompt 与参数，不调用模型")
    args = ap.parse_args()

    # 温度：命令行 > 环境变量 > config 原值
    if args.temperature is not None:
        os.environ["DR_LLM_TEMPERATURE"] = str(args.temperature)
    effective_temp = os.environ.get("DR_LLM_TEMPERATURE", "<未设置，沿用 config>")

    if args.draft:
        drafts = [Path(args.draft)]
    else:
        drafts = sorted(Path(args.draft_dir).glob(args.pattern))

    if not drafts:
        print(f"没有匹配到草稿：{args.draft or args.draft_dir}/{args.pattern}", file=sys.stderr)
        return 2

    # 延迟导入：dry-run 时也需要，但保持与项目其它脚本一致的导入习惯
    from langchain_core.messages import HumanMessage

    from deep_research.flywheel import critique_fingerprint, parse_critiques
    from deep_research.flywheel.parser import parse_checklist
    from deep_research.flywheel.rubric import build_checklist
    from deep_research.llm import get_chat_model
    from deep_research.prompts import RED_TEAM_CHECKLIST_PROMPT, RED_TEAM_PROMPT

    def build_prompt(draft: str, brief: str) -> str:
        if args.prompt == "checklist":
            return RED_TEAM_CHECKLIST_PROMPT.format(
                draft_report=draft, checklist=build_checklist()
            )
        return RED_TEAM_PROMPT.format(research_brief=brief, draft_report=draft)

    def count_items(raw: str) -> int:
        if args.prompt == "checklist":
            return len(parse_checklist(raw))
        return len(parse_critiques(raw))

    print(f"草稿 {len(drafts)} 份 × k={args.k}｜温度 {effective_temp}｜输出 {args.out}")

    if args.dry_run:
        d0 = _read(drafts[0])
        brief = args.brief or _guess_brief(d0)
        prompt = build_prompt(d0, brief)
        print("-" * 60)
        print(f"[dry] 草稿：{drafts[0]}（{len(d0)} 字）")
        print(f"[dry] 简报（截断）：{brief[:120]}")
        print(f"[dry] prompt 长度：{len(prompt)} 字")
        print(f"[dry] 指纹：{critique_fingerprint(d0)}")
        print(f"[dry] 总计将调用 {len(drafts) * args.k} 次模型")
        print("-" * 60)
        print("未调用模型。去掉 --dry-run 即真采样。")
        return 0

    model = get_chat_model("red_team")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    empty = 0
    with out_path.open("a", encoding="utf-8") as fd:
        for draft_path in drafts:
            draft = _read(draft_path)
            fp = critique_fingerprint(draft)
            brief = args.brief or _guess_brief(draft)
            prompt = build_prompt(draft, brief)

            for i in range(args.k):
                # PASS 也算一条样本：它代表"红队认为没问题"，
                # 后面会被 filters 判成 no_critique → trainable=False，
                # 但这个比例本身是有价值的指标（红队是不是过于宽松）。
                try:
                    resp = model.invoke([HumanMessage(content=prompt)])
                    raw = resp.content if isinstance(resp.content, str) else str(resp.content)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [ERR] {draft_path.name} #{i}: {str(exc)[:120]}", file=sys.stderr)
                    continue

                usage = getattr(resp, "usage_metadata", None) or {}
                row = {
                    "draft_path": str(draft_path),
                    "draft_fp": fp,
                    "sample_idx": i,
                    "temperature": os.environ.get("DR_LLM_TEMPERATURE", ""),
                    "raw": raw,
                    "n_critiques": count_items(raw),
                    "usage": usage,
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                fd.write(json.dumps(row, ensure_ascii=False) + "\n")
                fd.flush()
                written += 1
                if not row["n_critiques"]:
                    empty += 1
                print(f"  [{i + 1}/{args.k}] {draft_path.name} → 解析出 {row['n_critiques']} 条批评")

    print("-" * 60)
    print(f"写入 {written} 条 → {out_path}")
    if written:
        print(f"其中解析为 0 条的（PASS 或格式异常）：{empty} 条（{empty / written:.0%}）")
    print("下一步：python scripts/inspect_samples.py --raw " + str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
