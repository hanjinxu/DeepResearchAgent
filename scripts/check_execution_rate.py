#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测写作 agent 对批评的执行率。

这是开训前的第一道闸门，比 judge 稳定性更早该测。

为什么：
GRPO 训红队，奖励来自"这条批评成立吗"。但批评成立之后，要由**冻结的写作模型**
把它改掉——如果写作模型改不动，好批评也产不出改进，红队会收到"批评没用"的信号，
最后学会的是少说话。

判据：执行率 < 50% → 写作模型太弱，先换模型或简化批评表述；
      执行率 > 80% → 信号能传得动，可以开训。

用法：
    # 1) 生成批评（走 red_team）
    python scripts/sample_critiques.py --draft X.md --k 1 --out data/flywheel/exec_raw.jsonl

    # 2) 测执行率（走 writer，每条批评改一次）
    python scripts/check_execution_rate.py \
        --draft X.md --raw data/flywheel/exec_raw.jsonl --out data/flywheel/exec_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


REWRITE_PROMPT = """你是报告修订员。请按照下面每条审校意见修改报告，只改被指出的地方，
其余内容保持原样，不要新增没有被要求的章节。

<待修订报告>
{draft}
</待修订报告>

<审校意见>
{critiques}
</审校意见>

直接输出修订后的完整报告（Markdown），不要解释你改了什么。
"""

JUDGE_EXEC_PROMPT = """判断下面这条审校意见，是否已经被"修订后报告"真正落实。

只判落实与否，不评价改得好不好。

审校意见（位置：{target}）：
{problem}
建议改法：{suggestion}

<修订后报告>
{revised}
</修订后报告>

只输出 JSON：
{{"executed": true/false/"unknown", "evidence": "不超过 40 字，指出改后的对应文字"}}

- true：修订后报告里能看到针对这条意见的改动
- false：相关内容没有变化，或改动方向相反
- unknown：报告里找不到对应位置，无法判断
"""


def _fmt_critiques(crits) -> str:
    lines = []
    for i, c in enumerate(crits, 1):
        lines.append(
            f"{i}. 位置：{c.target_section}\n   问题：{c.problem}\n   建议：{c.suggestion or '（未给）'}"
        )
    return "\n".join(lines) or "（无）"


def main() -> int:
    ap = argparse.ArgumentParser(description="测写作 agent 对批评的执行率")
    ap.add_argument("--draft", required=True, help="原始草稿")
    ap.add_argument("--raw", required=True, help="sample_critiques.py 产出的 jsonl（取第一条非空样本）")
    ap.add_argument("--out", default="data/flywheel/exec_report.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from langchain_core.messages import HumanMessage

    from deep_research.flywheel import filter_critiques, parse_critiques
    from deep_research.flywheel.verifier import JUDGE_PROMPT  # noqa: F401  仅确认导入可用
    from deep_research.llm import get_chat_model
    from deep_research.memory.json_utils import extract_json_object

    draft = Path(args.draft).read_text(encoding="utf-8", errors="replace")

    # 选一条解析出最多批评的样本
    best = None
    for line in Path(args.raw).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        crits = parse_critiques(row.get("raw", ""))
        if crits and (best is None or len(crits) > len(best[1])):
            best = (row, crits)
    if best is None:
        print("raw 里没有可解析的批评，先跑 sample_critiques.py", file=sys.stderr)
        return 2

    row, crits = best
    fr = filter_critiques(crits, draft)
    usable = fr.usable
    if not usable:
        print(f"批评全部被过滤（{fr.reason}），无法测执行率", file=sys.stderr)
        return 2

    print(f"草稿：{args.draft}（{len(draft)} 字）")
    print(f"用于测试的批评：{len(usable)} 条（原 {len(crits)} 条，过滤 {fr.dropped_count} 条）")

    prompt = REWRITE_PROMPT.format(draft=draft, critiques=_fmt_critiques(usable))
    if args.dry_run:
        print("-" * 60)
        print(f"[dry] 改写 prompt 长度：{len(prompt)} 字")
        print(f"[dry] 将调用 writer 1 次 + judge {len(usable)} 次")
        print("-" * 60)
        return 0

    writer = get_chat_model("writer")
    judge = get_chat_model("evaluator")

    print(">>> 调用写作模型改写中……")
    resp = writer.invoke([HumanMessage(content=prompt)])
    revised = resp.content if isinstance(resp.content, str) else str(resp.content)

    rows = []
    executed = 0
    for c in usable:
        jp = JUDGE_EXEC_PROMPT.format(
            target=c.target_section[:120], problem=c.problem[:400],
            suggestion=c.suggestion[:300], revised=revised[:14000],
        )
        try:
            jr = judge.invoke([HumanMessage(content=jp)])
            text = jr.content if isinstance(jr.content, str) else str(jr.content)
            obj = extract_json_object(text)
            flag = obj.get("executed")
            ok = flag is True or str(flag).lower() == "true"
            evidence = str(obj.get("evidence", ""))[:100]
        except Exception as exc:  # noqa: BLE001
            ok, evidence = False, f"judge 失败: {str(exc)[:60]}"
        executed += 1 if ok else 0
        rows.append({
            "subitem": c.subitem, "severity": c.severity,
            "target": c.target_section[:80], "problem": c.problem[:120],
            "executed": ok, "evidence": evidence,
        })
        print(f"  [{'✓' if ok else '×'}] {c.subitem} {c.target_section[:36]}  {evidence}")

    rate = executed / len(usable)
    verdict = (
        "可以开训：写作能落实大部分批评，红队的信号能传下去"
        if rate >= 0.8 else
        "勉强：部分批评改进不去，红队可能学到'批评没用'"
        if rate >= 0.5 else
        "先别训：写作模型改不动，先换更大的写作模型或简化批评表述"
    )
    print("-" * 60)
    print(f"执行率：{executed}/{len(usable)} = {rate:.0%}")
    print(f"判读：{verdict}")

    out = {
        "draft": str(args.draft),
        "critiques_total": len(crits),
        "critiques_used": len(usable),
        "executed": executed,
        "rate": round(rate, 4),
        "verdict": verdict,
        "rows": rows,
        "revised_chars": len(revised),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告已写入：{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
