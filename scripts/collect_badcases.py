#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 bad case 汇总成一份可读的 Markdown，供人工复核与改提示词。

分三堆，因为处理方式不同：

    误报（invalid）        红队指错了 → 要么修提示词，要么说明是核实的锅
    被过滤（shortcut）     批评不可用（无定位/空话/复述/重复）→ 看红队的坏习惯
    无法判定（unverifiable） 规则覆盖不到 → 这批最值得人工看，决定要不要扩规则或加 judge

用法：
    python scripts/collect_badcases.py --samples data/flywheel/samples.json \
        --out data/flywheel/badcases.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _iter_samples(doc: dict):
    for g in doc.get("groups", []):
        draft_fp = g.get("prompt", "")
        for s in g.get("samples", []):
            yield draft_fp, s


def main() -> int:
    ap = argparse.ArgumentParser(description="汇总 bad case")
    ap.add_argument("--samples", default="data/flywheel/samples.json")
    ap.add_argument("--out", default="data/flywheel/badcases.md")
    ap.add_argument("--limit", type=int, default=40, help="每类最多列多少条")
    args = ap.parse_args()

    doc = json.loads(Path(args.samples).read_text(encoding="utf-8"))

    invalid = []
    filtered = []
    unverifiable = []

    for fp, s in _iter_samples(doc):
        crit_by_id = {c["critique_id"]: c for c in s.get("critiques", [])}
        for v in s.get("verifications", []):
            c = crit_by_id.get(v.get("critique_id", ""), {})
            item = {
                "draft": fp, "subitem": c.get("subitem", ""), "severity": c.get("severity", ""),
                "target": c.get("target_section", ""), "problem": c.get("problem", ""),
                "reason": v.get("reason", ""), "by": v.get("verified_by", ""),
            }
            if v.get("verdict") == "invalid":
                invalid.append(item)
            elif v.get("verdict") == "unverifiable":
                unverifiable.append(item)
        for h in s.get("shortcut_hits", []):
            filtered.append({
                "draft": fp, "rule": h.get("rule", ""), "detail": h.get("detail", ""),
            })

    rule_counter = Counter(x["rule"] for x in filtered)
    sub_counter = Counter(x["subitem"] for x in invalid)
    unver_counter = Counter(x["subitem"] for x in unverifiable)

    lines = ["# Bad case 汇总", ""]
    lines += [
        "| 类别 | 条数 |", "|---|---|",
        f"| 误报（红队指错） | {len(invalid)} |",
        f"| 被过滤（不可用） | {len(filtered)} |",
        f"| 无法判定 | {len(unverifiable)} |",
        "",
    ]

    if rule_counter:
        lines += ["## 过滤原因分布", "", "| 规则 | 条数 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in rule_counter.most_common()]
        lines += ["", "> 某一类特别多，说明红队有固定坏习惯。例如 `no_target` 多 = 不爱写定位，",
                  "> 该在提示词里强调格式，而不是放宽过滤器。", ""]

    if sub_counter:
        lines += ["## 误报按子项分布", "", "| 子项 | 条数 |", "|---|---|"]
        lines += [f"| {k or '未编号'} | {v} |" for k, v in sub_counter.most_common()]
        lines += ["", "> 集中在某几个子项，优先怀疑那几条核实规则是不是判太严。", ""]

    def _section(title: str, rows, render) -> None:
        lines.extend([f"## {title}（{len(rows)} 条）", ""])
        if not rows:
            lines.extend(["（无）", ""])
            return
        for i, r in enumerate(rows[: args.limit], 1):
            lines.append(render(i, r))
        if len(rows) > args.limit:
            lines.append(f"…… 其余 {len(rows) - args.limit} 条略")
        lines.append("")

    _section("误报", invalid, lambda i, r:
             f"{i}. **[{r['subitem'] or '未编号'}/{r['severity']}]** 位置：{r['target'][:60]}\n"
             f"   - 问题：{r['problem'][:160]}\n"
             f"   - 核实依据（{r['by']}）：{r['reason'][:120]}")

    _section("被过滤", filtered, lambda i, r:
             f"{i}. **{r['rule']}** — {r['detail'][:120]}")

    _section("无法判定（最该人工看）", unverifiable, lambda i, r:
             f"{i}. **[{r['subitem'] or '未编号'}]** 位置：{r['target'][:60]}\n"
             f"   - 问题：{r['problem'][:160]}")

    if unver_counter:
        lines += ["## 无法判定按子项分布", "", "| 子项 | 条数 |", "|---|---|"]
        lines += [f"| {k or '未编号'} | {v} |" for k, v in unver_counter.most_common()]
        lines += ["", "> 这里集中的子项，就是「规则覆盖不到的地方」——",
                  "> 要么补规则（如果可确定性判定），要么必须走 judge。", ""]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")

    print(f"误报 {len(invalid)}｜被过滤 {len(filtered)}｜无法判定 {len(unverifiable)}")
    if rule_counter:
        print("过滤原因：" + "，".join(f"{k} {v}" for k, v in rule_counter.most_common(3)))
    print(f"已写入：{out}")

    if len(unverifiable) > len(invalid) * 2 and unverifiable:
        print("\n⚠️ 无法判定的条数远多于误报：奖励信号大部分是空的。"
              "\n   下次跑 run_flywheel 去掉 --no-judge，让 judge 补上。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
