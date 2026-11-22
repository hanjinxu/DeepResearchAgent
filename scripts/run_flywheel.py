#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键跑完：解析 → 过滤 → 核实 → 打分 → 导出 GRPO 样本。

这是训练前的最后一步，跑完会给出三个数：
    1. 核实覆盖率（多少条批评真正被判了真假）
    2. 组内标准差（优势有没有区分度）
    3. 各子项的命中率（红队哪类问题最擅长）

任何一个不合格，先别接 verl。

用法：
    # 只跑规则核实（免费，覆盖低）
    python scripts/run_flywheel.py --raw data/flywheel/raw.jsonl \
        --out data/flywheel/samples.json --no-judge

    # 规则 + judge（覆盖高，要 token）
    python scripts/run_flywheel.py --raw data/flywheel/raw.jsonl \
        --out data/flywheel/samples.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deep_research.flywheel import (  # noqa: E402
    build_groups,
    cluster_failures,
    export_grpo,
    filter_critiques,
    group_variance_report,
    parse_critiques,
    score_sample,
    subitem_stats,
    summarize_doc,
)
from deep_research.flywheel.verifier import CombinedVerifier, JudgeVerifier  # noqa: E402


def _load_raw(path: Path):
    rows = []
    with path.open(encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="跑完飞轮全链，导出 GRPO 样本")
    ap.add_argument("--raw", default="data/flywheel/raw.jsonl")
    ap.add_argument("--out", default="data/flywheel/samples.json")
    ap.add_argument("--no-judge", action="store_true", help="只用规则核实（免费）")
    ap.add_argument("--no-quote-check", dest="require_quote", action="store_false",
                    help="关闭「必须引用原文」过滤（仅当使用旧版红队提示词时需要）")
    ap.add_argument("--min-group", type=int, default=2)
    args = ap.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f"找不到 {raw_path}，先跑 scripts/sample_critiques.py", file=sys.stderr)
        return 2

    rows = _load_raw(raw_path)
    if not rows:
        print("raw 为空", file=sys.stderr)
        return 2

    # 按草稿分组，同时缓存正文
    by_draft = defaultdict(list)
    drafts = {}
    for r in rows:
        fp = r.get("draft_fp", "")
        by_draft[fp].append(r)
        if fp not in drafts:
            p = r.get("draft_path", "")
            try:
                drafts[fp] = Path(p).read_text(encoding="utf-8", errors="replace")
            except OSError:
                drafts[fp] = ""

    judge = None
    if not args.no_judge:
        print(">>> 初始化 judge（evaluator 角色）……")
        judge = JudgeVerifier()

    all_scores = []
    coverage_all = []

    for fp, items in by_draft.items():
        draft = drafts.get(fp, "")
        verifier = CombinedVerifier(draft, judge=judge)

        for it in items:
            crits = parse_critiques(it.get("raw", ""))
            fr = filter_critiques(crits, draft, require_quote=args.require_quote)
            vers = verifier.verify_all(fr.usable)

            cov = verifier.coverage(vers)
            coverage_all.append(cov)

            s = score_sample(
                prompt=fp,
                critiques=fr.usable,
                verifications=vers,
                shortcut_hits=fr.dropped,
                filter_reason=fr.reason,
                trainable=fr.trainable,
            )
            all_scores.append(s)

    groups = build_groups(all_scores, drafts)
    doc = export_grpo(groups, args.out, min_group_size=args.min_group, include_advantages=True)

    # ---- 报告 ----
    print("\n" + "=" * 64)
    print(summarize_doc(doc))
    print("=" * 64)

    total_crit = sum(c["total"] for c in coverage_all)
    judged_crit = sum(c["judged"] for c in coverage_all)
    print(f"\n核实覆盖率：{judged_crit}/{total_crit} = "
          f"{(judged_crit / total_crit if total_crit else 0):.0%}"
          f"（其余 {total_crit - judged_crit} 条判为 unverifiable，不计分）")

    var_report = group_variance_report({g.prompt: [s.total for s in g.samples] for g in groups})
    print(f"\n组内方差：{var_report['verdict']}")
    for d in var_report["detail"][:5]:
        print(f"  {d['prompt'][:20]}  n={d['n']}  mean={d['mean']:.3f}  std={d['std']:.3f}"
              f"  {'OK' if d['ok'] else '⚠️'}")

    rows_stats = subitem_stats(all_scores)
    if rows_stats:
        print("\n各子项命中率（precision = 成立 / 被判真假的条数）：")
        for r in rows_stats:
            p = "—" if r["precision"] is None else f"{r['precision']:.0%}"
            print(f"  {r['subitem']:<8} 成立 {r['valid']:>3}｜误报 {r['invalid']:>3}｜"
                  f"无法判 {r['unverifiable']:>3}｜命中率 {p}")

    clusters = cluster_failures(all_scores, min_count=2)
    if clusters:
        print("\n高频失败（出现 ≥2 次）：")
        for c in clusters[:5]:
            print(f"  {c.signature:<28} ×{c.count:<3} {c.suggested_category}")

    print(f"\n样本已写入：{args.out}")

    # ---- 判读 ----
    print("\n" + "-" * 64)
    if not total_crit:
        print("没有解析出任何批评：先看 scripts/inspect_samples.py 的解析率。")
    elif judged_crit / total_crit < 0.3:
        print("核实覆盖率过低（<30%）：奖励基本是常数。")
        print("  → 要么开 judge（去掉 --no-judge），要么先补规则。")
    elif not var_report["usable_groups"]:
        print("组内有区分度的不足：检查温度是否 >0、每组 k 是否 ≥4。")
    else:
        print("可以进入下一步：把样本转成 verl 的 dataset（训练脚本尚未实现）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
