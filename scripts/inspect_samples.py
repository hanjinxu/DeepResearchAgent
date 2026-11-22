#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查采样质量：解析率、过滤率、批评类型分布、组内多样性。

这是"开训前的体检"，不依赖核实器。它回答三个问题：

    1. 解析器认不认得出红队的真实输出？（解析率）
    2. 排除掉多少不可用的批评？为什么？（过滤原因分布）
    3. 同一份草稿的 k 条批评，彼此差得够不够？（组内多样性）

⚠️ 这里不做"奖励"，因为没有核实器，reward 会退化成常数。
本脚本只报**结构指标**——它们是奖励的前置条件，不是奖励本身。
把"结构多样性"当成"训练信号"是最容易犯的错：四份批评措辞不同但都指同一处，
Jaccard 是 0，看起来"多样"，实际组内没有区分度。

用法：
    python scripts/inspect_samples.py --raw data/flywheel/raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deep_research.flywheel import filter_critiques, parse_critiques  # noqa: E402
from deep_research.flywheel.schema import SEVERITY_WEIGHT  # noqa: E402


def _load(path: Path) -> List[dict]:
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


def _critique_keys(crits) -> Set[str]:
    """把一条批评压成"指向"的指纹：子项 + 位置前缀。

    刻意不把措辞算进去——措辞不同但指同一处，等价于没有多样性。
    """
    return {
        f"{c.subitem or '?'}@{(c.target_section or '')[:16]}"
        for c in crits
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main() -> int:
    ap = argparse.ArgumentParser(description="检查采样质量（不需要核实器）")
    ap.add_argument("--raw", default="data/flywheel/raw.jsonl")
    ap.add_argument("--show", type=int, default=3, help="每份草稿打印几条样例")
    args = ap.parse_args()

    path = Path(args.raw)
    if not path.exists():
        print(f"文件不存在：{path}", file=sys.stderr)
        return 2

    rows = _load(path)
    if not rows:
        print("没有样本", file=sys.stderr)
        return 2

    # 草稿内容：优先用 draft_path 重读（raw.jsonl 只存路径不存正文）
    draft_cache: Dict[str, str] = {}
    for r in rows:
        p, fp = r.get("draft_path", ""), r.get("draft_fp", "")
        if p and fp and fp not in draft_cache:
            try:
                draft_cache[fp] = Path(p).read_text(encoding="utf-8", errors="replace")
            except OSError:
                draft_cache[fp] = ""

    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        groups[r.get("draft_fp", "?")].append(r)

    total = len(rows)
    parse_ok = 0
    all_dropped: Counter = Counter()
    subitem_counter: Counter = Counter()
    severity_counter: Counter = Counter()
    kept_total = 0
    dropped_total = 0

    print("=" * 68)
    print(f"样本文件：{path}")
    print(f"草稿 {len(groups)} 份｜采样 {total} 条｜平均 k = {total / max(len(groups), 1):.1f}")
    print("=" * 68)

    print("\n【每份草稿】")
    header = f"{'草稿':<28}{'k':>3}{'解析出':>7}{'可用':>5}{'过滤':>5}{'多样性':>8}"
    print(header)
    print("-" * len(header))

    low_diversity = 0
    rows_detail = []

    for fp, items in groups.items():
        draft = draft_cache.get(fp, "")
        keysets: List[Set[str]] = []
        n_parse_ok = 0
        n_kept = 0
        n_drop = 0
        name = Path(items[0].get("draft_path", fp)).name[:26]

        for it in items:
            crits = parse_critiques(it.get("raw", ""))
            if crits:
                n_parse_ok += 1
            fr = filter_critiques(crits, draft)
            n_kept += len(fr.usable)
            n_drop += fr.dropped_count
            for hit in fr.dropped:
                all_dropped[hit.rule] += 1
            for c in fr.usable:
                subitem_counter[c.subitem or "未编号"] += 1
                severity_counter[c.severity] += 1
            keysets.append(_critique_keys(fr.usable))

        pair_scores = [_jaccard(a, b) for a, b in combinations(keysets, 2)]
        diversity = 1.0 - (sum(pair_scores) / len(pair_scores)) if pair_scores else 0.0
        if diversity < 0.3:
            low_diversity += 1

        print(f"{name:<28}{len(items):>3}{n_parse_ok:>7}{n_kept:>5}{n_drop:>5}{diversity:>8.2f}")
        rows_detail.append((name, items, draft, diversity))

        parse_ok += n_parse_ok
        kept_total += n_kept
        dropped_total += n_drop

        if args.show > 0:
            for it in items[:1]:
                for c in filter_critiques(parse_critiques(it.get("raw", "")), draft).usable[:args.show]:
                    print(f"    · [{c.subitem}/{c.severity}] {c.problem[:52]}")

    print("\n【整体】")
    print(f"  解析率（能解析出 ≥1 条批评）  {parse_ok}/{total} = {parse_ok / total:.0%}")
    print(f"  可用批评 {kept_total} 条｜被过滤 {dropped_total} 条"
          + (f"（过滤率 {dropped_total / (kept_total + dropped_total):.0%}）"
             if (kept_total + dropped_total) else ""))

    if all_dropped:
        print("\n  过滤原因：")
        for rule, cnt in all_dropped.most_common():
            print(f"    {rule:<22}{cnt:>4}")

    if subitem_counter:
        print("\n  批评类型分布（子项）：")
        for sub, cnt in subitem_counter.most_common():
            print(f"    {sub:<8}{cnt:>4}")

    if severity_counter:
        w = sum(SEVERITY_WEIGHT.get(s, 0.6) * c for s, c in severity_counter.items())
        print(f"\n  严重度分布：" + "｜".join(f"{k} {v}" for k, v in severity_counter.most_common()))
        print(f"  加权总量 ≈ {w:.1f}（仅供比较，不是奖励）")

    print("\n【判读】")
    if low_diversity:
        print(f"  ⚠️ {low_diversity}/{len(groups)} 份草稿的组内多样性 < 0.3——"
              f"k 条批评指向几乎相同，组内没有区分度。")
        print("     先查温度是否 >0，再查提示词是否把输出压得太死。")
    elif len(groups) and total / len(groups) < 2:
        print("  样本不足：每组至少要 2 条才能算方差，建议 k ≥ 4。")
    else:
        print("  结构上可用：解析率与多样性都过关。接着补核实器算真实奖励。")

    print("\n  注：本脚本不计算奖励。没有核实器时 reward 是常数，")
    print("      报出来的任何'分数'都只是结构指标，不能拿去训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
