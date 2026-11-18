#***********************************************
#      Filename: attribution.py
#   Description: 失败归因——signature / 聚类 / 固化成回归用例
#***********************************************

"""三段式失败归因。

    1) signature —— 单次红队输出为什么不可用（一句话）
    2) cluster   —— 跨样本聚合，看哪类失败反复出现
    3) promote   —— 把高频失败固化成回归用例，以后每次改动都跑一遍

为什么要有第三段：
现在你只有 50 份历史报告，改一版 prompt 或换一个模型，无法判断
"上次那些问题还在不在"。把高频失败固化成固定用例之后，
才有"改进了没有"的客观证据——否则每次都只能凭感觉说"好像好一点"。

signature 直接用红队提示词里的子项编号（1.1~4.1），不另造分类：
模型已经在用这套编号组织审查方向，再叠一层自定义 taxonomy
只会造成同一件事两个名字，标注和训练都对不齐。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

from deep_research.flywheel.schema import SampleScore, stable_id


@dataclass
class FailureCluster:
    signature: str
    count: int = 0
    prompts: List[str] = field(default_factory=list)
    example_reasons: List[str] = field(default_factory=list)
    suggested_category: str = "critique_quality"

    def to_dict(self) -> Dict[str, object]:
        return {
            "signature": self.signature,
            "count": self.count,
            "prompts": self.prompts[:5],
            "example_reasons": self.example_reasons[:5],
            "suggested_category": self.suggested_category,
        }


def is_failure(score: SampleScore) -> bool:
    """判定一次红队输出是否算"失败"。

    两种失败：
      - 被 trainable 门拦下（产出的批评不可用）
      - 没有被拦，但奖励 ≤ 0（没有一条成立，或误报压过了命中）
    """
    if not score.trainable:
        return True
    return score.total <= 0.0


def failure_signature(score: SampleScore) -> str:
    """单次失败的签名。只取前两个子项，避免签名过长导致聚类粒度过细。"""
    if not score.trainable:
        rule = score.filter_reason or "filtered"
        return f"filtered:{rule}"

    subitems = sorted({c.subitem for c in score.critiques if c.subitem})
    head = "+".join(subitems[:2]) or "no_subitem"

    n_valid = sum(1 for v in score.verifications if v.verdict == "valid")
    n_invalid = sum(1 for v in score.verifications if v.verdict == "invalid")
    if n_valid == 0 and n_invalid > 0:
        return f"all_invalid:{head}"
    return f"weak:{head}"


def cluster_failures(
    scores: Sequence[SampleScore], min_count: int = 2
) -> List[FailureCluster]:
    """按签名聚类，只保留出现 ≥ min_count 次的簇，按频次降序。"""
    buckets: Dict[str, FailureCluster] = {}

    for s in scores:
        if not is_failure(s):
            continue
        sig = failure_signature(s)
        c = buckets.get(sig)
        if c is None:
            c = FailureCluster(signature=sig)
            buckets[sig] = c
        c.count += 1
        if s.prompt not in c.prompts:
            c.prompts.append(s.prompt)
        if len(c.example_reasons) < 5:
            reason = _first_reason(s)
            if reason:
                c.example_reasons.append(reason)

    out = [c for c in buckets.values() if c.count >= min_count]
    for c in out:
        c.suggested_category = _categorize(c.signature)
    out.sort(key=lambda c: c.count, reverse=True)
    return out


def _first_reason(score: SampleScore) -> str:
    if score.shortcut_hits:
        return f"{score.shortcut_hits[0].rule}: {score.shortcut_hits[0].detail}"
    for v in score.verifications:
        if v.verdict == "invalid":
            return f"误报: {v.reason or v.critique_id}"
    return ""


def _categorize(signature: str) -> str:
    if signature.startswith("filtered:no_target"):
        return "定位缺失"
    if signature.startswith("filtered:vague"):
        return "批评空泛"
    if signature.startswith("filtered:restates"):
        return "复述原文"
    if signature.startswith("filtered:duplicate"):
        return "重复批评"
    if signature.startswith("filtered:hallucinated"):
        return "法条幻觉"
    if signature.startswith("all_invalid"):
        return "系统性误报"
    return "批评偏弱"


def promote_to_regression(cluster: FailureCluster, draft: str = "", version: str = "v1") -> Dict[str, object]:
    """把一个高频失败簇固化成回归用例（可落 jsonl）。

    用例内容：给定这份草稿，红队**至少**要能提出某类批评，
    且不得复述原文、不得无定位。判定方式留给调用侧（可以人工，也可以跑一遍打分脚本）。
    """
    return {
        "case_id": stable_id("REG", cluster.signature, version, n=6),
        "version": version,
        "signature": cluster.signature,
        "category": cluster.suggested_category,
        "count": cluster.count,
        "draft_fingerprint": stable_id("draft", "".join(draft.split()), n=12) if draft else "",
        "expect": {
            "min_valid_subitem": _expected_subitem(cluster.signature),
            "must_not": ["restates_draft", "no_target"],
        },
        "notes": f"由生产失败聚类固化：{cluster.signature}（出现 {cluster.count} 次）",
    }


def _expected_subitem(signature: str) -> str:
    tail = signature.split(":", 1)[-1]
    for token in tail.replace("+", " ").split():
        if token and token[0].isdigit() and "." in token:
            return token
    return ""


def subitem_stats(scores: Iterable[SampleScore]) -> List[Dict[str, object]]:
    """按子项统计红队的命中率与误报率。

    这是"红队哪类问题最擅长、哪类最弱"的直接证据，
    比一个笼统的总分有用得多：它告诉你下一轮该补哪一类。
    """
    agg = defaultdict(lambda: {"valid": 0, "invalid": 0, "unverifiable": 0})

    for s in scores:
        by_id = {v.critique_id: v for v in s.verifications}
        for c in s.critiques:
            key = c.subitem or "unknown"
            verdict = by_id.get(c.critique_id).verdict if c.critique_id in by_id else "unverifiable"
            agg[key][verdict] += 1

    rows: List[Dict[str, object]] = []
    for subitem, cnt in agg.items():
        judged = cnt["valid"] + cnt["invalid"]
        rows.append({
            "subitem": subitem,
            "valid": cnt["valid"],
            "invalid": cnt["invalid"],
            "unverifiable": cnt["unverifiable"],
            "precision": round(cnt["valid"] / judged, 4) if judged else None,
        })
    rows.sort(key=lambda r: (r["precision"] is None, r["precision"] or 0))
    return rows
