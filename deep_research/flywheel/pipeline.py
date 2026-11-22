#***********************************************
#      Filename: pipeline.py
#   Description: 解析 → 过滤 → 打分的单入口
#***********************************************

"""把 parser / filters / reward 串成一步。

调用侧只需要记住一个函数：

    score_output(draft, red_team_raw_text, verifications) -> SampleScore

verifications 可以不传——那时所有批评按 unverifiable 计，
奖励里只有"不可验证超额惩罚"这一项，主要用来先跑通链路、
统计过滤率和批评类型分布（不需要核实也能做）。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from deep_research.flywheel.filters import filter_critiques
from deep_research.flywheel.parser import parse_critiques
from deep_research.flywheel.reward import score_sample
from deep_research.flywheel.schema import (
    Critique,
    SampleScore,
    Verification,
    critique_fingerprint,
)

__all__ = ["score_output", "score_group", "parse_critiques"]


def score_output(
    draft: str,
    raw: str,
    verifications: Optional[Iterable[Verification]] = None,
    *,
    law_validator=None,
    require_quote: bool = False,
) -> SampleScore:
    """把一次红队原始输出变成可训练样本（含过滤与打分）。"""
    crits: List[Critique] = parse_critiques(raw)
    fr = filter_critiques(crits, draft, law_validator=law_validator,
                          require_quote=require_quote)

    by_id: Dict[str, Verification] = {v.critique_id: v for v in (verifications or [])}
    kept_ver: List[Verification] = [
        by_id[c.critique_id] for c in fr.usable if c.critique_id in by_id
    ]

    return score_sample(
        prompt=critique_fingerprint(draft),
        critiques=fr.usable,
        verifications=kept_ver,
        shortcut_hits=fr.dropped,
        filter_reason=fr.reason,
        trainable=fr.trainable,
    )


def score_group(
    draft: str,
    raws: Sequence[str],
    verifications_per_sample: Optional[Sequence[Sequence[Verification]]] = None,
    *,
    law_validator=None,
) -> List[SampleScore]:
    """同一份草稿的 k 条采样。这是 GRPO 的一个"组"。"""
    out: List[SampleScore] = []
    for i, raw in enumerate(raws):
        ver = None
        if verifications_per_sample and i < len(verifications_per_sample):
            ver = verifications_per_sample[i]
        out.append(score_output(draft, raw, ver, law_validator=law_validator))
    return out
