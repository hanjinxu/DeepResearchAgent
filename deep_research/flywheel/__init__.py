#***********************************************
#      Filename: __init__.py
#   Description: 批评飞轮（红队 GRPO 的样本侧）
#***********************************************

"""批评飞轮：把红队输出变成可训练样本，并给出"该不该训"的判据。

职责边界（很重要，别越界）：

    本包只做   —— 解析红队输出、判断可用性、算奖励、聚失败、导样本
    本包不做   —— 调模型、跑 agent、训练、组内归一化的最终消费

依赖方向单向：flywheel 依赖 schema 层，不依赖 agents / memory / tools。
所以它可以离线单测，也可以在没有 GPU、没有 API key 的机器上跑。

典型用法：

    from deep_research.flywheel import score_group, build_groups, export_grpo

    samples = score_group(draft, [raw1, raw2, raw3, raw4])   # k=4
    doc = export_grpo(build_groups(samples), "data/flywheel/samples.json")

先跑这一步（不需要核实），看两件事：
    1. 过滤率 —— 红队有多少输出是"不可用"的
    2. 组内 std —— 四个样本分得开不开

两个数都不合格的话，先别接训练。

要算真实奖励还得有核实器：
    from deep_research.flywheel import CombinedVerifier, RuleVerifier

    verifier = CombinedVerifier(draft, judge=JudgeVerifier())   # 规则优先，判不了问 judge
    verifications = verifier.verify_all(critiques)
"""

from deep_research.flywheel.schema import (
    Critique,
    SampleScore,
    ShortcutHit,
    Verification,
    critique_fingerprint,
    severity_of,
    stable_id,
    SEVERITY_BY_SUBITEM,
    SEVERITY_WEIGHT,
)
from deep_research.flywheel.parser import parse_critiques
from deep_research.flywheel.filters import FilterResult, filter_critiques
from deep_research.flywheel.reward import (
    group_variance_report,
    normalize_advantages,
    score_sample,
)
from deep_research.flywheel.attribution import (
    FailureCluster,
    cluster_failures,
    failure_signature,
    is_failure,
    promote_to_regression,
    subitem_stats,
)
from deep_research.flywheel.export import (
    FORMAT,
    Group,
    build_groups,
    export_grpo,
    load_export,
    summarize_doc,
)
from deep_research.flywheel.pipeline import score_group, score_output
from deep_research.flywheel.verifier import (
    CombinedVerifier,
    JudgeVerifier,
    RuleVerifier,
    RuleFinding,
    split_sections,
)

__all__ = [
    "Critique",
    "Verification",
    "ShortcutHit",
    "SampleScore",
    "FilterResult",
    "FailureCluster",
    "Group",
    "FORMAT",
    "SEVERITY_BY_SUBITEM",
    "SEVERITY_WEIGHT",
    "critique_fingerprint",
    "severity_of",
    "stable_id",
    "parse_critiques",
    "filter_critiques",
    "score_sample",
    "score_output",
    "score_group",
    "normalize_advantages",
    "group_variance_report",
    "failure_signature",
    "is_failure",
    "cluster_failures",
    "promote_to_regression",
    "subitem_stats",
    "build_groups",
    "export_grpo",
    "load_export",
    "summarize_doc",
    "RuleVerifier",
    "JudgeVerifier",
    "CombinedVerifier",
    "RuleFinding",
    "split_sections",
]
