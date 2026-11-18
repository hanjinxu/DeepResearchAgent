#***********************************************
#      Filename: reward.py
#   Description: 批评奖励计算 + 组内优势归一化
#***********************************************

"""奖励与优势。

奖励设计（与 sentinel 那套的五项不同，这里只有三块）：

    + 命中且成立的批评，按严重度加权求和
    - 误报的批评，按严重度加权求和 × λ
    - 不可验证的批评，超出容忍条数后小额扣分

为什么不用"精度"（valid / total）：
精度会被"只提一条最确定的"策略刷满——提一条 critical 命中就是 100%，
这会让红队学会只挑最保险的浅层问题。加权求和天然带数量激励：
三条 critical 全对（3.0）优于一条（1.0）。

为什么误报要扣分而不是丢弃：
误报是唯一能告诉模型"你指错了"的信号。丢弃它，模型只会学到"少说话"，
而不是"说准"。λ=1.0 意味着误报和命中同权——宁可错杀，不可放过，
因为法务场景里一次误报会消耗 review 的信任成本。

为什么留一点不可验证的容忍额度：
有些批评（论据是否成立）需要人工判断，机器验不了。如果一并扣分，
红队会倾向于只提"规则可验"的浅层问题（链接失效、编号错），
而那恰恰是最不值钱的。
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

from deep_research.flywheel.schema import (
    Critique,
    SampleScore,
    ShortcutHit,
    Verification,
    SEVERITY_WEIGHT,
)

LAMBDA_INVALID = 1.0          # 误报惩罚系数
MU_UNVERIFIABLE = 0.10        # 不可验证的超额惩罚（每条）
FREE_UNVERIFIABLE = 1         # 容忍条数：一条无法核实不算问题

# 单样本奖励截断。防止某一条异常样本把组内均值拉飞，
# 连带影响同组其他三条的优势值。上下界按"4 条 critical 全中 = 4.0"留余量。
REWARD_CLAMP = (-3.0, 4.0)

# 组内标准差的可用下限。低于这个值说明 k 条样本几乎一样，
# 归一化后优势全是 0，训了等于没训。
MIN_GROUP_STD = 0.05


def _weight_of(crit: Critique) -> float:
    return SEVERITY_WEIGHT.get(crit.severity, 0.6)


def score_sample(
    prompt: str,
    critiques: Sequence[Critique],
    verifications: Sequence[Verification],
    *,
    shortcut_hits: Sequence[ShortcutHit] = (),
    filter_reason: str = "",
    trainable: bool = True,
) -> SampleScore:
    """给一次红队输出打分。

    prompt 是 GRPO 的分组键（用 critique_fingerprint(draft) 生成）。
    未提供核实的批评按 unverifiable 计——不知道对错就不能假设它对。
    """
    by_id: Dict[str, Verification] = {v.critique_id: v for v in verifications}

    valid_w = 0.0
    invalid_w = 0.0
    unverifiable = 0

    for crit in critiques:
        ver = by_id.get(crit.critique_id)
        verdict = ver.verdict if ver else "unverifiable"
        if verdict == "valid":
            valid_w += _weight_of(crit)
        elif verdict == "invalid":
            invalid_w += _weight_of(crit)
        else:
            unverifiable += 1

    unverifiable_penalty = MU_UNVERIFIABLE * max(0, unverifiable - FREE_UNVERIFIABLE)
    invalid_penalty = LAMBDA_INVALID * invalid_w

    total = valid_w - invalid_penalty - unverifiable_penalty
    lo, hi = REWARD_CLAMP
    total = max(lo, min(hi, total))

    return SampleScore(
        prompt=prompt,
        critiques=list(critiques),
        verifications=list(verifications),
        shortcut_hits=list(shortcut_hits),
        breakdown={
            "valid_weight": round(valid_w, 4),
            "invalid_penalty": round(-invalid_penalty, 4),
            "unverifiable_penalty": round(-unverifiable_penalty, 4),
            "n_valid": sum(1 for v in verifications if v.verdict == "valid"),
            "n_invalid": sum(1 for v in verifications if v.verdict == "invalid"),
            "n_unverifiable": unverifiable,
        },
        total=round(total, 4),
        trainable=trainable,
        filter_reason=filter_reason,
    )


def normalize_advantages(rewards: Sequence[float], eps: float = 1e-6) -> List[float]:
    """组内归一化：(r - mean) / (std + eps)。

    这是 GRPO 的核心那一步——不需要 critic，组内均值就是 baseline。
    std 接近 0 时全部返回 0：组内没有区分度，任何非零优势都是噪声放大。
    """
    if not rewards:
        return []
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5
    if std < eps:
        return [0.0] * n
    return [(r - mean) / (std + eps) for r in rewards]


def group_variance_report(
    groups: Dict[str, Iterable[float]],
    *,
    min_std: float = MIN_GROUP_STD,
) -> Dict[str, object]:
    """训练前的"方差体检"。

    这是整套方案唯一的 go / no-go 判据：
    如果绝大多数组的 std 都低于 min_std，说明 k 条样本高度雷同，
    训练跑起来优势全是 0，不如先回头查温度 / 提示词 / 草稿难度。

    返回可直接打印的结构，便于跑一次数据就出结论。
    """
    per_group: List[Dict[str, object]] = []
    usable = 0

    for prompt, values in groups.items():
        vals = [float(v) for v in values]
        if not vals:
            continue
        n = len(vals)
        mean = sum(vals) / n
        std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
        ok = std >= min_std and n > 1
        usable += 1 if ok else 0
        per_group.append({
            "prompt": prompt,
            "n": n,
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ok": ok,
        })

    total = len(per_group)
    per_group.sort(key=lambda d: d["std"])  # 最差的排前面，先看问题在哪
    return {
        "groups": total,
        "usable_groups": usable,
        "usable_ratio": round(usable / total, 4) if total else 0.0,
        "min_std": min_std,
        "verdict": _verdict(usable, total, min_std),
        "detail": per_group,
    }


def _verdict(usable: int, total: int, min_std: float) -> str:
    if total == 0:
        return "没有可评估的组：先导出样本"
    ratio = usable / total
    if ratio >= 0.8:
        return f"可以训：{usable}/{total} 组的组内标准差 ≥ {min_std}"
    if ratio >= 0.5:
        return f"勉强：{usable}/{total} 组有区分度，建议先提温度或增加草稿难度跨度"
    return f"不建议训：仅 {usable}/{total} 组有区分度，优势会大面积归零"
