#***********************************************
#      Filename: filters.py
#   Description: 样本可用性门（红队版"捷径过滤"）
#***********************************************

"""trainable 门：把"学不到东西"的批评剔掉。

与 reward 的分工（这一点容易做混，单独说明）：

    filters  判断「这条批评能不能用来训练」——看的是**可用性**
    reward   判断「这条批评对不对」——看的是**正确性**

所以本模块刻意不看验证结果。误报（verdict=invalid）在这里应当**通过**，
因为它是有价值的负样本：告诉模型"这条指错了"。把误报在这里丢掉，
模型就只会学到"少说话"，而不是"说准"。

真正该丢的是这四类——它们无法判对错，留着只会污染梯度：
    1. 没有定位（不知道在说哪一节，无法核实也无法执行）
    2. 空话（"建议加强论证"，没有可核实的具体主张）
    3. 复述草稿（把原文抄一遍当问题，没有新增信息）
    4. 同批重复（同一子项 + 同一位置出现两次，等价于刷数量）
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from deep_research.flywheel.parser import is_vague_suggestion
from deep_research.flywheel.schema import Critique, ShortcutHit

# 定位含糊的写法：指出这些等于没指
_VAGUE_TARGETS = {
    "全文", "整篇", "整体", "多处", "若干处", "部分内容", "各处", "文中", "正文",
    "report", "全文各节", "全部",
}

# 判断性词：出现任一，说明这段文字在做判断，不是单纯复述。
# 只用多字词——单字（过/少/多/未）会把"经过""至少""多源"这类正常表述误判成判断，
# 导致复述规则永久失效（等于没写这条规则）。
_JUDGMENT_RE = re.compile(
    r"错误|缺失|矛盾|不一致|未标|未注明|遗漏|误报|误判|不足|不应|必须|建议|"
    r"应改|需补|无法|不当|过多|过少|不匹配|脱节|悬空|站不住|不能支撑"
)

# 复述判定阈值。定得高（0.8）是因为批评天然会引用草稿原句，
# 阈值低会把"引用原句指出问题"误判成复述——那是好批评，不能杀。
RESTATE_OVERLAP = 0.80

MIN_USABLE_CRITIQUES = 1


@dataclass
class FilterResult:
    usable: List[Critique] = field(default_factory=list)
    dropped: List[ShortcutHit] = field(default_factory=list)
    trainable: bool = False
    reason: str = ""

    @property
    def dropped_count(self) -> int:
        return len(self.dropped)


def _norm(text: str) -> str:
    return "".join((text or "").split())


def _trigrams(text: str) -> set:
    s = _norm(text)
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def _overlap_in_draft(text: str, draft: str) -> float:
    """text 的 trigram 有多大比例能在 draft 里找到。"""
    t, d = _trigrams(text), _trigrams(draft)
    if not t or not d:
        return 0.0
    return len(t & d) / len(t)


def _is_vague_target(target: str) -> bool:
    s = _norm(target)
    if not s:
        return True
    if s in _VAGUE_TARGETS:
        return True
    # 只写了章节号但没有具体内容指认，仍算可用（章节号足以定位）
    return False


def _is_restatement(crit: Critique, draft: str) -> bool:
    """判断这条批评是不是把草稿原文抄了一遍。"""
    if not draft:
        return False
    text = f"{crit.problem} {crit.suggestion}".strip()
    if not text:
        return False
    if _JUDGMENT_RE.search(text):
        return False
    return _overlap_in_draft(text, draft) >= RESTATE_OVERLAP


def filter_critiques(
    critiques: Sequence[Critique],
    draft: str = "",
    *,
    min_usable: int = MIN_USABLE_CRITIQUES,
    law_validator: Optional[Callable[[str], bool]] = None,
) -> FilterResult:
    """按四类规则过滤，产出可用批评列表与样本级 trainable 判定。

    law_validator：可选的"法条是否真实存在"校验函数。
    不传时不拦——因为没有校验器就无从判断建议里引的法条是真是假，
    硬拦会误杀正确的法条引用。传了之后，校验不通过的批评记为
    hallucinated_law_ref 并剔除。
    """
    result = FilterResult()
    seen: Dict[str, int] = {}

    for crit in critiques:
        # 规则 1：没有定位
        if _is_vague_target(crit.target_section):
            result.dropped.append(
                ShortcutHit("no_target", crit.critique_id,
                            f"定位为空或含糊：{crit.target_section!r}")
            )
            continue

        # 规则 2：空话。
        # 注意：**不因"没写解决方法"而丢弃**。训练要核实的是问题本身，
        # 一条定位准确、问题成立的批评即使没给建议，也是有效样本；
        # 按格式完整性过滤会误杀，让红队学到"凑格式"而不是"找准问题"。
        if not crit.problem.strip() or is_vague_suggestion(crit.problem):
            result.dropped.append(
                ShortcutHit("vague_problem", crit.critique_id,
                            f"问题描述过于笼统：{crit.problem[:40]!r}")
            )
            continue

        # 规则 3：复述草稿
        if _is_restatement(crit, draft):
            result.dropped.append(
                ShortcutHit("restates_draft", crit.critique_id,
                            "批评内容几乎全部来自草稿原文，无新增信息")
            )
            continue

        # 规则 4：同批重复（同子项 + 同位置）
        dup_key = f"{crit.subitem}|{_norm(crit.target_section)[:40]}"
        if seen.get(dup_key):
            seen[dup_key] += 1
            result.dropped.append(
                ShortcutHit("duplicate_critique", crit.critique_id,
                            f"与同批第 {seen[dup_key] - 1} 条重复（{crit.subitem} @ {crit.target_section[:30]}）")
            )
            continue
        seen[dup_key] = 1

        # 规则 5：法条幻觉（需外部校验器）
        if law_validator is not None:
            refs = re.findall(r"《[^》]{2,30}》\s*第[一二三四五六七八九十百零\d]+条", crit.suggestion or "")
            bad = [r for r in refs if not law_validator(r)]
            if bad:
                result.dropped.append(
                    ShortcutHit("hallucinated_law_ref", crit.critique_id,
                                f"引用的法条不存在：{bad[0]}")
                )
                continue

        result.usable.append(crit)

    # 样本级判定：可用条数不足 → 不可训
    if len(result.usable) < min_usable:
        result.trainable = False
        # filter_reason 只记第一个原因，可读性优先
        result.reason = (
            result.dropped[0].rule if result.dropped
            else "no_critique"
        )
    else:
        result.trainable = True
        result.reason = ""

    return result
