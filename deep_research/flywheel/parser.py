#***********************************************
#      Filename: parser.py
#   Description: 把红队的自由文本输出解析成结构化批评
#***********************************************

"""解析红队输出。

为什么写解析器而不是让红队直接输出 JSON：
红队提示词已经约定三段式（【原文位置】/【问题】/【解决方法】），
并且这个格式对模型更稳——JSON 转义失败（尤其带引号、换行的法条引用）
是自由输出场景的高频故障。与其改 prompt 再赌一次格式遵循，
不如在本地把确定性格式解析干净。

解析器对残缺输出保持宽容：缺哪段就留空，交给 filters 去判"不可用"，
而不是在解析阶段直接丢弃——否则你无法区分"模型没说"和"解析器没认出来"。
"""

from __future__ import annotations

import re
from typing import List

from deep_research.flywheel.schema import SUBITEM_RE, Critique, severity_of

TAG_POS = "【原文位置】"
TAG_PROB = "【问题】"
TAG_FIX = "【解决方法】"

# 标签容错：模型偶尔会写成【 原文位置 】或【原文位置:】
_TAG_NORMALIZE = [
    (re.compile(r"【\s*原文位置\s*[:：]?\s*】"), TAG_POS),
    (re.compile(r"【\s*问题\s*[:：]?\s*】"), TAG_PROB),
    (re.compile(r"【\s*解决方法\s*[:：]?\s*】"), TAG_FIX),
    (re.compile(r"【\s*解决建议\s*】"), TAG_FIX),
]

# 泛化建议词：只有这类词而没有数字/编号指向时，判定为"空话"
_GENERIC_ADVICE = re.compile(r"加强|完善|优化|补充|丰富|提升|改进|注意|关注|细化|规范")
_HAS_REFERENCE = re.compile(r"\d")

MIN_PROBLEM_CHARS = 12


def _is_pass(text: str) -> bool:
    """判断是否为 PASS（无问题）。

    只在完全没有三段式标签时才认定 PASS——否则模型在批评正文里
    引用一句"这里不能写 PASS"就会被误判成通过。
    """
    if TAG_POS in text:
        return False
    stripped = re.sub(r"[\s`*_。.]", "", text).upper()
    return stripped in ("PASS", "通过", "无问题", "")


def _clean(segment: str) -> str:
    s = segment.strip()
    s = re.sub(r"^[：:\-\s]+", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _extract_subitem(problem: str) -> str:
    m = SUBITEM_RE.search(problem or "")
    return m.group(1) if m else ""


def parse_critiques(text: str) -> List[Critique]:
    """把一次红队输出解析成 Critique 列表。空列表表示 PASS 或无法解析。"""
    if not text or not text.strip():
        return []

    raw = text
    for pattern, repl in _TAG_NORMALIZE:
        raw = pattern.sub(repl, raw)

    if _is_pass(raw):
        return []

    # 以【原文位置】为分隔点切块；没有标签时整段当作一块
    blocks = re.split(r"(?=%s)" % re.escape(TAG_POS), raw)
    out: List[Critique] = []

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if TAG_POS not in block:
            # 完全没标签的散文本：整块当 problem，保留下来让 filters 判不可用
            problem = _clean(block)
            if problem:
                out.append(Critique(target_section="", problem=problem, suggestion=""))
            continue

        body = block.split(TAG_POS, 1)[1]

        if TAG_PROB in body:
            target_raw, rest = body.split(TAG_PROB, 1)
        else:
            target_raw, rest = body, ""

        if TAG_FIX in rest:
            problem_raw, suggestion_raw = rest.split(TAG_FIX, 1)
        else:
            problem_raw, suggestion_raw = rest, ""

        target = _clean(target_raw)
        problem = _clean(problem_raw)
        suggestion = _clean(suggestion_raw)

        if not (target or problem or suggestion):
            continue

        subitem = _extract_subitem(problem) or _extract_subitem(target)
        out.append(
            Critique(
                target_section=target,
                problem=problem,
                suggestion=suggestion,
                subitem=subitem,
                severity=severity_of(subitem),
            )
        )

    return out


def is_vague_suggestion(text: str) -> bool:
    """判断一句批评是不是纯空话（"建议加强论证"这类）。

    判据是"短 + 无任何编号指向 + 只有泛化动词"：
      - 够长（≥12 字）→ 不判空话，长句通常有实质内容
      - 含数字/编号（"补充 2.1 节的来源"）→ 有具体指向，不算空话
      - 剩下只有"建议加强/完善…"这类词 → 空话

    只拦"整句就是空话"的情况，不拦"具体建议里带了一个通用词"——
    后者是正常表达，拦了会误杀。
    """
    s = _clean(text or "")
    if not s:
        return True
    if len(s) >= MIN_PROBLEM_CHARS:
        return False
    if _HAS_REFERENCE.search(s):
        return False
    return bool(_GENERIC_ADVICE.search(s))
