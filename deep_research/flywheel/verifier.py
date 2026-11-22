#***********************************************
#      Filename: verifier.py
#   Description: 核实红队批评是否成立（奖励的信号源）
#***********************************************

"""核实器：判断"这条批评指出的问题，在草稿里是否真的存在"。

语义约定（很重要，别理解反了）：
    核实的是「草稿有没有这类错误」，不是「批评的措辞准不准」。
    草稿里确实存在该错误 → 提了它的批评 = valid
    草稿里不存在该错误       → 提了它的批评 = invalid（误报）
    规则判不了               → unverifiable（不计分）

为什么按"类"判而不是按"句"判：
红队的批评是自由文本，把"这条批评精确对应哪一行"做成可靠解析的成本，
远高于它带来的收益。而按类判已经足够训练出我们要的东西——
让红队学会**挑草稿真有问题的那几处**。这也正是它的决策空间：
同样的草稿，有的轨迹去比 2.3 与 3.2，有的去比 1.1 与 2.1，命中率不同。

两层结构：
    RuleVerifier    零成本、高精度、低覆盖。只处理子项与规则直接对应的情形。
    JudgeVerifier   调独立 judge 模型，覆盖高但要花 token。
    CombinedVerifier 规则优先，判不了再问 judge。

覆盖率的现实预期：规则只能覆盖一小部分（来源分级、时效标签、时间线日期）。
如果 CombinedVerifier 仍有大量 unverifiable，说明 judge 也在回避——
那时该检查 judge 的提示词，而不是放宽规则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from deep_research.flywheel.schema import Critique, Verification

# ===== 草稿结构解析 =====

_SECTION_RE = re.compile(r"^#{2,4}\s*(\d\.\d)\s*(.*)$", re.M)
_GENERIC_HEADING_RE = re.compile(r"^#{1,4}\s", re.M)

# 自媒体/低可信特征的来源类型关键词。
# 刻意不含"百科"：维基/百度百科属于聚合型来源，评为"中"是行业惯例，
# 纳入会制造大量误报（实测把"维基百科、百度百科→中"判成错，属误判）。
_SELF_MEDIA_HINTS = (
    "自媒体", "知乎", "微博", "微信公众号", "公众号", "论坛", "贴吧",
    "网友", "个人", "抖音", "快手", "小红书", "头条号", "百家号", "营销号",
)

# 时效标签：出现了这些，才认为"标了截止时间"
_TIME_MARKERS = re.compile(
    r"\d+\s*(?:h|H|小时|天|个工作日|工作日)|24h|48h|72h|立即|当日|即日|T\+\d"
)

_HIGH_TIERS = ("高",)
_MID_TIERS = ("中",)
_LOW_TIERS = ("低",)


def split_sections(draft: str) -> Dict[str, str]:
    """把草稿按 1.1 / 1.2 / 2.1 … 标题切成段。找不到编号标题时返回空 dict。"""
    marks: List[Tuple[str, int, int]] = []
    for m in _SECTION_RE.finditer(draft):
        marks.append((m.group(1), m.start(), m.end()))
    if not marks:
        return {}

    # 每段的结束点：下一个任意级别标题
    all_heads = [m.start() for m in _GENERIC_HEADING_RE.finditer(draft)]
    out: Dict[str, str] = {}
    for i, (num, start, body_start) in enumerate(marks):
        nxt_head = next((h for h in all_heads if h > start), len(draft))
        out[num] = draft[body_start:nxt_head]
    return out


def _table_rows(section: str) -> List[List[str]]:
    """抽出 markdown 表格的数据行（跳过表头与分隔线）。"""
    rows: List[List[str]] = []
    for line in section.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells if c):
            continue  # 分隔线
        rows.append(cells)
    return rows[1:] if rows else rows  # 去表头


def _has_tier(cell: str, tiers: Sequence[str]) -> bool:
    return any(t in cell for t in tiers)


# ===== 规则核实 =====

@dataclass
class RuleFinding:
    """一条规则对某类错误是否存在的判定。"""

    exists: Optional[bool]        # True=草稿确有该错 False=没有 None=判不了
    reason: str = ""


class RuleVerifier:
    """按子项分派到确定性规则。"""

    # 每条规则只处理"子项与规则直接对应"的情形，避免把不相干的批评判成 valid
    SUPPORTED = {"2.3", "3.3", "1.2"}

    def __init__(self, draft: str):
        self.draft = draft or ""
        self.sections = split_sections(self.draft)

    # ---- 2.3 来源分级：把自媒体类评成"高/中"就是错 ----
    def _check_source_tier(self) -> RuleFinding:
        sec = self.sections.get("2.3", "")
        if not sec:
            return RuleFinding(None, "草稿里没有 2.3 节")
        rows = _table_rows(sec)
        if not rows:
            return RuleFinding(None, "2.3 不是表格，无法解析评级")

        bad = []
        for row in rows:
            if len(row) < 2:
                continue
            src, tier_cell = row[0], row[1]
            if not any(h in src for h in _SELF_MEDIA_HINTS):
                continue
            # "低至中" 这种写法含"低"，按低处理——不判错
            if _has_tier(tier_cell, _LOW_TIERS):
                continue
            if _has_tier(tier_cell, _HIGH_TIERS) or _has_tier(tier_cell, _MID_TIERS):
                bad.append(f"{src}→{tier_cell}")

        if bad:
            return RuleFinding(True, f"2.3 存在自媒体来源被评为高/中：{bad[0]}")
        return RuleFinding(False, "2.3 的自媒体来源评级均为低，未发现错评")

    # ---- 3.3 时效标签：4.x 里完全没有时间标记就是缺失 ----
    def _check_timeliness(self) -> RuleFinding:
        sec4 = "".join(v for k, v in self.sections.items() if k.startswith("4."))
        if not sec4.strip():
            return RuleFinding(None, "草稿里没有 4.x 处置建议段")
        if _TIME_MARKERS.search(sec4):
            return RuleFinding(False, "4.x 已带时效标签，未发现缺失")
        return RuleFinding(True, "4.x 全段没有任何时限标记（h/小时/工作日/立即等）")

    # ---- 1.2 时间线：来源为空 / 来源日期早于事件时间 ----
    def _check_timeline_date(self) -> RuleFinding:
        sec = self.sections.get("1.2", "")
        if not sec:
            return RuleFinding(None, "草稿里没有 1.2 节")
        rows = _table_rows(sec)
        if not rows:
            return RuleFinding(None, "1.2 不是表格")

        # 先查来源为空——这是最常见也最容易判的一类。
        # 不加这一条会漏判：真实报告里"本报告出具日"那行来源就是 "—"，
        # 批评说"缺来源"是对的，只查日期倒置会判成误报。
        empty_src = [
            row[0][:20] for row in rows
            if len(row) >= 3 and row[2].strip() in ("", "—", "-", "--", "无", "N/A")
        ]
        if empty_src:
            return RuleFinding(True, f"1.2 有 {len(empty_src)} 行来源为空：{empty_src[0]}")

        # 再查日期倒置：只在同一行里同时出现两个日期时才比
        mismatches = []
        for row in rows:
            if len(row) < 3:
                continue
            when, src = row[0], row[2]
            d_when = _first_date(when)
            d_src = _first_date(src)
            if d_when and d_src and d_src < d_when:
                mismatches.append(f"{when} vs 来源 {d_src}")
        if mismatches:
            return RuleFinding(True, f"1.2 存在来源日期早于事件时间的行：{mismatches[0]}")
        return RuleFinding(False, "1.2 每行都有来源，且未发现日期倒置")

    def check(self, subitem: str) -> RuleFinding:
        if subitem == "2.3":
            return self._check_source_tier()
        if subitem == "3.3":
            return self._check_timeliness()
        if subitem == "1.2":
            return self._check_timeline_date()
        return RuleFinding(None, f"子项 {subitem or '未编号'} 没有对应规则")

    def verify(self, crit: Critique) -> Verification:
        f = self.check(crit.subitem)
        if f.exists is None:
            return Verification(crit.critique_id, "unverifiable", f.reason, "rule")
        verdict = "valid" if f.exists else "invalid"
        return Verification(crit.critique_id, verdict, f.reason, "rule")


_DATE_RE = re.compile(r"(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})")


def _first_date(text: str) -> Optional[Tuple[int, int, int]]:
    m = _DATE_RE.search(text or "")
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    except ValueError:
        return None


# ===== Judge 核实 =====

JUDGE_PROMPT = """你是法务报告的审校员。下面给你一份报告草稿的节选，以及一条审校意见。
请只判断一件事：**这条意见指出的问题，在草稿里是否真的存在**。

不要评价意见写得是否得体，不要补充新的问题，只判存在性。

草稿：
<draft>
{draft}
</draft>

审校意见（子项 {subitem}）：
<critique>
位置：{target}
问题：{problem}
</critique>

只输出 JSON，不要任何解释文字：
{{"exists": true/false/"unknown", "reason": "不超过 40 字的依据"}}

- true：草稿里确实存在这个问题
- false：草稿里不存在，意见属于误报
- unknown：仅凭草稿无法判断（需要外部信源、法条或人工）
"""


class JudgeVerifier:
    """用独立 judge 模型判存在性。

    必须用与 policy 不同源的模型——同源自评会把模型自己的偏好当成"事实"。
    本类不负责挑选模型，调用方传入 role（默认 evaluator）。
    """

    def __init__(self, model=None, role: str = "evaluator", max_draft_chars: int = 12000):
        if model is None:
            from deep_research.llm import get_chat_model

            model = get_chat_model(role)
        self.model = model
        self.max_draft_chars = max_draft_chars

    def verify(self, crit: Critique, draft: str) -> Verification:
        from langchain_core.messages import HumanMessage

        from deep_research.memory.json_utils import extract_json_object

        prompt = JUDGE_PROMPT.format(
            draft=(draft or "")[: self.max_draft_chars],
            subitem=crit.subitem or "未编号",
            target=crit.target_section[:120],
            problem=crit.problem[:400],
        )
        try:
            resp = self.model.invoke([HumanMessage(content=prompt)])
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            obj = extract_json_object(text)
        except Exception as exc:  # noqa: BLE001
            return Verification(crit.critique_id, "unverifiable", f"judge 调用失败: {str(exc)[:80]}", "judge")

        raw = obj.get("exists")
        reason = str(obj.get("reason", ""))[:120]
        if raw is True or str(raw).lower() in ("true", "yes", "1"):
            return Verification(crit.critique_id, "valid", reason, "judge")
        if raw is False or str(raw).lower() in ("false", "no", "0"):
            return Verification(crit.critique_id, "invalid", reason, "judge")
        return Verification(crit.critique_id, "unverifiable", reason or "judge 判为 unknown", "judge")


class CombinedVerifier:
    """规则优先，判不了再问 judge。

    规则命中时不花 token，这是省钱的关键：
    2.3 / 3.3 / 1.2 这三类在真实报告里占比不低，且判起来是确定性的。
    """

    def __init__(self, draft: str, judge: Optional[JudgeVerifier] = None):
        self.draft = draft or ""
        self.rule = RuleVerifier(self.draft)
        self.judge = judge

    def verify_all(self, critiques: Sequence[Critique]) -> List[Verification]:
        out: List[Verification] = []
        for c in critiques:
            v = self.rule.verify(c)
            if v.verdict == "unverifiable" and self.judge is not None:
                v = self.judge.verify(c, self.draft)
            out.append(v)
        return out

    def coverage(self, verifications: Sequence[Verification]) -> Dict[str, object]:
        """核实覆盖率：多少条真正被判了（valid+invalid）。"""
        n = len(verifications)
        judged = sum(1 for v in verifications if v.verdict in ("valid", "invalid"))
        return {
            "total": n,
            "judged": judged,
            "unverifiable": n - judged,
            "coverage": round(judged / n, 4) if n else 0.0,
        }
