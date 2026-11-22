#***********************************************
#      Filename: schema.py
#   Description: 批评飞轮的数据结构
#***********************************************

"""红队批评飞轮的结构定义。

三个设计决策，以及为什么这么定：

1. Critique 直接对位红队提示词的三段式输出
   （【原文位置】/【问题】/【解决方法】）。
   不另造 JSON schema 让模型重新输出——那会多一次调用、多一次格式失败，
   而这些字段本来就是模型已经在产出的内容。

2. subitem 直接复用提示词里的 10 个子项编号（1.1~4.1）作为错误分类。
   不重新发明 taxonomy：红队提示词已经用这套编号组织审查方向，
   再叠一层自定义分类只会导致"同一件事两个名字"，标注和训练都混乱。

3. severity 由 subitem 推导，而不是让模型自评。
   模型普遍高估自己指出问题的严重度（几乎全标"严重"），
   而子项编号到严重度的映射可以一次性标定、全批一致、可审计。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 子项编号 1.1 ~ 4.3。
# 前向排除 \d.vV —— 避免把 "v1.2" 这类版本号认成子项；
# 后向排除数量单位 —— 避免把 "2.2亿" "3.1%" 认成子项。
SUBITEM_RE = re.compile(r"(?<![\d.vV])([1-4]\.[1-3])(?![\d亿万%％倍个元人年余])")

# ===== 严重度标定 =====
# 依据：这个问题如果没被抓到，会不会直接导致法务决策错误。
#   critical —— 事实失实 / 风险定性错 / 风险等级错：结论从根上错
#   major    —— 遗漏冲突源 / 遗漏衍生风险 / 缺法定动作 / 风险与动作脱节
#   minor    —— 要素不全 / 建议不可执行 / 缺时效标签：影响执行细节，不动摇结论
SEVERITY_BY_SUBITEM: Dict[str, str] = {
    "1.1": "critical",   # 事实/传闻分级错误
    "1.2": "major",      # 多源冲突信息遗漏
    "1.3": "minor",      # 关键要素缺失
    "2.1": "critical",   # 风险定性错误
    "2.2": "critical",   # 风险等级误判
    "2.3": "major",      # 衍生风险遗漏
    "3.1": "minor",      # 建议不可执行
    "3.2": "major",      # 关键动作遗漏
    "3.3": "minor",      # 时效要求缺失
    "4.1": "major",      # 风险-动作脱节
}

SEVERITY_WEIGHT: Dict[str, float] = {
    "critical": 1.0,
    "major": 0.6,
    "minor": 0.2,
}

VALID_SEVERITIES = tuple(SEVERITY_WEIGHT.keys())


def severity_of(subitem: str, default: str = "major") -> str:
    """由子项编号推严重度；未知编号回落到 default。"""
    return SEVERITY_BY_SUBITEM.get(subitem, default)


def stable_id(prefix: str, *parts: str, n: int = 10) -> str:
    """跨进程稳定的短 ID。

    不能用内置 hash()：Python 每次启动的字符串哈希带随机盐，
    同一条批评在不同进程里会得到不同 ID，数据集比对和去重全部失效。
    """
    payload = "|".join(p for p in parts)
    return f"{prefix}-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:n]


@dataclass
class Critique:
    """红队的一条批评意见（对应提示词里的一个三段式块）。"""

    target_section: str            # 【原文位置】
    problem: str                   # 【问题】
    suggestion: str = ""           # 【解决方法】（可为空：训练核实的是问题本身）
    subitem: str = ""              # 1.1~4.1，从 problem 里抽取
    severity: str = ""             # 空 = 由 subitem 推导（不要在这里写默认值，
                                   # 否则默认值会掩盖推导结果，导致所有批评都是 major）
    critique_id: str = ""

    def __post_init__(self) -> None:
        # 子项编号允许从 problem 里回填：调用方常常只有一段文本，
        # 强制要求先解析再构造会让所有测试和一次性脚本都要重复一遍抽取逻辑。
        if not self.subitem:
            m = SUBITEM_RE.search(self.problem or "") or SUBITEM_RE.search(self.target_section or "")
            self.subitem = m.group(1) if m else ""
        if not self.severity or self.severity not in VALID_SEVERITIES:
            self.severity = severity_of(self.subitem)
        if not self.critique_id:
            self.critique_id = stable_id(
                "cr", self.subitem, self.target_section, self.problem[:80]
            )

    @property
    def weight(self) -> float:
        return SEVERITY_WEIGHT.get(self.severity, 0.6)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critique_id": self.critique_id,
            "subitem": self.subitem,
            "severity": self.severity,
            "target_section": self.target_section,
            "problem": self.problem,
            "suggestion": self.suggestion,
        }


@dataclass
class Verification:
    """一条批评的核实结果。

    verdict 三态而不是布尔：
      valid        —— 核实成立，正奖励
      invalid      —— 误报，负奖励（不是丢弃：误报是有价值的学习信号）
      unverifiable —— 无法机器核实（需要人工或 judge），不计分
    把 invalid 和 unverifiable 混成一个"不成立"，会让红队学到"只提敢保证的浅层问题"。
    """

    critique_id: str
    verdict: str                   # valid | invalid | unverifiable
    reason: str = ""
    verified_by: str = ""          # rule | judge | human

    VALID_VERDICTS = ("valid", "invalid", "unverifiable")

    def __post_init__(self) -> None:
        if self.verdict not in self.VALID_VERDICTS:
            raise ValueError(f"未知 verdict: {self.verdict}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critique_id": self.critique_id,
            "verdict": self.verdict,
            "reason": self.reason,
            "verified_by": self.verified_by,
        }


@dataclass
class ShortcutHit:
    """命中"捷径"规则——这类样本不该进训练集。"""

    rule: str
    critique_id: str = ""
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"rule": self.rule, "critique_id": self.critique_id, "detail": self.detail}


@dataclass
class SampleScore:
    """一次红队输出的完整打分结果。"""

    prompt: str                              # GRPO 分组键（草稿的指纹）
    critiques: List[Critique] = field(default_factory=list)
    verifications: List[Verification] = field(default_factory=list)
    shortcut_hits: List[ShortcutHit] = field(default_factory=list)
    breakdown: Dict[str, float] = field(default_factory=dict)
    total: float = 0.0
    trainable: bool = False
    filter_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "critiques": [c.to_dict() for c in self.critiques],
            "verifications": [v.to_dict() for v in self.verifications],
            "shortcut_hits": [s.to_dict() for s in self.shortcut_hits],
            "reward": {**self.breakdown, "total": round(self.total, 4)},
            "trainable": self.trainable,
            "filter_reason": self.filter_reason,
        }


def critique_fingerprint(text: str) -> str:
    """草稿指纹，用作 GRPO 的分组键。

    同一份草稿采出的 k 条批评必须落在同一组，否则组内比较失去意义。
    用内容指纹而不是文件名：同一份草稿复制多份（比如 A/B 实验）仍然同组。
    """
    norm = "".join((text or "").split())
    return stable_id("draft", norm, n=12)


@dataclass
class CheckVerdict:
    """红队对评分卡某一条的判定（新版「逐条核对」任务的输出单元）。

    与 Critique 的区别：Critique 是"自由发现的问题"，CheckVerdict 是"对指定检查项的判定"。
    后者的好处是**可以被规则核对**——rubrics.py 跑一遍就知道红队判得对不对。
    """

    check_id: str
    status: str                    # pass | fail | unknown
    reason: str = ""
    quote: str = ""                # 判 fail 时必须有的逐字引句

    STATUSES = ("pass", "fail", "unknown")

    def __post_init__(self) -> None:
        if self.status not in self.STATUSES:
            raise ValueError(f"未知状态: {self.status}")

    @property
    def passed(self) -> Optional[bool]:
        return {"pass": True, "fail": False}.get(self.status)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "check_id": self.check_id, "status": self.status,
            "passed": self.passed, "reason": self.reason, "quote": self.quote,
        }
