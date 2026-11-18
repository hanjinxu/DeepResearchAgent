#***********************************************
#      Filename: export.py
#   Description: 导出 GRPO 训练样本（按草稿分组）
#***********************************************

"""样本导出。

格式是对外契约，字段名不要随意改：
    format = "critique-grpo/v1"

与 sentinel 那套 grpo/v1 的差别，以及为什么：
它导的是"一条轨迹一个样本"，组内比较靠 prompt 字段相同；
这里导的是"一组批评一个组"，显式带 groups 结构。原因是红队一次输出
可能包含多条批评（最多 4 条），把 4 条拆成 4 个样本会让它们共享同一个
reward——那等于人为制造了 4 份相关性极高的数据，组内方差会被压低。

组内归一化不在本模块做（与训练侧的职责边界一致），
但 group 里会带 mean / std 供检查——这就是训练前的方差体检。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from deep_research.flywheel.reward import normalize_advantages
from deep_research.flywheel.schema import SampleScore

FORMAT = "critique-grpo/v1"

DESCRIPTION = (
    "每个 group 对应一份固定草稿，samples 是红队对它的 k 条独立采样；"
    "reward.total 用于组内优势，trainable=False 的样本应丢弃而不是当负样本。"
)


@dataclass
class Group:
    """一份草稿 + 红队对它的 k 条采样。"""

    prompt: str                  # 草稿指纹，GRPO 的分组键
    draft: str = ""
    samples: List[SampleScore] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def trainable_samples(self) -> List[SampleScore]:
        return [s for s in self.samples if s.trainable]

    def mean_std(self) -> tuple[float, float]:
        vals = [s.total for s in self.samples]
        if not vals:
            return 0.0, 0.0
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        return round(mean, 4), round(std, 4)


def _stats(samples: Sequence[SampleScore]) -> Dict[str, object]:
    total = len(samples)
    trainable = sum(1 for s in samples if s.trainable)
    reasons: Dict[str, int] = {}
    for s in samples:
        if not s.trainable and s.filter_reason:
            reasons[s.filter_reason] = reasons.get(s.filter_reason, 0) + 1
    return {
        "total": total,
        "trainable": trainable,
        "filtered": total - trainable,
        "filter_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def build_groups(samples: Iterable[SampleScore], drafts: Optional[Dict[str, str]] = None) -> List[Group]:
    """按 prompt（草稿指纹）把样本聚成组，保持首次出现顺序。"""
    drafts = drafts or {}
    buckets: Dict[str, Group] = {}
    for s in samples:
        g = buckets.get(s.prompt)
        if g is None:
            g = Group(prompt=s.prompt, draft=drafts.get(s.prompt, ""))
            buckets[s.prompt] = g
        g.samples.append(s)
    return list(buckets.values())


def export_grpo(
    groups: Sequence[Group],
    path: str | Path,
    *,
    min_group_size: int = 2,
    include_advantages: bool = True,
) -> Dict[str, object]:
    """写出样本文件，返回 payload（便于测试与日志）。

    min_group_size：少于这个条数的组无法算组内方差，默认 2（GRPO 建议 k≥4）。
    include_advantages：把组内归一化结果一并写出，供人工检查；
    训练侧应当按自己的 k 重算，不要直接消费这个字段。
    """
    all_samples: List[SampleScore] = [s for g in groups for s in g.samples]
    stat = _stats(all_samples)

    out_groups: List[Dict[str, object]] = []
    skipped: List[str] = []

    for g in groups:
        if g.n < min_group_size:
            skipped.append(g.prompt)
            continue
        mean, std = g.mean_std()
        payload: Dict[str, object] = {
            "prompt": g.prompt,
            "n": g.n,
            "mean": mean,
            "std": std,
            "usable": std >= 0.05,
            "samples": [s.to_dict() for s in g.samples],
        }
        if include_advantages:
            payload["advantages"] = [
                round(a, 4) for a in normalize_advantages([s.total for s in g.samples])
            ]
        out_groups.append(payload)

    doc: Dict[str, object] = {
        "format": FORMAT,
        "description": DESCRIPTION,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stats": {**stat, "groups": len(out_groups), "skipped_groups": skipped},
        "groups": out_groups,
    }

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return doc


def load_export(path: str | Path) -> Dict[str, object]:
    """读回样本文件（回归测试与后续训练脚本共用）。"""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8"))
    if doc.get("format") != FORMAT:
        raise ValueError(f"不是 {FORMAT} 格式：{doc.get('format')!r}")
    return doc


def summarize_doc(doc: Dict[str, object]) -> str:
    """一行摘要，可直接打印在训练脚本开头。"""
    st = doc.get("stats") or {}
    groups = doc.get("groups") or []
    usable = sum(1 for g in groups if g.get("usable"))
    return (
        f"{doc.get('format')}｜样本 {st.get('total', 0)} 条（可训 {st.get('trainable', 0)}）"
        f"｜组 {st.get('groups', 0)} 个（有区分度 {usable}）"
    )
