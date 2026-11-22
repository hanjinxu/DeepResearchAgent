#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""画训练曲线 / 打印指标摘要。

读 train_grpo.py 写的 train_log.jsonl，输出：
    - PNG 曲线（装了 matplotlib 时）
    - 终端摘要表（任何环境都有，AutoDL 上直接看）

四个关键指标的判读规则都写在图标题里，避免"画了但不会看"。

用法：
    python scripts/plot_train_log.py --log data/flywheel/grpo_run/train_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


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
    return [r for r in rows if "skipped" not in r]


def _bar(v: float, lo: float, hi: float, width: int = 24) -> str:
    if hi - lo < 1e-9:
        return "·" * width
    n = int(max(0.0, min(1.0, (v - lo) / (hi - lo))) * width)
    return "█" * n + "·" * (width - n)


def _summarize(rows: List[dict]) -> Dict[str, float]:
    if not rows:
        return {}
    def avg(key: str) -> float:
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0
    first, last = rows[0], rows[-1]
    return {
        "steps": len(rows),
        "loss_avg": avg("loss"),
        "kl_avg": avg("kl"),
        "reward_first": first.get("reward_mean", 0.0),
        "reward_last": last.get("reward_mean", 0.0),
        "reward_std_avg": avg("reward_std"),
        "adv_abs_avg": avg("adv_abs_mean"),
        "len_avg": avg("comp_len"),
    }


def _print_table(rows: List[dict]) -> None:
    if not rows:
        print("没有有效步（全被跳过）。检查：组内方差是否为 0、解析是否失败。")
        return

    cols = ["reward_mean", "reward_std", "adv_abs_mean", "kl", "comp_len"]
    ranges = {}
    for c in cols:
        vals = [r.get(c, 0.0) for r in rows]
        ranges[c] = (min(vals), max(vals))

    print(f"{'step':>5}  " + "  ".join(f"{c:>26}" for c in cols))
    print("-" * (7 + 28 * len(cols)))
    show = rows if len(rows) <= 40 else rows[:: max(1, len(rows) // 40)]
    for r in show:
        cells = []
        for c in cols:
            v = r.get(c, 0.0)
            lo, hi = ranges[c]
            cells.append(f"{v:>7.3f} {_bar(v, lo, hi, 16)}")
        print(f"{r.get('step', 0):>5}  " + "  ".join(cells))


def _verdict(s: Dict[str, float]) -> List[str]:
    out = []
    if not s:
        return ["没有数据"]
    if s["reward_std_avg"] < 0.02:
        out.append("⚠️ reward_std 长期接近 0：组内没有区分度，训练学不到东西。"
                   "先查温度、k、以及奖励是否被常数项主导。")
    if s["adv_abs_avg"] < 0.05:
        out.append("⚠️ adv_abs_mean 过小：即使有方差，信号也很弱。")
    if abs(s["kl_avg"]) > 2.0:
        out.append("⚠️ KL 偏离过大：策略在追噪声，考虑调大 --kl-coef 或降 lr。")
    if s["reward_last"] > s["reward_first"] + 0.05:
        out.append(f"✅ 奖励上升 {s['reward_first']:.3f} → {s['reward_last']:.3f}")
    elif s["reward_last"] < s["reward_first"] - 0.05:
        out.append(f"⚠️ 奖励下降 {s['reward_first']:.3f} → {s['reward_last']:.3f}，"
                   f"检查奖励符号与优势计算。")
    else:
        out.append("奖励基本持平：要么步数太少，要么任务对当前模型太容易/太难。")
    return out


def _plot(rows: List[dict], png: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    # 图里只用英文：AutoDL 容器通常没有中文字体，中文会渲染成方框。
    # 终端摘要那边用中文，两边各取所长。
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    panels = [
        ("reward_mean", "reward mean (should climb slowly)", axes[0][0]),
        ("reward_std", "in-group std (near 0 = no signal)", axes[0][1]),
        ("kl", "KL vs reference (fast rise = chasing noise)", axes[1][0]),
        ("comp_len", "completion length (drop = learns to stay silent)", axes[1][1]),
    ]
    for key, title, ax in panels:
        ys = [r.get(key, 0.0) for r in rows]
        ax.plot(steps, ys, linewidth=1.2, color="#185FA5")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
    fig.suptitle("Red-team GRPO training curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="训练日志可视化")
    ap.add_argument("--log", default="data/flywheel/grpo_run/train_log.jsonl")
    ap.add_argument("--png", default="", help="默认与日志同目录的 curves.png")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"找不到日志：{path}", file=sys.stderr)
        return 2

    rows = _load(path)
    s = _summarize(rows)

    print("=" * 72)
    print(f"日志：{path}｜有效步 {s.get('steps', 0)}")
    if s:
        print(f"奖励 {s['reward_first']:+.3f} → {s['reward_last']:+.3f}｜"
              f"组内 std 均值 {s['reward_std_avg']:.3f}｜"
              f"adv 均值 {s['adv_abs_avg']:.3f}｜"
              f"KL 均值 {s['kl_avg']:+.3f}｜输出长度 {s['len_avg']:.0f}")
    print("=" * 72)
    _print_table(rows)

    print("\n判读：")
    for line in _verdict(s):
        print("  " + line)

    png = Path(args.png) if args.png else path.parent / "curves.png"
    if _plot(rows, png):
        print(f"\n曲线已保存：{png}")
    else:
        print("\n（未装 matplotlib，跳过 PNG。pip install matplotlib 后可出图）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
