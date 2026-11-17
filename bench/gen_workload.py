#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
引擎层压测负载生成器（实验 1 · 问题 A 的第一步）

核心思想
--------
压测负载不能"随机编"。vLLM 自带的 `--dataset-name random` 只支持单一均匀分布，
而本项目的输入长度是**按角色分簇的混合分布**：

    网页摘要类   ~5.4K  （占调用绝大多数，prefill token 占比也最大）
    ReAct 累积   ~3~12K （随轮次线性增长）
    supervisor   ~2~6K
    最终报告     ~21K   （最长的一条）

用均匀分布压测，prefill 负载形状从第一步就是错的，测出来的 TTFT 没有参考价值。

本脚本的做法：**严格镜像 cost_estimate.py 里 simulate() 的调用序列**，
逐条还原一次完整研究的 LLM 调用清单（每次调用的 input/output token），
再展开成 vLLM bench 能直接吃的 jsonl。

镜像逻辑有一道保险：`--dry-run` 会拿镜像出来的序列去和 simulate() 的聚合值
逐个对账，对不上直接报错。所以 cost_estimate.py 改了参数，这里不会静默漂移。

两种前缀模式（用来分离"裸推理能力"和"agent 真实收益"）
-----------------------------------------------------
  isolated  每条请求从语料随机位置切窗口 → 前缀互不共享 → prefix cache 命中率 ≈ 0
            测的是引擎的**裸推理能力**，是量化对比该用的口径
  shared    每条请求前 N token 用同一段固定前缀（N = 该角色的 system prompt 长度）
            → prefix cache 真的会命中 → 测的是 **agent 场景下的真实吞吐**
            报出来的命中率必须一起写进报告，否则就是虚高

用法
----
    # 1. 只对账 + 打印分布，不写文件
    python bench/gen_workload.py --dry-run

    # 2. 生成基准档负载（约 222 条请求）
    python bench/gen_workload.py --out bench/workload/mix.jsonl

    # 3. 扫参用的小负载：只保留网页摘要这一簇（prefill 最密集、最快出信号）
    python bench/gen_workload.py --role-filter researcher_summarizer --max-requests 128 \
        --out bench/workload/prefill_heavy.jsonl

    # 4. agent 真实收益口径（带共享前缀）
    python bench/gen_workload.py --mode shared --out bench/workload/mix_shared.jsonl
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = BENCH_DIR / "workload" / "mix.jsonl"


# ---------------------------------------------------------------------------
# 一、调用名 → config.yml 里的 role → 走哪个 vLLM 实例
# ---------------------------------------------------------------------------
# 注意：role 决定这条请求落在哪个推理实例上。config.yml 里
# red_team 走 qwen_redteam(:8001)，其余全部走 qwen_general(:8000)。
# 分实例生成负载，才能分别对两个实例做容量规划。
ROLE_OF: Dict[str, str] = {
    "记忆·事件键抽取": "event_judge",
    "记忆·L3判重": "event_judge",
    "记忆·delta抽取": "event_judge",
    "记忆·归档": "event_judge",
    "研究简报": "draft",
    "报告初稿": "draft",
    "报告初稿(增量简)": "draft",
    "主线·supervisor": "supervisor",
    "子agent·ReAct": "researcher_main",
    "子agent·网页摘要": "researcher_summarizer",
    "子agent·网页正文": "researcher_summarizer",
    "子agent·压缩": "researcher_compressor",
    "对抗·红队": "red_team",
    "对抗·评分": "evaluator",
    "最终报告": "writer",
    "最终报告·findings": "writer",
}

INSTANCE_OF: Dict[str, str] = {"red_team": "qwen_redteam"}

# ---------------------------------------------------------------------------
# 一之二、把成本模型的"缓存分档"拆条合并回真实的一次调用
# ---------------------------------------------------------------------------
# cost_estimate 为了分别计费，把一次逻辑调用拆成两条：
#   「子agent·网页摘要」(1218 tok, 有输出) + 「子agent·网页正文」(4200 tok, 无输出)
#   「最终报告」(模板 4227) + 「最终报告·findings」(14300, 无输出)
# 计费上等价，但**压测上不等价**：真实请求是一次 5.4K / 18K 的 prefill，
# 拆成两次会显著改变 TTFT 分布（chunked prefill 的分块行为完全不同）。
# 默认合并，`--no-merge` 可关。
MERGE_INTO: Dict[str, str] = {
    "子agent·网页正文": "子agent·网页摘要",
    "最终报告·findings": "最终报告",
}


def merge_companions(calls: Sequence[Call]) -> Tuple[List[Call], List[str]]:
    """把「无输出」的伴随条目合并进它的宿主调用，还原真实单次请求形状。

    合并前提：两者 n 相同（同一次调用批同一批数据）。
    """
    merged: List[Call] = []
    notes: List[str] = []
    index = {c.name: c for c in calls}
    consumed = set()
    for c in calls:
        if c.name in consumed:
            continue
        host_name = MERGE_INTO.get(c.name)
        if host_name and host_name in index:
            host = index[host_name]
            if host.n == c.n:
                host.in_tokens += c.in_tokens
                consumed.add(c.name)
                notes.append(
                    f"{c.name}({c.in_tokens:,}) → {host_name}，"
                    f"合并后 {host.in_tokens:,} tok × {host.n} 次"
                )
                continue
        merged.append(c)
    return merged, notes


# ---------------------------------------------------------------------------
# 一之三、从真实运行日志提取经验长度分布（用来校准合成负载）
# ---------------------------------------------------------------------------
def parse_log_lengths(path: Path) -> Tuple[List[int], List[int]]:
    """从 LangChain 运行日志里抠出 (input_tokens, output_tokens) 序列。

    日志里每条 usage_metadata 都是真实发生过的一次调用——这是**不需要额外埋点**
    就能拿到的真实负载数据，比任何假设都可信。
    """
    import re

    if not path.exists():
        return [], []
    text = path.read_text(encoding="utf-8", errors="replace")
    ins = [int(x) for x in re.findall(r"input_tokens'?:\s*(\d+)", text)]
    outs = [int(x) for x in re.findall(r"output_tokens'?:\s*(\d+)", text)]
    return ins, outs


def _quantiles(xs: Sequence[int]) -> Dict[str, int]:
    if not xs:
        return {}
    s = sorted(xs)
    n = len(s)
    return {
        "n": n,
        "P50": s[min(n - 1, int(n * 0.50))],
        "P90": s[min(n - 1, int(n * 0.90))],
        "P99": s[min(n - 1, int(n * 0.99))],
        "max": s[-1],
    }


def compare_with_log(model_ins: Sequence[int], log_ins: Sequence[int]) -> bool:
    """把合成分布和真实日志分布摆在一起——差异本身就是结论。"""
    if not log_ins:
        return False
    m = _quantiles(model_ins)
    l = _quantiles(log_ins)
    print(f"\n{'-' * 88}")
    print(f"{'输入长度分位':<16}{'合成负载(成本模型)':>22}{'真实日志':>16}{'倍数':>10}")
    print("-" * 88)
    for k in ("P50", "P90", "P99", "max"):
        ratio = l[k] / m[k] if m.get(k) else 0.0
        flag = "  ← 模型低估" if ratio > 1.5 else ""
        print(f"{k:<16}{m.get(k, 0):>22,}{l.get(k, 0):>16,}{ratio:>9.2f}x{flag}")
    print("-" * 88)
    underestimated = l["max"] > m["max"] * 1.5
    if underestimated:
        print("结论：真实运行中的上下文累积远超成本模型假设。")
        print("      合成负载若按下限走，会低估 prefill 压力，TTFT 偏乐观。")
        print("      用 `--length-source log` 让合成负载直接采用实测长度分布")
        print("      （角色构成仍来自成本模型，两边各取所长）。")
    return underestimated


# 各角色 system prompt 的长度（token）。shared 模式下作为共享前缀长度。
# 数值取自 cost_estimate.PROMPT_TOKENS，这里只做"哪部分是他角色固定前缀"的映射。
SHARED_PREFIX_OF_ROLE: Dict[str, str] = {
    "researcher_summarizer": "summarize_sys",
    "researcher_main": "researcher_sys",
    "researcher_compressor": "compress_sys",
    "supervisor": "supervisor_sys",
    "red_team": "red_team",
    "evaluator": "evaluator",
    "draft": "draft_report",
    "writer": "final_report",
    "event_judge": "event_key_extract",
}


# ---------------------------------------------------------------------------
# 二、加载成本模型并镜像它的调用序列
# ---------------------------------------------------------------------------
def load_cost_model():
    """从仓库根目录加载 cost_estimate.py（它不在包里，用 importlib 直接拉）。"""
    path = REPO_ROOT / "cost_estimate.py"
    if not path.exists():
        raise SystemExit(f"找不到 {path}，请在仓库根目录下运行")
    spec = importlib.util.spec_from_file_location("cost_estimate", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Call:
    """一次 LLM 调用。n = 这个调用会被执行多少次（摘要类一次调用批 N 份网页）。"""

    name: str
    role: str
    in_tokens: int
    out_tokens: int
    n: int = 1
    cache: str = "static"

    @property
    def instance(self) -> str:
        return INSTANCE_OF.get(self.role, "qwen_general")

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "instance": self.instance,
            "in_tokens": self.in_tokens,
            "out_tokens": self.out_tokens,
            "n": self.n,
            "cache": self.cache,
        }


def iter_calls(
    cm: Any,
    *,
    sup_rounds: int,
    researchers_per_round: float,
    react_turns: int,
    searches_per_turn: int = 1,
    results_per_search: int = 3,
    is_legal: bool = True,
) -> List[Call]:
    """镜像 cost_estimate.simulate() 的调用序列，逐条 emit。

    ⚠️ 这个函数必须和 cost_estimate.simulate() 保持严格同构。
       _reconcile() 会拿聚合结果对账，改了一边不改另一边会立刻报错。
    """
    P = cm.PROMPT_TOKENS
    calls: List[Call] = []

    def add(name: str, i: int, o: int, n: int = 1, cache: str = "static") -> None:
        calls.append(
            Call(name=name, role=ROLE_OF.get(name, name), in_tokens=i,
                 out_tokens=o, n=n, cache=cache)
        )

    # --- 阶段 0：事件记忆层 ---
    add("记忆·事件键抽取", P["event_key_extract"], 120)
    if cm.P_L3_JUDGE > 0:
        add("记忆·L3判重", 1400 + 5 * 220, 180, 1, cache="hist")
    add("记忆·delta抽取", 6800, cm.DELTA_OUT, 1, cache="hist")

    # --- 阶段 1-2：简报 + 初稿 ---
    add("研究简报", P["research_brief"] + cm.CLARIFY_TOKENS, cm.BRIEF_OUT)
    add("报告初稿", P["draft_report"] + 400, cm.DRAFT_OUT, 1, cache="hist")

    # --- 阶段 3：supervisor 主循环 ---
    n_researchers = int(round(sup_rounds * researchers_per_round))
    sup_accum = 0
    tool_msg_per_round = 200 + researchers_per_round * cm.COMPRESS_OUT
    for _ in range(sup_rounds):
        add("主线·supervisor", P["supervisor_sys"] + sup_accum, 180, 1, cache="hist")
        sup_accum += tool_msg_per_round + 60

    # --- 阶段 4：researcher 子 agent ---
    for _ in range(n_researchers):
        accum = 0
        for _t in range(react_turns):
            add("子agent·ReAct", P["researcher_sys"] + accum, cm.AI_TOOLCALL_OUT, 1, cache="hist")
            n_sum = searches_per_turn * results_per_search
            add("子agent·网页摘要", P["summarize_sys"], cm.SUMMARY_OUT, n_sum, cache="static")
            add("子agent·网页正文", cm.PAGE_TOKENS, 0, n_sum, cache="fresh")
            accum += cm.AI_TOOLCALL_OUT + searches_per_turn * results_per_search * cm.TOOL_MSG_PER_RESULT
        add("子agent·压缩", P["compress_sys"] + accum, cm.COMPRESS_OUT, 1, cache="hist")

    # --- 阶段 5-6：对抗 ---
    n_red = max(1, round(sup_rounds * 0.25))
    add("对抗·红队", P["red_team"] + cm.BRIEF_OUT + cm.DRAFT_OUT, cm.REDTEAM_OUT, n_red, cache="hist")
    n_eval = max(1, round(sup_rounds * 0.3))
    add("对抗·评分", P["evaluator"] + cm.BRIEF_OUT + cm.DRAFT_OUT, cm.EVAL_OUT, n_eval, cache="hist")

    # --- 阶段 7-8：终稿 + 归档 ---
    findings = n_researchers * cm.COMPRESS_OUT + sup_rounds * 200
    tpl = P["final_legal"] if is_legal else P["final_report"]
    add("最终报告", tpl, cm.FINAL_OUT)
    add("最终报告·findings", findings + cm.DRAFT_OUT + cm.BRIEF_OUT, 0, 1, cache="fresh")
    add("记忆·归档", 800 + cm.FINAL_OUT, cm.DELTA_OUT, 1, cache="hist")

    return calls


def reconcile(cm: Any, calls: Sequence[Call], scenario: Tuple[str, int, float, int]) -> Dict[str, Any]:
    """对账：镜像序列的聚合值必须等于 cost_estimate.simulate() 的返回值。"""
    _name, r, n, t = scenario
    ref = cm.simulate(r, n, t)
    got_in = sum(c.in_tokens * c.n for c in calls)
    got_out = sum(c.out_tokens * c.n for c in calls)
    got_calls = sum(c.n for c in calls)
    got_searches = sum(c.n for c in calls if c.name == "子agent·ReAct")  # 每轮 ReAct 一次搜索

    checks = [
        ("原始输入 token", got_in, ref["in"]),
        ("输出 token", got_out, ref["out"]),
        ("LLM 调用次数", got_calls, ref["calls"]),
        ("搜索次数", got_searches, ref["searches"]),
    ]
    bad = [(k, a, b) for k, a, b in checks if a != b]
    if bad:
        print("❌ 镜像序列与 cost_estimate.simulate() 对不上，说明两边已漂移：", file=sys.stderr)
        for k, a, b in bad:
            print(f"   {k}: 镜像={a:,}  模型={b:,}", file=sys.stderr)
        raise SystemExit(2)

    return {"in": got_in, "out": got_out, "calls": got_calls, "searches": got_searches}


# ---------------------------------------------------------------------------
# 三、把 token 数变成真文本
# ---------------------------------------------------------------------------
CJK_RANGES = (
    (0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF),
    (0x3000, 0x303F), (0xFF00, 0xFFEF),
)


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return any(lo <= o <= hi for lo, hi in CJK_RANGES)


def estimate_tokens(text: str) -> int:
    """字符级估算，与 cost_estimate.py 的口径一致：中文 1 字 ≈ 1 token，其他 1 token ≈ 3.6 字符。"""
    cjk = sum(1 for ch in text if _is_cjk(ch))
    other = len(text) - cjk
    return int(round(cjk + other / 3.6))


class PromptSynthesizer:
    """把「目标 token 数」变成能喂给 vLLM 的真实文本。

    优先用真 tokenizer（transformers）精确控制长度；拿不到就退化成字符估算，
    并在 stderr 上明确告警——因为估算偏差会直接体现为实测 prompt_len 偏移。
    """

    def __init__(self, corpus: str, tokenizer_path: Optional[str] = None):
        if len(corpus) < 200_000:
            # 单条最长 ~21K token ≈ 21K 字符，语料太短会切不出长窗口
            corpus = (corpus * (200_000 // max(len(corpus), 1) + 1))
        self.corpus = corpus
        self.tokenizer = None
        if tokenizer_path:
            self.tokenizer = self._try_load(tokenizer_path)
        if self.tokenizer is None:
            print(
                "[WARN] 未加载 tokenizer，退化为字符估算（中文 1 字≈1 token）。"
                "实测 prompt_len 可能偏离目标值，建议用 --tokenizer 指向模型目录。",
                file=sys.stderr,
            )

    @staticmethod
    def _try_load(path: str):
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            return tok
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] 加载 tokenizer 失败（{path}）：{exc}", file=sys.stderr)
            return None

    def count(self, text: str) -> int:
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return estimate_tokens(text)

    def _window_chars_for(self, target_tokens: int) -> int:
        """二分出能落到目标 token 数的窗口长度（字符）。"""
        if self.tokenizer is None:
            return max(1, int(target_tokens * 1.05))  # 中文为主，略放一点余量
        lo, hi = 1, len(self.corpus)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.count(self.corpus[:mid]) < target_tokens:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def make(
        self,
        target_tokens: int,
        rng: random.Random,
        prefix_text: str = "",
        prefix_tokens: int = 0,
    ) -> str:
        body_tokens = max(64, target_tokens - prefix_tokens)
        need = self._window_chars_for(body_tokens)
        span = len(self.corpus) - need
        start = rng.randrange(0, max(span, 1))  # 随机起点 → 前缀互不共享
        return prefix_text + self.corpus[start:start + need]


def build_corpus() -> str:
    """从项目自带文档里拼一份中文语料。内容可信，长度可控。"""
    paths = [
        REPO_ROOT / "output_report_2026-08-15_余承东竹知了事件.md",
        REPO_ROOT / "output_report_2026-08-15.md",
        REPO_ROOT / "results" / "output_report_sample_1.md",
        REPO_ROOT / "results" / "output_report_sample_2.md",
    ]
    paths += sorted((REPO_ROOT / "deep_research" / "prompts").glob("*.py"))
    chunks: List[str] = []
    for p in paths:
        if p.exists():
            try:
                chunks.append(p.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    if not chunks:
        raise SystemExit("拼不出语料：找不到任何 md/py 源文件")
    return "\n\n".join(chunks)


# ---------------------------------------------------------------------------
# 四、展开 + 抽样 + 写 jsonl
# ---------------------------------------------------------------------------
@dataclass
class Request:
    role: str
    instance: str
    call_name: str
    target_in: int
    target_out: int
    cache: str
    prompt: str = field(default="", repr=False)

    def as_jsonl(self) -> Dict[str, Any]:
        # vLLM custom dataset 的字段名：prompt + output_tokens
        return {"prompt": self.prompt, "output_tokens": max(1, self.target_out)}


def expand(
    calls: Sequence[Call],
    runs: float,
    rng: random.Random,
    jitter: float,
) -> List[Request]:
    """把一个研究序列按 runs 倍展开成逐条请求；jitter 给长度加随机扰动。"""
    full = int(runs)
    frac = runs - full
    reps: List[float] = [1.0] * full
    if frac > 0:
        reps.append(frac)
    if not reps:
        reps = [1.0]

    out: List[Request] = []
    for rep_scale in reps:
        for c in calls:
            k = int(round(c.n * rep_scale))
            if c.n > 1 and rep_scale >= 1:
                k = max(1, k)
            for _ in range(k):
                j_in = 1.0
                j_out = 1.0
                if jitter > 0:
                    j_in = max(0.3, 1.0 + rng.uniform(-jitter, jitter))
                    j_out = max(0.3, 1.0 + rng.uniform(-jitter, jitter))
                out.append(
                    Request(
                        role=c.role,
                        instance=c.instance,
                        call_name=c.name,
                        target_in=int(c.in_tokens * j_in),
                        target_out=int(c.out_tokens * j_out),
                        cache=c.cache,
                    )
                )
    return out


def apply_quantile_mapping(reqs: List[Request], log_ins: Sequence[int], rng: random.Random) -> None:
    """分位映射：把合成负载的输入长度分布对齐到真实日志的经验分布。

    做法：按合成长度升序排位，把同分位的真实日志长度贴回去。
    保持"长短相对次序"不变（短调用仍短、长调用仍长），同时让分位点与实测一致。
    这是标准的直方图匹配做法，面试里好解释，也不是拍脑袋的缩放系数。
    """
    if not log_ins:
        return
    log_sorted = sorted(log_ins)
    m = len(log_sorted)
    order = sorted(range(len(reqs)), key=lambda i: reqs[i].target_in)
    n = len(order)
    for rank, idx in enumerate(order):
        q = (rank + 0.5) / n
        v = log_sorted[min(m - 1, int(q * m))]
        reqs[idx].target_in = max(64, int(v * (1.0 + rng.uniform(-0.03, 0.03))))


def describe(reqs: Sequence[Request]) -> None:
    """打印负载画像：按角色的簇、按实例的分布、token 总量。"""
    tot_in = sum(r.target_in for r in reqs)
    tot_out = sum(r.target_out for r in reqs)
    print(f"\n{'=' * 88}")
    print(f"负载画像：{len(reqs)} 条请求，输入合计 {tot_in:,} tok，输出合计 {tot_out:,} tok")
    print(f"输入:输出 = {tot_in / max(tot_out, 1):.2f} : 1"
          + ("   ← prefill 密集，压测要看 TTFT 而不是 TPOT" if tot_in > 5 * tot_out else ""))
    print(f"{'=' * 88}")

    by_role: Dict[str, List[Request]] = {}
    for r in reqs:
        by_role.setdefault(r.call_name, []).append(r)
    print(f"{'调用环节':<20}{'条数':>7}{'占比':>8}{'输入均值':>10}{'输入峰值':>10}"
          f"{'输出均值':>10}{'输入token占比':>14}")
    print("-" * 88)
    for name, rs in sorted(by_role.items(), key=lambda kv: -sum(x.target_in for x in kv[1])):
        s_in = sum(x.target_in for x in rs)
        print(f"{name:<20}{len(rs):>7}{len(rs) / len(reqs) * 100:>7.1f}%"
              f"{s_in / len(rs):>10,.0f}{max(x.target_in for x in rs):>10,}"
              f"{sum(x.target_out for x in rs) / len(rs):>10,.0f}{s_in / tot_in * 100:>13.1f}%")
    print("-" * 88)

    by_inst: Dict[str, int] = {}
    for r in reqs:
        by_inst[r.instance] = by_inst.get(r.instance, 0) + 1
    print("按实例拆分：" + "，".join(f"{k}={v} 条" for k, v in sorted(by_inst.items())))

    ins = sorted(r.target_in for r in reqs)
    n = len(ins)

    def pct(p: float) -> int:
        return ins[min(n - 1, int(n * p))]

    print(f"输入长度分位：P50={pct(.5):,}  P90={pct(.9):,}  P99={pct(.99):,}  max={ins[-1]:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="引擎层压测负载生成器")
    ap.add_argument("--scenario", default="基准", help="轻量/基准/重型（对齐 cost_estimate.SCENARIOS）")
    ap.add_argument("--runs", type=float, default=1.0, help="展开成几条研究的调用序列")
    ap.add_argument("--max-requests", type=int, default=0, help="请求数上限，0=不截断")
    ap.add_argument("--jitter", type=float, default=0.05, help="长度抖动比例，默认 ±5%%")
    ap.add_argument("--mode", choices=["isolated", "shared"], default="isolated",
                    help="isolated=前缀不共享（测裸推理）；shared=带角色固定前缀（测 agent 真实收益）")
    ap.add_argument("--shared-prefix-tokens", type=int, default=-1,
                    help="shared 模式下每条请求的共享前缀长度；-1=按角色自动取 system prompt 长度")
    ap.add_argument("--role-filter", default=None, help="只要某个 role 的请求，逗号分隔")
    ap.add_argument("--instance-filter", default=None, help="只要某个实例的请求（qwen_general/qwen_redteam）")
    ap.add_argument("--tokenizer", default=None, help="tokenizer 路径；不给则退化为字符估算")
    ap.add_argument("--no-merge", action="store_true",
                    help="不合并成本模型拆出的伴随条目（默认合并，还原真实单次请求形状）")
    ap.add_argument("--length-source", choices=["model", "log"], default="model",
                    help="输入长度取自成本模型还是真实日志的经验分布（分位映射）")
    ap.add_argument("--log", default=str(REPO_ROOT / "run_legal_audit.log"),
                    help="真实运行日志路径，用于 --length-source log 与分布对比")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dry-run", action="store_true", help="只对账 + 打印画像，不写文件")
    args = ap.parse_args()

    cm = load_cost_model()
    scen = next((s for s in cm.SCENARIOS if s[0] == args.scenario), None)
    if scen is None:
        raise SystemExit(f"未知场景 {args.scenario}，可选：{[s[0] for s in cm.SCENARIOS]}")

    name, r, nr, t = scen
    calls = iter_calls(cm, sup_rounds=r, researchers_per_round=nr, react_turns=t)
    agg = reconcile(cm, calls, scen)
    print(f"[OK] 与 cost_estimate.simulate() 对账通过：{name}档  "
          f"输入 {agg['in']:,} / 输出 {agg['out']:,} / 调用 {agg['calls']} / 搜索 {agg['searches']}")

    if not args.no_merge:
        calls, notes = merge_companions(calls)
        for note in notes:
            print(f"[合并] {note}")

    rng = random.Random(args.seed)
    reqs = expand(calls, args.runs, rng, args.jitter)

    if args.role_filter:
        want = {x.strip() for x in args.role_filter.split(",") if x.strip()}
        reqs = [x for x in reqs if x.role in want or x.call_name in want]
    if args.instance_filter:
        want_i = {x.strip() for x in args.instance_filter.split(",") if x.strip()}
        reqs = [x for x in reqs if x.instance in want_i]

    if args.max_requests and len(reqs) > args.max_requests:
        rng.shuffle(reqs)
        reqs = reqs[: args.max_requests]

    if not reqs:
        raise SystemExit("筛选后没有剩余请求，检查 --role-filter / --instance-filter")

    # --- 用真实运行日志校准长度分布 ---
    log_ins, _log_outs = parse_log_lengths(Path(args.log))
    if log_ins:
        compare_with_log([x.target_in for x in reqs], log_ins)
        if args.length_source == "log":
            apply_quantile_mapping(reqs, log_ins, rng)
            print("[校准] 已按真实日志经验分布做分位映射（角色构成不变，只改长度分位）")

    # 合成 prompt 文本
    corpus = build_corpus()
    synth = PromptSynthesizer(corpus, tokenizer_path=args.tokenizer)
    shared_prefix_tokens = args.shared_prefix_tokens
    prefix_cache = {}
    for role in {x.role for x in reqs}:
        key = SHARED_PREFIX_OF_ROLE.get(role)
        prefix_cache[role] = int(cm.PROMPT_TOKENS.get(key, 0)) if key else 0

    pref_text_cache: Dict[int, str] = {}
    for req in reqs:
        ptok = 0
        ptext = ""
        if args.mode == "shared":
            ptok = shared_prefix_tokens if shared_prefix_tokens >= 0 else prefix_cache.get(req.role, 0)
            ptok = min(ptok, max(0, req.target_in - 256))
            if ptok > 0:
                if ptok not in pref_text_cache:
                    pref_text_cache[ptok] = synth.make(ptok, random.Random(0))
                ptext = pref_text_cache[ptok]
        req.prompt = synth.make(req.target_in, rng, prefix_text=ptext, prefix_tokens=ptok)

    describe(reqs)

    if args.mode == "shared":
        tot_prefix = sum(min(prefix_cache.get(x.role, 0), x.target_in) for x in reqs)
        tot_in = sum(x.target_in for x in reqs)
        print(f"共享前缀模式：理论上界命中率 ≈ {tot_prefix / max(tot_in, 1) * 100:.1f}%"
              f"（实测值从 /metrics 的 prefix cache 指标读，必须一并写进报告）")

    if args.dry_run:
        print("\n[dry-run] 未写文件。")
        return

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for req in reqs:
            f.write(json.dumps(req.as_jsonl(), ensure_ascii=False) + "\n")

    meta = {
        "scenario": name,
        "runs": args.runs,
        "mode": args.mode,
        "jitter": args.jitter,
        "seed": args.seed,
        "role_filter": args.role_filter,
        "instance_filter": args.instance_filter,
        "requests": len(reqs),
        "total_input_tokens": sum(x.target_in for x in reqs),
        "total_output_tokens": sum(x.target_out for x in reqs),
        "by_instance": {k: sum(1 for x in reqs if x.instance == k)
                        for k in {x.instance for x in reqs}},
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"\n✅ 已写出 {out}（{len(reqs)} 条，{size_mb:.1f} MB）")
    print(f"   元信息：{out.with_suffix('.meta.json')}")
    print(f"   下一步：python bench/engine_bench.py --workload {out} --dry-run")


if __name__ == "__main__":
    main()
