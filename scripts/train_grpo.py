#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""自包含的 GRPO 训练脚本（红队专用，单轮生成任务）。

为什么不用 verl：
verl 面向"多轮 rollout + 分布式 actor/critic"的通用场景，API 随版本变动大。
而这里的任务是**单轮生成**：给定草稿 → 采 k 条批评 → 打分 → 更新。
用 verl 要为它适配 env、reward manager、数据集格式，调试成本远超收益。
这个脚本把所有环节摊开写，出问题能一眼定位，也方便面试时逐行讲。

⚠️ 与 verl 的等价性：
本脚本不做分布式、不做 colocate、不做权重同步到 vLLM。
rollout 与训练用同一个 HF 模型（就地采样），所以在 32G 卡上 8B + LoRA 是可行的，
代价是慢（attention 没走 PagedAttention）。

流程（每一步都能在日志里看到）：
    for step:
        prompt = 草稿 → 红队提示词
        k 条输出 = model.generate(temperature=0.9)
        reward   = parse → filter → verify → score
        adv      = (r - mean) / std          # 这是 GRPO 的核心
        loss     = -mean(adv × logp_pi) + β · KL(pi ‖ ref)
        backward + step

用法：
    # 冒烟测试（10 分钟内确认能跑，不追求效果）
    python scripts/train_grpo.py --smoke

    # 正式跑
    python scripts/train_grpo.py \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --raw data/flywheel/raw.jsonl \
        --steps 200 --k 4 --lr 1e-5 \
        --out data/flywheel/grpo_run1

判读指标（日志里每步都有）：
    reward_mean / reward_std  组内奖励的均值和离散度——std 长期接近 0 说明学不到东西
    adv_abs_mean              优势的绝对值均值，太小说明组内没区分度
    kl                        与参考策略的偏离，涨太快说明在追噪声
    comp_len                  输出长度，异常变短要警惕模型学会"少说话"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_drafts(raw_path: Path, limit: int = 0) -> Dict[str, str]:
    """从 raw.jsonl 里取出 (draft_fp → draft_path)，去重后读正文。"""
    seen: Dict[str, str] = {}
    with raw_path.open(encoding="utf-8") as fd:
        for line in fd:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            fp, p = row.get("draft_fp", ""), row.get("draft_path", "")
            if fp and fp not in seen:
                seen[fp] = p
    items = list(seen.items())
    random.shuffle(items)
    if limit:
        items = items[:limit]
    out: Dict[str, str] = {}
    for fp, p in items:
        try:
            out[fp] = Path(p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
    return out


def _truncate_middle(text: str, max_chars: int) -> str:
    """头尾保留、中间截断。

    草稿的关键信息在两端：开头是定性与时间线，结尾是风险等级与处置建议。
    直接截前 N 字会把"风险等级"整段丢掉，而那恰好是红队要挑错的地方。
    """
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n……（中段省略）……\n\n" + text[-half:]


def main() -> int:
    ap = argparse.ArgumentParser(description="自包含 GRPO 训练（红队）")
    ap.add_argument("--model", default="", help="基座模型路径；--smoke 时可留空")
    ap.add_argument("--raw", default="data/flywheel/raw.jsonl", help="用于取草稿清单")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--k", type=int, default=4, help="每组采样条数（GRPO 建议 ≥4）")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--kl-coef", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=800)
    ap.add_argument("--max-prompt-chars", type=int, default=12000,
                    help="草稿字符上限（头尾保留）；控制显存的主要旋钮")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--rollout-chunk", type=int, default=2,
                    help="一次 generate 多少条，OOM 就调小")
    ap.add_argument("--no-judge", action="store_true", help="只用规则核实（省 token）")
    ap.add_argument("--out", default="data/flywheel/grpo_run")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：极小配置，只验证能跑")
    args = ap.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 2)
        args.k = 2
        args.max_new_tokens = 120
        args.max_prompt_chars = 2000
        args.rollout_chunk = 1
        args.log_every = 1
        print(">>> 冒烟模式：steps=2, k=2, max_new_tokens=120, prompt≤2000 字")

    if not args.model:
        print("必须给 --model（LoRA 训练需要本地权重）", file=sys.stderr)
        return 2

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from deep_research.flywheel import (
        filter_critiques, normalize_advantages, parse_critiques, score_sample,
    )
    from deep_research.flywheel.verifier import CombinedVerifier, JudgeVerifier
    from deep_research.prompts import RED_TEAM_PROMPT

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    # ---------- 数据 ----------
    drafts = _load_drafts(Path(args.raw))
    if not drafts:
        print(f"从 {args.raw} 没读到草稿；先跑 scripts/sample_critiques.py", file=sys.stderr)
        return 2
    print(f"草稿池：{len(drafts)} 份")

    # ---------- 模型 ----------
    print(f">>> 加载 {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda:0", trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    ))
    model.print_trainable_parameters()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    # V100（Volta, SM70）没有 bf16，只能退到 fp16；fp16 反向传播会梯度下溢，
    # 必须配 GradScaler，否则几十步后 loss 变 NaN，看起来像"训崩了"其实只是没缩放。
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    if dtype == torch.float16:
        print(">>> 检测到不支持 bf16（V100 等 Volta 卡），已启用 fp16 + GradScaler")

    # ---------- 核实器 ----------
    judge = None if args.no_judge else JudgeVerifier()

    # ---------- 工具 ----------
    def build_prompt(draft: str) -> str:
        brief_hint = "；".join(
            ln.lstrip("#").strip() for ln in draft.splitlines()[:5] if ln.strip()
        )[:300]
        return RED_TEAM_PROMPT.format(
            research_brief=brief_hint or "（无简报）",
            draft_report=_truncate_middle(draft, args.max_prompt_chars),
        )

    @torch.no_grad()
    def rollout(prompt: str) -> List[str]:
        enc = tok(prompt, return_tensors="pt", truncation=True,
                  max_length=8192).to(model.device)
        outs: List[str] = []
        for _ in range(0, args.k, args.rollout_chunk):
            n = min(args.rollout_chunk, args.k - len(outs))
            gen = model.generate(
                **enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                num_return_sequences=n, max_new_tokens=args.max_new_tokens,
                pad_token_id=tok.pad_token_id, repetition_penalty=1.05,
            )
            for row in gen:
                outs.append(tok.decode(row[enc["input_ids"].shape[1]:], skip_special_tokens=True))
        return outs

    def reward_of(draft: str, texts: Sequence[str]) -> Tuple[List[float], List[dict]]:
        verifier = CombinedVerifier(draft, judge=judge)
        rewards, details = [], []
        for t in texts:
            crits = parse_critiques(t)
            fr = filter_critiques(crits, draft)
            vers = verifier.verify_all(fr.usable)
            s = score_sample(prompt="", critiques=fr.usable, verifications=vers,
                             shortcut_hits=fr.dropped, filter_reason=fr.reason,
                             trainable=fr.trainable)
            rewards.append(s.total)
            details.append({
                "trainable": s.trainable, "n_usable": len(fr.usable),
                "n_valid": s.breakdown.get("n_valid", 0),
                "n_invalid": s.breakdown.get("n_invalid", 0),
            })
        return rewards, details

    def seq_logp(text_prompt: str, completion: str, ref: bool = False) -> Tuple[torch.Tensor, int]:
        """返回 (completion 的逐 token logprob 均值, completion token 数)。

        ref=True 时禁用 LoRA adapter，得到参考策略的 logprob——
        KL 项就是 policy 与它的逐 token 差。这是 GRPO 里"不跑偏"的唯一约束。
        """
        import contextlib

        p_ids = tok(text_prompt, return_tensors="pt", truncation=True,
                    max_length=8192).input_ids
        c_ids = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids
        if c_ids.shape[1] == 0:
            return torch.zeros((), device=model.device), 0
        ids = torch.cat([p_ids, c_ids], dim=1).to(model.device)
        ctx = model.disable_adapter() if ref else contextlib.nullcontext()
        with ctx:
            logits = model(input_ids=ids).logits[:, :-1]
            labels = ids[:, 1:]
            lp = F.log_softmax(logits.float(), dim=-1).gather(
                -1, labels.unsqueeze(-1)).squeeze(-1)
        start = p_ids.shape[1] - 1
        lp_comp = lp[0, start:]
        return lp_comp.mean(), int(lp_comp.shape[0])

    # ---------- 训练 ----------
    print(f">>> 开始训练：{args.steps} 步｜k={args.k}｜lr={args.lr}｜kl={args.kl_coef}"
          f"｜grad_accum={args.grad_accum}")
    model.train()
    keys = list(drafts.items())
    skipped = 0
    accum = 0
    t0 = time.time()

    for step in range(1, args.steps + 1):
        fp, draft = keys[(step - 1) % len(keys)]
        prompt = build_prompt(draft)

        model.eval()   # 采样时关掉 LoRA dropout，否则同一条 prompt 的差异里混进 dropout 噪声
        texts = rollout(prompt)
        rewards, details = reward_of(draft, texts)
        model.train()

        mean_r = sum(rewards) / len(rewards)
        std_r = (sum((r - mean_r) ** 2 for r in rewards) / len(rewards)) ** 0.5
        advs = normalize_advantages(rewards)

        if std_r < 1e-6 or all(abs(a) < 1e-9 for a in advs):
            skipped += 1
            rec = {"step": step, "skipped": "zero_variance",
                   "reward_mean": round(mean_r, 4), "reward_std": round(std_r, 4),
                   "rewards": [round(r, 4) for r in rewards]}
            with log_path.open("a", encoding="utf-8") as fd:
                fd.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step % args.log_every == 0:
                print(f"[{step:>3}] 跳过（组内无方差）reward={mean_r:.3f} "
                      f"rewards={[round(r,2) for r in rewards]}")
            continue

        # 逐条算 logprob（不批量：序列长度差异大，桶排序收益不值得这层复杂度）
        loss_terms = []
        kl_terms = []
        comp_lens = []
        for text, adv in zip(texts, advs):
            lp_pi, n = seq_logp(prompt, text, ref=False)
            if n == 0:
                continue
            with torch.no_grad():
                lp_ref, _ = seq_logp(prompt, text, ref=True)
            loss_terms.append(-adv * lp_pi)
            kl_terms.append(lp_pi - lp_ref)
            comp_lens.append(n)

        if not loss_terms:
            skipped += 1
            continue

        policy_loss = torch.stack(loss_terms).mean()
        kl = torch.stack(kl_terms).mean()
        loss = policy_loss + args.kl_coef * kl

        # 真正的梯度累积：每 grad_accum 步才更新一次
        scaler.scale(loss / args.grad_accum).backward()
        accum += 1
        if accum >= args.grad_accum:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            accum = 0

        avg_len = sum(comp_lens) / len(comp_lens)
        adv_abs = sum(abs(a) for a in advs) / len(advs)
        rec = {
            "step": step, "loss": round(float(loss), 5),
            "policy_loss": round(float(policy_loss), 5),
            "kl": round(float(kl), 5),
            "reward_mean": round(mean_r, 4), "reward_std": round(std_r, 4),
            "adv_abs_mean": round(adv_abs, 4),
            "rewards": [round(r, 4) for r in rewards],
            "comp_len": round(avg_len, 1),
            "n_usable": [d["n_usable"] for d in details],
            "elapsed_s": round(time.time() - t0, 1),
        }
        with log_path.open("a", encoding="utf-8") as fd:
            fd.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if step % args.log_every == 0:
            print(f"[{step:>3}] loss={float(loss):+.4f} kl={float(kl):+.4f} "
                  f"r={mean_r:+.3f}±{std_r:.3f} adv={adv_abs:.3f} len={avg_len:.0f}"
                  f" usable={rec['n_usable']}")

        if step % 20 == 0:
            ckpt = out_dir / f"lora_step{step}"
            model.save_pretrained(ckpt)
            print(f"    ↳ 已存 {ckpt}")

    model.save_pretrained(out_dir / "lora_final")
    print(f"\n✅ 训练结束：{args.steps} 步（跳过 {skipped} 步）｜日志 {log_path}")
    print(f"   LoRA 已存 {out_dir / 'lora_final'}")
    print("\n下一步看曲线：")
    print(f"   python scripts/plot_train_log.py --log {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
