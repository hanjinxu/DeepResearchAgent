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
    ap.add_argument("--teacher-role", default="evaluator",
                    help="给 20 条内容项出参考答案的教师角色（必须与 policy 不同源）")
    ap.add_argument("--no-teacher", action="store_true",
                    help="调试用：无教师时奖励恒为 0，只验证链路能跑")
    ap.add_argument("--out", default="data/flywheel/grpo_run")
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：极小配置，只验证能跑")
    args = ap.parse_args()

    if args.smoke:
        args.steps = min(args.steps, 2)
        args.k = 2
        # 20 条判定至少要 300~500 token 才写得完，120 会截断成"解析不到 5 条"
        args.max_new_tokens = 700
        # 也不能压太狠：内容项要读完整节才能判，2000 字只剩个开头
        args.max_prompt_chars = 6000
        args.rollout_chunk = 1
        args.log_every = 1
        print(f">>> 冒烟模式：steps={args.steps}, k={args.k}, "
              f"max_new_tokens={args.max_new_tokens}, "
              f"prompt≤{args.max_prompt_chars} 字")

    if not args.model:
        print("必须给 --model（LoRA 训练需要本地权重）", file=sys.stderr)
        return 2

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from deep_research.flywheel import normalize_advantages
    from deep_research.prompts import RED_TEAM_CONTENT_PROMPT

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

    # ---------- 教师模型（给 20 条内容项出参考答案）----------
    # 必须与 policy 不同源：自己判自己，奖励没有意义。


    # ---------- 工具 ----------
    def to_chat(prompt: str) -> str:
        """把裸 prompt 包成 Qwen 的 chat 格式。

        vLLM 服务时走 /v1/chat/completions，会自动套 chat template；
        而这里的 rollout 和 logprob 计算如果直接喂裸文本，
        模型看到的输入分布和推理时不一致——实测后果是它不按指令输出，
        解析不到 5 条判定，整批样本被判定为不可训。
        """
        msgs = [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # 老版本 tokenizer 不认 enable_thinking
            return tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    def build_prompt(draft: str) -> str:
        """必须与 sample_critiques 用的 prompt 完全一致。

        踩过的坑：这里原本还用旧的 RED_TEAM_PROMPT（让它自由找问题），
        而采样脚本已经切到 20 条内容检查。模型收到"找问题"的指令后
        直接输出 PASS，解析出 0 条判定 → 全部不可训。
        报错信息只写"可训样本 0<2"，完全看不出是 prompt 不一致。
        """
        return RED_TEAM_CONTENT_PROMPT.format(
            draft_report=_truncate_middle(draft, args.max_prompt_chars),
            checklist=build_content_checklist(),
        )

    @torch.no_grad()
    def rollout(prompt: str) -> List[str]:
        enc = tok(to_chat(prompt), return_tensors="pt", truncation=True,
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

    # ---------- 教师标注（按草稿缓存）----------
    # 20 条内容项没有规则真值，只能让更强的模型当老师。
    # 关键在于**缓存**：草稿是固定的，几百步训练会反复用同一份草稿，
    # 每步都调一次 API 既慢又费钱。按草稿指纹缓存后，2 份草稿只调 2 次。
    from deep_research.flywheel.parser import parse_checklist
    from deep_research.flywheel.rubric import build_content_checklist
    teacher_path = out_dir / "teacher_labels.json"
    teacher_cache: Dict[str, Dict[str, bool]] = {}
    if teacher_path.exists():
        teacher_cache = json.loads(teacher_path.read_text(encoding="utf-8"))
        print(f">>> 复用教师标注缓存：{len(teacher_cache)} 份草稿")

    def get_chat_model_teacher():
        from deep_research.llm import get_chat_model

        return get_chat_model(args.teacher_role)

    teacher_model = None if args.no_teacher else get_chat_model_teacher()

    def teacher_labels(draft: str, fp: str) -> Dict[str, bool]:
        if fp in teacher_cache:
            return {k: bool(v) for k, v in teacher_cache[fp].items()}
        if teacher_model is None:
            # --no-teacher：不调 API，奖励恒为 0，只用于验证链路能跑通
            return {}
        prompt = RED_TEAM_CONTENT_PROMPT.format(
            draft_report=draft, checklist=build_content_checklist()
        )
        from langchain_core.messages import HumanMessage

        raw = teacher_model.invoke([HumanMessage(content=prompt)]).content
        labels = {v.check_id: v.passed for v in parse_checklist(raw) if v.passed is not None}
        teacher_cache[fp] = {k: bool(v) for k, v in labels.items()}
        teacher_path.write_text(json.dumps(teacher_cache, ensure_ascii=False, indent=2),
                                encoding="utf-8")
        print(f">>> 教师标注 {fp}: {len(labels)} 条可判")
        return labels

    MIN_TEACHER_ITEMS = 8

    def draft_is_usable(fp: str, draft: str) -> Tuple[bool, int]:
        """教师都没法判的草稿不能用来训练。

        实测：一份「AI Agent Memory 调研报告」不是法务骨架，20 条内容项
        只有 1 条可判。这类草稿参与训练只会让模型学到"随便判几条"。
        """
        try:
            n = len(teacher_labels(draft, fp))
        except Exception as exc:  # noqa: BLE001
            print(f"    [WARN] 教师标注失败 {fp}: {str(exc)[:80]}")
            return False, 0
        return n >= MIN_TEACHER_ITEMS, n

    def reward_of(draft: str, fp: str, texts: Sequence[str]) -> Tuple[List[float], List[dict]]:
        """奖励 = 分类别准确率的平均（balanced accuracy），再按覆盖率打折。

        为什么不用普通一致率：
            实测这份报告教师判 18 通过 / 2 不通过。模型只要全判"通过"
            就能拿 18/20 = 0.90，比它认真判还高——直接训出"全判通过"。
        为什么分类别：
            通过类和不过类各算一个准确率再平均，两类的权重一样。
            全判通过时"不过类"准确率为 0，总分封顶 0.5，占不到便宜。
        为什么乘覆盖率：
            模型只判 8 条、漏 12 条时，它能在判过的条目上全对。
            按覆盖率（judged/teacher）线性打折，漏判越多得分越低。
        """
        labels = teacher_labels(draft, fp)
        pos = [k for k, v in labels.items() if v]
        neg = [k for k, v in labels.items() if not v]

        rewards, details = [], []
        for t in texts:
            judged = {v.check_id: v.passed for v in parse_checklist(t) if v.passed is not None}

            acc_pos = (sum(1 for k in pos if judged.get(k) is True) / len(pos)) if pos else None
            acc_neg = (sum(1 for k in neg if judged.get(k) is False) / len(neg)) if neg else None
            parts = [a for a in (acc_pos, acc_neg) if a is not None]
            base = sum(parts) / len(parts) if parts else 0.0

            coverage = len(judged) / max(len(labels), 1)
            # 覆盖不到 80% 开始线性打折；判满则不打折
            factor = min(1.0, coverage / 0.8)
            rewards.append(round(base * factor, 4))

            details.append({
                "trainable": len(judged) >= 5,
                "n_items": len(judged), "coverage": round(coverage, 3),
                "acc_pos": None if acc_pos is None else round(acc_pos, 3),
                "acc_neg": None if acc_neg is None else round(acc_neg, 3),
                "base": round(base, 3),
            })
        return rewards, details

    def seq_logp(text_prompt: str, completion: str, ref: bool = False) -> Tuple[torch.Tensor, int]:
        """返回 (completion 的逐 token logprob 均值, completion token 数)。

        显存关键：只算 completion 位置的 logits。
            整条序列 [1, 12k, 151936] 的 bf16 logits 是 3.6G，
            `.float()` 再翻一倍到 7.3G —— 这是 40G 卡上 OOM 的直接原因。
            logits_to_keep=n_comp 让 lm_head 只算最后 n 个位置，
            显存降到原来的 1/17（700 vs 12000 个位置）。
        数值关键：用 F.cross_entropy 而不是手工 log_softmax + gather。
            cross_entropy 是融合实现，不会把整个 [n, V] 的 float32 softmax 落盘。
        """
        import contextlib

        p_ids = tok(to_chat(text_prompt), return_tensors="pt", truncation=True,
                    max_length=8192).input_ids
        c_ids = tok(completion, return_tensors="pt", add_special_tokens=False).input_ids
        n_comp = int(c_ids.shape[1])
        if n_comp == 0:
            return torch.zeros((), device=model.device), 0

        ids = torch.cat([p_ids, c_ids], dim=1).to(model.device)
        labels = ids[:, -n_comp:]          # 最后 n 个 token 就是 completion
        ctx = model.disable_adapter() if ref else contextlib.nullcontext()

        with ctx:
            try:
                logits = model(input_ids=ids, logits_to_keep=n_comp).logits
            except TypeError:
                # 老版本 transformers 不认 logits_to_keep，退回全量再切片
                logits = model(input_ids=ids).logits[:, -n_comp:, :]

            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                labels.reshape(-1),
                reduction="none",
            )
            lp_comp = -nll
            del logits

        return lp_comp.mean(), n_comp

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

        ok, n_label = draft_is_usable(fp, draft)
        if not ok:
            skipped += 1
            rec = {"step": step, "skipped": "draft_unusable",
                   "draft": fp, "teacher_items": n_label}
            with log_path.open("a", encoding="utf-8") as fd:
                fd.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step % args.log_every == 0:
                print(f"[{step:>3}] 跳过（该草稿教师只判出 {n_label} 条，"
                      f"低于 {MIN_TEACHER_ITEMS} 条门槛）")
            continue

        model.eval()   # 采样时关掉 LoRA dropout，否则同一条 prompt 的差异里混进 dropout 噪声
        texts = rollout(prompt)
        # 生成阶段的 KV cache 很占显存，先还给 allocator，
        # 否则紧接着算 logprob 时容易"明明该够却 OOM"
        torch.cuda.empty_cache()
        if args.smoke:
            print(f"--- 第一条生成（{len(texts[0])} 字）---")
            print(texts[0][:400].replace("\n", " ⏎ "))
            print("---")
        rewards, details = reward_of(draft, fp, texts)
        model.train()

        # ⚠️ 只用可训样本算优势，必须排除 trainable=False 的（PASS / 全被过滤）。
        # 否则：PASS 得 0 分，提 3 条错批评得 -2 分 → 优势指向"什么都不说"，
        # 模型会精确地学会沉默。这是整套奖励里最隐蔽的 reward hacking 入口：
        # 它不报错、不像 bug，只会让报告越来越"干净"而信息量越来越低。
        keep = [(t, r, d) for t, r, d in zip(texts, rewards, details) if d["trainable"]]
        n_drop = len(texts) - len(keep)

        if len(keep) < 2:
            skipped += 1
            rec = {"step": step, "skipped": "too_few_trainable",
                   "n_kept": len(keep), "n_drop": n_drop,
                   "rewards": [round(r, 4) for r in rewards]}
            with log_path.open("a", encoding="utf-8") as fd:
                fd.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step % args.log_every == 0:
                print(f"[{step:>3}] 跳过（可训样本 {len(keep)}<2，"
                      f"丢弃 {n_drop} 条不可训）")
            continue

        kept_texts = [t for t, _, _ in keep]
        kept_rewards = [r for _, r, _ in keep]
        kept_details = [d for _, _, d in keep]
        mean_r = sum(kept_rewards) / len(kept_rewards)
        std_r = (sum((r - mean_r) ** 2 for r in kept_rewards) / len(kept_rewards)) ** 0.5
        advs = normalize_advantages(kept_rewards)

        if std_r < 1e-6 or all(abs(a) < 1e-9 for a in advs):
            skipped += 1
            rec = {"step": step, "skipped": "zero_variance",
                   "reward_mean": round(mean_r, 4), "reward_std": round(std_r, 4),
                   "n_drop": n_drop, "rewards": [round(r, 4) for r in kept_rewards]}
            with log_path.open("a", encoding="utf-8") as fd:
                fd.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if step % args.log_every == 0:
                print(f"[{step:>3}] 跳过（组内无方差）reward={mean_r:.3f} "
                      f"rewards={[round(r,2) for r in kept_rewards]}")
            continue

        # 逐条算 logprob（不批量：序列长度差异大，桶排序收益不值得这层复杂度）
        loss_terms = []
        kl_terms = []
        comp_lens = []
        for text, adv in zip(kept_texts, advs):
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
            "step": step, "loss": round(float(loss.detach()), 5),
            "policy_loss": round(float(policy_loss.detach()), 5),
            "kl": round(float(kl.detach()), 5),
            "reward_mean": round(mean_r, 4), "reward_std": round(std_r, 4),
            "adv_abs_mean": round(adv_abs, 4),
            "rewards": [round(r, 4) for r in kept_rewards],
            "comp_len": round(avg_len, 1),
            "n_kept": len(kept_texts), "n_drop": n_drop,
            "elapsed_s": round(time.time() - t0, 1),
        }
        with log_path.open("a", encoding="utf-8") as fd:
            fd.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if step % args.log_every == 0:
            print(f"[{step:>3}] loss={float(loss.detach()):+.4f} "
                  f"kl={float(kl.detach()):+.4f} r={mean_r:+.3f}±{std_r:.3f} "
                  f"通过类={rec['acc_pos']:.2f} 不过类={rec['acc_neg']:.2f} "
                  f"len={avg_len:.0f} kept={len(kept_texts)}/{len(texts)}")

        if step % 20 == 0:
            ckpt = out_dir / f"lora_step{step}"
            model.save_pretrained(ckpt)
            print(f"    ↳ 已存 {ckpt}")

    if accum > 0:
        # 不足一个 grad_accum 的残留梯度直接提交，否则最后几步白算
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        print(f">>> 已提交最后 {accum} 步残留梯度")

    model.save_pretrained(out_dir / "lora_final")
    print(f"\n✅ 训练结束：{args.steps} 步（跳过 {skipped} 步）｜日志 {log_path}")
    print(f"   LoRA 已存 {out_dir / 'lora_final'}")
    print("\n下一步看曲线：")
    print(f"   python scripts/plot_train_log.py --log {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
