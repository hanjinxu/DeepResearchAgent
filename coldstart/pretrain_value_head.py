#***********************************************
#      Filename: pretrain_value_head.py
#   Description: Critic value head 冷启动预训练
#                加载 Qwen3-7B（或 SFT checkpoint），冻结 backbone，
#                只训 value head（Linear -> 1），用 rollout 数据做 MSE 回归
#
#   用法（AutoDL 上）：
#     source /root/autodl-tmp/envs/research/bin/activate
#     cd /root/autodl-tmp/project-Deep_Research
#
#     # 用 base 模型（如果没有 SFT checkpoint）
#     python coldstart/pretrain_value_head.py
#
#     # 用 SFT checkpoint（推荐）
#     COLDSTART_MODEL_PATH=/root/autodl-tmp/models/sft_checkpoint \
#       python coldstart/pretrain_value_head.py
#***********************************************

import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ===== 配置 =====
# 模型路径：优先读环境变量，默认用 base Qwen3-7B
MODEL_PATH = os.environ.get(
    "COLDSTART_MODEL_PATH",
    "/root/autodl-tmp/models/Qwen3-7B",
)

DATA_FILE = os.path.join(os.path.dirname(__file__), "rollout_data.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "critic_coldstart")

EPOCHS = 5
LR = 1e-4
MAX_LEN = 4096
LOG_EVERY = 5


def main():
    # 1. 加载数据
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: 数据文件不存在: {DATA_FILE}")
        print("请先运行: python coldstart/collect_rollout_data.py")
        sys.exit(1)

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"加载训练数据: {len(data)} 条")
    print(f"  return 范围: [{min(d['return'] for d in data):.4f}, {max(d['return'] for d in data):.4f}]")

    # 2. 加载模型
    print(f"\n加载模型: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. 替换 LM head 为 value head
    hidden_size = model.config.hidden_size
    value_head = torch.nn.Linear(hidden_size, 1, bias=False).to(model.dtype).to(model.device)
    model.lm_head = value_head
    print(f"  hidden_size={hidden_size}, value_head 参数数={sum(p.numel() for p in value_head.parameters())}")

    # 4. 冻结 backbone，只训 value head
    for p in model.parameters():
        p.requires_grad = False
    for p in value_head.parameters():
        p.requires_grad = True

    trainable = sum(p.numel() for p in value_head.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable} / {total} ({trainable/total*100:.4f}%)")

    optimizer = torch.optim.AdamW(value_head.parameters(), lr=LR)

    # 5. 训练
    print(f"\n开始训练: {EPOCHS} epochs, lr={LR}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        t0 = time.time()

        for i, item in enumerate(data):
            prompt = item["prompt"]
            target_val = item["return"]

            target = torch.tensor([target_val], dtype=torch.float32).to(model.device)

            # tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_LEN,
            ).to(model.device)

            # backbone forward（冻结，不反传）
            with torch.no_grad():
                hidden = model.model(**inputs).last_hidden_state

            # value head forward
            v_pred = value_head(hidden[:, -1, :]).squeeze(-1).float()

            # MSE loss
            loss = torch.nn.functional.mse_loss(v_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (i + 1) % LOG_EVERY == 0:
                print(
                    f"  Epoch {epoch+1}/{EPOCHS} [{i+1}/{len(data)}] "
                    f"loss={loss.item():.4f} v={v_pred.item():.4f} target={target_val:.4f}"
                )

        avg_loss = total_loss / len(data)
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1} done: avg_loss={avg_loss:.4f} ({elapsed:.0f}s)")
        print("-" * 60)

    # 6. 保存 checkpoint
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"\n{'=' * 60}")
    print(f"Critic 冷启动 checkpoint 已保存:")
    print(f"  {OUTPUT_DIR}")
    print(f"\n接入 verl PPO 时设置:")
    print(f"  critic.model.path={OUTPUT_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
