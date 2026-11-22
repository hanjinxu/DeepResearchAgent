# AutoDL 运行手册

本文只讲**怎么在 AutoDL 上把这套跑起来**。设计取舍见 `docs/` 下其它文档与代码注释。

---

## 0. 先确认你在哪一步

| 步骤 | 需要 GPU | 产出 | 判据 |
|---|---|---|---|
| 1. 测写作执行率 | 需要写作模型 | `exec_report.json` | 执行率 ≥ 50% 才继续 |
| 2. 采样红队 | 需要红队模型 | `raw.jsonl` | 解析率、多样性 |
| 3. 体检 + 导出 | 不需要（CPU 即可） | `samples.json` | 核实覆盖率、组内 std |
| 4. GRPO 训练 | 需要 | `train_log.jsonl` + `curves.png` | reward 上升、KL 不失控 |
| 5. 看曲线 | 不需要 | 终端摘要 / PNG | 见第 6 节判读规则 |

第 3 步只读文件、不调模型（除非开 judge），可以在本机跑。

训练用的是仓库自带的 `scripts/train_grpo.py`（单轮 GRPO，不依赖 verl）。
为什么不用 verl：本任务是单轮生成，verl 的 env / reward manager / 数据集适配
调试成本远超收益，而且版本耦合重。verl 仍然适合多轮、分布式场景。

---

## 1. 环境

镜像选 **PyTorch 2.8.0 / Python 3.12 / CUDA 12.8**，卡用 2×32G。

```bash
# 代码
cd /root/autodl-tmp
git clone <你的仓库地址> deepresearch && cd deepresearch

# ⚠️ config.yml 不在仓库里（含密钥，已停止跟踪）
# 从本地 scp 一份上来，或复制模板后填 key
cp config.example.yml config.yml

# 依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install vllm modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2. 下模型

```bash
modelscope download --model Qwen/Qwen3-8B \
  --local_dir /root/autodl-tmp/models/Qwen3-8B
```

放 `autodl-tmp` 而不是 `root`：系统盘通常只有 30~50G，模型和数据会撑爆。

---

## 3. 起两个 vLLM 实例

一张卡一个。8B 的 FP16 权重约 16.4G，32K 上下文的 KV 约 9G，单卡放得下。

```bash
# 卡0：写作（通用）
CUDA_VISIBLE_DEVICES=0 nohup vllm serve /root/autodl-tmp/models/Qwen3-8B \
  --served-model-name qwen3-8b \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  > /root/autodl-tmp/vllm_8000.log 2>&1 &

# 卡1：红队
CUDA_VISIBLE_DEVICES=1 nohup vllm serve /root/autodl-tmp/models/Qwen3-8B \
  --served-model-name qwen3-7b-redteam \
  --port 8001 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching \
  > /root/autodl-tmp/vllm_8001.log 2>&1 &
```

两个都在**同一台机器**上用 8000/8001，脚本走 `localhost`，不需要开公网端口。

> **V100 / Volta（SM70）必读**
>
> 当前实例若是 V100-32GB，上面两条命令要加三个参数，否则起不来或报错：
>
> ```bash
> export VLLM_ATTENTION_BACKEND=XFORMERS   # V100 不支持 FlashAttention-2（需 SM80+）
> # 命令内追加：
>   --dtype float16 \                        # V100 无 bf16，Qwen3 默认 bf16 必须降级
>   --enforce-eager \                        # 跳过 CUDA graph，V100 上易出错且更省显存
>   --max-model-len 16384                    # 先保守，跑通再试 32768
> ```
>
> 若报 `no supported attention backend`：换 `VLLM_ATTENTION_BACKEND=TORCH_SDPA`；
> 仍失败则降版本 `pip install vllm==0.6.6`（0.7+ 逐步移除 xformers，V100 反而更难）。

确认就绪：

```bash
curl -s http://localhost:8000/health && echo " 8000 OK"
curl -s http://localhost:8001/health && echo " 8001 OK"
```

---

## 4. 改配置

`config.yml` 里把三个角色切到本地：

```yaml
cognition:
  qwen_general:
    provider: openai_compatible
    base_url: http://localhost:8000/v1
    api_key: local-vllm
    default_model: qwen3-8b
    temperature: 0
    extra_body:
      chat_template_kwargs:
        enable_thinking: false      # 必须和训练时一致

  qwen_redteam:
    provider: openai_compatible
    base_url: http://localhost:8001/v1
    api_key: local-vllm
    default_model: qwen3-7b-redteam

roles:
  writer:      { backend: qwen_general, handle: qwen3-8b, temperature: 0 }
  draft:       { backend: qwen_general, handle: qwen3-8b, temperature: 0 }
  red_team:    { backend: qwen_redteam, handle: qwen3-7b-redteam }
  # planner 与搜索思考仍走 API（max/plus 档），不在这里
```

**红队的 temperature 不要写死在 config 里**，用环境变量传（见下）。

---

## 5. 跑起来

### 5.1 测写作执行率（第一道闸门）

```bash
# 先采一条批评
DR_LLM_TEMPERATURE=0.9 python scripts/sample_critiques.py \
  --draft output_report_2026-08-15_余承东竹知了事件.md \
  --k 1 --out data/flywheel/exec_raw.jsonl

# 测 writer 能不能落实
python scripts/check_execution_rate.py \
  --draft output_report_2026-08-15_余承东竹知了事件.md \
  --raw data/flywheel/exec_raw.jsonl \
  --out data/flywheel/exec_report.json
```

**执行率 < 50% → 停。** 换更大的写作模型，或先简化批评表述。这一步不过，训了也是白训。

### 5.2 采样红队

```bash
DR_LLM_TEMPERATURE=0.9 python scripts/sample_critiques.py \
  --draft-dir . --pattern "output_report_*.md" \
  --k 4 --out data/flywheel/raw.jsonl
```

`DR_LLM_TEMPERATURE` 不能漏——config 里是 0 的话，k 条一模一样，advantage 全 0。

### 5.3 体检

```bash
python scripts/inspect_samples.py --raw data/flywheel/raw.jsonl
```

看三个数：解析率（<80% 说明解析器要调）、过滤原因分布、组内多样性（<0.3 说明采样没差异）。

### 5.4 核实 + 导出

```bash
# 只用规则（免费，覆盖低）
python scripts/run_flywheel.py --raw data/flywheel/raw.jsonl \
  --out data/flywheel/samples.json --no-judge

# 规则 + judge（覆盖高，要 token）
python scripts/run_flywheel.py --raw data/flywheel/raw.jsonl \
  --out data/flywheel/samples.json
```

看两个数：**核实覆盖率**（<30% 说明奖励基本是常数）、**组内 std**（不可用组占比高就别训）。

### 5.5 收集 bad case

```bash
python scripts/collect_badcases.py \
  --samples data/flywheel/samples.json \
  --out data/flywheel/badcases.md
```

输出分三堆：误报（红队指错）、被过滤（不可用）、无法判定。
**第三堆最值得人工看**——它告诉你规则覆盖不到哪里，决定要不要扩规则或必须走 judge。

### 5.6 GRPO 训练

**先冒烟**（10 分钟，确认能跑通，不追求效果）：

```bash
# 训练要 LoRA，写作实例可以让位；这里只训红队，卡1 够用
python scripts/train_grpo.py --smoke \
  --model /root/autodl-tmp/models/Qwen3-8B \
  --raw data/flywheel/raw.jsonl \
  --out data/flywheel/grpo_smoke
```

冒烟只看三件事：能生成、能算奖励、能 backward。OOM 就调小 `--max-prompt-chars`。

**正式跑**：

```bash
python scripts/train_grpo.py \
  --model /root/autodl-tmp/models/Qwen3-8B \
  --raw data/flywheel/raw.jsonl \
  --steps 200 --k 4 --lr 1e-5 --kl-coef 0.04 \
  --max-prompt-chars 12000 --rollout-chunk 2 \
  --out data/flywheel/grpo_run1
```

关键参数与显存的关系：

| 参数 | 作用 | 加大后 |
|---|---|---|
| `--max-prompt-chars` | 草稿进 prompt 的字符上限（头尾保留） | 覆盖更多上下文，显存涨 |
| `--rollout-chunk` | 一次 generate 几条 | 快，但显存涨 |
| `--k` | 每组采样条数 | 组内方差更稳，成本线性涨 |

> **V100 上训练的两个额外注意**
>
> 1. 脚本会自动检测 bf16 不支持 → 退到 fp16 并启用 `GradScaler`。
>    如果看到 loss 变 NaN，先确认日志里有 `已启用 fp16 + GradScaler` 这行。
> 2. fp16 下梯度更容易下溢，`--lr` 建议先用 5e-6，别直接上 1e-5。

### 5.7 看曲线

```bash
python scripts/plot_train_log.py --log data/flywheel/grpo_run1/train_log.jsonl
```

终端会出一张 ASCII 表 + 判读结论；装了 matplotlib 还会存 `curves.png`。

**四个指标怎么读**：

- `reward_mean` 应缓慢上升。不升 = 奖励没区分度，或步数太少。
- `reward_std` 长期接近 0 = 组内没信号，白训。这是最先该看的。
- `kl` 涨太快 = 在追噪声，调大 `--kl-coef` 或降 lr。
- `comp_len` 突然变短 = 模型学会"少说话"，检查是否有"少提批评更安全"的激励。

---

## 6. 常见问题

**Q: 采样时报 `Connection refused`**
vLLM 还没起完。8B 加载 + CUDA graph 捕获通常 2~4 分钟，看 `vllm_8001.log` 的最后一行。

**Q: 红队输出解析不出来**
看 `inspect_samples.py` 的解析率。如果模型不按三段式输出，先把 `deep_research/prompts/red_team.py` 的输出格式约定再强调一遍，别急着改解析器。

**Q: 组内 std 全是 0**
按顺序查三件事：温度是不是 0；批处理是否把 k 条合成了一次调用；提示词是不是把输出压得太死（比如强制"最多 1 条"）。

**Q: 显存不够**
两个实例同时起会占满两张卡，训练时没卡可用。训练阶段停掉写作实例，或改用 AWQ 量化版（每个约 6G）。

**Q: 训练怎么跑**
用 `scripts/train_grpo.py`（见 5.6）。它自己完成 rollout + 打分 + 反向传播，
不依赖 verl。训练时**停掉写作实例**，把那张卡空出来——单卡 32G 跑 8B + LoRA 刚好。
