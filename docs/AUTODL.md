# AutoDL 运行手册

本文只讲**怎么在 AutoDL 上把这套跑起来**。设计取舍见 `docs/` 下其它文档与代码注释。

---

## 0. 先确认你在哪一步

| 步骤 | 需要 GPU | 产出 | 判据 |
|---|---|---|---|
| 1. 测写作执行率 | 需要写作模型 | `exec_report.json` | 执行率 ≥ 50% 才继续 |
| 2. 采样红队 | 需要红队模型 | `raw.jsonl` | 解析率、多样性 |
| 3. 体检 + 导出 | 不需要（CPU 即可） | `samples.json` | 核实覆盖率、组内 std |
| 4. verl 训练 | 需要 | 训练曲线 | **脚本尚未实现** |

第 3 步只读文件、不调模型（除非开 judge），可以在本机跑。

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
**还没实现。** 仓库里目前只有样本侧（解析/过滤/核实/打分/导出），verl 的 dataset 适配和 reward 接口要另写。
