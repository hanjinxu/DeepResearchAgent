#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
后训练权重的量化脚本（llm-compressor）

用途
----
对**后训练之后合并好的 BF16 权重**做量化，产出 vLLM 可直接加载的压缩 checkpoint。

为什么必须对「后训练后的权重」做量化
------------------------------------
AWQ 的 scale / GPTQ 的 Hessian 都是基于权重在**校准数据上的激活分布**搜出来的。
后训练改变了权重分布 → 激活分布也变 → base 模型上搜出的 scale 不再适配。
⇒ 校准必须基于最终权重。这是"量化放在后训练之后"的技术实质，不是流程偏好。

⚠️ 环境隔离（官方明确建议）
    llm-compressor 与 vLLM 的 torch 版本约束会打架，分两个环境：
      环境 A（本脚本）：pip install llmcompressor datasets transformers
      环境 B（推理/评测）：pip install vllm "lm_eval[vllm,api]"
    lm_eval 评测量化模型时必须加 add_bos_token=True，否则数字不可比。

⚠️ AutoAWQ 已废弃（仓库 2025-05 归档），不要再用 autoawq，统一走 llm-compressor。

用法
----
# 0. 只打印配方，不加载模型（本地也能跑，用来确认参数）
python bench/quantize.py --scheme gptq --model /path/to/merged-bf16 --dry-run

# 1. FP8 —— data-free，不需要校准集，最快（9B 上约 2 分钟）
python bench/quantize.py --scheme fp8 --model /root/models/merged-bf16

# 2. GPTQ W4A16 —— 通用语料校准
python bench/quantize.py --scheme gptq --model /root/models/merged-bf16 \
    --calib-source general --calib-samples 512 --max-seq-len 2048

# 3. GPTQ W4A16 —— 领域语料校准（⭐ 只有你有的那组对比）
python bench/quantize.py --scheme gptq --model /root/models/merged-bf16 \
    --calib-source domain --calib-samples 512 --max-seq-len 2048

# 4. AWQ（两段式 recipe，最慢，9B 上约 30 分钟）
python bench/quantize.py --scheme awq --model /root/models/merged-bf16 \
    --calib-source domain --calib-samples 256 --max-seq-len 512

# 5. 自己录制的 prompt 做校准（最贴真实分布，推荐）
python bench/quantize.py --scheme gptq --model /root/models/merged-bf16 \
    --calib-source file --calib-file /root/calib/prompts.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent

# 永远不量化这些：embedding / lm_head / 视觉塔 等
# 官方 Qwen 示例就是用正则排除 lm_head 和 embed_tokens
DEFAULT_IGNORE = [r"re:.*lm_head", r"re:.*embed_tokens$"]

SCHEMES = ("fp8", "gptq", "awq", "w8a8")


# ---------------------------------------------------------------------------
# 一、校准数据
# ---------------------------------------------------------------------------
def collect_domain_texts(max_chars_per_chunk: int = 3000) -> List[str]:
    """从项目自带的文件里抽中文舆情语料，作为领域校准集。

    优先用真实产出（报告），因为它的分布最接近模型实际处理的内容；
    prompt 模板的字符串字面量也一并收进来，覆盖指令部分。
    """
    texts: List[str] = []

    # 1. 项目产出的报告（最接近模型实际输入的内容）
    for pat in ("output_report_*.md", "results/*.md"):
        for p in sorted(REPO_ROOT.glob(pat)):
            try:
                txt = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i in range(0, len(txt), max_chars_per_chunk):
                chunk = txt[i:i + max_chars_per_chunk].strip()
                if len(chunk) > 200:
                    texts.append(chunk)

    # 2. prompts 目录里的字符串字面量（覆盖指令部分）
    for p in sorted((REPO_ROOT / "deep_research" / "prompts").glob("*.py")):
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.findall(r'"""(.*?)"""', src, flags=re.S):
            m = m.strip()
            if len(m) > 200:
                texts.append(m)

    return texts


def load_texts_from_file(path: Path) -> List[str]:
    """从 jsonl / txt 读校准文本。

    jsonl 支持三种字段名：text / content / prompt
    （后续用 replay 录制真实 prompt 时直接写 {"prompt": "..."} 即可）
    """
    if not path.exists():
        raise SystemExit(f"校准文件不存在：{path}")
    texts: List[str] = []
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".jsonl":
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("text", "content", "prompt"):
                if isinstance(obj.get(key), str) and obj[key].strip():
                    texts.append(obj[key])
                    break
    else:
        texts = [blk.strip() for blk in raw.split("\n\n") if len(blk.strip()) > 200]
    return texts


def build_calibration_dataset(
    source: str,
    tokenizer: Any,
    *,
    calib_file: Optional[str],
    num_samples: int,
    max_seq_len: int,
    general_dataset: str,
) -> Any:
    """构造 llm-compressor 能吃的校准集。返回 (dataset, 说明字符串)。"""
    from datasets import Dataset  # type: ignore

    if source == "general":
        # 官方示例的通用指令语料
        return None, f"通用语料 {general_dataset}（由 oneshot 内部按 splits 截取前 {num_samples} 条）"

    if source == "domain":
        texts = collect_domain_texts()
    else:  # file
        if not calib_file:
            raise SystemExit("--calib-source file 时必须给 --calib-file")
        texts = load_texts_from_file(Path(calib_file))

    if not texts:
        raise SystemExit(f"校准语料为空（source={source}）")

    # 循环补齐到需要的条数
    need = max(num_samples, 1)
    pool: List[str] = []
    while len(pool) < need:
        pool.extend(texts)
    pool = pool[: max(need * 2, need)]

    tokenized = tokenizer(
        pool,
        padding=False,
        truncation=True,
        max_length=max_seq_len,
        add_special_tokens=False,
    )
    ds = Dataset.from_dict({"input_ids": tokenized["input_ids"]})
    return ds, f"自定义语料 {len(pool)} 条（来自 {source}），已 tokenize 到 {max_seq_len}"


# ---------------------------------------------------------------------------
# 二、配方
# ---------------------------------------------------------------------------
def build_recipe(scheme: str, ignore: Sequence[str], dampening_frac: float,
                 actorder: Optional[str]) -> Any:
    """按 scheme 构造 llm-compressor 的 recipe。

    三种配方的差别（这是量化参数对比的核心）：
      fp8   —— QuantizationModifier(FP8_DYNAMIC)，data-free，权重+激活都压到 FP8
      gptq  —— GPTQModifier(W4A16)，需要校准；Hessian 引导的逐列量化
      awq   —— [AWQModifier(duo_scaling="both"), QuantizationModifier(W4A16_ASYM)]
               AWQ 先搜 activation-aware 的 scale，再交给 quantizer 落盘
      w8a8  —— QuantizationModifier(W8A8)，需要校准；激活静态量化，风险最大
    """
    from llmcompressor.modifiers.gptq import GPTQModifier  # type: ignore
    from llmcompressor.modifiers.quantization import QuantizationModifier  # type: ignore

    if scheme == "fp8":
        return QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=list(ignore))

    if scheme == "gptq":
        kwargs: Dict[str, Any] = {
            "targets": "Linear",
            "scheme": "W4A16",
            "ignore": list(ignore),
            "dampening_frac": dampening_frac,
        }
        if actorder:
            kwargs["actorder"] = actorder
        return GPTQModifier(**kwargs)

    if scheme == "awq":
        from llmcompressor.modifiers.transform.awq import AWQModifier  # type: ignore

        return [
            AWQModifier(duo_scaling="both"),
            QuantizationModifier(targets="Linear", scheme="W4A16_ASYM", ignore=list(ignore)),
        ]

    if scheme == "w8a8":
        return QuantizationModifier(targets="Linear", scheme="W8A8", ignore=list(ignore))

    raise SystemExit(f"未知 scheme {scheme}，可选：{SCHEMES}")


def default_out_dir(model: str, scheme: str, source: str, samples: int) -> str:
    base = Path(model).name.rstrip("/")
    tag = f"{scheme.upper()}"
    if scheme != "fp8":
        tag += f"-{source}-{samples}"
    return str(Path(model).parent / f"{base}-{tag}")


# ---------------------------------------------------------------------------
# 三、主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="后训练权重的量化（llm-compressor）")
    ap.add_argument("--model", required=True, help="后训练合并后的 BF16 权重目录")
    ap.add_argument("--scheme", required=True, choices=SCHEMES)
    ap.add_argument("--calib-source", choices=["general", "domain", "file"], default="domain",
                    help="general=通用语料 / domain=项目自带中文语料 / file=自己录制的 prompt")
    ap.add_argument("--calib-file", default=None, help="--calib-source file 时的 jsonl/txt")
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--general-dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--dampening-frac", type=float, default=0.01,
                    help="GPTQ 专用。调低可能提升精度，但可能数值不稳")
    ap.add_argument("--actorder", default=None, choices=[None, "weight", "group"],
                    help="GPTQ 专用。weight 可提升精度且不增加推理延迟")
    ap.add_argument("--ignore", action="append", default=None,
                    help="额外忽略的层（可重复）。默认已忽略 lm_head / embed_tokens")
    ap.add_argument("--out", default=None, help="输出目录，默认按 scheme+校准集自动命名")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印配方与校准集信息，不加载模型、不量化")
    args = ap.parse_args()

    model_path = Path(args.model)
    ignore = list(DEFAULT_IGNORE) + list(args.ignore or [])
    out_dir = args.out or default_out_dir(args.model, args.scheme, args.calib_source, args.calib_samples)

    print("=" * 84)
    print("量化配置")
    print("=" * 84)
    print(f"  输入权重      {args.model}")
    print(f"  权重是否存在  {'✅' if model_path.exists() else '❌ 不存在（脚本会继续，真跑时才会报错）'}")
    print(f"  量化方案      {args.scheme.upper()}")
    print(f"  校准来源      {args.calib_source}")
    print(f"  校准样本数    {args.calib_samples}")
    print(f"  max_seq_len   {args.max_seq_len}")
    if args.scheme == "gptq":
        print(f"  dampening     {args.dampening_frac}")
        print(f"  actorder      {args.actorder or '（未设置）'}")
    print(f"  忽略层        {ignore}")
    print(f"  输出目录      {out_dir}")
    print("=" * 84)

    if args.scheme == "fp8":
        print("\n[提示] FP8_DYNAMIC 是 data-free，不需要校准集，通常 1~2 分钟完成。")

    if args.dry_run:
        # 只统计校准语料，不加载模型
        if args.calib_source in ("domain", "file"):
            texts = (collect_domain_texts() if args.calib_source == "domain"
                     else load_texts_from_file(Path(args.calib_file or "")))
            total_chars = sum(len(t) for t in texts)
            print(f"\n[dry-run] 校准语料：{len(texts)} 段，合计 {total_chars:,} 字符，"
                  f"平均 {total_chars // max(len(texts), 1)} 字符/段")
            if texts:
                head = texts[0][:120].replace("\n", " ")
                print(f"[dry-run] 样例开头：{head}...")
            if len(texts) < 128:
                print(f"[dry-run] ⚠️ 语料只有 {len(texts)} 段，偏少——校准样本数会被循环补齐，")
                print(f"[dry-run]    重复样本会削弱 activation-aware 量化的效果。")
                print(f"[dry-run]    建议：扩大语料，或改用真实录制的 prompt：")
                print(f"[dry-run]      --calib-source file --calib-file /root/calib/prompts.jsonl")
        print("\n[dry-run] 未加载模型、未量化。")
        return

    # ---- 真跑 ----
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        from llmcompressor import oneshot  # type: ignore
    except ImportError as exc:
        print(f"\n❌ 缺少依赖：{exc}", file=sys.stderr)
        print("   环境 A 需要：pip install llmcompressor datasets transformers", file=sys.stderr)
        print("   注意 llm-compressor 与 vLLM 官方建议分环境安装。", file=sys.stderr)
        raise SystemExit(3)

    print("\n加载 tokenizer 与模型（BF16）...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype="auto",
                                                 trust_remote_code=True)

    recipe = build_recipe(args.scheme, ignore, args.dampening_frac, args.actorder)

    oneshot_kwargs: Dict[str, Any] = {
        "model": model,
        "recipe": recipe,
        "output_dir": out_dir,
    }

    if args.scheme != "fp8":
        ds, desc = build_calibration_dataset(
            args.calib_source, tokenizer,
            calib_file=args.calib_file,
            num_samples=args.calib_samples,
            max_seq_len=args.max_seq_len,
            general_dataset=args.general_dataset,
        )
        if ds is None:
            oneshot_kwargs["dataset"] = args.general_dataset
            oneshot_kwargs["splits"] = f"train_sft[:{args.calib_samples}]"
        else:
            oneshot_kwargs["dataset"] = ds
        oneshot_kwargs["num_calibration_samples"] = args.calib_samples
        oneshot_kwargs["max_seq_length"] = args.max_seq_len
        print(f"校准集：{desc}")

    print("\n开始量化（耗时因 scheme 而异：FP8 分钟级，GPTQ 十分钟级，AWQ 半小时级）...")
    oneshot(**oneshot_kwargs)

    tokenizer.save_pretrained(out_dir)
    print(f"\n✅ 量化完成：{out_dir}")
    print(f"   下一步：./bench/vllm_matrix.sh {args.scheme} "
          f"MODEL_PATH={out_dir}")


if __name__ == "__main__":
    main()
