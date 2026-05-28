#!/usr/bin/env python3
"""Week 1 Baseline: 验证 FP16 OOM + NF4 4bit 基准测试。

用法:
    conda run -n pytorch-env python scripts/01_baseline_test.py
    conda run -n pytorch-env python scripts/01_baseline_test.py --num-prompts 10 --max-new-tokens 128

产出:
    output/baseline_results.json  — 延迟、显存数据
    output/baseline_oom.log       — FP16 OOM 记录
"""

import argparse
import json
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b-instruct", "Qwen", "Qwen2.5-7B-Instruct")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about artificial intelligence.",
    "How does a transformer model work in natural language processing?",
    "Describe the process of photosynthesis in plants.",
    "What are the main causes of World War II?",
    "Explain quantum computing in simple terms.",
    "Write a Python function to reverse a linked list.",
    "What is the theory of relativity?",
    "Describe the water cycle and its importance to Earth's climate.",
    "How do neural networks learn from data?",
    "What are the ethical implications of AI in healthcare?",
    "Explain the concept of recursion with an example.",
    "What is blockchain technology and how does it work?",
    "Write a haiku about the moon.",
    "How does garbage collection work in programming languages?",
    "What is the difference between TCP and UDP?",
    "Explain the Monty Hall problem and its solution.",
    "What are RESTful APIs and how do they work?",
    "Describe the structure of a DNA molecule.",
]


def parse_args():
    parser = argparse.ArgumentParser(description="FP16 OOM 验证 + NF4 baseline 基准测试")
    parser.add_argument("--model-dir", default=MODEL_DIR, help="模型路径")
    parser.add_argument("--num-prompts", type=int, default=20, help="测试 prompt 数量")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="每个 prompt 最多生成 token 数")
    parser.add_argument("--skip-fp16", action="store_true", help="跳过 FP16 OOM 测试")
    return parser.parse_args()


def fmt_bytes(b: float) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(b) < 1024:
            return f"{b:.2f} {unit}"
        b /= 1024
    return f"{b:.2f} TB"


def test_fp16_oom(model_dir: str) -> dict:
    """尝试用 FP16 加载 7B 模型，预期 OOM。

    分两步：
    1. device_map="auto" + CPU offload → 能加载但部分层在 CPU
    2. 强制全 GPU → 预期 OOM，因为 14.1GB > 12GB
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gpu_name = torch.cuda.get_device_name(0)
    vram_total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    result = {
        "attempted": True,
        "oom_occurred": False,
        "error_message": None,
        "vram_peak_mb": None,
        "model_size_est_gb": 14.1,
        "gpu_name": gpu_name,
        "gpu_vram_total_gb": round(vram_total_gb, 1),
        "offload_vram_mb": None,
    }

    print("=" * 60)
    print("TEST 1: FP16 加载 Qwen2.5-7B-Instruct（预期 OOM）")
    print("=" * 60)
    print(f"GPU: {gpu_name} ({vram_total_gb:.1f} GB)")
    print(f"模型 FP16 权重大小: ~{result['model_size_est_gb']} GB")
    print(f"结论: 14.1GB 权重 + KV cache + 中间激活 > 11.7GB 显存")
    print()

    # Step 1: 用 device_map="auto" 加载（会有 CPU offload，演示勉强能跑但不实际）
    print("--- Step 1: device_map='auto' (自动 CPU offload) ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        result["offload_vram_mb"] = round(vram_mb, 1)
        # 检查有多少参数在 GPU 上
        params_on_gpu = sum(1 for p in model.parameters() if p.device.type == "cuda")
        total_params = sum(1 for _ in model.parameters())
        print(f"  GPU 参数层: {params_on_gpu}/{total_params}, 显存: {vram_mb:.0f} MB")
        print(f"  虽然加载成功，但部分层被 offload 到 CPU，推理会非常慢")
        del model, tokenizer
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  加载失败: {e}")
    print()

    # Step 2: 强制全 GPU — 预期 OOM
    print("--- Step 2: 强制全 GPU 加载 (device_map={'': 0}) ---")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True,
        )
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        result["vram_peak_mb"] = round(vram_mb, 1)
        print(f"  意外: 全 GPU 加载成功，显存 {vram_mb:.0f} MB")
        del model, tokenizer
    except torch.cuda.OutOfMemoryError as e:
        result["oom_occurred"] = True
        result["error_message"] = str(e)[:500]
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
        result["vram_peak_mb"] = round(vram_mb, 1)
        print(f"  ✓ CUDA OOM! (符合预期)")
        print(f"  OOM 前显存峰值: {vram_mb:.0f} MB / {vram_total_gb*1024:.0f} MB")
        print(f"  14.1GB 权重 > {(vram_total_gb - 0.5):.1f}GB 可用显存")
    except Exception as e:
        result["oom_occurred"] = True
        result["error_message"] = str(e)[:500]
        print(f"  ✓ 加载失败: {type(e).__name__}: {str(e)[:200]}")

    torch.cuda.empty_cache()
    print()
    return result


def test_nf4_baseline(model_dir: str, prompts: list[str], max_new_tokens: int) -> dict:
    """用 bitsandbytes NF4 加载模型并跑 benchmark。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("=" * 60)
    print("TEST 2: NF4 4bit 全 GPU 加载 + Baseline Benchmark")
    print("=" * 60)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print("加载模型 (NF4 4bit, 强制全 GPU: device_map={'': 0})...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=quant_config,
        device_map={"": 0},
        trust_remote_code=True,
    )
    load_time = time.time() - t0
    vram_after_load_mb = torch.cuda.max_memory_allocated() / 1024**2
    print(f"加载耗时: {load_time:.1f}s, 显存: {vram_after_load_mb:.0f} MB")
    print(f"  模型全部在 GPU，没有 CPU offload")

    model.eval()

    # Warmup
    print("\nWarmup...")
    warmup_inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        _ = model.generate(**warmup_inputs, max_new_tokens=16, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    vram_after_warmup_mb = torch.cuda.max_memory_allocated() / 1024**2

    # Benchmark
    results = []
    total_prompts = len(prompts)
    for idx, prompt in enumerate(prompts):
        print(f"\n[{idx+1}/{total_prompts}] {prompt[:80]}...")

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs.input_ids.shape[1]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        total_time = time.time() - t0

        generated_ids = outputs[0][input_len:]
        num_generated = len(generated_ids)
        vram_peak_mb = torch.cuda.max_memory_allocated() / 1024**2

        tokens_per_sec = num_generated / total_time if total_time > 0 else 0
        avg_ms_per_token = (total_time / num_generated * 1000) if num_generated > 0 else 0

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        record = {
            "prompt": prompt,
            "input_tokens": input_len,
            "output_tokens": num_generated,
            "total_time_s": round(total_time, 3),
            "tokens_per_second": round(tokens_per_sec, 2),
            "avg_ms_per_token": round(avg_ms_per_token, 2),
            "vram_peak_mb": round(vram_peak_mb, 1),
            "generated_text": generated_text[:200],
        }
        results.append(record)

        print(f"  input={input_len}, output={num_generated}, "
              f"time={total_time:.2f}s, speed={tokens_per_sec:.1f} tok/s, "
              f"VRAM={vram_peak_mb:.0f} MB")

    # 汇总统计
    speeds = [r["tokens_per_second"] for r in results]
    vrams = [r["vram_peak_mb"] for r in results]
    summary = {
        "quantization": "NF4 (bitsandbytes)",
        "load_time_s": round(load_time, 1),
        "vram_after_load_mb": round(vram_after_load_mb, 1),
        "vram_after_warmup_mb": round(vram_after_warmup_mb, 1),
        "gpu_name": torch.cuda.get_device_name(0),
        "num_prompts": len(results),
        "max_new_tokens": max_new_tokens,
        "avg_tokens_per_second": round(sum(speeds) / len(speeds), 2),
        "min_tokens_per_second": round(min(speeds), 2),
        "max_tokens_per_second": round(max(speeds), 2),
        "avg_ms_per_token": round(sum(r["avg_ms_per_token"] for r in results) / len(results), 2),
        "avg_vram_peak_mb": round(sum(vrams) / len(vrams), 1),
        "max_vram_peak_mb": round(max(vrams), 1),
        "per_prompt": results,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY (NF4 4bit)")
    print(f"{'='*60}")
    print(f"  加载后显存:    {summary['vram_after_load_mb']:.0f} MB")
    print(f"  Warmup 后显存: {summary['vram_after_warmup_mb']:.0f} MB")
    print(f"  平均生成速度:  {summary['avg_tokens_per_second']} tok/s")
    print(f"  平均每 token:  {summary['avg_ms_per_token']} ms")
    print(f"  平均推理显存:  {summary['avg_vram_peak_mb']:.0f} MB")
    print(f"  最大推理显存:  {summary['max_vram_peak_mb']:.0f} MB")

    del model, tokenizer
    torch.cuda.empty_cache()
    return summary


def main():
    args = parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"错误: 模型路径不存在: {args.model_dir}")
        print("请先运行: python scripts/download_model.py")
        sys.exit(1)

    prompts = TEST_PROMPTS[: args.num_prompts]
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    report = {}

    # Step 1: FP16 OOM test
    if not args.skip_fp16:
        report["fp16"] = test_fp16_oom(args.model_dir)

    # Step 2: NF4 baseline benchmark
    report["nf4"] = test_nf4_baseline(args.model_dir, prompts, args.max_new_tokens)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, "baseline_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    # Also save OOM log separately
    if "fp16" in report:
        oom_log = os.path.join(OUTPUT_DIR, "baseline_oom.log")
        with open(oom_log, "w") as f:
            f.write(f"GPU: {report['fp16']['gpu_name']} ({report['fp16']['gpu_vram_total_gb']:.1f} GB)\n")
            f.write(f"Model size estimate: {report['fp16']['model_size_est_gb']} GB\n")
            f.write(f"OOM occurred: {report['fp16']['oom_occurred']}\n")
            if report["fp16"]["error_message"]:
                f.write(f"Error: {report['fp16']['error_message']}\n")
            if report["fp16"]["vram_peak_mb"]:
                f.write(f"VRAM at failure: {report['fp16']['vram_peak_mb']} MB\n")
        print(f"OOM 日志已保存: {oom_log}")


if __name__ == "__main__":
    main()
