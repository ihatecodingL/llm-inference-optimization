#!/usr/bin/env python3
"""Week 2: 量化模型全对比 Benchmark。

对 6 种 GGUF 精度跑:
  1. llama-perplexity → PPL (WikiText-2 test set, ~237K tokens)
  2. llama-bench      → Prefill/Decode 速度 (tok/s) + 显存

用法:
    conda run -n pytorch-env python scripts/03_quant_benchmark.py
    conda run -n pytorch-env python scripts/03_quant_benchmark.py --skip-perplexity  # 只测速度
    conda run -n pytorch-env python scripts/03_quant_benchmark.py --skip-bench       # 只测 PPL

产出:
    output/benchmark_results.json — 完整对比数据
    output/figures/                — 图表 (后续用 plot_results.py 生成)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", os.path.join(PROJECT_ROOT, "..", "llama.cpp"))
BENCH_BIN = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-bench")
PERPLEXITY_BIN = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-perplexity")
WIKITEXT_TEST = os.path.join(PROJECT_ROOT, "data", "wiki.test.raw")

MODELS = {
    "FP16":   os.path.join(GGUF_DIR, "qwen2.5-7b-fp16.gguf"),
    "Q8_0":   os.path.join(GGUF_DIR, "qwen2.5-7b-Q8_0.gguf"),
    "Q5_K_M": os.path.join(GGUF_DIR, "qwen2.5-7b-Q5_K_M.gguf"),
    "Q4_K_M": os.path.join(GGUF_DIR, "qwen2.5-7b-Q4_K_M.gguf"),
    "Q4_0":   os.path.join(GGUF_DIR, "qwen2.5-7b-Q4_0.gguf"),
    "IQ4_NL": os.path.join(GGUF_DIR, "qwen2.5-7b-IQ4_NL.gguf"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="量化模型全对比 Benchmark")
    parser.add_argument("--skip-perplexity", action="store_true", help="跳过 PPL 测试")
    parser.add_argument("--skip-bench", action="store_true", help="跳过速度 Benchmark")
    parser.add_argument("--models", default="FP16,Q8_0,Q5_K_M,Q4_K_M,Q4_0,IQ4_NL",
                        help="逗号分隔的模型列表")
    parser.add_argument("--n-gpu-layers", type=int, default=99, help="GPU offload 层数")
    return parser.parse_args()


def ensure_wikitext():
    """下载 WikiText-2 测试集（通过 HuggingFace datasets）。"""
    if os.path.isfile(WIKITEXT_TEST):
        with open(WIKITEXT_TEST) as f:
            lines = f.readlines()
        print(f"WikiText-2 已存在: {WIKITEXT_TEST} ({len(lines)} lines)\n")
        return WIKITEXT_TEST

    os.makedirs(os.path.dirname(WIKITEXT_TEST), exist_ok=True)

    print("下载 WikiText-2 数据集 (HuggingFace)...")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test", trust_remote_code=False)
    print(f"  Test samples: {len(ds)}")

    with open(WIKITEXT_TEST, "w") as f:
        for item in ds:
            text = item["text"].strip()
            if text:
                f.write(text + "\n")

    with open(WIKITEXT_TEST) as f:
        lines = f.readlines()
    print(f"  已保存: {WIKITEXT_TEST} ({len(lines)} lines)\n")
    return WIKITEXT_TEST


def run_perplexity(model_path: str, model_name: str, test_file: str, n_gpu_layers: int) -> dict:
    """用 llama-perplexity 测 PPL。"""
    print(f"[{model_name}] 测 PPL...")

    cmd = [
        PERPLEXITY_BIN,
        "-m", model_path,
        "-f", test_file,
        "-ngl", str(n_gpu_layers),
        "--no-warmup",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    elapsed = time.time() - t0

    output = result.stdout + result.stderr

    if result.returncode != 0:
        print(f"  FAILED (exit={result.returncode})")
        return {"model": model_name, "error": output[-500:]}

    # 解析 PPL 输出。llama-perplexity 输出格式类似:
    # [1]4.1234,[2]5.6789,...  (chunk-level PPL)
    # 最后一行通常是: Final estimate: PPL = 6.1234 +/- 0.0456
    ppl_match = re.search(r"Final estimate.*?PPL\s*=\s*([\d.]+)", output)
    if ppl_match:
        ppl = float(ppl_match.group(1))
    else:
        # 尝试从 chunks 计算平均
        chunk_ppls = re.findall(r"\](\d+\.\d+),?", output)
        if chunk_ppls:
            ppl = sum(float(p) for p in chunk_ppls) / len(chunk_ppls)
        else:
            ppl = None

    print(f"  PPL={ppl:.4f}, 耗时={elapsed:.1f}s" if ppl else f"  未找到 PPL, 耗时={elapsed:.1f}s")

    return {
        "model": model_name,
        "ppl": round(ppl, 4) if ppl else None,
        "time_s": round(elapsed, 1),
    }


def run_bench(model_path: str, model_name: str, n_gpu_layers: int) -> dict:
    """用 llama-bench 测推理速度。"""
    print(f"[{model_name}] 测速度...")

    cmd = [
        BENCH_BIN,
        "-m", model_path,
        "-ngl", str(n_gpu_layers),
        "-p", "512",
        "-n", "128",
        "-b", "2048",
        "-r", "3",
        "-o", "json",
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"  FAILED (exit={result.returncode})")
        return {"model": model_name, "error": result.stderr[-500:]}

    # llama-bench JSON 输出: 一个 JSON 数组，包含 prefill 和 decode 两条记录
    # prefill: n_prompt>0, n_gen=0  → avg_ts = prompt processing tok/s
    # decode:  n_prompt=0, n_gen>0  → avg_ts = text generation tok/s
    try:
        entries = json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  无法解析 benchmark JSON")
        return {"model": model_name, "error": "json parse failed", "raw": result.stdout[-500:]}

    prefill = next((e for e in entries if e.get("n_prompt", 0) > 0), None)
    decode = next((e for e in entries if e.get("n_gen", 0) > 0), None)

    pp_tok = prefill["avg_ts"] if prefill else 0
    tg_tok = decode["avg_ts"] if decode else 0
    model_size = entries[0].get("model_size", 0) if entries else 0
    model_size_mb = model_size / 1024**2

    print(f"  Prefill={pp_tok:.1f} tok/s, Decode={tg_tok:.1f} tok/s, "
          f"Model={model_size_mb:.0f} MB, 耗时={elapsed:.1f}s")

    return {
        "model": model_name,
        "prefill_tok_s": round(pp_tok, 2),
        "decode_tok_s": round(tg_tok, 2),
        "decode_ms_per_token": round(1000 / tg_tok, 2) if tg_tok > 0 else None,
        "model_size_mb": round(model_size_mb, 1),
        "bench_time_s": round(elapsed, 1),
    }


def main():
    args = parse_args()
    selected = [m.strip() for m in args.models.split(",") if m.strip()]

    for name in selected:
        if name not in MODELS:
            print(f"错误: 未知模型 '{name}'，可选: {list(MODELS.keys())}")
            sys.exit(1)
        if not os.path.isfile(MODELS[name]):
            print(f"错误: 模型文件不存在: {MODELS[name]}")
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    results = {
        "gpu": "NVIDIA GeForce RTX 3060 12GB",
        "n_gpu_layers": args.n_gpu_layers,
        "perplexity": [],
        "bench": [],
    }

    # Step 1: Perplexity (质量评估)
    if not args.skip_perplexity:
        ppl_file = ensure_wikitext()
        print("=" * 60)
        print("Step 1: Perplexity 测试 (WikiText-2)")
        print("=" * 60)
        for name in selected:
            ppl_result = run_perplexity(MODELS[name], name, ppl_file, args.n_gpu_layers)
            results["perplexity"].append(ppl_result)
            print()
    else:
        print("跳过 PPL 测试 (--skip-perplexity)\n")

    # Step 2: Speed Benchmark
    if not args.skip_bench:
        print("=" * 60)
        print("Step 2: 推理速度 Benchmark (llama-bench)")
        print("=" * 60)
        for name in selected:
            bench_result = run_bench(MODELS[name], name, args.n_gpu_layers)
            results["bench"].append(bench_result)
            print()
    else:
        print("跳过 Speed Benchmark (--skip-bench)\n")

    # 汇总对比表
    print("=" * 80)
    print("汇总对比")
    print("=" * 80)

    # 合并 PPL + Bench 数据
    ppl_map = {r["model"]: r.get("ppl") for r in results["perplexity"]}
    bench_map = {r["model"]: r for r in results["bench"]}

    header = f"{'Model':<8} {'File Size':>8} {'PPL':>8} {'Prefill':>10} {'Decode':>10} {'ms/tok':>8}"
    print(header)
    print("-" * len(header))

    combined = []
    for name in selected:
        ppl = ppl_map.get(name, "?")
        b = bench_map.get(name, {})
        size_mb = b.get("model_size_mb", 0)
        size_gb = size_mb / 1024 if size_mb else "?"

        if isinstance(size_gb, float):
            size_str = f"{size_gb:.1f} GB"
        else:
            size_str = str(size_gb)

        pp_tok = b.get("prefill_tok_s", "?")
        tg_tok = b.get("decode_tok_s", "?")
        ms_tok = b.get("decode_ms_per_token", "?")

        pp_str = f"{pp_tok:.0f} tok/s" if isinstance(pp_tok, float) else str(pp_tok)
        tg_str = f"{tg_tok:.1f} tok/s" if isinstance(tg_tok, float) else str(tg_tok)
        ms_str = f"{ms_tok:.1f} ms" if isinstance(ms_tok, float) else str(ms_tok)
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)

        print(f"{name:<8} {size_str:>8} {ppl_str:>8} {pp_str:>10} {tg_str:>10} {ms_str:>8}")

        combined.append({
            "model": name,
            "file_size_gb": round(size_gb, 2) if isinstance(size_gb, float) else None,
            "ppl": ppl if isinstance(ppl, float) else None,
            "prefill_tok_s": pp_tok if isinstance(pp_tok, float) else None,
            "decode_tok_s": tg_tok if isinstance(tg_tok, float) else None,
            "decode_ms_per_token": ms_tok if isinstance(ms_tok, float) else None,
        })

    results["comparison"] = combined

    # Save
    output_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {output_path}")

    # Also save a markdown table for easy reading
    md_path = os.path.join(OUTPUT_DIR, "benchmark_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Qwen2.5-7B 量化 Benchmark 结果\n\n")
        f.write(f"GPU: {results['gpu']}, n_gpu_layers: {args.n_gpu_layers}\n\n")
        f.write("| Model | File Size | PPL | Prefill | Decode | ms/tok |\n")
        f.write("|-------|----------|-----|---------|--------|--------|\n")
        for c in combined:
            size = f"{c['file_size_gb']:.1f} GB" if c['file_size_gb'] else "?"
            ppl = f"{c['ppl']:.2f}" if c['ppl'] else "?"
            pp = f"{c['prefill_tok_s']:.0f} tok/s" if c['prefill_tok_s'] else "?"
            tg = f"{c['decode_tok_s']:.1f} tok/s" if c['decode_tok_s'] else "?"
            ms = f"{c['decode_ms_per_token']:.1f} ms" if c['decode_ms_per_token'] else "?"
            f.write(f"| {c['model']} | {size} | {ppl} | {pp} | {tg} | {ms} |\n")
    print(f"Markdown 已保存: {md_path}")


if __name__ == "__main__":
    main()
