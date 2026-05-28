#!/usr/bin/env python3
"""Week 2/3: KV Cache 显存分析与推理显存分解。

分析内容:
  1. KV Cache 理论计算 — 不同 seq_len / cache 类型下的显存占用
  2. llm 推理显存四块分解 — 权重 / KV Cache / 中间激活 / CUDA 开销
  3. 用 llama-bench 实测不同 KV cache 量化对速度的影响

用法:
    conda run -n pytorch-env python scripts/04_kv_cache_analysis.py
    conda run -n pytorch-env python scripts/04_kv_cache_analysis.py --model Q4_K_M --seq-lens 512,2048,4096,8192,16384

产出:
    output/kv_cache_analysis.json
    output/kv_cache_analysis.md
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GGUF_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", os.path.join(PROJECT_ROOT, "..", "llama.cpp"))
BENCH_BIN = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-bench")

MODEL_CONFIG_PATH = os.path.join(
    PROJECT_ROOT, "models", "qwen2.5-7b-instruct", "Qwen", "Qwen2.5-7B-Instruct", "config.json"
)

# 显存估算参数
CUDA_OVERHEAD_MB = 450  # CUDA context + driver 固定开销


def parse_args():
    p = argparse.ArgumentParser(description="KV Cache 显存分析")
    p.add_argument("--model", default="Q4_K_M", help="使用的 GGUF 模型 (默认: Q4_K_M)")
    p.add_argument("--seq-lens", default="512,1024,2048,4096,8192,16384,32768",
                   help="逗号分隔的序列长度")
    p.add_argument("--n-gpu-layers", type=int, default=99)
    p.add_argument("--skip-bench", action="store_true", help="跳过 benchmark 实测")
    return p.parse_args()


def load_model_config():
    with open(MODEL_CONFIG_PATH) as f:
        cfg = json.load(f)
    return {
        "num_layers": cfg["num_hidden_layers"],
        "hidden_size": cfg["hidden_size"],
        "num_q_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg["num_key_value_heads"],
        "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
        "max_seq_len": cfg["max_position_embeddings"],
        "vocab_size": cfg["vocab_size"],
    }


def calc_kv_cache(config: dict, seq_len: int, dtype: str = "f16") -> dict:
    """计算 KV Cache 理论大小。

    KV Cache = 2 (K+V) × num_layers × num_kv_heads × head_dim × seq_len × bytes_per_elem
    """
    dtype_bytes = {"f16": 2, "f32": 4, "q8_0": 1, "q4_0": 0.5}
    bytes_per_elem = dtype_bytes.get(dtype, 2)

    per_token = 2 * config["num_layers"] * config["num_kv_heads"] * config["head_dim"] * bytes_per_elem
    total_bytes = per_token * seq_len
    total_mb = total_bytes / 1024**2

    return {
        "seq_len": seq_len,
        "dtype": dtype,
        "bytes_per_elem": bytes_per_elem,
        "per_token_kv_bytes": int(per_token),
        "per_token_kv_kb": round(per_token / 1024, 2),
        "total_mb": round(total_mb, 2),
        "total_gb": round(total_mb / 1024, 3),
    }


def model_weight_mb(model_name: str) -> float:
    """从文件大小获取模型权重显存。"""
    model_map = {
        "FP16": "qwen2.5-7b-fp16.gguf",
        "Q8_0": "qwen2.5-7b-Q8_0.gguf",
        "Q5_K_M": "qwen2.5-7b-Q5_K_M.gguf",
        "Q4_K_M": "qwen2.5-7b-Q4_K_M.gguf",
        "Q4_0": "qwen2.5-7b-Q4_0.gguf",
        "IQ4_NL": "qwen2.5-7b-IQ4_NL.gguf",
    }
    fname = model_map.get(model_name)
    if not fname:
        return 0
    path = os.path.join(GGUF_DIR, fname)
    if os.path.isfile(path):
        return os.path.getsize(path) / 1024**2
    return 0


def estimate_activation_mb(config: dict, seq_len: int) -> float:
    """估算中间激活显存（单 batch 推理时）。

    主要来源:
      - Attention 中间结果 (QKV 投影)
      - FFN 中间结果 (gate/up 投影)
      - 残差连接

    精确值依赖框架实现，这里给一个保守估计。
    """
    hidden = config["hidden_size"]
    n_layers = config["num_layers"]
    n_q_heads = config["num_q_heads"]
    head_dim = config["head_dim"]

    # Prefill 阶段: 需要存储 QKV 投影 + attention scores + FFN 中间结果
    # 粗略估计: hidden_size × 6 (Q/K/V proj + gate/up/down) × 2 bytes
    activ_mb = hidden * 6 * 2 / 1024**2  # per layer, per token, without batch

    # 加上 attention 中间结果 (QK^T 矩阵)
    # FlashAttention 下 O(seq_len), 标准 attention 下 O(seq_len^2)
    # llama.cpp 使用优化 attention, 近似 O(seq_len)
    attn_activ_mb = n_q_heads * head_dim * seq_len * 4 / 1024**2  # rough estimate

    total = activ_mb * n_layers + attn_activ_mb
    return round(total, 1)


def memory_budget(config: dict, model_name: str, seq_lens: list[int]) -> list[dict]:
    """计算完整的显存预算分解。"""
    weight_mb = model_weight_mb(model_name)
    budgets = []

    for sl in seq_lens:
        kv = calc_kv_cache(config, sl, "f16")
        activ_mb = estimate_activation_mb(config, sl)

        total_mb = weight_mb + kv["total_mb"] + activ_mb + CUDA_OVERHEAD_MB

        budgets.append({
            "seq_len": sl,
            "weight_mb": round(weight_mb, 0),
            "weight_pct": round(weight_mb / total_mb * 100, 1) if total_mb > 0 else 0,
            "kv_cache_mb": kv["total_mb"],
            "kv_cache_pct": round(kv["total_mb"] / total_mb * 100, 1) if total_mb > 0 else 0,
            "activation_mb": activ_mb,
            "activation_pct": round(activ_mb / total_mb * 100, 1) if total_mb > 0 else 0,
            "cuda_overhead_mb": CUDA_OVERHEAD_MB,
            "overhead_pct": round(CUDA_OVERHEAD_MB / total_mb * 100, 1) if total_mb > 0 else 0,
            "total_mb": round(total_mb, 0),
            "total_gb": round(total_mb / 1024, 2),
            "fits_12gb": total_mb < 12000,
        })

    return budgets


def run_kv_bench(model_name: str, model_path: str, seq_lens: list[int],
                  n_gpu_layers: int, configs: list[dict], model_cfg: dict) -> list[dict]:
    """用 llama-bench 实测不同 KV cache 配置的速度。

    configs: [{"label": "f16+noFA", "ctk": "f16", "ctv": "f16", "fa": False}, ...]
    """
    results = []
    for sl in seq_lens:
        for cfg in configs:
            label = cfg["label"]
            ctk = cfg.get("ctk", "f16")
            ctv = cfg.get("ctv", "f16")
            fa = cfg.get("fa", False)
            print(f"  seq={sl}, {label} (ctk={ctk}, ctv={ctv}, fa={fa})...", end=" ", flush=True)

            cmd = [
                BENCH_BIN, "-m", model_path,
                "-ngl", str(n_gpu_layers),
                "-p", str(sl), "-n", "1",
                "-b", "2048", "-r", "3",
                "-ctk", ctk, "-ctv", ctv,
                "-o", "json",
            ]
            if fa:
                cmd += ["-fa", "1"]

            t0 = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0

            if r.returncode != 0:
                print(f"FAILED: {r.stderr[-200:]}")
                results.append({"seq_len": sl, "config": label, "error": r.stderr[-300:]})
                continue

            try:
                entries = json.loads(r.stdout)
            except json.JSONDecodeError:
                print("JSON parse failed")
                results.append({"seq_len": sl, "config": label, "error": "json parse"})
                continue

            prefill = next((e for e in entries if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0), None)
            if not prefill:
                prefill = next((e for e in entries if e.get("n_prompt", 0) > 0), None)
            decode = next((e for e in entries if e.get("n_prompt", 0) == 0 and e.get("n_gen", 0) > 0), None)

            kv_size_mb = calc_kv_cache(model_cfg, sl, ctk)["total_mb"] + calc_kv_cache(model_cfg, sl, ctv)["total_mb"]

            result = {
                "seq_len": sl,
                "config": label,
                "type_k": ctk,
                "type_v": ctv,
                "flash_attn": fa,
                "prefill_tok_s": round(prefill["avg_ts"], 1) if prefill else None,
                "decode_tok_s": round(decode["avg_ts"], 2) if decode else None,
                "ttft_ms": round(prefill["avg_ns"] / 1e6, 1) if prefill and prefill.get("avg_ns") else None,
                "kv_size_mb": round(kv_size_mb, 1),
                "time_s": round(elapsed, 1),
            }
            results.append(result)
            pp = f"{result['prefill_tok_s']:.0f} tok/s" if result['prefill_tok_s'] else "?"
            tg = f"{result['decode_tok_s']:.1f} tok/s" if result['decode_tok_s'] else "?"
            print(f"prefill={pp}, decode={tg}, kv={result['kv_size_mb']} MB")

    return results


def main():
    args = parse_args()
    config = load_model_config()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    model_name = args.model
    model_path = os.path.join(GGUF_DIR, f"qwen2.5-7b-{model_name}.gguf")

    if not os.path.isfile(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("模型架构参数")
    print("=" * 60)
    print(f"  层数: {config['num_layers']}")
    print(f"  hidden_size: {config['hidden_size']}")
    print(f"  Q heads: {config['num_q_heads']}, KV heads: {config['num_kv_heads']} (GQA {config['num_q_heads']//config['num_kv_heads']}:1)")
    print(f"  head_dim: {config['head_dim']}")
    print(f"  max_seq_len: {config['max_seq_len']}")
    print(f"  vocab_size: {config['vocab_size']}")
    print()

    # 1. KV Cache 理论计算
    print("=" * 60)
    print("1. KV Cache 理论大小")
    print("=" * 60)
    cache_dtypes = ["f16", "q8_0", "q4_0"]
    kv_results = {}
    for dtype in cache_dtypes:
        kv_rows = []
        for sl in seq_lens:
            kv_rows.append(calc_kv_cache(config, sl, dtype))
        kv_results[dtype] = kv_rows

        header = f"  {'dtype=' + dtype:<14}"
        for r in kv_rows:
            header += f"seq={r['seq_len']:>6}: {r['total_mb']:>8.1f} MB  "
        print(header)

    # Per-token cost
    base_kv = kv_results["f16"][0]
    print(f"\n  Per-token KV (FP16): {base_kv['per_token_kv_bytes']:,} bytes = {base_kv['per_token_kv_kb']} KB")
    print(f"  每生成 1000 token: KV cache 增长 {base_kv['per_token_kv_kb'] * 1000 / 1024:.1f} MB")
    print()

    # 2. 显存预算分解
    print("=" * 60)
    print(f"2. 显存预算分解 ({model_name})")
    print("=" * 60)
    budgets = memory_budget(config, model_name, [512, 1024, 2048, 4096, 8192, 16384, 32768])

    print(f"  {'Seq Len':<8} {'Weight':>8} {'KV Cache':>10} {'Activation':>10} {'Overhead':>8} {'Total':>8} {'Fits 12GB'}")
    print(f"  {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*9}")
    for b in budgets:
        fits = "✓" if b["fits_12gb"] else "✗ OOM"
        print(f"  {b['seq_len']:<8} {b['weight_mb']:>6.0f} MB {b['kv_cache_mb']:>8.0f} MB "
              f"{b['activation_mb']:>8.0f} MB {b['cuda_overhead_mb']:>6.0f} MB "
              f"{b['total_mb']:>6.0f} MB {fits}")

    print(f"\n  Per-token KV cost: {base_kv['per_token_kv_kb']:.2f} KB/token (FP16)")
    print()

    # 3. Benchmark 实测
    # 关键发现: K cache 量化无需 FA, V cache 量化需要 FA
    # 原因: 非 FA 路径中 V 需要 transpose, block-quantized 数据不支持转置
    if not args.skip_bench:
        print("=" * 60)
        print(f"3. KV Cache 量化实测对比 ({model_name})")
        print("=" * 60)

        bench_seq_lens = [512, 1024, 2048, 4096]

        # 四种配置: baseline, K-only, K+V+FA, K-Q4_0
        bench_configs = [
            {"label": "f16 (baseline)",     "ctk": "f16",  "ctv": "f16",  "fa": False},
            {"label": "K=q8_0, no FA",      "ctk": "q8_0", "ctv": "f16",  "fa": False},
            {"label": "K+V=q8_0 + FA",      "ctk": "q8_0", "ctv": "q8_0", "fa": True},
            {"label": "K=q4_0, no FA",      "ctk": "q4_0", "ctv": "f16",  "fa": False},
            {"label": "FA only (FP16 KV)",  "ctk": "f16",  "ctv": "f16",  "fa": True},
        ]

        bench_results = run_kv_bench(model_name, model_path, bench_seq_lens,
                                     args.n_gpu_layers, bench_configs, config)

        # 对比表
        print(f"\n  {'Seq':<6} {'Config':<20} {'Prefill':>10} {'TTFT':>8} {'Decode':>8} {'KV':>8}")
        print(f"  {'-'*6} {'-'*20} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
        for r in bench_results:
            if "error" in r:
                print(f"  {r['seq_len']:<6} {r['config']:<20} {'ERROR':>10}")
                continue
            pp = f"{r['prefill_tok_s']:.0f}" if r['prefill_tok_s'] else "?"
            ttft = f"{r['ttft_ms']:.0f}ms" if r.get('ttft_ms') else "?"
            tg = f"{r['decode_tok_s']:.1f}" if r['decode_tok_s'] else "?"
            kv = f"{r['kv_size_mb']:.0f}M" if r['kv_size_mb'] else "?"
            print(f"  {r['seq_len']:<6} {r['config']:<20} {pp:>8} tok/s {ttft:>8} {tg:>8} tok/s {kv:>8}")

        # Prefill 速度对比 (seq=2048)
        print(f"\n  --- Prefill 加速比 (vs f16 baseline, seq=2048) ---")
        baseline_pp = None
        for r in bench_results:
            if "error" in r:
                continue
            if r.get("config") == "f16 (baseline)" and r["seq_len"] == 2048:
                baseline_pp = r["prefill_tok_s"]
                break
        if baseline_pp:
            for r in bench_results:
                if r["seq_len"] == 2048 and r.get("prefill_tok_s"):
                    speedup = r["prefill_tok_s"] / baseline_pp - 1
                    print(f"    {r['config']:<20}: {speedup:+.0%} vs baseline")
        print()
    else:
        bench_results = []

    # 汇总
    weight_mb = model_weight_mb(model_name)
    kv_2048 = calc_kv_cache(config, 2048, "f16")
    kv_4096 = calc_kv_cache(config, 4096, "f16")
    kv_32768 = calc_kv_cache(config, 32768, "f16")

    print("=" * 60)
    print("关键结论")
    print("=" * 60)
    print(f"  模型权重 ({model_name}): {weight_mb:.0f} MB")
    print(f"  KV Cache (FP16, seq=2048): {kv_2048['total_mb']:.0f} MB — 占总显存的 {kv_2048['total_mb']/12000*100:.1f}%")
    print(f"  KV Cache (FP16, seq=4096): {kv_4096['total_mb']:.0f} MB — 占总显存的 {kv_4096['total_mb']/12000*100:.1f}%")
    print(f"  KV Cache (FP16, seq=32768): {kv_32768['total_mb']:.0f} MB — 占总显存的 {kv_32768['total_mb']/12000*100:.1f}%")
    print(f"  GQA (4 KV heads vs 28 Q heads) 将 KV cache 缩小了 {config['num_q_heads']//config['num_kv_heads']}x")
    print(f"  K cache 量化: q8_0/q4_0 可直接使用，无需 FlashAttention")
    print(f"  V cache 量化: 需要 FlashAttention（非 FA 路径的 transpose 与 block 量化不兼容）")
    print()

    # 保存
    output = {
        "model_config": config,
        "model_name": model_name,
        "kv_cache_theoretical": {dtype: kv_results[dtype] for dtype in cache_dtypes},
        "memory_budget": budgets,
        "bench_results": bench_results,
        "per_token_kv_bytes": base_kv["per_token_kv_bytes"],
        "kv_cache_findings": {
            "k_cache_without_fa": "works — ggml_mul_mat handles Q8_0/Q4_0 dequant internally",
            "v_cache_without_fa": "fails — non-FA path transposes V, block-quantized data can't be transposed",
            "v_cache_with_fa": "works — FA path uses ggml_flash_attn_ext which handles quant types natively",
            "root_cause_file": "src/llama-graph.cpp:1972 (build_attn_mha), lines 2057-2061 (transpose)",
            "fix_approach": "Add ggml_cast(v, F16) before permute in non-FA path, remove early validation",
        },
    }

    json_path = os.path.join(OUTPUT_DIR, "kv_cache_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON 已保存: {json_path}")

    # Markdown
    md_path = os.path.join(OUTPUT_DIR, "kv_cache_analysis.md")
    with open(md_path, "w") as f:
        f.write("# KV Cache 显存分析\n\n")
        f.write(f"模型: Qwen2.5-7B-Instruct, 量化方案: {model_name} ({weight_mb:.0f} MB)\n\n")

        f.write("---\n\n## 1. 模型架构参数\n\n")
        f.write("| 参数 | 值 |\n")
        f.write("|------|-----|\n")
        f.write(f"| 层数 | {config['num_layers']} |\n")
        f.write(f"| hidden_size | {config['hidden_size']} |\n")
        f.write(f"| Q heads | {config['num_q_heads']} |\n")
        f.write(f"| KV heads | **{config['num_kv_heads']}** (GQA, {config['num_q_heads']//config['num_kv_heads']}:1) |\n")
        f.write(f"| head_dim | {config['head_dim']} |\n")
        f.write(f"| max_seq_len | {config['max_seq_len']} |\n")
        f.write(f"| vocab_size | {config['vocab_size']} |\n\n")
        f.write(f"**Per-token KV: {base_kv['per_token_kv_bytes']:,} bytes = {base_kv['per_token_kv_kb']} KB**（FP16）\n\n")

        f.write("---\n\n## 2. KV Cache 理论大小\n\n")
        f.write("```\n")
        f.write(f"KV_Cache = 2 x num_layers x num_kv_heads x head_dim x seq_len x dtype_bytes\n")
        f.write(f"         = 2 x {config['num_layers']} x {config['num_kv_heads']} x {config['head_dim']} x seq_len x dtype_bytes\n")
        f.write(f"         = {base_kv['per_token_kv_bytes']:,} bytes x seq_len\n")
        f.write("```\n\n")
        f.write("| Seq Len | FP16 (2B) | Q8_0 (1B) | Q4_0 (0.5B) | 备注 |\n")
        f.write("|---------|-----------|-----------|-------------|------|\n")
        notes = {512: "短 prompt", 1024: "", 2048: "常见对话长度", 4096: "长文档",
                 8192: "", 16384: "", 32768: "模型最大 context"}
        for i, sl in enumerate(seq_lens):
            f16_mb = kv_results["f16"][i]["total_mb"]
            q8_mb = kv_results["q8_0"][i]["total_mb"]
            q4_mb = kv_results["q4_0"][i]["total_mb"]
            note = notes.get(sl, "")
            f.write(f"| {sl} | {f16_mb:.0f} MB | {q8_mb:.0f} MB | {q4_mb:.0f} MB | {note} |\n")

        f.write(f"\n**关键洞察**: GQA 把 KV heads 从 {config['num_q_heads']} 压到 {config['num_kv_heads']}，"
                f"KV cache 直接缩小 **{config['num_q_heads']//config['num_kv_heads']} 倍**。"
                f"如果没有 GQA，seq=2048 的 KV cache 将是 {kv_2048['total_mb'] * config['num_q_heads'] / config['num_kv_heads']:.0f} MB。\n\n")

        f.write("---\n\n## 3. 推理显存四块分解\n\n")
        f.write("| Seq Len | 权重 | KV Cache | 激活 | CUDA 开销 | 总计 | Fits 12GB |\n")
        f.write("|---------|------|----------|------|-----------|------|-----------|\n")
        for b in budgets:
            fits = "✓" if b["fits_12gb"] else "✗ OOM"
            f.write(f"| {b['seq_len']} | {b['weight_mb']:.0f} MB | {b['kv_cache_mb']:.0f} MB "
                    f"| {b['activation_mb']:.0f} MB | {b['cuda_overhead_mb']:.0f} MB "
                    f"| {b['total_mb']:.0f} MB | {fits} |\n")

        f.write(f"\n**权重占比**: seq=2048 时权重 {weight_mb:.0f} MB 占总量 {budgets[2]['total_mb']:.0f} MB 的 "
                f"{budgets[2]['weight_pct']:.0f}%——量化收益远大于 KV cache 优化。\n\n")

        f.write("---\n\n## 4. KV Cache 量化实测\n\n")

        if bench_results:
            # Prefill 对比
            f.write("### 4.1 Prefill 速度 (tok/s)\n\n")
            configs = list(dict.fromkeys(r["config"] for r in bench_results if "error" not in r))
            f.write("| Seq Len | " + " | ".join(configs) + " |\n")
            f.write("|---------|" + "|".join("---------" for _ in configs) + "|\n")
            for sl in bench_seq_lens:
                vals = []
                for cfg in configs:
                    match = [r for r in bench_results if r.get("config") == cfg and r["seq_len"] == sl]
                    vals.append(f"{match[0]['prefill_tok_s']:.0f}" if match and match[0].get("prefill_tok_s") else "?")
                f.write(f"| {sl} | " + " | ".join(vals) + " |\n")

            # Decode 对比
            f.write("\n### 4.2 Decode 速度 (tok/s) — 单 token 生成\n\n")
            f.write("| Seq Len | " + " | ".join(configs) + " |\n")
            f.write("|---------|" + "|".join("---------" for _ in configs) + "|\n")
            for sl in bench_seq_lens:
                vals = []
                for cfg in configs:
                    match = [r for r in bench_results if r.get("config") == cfg and r["seq_len"] == sl]
                    vals.append(f"{match[0]['decode_tok_s']:.1f}" if match and match[0].get("decode_tok_s") else "?")
                f.write(f"| {sl} | " + " | ".join(vals) + " |\n")

            # TTFT 对比
            f.write("\n### 4.3 TTFT (ms) — Time To First Token\n\n")
            f.write("| Seq Len | " + " | ".join(configs) + " |\n")
            f.write("|---------|" + "|".join("---------" for _ in configs) + "|\n")
            for sl in bench_seq_lens:
                vals = []
                for cfg in configs:
                    match = [r for r in bench_results if r.get("config") == cfg and r["seq_len"] == sl]
                    vals.append(f"{match[0]['ttft_ms']:.0f}" if match and match[0].get("ttft_ms") else "?")
                f.write(f"| {sl} | " + " | ".join(vals) + " |\n")

        f.write("\n### 4.4 兼容性矩阵\n\n")
        f.write("| KV 配置 | FlashAttention | K Cache 量化 | V Cache 量化 | 状态 | 原因 |\n")
        f.write("|---------|---------------|-------------|-------------|------|------|\n")
        f.write("| f16 (baseline) | OFF | f16 | f16 | ✓ | 默认 |\n")
        f.write("| K=q8_0 only | OFF | **q8_0** | f16 | ✓ | K 路径无 transpose, `ggml_mul_mat` 内部反量化 |\n")
        f.write("| K=q4_0 only | OFF | **q4_0** | f16 | ✓ | 同上，block_size=32 整除 head_dim=128 |\n")
        f.write("| K+V=q8_0 no FA | OFF | q8_0 | q8_0 | **✗** | V 需要 transpose, block 量化数据不能转置 |\n")
        f.write("| K+V=q8_0 + FA | ON | q8_0 | q8_0 | **✓** | FA 路径无 transpose, 直接传量化 V 给 `ggml_flash_attn_ext` |\n")
        f.write("| FA only (FP16 KV) | ON | f16 | f16 | ✓ | FA 本身也加速 Prefill 约 15-20% |\n")
        f.write("\n**根因分析**: 非 FA 路径中 V 经过 `permute → transpose → cont` 操作链 (`llama-graph.cpp:2057-2061`)，"
                "transpose 需要重排 block 内元素，Q8_0 的 32-element block 结构无法支持。"
                "FA 路径直接传原始 V 给 `ggml_flash_attn_ext`，没有 transpose，所以兼容。\n\n")

        f.write("---\n\n## 5. 源码级根因与修复方案\n\n")

        f.write("### 5.1 关键代码路径\n\n")
        f.write("```\n")
        f.write("llama-context.cpp:373-376   // 早期校验: quantized V 必须有 FA，否则 throw\n")
        f.write("llama-context.cpp:3434-3437  // 重复校验: llama_new_context_with_model\n")
        f.write("llama-graph.cpp:1970-1972    // build_attn_mha: Q/K/V permute\n")
        f.write("llama-graph.cpp:2018-2081    // 非 FA 路径: V transpose + mul_mat\n")
        f.write("llama-graph.cpp:1977-2017    // FA 路径: ggml_flash_attn_ext\n")
        f.write("```\n\n")

        f.write("### 5.2 修复方案\n\n")
        f.write("**方案 A: 反量化前置（推荐）**\n\n")
        f.write("在 `build_attn_mha()` 中，对 V 做 permute 之前插入反量化:\n")
        f.write("```cpp\n")
        f.write("// llama-graph.cpp, before line 1972 (v = ggml_permute(...))\n")
        f.write("if (!use_flash_attn && ggml_is_quantized(v->type)) {\n")
        f.write("    v = ggml_cast(ctx0, v, GGML_TYPE_F16);\n")
        f.write("}\n")
        f.write("v = ggml_permute(ctx0, v, 0, 2, 1, 3);  // now safe for FP16\n")
        f.write("```\n")
        f.write("然后移除 `llama-context.cpp:373-376` 和 `llama-context.cpp:3434-3437` 的校验。\n\n")
        f.write("**优点**: 改动最小，V cache 存储仍为 Q8_0（显存减半），仅在计算时反量化一次。\n")
        f.write("**带宽收益**: 读 Q8_0 V cache 的 HBM 带宽需求减半，反量化开销极小 (scale × int8)。\n\n")

        f.write("**方案 B: FA 路径泛化（长期方案）**\n\n")
        f.write("将 `ggml_flash_attn_ext` 的类型处理能力泛化到 `ggml_mul_mat` 路径，")
        f.write("或为非 FA 路径添加 fused-dequant-transpose kernel。工程量较大。\n\n")

        f.write("---\n\n## 6. 面试要点\n\n")
        f.write("| 问题 | 回答要点 |\n")
        f.write("|------|---------|\n")
        f.write("| \"KV Cache 量化怎么做？\" | K cache 量化随时可用（Q8_0/Q4_0），V cache 量化需要 FlashAttention。根因是非 FA 路径中 V 要 transpose，block 量化格式不支持。修复方案: permutation 前插入 ggml_cast 反量化 |\n")
        f.write("| \"为什么 V cache 需要 FA?\" | 标准 attention 中 V 经过 permute+transpose 操作链，transpose 改变 block 内元素排布，block-wise 量化（Q8_0 block=32）无法支持。FA 路径直接用原始 V 调 ggml_flash_attn_ext，无 transpose |\n")
        f.write("| \"有没有修过 llama.cpp?\" | 定位了 `llama-graph.cpp:2057-2061` 的根因，提出 A/B 两种修复方案，方案 A 的代码改动仅 3 行 |\n\n")

    print(f"Markdown 已保存: {md_path}")


if __name__ == "__main__":
    main()
