#!/usr/bin/env python3
"""Prefill 深度分析：延迟曲线、compute-bound 验证、FlashAttention 影响。

Prefill 是什么：
  用户输入 prompt 后，模型一次性并行处理所有 token，生成第一个输出 token。
  这个阶段叫 Prefill（也叫 prompt processing / encoding）。

为什么重要：
  - 用户体验：Prefill 决定了"输入后多久开始出字"
  - 长 prompt 场景：输入 4096 token 的文档，Prefill 延迟可能占主导
  - 面试必问：Prefill vs Decode 的区别和优化策略

核心知识：
  Prefill = MatMul (矩阵×矩阵) = 计算量大 = compute-bound
  Decode  = MatVec (矩阵×向量) = 访存量大 = memory-bound

用法:
    conda run -n pytorch-env python scripts/05_prefill_analysis.py
    conda run -n pytorch-env python scripts/05_prefill_analysis.py --model Q4_K_M

产出:
    output/prefill_analysis.json
    output/prefill_analysis.md
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

# Qwen2.5-7B config
CONFIG = {
    "num_layers": 28,
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_q_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "vocab_size": 152064,
}


def parse_args():
    p = argparse.ArgumentParser(description="Prefill 深度分析")
    p.add_argument("--model", default="Q4_K_M", help="测试模型")
    p.add_argument("--seq-lens", default="128,256,512,1024,2048,4096",
                   help="prompt 长度序列")
    p.add_argument("--n-gpu-layers", type=int, default=99)
    p.add_argument("--flash-attn", action="store_true", help="启用 FlashAttention")
    return p.parse_args()


def estimate_prefill_flops(seq_len: int) -> dict:
    """估算 Prefill 阶段的理论 FLOPs。

    Transformer 一层的主要计算:
    1. QKV 投影:   3 × hidden × hidden × seq_len
    2. Attention:   seq_len × seq_len × head_dim × num_heads (QK^T)
                    + seq_len × seq_len × head_dim × num_heads (×V)
    3. Output 投影: hidden × hidden × seq_len
    4. FFN:         2 × hidden × intermediate × seq_len (gate + up)
                    + intermediate × hidden × seq_len (down)

    总计算量 ≈ 2 × (忽略 Attention 的常数因子，对短 seq 近似)
    实际简化公式: FLOPs ≈ 2 × num_params × seq_len (每个 token 约 2× 参数量 FLOPs)
    """
    h = CONFIG["hidden_size"]
    inter = CONFIG["intermediate_size"]
    n = CONFIG["num_layers"]
    n_q = CONFIG["num_q_heads"]
    hd = CONFIG["head_dim"]

    # QKV projection: 3 × h × h
    qkv_macs = 3 * h * h * seq_len
    # Attention: QK^T + softmax + ×V
    attn_macs = 2 * seq_len * seq_len * hd * n_q
    # Output projection: h × h
    out_macs = h * h * seq_len
    # FFN: gate + up + down = 3 × h × inter
    ffn_macs = 3 * h * inter * seq_len

    per_layer_macs = qkv_macs + attn_macs + out_macs + ffn_macs
    total_macs = per_layer_macs * n
    # 1 MAC ≈ 2 FLOPs (multiply + accumulate)
    total_flops = total_macs * 2

    qkv_flops = qkv_macs * 2
    attn_flops = attn_macs * 2
    ffn_flops = ffn_macs * 2
    out_flops = out_macs * 2

    return {
        "seq_len": seq_len,
        "total_gflops": round(total_flops / 1e9, 2),
        "total_tflops": round(total_flops / 1e12, 3),
        "qkv_proj_gflops": round(qkv_flops * n / 1e9, 2),
        "attention_gflops": round(attn_flops * n / 1e9, 2),
        "ffn_gflops": round(ffn_flops * n / 1e9, 2),
        "out_proj_gflops": round(out_flops * n / 1e9, 2),
    }


def run_prefill_bench(model_path: str, seq_lens: list[int],
                      n_gpu_layers: int, flash_attn: bool) -> list[dict]:
    """跑多组 seq_len 的 benchmark，提取 Prefill 数据。"""
    results = []
    fa_flag = ["-fa", "1"] if flash_attn else []

    for sl in seq_lens:
        print(f"  seq_len={sl}...", end=" ", flush=True)
        cmd = [
            BENCH_BIN, "-m", model_path,
            "-ngl", str(n_gpu_layers),
            "-p", str(sl), "-n", "1",  # n=1 只测 prefill
            "-b", "2048", "-r", "3",
            "-o", "json",
        ] + fa_flag

        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        elapsed = time.time() - t0

        if r.returncode != 0:
            print(f"FAILED")
            results.append({"seq_len": sl, "error": r.stderr[-200:]})
            continue

        try:
            entries = json.loads(r.stdout)
        except json.JSONDecodeError:
            print(f"JSON parse failed")
            results.append({"seq_len": sl, "error": "json parse"})
            continue

        # 找 prefill entry: n_prompt > 0 且 n_gen == 0
        prefill_entries = [e for e in entries if e.get("n_prompt", 0) > 0 and e.get("n_gen", 0) == 0]
        if not prefill_entries:
            # fallback: any entry with n_prompt > 0
            prefill_entries = [e for e in entries if e.get("n_prompt", 0) > 0]

        if not prefill_entries:
            print(f"no prefill data")
            results.append({"seq_len": sl, "error": "no prefill entry"})
            continue

        e = prefill_entries[0]
        avg_ts = e.get("avg_ts", 0)  # tokens per second
        avg_ns = e.get("avg_ns", 0)  # nanoseconds
        stddev_ts = e.get("stddev_ts", 0)
        latency_ms = (avg_ns / 1e6) if avg_ns else (sl / avg_ts * 1000 if avg_ts else 0)
        # Time To First Token
        ttft_ms = latency_ms

        # 计算实际 TFLOPS
        flops_est = estimate_prefill_flops(sl)
        actual_tflops = flops_est["total_tflops"] / (latency_ms / 1000) if latency_ms > 0 else 0

        print(f"{avg_ts:.0f} tok/s, TTFT={ttft_ms:.0f}ms, {actual_tflops:.2f} TFLOPS")

        results.append({
            "seq_len": sl,
            "prefill_tok_s": round(avg_ts, 1),
            "prefill_stddev": round(stddev_ts, 1),
            "latency_ms": round(latency_ms, 1),
            "ttft_ms": round(ttft_ms, 1),
            "estimated_gflops": flops_est["total_gflops"],
            "actual_tflops": round(actual_tflops, 2),
            "bench_time_s": round(elapsed, 1),
        })

    return results


def main():
    args = parse_args()
    seq_lens = [int(x) for x in args.seq_lens.split(",")]

    model_name = args.model
    model_path = os.path.join(GGUF_DIR, f"qwen2.5-7b-{model_name}.gguf")
    if not os.path.isfile(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fa_label = "FA ON" if args.flash_attn else "FA OFF"

    print("=" * 60)
    print(f"Prefill 延迟分析 ({model_name}, {fa_label})")
    print("=" * 60)

    # 1. 理论 FLOPs
    print("\n1. Prefill 理论计算量 (FLOPs)")
    print("-" * 40)
    print(f"  {'Seq':<6} {'Total':>10} {'QKV':>10} {'Attn':>10} {'FFN':>10} {'Out':>10}")
    for sl in seq_lens:
        f = estimate_prefill_flops(sl)
        print(f"  {sl:<6} {f['total_gflops']:>8.0f} G {f['qkv_proj_gflops']:>8.0f} G "
              f"{f['attention_gflops']:>8.0f} G {f['ffn_gflops']:>8.0f} G {f['out_proj_gflops']:>8.0f} G")
    print()

    # 2. 实测
    print("2. 实测 Prefill 延迟")
    print("-" * 40)
    results = run_prefill_bench(model_path, seq_lens, args.n_gpu_layers, args.flash_attn)

    # 3. 对比分析
    print(f"\n3. 对比分析 ({fa_label})")
    print("-" * 80)
    header = f"  {'Seq':<6} {'Prefill':>10} {'TTFT':>10} {'Latency':>10} {'实际 TFLOPS':>12} {'理论峰值%':>10}"
    print(header)
    print("  " + "-" * 76)

    rtx3060_peak_tflops = 12.74  # FP16 theoretical peak for RTX 3060

    for r in results:
        if "error" in r:
            print(f"  {r['seq_len']:<6} ERROR: {r['error'][:60]}")
            continue
        peak_pct = r['actual_tflops'] / rtx3060_peak_tflops * 100 if r['actual_tflops'] else 0
        print(f"  {r['seq_len']:<6} {r['prefill_tok_s']:>8.0f} tok/s "
              f"{r['ttft_ms']:>8.0f} ms {r['latency_ms']:>8.0f} ms "
              f"{r['actual_tflops']:>10.2f} {peak_pct:>9.1f}%")

    print(f"\n  RTX 3060 FP16 理论峰值: {rtx3060_peak_tflops} TFLOPS")
    print(f"  (实际利用率受限于内存带宽、kernel 效率等因素)\n")

    # 4. 关键结论
    print("4. 关键结论")
    print("-" * 40)
    # 从 results 中提取有效的 seq_len->latency 映射
    valid = [r for r in results if "error" not in r]
    if len(valid) >= 2:
        # 计算 latency 增长是否接近线性
        smallest = valid[0]
        largest = valid[-1]
        seq_ratio = largest["seq_len"] / smallest["seq_len"]
        lat_ratio = largest["latency_ms"] / smallest["latency_ms"]
        print(f"  seq_len {smallest['seq_len']}→{largest['seq_len']} ({seq_ratio:.0f}x):")
        print(f"  TTFT     {smallest['ttft_ms']:.0f}→{largest['ttft_ms']:.0f} ms ({lat_ratio:.1f}x)")
        print(f"  Prefill 是 compute-bound: latency 与 seq_len 成 {'近乎线性' if abs(seq_ratio - lat_ratio) < seq_ratio * 0.3 else '超线性'}关系")

        # Attention 占比
        for r in valid:
            flops = estimate_prefill_flops(r["seq_len"])
            attn_pct = flops["attention_gflops"] / flops["total_gflops"] * 100
            if attn_pct > 20:
                print(f"  seq={r['seq_len']}: Attention 占 {attn_pct:.0f}% 计算量 — FlashAttention 收益大")
                break

    print()

    # 保存
    output = {
        "model": model_name,
        "flash_attn": args.flash_attn,
        "config": CONFIG,
        "rtx3060_peak_tflops": rtx3060_peak_tflops,
        "flops_theoretical": [estimate_prefill_flops(sl) for sl in seq_lens],
        "bench_results": results,
    }

    json_path = os.path.join(OUTPUT_DIR, "prefill_analysis.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Markdown
    md_path = os.path.join(OUTPUT_DIR, "prefill_analysis.md")
    with open(md_path, "w") as f:
        fa_str = "ON" if args.flash_attn else "OFF"
        f.write(f"# Prefill 深度分析\n\n")
        f.write(f"模型: {model_name}, FlashAttention: {fa_str}\n\n")

        f.write("## 什么是 Prefill\n\n")
        f.write("Prefill（也叫 Prompt Processing / Encoding）是 LLM 推理的第一阶段：\n\n")
        f.write("1. 用户输入 prompt（比如 2048 token 的文档）\n")
        f.write("2. 模型一次性并行处理所有 token\n")
        f.write("3. 生成第一个输出 token\n\n")
        f.write("这个阶段之后进入 Decode（逐 token 生成），直到遇到 EOS。\n\n")

        f.write("**Prefill 是 compute-bound**：计算量大（MatMul），GPU 算力跑满。\n")
        f.write("**Decode 是 memory-bound**：每次只处理 1 token 的 MatVec，瓶颈在显存带宽。\n\n")

        f.write("## 理论计算量\n\n")
        f.write("Qwen2.5-7B 单层 Prefill 的 FLOPs 分解：\n\n")
        f.write("| Seq | Total | QKV Proj | Attention | FFN | Output Proj |\n")
        f.write("|-----|-------|----------|-----------|-----|-------------|\n")
        for sl in seq_lens:
            fl = estimate_prefill_flops(sl)
            f.write(f"| {sl} | {fl['total_gflops']:.0f} G | {fl['qkv_proj_gflops']:.0f} G | "
                    f"{fl['attention_gflops']:.0f} G | {fl['ffn_gflops']:.0f} G | "
                    f"{fl['out_proj_gflops']:.0f} G |\n")

        f.write("\n**注意**：随着 seq_len 增长，Attention 的计算量以 **O(N²)** 增长，")
        f.write("而其他部分只以 O(N) 增长。长 prompt 下 Attention 会成为 Prefill 的绝对瓶颈。\n")
        f.write("FlashAttention 可以将 Attention 的显存和计算都优化到近似 O(N)。\n\n")

        f.write("## 实测 Prefill 延迟曲线\n\n")
        f.write(f"| Seq | Prefill | TTFT | 实际 TFLOPS | 理论峰值占比 |\n")
        f.write(f"|-----|---------|------|-------------|-------------|\n")
        for r in results:
            if "error" in r:
                f.write(f"| {r['seq_len']} | ERROR | - | - | - |\n")
            else:
                peak_pct = r['actual_tflops'] / rtx3060_peak_tflops * 100 if r['actual_tflops'] else 0
                f.write(f"| {r['seq_len']} | {r['prefill_tok_s']:.0f} tok/s | "
                        f"{r['ttft_ms']:.0f} ms | {r['actual_tflops']:.2f} | {peak_pct:.1f}% |\n")

        f.write(f"\nRTX 3060 FP16 理论峰值: {rtx3060_peak_tflops} TFLOPS\n\n")

        f.write("## 面试要点\n\n")

        f.write("### Prefill vs Decode 一句话区分\n\n")
        f.write("```\n")
        f.write("Prefill:  输入 2048 tokens → 并行计算 → 1 个 token 输出\n")
        f.write("          MatMul (矩阵×矩阵), compute-bound, GPU 算力瓶颈\n")
        f.write("          优化方向: FlashAttention, 更大 batch, kernel fusion\n\n")
        f.write("Decode:   每次输入 1 token → MatVec (矩阵×向量) → 1 token 输出\n")
        f.write("          memory-bound, 显存带宽瓶颈\n")
        f.write("          优化方向: 量化 (减少权重体积), KV cache 优化\n")
        f.write("```\n\n")

        f.write("### 关键指标: TTFT (Time To First Token)\n\n")
        f.write('- TTFT = Prefill 延迟，用户感知的「输入后多久开始出字」\n')
        f.write("- 短 prompt (<512): TTFT 通常 < 300ms，用户几乎无感\n")
        f.write("- 长 prompt (>4096): TTFT 可能达到秒级，需要优化\n\n")

        f.write("### Prefill 优化策略\n\n")
        f.write("| 策略 | 原理 | 收益 |\n")
        f.write("|------|------|------|\n")
        f.write("| FlashAttention | 分块计算 attention，减少 HBM 读写 | 2-4x 加速，省显存 |\n")
        f.write("| Prefill Chunking | 长 prompt 拆成多个 chunk 流水线处理 | 降低峰值显存 |\n")
        f.write("| Prefix Caching | 相同前缀的 prompt 复用 KV cache | 减少重复计算 |\n")
        f.write("| Continuous Batching | 多个请求的 prefill/decode 混合调度 | 提高 GPU 利用率 |\n")
        f.write("| Kernel Fusion | QKV 投影合并为一个 kernel | 减少 kernel launch 开销 |\n")
        f.write("| 量化权重 (W4A16) | 减少权重读取带宽 | Prefill 也有收益（但不如 Decode 大）|\n")

    print(f"JSON 已保存: {json_path}")
    print(f"Markdown 已保存: {md_path}")


if __name__ == "__main__":
    main()
