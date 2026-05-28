#!/usr/bin/env python3
"""Generate all benchmark visualization figures for the Qwen2.5-7B quantization project.

Data sources (output/):
  - benchmark_results.json: PPL + speed comparison across 6 GGUF quant types
  - kv_cache_analysis.json: KV cache size, prefill/decode/TTFT by seq_len & config
  - prefill_analysis.json: FLOPs analysis and compute-bound verification
  - ncu_per_kernel.txt: NCU profiling metrics (parsed manually for summary)

Output: output/figures/*.png
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Global style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f8f8",
    "axes.grid": True,
    "grid.alpha": 0.4,
})

# Color palette
COLORS = {
    "FP16": "#e74c3c",
    "Q8_0": "#e67e22",
    "Q5_K_M": "#2ecc71",
    "Q4_K_M": "#3498db",
    "Q4_0": "#9b59b6",
    "IQ4_NL": "#1abc9c",
    "f16 (baseline)": "#333333",
    "K=q8_0, no FA": "#e67e22",
    "K+V=q8_0 + FA": "#3498db",
    "K=q4_0, no FA": "#9b59b6",
    "FA only (FP16 KV)": "#2ecc71",
}
QUANT_COLORS = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6", "#1abc9c"]
KV_COLORS = ["#333333", "#e67e22", "#3498db", "#9b59b6", "#2ecc71"]


def load_json(name):
    with open(OUTPUT_DIR / name) as f:
        return json.load(f)


# ── Figure 1: PPL vs Decode Speed Tradeoff ──────────────────────────
def fig_ppl_vs_speed():
    data = load_json("benchmark_results.json")
    comp = [c for c in data["comparison"] if c["decode_tok_s"] is not None]

    models = [c["model"] for c in comp]
    ppl_deltas = [c["ppl_delta"] for c in comp]
    speeds = [c["decode_tok_s"] for c in comp]
    sizes = [c["file_size_gb"] for c in comp]
    colors = [COLORS.get(m, "#888888") for m in models]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    scatter = ax.scatter(ppl_deltas, speeds, c=colors, s=[s * 80 for s in sizes],
                         edgecolors="white", linewidth=0.8, zorder=5, alpha=0.9)

    for m, dx, dy in zip(models, ppl_deltas, speeds):
        offset = 1.2 if m != "Q5_K_M" else -0.5
        ax.annotate(m, (dx, dy), textcoords="offset points", xytext=(8, offset * 6),
                    fontsize=9, fontweight="bold", color=COLORS.get(m))

    ax.set_xlabel("PPL Increase vs FP16 (lower = better)")
    ax.set_ylabel("Decode Speed (tok/s, higher = better)")
    ax.set_title("Qwen2.5-7B GGUF Quantization: Quality vs Speed Tradeoff")
    # Annotate "ideal" direction
    ax.annotate("Better →", xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=9, color="#27ae60",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#eafaf1", alpha=0.8))
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_ppl_vs_speed.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: Model Size + PPL grouped bar ──────────────────────────
def fig_model_size_ppl():
    data = load_json("benchmark_results.json")
    comp = [c for c in data["comparison"] if c["decode_tok_s"] is not None]

    models = [c["model"] for c in comp]
    sizes = [c["file_size_gb"] for c in comp]
    ppls = [c["ppl"] for c in comp]
    colors = [COLORS.get(m, "#888888") for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # Size bars
    bars1 = ax1.bar(models, sizes, color=colors, edgecolor="white", linewidth=0.8)
    ax1.set_title("Model File Size")
    ax1.set_ylabel("GB")
    ax1.set_ylim(0, max(sizes) * 1.2)
    for bar, sz in zip(bars1, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                 f"{sz:.1f}", ha="center", fontsize=8, fontweight="bold")

    # PPL bars
    ppl_base = 7.72  # FP16
    bars2 = ax2.bar(models, ppls, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_title("Perplexity (WikiText-2, lower = better)")
    ax2.axhline(y=ppl_base, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"FP16 baseline ({ppl_base:.2f})")
    ax2.legend(fontsize=8)
    for bar, ppl in zip(bars2, ppls):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.15,
                 f"{ppl:.3f}", ha="center", fontsize=8, fontweight="bold", color="white")

    fig.suptitle("GGUF Quantization: Model Size & Perplexity", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_model_size_ppl.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 3: KV Cache Size vs Sequence Length ──────────────────────
def fig_kv_cache_size():
    data = load_json("kv_cache_analysis.json")
    theory = data["kv_cache_theoretical"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for dtype_key, label, color, ls in [
        ("f16", "FP16 (2B/elem)", "#333333", "-"),
        ("q8_0", "Q8_0 (1B/elem)", "#e67e22", "--"),
        ("q4_0", "Q4_0 (0.5B/elem)", "#3498db", ":"),
    ]:
        entries = theory[dtype_key]
        seq_lens = [e["seq_len"] for e in entries]
        mbs = [e["total_mb"] for e in entries]
        ax.plot(seq_lens, mbs, color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=5, label=label)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("KV Cache Size (MB)")
    ax.set_title("KV Cache Size vs Sequence Length (28 layers, 4 KV heads)")
    ax.legend()
    ax.set_xlim(0, 4200)
    ax.set_ylim(0, None)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_kv_cache_size.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 4: Prefill Speed vs Seq Len by KV Config ─────────────────
def fig_kv_prefill():
    data = load_json("kv_cache_analysis.json")
    results = data["bench_results"]

    # Group by config label
    configs_order = ["f16 (baseline)", "K=q8_0, no FA", "K+V=q8_0 + FA",
                     "K=q4_0, no FA", "FA only (FP16 KV)"]
    config_data = {}
    for r in results:
        cfg = r["config"]
        if cfg not in config_data:
            config_data[cfg] = {"seq_lens": [], "prefill": []}
        config_data[cfg]["seq_lens"].append(r["seq_len"])
        config_data[cfg]["prefill"].append(r["prefill_tok_s"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cfg in enumerate(configs_order):
        if cfg not in config_data:
            continue
        d = config_data[cfg]
        ax.plot(d["seq_lens"], d["prefill"], color=KV_COLORS[i], linewidth=2,
                marker="o", markersize=6, label=cfg)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Prefill Speed (tok/s)")
    ax.set_title("Prefill Speed vs Sequence Length by KV Cache Configuration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_kv_prefill_speed.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 5: Decode Speed vs Seq Len by KV Config ──────────────────
def fig_kv_decode():
    data = load_json("kv_cache_analysis.json")
    results = data["bench_results"]
    configs_order = ["f16 (baseline)", "K=q8_0, no FA", "K+V=q8_0 + FA",
                     "K=q4_0, no FA", "FA only (FP16 KV)"]
    config_data = {}
    for r in results:
        cfg = r["config"]
        if cfg not in config_data:
            config_data[cfg] = {"seq_lens": [], "decode": []}
        config_data[cfg]["seq_lens"].append(r["seq_len"])
        config_data[cfg]["decode"].append(r["decode_tok_s"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cfg in enumerate(configs_order):
        if cfg not in config_data:
            continue
        d = config_data[cfg]
        ax.plot(d["seq_lens"], d["decode"], color=KV_COLORS[i], linewidth=2,
                marker="s", markersize=6, label=cfg)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Decode Speed (tok/s)")
    ax.set_title("Decode Speed vs Sequence Length by KV Cache Configuration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_kv_decode_speed.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 6: TTFT vs Seq Len by KV Config ──────────────────────────
def fig_kv_ttft():
    data = load_json("kv_cache_analysis.json")
    results = data["bench_results"]
    configs_order = ["f16 (baseline)", "K=q8_0, no FA", "K+V=q8_0 + FA",
                     "K=q4_0, no FA", "FA only (FP16 KV)"]
    config_data = {}
    for r in results:
        cfg = r["config"]
        if cfg not in config_data:
            config_data[cfg] = {"seq_lens": [], "ttft": []}
        config_data[cfg]["seq_lens"].append(r["seq_len"])
        config_data[cfg]["ttft"].append(r["ttft_ms"])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, cfg in enumerate(configs_order):
        if cfg not in config_data:
            continue
        d = config_data[cfg]
        ax.plot(d["seq_lens"], d["ttft"], color=KV_COLORS[i], linewidth=2,
                marker="D", markersize=6, label=cfg)

    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Time-To-First-Token vs Sequence Length by KV Cache Configuration")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_kv_ttft.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 7: Memory Budget Breakdown ───────────────────────────────
def fig_memory_budget():
    data = load_json("kv_cache_analysis.json")
    budgets = data.get("memory_budget", [])

    seq_labels = [f"{b['seq_len']}" for b in budgets]
    weights = np.array([b["weight_mb"] for b in budgets])
    kv_caches = np.array([b["kv_cache_mb"] for b in budgets])
    activations = np.array([b["activation_mb"] for b in budgets])
    overheads = np.array([b["cuda_overhead_mb"] for b in budgets])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(seq_labels, weights, color="#3498db", label="Weights (Q4_K_M, 4466 MB)", zorder=3)
    ax.bar(seq_labels, kv_caches, bottom=weights, color="#e67e22", label="KV Cache", zorder=3)
    ax.bar(seq_labels, activations, bottom=weights + kv_caches, color="#2ecc71", label="Activations", zorder=3)
    ax.bar(seq_labels, overheads, bottom=weights + kv_caches + activations, color="#95a5a6", label="CUDA Overhead (~450 MB)", zorder=3)

    ax.axhline(y=12022, color="red", linestyle="--", linewidth=1.5, alpha=0.7, label="12 GB VRAM Limit")
    ax.set_xlabel("Sequence Length")
    ax.set_ylabel("Memory (MB)")
    ax.set_title("GPU Memory Budget Decomposition (Q4_K_M, RTX 3060 12GB)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_memory_budget.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 8: Prefill TTFT + FlashAttention Comparison ────────────
def fig_prefill_compute():
    data = load_json("prefill_analysis.json")
    theory = data["flops_theoretical"]

    # Support both old format (bench_results) and new format (bench_fa_off / bench_fa_on)
    if "bench_fa_off" in data:
        fa_off = data["bench_fa_off"]
        fa_on = data["bench_fa_on"]
    else:
        fa_off = data["bench_results"]
        fa_on = None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: Pie chart of FLOPs breakdown at seq=2048
    colors_ops = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
    labels_ops = ["QKV Proj", "Attention", "FFN", "Output Proj"]
    for t in theory:
        if t["seq_len"] == 2048:
            ops = [t["qkv_proj_gflops"] / 1000, t["attention_gflops"] / 1000,
                   t["ffn_gflops"] / 1000, t["out_proj_gflops"] / 1000]
            wedges, texts, autotexts = ax1.pie(
                ops, labels=labels_ops, colors=colors_ops,
                autopct="%1.1f%%", startangle=90,
                explode=(0.02, 0.1, 0.02, 0.02))
            for at in autotexts:
                at.set_fontsize(8)
            break
    ax1.set_title(f"Prefill FLOPs Breakdown (seq=2048)\nTotal: {theory[4]['total_tflops']:.1f} TFLOPs")

    # Right: TTFT comparison FA OFF vs FA ON
    seq_lens = [b["seq_len"] for b in fa_off]
    ttft_off = [b["ttft_ms"] for b in fa_off]

    ax2.plot(seq_lens, ttft_off, color="#e74c3c", linewidth=2.5,
             marker="o", markersize=8, label="FA OFF (baseline)", zorder=5)

    if fa_on:
        ttft_on = [b["ttft_ms"] for b in fa_on]
        speedups = [b["speedup_vs_baseline"] for b in fa_on]
        ax2.plot(seq_lens, ttft_on, color="#27ae60", linewidth=2.5,
                 marker="s", markersize=8, label="FA ON", zorder=5)
        ax2.fill_between(seq_lens, ttft_on, ttft_off, alpha=0.12, color="#27ae60")

        # Annotate speedup at key points
        for sl, toff, ton, sp in zip(seq_lens, ttft_off, ttft_on, speedups):
            if sp >= 1.1:  # Only annotate >= 10% speedup
                ax2.annotate(f"{sp:.2f}x", (sl, ton), textcoords="offset points",
                            xytext=(0, -18), fontsize=8, fontweight="bold",
                            color="#27ae60", ha="center")

    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("TTFT (ms)")
    ax2.set_title("Prefill TTFT: FlashAttention OFF vs ON")
    ax2.legend(fontsize=9)
    ax2.set_xlim(0, 4300)
    ax2.set_ylim(0, None)

    fig.suptitle("Prefill Phase Analysis: FLOPs Breakdown + FlashAttention Speedup", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_prefill_compute.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 9: Decode Step Time Breakdown (Pie) ──────────────────────
def fig_decode_breakdown():
    # Data from NCU profiling (ncu_profiling.md section 6)
    labels = [
        "FFN MatVec\n(gate+up+down)\n22.4 ms",
        "Attn Output MatVec\n1.06 ms",
        "Attn QK MatVec\n0.52 ms",
        "Quantization\n0.81 ms",
        "RMS Norm\n0.34 ms",
        "RoPE + KV Write\n0.40 ms",
        "FlashAttention\n0.03 ms",
    ]
    times = [22.4, 1.06, 0.52, 0.81, 0.34, 0.40, 0.03]
    colors_pie = ["#e74c3c", "#e67e22", "#f39c12", "#3498db", "#2ecc71", "#9b59b6", "#95a5a6"]
    explode = (0.05, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02)

    fig, ax = plt.subplots(figsize=(7, 6))
    wedges, texts, autotexts = ax.pie(
        times, labels=labels, colors=colors_pie, autopct="%1.1f%%",
        startangle=90, explode=explode, labeldistance=1.15)
    for at in autotexts:
        at.set_fontsize(9)
        at.set_fontweight("bold")
    for t in texts:
        t.set_fontsize(8)

    ax.set_title("Single Decode Step Time Breakdown\n(Q4_K_M, 28 layers, RTX 3060)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_decode_breakdown.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 10: Roofline Model ───────────────────────────────────────
def fig_roofline():
    peak_bw = 360       # GB/s (RTX 3060)
    peak_flops = 12740  # GFLOPS (12.74 TFLOPS)

    # RTX 3060 has L2 cache bandwidth too, but for simplicity use DRAM
    # Ridge point: peak_flops / peak_bw = 35.4 FLOPs/Byte

    # Kernel data points from NCU
    kernels = {
        "FFN matvec (Q4_K,\ngrid=18944)":   {"oi": 31.6, "perf": 0.824 * peak_flops},
        "Attn out (Q4_K,\ngrid=3584)":      {"oi": 3.1,  "perf": 0.581 * peak_flops},
        "Attn QK (Q4_K,\ngrid=512)":        {"oi": 6.1,  "perf": 0.464 * peak_flops},
        "FFN matvec (Q6_K,\ngrid=3584)":    {"oi": 3.1,  "perf": 0.778 * peak_flops},
        "Attn QK (Q6_K,\ngrid=512)":        {"oi": 6.1,  "perf": 0.547 * peak_flops},
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    # Roofline: bandwidth ceiling
    oi_range = np.logspace(-1, 3, 500)
    bw_ceiling = np.minimum(peak_bw * oi_range, peak_flops)
    ax.loglog(oi_range, bw_ceiling, color="#2c3e50", linewidth=2.5, label="RTX 3060 Roofline (360 GB/s, 12.74 TFLOPS)")

    # Ridge line
    ridge_x = peak_flops / peak_bw
    ax.axvline(x=ridge_x, color="#7f8c8d", linestyle=":", linewidth=1, alpha=0.7)
    ax.annotate(f"Ridge\n{ridge_x:.1f} FLOPs/Byte", xy=(ridge_x, 100), fontsize=8, color="#7f8c8d",
                ha="left", va="bottom")

    # Plot kernel points
    kernel_colors = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    for i, (name, k) in enumerate(kernels.items()):
        ax.plot(k["oi"], k["perf"], marker="o", markersize=10, color=kernel_colors[i],
                markeredgecolor="white", markeredgewidth=1, zorder=5)
        ax.annotate(name, (k["oi"], k["perf"]), textcoords="offset points",
                    xytext=(10, 5), fontsize=8, color=kernel_colors[i],
                    arrowprops=dict(arrowstyle="->", color=kernel_colors[i], lw=0.8),
                    fontweight="bold")

    ax.set_xlabel("Operational Intensity (FLOPs/Byte)")
    ax.set_ylabel("Performance (GFLOPS)")
    ax.set_title("Roofline Model: Qwen2.5-7B Q4_K_M Decode Kernels on RTX 3060")
    ax.legend(fontsize=8, loc="lower right")

    # Annotations
    ax.annotate("Memory-Bound\nRegion", xy=(5, 500), fontsize=11, color="#e74c3c",
                ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#fdedec", alpha=0.7))
    ax.annotate("Compute-Bound\nRegion", xy=(100, 8000), fontsize=11, color="#27ae60",
                ha="center", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#eafaf1", alpha=0.7))

    ax.set_xlim(0.5, 200)
    ax.set_ylim(50, 20000)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_roofline.png", bbox_inches="tight")
    plt.close(fig)


# ── Figure 11: KV Cache Speedup Summary (Grouped Bar) ───────────────
def fig_kv_speedup():
    data = load_json("kv_cache_analysis.json")
    results = data["bench_results"]

    # Pick seq=2048 for comparison
    seq_target = 2048
    configs = []
    prefill_speeds = []
    decode_speeds = []
    ttfts = []
    kv_sizes = []

    for r in results:
        if r["seq_len"] == seq_target:
            configs.append(r["config"])
            prefill_speeds.append(r["prefill_tok_s"])
            decode_speeds.append(r["decode_tok_s"])
            ttfts.append(r["ttft_ms"])
            kv_sizes.append(r["kv_size_mb"])

    # Sort consistently
    order = ["f16 (baseline)", "K=q8_0, no FA", "K+V=q8_0 + FA",
             "K=q4_0, no FA", "FA only (FP16 KV)"]
    ordered = sorted(zip(configs, prefill_speeds, decode_speeds, ttfts, kv_sizes),
                     key=lambda x: order.index(x[0]) if x[0] in order else 99)
    configs = [o[0] for o in ordered]
    prefill_speeds = [o[1] for o in ordered]
    decode_speeds = [o[2] for o in ordered]
    ttfts = [o[3] for o in ordered]
    kv_sizes = [o[4] for o in ordered]

    short_labels = ["f16\nbaseline", "K=q8_0\nno FA", "K+V=q8_0\n+ FA", "K=q4_0\nno FA", "FA only\n(FP16 KV)"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    c = KV_COLORS[:len(configs)]

    # Prefill speed
    ax = axes[0, 0]
    bars = ax.bar(short_labels, prefill_speeds, color=c, edgecolor="white")
    ax.set_title(f"Prefill Speed @ seq={seq_target}")
    ax.set_ylabel("tok/s")
    for bar, v in zip(bars, prefill_speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                str(v), ha="center", fontsize=8, fontweight="bold")

    # Decode speed
    ax = axes[0, 1]
    bars = ax.bar(short_labels, decode_speeds, color=c, edgecolor="white")
    ax.set_title(f"Decode Speed @ seq={seq_target}")
    ax.set_ylabel("tok/s")
    for bar, v in zip(bars, decode_speeds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", fontsize=8, fontweight="bold")

    # TTFT
    ax = axes[1, 0]
    bars = ax.bar(short_labels, ttfts, color=c, edgecolor="white")
    ax.set_title(f"TTFT @ seq={seq_target}")
    ax.set_ylabel("ms")
    for bar, v in zip(bars, ttfts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(v), ha="center", fontsize=8, fontweight="bold")

    # KV Cache size
    ax = axes[1, 1]
    bars = ax.bar(short_labels, kv_sizes, color=c, edgecolor="white")
    ax.set_title(f"KV Cache Size @ seq={seq_target}")
    ax.set_ylabel("MB")
    for bar, v in zip(bars, kv_sizes):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{v:.0f}", ha="center", fontsize=8, fontweight="bold")

    fig.suptitle("KV Cache Configuration Comparison at Sequence Length 2048", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_kv_cache_speedup.png", bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print("Generating figures...")
    fig_ppl_vs_speed()
    print("  [1/11] PPL vs Speed tradeoff")
    fig_model_size_ppl()
    print("  [2/11] Model size & PPL")
    fig_kv_cache_size()
    print("  [3/11] KV cache size vs seq_len")
    fig_kv_prefill()
    print("  [4/11] Prefill speed by KV config")
    fig_kv_decode()
    print("  [5/11] Decode speed by KV config")
    fig_kv_ttft()
    print("  [6/11] TTFT by KV config")
    fig_memory_budget()
    print("  [7/11] Memory budget breakdown")
    fig_prefill_compute()
    print("  [8/11] Prefill compute analysis")
    fig_decode_breakdown()
    print("  [9/11] Decode time breakdown")
    fig_roofline()
    print("  [10/11] Roofline model")
    fig_kv_speedup()
    print("  [11/11] KV cache speedup summary")

    print(f"\nAll figures saved to {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.png")):
        size_kb = os.path.getsize(f) / 1024
        print(f"  {f.name} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
