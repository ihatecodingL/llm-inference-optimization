# Qwen2.5-7B 量化推理优化

Qwen2.5-7B-Instruct 的 FP16 权重要 14.9 GB，RTX 3060 只有 12 GB 显存。本项目通过 GGUF 量化将模型压缩至 4-8 GB，在可控精度损失下实现消费级 GPU 推理，并通过 NCU profiling 定位 decode 阶段性能瓶颈。

## 环境

| 组件 | 规格 |
|------|------|
| GPU | NVIDIA GeForce RTX 3060 12GB (GA106) |
| Compute Capability | 8.6 (Ampere), 28 SMs |
| 显存带宽 | 360 GB/s |
| FP16 峰值算力 | 12.74 TFLOPS |
| CUDA | 12.2 |
| PyTorch | 2.5.1 |
| llama.cpp | CUDA backend (commit 35a74c8fb) |
| 模型 | Qwen2.5-7B-Instruct (7.6B params, 28 layers, GQA 7:1) |

## 量化方案对比

FP16 14.9 GB > 12 GB 显存，加载即 OOM。6 种 GGUF 量化方案在 WikiText-2 上的完整对比：

| 方案 | 文件大小 | PPL (↓) | Δ PPL | Prefill | Decode (↑) | ms/tok |
|------|----------|---------|-------|---------|------------|--------|
| FP16 | 14.2 GB | 7.7206 | 基准 | — | — | — |
| Q8_0 | 7.5 GB | 7.7256 | +0.005 | 2124 tok/s | 39.3 tok/s | 25.4 ms |
| Q5_K_M | 5.1 GB | 7.7744 | +0.054 | 1978 tok/s | 54.8 tok/s | 18.2 ms |
| **Q4_K_M** | **4.4 GB** | **7.8435** | **+0.123** | **1995 tok/s** | **61.0 tok/s** | **16.4 ms** |
| Q4_0 | 4.1 GB | 7.9947 | +0.274 | 2193 tok/s | 67.0 tok/s | 14.9 ms |
| IQ4_NL | 4.2 GB | 7.9399 | +0.219 | 2178 tok/s | 65.9 tok/s | 15.2 ms |

> FP16 无法在 GPU 上加载。PPL 使用 CPU-only (`-ngl 0`) 测得作为质量基线。

**推荐: Q4_K_M** — 体积压缩 69%，PPL 仅损失 0.12（相对 1.6%），decode 速度从不可用到 61 tok/s。

![PPL vs Speed](output/figures/01_ppl_vs_speed.png)

## 关键发现

### 1. Decode 是 Memory-Bound（NCU Profiling 实测）

NCU per-kernel profiling 定位到 FFN 的 `mul_mat_vec_q` kernel 占单次 decode 的 **88%**（22.4 ms）：

| Kernel | Grid | Duration | DRAM 利用率 |
|--------|------|----------|------------|
| FFN matvec (gate+up+down) | 18944 | 266 µs × 84 | 82.4% |
| Attn output matvec | 3584 | 38 µs × 28 | 58.1% |
| Attn QK matvec | 512 | 19 µs × 28 | 46.4% |

Roofline 分析：OI = 31.6 FLOPs/Byte < Ridge 35.4，确认 memory-bound。DRAM 利用率 82.4%，已接近 360 GB/s 理论带宽上限。

![Decode Breakdown](output/figures/09_decode_breakdown.png)
![Roofline](output/figures/10_roofline.png)

### 2. KV Cache 量化兼容性

K cache (Q8_0/Q4_0) 可直接使用。V cache 量化需要 FlashAttention：

| 配置 | K Cache | V Cache | FA | 状态 |
|------|---------|---------|----|------|
| f16 baseline | f16 | f16 | OFF | ✓ |
| K=q8_0 only | **q8_0** | f16 | OFF | ✓ |
| K=q4_0 only | **q4_0** | f16 | OFF | ✓ |
| K+V=q8_0 no FA | q8_0 | q8_0 | OFF | **✗** |
| K+V=q8_0 + FA | q8_0 | q8_0 | **ON** | ✓ |

**根因**: 非 FA 路径中 V 经过 `permute → transpose → cont` (`llama-graph.cpp:2057-2061`)，transpose 需重排 block 内元素，block-wise 量化（Q8_0 block=32）无法支持。FA 路径直接传原始 V 给 `ggml_flash_attn_ext`，无 transpose。

修复方案 A：在 permute 前插入 `ggml_cast(v, F16)` 反量化（3 行改动），V cache 存储仍为 Q8_0，仅在计算时反量化一次。

![KV Cache Speedup](output/figures/11_kv_cache_speedup.png)

### 3. Prefill 是 Compute-Bound + FlashAttention 实测

实测 seq_len 从 128 增大到 4096（32 倍），TTFT 从 90 ms 增大到 3493 ms（39 倍），近乎线性——证实 compute-bound。

FlashAttention 同 session 对照测试：

| Seq Len | FA OFF TTFT | FA ON TTFT | 加速比 |
|---------|------------|------------|--------|
| 128 | 90 ms | 89 ms | 1.02x |
| 512 | 347 ms | 321 ms | 1.08x |
| 1024 | 727 ms | 639 ms | 1.14x |
| 2048 | 1,804 ms | **1,314 ms** | **1.37x** |
| 4096 | 3,493 ms | **2,699 ms** | **1.29x** |

FA 在 seq≥1024 时开始有效，2048 时加速最大（37%），长 prompt 场景必开。

Decode 速度在所有 seq_len 下恒定 ~62 tok/s——完美证明 memory-bound（只关心权重大小，不关心 seq_len）。

![Prefill FA Comparison](output/figures/08_prefill_compute.png)

## 项目结构

```
llm_infer_optim/
├── README.md
├── configs/
│   └── quant_configs.json              # 量化方案配置与参数
├── scripts/
│   ├── 01_baseline_test.py             # FP16 OOM 验证 + NF4 baseline
│   ├── 02_convert_gguf.py              # HF → GGUF 6 种精度转换
│   ├── 03_quant_benchmark.py           # PPL + 推理速度全对比
│   ├── 04_kv_cache_analysis.py         # KV Cache 显存分析 + 量化实测
│   └── 05_prefill_analysis.py          # Prefill TTFT 曲线 + compute-bound 验证
├── plot_results.py                     # 11 张可视化图表
├── docs/
│   ├── quantization_theory.md          # Q4_0 / K-Quants / IQ4_NL 原理笔记
│   ├── prefill_engineering.md          # Prefill 工程全面指南
│   └── kernel_fusion_analysis.md       # Kernel Fusion 分析：14 种已有融合 + 未实现原因
└── output/
    ├── benchmark_results.{json,md}      # 量化对比数据与报告
    ├── kv_cache_analysis.{json,md}      # KV Cache 分析数据与报告
    ├── prefill_analysis.{json,md}       # Prefill 分析数据与报告
    ├── ncu_profiling.md                # NCU Decode Profiling 完整分析
    └── figures/                        # 11 张可视化图表
```

## 复现

```bash
# 1. 编译 llama.cpp (CUDA backend)
cd /path/to/llama.cpp
mkdir -p build && cd build
cmake -DGGML_CUDA=ON .. && make -j

# 2. 下载模型
python scripts/download_model.py

# 3. 转换 + 量化
python scripts/02_convert_gguf.py

# 4. Benchmark
python scripts/03_quant_benchmark.py

# 5. KV Cache 分析
python scripts/04_kv_cache_analysis.py

# 6. Prefill 分析
python scripts/05_prefill_analysis.py

# 7. 生成图表
python plot_results.py
```

## 参考资料

- [GGUF 格式](https://github.com/ggerganov/ggml)
- [K-Quants PR](https://github.com/ggerganov/llama.cpp/pull/1684)
- [llama.cpp 源码](https://github.com/ggerganov/llama.cpp)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
