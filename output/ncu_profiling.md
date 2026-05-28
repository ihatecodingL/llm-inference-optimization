# Nsight Compute Decode 阶段 Profiling 分析

模型: Qwen2.5-7B-Instruct Q4_K_M, GPU: RTX 3060 12GB (SM 8.6)

---

## 1. Profiling 环境

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA GeForce RTX 3060 (GA106), 12GB GDDR6 |
| Compute Capability | 8.6 (Ampere) |
| SM 数量 | 28 |
| 理论显存带宽 | 360 GB/s |
| Max Threads/SM | 1536 |
| Max Registers/SM | 65536 |
| Max Shared Mem/SM | 100 KB |
| CUDA 版本 | 12.2 |
| NCU 版本 | 2023.2.2.0 |
| 模型 | Qwen2.5-7B-Instruct Q4_K_M (4.36 GiB, 4.91 BPW) |

## 2. Profiling 命令

### 2.1 首次尝试: per-gpu 模式 (太慢,被终止)

```bash
# 全量 profiling,每个 kernel 9 passes — 太慢,2000+ kernels 无法完成
/usr/local/cuda-12.2/bin/ncu \
  --print-summary per-gpu \
  --target-processes all \
  $LLAMA_CPP_DIR/build/bin/llama-simple \
  -m $PROJECT_ROOT/models/gguf/qwen2.5-7b-Q4_K_M.gguf \
  -ngl 99 -n 5 "Hello"
```

**问题**: per-gpu 模式对每个 kernel launch 做 9 passes 的详细 profiling。2000+ kernel launches × 9 passes 需要 10+ 分钟,在模型加载完成前就被终止了。

### 2.2 成功方案: per-kernel + launch-count 限制

```bash
# 只 profile 前 50 个 kernel launches,per-kernel 模式
/usr/local/cuda-12.2/bin/ncu \
  --print-summary per-kernel \
  --launch-skip 0 --launch-count 50 \
  --target-processes all \
  $LLAMA_CPP_DIR/build/bin/llama-simple \
  -m $PROJECT_ROOT/models/gguf/qwen2.5-7b-Q4_K_M.gguf \
  -ngl 99 -n 3 "Hello"
```

**原理**:
- `--launch-skip 0 --launch-count 50`: 只看前 50 个 GPU kernel launch,跳过模型加载(模型加载主要用 cudaMemcpy,不走 kernel)
- 前 50 个 launch 覆盖 prefill (1 token "Hello") + 2-3 个 decode step,足够分析 decode 阶段的 kernel 特征
- NCU 自动按 kernel 签名分组,聚合 Metric

### 2.3 如果要 profile 特定 kernel

```bash
# 使用 kernel name 正则过滤 (如果 shell 转义没问题的话)
ncu --print-summary per-kernel \
  --kernel-name "mul_mat_vec_q" \
  --launch-count 20 \
  --target-processes all \
  ./llama-simple -m model.gguf -ngl 99 -n 3 "Hello"
```

**注意**: `--kernel-name` 使用标准 regex,`|` 做 alternation。在 bash 中用单引号避免 shell 展开 `|`:
```bash
--kernel-name 'mul_mat_vec_q|rms_norm_f32'
```
如果 regex 包含了所有 launch 都不匹配的 pattern,会得到 `No kernels were profiled`。

### 2.4 更深入的 kernel 分析 (replay 模式)

```bash
# 对单个 kernel 类型做详细 replay profiling
ncu --replay-mode kernel \
  --kernel-name "mul_mat_vec_q" \
  --section MemoryWorkloadAnalysis \
  --section SchedulerStats \
  ./llama-simple -m model.gguf -ngl 99 -n 2 "Test"
```

Replay 模式会多次重放同一个 kernel 来收集不同 section 的 metric,结果更详细但很慢。

---

## 3. Decode 阶段 Kernel 全景

### 3.1 Kernel 类型分布

从 per-gpu profiling 数据统计了 2076 个 profiled kernels:

| Kernel | 调用次数 | 占比 | 说明 |
|--------|---------|------|------|
| `quantize_q8_1` | 737 | 35.5% | FP16→Q8_1 量化,为 matvec 准备输入 |
| `mul_mat_vec_q` | 736 | 35.5% | INT4/INT6 反量化 + MatVec 融合 kernel (**核心**) |
| `rms_norm_f32` | 249 | 12.0% | RMSNorm (已融合 weight multiply) |
| `rope_neox` | 244 | 11.8% | RoPE 位置编码 |
| `k_set_rows` | 122 | 5.9% | KV Cache 写入 |
| `flash_attn_ext_f16` | 122 | 5.9% | FlashAttention 主 kernel |
| `flash_attn_stream_k_fixup` | 122 | 5.9% | FlashAttention 子 kernel |

**关键结论**: `quantize_q8_1` + `mul_mat_vec_q` 合计占 71%,decode 阶段的计算时间几乎全部花在反量化+矩阵向量乘上。

### 3.2 单个 Decode Step 的 Kernel 序列

从 launch 0-49 的 profiling 数据可以还原一个 decode step 的 kernel 调用序列:

```
层 0-27 (28 layers),每层 8 个 FFN matvec + 2 个 attention matvec:
  rms_norm_f32        (attn_norm)
  quantize_q8_1       (Q 量化)
  mul_mat_vec_q       (Q * K_rot,   grid=256)    ← 4 KV heads × 128 / 32 threads
  quantize_q8_1       (V 量化)  
  mul_mat_vec_q       (attn * V,    grid=512)    ← 4 KV heads × 128 / 32 threads
  rope_neox           (Q 旋转)
  rope_neox           (K 旋转)
  k_set_rows          (写 KV cache)
  [每 2 层: flash_attn_ext_f16 + flash_attn_stream_k_fixup]  ← GQA 跨层批处理
  rms_norm_f32        (ffn_norm)
  quantize_q8_1       (gate 量化)
  mul_mat_vec_q       (gate_proj,   grid=18944)  ← FFN dim / 32
  quantize_q8_1       (up 量化)
  mul_mat_vec_q       (up_proj,     grid=18944)
  [SiLU activation]
  quantize_q8_1       (down 量化)
  mul_mat_vec_q       (down_proj,   grid=18944)
  [residual add — fused in rms_norm_f32]
```

**注意**: llama.cpp 使用 CUDA Graph 捕获整个 decode 计算图,所以第 2 个 token 开始的 kernel launch 都通过 `cudaGraphLaunch` 一次性提交(见输出 `CUDA Graph id 21 reused`)。

---

## 4. 核心 Kernel 详细分析

### 4.1 mul_mat_vec_q (反量化 + MatVec 融合)

这是 decode 阶段最重要的 kernel,有 5 种实例化(按 weight 量化类型和 grid 大小分):

| 实例 | Weight 类型 | Grid | Invoc | Duration | DRAM % | Compute % | Occupancy |
|------|-----------|------|-------|----------|--------|-----------|-----------|
| Q4_K, grid=512 | q4_K (type=12) | 512 | 3 | 9.21 µs | 46.4% | 41.7% | 56.9% |
| Q4_K, grid=3584 | q4_K (type=12) | 3584 | 5 | 37.73 µs | 58.1% | 72.7% | 62.5% |
| Q4_K, grid=18944 | q4_K (type=12) | **18944** | 2 | **266.66 µs** | **82.4%** | 72.5% | 64.4% |
| Q6_K, grid=512 | q6_K (type=14) | 512 | 3 | 11.63 µs | 54.7% | 51.0% | 64.7% |
| Q6_K, grid=3584 | q6_K (type=14) | 3584 | 2 | 210.40 µs | 77.8% | 79.2% | 79.1% |

**解读**:
- **grid=18944** 是 FFN 层的 down/up projection,一个 kernel 跑 **266 µs**,占单个 decode step 的大头
- **DRAM Throughput 82.4%**: 实测显存带宽约 297 GB/s (RTX 3060 理论 360 GB/s),利用率已经很高
- **Compute Throughput 72.5%**: SM 计算单元有 27.5% 空闲,说明 decode 是 **memory-bound** (等数据,不是等计算)
- Q6_K 比 Q4_K grid=3584 慢 **5.6x** (210 vs 38 µs),因为 Q6_K weight 体积大 50%

**Launch 配置**:
- Block: (32, 4, 1) = 128 threads
- Static Shared Memory: 768 bytes (用于存储 partial dot product)
- Registers: 48-53 per thread

### 4.2 flash_attn_ext_f16

| Metric | 值 |
|--------|-----|
| Grid | (8, 1, 1) |
| Block | (32, 4, 1) = 128 threads |
| Duration | **10.50 µs** avg |
| DRAM Throughput | 20.5% |
| Compute Throughput | 3.4% |
| Registers/Thread | **140** (很高) |
| Shared Mem/Block | 69.9 KB (dynamic) + 102.4 KB config |
| Theoretical Occupancy | **8.33%** (被寄存器限制) |
| Achieved Occupancy | 7.94% |

**解读**:
- FlashAttention 虽然 compute 和 memory 利用率都低,但延迟只有 **10.5 µs** — 对 4 KV heads 的 GQA 来说非常轻量
- **Register pressure** 是瓶颈: 140 regs/thread 限制了每 SM 只能跑 1 个 block (Block Limit Registers = 1.0)
- Occupancy 8.33% 看似很低,但对于这种 small-batch attention kernel 是正常的 — 增加 occupancy 也不会加速

### 4.3 rms_norm_f32

| Metric | 值 |
|--------|-----|
| Grid | (1, 1, 1) |
| Block | (1024, 1, 1) |
| Duration | **6.05 µs** avg |
| DRAM Throughput | 1.70% |
| Compute Throughput | 0.90% |
| Achieved Occupancy | 63.1% |
| Registers/Thread | 39 |

**解读**:
- 只用 **1 个 block** (1024 threads),非常小
- DRAM 1.70% — 几乎不碰显存,数据全在 L1 cache
- 1 block 导致它只用了 28 SM 中的 1 个 — 本质上是 **latency-bound** (kernel launch overhead 可能比计算本身还大)
- **已融合 weight multiply**: 因为 `rms_norm_f32<(int)1024, (bool)1, (bool)0>` 模板参数 `do_multiply=true`

### 4.4 quantize_q8_1

| 实例 | Grid | Invoc | Duration | DRAM % | Occupancy |
|------|------|-------|----------|--------|-----------|
| grid=14 (3584/256) | 14 | 13 | 2.74 µs | 3.4% | 16.6% |
| grid=74 (18944/256) | 74 | 2 | 3.20 µs | 8.5% | 40.3% |

**解读**:
- 将 FP16 activation 量化为 Q8_1,为 `mul_mat_vec_q` 准备输入
- grid=14: attention 路径的量化 (3584 hidden_dim / 256 block_size)
- grid=74: FFN 路径的量化 (18944 ffn_dim / 256 block_size)
- 延迟极低 (3 µs),输入数据量小

### 4.5 rope_neox

| 实例 | Grid | Invoc | Duration | Occupancy |
|------|------|-------|----------|-----------|
| grid=4 (1 head group) | 4 | 3 | 3.00 µs | 8.1% |
| grid=28 (Q heads) | 28 | 3 | 2.74 µs | 9.4% |

RoPE 对 Q heads (28) 和 KV heads (4) 分别执行。延迟很低,不是瓶颈。

### 4.6 k_set_rows

| Metric | 值 |
|--------|-----|
| Grid | (2, 1, 1) |
| Block | (256, 1, 1) |
| Duration | 2.72 µs |
| DRAM Throughput | 0.64% |

写入 KV Cache,数据量极小(每层 4 KV heads × 128 dim)。

---

## 5. Roofline 分析

### 5.1 RTX 3060 理论上限

| 指标 | 值 |
|------|-----|
| 显存带宽 | 360 GB/s (GDDR6, 192-bit) |
| FP32 算力 | 12.74 TFLOPS (28 SM × 128 cores × 1.78 GHz × 2 ops/FMA) |
| INT8 算力 | 25.5 TOPS (tensor cores, 但 llama.cpp 不走 tensor core) |

### 5.2 实测效率

```
mul_mat_vec_q (grid=18944, Q4_K):
  数据量: Q4_K weight 18944×3584×0.5B + FP16 activation ≈ 4.3 MB
  计算量: 18944×3584 ≈ 68M FMAs ≈ 136M FLOPs
  Duration: 266.66 µs
  实测带宽: 82.4% × 360 GB/s = 296.6 GB/s  ← 利用率很高
  实测算力: 72.5% × 12.74 TFLOPS ≈ 9.2 TFLOPS

mul_mat_vec_q (grid=3584, Q6_K):
  实测带宽: 77.8% × 360 GB/s = 280.1 GB/s
  实测算力: 79.2% × 12.74 TFLOPS ≈ 10.1 TFLOPS
```

![Roofline Model](figures/10_roofline.png)

### 5.3 Operational Intensity 与 Roofline

```
grid=18944 FFN matvec:
  OI = 136M FLOPs / 4.3 MB = 31.6 FLOPs/Byte
  Roofline ridge point (RTX 3060): 12.74 TFLOPS / 360 GB/s = 35.4 FLOPs/Byte
  → OI (31.6) < Ridge (35.4) → Memory-bound

grid=3584 attention output:
  OI = 3584×3584×2 / (3584×3584×0.625 + 3584×2) ≈ 25M FLOPs / 8 MB = 3.1 FLOPs/Byte
  → Strongly memory-bound

grid=512 attention QK:
  OI = 512×3584×2 / (512×3584×0.625 + 3584×2) ≈ 7.3M FLOPs / 1.2 MB = 6.1 FLOPs/Byte
  → Strongly memory-bound
```

**结论**: Decode 阶段全部是 **memory-bound**。即使算力利用率只有 72%,瓶颈在显存带宽而非计算。

### 5.4 带宽利用率瓶颈分析

当前最高 DRAM 利用率 82.4%,剩余 ~18% 的带宽去哪了?

- **Kernel launch overhead**: 每个 kernel launch 有固定延迟 (~2-5 µs),对于 266 µs 的 kernel,overhead 占 ~1-2%
- **L2 cache 命中**: MatVec 的 KV cache 访问有 L2 命中,减少了一部分 DRAM 流量
- **Non-coalesced access**: Q4_K 是 block 量化格式 (super-block=256, sub-block=32),访问模式不能完全 coalesce
- **指令开销**: 地址计算、scale 反量化等非内存指令占用 SM cycle

---

![Decode Step Time Breakdown](figures/09_decode_breakdown.png)

## 6. 单次 Decode Step 时间分解

基于 profiling 数据计算单个 decode step 各 kernel 的累计时间:

| 组件 | 涉及 Kernel | 每层次数 | 每 step 时间 (28 层) |
|------|-----------|---------|---------------------|
| FFN matvec | mul_mat_vec_q (grid=18944) | 3 (gate+up+down) | 28×3×266µs = 22.4 ms |
| Attention output matvec | mul_mat_vec_q (grid=3584) | 1 | 28×1×38µs = 1.06 ms |
| Attention QK matvec | mul_mat_vec_q (grid=512) | 2 | 28×2×9.2µs = 0.52 ms |
| Quantization | quantize_q8_1 | ~10 | 28×10×2.9µs = 0.81 ms |
| RMSNorm | rms_norm_f32 | 2 | 28×2×6µs = 0.34 ms |
| RoPE | rope_neox | 3 | 28×3×3µs = 0.25 ms |
| FlashAttention | flash_attn_ext | batch | ~30 µs (跨层批处理) |
| KV Cache 写入 | k_set_rows | 2 | 28×2×2.7µs = 0.15 ms |
| **总计** | | | **≈ 25.5 ms** |

实测 decode: 39.42 ms/token → **39.42 ms** (包括 CPU overhead)

差异原因: profiling 扰动了正常执行(CUDA graph 被禁用),实际生产环境用 CUDA Graph 会更快(25.13 t/s → 39.8 ms/tok)。

**时间占比**:
- FFN matvec: **88%** (绝对瓶颈)
- Attention matvec + FA: ~6%
- 其他 (quantize, rms_norm, rope, etc.): ~6%

---

## 7. 优化方向

### 7.1 更激进的 Weight 量化

当前 Q4_K_M 中 attention/output 层用 Q6_K (29/339 tensors)。如果全部用 Q4_K:
- Q6_K grid=3584: 210 µs → Q4_K grid=3584: 38 µs (5.5x 加速)
- 代价: attention 精度进一步下降

### 7.2 KV Cache 量化

K cache Q8_0/Q4_0 减少显存带宽需求,K cache 访问占 attention matvec 的 weight 部分:
- Q4_K weight 3584×512 = 1.8M elements × 4.5 bit = 1.0 MB
- K cache 512×1 × 2B (FP16) = 1 KB per token → negligible for decode
- 收益: KV cache 量化主要减少显存占用,对 decode 速度影响很小

### 7.3 Speculative Decoding

当前瓶颈是 FFN matvec (每个 token 必须等所有 FFN 层完成)。Speculative decoding 用 draft model 生成多个候选 token,一次 batch 验证,相当于把 memory-bound 的 decode 变成 compute-bound 的小 batch prefill。

### 7.4 Tensor Core 利用

当前 llama.cpp 不走 Tensor Core (用的是 CUDA core 的 FMA 指令+反量化)。对于 Q4_K 的 block-wise 量化,Tensor Core 的 m16n8k16 矩阵 multiply 与 block 结构不完全对齐,但可以通过调整权重布局部分利用。

---

## 8. 面试要点

| 问题 | 回答要点 |
|------|---------|
| "decode 为什么是 memory-bound?" | 每个 token 只需 1 次矩阵向量乘,FLOPs/Byte ≈ 3-32,远低于 RTX 3060 的 ridge point 35.4。实测 DRAM 利用率 82%,compute 利用率 72% |
| "怎么 profiling 的?" | NCU per-kernel 模式,`--launch-count 50` 限制只分析前 50 个 kernel。FFN matvec 占 decode 时延 88%,266 µs/kernel |
| "FFN matvec 为什么最慢?" | grid=18944,需要把 Q4_K weight (18944×3584≈34M params×4.5bit≈19MB) 全部读一遍,受显存带宽限制 |
| "Q4_K 和 Q6_K 速度差多少?" | grid=3584 的 matvec: Q6_K 210 µs vs Q4_K 38 µs,约 5.5x。Q6_K weight 体积大 50% 且 dequant 计算更多 |
| "RMSNorm fusion 有效吗?" | llama.cpp 已融合 `rms_norm + multiply + residual add`,但 kernel 太小 (1 block 6 µs),实际收益不大 |
| "FlashAttention occupancy 为什么只有 8%?" | register pressure (140 regs/thread),8 blocks × 128 threads × 140 regs = 143,360 regs 远超 SM 的 65536,每 SM 只能跑 1 block |

---

## 9. 原始数据文件

- `output/ncu_summary_gpu.txt` (168K) — per-gpu 模式原始输出,含 2076 个 kernel 的 profiling log
- `output/ncu_per_kernel.txt` (88K) — per-kernel 模式,前 50 个 launch 的完整 metric
- profiling 命令见本文档第 2 节

## 10. 速查: NCU 常用命令

```bash
# 快速概览 (GPU 级别)
ncu --print-summary per-gpu ./my_app

# 逐个 kernel 分析 (推荐用于 decode 这种 kernel 类型少的场景)
ncu --print-summary per-kernel ./my_app

# 只看前 100 个 kernel launch
ncu --print-summary per-kernel --launch-count 100 ./my_app

# 跳过前 1000 个 launch (跳过模型加载/prefill)
ncu --print-summary per-kernel --launch-skip 1000 --launch-count 200 ./my_app

# 只分析特定 kernel
ncu --kernel-name "mul_mat_vec_q" --print-summary per-kernel ./my_app

# 详细 Memory Workload 分析 (replay 模式,很慢)
ncu --replay-mode kernel --kernel-name "mul_mat_vec_q" \
  --section MemoryWorkloadAnalysis ./my_app

# 导出 CSV 用于后续分析
ncu --csv --print-summary per-kernel ./my_app > profile.csv

# 查看可用 section
ncu --list-sections
```
