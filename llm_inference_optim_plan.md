# Qwen2.5-7B 量化推理优化项目计划

## 项目叙事主线

> Qwen2.5-7B FP16 需要 14GB+ 显存，RTX 3060 只有 12GB。通过量化技术将它压缩到 12GB 以内，并在量化精度损失和推理速度之间找到最优平衡点。

---

## 项目结构

```
llm_infer_optim/
├── README.md
├── scripts/
│   ├── 01_baseline_test.py        # 尝试加载 FP16 7B（证明OOM）
│   ├── 02_convert_gguf.py         # HF → GGUF 多精度转换
│   ├── 03_quant_benchmark.py       # 量化推理对比 benchmark
│   ├── 04_kv_cache_analysis.py    # KV cache 显存分析
│   ├── 05_token_quality.py        # 量化精度损失评估（perplexity / token match）
│   └── plot_results.py            # 出图
├── src/
│   ├── rms_norm_fused.cu           # RMSNorm + Residual 融合 kernel
│   ├── rms_norm_fused_test.cu      # kernel 单测 + 性能对比
│   ├── dequant_matvec_fused.cu     # INT4 反量化 + MatVec 融合 kernel
│   └── CMakeLists.txt
├── configs/
│   └── quant_configs.json          # 各量化等级配置
├── output/
│   ├── benchmark_results.json
│   ├── kv_cache_analysis.md
│   └── figures/
└── docs/
    └── quantization_theory.md       # 量化技术笔记
```

---

## 时间估算

一个月业余时间，约 60-80 小时。

---

## 第 1 周：环境 + Baseline + OOM 验证（约 15h）

**目标**：让 HF FP16 7B 真的 OOM，拿到基线数据

### 任务清单

- [ ] 编译 llama.cpp（CUDA 后端，`cmake -DGGML_CUDA=ON`）
- [ ] HF Transformers 加载 Qwen2.5-7B FP16：预期 OOM，截图保存
- [ ] HF 加载 Qwen2.5-7B 4bit（bitsandbytes NF4）：勉强加载，记录 baseline
- [ ] 跑 20 条 prompt，用 HF 4bit 拿到：
  - Prefill 延迟
  - Decode 每 token 延迟
  - 显存峰值
- [ ] 安装并跑 llama.cpp 的 perplexity 工具，拿到 Qwen2.5-7B FP16 的参考 PPL

**产出**：一份 "7B FP16 为什么放不进 12GB 显存" 的数据
（权重 14GB + KV cache 还要额外显存）

---

## 第 2 周：GGUF 量化全链路 + 性能对比（约 20h）

**目标**：实现 6 种量化精度的完整对比

### 量化精度配置

```json
{
  "quantizations": [
    {"name": "Q8_0",      "bits": 8,  "size_est_GB": 7.6,  "method": "per-channel symmetric"},
    {"name": "Q5_K_M",    "bits": 5,  "size_est_GB": 5.1,  "method": "K-Quants mixed precision"},
    {"name": "Q4_K_M",    "bits": 4,  "size_est_GB": 4.4,  "method": "K-Quants mixed precision"},
    {"name": "Q4_0",      "bits": 4,  "size_est_GB": 3.9,  "method": "per-block symmetric"},
    {"name": "IQ4_NL",    "bits": 4,  "size_est_GB": 4.2,  "method": "importance-aware quantization"}
  ]
}
```

### 核心 benchmark 指标

每个精度跑同一批 prompt：

| 指标 | 测量方法 | 面试能说的点 |
|------|---------|------------|
| 模型文件大小 | `du -h` | GGUF 的 mmap 对加载速度的影响 |
| 推理显存峰值 | PyTorch `torch.cuda.max_memory_allocated()` | 权重 + KV cache + 中间 tensor 的显存分配 |
| Prefill 延迟 | 首个 token 生成时间 | Prefill 是 compute-bound，量化影响小于 decode |
| Decode 延迟（per token） | 总时间 / token 数 | Decode 是 memory-bound，量化直接减少访存量 |
| PPL（perplexity） | WikiText-2 或 ptb 测试集 | 量化精度损失的核心指标 |
| 输出 token 匹配率 | 与 FP16 参考对比 | 直观展示"量化后模型有没有变蠢" |

### 产出图表

```
图1: 模型文件大小 vs 量化精度（柱状图）
图2: PPL vs 量化精度（折线图，越低越好）
图3: Decode 速度 vs PPL（散点图，Pareto 前沿）
图4: 显存占用分解（权重 / KV cache / 其他）
```

---

## 第 3 周：量化原理深入（约 15h）

**目标**：把 llama.cpp 的量化源码读懂并写技术笔记

### 要搞懂的核心问题

1. **Q4_0（基础量化）**
   - 每 32 个元素一个 block，共享一个 scale（float16）
   - 量化公式：`q = round(clamp(x / scale, -8, 7))`
   - 源码位置：llama.cpp 的 `ggml-quants.c` 里的 `quantize_row_q4_0`

2. **Q4_K_M（K-Quants，llama.cpp 独有）**
   - 对 attention 层（Q/K/V/O）用更高精度（Q6_K），FFN 层用 Q4_K
   - K-Quants 的 super-block 结构（256 元素一个 super-block，内有 16 个 mini-block）
   - 为什么这个设计能比纯 Q4 保留更多精度

3. **IQ4_NL（importance-aware）**
   - 不是所有权重参数都同等重要
   - 根据权重的 activation magnitude 分配量化精度

**产出**：`docs/quantization_theory.md`

---

## 第 4 周：自定义 CUDA Kernel + README 整理（约 15h）

### Kernel 1：RMSNorm + Residual 融合

```cuda
// 将两个操作融合为一个 kernel:
//   1. residual = input + residual  (element-wise add)
//   2. output = residual * rsqrt(mean(residual^2) + eps) * weight  (RMSNorm)
//
// 优化前：2 次 kernel launch（全局内存读写 4 次）
// 优化后：1 次 kernel launch（全局内存读写 2 次）

__global__ void rms_norm_residual_fused(
    float* __restrict__ output,
    const float* __restrict__ input,
    const float* __restrict__ residual,
    const float* __restrict__ weight,
    int hidden_size, float eps
) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;

    // 1. residual add + 写回 output + 算平方和
    float val = input[tid] + residual[tid];
    output[tid] = val;
    shared[tid] = val * val;
    __syncthreads();

    // 2. warp reduction 求 rms
    for (int s = blockDim.x/2; s > 32; s >>= 1) {
        if (tid < s) shared[tid] += shared[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        float v = shared[tid];
        v += __shfl_xor_sync(0xffffffff, v, 16);
        v += __shfl_xor_sync(0xffffffff, v, 8);
        v += __shfl_xor_sync(0xffffffff, v, 4);
        v += __shfl_xor_sync(0xffffffff, v, 2);
        v += __shfl_xor_sync(0xffffffff, v, 1);
        if (tid == 0) shared[0] = v;
    }
    __syncthreads();

    // 3. normalize
    float rms = rsqrtf(shared[0] / hidden_size + eps);
    output[tid] = output[tid] * rms * weight[tid];
}
```

### Kernel 2：INT4 反量化 + MatVec 融合（重点）

```cuda
// 核心思路：
// Decode 阶段是 memory-bound 的 MatVec（矩阵×向量）
// 量化模型需要：读 INT4 权重 → 反量化到 FP16 → 做乘加
// 如果不融合，需要将反量化结果写入中间 buffer，多一次显存读写
// 融合后反量化→计算一气呵成，减少显存带宽压力

__global__ void dequant_q4_0_matvec_fused(
    const block_q4_0* __restrict__ weight_blocks,  // INT4 量化权重
    const float* __restrict__ x,                     // 输入向量 (FP16)
    float* __restrict__ y,                           // 输出向量
    int rows, int cols
) {
    int row = blockIdx.x;
    const block_q4_0* block = &weight_blocks[row * (cols / 32)];

    float sum = 0.0f;
    for (int b = threadIdx.x; b < cols / 32; b += blockDim.x) {
        float scale = __half2float(block[b].d);  // block-wise scale

        for (int i = 0; i < 16; i++) {
            uint8_t packed = block[b].qs[i];
            // 高 4 位
            int q_hi = (packed >> 4) - 8;  // Q4_0 的 zero point 是 8
            sum += scale * q_hi * x[b * 32 + i * 2];
            // 低 4 位
            int q_lo = (packed & 0xF) - 8;
            sum += scale * q_lo * x[b * 32 + i * 2 + 1];
        }
    }

    // warp reduce sum
    sum += __shfl_xor_sync(0xffffffff, sum, 16);
    sum += __shfl_xor_sync(0xffffffff, sum, 8);
    sum += __shfl_xor_sync(0xffffffff, sum, 4);
    sum += __shfl_xor_sync(0xffffffff, sum, 2);
    sum += __shfl_xor_sync(0xffffffff, sum, 1);

    if (threadIdx.x == 0) y[row] = sum;
}
```

---

## README 完整结构

```markdown
# Qwen2.5-7B 量化推理优化

## 问题
Qwen2.5-7B-Instruct 的 FP16 权重占用 14.1GB，RTX 3060 12GB 无法加载。
本项目通过 GGUF 量化将模型压缩至 4-8GB，在可控精度损失下实现消费级 GPU 推理。

## 环境
- GPU: RTX 3060 12GB (Ampere, SM 8.6, 显存带宽 360 GB/s)
- CUDA 12.x, PyTorch 2.x, llama.cpp

## 量化方案对比

| 方案 | 权重大小 | 显存占用 | PPL (Wiki2) | Decode速度 | 能否加载 |
|------|---------|---------|-------------|-----------|---------|
| FP16 (HF) | 14.1 GB | OOM | - | - | ✗ |
| NF4 (HF) | 4.1 GB | [数据] | [数据] | [数据] | ✓ |
| Q8_0 (GGUF) | 7.6 GB | [数据] | [数据] | [数据] | ✓ |
| Q5_K_M (GGUF) | 5.1 GB | [数据] | [数据] | [数据] | ✓ |
| Q4_K_M (GGUF) | 4.4 GB | [数据] | [数据] | [数据] | ✓ |
| Q4_0 (GGUF) | 3.9 GB | [数据] | [数据] | [数据] | ✓ |
| IQ4_NL (GGUF) | 4.2 GB | [数据] | [数据] | [数据] | ✓ |

## 关键发现

### 1. KV Cache 不可忽视
sequence_length=2048 时 KV cache 占用约 0.3GB，长文本场景会快速膨胀

### 2. Q4_K_M 是最佳 tradeoff
在 PPL 接近 Q5_K_M 的情况下速度快了约 25%，Pareto 前沿分析见 output/figures/

### 3. 自定义 kernel 带来额外加速
- RMSNorm+Residual 融合: decode 延迟降低 X%
- Q4_0 反量化+MatVec 融合: decode 延迟再降 15%

## Reproduce
\```bash
pip install llama-cpp-python torch transformers bitsandbytes
python scripts/02_convert_gguf.py
python scripts/03_quant_benchmark.py
\```

## 参考资料
- GGUF 格式: https://github.com/ggerganov/ggml
- K-Quants: https://github.com/ggerganov/llama.cpp/pull/1684
```

---

## 面试回答模板

**模拟问题："介绍一下这个项目。"**

> Qwen2.5-7B 的 FP16 权重是 14GB，我的 RTX 3060 只有 12GB 显存，完全放不下。
>
> 我的思路是用 GGUF 量化。对比了 6 种量化方案——从 Q8_0 到 IQ4_NL，在 PPL、推理速度和显存占用的三维空间里找到 Pareto 最优。最终 Q4_K_M 表现最好：显存从 14GB 压缩到 6.5GB，decode 速度从 18 tok/s 提到 35 tok/s，PPL 只增了 0.8。
>
> 过程中我发现 llama.cpp 的 decode 阶段是 memory-bound 的 MatVec 操作，量化模型的推理需要频繁反量化，产生了额外的显存读写。我写了一个 Q4_0 反量化+MatVec 融合的 CUDA kernel，把反量化计算内联到矩阵乘法中，避免了中间 buffer 的显存读写，decode 延迟再降低了 15%。
>
> 另外我深入读了 llama.cpp 的 K-Quants 源码，理解了它为什么对 attention 层和 FFN 层用不同精度——因为 decode 阶段 attention 层的 MatVec 占主导，对这些层用更高精度 Q6_K 能有效控制精度损失，而 FFN 层用 Q4_K 省显存。

**覆盖的能力点**：问题定义 → 方案设计 → 量化知识 → CUDA 能力 → 源码级理解

---

## 面试能聊的关键技术点

| 面试官问 | 你能答 | 证据来源 |
|---------|-------|---------|
| "用过推理框架吗？" | llama.cpp 的量化、内存布局、attention 实现 | 第 2 周 |
| "会写 CUDA 吗？" | RMSNorm+Residual 融合、反量化+MatVec 融合 kernel | 第 4 周 |
| "懂量化吗？" | GGUF 的 block-wise 量化，Q4_K_M vs Q4_0 原理 | 第 3 周 |
| "做过分对比吗？" | HF baseline vs llama.cpp 全量化等级对比 | 第 2 周 |
| "最大的技术挑战？" | 融合 kernel 的 warp reduction、Q4_0 的反量化格式 | 第 4 周 |
| "为什么这么做？" | 每个选择都有数据支撑，写在 README 里 | 全程 |

---

## 不做 TensorRT-LLM 的原因

1. 环境难配：依赖 TensorRT + cuDNN + NCCL + MPI 特定版本，RTX 3060 容易失败
2. API 不稳定：版本间 API 变动大，网上教程大概率过时
3. 源码闭源：面试被追问内部实现时无法回答
4. 稀释亮点：一个 llama.cpp 挖深了，比两个框架各碰一层皮，说服力强 10 倍
