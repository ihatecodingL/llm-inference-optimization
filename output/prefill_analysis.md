# Prefill 深度分析

模型: Qwen2.5-7B-Instruct Q4_K_M (4.4 GB), 测试日期: 2026-05-28

---

## 1. 什么是 Prefill

LLM 推理分两个阶段：

```
用户输入: "Translate to French: Hello, how are you?"
         │
         ▼
┌─────────────────────────────────────────────┐
│  PREFILL (Prompt Processing / Encoding)      │
│  - 一次性并行处理所有 prompt token            │
│  - 计算: MatMul (矩阵 × 矩阵)                │
│  - 特性: compute-bound (GPU 算力瓶颈)        │
│  - 输出: 第一个 token                        │
│  - 耗时: TTFT (Time To First Token)          │
└─────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  DECODE (自回归生成)                         │
│  - 每次处理 1 个 token                        │
│  - 计算: MatVec (矩阵 × 向量)                │
│  - 特性: memory-bound (显存带宽瓶颈)          │
│  - 逐 token 输出，直到 EOS                    │
└─────────────────────────────────────────────┘
```

**面试一句话**: "Prefill 并行处理所有 prompt token 做 MatMul，compute-bound；Decode 每次一个 token 做 MatVec，memory-bound。"

---

## 2. Prefill 计算量构成

Qwen2.5-7B 一层 Transformer 在 Prefill 时的主要计算：

| 操作 | 类型 | 计算量 (per token) |
|------|------|-------------------|
| Q, K, V 投影 | MatMul | 3 × hidden_size² |
| Attention (QKᵀ + softmax + ×V) | MatMul + element-wise | 2 × seq_len × head_dim × n_heads |
| Output 投影 | MatMul | hidden_size² |
| FFN (gate, up, down) | MatMul | 3 × hidden_size × intermediate_size |

**关键观察**: Attention 的计算量是 **O(seq_len²)**，其他都是 O(seq_len)。短 prompt 时 FFN 主导，长 prompt 时 Attention 主导。

| Seq Len | Total | QKV Proj | Attention | FFN | Output Proj |
|---------|-------|----------|-----------|-----|-------------|
| 128 | 1,835 G | 276 G | **7 G** (0.4%) | 1,460 G | 92 G |
| 512 | 7,418 G | 1,105 G | **105 G** (1.4%) | 5,840 G | 368 G |
| 2048 | 30,937 G | 4,420 G | **1,684 G** (5.4%) | 23,360 G | 1,473 G |
| 4096 | 65,241 G | 8,839 G | **6,735 G** (10.3%) | 46,721 G | 2,946 G |

seq=4096 时 Attention 已占 10%，继续增长会快速成为绝对瓶颈。这就是 FlashAttention 的价值。

![Prefill Compute Analysis](figures/08_prefill_compute.png)

---

## 3. 实测 TTFT 曲线 + FlashAttention 对比

> **2026-05-28 同 session 对照测试**：FA OFF 和 FA ON 在相同 GPU 状态下连续测试，确保对照有效。

### 3.1 数据表

| Seq Len | FA OFF Prefill | FA OFF TTFT | FA ON Prefill | FA ON TTFT | **TTFT 加速** | 备注 |
|---------|---------------|-------------|---------------|------------|-------------|------|
| 128 | 1,420 tok/s | 90 ms | 1,448 tok/s | 89 ms | **1.02x** | Attention 仅 0.4%，FA 几乎无收益 |
| 256 | 1,505 tok/s | 170 ms | 1,605 tok/s | 160 ms | **1.07x** | 开始有微弱加速 |
| 512 | 1,474 tok/s | 347 ms | 1,598 tok/s | 321 ms | **1.08x** | |
| 1024 | 1,408 tok/s | 727 ms | 1,602 tok/s | 639 ms | **1.14x** | Attention 占比 ~3% |
| 2048 | 1,136 tok/s | 1,804 ms | 1,559 tok/s | **1,314 ms** | **1.37x** | Attention 占比 5.4%，FA 显著有效 |
| 4096 | 1,173 tok/s | 3,493 ms | 1,518 tok/s | **2,699 ms** | **1.29x** | Attention 占比 10.3%，FA 省 794ms |

### 3.2 TTFT 可视化

```
TTFT (ms)
 3500 ┤                                          ● FA OFF (3493ms)
      ┤
 3000 ┤
      ┤                              ● FA ON (2699ms)
 2500 ┤
      ┤
 2000 ┤                     ● FA OFF (1804ms)
      ┤
 1500 ┤                ● FA ON (1314ms)
      ┤
 1000 ┤           ● FA OFF (727ms)
      ┤      ● FA ON (639ms)
  500 ┤ ● FA OFF (347ms)  ← 512
      ┤ ● FA ON (321ms)
    0 └─────┴─────┴─────┴─────┴─────┴───── seq_len
        0    512   1024  1536  2048  2560  3072  3584  4096
```

### 3.3 FA 收益随 seq_len 增大

```
seq_len: 128    256    512    1024   2048   4096
FA加速:  1.02x  1.07x  1.08x  1.14x  1.37x  1.29x
          ▏      ▏      ▏      ▏      ██     ██
```

### 3.4 解读

- **seq_len < 512**：FA 收益 <10%，因为 Attention 计算量占比极小（<2%），FFN matmul 主导
- **seq_len = 1024**：FA 开始有效（14% 加速），Attention 占比 ~3%
- **seq_len = 2048**：FA 收益最大（37% 加速），Attention 占比 5.4%
- **seq_len = 4096**：FA 加速 29%，Attention 占比 10.3%。加速比 2048 时略降可能是因为 FA kernel 自身的开销开始体现

**关键结论**：FlashAttention 在 seq ≥ 1024 时开始产生可感知的收益，在 2048 附近达到最佳加速比。对于 RAG、长文档处理等场景，FA 是必开的优化。

### 3.5 关于 TFLOPS 数字

实测 TFLOPS 超过 RTX 3060 的 FP16 CUDA Core 理论峰值（12.74 TFLOPS）。原因：

1. **Tensor Cores**: RTX 3060 的 Tensor Cores 提供 ~25.6 TFLOPS（无稀疏化），实际推理使用 Tensor Cores
2. **FLOPs 估算偏大**: 理论公式 `2×P×seq_len` 包含了 softmax、RMSNorm 等小操作，实际有效 FLOPs 低于估算

**面试时不要报 TFLOPS 数字，报 TTFT 就够了** — 面试官想知道的是"2048 token 的 prompt 多久出第一个字"。

---

## 4. Prefill vs Decode 核心区别

| | Prefill | Decode |
|---|---------|--------|
| **做什么** | 并行处理所有 prompt token | 逐 token 自回归生成 |
| **计算类型** | MatMul (矩阵×矩阵) | MatVec (矩阵×向量) |
| **瓶颈** | **GPU 算力** (compute-bound) | **显存带宽** (memory-bound) |
| **延迟特性** | 随 seq_len 线性增长 | 每 token 基本恒定 |
| **量化收益** | 中等（权重体积影响 MatMul） | **大**（权重体积直接影响带宽） |
| **FlashAttention 收益** | seq≥1024 时显著 (1.14-1.37x) | 几乎无影响 |
| **关键优化** | FlashAttention, Chunked Prefill | 量化, KV Cache 优化, Speculative Decoding |

### 实测证据（同一个 Q4_K_M 模型）

```
seq_len   128     256     512     1024    2048    4096
Prefill:  1420    1505    1474    1408    1136    1173   tok/s (FA OFF)
Decode:   62.0    62.0    62.0    61.7    61.7    61.7   tok/s (恒定!)
```

Decode 速度在所有 seq_len 下恒定 ~62 tok/s，完美证明"Decode 是 memory-bound"——它只关心模型权重大小，不关心 seq_len。

---

## 5. Prefill 优化策略

| 策略 | 原理 | 本项目实测 | 适用场景 |
|------|------|-----------|---------|
| **FlashAttention** | 分块计算 attention，避免完整 attention matrix 写入 HBM | ✅ seq=2048 加速 37%，seq=4096 加速 29% | 长 prompt (seq>1024) |
| **量化权重 (W4A16)** | 减少权重读取带宽 | ✅ Q4_K_M 4.4GB，decode 61 tok/s | 所有场景 |
| Prefill Chunking | 将长 prompt 拆成多个 chunk，GPU 轮流处理 | 未测 | 超长 prompt + 并发服务 |
| Prefix Caching | 相同 system prompt 的 KV cache 复用 | 未测 | 多轮对话 |
| Continuous Batching | 多个请求 prefill/decode 混合调度 | 未测（单用户项目） | 高并发服务 |
| Kernel Fusion | Q/K/V 投影合并为一个 kernel | 分析见 `docs/kernel_fusion_analysis.md` | 所有场景 |

---

## 6. 面试模拟

**Q: "Prefill 和 Decode 有什么区别？"**

> Prefill 阶段并行处理所有 prompt token，做 MatMul，是 compute-bound——我实测 seq_len 从 128 到 4096（32 倍），TTFT 从 90ms 增大到 3493ms（39 倍），近乎线性。
> Decode 阶段每次处理 1 个 token，做 MatVec，是 memory-bound——我实测 seq_len 从 128 到 4096，Decode 速度恒定为 62 tok/s，完全不受 seq_len 影响。
> 所以优化策略不同：Prefill 用 FlashAttention、Chunking；Decode 用量化减权重、优化 KV cache。

**Q: "FlashAttention 实际收益有多大？"**

> 我实测了同 session 下 FA ON vs OFF 的对照。seq=512 以下几乎无收益（<10%），因为 Attention 占比太小。seq=2048 时加速 37%（TTFT 从 1804ms 降到 1314ms），seq=4096 时加速 29%（从 3493ms 降到 2699ms）。所以 FA 在长 prompt 场景是必开的，短 prompt 用处不大。

**Q: "TTFT 多少是合理的？"**

> 看场景。对话式 AI 的 prompt 通常 <200 token，TTFT <100ms 用户无感。RAG 场景输入 2000+ token 文档，无 FA 时 TTFT 约 1.8s，开 FA 后降到 1.3s。代码补全场景需要 <50ms。我们的数据：512 token → 321ms (FA ON), 2048 token → 1314ms (FA ON), 4096 token → 2699ms (FA ON)。

**Q: "FlashAttention 为什么对 Decode 没帮助？"**

> Decode 每次只处理 1 个 token，Attention 计算的是 1 个 query 对所有历史 key 的点积，不存在 N×N 的 attention matrix。FA 的核心优化是避免将 N×N attention matrix 写入 HBM，Decode 根本就没有这个 N×N 矩阵。所以 FA 对 Decode 几乎没有加速效果——我们的 NCU profiling 也证实了这一点，FlashAttention kernel 在 decode 中仅占 0.03ms。
