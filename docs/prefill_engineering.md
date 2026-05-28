# LLM 推理 Prefill 工程详解

> 面向 LLM 推理开发者的 Prefill 技术全链路指南。
> 涵盖原理、数学、优化策略、工程实践。

---

## 目录

1. [什么是 Prefill](#1-什么是-prefill)
2. [Prefill 阶段的计算分解](#2-prefill-阶段的计算分解)
3. [Prefill 内存分析](#3-prefill-内存分析)
4. [Roofline 模型：为什么 Prefill 是 compute-bound](#4-roofline-模型为什么-prefill-是-compute-bound)
5. [关键指标：TTFT](#5-关键指标ttft)
6. [优化策略](#6-优化策略)
7. [Prefill 在推理系统中的位置](#7-prefill-在推理系统中的位置)
8. [工程实践：以 llama.cpp 为例](#8-工程实践以-llamacpp-为例)
9. [面试要点](#9-面试要点)

---

## 1. 什么是 Prefill

### 1.1 LLM 推理的两阶段

大语言模型是自回归的（autoregressive）：每生成一个 token，把它追加到输入后面，再生成下一个。这自然把推理分成了两个阶段：

```
时间轴 ──────────────────────────────────────────────────────►

┌──────────────────────┐  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
│      PREFTLL          │  │ D_1 │ │ D_2 │ │ D_3 │ │ D_4 │ ...
│  (prompt processing)  │  │     │ │     │ │     │ │     │
└──────────────────────┘  └─────┘ └─────┘ └─────┘ └─────┘
         ↑                    ↑       ↑       ↑       ↑
    处理所有                 逐 token 自回归生成
    prompt token             (每次只处理 1 个 token)
```

**Prefill (Prompt Processing / Encoding / Init)**:
- 输入：用户 prompt 的全部 token（比如 2048 个）
- 计算：一次性并行处理
- 输出：生成第一个 token
- 耗时：TTFT (Time To First Token)

**Decode (Generation / Auto-regressive)**:
- 输入：每次 1 个 token（上一轮的输出）
- 计算：逐 token 串行
- 输出：持续生成 token 直到遇到 EOS
- 耗时：TPOT (Time Per Output Token) × 输出长度

### 1.2 Prefill 的内部流程

以一个 seq_len=4 的 prompt "Hello, how are you" 为例：

```
Step 1: Tokenization
"Hello, how are you"  →  [token_0, token_1, token_2, token_3]

Step 2: Embedding
[token_0, token_1, token_2, token_3]  →  [emb_0, emb_1, emb_2, emb_3]
                                           shape: (4, hidden_size)

Step 3: Transformer Layers (逐层)
For each layer:
  ┌─────────────────────────────────────────────────────┐
  │  a) RMSNorm (input)                                  │
  │  b) QKV Projection: (4, h) × (h, 3h) → (4, 3h)     │  ← MatMul
  │     split into Q, K, V: each (4, h)                  │
  │  c) Attention: Q×Kᵀ → (4,4) softmax × V → (4, h)    │  ← MatMul × 2
  │  d) Output Projection: (4, h) × (h, h) → (4, h)     │  ← MatMul
  │  e) Residual Add                                     │
  │  f) RMSNorm (pre-FFN)                                │
  │  g) FFN Gate/Up: (4, h) × (h, inter) → (4, inter)   │  ← MatMul × 2
  │  h) FFN Down: (4, inter) × (h, h) → (4, h)          │  ← MatMul
  │  i) Residual Add                                     │
  └─────────────────────────────────────────────────────┘

Step 4: LM Head
hidden_state → (4, vocab_size) → 取最后一个位置 → token 预测
```

**关键观察**: 每一步的 MatMul 都是 `(seq_len, hidden) × (hidden, *)`，seq_len 直接决定了计算量。

### 1.3 与 Decode 的本质区别

```
Prefill (seq_len=2048):
  input:  (2048, hidden_size)
  QKV:    (2048, hidden) × (hidden, 3h)  = 2048 × hidden × 3h ≈ 77G MACs
  FFN:    (2048, hidden) × (hidden, 4h)  = 2048 × 4h² ≈ 103G MACs
  → 大量并行计算，GPU 利用率高

Decode (seq_len=1):
  input:  (1, hidden_size)
  QKV:    (1, hidden) × (hidden, 3h)     = 1 × hidden × 3h ≈ 39M MACs
  FFN:    (1, hidden) × (hidden, 4h)     = 1 × 4h² ≈ 51M MACs
  → 计算量极小，GPU 大部分时间在等数据从显存传过来
```

**Decode 处理 1 个 token 的计算量只有 Prefill 处理 2048 个 token 的 1/2048，但它需要读完整份模型权重（~4.4 GB）。这 4.4 GB 的读取时间是固定的，不管你算 1 个还是 2048 个 token。**

---

## 2. Prefill 阶段的计算分解

### 2.1 FLOPs 公式

对于 Qwen2.5-7B 的配置：

```
hidden_size (h)         = 3584
intermediate_size (int) = 18944
num_layers (L)          = 28
num_q_heads             = 28
num_kv_heads            = 4   (GQA)
head_dim (d)            = 128
```

**单层单 token 的 MACs**：

```
QKV 投影:     3 × h²           = 3 × 3584²      ≈ 38.5M
QK^T:         seq_len × d × nq = seq_len × 3584  (GQA: 只算 Q heads)
Attention ×V: seq_len × d × nq = seq_len × 3584
Output 投影:  h²               = 3584²           ≈ 12.8M
FFN (gate+up):2 × h × int      = 2 × 3584 × 18944 ≈ 135.7M
FFN (down):   int × h          = 18944 × 3584    ≈ 67.9M
───────────────────────────────────────────────────
单层单 token: ≈ 255M + 2 × seq_len × 3584 MACs
28 层单 token:≈ 7.14G + 56 × seq_len × 3584 MACs
```

**总 Prefill MACs** = seq_len × (per_token_MACs)

### 2.2 Attention 的 O(N²) 问题

Attention 部分（QK^T + ×V）每层的 MACs = 2 × seq_len² × d × nq_heads / n_kv_heads

```
seq_len = 128:    2 × 128² × 128 × 28 / 4  ≈ 7.3M     (占总量 <1%)
seq_len = 512:    2 × 512² × 128 × 28 / 4  ≈ 117M     (占总量 ~1.5%)
seq_len = 2048:   2 × 2048² × 128 × 28 / 4 ≈ 1.9G     (占总量 ~5%)
seq_len = 4096:   2 × 4096² × 128 × 28 / 4 ≈ 7.5G     (占总量 ~10%)
seq_len = 8192:   2 × 8192² × 128 × 28 / 4 ≈ 30G      (占总量 ~20%)
seq_len = 32768:  2 × 32768² × 128 × 28 / 4≈ 481G     (占总量 ~50%)
```

**关键转折点**：seq_len < 2048 时 FFN 是计算主力；seq_len > 4096 后 Attention 快速成为瓶颈。这就是 FlashAttention 的价值所在。

### 2.3 项目实测数据（Q4_K_M, RTX 3060）

| Seq | Prefill 吞吐 | TTFT | 说明 |
|-----|-------------|------|------|
| 128 | 1,944 tok/s | 66 ms | 短 prompt |
| 256 | 2,052 tok/s | 125 ms | |
| 512 | 2,048 tok/s | 250 ms | 常见对话 |
| 1024 | 1,945 tok/s | 527 ms | |
| 2048 | 1,826 tok/s | 1,122 ms | RAG 文档 |
| 4096 | 1,628 tok/s | 2,515 ms | 长上下文 |

**观察**：
1. Prefill 吞吐随 seq_len 缓慢下降（1944→1628 tok/s），说明 compute-bound 但非纯线性
2. TTFT 随 seq_len 近乎线性增长（66→2515ms，38x / 32x ≈ 1.2x 超线性），因为有 Attention O(N²) 的额外开销

---

## 3. Prefill 内存分析

### 3.1 显存使用

Prefill 阶段的显存分配（以 Qwen2.5-7B Q4_K_M, seq_len=2048 为例）：

```
┌────────────────────────────────────────┐
│ 模型权重: 4,466 MB (固定)              │  ← 量化收益的主要来源
├────────────────────────────────────────┤
│ KV Cache: 112 MB                       │  ← Prefill 写入，Decode 读取
│ (seq_len × 2 × L × n_kv_heads × d × 2B)│
├────────────────────────────────────────┤
│ 中间激活: ~29 MB                       │  ← Prefill 期间存在
│ (QKV 投影结果 + FFN 中间 + residuals)  │     Decode 阶段极小(~1 MB)
├────────────────────────────────────────┤
│ CUDA Context: ~450 MB                  │  ← 固定开销
└────────────────────────────────────────┘
总计: ~5,057 MB / 12,022 MB
```

**Prefill 的临时显存压力**：虽然总量 OK，但 Prefill 期间需要同时存 QKV 投影结果和 attention 中间结果。如果不用 FlashAttention，标准的 QK^T 矩阵需要 `seq_len² × n_heads × 4 bytes`：

```
seq_len=2048:  2048² × 28 × 4 ≈ 470 MB   (attention matrix alone)
seq_len=4096:  4096² × 28 × 4 ≈ 1.9 GB   (可能触发 OOM!)
seq_len=8192:  8192² × 28 × 4 ≈ 7.5 GB   (必定 OOM)
```

**FlashAttention 将此降到 O(seq_len)** 而不是 O(seq_len²)，这就是它如此重要的原因。

### 3.2 Prefill 结束后的显存状态

```
Prefill 结束 → 中间激活释放（Q, K, V 投影结果丢弃）
            → KV Cache 保留（供后续 Decode 使用）
            → 显存占用回落到: 权重 + KV Cache + CUDA Context
```

---

## 4. Roofline 模型：为什么 Prefill 是 compute-bound

### 4.1 概念

Roofline 模型用两个参数描述一个操作的瓶颈：

```
算术强度 (Operational Intensity) = FLOPs / Bytes_read
                                  = 计算量 / 访存量

如果 算术强度 > GPU 的 FLOPs/Byte 比值 → compute-bound
如果 算术强度 < GPU 的 FLOPs/Byte 比值 → memory-bound
```

### 4.2 计算 Prefill 和 Decode 的算术强度

**RTX 3060**:
- 峰值算力: ~12.7 TFLOPS (CUDA cores FP16) / ~25.6 TFLOPS (Tensor cores)
- 显存带宽: 360 GB/s
- 临界点: 12.7T / 360G ≈ 35 FLOPs/Byte (CUDA cores)
- 临界点: 25.6T / 360G ≈ 71 FLOPs/Byte (Tensor cores)

**Prefill (seq_len=2048, 一层 Transformer)**:

```
计算量: ~7.1G MACs ≈ 14.2G FLOPs (per token per layer × seq_len)
访存量: 读权重 (~256M bytes for QKV+FFN weights at 4-bit for Q4_K_M)
        读输入 (~29K bytes, seq_len × hidden × 2B)
        写输出 (~29K bytes)
        ≈ 256 MB per layer

算术强度 ≈ 14.2G / 256M ≈ 55 FLOPs/Byte
→ 接近临界值，偏向 compute-bound
```

**Prefill (seq_len=128)**:

```
计算量: ~0.44G FLOPs (per layer)
访存量: ~256 MB
算术强度 ≈ 0.44G / 256M ≈ 1.7 FLOPs/Byte
→ 明显 memory-bound！短 prompt 下 Prefill 也是 memory-bound
```

**Decode (seq_len=1)**:

```
计算量: ~14.2G / 2048 ≈ 7M FLOPs (per layer, 1 token)
访存量: ~256 MB (仍然需要读全部权重)
算术强度 ≈ 7M / 256M ≈ 0.027 FLOPs/Byte
→ 严重 memory-bound
```

### 4.3 结论

```
算术强度:
  Decode (seq=1)     ▏ 0.03   严重 memory-bound
  Prefill (seq=128)  ▏ 1.7    memory-bound
  Prefill (seq=512)  ▏ 14     transitional
  Prefill (seq=2048) █ 55     compute-bound
  Prefill (seq=4096) █ 110    strongly compute-bound

GPU 临界值: ~35 (CUDA cores) / ~71 (Tensor cores)
```

**这就是为什么：**
- **短 prompt 不需要特别的 Prefill 优化**（memory-bound，和 Decode 类似）
- **长 prompt 必须关注 Prefill**（compute-bound，但 Attention O(N²) 会超出算力预算）
- **Decode 永远是 memory-bound**（量化是最有效的优化手段）

---

## 5. 关键指标：TTFT

### 5.1 定义

**TTFT (Time To First Token)** = 从用户发送请求到第一个 token 生成完毕的时间。

```
TTFT = Tokenization + Embedding + Prefill + 第一个 token 的 Decode
     ≈ Prefill (占主导)
```

### 5.2 TTFT 要求

| 场景 | 典型 prompt 长度 | TTFT 要求 | 原因 |
|------|-----------------|----------|------|
| 代码补全 | ~200 token | < 50 ms | 实时交互 |
| 对话 | ~200-500 token | < 300 ms | 用户等待感 |
| RAG | ~2000-4000 token | < 2 s | 检索后阅读 |
| 文档摘要 | ~8000 token | < 5 s | 离线批处理 |
| 长文档分析 | ~32K+ token | < 30 s | 可接受等待 |

### 5.3 TTFT 的来源分解

以 seq_len=2048, Q4_K_M 为例：

```
Tokenizer:      ~5 ms     (CPU)
Embedding:      ~2 ms     (GPU kernel)
28 Layers:      ~1100 ms  (GPU, 主体)
  - QKV 投影:  ~200 ms
  - Attention:  ~100 ms  (QK^T + softmax + ×V)
  - Output:     ~80 ms
  - FFN:        ~700 ms  (gate + up + down)
  - Other:      ~20 ms   (RMSNorm, residual, etc.)
LM Head:        ~10 ms    (GPU kernel)
─────────────────────────
TTFT:           ~1122 ms
```

**优化的着力点**：FFN (700ms) 和 Attention (100ms，但会随 seq_len² 增长)。

---

## 6. 优化策略

### 6.1 FlashAttention

**问题**：标准 Attention 需要存储完整的 `QK^T` 矩阵 `(seq_len, seq_len)`，这导致：
- 显存开销 O(N²)
- 大量的 HBM 读写（写入 QK^T 矩阵，再读出做 softmax，再写入，再读出 ×V）

**FlashAttention 的解法**：分块计算（tiling）+ 在线 softmax（online rescaling）

```
标准 Attention:
  1. S = Q × K^T           → 写入 HBM (size: seq² × n_heads × 4B)
  2. P = softmax(S)        → 读 HBM → 写 HBM
  3. O = P × V             → 读 HBM → 写 HBM
  总计 HBM 读写: ~seq² × n_heads × 4B × 3

FlashAttention:
  1. 将 Q, K, V 分块加载到 SRAM (on-chip shared memory)
  2. 在 SRAM 内完成: S = Q_block × K_block^T
  3. 在线 softmax（不写回 HBM）
  4. 累加到输出
  总计 HBM 读写: ~seq × h × 2B × 3  (不再有 seq² 项!)
```

**收益**：
- 速度: Prefill 2-4x 加速（长 prompt 效果更明显）
- 显存: Attention 中间结果从 O(N²) 降到 O(N)
- llama.cpp 用法: `-fa 1` 或 `--flash-attn`

### 6.2 Prefill Chunking (Split Prefill)

**问题**：一个超长 prompt（32K token）的 Prefill 耗时很长，而且期间 GPU 无法做其他事。

**解法**：把长 prompt 拆成多个 chunk，分批次计算：

```
不分块 (Naive):
  [============ 32K Prefill ============] → [Decode 1] → [Decode 2]...
  TTFT = 20s (用户等疯了)

分块 (Chunked):
  [Chunk 1: 2K] → [Chunk 2: 2K] → ... → [Chunk 16: 2K] → [Decode 1] → ...
  每次 Chunk 之间可以插入其他请求的 Decode
```

**好处**：
1. 每个 chunk 的 Prefill 短（~1s），可以和其他请求的 Decode 交错调度
2. 降低 Prefill 的显存峰值（不需要一次性分配所有中间结果）
3. 改善并发服务的延迟公平性

### 6.3 Prefix Caching

**问题**：多轮对话中，system prompt 和 history 是重复的：

```
Round 1: [system_prompt] + "What is AI?"
Round 2: [system_prompt] + "What is AI?" + "Tell me more" + "Sure, AI is..."
         ↑ 这部分 KV Cache 和 Round 1 完全一样
```

**解法**：对相同的 token 前缀，复用已计算的 KV Cache。

```
实现方式:
  1. 用前缀的 token hash 或 Radix Tree 索引已缓存的 KV Cache
  2. 新请求匹配到相同前缀 → 直接拷贝 KV Cache
  3. 只计算新的 suffix 部分

收益:
  - 多轮对话: 通常省 50-90% Prefill (大部分 prompt 是重复的 history)
  - RAG: 如果多个请求共享同一个 system prompt
  - 不适合每请求都不同的独立 prompt
```

### 6.4 Continuous Batching

**问题**：传统 static batching 要求一个 batch 内所有请求同时 Prefill 同时 Decode。但实际负载是流式的，请求到达时间各不相同。

**解法**：iteration-level 调度，每次 GPU 计算可以混合 Prefill 和 Decode：

```
Iteration 1: [Req_A Prefill(512)] + [Req_B Decode] + [Req_C Decode]
Iteration 2: [Req_A Decode]       + [Req_D Prefill(2048)]
Iteration 3: [Req_A Decode]       + [Req_B Decode] + [Req_D Decode]
...

好处:
  - GPU 利用率显著提高（不用等最慢的请求）
  - TTFT 改善（新请求不必等当前 batch 的 Decode 完成）

这是 vLLM 的核心竞争力。
```

### 6.5 Kernel Fusion

**问题**：Prefill 的每层 Transformer 有多个小 kernel（RMSNorm、QKV 投影、Attention、FFN 等），每个 kernel launch 有固定开销。

**解法**：融合相邻的小操作：

```
融合前:
  kernel_1: rms_norm(input)          → 读 input, 写 normed_input
  kernel_2: matmul_qkv(normed_input) → 读 normed_input, 写 Q, K, V
  kernel_3: attention(Q, K, V)       → 读 Q, K, V, 写 attn_output
  共 3 次 kernel launch, 6 次 HBM 读写

融合后:
  kernel_1: rms_norm + QKV_proj_fused(input)  → 读 input, 写 Q, K, V
  kernel_2: attention(Q, K, V)                 → 读 Q, K, V, 写 attn_output
  共 2 次 kernel launch, 4 次 HBM 读写
```

项目计划中的 `rms_norm_fused.cu` 就是做这个。

### 6.6 量化权重 (W4A16)

**对 Prefill 的影响**：

- Prefill 也是大量 MatMul，需要读权重。量化减少了 4x 的权重读取量。
- 但 Prefill 的 compute-bound 特性使收益不如 Decode 大（Decode 是纯 memory-bound）。

**项目实测**：

```
Prefill (seq=2048): Q8_0=2124 tok/s vs Q4_0=2193 tok/s  (差异 ~3%)
Decode  (seq=2048): Q8_0=39 tok/s  vs Q4_0=67 tok/s    (差异 ~71%)
```

**结论**：量化权重对 Prefill 有小幅帮助（减少权重读取带宽），但主要收益在 Decode。

### 6.7 优化策略选择指南

```
你的 prompt 有多长？
  ├── 短 (<512): 不需要特殊 Prefill 优化，量化权重 + FlashAttention 足够
  ├── 中 (512-4096): FlashAttention 显著
  ├── 长 (4K-32K): FlashAttention + Chunked Prefill + Prefix Caching
  └── 超长 (>32K): 以上全要 + 考虑 sparse attention / sliding window

你在做单用户还是多用户服务？
  ├── 单用户 (你的 RTX 3060 项目): 量化 + FlashAttention
  └── 多用户服务: Continuous Batching + Prefix Caching + Chunked Prefill

你的瓶颈在哪？
  ├── TTFT 太高: 优化 Prefill (FlashAttention, Chunking)
  ├── TPOT 太高: 优化 Decode (量化, KV cache 量化, speculative decoding)
  └── 吞吐太低: 提高 GPU 利用率 (Continuous Batching, larger batch)
```

---

## 7. Prefill 在推理系统中的位置

### 7.1 完整的推理引擎架构

```
                   ┌──────────────────────────┐
                   │     Request Scheduler      │
                   │  - 决定何时做 Prefill       │
                   │  - 决定 batch 组成          │
                   │  - Prefill/Decode 混合调度   │
                   └──────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ Tokenizer│   │  KV Cache │   │  Model   │
        │ (CPU)    │   │  Manager  │   │  Runner  │
        └──────────┘   └──────────┘   └──────────┘
                              │               │
                    ┌─────────┴───────┐       │
                    │ Block Allocator  │       │
                    │ (PagedAttention) │       │
                    └─────────────────┘       │
                                              ▼
                              ┌──────────────────────────┐
                              │     GPU Kernel Executor    │
                              │  - Prefill Kernels         │
                              │    · flash_attn_fwd        │
                              │    · rms_norm_fused        │
                              │    · matmul_qkv            │
                              │    · ffn_gate_up           │
                              │    · ffn_down              │
                              │  - Decode Kernels          │
                              │    · dequant_matvec_fused  │
                              │    · flash_attn_decode     │
                              └──────────────────────────┘
```

### 7.2 llama.cpp 的 Prefill 实现

llama.cpp 中 Prefill 和 Decode 使用相同的 `llama_decode()` 入口，区别在于输入的 token 数量：

```c
// llama.cpp 内部 (简化)
int llama_decode(llama_context * ctx, llama_batch batch) {
    // batch.n_tokens > 1 → Prefill 逻辑
    // batch.n_tokens == 1 → Decode 逻辑

    for (int il = 0; il < n_layers; il++) {
        // 1. RMSNorm
        ggml_rms_norm(ctx, input);

        // 2. QKV projection (batch 模式: MatMul)
        ggml_mul_mat(ctx, weight_q, normed_input);  // (n_tokens, h) × (h, h)
        ggml_mul_mat(ctx, weight_k, normed_input);
        ggml_mul_mat(ctx, weight_v, normed_input);

        // 3. Attention
        if (flash_attn) {
            ggml_flash_attn(ctx, Q, K, V);  // fused kernel
        } else {
            ggml_mul_mat(ctx, Q, K_T);       // QK^T
            ggml_softmax(ctx, attn_scores);
            ggml_mul_mat(ctx, attn_scores, V);
        }

        // 4. Output projection
        ggml_mul_mat(ctx, weight_o, attn_output);

        // 5. FFN
        ggml_mul_mat(ctx, weight_gate, normed_ffn);
        ggml_mul_mat(ctx, weight_up, normed_ffn);
        ggml_silu(ctx, gate_activated);
        ggml_mul(ctx, gate_activated, up_result);
        ggml_mul_mat(ctx, weight_down, ffn_hidden);
    }

    // LM Head (最后一个 token 的 hidden state)
    ggml_mul_mat(ctx, weight_output, last_hidden_state);
}
```

关键点：**Prefill 和 Decode 的区别只在 tensor shape**——n_tokens 不同。MatMul kernel 对不同 shape 的性能特性完全不同。

---

## 8. 工程实践：以 llama.cpp 为例

### 8.1 用 llama-bench 测量 Prefill

```bash
# 只测 Prefill: n_gen=1
llama-bench -m model.gguf -p 2048 -n 1 -ngl 99

# 对比不同 prompt 长度
for seq in 128 256 512 1024 2048 4096; do
    llama-bench -m model.gguf -p $seq -n 1 -ngl 99 -o json
done

# 开 FlashAttention 测 Prefill
llama-bench -m model.gguf -p 2048 -n 1 -ngl 99 -fa 1

# 对比不同 KV cache 类型
llama-bench -m model.gguf -p 2048 -n 1 -ngl 99 -ctk f16 -ctv f16
llama-bench -m model.gguf -p 2048 -n 1 -ngl 99 -ctk q8_0 -ctv q8_0
```

### 8.2 在你的项目中复现

```bash
cd /path/to/llm_infer_optim

# 方式 1: 用包装好的脚本
conda run -n pytorch-env python scripts/05_prefill_analysis.py --model Q4_K_M

# 方式 2: 直接调 llama-bench
/path/to/llama.cpp/build/bin/llama-bench \
    -m models/gguf/qwen2.5-7b-Q4_K_M.gguf \
    -p 2048 -n 1 -ngl 99 -r 3
```

### 8.3 关键源码位置

| 想了解什么 | 看哪里 |
|-----------|--------|
| Prefill 的 MatMul kernel | `llama.cpp/ggml/src/ggml-cuda/mmq.cu` |
| FlashAttention 实现 | `llama.cpp/ggml/src/ggml-cuda/flash_attn.cu` |
| Q4_0 反量化 (Prefill 也要用) | `llama.cpp/ggml/src/ggml-cuda/dequant.cuh` |
| llama_decode 主流程 | `llama.cpp/src/llama.cpp` → `llama_decode_internal()` |
| KV cache 分配和管理 | `llama.cpp/src/llama.cpp` → `llama_kv_cache_init()` |
| Attention 计算图构建 | `llama.cpp/src/llama.cpp` → `llama_build_attn()` |

---

## 9. 面试要点

### 9.1 面试话术模板

**"介绍一下你对 LLM 推理 Prefill 的理解"**

> LLM 推理分 Prefill 和 Decode 两阶段。Prefill 一次性并行处理所有 prompt token，本质是 MatMul（矩阵乘矩阵），是 compute-bound 的。我实测 Qwen2.5-7B 在 RTX 3060 上，seq_len 从 128 增到 4096（32 倍），TTFT 从 66ms 增到 2515ms（38 倍），近乎线性，验证了 compute-bound 特性。
>
> Prefill 的主要瓶颈有两个：一是 FFN 的 MatMul（占计算量 ~70%），二是 Attention 的 O(N²) 增长。短 prompt 时 FFN 主导，长 prompt (>4096) 时 Attention 可能占 10%+ 并在持续增长。
>
> 优化 Prefill 的核心手段是 FlashAttention——通过分块计算和在线 softmax 把 Attention 的 HBM 读写从 O(N²) 降到 O(N)。做服务时还需要 Continuous Batching 和 Prefill Chunking 来平衡延迟和吞吐。

### 9.2 常见追问

**Q: "什么时候 Prefill 也是 memory-bound？"**

> 短 prompt。我可以用 Roofline 模型解释：seq_len=128 时算术强度约 1.7 FLOPs/Byte，远低于 RTX 3060 的临界值 ~35，此时 Prefill 也是 memory-bound。量化权重对短 prompt Prefill 也有帮助。

**Q: "FlashAttention 为什么能加速 Prefill？"**

> 标准 Attention 生成 `seq_len × seq_len` 的完整 QK^T 矩阵写入 HBM，再读出做 softmax。FlashAttention 把 Q、K、V 分块加载到 SRAM（片上共享内存），在 SRAM 内完成矩阵乘和 softmax，结果直接累加——避开了 O(N²) 的 HBM 读写。长 prompt 下（比如 8192 token）QK^T 矩阵本身就要 750 MB，FlashAttention 把它省掉了。

**Q: "Prefill Chunking 和 Continuous Batching 怎么配合？"**

> 当一个超长 prompt 进入系统，如果一次性 Prefill 会占 GPU 太久（比如 5 秒），其他请求的 Decode 就卡住了。Chunked Prefill 把长 prompt 拆成小块（比如每块 2K token），每个 chunk 之间可以插入其他请求的 Decode。Scheduler 在 iteration level 决定本轮是否 Prefill 一个 chunk 还是 Decode 若干请求——这就是 Continuous Batching 的实现方式。

**Q: "你的项目里对 Prefill 做了什么？"**

> 我做了完整的 Prefill 延迟曲线测量（128~4096 token，TTFT 从 66ms 到 2.5s）。计算了理论 FLOPs 分解，确认了 Attention O(N²) 的转折点。计划写一个 CUDA kernel 把 RMSNorm 和 Residual Add 融合，减少 Prefill 的 kernel launch 开销。另外 KV cache 分析确认了 GQA 对 Prefill 内存的好处——4 KV heads vs 28 Q heads 省了 7 倍 KV cache。
