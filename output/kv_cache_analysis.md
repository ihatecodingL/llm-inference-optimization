# KV Cache 显存分析

模型: Qwen2.5-7B-Instruct, 量化方案: Q4_K_M (4466 MB)

---

## 1. 模型架构参数

| 参数 | 值 |
|------|-----|
| 层数 | 28 |
| hidden_size | 3584 |
| Q heads | 28 |
| KV heads | **4** (GQA, 7:1) |
| head_dim | 128 |
| max_seq_len | 32768 |
| vocab_size | 152064 |

**Per-token KV: 57,344 bytes = 56.0 KB**（FP16）

---

![KV Cache Size vs Seq Len](figures/03_kv_cache_size.png)

## 2. KV Cache 理论大小

```
KV_Cache = 2 x num_layers x num_kv_heads x head_dim x seq_len x dtype_bytes
         = 2 x 28 x 4 x 128 x seq_len x dtype_bytes
         = 57,344 bytes x seq_len
```

| Seq Len | FP16 (2B) | Q8_0 (1B) | Q4_0 (0.5B) | 备注 |
|---------|-----------|-----------|-------------|------|
| 512 | 28 MB | 14 MB | 7 MB | 短 prompt |
| 1024 | 56 MB | 28 MB | 14 MB |  |
| 2048 | 112 MB | 56 MB | 28 MB | 常见对话长度 |
| 4096 | 224 MB | 112 MB | 56 MB | 长文档 |

**关键洞察**: GQA 把 KV heads 从 28 压到 4，KV cache 直接缩小 **7 倍**。如果没有 GQA，seq=2048 的 KV cache 将是 784 MB。

---

## 3. 推理显存四块分解

![Memory Budget Breakdown](figures/07_memory_budget.png)

| Seq Len | 权重 | KV Cache | 激活 | CUDA 开销 | 总计 | Fits 12GB |
|---------|------|----------|------|-----------|------|-----------|
| 512 | 4466 MB | 28 MB | 8 MB | 450 MB | 4952 MB | ✓ |
| 1024 | 4466 MB | 56 MB | 15 MB | 450 MB | 4987 MB | ✓ |
| 2048 | 4466 MB | 112 MB | 29 MB | 450 MB | 5057 MB | ✓ |
| 4096 | 4466 MB | 224 MB | 57 MB | 450 MB | 5197 MB | ✓ |
| 8192 | 4466 MB | 448 MB | 113 MB | 450 MB | 5477 MB | ✓ |
| 16384 | 4466 MB | 896 MB | 225 MB | 450 MB | 6037 MB | ✓ |
| 32768 | 4466 MB | 1792 MB | 449 MB | 450 MB | 7157 MB | ✓ |

**权重占比**: seq=2048 时权重 4466 MB 占总量 5057 MB 的 88%——量化收益远大于 KV cache 优化。

---

## 4. KV Cache 量化实测

![Prefill Speed by KV Config](figures/04_kv_prefill_speed.png)

### 4.1 Prefill 速度 (tok/s)

| Seq Len | f16 (baseline) | K=q8_0, no FA | K+V=q8_0 + FA | K=q4_0, no FA | FA only (FP16 KV) |
|---------|---------|---------|---------|---------|---------|
| 512 | 2056 | 2028 | 2205 | 2050 | 2235 |
| 1024 | 1953 | 1936 | 2182 | 1917 | 2216 |
| 2048 | 1829 | 1782 | 2139 | 1774 | 2174 |
| 4096 | 1628 | 1570 | 2054 | 1577 | 2086 |

![Decode Speed by KV Config](figures/05_kv_decode_speed.png)

### 4.2 Decode 速度 (tok/s) — 单 token 生成

| Seq Len | f16 (baseline) | K=q8_0, no FA | K+V=q8_0 + FA | K=q4_0, no FA | FA only (FP16 KV) |
|---------|---------|---------|---------|---------|---------|
| 512 | 59.4 | 61.2 | 62.4 | 61.8 | 66.2 |
| 1024 | 59.1 | 56.1 | 64.4 | 57.1 | 61.0 |
| 2048 | 63.2 | 61.0 | 60.0 | 58.1 | 65.8 |
| 4096 | 58.3 | 56.8 | 59.5 | 59.0 | 64.3 |

![TTFT by KV Config](figures/06_kv_ttft.png)

### 4.3 TTFT (ms) — Time To First Token

| Seq Len | f16 (baseline) | K=q8_0, no FA | K+V=q8_0 + FA | K=q4_0, no FA | FA only (FP16 KV) |
|---------|---------|---------|---------|---------|---------|
| 512 | 249 | 252 | 232 | 250 | 229 |
| 1024 | 524 | 529 | 469 | 534 | 462 |
| 2048 | 1120 | 1149 | 958 | 1154 | 942 |
| 4096 | 2516 | 2609 | 1995 | 2598 | 1964 |

### 4.4 兼容性矩阵

| KV 配置 | FlashAttention | K Cache 量化 | V Cache 量化 | 状态 | 原因 |
|---------|---------------|-------------|-------------|------|------|
| f16 (baseline) | OFF | f16 | f16 | ✓ | 默认 |
| K=q8_0 only | OFF | **q8_0** | f16 | ✓ | K 路径无 transpose, `ggml_mul_mat` 内部反量化 |
| K=q4_0 only | OFF | **q4_0** | f16 | ✓ | 同上，block_size=32 整除 head_dim=128 |
| K+V=q8_0 no FA | OFF | q8_0 | q8_0 | **✗** | V 需要 transpose, block 量化数据不能转置 |
| K+V=q8_0 + FA | ON | q8_0 | q8_0 | **✓** | FA 路径无 transpose, 直接传量化 V 给 `ggml_flash_attn_ext` |
| FA only (FP16 KV) | ON | f16 | f16 | ✓ | FA 本身也加速 Prefill 约 15-20% |

**根因分析**: 非 FA 路径中 V 经过 `permute → transpose → cont` 操作链 (`llama-graph.cpp:2057-2061`)，transpose 需要重排 block 内元素，Q8_0 的 32-element block 结构无法支持。FA 路径直接传原始 V 给 `ggml_flash_attn_ext`，没有 transpose，所以兼容。

![KV Cache Speedup Summary @ seq=2048](figures/11_kv_cache_speedup.png)

---

## 5. 源码级根因与修复方案

### 5.1 关键代码路径

```
llama-context.cpp:373-376   // 早期校验: quantized V 必须有 FA，否则 throw
llama-context.cpp:3434-3437  // 重复校验: llama_new_context_with_model
llama-graph.cpp:1970-1972    // build_attn_mha: Q/K/V permute
llama-graph.cpp:2018-2081    // 非 FA 路径: V transpose + mul_mat
llama-graph.cpp:1977-2017    // FA 路径: ggml_flash_attn_ext
```

### 5.2 修复方案

**方案 A: 反量化前置（推荐）**

在 `build_attn_mha()` 中，对 V 做 permute 之前插入反量化:
```cpp
// llama-graph.cpp, before line 1972 (v = ggml_permute(...))
if (!use_flash_attn && ggml_is_quantized(v->type)) {
    v = ggml_cast(ctx0, v, GGML_TYPE_F16);
}
v = ggml_permute(ctx0, v, 0, 2, 1, 3);  // now safe for FP16
```
然后移除 `llama-context.cpp:373-376` 和 `llama-context.cpp:3434-3437` 的校验。

**优点**: 改动最小，V cache 存储仍为 Q8_0（显存减半），仅在计算时反量化一次。
**带宽收益**: 读 Q8_0 V cache 的 HBM 带宽需求减半，反量化开销极小 (scale × int8)。

**方案 B: FA 路径泛化（长期方案）**

将 `ggml_flash_attn_ext` 的类型处理能力泛化到 `ggml_mul_mat` 路径，或为非 FA 路径添加 fused-dequant-transpose kernel。工程量较大。

---

## 6. 面试要点

| 问题 | 回答要点 |
|------|---------|
| "KV Cache 量化怎么做？" | K cache 量化随时可用（Q8_0/Q4_0），V cache 量化需要 FlashAttention。根因是非 FA 路径中 V 要 transpose，block 量化格式不支持。修复方案: permutation 前插入 ggml_cast 反量化 |
| "为什么 V cache 需要 FA?" | 标准 attention 中 V 经过 permute+transpose 操作链，transpose 改变 block 内元素排布，block-wise 量化（Q8_0 block=32）无法支持。FA 路径直接用原始 V 调 ggml_flash_attn_ext，无 transpose |
| "有没有修过 llama.cpp?" | 定位了 `llama-graph.cpp:2057-2061` 的根因，提出 A/B 两种修复方案，方案 A 的代码改动仅 3 行 |

