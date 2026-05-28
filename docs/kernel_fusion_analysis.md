# Kernel Fusion 分析：已实现 vs 未实现 & 原因

基于 llama.cpp CUDA backend 源码分析（commit 35a74c8fb），结合 RTX 3060 (SM 8.6) 的 dispatch 路径。

---

## 1. llama.cpp 已有融合（14 种）

| 融合 | 文件 | 关键行 | 图模式 |
|------|------|--------|--------|
| RMSNorm + Mul | `norm.cu` | 76-155 | `{RMS_NORM, MUL}` |
| RMSNorm + Mul + Add（Residual） | `norm.cu` | 76-155, 562 | `{RMS_NORM, MUL, ADD}` |
| Dequant + MatVec（量化权重） | `mmvq.cu` | 398 | 原生，无独立反量化步骤 |
| **FFN gate+up+SwiGLU（Decode）** | `mmvq.cu` | 396-597 | `{MUL_MAT, ADD, MUL_MAT, ADD, GLU}` |
| F32/F16 MatVec + SwiGLU | `mmvf.cu` | 7-376 | 同上 |
| SiLU + Mul | `unary.cu` | 260-284 | `{UNARY(SILU), MUL}` |
| ReLU + Sqr | `unary.cu` | 632 | `{UNARY(RELU), SQR}` |
| Chained ADD（最多 8 个） | `binbcast.cu` | 441-469 | N 个连续 ADD |
| Chained MUL（最多 8 个） | `binbcast.cu` | 471-498 | N 个连续 MUL |
| Rope + View + SetRows | `rope.cuh` | 9 | `{ROPE, VIEW, SET_ROWS}` |
| Snake（mul→sin→sqr→mul→add） | `snake.cu` | 66 | `{MUL, SIN, SQR, MUL, ADD}` |
| Softcap（Scale+Tanh+Scale） | `softcap.cu` | 3, 22 | 单 op |
| SSM Conv + Add + SiLU | `ssm-conv.cu` | — | `{SSM_CONV, ADD, UNARY(SILU)}` |
| Top-K MoE pipeline | `topk-moe.cu` | — | softmax + argsort + get_rows |

### 关键发现：你计划的两个 kernel 早已实现

原计划（`llm_inference_optim_plan.md` 第 4 周）：
- **RMSNorm + Residual 融合** → `norm.cu` 的 `rms_norm_f32<do_add=true>` 模板
- **INT4 Dequant + MatVec 融合** → `mmvq.cu` 的 `mul_mat_vec_q` kernel 天然设计

**这就是为什么第 4 周从"写 CUDA kernel"改为"NCU profiling"** —— kernel 已经有了，该做的是 profiling 验证。

---

## 2. 看似可行、实际不成立的融合

### 2.1 Prefill FFN gate+up 融合 → 不成立

**直觉**：decode 的 mmvq 已经有 `has_fusion` 把 gate+up+SwiGLU 融了，prefill 的 mmq 应该也能做。

**实际**：RTX 3060 的 prefill dispatch 逻辑（`ggml-cuda.cu:2538-2622`）：

```
src1->ne[1] < MMVQ_MAX_BATCH_SIZE (8)
  → MMVQ (decode, has_fusion ✅)

src1->ne[1] < MMQ_DP4A_MAX_BATCH_SIZE (64)
  → MMQ dp4a (no fusion ❌)

src1->ne[1] >= 64  ← RTX 3060 所有实际 prefill 都走这里
  → cuBLAS (closed source, 改不了 ❌)
```

`mmq.cu:319-320` 的判断：
```cpp
// RTX 3060 (cc=86): fp16_mma_hardware_available = true
return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
//           ↑ false                          ↑ 64
//   所以只有 ne11 < 64 走 MMQ，否则 cuBLAS
```

seq_len >= 64 走 cuBLAS batched GEMM（`ggml_cuda_mul_mat_batched_cublas`），这是闭源库，无法添加融合。seq_len < 64 走 MMQ，但此时 prefill 耗时本身极短（<50ms），融合收益可忽略。

**结论**：官方已经检测到了 prefill 的 `{MUL_MAT, MUL_MAT, GLU}` 图模式（`ggml-cuda.cu:3643-3652`），但 dispatch 时 `ggml_cuda_should_fuse_mul_mat_vec_q` 明确要求 `ncols_dst == 1`（`ggml-cuda.cu:2488`），故意排除了 prefill。这是有意识的设计选择，不是遗漏。

### 2.2 Q/K/V 投影合并 → 收益太小

Q、K、V 三个投影读同一个 hidden state x，理论上可以一次读完算三个。

**Decode 侧**：QKV 投影总共占 decode 时间的 ~8%（Q: 0.52ms, K: ~0.3ms, V: ~0.3ms ≈ 1.1ms / 25ms）。即使完美融合省一半，也不过 0.5ms（2%）。且 Q 输出维度 3584，K/V 各 512，尺寸不对称增加了实现复杂度。

**Prefill 侧**：同样遇 cuBLAS 问题。

### 2.3 Attention output matvec + post-attn RMSNorm → micro-optimization

Decode 的 attention output matvec（1.06ms）直接接 RMSNorm（0.17ms）。把 RMSNorm 内联到 matvec 结果写回前，省 0.17ms。属于 micro-optimization，<1% 整体提升。

---

## 3. 核心洞察：为什么 Fusion 在 Decode 比 Prefill 重要得多

```
                     Decode                        Prefill
                     ──────                        ───────
操作类型              MatVec (矩阵×向量)             MatMul (矩阵×矩阵)
瓶颈                  Memory-bound                  Compute-bound
瓶颈物理量            显存带宽 (360 GB/s)            GPU 算力 (12.74 TFLOPS)
Fusion 省什么         省 HBM 读/写                  省 HBM 读/写 + kernel launch
Fusion 收益机制       减少带宽压力 → 直接加速         减少带宽压力 → 不直接加速
                                                （算力才是瓶颈）

典型数据量:
  src1 (hidden state)   3584 × 2B = 7 KB             2048 × 3584 × 2B = 14.7 MB
  权重 (gate)          18944 × 3584 × 0.5B = 34 MB   同左
  权重 (up)            18944 × 3584 × 0.5B = 34 MB   同左
  
  Fusion 省            7 KB (0.01% of total)         14.7 MB (~18% of weight reads)
  
  但！                  memory-bound → 省带宽就加速    compute-bound → 14.7MB 可能在 L2 里
                                                    RTX 3060 L2 = 3MB，14.7MB 放不下
                                                    但 cuBLAS 处理了，改不了
```

**一句话**：Decode 的 fusion 是"省一次显存读 ≈ 省一次瓶颈资源"，Prefill 的 fusion 是"省一次显存读 ≈ 省一种不紧缺的资源"。

---

## 4. 面试要点

### 如果面试官问"你为什么不做 kernel fusion？"

> 我一开始也想做。研究了 llama.cpp CUDA backend 后发现，我计划的两个 kernel（RMSNorm+Residual、Dequant+MatVec）官方已经实现并融合了。我把计划从"写 kernel"改成了"NCU profiling 验证已有 kernel 的性能"。
>
> 然后我进一步分析了是否还有其他融合机会。我发现 FFN gate+up 在 decode 路径已融合（`mmvq.cu` 的 `has_fusion`），但 prefill 路径没有。深入看 dispatch 逻辑后发现，RTX 3060 的 prefill（seq>=64）走 cuBLAS，是闭源库，改不了。而 seq<64 的 MMQ 路径，prefill 是 compute-bound 的，fusion 省带宽但不省算力，收益微乎其微。
>
> 所以结论是：对于我的 GPU 和场景，现有融合已经覆盖了真正有效的地方，没有明显的遗漏。这个分析过程本身比"我写了一个 fusion kernel"更有价值。

### 关键数字

| 问题 | 答案 |
|------|------|
| llama.cpp 有多少种 kernel fusion？ | 14 种，覆盖 RMSNorm+Residual、Dequant+MatVec、FFN gate+up+SwiGLU 等 |
| Decode FFN 已经融合了吗？ | 是，`mmvq.cu:396` 的 `has_fusion` 模板参数 |
| Prefill 为什么没做？ | RTX 3060 走 cuBLAS（闭源），且 prefill 是 compute-bound，fusion 收益小 |
| 你自己写了 kernel 吗？ | 没有——用 NCU profiling 替代，验证已有 kernel 的性能表现 |

---

## 5. 源码引用速查

| 要查什么 | 文件:行号 |
|---------|----------|
| MMVQ fusion kernel | `ggml/src/ggml-cuda/mmvq.cu:396-597` |
| MMQ dispatch 条件 | `ggml/src/ggml-cuda/mmq.cu:267-371` |
| RMSNorm+Residual fusion | `ggml/src/ggml-cuda/norm.cu:76-155` |
| Graph pattern matching | `ggml/src/ggml-cuda/ggml-cuda.cu:3623-3711` |
| Fusion dispatch (decode) | `ggml/src/ggml-cuda/ggml-cuda.cu:3998-4125` |
| `ncols_dst == 1` 约束 | `ggml/src/ggml-cuda/ggml-cuda.cu:2488` |
| `GGML_CUDA_DISABLE_FUSION` | `ggml/src/ggml-cuda/ggml-cuda.cu:3832` |
