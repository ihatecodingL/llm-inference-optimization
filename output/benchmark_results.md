# Qwen2.5-7B 量化 Benchmark 结果

GPU: NVIDIA GeForce RTX 3060 12GB, n_gpu_layers: 99 (FP16 用 CPU -ngl 0, 其余用 GPU)

## 核心结论：Q4_K_M 是最佳权衡点

FP16 权重 14.9 GB，RTX 3060 只有 12 GB 显存，**无法加载**。
Q4_K_M 将模型压缩至 4.4 GB，PPL 仅损失 0.12（7.72 → 7.84），decode 速度从无法运行提升至 61 tok/s。

![PPL vs Decode Speed Tradeoff](figures/01_ppl_vs_speed.png)

![Model Size & PPL](figures/02_model_size_ppl.png)

## 完整对比表

| Model | File Size | PPL (↓) | Δ PPL | Prefill | Decode (↑) | ms/tok |
|-------|----------|---------|-------|---------|-------------|--------|
| FP16  | 14.2 GB  | 7.7206  | 基准  | —       | —           | —      |
| Q8_0  | 7.5 GB   | 7.7256  | +0.005 | 2124 tok/s | 39.3 tok/s | 25.4 ms |
| Q5_K_M | 5.1 GB  | 7.7744  | +0.054 | 1978 tok/s | 54.8 tok/s | 18.2 ms |
| Q4_K_M | 4.4 GB  | 7.8435  | +0.123 | 1995 tok/s | **61.0 tok/s** | 16.4 ms |
| Q4_0   | 4.1 GB  | 7.9947  | +0.274 | 2193 tok/s | **67.0 tok/s** | 14.9 ms |
| IQ4_NL | 4.2 GB  | 7.9399  | +0.219 | 2178 tok/s | 65.9 tok/s | 15.2 ms |

> **FP16 说明**: 14.9 GB > 12 GB GPU 显存，加载即 OOM。PPL 使用 CPU 推理（-ngl 0）测得，作为质量参考基线。速度无法测量（GPU 加载失败）。

## 解读

- **Q8_0**: PPL 几乎无损（Δ=0.005），但体积仍然较大（7.5 GB）、速度最慢（39 tok/s）。适合对精度要求极高的场景。
- **Q5_K_M**: 体积 5.1 GB，PPL 多 0.05，速度提升 39%。不错的中间选项。
- **Q4_K_M** ✦: PPL 多 0.12（1.6% 相对损失），体积省 69%，速度翻倍。**最佳性价比。**
- **Q4_0**: 最快（67 tok/s），但 PPL 损失最大（0.27）。适合对速度敏感、对质量容忍度高的场景。
- **IQ4_NL**: 介于 Q4_K_M 和 Q4_0 之间，importance-aware 量化在 PPL 上优于 Q4_0（7.94 vs 7.99），但体积稍大。

## 数据来源

- PPL: llama-perplexity, WikiText-2 test set (579 chunks), `--no-warmup`
- Speed: llama-bench, `-p 512 -n 128 -b 2048 -r 3`
- FP16 PPL 使用 CPU-only (`-ngl 0`)，耗时 ~17.5 分钟
