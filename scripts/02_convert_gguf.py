#!/usr/bin/env python3
"""Week 2: HF → GGUF 转换 + 多精度量化。

流程:
  1. HF FP16 → GGUF FP16 (via llama.cpp convert_hf_to_gguf.py)
  2. GGUF FP16 → Q8_0 / Q5_K_M / Q4_K_M / Q4_0 / IQ4_NL (via llama-quantize)

用法:
    conda run -n pytorch-env python scripts/02_convert_gguf.py
    conda run -n pytorch-env python scripts/02_convert_gguf.py --skip-fp16  # 只做量化
    conda run -n pytorch-env python scripts/02_convert_gguf.py --quants Q8_0,Q4_K_M  # 只选几种

产出:
    models/gguf/
      qwen2.5-7b-fp16.gguf     (~14 GB)
      qwen2.5-7b-Q8_0.gguf     (~7.6 GB)
      qwen2.5-7b-Q5_K_M.gguf   (~5.1 GB)
      qwen2.5-7b-Q4_K_M.gguf   (~4.4 GB)
      qwen2.5-7b-Q4_0.gguf     (~3.9 GB)
      qwen2.5-7b-IQ4_NL.gguf   (~4.2 GB)
"""

import argparse
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_MODEL_DIR = os.path.join(
    PROJECT_ROOT, "models", "qwen2.5-7b-instruct", "Qwen", "Qwen2.5-7B-Instruct"
)
GGUF_OUT_DIR = os.path.join(PROJECT_ROOT, "models", "gguf")
LLAMA_CPP_DIR = os.environ.get("LLAMA_CPP_DIR", os.path.join(PROJECT_ROOT, "..", "llama.cpp"))
CONVERT_SCRIPT = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
QUANTIZE_BIN = os.path.join(LLAMA_CPP_DIR, "build", "bin", "llama-quantize")

QUANT_CONFIGS = {
    "Q8_0":   {"desc": "8-bit per-channel symmetric",       "est_gb": 7.6},
    "Q5_K_M": {"desc": "5-bit K-Quants mixed precision",    "est_gb": 5.1},
    "Q4_K_M": {"desc": "4-bit K-Quants mixed precision",    "est_gb": 4.4},
    "Q4_0":   {"desc": "4-bit per-block symmetric",         "est_gb": 3.9},
    "IQ4_NL": {"desc": "4-bit importance-aware non-linear", "est_gb": 4.2},
}


def parse_args():
    parser = argparse.ArgumentParser(description="HF → GGUF 转换 + 多精度量化")
    parser.add_argument("--hf-model-dir", default=HF_MODEL_DIR)
    parser.add_argument("--out-dir", default=GGUF_OUT_DIR)
    parser.add_argument("--llama-dir", default=LLAMA_CPP_DIR)
    parser.add_argument(
        "--quants",
        default="Q8_0,Q5_K_M,Q4_K_M,Q4_0,IQ4_NL",
        help="逗号分隔的量化类型",
    )
    parser.add_argument("--skip-fp16", action="store_true", help="跳过 FP16 转换")
    parser.add_argument("--skip-quant", action="store_true", help="跳过量化步骤")
    return parser.parse_args()


def find_gguf_scripts():
    """找到 llama.cpp 的 gguf-py 和相关脚本。"""
    gguf_py = os.path.join(LLAMA_CPP_DIR, "gguf-py")
    if os.path.isdir(gguf_py):
        sys.path.insert(0, str(gguf_py))
    else:
        print(f"WARNING: gguf-py not found at {gguf_py}, conversion may fail")


def run_cmd(cmd: list[str], desc: str) -> bool:
    print(f"[{desc}]")
    print(f"  {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=LLAMA_CPP_DIR)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"  OK ({elapsed:.1f}s)\n")
        return True
    else:
        print(f"  FAILED (exit={result.returncode}, {elapsed:.1f}s)\n")
        return False


def convert_hf_to_gguf_fp16(hf_dir: str, out_path: str) -> bool:
    """用 llama.cpp 的 convert_hf_to_gguf.py 把 HF 模型转成 GGUF FP16。"""
    if os.path.exists(out_path):
        size_gb = os.path.getsize(out_path) / 1024**3
        print(f"FP16 GGUF 已存在: {out_path} ({size_gb:.1f} GB)，跳过转换\n")
        return True

    print("=" * 60)
    print("Step 1: HF → GGUF FP16")
    print("=" * 60)
    print(f"  源: {hf_dir}")
    print(f"  目标: {out_path}")

    # convert_hf_to_gguf.py expects: model_dir --outtype f16 --outfile output.gguf
    cmd = [
        sys.executable,
        CONVERT_SCRIPT,
        hf_dir,
        "--outtype", "f16",
        "--outfile", out_path,
    ]
    return run_cmd(cmd, "convert_hf_to_gguf FP16")


def quantize_model(fp16_path: str, quant_type: str, out_path: str) -> bool:
    """对 GGUF FP16 模型进行量化。"""
    if os.path.exists(out_path):
        size_gb = os.path.getsize(out_path) / 1024**3
        print(f"  [{quant_type}] 已存在 ({size_gb:.1f} GB)，跳过")
        return True

    cmd = [QUANTIZE_BIN, fp16_path, out_path, quant_type]
    return run_cmd(cmd, f"quantize {quant_type}")


def main():
    args = parse_args()
    quants = [q.strip() for q in args.quants.split(",") if q.strip()]

    if not os.path.isdir(args.hf_model_dir):
        print(f"错误: HF 模型路径不存在: {args.hf_model_dir}")
        sys.exit(1)
    if not os.path.isfile(CONVERT_SCRIPT):
        print(f"错误: 找不到 convert_hf_to_gguf.py: {CONVERT_SCRIPT}")
        sys.exit(1)
    if not os.path.isfile(QUANTIZE_BIN):
        print(f"错误: 找不到 llama-quantize: {QUANTIZE_BIN}")
        print("请先编译 llama.cpp: cd llama.cpp && mkdir build && cd build && cmake -DGGML_CUDA=ON .. && make -j")
        sys.exit(1)

    # 确保 gguf-py 在 PYTHONPATH 中
    find_gguf_scripts()

    os.makedirs(args.out_dir, exist_ok=True)

    fp16_path = os.path.join(args.out_dir, "qwen2.5-7b-fp16.gguf")

    results = {
        "hf_model": args.hf_model_dir,
        "gguf_dir": args.out_dir,
        "fp16_path": fp16_path,
        "quantizations": [],
    }

    # Step 1: HF → GGUF FP16
    if not args.skip_fp16:
        ok = convert_hf_to_gguf_fp16(args.hf_model_dir, fp16_path)
        if not ok:
            print("FP16 转换失败，终止")
            sys.exit(1)
        results["fp16_size_gb"] = round(os.path.getsize(fp16_path) / 1024**3, 2)
    else:
        print("跳过 FP16 转换 (--skip-fp16)\n")

    if not os.path.isfile(fp16_path):
        print(f"错误: FP16 GGUF 文件不存在: {fp16_path}")
        sys.exit(1)

    # Step 2: 多精度量化
    if not args.skip_quant:
        print("=" * 60)
        print(f"Step 2: GGUF FP16 → {len(quants)} 种量化精度")
        print("=" * 60)
        fp16_size = os.path.getsize(fp16_path)
        print(f"  源: {fp16_path} ({fp16_size/1024**3:.1f} GB)\n")

        for qt in quants:
            config = QUANT_CONFIGS.get(qt, {"desc": "unknown", "est_gb": "?"})
            out_name = f"qwen2.5-7b-{qt}.gguf"
            out_path = os.path.join(args.out_dir, out_name)

            print(f"--- {qt} ({config['desc']}), 预计 {config['est_gb']} GB ---")
            ok = quantize_model(fp16_path, qt, out_path)
            if ok:
                actual_gb = os.path.getsize(out_path) / 1024**3
                results["quantizations"].append({
                    "type": qt,
                    "desc": config["desc"],
                    "est_gb": config["est_gb"],
                    "actual_gb": round(actual_gb, 2),
                    "path": out_path,
                })
                print(f"  实际大小: {actual_gb:.2f} GB\n")
            else:
                print(f"  {qt} 量化失败!\n")
    else:
        print("跳过量化 (--skip-quant)\n")

    # 汇总
    print("=" * 60)
    print("转换 & 量化完成")
    print("=" * 60)
    print(f"{'类型':<10} {'预计':>8} {'实际':>8} {'状态':>8}")
    print("-" * 40)
    for q in results["quantizations"]:
        print(f"{q['type']:<10} {q['est_gb']:>6.1f} GB {q['actual_gb']:>6.2f} GB {'✓':>8}")
    print()

    # 保存结果
    result_path = os.path.join(PROJECT_ROOT, "output", "convert_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已保存: {result_path}")


if __name__ == "__main__":
    main()
