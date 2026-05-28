#!/usr/bin/env python3
"""从 ModelScope 下载 Qwen2.5-7B-Instruct 原始模型（FP16）。

用法:
    python scripts/download_model.py                          # 默认路径 models/qwen2.5-7b-instruct/
    python scripts/download_model.py --model-dir /data/models/qwen2.5-7b/
    python scripts/download_model.py --model Qwen/Qwen2.5-1.5B-Instruct  # 下载其他模型

依赖: pip install modelscope
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "models", "qwen2.5-7b-instruct")


def parse_args():
    parser = argparse.ArgumentParser(description="从 ModelScope 下载模型")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ModelScope 模型 ID（默认: {DEFAULT_MODEL}）",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_DIR,
        help=f"本地保存路径（默认: {DEFAULT_DIR}）",
    )
    parser.add_argument(
        "--revision",
        default="master",
        help="模型分支/tag（默认: master）",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["*.pth", "*.safetensors"],
        help="排除的文件模式（默认: *.pth *.safetensors 单独下载以支持断点续传）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from modelscope import snapshot_download
    except ImportError:
        print("错误: 需要安装 modelscope，请运行: pip install modelscope")
        sys.exit(1)

    os.makedirs(args.model_dir, exist_ok=True)

    print(f"模型: {args.model}")
    print(f"保存路径: {args.model_dir}")
    print(f"分支: {args.revision}")
    print()

    snapshot_download(
        args.model,
        cache_dir=args.model_dir,
        revision=args.revision,
    )

    print(f"\n下载完成，模型文件位于: {args.model_dir}")


if __name__ == "__main__":
    main()
