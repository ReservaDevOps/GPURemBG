from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List

from .pipeline import BackgroundRemover, MattingConfig, MODEL_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPU-accelerated background remover with multiple matting backends.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where RGBA outputs will be written. Each model writes to its own sub-folder.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["u2net"],
        choices=list(MODEL_REGISTRY.keys()),
        help=(
            "List of models to benchmark. Available: "
            + ", ".join(MODEL_REGISTRY.keys())
        ),
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path("~/.cache/gpurembg").expanduser(),
        help="Directory used to cache downloaded model weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="CUDA device identifier, e.g. cuda:0.",
    )
    parser.add_argument(
        "--fp16",
        dest="use_fp16",
        action="store_true",
        default=True,
        help="Enable automatic mixed precision for Tensor Core acceleration (default).",
    )
    parser.add_argument(
        "--no-fp16",
        dest="use_fp16",
        action="store_false",
        help="Disable FP16 inference.",
    )
    parser.add_argument(
        "--alpha-threshold",
        type=float,
        default=None,
        help="Optional hard threshold [0,1] applied to the alpha matte.",
    )
    parser.add_argument(
        "--refine-dilate",
        type=int,
        default=0,
        help="Optional number of 3x3 dilation iterations applied after thresholding.",
    )
    parser.add_argument(
        "--refine-feather",
        type=int,
        default=0,
        help="Optional gaussian blur radius (pixels) to feather mask edges.",
    )
    parser.add_argument(
        "--no-refine",
        dest="refine_foreground",
        action="store_false",
        help="Skip post-processing and only attach the predicted alpha channel.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite outputs even if the file already exists.",
    )
    parser.add_argument(
        "--json",
        dest="json_report",
        type=Path,
        default=None,
        help="Optional path to write a JSON benchmarking report.",
    )
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    args.input_dir = args.input_dir.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.weights_dir = args.weights_dir.expanduser()

    if not args.input_dir.exists():
        raise SystemExit(f"Input directory {args.input_dir} does not exist.")

    results: Dict[str, Dict[str, float]] = {}

    for model_name in args.models:
        print(f"[+] Running model: {model_name}")
        config = MattingConfig(
            model_name=model_name,
            weights_dir=args.weights_dir,
            use_fp16=args.use_fp16,
            device=args.device,
            alpha_threshold=args.alpha_threshold,
            refine_foreground=args.refine_foreground,
            refine_dilate=max(0, args.refine_dilate),
            refine_feather=max(0, args.refine_feather),
        )
        remover = BackgroundRemover(config)
        model_output = args.output_dir / model_name
        model_output.mkdir(parents=True, exist_ok=True)
        timings = remover.process_directory(
            args.input_dir,
            model_output,
            overwrite=args.overwrite,
        )
        if not timings:
            print("    No images processed (perhaps outputs already exist?).")
            continue

        total_time = sum(timings.values())
        avg_time = mean(timings.values())
        print(
            f"    Processed {len(timings)} images | total {total_time:.2f}s | avg {avg_time:.3f}s"
        )
        results[model_name] = {
            "images": len(timings),
            "total_seconds": total_time,
            "avg_seconds": avg_time,
        }

    if args.json_report and results:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        with args.json_report.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"[+] Wrote benchmarking report to {args.json_report}")


if __name__ == "__main__":
    run()
