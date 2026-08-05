# SPDX-License-Identifier: MIT
r"""``python -m herbert_nn.evaluate`` -- load a checkpoint, score it on a held-out split.

Loads the model architecture and weights from a checkpoint (as written by
``herbert_nn.train``), reloads the exact preprocessing cache it was trained
against, and computes per-head metrics (mouse MAE, discrete accuracy/F1/AUC,
block-placement top-1/top-3 accuracy) on the requested split. Metrics are
written as JSON, by default alongside the checkpoint.

Example:
    python -m herbert_nn.evaluate --checkpoint runs/default/2026-08-04_12-00-00/best.pt \
        --split test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from herbert_nn.data.cache import load_cache_bundle
from herbert_nn.data.dataset import build_dataset
from herbert_nn.eval.metrics import collect_predictions, compute_metrics
from herbert_nn.logging_utils import configure_logging
from herbert_nn.training.checkpoint import build_model_from_checkpoint, load_checkpoint
from herbert_nn.training.train_loop import resolve_device

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this CLI."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained herbert_nn checkpoint on a held-out data split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .pt checkpoint file."
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate on.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write metrics.json into. Defaults to the checkpoint's parent directory.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Evaluation batch size."
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="DataLoader worker processes."
    )
    parser.add_argument(
        "--device", default="auto", help='Device to run on: "auto", "cpu", or "cuda".'
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=None,
        help="Override the GRU window length used for evaluation (defaults to the cache's configured value).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=None,
        help="Override the GRU window stride used for evaluation (defaults to the cache's configured value).",
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level.upper()))

    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint(checkpoint_path)
    model = build_model_from_checkpoint(checkpoint)
    device = resolve_device(args.device)
    model.to(device)

    cache_bundle = load_cache_bundle(checkpoint["cache_path"])
    resolved_data_cfg = cache_bundle.manifest.resolved_config
    # `or` would treat an explicit `--window-length 0` the same as "not passed", falling
    # back to the cache's configured value instead of the (invalid, but explicit) 0.
    window_length = (
        args.window_length
        if args.window_length is not None
        else int(resolved_data_cfg["window_length"])
    )
    window_stride = (
        args.window_stride
        if args.window_stride is not None
        else int(resolved_data_cfg["window_stride"])
    )

    family = checkpoint["model_cfg"]["family"]
    tensors = cache_bundle.load_split(args.split)
    boundaries = cache_bundle.manifest.split_boundaries[args.split]
    dataset = build_dataset(tensors, boundaries, family, window_length, window_stride)
    if len(dataset) == 0:
        raise RuntimeError(
            f"The {args.split!r} split produced 0 samples for family={family!r}."
        )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    logger.info(
        "Evaluating %s checkpoint on split=%s (%d samples)...",
        family,
        args.split,
        len(dataset),
    )
    collected = collect_predictions(model, loader, device)
    metrics = compute_metrics(collected)
    metrics["_meta"] = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": len(dataset),
        "model_family": family,
    }

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"metrics_{args.split}.json"
    output_path.write_text(json.dumps(metrics, indent=2))
    logger.info("Wrote evaluation metrics to %s", output_path)
    logger.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
