# SPDX-License-Identifier: MIT
r"""``python -m herbert_nn.preprocess`` -- raw JSONL sessions -> cached tensors.

Parses every ``*.jsonl`` session log under ``--raw-dir``, validates it
against the BridgeLogger schema, engineers features, fits normalization
statistics and categorical vocabularies on a session-level training split,
and writes gzip-compressed tensors + a manifest to a content-hashed
directory under ``--cache-dir``. Re-running with the same raw files and
config reuses the existing cache; use ``--force-rebuild`` to recompute.

Example:
    python -m herbert_nn.preprocess --raw-dir data/raw --cache-dir data/cache \
        --window-length 32
"""

from __future__ import annotations

import argparse
import logging

from herbert_nn.data.cache import build_or_load_cache
from herbert_nn.data.config import PreprocessConfig
from herbert_nn.logging_utils import configure_logging

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess raw BridgeLogger .jsonl session files into a cached, "
            "normalized tensor dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        required=True,
        help="Directory containing raw *.jsonl session log files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/cache",
        help="Root directory under which the content-hashed cache is written.",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=32,
        help="GRU sliding-window length in ticks (also stored in the cache manifest).",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Stride, in ticks, between window starts.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training split fraction of sessions.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split fraction of sessions.",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="Test split fraction of sessions."
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Seed controlling the session train/val/test shuffle.",
    )
    parser.add_argument(
        "--block-grid-width",
        type=int,
        default=None,
        help="Expected block_grid width; auto-detected if omitted.",
    )
    parser.add_argument(
        "--block-grid-height",
        type=int,
        default=None,
        help="Expected block_grid height; auto-detected if omitted.",
    )
    parser.add_argument(
        "--block-grid-depth",
        type=int,
        default=None,
        help="Expected block_grid depth; auto-detected if omitted.",
    )
    parser.add_argument(
        "--item-type-vocab-size",
        type=int,
        default=128,
        help="Max vocabulary size for held_item.item_id.",
    )
    parser.add_argument(
        "--kit-type-vocab-size",
        type=int,
        default=32,
        help="Max vocabulary size for match.kit.",
    )
    parser.add_argument(
        "--place-block-type-vocab-size",
        type=int,
        default=32,
        help="Max vocabulary size for input.place_block_type (BlockPlacementHead classes).",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Ignore any existing cache for this config and rebuild from raw data.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Python logging level, e.g. DEBUG, INFO, WARNING.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level.upper()))

    config = PreprocessConfig(
        raw_dir=args.raw_dir,
        cache_dir=args.cache_dir,
        window_length=args.window_length,
        window_stride=args.window_stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        split_seed=args.split_seed,
        block_grid_width=args.block_grid_width,
        block_grid_height=args.block_grid_height,
        block_grid_depth=args.block_grid_depth,
        item_type_vocab_size=args.item_type_vocab_size,
        kit_type_vocab_size=args.kit_type_vocab_size,
        place_block_type_vocab_size=args.place_block_type_vocab_size,
    )
    bundle = build_or_load_cache(config, force_rebuild=args.force_rebuild)
    logger.info(
        "Cache ready at %s | schema_version=%s | counts=%s | block_grid_shape=%s",
        bundle.cache_path,
        bundle.manifest.schema_version,
        bundle.manifest.counts,
        bundle.manifest.block_grid_shape,
    )


if __name__ == "__main__":
    main()
