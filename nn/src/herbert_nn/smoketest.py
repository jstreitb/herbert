# SPDX-License-Identifier: MIT
"""``python -m herbert_nn.smoketest`` -- fast end-to-end pipeline validation.

Runs one epoch over 200 randomly-generated (but shape/dtype-correct)
synthetic samples through a real (tiny) model, a real forward + backward
pass, the real composite loss, and a real checkpoint save -- to confirm the
whole model/loss/training-engine wiring is correct without needing any real
recorded session data or a preprocessing pass. Designed to finish in well
under 60 seconds on CPU alone.

Example:
    python -m herbert_nn.smoketest
    python -m herbert_nn.smoketest --model gru --output-dir /tmp/smoketest
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from torch.optim import AdamW
from torch.utils.data import DataLoader

from herbert_nn.logging_utils import configure_logging
from herbert_nn.models.base import build_model
from herbert_nn.models.losses import CompositeLoss
from herbert_nn.training.checkpoint import save_checkpoint
from herbert_nn.training.engine import run_epoch
from herbert_nn.training.seed import set_seed
from herbert_nn.training.smoke import (
    SMOKE_WINDOW_LENGTH,
    SyntheticDataset,
    smoke_data_meta,
)
from herbert_nn.training.train_loop import resolve_device

logger = logging.getLogger(__name__)

_MLP_MODEL_CFG = {
    "family": "mlp",
    "hidden_dims": [32, 16],
    "dropout": 0.1,
    "block_cell_embed_dim": 4,
    "item_type_embed_dim": 8,
    "kit_type_embed_dim": 4,
    "held_item_embed_dim": 4,
    "hotbar_slot_embed_dim": 4,
}
_GRU_MODEL_CFG = {
    "family": "gru",
    "hidden_size": 32,
    "num_layers": 1,
    "dropout": 0.0,
    "trunk_hidden_dims": [16],
    "block_cell_embed_dim": 4,
    "item_type_embed_dim": 8,
    "kit_type_embed_dim": 4,
    "held_item_embed_dim": 4,
    "hotbar_slot_embed_dim": 4,
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this CLI."""
    parser = argparse.ArgumentParser(
        description="Fast end-to-end smoke test of the herbert_nn model/loss/training pipeline "
        "on synthetic data (no real session data required).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="mlp",
        choices=["mlp", "gru"],
        help="Which policy family to smoke-test.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of synthetic samples for the epoch.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--output-dir",
        default="runs/smoketest",
        help="Directory to write the checkpoint into.",
    )
    parser.add_argument(
        "--device", default="cpu", help='Device to run on: "auto", "cpu", or "cuda".'
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level.upper()))

    start = time.perf_counter()
    set_seed(args.seed, deterministic_cudnn=False)
    device = resolve_device(args.device)

    data_meta = smoke_data_meta()
    window_length = SMOKE_WINDOW_LENGTH if args.model == "gru" else None
    dataset = SyntheticDataset(args.num_samples, data_meta, window_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    model_cfg = _GRU_MODEL_CFG if args.model == "gru" else _MLP_MODEL_CFG
    model = build_model(model_cfg, data_meta).to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    loss_fn = CompositeLoss().to(device)

    logger.info(
        "Smoke-testing %s policy: %d samples, batch_size=%d, device=%s, %d params",
        args.model,
        len(dataset),
        args.batch_size,
        device,
        sum(p.numel() for p in model.parameters()),
    )

    train_metrics = run_epoch(
        model,
        loader,
        loss_fn,
        device,
        optimizer=optimizer,
        amp=False,
        grad_accum_steps=1,
        grad_clip_norm=1.0,
        is_train=True,
    )
    eval_metrics = run_epoch(model, loader, loss_fn, device, amp=False, is_train=False)

    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / "smoketest.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        model_cfg,
        data_meta,
        cache_path="<synthetic-smoketest-data>",
        epoch=1,
        global_step=len(loader),
        best_val_loss=eval_metrics["total"],
        optimizer=optimizer,
    )

    elapsed = time.perf_counter() - start
    logger.info(
        "SMOKE TEST PASSED (%s): train_loss=%.5f eval_loss=%.5f | checkpoint=%s | elapsed=%.2fs",
        args.model,
        train_metrics["total"],
        eval_metrics["total"],
        checkpoint_path,
        elapsed,
    )
    if elapsed > 60:
        logger.warning(
            "Smoke test took %.1fs (>60s) -- this environment is unusually slow, or something "
            "is wrong with the fast path.",
            elapsed,
        )


if __name__ == "__main__":
    main()
