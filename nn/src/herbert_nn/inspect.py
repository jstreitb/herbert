# SPDX-License-Identifier: MIT
r"""``python -m herbert_nn.inspect`` -- qualitative replay inspector.

Given a single held-out session ``.jsonl`` file and a trained checkpoint,
runs the model tick-by-tick over that session and writes:

* A CSV with one row per tick: actual vs. predicted values for every head.
* A matplotlib figure comparing predicted vs. actual actions over time.

This is a qualitative sanity-checking tool, not a metrics tool (see
``herbert_nn.evaluate`` for aggregate held-out metrics) -- use it to eyeball
whether the policy's aim/timing/placement decisions look at all like the
recorded player's.

Example:
    python -m herbert_nn.inspect --session data/raw/session_0007.jsonl \
        --checkpoint runs/default/2026-08-04_12-00-00/best.pt --output-dir inspect_out
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import TypedDict, cast

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch

from herbert_nn.constants import DISCRETE_ACTION_NAMES
from herbert_nn.data.cache import encode_session_for_inference, load_cache_bundle
from herbert_nn.data.features import SessionArrays, encode_session_raw
from herbert_nn.data.vocab import CategoricalVocab
from herbert_nn.logging_utils import configure_logging
from herbert_nn.schemas.registry import load_session
from herbert_nn.schemas.v1_0_0 import TickRecordV1
from herbert_nn.training.checkpoint import build_model_from_checkpoint, load_checkpoint
from herbert_nn.training.train_loop import resolve_device

logger = logging.getLogger(__name__)

_FEATURE_KEYS = (
    "continuous",
    "block_grid_cells",
    "hotbar_slot_index",
    "hotbar_item_type",
    "opponent_held_item_category",
    "match_kit_type",
)


class SessionPredictions(TypedDict):
    """Per-tick predictions for one session, as produced by :func:`_predict_mlp`/:func:`_predict_gru`.

    ``mouse``/``discrete_prob``/``movement`` are always fully populated (NaN-filled for GRU
    ticks with no valid preceding window). ``block_logits`` is ``None`` only when a GRU
    session has no ticks long enough to form a single window (see :func:`_predict_gru`).
    """

    mouse: np.ndarray
    discrete_prob: np.ndarray
    block_logits: np.ndarray | None
    movement: np.ndarray


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argparse parser for this CLI."""
    parser = argparse.ArgumentParser(
        description="Replay inspector: compare a checkpoint's per-tick predictions against a "
        "held-out recorded session.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--session", required=True, help="Path to a single raw session .jsonl file."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to a .pt checkpoint file."
    )
    parser.add_argument(
        "--output-dir",
        default="inspect_out",
        help="Directory to write the CSV and figure into.",
    )
    parser.add_argument(
        "--device", default="auto", help='Device to run on: "auto", "cpu", or "cuda".'
    )
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    return parser


@torch.no_grad()
def _predict_mlp(
    model: torch.nn.Module, tensors: dict[str, torch.Tensor], device: torch.device
) -> SessionPredictions:
    """Predict every tick independently (MLPPolicy has no history requirement)."""
    batch = {k: tensors[k].to(device) for k in _FEATURE_KEYS}
    output = model(batch)
    return {
        "mouse": output["mouse"].cpu().numpy(),
        "discrete_prob": torch.sigmoid(output["discrete"]).cpu().numpy(),
        "block_logits": output["block_placement"].cpu().numpy(),
        "movement": output["movement"].cpu().numpy(),
    }


@torch.no_grad()
def _predict_gru(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    window_length: int,
) -> SessionPredictions:
    """Predict tick ``t`` for every ``t >= window_length - 1`` using the preceding window.

    Ticks before the first full window (``t < window_length - 1``) have no
    prediction and are filled with NaN.
    """
    num_ticks = tensors["continuous"].shape[0]
    mouse = np.full((num_ticks, 2), np.nan, dtype=np.float32)
    discrete_prob = np.full(
        (num_ticks, len(DISCRETE_ACTION_NAMES)), np.nan, dtype=np.float32
    )
    movement = np.full((num_ticks, 2), np.nan, dtype=np.float32)

    valid_ends = list(range(window_length - 1, num_ticks))
    if not valid_ends:
        logger.warning(
            "Session has %d ticks, shorter than window_length=%d; no GRU predictions possible.",
            num_ticks,
            window_length,
        )
        return {
            "mouse": mouse,
            "discrete_prob": discrete_prob,
            "block_logits": None,
            "movement": movement,
        }

    windows = {
        key: torch.stack(
            [tensors[key][end - window_length + 1 : end + 1] for end in valid_ends],
            dim=0,
        ).to(device)
        for key in _FEATURE_KEYS
    }
    output = model(windows)
    pred_mouse = output["mouse"].cpu().numpy()
    pred_discrete = torch.sigmoid(output["discrete"]).cpu().numpy()
    pred_block = output["block_placement"].cpu().numpy()
    pred_movement = output["movement"].cpu().numpy()
    block_logits = np.full((num_ticks, pred_block.shape[1]), np.nan, dtype=np.float32)
    for i, end in enumerate(valid_ends):
        mouse[end] = pred_mouse[i]
        discrete_prob[end] = pred_discrete[i]
        block_logits[end] = pred_block[i]
        movement[end] = pred_movement[i]
    return {
        "mouse": mouse,
        "discrete_prob": discrete_prob,
        "block_logits": block_logits,
        "movement": movement,
    }


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
    family = checkpoint["model_cfg"]["family"]

    cache_bundle = load_cache_bundle(checkpoint["cache_path"])
    manifest = cache_bundle.manifest

    session_path = Path(args.session)
    header, raw_records = load_session(session_path)
    # `load_session` is version-generic (returns `list[BaseModel]`); this
    # module only supports schema_version "1.0.0" today, matching
    # herbert_nn.data.features.encode_session_raw's signature.
    session_id: str = getattr(header, "session_id")  # noqa: B009
    records = cast(list[TickRecordV1], raw_records)
    logger.info(
        "Loaded session %s (%s) with %d ticks.", session_id, session_path, len(records)
    )

    raw_arrays = encode_session_raw(records, session_id, manifest.block_grid_shape)
    tensors = encode_session_for_inference(raw_arrays, manifest)

    place_block_type_vocab: CategoricalVocab = manifest.place_block_type_vocab

    preds: SessionPredictions
    if family == "mlp":
        preds = _predict_mlp(model, tensors, device)
    elif family == "gru":
        window_length = int(manifest.resolved_config["window_length"])
        preds = _predict_gru(model, tensors, device, window_length)
    else:
        raise ValueError(f"Unknown model family {family!r} in checkpoint.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{session_id}_{family}"
    csv_path = output_dir / f"{stem}.csv"
    fig_path = output_dir / f"{stem}.png"

    _write_csv(csv_path, records, raw_arrays, preds, place_block_type_vocab)
    _write_figure(fig_path, records, raw_arrays, preds, session_id)
    logger.info("Wrote replay CSV to %s and figure to %s", csv_path, fig_path)


def _write_csv(
    csv_path: Path,
    records: list[TickRecordV1],
    raw_arrays: SessionArrays,
    preds: SessionPredictions,
    place_block_type_vocab: CategoricalVocab,
) -> None:
    fieldnames = [
        "tick",
        "timestamp",
        "actual_d_yaw",
        "pred_d_yaw",
        "actual_d_pitch",
        "pred_d_pitch",
        "actual_forward",
        "pred_forward",
        "actual_strafe",
        "pred_strafe",
        *[f"actual_{name}" for name in DISCRETE_ACTION_NAMES],
        *[f"pred_{name}_prob" for name in DISCRETE_ACTION_NAMES],
        "actual_place_block_type",
        "pred_place_block_type_top1",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for i, record in enumerate(records):
            row = {
                "tick": record.tick,
                "timestamp": record.timestamp,
                "actual_d_yaw": raw_arrays.mouse_target[i, 0],
                "pred_d_yaw": preds["mouse"][i, 0],
                "actual_d_pitch": raw_arrays.mouse_target[i, 1],
                "pred_d_pitch": preds["mouse"][i, 1],
                "actual_forward": raw_arrays.movement_target[i, 0],
                "pred_forward": preds["movement"][i, 0],
                "actual_strafe": raw_arrays.movement_target[i, 1],
                "pred_strafe": preds["movement"][i, 1],
                "actual_place_block_type": raw_arrays.place_block_type_raw[i] or "",
            }
            for j, name in enumerate(DISCRETE_ACTION_NAMES):
                row[f"actual_{name}"] = int(raw_arrays.discrete_target[i, j])
                row[f"pred_{name}_prob"] = preds["discrete_prob"][i, j]
            block_logits = preds["block_logits"]
            if block_logits is not None and not np.isnan(block_logits[i]).all():
                top1_idx = int(np.nanargmax(block_logits[i]))
                row["pred_place_block_type_top1"] = place_block_type_vocab.decode(
                    top1_idx
                )
            else:
                row["pred_place_block_type_top1"] = ""
            writer.writerow(row)


def _write_figure(
    fig_path: Path,
    records: list[TickRecordV1],
    raw_arrays: SessionArrays,
    preds: SessionPredictions,
    session_id: str,
) -> None:
    ticks = [r.tick for r in records]
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), sharex=True)

    axes[0].plot(ticks, raw_arrays.mouse_target[:, 0], label="actual d_yaw", alpha=0.7)
    axes[0].plot(
        ticks, preds["mouse"][:, 0], label="pred d_yaw", alpha=0.7, linestyle="--"
    )
    axes[0].set_ylabel("d_yaw")
    axes[0].legend(loc="upper right")
    axes[0].set_title(f"Replay inspection: session {session_id}")

    axes[1].plot(
        ticks, raw_arrays.mouse_target[:, 1], label="actual d_pitch", alpha=0.7
    )
    axes[1].plot(
        ticks, preds["mouse"][:, 1], label="pred d_pitch", alpha=0.7, linestyle="--"
    )
    axes[1].set_ylabel("d_pitch")
    axes[1].legend(loc="upper right")

    for j, name in enumerate(DISCRETE_ACTION_NAMES):
        axes[2].plot(
            ticks, preds["discrete_prob"][:, j], label=f"pred {name} prob", alpha=0.6
        )
        actual = raw_arrays.discrete_target[:, j]
        active_ticks = [t for t, a in zip(ticks, actual, strict=True) if a > 0.5]
        axes[2].scatter(
            active_ticks,
            [1.02 + 0.05 * j] * len(active_ticks),
            marker="|",
            s=40,
            label=f"actual {name}",
        )
    axes[2].set_ylabel("action probability / actual (markers)")
    axes[2].legend(loc="upper right", fontsize="x-small", ncol=2)

    axes[3].plot(
        ticks, raw_arrays.movement_target[:, 0], label="actual forward", alpha=0.7
    )
    axes[3].plot(
        ticks, preds["movement"][:, 0], label="pred forward", alpha=0.7, linestyle="--"
    )
    axes[3].plot(
        ticks, raw_arrays.movement_target[:, 1], label="actual strafe", alpha=0.7
    )
    axes[3].plot(
        ticks, preds["movement"][:, 1], label="pred strafe", alpha=0.7, linestyle="--"
    )
    axes[3].set_ylabel("movement (-1/0/1)")
    axes[3].set_xlabel("tick")
    axes[3].legend(loc="upper right", fontsize="x-small", ncol=2)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
