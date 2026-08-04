"""Full training loop orchestration, driven by a resolved Hydra config.

Shared by the ``herbert_nn.train`` CLI. Builds the cache/datasets/model/
optimizer/scheduler, runs the epoch loop with early stopping, and writes
checkpoints + TensorBoard logs + a final ``metrics.json`` into the Hydra run
directory.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, cast

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from herbert_nn.data.cache import build_or_load_cache
from herbert_nn.data.config import PreprocessConfig
from herbert_nn.data.dataset import build_dataset
from herbert_nn.models.base import DataMeta, build_model
from herbert_nn.models.losses import CompositeLoss
from herbert_nn.training.checkpoint import save_checkpoint
from herbert_nn.training.early_stopping import EarlyStopping
from herbert_nn.training.engine import run_epoch
from herbert_nn.training.scheduler import build_lr_scheduler
from herbert_nn.training.seed import set_seed

logger = logging.getLogger(__name__)


def resolve_device(requested: str) -> torch.device:
    """Resolve the ``device`` config value ("auto"/"cpu"/"cuda") to a concrete device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def data_meta_from_manifest(manifest: Any) -> DataMeta:
    """Build a :class:`DataMeta` from a :class:`herbert_nn.data.cache.CacheManifest`."""
    return DataMeta(
        block_grid_shape=manifest.block_grid_shape,
        item_type_vocab_size=manifest.item_type_vocab.size,
        kit_type_vocab_size=manifest.kit_type_vocab.size,
        place_block_type_vocab_size=manifest.place_block_type_vocab.size,
    )


def run_training(cfg: DictConfig, run_dir: Path) -> dict[str, Any]:
    """Execute a full training run described by ``cfg``.

    Args:
        cfg: The fully-composed Hydra config (``config.yaml`` + overrides).
        run_dir: Directory to write checkpoints, TensorBoard logs, and
            ``metrics.json`` into (typically Hydra's own run output dir).

    Returns:
        The summary dict also written to ``run_dir / "metrics.json"``.
    """
    set_seed(int(cfg.seed))
    device = resolve_device(cfg.device)
    logger.info("Training on device=%s", device)

    data_cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg.data, resolve=True))
    preprocess_cfg = PreprocessConfig(**data_cfg_dict)
    cache_bundle = build_or_load_cache(preprocess_cfg)
    data_meta = data_meta_from_manifest(cache_bundle.manifest)

    family = cfg.model.family
    train_tensors = cache_bundle.load_split("train")
    val_tensors = cache_bundle.load_split("val")
    train_boundaries = cache_bundle.manifest.split_boundaries["train"]
    val_boundaries = cache_bundle.manifest.split_boundaries["val"]

    train_dataset = build_dataset(
        train_tensors, train_boundaries, family, cfg.data.window_length, cfg.data.window_stride
    )
    val_dataset = build_dataset(
        val_tensors, val_boundaries, family, cfg.data.window_length, cfg.data.window_stride
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError(
            f"Empty train ({len(train_dataset)}) or val ({len(val_dataset)}) dataset for "
            f"family={family!r}; with too few ticks per session and window_length="
            f"{cfg.data.window_length}, no windows may be constructible. Reduce window_length "
            "or record longer sessions."
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=True,
        num_workers=int(cfg.training.num_workers),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg.training.batch_size),
        shuffle=False,
        num_workers=int(cfg.training.num_workers),
        drop_last=False,
    )

    model_cfg_dict = cast(dict[str, Any], OmegaConf.to_container(cfg.model, resolve=True))
    model = build_model(model_cfg_dict, data_meta).to(device)
    logger.info(
        "Model (%s): %d trainable parameters",
        family,
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.training.optimizer.lr),
        weight_decay=float(cfg.training.optimizer.weight_decay),
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / max(1, cfg.training.grad_accum_steps)))
    total_steps = steps_per_epoch * int(cfg.training.epochs)
    scheduler = build_lr_scheduler(
        optimizer,
        warmup_steps=int(cfg.training.warmup_steps),
        total_steps=total_steps,
        min_lr_ratio=float(cfg.training.min_lr_ratio),
    )
    use_amp = bool(cfg.training.amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device="cuda", enabled=use_amp)

    loss_fn = CompositeLoss(
        mouse_weight=float(cfg.training.loss_weights.mouse),
        discrete_weight=float(cfg.training.loss_weights.discrete),
        block_placement_weight=float(cfg.training.loss_weights.block_placement),
        huber_delta=float(cfg.training.huber_delta),
    ).to(device)

    early_stopping = EarlyStopping(
        patience=int(cfg.training.early_stopping_patience),
        min_delta=float(cfg.training.early_stopping_min_delta),
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    wandb_run = _maybe_init_wandb(cfg, run_dir)

    history: list[dict[str, Any]] = []
    global_step = 0
    final_epoch = 0
    for epoch in range(1, int(cfg.training.epochs) + 1):
        final_epoch = epoch
        train_metrics = run_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            amp=use_amp,
            grad_accum_steps=int(cfg.training.grad_accum_steps),
            grad_clip_norm=cfg.training.grad_clip_norm,
            is_train=True,
        )
        val_metrics = run_epoch(model, val_loader, loss_fn, device, amp=use_amp, is_train=False)
        global_step += len(train_loader)

        logger.info(
            "epoch %d/%d | train_loss=%.5f val_loss=%.5f | train(mouse=%.5f discrete=%.5f block=%.5f) "
            "val(mouse=%.5f discrete=%.5f block=%.5f)",
            epoch,
            int(cfg.training.epochs),
            train_metrics["total"],
            val_metrics["total"],
            train_metrics["mouse"],
            train_metrics["discrete"],
            train_metrics["block_placement"],
            val_metrics["mouse"],
            val_metrics["discrete"],
            val_metrics["block_placement"],
        )
        for split_name, metrics in (("train", train_metrics), ("val", val_metrics)):
            for k, v in metrics.items():
                writer.add_scalar(f"{split_name}/{k}_loss", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)
        if wandb_run is not None:
            wandb_run.log(
                {f"train/{k}": v for k, v in train_metrics.items()}
                | {f"val/{k}": v for k, v in val_metrics.items()}
                | {"epoch": epoch, "lr": optimizer.param_groups[0]["lr"]}
            )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        is_best = early_stopping.step(val_metrics["total"])
        cache_path_str = str(cache_bundle.cache_path)
        save_checkpoint(
            run_dir / "last.pt",
            model,
            model_cfg_dict,
            data_meta,
            cache_path_str,
            epoch,
            global_step,
            early_stopping.best_value,
            optimizer=optimizer,
            extra={"resolved_config": OmegaConf.to_container(cfg, resolve=True)},
        )
        if is_best:
            save_checkpoint(
                run_dir / "best.pt",
                model,
                model_cfg_dict,
                data_meta,
                cache_path_str,
                epoch,
                global_step,
                early_stopping.best_value,
                optimizer=None,
                extra={"resolved_config": OmegaConf.to_container(cfg, resolve=True)},
            )

        if early_stopping.should_stop:
            logger.info("Stopping early at epoch %d.", epoch)
            break

    writer.close()
    if wandb_run is not None:
        wandb_run.finish()

    summary = {
        "experiment_name": str(cfg.experiment_name),
        "model_family": family,
        "epochs_run": final_epoch,
        "best_val_loss": early_stopping.best_value,
        "final_train_metrics": history[-1]["train"] if history else {},
        "final_val_metrics": history[-1]["val"] if history else {},
        "history": history,
        "cache_path": str(cache_bundle.cache_path),
        "run_dir": str(run_dir),
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    logger.info("Training complete. Summary written to %s", run_dir / "metrics.json")
    return summary


def _maybe_init_wandb(cfg: DictConfig, run_dir: Path) -> Any:
    """Guarded, optional Weights & Biases init. Returns ``None`` if disabled/unavailable."""
    if not bool(cfg.training.use_wandb):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning(
            "training.use_wandb=true but the 'wandb' package is not installed "
            "(install the 'wandb' extra: pip install -e '.[wandb]'). Skipping W&B logging."
        )
        return None
    run = wandb.init(
        project=str(cfg.training.wandb_project),
        name=f"{cfg.experiment_name}-{run_dir.name}",
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(run_dir),
    )
    return run
