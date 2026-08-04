"""``python -m herbert_nn.train`` -- Hydra-driven full training run.

Examples:
    Train the MLP baseline with defaults::

        python -m herbert_nn.train data.raw_dir=data/raw

    Train the GRU variant with a custom learning rate and batch size::

        python -m herbert_nn.train model=gru training.optimizer.lr=1e-4 training.batch_size=128

    Use the example experiment config::

        python -m herbert_nn.train +experiment=gru_baseline data.raw_dir=data/raw

Every run writes to ``runs/{experiment_name}/{timestamp}/``: a snapshot of
the resolved config (written automatically by Hydra under ``.hydra/``),
``best.pt`` / ``last.pt`` checkpoints, TensorBoard event files, and a final
``metrics.json``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from herbert_nn.logging_utils import configure_logging
from herbert_nn.training.train_loop import run_training

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point: resolve the run directory and delegate to :func:`run_training`."""
    configure_logging()
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    logger.info("Hydra run dir: %s", run_dir)
    run_training(cfg, run_dir)


if __name__ == "__main__":
    main()
