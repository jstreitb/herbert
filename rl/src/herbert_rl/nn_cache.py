"""Read an `/nn` preprocessing cache's `manifest.json` without importing `herbert_nn`.

A `/nn` checkpoint's weights (in particular the `FeatureEncoder`'s embedding tables and the
`MouseHead`/`DiscreteHead` linear layers spliced into the PPO action head -- see
`policy/checkpoint_adapter.py`) were trained against inputs standardized/vocab-encoded by one
specific preprocessing cache. Feeding that checkpoint un-normalized or differently-vocab-encoded
features would silently produce garbage. `manifest.json` (written by
`herbert_nn.data.cache.build_or_load_cache`) is plain JSON, so we can load exactly the fitted
`Standardizer` and `CategoricalVocab` objects it embeds using `/rl`'s own hand-synced copies of
those classes (`normalization.py`, `vocab.py`) -- no dependency on the `herbert_nn` package
itself, only on the on-disk JSON format it happens to write, which is a much narrower and more
stable contract than the package's Python API.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from herbert_rl.normalization import Standardizer
from herbert_rl.vocab import CategoricalVocab

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NNCacheStats:
    """Everything `/rl`'s feature encoder needs from an `/nn` preprocessing cache."""

    block_grid_shape: tuple[int, int, int]
    standardizer: Standardizer
    item_type_vocab: CategoricalVocab
    kit_type_vocab: CategoricalVocab

    @property
    def num_block_cells(self) -> int:
        w, h, d = self.block_grid_shape
        return w * h * d


def load_nn_cache_stats(manifest_path: str | Path) -> NNCacheStats:
    """Load fitted normalization/vocab stats from an `/nn` cache's `manifest.json`.

    Args:
        manifest_path: Path to either the `manifest.json` file itself, or the cache directory
            containing it (e.g. the value embedded in a checkpoint's ``cache_path`` field --
            note the RL trainer must be pointed at a manifest that still exists on disk; a
            checkpoint's ``cache_path`` may reference a cache directory local to whichever
            machine trained it, so re-point `nn_cache_manifest_path` in `/rl/conf` if training
            RL on a different machine than the BC checkpoint was produced on).

    Returns:
        The parsed :class:`NNCacheStats`.

    Raises:
        FileNotFoundError: If no `manifest.json` is found at/under ``manifest_path``.
    """
    path = Path(manifest_path)
    if path.is_dir():
        path = path / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"No /nn cache manifest found at {path}. Set `nn_cache_manifest_path` in "
            "rl/conf/config.yaml (or via CLI override) to the manifest.json (or its parent "
            "cache directory) that the BC checkpoint being fine-tuned was trained against."
        )
    data = json.loads(path.read_text())
    stats = NNCacheStats(
        block_grid_shape=tuple(data["block_grid_shape"]),  # type: ignore[arg-type]
        standardizer=Standardizer.from_dict(data["standardizer"]),
        item_type_vocab=CategoricalVocab.from_dict(data["item_type_vocab"]),
        kit_type_vocab=CategoricalVocab.from_dict(data["kit_type_vocab"]),
    )
    logger.info(
        "Loaded /nn cache stats from %s: block_grid_shape=%s item_type_vocab=%d kit_type_vocab=%d",
        path,
        stats.block_grid_shape,
        stats.item_type_vocab.size,
        stats.kit_type_vocab.size,
    )
    return stats
