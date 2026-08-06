# SPDX-License-Identifier: MIT
"""Preprocessing configuration dataclass, shared by the CLI, Hydra configs, and cache hashing."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class PreprocessConfig:
    """Fully-resolved preprocessing configuration.

    An instance of this class is built from the Hydra ``data`` config group
    (see ``conf/data/default.yaml``) or directly via argparse for the
    standalone ``herbert_nn.preprocess`` CLI. It is the single object hashed
    (together with the schema version and the discovered raw file list) to
    produce the cache directory name -- see :mod:`herbert_nn.data.cache`.
    """

    raw_dir: str
    """Directory containing raw ``*.jsonl`` session log files."""

    cache_dir: str = "data/cache"
    """Root directory under which content-hashed preprocessing caches are stored."""

    window_length: int = 32
    """Number of consecutive ticks in each GRU sliding-window sample."""

    window_stride: int = 1
    """Step (in ticks) between consecutive window start positions."""

    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42

    block_grid_width: int | None = None
    block_grid_height: int | None = None
    block_grid_depth: int | None = None
    """Expected block-grid dimensions. If any is ``None``, the shape observed
    on the very first tick of the first session encountered is adopted as the
    canonical shape and enforced across the whole dataset."""

    item_type_vocab_size: int = 128
    kit_type_vocab_size: int = 32
    place_block_type_vocab_size: int = 32

    feature_schema_version: int = 2
    """Bumped whenever the per-tick feature/target tensor layout changes (independent of
    field values above), so that existing caches built against an older layout are
    invalidated rather than silently reused (see :mod:`herbert_nn.data.cache`'s docstring).
    Version 2 added ``movement_target``."""

    def block_grid_shape(self) -> tuple[int, int, int] | None:
        """Return the configured canonical block-grid shape, if fully specified."""
        if (
            self.block_grid_width is not None
            and self.block_grid_height is not None
            and self.block_grid_depth is not None
        ):
            return (
                self.block_grid_width,
                self.block_grid_height,
                self.block_grid_depth,
            )
        return None

    def to_dict(self) -> dict:
        """Return a plain, JSON-serializable dict of this config."""
        return asdict(self)
