# SPDX-License-Identifier: MIT
"""Content-hashed preprocessing cache: raw JSONL -> normalized tensors on disk.

The cache directory name is a hash of the resolved preprocessing config plus
the schema version(s) encountered plus the sorted list of raw input
filenames, so that changing any of the following automatically invalidates
(recomputes) the cache rather than silently serving stale data:

* The schema version of the raw session files.
* Any preprocessing knob (window length/stride, split ratios/seed, vocab
  size caps, block-grid shape override, ...).
* The set of raw files being preprocessed.

Each cache directory contains one gzip-compressed ``torch.save`` file per
split (``train.pt.gz`` / ``val.pt.gz`` / ``test.pt.gz``) plus a
``manifest.json`` with everything needed to *interpret* those tensors
(fitted normalization statistics, vocabularies, block-grid shape, and the
resolved config/session split itself) without re-reading raw data.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from herbert_nn.data.config import PreprocessConfig
from herbert_nn.data.features import SessionArrays, encode_session_raw
from herbert_nn.data.normalization import Standardizer
from herbert_nn.data.split import SessionSplit, split_sessions
from herbert_nn.data.vocab import CategoricalVocab
from herbert_nn.schemas.registry import (
    SchemaVersionError,
    load_session,
    parse_header_line,
)

logger = logging.getLogger(__name__)

_SPLIT_NAMES = ("train", "val", "test")


@dataclass
class SessionBoundary:
    """Half-open ``[start, end)`` row range owned by one session within a split tensor."""

    session_id: str
    start: int
    end: int


@dataclass
class CacheManifest:
    """Everything needed to interpret cached tensors without re-reading raw data."""

    schema_version: str
    config_hash: str
    resolved_config: dict[str, Any]
    block_grid_shape: tuple[int, int, int]
    standardizer: Standardizer
    item_type_vocab: CategoricalVocab
    kit_type_vocab: CategoricalVocab
    place_block_type_vocab: CategoricalVocab
    split_session_ids: dict[str, list[str]]
    split_boundaries: dict[str, list[SessionBoundary]]
    counts: dict[str, int]
    created_at: str

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-safe dict written to ``manifest.json``."""
        return {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "resolved_config": self.resolved_config,
            "block_grid_shape": list(self.block_grid_shape),
            "standardizer": self.standardizer.to_dict(),
            "item_type_vocab": self.item_type_vocab.to_dict(),
            "kit_type_vocab": self.kit_type_vocab.to_dict(),
            "place_block_type_vocab": self.place_block_type_vocab.to_dict(),
            "split_session_ids": self.split_session_ids,
            "split_boundaries": {
                split: [[b.session_id, b.start, b.end] for b in boundaries]
                for split, boundaries in self.split_boundaries.items()
            },
            "counts": self.counts,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> CacheManifest:
        """Deserialize a manifest previously produced by :meth:`to_json_dict`."""
        return cls(
            schema_version=data["schema_version"],
            config_hash=data["config_hash"],
            resolved_config=data["resolved_config"],
            block_grid_shape=tuple(data["block_grid_shape"]),
            standardizer=Standardizer.from_dict(data["standardizer"]),
            item_type_vocab=CategoricalVocab.from_dict(data["item_type_vocab"]),
            kit_type_vocab=CategoricalVocab.from_dict(data["kit_type_vocab"]),
            place_block_type_vocab=CategoricalVocab.from_dict(
                data["place_block_type_vocab"]
            ),
            split_session_ids=data["split_session_ids"],
            split_boundaries={
                split: [SessionBoundary(sid, start, end) for sid, start, end in rows]
                for split, rows in data["split_boundaries"].items()
            },
            counts=data["counts"],
            created_at=data["created_at"],
        )


@dataclass
class CacheBundle:
    """A resolved cache directory: manifest + helpers to load split tensors."""

    cache_path: Path
    manifest: CacheManifest

    def load_split(self, split: str) -> dict[str, torch.Tensor]:
        """Load one split's tensors (``"train"``, ``"val"``, or ``"test"``)."""
        if split not in _SPLIT_NAMES:
            raise ValueError(
                f"Unknown split {split!r}, expected one of {_SPLIT_NAMES}."
            )
        path = self.cache_path / f"{split}.pt.gz"
        with gzip.open(path, "rb") as fh:
            # torch's stubs don't cover GzipFile, but it satisfies the
            # IO[bytes] protocol torch.load actually accepts at runtime.
            return torch.load(fh, map_location="cpu", weights_only=False)  # type: ignore[arg-type]


def _discover_raw_files(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No *.jsonl session files found under {raw_dir}.")
    return files


def _compute_config_hash(
    config: PreprocessConfig, schema_version: str, raw_files: list[Path]
) -> str:
    payload = {
        "schema_version": schema_version,
        "config": config.to_dict(),
        "raw_files": sorted(f.name for f in raw_files),
    }
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _cache_dir_for(config: PreprocessConfig, config_hash: str) -> Path:
    return Path(config.cache_dir) / config_hash


def _load_all_sessions(
    raw_files: list[Path],
) -> tuple[str, dict[str, list]]:
    """Parse every raw file, returning the shared schema version and per-session records."""
    schema_version: str | None = None
    sessions: dict[str, list] = {}
    for path in raw_files:
        header, records = load_session(path)
        # `header`'s concrete type depends on the dispatched schema version;
        # every version's header model is required to declare these fields
        # (see herbert_nn.schemas.registry).
        this_version: str = getattr(header, "schema_version")  # noqa: B009
        if schema_version is None:
            schema_version = this_version
        elif this_version != schema_version:
            raise ValueError(
                f"Mixed schema versions in {path.parent}: found {schema_version!r} and "
                f"{this_version!r}. Preprocess each schema version into a separate cache."
            )
        session_id: str = getattr(header, "session_id")  # noqa: B009
        if session_id in sessions:
            raise ValueError(
                f"Duplicate session_id {session_id!r} encountered in {path} "
                "(already seen in another raw file)."
            )
        if not records:
            logger.warning(
                "Session %s (%s) has zero tick records; skipping.", session_id, path
            )
            continue
        sessions[session_id] = records
    if schema_version is None:
        raise FileNotFoundError("No valid sessions found to preprocess.")
    return schema_version, sessions


def _concat_session_arrays(
    session_ids: list[str], arrays_by_session: dict[str, SessionArrays]
) -> tuple[dict[str, np.ndarray], list[SessionBoundary]]:
    boundaries: list[SessionBoundary] = []
    offset = 0
    parts: dict[str, list[np.ndarray]] = {
        "tick": [],
        "continuous": [],
        "block_grid_cells": [],
        "hotbar_slot_index": [],
        "opponent_held_item_category": [],
        "mouse_target": [],
        "discrete_target": [],
        "place_mask": [],
        "movement_target": [],
    }
    raw_lists: dict[str, list] = {
        "hotbar_item_type_raw": [],
        "match_kit_type_raw": [],
        "place_block_type_raw": [],
    }
    for sid in session_ids:
        arr = arrays_by_session[sid]
        n = arr.tick.shape[0]
        boundaries.append(SessionBoundary(session_id=sid, start=offset, end=offset + n))
        offset += n
        for key in parts:
            parts[key].append(getattr(arr, key))
        for key in raw_lists:
            raw_lists[key].extend(getattr(arr, key))

    concatenated = {key: np.concatenate(vals, axis=0) for key, vals in parts.items()}
    concatenated.update(raw_lists)  # type: ignore[arg-type]
    return concatenated, boundaries


def load_cache_bundle(cache_path: str | Path) -> CacheBundle:
    """Load a :class:`CacheBundle` directly from a known cache directory path.

    Unlike :func:`build_or_load_cache`, this does not need (or recompute) a
    :class:`PreprocessConfig` -- it's used by ``herbert_nn.evaluate`` /
    ``herbert_nn.inspect`` to reload the exact cache a checkpoint was trained
    against, using the ``cache_path`` string embedded in the checkpoint.

    Args:
        cache_path: Path to a cache directory containing ``manifest.json``.

    Returns:
        The resolved :class:`CacheBundle`.

    Raises:
        FileNotFoundError: If ``cache_path`` has no ``manifest.json``.
    """
    cache_path = Path(cache_path)
    manifest_path = cache_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest.json under {cache_path}; this checkpoint's cache may have been "
            "deleted. Re-run herbert_nn.preprocess with the same config to rebuild it "
            "(the checkpoint's embedded resolved data config can be found in its 'extra' field)."
        )
    manifest = CacheManifest.from_json_dict(json.loads(manifest_path.read_text()))
    return CacheBundle(cache_path=cache_path, manifest=manifest)


def _encode_tensors(
    data: dict[str, Any],
    standardizer: Standardizer,
    item_type_vocab: CategoricalVocab,
    kit_type_vocab: CategoricalVocab,
    place_block_type_vocab: CategoricalVocab,
) -> dict[str, torch.Tensor]:
    """Apply a fitted standardizer + vocabularies to raw per-tick arrays, producing tensors.

    ``data`` must have the same keys as :class:`herbert_nn.data.features.SessionArrays`
    (minus ``session_id``), either for one session or for many sessions already
    concatenated together (see :func:`_concat_session_arrays`).
    """
    return {
        "tick": torch.from_numpy(data["tick"]),
        "continuous": torch.from_numpy(standardizer.transform(data["continuous"])),
        "block_grid_cells": torch.from_numpy(data["block_grid_cells"]),
        "hotbar_slot_index": torch.from_numpy(data["hotbar_slot_index"]),
        "hotbar_item_type": torch.tensor(
            [item_type_vocab.encode(v) for v in data["hotbar_item_type_raw"]],
            dtype=torch.int64,
        ),
        "opponent_held_item_category": torch.from_numpy(
            data["opponent_held_item_category"]
        ),
        "match_kit_type": torch.tensor(
            [kit_type_vocab.encode(v) for v in data["match_kit_type_raw"]],
            dtype=torch.int64,
        ),
        "mouse_target": torch.from_numpy(data["mouse_target"]),
        "discrete_target": torch.from_numpy(data["discrete_target"]),
        "place_block_type": torch.tensor(
            [place_block_type_vocab.encode(v) for v in data["place_block_type_raw"]],
            dtype=torch.int64,
        ),
        "place_mask": torch.from_numpy(data["place_mask"]),
        "movement_target": torch.from_numpy(data["movement_target"]),
    }


def encode_session_for_inference(
    arrays: SessionArrays, manifest: CacheManifest
) -> dict[str, torch.Tensor]:
    """Apply an existing cache's fitted normalizer/vocabs to one (held-out) session's raw arrays.

    Used by ``herbert_nn.inspect`` to run a trained model over a session that
    was not necessarily part of the training cache, using that cache's exact
    standardization/vocabulary so features line up with what the model saw
    during training.

    Args:
        arrays: Raw (pre-normalization) arrays for one session, from
            :func:`herbert_nn.data.features.encode_session_raw`.
        manifest: The manifest of the cache whose normalizer/vocabs to apply
            (typically loaded via :func:`load_cache_bundle`).

    Returns:
        A tensor dict in the same shape/format as
        :meth:`CacheBundle.load_split`, for a single session.
    """
    data = {
        "tick": arrays.tick,
        "continuous": arrays.continuous,
        "block_grid_cells": arrays.block_grid_cells,
        "hotbar_slot_index": arrays.hotbar_slot_index,
        "hotbar_item_type_raw": arrays.hotbar_item_type_raw,
        "opponent_held_item_category": arrays.opponent_held_item_category,
        "match_kit_type_raw": arrays.match_kit_type_raw,
        "mouse_target": arrays.mouse_target,
        "discrete_target": arrays.discrete_target,
        "place_block_type_raw": arrays.place_block_type_raw,
        "place_mask": arrays.place_mask,
        "movement_target": arrays.movement_target,
    }
    return _encode_tensors(
        data,
        manifest.standardizer,
        manifest.item_type_vocab,
        manifest.kit_type_vocab,
        manifest.place_block_type_vocab,
    )


def build_or_load_cache(
    config: PreprocessConfig, force_rebuild: bool = False
) -> CacheBundle:
    """Build (or load an already-valid) preprocessing cache for ``config``.

    Args:
        config: Resolved preprocessing configuration.
        force_rebuild: If ``True``, ignore any existing cache at the computed
            hash path and rebuild from raw data.

    Returns:
        A :class:`CacheBundle` pointing at valid cached tensors on disk.
    """
    raw_dir = Path(config.raw_dir)
    raw_files = _discover_raw_files(raw_dir)

    # We need the schema version to compute the hash, which means reading
    # headers before we know whether we can skip the (expensive) full parse.
    # Reading just the header lines is cheap, so do a lightweight pre-pass.
    schema_version = _peek_schema_version(raw_files[0])
    config_hash = _compute_config_hash(config, schema_version, raw_files)
    cache_path = _cache_dir_for(config, config_hash)
    manifest_path = cache_path / "manifest.json"

    if not force_rebuild and manifest_path.exists():
        logger.info("Reusing existing preprocessing cache at %s", cache_path)
        manifest = CacheManifest.from_json_dict(json.loads(manifest_path.read_text()))
        return CacheBundle(cache_path=cache_path, manifest=manifest)

    logger.info(
        "Building preprocessing cache at %s (this may take a while)...", cache_path
    )
    return _build_cache(config, cache_path, config_hash)


def _peek_schema_version(path: Path) -> str:
    """Cheaply read just the ``schema_version`` field from a session file's header line.

    Reuses :func:`herbert_nn.schemas.registry.parse_header_line` so a malformed or empty
    first raw file produces the same clear, actionable error as the full parse in
    :func:`_load_all_sessions` would, rather than a raw, unhelpful JSON decode error.

    Args:
        path: Path to a session ``.jsonl`` file.

    Returns:
        The header's ``schema_version`` string.

    Raises:
        SchemaVersionError: If the file is empty, its header line is not valid JSON,
            is missing ``schema_version``, or declares an unsupported version.
        pydantic.ValidationError: If the header line fails full validation.
    """
    with path.open("r", encoding="utf-8") as fh:
        first_line = fh.readline()
    if not first_line.strip():
        raise SchemaVersionError(
            f"{path}: file is empty, expected a session header line."
        )
    header = parse_header_line(first_line)
    # `header`'s concrete type depends on the dispatched schema version; see the
    # herbert_nn.schemas.registry module docstring.
    return str(getattr(header, "schema_version"))  # noqa: B009


def _build_cache(
    config: PreprocessConfig, cache_path: Path, config_hash: str
) -> CacheBundle:
    raw_dir = Path(config.raw_dir)
    raw_files = _discover_raw_files(raw_dir)
    schema_version, sessions_by_id = _load_all_sessions(raw_files)

    # Determine the canonical block-grid shape.
    configured_shape = config.block_grid_shape()
    if configured_shape is not None:
        block_grid_shape = configured_shape
    else:
        first_records = next(iter(sessions_by_id.values()))
        bg = first_records[0].block_grid
        block_grid_shape = (bg.width, bg.height, bg.depth)
        logger.info(
            "Adopted canonical block_grid_shape=%s from first session.",
            block_grid_shape,
        )

    arrays_by_session: dict[str, SessionArrays] = {
        sid: encode_session_raw(records, sid, block_grid_shape)
        for sid, records in sessions_by_id.items()
    }

    split: SessionSplit = split_sessions(
        list(arrays_by_session.keys()),
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        seed=config.split_seed,
    )
    split_session_ids = {"train": split.train, "val": split.val, "test": split.test}

    concatenated: dict[str, dict[str, np.ndarray]] = {}
    boundaries: dict[str, list[SessionBoundary]] = {}
    for split_name in _SPLIT_NAMES:
        sids = sorted(split_session_ids[split_name])
        concatenated[split_name], boundaries[split_name] = _concat_session_arrays(
            sids, arrays_by_session
        )

    # Fit vocabularies + standardizer on TRAIN ONLY.
    train_data = concatenated["train"]
    item_type_vocab = CategoricalVocab(
        "hotbar_item_type", config.item_type_vocab_size
    ).fit(train_data["hotbar_item_type_raw"])
    kit_type_vocab = CategoricalVocab("match_kit_type", config.kit_type_vocab_size).fit(
        train_data["match_kit_type_raw"]
    )
    place_block_type_vocab = CategoricalVocab(
        "place_block_type", config.place_block_type_vocab_size
    ).fit(train_data["place_block_type_raw"])

    standardizer = Standardizer().fit(train_data["continuous"])

    cache_path.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split_name in _SPLIT_NAMES:
        data = concatenated[split_name]
        tensors = _encode_tensors(
            data, standardizer, item_type_vocab, kit_type_vocab, place_block_type_vocab
        )
        counts[split_name] = int(data["tick"].shape[0])
        out_path = cache_path / f"{split_name}.pt.gz"
        with gzip.open(out_path, "wb", compresslevel=6) as fh:
            torch.save(tensors, fh)  # type: ignore[arg-type]
        logger.info(
            "Wrote %s (%d ticks) to %s", split_name, counts[split_name], out_path
        )

    manifest = CacheManifest(
        schema_version=schema_version,
        config_hash=config_hash,
        resolved_config=config.to_dict(),
        block_grid_shape=block_grid_shape,
        standardizer=standardizer,
        item_type_vocab=item_type_vocab,
        kit_type_vocab=kit_type_vocab,
        place_block_type_vocab=place_block_type_vocab,
        split_session_ids=split_session_ids,
        split_boundaries=boundaries,
        counts=counts,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    (cache_path / "manifest.json").write_text(
        json.dumps(manifest.to_json_dict(), indent=2)
    )
    logger.info("Cache build complete: %s", cache_path)
    return CacheBundle(cache_path=cache_path, manifest=manifest)
