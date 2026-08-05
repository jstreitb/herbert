# SPDX-License-Identifier: MIT
"""Schema-version registry and version-dispatching parse helpers.

To add support for a new ``schema_version`` (e.g. ``"1.1.0"``):

1. Create ``herbert_nn/schemas/v1_1_0.py`` modeled after
   :mod:`herbert_nn.schemas.v1_0_0`, with its own ``SessionHeaderV1_1_0`` /
   ``TickRecordV1_1_0`` (or similarly named) Pydantic models. Do not import
   from or mutate the previous version's module -- each version module must
   be a fully self-contained, frozen description of that exact wire format.
2. Add one line to :data:`SCHEMA_REGISTRY` below mapping the new version
   string to a :class:`SchemaModels` pointing at the new models.

Everything else (preprocessing, dataset building, CLI tools) dispatches
through :func:`get_models_for_version` / :func:`load_session` and therefore
needs no changes to support the new version.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from pydantic import BaseModel, ValidationError

from herbert_nn.schemas import v1_0_0

PathLike = str | Path


class SchemaVersionError(Exception):
    """Raised when a session file declares an unknown/unsupported ``schema_version``."""


class SessionParseError(Exception):
    """Raised by :func:`load_session` to add file/line context to a parse failure.

    The underlying error (a :class:`SchemaVersionError` or a
    :class:`pydantic.ValidationError`) is always chained as ``__cause__``,
    so callers that need the structured Pydantic error details (e.g. to list
    every invalid field) can still access ``exc.__cause__``.
    """


@dataclass(frozen=True)
class SchemaModels:
    """The pair of Pydantic models that implement one schema version."""

    header_model: type[BaseModel]
    record_model: type[BaseModel]


#: Registry of all supported schema versions. Add new versions here.
SCHEMA_REGISTRY: dict[str, SchemaModels] = {
    v1_0_0.SCHEMA_VERSION: SchemaModels(
        header_model=v1_0_0.SessionHeaderV1,
        record_model=v1_0_0.TickRecordV1,
    ),
}


def get_models_for_version(schema_version: str) -> SchemaModels:
    """Look up the Pydantic models registered for a ``schema_version`` string.

    Args:
        schema_version: The ``schema_version`` value read from a session
            header, e.g. ``"1.0.0"``.

    Returns:
        The :class:`SchemaModels` registered for that version.

    Raises:
        SchemaVersionError: If no models are registered for
            ``schema_version``. The message lists the versions that *are*
            supported, to make upgrading/downgrading straightforward.
    """
    try:
        return SCHEMA_REGISTRY[schema_version]
    except KeyError as exc:
        supported = ", ".join(sorted(SCHEMA_REGISTRY)) or "<none registered>"
        raise SchemaVersionError(
            f"Unsupported schema_version {schema_version!r}. "
            f"This build of herbert_nn supports: {supported}. "
            "If this file was produced by a newer /mod build, upgrade herbert_nn; "
            "if it predates the registry, add a versioned schema module "
            "(see herbert_nn.schemas.registry docstring)."
        ) from exc


def parse_header_line(line: str) -> BaseModel:
    """Parse the first line of a session ``.jsonl`` file into a header model.

    Args:
        line: Raw JSON text of the header line (including or excluding a
            trailing newline; both are accepted).

    Returns:
        A validated header model instance for the version declared in the line.

    Raises:
        SchemaVersionError: If the declared ``schema_version`` is unsupported,
            or the line is not valid JSON, or lacks a ``schema_version`` field.
        pydantic.ValidationError: If the line fails validation against the
            resolved header model (e.g. missing/malformed fields).
    """
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaVersionError(
            f"Session header line is not valid JSON: {exc}"
        ) from exc
    if not isinstance(raw, dict) or "schema_version" not in raw:
        raise SchemaVersionError(
            "Session header line is missing the required 'schema_version' field."
        )
    models = get_models_for_version(str(raw["schema_version"]))
    return models.header_model.model_validate(raw)


def parse_record_line(line: str, schema_version: str) -> BaseModel:
    """Parse a single tick-record JSON line for a known schema version.

    Args:
        line: Raw JSON text of the record line.
        schema_version: The session's schema version (as read from its header),
            used to select which record model to validate against.

    Returns:
        A validated tick-record model instance.

    Raises:
        SchemaVersionError: If ``schema_version`` is unsupported, or the line
            is not valid JSON.
        pydantic.ValidationError: If the line fails validation against the
            resolved record model -- e.g. a wrong type, an out-of-enum
            string, or a missing required field. The raised error carries
            precise field-path information from Pydantic.
    """
    models = get_models_for_version(schema_version)
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise SchemaVersionError(f"Tick record line is not valid JSON: {exc}") from exc
    return models.record_model.model_validate(raw)


@dataclass
class ParsedSession:
    """A fully parsed session: validated header plus an iterator of records."""

    header: BaseModel
    records: Iterator[BaseModel]


def load_session(path: PathLike) -> tuple[BaseModel, list[BaseModel]]:
    """Load and fully validate a session ``.jsonl`` file.

    This eagerly reads and validates every line (appropriate for the
    file sizes produced by multi-hour single-session recordings at Minecraft
    tick rates; callers that need a lazy/streaming variant can reimplement
    this using :func:`parse_header_line` / :func:`parse_record_line` directly
    over an open file handle).

    Args:
        path: Path to the ``.jsonl`` session file.

    Returns:
        A tuple ``(header, records)`` where ``header`` is the validated
        session header model and ``records`` is a list of validated
        per-tick record models in file order.

    Raises:
        SessionParseError: If the header or any record line fails to parse
            or validate. Wraps the underlying :class:`SchemaVersionError` or
            :class:`pydantic.ValidationError` (available as ``.__cause__``)
            with the offending file path and 1-indexed line number.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        header, records = _load_session_from_filehandle(fh, path)
    return header, records


def _load_session_from_filehandle(
    fh: TextIO, path: Path
) -> tuple[BaseModel, list[BaseModel]]:
    first_line = fh.readline()
    if not first_line.strip():
        raise SessionParseError(
            f"{path}: file is empty, expected a session header line."
        )
    try:
        header = parse_header_line(first_line)
    except (SchemaVersionError, ValidationError) as exc:
        raise SessionParseError(f"{path}: line 1 (header): {exc}") from exc

    # `header`'s concrete type depends on the dispatched schema version, so
    # mypy only sees the common `BaseModel` base; `schema_version` is a field
    # every version's header model is required to declare (see the
    # herbert_nn.schemas.registry module docstring).
    schema_version: str = getattr(header, "schema_version")  # noqa: B009
    records: list[BaseModel] = []
    for line_number, line in enumerate(fh, start=2):
        if not line.strip():
            continue
        try:
            records.append(parse_record_line(line, schema_version))
        except (SchemaVersionError, ValidationError) as exc:
            raise SessionParseError(f"{path}: line {line_number}: {exc}") from exc
    return header, records
