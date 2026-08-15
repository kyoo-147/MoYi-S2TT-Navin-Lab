"""Content-addressed SQLite cache for offline teacher predictions."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from ..config import StrictModel
from ..teachers.base import GenerationConfig, Teacher, TeacherInput, TeacherPrediction, TeacherSpec


class CachedPrediction(StrictModel):
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_id: str
    text: str
    confidence_proxy: float | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    teacher_id: str
    teacher_revision: str
    teacher_license: str
    generation_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    generated_at: str


def prediction_cache_key(
    spec: TeacherSpec, teacher_input: TeacherInput, config: GenerationConfig
) -> str:
    if not spec.revision:
        raise ValueError("teacher inference requires a pinned teacher revision")
    payload = {
        "teacher_id": spec.id,
        "teacher_revision": spec.revision,
        "input": teacher_input.model_dump(mode="json"),
        "generation_config": config.model_dump(mode="json"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class PredictionCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS teacher_predictions (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL
            )
            """
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> PredictionCache:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def get(self, key: str) -> CachedPrediction | None:
        row = self._connection.execute(
            "SELECT payload FROM teacher_predictions WHERE cache_key = ?", (key,)
        ).fetchone()
        return CachedPrediction.model_validate_json(row[0]) if row else None

    def put(self, prediction: CachedPrediction) -> None:
        payload = prediction.model_dump_json()
        self._connection.execute(
            "INSERT OR REPLACE INTO teacher_predictions(cache_key, payload) VALUES (?, ?)",
            (prediction.cache_key, payload),
        )
        self._connection.commit()


def _cached_prediction(
    key: str,
    spec: TeacherSpec,
    config: GenerationConfig,
    prediction: TeacherPrediction,
) -> CachedPrediction:
    if not spec.revision:
        raise ValueError("teacher inference requires a pinned teacher revision")
    return CachedPrediction(
        cache_key=key,
        source_id=prediction.source_id,
        text=prediction.text,
        confidence_proxy=prediction.confidence_proxy,
        metadata=prediction.metadata,
        teacher_id=spec.id,
        teacher_revision=spec.revision,
        teacher_license=spec.license,
        generation_config_sha256=config.sha256,
        generated_at=datetime.now(UTC).isoformat(),
    )


def generate_cached(
    teacher: Teacher,
    inputs: Sequence[TeacherInput],
    config: GenerationConfig,
    cache: PredictionCache,
) -> list[CachedPrediction]:
    """Generate only cache misses and return results in input order."""

    direction_inputs = [
        item
        for item in inputs
        if f"{item.src_lang}-{item.tgt_lang}" in teacher.spec.directions
    ]
    if len(direction_inputs) != len(inputs):
        raise ValueError(f"teacher {teacher.spec.id} does not support every requested direction")

    keys = [prediction_cache_key(teacher.spec, item, config) for item in inputs]
    found = [cache.get(key) for key in keys]
    missing_indices = [index for index, value in enumerate(found) if value is None]
    if missing_indices:
        missing_inputs = [inputs[index] for index in missing_indices]
        predictions = list(teacher.generate(missing_inputs, config))
        if len(predictions) != len(missing_inputs):
            raise ValueError("teacher returned a different prediction count")
        for index, prediction in zip(missing_indices, predictions, strict=True):
            expected_source_id = inputs[index].source_id
            if prediction.source_id != expected_source_id:
                message = (
                    f"teacher output order mismatch: expected {expected_source_id}, "
                    f"got {prediction.source_id}"
                )
                raise ValueError(message)
            cached = _cached_prediction(keys[index], teacher.spec, config, prediction)
            cache.put(cached)
            found[index] = cached
    return [value for value in found if value is not None]
