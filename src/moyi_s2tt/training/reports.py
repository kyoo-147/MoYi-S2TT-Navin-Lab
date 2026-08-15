"""Sanitized training evidence with explicit unavailable states."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from ..config import StrictModel


class TrainingEvidence(StrictModel):
    schema_version: int = 1
    run_id: str
    status: Literal["VERIFIED", "FAILED", "BLOCKED", "NOT_RUN"]
    objective: Literal["overfit", "smoke", "baseline", "sequence_kd"]
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    model_revision: str
    evaluation_freeze_sha256: str
    target_kinds: tuple[str, ...]
    seed: int
    selected_checkpoint: str | None = None
    checkpoint_metric: str
    quality_metrics: dict[str, float] = Field(default_factory=dict)
    training_seconds: float | None = None
    parameter_count: int | None = None
    peak_gpu_memory_bytes: int | None = None
    failures: tuple[str, ...] = ()
    variance_runs: int = 0
    limitations: tuple[str, ...] = ()
