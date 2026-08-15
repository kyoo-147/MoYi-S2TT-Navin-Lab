"""Fail-closed experiment, data-selection, and Trainer resume contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, model_validator

from ..config import StrictModel
from ..data.manifest import ManifestRecord
from ..directions import DIRECTION_KEYS
from ..runtime.checkpointing import canonical_sha256


class TrainingConfig(StrictModel):
    schema_version: int = 1
    id: str
    direction: str
    mode: Literal["overfit", "smoke", "baseline", "sequence_kd"]
    model_id: str
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_license: str
    seed: int
    max_audio_seconds: float = Field(gt=0, le=30)
    max_rows: int = Field(gt=0)
    validation_fraction: float = Field(ge=0, lt=1)
    max_steps: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    per_device_batch_size: int = Field(gt=0)
    gradient_accumulation_steps: int = Field(gt=0)
    save_steps: int = Field(gt=0)
    eval_steps: int = Field(gt=0)
    mixed_precision: Literal["fp16", "bf16", "fp32"]
    source_language: str
    target_language: str
    task: Literal["translate"]
    expected_parameter_range_millions: tuple[int, int]
    allowed_target_kinds: tuple[Literal["gold", "teacher"], ...] = ("gold", "teacher")
    checkpoint_metric: str = "eval_loss"
    greater_is_better: bool = False

    @model_validator(mode="after")
    def validate_training(self) -> TrainingConfig:
        if self.direction not in DIRECTION_KEYS:
            raise ValueError(f"unsupported training direction: {self.direction}")
        if self.mode == "overfit" and self.validation_fraction != 0:
            raise ValueError("overfit mode must train on every selected row")
        if self.expected_parameter_range_millions[0] > self.expected_parameter_range_millions[1]:
            raise ValueError("expected parameter range is reversed")
        return self

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))


class TrainingRunContract(StrictModel):
    schema_version: int = 1
    run_id: str
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_id: str
    model_revision: str
    git_commit: str
    selected_ids: tuple[str, ...]


def load_training_config(path: Path) -> TrainingConfig:
    value: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    return TrainingConfig.model_validate(value)


def select_training_rows(
    rows: Sequence[ManifestRecord], config: TrainingConfig
) -> tuple[ManifestRecord, ...]:
    eligible = [
        row
        for row in rows
        if row.split == "train"
        and row.duration_s <= config.max_audio_seconds
        and row.materialization_status == "ready"
        and row.filter_decision != "reject"
        and row.target_kind in config.allowed_target_kinds
    ]
    ordered = sorted(
        eligible,
        key=lambda row: hashlib.sha256(f"{config.seed}:{row.id}".encode()).hexdigest(),
    )
    selected = tuple(ordered[: config.max_rows])
    if len(selected) != config.max_rows:
        raise ValueError(
            f"training requested {config.max_rows} rows but only {len(selected)} are eligible"
        )
    return selected


def data_sha256(rows: Sequence[ManifestRecord]) -> str:
    payload = [
        {
            "id": row.id,
            "audio_sha256": row.audio_sha256,
            "target": row.tgt_text,
            "target_kind": row.target_kind,
        }
        for row in rows
    ]
    return canonical_sha256(payload)


def make_run_contract(
    config: TrainingConfig, rows: Sequence[ManifestRecord], git_commit: str
) -> TrainingRunContract:
    return TrainingRunContract(
        run_id=config.id,
        config_sha256=config.sha256,
        data_sha256=data_sha256(rows),
        model_id=config.model_id,
        model_revision=config.model_revision,
        git_commit=git_commit,
        selected_ids=tuple(row.id for row in rows),
    )


def require_compatible_run(path: Path, expected: TrainingRunContract) -> None:
    actual = TrainingRunContract.model_validate_json(path.read_text(encoding="utf-8"))
    if actual != expected:
        raise ValueError("training resume contract mismatch")


def write_run_contract(path: Path, contract: TrainingRunContract) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")


def require_trainer_checkpoint(path: Path) -> None:
    required = {
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    }
    missing = sorted(name for name in required if not (path / name).is_file())
    if missing:
        raise ValueError(f"incomplete Trainer checkpoint, missing: {missing}")
