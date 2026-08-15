"""Validated configuration loading for languages, students, and experiments."""

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .directions import DIRECTION_KEYS, Direction


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DirectionConfig(StrictModel):
    key: str
    source_language: str
    target_language: str
    enabled: bool = False
    target_control_token: str
    teacher_chain: tuple[str, ...] = ()
    sampling_weight: float = Field(default=1.0, gt=0)

    @model_validator(mode="after")
    def validate_direction(self) -> "DirectionConfig":
        direction = Direction(self.source_language, self.target_language)
        if self.key != direction.key:
            raise ValueError(f"direction key {self.key!r} does not match {direction.key!r}")
        if self.key not in DIRECTION_KEYS:
            raise ValueError(f"direction is not registered: {self.key}")
        return self


class StudentTierConfig(StrictModel):
    name: Literal["tiny", "mobile", "base", "plus"]
    role: str
    parameter_min_millions: int = Field(gt=0)
    parameter_max_millions: int = Field(gt=0)
    deployment_gate: str

    @model_validator(mode="after")
    def validate_range(self) -> "StudentTierConfig":
        if self.parameter_min_millions > self.parameter_max_millions:
            raise ValueError("parameter range is reversed")
        return self


class ExperimentConfig(StrictModel):
    name: str
    direction: str
    student_tier: Literal["tiny", "mobile", "base", "plus"]
    objective: Literal["smoke", "sequence_kd"]
    seed: int = 147
    max_audio_seconds: float = Field(default=30.0, gt=0, le=30)
    data_manifest: str
    output_dir: str

    @model_validator(mode="after")
    def validate_experiment(self) -> "ExperimentConfig":
        if self.direction not in DIRECTION_KEYS:
            raise ValueError(f"unsupported experiment direction: {self.direction}")
        return self


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object in {path}")
    return value


def load_direction(path: Path) -> DirectionConfig:
    return DirectionConfig.model_validate(_read_yaml(path))


def load_student_tier(path: Path) -> StudentTierConfig:
    return StudentTierConfig.model_validate(_read_yaml(path))


def load_experiment(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(_read_yaml(path))


def validate_config_tree(root: Path) -> dict[str, int]:
    language_paths = sorted((root / "configs" / "languages").glob("*.yaml"))
    student_paths = sorted((root / "configs" / "students").glob("*.yaml"))
    experiment_paths = sorted((root / "configs" / "experiments").glob("*.yaml"))

    directions = [load_direction(path) for path in language_paths]
    students = [load_student_tier(path) for path in student_paths]
    experiments = [load_experiment(path) for path in experiment_paths]

    keys = {config.key for config in directions}
    if keys != DIRECTION_KEYS:
        missing = sorted(DIRECTION_KEYS - keys)
        extra = sorted(keys - DIRECTION_KEYS)
        raise ValueError(f"direction config mismatch: missing={missing}, extra={extra}")
    if sum(config.enabled for config in directions) != 1:
        raise ValueError("exactly one direction must be enabled in the foundation checkpoint")
    if not next(config for config in directions if config.enabled).key == "vi-en":
        raise ValueError("VI->EN must be the first enabled direction")

    return {
        "directions": len(directions),
        "student_tiers": len(students),
        "experiments": len(experiments),
    }
