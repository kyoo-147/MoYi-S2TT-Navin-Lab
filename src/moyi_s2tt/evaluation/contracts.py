"""Frozen evaluation IDs, human-review status, and contamination gates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import model_validator

from ..config import StrictModel
from ..data.manifest import LanguageCode
from ..directions import DIRECTION_KEYS

CriticalCategory = Literal[
    "number",
    "decimal",
    "unit",
    "negation",
    "proper_noun",
    "machine_id",
    "safety_command",
    "industrial_term",
]
EvaluationSplit = Literal["validation", "test"]
ReferenceKind = Literal["dataset_reference", "human_reviewed"]
ReviewStatus = Literal["pending", "accepted", "rejected"]


class FrozenEvaluationSet(StrictModel):
    schema_version: int = 1
    id: str
    direction: str
    source_dataset: str
    source_revision: str
    source_locale: str
    target_locale: str
    validation_semantic_ids: tuple[str, ...]
    test_semantic_ids: tuple[str, ...]
    training_exclusion_semantic_ids: tuple[str, ...]

    @model_validator(mode="after")
    def validate_freeze(self) -> FrozenEvaluationSet:
        if self.direction not in DIRECTION_KEYS:
            raise ValueError(f"unsupported evaluation direction: {self.direction}")
        groups = (
            self.validation_semantic_ids,
            self.test_semantic_ids,
            self.training_exclusion_semantic_ids,
        )
        if any(tuple(sorted(values, key=_numeric_sort_key)) != values for values in groups):
            raise ValueError("frozen IDs must use deterministic numeric-aware sorting")
        if any(len(values) != len(set(values)) for values in groups):
            raise ValueError("frozen ID groups must not contain duplicates")
        if set(self.validation_semantic_ids) & set(self.test_semantic_ids):
            raise ValueError("validation and test semantic IDs overlap")
        expected = set(self.validation_semantic_ids) | set(self.test_semantic_ids)
        if set(self.training_exclusion_semantic_ids) != expected:
            raise ValueError("training exclusions must equal frozen validation and test IDs")
        return self

    @property
    def sha256(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class EvaluationReference(StrictModel):
    id: str
    frozen_set_id: str
    source_manifest_id: str
    semantic_group_id: str
    split: EvaluationSplit
    src_lang: LanguageCode
    tgt_lang: LanguageCode
    source_text: str
    reference_text: str
    reference_kind: ReferenceKind
    review_status: ReviewStatus
    critical_categories: tuple[CriticalCategory, ...] = ()
    reviewer_id: str | None = None
    review_notes: str = ""

    @model_validator(mode="after")
    def validate_review(self) -> EvaluationReference:
        if (
            self.reference_kind == "human_reviewed"
            and self.review_status == "accepted"
            and not self.reviewer_id
        ):
            raise ValueError("accepted human references require a reviewer ID")
        return self


class TrainingRecord(Protocol):
    @property
    def id(self) -> str: ...

    @property
    def semantic_group_id(self) -> str: ...


class ContaminationIssue(StrictModel):
    record_id: str
    semantic_group_id: str


def _numeric_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def find_evaluation_contamination(
    records: Iterable[TrainingRecord], frozen: FrozenEvaluationSet
) -> list[ContaminationIssue]:
    excluded = set(frozen.training_exclusion_semantic_ids)
    return [
        ContaminationIssue(record_id=record.id, semantic_group_id=record.semantic_group_id)
        for record in records
        if record.semantic_group_id in excluded
    ]


def require_evaluation_clean(
    records: Iterable[TrainingRecord], frozen: FrozenEvaluationSet
) -> None:
    issues = find_evaluation_contamination(records, frozen)
    if issues:
        preview = ", ".join(f"{issue.record_id}:{issue.semantic_group_id}" for issue in issues[:10])
        raise ValueError(f"training/evaluation contamination ({len(issues)} row(s)): {preview}")


def load_frozen_evaluation(path: Path) -> FrozenEvaluationSet:
    return FrozenEvaluationSet.model_validate_json(path.read_text(encoding="utf-8"))


def write_frozen_evaluation(path: Path, frozen: FrozenEvaluationSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(frozen.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
