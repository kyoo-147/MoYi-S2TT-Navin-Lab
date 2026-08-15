"""Private pseudo-label candidates retaining every teacher stage and failure."""

from __future__ import annotations

import hashlib
import json
from typing import Protocol

from pydantic import Field, model_validator

from ..config import StrictModel
from ..data.source import SourceRecord
from .cache import CachedPrediction


class LanguageIdentifier(Protocol):
    id: str
    revision: str

    def is_language(self, text: str, expected_language: str) -> bool: ...


class SemanticAgreementProvider(Protocol):
    id: str
    revision: str

    def score(self, left: str, right: str) -> float: ...


class PseudoLabelCandidate(StrictModel):
    source: SourceRecord
    asr: CachedPrediction | None = None
    mt: CachedPrediction | None = None
    direct_s2tt: CachedPrediction | None = None
    failure_stage: str | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> PseudoLabelCandidate:
        if bool(self.failure_stage) != bool(self.failure_reason):
            raise ValueError("failure stage and reason must be recorded together")
        if not self.failure_stage and not (self.asr and self.mt and self.direct_s2tt):
            raise ValueError("successful candidates require ASR, MT, and direct S2TT outputs")
        return self

    @property
    def teacher_chain_sha256(self) -> str:
        payload = {
            name: value.model_dump(mode="json") if value else None
            for name, value in (
                ("asr", self.asr),
                ("mt", self.mt),
                ("direct_s2tt", self.direct_s2tt),
            )
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


class FilteringAggregate(StrictModel):
    schema_version: int = 1
    status: str
    total_rows: int
    accepted_rows: int
    rejected_rows: int
    accepted_hours: float
    rejected_hours: float
    reject_reasons: dict[str, int] = Field(default_factory=dict)
    rows_by_domain: dict[str, dict[str, int]] = Field(default_factory=dict)
    rows_by_duration_bucket: dict[str, dict[str, int]] = Field(default_factory=dict)
    human_audit_sample_ids: tuple[str, ...] = ()
    false_accept_count: int | None = None
    false_reject_count: int | None = None
    limitations: tuple[str, ...] = ()
