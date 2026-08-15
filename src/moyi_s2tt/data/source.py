"""Canonical source-audio records shared by licensed dataset adapters."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Literal, Protocol

from pydantic import Field, model_validator

from ..config import StrictModel
from .manifest import Domain, LanguageCode, MaterializationStatus, Split
from .splits import find_split_leakage, normalized_text_hash


class DatasetDescriptor(StrictModel):
    id: str
    revision: str
    license: str
    access: Literal["public", "authenticated", "gated", "hold"]
    homepage_url: str
    terms_url: str | None = None
    redistribution: Literal["allowed", "restricted", "unverified"]


class DatasetAdapter(Protocol):
    @property
    def descriptor(self) -> DatasetDescriptor: ...


class SourceRecord(StrictModel):
    """One source utterance before a gold or teacher translation is attached."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]+$")
    source_item_id: str
    semantic_group_id: str
    source_locale: str
    source_audio_ref: str
    audio_path: str | None = None
    audio_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    duration_s: float = Field(gt=0, le=30)
    source_sample_rate: int | None = Field(default=None, gt=0)
    source_channels: int | None = Field(default=None, gt=0)
    source_codec: str | None = None
    speaker_id: str | None = None
    src_lang: LanguageCode
    src_text: str = Field(min_length=1)
    domain: Domain
    split: Split
    source_dataset: str
    source_revision: str
    source_license: str
    materialization_status: MaterializationStatus = "metadata_only"
    quality_flags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_materialization(self) -> SourceRecord:
        if self.materialization_status == "ready":
            required = (
                self.audio_path,
                self.audio_sha256,
                self.source_sample_rate,
                self.source_channels,
                self.source_codec,
            )
            if not all(required):
                raise ValueError(
                    "ready source records require path, hash, and probed audio metadata"
                )
        return self


class SourceIssue(StrictModel):
    source_item: str
    reason: str


class SourceBatch(StrictModel):
    records: tuple[SourceRecord, ...]
    issues: tuple[SourceIssue, ...] = ()


class SplitSummary(StrictModel):
    rows: int
    hours: float
    speakers: int
    missing_speaker_rows: int
    duration_p50_s: float | None
    duration_p95_s: float | None
    duration_max_s: float | None


class SourceAcceptanceReport(StrictModel):
    source_dataset: str
    source_revision: str
    source_license: str
    splits: dict[str, SplitSummary]
    total_rows: int
    total_hours: float
    unique_speakers: int
    rejected_rows: int
    rejection_reasons: dict[str, int]
    duplicate_source_item_rows: int
    duplicate_audio_rows: int
    repeated_semantic_groups: int
    repeated_text_groups: int
    split_leakage_issues: int


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def build_source_report(batch: SourceBatch) -> SourceAcceptanceReport:
    if not batch.records:
        raise ValueError("cannot report an empty source batch")
    first = batch.records[0]
    if any(record.source_dataset != first.source_dataset for record in batch.records):
        raise ValueError("source report cannot mix datasets")
    if any(record.source_revision != first.source_revision for record in batch.records):
        raise ValueError("source report cannot mix revisions")

    summaries: dict[str, SplitSummary] = {}
    for split in ("train", "validation", "test"):
        records = [record for record in batch.records if record.split == split]
        durations = [record.duration_s for record in records]
        summaries[split] = SplitSummary(
            rows=len(records),
            hours=sum(durations) / 3600,
            speakers=len({record.speaker_id for record in records if record.speaker_id}),
            missing_speaker_rows=sum(record.speaker_id is None for record in records),
            duration_p50_s=_percentile(durations, 0.5),
            duration_p95_s=_percentile(durations, 0.95),
            duration_max_s=max(durations) if durations else None,
        )
    reasons = Counter(issue.reason for issue in batch.issues)
    source_items = Counter(record.source_item_id for record in batch.records)
    audio_hashes = Counter(record.audio_sha256 for record in batch.records if record.audio_sha256)
    semantic_groups = Counter(record.semantic_group_id for record in batch.records)
    text_groups = Counter(normalized_text_hash(record.src_text) for record in batch.records)
    return SourceAcceptanceReport(
        source_dataset=first.source_dataset,
        source_revision=first.source_revision,
        source_license=first.source_license,
        splits=summaries,
        total_rows=len(batch.records),
        total_hours=sum(record.duration_s for record in batch.records) / 3600,
        unique_speakers=len({record.speaker_id for record in batch.records if record.speaker_id}),
        rejected_rows=len(batch.issues),
        rejection_reasons=dict(sorted(reasons.items())),
        duplicate_source_item_rows=sum(
            count - 1 for count in source_items.values() if count > 1
        ),
        duplicate_audio_rows=sum(count - 1 for count in audio_hashes.values() if count > 1),
        repeated_semantic_groups=sum(count > 1 for count in semantic_groups.values()),
        repeated_text_groups=sum(count > 1 for count in text_groups.values()),
        split_leakage_issues=len(find_split_leakage(batch.records)),
    )


def iter_source_manifest(path: Path) -> Iterator[SourceRecord]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield SourceRecord.model_validate_json(line)
            except ValueError as error:
                raise ValueError(f"invalid source row {path}:{line_number}: {error}") from error


def write_source_manifest(path: Path, records: Iterable[SourceRecord]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = json.dumps(
                record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True
            )
            handle.write(payload + "\n")
            count += 1
    return count
