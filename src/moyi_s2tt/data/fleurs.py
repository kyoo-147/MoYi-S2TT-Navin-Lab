"""Pinned FLEURS TSV parsing and cross-locale metadata alignment."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..directions import DIRECTION_KEYS
from .manifest import LanguageCode, ManifestRecord, Split

FLEURS_DATASET = "google/fleurs"
FLEURS_LICENSE = "CC-BY-4.0"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_SAMPLE_RATE = 16_000
FLEURS_LOCALES = {
    "vi": "vi_vn",
    "en": "en_us",
    "zh": "cmn_hans_cn",
    "ko": "ko_kr",
}


@dataclass(frozen=True)
class FleursRecord:
    sentence_id: str
    audio_file: str
    raw_transcription: str
    normalized_transcription: str
    num_samples: int
    gender: str

    @property
    def duration_s(self) -> float:
        return self.num_samples / FLEURS_SAMPLE_RATE


def read_fleurs_tsv(path: Path) -> list[FleursRecord]:
    """Read the seven-column TSV format stored in the pinned FLEURS repository."""

    records: list[FleursRecord] = []
    seen_audio_files: set[str] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = line.rstrip("\r\n").split("\t")
            if len(row) not in {6, 7}:
                message = f"expected 6 or 7 columns at {path}:{line_number}, found {len(row)}"
                raise ValueError(message)
            sentence_id, audio_file, raw, normalized, _characters, samples, *optional = row
            gender = optional[0] if optional else "UNKNOWN"
            if audio_file in seen_audio_files:
                raise ValueError(f"duplicate audio file {audio_file!r} in {path}")
            seen_audio_files.add(audio_file)
            try:
                num_samples = int(samples)
            except ValueError as error:
                message = f"invalid sample count at {path}:{line_number}: {samples!r}"
                raise ValueError(message) from error
            if num_samples <= 0:
                raise ValueError(f"non-positive sample count at {path}:{line_number}")
            records.append(
                FleursRecord(
                    sentence_id=sentence_id,
                    audio_file=audio_file,
                    raw_transcription=raw.strip(),
                    normalized_transcription=normalized.strip(),
                    num_samples=num_samples,
                    gender=gender,
                )
            )
    return records


def _stable_id(
    revision: str, source_locale: str, target_locale: str, split: Split, record: FleursRecord
) -> str:
    payload = "\0".join(
        (
            FLEURS_DATASET,
            revision,
            source_locale,
            target_locale,
            split,
            record.sentence_id,
            record.audio_file,
        )
    )
    return f"fleurs-{hashlib.sha256(payload.encode()).hexdigest()}"


def align_fleurs_records(
    source_records: Iterable[FleursRecord],
    target_records: Iterable[FleursRecord],
    *,
    src_lang: LanguageCode,
    tgt_lang: LanguageCode,
    split: Split,
    revision: str = FLEURS_REVISION,
) -> list[ManifestRecord]:
    """Inner-join locale metadata by sentence ID without crossing official splits."""

    direction = f"{src_lang}-{tgt_lang}"
    if direction not in DIRECTION_KEYS:
        raise ValueError(f"unsupported direction: {direction}")
    source_locale = FLEURS_LOCALES[src_lang]
    target_locale = FLEURS_LOCALES[tgt_lang]
    source_rows = list(source_records)
    target_texts: dict[str, set[str]] = {}
    for target in target_records:
        target_texts.setdefault(target.sentence_id, set()).add(target.raw_transcription)
    ambiguous_targets = [key for key, values in target_texts.items() if len(values) != 1]
    if ambiguous_targets:
        raise ValueError(f"inconsistent target text for sentence IDs: {ambiguous_targets[:10]}")
    target_by_id = {key: next(iter(values)) for key, values in target_texts.items()}
    aligned_sources = sorted(
        (record for record in source_rows if record.sentence_id in target_by_id),
        key=lambda record: (int(record.sentence_id), record.audio_file),
    )

    source_split = "dev" if split == "validation" else split
    output: list[ManifestRecord] = []
    for source in aligned_sources:
        if source.duration_s > 30:
            continue
        output.append(
            ManifestRecord(
                id=_stable_id(revision, source_locale, target_locale, split, source),
                source_item_id=f"{source.sentence_id}:{source.audio_file}",
                semantic_group_id=source.sentence_id,
                source_locale=source_locale,
                source_audio_ref=f"data/{source_locale}/audio/{source_split}/{source.audio_file}",
                duration_s=source.duration_s,
                speaker_id=None,
                src_lang=src_lang,
                tgt_lang=tgt_lang,
                src_text=source.raw_transcription,
                tgt_text=target_by_id[source.sentence_id],
                target_kind="gold",
                domain="general_read",
                split=split,
                source_dataset=FLEURS_DATASET,
                source_revision=revision,
                source_license=FLEURS_LICENSE,
                materialization_status="metadata_only",
                quality_flags=(
                    "metadata_only",
                    "missing_speaker_id",
                    "requires_manual_alignment_audit",
                ),
            )
        )
    return output
