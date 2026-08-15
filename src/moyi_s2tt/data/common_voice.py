"""Mozilla Common Voice Scripted Speech v26 metadata and audio adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .manifest import LanguageCode, MaterializationStatus, Split
from .source import DatasetDescriptor, SourceBatch, SourceIssue, SourceRecord

COMMON_VOICE_DATASET = "mozilla/common-voice-scripted-speech"
COMMON_VOICE_RELEASE = "26.0"
COMMON_VOICE_DATE = "2026-06-12"
COMMON_VOICE_REVISION = "cv-corpus-26.0-2026-06-12"
COMMON_VOICE_STATS_REVISION = "f99d8239d2796131b73ac99f92ee7cb4443bf3ba"
COMMON_VOICE_LICENSE = "CC0-1.0"
COMMON_VOICE_LOCALES: dict[LanguageCode, str] = {
    "vi": "vi",
    "en": "en",
    "zh": "zh-CN",
    "ko": "ko",
}
COMMON_VOICE_DESCRIPTOR = DatasetDescriptor(
    id=COMMON_VOICE_DATASET,
    revision=COMMON_VOICE_REVISION,
    license=COMMON_VOICE_LICENSE,
    access="authenticated",
    homepage_url="https://commonvoice.mozilla.org/en/datasets",
    terms_url="https://commonvoice.mozilla.org/terms",
    redistribution="restricted",
)
_REQUIRED_FIELDS = {
    "client_id",
    "path",
    "sentence_id",
    "sentence",
    "up_votes",
    "down_votes",
    "locale",
}


@dataclass(frozen=True)
class AudioMetadata:
    duration_s: float
    sample_rate: int
    channels: int
    codec: str


@dataclass(frozen=True)
class CommonVoiceAdapter:
    language: LanguageCode

    @property
    def descriptor(self) -> DatasetDescriptor:
        return COMMON_VOICE_DESCRIPTOR

    @property
    def locale(self) -> str:
        return COMMON_VOICE_LOCALES[self.language]


def probe_audio_ffprobe(path: Path) -> AudioMetadata:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,sample_rate,channels:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise ValueError(f"no audio stream found in {path}")
    stream = streams[0]
    return AudioMetadata(
        duration_s=float(payload["format"]["duration"]),
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
        codec=str(stream["codec_name"]),
    )


def read_clip_durations(path: Path) -> dict[str, float]:
    durations: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["clip", "duration[ms]"]:
            raise ValueError(f"unexpected clip-duration header in {path}: {reader.fieldnames}")
        for line_number, row in enumerate(reader, start=2):
            clip = row["clip"].strip()
            try:
                duration_s = float(row["duration[ms]"]) / 1000
            except ValueError as error:
                raise ValueError(f"invalid duration at {path}:{line_number}") from error
            if not clip or duration_s <= 0:
                raise ValueError(f"invalid clip duration row at {path}:{line_number}")
            if clip in durations:
                raise ValueError(f"duplicate clip duration for {clip!r}")
            durations[clip] = duration_s
    return durations


def _safe_clip_name(value: str) -> str | None:
    candidate = PurePosixPath(value)
    if (
        not value
        or "/" in value
        or "\\" in value
        or ":" in value
        or candidate.name != value
        or candidate.suffix.casefold() != ".mp3"
    ):
        return None
    return value


def _stable_id(locale: str, split: Split, path: str) -> str:
    payload = "\0".join((COMMON_VOICE_DATASET, COMMON_VOICE_REVISION, locale, split, path))
    return f"commonvoice-{hashlib.sha256(payload.encode()).hexdigest()}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common_voice_split(
    tsv_path: Path,
    duration_path: Path,
    *,
    adapter: CommonVoiceAdapter,
    split: Split,
    clips_root: Path | None = None,
    audio_probe: Callable[[Path], AudioMetadata] = probe_audio_ffprobe,
    duration_tolerance_s: float = 0.25,
) -> SourceBatch:
    """Validate one official split and return accepted source-only records."""

    durations = read_clip_durations(duration_path)
    records: list[SourceRecord] = []
    issues: list[SourceIssue] = []
    seen_paths: set[str] = set()
    with tsv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or ())
        missing = _REQUIRED_FIELDS - fields
        if missing:
            raise ValueError(f"missing Common Voice columns in {tsv_path}: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            raw_path = row["path"].strip()
            item = raw_path or f"line-{line_number}"
            clip_name = _safe_clip_name(raw_path)
            if clip_name is None:
                issues.append(SourceIssue(source_item=item, reason="unsafe_or_non_mp3_path"))
                continue
            if clip_name in seen_paths:
                issues.append(SourceIssue(source_item=item, reason="duplicate_clip_path"))
                continue
            seen_paths.add(clip_name)
            if row["locale"].strip() != adapter.locale:
                issues.append(SourceIssue(source_item=item, reason="locale_mismatch"))
                continue
            sentence_id = row["sentence_id"].strip()
            sentence = row["sentence"].strip()
            if not sentence_id or not sentence:
                issues.append(SourceIssue(source_item=item, reason="missing_sentence_metadata"))
                continue
            try:
                up_votes = int(row["up_votes"])
                down_votes = int(row["down_votes"])
            except ValueError:
                issues.append(SourceIssue(source_item=item, reason="invalid_vote_count"))
                continue
            if up_votes + down_votes < 2 or up_votes <= down_votes:
                issues.append(SourceIssue(source_item=item, reason="clip_not_validated"))
                continue
            duration_s = durations.get(clip_name)
            if duration_s is None:
                issues.append(SourceIssue(source_item=item, reason="missing_clip_duration"))
                continue
            if duration_s > 30:
                issues.append(SourceIssue(source_item=item, reason="duration_over_30_seconds"))
                continue

            speaker_id = row["client_id"].strip() or None
            quality_flags: set[str] = set()
            audio_path: str | None = None
            audio_sha256: str | None = None
            source_sample_rate: int | None = None
            source_channels: int | None = None
            source_codec: str | None = None
            materialization_status: MaterializationStatus = "metadata_only"
            if clips_root is None:
                quality_flags.add("metadata_only")
            else:
                root = clips_root.resolve()
                candidate = (root / clip_name).resolve()
                if candidate.parent != root:
                    issues.append(SourceIssue(source_item=item, reason="unsafe_audio_resolution"))
                    continue
                if not candidate.is_file():
                    issues.append(SourceIssue(source_item=item, reason="audio_file_missing"))
                    continue
                try:
                    metadata = audio_probe(candidate)
                except (OSError, ValueError, subprocess.SubprocessError, KeyError, TypeError):
                    issues.append(SourceIssue(source_item=item, reason="audio_probe_failed"))
                    continue
                if abs(metadata.duration_s - duration_s) > duration_tolerance_s:
                    issues.append(SourceIssue(source_item=item, reason="duration_probe_mismatch"))
                    continue
                if (
                    metadata.duration_s <= 0
                    or metadata.sample_rate <= 0
                    or metadata.channels <= 0
                ):
                    issues.append(SourceIssue(source_item=item, reason="audio_probe_invalid"))
                    continue
                if metadata.codec.casefold() not in {"mp3", "mp3float"}:
                    issues.append(SourceIssue(source_item=item, reason="audio_codec_mismatch"))
                    continue
                if metadata.duration_s > 30:
                    issues.append(
                        SourceIssue(
                            source_item=item, reason="probed_duration_over_30_seconds"
                        )
                    )
                    continue
                audio_path = str(candidate)
                audio_sha256 = _sha256(candidate)
                source_sample_rate = metadata.sample_rate
                source_channels = metadata.channels
                source_codec = metadata.codec
                duration_s = metadata.duration_s
                materialization_status = "ready"
            if speaker_id is None:
                quality_flags.add("missing_speaker_id")

            records.append(
                SourceRecord(
                    id=_stable_id(adapter.locale, split, clip_name),
                    source_item_id=clip_name,
                    semantic_group_id=sentence_id,
                    source_locale=adapter.locale,
                    source_audio_ref=(
                        f"{COMMON_VOICE_REVISION}/{adapter.locale}/clips/{clip_name}"
                    ),
                    audio_path=audio_path,
                    audio_sha256=audio_sha256,
                    duration_s=duration_s,
                    source_sample_rate=source_sample_rate,
                    source_channels=source_channels,
                    source_codec=source_codec,
                    speaker_id=speaker_id,
                    src_lang=adapter.language,
                    src_text=sentence,
                    domain="general_read",
                    split=split,
                    source_dataset=COMMON_VOICE_DATASET,
                    source_revision=COMMON_VOICE_REVISION,
                    source_license=COMMON_VOICE_LICENSE,
                    materialization_status=materialization_status,
                    quality_flags=tuple(sorted(quality_flags)),
                )
            )
    return SourceBatch(records=tuple(records), issues=tuple(issues))
