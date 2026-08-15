from pathlib import Path

import yaml

from moyi_s2tt.data.common_voice import (
    COMMON_VOICE_DESCRIPTOR,
    AudioMetadata,
    CommonVoiceAdapter,
    load_common_voice_split,
)
from moyi_s2tt.data.fleurs import FLEURS_DESCRIPTOR, FleursSourceAdapter, read_fleurs_tsv
from moyi_s2tt.data.source import (
    DatasetAdapter,
    SourceBatch,
    build_source_report,
    iter_source_manifest,
    write_source_manifest,
)
from moyi_s2tt.data.splits import find_split_leakage

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "common_voice"
FLEURS_FIXTURES = Path(__file__).parent / "fixtures" / "fleurs"


def assert_adapter(adapter: DatasetAdapter) -> None:
    assert adapter.descriptor.revision


def test_dataset_adapters_share_descriptors() -> None:
    common_voice = CommonVoiceAdapter("vi")
    fleurs = FleursSourceAdapter("vi")
    assert_adapter(common_voice)
    assert_adapter(fleurs)
    assert common_voice.descriptor == COMMON_VOICE_DESCRIPTOR
    assert fleurs.descriptor == FLEURS_DESCRIPTOR
    assert common_voice.descriptor.redistribution == "restricted"


def test_common_voice_metadata_parser_keeps_only_valid_rows() -> None:
    batch = load_common_voice_split(
        FIXTURES / "train.tsv",
        FIXTURES / "clip_durations.tsv",
        adapter=CommonVoiceAdapter("vi"),
        split="train",
    )
    assert [record.source_item_id for record in batch.records] == ["clip-a.mp3", "clip-b.mp3"]
    assert batch.records[0].speaker_id == "speaker-a"
    assert batch.records[1].speaker_id is None
    assert batch.records[0].materialization_status == "metadata_only"
    assert set(issue.reason for issue in batch.issues) == {
        "clip_not_validated",
        "duration_over_30_seconds",
        "unsafe_or_non_mp3_path",
    }


def test_common_voice_materialization_hashes_and_probes_audio(tmp_path: Path) -> None:
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "clip-a.mp3").write_bytes(b"synthetic-audio-a")
    (clips / "clip-b.mp3").write_bytes(b"synthetic-audio-b")

    def probe(path: Path) -> AudioMetadata:
        duration = 2.0 if path.name == "clip-a.mp3" else 2.5
        return AudioMetadata(duration_s=duration, sample_rate=48_000, channels=1, codec="mp3")

    batch = load_common_voice_split(
        FIXTURES / "train.tsv",
        FIXTURES / "clip_durations.tsv",
        adapter=CommonVoiceAdapter("vi"),
        split="train",
        clips_root=clips,
        audio_probe=probe,
    )
    assert len(batch.records) == 2
    assert all(record.materialization_status == "ready" for record in batch.records)
    assert all(record.audio_sha256 for record in batch.records)
    assert batch.records[0].source_sample_rate == 48_000


def test_source_report_and_manifest_round_trip(tmp_path: Path) -> None:
    batches = [
        load_common_voice_split(
            FIXTURES / filename,
            FIXTURES / "clip_durations.tsv",
            adapter=CommonVoiceAdapter("vi"),
            split=split,
        )
        for filename, split in (
            ("train.tsv", "train"),
            ("dev.tsv", "validation"),
            ("test.tsv", "test"),
        )
    ]
    combined = SourceBatch(
        records=tuple(record for batch in batches for record in batch.records),
        issues=tuple(issue for batch in batches for issue in batch.issues),
    )
    report = build_source_report(combined)
    assert report.total_rows == 4
    assert report.splits["train"].rows == 2
    assert report.splits["validation"].rows == 1
    assert report.unique_speakers == 3
    assert report.rejection_reasons["duration_over_30_seconds"] == 1
    assert report.split_leakage_issues == 0
    assert report.duplicate_audio_rows == 0

    path = tmp_path / "source.jsonl"
    assert write_source_manifest(path, combined.records) == 4
    assert tuple(iter_source_manifest(path)) == combined.records


def test_source_leakage_detects_speaker_crossing() -> None:
    batch = load_common_voice_split(
        FIXTURES / "dev.tsv",
        FIXTURES / "clip_durations.tsv",
        adapter=CommonVoiceAdapter("vi"),
        split="validation",
    )
    original = batch.records[0]
    crossing = original.model_copy(
        update={
            "id": "commonvoice-crossing-test",
            "source_item_id": "other.mp3",
            "semantic_group_id": "other-sentence",
            "src_text": "Nội dung khác.",
            "split": "test",
        }
    )
    issues = find_split_leakage([original, crossing])
    assert {issue.key_type for issue in issues} == {"speaker"}


def test_fleurs_source_adapter_uses_same_source_contract() -> None:
    records = read_fleurs_tsv(FLEURS_FIXTURES / "vi_vn-train.tsv")
    batch = FleursSourceAdapter("vi").adapt(records, "train")
    assert len(batch.records) == 4
    assert batch.records[0].source_dataset == "google/fleurs"
    assert batch.records[0].materialization_status == "metadata_only"


def test_pinned_common_voice_registry_is_truthful() -> None:
    path = ROOT / "data" / "registry" / "common-voice-26-metadata-audit.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["release"] == "26.0"
    assert payload["public_stats"]["repository_revision"] == (
        "f99d8239d2796131b73ac99f92ee7cb4443bf3ba"
    )
    assert payload["locales"]["vi"]["validated_hours"] == 7.8
    assert payload["access"]["downloaded_in_this_checkpoint"] is False
