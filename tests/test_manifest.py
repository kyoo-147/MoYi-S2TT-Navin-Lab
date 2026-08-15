from pathlib import Path

import pytest

from moyi_s2tt.data.manifest import ManifestRecord, iter_manifest, write_manifest


def record(**overrides: object) -> ManifestRecord:
    values: dict[str, object] = {
        "id": "fixture-row-1",
        "source_item_id": "101",
        "semantic_group_id": "101",
        "source_locale": "vi_vn",
        "source_audio_ref": "data/vi_vn/audio/train/101.wav",
        "duration_s": 2.0,
        "src_lang": "vi",
        "tgt_lang": "en",
        "src_text": "khởi động máy",
        "tgt_text": "start the machine",
        "target_kind": "gold",
        "domain": "industrial",
        "split": "train",
        "source_dataset": "fixture",
        "source_revision": "fixture-v1",
        "source_license": "CC0-1.0",
    }
    values.update(overrides)
    return ManifestRecord.model_validate(values)


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    expected = record()
    assert write_manifest(path, [expected]) == 1
    assert list(iter_manifest(path)) == [expected]


def test_ready_record_requires_materialized_audio() -> None:
    with pytest.raises(ValueError, match="ready records require"):
        record(materialization_status="ready")


def test_teacher_target_requires_complete_provenance() -> None:
    with pytest.raises(ValueError, match="teacher targets require"):
        record(target_kind="teacher", teacher_id="teacher-a")

    teacher = record(
        target_kind="teacher",
        teacher_id="teacher-a",
        teacher_revision="revision-a",
        generation_config_sha256="a" * 64,
    )
    assert teacher.teacher_revision == "revision-a"
