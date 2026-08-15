from pathlib import Path

import pytest

from moyi_s2tt.data.manifest import ManifestRecord
from moyi_s2tt.training.contracts import load_training_config, select_training_rows

ROOT = Path(__file__).resolve().parents[1]


def teacher_row() -> ManifestRecord:
    return ManifestRecord(
        id="teacher-row",
        source_item_id="clip.wav",
        semantic_group_id="train-id",
        source_locale="vi_vn",
        source_audio_ref="audio/clip.wav",
        audio_path="audio/clip.wav",
        audio_sha256="a" * 64,
        duration_s=2,
        src_lang="vi",
        tgt_lang="en",
        src_text="xin chao",
        tgt_text="hello",
        target_kind="teacher",
        domain="general_read",
        split="train",
        source_dataset="fixture",
        source_revision="v1",
        source_license="fixture",
        teacher_id="fixture/teacher",
        teacher_revision="v1",
        teacher_license="fixture",
        generation_config_sha256="b" * 64,
        teacher_generated_at="2026-08-15T00:00:00Z",
        materialization_status="ready",
        filter_decision="keep",
    )


def test_baseline_rejects_teacher_target_contamination() -> None:
    config = load_training_config(ROOT / "configs/training/vi-en-non-kd-baseline.yaml")
    assert config.allowed_target_kinds == ("gold",)
    with pytest.raises(ValueError, match="only 0 are eligible"):
        select_training_rows([teacher_row()], config.model_copy(update={"max_rows": 1}))
