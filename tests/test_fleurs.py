from pathlib import Path

import pytest

from moyi_s2tt.data.fleurs import FLEURS_REVISION, align_fleurs_records, read_fleurs_tsv

FIXTURES = Path(__file__).parent / "fixtures" / "fleurs"


def test_fleurs_fixture_alignment_is_deterministic() -> None:
    source = read_fleurs_tsv(FIXTURES / "vi_vn-train.tsv")
    target = read_fleurs_tsv(FIXTURES / "en_us-train.tsv")

    rows = align_fleurs_records(source, target, src_lang="vi", tgt_lang="en", split="train")

    assert [row.semantic_group_id for row in rows] == ["101", "101", "202"]
    assert [row.source_item_id for row in rows] == [
        "101:vi-101.wav",
        "101:vi-101b.wav",
        "202:vi-202.wav",
    ]
    assert rows[0].tgt_text == "Start machine number three."
    assert rows[0].duration_s == 2.0
    assert rows[0].source_revision == FLEURS_REVISION
    assert rows[0].materialization_status == "metadata_only"
    assert "requires_manual_alignment_audit" in rows[0].quality_flags
    assert rows == align_fleurs_records(
        source, target, src_lang="vi", tgt_lang="en", split="train"
    )


def test_fleurs_maps_validation_back_to_dev_audio_reference() -> None:
    source = read_fleurs_tsv(FIXTURES / "vi_vn-train.tsv")
    target = read_fleurs_tsv(FIXTURES / "en_us-train.tsv")
    rows = align_fleurs_records(
        source, target, src_lang="vi", tgt_lang="en", split="validation"
    )
    assert "/audio/dev/" in rows[0].source_audio_ref


def test_fleurs_rejects_unknown_direction() -> None:
    source = read_fleurs_tsv(FIXTURES / "vi_vn-train.tsv")
    with pytest.raises(ValueError, match="unsupported direction"):
        align_fleurs_records(source, source, src_lang="en", tgt_lang="ko", split="train")


def test_fleurs_rejects_malformed_tsv(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.tsv"
    malformed.write_text("only\ttwo\n", encoding="utf-8")
    with pytest.raises(ValueError, match="expected 6 or 7 columns"):
        read_fleurs_tsv(malformed)
