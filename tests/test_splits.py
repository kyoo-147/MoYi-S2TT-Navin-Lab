import pytest

from moyi_s2tt.data.manifest import ManifestRecord
from moyi_s2tt.data.splits import find_split_leakage, require_leakage_safe


def record(identifier: str, split: str, source_item_id: str, text: str) -> ManifestRecord:
    return ManifestRecord.model_validate(
        {
            "id": identifier,
            "source_item_id": source_item_id,
            "semantic_group_id": source_item_id,
            "source_locale": "vi_vn",
            "source_audio_ref": f"audio/{identifier}.wav",
            "duration_s": 1.0,
            "src_lang": "vi",
            "tgt_lang": "en",
            "src_text": text,
            "tgt_text": "target",
            "target_kind": "gold",
            "domain": "general_read",
            "split": split,
            "source_dataset": "fixture",
            "source_revision": "v1",
            "source_license": "CC0-1.0",
        }
    )


def test_leakage_detects_source_and_normalized_text_across_splits() -> None:
    rows = [
        record("row-train", "train", "same-id", "  Đóng   VAN "),
        record("row-test", "test", "same-id", "đóng van"),
    ]
    issues = find_split_leakage(rows)
    assert {issue.key_type for issue in issues} == {
        "semantic_group",
        "source_item",
        "source_text",
    }
    with pytest.raises(ValueError, match="split leakage detected"):
        require_leakage_safe(rows)


def test_leakage_safe_rows_are_materialized_once() -> None:
    rows = [
        record("row-train", "train", "train-id", "train text"),
        record("row-test", "test", "test-id", "test text"),
    ]
    assert require_leakage_safe(iter(rows)) == rows
