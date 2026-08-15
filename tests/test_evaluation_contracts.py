from pathlib import Path

import pytest

from moyi_s2tt.data.source import SourceRecord
from moyi_s2tt.evaluation.contracts import (
    EvaluationReference,
    FrozenEvaluationSet,
    find_evaluation_contamination,
    load_frozen_evaluation,
    require_evaluation_clean,
)

ROOT = Path(__file__).resolve().parents[1]


def source(semantic_group_id: str) -> SourceRecord:
    return SourceRecord(
        id=f"source-{semantic_group_id}",
        source_item_id=f"{semantic_group_id}.wav",
        semantic_group_id=semantic_group_id,
        source_locale="vi_vn",
        source_audio_ref=f"audio/{semantic_group_id}.wav",
        duration_s=1.0,
        src_lang="vi",
        src_text="fixture text",
        domain="general_read",
        split="train",
        source_dataset="fixture",
        source_revision="v1",
        source_license="fixture-only",
    )


def test_checked_in_fleurs_evaluation_is_frozen() -> None:
    frozen = load_frozen_evaluation(ROOT / "data" / "evaluation" / "vi-en-fleurs-v1.json")
    assert len(frozen.validation_semantic_ids) == 149
    assert len(frozen.test_semantic_ids) == 347
    assert len(frozen.training_exclusion_semantic_ids) == 496
    assert frozen.sha256 == "0a8ef907dd1b3f123e6f05daa1010d1fc190714c376075cca43b96935ddd0901"


def test_frozen_contract_rejects_overlap_and_incomplete_exclusions() -> None:
    with pytest.raises(ValueError, match="overlap"):
        FrozenEvaluationSet(
            id="fixture",
            direction="vi-en",
            source_dataset="fixture",
            source_revision="v1",
            source_locale="vi",
            target_locale="en",
            validation_semantic_ids=("1",),
            test_semantic_ids=("1",),
            training_exclusion_semantic_ids=("1",),
        )


def test_training_contamination_fails_closed() -> None:
    frozen = load_frozen_evaluation(ROOT / "data" / "evaluation" / "vi-en-fleurs-v1.json")
    contaminated = source(frozen.test_semantic_ids[0])
    clean = source("not-a-frozen-id")
    issues = find_evaluation_contamination([clean, contaminated], frozen)
    assert [issue.record_id for issue in issues] == [contaminated.id]
    with pytest.raises(ValueError, match="training/evaluation contamination"):
        require_evaluation_clean([contaminated], frozen)
    require_evaluation_clean([clean], frozen)


def test_accepted_human_reference_requires_reviewer() -> None:
    values = {
        "id": "reference-1",
        "frozen_set_id": "industrial-v1",
        "source_manifest_id": "source-1",
        "semantic_group_id": "prompt-1",
        "split": "test",
        "src_lang": "vi",
        "tgt_lang": "en",
        "source_text": "dừng máy",
        "reference_text": "stop the machine",
        "reference_kind": "human_reviewed",
        "review_status": "accepted",
        "critical_categories": ("safety_command",),
    }
    with pytest.raises(ValueError, match="reviewer ID"):
        EvaluationReference.model_validate(values)
    reference = EvaluationReference.model_validate({**values, "reviewer_id": "reviewer-1"})
    assert reference.reference_kind == "human_reviewed"
