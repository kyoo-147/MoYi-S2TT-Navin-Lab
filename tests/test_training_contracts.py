from pathlib import Path

import pytest

from moyi_s2tt.data.manifest import ManifestRecord
from moyi_s2tt.training.contracts import (
    load_training_config,
    make_run_contract,
    require_compatible_run,
    require_trainer_checkpoint,
    select_training_rows,
    write_run_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def row(identifier: str) -> ManifestRecord:
    return ManifestRecord(
        id=identifier,
        source_item_id=f"{identifier}.wav",
        semantic_group_id=identifier,
        source_locale="vi_vn",
        source_audio_ref=f"audio/{identifier}.wav",
        audio_path=f"audio/{identifier}.wav",
        audio_sha256="a" * 64,
        duration_s=2,
        src_lang="vi",
        tgt_lang="en",
        src_text="xin chao",
        tgt_text="hello",
        target_kind="gold",
        domain="general_read",
        split="train",
        source_dataset="fixture",
        source_revision="v1",
        source_license="fixture-only",
        materialization_status="ready",
    )


def test_tiny_configs_pin_39m_initialization_and_budget() -> None:
    overfit = load_training_config(ROOT / "configs/training/vi-en-tiny-overfit.yaml")
    smoke = load_training_config(ROOT / "configs/training/vi-en-tiny-smoke.yaml")
    assert overfit.model_revision == "169d4a4341b33bc18d8881c4b69c2e104e1cc0af"
    assert overfit.max_rows == 32
    assert smoke.max_rows == 500
    assert smoke.gradient_accumulation_steps == 4


def test_selection_and_resume_contract_fail_closed(tmp_path: Path) -> None:
    config = load_training_config(
        ROOT / "configs/training/vi-en-tiny-overfit.yaml"
    ).model_copy(update={"max_rows": 2})
    selected = select_training_rows([row("row-c"), row("row-a"), row("row-b")], config)
    assert len(selected) == 2
    contract = make_run_contract(config, selected, "commit-a")
    path = tmp_path / "run-contract.json"
    write_run_contract(path, contract)
    require_compatible_run(path, contract)
    changed = contract.model_copy(update={"git_commit": "commit-b"})
    with pytest.raises(ValueError, match="resume contract mismatch"):
        require_compatible_run(path, changed)


def test_trainer_checkpoint_requires_optimizer_scheduler_and_rng(tmp_path: Path) -> None:
    (tmp_path / "trainer_state.json").write_text("{}")
    with pytest.raises(ValueError, match="optimizer.pt"):
        require_trainer_checkpoint(tmp_path)
    for name in ("optimizer.pt", "scheduler.pt", "rng_state.pth"):
        (tmp_path / name).write_bytes(b"fixture")
    require_trainer_checkpoint(tmp_path)
