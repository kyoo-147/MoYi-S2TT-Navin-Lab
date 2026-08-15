import hashlib
import random
from pathlib import Path

import pytest

from moyi_s2tt.runtime.checkpointing import (
    CheckpointManager,
    CheckpointState,
    canonical_sha256,
    capture_python_rng,
    require_private_root,
    restore_python_rng,
)


def test_interrupted_rng_and_cursor_resume_matches_uninterrupted(tmp_path: Path) -> None:
    config_hash = canonical_sha256({"seed": 147, "batch": 4})
    random.seed(147)
    uninterrupted = [random.random() for _ in range(8)]

    random.seed(147)
    before = [random.random() for _ in range(3)]
    artifact = b'{"optimizer_step":3,"scheduler_step":3,"scaler":1.0}'
    state = CheckpointState(
        run_id="smoke",
        global_step=3,
        epoch=0,
        dataset_cursor=12,
        config_sha256=config_hash,
        dataset_revision="fixture-v1",
        git_commit="fixture-commit",
        rng_state={"python": capture_python_rng()},
        artifact_sha256={"trainer-state.json": hashlib.sha256(artifact).hexdigest()},
    )
    manager = CheckpointManager(tmp_path, "smoke")
    manager.save(state, {"trainer-state.json": artifact})

    random.seed(999)
    restored, artifacts = manager.load_latest(
        config_sha256=config_hash, dataset_revision="fixture-v1"
    )
    restore_python_rng(restored.rng_state["python"])
    after = [random.random() for _ in range(5)]

    assert before + after == uninterrupted
    assert restored.dataset_cursor == 12
    assert artifacts["trainer-state.json"] == artifact


def test_resume_rejects_incompatible_config(tmp_path: Path) -> None:
    artifact = b"state"
    state = CheckpointState(
        run_id="smoke",
        global_step=1,
        epoch=0,
        dataset_cursor=1,
        config_sha256="a" * 64,
        dataset_revision="v1",
        git_commit="commit",
        rng_state={"python": capture_python_rng()},
        artifact_sha256={"state.bin": hashlib.sha256(artifact).hexdigest()},
    )
    manager = CheckpointManager(tmp_path, "smoke")
    manager.save(state, {"state.bin": artifact})
    with pytest.raises(ValueError, match="config hash mismatch"):
        manager.load_latest(config_sha256="b" * 64, dataset_revision="v1")


def test_private_root_must_be_outside_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    with pytest.raises(ValueError, match="outside"):
        require_private_root(repo, {"MOYI_PRIVATE_ROOT": str(repo / "private")})
    external = tmp_path / "external"
    assert require_private_root(repo, {"MOYI_PRIVATE_ROOT": str(external)}) == external
