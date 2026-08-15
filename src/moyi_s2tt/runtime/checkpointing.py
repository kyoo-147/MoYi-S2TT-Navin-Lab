"""Atomic, content-verified checkpoints for interruptible Colab jobs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from ..config import StrictModel


class CheckpointState(StrictModel):
    schema_version: int = 1
    run_id: str
    global_step: int = Field(ge=0)
    epoch: int = Field(ge=0)
    dataset_cursor: int = Field(ge=0)
    config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_revision: str
    git_commit: str
    rng_state: dict[str, Any]
    artifact_sha256: dict[str, str]

    @model_validator(mode="after")
    def validate_artifacts(self) -> CheckpointState:
        unsafe = (
            "/" in name or "\\" in name or name in {"", ".", ".."}
            for name in self.artifact_sha256
        )
        if any(unsafe):
            raise ValueError("checkpoint artifact names must be flat safe filenames")
        return self


class LatestPointer(StrictModel):
    checkpoint: str
    run_id: str
    global_step: int
    config_sha256: str
    dataset_revision: str


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def capture_python_rng() -> dict[str, Any]:
    version, values, gaussian = random.getstate()
    return {"version": version, "values": list(values), "gaussian": gaussian}


def restore_python_rng(value: Mapping[str, Any]) -> None:
    random.setstate((int(value["version"]), tuple(value["values"]), value["gaussian"]))


def require_private_root(repo_root: Path, environment: Mapping[str, str] = os.environ) -> Path:
    raw = environment.get("MOYI_PRIVATE_ROOT")
    if not raw:
        raise ValueError("MOYI_PRIVATE_ROOT is required for checkpoints and private artifacts")
    root = Path(raw).expanduser().resolve()
    repository = repo_root.resolve()
    if root == repository or repository in root.parents or root in repository.parents:
        raise ValueError("MOYI_PRIVATE_ROOT must be outside the public repository tree")
    root.mkdir(parents=True, exist_ok=True)
    return root


class CheckpointManager:
    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root.resolve() / "checkpoints" / run_id
        self.run_id = run_id
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def latest_path(self) -> Path:
        return self.root / "latest.json"

    def save(self, state: CheckpointState, artifacts: Mapping[str, bytes]) -> Path:
        if state.run_id != self.run_id:
            raise ValueError("checkpoint run ID does not match manager")
        actual_hashes = {
            name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()
        }
        if actual_hashes != state.artifact_sha256:
            raise ValueError("checkpoint artifact hashes do not match state")
        destination = self.root / f"step-{state.global_step:08d}"
        if destination.exists():
            raise FileExistsError(f"checkpoint already exists: {destination}")
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self.root))
        try:
            for name, content in artifacts.items():
                (temporary / name).write_bytes(content)
            (temporary / "state.json").write_text(
                json.dumps(state.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, destination)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        pointer = LatestPointer(
            checkpoint=destination.name,
            run_id=state.run_id,
            global_step=state.global_step,
            config_sha256=state.config_sha256,
            dataset_revision=state.dataset_revision,
        )
        pointer_tmp = self.latest_path.with_suffix(".json.tmp")
        pointer_tmp.write_text(pointer.model_dump_json(indent=2) + "\n", encoding="utf-8")
        os.replace(pointer_tmp, self.latest_path)
        return destination

    def load_latest(
        self, *, config_sha256: str, dataset_revision: str
    ) -> tuple[CheckpointState, dict[str, bytes]]:
        if not self.latest_path.is_file():
            raise FileNotFoundError(f"no latest checkpoint for {self.run_id}")
        pointer = LatestPointer.model_validate_json(self.latest_path.read_text(encoding="utf-8"))
        if pointer.run_id != self.run_id:
            raise ValueError("latest pointer run ID mismatch")
        if pointer.config_sha256 != config_sha256:
            raise ValueError("resume config hash mismatch")
        if pointer.dataset_revision != dataset_revision:
            raise ValueError("resume dataset revision mismatch")
        directory = self.root / pointer.checkpoint
        state_text = (directory / "state.json").read_text(encoding="utf-8")
        state = CheckpointState.model_validate_json(state_text)
        artifacts = {name: (directory / name).read_bytes() for name in state.artifact_sha256}
        actual = {name: hashlib.sha256(content).hexdigest() for name, content in artifacts.items()}
        if actual != state.artifact_sha256:
            raise ValueError("checkpoint artifact integrity failure")
        return state, artifacts
