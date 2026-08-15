"""Generate clean thin notebooks that delegate to the shared runtime runner."""

from __future__ import annotations

import json
from pathlib import Path

BOOTSTRAP = '''import os
import subprocess
from pathlib import Path

if not Path("pyproject.toml").is_file():
    revision = os.environ.get("MOYI_REPO_REVISION", "main")
    checkout = Path("/content/moyi-s2tt-lab")
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            revision,
            "https://github.com/kyoo-147/MoYi-S2TT-Navin-Lab.git",
            str(checkout),
        ],
        check=True,
    )
    os.chdir(checkout)
'''


TEACHER_COMMAND = '''import os
import subprocess

private_root = os.environ.get("MOYI_PRIVATE_ROOT")
assert private_root, "Set MOYI_PRIVATE_ROOT to private Drive storage"
subprocess.run(["python", "scripts/materialize_fleurs_audit.py", "--limit", "100"], check=True)
subprocess.run(
    [
        "python",
        "scripts/run_teacher_audit.py",
        "--config",
        "configs/inference/vi-en-teacher-audit.yaml",
        "--manifest",
        f"{private_root}/fleurs-audit/source.jsonl",
        "--audio-root",
        f"{private_root}/fleurs-audit",
        "--report",
        "data/evidence/vi-en-teacher-audit-v1.json",
    ],
    check=True,
)
'''


NOTEBOOKS = {
    "00_environment_check.ipynb": "environment",
    "01_data_smoke.ipynb": "data",
    "02_teacher_generation_vi_en.ipynb": "teacher",
    "03_student_smoke_vi_en.ipynb": "tiny",
    "04_sequence_kd_vi_en.ipynb": "sequence-kd",
    "05_evaluation.ipynb": "evaluation",
    "06_export_quantize.ipynb": "export",
}


def notebook(command: str) -> dict[str, object]:
    if command == "environment":
        runtime_command = "!python -m moyi_s2tt.runtime.runner environment"
    elif command == "teacher":
        runtime_command = TEACHER_COMMAND
    else:
        runtime_command = f"!python -m moyi_s2tt.runtime.runner stage {command}"
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MoYi S2TT Colab stage\n",
                    "This clean notebook delegates to versioned package code. ",
                    "Private paths come from `MOYI_PRIVATE_ROOT`.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [BOOTSTRAP],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["%pip install -r requirements/colab.lock.txt\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [runtime_command + "\n"],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    root = Path("notebooks")
    root.mkdir(exist_ok=True)
    for filename, command in NOTEBOOKS.items():
        (root / filename).write_text(
            json.dumps(notebook(command), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
