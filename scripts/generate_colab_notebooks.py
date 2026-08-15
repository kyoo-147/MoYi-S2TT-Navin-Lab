"""Generate clean thin notebooks that delegate to the shared runtime runner."""

from __future__ import annotations

import json
from pathlib import Path

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
    runtime_command = (
        "!python -m moyi_s2tt.runtime.runner environment"
        if command == "environment"
        else f"!python -m moyi_s2tt.runtime.runner stage {command}"
    )
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# MoYi S2TT Colab stage\\n",
                    "This clean notebook delegates to versioned package code. ",
                    "Private paths come from `MOYI_PRIVATE_ROOT`.\\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["%pip install -r requirements/colab.lock.txt\\n"],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [runtime_command + "\\n"],
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
