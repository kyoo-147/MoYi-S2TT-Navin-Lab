import json
from pathlib import Path

from moyi_s2tt.runtime.colab import choose_profile, environment_report
from moyi_s2tt.runtime.runner import main
from tools.check_notebooks import check_notebook

ROOT = Path(__file__).resolve().parents[1]


def test_gpu_profiles_are_budgeted_for_t4_first() -> None:
    assert choose_profile("Tesla T4", 15_000).teacher_batch_size == 2
    assert choose_profile("NVIDIA L4", 23_000).student_batch_size == 8
    assert choose_profile("NVIDIA A100", 40_000).precision == "bf16"
    assert choose_profile(None, None).accelerator == "CPU"


def test_environment_report_is_truthful_on_current_host() -> None:
    report = environment_report()
    assert report.python
    assert report.accelerator
    assert "torch" in report.packages


def test_shared_runner_environment(capsys: object) -> None:
    assert main(["environment"]) == 0
    payload = json.loads(capsys.readouterr().out)  # type: ignore[attr-defined]
    assert payload["accelerator"]


def test_all_colab_notebooks_are_clean_and_thin() -> None:
    notebooks = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebooks) == 7
    for path in notebooks:
        assert check_notebook(path) == []
        payload = json.loads(path.read_text(encoding="utf-8"))
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in payload["cells"]
            if cell["cell_type"] == "code"
        )
        assert (
            "moyi_s2tt.runtime.runner" in code
            or "scripts/run_teacher_audit.py" in code
        )
        assert "drive.google.com" not in code
