"""Truthful Colab/GPU environment detection and budget profiles."""

from __future__ import annotations

import importlib.metadata
import platform
import subprocess
import sys
from dataclasses import dataclass

from ..config import StrictModel


@dataclass(frozen=True)
class RuntimeProfile:
    accelerator: str
    precision: str
    teacher_batch_size: int
    student_batch_size: int


class EnvironmentReport(StrictModel):
    python: str
    platform: str
    accelerator: str
    gpu_memory_mib: int | None
    profile: dict[str, str | int]
    packages: dict[str, str | None]


def choose_profile(gpu_name: str | None, memory_mib: int | None) -> RuntimeProfile:
    name = (gpu_name or "").casefold()
    if "a100" in name:
        return RuntimeProfile("A100", "bf16", 8, 16)
    if "l4" in name:
        return RuntimeProfile("L4", "fp16", 4, 8)
    if "t4" in name:
        return RuntimeProfile("T4", "fp16", 2, 4)
    if gpu_name:
        return RuntimeProfile(f"OTHER_GPU:{gpu_name}", "fp16", 1, 2)
    return RuntimeProfile("CPU", "fp32", 1, 1)


def detect_gpu() -> tuple[str | None, int | None]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    first = result.stdout.strip().splitlines()[0]
    name, memory = (part.strip() for part in first.rsplit(",", maxsplit=1))
    return name, int(memory)


def environment_report() -> EnvironmentReport:
    gpu_name, memory = detect_gpu()
    profile = choose_profile(gpu_name, memory)
    packages: dict[str, str | None] = {}
    package_names = (
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "tokenizers",
        "onnx",
        "onnxruntime",
    )
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return EnvironmentReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        accelerator=profile.accelerator,
        gpu_memory_mib=memory,
        profile={
            "precision": profile.precision,
            "teacher_batch_size": profile.teacher_batch_size,
            "student_batch_size": profile.student_batch_size,
        },
        packages=packages,
    )
