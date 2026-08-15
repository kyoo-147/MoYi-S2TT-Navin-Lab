"""Fail closed when private, oversized, secret, data, or model files enter Git."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ALLOWED_ROOTS = {
    ".github",
    "apps",
    "configs",
    "data",
    "models",
    "native",
    "notebooks",
    "scripts",
    "src",
    "tests",
    "tools",
}
ALLOWED_ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    ".pre-commit-config.yaml",
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "uv.lock",
}
FORBIDDEN_PARTS = {
    "artifacts",
    "briefs",
    "checkpoints",
    "docs",
    "outputs",
    "private",
    "proposals",
    "research-notes",
    "secrets",
}
FORBIDDEN_DATA_PARTS = {"downloads", "interim", "processed", "pseudo_labels", "raw"}
FORBIDDEN_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".flac",
    ".gguf",
    ".m4a",
    ".mp3",
    ".onnx",
    ".pcm",
    ".pt",
    ".pth",
    ".qnn",
    ".safetensors",
    ".tflite",
    ".wav",
}
MAX_FILE_BYTES = 2 * 1024 * 1024
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".csv",
    ".ini",
    ".ipynb",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".tsv",
    ".yaml",
    ".yml",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:ghp|github_pat)_[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "Google API key": re.compile(r"AIza[0-9A-Za-z_-]{20,}"),
    "private key": re.compile(r"BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY"),
}
PRIVATE_PHRASES = (
    "organizer-" + "confidential",
    "technical proposal " + "draft",
    "private workspace document" + " — do not push",
)


def candidate_paths(root: Path) -> list[Path]:
    command = [
        "git",
        "ls-files",
        "--cached",
        "--others",
        "--exclude-standard",
        "-z",
    ]
    output = subprocess.check_output(command, cwd=root)
    return [root / value.decode() for value in output.split(b"\0") if value]


def check_paths(root: Path, paths: Iterable[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        parts = set(relative.parts)
        top_level = relative.parts[0]
        if len(relative.parts) == 1:
            if top_level not in ALLOWED_ROOT_FILES:
                errors.append(f"unapproved root file: {relative}")
        elif top_level not in ALLOWED_ROOTS:
            errors.append(f"unapproved top-level path: {relative}")
        if parts & FORBIDDEN_PARTS:
            errors.append(f"private path is forbidden: {relative}")
        if top_level == "data" and parts & FORBIDDEN_DATA_PARTS:
            errors.append(f"raw/derived data path is forbidden: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"binary data/model suffix is forbidden: {relative}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds {MAX_FILE_BYTES} bytes: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"unreviewed binary content: {relative}")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} detected: {relative}")
        lowered = text.lower()
        for phrase in PRIVATE_PHRASES:
            if phrase in lowered:
                errors.append(f"private phrase detected in {relative}: {phrase}")
    return errors


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    errors = check_paths(root, candidate_paths(root))
    if errors:
        print("Public boundary check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print("Public boundary check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
