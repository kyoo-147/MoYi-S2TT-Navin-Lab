"""Fetch and verify pinned public Common Voice release statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

REVISION = "f99d8239d2796131b73ac99f92ee7cb4443bf3ba"
STATS_PATH = "datasets/scripted-speech/cv-corpus-26.0-2026-06-12.json"
STATS_SHA256 = "1fc8baffece087182d769cdeb8af5ba35d603b1c7963ff21285422355edacac2"
LOCALES = ("vi", "en", "zh-CN", "ko")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("data/downloads/common-voice-26-stats.json")
    )
    args = parser.parse_args()
    url = f"https://raw.githubusercontent.com/common-voice/cv-dataset/{REVISION}/{STATS_PATH}"
    request = urllib.request.Request(url, headers={"User-Agent": "moyi-s2tt-lab/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        content = response.read()
    digest = hashlib.sha256(content).hexdigest()
    if digest != STATS_SHA256:
        raise ValueError(f"Common Voice stats checksum mismatch: {digest}")
    payload = json.loads(content)
    selected = {locale: payload["locales"][locale] for locale in LOCALES}
    output = {
        "release": "26.0",
        "release_date": "2026-06-12",
        "repository_revision": REVISION,
        "stats_sha256": digest,
        "locales": selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
