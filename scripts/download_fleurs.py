"""Download only pinned FLEURS metadata; audio remains an explicit separate action."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from moyi_s2tt.data.fleurs import FLEURS_LOCALES, FLEURS_REVISION

_SPLITS = ("train", "dev", "test")


def download(url: str, destination: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "moyi-s2tt-lab/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
        content = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default=FLEURS_REVISION)
    parser.add_argument("--locale", action="append", choices=sorted(FLEURS_LOCALES.values()))
    parser.add_argument("--split", action="append", choices=_SPLITS)
    parser.add_argument("--output", type=Path, default=Path("data/downloads/fleurs"))
    args = parser.parse_args()
    locales = args.locale or [FLEURS_LOCALES["vi"], FLEURS_LOCALES["en"]]
    splits = args.split or list(_SPLITS)
    results = []
    for locale in locales:
        for split in splits:
            url = (
                "https://huggingface.co/datasets/google/fleurs/resolve/"
                f"{args.revision}/data/{locale}/{split}.tsv"
            )
            destination = args.output / args.revision / locale / f"{split}.tsv"
            digest = download(url, destination)
            results.append(
                {"locale": locale, "split": split, "sha256": digest, "path": str(destination)}
            )
    print(json.dumps({"revision": args.revision, "files": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
