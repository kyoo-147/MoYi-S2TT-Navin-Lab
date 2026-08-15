"""Download and verify a Common Voice archive through the official MDC SDK."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

EXPECTED_ARCHIVE_SHA256 = {
    "vi": "5cc6d3ee20dcddd63f7d29357e23e4232251efb5a8d13c7f3833b615c489d7b3",
    "en": "6809228e6ab506d18f6a1ebc830056450f8266c8f513d6038bdb0fc88a49e6cb",
    "zh-CN": "4d8322c782e425c9af2d41c476eb63cc81db83684e4e5d64a1bdee2c1acc84e9",
    "ko": "169b269b86f0acd959d3f2fd1c098838ec458dbe0d4077a62b4bdb31ab2e8c4b",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_id", help="MDC dataset ID or slug shown after accepting terms")
    parser.add_argument("--locale", choices=tuple(EXPECTED_ARCHIVE_SHA256), default="vi")
    parser.add_argument("--output", type=Path, default=Path("data/downloads/common-voice"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("MDC_API_KEY"):
        parser.error("MDC_API_KEY is required; accept dataset terms in MDC before downloading")
    try:
        from datacollective import download_dataset, get_dataset_details
    except ImportError as error:
        raise SystemExit("install the data extra: uv sync --extra data") from error

    details: Any = get_dataset_details(args.dataset_id)
    license_value = str(
        getattr(details, "licenseAbbreviation", None) or getattr(details, "license", "")
    )
    locale_value = str(getattr(details, "locale", ""))
    name_value = str(getattr(details, "name", ""))
    expected_hash = EXPECTED_ARCHIVE_SHA256[args.locale]
    if locale_value != args.locale:
        raise ValueError(f"MDC locale mismatch: expected {args.locale}, got {locale_value}")
    if "26.0" not in name_value:
        raise ValueError(f"MDC dataset is not Common Voice Scripted Speech 26.0: {name_value}")
    if "CC0" not in license_value.replace("-", ""):
        raise ValueError(f"unexpected MDC license: {license_value}")
    api_checksum = str(getattr(details, "checksum", "") or "").removeprefix("sha256:")
    if api_checksum and api_checksum != expected_hash:
        raise ValueError(f"MDC metadata checksum mismatch: {api_checksum}")

    print(
        json.dumps(
            {
                "dataset_id": args.dataset_id,
                "expected_sha256": expected_hash,
                "license": license_value,
                "locale": locale_value,
                "name": name_value,
            },
            sort_keys=True,
        )
    )
    archive = download_dataset(
        args.dataset_id,
        download_directory=str(args.output),
        overwrite_existing=args.overwrite,
        enable_logging=True,
    )
    actual_hash = sha256_file(archive)
    if actual_hash != expected_hash:
        raise ValueError(f"downloaded Common Voice archive checksum mismatch: {actual_hash}")
    print(
        json.dumps(
            {
                "archive": str(archive),
                "sha256": actual_hash,
                "source": "Mozilla Data Collective",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
