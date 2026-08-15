"""Prepare one leakage-aware FLEURS cross-locale JSONL manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from moyi_s2tt.data.fleurs import FLEURS_REVISION, align_fleurs_records, read_fleurs_tsv
from moyi_s2tt.data.manifest import LanguageCode, Split, write_manifest
from moyi_s2tt.data.splits import require_leakage_safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-tsv", type=Path, required=True)
    parser.add_argument("--target-tsv", type=Path, required=True)
    parser.add_argument("--source-language", choices=("vi", "en", "zh", "ko"), required=True)
    parser.add_argument("--target-language", choices=("vi", "en", "zh", "ko"), required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--revision", default=FLEURS_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = align_fleurs_records(
        read_fleurs_tsv(args.source_tsv),
        read_fleurs_tsv(args.target_tsv),
        src_lang=cast(LanguageCode, args.source_language),
        tgt_lang=cast(LanguageCode, args.target_language),
        split=cast(Split, args.split),
        revision=args.revision,
    )
    records = require_leakage_safe(records)
    count = write_manifest(args.output, records)
    print(json.dumps({"rows": count, "output": str(args.output), "metadata_only": True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
