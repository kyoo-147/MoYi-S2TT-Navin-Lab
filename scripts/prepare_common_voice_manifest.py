"""Prepare and report a Common Voice source-only manifest from an extracted archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from moyi_s2tt.data.common_voice import CommonVoiceAdapter, load_common_voice_split
from moyi_s2tt.data.manifest import LanguageCode, Split
from moyi_s2tt.data.source import SourceBatch, build_source_report, write_source_manifest
from moyi_s2tt.data.splits import require_leakage_safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locale-root", type=Path, required=True)
    parser.add_argument("--language", choices=("vi", "en", "zh", "ko"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    adapter = CommonVoiceAdapter(cast(LanguageCode, args.language))
    batches = []
    split_files = (("train.tsv", "train"), ("dev.tsv", "validation"), ("test.tsv", "test"))
    for filename, split in split_files:
        batches.append(
            load_common_voice_split(
                args.locale_root / filename,
                args.locale_root / "clip_durations.tsv",
                adapter=adapter,
                split=cast(Split, split),
                clips_root=None if args.metadata_only else args.locale_root / "clips",
            )
        )
    records = [record for batch in batches for record in batch.records]
    issues = tuple(issue for batch in batches for issue in batch.issues)
    require_leakage_safe(records)
    combined = SourceBatch(records=tuple(records), issues=issues)
    count = write_source_manifest(args.output, records)
    report = build_source_report(combined)
    payload = {"rows": count, "report": report.model_dump(mode="json")}
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
