"""Stream and materialize a bounded private FLEURS training audit batch."""

from __future__ import annotations

import argparse
import hashlib
import importlib
from pathlib import Path
from typing import Any

from moyi_s2tt.data.fleurs import FLEURS_REVISION
from moyi_s2tt.data.source import SourceRecord, write_source_manifest
from moyi_s2tt.evaluation.contracts import load_frozen_evaluation
from moyi_s2tt.runtime.checkpointing import require_private_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.limit <= 500:
        raise ValueError("audit limit must be between 1 and 500")

    repo = Path(__file__).resolve().parents[1]
    private_root = require_private_root(repo)
    output = (args.output or private_root / "fleurs-audit").resolve()
    if private_root != output and private_root not in output.parents:
        raise ValueError("audit materialization output must remain under MOYI_PRIVATE_ROOT")
    audio_root = output / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    frozen = load_frozen_evaluation(repo / "data/evaluation/vi-en-fleurs-v1.json")
    excluded = set(frozen.training_exclusion_semantic_ids)

    try:
        datasets = importlib.import_module("datasets")
        soundfile = importlib.import_module("soundfile")
    except ModuleNotFoundError as error:
        raise RuntimeError("materialization requires the pinned Colab packages") from error

    stream: Any = datasets.load_dataset(
        "google/fleurs",
        "vi_vn",
        split="train",
        revision=FLEURS_REVISION,
        streaming=True,
    )
    records: list[SourceRecord] = []
    for example in stream:
        semantic_id = str(example["id"])
        if semantic_id in excluded:
            continue
        audio = example["audio"]
        samples = audio["array"]
        sample_rate = int(audio["sampling_rate"])
        duration = len(samples) / sample_rate
        if not 0 < duration <= 30:
            continue
        item_id = Path(str(example["path"])).stem
        destination = audio_root / f"{item_id}.wav"
        soundfile.write(destination, samples, sample_rate, subtype="PCM_16")
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        records.append(
            SourceRecord(
                id=f"fleurs-vi_vn-train-{item_id.lower()}",
                source_item_id=str(example["path"]),
                semantic_group_id=semantic_id,
                source_locale="vi_vn",
                source_audio_ref=f"audio/{destination.name}",
                audio_path=f"audio/{destination.name}",
                audio_sha256=digest,
                duration_s=duration,
                source_sample_rate=sample_rate,
                source_channels=1,
                source_codec="wav-pcm16",
                src_lang="vi",
                src_text=str(example["transcription"]),
                domain="general_read",
                split="train",
                source_dataset="google/fleurs",
                source_revision=FLEURS_REVISION,
                source_license="CC-BY-4.0",
                materialization_status="ready",
            )
        )
        if len(records) == args.limit:
            break
    if len(records) != args.limit:
        raise ValueError(f"stream ended after {len(records)} eligible rows; requested {args.limit}")
    manifest = output / "source.jsonl"
    write_source_manifest(manifest, records)
    print(f"{manifest}: {len(records)} private training rows materialized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
