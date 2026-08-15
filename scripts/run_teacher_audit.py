"""Run the pinned VI→EN teacher chain on private, materialized training audio."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from moyi_s2tt.data.source import SourceRecord, iter_source_manifest
from moyi_s2tt.distillation.cache import (
    PredictionCache,
    generate_cached,
    prediction_cache_key,
)
from moyi_s2tt.evaluation.contracts import load_frozen_evaluation, require_evaluation_clean
from moyi_s2tt.runtime.checkpointing import require_private_root
from moyi_s2tt.runtime.colab import environment_report
from moyi_s2tt.teachers.audit import (
    TeacherAuditEvidence,
    load_audit_config,
    write_sanitized_evidence,
)
from moyi_s2tt.teachers.base import TeacherInput
from moyi_s2tt.teachers.catalog import load_teacher_spec
from moyi_s2tt.teachers.huggingface import HuggingFaceNllbTeacher, HuggingFaceWhisperTeacher


def release_accelerator() -> None:
    gc.collect()
    try:
        import torch  # type: ignore[import-not-found]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ModuleNotFoundError:
        pass


def word_error_rate(reference: str, prediction: str) -> float:
    expected = reference.casefold().split()
    actual = prediction.casefold().split()
    previous = list(range(len(actual) + 1))
    for index, expected_word in enumerate(expected, start=1):
        current = [index]
        for other_index, actual_word in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[other_index] + 1,
                    previous[other_index - 1] + (expected_word != actual_word),
                )
            )
        previous = current
    return previous[-1] / max(1, len(expected))


def count_hits(
    cache: PredictionCache, teacher: Any, inputs: list[TeacherInput], config: Any
) -> int:
    return sum(
        cache.get(prediction_cache_key(teacher.spec, item, config)) is not None
        for item in inputs
    )


def timed_generate(
    cache: PredictionCache, teacher: Any, inputs: list[TeacherInput], config: Any
) -> tuple[list[Any], int, float]:
    hits = count_hits(cache, teacher, inputs, config)
    started = time.perf_counter()
    outputs = generate_cached(teacher, inputs, config, cache)
    return outputs, hits, time.perf_counter() - started


def resolve_audio(record: SourceRecord, audio_root: Path) -> str:
    if not record.audio_path:
        raise ValueError(f"source row {record.id} is not materialized")
    path = Path(record.audio_path)
    resolved = path if path.is_absolute() else audio_root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"audio is missing for {record.id}: {resolved}")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    if digest != record.audio_sha256:
        raise ValueError(f"audio hash mismatch for {record.id}")
    return str(resolved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    private_root = require_private_root(repo)
    config = load_audit_config(args.config)
    limit = min(args.limit or config.max_rows, config.max_rows)
    records = [
        row
        for row in iter_source_manifest(args.manifest)
        if row.split == config.source_split and row.duration_s <= config.max_audio_seconds
    ][:limit]
    if len(records) != limit:
        raise ValueError(f"requested {limit} audit rows, found {len(records)} eligible rows")
    frozen = load_frozen_evaluation(repo / config.frozen_evaluation)
    require_evaluation_clean(records, frozen)

    inputs = [
        TeacherInput(
            source_id=row.id,
            src_lang="vi",
            tgt_lang="en",
            audio_path=resolve_audio(row, args.audio_root),
            audio_sha256=row.audio_sha256,
        )
        for row in records
    ]
    audit_root = private_root / "teacher-audits" / config.id
    audit_root.mkdir(parents=True, exist_ok=True)
    cache_hits: dict[str, int] = {}
    elapsed: dict[str, float] = {}

    with PredictionCache(audit_root / "predictions.sqlite3") as cache:
        asr_spec = load_teacher_spec(repo / config.asr.spec)
        asr_config = config.asr.generation(config.seed)
        asr_teacher = HuggingFaceWhisperTeacher(asr_spec)
        asr, cache_hits["asr"], elapsed["asr"] = timed_generate(
            cache, asr_teacher, inputs, asr_config
        )
        del asr_teacher
        release_accelerator()

        mt_spec = load_teacher_spec(repo / config.mt.spec)
        mt_config = config.mt.generation(config.seed)
        mt_inputs = [
            TeacherInput(
                source_id=item.source_id,
                src_lang="vi",
                tgt_lang="en",
                source_text=prediction.text,
            )
            for item, prediction in zip(inputs, asr, strict=True)
        ]
        mt_teacher = HuggingFaceNllbTeacher(
            mt_spec,
            str(config.mt.parameters["source_language_token"]),
            str(config.mt.parameters["target_language_token"]),
        )
        mt, cache_hits["mt"], elapsed["mt"] = timed_generate(
            cache, mt_teacher, mt_inputs, mt_config
        )
        del mt_teacher
        release_accelerator()

        direct_spec = load_teacher_spec(repo / config.direct_s2tt.spec)
        direct_config = config.direct_s2tt.generation(config.seed)
        direct_teacher = HuggingFaceWhisperTeacher(direct_spec)
        direct, cache_hits["direct_s2tt"], elapsed["direct_s2tt"] = timed_generate(
            cache, direct_teacher, inputs, direct_config
        )
        del direct_teacher
        release_accelerator()

    private_output = audit_root / "predictions.jsonl"
    with private_output.open("w", encoding="utf-8", newline="\n") as handle:
        for row, asr_value, mt_value, direct_value in zip(records, asr, mt, direct, strict=True):
            handle.write(
                json.dumps(
                    {
                        "source_id": row.id,
                        "source_transcript": row.src_text,
                        "asr_teacher": asr_value.model_dump(mode="json"),
                        "mt_teacher": mt_value.model_dump(mode="json"),
                        "direct_s2tt_teacher": direct_value.model_dump(mode="json"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    report = environment_report()
    evidence = TeacherAuditEvidence(
        audit_id=config.id,
        status="VERIFIED_REAL_INFERENCE",
        rows_requested=limit,
        rows_completed=len(direct),
        cache_hits=cache_hits,
        elapsed_seconds={key: round(value, 3) for key, value in elapsed.items()},
        teacher_revisions={
            "asr": asr_spec.revision or "",
            "mt": mt_spec.revision or "",
            "direct_s2tt": direct_spec.revision or "",
        },
        generation_sha256={
            "asr": asr_config.sha256,
            "mt": mt_config.sha256,
            "direct_s2tt": direct_config.sha256,
        },
        gpu=report.accelerator,
        package_versions=report.packages,
        aggregate={
            "asr_wer_mean": round(
                sum(
                    word_error_rate(row.src_text, value.text)
                    for row, value in zip(records, asr, strict=True)
                )
                / len(records),
                6,
            ),
            "mt_direct_similarity_mean": round(
                sum(
                    SequenceMatcher(
                        None, mt_value.text.casefold(), direct_value.text.casefold()
                    ).ratio()
                    for mt_value, direct_value in zip(mt, direct, strict=True)
                )
                / len(records),
                6,
            ),
            "private_output_sha256": hashlib.sha256(private_output.read_bytes()).hexdigest(),
        },
        limitations=(
            "Teacher confidence proxies are not calibrated or compared across models.",
            "Audit outputs remain teacher labels in private storage and are never gold references.",
        ),
    )
    if args.report:
        write_sanitized_evidence(args.report, evidence)
    print(evidence.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
