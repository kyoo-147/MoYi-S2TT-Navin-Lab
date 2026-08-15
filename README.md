# MoYi S2TT Navin Lab

Training, distillation, evaluation, and mobile export tooling for compact multilingual speech-to-text translation (S2TT) models.

## Scope

The shared code is direction-agnostic and reserves six Vietnamese-centered directions:

- `vi-en`, `en-vi`
- `vi-zh`, `zh-vi`
- `vi-ko`, `ko-vi`

The first active research slice is **Vietnamese to English (`vi-en`)**. Other directions remain disabled until the same data, quality, and deployment gates can be applied truthfully.

Model experiments use four evidence-driven tiers:

| Tier | Intended role |
|---|---|
| `tiny` | 30–50M pipeline smoke proof |
| `mobile` | 70–90M low-latency candidate |
| `base` | 120–160M primary quality candidate |
| `plus` | 200–250M optional measured upper bound |

Parameter ranges are targets, not released model claims.

## Repository boundary

This repository tracks source code, tests, clean notebooks, configuration, dataset provenance metadata, and tiny approved fixtures. It does **not** track raw datasets, bulk pseudo-labels, checkpoints, model weights, private recordings, credentials, or internal competition/business material.

Dataset and model licenses remain asset-specific. No dataset or checkpoint is redistributed merely because a downloader or manifest exists here.

## Development

Install [uv](https://docs.astral.sh/uv/) and run:

```bash
uv sync --all-groups
uv run ruff check .
uv run mypy src
uv run pytest
uv run python tools/check_public_boundary.py
uv run python tools/check_notebooks.py
```

Validate the six direction contracts:

```bash
uv run moyi-s2tt validate-configs
uv run moyi-s2tt list-directions
```

## Licensed source adapters

FLEURS and Common Voice share a canonical source-audio contract. Common Voice Scripted Speech 26.0 is pinned for VI/EN/ZH-CN/KO metadata, with VI first. Its content is listed as CC0-1.0, while current access terms require Mozilla Data Collective authentication and terms acceptance and restrict mirroring. No Common Voice archive or row is redistributed here, and the source remains unaccepted until a real archive checksum, TSV parse, audio probe, and cross-split leakage report pass.

## Resumable Colab workflow

Seven output-free notebooks under `notebooks/` delegate to one shared runtime runner. Package versions are pinned in `requirements/colab.lock.txt`; T4 is the required budget baseline, with conservative L4/A100 profiles. Private inputs and checkpoints must be rooted outside this checkout through `MOYI_PRIVATE_ROOT`—no Drive ID is embedded in the repository.

Atomic checkpoints record config and dataset revisions, Git commit, global step, epoch, data cursor, RNG state, and SHA-256 hashes for caller-supplied optimizer/scheduler/scaler artifacts. Resume fails closed on incompatible config/data or corrupt payloads.

```bash
uv run python -m moyi_s2tt.runtime.runner environment
uv run python tools/check_notebooks.py
```

The foundation and interruption test are locally verified. Fresh Colab/T4 execution remains `UNVERIFIED` until a real hosted run is captured; later stage notebooks intentionally return `execution_not_implemented` until their corresponding checkpoints land.

## Frozen evaluation

`vi-en-fleurs-v1` freezes 149 aligned validation and 347 aligned test semantic IDs at the pinned FLEURS revision. Their 496-ID union is a mandatory exclusion set for training and pseudo-label generation. Conversation and industrial human evaluation remain explicitly unavailable; FLEURS read speech is not used to claim those domains.

```bash
uv run moyi-s2tt validate-evaluation data/evaluation/vi-en-fleurs-v1.json
```

## Frozen non-KD baseline

`vi-en-non-kd-baseline-v1` freezes the same Tiny initialization, seed, 500-row/500-step budget, evaluation freeze, and `eval_loss` checkpoint rule used for later KD comparison. Its data policy permits only genuine `gold` references and rejects teacher/synthetic contamination. The run is truthfully `BLOCKED`: no accepted genuine VI→EN speech-translation training manifest exists yet.

## Whisper Tiny training gate

The 39M `openai/whisper-tiny` initialization is pinned for two VI→EN pipeline gates: a deterministic 32-row overfit run and a bounded 500-row smoke run. Selection hashes seed/source IDs, validates audio hashes and frozen-evaluation exclusion, and uses length-bounded materialized audio only.

Before resuming, the runner compares config/data/model/selected-ID contracts and rejects incomplete Hugging Face checkpoints missing optimizer, scheduler, trainer, or RNG state. All weights and decoded examples stay under `MOYI_PRIVATE_ROOT`; only aggregate evidence may enter Git. GPU runs remain `NOT_RUN` until accepted private labels and signed-in Colab are available.

## Resumable pseudo-label shards

`ShardRunner` sorts source IDs, hashes each shard input, writes JSONL atomically, and records attempts/output hashes in SQLite. Completed shards are integrity-checked and skipped on resume. Private `PseudoLabelCandidate` rows retain ASR, MT, direct-S2TT, revisions, licenses, timestamps, generation hashes, or an explicit failure stage/reason.

Filtering adds source-transcript WER and duration/text-length gates to number, unit, negation, repetition, language-ID, and multi-teacher agreement checks. Reports separate accepted/rejected counts and hours by reason/domain/duration. Human false-accept/false-reject calibration remains pending; deterministic thresholds are not described as calibrated.

## Pinned teacher audit

The first VI→EN audit contract pins Whisper Turbo ASR, NLLB-200 distilled 600M MT, and Whisper large-v3 direct speech translation to immutable Hugging Face commits. `openai/whisper-tiny` is pinned as the 39M smoke initialization. These are reproducibility declarations, not approvals or quality evidence.

`scripts/materialize_fleurs_audit.py` streams at most 100 leakage-clean FLEURS training rows into `MOYI_PRIVATE_ROOT`; `scripts/run_teacher_audit.py` validates every audio hash, reuses the content-addressed cache, and emits only sanitized aggregates. NLLB remains noncommercial research-only, and SeamlessM4T is `HOLD`.

```bash
export MOYI_PRIVATE_ROOT=/path/outside/this/repository
uv run python scripts/materialize_fleurs_audit.py --limit 100
# Real inference requires the pinned Colab environment and materialized private manifest.
```

## Offline teacher boundary

Teacher candidates for Whisper, NLLB, OPUS-MT, and Seamless are declarative and validated with:

```bash
uv run moyi-s2tt validate-teachers
```

No teacher is approved or pinned yet, so production inference intentionally refuses to run. The shared interface and SQLite cache key every prediction by input content, teacher revision, and generation configuration. Cached labels retain teacher/license provenance and remain explicitly unfiltered until later quality gates accept or reject them. Bulk labels and cache databases are ignored by Git.

## Label filtering

Teacher labels remain non-gold. The first deterministic VI→EN gates reject changed numbers, abbreviated units, missing negation, excessive repetition, invalid target-language checks, and optional low lexical teacher agreement. Confidence thresholds remain unset until each teacher proxy is calibrated. Lexical overlap is explicitly a triage proxy, not semantic equivalence; accepted rows still require downstream audit and frozen evaluation.

## Status

The repository foundation and canonical manifest contracts are implemented. Pinned FLEURS metadata tooling can produce leakage-checked, metadata-only VI→EN rows; no FLEURS audio has been accepted or redistributed. No model-quality, accepted dataset-scale, mobile-latency, or accelerator-placement result is claimed until a versioned evidence artifact is produced.

## License

No license has been granted yet. Source is visible for review; reuse rights remain reserved until a license is selected.
