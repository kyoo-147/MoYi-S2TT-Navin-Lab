# Data boundary

Only schemas, source registries, checksums, manifests, and tiny redistribution-approved fixtures belong in Git.

Raw audio, derived datasets, pseudo-label corpora, private recordings, and model artifacts must remain in ignored local storage or separately approved remote storage. A downloader does not grant redistribution rights. Every accepted row must preserve its source revision, license, split, and teacher provenance.

## Pinned FLEURS metadata smoke

Download the VI/EN TSV metadata from the pinned public revision (audio is not downloaded):

```bash
uv run python scripts/download_fleurs.py
```

Prepare one metadata-only aligned manifest after mapping FLEURS `dev.tsv` to the canonical `validation` split:

```bash
uv run python scripts/prepare_fleurs_manifest.py \
  --source-tsv data/downloads/fleurs/<revision>/vi_vn/train.tsv \
  --target-tsv data/downloads/fleurs/<revision>/en_us/train.tsv \
  --source-language vi --target-language en --split train \
  --output data/manifests/vi-en-fleurs-train.jsonl
uv run moyi-s2tt validate-manifest data/manifests/vi-en-fleurs-train.jsonl
uv run python scripts/sample_alignment_audit.py \
  data/manifests/vi-en-fleurs-train.jsonl \
  --output data/downloads/vi-en-alignment-audit.tsv --size 100
```

Generated rows remain `metadata_only`: they intentionally have no audio hash or speaker ID and carry a manual-alignment-audit flag. They are not training-ready until audio is lawfully materialized, hashed, audited, and the manifest passes cross-split checks.

## Common Voice Scripted Speech 26.0

The public release registry pins `cv-corpus-26.0-2026-06-12`. The Vietnamese archive is listed as CC0-1.0, but current Common Voice terms provide access exclusively through Mozilla Data Collective (MDC), require web terms acceptance, and ask users not to mirror or redistribute dataset copies.

Verify the pinned public statistics without an MDC credential:

```bash
uv run python scripts/sync_common_voice_stats.py
```

To download after accepting the exact Vietnamese v26 terms in MDC, keep the credential outside Git and use the dataset ID/slug shown by MDC:

```bash
uv sync --extra data
MDC_API_KEY=... uv run python scripts/download_common_voice.py <MDC_DATASET_ID>
```

After extracting the official archive, generate a source-only manifest. Omit `--metadata-only` to require every accepted MP3 to exist, hash successfully, and pass `ffprobe` duration/codec/sample-rate/channel inspection:

```bash
uv run python scripts/prepare_common_voice_manifest.py \
  --locale-root data/downloads/common-voice/cv-corpus-26.0-2026-06-12/vi \
  --language vi --metadata-only \
  --output data/manifests/common-voice-26-vi.jsonl
```

Downloaded archives, extracted audio, generated manifests, credentials, and SDK logs are not public repository artifacts.

## Frozen evaluation IDs

Regenerate the VI→EN FLEURS freeze from the pinned, ignored metadata download and verify its canonical hash:

```bash
uv run python scripts/freeze_fleurs_evaluation.py \
  --source-root data/downloads/fleurs/<revision>/vi_vn \
  --target-root data/downloads/fleurs/<revision>/en_us
uv run moyi-s2tt validate-evaluation data/evaluation/vi-en-fleurs-v1.json
```

Every future training or pseudo-label pipeline must call the contamination gate against the 496 frozen exclusion IDs.
