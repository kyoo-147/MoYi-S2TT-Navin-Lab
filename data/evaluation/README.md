# Frozen evaluation boundary

`vi-en-fleurs-v1.json` freezes the aligned FLEURS validation and test semantic IDs at the pinned dataset revision. Its union is a mandatory exclusion set for every training source and pseudo-label run.

The file contains IDs only, not transcripts or audio. FLEURS is a controlled general/read-speech anchor; it is not evidence for conversation or industrial performance. Human conversation and industrial evaluation sets remain `UNAVAILABLE` until approved recordings and accepted human references exist.
