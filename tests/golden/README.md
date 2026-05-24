# Golden Seed Set

This directory contains the curated regression set the evaluator runs against to
detect score drift across prompt/model changes.

## Why

Without this, every prompt tweak silently re-scales the radar's outputs. The golden
set is our anchor: change the harness, then `radar regression` MUST stay green.

## How to extend

1. Pick a high-signal historical resource the Lab clearly liked, hated, or watched.
2. Add it to `seeds.yaml` with a hand-set `expected_aggregate` and `expected_band`.
3. Use a generous `tolerance` (default 0.7) — we want to catch drift, not noise.
4. Run `radar regression` to verify it passes BEFORE checking the change in.

## Bands

- `strong_recommend` — aggregate >= 3.5
- `watch`            — 2.5 <= aggregate < 3.5
- `monitor`          — aggregate < 2.5

Bands are recomputed from `expected_aggregate` against the configured thresholds in
`settings.py`; if you change the thresholds, re-run regression and adjust expected
aggregates if needed.
