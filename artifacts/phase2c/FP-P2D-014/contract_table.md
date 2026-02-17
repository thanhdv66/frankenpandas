# FP-P2D-014 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-014` |
| input_contract | two DataFrame payloads (`frame`, `frame_right`), join policy (`join_type`) for merge operations, and optional merge key (`merge_on`) |
| output_contract | DataFrame output with deterministic row order and deterministic suffixing for duplicate non-key columns (`_left`, `_right`) |
| error_contract | invalid payloads or concat column mismatch fail closed with explicit error strings |
| null_contract | unmatched side rows materialize null values in non-key columns |
| index_alignment_contract | column-key merges preserve merge engine ordering; index-key merges use deterministic synthetic key column (`__index_key` default) |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | divergence only in explicit allowlisted defensive classes |
| excluded_scope | multi-column merge keys, custom suffixes, `indicator`, `validate`, column-wise concat (`axis=1`) |
| oracle_tests | pandas `merge` (inner/left/right/outer, key-on-column and synthetic key-on-index) + pandas `concat(axis=0)` |
| performance_sentinels | duplicate-key cartesian growth, unmatched-row null fill path, suffix collision path |
| compatibility_risks | key coercion parity, right/outer ordering stability, index-key synthetic-column drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
