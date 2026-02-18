# FP-P2D-032 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-032` |
| input_contract | two DataFrame payloads (`frame`, `frame_right`) with `concat_axis=0` and `concat_join=outer`; payloads may include explicit `column_order` |
| output_contract | `concat(axis=0, join='outer', sort=False)` preserves first-seen column order from left frame then appends newly-seen right columns |
| error_contract | invalid `concat_axis` and invalid `concat_join` fail closed with explicit compatibility errors |
| null_contract | existing null values are preserved; null fill introduced only for missing source columns |
| index_alignment_contract | output index is left labels followed by right labels with duplicates preserved |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | divergence only in explicit allowlisted defensive classes |
| selector_scope | `dataframe_concat` with `concat_axis=0` and `concat_join='outer'` plus column-order observability |
| excluded_scope | MultiIndex concat behavior, >2-frame packet matrices, axis=1 duplicate-index recovery |
| oracle_tests | pandas `concat(axis=0, join='outer', sort=False)` with non-lexical and disjoint schema ordering fixtures |
| performance_sentinels | union-column materialization cost, null-fill allocation, duplicate-index stability |
| compatibility_risks | column-order drift, union-column drift, null-fill drift, selector normalization drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
