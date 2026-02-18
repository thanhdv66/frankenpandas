# FP-P2D-031 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-031` |
| input_contract | two DataFrame payloads (`frame`, `frame_right`) with `concat_axis=0` and `concat_join=outer` |
| output_contract | `concat(axis=0, join='outer')` materializes row-wise concat over unioned columns with null fill for missing cells |
| error_contract | invalid `concat_axis` and invalid `concat_join` fail closed with explicit compatibility errors |
| null_contract | existing null values are preserved; new nulls are inserted only where a source frame does not provide a column value |
| index_alignment_contract | output index is left labels followed by right labels with duplicates preserved |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | divergence only in explicit allowlisted defensive classes |
| selector_scope | `dataframe_concat` with `concat_axis=0` and `concat_join='outer'` |
| excluded_scope | MultiIndex concat behavior, >2-frame packet matrices, axis=1 duplicate-index recovery |
| oracle_tests | pandas `concat(axis=0, join='outer', sort=False)` over overlap/disjoint/null/empty/error matrices |
| performance_sentinels | union-column materialization cost, null-fill allocation, duplicate-index stability |
| compatibility_risks | column union drift, null-fill drift, selector normalization drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
