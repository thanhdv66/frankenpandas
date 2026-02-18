# FP-P2D-029 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-029` |
| input_contract | two DataFrame payloads (`frame`, `frame_right`) with `concat_axis=1` and `concat_join` selector |
| output_contract | `concat_axis=1`, `concat_join=inner` materializes deterministic intersection-index concat |
| error_contract | invalid `concat_join`, duplicate output columns, duplicate input index labels, and unsupported `axis=0, join=inner` fail closed |
| null_contract | existing null values are preserved for intersected rows |
| index_alignment_contract | intersection index preserves left frame label order (`sort=False` compatible) |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | divergence only in explicit allowlisted defensive classes |
| selector_scope | `dataframe_concat` with `concat_axis=1` and `concat_join in {outer, inner}` where this packet targets `inner` |
| excluded_scope | axis=0 inner join column-intersection semantics, MultiIndex concat, duplicate-column-preserving output model |
| oracle_tests | pandas `concat(axis=1, join='inner', sort=False)` for numeric/utf8/disjoint/null/error matrices |
| performance_sentinels | intersection index filtering cost, sparse overlap behavior, deterministic ordering under non-monotonic labels |
| compatibility_risks | index-order drift, join selector normalization drift, fail-closed error contract drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |

