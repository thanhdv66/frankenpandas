# FP-P2D-015 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-015` |
| input_contract | scalar series payload (`left`) containing numeric, boolean, and missing values |
| output_contract | scalar reduction output for `nan_sum`, `nan_mean`, `nan_min`, `nan_max`, `nan_std`, `nan_var`, `nan_count` |
| error_contract | malformed payloads fail closed with explicit error text |
| null_contract | missing values (`null`, `na_n`) are skipped; empty/insufficient populations return NaN-style null for mean/min/max/std/var |
| index_alignment_contract | index labels are irrelevant for scalar nanops reductions; only value multiset matters |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | divergence only in explicit allowlisted defensive classes |
| excluded_scope | rolling/window nanops, quantile/median packet, axis-wise DataFrame nanops |
| oracle_tests | pandas-backed scalar reductions with `skipna=True`, `ddof=1` for std/var |
| performance_sentinels | dense numeric scan throughput, mixed-type coercion overhead |
| compatibility_risks | empty/all-missing behavior, ddof handling, bool-to-numeric coercion parity |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
