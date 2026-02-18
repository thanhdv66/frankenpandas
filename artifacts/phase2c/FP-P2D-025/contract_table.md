# FP-P2D-025 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-025` |
| input_contract | DataFrame `loc_labels` / `iloc_positions` selectors with optional column selector (`column_order`) |
| output_contract | deterministic row+column subset materialization for valid selectors |
| error_contract | missing row labels, missing columns, duplicate column selectors, and out-of-bounds iloc fail closed with explicit diagnostics |
| null_contract | selection preserves source scalar/null values without coercive mutation |
| index_contract | selected index preserves selector order and duplicate row requests |
| columns_contract | selected columns constrained by explicit selector; empty selector yields empty-column frame |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | bounded divergence only in explicitly allowlisted defensive categories |
| selector_scope | list-like row selectors + optional list-like column selectors |
| excluded_scope | boolean indexers, slice/range indexers, callable selectors, MultiIndex axis selectors |
| oracle_tests | pandas `df.loc[[...], [...]]` and `df.iloc[[...], [...]]` parity baseline |
| performance_sentinels | selector traversal and materialization cost under duplicate-row requests |
| compatibility_risks | row-order drift, duplicate-row handling drift, error-taxonomy drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
