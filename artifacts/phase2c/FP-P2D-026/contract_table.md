# FP-P2D-026 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-026` |
| input_contract | DataFrame `head_n` / `tail_n` cardinality selectors |
| output_contract | deterministic prefix/suffix row materialization preserving source columns and scalar values |
| error_contract | missing `frame`, missing `head_n`, or missing `tail_n` fails closed with explicit diagnostics |
| null_contract | null scalars remain unchanged through head/tail materialization |
| index_contract | output index is exact prefix (`head`) or suffix (`tail`) in original order |
| columns_contract | output columns match input schema exactly and preserve per-column ordering |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | bounded divergence only in explicitly allowlisted defensive categories |
| selector_scope | non-negative integer `n` for DataFrame head/tail |
| excluded_scope | negative `n`, axis=1 column head/tail, MultiIndex-specific truncation behaviors |
| oracle_tests | pandas `df.head(n)` and `df.tail(n)` parity baseline |
| performance_sentinels | bounded copy cost proportional to selected rows |
| compatibility_risks | off-by-one row truncation drift, index-order drift, null propagation drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
