# FP-P2C-010 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-010` |
| input_contract | series/DataFrame with label or positional indexer specification |
| output_contract | subset series/DataFrame preserving dtype and index contracts |
| error_contract | length mismatch, type mismatch, and missing label errors fail closed |
| null_contract | null values in boolean mask treated as false (not selected) |
| index_alignment_contract | filter preserves selected labels in original order; head/tail preserves positional order |
| strict_mode_policy | fail-closed on unsupported branch surfaces |
| hardened_mode_policy | allowlisted bounded continuation with decision ledger |
| excluded_scope | full loc/iloc API (scalar, list, slice, callable), setitem assignment paths, MultiIndex |
| oracle_tests | pandas `loc[]`, `iloc[]`, boolean indexing contract slices via oracle adapter |
| performance_sentinels | large boolean mask evaluation, index label lookup performance |
| compatibility_risks | boolean mask null handling, label-based vs position-based indexing confusion |
| raptorq_artifacts | pending conformance harness integration |
