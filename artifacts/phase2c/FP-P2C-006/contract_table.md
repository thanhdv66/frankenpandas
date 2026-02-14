# FP-P2C-006 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-006` |
| input_contract | two indexed series or DataFrames with join key specification |
| output_contract | joined/concatenated series or DataFrame with deterministic index and value ordering |
| error_contract | unsupported join modes, mismatched column sets, and invalid alignment vectors fail closed |
| null_contract | unmatched keys produce missing values; null key handling is mode-gated |
| index_alignment_contract | join preserves key multiplicity semantics; concat preserves input ordering |
| strict_mode_policy | fail-closed on unsupported mode/metadata combinations |
| hardened_mode_policy | bounded allowlisted defenses with explicit divergence logging |
| excluded_scope | multi-column merges, right/outer join modes, axis-1 concat, full sort matrix |
| oracle_tests | pandas `merge()` and `concat()` contract slices via oracle adapter |
| performance_sentinels | duplicate key explosion, large union alignment overhead |
| compatibility_risks | join ordering drift, null-key handling mismatch across join modes |
| raptorq_artifacts | pending conformance harness integration |
