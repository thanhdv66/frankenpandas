# FP-P2C-007 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-007` |
| input_contract | series or column with mixed valid/missing values |
| output_contract | reduced scalar, filled series, or filtered series with deterministic missing semantics |
| error_contract | incompatible fill type or coercion failure fails closed |
| null_contract | missing propagation is monotonic; reductions skip missing by default; fillna replaces; dropna removes |
| index_alignment_contract | fillna/dropna preserve or contract index; reductions return scalar |
| strict_mode_policy | fail-closed on unknown coercion/reduction ambiguity |
| hardened_mode_policy | explicit bounded recovery with evidence logging |
| excluded_scope | skipna=False paths, interpolation, bfill/ffill, min_count parameter |
| oracle_tests | pandas `fillna()`, `dropna()`, `sum()`, `mean()` contract slices via oracle adapter |
| performance_sentinels | dense null columns, large validity mask operations |
| compatibility_risks | NaN-vs-Null conflation in edge cases, all-NaN reduction semantics |
| raptorq_artifacts | pending conformance harness integration |
