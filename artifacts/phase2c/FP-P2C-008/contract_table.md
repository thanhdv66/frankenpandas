# FP-P2C-008 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-008` |
| input_contract | file path or string data with format specification (CSV/JSON) |
| output_contract | DataFrame or Series with deterministic dtype inference and index assignment |
| error_contract | malformed input fails closed with deterministic error diagnostics |
| null_contract | empty/missing cells in CSV produce Null; JSON null produces Null |
| index_alignment_contract | index_col lifts column to index; default is sequential integer index |
| strict_mode_policy | fail-closed on unsupported metadata/features |
| hardened_mode_policy | bounded parser recovery for allowlisted corruption classes |
| excluded_scope | non-CSV/JSON formats, compression, chunked reads, non-UTF-8 encoding |
| oracle_tests | pandas `read_csv()`, `read_json()`, `to_csv()`, `to_json()` slices via oracle adapter |
| performance_sentinels | large file parse latency, dtype inference overhead |
| compatibility_risks | dtype inference divergence, CSV quoting edge cases, JSON orient compatibility |
| raptorq_artifacts | pending conformance harness integration |
