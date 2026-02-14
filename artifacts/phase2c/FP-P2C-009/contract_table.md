# FP-P2C-009 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2C-009` |
| input_contract | column/frame construction from typed value arrays with validity masks |
| output_contract | typed columnar storage with per-element validity tracking and dtype consistency |
| error_contract | length/type mismatches fail closed at construction boundaries |
| null_contract | packed bitvec validity mask tracks per-element nullity; validity propagated through all operations |
| index_alignment_contract | column length must match index length in DataFrame; reindex produces new column with validity |
| strict_mode_policy | invariant breach is fail-closed |
| hardened_mode_policy | bounded containment with mandatory forensic logging |
| excluded_scope | BlockManager consolidation, in-place mutation, Arrow-compatible memory layouts |
| oracle_tests | pandas DataFrame construction and storage-observable behavior via oracle adapter |
| performance_sentinels | large column allocation, validity mask bit operations, dtype promotion chains |
| compatibility_risks | per-column vs block storage model divergence from pandas internals |
| raptorq_artifacts | pending conformance harness integration |
