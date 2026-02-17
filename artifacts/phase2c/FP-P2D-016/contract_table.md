# FP-P2D-016 Contract Table

| Field | Contract |
|---|---|
| packet_id | `FP-P2D-016` |
| input_contract | CSV payload string in `csv_input` covering quoting/escaping/newline/UTF-8/mixed-type/error cases |
| output_contract | `csv_round_trip` returns boolean parity for parse→write→reparse semantic equivalence |
| error_contract | malformed CSV inputs must fail closed with explicit parser error text |
| null_contract | empty CSV fields round-trip as missing values without row/column drift |
| dtype_contract | numeric/boolean/utf8 mixed columns retain semantic cell values across round-trip |
| strict_mode_policy | zero critical drift tolerated |
| hardened_mode_policy | bounded divergence only in explicitly allowlisted defensive categories |
| parser_formatter_scope | quoting, escaped quotes, embedded newline cells, CRLF and no-trailing-newline variants |
| encoding_scope | UTF-8 unicode header/value round-trip fidelity |
| excluded_scope | custom delimiter/encoding options, non-UTF8 byte streams, chunked CSV reader paths |
| oracle_tests | pandas `read_csv` + `to_csv(index=False)` + reparse semantic-equivalence check |
| performance_sentinels | parser throughput on mixed rows, formatter overhead for quoted/escaped fields |
| compatibility_risks | parser error-class drift, bool/float coercion drift, unicode normalization drift |
| raptorq_artifacts | parity report, RaptorQ sidecar, and decode proof emitted per packet run |
