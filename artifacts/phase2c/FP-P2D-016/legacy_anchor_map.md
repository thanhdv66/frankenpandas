# FP-P2D-016 Legacy Anchor Map

Packet: `FP-P2D-016`
Subsystem: CSV parser+formatter edge-case parity matrix

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/io/parsers/readers.py` (`read_csv`, parser option normalization)
- `legacy_pandas_code/pandas/pandas/io/formats/csvs.py` (`DataFrame.to_csv` formatting behavior)
- `legacy_pandas_code/pandas/pandas/io/parsers/c_parser_wrapper.py` (C-engine parser error semantics)

## Extracted Behavioral Contract

1. Quoted fields preserve commas, escaped quotes, and embedded newlines as cell content.
2. CRLF and missing final newline variants parse without row/column loss.
3. UTF-8 header/value payloads round-trip without semantic corruption.
4. Mixed numeric/boolean/null columns round-trip with semantic value parity.
5. Malformed rows (ragged/extra fields/unclosed quotes) fail closed with parser errors.

## Rust Slice Implemented

- `crates/fp-io/src/lib.rs`: `read_csv_str`, `write_csv_string`
- `crates/fp-conformance/src/lib.rs`: `csv_round_trip` execution + differential/error comparator wiring
- `crates/fp-conformance/oracle/pandas_oracle.py`: pandas-backed `csv_round_trip` live oracle branch

## Type Inventory

- `fp_conformance::FixtureOperation::CsvRoundTrip`
- `fp_conformance::PacketFixture` fields: `csv_input`, `expected_bool`, `expected_error_contains`
- `fp_types::Scalar` variants observed via CSV inference: `Int64`, `Float64`, `Bool`, `Utf8`, `Null`

## Rule Ledger

1. Round-trip pass requires semantic equality between parsed frame and reparsed formatted frame.
2. Column-order and row-count drift are treated as critical mismatches.
3. Expected-error fixtures pass only when parser errors are observed (strict text containment in fixture mode).
4. Live oracle expected-error cases resolve to error-any semantics (fail-closed on success).

## Hidden Assumptions

1. Packet scope uses default CSV delimiter and UTF-8 string inputs.
2. Semantic equality intentionally ignores superficial CSV formatting differences.

## Undefined-Behavior Edges

1. Non-UTF8 byte-stream ingestion paths (`read_csv(path)` decode layer).
2. Parser option parity (`quotechar`, `escapechar`, delimiter override) outside current fixture schema.
