# FP-P2C-008 Legacy Anchor Map

Packet: `FP-P2C-008`
Subsystem: IO first-wave contract (CSV + JSON)

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/io/parsers/readers.py` (`read_csv`, `TextFileReader`)
- `legacy_pandas_code/pandas/pandas/io/parsers/c_parser_wrapper.py` (C-backed CSV parser)
- `legacy_pandas_code/pandas/pandas/io/json/_json.py` (`read_json`, `to_json`, orient parameter dispatch)
- `legacy_pandas_code/pandas/pandas/io/common.py` (IO path resolution, compression, encoding)

## Extracted Behavioral Contract

1. CSV parser normalization is deterministic for scoped dialect/support (delimiter, header, index_col).
2. Malformed input paths are fail-closed with deterministic diagnostics.
3. Round-trip stability is preserved for supported schema/value surface.
4. JSON orient modes (records, columns, split, values, index) produce deterministic output shapes.
5. Type inference from CSV produces consistent dtypes across identical inputs.

## Rust Slice Implemented

- `crates/fp-io/src/lib.rs`: `read_csv`, `write_csv`, `read_csv_with_options`, `read_json_str`, `write_json_string`
- CSV options: delimiter, has_header, index_col, skip_rows
- JSON orientations: records, columns, split, values, index

## Type Inventory

- `fp_io::IoError`
  - variants: `Csv`, `Io`, `Frame`, `Column`, `JsonFormat`, `JsonSerialize`
- `fp_io::CsvReadOptions`
  - fields: `delimiter: u8`, `has_header: bool`, `index_col: Option<usize>`, `skip_rows: usize`
- `fp_io::JsonOrient`
  - variants: `Records`, `Columns`, `Split`, `Values`, `Index`

## Rule Ledger

1. CSV read:
   - first row is header by default (`has_header: true`),
   - dtype inference: integer -> Int64, float -> Float64, else Utf8,
   - missing values: empty cells produce `Scalar::Null(NullKind::Null)`,
   - index_col lifts specified column to index labels.
2. CSV write:
   - header row includes column names,
   - missing values written as empty string.
3. JSON read:
   - orient parameter determines parse strategy,
   - `records`: array of objects,
   - `columns`: {col_name: {idx: val}},
   - `split`: {index, columns, data} object.
4. JSON write:
   - orient parameter determines serialization format,
   - round-trip stable for supported orient modes.

## Error Ledger

- `IoError::Csv` for malformed CSV (parse failures, encoding issues).
- `IoError::Io` for file system errors (not found, permission denied).
- `IoError::JsonFormat` for invalid JSON structure or orient mismatch.
- `IoError::Frame` propagation for DataFrame construction failures.
- `IoError::Column` propagation for column construction failures.

## Hidden Assumptions

1. First wave scoped to CSV and JSON formats only.
2. Encoding is UTF-8 only; no codepage support.
3. Compression not yet supported.
4. CSV parser is pure Rust (no C-backed parser).
5. File-based read/write wraps string-based operations.

## Undefined-Behavior Edges

1. Full CSV option matrix (quoting, escaping, encoding, thousands separator).
2. Full JSON option matrix (date_format, default_handler, lines mode).
3. Excel, Parquet, SQL, HDF5, and other IO formats.
4. Chunked/streaming read for large files.
5. Compression/decompression pipelines.
