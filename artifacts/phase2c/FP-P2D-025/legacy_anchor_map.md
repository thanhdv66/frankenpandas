# FP-P2D-025 Legacy Anchor Map

Packet: `FP-P2D-025`
Subsystem: DataFrame `loc`/`iloc` multi-axis selector parity

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/indexing.py` (`_LocIndexer`, `_iLocIndexer` list-like selector paths)
- `legacy_pandas_code/pandas/pandas/core/frame.py` (DataFrame selector materialization and axis contracts)
- `legacy_pandas_code/pandas/pandas/core/indexes/base.py` (label matching and positional normalization semantics)

## Extracted Behavioral Contract

1. Row selection respects requested order and preserves duplicate requests.
2. Optional column selector constrains output columns and fails on unknown labels.
3. Negative iloc positions resolve from the end; out-of-bounds fails closed.
4. Empty explicit column selector yields an empty-column frame with selected index preserved.

## Rust Slice Implemented

- `crates/fp-frame/src/lib.rs`: `loc_with_columns` / `iloc_with_columns` selector paths
- `crates/fp-conformance/src/lib.rs`: fixture execution + differential handling for DataFrame loc/iloc column selectors
- `crates/fp-conformance/fixtures/packets/fp_p2d_025_*`: packet fixture matrix

## Type Inventory

- `fp_frame::DataFrame` selector APIs: `loc`, `loc_with_columns`, `iloc`, `iloc_with_columns`
- `fp_conformance::PacketFixture` fields: `loc_labels`, `iloc_positions`, `column_order`
- `fp_index::IndexLabel` selector label model

## Rule Ledger

1. Missing row labels are compatibility-gate errors.
2. Missing or duplicate column selectors are compatibility-gate errors.
3. iloc normalization must be deterministic for negative positions.
4. Selector materialization preserves scalar/null payloads exactly.

## Hidden Assumptions

1. Column selector duplicates are fail-closed (DataFrame storage model is unique-keyed).
2. Column-order display semantics remain map-key deterministic in the current frame representation.

## Undefined-Behavior Edges

1. MultiIndex row/column selector semantics.
2. Callable/boolean/slice selectors and axis-broadcasting behavior.
