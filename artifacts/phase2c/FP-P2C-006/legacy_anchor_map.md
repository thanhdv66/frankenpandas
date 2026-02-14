# FP-P2C-006 Legacy Anchor Map

Packet: `FP-P2C-006`
Subsystem: join + concat semantics

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/reshape/merge.py` (`merge`, `_MergeOperation`, `get_join_indexers`)
- `legacy_pandas_code/pandas/pandas/core/reshape/concat.py` (`concat`, `_Concatenator`, `_get_result`, `_make_concat_multiindex`)
- `legacy_pandas_code/pandas/pandas/core/indexes/base.py` (`Index.join`, `Index.union`, `Index.intersection`)

## Extracted Behavioral Contract

1. Join cardinality matches key multiplicity semantics for each join mode (inner, left, right, outer).
2. Duplicate keys expand to deterministic cross-product cardinality with nested-loop ordering.
3. Concat axis-0 preserves declared index ordering with optional deduplication.
4. Concat axis-1 performs column-union alignment with deterministic column ordering.
5. Null-side handling is deterministic: left join inserts missing for unmatched right keys; right join mirrors.

## Rust Slice Implemented

- `crates/fp-join/src/lib.rs`: `inner_join`, `left_join` with indexed series join semantics
- `crates/fp-frame/src/lib.rs`: `concat_series`, `concat_dataframes` for axis-0 concatenation

## Type Inventory

- `fp_join::JoinError`
  - variants: `Frame`, `Index`, `Column`
- `fp_join::JoinMode` (planned)
  - variants: `Inner`, `Left`, `Right`, `Outer`
- `fp_frame::Series`
  - fields: `name: String`, `index: Index`, `column: Column`
- `fp_frame::DataFrame`
  - fields: `index: Index`, `columns: BTreeMap<String, Column>`

## Rule Ledger

1. Inner join:
   - output contains only labels present in both inputs,
   - duplicate keys produce cross-product expansion (left_count * right_count rows per key).
2. Left join:
   - output preserves all left labels in left order,
   - unmatched right labels produce missing values.
3. Concat axis-0:
   - index labels concatenate in input order,
   - duplicate labels from different inputs are preserved.
4. Concat axis-1:
   - requires matching row counts or explicit index alignment,
   - column name collisions are currently rejected.

## Error Ledger

- `JoinError::Frame` propagation for series construction failures.
- `JoinError::Index` propagation for invalid alignment vectors.
- `JoinError::Column` propagation for reindex/column construction failures.
- `FrameError` for concat with mismatched column sets (DataFrame) or empty input list.

## Hidden Assumptions

1. Scoped join modes: only `inner` and `left` currently implemented.
2. Concat scoped to axis-0 only; axis-1 concat deferred.
3. Multi-column merge keys not yet supported.
4. Sort parameter interaction with join output ordering not yet scoped.

## Undefined-Behavior Edges

1. Full merge option matrix (suffixes, indicator, validate parameters).
2. Right and outer join modes.
3. Axis-1 concat with index alignment.
4. MultiIndex join/concat behavior.
5. Sort semantics in merge output.
