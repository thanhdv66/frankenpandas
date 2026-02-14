# FP-P2C-010 Legacy Anchor Map

Packet: `FP-P2C-010`
Subsystem: full `loc`/`iloc` branch-path semantics

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/indexing.py` (`_LocIndexer`, `_iLocIndexer`, `check_bool_indexer`, `convert_missing_indexer`)
- `legacy_pandas_code/pandas/pandas/core/indexes/base.py` (`Index.get_loc`, `Index.get_indexer`, `Index.slice_locs`)
- `legacy_pandas_code/pandas/pandas/core/series.py` (`__getitem__`, `__setitem__`, `_get_with`)

## Extracted Behavioral Contract

1. `loc` uses label-based indexing: scalar, list, slice, boolean mask.
2. `iloc` uses integer-position indexing: scalar, list, slice, boolean mask.
3. Branch-path decisions are deterministic and mode-consistent.
4. Boolean/indexer coercion semantics preserve pandas-observable contracts.
5. Missing-label access raises `KeyError` in strict mode.
6. Boolean mask must match index length exactly.

## Rust Slice Implemented

- `crates/fp-frame/src/lib.rs`: `Series::filter()` (boolean mask indexing), `DataFrame::filter_rows()`
- `crates/fp-frame/src/lib.rs`: `DataFrame::head()`, `DataFrame::tail()` (positional slicing)
- `crates/fp-index/src/lib.rs`: `Index::position_of()`, `Index::positions_of_all()`

## Type Inventory

- `fp_index::Index`
  - fields: `labels: Vec<IndexLabel>`, `duplicate_cache: OnceCell<bool>`
- `fp_index::IndexLabel`
  - variants: `Int64`, `Utf8`
- `fp_frame::Series`
  - methods: `filter(&self, mask: &Series)` for boolean indexing
- `fp_frame::DataFrame`
  - methods: `filter_rows(&self, mask: &Series)`, `head(n)`, `tail(n)`

## Rule Ledger

1. Boolean mask indexing:
   - mask must be Bool-typed Series,
   - mask length must match target length,
   - null values in mask are treated as false (not selected).
2. Positional slicing:
   - `head(n)` returns first n rows,
   - `tail(n)` returns last n rows,
   - out-of-bounds n is clamped to available rows.
3. Label-based lookup (planned):
   - `position_of(label)` returns first matching position,
   - missing labels produce None or error depending on mode.

## Error Ledger

- `FrameError::LengthMismatch` for boolean mask length disagreement.
- `ColumnError::TypeMismatch` for non-Bool mask in filter operations.
- `IndexError` for invalid position or label lookups.

## Hidden Assumptions

1. Full `loc`/`iloc` API not yet implemented; current coverage is boolean mask filtering and head/tail.
2. Coercion and indexer normalization logic spans multiple helper paths requiring branch matrix extraction.
3. Scalar label lookup, list-of-labels selection, and label-slice selection are deferred.

## Undefined-Behavior Edges

1. Full `loc` branch matrix (scalar, list, slice, callable, boolean).
2. Full `iloc` branch matrix (scalar, list, slice, boolean).
3. Mixed indexer combinations (e.g., `loc[bool_mask, label_list]`).
4. `__setitem__` assignment paths for both `loc` and `iloc`.
5. Advanced boolean indexer with MultiIndex.
