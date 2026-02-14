# FP-P2C-007 Legacy Anchor Map

Packet: `FP-P2C-007`
Subsystem: missingness + nanops reductions

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/missing.py` (`mask_missing`, `clean_fill_method`, `interpolate_2d_inplace`)
- `legacy_pandas_code/pandas/pandas/core/nanops.py` (`nansum`, `nanmean`, `nanmedian`, `nanvar`, `nancorr`, `_ensure_numeric`)
- `legacy_pandas_code/pandas/pandas/core/dtypes/missing.py` (`isna`, `notna`, `is_valid_na_for_dtype`)

## Extracted Behavioral Contract

1. Missing propagation is monotonic under composed operations: null in -> null out, never silently dropped.
2. NaN/NaT/null distinctions are preserved at observable API boundaries.
3. Reduction defaults and numeric coercion are deterministic: `skipna=True` by default.
4. `nansum` treats all-NaN groups as 0.0 (not NaN), matching pandas behavior.
5. `nanmean` of empty or all-NaN returns NaN.
6. Fillna replaces missing values with specified scalar, preserving dtype via cast.
7. Dropna removes rows with any missing values.

## Rust Slice Implemented

- `crates/fp-types/src/lib.rs`: `Scalar::is_missing()`, `NullKind` enum, `fill_na()`, `dropna()`, `nansum()`, `nanmean()`
- `crates/fp-columnar/src/lib.rs`: `Column::fillna()`, `Column::dropna()`, `ValidityMask` null tracking
- `crates/fp-frame/src/lib.rs`: `Series::fillna()`, `Series::dropna()`, `Series::sum()`, `Series::mean()`, `Series::count()`

## Type Inventory

- `fp_types::NullKind`
  - variants: `Null`, `NaN`, `NaT`
- `fp_types::Scalar`
  - variants: `Int64`, `Float64`, `Utf8`, `Bool`, `Null(NullKind)`
- `fp_columnar::ValidityMask`
  - fields: `bits: Vec<u64>`, `len: usize`
- `fp_columnar::Column`
  - fields: `dtype: DType`, `values: Vec<Scalar>`, `validity: ValidityMask`

## Rule Ledger

1. Missing detection:
   - `Scalar::is_missing()` returns true for all `Null` variants (Null, NaN, NaT).
2. Reduction semantics:
   - `nansum`: skip missing, return 0.0 for all-missing input,
   - `nanmean`: skip missing, return NaN for all-missing input,
   - `count`: count of non-missing values only.
3. Fill semantics:
   - `fillna`: cast fill value to column dtype, replace missing only.
4. Drop semantics:
   - `dropna`: remove all rows where value is missing.
5. Validity tracking:
   - `ValidityMask` uses packed bitvec (1 bit per element, Vec<u64>),
   - validity propagation in arithmetic: null op anything = null.

## Error Ledger

- `ColumnError::TypeMismatch` for fillna with incompatible scalar type.
- `ColumnError::LengthMismatch` for validity mask vs values length disagreement.
- `TypeError` for numeric coercion failures in reductions.

## Hidden Assumptions

1. NaN and Null are both treated as missing in reductions (no NaN-vs-Null distinction in aggregation).
2. dtype-specific missing marker normalization is centralized in scalar/column contracts.
3. `skipna` parameter is always true (no `skipna=False` path implemented).

## Undefined-Behavior Edges

1. Full nanops option matrix (`skipna=False`, `min_count` parameter).
2. `interpolate` and `bfill`/`ffill` methods.
3. NaN-vs-NaT distinction in datetime operations.
4. Full `isna`/`notna` API surface with dtype-aware dispatch.
