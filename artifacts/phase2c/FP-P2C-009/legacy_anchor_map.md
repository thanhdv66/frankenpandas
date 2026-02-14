# FP-P2C-009 Legacy Anchor Map

Packet: `FP-P2C-009`
Subsystem: BlockManager + storage invariants

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/internals/managers.py` (`BaseBlockManager`, `BlockManager`, `SingleBlockManager`, `create_block_manager_from_blocks`, `_consolidate`)
- `legacy_pandas_code/pandas/pandas/core/internals/blocks.py` (`Block`, `DatetimeTZBlock`, `ExtensionBlock`)
- `legacy_pandas_code/pandas/pandas/core/internals/construction.py` (`arrays_to_mgr`, `dict_to_mgr`)

## Extracted Behavioral Contract

1. Block placement and axis mappings are internally consistent (`blknos`/`blklocs`-class invariants).
2. Consolidation rules preserve observable dtype/null semantics.
3. Storage transforms do not silently corrupt downstream frame/index contracts.
4. Column-major storage maps dtype-homogeneous columns into contiguous blocks.

## Rust Slice Implemented

- `crates/fp-columnar/src/lib.rs`: `Column` with per-column `ValidityMask`, `DType`, `Vec<Scalar>`
- `crates/fp-frame/src/lib.rs`: `DataFrame` with `BTreeMap<String, Column>` storage

## Type Inventory

- `fp_columnar::Column`
  - fields: `dtype: DType`, `values: Vec<Scalar>`, `validity: ValidityMask`
- `fp_columnar::ValidityMask`
  - fields: `bits: Vec<u64>`, `len: usize`
- `fp_types::DType`
  - variants: `Int64`, `Float64`, `Utf8`, `Bool`
- `fp_frame::DataFrame`
  - fields: `index: Index`, `columns: BTreeMap<String, Column>`

## Rule Ledger

1. Storage model:
   - FrankenPandas uses per-column storage (no BlockManager consolidation),
   - each Column owns its values and validity mask independently.
2. Dtype contract:
   - Column dtype is inferred at construction and locked,
   - arithmetic between columns may promote dtype (e.g., Int64 + Float64 -> Float64).
3. Validity invariant:
   - ValidityMask length must equal values length,
   - validity is propagated through all operations (null in -> null out).
4. Column independence:
   - columns in a DataFrame are independent; no cross-column block consolidation.

## Error Ledger

- `ColumnError::LengthMismatch` for validity mask vs values length disagreement.
- `ColumnError::TypeMismatch` for incompatible dtype operations.
- `FrameError::LengthMismatch` for column vs index length disagreement.

## Hidden Assumptions

1. No BlockManager equivalent: FrankenPandas uses per-column storage, not blocked storage.
2. No consolidation pass: each column is independently typed and stored.
3. Storage invariants are simpler than pandas due to per-column model.
4. Low-level storage invariants may require dedicated witness ledgers beyond API-level fixtures.

## Undefined-Behavior Edges

1. Full BlockManager operation matrix and internals migration boundaries.
2. In-place mutation semantics (pandas copy-on-write vs FrankenPandas immutable columns).
3. Memory layout optimization (Arrow-compatible layouts, SIMD alignment).
4. Block consolidation and deconsolidation paths.
