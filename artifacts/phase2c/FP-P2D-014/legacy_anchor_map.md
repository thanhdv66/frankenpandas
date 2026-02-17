# FP-P2D-014 Legacy Anchor Map

Packet: `FP-P2D-014`
Subsystem: DataFrame merge/join/concat parity matrix

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/reshape/merge.py` (`merge`, `MergeOperation`)
- `legacy_pandas_code/pandas/pandas/core/frame.py` (`DataFrame.merge`, `DataFrame.join`)
- `legacy_pandas_code/pandas/pandas/core/reshape/concat.py` (`concat`)

## Extracted Behavioral Contract

1. Merge join type controls row inclusion (`inner`, `left`, `right`, `outer`).
2. Duplicate key pairs expand via cartesian product.
3. Name collisions for non-key columns are suffix-disambiguated.
4. `axis=0` concat appends rows while preserving index labels and duplicates.
5. Index-key merge path is modeled via deterministic synthetic key materialization.

## Rust Slice Implemented

- `crates/fp-join/src/lib.rs`: `merge_dataframes`
- `crates/fp-frame/src/lib.rs`: `concat_dataframes`
- `crates/fp-conformance/src/lib.rs`: packet fixture execution + diff comparators for DataFrame merge/index-merge/concat
- `crates/fp-conformance/oracle/pandas_oracle.py`: pandas-backed merge/concat oracle adapters

## Type Inventory

- `fp_conformance::FixtureOperation`
  - variants: `DataFrameMerge`, `DataFrameMergeIndex`, `DataFrameConcat`
- `fp_conformance::FixtureJoinType`
  - variants: `Inner`, `Left`, `Right`, `Outer`
- `fp_join::MergedDataFrame`
  - fields: `index`, `columns`

## Rule Ledger

1. Merge ordering follows join strategy and input encounter order.
2. Concat requires matching column sets; mismatches are hard errors.
3. Index-key merge synthesizes a deterministic key column before merge.
4. Differential comparisons enforce exact index and value equality with semantic null matching.

## Hidden Assumptions

1. Index-key merge currently maps to synthetic-column semantics instead of native index-index output index preservation.
2. Merge key coercion is currently bounded to scalar types expressible via fixture schema.

## Undefined-Behavior Edges

1. Multi-key DataFrame merges (`on=[...]`) are not yet covered.
2. Custom merge suffix arguments are not yet exposed in fixtures.
3. Concat `axis=1` alignment behavior is not yet part of this packet.
