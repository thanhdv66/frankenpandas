# FP-P2D-029 Legacy Anchor Map

Packet: `FP-P2D-029`  
Subsystem: DataFrame concat axis=1 `join='inner'` parity

## Pandas Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/reshape/concat.py` (`concat`, `_Concatenator`, axis + join selector behavior)
- `legacy_pandas_code/pandas/pandas/core/frame.py` (DataFrame alignment and row-label intersection materialization)

## FrankenPandas Anchors

- `crates/fp-frame/src/lib.rs`: `ConcatJoin`, `concat_dataframes_with_axis_join`, axis=1 intersection logic
- `crates/fp-conformance/src/lib.rs`: `concat_join` fixture/oracle plumbing and packet execution
- `crates/fp-conformance/oracle/pandas_oracle.py`: `concat_join` selector validation + pandas bridge
- `crates/fp-conformance/fixtures/packets/fp_p2d_029_*`: axis=1 inner join fixture matrix

## Behavioral Commitments

1. `concat(axis=1, join='inner')` uses intersection of row labels while preserving left-order.
2. Existing null values are preserved through intersection materialization.
3. Unsupported selector combinations remain fail-closed with explicit diagnostics.

## Open Gaps

1. Axis=0 `join='inner'` column-intersection semantics.
2. MultiIndex concat join semantics.
3. Duplicate-column-preserving output model parity.

