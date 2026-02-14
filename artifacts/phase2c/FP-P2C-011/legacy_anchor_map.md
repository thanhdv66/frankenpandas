# FP-P2C-011 Legacy Anchor Map

Packet: `FP-P2C-011`
Subsystem: full GroupBy planner split/apply/combine + aggregate matrix

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/groupby/grouper.py` (`Grouper`, `Grouping`, `get_grouper`)
- `legacy_pandas_code/pandas/pandas/core/groupby/ops.py` (`WrappedCythonOp`, `BaseGrouper`, `BinGrouper`, `DataSplitter`)
- `legacy_pandas_code/pandas/pandas/core/groupby/groupby.py` (`GroupBy`, `SeriesGroupBy`, `DataFrameGroupBy`)
- `legacy_pandas_code/pandas/pandas/core/groupby/generic.py` (aggregation dispatch)

## Extracted Behavioral Contract

1. Planner decisions preserve grouping key determinism and output ordering contracts.
2. Aggregate matrix preserves dtype/null semantics and default-option behavior.
3. Split/apply/combine orchestration is reproducible under strict/hardened policy gates.
4. First-seen key encounter order defines output ordering when `sort=False`.
5. `dropna=True` excludes null keys from grouping.
6. Each aggregate function has defined semantics for empty groups and all-null groups.

## Rust Slice Implemented

- `crates/fp-groupby/src/lib.rs`: `groupby_sum`, `groupby_mean`, `groupby_count`, `groupby_min`, `groupby_max`, `groupby_first`, `groupby_last`, `groupby_std`, `groupby_var`, `groupby_median`
- `crates/fp-groupby/src/lib.rs`: `groupby_agg` with `AggFunc` enum for generic dispatch
- `crates/fp-groupby/src/lib.rs`: dense Int64 fast path + arena-backed aggregation

## Type Inventory

- `fp_groupby::AggFunc`
  - variants: `Sum`, `Mean`, `Count`, `Min`, `Max`, `First`, `Last`, `Std`, `Var`, `Median`
- `fp_groupby::GroupByOptions`
  - fields: `dropna: bool`
- `fp_groupby::GroupByExecutionOptions`
  - fields: `use_arena: bool`, `arena_budget_bytes: usize`
- `fp_groupby::GroupByError`
  - variants: `Frame`, `Index`, `Column`
- `fp_groupby::GroupKeyRef`
  - variants: `Bool`, `Int64`, `FloatBits`, `Utf8`, `Null`

## Rule Ledger

1. Group materialization:
   - first-seen key order defines output ordering,
   - duplicate keys accumulated within same group.
2. Alignment:
   - fast path: if keys/values indexes match and are duplicate-free, skip alignment,
   - otherwise align via `align_union`.
3. Aggregate semantics:
   - `Sum`: sum of non-null values (0.0 for all-null group),
   - `Mean`: mean of non-null values (Null for all-null group),
   - `Count`: count of non-null values,
   - `Min`/`Max`: min/max of non-null values (Null for all-null group),
   - `First`/`Last`: first/last non-null value (Null for all-null group),
   - `Std`: sample standard deviation (ddof=1, Null for single-value group),
   - `Var`: sample variance (ddof=1, Null for single-value group),
   - `Median`: median of non-null values (Null for all-null group).
4. Dense optimization:
   - Int64 keys with span <= 65_536 use dense bucket path,
   - arena-backed path uses bumpalo for intermediates.

## Error Ledger

- `GroupByError::Frame` propagation for series construction failures.
- `GroupByError::Index` propagation for invalid alignment plan.
- `GroupByError::Column` propagation for output column construction failures.

## Hidden Assumptions

1. Multi-aggregate planner behavior has high interaction complexity.
2. Single-key series groupby only; multi-key DataFrame groupby deferred.
3. No `transform`, `filter`, or `apply` paths implemented.
4. No `observed` parameter for categorical key handling.

## Undefined-Behavior Edges

1. Full aggregate planner matrix (custom aggregation functions, named aggregation).
2. Multi-key DataFrame groupby semantics.
3. Categorical key handling with `observed` parameter.
4. `transform`, `filter`, `apply` paths.
5. Rolling/expanding window aggregation.
6. `resample` time-based groupby semantics.
