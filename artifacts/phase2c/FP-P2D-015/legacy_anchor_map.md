# FP-P2D-015 Legacy Anchor Map

Packet: `FP-P2D-015`
Subsystem: null-skipping scalar nanops parity matrix

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/nanops.py` (`nansum`, `nanmean`, `nanmin`, `nanmax`, `nanvar`, `nanstd`)
- `legacy_pandas_code/pandas/pandas/core/series.py` (`Series.count`, scalar reductions)

## Extracted Behavioral Contract

1. Missing values are skipped for scalar reductions.
2. `sum` over empty/all-missing inputs yields zero.
3. `mean`, `min`, `max`, `std`, `var` over empty/all-missing (or insufficient ddof) yield NaN-style missing outputs.
4. `count` reports number of non-missing elements.
5. `std`/`var` use sample-statistics semantics (`ddof=1`).

## Rust Slice Implemented

- `crates/fp-types/src/lib.rs`: `nansum`, `nanmean`, `nanmin`, `nanmax`, `nanstd`, `nanvar`, `nancount`
- `crates/fp-conformance/src/lib.rs`: nanops fixture execution and differential comparator wiring
- `crates/fp-conformance/oracle/pandas_oracle.py`: pandas nanops scalar oracle handlers

## Type Inventory

- `fp_conformance::FixtureOperation`
  - variants: `NanSum`, `NanMean`, `NanMin`, `NanMax`, `NanStd`, `NanVar`, `NanCount`
- `fp_types::Scalar`
  - variants involved: `Bool`, `Int64`, `Float64`, `Null(Null|NaN)`

## Rule Ledger

1. Non-missing numeric-compatible values participate in reductions.
2. Boolean values are coerced numerically (`false=0`, `true=1`).
3. Missing markers are excluded from aggregates.
4. `std`/`var` use `ddof=1`; insufficient populations return missing.

## Hidden Assumptions

1. Non-numeric string coercions are out-of-scope for this packet.
2. Packet targets scalar series reductions, not DataFrame axis reductions.

## Undefined-Behavior Edges

1. Locale/encoding-dependent numeric parsing from string-like values.
2. Rolling/window nanops and grouped nanops interactions.
