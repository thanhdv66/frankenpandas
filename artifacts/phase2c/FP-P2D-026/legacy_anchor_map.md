# FP-P2D-026 Legacy Anchor Map

Packet: `FP-P2D-026`
Subsystem: DataFrame `head`/`tail` parity

## Legacy Anchors

- `legacy_pandas_code/pandas/pandas/core/generic.py` (`NDFrame.head` / `NDFrame.tail` contracts)
- `legacy_pandas_code/pandas/pandas/core/frame.py` (DataFrame row-slice materialization)
- `legacy_pandas_code/pandas/pandas/core/indexes/base.py` (index preservation through row truncation)

## Extracted Behavioral Contract

1. `head(n)` returns the first `min(n, len(df))` rows.
2. `tail(n)` returns the last `min(n, len(df))` rows.
3. `n=0` returns empty-index frames with unchanged column schema.
4. Scalar/null payloads are copied without coercive mutation.

## Rust Slice Implemented

- `crates/fp-frame/src/lib.rs`: `DataFrame::head` / `DataFrame::tail`
- `crates/fp-conformance/src/lib.rs`: fixture + differential operation handling for `dataframe_head` / `dataframe_tail`
- `crates/fp-conformance/oracle/pandas_oracle.py`: live-oracle ops for DataFrame head/tail
- `crates/fp-conformance/fixtures/packets/fp_p2d_026_*`: packet fixture matrix

## Type Inventory

- `fp_frame::DataFrame` selector APIs: `head`, `tail`
- `fp_conformance::PacketFixture` fields: `head_n`, `tail_n`
- `fp_index::IndexLabel` index model for row-selection output

## Rule Ledger

1. Prefix/suffix selection preserves original index ordering.
2. Column schema remains unchanged regardless of `n`.
3. `n` greater than frame length saturates to full frame.
4. Empty-output frames preserve deterministic empty-column vectors.

## Hidden Assumptions

1. Current fixture matrix uses non-negative integer `n` values.
2. Column-order semantics follow the current DataFrame storage model.

## Undefined-Behavior Edges

1. Negative `n` semantics (`pandas` allows `head(-k)` / `tail(-k)` contracts).
2. Axis=1 head/tail behavior and MultiIndex axis interactions.
