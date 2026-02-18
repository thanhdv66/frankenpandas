# FP-P2D-031 Legacy Anchor Map

Packet: `FP-P2D-031`
Scope: DataFrame concat axis=0 with `join='outer'` union-column semantics.

Primary pandas anchors:

1. `pandas/core/reshape/concat.py`
   - `concat(...)`
   - `_Concatenator.get_result(...)`
   - axis=0 join normalization and block assembly paths
2. `pandas/core/frame.py`
   - DataFrame construction and column materialization invariants used by concat output
3. `pandas/core/indexes/base.py`
   - index append/ordering behavior observed in row-wise concat

Behavior essence extracted:

- axis=0 concat preserves input frame row order; output index is left labels then right labels.
- `join='outer'` yields union of columns; missing cells become null/NaN.
- invalid axis and invalid join selector values are rejected.
- existing null payloads are preserved through concat materialization.
