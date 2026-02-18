# FP-P2D-032 Legacy Anchor Map

Packet: `FP-P2D-032`
Scope: DataFrame concat axis=0 with `join='outer'` and observable first-seen column order.

Primary pandas anchors:

1. `pandas/core/reshape/concat.py`
   - `concat(...)`
   - `_Concatenator.get_result(...)`
   - axis=0 `join='outer'` with `sort=False` preserving first-seen non-concat axis labels
2. `pandas/core/frame.py`
   - DataFrame column label ordering and block construction invariants
3. `pandas/core/indexes/base.py`
   - index append/ordering behavior observed in row-wise concat

Behavior essence extracted:

- axis=0 concat preserves input frame row order; output index is left labels then right labels.
- `join='outer'` yields union of columns with `sort=False` first-seen ordering (left first, unseen right appended by encounter order).
- missing cells become null/NaN.
- invalid axis and invalid join selector values are rejected.
- existing null payloads are preserved through concat materialization.
