# FP-P2D-031 Risk Note

Primary risk: axis=0 outer-join concat can drift on union-column materialization and null-fill placement, causing downstream schema and value-shape regressions.

Mitigations in this packet:

1. Overlap/disjoint fixture matrix locks union-column + null-fill behavior.
2. Empty-schema and null-preservation cases validate sparse-input stability.
3. Error fixtures enforce fail-closed selector validation for invalid axis/join values.

Invariant hooks:

- `FP-I1` (shape consistency): output row count equals `len(left) + len(right)`.
- `FP-I3` (null propagation): existing nulls are preserved and new nulls are only introduced for missing source columns.
- `FP-I4` (determinism): output index and column values are stable across repeated runs.
- `FP-I7` (fail-closed semantics): invalid selector inputs return explicit errors.

Residual risk:

- MultiIndex concat semantics remain out of scope.
- >2-frame axis=0 outer concat packet matrices remain pending.
- axis=1 duplicate-index concat remains fail-closed in current runtime.
