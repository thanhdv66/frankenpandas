# FP-P2D-032 Risk Note

Primary risk: axis=0 outer concat can drift on observable column ordering (`sort=False`) even when values are otherwise correct, causing downstream selector and serialization mismatches.

Mitigations in this packet:

1. Non-lexical overlap/disjoint fixture matrix locks first-seen output column order.
2. Empty-schema, duplicate-index, and null-preservation fixtures validate order + value stability together.
3. Error fixtures enforce fail-closed selector validation for invalid axis/join values.

Invariant hooks:

- `FP-I1` (shape consistency): output row count equals `len(left) + len(right)`.
- `FP-I3` (null propagation): existing nulls are preserved and new nulls are introduced only for missing source columns.
- `FP-I4` (determinism): output index, column order, and values are stable across repeated runs.
- `FP-I7` (fail-closed semantics): invalid selector inputs return explicit errors.

Residual risk:

- MultiIndex concat semantics remain out of scope.
- >2-frame axis=0 outer concat packet matrices remain pending.
- Global DataFrame column-order propagation across every operation remains an incremental workstream.
