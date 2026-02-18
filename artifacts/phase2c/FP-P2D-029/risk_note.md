# FP-P2D-029 Risk Note

Primary risk: inner-join selector drift in axis=1 concat can silently reorder or drop rows inconsistently across columns, producing subtle downstream parity regressions.

Mitigations in this packet:

1. Fixtures lock deterministic left-order intersection behavior for numeric and utf8 labels.
2. Null-preservation case ensures intersection filtering does not coerce/drop existing nulls.
3. Error matrix enforces fail-closed behavior for unsupported selector values and ambiguous input shapes.

Invariant hooks:

- `FP-I1` (shape consistency): output row count equals intersection label cardinality.
- `FP-I4` (determinism): repeated axis=1 inner concat yields stable label/value order.
- `FP-I7` (fail-closed semantics): unsupported selector combinations return explicit errors.

Residual risk:

- Axis=0 `join='inner'` remains intentionally unsupported in this slice.
- Duplicate-column parity remains blocked by current unique-keyed column storage model.

