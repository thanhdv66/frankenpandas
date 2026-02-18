# FP-P2D-025 Risk Note

Primary risk: row/column selector parity drift in DataFrame `loc`/`iloc` can silently reorder data, drop duplicates, or emit unstable diagnostics.

Mitigations:
1. Packet matrix covers row+column subset semantics for both loc and iloc.
2. Error fixtures lock fail-closed diagnostics for missing labels/columns, duplicate column selectors, and out-of-bounds iloc.
3. Differential harness enforces strict gate failure on selector drift.

## Invariant Ledger Hooks

- `FP-I1` (shape consistency): selector outputs have deterministic row/column cardinality.
- `FP-I4` (determinism): repeated selectors yield stable output ordering.
- `FP-I7` (fail-closed semantics): malformed selectors reject explicitly.
