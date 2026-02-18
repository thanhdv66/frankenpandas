# FP-P2D-026 Risk Note

Primary risk: prefix/suffix row-selection drift can silently truncate the wrong rows or reorder index labels, causing downstream aggregation and join parity failures.

Mitigations:
1. Packet matrix locks prefix, suffix, zero-row, and `n > len` saturation behavior.
2. Null-preservation fixtures guard scalar/null propagation.
3. Differential harness enforces strict gate failure on row or value drift.

## Invariant Ledger Hooks

- `FP-I1` (shape consistency): output row count is deterministic for each `n`.
- `FP-I4` (determinism): head/tail calls preserve stable index ordering.
- `FP-I7` (fail-closed semantics): malformed fixture payloads reject explicitly.
