# FP-P2D-014 Risk Note

Primary risk: merge ordering and null-materialization behavior can silently drift when join type or duplicate-key handling changes.

Mitigations:
1. Dedicated fixtures lock row-order contracts across `inner`/`left`/`right`/`outer` merge paths.
2. Suffix-collision fixtures assert deterministic `_left`/`_right` naming behavior.
3. Index-key merge fixtures assert deterministic synthetic-key behavior with both duplicate and unmatched keys.
4. Concat fixtures validate duplicate-index preservation and explicit mismatch failure semantics.

## Invariant Ledger Hooks

- `FP-I2` (missingness monotonicity): unmatched merge rows emit deterministic nulls in non-key columns.
- `FP-I4` (index/order determinism): merge/concat outputs preserve documented encounter-order contracts.
- merge/concat parity lock: packet fixtures and differential report act as regression sentinels.
