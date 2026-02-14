# FP-P2C-009 Risk Note

Primary risk: FrankenPandas per-column storage model diverges architecturally from pandas BlockManager; observable behavior must match despite different internal representation.

Mitigations:
1. API-level behavior tests validate observable semantics regardless of storage model.
2. Per-column model simplifies invariant checking (no cross-column consolidation bugs).
3. ValidityMask packed bitvec provides efficient null tracking with clear invariants.
4. Dtype is locked at construction time, preventing silent mutation.

## Isomorphism Proof Hook

- storage-observable equivalence: per-column storage produces same API behavior as BlockManager
- validity invariant: ValidityMask.len == Column.values.len always holds
- dtype lock: column dtype is immutable after construction
- null propagation: validity mask correctly propagated through arithmetic and reindex

## Invariant Ledger Hooks

- `FP-I6` (storage invariant consistency):
  - evidence: `artifacts/phase2c/FP-P2C-009/contract_table.md`, validity mask contracts
- `FP-I3` (dtype/coercion determinism):
  - evidence: dtype lock and promotion rules in legacy anchor map
- `FP-I2` (missingness monotonicity):
  - evidence: validity propagation contract
