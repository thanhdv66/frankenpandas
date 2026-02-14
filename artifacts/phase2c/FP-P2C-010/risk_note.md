# FP-P2C-010 Risk Note

Primary risk: `loc`/`iloc` branch matrix is extremely complex in pandas with many coercion paths; incomplete coverage may cause silent behavior divergence.

Mitigations:
1. Strict mode fails closed on unsupported indexer types.
2. Boolean mask handling is explicit: null = false, type must be Bool.
3. Positional slicing (head/tail) has simple, well-defined semantics.
4. Branch coverage tracked explicitly in rule ledger.

## Isomorphism Proof Hook

- boolean filter: null-as-false matches pandas observable behavior
- head/tail: positional slicing matches pandas exactly
- index preservation: filtered output preserves original label order for selected rows
- type safety: non-Bool mask rejected at operation boundary

## Invariant Ledger Hooks

- `FP-I4` (index/order determinism):
  - evidence: `artifacts/phase2c/FP-P2C-010/contract_table.md`, filter order preservation
- `FP-I2` (missingness monotonicity):
  - evidence: null-as-false boolean mask contract
- indexer type safety:
  - evidence: error ledger type mismatch enforcement
