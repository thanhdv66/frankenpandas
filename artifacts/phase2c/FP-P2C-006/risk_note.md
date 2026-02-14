# FP-P2C-006 Risk Note

Primary risk: join cardinality explosion with duplicate keys can produce unexpected output sizes; concat ordering semantics may drift from pandas-observable behavior under index alignment.

Mitigations:
1. Strict packet gate enforces zero failed fixtures for scoped join modes.
2. Hardened divergence budget is explicit and allowlisted.
3. Cross-product expansion is bounded by explicit cardinality checks.
4. Concat preserves input ordering deterministically.

## Isomorphism Proof Hook

- join cardinality: cross-product expansion matches nested-loop semantics
- ordering preserved: left join maintains left-side label order
- null handling: unmatched keys produce missing values deterministically
- concat ordering: input order is preserved across concatenation

## Invariant Ledger Hooks

- `FP-I1` (join cardinality determinism):
  - evidence: `artifacts/phase2c/FP-P2C-006/contract_table.md`, join mode specifications
- `FP-I2` (missingness monotonicity):
  - evidence: null-side handling contract in legacy anchor map
- `FP-I4` (index/order determinism):
  - evidence: concat ordering contract, join left-order preservation
