# FP-P2C-008 Risk Note

Primary risk: CSV dtype inference may diverge from pandas for ambiguous input (mixed integer/float columns, null representations). JSON orient compatibility surface is large.

Mitigations:
1. Strict mode rejects unsupported features/metadata rather than guessing.
2. Round-trip tests validate read-write stability for supported formats.
3. Dtype inference follows explicit priority: try Int64 -> Float64 -> Utf8.
4. JSON orient parameter explicitly selects parse strategy.

## Isomorphism Proof Hook

- CSV round-trip: write then read produces identical DataFrame for supported dtypes
- JSON round-trip: write then read preserves values for supported orient modes
- dtype inference: deterministic and consistent across identical inputs
- missing value handling: empty CSV cells and JSON null both produce Scalar::Null

## Invariant Ledger Hooks

- `FP-I3` (dtype/coercion determinism):
  - evidence: `artifacts/phase2c/FP-P2C-008/contract_table.md`, dtype inference rules
- `FP-I5` (round-trip stability):
  - evidence: CSV and JSON round-trip contracts in legacy anchor map
- IO error determinism:
  - evidence: error ledger in legacy anchor map, fail-closed malformed input policy
