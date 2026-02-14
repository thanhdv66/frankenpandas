# FP-P2C-007 Risk Note

Primary risk: missing value semantics differ subtly between NaN, NaT, and None in pandas; our unified `is_missing()` treatment may diverge from pandas in dtype-aware contexts.

Mitigations:
1. Strict mode fails closed on unrecognized coercion or reduction ambiguity.
2. ValidityMask packed bitvec tracks per-element validity efficiently.
3. Reduction functions explicitly document all-missing return values (0.0 for sum, NaN for mean).
4. Fill value cast to target dtype prevents silent type coercion.

## Isomorphism Proof Hook

- missing propagation: null input always produces null output in arithmetic
- reduction semantics: nansum/nanmean match pandas skipna=True defaults
- fillna contract: only missing values replaced, non-missing preserved
- dropna contract: removes exactly those rows where is_missing() is true

## Invariant Ledger Hooks

- `FP-I2` (missingness monotonicity):
  - evidence: `artifacts/phase2c/FP-P2C-007/contract_table.md`, null propagation rules
- `FP-I3` (dtype/coercion determinism):
  - evidence: fillna cast semantics in legacy anchor map
- reduction contract lock:
  - evidence: nansum/nanmean behavior specifications
