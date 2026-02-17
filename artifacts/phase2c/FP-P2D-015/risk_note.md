# FP-P2D-015 Risk Note

Primary risk: subtle mismatches in missingness semantics (`null` vs `na_n`) and `ddof=1` behavior for `std`/`var` can cause false parity confidence.

Mitigations:
1. Strict+hardened fixtures explicitly cover mixed values, all-missing inputs, and insufficient-population std/var cases.
2. Oracle adapter uses pandas scalar reductions with explicit `skipna=True` and `ddof=1` for std/var.
3. Conformance comparator uses semantic scalar equality that treats NaN/null-kind parity correctly.

## Invariant Ledger Hooks

- `FP-I2` (missingness monotonicity): missing inputs never increase count; missing-only sets produce documented neutral/NaN outputs.
- `FP-I4` (determinism): scalar nanops outputs are deterministic for fixed inputs.
- nanops parity lock: packet fixtures and differential logs serve as regression sentinels.
