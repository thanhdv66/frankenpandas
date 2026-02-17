# FP-P2D-016 Risk Note

Primary risk: CSV parser/formatter edge cases can silently regress by preserving shape while mutating cell semantics, especially around quoted text, unicode payloads, and malformed-row handling.

Mitigations:
1. Packet matrix explicitly covers quoting, escaped quotes, embedded newlines, CRLF/no-trailing-newline, UTF-8, mixed dtypes, and malformed parser-error cases.
2. `csv_round_trip` conformance execution now enforces semantic DataFrame equality (not only column/row shape checks).
3. Live oracle branch uses pandas `read_csv` + `to_csv` + reparse to keep expected success/error behavior anchored to legacy semantics.

## Invariant Ledger Hooks

- `FP-I2` (missingness monotonicity): empty CSV fields remain missing across round-trip.
- `FP-I4` (determinism): identical CSV input yields deterministic round-trip pass/fail outcome.
- `FP-I7` (fail-closed parser semantics): malformed rows must fail rather than auto-repair silently.
