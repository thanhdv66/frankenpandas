# COMPAT_CLOSURE Unit/Property Test + Structured Logging Evidence

Bead: `bd-2gi.29.5` [COMPAT-CLOSURE-E]  
Subsystem: `fp-conformance` compat-closure validation + forensic logging  
Primary file: `crates/fp-conformance/src/lib.rs`

---

## 1. Scope

This evidence note records the `bd-2gi.29.5` implementation surface:

1. Compat-closure deterministic structured log schema and emission.
2. Unit + property-style tests for schema completeness, determinism, and mode-split boundaries.
3. Compatibility-matrix coverage floor enforcement/reporting for `CC-001..CC-009`.
4. Reproducible replay commands for local and CI workflows.

---

## 2. Structured Log Schema

`CompatClosureCaseLog` is emitted through forensic event `compat_closure_case` and includes all required fields:

```json
{
  "ts_utc": 1700000000000,
  "suite_id": "COMPAT-CLOSURE-E",
  "test_id": "series_add_strict",
  "api_surface_id": "CC-004",
  "packet_id": "FP-P2C-001",
  "mode": "strict",
  "seed": 12345,
  "input_digest": "<sha256>",
  "output_digest": "<sha256>",
  "env_fingerprint": "<sha256>",
  "artifact_refs": [
    "artifacts/phase2c/FP-P2C-001/parity_report.json",
    "artifacts/phase2c/FP-P2C-001/parity_report.raptorq.json",
    "artifacts/phase2c/FP-P2C-001/parity_report.decode_proof.json",
    "artifacts/phase2c/FP-P2C-001/parity_gate_result.json",
    "artifacts/phase2c/FP-P2C-001/parity_mismatch_corpus.json"
  ],
  "duration_ms": 5,
  "outcome": "pass",
  "reason_code": "ok"
}
```

Determinism contract:

1. `seed`, `input_digest`, and `output_digest` are pure derivations from packet/case/mode/result inputs.
2. `env_fingerprint` is stable for a fixed repo/toolchain environment.
3. Same inputs plus same `ts_utc` produce byte-identical serialized payloads.

---

## 3. Test Inventory

Added tests in `crates/fp-conformance/src/lib.rs`:

1. `compat_closure_case_log_contains_required_fields`  
Validates required schema keys and pass-path reason code.

2. `compat_closure_case_log_is_deterministic_for_same_inputs`  
Validates byte-identical serialized output for identical inputs.

3. `compat_closure_mode_split_contracts_hold_across_seed_span`  
Property-style (seed sweep) validation for strict fail-closed (`CC-008`) and hardened bounded repair (`CC-009`).

4. `compat_closure_coverage_report_is_complete`  
Asserts matrix coverage report has zero uncovered rows and meets `100%` floor.

5. `e2e_emits_compat_closure_case_events`  
Verifies end-to-end forensic stream includes `compat_closure_case` events with required structured fields.

---

## 4. Coverage Floor + Matrix Completeness

Coverage report is produced by `build_compat_closure_coverage_report()` and enforces:

1. Required rows: `CC-001..CC-009`.
2. Achieved coverage: computed from fixture operation mapping + runtime mode-split checks.
3. Floor: `100%` (no uncovered scoped rows).

If any required row is uncovered, report completeness fails and tests fail closed.

---

## 5. Replay Commands

Targeted tests:

```bash
cargo test -p fp-conformance --lib compat_closure_case_log_contains_required_fields -- --nocapture
cargo test -p fp-conformance --lib compat_closure_case_log_is_deterministic_for_same_inputs -- --nocapture
cargo test -p fp-conformance --lib compat_closure_mode_split_contracts_hold_across_seed_span -- --nocapture
cargo test -p fp-conformance --lib compat_closure_coverage_report_is_complete -- --nocapture
cargo test -p fp-conformance --lib e2e_emits_compat_closure_case_events -- --nocapture
```

Required quality gates (offloaded via `rch`):

```bash
rch exec -- cargo check --workspace --all-targets
rch exec -- cargo clippy --workspace --all-targets -- -D warnings
rch exec -- cargo fmt --check
```

---

## 6. Invariant Checklist

| Invariant | Status | Evidence Surface |
|---|---|---|
| Structured log schema includes required compat-closure fields | enforced | `compat_closure_case_log_contains_required_fields` |
| Structured log payload is deterministic for fixed inputs | enforced | `compat_closure_case_log_is_deterministic_for_same_inputs` |
| Strict mode remains fail-closed for unknown features | enforced | `compat_closure_mode_split_contracts_hold_across_seed_span` |
| Hardened mode over-cap admission remains bounded-repair | enforced | `compat_closure_mode_split_contracts_hold_across_seed_span` |
| Compatibility matrix rows are fully covered for scoped closure set | enforced | `compat_closure_coverage_report_is_complete` |
| E2E forensic output carries compat-closure structured events | enforced | `e2e_emits_compat_closure_case_events` |
