# COMPAT_CLOSURE Differential + Fault-Injection Validation Evidence

Bead: `bd-2gi.29.6` [COMPAT-CLOSURE-F]  
Subsystem: `fp-conformance` differential harness + deterministic fault injection  
Primary files:
- `crates/fp-conformance/src/lib.rs`
- `crates/fp-conformance/src/bin/fp-conformance-cli.rs`

---

## 1. Scope

This bead extends COMPAT-CLOSURE validation with:

1. Deterministic differential validation log entries with required replay identifiers.
2. Deterministic fault-injection workflow that classifies strict violations vs hardened allowlisted outcomes.
3. Artifact writers for differential JSONL and fault-injection JSON reports.
4. CLI wiring for explicit artifact production in local/CI runs.

---

## 2. Differential Validation Log Contract

`DifferentialValidationLogEntry` fields (required):

- `packet_id`
- `case_id`
- `mode`
- `trace_id`
- `oracle_source`
- `mismatch_class`
- `replay_key`

Generation path:

1. `run_differential_by_id(...)` produces `DifferentialReport`.
2. `build_differential_validation_log(...)` emits deterministic, sorted entries.
3. `write_differential_validation_log(...)` writes JSONL artifact (`differential_validation_log.jsonl`).

---

## 3. Fault-Injection Validation Contract

`run_fault_injection_validation_by_id(...)` runs deterministic mode-split validation over packet fixtures and emits `FaultInjectionValidationReport` with entries classified as:

- `strict_violation`
- `hardened_allowlisted`

Each entry includes deterministic replay anchors:

- `trace_id`
- `replay_key`
- `mismatch_class`

If a fixture has no natural drift, a deterministic synthetic drift record is injected to ensure classification coverage remains explicit and reproducible.

---

## 4. Artifact Outputs

Per packet (`artifacts/phase2c/<packet_id>/`):

1. `differential_validation_log.jsonl`
2. `fault_injection_validation.json`

Both artifacts are deterministic for fixed fixture set and runtime mode configuration.

---

## 5. CLI Integration

`fp-conformance-cli` now supports:

1. `--write-differential-validation`
2. `--write-fault-injection`

These flags generate closure artifacts per packet after packet suite execution.

---

## 6. Test Coverage

Added tests in `crates/fp-conformance/src/lib.rs`:

1. `differential_validation_log_contains_required_fields`
2. `differential_validation_log_writes_jsonl`
3. `fault_injection_validation_classifies_strict_vs_hardened`
4. `fault_injection_validation_report_writes_json`

These tests verify schema completeness, deterministic artifact shape, and explicit strict/hardened taxonomy split.

---

## 7. Replay Commands

Targeted:

```bash
cargo test -p fp-conformance --lib differential_validation_log_contains_required_fields -- --nocapture
cargo test -p fp-conformance --lib differential_validation_log_writes_jsonl -- --nocapture
cargo test -p fp-conformance --lib fault_injection_validation_classifies_strict_vs_hardened -- --nocapture
cargo test -p fp-conformance --lib fault_injection_validation_report_writes_json -- --nocapture
```

Artifact generation:

```bash
cargo run -p fp-conformance --bin fp-conformance-cli -- --packet-id FP-P2C-001 --write-differential-validation --write-fault-injection
```

Quality gates (offloaded with `rch`):

```bash
rch exec -- cargo check --workspace --all-targets
rch exec -- cargo clippy --workspace --all-targets -- -D warnings
rch exec -- cargo fmt --check
```
