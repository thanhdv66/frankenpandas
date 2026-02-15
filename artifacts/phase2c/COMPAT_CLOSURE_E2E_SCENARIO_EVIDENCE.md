# COMPAT_CLOSURE E2E Scenario Matrix + Replay/Forensics Evidence

Bead: `bd-2gi.29.7` [COMPAT-CLOSURE-G]  
Subsystem: `fp-conformance` E2E scenario synthesis + replay bundle outputs  
Primary files:
- `crates/fp-conformance/src/lib.rs`
- `crates/fp-conformance/src/bin/fp-conformance-cli.rs`

---

## 1. Scope

This bead adds a deterministic COMPAT-CLOSURE E2E scenario layer on top of existing forensic events.

Delivered capabilities:

1. Scenario-step report builder from E2E forensic stream.
2. Step-level fields required for replay/triage:
   - `command_or_api`
   - `input_ref`
   - `output_ref`
   - `duration_ms`
   - `retry_count`
   - `outcome`
   - `reason_code`
3. Failure-injection linkage from `bd-2gi.29.6` fault-validation output.
4. CLI flag to emit scenario reports.

---

## 2. Data Model

### 2.1 Scenario Kind

- `golden_journey`
- `regression`
- `failure_injection`

### 2.2 Step Schema

`CompatClosureE2eScenarioStep` fields:

- `scenario_id`
- `packet_id`
- `mode`
- `trace_id`
- `step_id`
- `kind`
- `command_or_api`
- `input_ref`
- `output_ref`
- `duration_ms`
- `retry_count`
- `outcome`
- `reason_code`
- `replay_cmd`

### 2.3 Report Schema

`CompatClosureE2eScenarioReport` fields:

- `suite_id` (`COMPAT-CLOSURE-G`)
- `scenario_count`
- `pass_count`
- `fail_count`
- `steps`

---

## 3. Build and Write Pipeline

1. Run E2E: `run_e2e_suite(...)`
2. Build scenario report: `build_compat_closure_e2e_scenario_report(...)`
3. Optionally merge fault-injection report(s) from `run_fault_injection_validation_by_id(...)`
4. Write report: `write_compat_closure_e2e_scenario_report(...)`

Artifact path:

- single-packet scope: `artifacts/phase2c/<packet_id>/compat_closure_e2e_scenarios.json`
- multi-packet scope: `artifacts/phase2c/compat_closure_e2e_scenarios.json`

---

## 4. CLI Integration

`fp-conformance-cli` now supports:

- `--write-e2e-scenarios`

This runs E2E for the selected scope and emits COMPAT-CLOSURE scenario report artifacts.

---

## 5. Test Coverage

Added tests in `crates/fp-conformance/src/lib.rs`:

1. `compat_closure_e2e_scenario_report_contains_required_step_fields`
2. `compat_closure_e2e_scenario_report_includes_failure_injection_steps`
3. `compat_closure_e2e_scenario_report_writes_json`

These tests ensure schema completeness, deterministic report shape, and failure-injection linkage.

---

## 6. Replay Commands

Targeted tests:

```bash
cargo test -p fp-conformance --lib compat_closure_e2e_scenario_report_contains_required_step_fields -- --nocapture
cargo test -p fp-conformance --lib compat_closure_e2e_scenario_report_includes_failure_injection_steps -- --nocapture
cargo test -p fp-conformance --lib compat_closure_e2e_scenario_report_writes_json -- --nocapture
```

Artifact generation example:

```bash
cargo run -p fp-conformance --bin fp-conformance-cli -- --packet-id FP-P2C-001 --write-e2e-scenarios
```

Quality gates (offloaded via `rch`):

```bash
rch exec -- cargo check --workspace --all-targets
rch exec -- cargo clippy --workspace --all-targets -- -D warnings
rch exec -- cargo fmt --check
```
