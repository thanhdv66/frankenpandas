# DOC-PASS-09: Unit/E2E Test Corpus and Logging Evidence Crosswalk

**Bead:** `bd-2gi.23.10`  
**Date:** 2026-02-15  
**Status:** Complete  
**Scope:** map major documented behaviors to unit/property/differential/e2e evidence and document replay/forensics logging fields plus coverage gaps.

---

## 1. Method and Scope Boundaries

This crosswalk uses source-anchored evidence from:

1. crate-local unit tests (`#[cfg(test)]`) across all workspace crates,
2. centralized property tests in `fp-conformance`,
3. differential + conformance harness code paths,
4. E2E orchestration + failure/CI forensics surfaces,
5. Phase2C testing/logging doctrine artifacts.

Primary anchors:

- test placement doctrine: `artifacts/phase2c/TEST_CONVENTIONS.md:12`, `artifacts/phase2c/TEST_CONVENTIONS.md:44`
- reliability gates: `artifacts/phase2c/COVERAGE_FLAKE_BUDGETS.md:124`
- replay/forensics UX: `artifacts/phase2c/FAILURE_FORENSICS_UX.md:19`
- harness/differential/e2e implementation: `crates/fp-conformance/src/lib.rs:87`, `crates/fp-conformance/src/lib.rs:546`, `crates/fp-conformance/src/lib.rs:620`, `crates/fp-conformance/src/lib.rs:3500`, `crates/fp-conformance/src/lib.rs:3782`

---

## 2. Evidence Inventory Snapshot

### 2.1 Unit Test Surfaces

Unit-test modules exist in every crate:

- `crates/fp-types/src/lib.rs:323`
- `crates/fp-columnar/src/lib.rs:1033`
- `crates/fp-index/src/lib.rs:736`
- `crates/fp-frame/src/lib.rs:909`
- `crates/fp-expr/src/lib.rs:200`
- `crates/fp-groupby/src/lib.rs:1130`
- `crates/fp-join/src/lib.rs:507`
- `crates/fp-io/src/lib.rs:488`
- `crates/fp-runtime/src/lib.rs:568`
- `crates/fp-conformance/src/lib.rs:3847`

### 2.2 Property Test Surface

Centralized property suite: `crates/fp-conformance/tests/proptest_properties.rs:1`.

Coverage families include:

- alignment invariants: `crates/fp-conformance/tests/proptest_properties.rs:109`
- duplicate detection determinism: `crates/fp-conformance/tests/proptest_properties.rs:183`
- series add invariants: `crates/fp-conformance/tests/proptest_properties.rs:210`
- join invariants: `crates/fp-conformance/tests/proptest_properties.rs:298`
- groupby invariants: `crates/fp-conformance/tests/proptest_properties.rs:409`
- scalar/null semantics + serialization round trips: `crates/fp-conformance/tests/proptest_properties.rs:476`, `crates/fp-conformance/tests/proptest_properties.rs:532`
- arena/global isomorphism + fallback invariants: `crates/fp-conformance/tests/proptest_properties.rs:619`

### 2.3 Differential Fixture Operation Coverage

The differential engine implements all listed fixture operations in both report and differential paths:

- operation enum: `crates/fp-conformance/src/lib.rs:123`
- execution path switch: `crates/fp-conformance/src/lib.rs:1960`
- differential path switch: `crates/fp-conformance/src/lib.rs:2741`

Current fixture corpus operation counts (derived from `crates/fp-conformance/fixtures/*.json`):

| Operation | Fixture count |
|---|---|
| `series_add` | 5 |
| `series_join` | 3 |
| `groupby_sum` | 3 |
| `series_concat` | 2 |
| `index_has_duplicates` | 2 |
| `index_align_union` | 2 |
| `csv_round_trip` | 2 |
| `column_dtype_check` | 2 |
| `series_head` | 1 |
| `series_filter` | 1 |
| `nan_sum` | 1 |
| `index_first_positions` | 1 |
| `groupby_mean` | 1 |
| `groupby_count` | 1 |
| `fill_na` | 1 |
| `drop_na` | 1 |

### 2.4 E2E and Replay Entry Points

- smoke and basic packet execution tests: `crates/fp-conformance/tests/smoke.rs:5`
- end-to-end scenario integration tests: `crates/fp-conformance/tests/ag_e2e.rs:21`
- E2E orchestrator: `crates/fp-conformance/src/lib.rs:3500`
- failure forensics builder: `crates/fp-conformance/src/lib.rs:3782`
- CLI replay/gate surface: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:8`
- script replay entry points: `scripts/phase2c_gate_check.sh:11`, `scripts/governance_gate_check.sh:11`

---

## 3. Major Behavior -> Evidence Crosswalk

| Behavior contract | Unit evidence | Property evidence | Differential evidence | E2E / scenario evidence | Logging + replay evidence | Status |
|---|---|---|---|---|---|---|
| Alignment union plan validity + position shape | `crates/fp-index/src/lib.rs:736` | `crates/fp-conformance/tests/proptest_properties.rs:109` | `crates/fp-conformance/src/lib.rs:2786` | `crates/fp-conformance/tests/ag_e2e.rs:96` | drift categories include `shape`/`index`: `crates/fp-conformance/src/lib.rs:277` | covered |
| Duplicate index detection + strict rejection path | `crates/fp-frame/src/lib.rs:909`, `crates/fp-runtime/src/lib.rs:796` | `crates/fp-conformance/tests/proptest_properties.rs:183` | `crates/fp-conformance/src/lib.rs:2800` | workflow contract: `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:159` | policy evidence records: `crates/fp-runtime/src/lib.rs:82` | covered |
| Series add alignment + null propagation | `crates/fp-frame/src/lib.rs:909` | `crates/fp-conformance/tests/proptest_properties.rs:210` | `crates/fp-conformance/src/lib.rs:2741` | `crates/fp-conformance/tests/ag_e2e.rs:102` | failure replay command from case id: `crates/fp-conformance/src/lib.rs:3801` | covered |
| Join semantics (inner/left/right/outer) | `crates/fp-join/src/lib.rs:507` | `crates/fp-conformance/tests/proptest_properties.rs:298` | `crates/fp-conformance/src/lib.rs:2753` | `crates/fp-conformance/tests/ag_e2e.rs:140` | packet gate enforcement: `crates/fp-conformance/src/lib.rs:620` | covered |
| GroupBy sum behavior + dropna invariants | `crates/fp-groupby/src/lib.rs:1130` | `crates/fp-conformance/tests/proptest_properties.rs:409` | `crates/fp-conformance/src/lib.rs:2769` | `crates/fp-conformance/tests/ag_e2e.rs:171` | per-case evidence counts in `CaseEnd`: `crates/fp-conformance/src/lib.rs:3356` | covered |
| Scalar coercion, missingness, JSON round trips | `crates/fp-types/src/lib.rs:323` | `crates/fp-conformance/tests/proptest_properties.rs:476`, `crates/fp-conformance/tests/proptest_properties.rs:532` | `crates/fp-conformance/src/lib.rs:2838`, `crates/fp-conformance/src/lib.rs:2847`, `crates/fp-conformance/src/lib.rs:2860` | workflow corpus includes null/NaN paths: `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:191` | mismatch corpus + failure digest replay: `crates/fp-conformance/src/lib.rs:3812` | covered |
| CSV ingest/export + dtype checks | `crates/fp-io/src/lib.rs:488` | property round-trip primitives: `crates/fp-conformance/tests/proptest_properties.rs:532` | `crates/fp-conformance/src/lib.rs:2869`, `crates/fp-conformance/src/lib.rs:2884` | workflows: `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:41`, `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:128` | artifact writes logged as `ArtifactWritten`: `crates/fp-conformance/src/lib.rs:3363` | partial (low fixture count) |
| Differential taxonomy and parity gating | `crates/fp-conformance/src/lib.rs:3847` | N/A | `crates/fp-conformance/src/lib.rs:265`, `crates/fp-conformance/src/lib.rs:292` | E2E gate events: `crates/fp-conformance/src/lib.rs:3548` | CLI `--require-green`: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:36` | covered |
| E2E lifecycle orchestration | `crates/fp-conformance/src/lib.rs:4263` | N/A | N/A | orchestrator path: `crates/fp-conformance/src/lib.rs:3500` | forensic JSONL events: `crates/fp-conformance/src/lib.rs:3329` | covered with field gaps |
| Failure forensics digest and one-command replay | `crates/fp-conformance/src/lib.rs:4530` | N/A | consumes differential results | used by E2E report: `crates/fp-conformance/src/lib.rs:3782` | replay string format: `crates/fp-conformance/src/lib.rs:3801`; UX contract: `artifacts/phase2c/FAILURE_FORENSICS_UX.md:24` | covered |
| CI reliability gate forensic traceability | `crates/fp-conformance/src/lib.rs:4856` | N/A | N/A | CI pipeline invokes E2E gate: `crates/fp-conformance/src/lib.rs:1706` | `CiGateForensicsEntry.repro_cmd`: `crates/fp-conformance/src/lib.rs:1531` | covered |
| ASUPERSYNC structured test logs | `crates/fp-runtime/src/lib.rs:629` | property-style checks: `crates/fp-runtime/src/lib.rs:664`, `crates/fp-runtime/src/lib.rs:700`, `crates/fp-runtime/src/lib.rs:738` | N/A | N/A | schema + replay contract: `artifacts/phase2c/ASUPERSYNC_TEST_LOGGING_EVIDENCE.md:51` | covered (runtime-scope only) |

---

## 4. Logging Field Crosswalk for Replay/Forensics

| Layer | Fields currently emitted | Required replay/forensics fields status | Source anchors |
|---|---|---|---|
| Runtime structured unit/property logs (`fp-runtime`) | `packet_id`, `case_id`, `mode`, `seed`, `trace_id`, `assertion_path`, `result`, `replay_cmd` | full coverage for the ASUPERSYNC test layer | `crates/fp-runtime/src/lib.rs:580`, `crates/fp-runtime/src/lib.rs:610` |
| Differential result objects (`fp-conformance`) | `packet_id`, `case_id`, `mode`, `oracle_source`, `status`, `drift_records`, `evidence_records` | has core identity + oracle context; no explicit `replay_key` field | `crates/fp-conformance/src/lib.rs:301` |
| Drift records | `category`, `level`, `location`, `message` | supports mismatch taxonomy; no dedicated `mismatch_class` alias field | `crates/fp-conformance/src/lib.rs:292` |
| E2E forensic events | suite/packet/case/artifact/gate/error events, plus `elapsed_us` on `CaseEnd` | has lifecycle trace; missing explicit `scenario_id`, `trace_id`, `step_id`, `decision_action`; `elapsed_us` currently hardcoded `0` in retrospective path | `crates/fp-conformance/src/lib.rs:3329`, `crates/fp-conformance/src/lib.rs:3535` |
| Failure digest | `packet_id`, `case_id`, `operation`, `mode`, `mismatch_summary`, `replay_command`, `artifact_path` | directly supports one-command replay and artifact lookup | `crates/fp-conformance/src/lib.rs:3698`, `crates/fp-conformance/src/lib.rs:3801` |
| CI gate forensics | `rule_id`, `gate`, `elapsed_ms`, `commands`, `repro_cmd` | includes deterministic reproduction command per failing gate | `crates/fp-conformance/src/lib.rs:1531`, `crates/fp-conformance/src/lib.rs:1569` |
| Drift history ledger | `ts_unix_ms`, `packet_id`, `suite`, `fixture_count`, `passed`, `failed`, `strict_failed`, `hardened_failed`, `gate_pass`, `report_hash` | supports run-level auditability; no per-case replay keys | `crates/fp-conformance/src/lib.rs:440`, `crates/fp-conformance/src/lib.rs:652` |

---

## 5. Replay Command Registry

| Replay target | Command | Anchor |
|---|---|---|
| Full packet conformance gate | `cargo run -p fp-conformance --bin fp-conformance-cli -- --oracle fixture --write-artifacts --require-green` | `scripts/phase2c_gate_check.sh:11` |
| Governance gate report | `cargo run -p fp-conformance --bin fp-governance-gate -- --json-out artifacts/ci/governance_gate_report.json` | `scripts/governance_gate_check.sh:11` |
| Single conformance failure case | `cargo test -p fp-conformance -- <case_id> --nocapture` | `crates/fp-conformance/src/lib.rs:3801` |
| Property failure seed replay | `PROPTEST_SEED="<seed>" cargo test -p fp-conformance --test proptest_properties -- <test_name>` | `artifacts/phase2c/TEST_CONVENTIONS.md:375` |
| ASUPERSYNC property test replay | `cargo test -p fp-runtime -- asupersync_property_hardened_over_cap_forces_repair --nocapture` | `artifacts/phase2c/ASUPERSYNC_TEST_LOGGING_EVIDENCE.md:81` |
| ASUPERSYNC suite replay | `cargo test -p fp-runtime -- asupersync_ --nocapture` | `artifacts/phase2c/ASUPERSYNC_TEST_LOGGING_EVIDENCE.md:87` |

---

## 6. Coverage Gaps and Prioritized Follow-Ups

| Priority | Gap | Impact | Proposed follow-up |
|---|---|---|---|
| `P0` | E2E forensic `CaseEnd.elapsed_us` is currently `0` in retrospective run path | loses actionable per-case latency evidence for forensics/perf gates | capture real per-case timings inside execution loop and emit non-zero deterministic durations (`crates/fp-conformance/src/lib.rs:3540`) |
| `P0` | E2E logs do not currently emit explicit `scenario_id`, `trace_id`, `step_id`, `decision_action` fields | weakens contract alignment with stricter replay/forensics schemas | extend `ForensicEventKind::CaseStart/CaseEnd` payload and E2E emitter pipeline to include these fields |
| `P1` | Differential results have no dedicated `replay_key` and no explicit `mismatch_class` field name | replay/mismatch workflows require deriving from free-form fields | add explicit fields on `DifferentialResult` / `DriftRecord` and include them in mismatch corpus |
| `P1` | CSV/dtype and nan/fill/drop operations have low fixture multiplicity (mostly 1-2 cases) | higher risk of edge-case blind spots | expand fixture matrix per operation with strict+hardened+edge triples per coverage doctrine |
| `P1` | ASUPERSYNC structured log contract is runtime-local; no equivalent schema enforced for broader conformance/e2e logs | inconsistent operator experience and partial replay fidelity | unify schema contract across `fp-runtime` test logs and conformance forensic events |
| `P2` | Live-oracle operation surface in `pandas_oracle.py` remains intentionally narrow | parity confidence limited for unsupported ops | extend oracle dispatch with additional operations once fixture coverage expands |

---

## 7. Acceptance Criteria Check

| Requirement | Status | Evidence |
|---|---|---|
| Each major documented behavior maps to tests/e2e/log artifacts | complete | Section 3 |
| Coverage gaps are explicit and prioritized | complete | Section 6 |
| Logging fields required for replay/forensics are documented | complete | Sections 4 and 5 |

