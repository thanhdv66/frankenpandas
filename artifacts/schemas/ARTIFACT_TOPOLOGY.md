# Artifact Topology Lock

Canonical artifact schema registry and topology rules for FrankenPandas Phase-2C.

Frozen per bead **bd-2gi.3**: Contract Schema + Artifact Topology Lock.

---

## Schema Registry

All JSON Schema files in this directory are the single source of truth for artifact structure. Any serialization or deserialization of these artifacts must conform to these schemas.

| Schema File | Artifact | Format | Rust Struct | Emitter |
|---|---|---|---|---|
| `shared_definitions.schema.json` | (shared types) | N/A | `Scalar`, `IndexLabel`, enums | N/A |
| `packet_fixture.schema.json` | Test fixture | JSON | `PacketFixture` | manual / proptest |
| `parity_report.schema.json` | Parity report | JSON | `PacketParityReport` | `run_packet_suite()` |
| `parity_gate.schema.json` | Gate config | YAML | (deserialized map) | manual |
| `gate_result.schema.json` | Gate result | JSON | `PacketGateResult` | `evaluate_parity_gate()` |
| `raptorq_sidecar.schema.json` | RaptorQ sidecar | JSON | `RaptorQSidecarArtifact` | `generate_raptorq_sidecar()` |
| `decode_proof_artifact.schema.json` | Decode proof | JSON | (custom wrapper) | `run_raptorq_decode_recovery_drill()` |
| `mismatch_corpus.schema.json` | Mismatch corpus | JSON | (custom wrapper) | `write_packet_artifacts()` |
| `drift_history_entry.schema.json` | Drift history | JSONL | `PacketDriftHistoryEntry` | `append_phase2c_drift_history()` |
| `fixture_manifest.schema.json` | Fixture manifest | JSON | (custom wrapper) | manual |
| `test_log_entry.schema.json` | Structured test log | eprintln | N/A | `#[test]` functions |

---

## Artifact File Layout

```
artifacts/
  schemas/                              # THIS DIRECTORY - frozen schemas
    shared_definitions.schema.json
    packet_fixture.schema.json
    parity_report.schema.json
    parity_gate.schema.json
    gate_result.schema.json
    raptorq_sidecar.schema.json
    decode_proof_artifact.schema.json
    mismatch_corpus.schema.json
    drift_history_entry.schema.json
    fixture_manifest.schema.json
    test_log_entry.schema.json
    ARTIFACT_TOPOLOGY.md                # THIS FILE
  phase2c/
    drift_history.jsonl                 # Append-only drift ledger (JSONL)
    ESSENCE_EXTRACTION_LEDGER.md        # Foundation ledger (bd-2gi.1)
    TEST_CONVENTIONS.md                 # Test conventions + log contract (bd-2gi.5)
    COVERAGE_FLAKE_BUDGETS.md           # Reliability SLOs + gate definitions (bd-2gi.22)
    USER_WORKFLOW_SCENARIOS.md          # User workflow scenario corpus (bd-2gi.20)
    FAILURE_FORENSICS_UX.md             # Failure diagnostics design (bd-2gi.21)
    PERFORMANCE_BASELINES.md            # Performance protocol + baselines (bd-2gi.8)
    ADVERSARIAL_FUZZ_CORPUS.md          # Adversarial corpus + fuzz workflow (bd-2gi.7)
    FP-P2C-{NNN}/                       # Per-packet artifact directory
      parity_gate.yaml                  # Gate configuration
      parity_report.json                # Parity report
      parity_report.raptorq.json        # RaptorQ sidecar for parity report
      parity_report.decode_proof.json   # Decode proof artifact
      parity_gate_result.json           # Gate evaluation result
      parity_mismatch_corpus.json       # Failed cases corpus
      fixture_manifest.json             # Fixture inventory
      legacy_anchor_map.md              # Behavioral anchors (Markdown)
      contract_table.md                 # Input/output contracts (Markdown)
      risk_note.md                      # Risk assessment (Markdown)
  perf/
    ROUND{N}_BASELINE.md
    ROUND{N}_OPPORTUNITY_MATRIX.md
    ROUND{N}_ISOMORPHISM_PROOF.md
    ROUND{N}_RECOMMENDATION_CONTRACT.md

crates/fp-conformance/
  fixtures/
    packets/
      fp_p2c_{NNN}_*.json              # Individual fixture files
    smoke_case.json                     # Smoke test fixture
  oracle/
    pandas_oracle.py                    # Live oracle subprocess
```

---

## Topology Rules

### Rule T1: Schema-First Serialization
Every JSON/YAML artifact emitted by fp-conformance or fp-runtime must validate against its corresponding schema in this directory. New artifact types require a schema file here before code can emit them.

### Rule T2: Packet Directory Convention
Per-packet artifacts live under `artifacts/phase2c/FP-P2C-{NNN}/` where `{NNN}` is a zero-padded three-digit packet number. Every packet directory must contain at minimum: `parity_gate.yaml`, `fixture_manifest.json`, `legacy_anchor_map.md`, `contract_table.md`, `risk_note.md`.

### Rule T3: Fixture File Naming
Fixture JSON files follow the pattern `fp_p2c_{NNN}_{description}_{mode}.json` and live under `crates/fp-conformance/fixtures/packets/`. The `case_id` field inside must match the filename stem.

### Rule T4: Drift History Append-Only
`drift_history.jsonl` is append-only. Rows are never modified or deleted. Each row conforms to `drift_history_entry.schema.json`.

### Rule T5: RaptorQ Sidecar Pairing
Every `parity_report.json` must have a corresponding `parity_report.raptorq.json` sidecar and `parity_report.decode_proof.json`. The sidecar's `source_hash` must match the SHA-256 of the parity report content.

### Rule T6: Gate Configuration Immutability
Once a `parity_gate.yaml` is set for a packet, its thresholds may only be tightened (lower budgets), never loosened, without explicit documented justification in a bead comment.

### Rule T7: Markdown Artifacts Are Not Machine-Parsed
`legacy_anchor_map.md`, `contract_table.md`, and `risk_note.md` are human-readable documentation artifacts. Their structure is conventional (section headings) but not schema-enforced. Machine-readable contracts live in the JSON artifacts.

### Rule T8: Schema Versioning
Schema changes require a new bead. The `$id` field in each schema serves as the version anchor. Breaking changes (field removal, type narrowing, new required fields) are forbidden without migration tooling.

### Rule T9: Cross-Reference Integrity
- `fixture_manifest.json` entries must reference existing fixture files
- `parity_report.json` case results must reference valid packet_id and case_id pairs
- `gate_result.json` must be evaluable against its packet's `parity_gate.yaml`

### Rule T10: Shared Type Stability
The `Scalar` and `IndexLabel` tagged-union formats in `shared_definitions.schema.json` are load-bearing across all fixture and report formats. Changes to their structure require coordinated migration of all existing artifacts.

---

## Markdown Artifact Templates

### legacy_anchor_map.md Sections
1. Packet ID and subsystem identifier
2. Legacy Anchors (pandas source file references)
3. Extracted Behavioral Contract
4. Rust Slice Implementation reference
5. Type Inventory
6. Rule Ledger
7. Error Ledger
8. Hidden Assumptions
9. Undefined-Behavior Edges

### contract_table.md Columns
packet_id | input_contract | output_contract | error_contract | null_contract | index_alignment_contract | strict_mode_policy | hardened_mode_policy | excluded_scope | oracle_tests | performance_sentinels | compatibility_risks | raptorq_artifacts

### risk_note.md Sections
1. Primary risk statement
2. Mitigations
3. Isomorphism Proof Hook
4. Invariant Ledger Hooks (FP-I1, FP-I2, FP-I4, ...)

---

## Relationship Diagram

```
parity_gate.yaml ──evaluates──> parity_report.json ──produces──> gate_result.json
                                      │
                                      ├──> parity_mismatch_corpus.json (failed subset)
                                      ├──> parity_report.raptorq.json (durability sidecar)
                                      │         └──> parity_report.decode_proof.json
                                      └──> drift_history.jsonl (append per run)

fixture_manifest.json ──references──> fixtures/packets/*.json (PacketFixture)
                                            │
                                            └──> consumed by run_packet_suite()
                                                       └──> produces parity_report.json

legacy_anchor_map.md ──informs──> contract_table.md ──informs──> risk_note.md
```
