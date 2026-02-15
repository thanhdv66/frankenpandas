# COMPAT_CLOSURE Anchor Map + Behavior/Workflow Extraction Ledger

Bead: `bd-2gi.29.1` [COMPAT-CLOSURE-A]  
Subsystem: Drop-in compatibility closure surfaces across `fp-types`, `fp-index`, `fp-frame`, `fp-runtime`, `fp-conformance`, and operator workflows

---

## 1. Scope and Purpose

This ledger extracts the compatibility-closure substrate for FrankenPandas from:

1. Legacy oracle behavior contracts.
2. Current Rust implementation anchors.
3. Operational workflows that generate and verify compatibility evidence.

The objective is to make downstream COMPAT-CLOSURE beads (`bd-2gi.29.2+`) mechanical:
contract tables, policy matrices, threat models, validation plans, and final attestation should all trace back to concrete anchors in this document.

---

## 2. Legacy and Spec Anchors

### 2.1 Legacy Oracle

- Legacy oracle root: `/dp/frankenpandas/legacy_pandas_code/pandas`  
- Live adapter: `crates/fp-conformance/oracle/pandas_oracle.py`
  - Strict import/fail-closed behavior: `crates/fp-conformance/oracle/pandas_oracle.py:30-68`
  - Supported operation dispatch:
    - `series_add`: `crates/fp-conformance/oracle/pandas_oracle.py:121-141`
    - `series_join`: `crates/fp-conformance/oracle/pandas_oracle.py:144-180`
    - `groupby_sum`: `crates/fp-conformance/oracle/pandas_oracle.py:183-218`
    - `index_align_union`: `crates/fp-conformance/oracle/pandas_oracle.py:221-248`
    - `index_has_duplicates`: `crates/fp-conformance/oracle/pandas_oracle.py:251-257`
    - `index_first_positions`: `crates/fp-conformance/oracle/pandas_oracle.py:260-273`
  - Unsupported operation fail-closed behavior: `crates/fp-conformance/oracle/pandas_oracle.py:276-290`

### 2.2 Program-Level Compatibility Contract

- Prime directive and AACE identity: `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:3-20`
- Strict vs hardened mode doctrine: `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:56-74`
- Compatibility focus (alignment, dtype coercion, null behavior, join/groupby contracts): `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:66-69`
- Conformance process and assurance tiers: `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:124-141`
- V1 conformance matrix families:
  - construction/dtype, index alignment, null/NaN, groupby, join, sort/filter, CSV, mixed E2E
  - `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:249-260`
- Packet contract and fail-closed rules: `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:262-309`
- Security/compatibility threat matrix: `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:310-321`
- CI gate topology contract: `COMPREHENSIVE_SPEC_FOR_FRANKENPANDAS_V1.md:342-353`

### 2.3 Porting and Parity Tracking Anchors

- Spec-first clean-room method and differential parity mandate: `PLAN_TO_PORT_PANDAS_TO_RUST.md:3-13`
- Mandatory exit criteria (including RaptorQ artifacts): `PLAN_TO_PORT_PANDAS_TO_RUST.md:45-52`
- Current parity matrix and known open gaps:
  - `FEATURE_PARITY.md:11-19`
  - `FEATURE_PARITY.md:41-47`
- Current packet evidence index and drift history anchor:
  - `FEATURE_PARITY.md:23-36`

---

## 3. Target Implementation Anchors (FrankenPandas)

### 3.1 Semantics Core

- DType and scalar/null model:
  - `DType`: `crates/fp-types/src/lib.rs:8-14`
  - `NullKind`: `crates/fp-types/src/lib.rs:18-22`
  - `Scalar`: `crates/fp-types/src/lib.rs:26-32`
  - Missingness + NaN semantics: `crates/fp-types/src/lib.rs:47-77`
  - DType promotion: `crates/fp-types/src/lib.rs:130-144`
  - Cast semantics (lossy and invalid cast guards): `crates/fp-types/src/lib.rs:156-206`

- Alignment semantics:
  - `align_union`: `crates/fp-index/src/lib.rs:519-546`
  - `validate_alignment_plan`: `crates/fp-index/src/lib.rs:548-556`

- Arithmetic/alignment/policy integration in Series:
  - `binary_op_with_policy`: `crates/fp-frame/src/lib.rs:107-154`
  - Strict-mode duplicate rejection path: `crates/fp-frame/src/lib.rs:114-122`
  - Alignment admission decision path: `crates/fp-frame/src/lib.rs:131-136`

- Numeric kernel null/NaN propagation:
  - `binary_numeric` and scalar fallback: `crates/fp-columnar/src/lib.rs:667-725`
  - NaN-aware validity: `crates/fp-columnar/src/lib.rs:650-660`

### 3.2 Join/GroupBy Compatibility Surfaces

- Join semantics:
  - `JoinType`: `crates/fp-join/src/lib.rs:12-17`
  - `join_series` and `join_series_with_options`: `crates/fp-join/src/lib.rs:58-74`
  - Join cardinality/left-right-outer behavior materialization: `crates/fp-join/src/lib.rs:198-257`

- GroupBy semantics:
  - `groupby_sum`: `crates/fp-groupby/src/lib.rs:58-73`
  - Alignment before aggregation: `crates/fp-groupby/src/lib.rs:96-107`
  - `dropna` behavior and first-seen ordering accumulation: `crates/fp-groupby/src/lib.rs:166-190`

### 3.3 Runtime Strict/Hardened Policy and Evidence

- Mode/action enums: `crates/fp-runtime/src/lib.rs:10-21`
- Evidence records: `crates/fp-runtime/src/lib.rs:78-87`
- Evidence ledger: `crates/fp-runtime/src/lib.rs:123-144`
- `RuntimePolicy::strict`/`RuntimePolicy::hardened`: `crates/fp-runtime/src/lib.rs:155-170`
- Unknown feature decision + fail-closed strict override: `crates/fp-runtime/src/lib.rs:172-204`
- Join admission decision + hardened repair override: `crates/fp-runtime/src/lib.rs:206-249`
- Decision engine + expected-loss card: `crates/fp-runtime/src/lib.rs:272-322`
- RaptorQ envelope placeholders (runtime-level metadata contract): `crates/fp-runtime/src/lib.rs:324-340`

### 3.4 Conformance and Evidence Pipeline

- Differential taxonomy:
  - `DriftLevel`: `crates/fp-conformance/src/lib.rs:265-272`
  - `ComparisonCategory`: `crates/fp-conformance/src/lib.rs:277-288`
  - `DriftRecord`: `crates/fp-conformance/src/lib.rs:292-297`
  - `DifferentialReport`: `crates/fp-conformance/src/lib.rs:363-367`
  - `PacketParityReport`: `crates/fp-conformance/src/lib.rs:370-385`

- Packet execution and gates:
  - `run_packets_grouped`: `crates/fp-conformance/src/lib.rs:583-608`
  - `write_grouped_artifacts`: `crates/fp-conformance/src/lib.rs:610-618`
  - `enforce_packet_gates`: `crates/fp-conformance/src/lib.rs:620-650`
  - `append_phase2c_drift_history`: `crates/fp-conformance/src/lib.rs:652-690`
  - `evaluate_parity_gate`: `crates/fp-conformance/src/lib.rs:846-940`

- RaptorQ evidence emission:
  - sidecar + decode proof + mismatch corpus writing: `crates/fp-conformance/src/lib.rs:774-844`

- Failure forensics API:
  - `E2eReport`: `crates/fp-conformance/src/lib.rs:3471-3479`
  - `FailureForensicsReport`: `crates/fp-conformance/src/lib.rs:3729-3736`
  - `build_failure_forensics`: `crates/fp-conformance/src/lib.rs:3782-3837`

### 3.5 Operator Entry Points

- CLI orchestration flags and behavior:
  - `--packet-id`, `--oracle`, `--write-artifacts`, `--require-green`, `--write-drift-history`, `--allow-system-pandas-fallback`
  - `crates/fp-conformance/src/bin/fp-conformance-cli.rs:18-44`
  - Gate enforcement wiring: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:75-77`
  - Artifact output contract: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:79-97`

- Blocking gate script:
  - `scripts/phase2c_gate_check.sh:1-14`

- Failure/replay UX contract:
  - `artifacts/phase2c/FAILURE_FORENSICS_UX.md:8-33`
  - `artifacts/phase2c/FAILURE_FORENSICS_UX.md:105-141`

- Golden workflow corpus:
  - `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:8-35`
  - Scenario map and complexity tiers: `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:333-374`

---

## 4. Extracted Behavioral Contract

### 4.1 Nominal Conditions

1. **DType/null semantics are explicit and closed over known variants.**  
   `DType`, `NullKind`, and tagged `Scalar` encode value/nullness state directly (`crates/fp-types/src/lib.rs:8-32`).

2. **Common dtype promotion is deterministic and explicit.**  
   Mixed numeric promotions resolve through `common_dtype` (`bool+int64 -> int64`, `int64+float64 -> float64`) and invalid pairs fail (`crates/fp-types/src/lib.rs:130-144`).

3. **Series arithmetic is alignment-first.**  
   Binary ops run `align_union` + `validate_alignment_plan` before kernel execution (`crates/fp-frame/src/lib.rs:125-129`).

4. **Strict mode blocks duplicate-label arithmetic.**  
   Duplicate index detection triggers `decide_unknown_feature`; strict mode returns `DuplicateIndexUnsupported` (`crates/fp-frame/src/lib.rs:114-122`).

5. **Hardened mode admits bounded repairs at policy boundaries.**  
   Join admission uses expected-loss selection with optional cap override (`crates/fp-runtime/src/lib.rs:206-249`).

6. **Null/NaN propagation in arithmetic is deterministic.**  
   Missing operands produce missing outputs; NaN-class missing is preserved as NaN sentinel (`crates/fp-columnar/src/lib.rs:694-699`).

7. **Join output shape and null injection obey join type contract.**  
   Inner/left/right/outer materialization paths are explicit in `join_series_with_global_allocator` (`crates/fp-join/src/lib.rs:198-257`).

8. **GroupBy uses first-seen key ordering and configurable null-key treatment.**  
   `dropna` handling and stable ordering map are explicit in aggregation loops (`crates/fp-groupby/src/lib.rs:166-190`).

9. **Differential drift is taxonomy-classified, not opaque string compare only.**  
   Categories (`value/type/shape/index/nullness`) and severities (`critical/non_critical/informational`) are first-class (`crates/fp-conformance/src/lib.rs:265-297`).

10. **Packet gates are enforced from machine-checkable budgets.**  
    `evaluate_parity_gate` validates packet id/suite/fixture count and strict/hardened budgets (`crates/fp-conformance/src/lib.rs:846-940`).

11. **Conformance evidence is durable and replayable.**  
    Each packet emits parity report, gate result, mismatch corpus, RaptorQ sidecar, and decode proof (`crates/fp-conformance/src/lib.rs:774-844`).

12. **Operator command contract is explicit and fail-closed capable.**  
    `--require-green` converts drift into non-zero CLI exit (`crates/fp-conformance/src/bin/fp-conformance-cli.rs:75-77`).

### 4.2 Edge Conditions

13. **Zero-fixture suites are never considered green.**  
    `PacketParityReport::is_green()` requires `failed == 0 && fixture_count > 0` (`crates/fp-conformance/src/lib.rs:380-385`).

14. **Clock skew degrades decision timestamps to zero instead of crashing.**  
    `now_unix_ms().unwrap_or_default()` in `decide()` yields `ts_unix_ms = 0` on error (`crates/fp-runtime/src/lib.rs:307-309`).

15. **Unsupported label/scalar kinds fail closed in oracle adapter.**  
    `label_from_json` and `scalar_from_json` raise `OracleError` on unknown kind (`crates/fp-conformance/oracle/pandas_oracle.py:70-97`).

16. **Oracle join surface is intentionally narrow.**  
    `series_join` accepts only `inner|left`; invalid join types are rejected (`crates/fp-conformance/oracle/pandas_oracle.py:147-152`).

17. **Gate config mismatch is itself a gate failure reason.**  
    Packet id/suite mismatches append hard reasons in `evaluate_parity_gate` (`crates/fp-conformance/src/lib.rs:887-901`).

18. **Drift history ledger is append-only JSONL.**  
    Entries append with hash of parity report bytes (`crates/fp-conformance/src/lib.rs:663-687`).

19. **Failure UX contract truncates mismatch summary and emphasizes replay commands.**  
    Expected digest format is documented and deterministic (`artifacts/phase2c/FAILURE_FORENSICS_UX.md:19-33`).

20. **Workflow corpus defines out-of-scope behaviors explicitly.**  
    Multi-threaded access, datetime ops, MultiIndex, and window functions are listed as non-goals (`artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:378-391`).

### 4.3 Adversarial Conditions

21. **Unknown oracle operations are hard rejected.**  
    `dispatch` raises on unsupported operation string (`crates/fp-conformance/oracle/pandas_oracle.py:276-290`).

22. **Strict-legacy oracle import can be configured fail-closed.**  
    Without fallback allowance, import failures return oracle error and non-zero exit (`crates/fp-conformance/oracle/pandas_oracle.py:57-68`, `crates/fp-conformance/oracle/pandas_oracle.py:307-319`).

23. **Join-explosion risk is policy-mediated, not silently ignored.**  
    Runtime join admission uses estimated row count evidence and hardened cap override (`crates/fp-runtime/src/lib.rs:217-243`).

24. **Artifact corruption is surfaced through scrub/decode artifacts.**  
    Sidecar generation + decode recovery drill + status propagation are mandatory artifact outputs (`crates/fp-conformance/src/lib.rs:778-810`).

25. **Gate-enforced command is release-blocking by contract.**  
    `--require-green` path in CLI and `phase2c_gate_check.sh` script operationalize hard failures (`crates/fp-conformance/src/bin/fp-conformance-cli.rs:75-77`, `scripts/phase2c_gate_check.sh:11-14`).

---

## 5. Operator Workflow Ledger

### W1. Packet Gate Run (Fixture Oracle, Blocking)

**Intent:** produce canonical packet artifacts and fail fast on drift.

1. Run packet suite with artifacts and gate enforcement:
   - reference command path: `scripts/phase2c_gate_check.sh:11-14`
2. CLI executes grouped packets:
   - `run_packets_grouped`: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:62`
3. Artifacts are emitted:
   - `write_grouped_artifacts`: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:79-92`
4. Drift history appends:
   - `append_phase2c_drift_history`: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:94-97`
5. Any parity/gate failure returns non-zero:
   - `enforce_packet_gates`: `crates/fp-conformance/src/bin/fp-conformance-cli.rs:75-77`

### W2. Live Oracle Verification (Legacy-Strict or Fallback)

**Intent:** compare against pandas behavior beyond frozen fixtures.

1. Use `--oracle live` in CLI (`crates/fp-conformance/src/bin/fp-conformance-cli.rs:25-31`).
2. Oracle adapter imports legacy pandas first (`crates/fp-conformance/oracle/pandas_oracle.py:46-55`).
3. In strict legacy mode with no fallback, import failures are hard errors (`crates/fp-conformance/oracle/pandas_oracle.py:57-60`).
4. Optional fallback mode allows system pandas import (`crates/fp-conformance/oracle/pandas_oracle.py:39-42`, `crates/fp-conformance/oracle/pandas_oracle.py:62-67`).

### W3. Failure Triage and Replay

**Intent:** collapse failure-to-repro cycle to one command.

1. Build/report forensics object in Rust:
   - `build_failure_forensics`: `crates/fp-conformance/src/lib.rs:3782-3837`
2. Read digest format contract:
   - `artifacts/phase2c/FAILURE_FORENSICS_UX.md:19-33`
3. Replay fixture or packet with command from digest:
   - replay patterns documented at `artifacts/phase2c/FAILURE_FORENSICS_UX.md:107-141`
4. Use `parity_mismatch_corpus.json` for full mismatch payload:
   - artifact mapping at `artifacts/phase2c/FAILURE_FORENSICS_UX.md:87-93`

### W4. Compatibility Closure Tracking

**Intent:** map implemented packet evidence to full drop-in closure backlog.

1. Read open parity matrix gaps:
   - `FEATURE_PARITY.md:11-19`
2. Correlate green packet evidence:
   - `FEATURE_PARITY.md:23-31`
3. Tie unresolved families to COMPAT-CLOSURE backlog:
   - `bd-2gi.29.*` bead stack (Beads graph)
4. Maintain drift history continuity:
   - `artifacts/phase2c/drift_history.jsonl` (`FEATURE_PARITY.md:35`)

---

## 6. Type Inventory (Compatibility-Critical)

### 6.1 Core Data Semantics Types

- `DType`, `NullKind`, `Scalar` (`crates/fp-types/src/lib.rs:8-32`)
- `TypeError` and cast/promotion helpers (`crates/fp-types/src/lib.rs:112-206`)
- `Index`, `IndexLabel`, `AlignmentPlan` via `fp-index` (alignment entrypoints at `crates/fp-index/src/lib.rs:519-556`)

### 6.2 Runtime Decision and Audit Types

- `RuntimeMode`, `DecisionAction`, `IssueKind` (`crates/fp-runtime/src/lib.rs:10-30`)
- `CompatibilityIssue`, `EvidenceTerm`, `DecisionMetrics`, `DecisionRecord` (`crates/fp-runtime/src/lib.rs:32-87`)
- `EvidenceLedger` (`crates/fp-runtime/src/lib.rs:123-144`)
- `RuntimePolicy` (`crates/fp-runtime/src/lib.rs:147-250`)
- `RaptorQEnvelope`, `RaptorQMetadata`, `ScrubStatus`, `DecodeProof` (`crates/fp-runtime/src/lib.rs:324-376`)

### 6.3 Differential/Gate Types

- `CaseStatus`, `CaseResult` (`crates/fp-conformance/src/lib.rs:242-258`)
- `DriftLevel`, `ComparisonCategory`, `DriftRecord` (`crates/fp-conformance/src/lib.rs:265-297`)
- `DifferentialResult`, `DifferentialReport`, `DriftSummary` (`crates/fp-conformance/src/lib.rs:301-367`)
- `PacketParityReport`, `PacketGateResult` (`crates/fp-conformance/src/lib.rs:370-397`)
- `PacketDriftHistoryEntry` (`crates/fp-conformance/src/lib.rs:440-451`)
- `CiGate`, `CiPipelineResult` (`crates/fp-conformance/src/lib.rs:1376`, `crates/fp-conformance/src/lib.rs:1522`)
- `E2eReport`, `FailureForensicsReport` (`crates/fp-conformance/src/lib.rs:3471`, `crates/fp-conformance/src/lib.rs:3729`)

---

## 7. Rule Ledger

1. Compatibility closure is evidence-driven, not assertion-driven: all claims must map to packet artifacts and gate results.
2. Strict mode must fail closed for unknown/incompatible features (`crates/fp-runtime/src/lib.rs:198-200`).
3. Hardened mode may repair but cannot silently suppress evidence recording (`crates/fp-runtime/src/lib.rs:246-248`).
4. Series arithmetic must align by union index before value ops (`crates/fp-frame/src/lib.rs:125-129`).
5. Alignment plans must pass shape validation (`crates/fp-index/src/lib.rs:548-556`).
6. Missingness and NaN handling must propagate deterministically (`crates/fp-columnar/src/lib.rs:694-699`).
7. Differential drift must be categorized by value/type/shape/index/nullness (`crates/fp-conformance/src/lib.rs:277-288`).
8. Packet parity is green only when fixture_count > 0 and failed == 0 (`crates/fp-conformance/src/lib.rs:380-385`).
9. Gate evaluation must check packet/suite consistency and budget thresholds (`crates/fp-conformance/src/lib.rs:887-940`).
10. Artifact bundle per packet includes parity report, gate result, mismatch corpus, RaptorQ sidecar, and decode proof (`crates/fp-conformance/src/lib.rs:774-844`).
11. Drift history must append immutable JSONL summaries with report hash (`crates/fp-conformance/src/lib.rs:668-687`).
12. Oracle adapter must fail on unsupported operation or schema kind (`crates/fp-conformance/oracle/pandas_oracle.py:70-97`, `crates/fp-conformance/oracle/pandas_oracle.py:276-290`).
13. Replay commands are part of required failure UX contract (`artifacts/phase2c/FAILURE_FORENSICS_UX.md:11-32`).
14. Scenario corpus is authoritative for nominal multi-step user workflows (`artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:10-35`).
15. Declared non-goals must remain explicit to avoid accidental closure claims (`artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:378-391`).

---

## 8. Error Ledger

1. **Clock skew fallback:** runtime decision timestamp defaults to `0` on clock error (`crates/fp-runtime/src/lib.rs:307-309`).
2. **Strict duplicate-index arithmetic rejection:** `DuplicateIndexUnsupported` in strict mode (`crates/fp-frame/src/lib.rs:120-122`).
3. **Oracle unsupported operation:** hard `OracleError` (`crates/fp-conformance/oracle/pandas_oracle.py:290`).
4. **Oracle invalid join type:** hard `OracleError` for non `inner|left` (`crates/fp-conformance/oracle/pandas_oracle.py:150-151`).
5. **Gate enforcement failure aggregation:** multiple gate/parity failures are folded into one hard CLI error (`crates/fp-conformance/src/lib.rs:642-648`).
6. **Packet gate config mismatch:** packet-id/suite mismatches generate explicit rejection reasons (`crates/fp-conformance/src/lib.rs:887-901`).
7. **Mismatched column lengths in arithmetic:** `LengthMismatch` from `binary_numeric` (`crates/fp-columnar/src/lib.rs:668-673`).
8. **Incompatible dtype promotion:** `TypeError::IncompatibleDtypes` (`crates/fp-types/src/lib.rs:140-141`).

---

## 9. Hidden Assumptions

1. Current live-oracle coverage assumes operation set in `dispatch` is sufficient for active packet families (`crates/fp-conformance/oracle/pandas_oracle.py:276-289`).
2. `FEATURE_PARITY.md` status matrix is treated as up-to-date planning truth for closure readiness (`FEATURE_PARITY.md:11-19`).
3. Strict/hardened policy behavior is currently concentrated in runtime + series arithmetic and may not yet cover all future API families.
4. Replay command examples in forensics UX assume local cargo execution paths and deterministic fixture naming (`artifacts/phase2c/FAILURE_FORENSICS_UX.md:107-141`).
5. Packet gate budget logic assumes `parity_gate.yaml` exists and is valid for every packet (`crates/fp-conformance/src/lib.rs:854-856`).

---

## 10. Undefined/Gap Edges for Compatibility Closure

1. Full DataFrame-level merge/concat parity remains open (not yet closure-green): `FEATURE_PARITY.md:17`.
2. Full `loc/iloc` parity remains open: `FEATURE_PARITY.md:15`.
3. Full nanops matrix remains open: `FEATURE_PARITY.md:18`.
4. CSV parser/formatter parity matrix remains open: `FEATURE_PARITY.md:19`.
5. Extended API families listed in scenario non-goals (MultiIndex, datetime, window functions) are not yet represented in packet closure artifacts: `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:378-391`.
6. Decision timestamp reliability under clock skew still uses fallback `0`; no explicit mitigation pipeline in runtime yet (`crates/fp-runtime/src/lib.rs:264-309`).
7. Live oracle fallback (`--allow-system-pandas-fallback`) can weaken strict provenance if enabled outside controlled runs (`crates/fp-conformance/src/bin/fp-conformance-cli.rs:42-44`).

---

## 11. Downstream Hand-off Notes (for bd-2gi.29.2 and bd-2gi.29.3)

Use this anchor map to construct the COMPAT-CLOSURE contract table and strict/hardened policy matrix by:

1. Converting Rule Ledger items into machine-checkable contract rows.
2. Mapping each rule to strict-mode and hardened-mode expected outcomes.
3. Linking each rule to evidence artifacts (`parity_report`, `parity_gate_result`, `mismatch_corpus`, `drift_history`).
4. Marking each Undefined/Gap edge as explicit open contract item (not silent debt).
5. Emitting a fail-closed threat matrix in `artifacts/phase2c/COMPAT_CLOSURE_THREAT_MODEL.md` with explicit abuse/drift classes, compatibility envelopes, and testable controls keyed to `CC-*`/`FC-*` rows.
6. Emitting an integration/module-seam plan in `artifacts/phase2c/COMPAT_CLOSURE_RUST_INTEGRATION.md` that maps contract/threat rows to concrete ownership and phased execution for `29.5..29.9`.
