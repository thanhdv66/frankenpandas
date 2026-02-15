# FRANKENTUI Contract Table + Strict/Hardened Policy Matrix

Bead: `bd-2gi.28.2` [FRANKENTUI-B]
Subsystem: FRANKENTUI -- terminal user interface operator cockpit (conformance, performance, forensics, decision dashboards)
Source anchor: `FRANKENTUI_ANCHOR_MAP.md` (bd-2gi.28.1)

---

## 1. Summary Contract (Tabular)

| Field | Contract |
|---|---|
| packet_id | `FRANKENTUI-B` |
| input_contract | Artifact JSON files (parity reports, gate results, mismatch corpora, RaptorQ sidecars, decode proofs); `drift_history.jsonl` (JSONL line-stream); `EvidenceLedger` (in-memory `Vec<DecisionRecord>`); `ForensicLog` (in-memory event vec); `CiPipelineResult` (gate pipeline); `ConformalGuard` (calibration state); `RuntimePolicy` (mode configuration) |
| output_contract | Rendered TUI views only (no file mutations); optional clipboard export of replay commands; optional screenshot/text export of dashboard state; all output is ephemeral terminal rendering |
| error_contract | Missing artifact files degrade individual panels (not global crash); malformed JSON/JSONL lines are skipped with warning count; terminal too small triggers minimum-size gate; clock-skew timestamps displayed as "unknown"; IO errors on artifact directory degrade entire conformance dashboard while other dashboards remain functional |
| null_contract | `ts_unix_ms = 0` displayed as "unknown" (not "1970-01-01"); uncalibrated conformal guard shows "Uncalibrated (N/2 scores)"; placeholder RaptorQ envelopes (`source_hash = "blake3:placeholder"`) render as "placeholder" status; `hardened_join_row_cap = None` displays as "unlimited"; empty evidence ledger shows "No decisions recorded" |
| strict_mode_policy | Conformance dashboard highlights strict-mode failures with distinct color; gate evaluation surfaces `strict_failed` counts prominently; policy provenance panel shows `fail_closed_unknown_features = true`; decision cards carry `mode: Strict` badge; all strict-only gate budget violations flagged |
| hardened_mode_policy | Conformance dashboard surfaces hardened-mode divergence budgets; gate evaluation shows `hardened_failed` counts alongside allowlisted categories; policy provenance panel shows `hardened_join_row_cap` value; decision cards carry `mode: Hardened` badge with repair-permitted indicator |
| excluded_scope | Artifact file writes (read-only consumer); live conformance run execution; policy modification; async runtime; multi-user or remote access; authentication/authorization |
| oracle_tests | Planned: FTUI rendering integration tests validating panel content against known artifact fixtures; feature-flag isolation tests (asupersync enabled/disabled); terminal size boundary tests |
| performance_sentinels | Render budget: 16ms per frame (60 FPS target); artifact parse: <100ms per file; drift history: line-streaming with <500ms initial load for 10K lines; forensic log: virtual scrolling at >1000 events; mismatch corpus: paginated at >100 entries; evidence terms: capped display at 50 with "and N more" |
| compatibility_risks | `frankentui` crate does not exist yet (unwritten dependency); feature-flag interaction with `asupersync` requires conditional compilation; version coupling to `fp-conformance` and `fp-runtime` Serialize/Deserialize contracts; terminal backend choice (crossterm/termion/termwiz) unspecified |
| raptorq_artifacts | Read-only display of `RaptorQSidecarArtifact` metadata, `DecodeProofArtifact` status, and `RaptorQEnvelope` provenance; no encode/decode operations |

---

## 2. Input Contract (Detailed)

### 2.1 Artifact Files (Disk)

| Source File | Type | Parse Contract | Access Pattern |
|---|---|---|---|
| `artifacts/phase2c/{packet_id}/parity_report.json` | `PacketParityReport` | JSON; `serde_json::from_str` | Lazy on-demand; re-read on refresh |
| `artifacts/phase2c/{packet_id}/parity_gate_result.json` | `PacketGateResult` | JSON; `serde_json::from_str` | Lazy on-demand; re-read on refresh |
| `artifacts/phase2c/{packet_id}/parity_mismatch_corpus.json` | `Vec<CaseResult>` | JSON; `serde_json::from_str` | Lazy on-demand; paginated display |
| `artifacts/phase2c/{packet_id}/parity_report.raptorq.json` | `RaptorQSidecarArtifact` | JSON; `serde_json::from_str` | Lazy on-demand |
| `artifacts/phase2c/{packet_id}/parity_report.decode_proof.json` | `DecodeProofArtifact` | JSON; `serde_json::from_str` | Lazy on-demand |
| `artifacts/phase2c/drift_history.jsonl` | `Vec<PacketDriftHistoryEntry>` | JSONL; one `serde_json::from_str` per line | Line-streaming; malformed lines skipped |
| `artifacts/schemas/*.schema.json` | JSON Schema | Not parsed by FTUI | Reference only; not consumed at runtime |

**Preconditions:**
- All artifact JSON files are UTF-8 encoded (produced by `serde_json::to_string()`).
- File paths follow deterministic scheme: `artifacts/phase2c/{packet_id}/{artifact_file}`.
- FTUI must not hold file locks on any artifact file (concurrent conformance writes possible).
- Packet IDs follow `FP-P2C-NNN` naming convention (currently FP-P2C-001 through FP-P2C-011).

### 2.2 In-Memory Types (fp-runtime)

| Type | Source | Fields Consumed | FTUI Usage |
|---|---|---|---|
| `DecisionRecord` | `EvidenceLedger::records()` | `ts_unix_ms`, `mode`, `action`, `issue`, `prior_compatible`, `metrics`, `evidence` | Decision dashboard cards, evidence trace drilldown |
| `GalaxyBrainCard` | `decision_to_card(&DecisionRecord)` | `title`, `equation`, `substitution`, `intuition` | Galaxy-brain card rendering (4-line plain text or styled widget) |
| `EvidenceLedger` | Direct reference | `records()` slice | Decision count, record enumeration |
| `RuntimePolicy` | Direct reference | `mode`, `fail_closed_unknown_features`, `hardened_join_row_cap` | Policy provenance panel |
| `ConformalGuard` | Direct reference | `is_calibrated()`, `calibration_count()`, `conformal_quantile()`, `empirical_coverage()`, `coverage_alert()` | Conformal guard dashboard panel |
| `ConformalPredictionSet` | `ConformalGuard::evaluate()` | `quantile_threshold`, `current_score`, `bayesian_action_in_set`, `admissible_actions`, `empirical_coverage` | Per-decision conformal status |
| `RaptorQEnvelope` | Deserialized from artifact | `artifact_id`, `artifact_type`, `source_hash`, `raptorq`, `scrub`, `decode_proofs` | RaptorQ metadata display |

### 2.3 In-Memory Types (fp-conformance)

| Type | Source | Fields Consumed | FTUI Usage |
|---|---|---|---|
| `PacketParityReport` | Deserialized from artifact | `suite`, `packet_id`, `fixture_count`, `passed`, `failed`, `results` | Conformance dashboard per-packet cards |
| `PacketGateResult` | Deserialized from artifact | `packet_id`, `pass`, `strict_total`, `strict_failed`, `hardened_total`, `hardened_failed`, `reasons` | Gate status badges, strict/hardened failure counts |
| `PacketDriftHistoryEntry` | Parsed from JSONL | `ts_unix_ms`, `packet_id`, `fixture_count`, `passed`, `failed`, `strict_failed`, `hardened_failed`, `gate_pass`, `report_hash` | Drift trend visualization |
| `ForensicLog` | In-memory or deserialized | `events: Vec<ForensicEvent>` | Forensic event stream timeline |
| `ForensicEventKind` | Embedded in `ForensicEvent` | All 10 variants | Event classification, filtering, color-coding |
| `CiPipelineResult` | In-memory | `gates`, `all_passed`, `first_failure`, `elapsed_ms` | CI pipeline status bar |
| `CiGateResult` | Embedded in `CiPipelineResult` | `gate`, `passed`, `elapsed_ms`, `summary`, `errors` | Per-gate drill-down |
| `FailureDigest` | Embedded in `FailureForensicsReport` | `packet_id`, `case_id`, `operation`, `mode`, `mismatch_summary`, `replay_command`, `artifact_path` | Failure digest list with replay commands |
| `E2eReport` | In-memory | `is_green()`, `total_fixtures`, `total_passed`, `total_failed`, `forensic_log`, `gate_results` | Top-level E2E summary |
| `DriftRecord` | Embedded in `DifferentialResult` | `category`, `level`, `location`, `message` | Drift detail drilldown |
| `DriftSummary` | Embedded in `DifferentialReport` | `total_drift_records`, `critical_count`, `non_critical_count`, `informational_count`, `categories` | Drift distribution display |

---

## 3. Output Contract (Detailed)

### 3.1 Rendered Views (Terminal)

| View | Content | Mutation | Persistence |
|---|---|---|---|
| Conformance dashboard | Per-packet cards: gate state, fixture pass/fail, drift sparkline | None (read-only render) | Ephemeral |
| Decision dashboard | Evidence ledger entries, galaxy-brain cards, conformal guard status | None | Ephemeral |
| Forensics dashboard | Failure digest list, forensic event timeline, artifact cross-reference | None | Ephemeral |
| CI pipeline view | 9-gate ordered status bar with per-gate elapsed time | None | Ephemeral |
| Policy provenance panel | Active `RuntimeMode`, `fail_closed_unknown_features`, `hardened_join_row_cap` | None | Ephemeral |
| Conformal guard panel | Calibration status, quantile threshold, empirical coverage, alert badge | None | Ephemeral |
| Drift trend chart | Time-series pass rate per packet with gate overlay | None | Ephemeral |

### 3.2 Auxiliary Outputs

| Output | Trigger | Content | Guarantee |
|---|---|---|---|
| Clipboard copy | User selects replay command | Verbatim `FailureDigest::replay_command` string | Exact string, no modification |
| Text export (planned) | User requests dashboard snapshot | Plain-text rendering of current view | Matches `render_plain()` canonical format where applicable |
| Screenshot (planned) | User requests terminal capture | Terminal buffer contents | Platform-dependent; not guaranteed on all terminals |

**Post-conditions:**
- FTUI never writes to `drift_history.jsonl`, parity reports, gate results, or any other artifact file.
- FTUI never modifies `EvidenceLedger`, `RuntimePolicy`, or `ConformalGuard` state.
- All FTUI output is side-effect-free terminal rendering or user-initiated clipboard operations.

---

## 4. Error Contract (Detailed)

### 4.1 File System Errors

| Error | Trigger | Handling | Scope |
|---|---|---|---|
| Artifact file not found | `{packet_id}/parity_report.json` missing | Display "artifact not found: {path}" in packet panel | Single packet panel degraded |
| Artifact directory inaccessible | `artifacts/phase2c/` missing or unreadable | Display "artifact directory not accessible: {path}" at dashboard level | Entire conformance dashboard degraded; other dashboards functional |
| Artifact JSON parse failure | Invalid JSON in artifact file | Display "parse error: {path}: {error}" in packet panel; skip artifact | Single packet data missing |
| Drift history file missing | `drift_history.jsonl` does not exist | Display "No drift history available" in trend panel | Drift trend panel empty but functional |
| Drift history malformed line | Non-JSON line in JSONL | Skip line; increment "skipped lines" counter in footer | Drift trend may have gaps |
| Drift history trailing incomplete | Concurrent write produces partial last line | Tolerate and skip trailing incomplete line | Last data point may be missing until next refresh |

### 4.2 Terminal Errors

| Error | Trigger | Handling | Scope |
|---|---|---|---|
| Terminal too small | Dimensions below minimum layout threshold | Display "terminal too small (min: WxH)" message; refuse to render | Dashboard blocked; auto-resumes on resize |
| Terminal resize during render | `SIGWINCH` received mid-render | Abort current render; recompute layout; re-render | Single frame dropped |
| Terminal capability missing | No color support or unsupported escape codes | Fall back to monochrome rendering | Visual degradation only |

### 4.3 Data Errors

| Error | Trigger | Handling | Scope |
|---|---|---|---|
| Clock skew timestamp (`ts_unix_ms = 0`) | `DecisionRecord` or `ForensicEvent` carries epoch-zero timestamp | Display as "unknown" or "epoch"; do not sort as earliest | Chronological ordering degraded for affected records |
| Empty evidence ledger | `EvidenceLedger::records()` returns empty slice | Display "No decisions recorded"; suppress galaxy-brain card panel | Decision dashboard minimal content |
| Oversized forensic log (>10K events) | `ForensicLog::events` exceeds virtual scrolling threshold | Virtual scrolling; render visible window only; footer shows "N of M events" | Large logs navigable without UI freeze |
| Oversized mismatch corpus (>100 entries) | Large number of failing fixtures | Paginate with "showing N of M mismatches" | Truncated display with navigation |
| Excessive evidence terms (>50 per record) | `DecisionRecord::evidence` has unbounded length | Cap rendered terms at 50 with "and N more" indicator | Single record's detail view truncated |
| Posterior display extremes | `posterior_compatible` very close to 0.0 or 1.0 | Render with 4-decimal fixed precision (e.g., "0.0000" or "1.0000") | No scientific notation artifacts |

### 4.4 Feature Flag Errors

| Error | Trigger | Handling | Scope |
|---|---|---|---|
| Asupersync panel without feature | FTUI code references `outcome_to_action()` without `#[cfg(feature = "asupersync")]` | Compile-time error | Build failure (correct behavior) |
| Asupersync feature disabled at runtime | `asupersync` feature not enabled | Suppress asupersync-specific panels without error | Panel absence; no runtime degradation |

---

## 5. Null/Missing Contract (Detailed)

| Scenario | Null Representation | Display Behavior | Risk |
|---|---|---|---|
| Clock skew in `DecisionRecord` | `ts_unix_ms = 0` | Render as "unknown" or "epoch" | Indistinguishable from genuine epoch-0 records; chronological sort degraded |
| Clock skew in `ForensicEvent` | `ts_unix_ms = 0` | Render as "unknown" in timeline | Event may appear out-of-order in timeline |
| Empty evidence vector | `evidence: Vec::new()` | Render "no evidence terms" in decision detail | Decision based solely on prior + loss matrix; card still renderable |
| Empty drift history file | 0-byte file or missing file | Render "No drift history available" | Trend chart empty; no crash |
| Uncalibrated conformal guard | `is_calibrated() = false` (< 2 scores) | Render "Uncalibrated (N/2 scores)" | No quantile threshold displayed; suppress conformal metrics |
| Conformal quantile `None` | `conformal_quantile()` returns `None` | Render threshold as "--" or "N/A" | Guard admits all actions (threshold = infinity) |
| No hardened join row cap | `hardened_join_row_cap = None` | Display as "unlimited" | All joins admitted; no cap enforcement indicator |
| Placeholder RaptorQ envelope | `source_hash = "blake3:placeholder"`, `k = 0`, `repair_symbols = 0` | Render as "placeholder" status with distinct styling | Not corruption or missing data; intentional sentinel |
| Placeholder scrub status | `last_ok_unix_ms = 0`, `status = "ok"` | Render scrub time as "never scrubbed" | Misleading "ok" status must be contextualized with zero timestamp |
| Missing `artifact_path` in FailureDigest | `artifact_path: None` | Omit artifact link from failure digest display | Artifact not available for navigation |
| Missing `packet_id` in report | `packet_id: None` | Display as "unknown" | Packet grouping and navigation degraded |
| Empty `reasons` in gate result | `reasons: Vec::new()` | Display "no failure reasons reported" | Gate failed but no explanation available |
| Coverage alert suppressed | `total_count < 100` | No alert badge regardless of coverage rate | By design: insufficient data for alert |

---

## 6. Strict Mode Policy Matrix

| FTUI Surface | Strict Mode Behavior | Mechanism | Display Artifact |
|---|---|---|---|
| **Conformance Dashboard** | | | |
| Gate status badge | Strict gate uses `critical_drift_budget = 0` (zero tolerance) | `PacketGateResult::strict_failed` count | Red/green badge with strict failure count |
| Fixture pass/fail ratio | All critical drift failures block the gate | `CaseResult` with `mode: Strict` | Per-case PASS/FAIL with mode annotation |
| Drift summary | Critical drift records flagged as gate-blocking | `DriftLevel::Critical` count in `DriftSummary` | Critical count highlighted; zero budget displayed |
| **Decision Dashboard** | | | |
| Galaxy-brain cards | Cards carry `mode: Strict` badge | `DecisionRecord::mode = RuntimeMode::Strict` | "STRICT" mode badge on each card |
| Policy provenance | `fail_closed_unknown_features = true` displayed | `RuntimePolicy::strict()` fields | "Unknown features: REJECT (fail-closed)" |
| Join row cap | "N/A" (no cap in strict mode) | `hardened_join_row_cap = None` | "Join row cap: N/A (strict mode)" |
| Unknown feature decisions | Always REJECT regardless of posterior | `decide_unknown_feature()` override | Decision card shows "REJECT (fail-closed override)" |
| **Forensics Dashboard** | | | |
| Failure digest mode | Strict-mode failures flagged distinctly | `FailureDigest::mode = Strict` | "[Strict]" badge on each failure entry |
| Gate failure reasons | Strict budget violations listed | `PacketGateResult::reasons` | "Strict: N critical drift records exceed budget 0" |
| **CI Pipeline View** | | | |
| G6 Conformance gate | Strict gate evaluation applies | `evaluate_parity_gate()` with strict config | Per-packet strict pass/fail in G6 detail |
| **Conformal Guard Panel** | | | |
| Guard behavior | Same as hardened (mode-independent) | `ConformalGuard::evaluate()` | No strict-specific conformal modification |
| **Drift Trend Chart** | | | |
| Strict failure overlay | `strict_failed` count per data point | `PacketDriftHistoryEntry::strict_failed` | Strict failure trend line overlaid on pass rate |

**Strict mode FTUI invariants:**
- SINV-FTUI-FAIL-CLOSED-VISIBLE: When `fail_closed_unknown_features = true`, every unknown-feature decision card must display "fail-closed override" annotation.
- SINV-FTUI-STRICT-BADGE: Every `DecisionRecord` with `mode: Strict` must render a visible "STRICT" mode indicator.
- SINV-FTUI-ZERO-BUDGET: Strict drift budget (critical = 0) must be explicitly displayed in gate detail views.

---

## 7. Hardened Mode Policy Matrix

| FTUI Surface | Hardened Mode Behavior | Mechanism | Display Artifact |
|---|---|---|---|
| **Conformance Dashboard** | | | |
| Gate status badge | Hardened gate uses `divergence_budget_percent` tolerance | `PacketGateResult::hardened_failed` count | Amber/green badge with hardened failure count vs budget |
| Fixture pass/fail ratio | Non-critical drift within budget does not block gate | `CaseResult` with `mode: Hardened` | Per-case PASS/FAIL with mode annotation |
| Drift summary | Allowlisted categories shown separately | `HardenedGateConfig::allowlisted_divergence_categories` | "Allowlisted divergences: {categories}" |
| **Decision Dashboard** | | | |
| Galaxy-brain cards | Cards carry `mode: Hardened` badge | `DecisionRecord::mode = RuntimeMode::Hardened` | "HARDENED" mode badge on each card |
| Policy provenance | `fail_closed_unknown_features = false` + join cap displayed | `RuntimePolicy::hardened()` fields | "Unknown features: Bayesian (repair permitted)" |
| Join row cap | Explicit cap value displayed | `hardened_join_row_cap = Some(N)` | "Join row cap: N rows" |
| Join admission decisions | Over-cap forced to REPAIR with override indicator | `decide_join_admission()` cap override | Decision card shows "REPAIR (cap override: estimated > cap)" |
| Unknown feature decisions | Bayesian argmin (may Allow, Reject, or Repair) | `decide_unknown_feature()` no override | Decision card shows pure Bayesian result |
| **Forensics Dashboard** | | | |
| Failure digest mode | Hardened-mode failures flagged distinctly | `FailureDigest::mode = Hardened` | "[Hardened]" badge on each failure entry |
| Gate failure reasons | Hardened budget violations with allowlist context | `PacketGateResult::reasons` | "Hardened: N divergences exceed M% budget" |
| **CI Pipeline View** | | | |
| G6 Conformance gate | Hardened gate evaluation applies | `evaluate_parity_gate()` with hardened config | Per-packet hardened pass/fail in G6 detail |
| **Conformal Guard Panel** | | | |
| Guard behavior | Same as strict (mode-independent) | `ConformalGuard::evaluate()` | No hardened-specific conformal modification |
| **Drift Trend Chart** | | | |
| Hardened failure overlay | `hardened_failed` count per data point | `PacketDriftHistoryEntry::hardened_failed` | Hardened failure trend line overlaid on pass rate |

**Hardened mode FTUI invariants:**
- HINV-FTUI-CAP-VISIBLE: When `hardened_join_row_cap = Some(N)`, the cap value must be displayed in the policy provenance panel.
- HINV-FTUI-REPAIR-OVERRIDE: When a join admission decision is forced to `Repair` by cap override, the decision card must annotate "cap override".
- HINV-FTUI-BUDGET-DISPLAY: Hardened divergence budget percentage must be displayed alongside actual divergence percentage in gate detail views.

---

## 8. Strict vs Hardened Comparison Matrix

| Property | Strict (FTUI Display) | Hardened (FTUI Display) | Visual Delta |
|---|---|---|---|
| Mode badge color | Red/high-severity indicator | Amber/medium-severity indicator | Distinct color coding per mode |
| `fail_closed_unknown_features` | "REJECT (fail-closed)" | "Bayesian (repair permitted)" | Text and color differ |
| `hardened_join_row_cap` | "N/A" | "{N} rows" or "unlimited" | Presence/absence of cap value |
| Unknown feature decision card | Always shows "Reject" with override annotation | Shows Bayesian argmin result (Allow/Reject/Repair) | Override annotation present only in strict |
| Over-cap join decision | N/A (no cap in strict mode) | "REPAIR (cap override)" | Only rendered in hardened mode |
| Critical drift budget | "0 (zero tolerance)" | "N% (with allowlist)" | Numeric budget + allowlist context |
| Gate failure display | "Strict: critical drift count > 0" | "Hardened: divergence % > budget %" | Different failure criteria text |
| Drift trend strict_failed line | Prominent (primary failure metric) | Secondary (complementary to hardened_failed) | Visual weight differs by mode |
| Drift trend hardened_failed line | Secondary (may be N/A) | Prominent (primary failure metric) | Visual weight differs by mode |
| Evidence ledger per mode | Count of strict-mode records | Count of hardened-mode records | Per-mode record count in policy panel |
| Conformal guard | Identical behavior | Identical behavior | No visual difference |
| Outcome bridge | Identical mapping | Identical mapping | No visual difference |
| Default policy indicator | "DEFAULT" tag (strict is Default impl) | No default tag | Tag present only on strict |

---

## 9. Performance Sentinels

### 9.1 Render Time Budgets

| Operation | Budget | Enforcement | Degradation Path |
|---|---|---|---|
| Frame render (full dashboard) | 16ms (60 FPS target) | Frame timing measurement | Drop to 30 FPS (33ms) on complex views |
| Single panel render | 4ms | Per-panel timing | Skip panel animation; render static |
| Drift trend chart render | 8ms | Chart-specific timing | Downsample data points to fit budget |
| Forensic event list render | 4ms (visible window only) | Virtual scrolling window | Reduce visible window size |
| Galaxy-brain card render | 1ms per card | Per-card timing | Batch rendering with lazy layout |

### 9.2 Data Volume Limits

| Data Source | Soft Limit | Hard Limit | Over-Limit Behavior |
|---|---|---|---|
| `drift_history.jsonl` lines | 10,000 lines | 100,000 lines | Soft: warn "large history file"; Hard: read only last 100K lines |
| `ForensicLog::events` | 1,000 events | 50,000 events | Soft: enable virtual scrolling; Hard: display last 50K events with "truncated" warning |
| `parity_mismatch_corpus.json` entries | 100 entries | 10,000 entries | Soft: paginate at 100/page; Hard: truncate with "showing 100 of N" |
| `EvidenceLedger::records` | 500 records | 10,000 records | Soft: paginate; Hard: display last 10K with "truncated" indicator |
| `DecisionRecord::evidence` terms | 50 terms | 1,000 terms | Soft: display 50 with "and N more"; Hard: cap at 50 |
| Packet count | 11 packets | 100 packets | Soft: scrollable list; Hard: paginate at 50/page |
| `CiGateResult::errors` per gate | 10 errors | 100 errors | Soft: show all; Hard: truncate with "and N more errors" |

### 9.3 Virtual Scrolling Thresholds

| List Type | Threshold | Visible Window | Pre-fetch Buffer |
|---|---|---|---|
| Forensic event timeline | >1,000 events | 50 events | 25 events above/below |
| Mismatch corpus | >100 entries | 25 entries | 10 entries above/below |
| Evidence ledger records | >500 records | 50 records | 25 records above/below |
| Drift history entries | >1,000 entries | 100 entries | 50 entries above/below |
| Failure digest list | >50 digests | 20 digests | 10 digests above/below |

### 9.4 Artifact Parse Budgets

| Artifact Type | Parse Budget | Size Expectation | Over-Budget Path |
|---|---|---|---|
| `parity_report.json` | 50ms | <100 KB | Warn "slow parse"; cache parsed result |
| `parity_gate_result.json` | 10ms | <10 KB | Cache parsed result |
| `parity_mismatch_corpus.json` | 100ms | <1 MB | Stream-parse if >1 MB |
| `parity_report.raptorq.json` | 50ms | <500 KB | Cache parsed result |
| `parity_report.decode_proof.json` | 10ms | <10 KB | Cache parsed result |
| `drift_history.jsonl` (full) | 500ms | <10 MB | Incremental read by seek position |

---

## 10. Compatibility Risks

### 10.1 Frankentui Crate Dependency Risk

| Risk ID | Description | Severity | Mitigation |
|---|---|---|---|
| FT-CR-01 | `frankentui` crate does not exist; no published API, no source code | Critical | Design FTUI against abstract `FtuiWidget` trait; defer concrete backend binding |
| FT-CR-02 | Terminal backend choice (crossterm/termion/termwiz) undecided | Medium | Use trait-based terminal abstraction; allow backend swap without API change |
| FT-CR-03 | `frankentui` widget model may not support virtual scrolling | Medium | Implement custom paginator (`FtuiPaginator`) independent of widget framework |
| FT-CR-04 | `frankentui` event loop model (sync vs async) unknown | High | Design FTUI with synchronous event loop (consistent with fp-runtime); plan async adapter if needed |

### 10.2 Feature Flag Interactions

| Risk ID | Description | Severity | Mitigation |
|---|---|---|---|
| FT-CR-05 | `asupersync` feature flag in `fp-runtime` gates `outcome_to_action()` | Medium | All asupersync-dependent FTUI panels must use `#[cfg(feature = "asupersync")]` conditional compilation |
| FT-CR-06 | FTUI must render identically with/without asupersync enabled | Medium | Integration tests under both feature configurations |
| FT-CR-07 | Adding asupersync panels must not break non-asupersync builds | Low | Feature-gated module separation; no unconditional imports |

### 10.3 Serialization Contract Coupling

| Risk ID | Description | Severity | Mitigation |
|---|---|---|---|
| FT-CR-08 | FTUI depends on `Serialize`/`Deserialize` stability of all consumed types | High | Pin serde format contracts; schema validation against `artifacts/schemas/*.schema.json` |
| FT-CR-09 | `RuntimePolicy` does not implement `Serialize` (no serde derives) | Low | FTUI accesses `RuntimePolicy` via direct struct field access, not deserialization |
| FT-CR-10 | Artifact schema evolution may break FTUI deserialization | Medium | Version-aware deserialization with fallback to previous schema version |
| FT-CR-11 | `serde(rename_all = "snake_case")` on enums means FTUI must match exact JSON string representations | Low | Use `serde_json::from_str` (inherits rename rules automatically) |

### 10.4 Version Coupling

| Aspect | Current State | Risk |
|---|---|---|
| `fp-runtime` dependency | Workspace crate; always in-sync | Low: simultaneous version evolution |
| `fp-conformance` dependency | Workspace crate; always in-sync | Low: simultaneous version evolution |
| `frankentui` dependency | Not yet created; external crate | High: API stability unknown |
| `serde` / `serde_json` | Stable ecosystem crates | Low: backward-compatible evolution |
| Terminal backend crate | Not yet chosen | Medium: API surface varies significantly between crossterm/termion/termwiz |

### 10.5 Data Model Assumptions

| Risk ID | Description | Severity | Mitigation |
|---|---|---|---|
| FT-CR-12 | 9-gate CI pipeline assumed complete; `CiGate` enum may gain variants | Low | Exhaustive match on `CiGate` triggers compile error on new variants (correct) |
| FT-CR-13 | 10-variant `ForensicEventKind` assumed complete; may gain variants | Low | Exhaustive match triggers compile error on new variants (correct) |
| FT-CR-14 | `FP-P2C-NNN` naming convention assumed for packet IDs | Low | Regex-based parsing with fallback to raw string display |
| FT-CR-15 | `drift_history.jsonl` assumed append-only and never truncated | Medium | FTUI should handle file shrinkage gracefully (re-read from beginning) |

---

## 11. Machine-Checkable Invariant Summary

| Invariant ID | Statement | Checkable By | Status |
|---|---|---|---|
| INV-FTUI-READ-ONLY | FTUI never writes to artifact files, drift history, or evidence ledger | Code review; no `fs::write` / `File::create` in FTUI crate | Planned |
| INV-FTUI-NO-PANIC-ON-MISSING | Missing artifact files produce panel-level degradation, not panic | Integration test: remove artifact file, verify FTUI renders | Planned |
| INV-FTUI-NO-PANIC-ON-PARSE | Malformed artifact JSON produces panel-level error, not panic | Integration test: corrupt artifact file, verify FTUI renders | Planned |
| INV-FTUI-JSONL-SKIP-MALFORMED | Malformed `drift_history.jsonl` lines are skipped, not fatal | Unit test: inject malformed lines, verify skip count | Planned |
| INV-FTUI-JSONL-TRAILING-TOLERANT | Trailing incomplete line in JSONL is tolerated | Unit test: truncate last line, verify no error | Planned |
| INV-FTUI-CLOCK-SKEW-DISPLAY | `ts_unix_ms = 0` renders as "unknown", not "1970-01-01" | Unit test: render record with `ts_unix_ms = 0` | Planned |
| INV-FTUI-UNCALIBRATED-DISPLAY | Uncalibrated conformal guard displays "Uncalibrated (N/2 scores)" | Unit test: `is_calibrated() = false` path | Planned |
| INV-FTUI-PLACEHOLDER-DISPLAY | RaptorQ placeholder envelopes render as "placeholder" status | Unit test: render `RaptorQEnvelope::placeholder()` | Planned |
| INV-FTUI-STRICT-BADGE | All strict-mode decision records display "STRICT" mode indicator | Integration test: ledger with strict records | Planned |
| INV-FTUI-HARDENED-BADGE | All hardened-mode decision records display "HARDENED" mode indicator | Integration test: ledger with hardened records | Planned |
| INV-FTUI-FAIL-CLOSED-ANNOTATION | Strict fail-closed override annotated on decision cards | Integration test: `decide_unknown_feature()` in strict mode | Planned |
| INV-FTUI-CAP-OVERRIDE-ANNOTATION | Hardened cap override annotated on join admission cards | Integration test: `decide_join_admission()` over cap | Planned |
| INV-FTUI-VIRTUAL-SCROLL | Lists exceeding threshold use virtual scrolling, not full render | Performance test: inject >10K forensic events, verify render time <16ms | Planned |
| INV-FTUI-EVIDENCE-CAP | Evidence term display capped at 50 with "and N more" indicator | Unit test: record with 100 evidence terms | Planned |
| INV-FTUI-MISMATCH-TRUNCATE | Mismatch summary truncated to 200 characters with ellipsis | Unit test: `mismatch_summary` > 200 chars | Planned |
| INV-FTUI-FEATURE-ISOLATED | Asupersync panels compile-gated with `#[cfg(feature = "asupersync")]` | Compiler; build without asupersync feature | Planned |
| INV-FTUI-FEATURE-NO-RUNTIME-ERROR | Missing asupersync feature produces panel absence, not runtime error | Integration test: build without feature, verify no error on launch | Planned |
| INV-FTUI-GATE-ORDER | CI gates rendered in `CiGate::order()` sequence (G1=1 through G8=9) | Unit test: verify render order matches `pipeline()` order | Planned |
| INV-FTUI-FORENSIC-FILTER | Forensic events filterable by all 10 `ForensicEventKind` variants | Integration test: toggle each filter, verify correct event set | Planned |
| INV-FTUI-POSTERIOR-PRECISION | Posterior values rendered with 4-decimal fixed precision | Unit test: extreme posteriors (1e-15, 1-1e-15) render as "0.0000", "1.0000" | Planned |
| INV-FTUI-TERMINAL-MIN-SIZE | Below minimum terminal dimensions, FTUI displays size message, not garbled output | Integration test: set terminal to 20x5, verify message | Planned |
| INV-FTUI-REFRESH-REREAD | Manual refresh re-reads artifact files to capture updates | Integration test: modify artifact on disk, refresh, verify new data displayed | Planned |
| INV-FTUI-NO-FILE-LOCKS | FTUI does not hold file locks on artifact files during read | Code review; use `fs::read_to_string()` not persistent file handles | Planned |
| INV-FTUI-PACKET-SORT | Packets sorted by `FP-P2C-NNN` numeric suffix | Unit test: verify sort order for FP-P2C-001 through FP-P2C-011 | Planned |
| INV-FTUI-REPLAY-VERBATIM | Clipboard copy of replay command is verbatim `FailureDigest::replay_command` | Unit test: compare clipboard content to source field | Planned |
| INV-FTUI-COVERAGE-ALERT-BADGE | Coverage alert badge visible when `coverage_alert() = true` | Unit test: inject 100+ decisions with low coverage, verify badge | Planned |
| INV-FTUI-ALL-GREEN-SUMMARY | When `E2eReport::is_green()`, display "ALL GREEN: N/N fixtures passed" | Unit test: green report renders summary-only view | Planned |
| INV-FTUI-REPORT-HASH-DEDUP | Identical `report_hash` values in drift history indicate reruns | Display: show "rerun" indicator when consecutive entries share hash | Planned |
| INV-FTUI-NO-UNSAFE | FTUI crate uses `#![forbid(unsafe_code)]` | Compiler | Planned |

---

## 12. Appendix: ForensicEventKind Display Mapping

| Variant | Icon/Prefix | Color Class | Filter Label |
|---|---|---|---|
| `SuiteStart` | `[>>]` | Informational (blue) | "Suite Start" |
| `SuiteEnd` | `[<<]` | Informational (blue) | "Suite End" |
| `PacketStart` | `[>]` | Neutral (white) | "Packet Start" |
| `PacketEnd` | `[<]` | Neutral (white) | "Packet End" |
| `CaseStart` | `[.]` | Neutral (dim) | "Case Start" |
| `CaseEnd` | `[*]` | Pass=green / Fail=red | "Case End" |
| `ArtifactWritten` | `[W]` | Success (green) | "Artifact Written" |
| `GateEvaluated` | `[G]` | Pass=green / Fail=red | "Gate Evaluated" |
| `DriftHistoryAppended` | `[D]` | Informational (cyan) | "Drift Appended" |
| `Error` | `[!]` | Error (bright red, bold) | "Error" |

---

## 13. Appendix: CiGate Display Mapping

| Gate | Order | Label | Status Symbol (Pass) | Status Symbol (Fail) |
|---|---|---|---|---|
| `G1Compile` | 1 | "G1: Compile + Format" | `[PASS]` (green) | `[FAIL]` (red) |
| `G2Lint` | 2 | "G2: Lint (Clippy)" | `[PASS]` (green) | `[FAIL]` (red) |
| `G3Unit` | 3 | "G3: Unit Tests" | `[PASS]` (green) | `[FAIL]` (red) |
| `G4Property` | 4 | "G4: Property Tests" | `[PASS]` (green) | `[FAIL]` (red) |
| `G4_5Fuzz` | 5 | "G4.5: Fuzz Regression" | `[PASS]` (green) | `[FAIL]` (red) |
| `G5Integration` | 6 | "G5: Integration Tests" | `[PASS]` (green) | `[FAIL]` (red) |
| `G6Conformance` | 7 | "G6: Conformance" | `[PASS]` (green) | `[FAIL]` (red) |
| `G7Coverage` | 8 | "G7: Coverage Floor" | `[PASS]` (green) | `[FAIL]` (red) |
| `G8E2e` | 9 | "G8: E2E Pipeline" | `[PASS]` (green) | `[FAIL]` (red) |

**Pipeline header format:**
- All pass: `CI PIPELINE: ALL GREEN (N/N gates passed in {elapsed}ms)`
- Any fail: `CI PIPELINE: FAILED (N/M gates passed)` with `first_failure` gate highlighted

---

## 14. Appendix: Dashboard Navigation Taxonomy

| Dashboard | Key (planned) | Panel Count | Primary Data Source |
|---|---|---|---|
| Conformance | `1` or `C` | 4 (packet list, gate status, drift trend, detail) | Artifact files + drift history |
| Decision | `2` or `D` | 4 (evidence ledger, galaxy-brain cards, conformal guard, policy) | `EvidenceLedger` + `ConformalGuard` + `RuntimePolicy` |
| Forensics | `3` or `F` | 3 (failure digests, forensic timeline, artifact index) | `FailureForensicsReport` + `ForensicLog` |
| CI Pipeline | `4` or `P` | 2 (pipeline bar, gate detail) | `CiPipelineResult` |
| Global | `?` or `h` | 1 (help overlay) | Static |

**Navigation invariant:** Dashboard switching must not lose scroll position or filter state of the previously active dashboard within the same session.

---

## Changelog

- **bd-2gi.28.2** (2026-02-14): Initial FRANKENTUI contract table. Defines input/output/error/null contracts, strict and hardened policy matrices with per-surface breakdown, strict-vs-hardened comparison, performance sentinels (render budgets, data volume limits, virtual scrolling thresholds, artifact parse budgets), compatibility risks (frankentui crate dependency, feature flag interactions, serialization coupling, version coupling, data model assumptions), and 29 machine-checkable invariants. Appendices cover ForensicEventKind display mapping, CiGate display mapping, and dashboard navigation taxonomy.
