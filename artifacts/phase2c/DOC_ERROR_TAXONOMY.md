# DOC-PASS-07: Error Taxonomy, Failure Modes, and Recovery Semantics

**Bead:** `bd-2gi.23.8`  
**Date:** 2026-02-15  
**Status:** Complete  
**Primary scope:** user-visible error semantics, strict/hardened behavior splits, and fail/recovery contracts across runtime + ASUPERSYNC + conformance layers.

---

## 1. Executive Summary

FrankenPandas currently exposes an explicit, typed error surface in crate APIs (`Result<T, Error>` with `thiserror` enums), a policy-decision surface (`DecisionAction` + `EvidenceLedger`) for strict/hardened runtime governance, and a conformance/oracle drift surface (`DriftRecord` + gate enforcement). The dominant doctrine is:

1. Strict mode fails closed on unknown compatibility surfaces.
2. Hardened mode allows bounded repair paths with explicit evidence.
3. Differential drift is taxonomy-classified (not opaque message-only failures).
4. Recovery paths are deterministic and replayable where currently implemented.

The tables below are source-anchored and intended to be machine-auditable.

---

## 2. Error Channel Model (User vs Internal)

| Channel ID | Audience | Contract | Primary carriers | Source anchors |
|---|---|---|---|---|
| `U-API` | library callers | Typed `Result<T, Error>` return values | `TypeError`, `ColumnError`, `FrameError`, `IndexError`, `JoinError`, `GroupByError`, `ExprError`, `IoError` | `crates/fp-types/src/lib.rs:113`, `crates/fp-columnar/src/lib.rs:473`, `crates/fp-frame/src/lib.rs:15`, `crates/fp-index/src/lib.rs:439`, `crates/fp-join/src/lib.rs:27`, `crates/fp-groupby/src/lib.rs:25`, `crates/fp-expr/src/lib.rs:46`, `crates/fp-io/src/lib.rs:14` |
| `U-CONFORMANCE` | CI/operators | run/gate failures with explicit parity context | `HarnessError`, `PacketGateResult.pass=false`, process exit code | `crates/fp-conformance/src/lib.rs:454`, `crates/fp-conformance/src/lib.rs:388` |
| `U-ORACLE` | CI/operators | fail-closed oracle adapter errors | `OracleError` JSON payload + non-zero exit | `crates/fp-conformance/oracle/pandas_oracle.py:22`, `crates/fp-conformance/oracle/pandas_oracle.py:307` |
| `I-POLICY` | runtime forensics | structured decision evidence, not a Rust `Error` enum | `DecisionRecord`, `DecisionMetrics`, `EvidenceLedger`, `DecisionAction` | `crates/fp-runtime/src/lib.rs:73`, `crates/fp-runtime/src/lib.rs:82`, `crates/fp-runtime/src/lib.rs:126`, `crates/fp-runtime/src/lib.rs:20` |
| `I-ASUPERSYNC` | runtime forensics/operators | transport/integrity/recovery error + outcome channels | `AsupersyncError`, `TransferStatus`, `RecoveryOutcome`, `RecoveryReport` | `crates/fp-runtime/src/asupersync/error.rs:6`, `crates/fp-runtime/src/asupersync/transport.rs:13`, `crates/fp-runtime/src/asupersync/recovery.rs:11`, `crates/fp-runtime/src/asupersync/recovery.rs:25` |
| `I-DRIFT` | parity governance | structured mismatch taxonomy | `DriftLevel`, `ComparisonCategory`, `DriftRecord`, `DifferentialResult` | `crates/fp-conformance/src/lib.rs:265`, `crates/fp-conformance/src/lib.rs:277`, `crates/fp-conformance/src/lib.rs:292`, `crates/fp-conformance/src/lib.rs:301` |

---

## 3. Crate-Level Error Taxonomy

### 3.1 Validation / Coercion / Data-Shape Errors

| Domain | Type | Variant classes | User-visible semantics | Recovery semantics |
|---|---|---|---|---|
| Type algebra | `TypeError` | incompatible dtypes, invalid cast, lossy float->int, invalid bool coercion, non-numeric, missing value | Operation rejected with exact coercion reason | Callers must normalize dtype/value domain before operation; no silent coercion fallback |
| Column kernels | `ColumnError` | length mismatch, wrapped `TypeError` | Binary/reindex ops reject invalid shape/type | Resolve upstream shape/type mismatch; rerun |
| Index alignment | `IndexError` | invalid alignment vector shape | Alignment plan rejected before materialization | Recompute plan via `align_*` helpers; fail closed on malformed vectors |
| Frame operations | `FrameError` | index/column length mismatch, duplicate index unsupported (strict slice), compatibility rejected, wrapped column/index errors | Explicit construction/alignment rejection | In strict mode, duplicate index branch is terminal; in hardened, policy may permit continuation |
| Expression planner | `ExprError` | unknown series ref, unanchored literal, wrapped frame errors | Planner refuses unresolved or index-free literals | Bind missing series/index anchor; rerun |
| IO | `IoError` | missing headers, JSON format error, wrapped csv/json/io/utf8/column/frame errors | Parsing/serialization failures returned directly | Fix malformed source payload; retry ingestion |

Source anchors: `crates/fp-types/src/lib.rs:113`, `crates/fp-columnar/src/lib.rs:473`, `crates/fp-index/src/lib.rs:439`, `crates/fp-frame/src/lib.rs:15`, `crates/fp-expr/src/lib.rs:46`, `crates/fp-io/src/lib.rs:14`.

### 3.2 Runtime Compatibility / Policy Errors

| Domain | Type | Variant classes | User-visible semantics | Recovery semantics |
|---|---|---|---|---|
| Runtime infra | `RuntimeError` | `ClockSkew` | Internal timestamp source invalid; decision timestamp defaults to `0` in current path | Investigate clock env; avoid depending on wallclock uniqueness for correctness |
| Policy issue classes | `IssueKind` | `UnknownFeature`, `MalformedInput`, `JoinCardinality`, `PolicyOverride` | Encodes why decision was taken | Drives evidence ledger + operator diagnosis |
| Policy action classes | `DecisionAction` | `Allow`, `Reject`, `Repair` | Governs admission control boundary | `Repair` routes to bounded recovery paths; `Reject` fails closed |

Source anchors: `crates/fp-runtime/src/lib.rs:262`, `crates/fp-runtime/src/lib.rs:28`, `crates/fp-runtime/src/lib.rs:20`, `crates/fp-runtime/src/lib.rs:311`.

### 3.3 ASUPERSYNC Recovery/Transport Errors

| Domain | Type | Variant classes | User-visible semantics | Recovery semantics |
|---|---|---|---|---|
| ASUPERSYNC integration | `AsupersyncError` | config invalid, capability denied, artifact missing, integrity mismatch, codec error, transport error, recovery exhausted | Explicit root-cause enum variant with domain context fields | Retry path is policy-governed; terminal on `RecoveryExhausted`/permanent failure |
| Transport state | `TransferStatus` | `Completed`, `RetryableFailure`, `PermanentFailure` | Transfer lifecycle state for reports | `RetryableFailure` can schedule retry; `PermanentFailure` rejects |
| Recovery classification | `RecoveryOutcome` | `Recovered`, `RetryScheduled`, `Rejected` | Canonical recovery decision state | Determined by transfer status + retry budget |

Source anchors: `crates/fp-runtime/src/asupersync/error.rs:6`, `crates/fp-runtime/src/asupersync/transport.rs:13`, `crates/fp-runtime/src/asupersync/recovery.rs:11`, `crates/fp-runtime/src/asupersync/recovery.rs:42`.

### 3.4 Conformance / Oracle / Drift Taxonomy

| Domain | Type | Variant classes | User-visible semantics | Recovery semantics |
|---|---|---|---|---|
| Harness execution | `HarnessError` | IO/JSON/YAML/frame wraps, fixture format, oracle unavailable/failed, raptorq error | Suite run returns structured failure with context | Fix fixture/oracle/runtime environment, rerun targeted suite |
| Drift severity | `DriftLevel` | `Critical`, `NonCritical`, `Informational` | Gate strictness maps to severity budgets | `Critical` blocks gates; non-critical/informational are budgeted/documented |
| Drift dimensions | `ComparisonCategory` | `Value`, `Type`, `Shape`, `Index`, `Nullness` | Mismatch is classified by semantic dimension | Enables deterministic triage and budget accounting |
| Oracle adapter | `OracleError` (python side) | import/setup errors, unsupported label/scalar kinds, unsupported operation | JSON error payload + non-zero exit (fail-closed in strict legacy mode) | Fix request or environment; no silent fallback unless explicitly allowed |

Source anchors: `crates/fp-conformance/src/lib.rs:454`, `crates/fp-conformance/src/lib.rs:265`, `crates/fp-conformance/src/lib.rs:277`, `crates/fp-conformance/oracle/pandas_oracle.py:22`, `crates/fp-conformance/oracle/pandas_oracle.py:57`, `crates/fp-conformance/oracle/pandas_oracle.py:77`, `crates/fp-conformance/oracle/pandas_oracle.py:290`.

---

## 4. Failure Mode Matrix (Trigger / Impact / Recovery)

| FM ID | Trigger | Impact | Detection surface | Recovery behavior |
|---|---|---|---|---|
| `FM-001` | Unknown feature decision in strict mode (`fail_closed_unknown_features=true`) | Deterministic rejection (fail-closed) | `DecisionRecord` + `DecisionAction::Reject` | Implement feature or reroute through explicitly bounded hardened flow |
| `FM-002` | Hardened join estimate exceeds configured cap | Forced `Repair` action at admission boundary | ledger record with `IssueKind::JoinCardinality` | bounded repair path; avoid unbounded join materialization |
| `FM-003` | Alignment vectors length mismatch | alignment plan invalid; operation aborted | `IndexError::InvalidAlignmentVectors` | rebuild plan via `align_*`; no partial materialization |
| `FM-004` | Series/DataFrame index length differs from column length | construction fails | `FrameError::LengthMismatch` | fix data construction before operation |
| `FM-005` | Duplicate labels in strict-path arithmetic slice | operation denied | `FrameError::DuplicateIndexUnsupported` | deduplicate/reindex input or run non-strict mode with evidence logging |
| `FM-006` | Invalid dtype coercion (e.g., non-integer float to int64) | coercion fails before kernel execution | `TypeError::*` | explicit cast or pre-clean data; no lossy implicit cast |
| `FM-007` | CSV headers missing | ingestion fails at parser setup | `IoError::MissingHeaders` | supply headers or correct source options |
| `FM-008` | ASUPERSYNC capability requirements not met | transport/remote path denied | `AsupersyncError::CapabilityDenied` | elevate capability set intentionally; otherwise remain fail-closed |
| `FM-009` | ASUPERSYNC integrity digest mismatch | artifact rejected | `AsupersyncError::IntegrityMismatch` | retry/recover with fresh payload; preserve mismatch evidence |
| `FM-010` | ASUPERSYNC repeated receive/verify failures beyond max attempts | recovery terminates | `AsupersyncError::RecoveryExhausted` | escalate to operator; artifact remains unrecovered |
| `FM-011` | Oracle strict legacy import failure | conformance oracle returns error, exits non-zero | `OracleError` payload + exit code `1` | fix legacy root/env; no silent fallback unless flag-enabled |
| `FM-012` | Unsupported operation requested from oracle adapter | fixture run fails closed | `OracleError("unsupported operation")` | update oracle mapping and fixture operation support |
| `FM-013` | Differential drift exceeds gate budgets | packet gate fails, CI can block | `PacketGateResult.pass=false` + drift summary | investigate drift record category/level, patch behavior or update allowed budgets |
| `FM-014` | System clock before UNIX epoch during decision timestamping | timestamp degrades to `0` | `RuntimeError::ClockSkew` (internal) + `ts_unix_ms=0` fallback | environment repair; semantics currently non-fatal but audit signal must be preserved |

Source anchors: `crates/fp-runtime/src/lib.rs:152`, `crates/fp-runtime/src/lib.rs:201`, `crates/fp-runtime/src/lib.rs:245`, `crates/fp-index/src/lib.rs:548`, `crates/fp-frame/src/lib.rs:36`, `crates/fp-frame/src/lib.rs:114`, `crates/fp-types/src/lib.rs:113`, `crates/fp-io/src/lib.rs:66`, `crates/fp-runtime/src/asupersync/mod.rs:21`, `crates/fp-runtime/src/asupersync/integrity.rs:33`, `crates/fp-runtime/src/asupersync/recovery.rs:96`, `crates/fp-conformance/oracle/pandas_oracle.py:57`, `crates/fp-conformance/oracle/pandas_oracle.py:290`, `crates/fp-conformance/src/lib.rs:388`, `crates/fp-runtime/src/lib.rs:311`.

---

## 5. Strict vs Hardened Semantics (Explicit)

| Surface | Strict behavior | Hardened behavior | Anchor |
|---|---|---|---|
| Unknown feature compatibility issue | Fail-closed override to `Reject` | Bayesian argmin can choose non-reject path (`fail_closed_unknown_features=false`) | `crates/fp-runtime/src/lib.rs:175`, `crates/fp-runtime/src/lib.rs:201`, `crates/fp-runtime/src/lib.rs:169` |
| Duplicate-index arithmetic in `Series::binary_op_with_policy` | Returns `FrameError::DuplicateIndexUnsupported` | Continues past duplicate warning branch (subject to policy decisions) | `crates/fp-frame/src/lib.rs:114`, `crates/fp-frame/src/lib.rs:120` |
| Join admission over cap | No forced cap override path | Explicit forced `Repair` when `estimated_rows > cap` | `crates/fp-runtime/src/lib.rs:245` |
| Oracle legacy import | `--strict-legacy` fails closed without system fallback | fallback allowed only with explicit opt-in flag (`--allow-system-pandas-fallback`) | `crates/fp-conformance/oracle/pandas_oracle.py:34`, `crates/fp-conformance/oracle/pandas_oracle.py:39`, `crates/fp-conformance/oracle/pandas_oracle.py:57` |
| Differential drift gating | Critical drift blocks strict budgets | Hardened divergence is budgeted/allowlisted | `crates/fp-conformance/src/lib.rs:265`, `crates/fp-conformance/src/lib.rs:487`, `crates/fp-conformance/src/lib.rs:620` |

---

## 6. Recovery Semantics

### 6.1 Policy-Level Recovery

`RuntimePolicy` emits `DecisionAction` for each compatibility issue; this is the first recovery boundary:

1. `Allow`: proceed with operation.
2. `Repair`: continue through bounded remediation path.
3. `Reject`: fail closed.

Anchor: `crates/fp-runtime/src/lib.rs:20`, `crates/fp-runtime/src/lib.rs:175`, `crates/fp-runtime/src/lib.rs:209`.

### 6.2 ASUPERSYNC Recovery Loop

`recover_once()` enforces bounded retry behavior:

1. Immediate config hard-fail when `max_attempts == 0`.
2. Retryable receive/verify failures loop while policy allows.
3. Terminal `RecoveryExhausted` when retry budget is spent.
4. Successful integrity verification yields `RecoveryReport { outcome: Recovered }`.

Anchors: `crates/fp-runtime/src/asupersync/recovery.rs:80`, `crates/fp-runtime/src/asupersync/recovery.rs:90`, `crates/fp-runtime/src/asupersync/recovery.rs:96`, `crates/fp-runtime/src/asupersync/recovery.rs:104`, `crates/fp-runtime/src/asupersync/recovery.rs:106`.

### 6.3 Conformance Recovery / Forensics

Drift handling is explicitly classified and budgeted rather than treated as opaque pass/fail:

1. Every mismatch is categorized (`value/type/shape/index/nullness`).
2. Every mismatch has severity (`critical/non_critical/informational`).
3. Gate evaluation can block progression and force remediation.

Anchors: `crates/fp-conformance/src/lib.rs:277`, `crates/fp-conformance/src/lib.rs:292`, `crates/fp-conformance/src/lib.rs:353`, `crates/fp-conformance/src/lib.rs:620`.

---

## 7. User-Facing Error Semantics

### 7.1 Library API Guarantees

1. API calls fail via typed enums; no exception unwinding model.
2. Error messages are deterministic and include key context fields (e.g., expected/observed lengths, artifact IDs).
3. Strict-mode compatibility rejections are explicit (`FrameError::CompatibilityRejected`, strict duplicate-index rejection path).

Anchors: `crates/fp-frame/src/lib.rs:20`, `crates/fp-frame/src/lib.rs:133`, `crates/fp-io/src/lib.rs:15`, `crates/fp-runtime/src/asupersync/error.rs:16`.

### 7.2 Operator/CI Guarantees

1. Oracle adapter emits normalized JSON with an `error` field on failure.
2. Packet parity gate emits structured gate results and can be used as hard CI blocker.
3. Drift records preserve category + severity + location + message for triage.

Anchors: `crates/fp-conformance/oracle/pandas_oracle.py:308`, `crates/fp-conformance/src/lib.rs:388`, `crates/fp-conformance/src/lib.rs:292`.

---

## 8. Replay and Logging Crosswalk

| Evidence path | Required or available fields | Replay contract |
|---|---|---|
| ASUPERSYNC test logs | `packet_id`, `case_id`, `mode`, `seed`, `trace_id`, `assertion_path`, `result`, `replay_cmd` | deterministic replay command per test case |
| Policy evidence ledger | mode/action/issue/prior/metrics/evidence terms | reconstruct admission decision rationale and expected-loss basis |
| Differential drift records | category/level/location/message | localize parity regressions by semantics class |
| Oracle adapter failures | `error` field + non-zero exit code | rerun identical payload against oracle script |

Anchors: `crates/fp-runtime/src/lib.rs:579`, `crates/fp-runtime/src/lib.rs:610`, `artifacts/phase2c/ASUPERSYNC_TEST_LOGGING_EVIDENCE.md:51`, `crates/fp-runtime/src/lib.rs:82`, `crates/fp-conformance/src/lib.rs:292`, `crates/fp-conformance/oracle/pandas_oracle.py:307`.

---

## 9. Open Risks and Follow-ups

1. `RuntimeError::ClockSkew` is currently absorbed into `ts_unix_ms=0` via `unwrap_or_default`, which weakens incident visibility unless downstream alerts key on zero timestamps.
2. `HarnessError::RaptorQ(String)` is stringly-typed and may hide root-cause structure needed for automated remediation.
3. Differential/adversarial error corpus minimization and replay workflow remain primarily in conformance artifacts; broader API-level replay tooling is still uneven across crates.

Anchors: `crates/fp-runtime/src/lib.rs:311`, `crates/fp-conformance/src/lib.rs:470`, `artifacts/phase2c/COMPAT_CLOSURE_CONTRACT_TABLE.md:46`.

---

## 10. Acceptance Criteria Check

| Requirement | Status | Evidence |
|---|---|---|
| Failure mode matrix with trigger/impact/recovery | complete | Section 4 |
| User-facing error semantics documented | complete | Sections 2 and 7 |
| Recovery and fail-closed behavior explicit | complete | Sections 5 and 6 |

