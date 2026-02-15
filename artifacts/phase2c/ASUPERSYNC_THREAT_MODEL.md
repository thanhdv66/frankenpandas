# ASUPERSYNC Security + Compatibility Threat Model

Bead: `bd-2gi.27.3` [ASUPERSYNC-C]
Subsystem: `fp-runtime` asupersync outcome bridge + artifact synchronization integration
Source anchors: `ASUPERSYNC_ANCHOR_MAP.md` (bd-2gi.27.1), `ASUPERSYNC_CONTRACT_TABLE.md` (bd-2gi.27.2)
Reference spec: `COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1.md` section 4 ("Asupersync Deep Integration")

Doctrine: **fail-closed on unknown/unsafe paths**. Every threat is evaluated against
the strict/hardened mode split. Where behavior is undefined or unspecified, the
fail-closed doctrine requires the system to reject, not guess.

---

## 1. Threat Surface Enumeration

This section catalogs every attack vector and failure mode arising from the
asupersync integration surface. Surfaces are numbered ATS-N (Asupersync Threat
Surface) to distinguish from the general TS-N surfaces in
`SECURITY_COMPATIBILITY_THREAT_MATRIX.md`.

### ATS-1: Outcome Bridge (`outcome_to_action`)

| Property | Value |
|---|---|
| Entry point | `fp-runtime/src/lib.rs` line 555, `#[cfg(feature = "asupersync")]` |
| Input class | `&asupersync::Outcome<T, E>` from external crate |
| Fail-closed behavior | Exhaustive match; `#[must_use]` on return |

**Threats:**
- **ATS-1.1 Variant evolution:** If asupersync adds a fifth `Outcome` variant (e.g.,
  `Timeout`), the exhaustive match will fail to compile. This is **correct fail-closed
  behavior**. However, a semver-compatible release could theoretically add `#[non_exhaustive]`
  to the enum, which would force a wildcard arm and silently drop new variant semantics.
- **ATS-1.2 Panic payload information leak:** `Outcome::Panicked(PanicPayload)` carries
  a panic message that may contain stack traces, memory addresses, or sensitive data.
  The bridge discards this payload entirely (maps to `Reject`). While safe from a
  leak perspective, this loses diagnostic information that supervision would need.
- **ATS-1.3 Error severity conflation:** `Outcome::Err(E)` universally maps to `Repair`
  regardless of whether the error is transient (network timeout) or permanent (schema
  corruption). A permanent error routed to `Repair` wastes retry budget and delays
  correct rejection.
- **ATS-1.4 Stateless bridge bypass:** The bridge is a pure function with no ledger
  integration. A caller could invoke `outcome_to_action()`, receive `Allow`, and
  proceed without ever recording the decision. The `#[must_use]` attribute only warns
  on unused return values; it does not enforce that a `DecisionRecord` is created.

### ATS-2: Feature Gate Boundary

| Property | Value |
|---|---|
| Entry point | `fp-runtime/Cargo.toml` feature `asupersync` |
| Input class | Cargo feature flag at compile time |
| Fail-closed behavior | Function does not exist when feature is disabled |

**Threats:**
- **ATS-2.1 Feature flag confusion:** If a downstream crate enables the `asupersync`
  feature but does not gate its own call sites, the dependency is pulled but may not
  be correctly integrated. No runtime check verifies that the feature is intentionally
  enabled by the end user.
- **ATS-2.2 Implicit feature unification:** Cargo feature unification means any crate
  in the dependency graph that enables `fp-runtime/asupersync` will enable it for ALL
  users of `fp-runtime` in the same build. This could pull asupersync's entire
  transitive dependency tree unexpectedly.
- **ATS-2.3 Default-features assumption:** `default-features = false` on the asupersync
  dependency assumes `Outcome<T, E>` is available without any feature flags. If
  asupersync restructures so that `Outcome` requires a feature, the build breaks.
  This is a correct fail-closed outcome but could be surprising.

### ATS-3: Clock and Timestamp (`now_unix_ms`)

| Property | Value |
|---|---|
| Entry point | `fp-runtime/src/lib.rs` line 264, `now_unix_ms()` |
| Input class | Ambient `SystemTime::now()` call |
| Fail-closed behavior | Returns `Err(RuntimeError::ClockSkew)` on pre-epoch clock |

**Threats:**
- **ATS-3.1 INV-NO-AMBIENT-AUTHORITY violation:** Direct `SystemTime::now()` call
  violates the ambient authority prohibition. Under lab-mode deterministic testing,
  this produces non-deterministic timestamps that poison trace reproducibility.
  **This is the single most significant invariant violation in fp-runtime.**
- **ATS-3.2 Silent zero-timestamp sentinel:** `decide()` catches `ClockSkew` via
  `unwrap_or_default()`, producing `ts_unix_ms = 0`. Records with timestamp 0 are
  indistinguishable from genuine UNIX epoch records. No warning is logged, no
  metric is incremented, and the decision proceeds as if nothing happened.
- **ATS-3.3 Monotonicity violation:** No guarantee that successive `now_unix_ms()`
  calls return monotonically increasing values. NTP adjustments, VM clock jitter,
  or container time-sync failures can produce backward time jumps, causing
  `DecisionRecord` sequences with non-monotonic timestamps. The evidence ledger
  has no invariant check for timestamp ordering.
- **ATS-3.4 Lab mode non-determinism:** Under `LabRuntime`, all time must flow
  through `Cx` virtual clocks. Direct `SystemTime::now()` in `now_unix_ms()` will
  produce real wall-clock time that varies across runs, defeating the purpose of
  deterministic testing entirely.

### ATS-4: RaptorQ Integrity

| Property | Value |
|---|---|
| Entry point | `RaptorQEnvelope::placeholder()`, planned `RaptorQSenderBuilder`/`RaptorQReceiverBuilder` |
| Input class | Artifact metadata, repair symbols, decode proofs |
| Fail-closed behavior | None; placeholder metadata is trusted structurally |

**Threats:**
- **ATS-4.1 Placeholder misinterpretation:** `RaptorQEnvelope::placeholder()` creates
  envelopes with `source_hash: "blake3:placeholder"` and `k: 0`. If a consumer
  fails to check for the sentinel hash, it may attempt decode with zero source
  blocks, producing undefined behavior in the RaptorQ decoder.
- **ATS-4.2 Unauthenticated symbols:** FrankenPandas does not use asupersync's
  `SecurityContext` or `AuthenticatedSymbol`. When actual RaptorQ encoding is wired,
  symbols will be accepted without epoch-scoped authentication tags. An adversary
  with write access to the symbol stream can inject forged repair symbols that
  decode to arbitrary data.
- **ATS-4.3 Decode proof replay:** `DecodeProof` contains `ts_unix_ms`, `reason`,
  `recovered_blocks`, and `proof_hash` but no artifact-scoped nonce or sequence
  number. A proof from one artifact can be replayed against another, falsely
  claiming successful decode.
- **ATS-4.4 Unbounded proof accumulation:** `decode_proofs: Vec<DecodeProof>` has
  no count limit. An adversary generating repeated decode attempts can grow this
  vector unboundedly, causing memory exhaustion.
- **ATS-4.5 Scrub status trust:** `ScrubStatus` with `status: "ok"` and
  `last_ok_unix_ms: 0` on placeholders claims the artifact is healthy despite
  never being scrubbed. This is a false-positive integrity signal.

### ATS-5: Cancellation and Budget Absence

| Property | Value |
|---|---|
| Entry point | All `fp-runtime` operations; all FrankenPandas kernel entry points |
| Input class | Long-running operations without cooperative cancellation |
| Fail-closed behavior | None; operations run to completion or OOM |

**Threats:**
- **ATS-5.1 Cancellation starvation:** Without `Cx::checkpoint()` calls, FrankenPandas
  operations cannot be cooperatively cancelled. A cancelled asupersync task hosting
  FrankenPandas logic (e.g., alignment on a 100M-row dataset) will block the
  cancellation signal until the operation completes, potentially holding resources
  for minutes.
- **ATS-5.2 Deadline bypass:** No budget enforcement exists. A malicious or
  misconfigured caller can trigger unbounded computation (e.g., Cartesian join
  explosion) that ignores any deadline set by a parent `Cx` scope.
- **ATS-5.3 Resource starvation via polling:** Without poll-quota tracking, a tight
  loop in alignment or groupby can starve other tasks on the same runtime thread,
  violating asupersync's cooperative scheduling contract.
- **ATS-5.4 Obligation leak risk:** If the evidence ledger `push()` were to be
  modeled as a two-phase obligation (reserve -> commit/abort), the current
  fire-and-forget implementation would constitute a permanent obligation leak
  under asupersync's linear resource discipline.

### ATS-6: Supervision and Recovery

| Property | Value |
|---|---|
| Entry point | Planned supervision integration; not yet implemented |
| Input class | `Outcome` variants from asupersync task execution |
| Fail-closed behavior | None; no supervision loop exists |

**Threats:**
- **ATS-6.1 Panicked task restart:** Without INV-SUPERVISION-MONOTONE enforcement,
  a naive supervision loop could restart a panicked task, violating the invariant
  that programming errors must not be retried. This produces infinite restart loops.
- **ATS-6.2 Transient error misclassification:** The `Outcome::Err -> Repair`
  mapping does not distinguish transient from permanent errors. A supervision
  policy that restarts all `Repair` actions will retry permanent failures until
  budget exhaustion.
- **ATS-6.3 Escalation gap:** No escalation path exists. When a component fails
  repeatedly, there is no mechanism to escalate to a parent supervisor, degrade
  functionality, or alert an operator.

---

## 2. Threat Matrix

| ID | Threat | Severity | Likelihood | Impact | Mitigation | Mode Behavior |
|---|---|---|---|---|---|---|
| ATS-1.1 | `Outcome` variant evolution breaks bridge | LOW | LOW | Compile error (correct) | Exhaustive match guards against silent breakage | Both: fail at compile time |
| ATS-1.2 | Panic payload discarded; diagnostic info lost | LOW | MEDIUM | Reduced debuggability | Extract panic message into `DecisionRecord.issue.detail` | Both: payload discarded |
| ATS-1.3 | Error severity conflation (all Err -> Repair) | MEDIUM | HIGH | Wasted retry budget on permanent errors | Add error classifier; map permanent errors to `Reject` | Both: unconditional `Repair` |
| ATS-1.4 | Decision not recorded after bridge call | MEDIUM | MEDIUM | Audit trail gap | Wrap bridge in method that mandates ledger push | Both: caller responsibility |
| ATS-2.1 | Feature flag confusion in downstream crates | LOW | LOW | Unexpected dependency pull | Document feature gate requirements in crate docs | Both: compile-time |
| ATS-2.2 | Cargo feature unification pulls asupersync globally | MEDIUM | LOW | Unwanted transitive deps in workspace | Use workspace-level feature management | Both: build-time |
| ATS-2.3 | `default-features = false` breaks if Outcome moves behind feature | LOW | LOW | Compile error (correct) | Pin exact asupersync features in Cargo.toml | Both: fail at compile time |
| ATS-3.1 | **INV-NO-AMBIENT-AUTHORITY violation (SystemTime::now)** | **HIGH** | **CERTAIN** | **Non-deterministic timestamps; lab mode broken** | **Route time through Cx clock trait** | Both: violated |
| ATS-3.2 | Silent zero-timestamp on clock skew | MEDIUM | LOW | Indistinguishable sentinel values | Log warning; use Option<u64> instead of 0 sentinel | Strict: should reject; Hardened: should log |
| ATS-3.3 | Non-monotonic timestamps in evidence ledger | MEDIUM | LOW | Corrupted temporal ordering | Add monotonicity check in `push()` | Strict: reject; Hardened: log and accept |
| ATS-3.4 | Lab mode time non-determinism | HIGH | CERTAIN (when lab enabled) | Irreproducible tests | Replace `SystemTime::now()` with `cx.time()` | Both: blocks lab mode |
| ATS-4.1 | Placeholder envelope misinterpreted as real | LOW | MEDIUM | Decode with zero source blocks | Validate sentinel hash before decode attempt | Strict: reject placeholder in decode path; Hardened: warn |
| ATS-4.2 | Unauthenticated RaptorQ symbols | HIGH | LOW (current); HIGH (when wired) | Forged artifact reconstruction | Integrate `SecurityContext` + `AuthenticatedSymbol` | Strict: reject unauthenticated; Hardened: accept with warning |
| ATS-4.3 | Decode proof cross-artifact replay | MEDIUM | LOW | False integrity attestation | Add `artifact_id` + nonce to `DecodeProof` | Both: enforce proof binding |
| ATS-4.4 | Unbounded decode_proofs vector | LOW | LOW | Memory exhaustion | Cap vector at 1000 entries | Both: enforce cap |
| ATS-4.5 | Placeholder scrub status claims "ok" | LOW | MEDIUM | False-positive integrity signal | Use `status: "placeholder"` instead of `"ok"` | Both: distinguish placeholder |
| ATS-5.1 | Cancellation starvation on long operations | HIGH | MEDIUM | Resource holding; blocked cleanup | Insert `cx.checkpoint()` at kernel boundaries | Both: blocks cancellation |
| ATS-5.2 | Deadline bypass via unbounded computation | HIGH | MEDIUM | Resource exhaustion DoS | Enforce `cx.scope_with_budget()` at entry points | Both: no enforcement |
| ATS-5.3 | Polling starvation of sibling tasks | MEDIUM | MEDIUM | Latency spikes for colocated work | Periodic `cx.checkpoint()` in tight loops | Both: no enforcement |
| ATS-5.4 | Evidence ledger obligation leak under Cx | LOW | LOW (planned) | Leaked obligation state | Model push as two-phase obligation | Both: not applicable yet |
| ATS-6.1 | Panicked task restart (violates monotone) | HIGH | LOW (planned) | Infinite restart loop | Enforce INV-SUPERVISION-MONOTONE in supervisor | Both: no supervisor yet |
| ATS-6.2 | Transient error misclassification | MEDIUM | MEDIUM | Wasted retry budget | Add error classification layer | Both: all errors treated same |
| ATS-6.3 | No escalation path for repeated failures | MEDIUM | LOW (planned) | Silent degradation | Implement supervision tree | Both: no escalation |

---

## 3. Compatibility Envelope

### 3.1 Version Coupling Risks

| Coupling Point | Current State | Risk Level | Fail-Closed Behavior |
|---|---|---|---|
| asupersync crate version | `"0.1.1"` (semver range `>=0.1.1, <0.2.0`) | MEDIUM | Patch bumps accepted; minor bump accepted if compatible |
| `Outcome` enum variant count | 4 variants, exhaustive match | LOW | Fifth variant = compile error (correct) |
| `Outcome` enum `#[non_exhaustive]` | Not currently marked | HIGH (if changed) | Would require wildcard arm; lose compile-time safety |
| `default-features = false` surface | Assumes `Outcome` in minimal build | LOW | Feature restructure = compile error (correct) |
| Transitive dependency versions | 11 transitive deps (see 6.1) | MEDIUM | Lockfile pins; `cargo audit` guards |
| Rust edition compatibility | fp-runtime uses `edition = "2024"` | LOW | asupersync must support edition 2024 |

### 3.2 Feature Flag Interaction Matrix

| Feature Combination | Status | Risk |
|---|---|---|
| `default = []` (asupersync disabled) | Primary configuration | None; all asupersync code elided |
| `asupersync` enabled alone | Supported | Pulls asupersync + 11 transitive deps |
| `asupersync` + future `lab` feature | Undefined | Would pull `LabRuntime`; untested combination |
| `asupersync` + future `transport` feature | Undefined | Would pull network stack; significant surface expansion |
| Workspace-wide feature unification | Risk: any crate enabling `fp-runtime/asupersync` enables it for all | Document in workspace Cargo.toml |

### 3.3 API Stability Guarantees

| API Surface | Stability | Breaking Change Risk |
|---|---|---|
| `outcome_to_action<T, E>(&Outcome<T, E>) -> DecisionAction` | Stable (signature change requires major bump) | LOW: generic, no concrete types exposed |
| `DecisionAction` enum variants | Stable (adding variant is breaking for downstream match) | LOW: adding `Retry` would be breaking |
| `RaptorQEnvelope` struct fields | Semi-stable (adding fields breaks deserialization without `#[serde(default)]`) | MEDIUM: planned fields for real encode |
| `EvidenceLedger::push()` signature | Stable | LOW |
| `RuntimePolicy::decide_*` methods | Stable (return type and side effects fixed) | LOW |
| `ConformalGuard` public interface | Semi-stable (internal score representation may change) | MEDIUM |

### 3.4 Cross-Crate Dependency Isolation

Currently, no other crate in the FrankenPandas workspace depends on asupersync
or on `fp-runtime`'s `asupersync` feature. This isolation is critical:

- `fp-types` -- no asupersync dependency (pure data types)
- `fp-index` -- no asupersync dependency (alignment primitives)
- `fp-columnar` -- no asupersync dependency (columnar kernels)
- `fp-frame` -- no asupersync dependency (DataFrame operations)
- `fp-groupby` -- no asupersync dependency (aggregation)
- `fp-join` -- no asupersync dependency (join operations)
- `fp-io` -- no asupersync dependency (CSV/JSON IO)
- `fp-expr` -- no asupersync dependency (expression planning)
- `fp-conformance` -- no asupersync dependency (conformance harness)

**Invariant (INV-FEATURE-NO-TRANSITIVE):** No workspace crate other than
`fp-runtime` may declare an asupersync dependency. Violation of this invariant
would create diamond dependency risks and capability leakage.

---

## 4. Fail-Closed Doctrine Application

For each threat, the fail-closed doctrine prescribes the strictest safe behavior.
Hardened mode may relax some strictures with evidence recording.

### 4.1 Outcome Bridge Fail-Closed Rules

| Scenario | Strict Mode | Hardened Mode |
|---|---|---|
| Unknown `Outcome` variant (if `#[non_exhaustive]`) | **REJECT** -- wildcard arm maps to `Reject` | **REJECT** -- same; cannot infer semantics |
| `Outcome::Err` with unknown error type | **REPAIR** -- current behavior; consider REJECT | **REPAIR** -- current behavior; record evidence |
| `Outcome::Panicked` | **REJECT** -- correct; must not restart | **REJECT** -- correct; must not restart |
| Bridge call without ledger recording | **VIOLATION** -- strict mode mandates audit trail | **WARNING** -- hardened mode should warn but not block |

### 4.2 Clock and Timestamp Fail-Closed Rules

| Scenario | Strict Mode | Hardened Mode |
|---|---|---|
| `SystemTime::now()` returns pre-epoch | **REJECT decision** -- do not proceed with zero timestamp | **LOG + PROCEED** -- record clock skew evidence, continue |
| Non-monotonic timestamp detected | **REJECT decision** -- temporal ordering violated | **LOG + PROCEED** -- record anomaly, accept decision |
| Lab mode with real clock | **ABORT** -- deterministic testing is impossible | **ABORT** -- lab mode is meaningless without virtual time |

### 4.3 RaptorQ Fail-Closed Rules

| Scenario | Strict Mode | Hardened Mode |
|---|---|---|
| Placeholder envelope in decode path | **REJECT** -- cannot decode from placeholder | **REJECT** -- same; no degraded decode |
| Unauthenticated symbol received | **REJECT** -- no auth tag = untrusted | **ACCEPT with evidence** -- record unauthenticated origin |
| Decode proof without artifact binding | **REJECT** -- unbound proof is meaningless | **ACCEPT with warning** -- log unbounded proof |
| Decode proof count exceeds cap | **REJECT** -- possible DoS | **TRUNCATE + LOG** -- drop oldest proofs |
| Verification failure on sidecar | **ABORT** -- integrity cannot be assured | **LOG + DEGRADE** -- mark artifact as unverified |

### 4.4 Cancellation Fail-Closed Rules

| Scenario | Strict Mode | Hardened Mode |
|---|---|---|
| Cancellation requested but no checkpoint available | **Operation must complete then fail** | **Same** -- no partial cancel possible |
| Budget exceeded (deadline) | **ABORT** -- return error immediately | **ABORT** -- budgets are non-negotiable |
| Budget exceeded (poll quota) | **ABORT** -- return error immediately | **ABORT** -- polling limits prevent starvation |
| Budget exceeded (cost quota) | **ABORT** -- return error immediately | **ABORT** -- cost limits prevent OOM |

### 4.5 Supervision Fail-Closed Rules

| Scenario | Strict Mode | Hardened Mode |
|---|---|---|
| `Panicked` outcome received by supervisor | **STOP** -- never restart panicked task | **STOP** -- same; INV-SUPERVISION-MONOTONE |
| `Cancelled` outcome received by supervisor | **STOP** -- external directive is authoritative | **STOP** -- same |
| `Err` outcome, restart budget exhausted | **ESCALATE** -- bubble to parent supervisor | **ESCALATE** -- same |
| `Err` outcome, restart budget available | **RESTART** -- if error is transient | **RESTART** -- with exponential backoff |
| No supervision configured | **CRASH** -- unsupervised failure is a bug | **LOG + CRASH** -- same |

---

## 5. INV-NO-AMBIENT-AUTHORITY Violation Analysis

### 5.1 The Violation

**Location:** `fp-runtime/src/lib.rs`, line 264-270:
```rust
fn now_unix_ms() -> Result<u64, RuntimeError> {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| RuntimeError::ClockSkew)?
        .as_millis();
    Ok(ms as u64)
}
```

**Invariant violated:** INV-NO-AMBIENT-AUTHORITY, as defined in FrankenSQLite spec
section 4.1.1:

> FrankenSQLite crates MUST NOT call ambient side-effect APIs directly. In
> particular, database crates MUST NOT call:
> - `std::time::SystemTime::now()` / `Instant::now()` (use Cx time/budget clocks)

### 5.2 Impact Assessment

| Dimension | Impact | Severity |
|---|---|---|
| **Deterministic testing** | `now_unix_ms()` produces real wall-clock time under `LabRuntime`. Two runs of the same test with the same seed will produce different `ts_unix_ms` values in `DecisionRecord`. Trace comparison, replay verification, and oracle assertions on timestamps all fail. | HIGH |
| **Capability security** | `now_unix_ms()` does not require any capability token. Any code path reaching `decide()` can observe wall-clock time without permission. Under the Cx model, time observation requires `TIME` capability. | MEDIUM |
| **Cancel-correctness** | `SystemTime::now()` is not a cancellation checkpoint. Under Cx integration, `cx.time()` would also serve as an implicit checkpoint. The current code misses this checkpoint opportunity. | LOW |
| **Budget enforcement** | The `decide()` function does not check budget deadlines. Under Cx integration, `cx.checkpoint()` inside `now_unix_ms()` would enforce deadline budgets. The current code bypasses this enforcement. | MEDIUM |
| **Audit compliance** | The FrankenSQLite spec requires a compile-time audit gate (`asupersync audit::ambient` pattern) to catch ambient authority violations in CI. FrankenPandas has no such gate. | MEDIUM |

### 5.3 Callers of the Violated Function

`now_unix_ms()` is called exactly once, in `decide()` at line 308:
```rust
ts_unix_ms: now_unix_ms().unwrap_or_default(),
```

`decide()` is called by:
- `RuntimePolicy::decide_unknown_feature()` (line 197)
- `RuntimePolicy::decide_join_admission()` (line 240)

These are the only two paths that produce timestamps. All `DecisionRecord`
instances in the evidence ledger carry potentially non-deterministic timestamps.

### 5.4 Remediation Path

**Phase 1 (minimal, no Cx dependency):** Inject a clock function.

```rust
// Replace ambient call with injectable clock
pub type ClockFn = fn() -> Result<u64, RuntimeError>;

fn system_clock() -> Result<u64, RuntimeError> {
    let ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| RuntimeError::ClockSkew)?
        .as_millis();
    Ok(ms as u64)
}

fn decide_with_clock(
    mode: RuntimeMode,
    issue: CompatibilityIssue,
    prior_compatible: f64,
    loss: LossMatrix,
    evidence: Vec<EvidenceTerm>,
    clock: ClockFn,
) -> DecisionRecord {
    // ... use clock() instead of now_unix_ms() ...
}
```

**Phase 2 (Cx integration):** Accept `&Cx<TimeCaps>` and use `cx.time()`.

```rust
fn decide_cx<C: HasTime>(
    cx: &Cx<C>,
    mode: RuntimeMode,
    issue: CompatibilityIssue,
    prior_compatible: f64,
    loss: LossMatrix,
    evidence: Vec<EvidenceTerm>,
) -> DecisionRecord {
    cx.checkpoint().expect("cancellation");
    let ts = cx.time().unix_ms();
    // ...
}
```

**Phase 3 (audit gate):** Add CI lint to deny `SystemTime::now` and
`Instant::now` in fp-runtime source files.

### 5.5 Risk if Unremediated

If the violation persists when lab-mode testing is introduced:
1. All `LabRuntime` tests involving the decision engine will produce non-deterministic
   `ts_unix_ms` values.
2. Trace comparison between runs will fail on timestamp fields.
3. Oracle assertions on temporal ordering will be unreliable.
4. The conformal guard's coverage tracking (which depends on decision ordering)
   will not be reproducible.
5. Any e-process monitor asserting temporal invariants on the evidence ledger will
   produce false positives.

The violation is tolerable in the current synchronous, non-lab context. It becomes
**blocking** the moment any of the following are introduced: `LabRuntime` testing,
trace-based replay, deterministic CI, or temporal oracle assertions.

---

## 6. Dependency Chain Risks

### 6.1 Transitive Dependency Tree

When the `asupersync` feature is enabled, the following transitive dependencies
are pulled (as recorded in the anchor map):

| Dependency | Purpose | Risk |
|---|---|---|
| `base64` | Encoding for wire formats | LOW -- widely audited, no unsafe |
| `bincode` | Binary serialization | MEDIUM -- deserialization of untrusted data is risky |
| `crossbeam-queue` | Lock-free concurrent queues | LOW -- well-maintained, audited unsafe |
| `getrandom` | OS-level randomness | LOW -- platform-specific; links to libc |
| `libc` | FFI bindings to C standard library | MEDIUM -- unsafe FFI; platform-dependent |
| `nix` | Unix system call wrappers | MEDIUM -- unsafe FFI; Unix-only |
| `parking_lot` | Efficient mutex/rwlock | LOW -- well-maintained, audited unsafe |
| `pin-project` | Safe pin projections | LOW -- proc macro; audited |
| `polling` | Portable async I/O polling | MEDIUM -- platform-specific unsafe |
| `rmp-serde` | MessagePack serialization | LOW -- serde-based |
| `serde` | Serialization framework | LOW -- already a workspace dependency |

### 6.2 Supply Chain Attack Vectors

| Vector | Description | Mitigation |
|---|---|---|
| **Typosquatting** | Misspelled crate name in Cargo.toml | Pin exact crate name; verify checksum |
| **Dependency confusion** | Private registry shadowed by crates.io | FrankenPandas uses only crates.io |
| **Malicious patch release** | asupersync 0.1.2 contains backdoor | Pin exact version or use `=0.1.1`; run `cargo audit` |
| **Transitive vuln** | Vulnerability in `bincode`, `nix`, or `libc` | Regular `cargo audit`; Dependabot/RenovateBot |
| **Build script injection** | Dependency build.rs executes arbitrary code | Audit build scripts of new deps; use `cargo-deny` |

### 6.3 Binary Size and Compile Time Impact

| Metric | Without asupersync | With asupersync | Delta |
|---|---|---|---|
| Transitive dependencies | ~15 (fp-runtime baseline) | ~26 (+11 from asupersync) | +73% dependency count |
| Compile time (estimated) | Baseline | +15-30s for asupersync tree | Moderate increase |
| Binary size (estimated) | Baseline | +200-500KB (GF(256) tables, transport abstractions) | Noticeable increase |

### 6.4 Platform Compatibility Risks

| Dependency | Platform Risk |
|---|---|
| `nix` | Unix-only; Windows builds will fail or degrade |
| `libc` | Platform-specific behavior; musl vs glibc differences |
| `polling` | Uses epoll (Linux), kqueue (macOS), IOCP (Windows) -- different behavior per platform |
| `getrandom` | Different entropy sources per platform; WASM needs special configuration |

**Fail-closed rule:** If any transitive dependency fails to compile on the target
platform, the `asupersync` feature must be disabled. The `default = []` configuration
ensures this is the fallback behavior.

### 6.5 `unsafe` Code Audit Surface

| Dependency | Contains `unsafe` | Audit Status |
|---|---|---|
| `crossbeam-queue` | Yes (lock-free internals) | Well-audited; RustSec clean |
| `parking_lot` | Yes (low-level synchronization) | Well-audited; RustSec clean |
| `libc` | Yes (FFI bindings) | Foundational crate; widely used |
| `nix` | Yes (syscall wrappers) | Actively maintained; reviewed |
| `polling` | Yes (platform I/O) | smol-rs ecosystem; reviewed |
| `getrandom` | Yes (OS entropy) | Rust project-adjacent; reviewed |
| `base64` | No | Safe |
| `bincode` | No (recent versions) | Safe |
| `pin-project` | No (proc macro) | Safe |
| `rmp-serde` | No | Safe |
| `serde` | Minimal unsafe | Well-audited |

**Note:** `fp-runtime` itself uses `#![forbid(unsafe_code)]`, ensuring no unsafe
code in the FrankenPandas integration layer regardless of what dependencies contain.

---

## 7. Recommendations

Prioritized list of security hardening steps, ordered by impact and urgency.

### P0: Critical (Must Fix Before Lab Mode)

**R-01: Remediate INV-NO-AMBIENT-AUTHORITY violation in `now_unix_ms()`.**
- Replace `SystemTime::now()` with an injectable clock function (Phase 1).
- Provide a `SystemClock` default and a `FakeClock` for testing.
- Wire Cx time capability when Cx integration lands (Phase 2).
- Add CI lint to prevent future ambient authority regressions (Phase 3).
- **Blocked by:** Nothing. Can be done immediately.
- **Blocks:** Lab-mode deterministic testing, trace replay, temporal oracles.

**R-02: Eliminate silent zero-timestamp sentinel.**
- Change `now_unix_ms()` error handling from `unwrap_or_default()` to explicit
  error propagation or `Option<u64>` in `DecisionRecord.ts_unix_ms`.
- In strict mode, `decide()` must return an error (not proceed with ts=0).
- In hardened mode, `decide()` may proceed but must record the clock skew
  as evidence in the decision record.
- **Blocked by:** R-01 (same code path).
- **Blocks:** Reliable audit trail.

### P1: High (Must Fix Before RaptorQ Wiring)

**R-03: Integrate `SecurityContext` + `AuthenticatedSymbol` for RaptorQ.**
- When actual RaptorQ encode/decode is wired, all symbols must carry
  epoch-scoped authentication tags.
- Strict mode must reject any unauthenticated symbol.
- Hardened mode must log unauthenticated symbols with degraded confidence.
- **Blocked by:** Actual RaptorQ pipeline wiring.
- **Blocks:** Secure artifact synchronization.

**R-04: Add artifact binding to `DecodeProof`.**
- Add `artifact_id: String` and `nonce: u64` fields to `DecodeProof`.
- Validate that proof `artifact_id` matches envelope `artifact_id` before
  accepting the proof.
- **Status (2026-02-15):** Partial mitigation landed in
  `verify_packet_sidecar_integrity()` by enforcing decode-proof hash pairing
  between sidecar envelope and decode-proof artifact plus `sha256:` prefix
  validation (`sidecar_integrity_fails_when_decode_proof_hash_mismatches_sidecar`).
  Full `artifact_id + nonce` binding is still pending.
- **Blocked by:** Nothing.
- **Blocks:** Proof replay prevention.

**R-05: Add error severity classification to outcome bridge.**
- Extend `outcome_to_action` (or a new `outcome_to_action_classified`) to
  inspect the error type `E` when it implements a classification trait.
- Map permanent errors to `Reject`, transient errors to `Repair`.
- Preserve current behavior as fallback when `E` does not implement the trait.
- **Blocked by:** asupersync error classification API.
- **Blocks:** Correct supervision retry policy.

### P2: Medium (Should Fix Before Production Use)

**R-06: Insert cancellation checkpoints at kernel boundaries.**
- Identify FrankenPandas hot loops: alignment iteration (`align_union`),
  groupby accumulation (`groupby_sum`), join cross-product expansion
  (`join_series`), CSV row parsing (`read_csv_str`).
- When Cx integration lands, insert `cx.checkpoint()` calls at each
  iteration boundary.
- Before Cx integration, add a simple `check_cancelled(flag: &AtomicBool)`
  pattern as a stepping stone.
- **Blocked by:** Cx integration design.
- **Blocks:** Cooperative cancellation.

**R-07: Cap `decode_proofs` vector in `RaptorQEnvelope`.**
- Add a constant `MAX_DECODE_PROOFS = 1000`.
- Enforce the cap in any method that adds proofs.
- In strict mode, reject the envelope if the cap is reached.
- In hardened mode, evict the oldest proof.
- **Blocked by:** Nothing.
- **Blocks:** Memory exhaustion prevention.

**R-08: Change placeholder `ScrubStatus` from `"ok"` to `"placeholder"`.**
- This prevents false-positive integrity signals from placeholder envelopes.
- All code that checks `scrub.status == "ok"` must be updated.
- **Blocked by:** Nothing.
- **Blocks:** Accurate integrity reporting.

**R-09: Add `prior_compatible` clamping in `decide()`.**
- Clamp `prior_compatible` to `(1e-10, 1.0 - 1e-10)` to prevent
  infinity/NaN in log-odds computation.
- This addresses CR-05 from the contract table.
- **Blocked by:** Nothing.
- **Blocks:** Numerical stability of Bayesian engine.

### P3: Low (Deferred to Full Integration)

**R-10: Implement Cx capability narrowing for FrankenPandas call stack.**
- Define FrankenPandas-specific capability profiles:
  - `FpFullCaps` -- entry points (DataFrame operations)
  - `FpComputeCaps` -- pure computation (alignment, expression evaluation)
  - `FpStorageCaps` -- I/O operations (CSV/JSON read/write)
- Thread `&Cx<Caps>` through the call stack.
- **Blocked by:** Cx integration design; asupersync Cx API stability.
- **Blocks:** Full capability security.

**R-11: Implement supervision tree for FrankenPandas services.**
- Define supervision policies for any background services (if introduced).
- Enforce INV-SUPERVISION-MONOTONE: never restart panicked tasks.
- Use restart budgets with exponential backoff for transient errors.
- **Blocked by:** Background service architecture decision.
- **Blocks:** Resilient operation under failure.

**R-12: Add compile-time audit gate for ambient authority.**
- Implement a CI lint (clippy lint, custom lint, or build script check)
  that denies `SystemTime::now`, `Instant::now`, `thread_rng`, `getrandom`,
  `std::fs`, `std::net`, and `std::thread::spawn` in fp-runtime source.
- Follow the asupersync `audit::ambient` pattern from the FrankenSQLite spec.
- **Blocked by:** R-01 (must fix violation before gate catches it).
- **Blocks:** Preventing future ambient authority regressions.

**R-13: Implement budget enforcement at FrankenPandas entry points.**
- Wrap all public API entry points in `cx.scope_with_budget()`.
- Define default budgets for common operations (alignment, groupby, join, IO).
- Document budget semantics in the crate-level docs.
- **Blocked by:** Cx integration; budget calibration data.
- **Blocks:** Resource exhaustion prevention.

**R-14: Add `cargo-deny` configuration for asupersync dependency tree.**
- Deny known-vulnerable versions of transitive dependencies.
- Deny duplicate dependency versions.
- Deny `unsafe` in new direct dependencies (transitive unsafe is accepted
  for established crates).
- Run in CI as a gating check.
- **Blocked by:** Nothing.
- **Blocks:** Supply chain risk reduction.

---

## Drift Gates

These conditions must hold for this threat model to remain valid. If any gate
is violated, this document must be re-evaluated.

1. `#![forbid(unsafe_code)]` remains on `fp-runtime` (memory safety boundary).
2. `asupersync` remains an optional, default-off dependency in `fp-runtime` only.
3. No other workspace crate declares an asupersync dependency (INV-FEATURE-NO-TRANSITIVE).
4. The `Outcome<T, E>` enum has exactly four variants (INV-OUTCOME-EXHAUSTIVE).
5. The outcome bridge remains stateless and side-effect-free (INV-OUTCOME-STATELESS).
6. The evidence ledger remains append-only (INV-LEDGER-APPEND-ONLY).
7. `RaptorQEnvelope` remains structural metadata only (no actual encode/decode).
8. `SystemTime::now()` remains the sole ambient authority violation (scope of R-01).

If asupersync releases a version > 0.1.x, or if additional asupersync modules
(transport, lab, security) are integrated, this threat model must be extended to
cover the expanded surface.

---

## Appendix A: Invariant Cross-Reference

| Invariant | Defined In | Status in FrankenPandas | Threat Impact |
|---|---|---|---|
| INV-NO-AMBIENT-AUTHORITY | FrankenSQLite spec 4.1.1 | **VIOLATED** (`SystemTime::now()`) | ATS-3.1, ATS-3.4 |
| INV-SUPERVISION-MONOTONE | FrankenSQLite spec 4.14 | Planned (no supervisor yet) | ATS-6.1 |
| INV-OUTCOME-EXHAUSTIVE | Contract table 12 | Enforced (exhaustive match) | ATS-1.1 |
| INV-OUTCOME-STATELESS | Contract table 12 | Enforced (no side effects) | ATS-1.4 |
| INV-OUTCOME-MUST-USE | Contract table 12 | Enforced (`#[must_use]`) | ATS-1.4 |
| INV-FEATURE-ISOLATED | Contract table 12 | Enforced (`default = []`) | ATS-2.1, ATS-2.2 |
| INV-FEATURE-NO-TRANSITIVE | Contract table 12 | Enforced (workspace audit) | ATS-2.2 |
| INV-LEDGER-APPEND-ONLY | Contract table 12 | Enforced (API surface) | ATS-5.4 |
| INV-NO-UNSAFE | Contract table 12 | Enforced (`#![forbid(unsafe_code)]`) | Dependency chain (sec 6) |
| INV-CONFORMAL-ALPHA-CLAMPED | Contract table 12 | Enforced (`.clamp()`) | N/A |
| INV-STRICT-FAIL-CLOSED | Contract table 12 | Enforced (post-decision override) | ATS-1.3 |
| INV-HARDENED-CAP-REPAIR | Contract table 12 | Enforced (post-decision override) | ATS-1.3 |
| INV-PLACEHOLDER-SENTINEL | Contract table 12 | Enforced (`"blake3:placeholder"`) | ATS-4.1, ATS-4.5 |
| INV-OBLIGATION-RESOLVE | Contract table 12 | Planned (no obligation model) | ATS-5.4 |

---

## Appendix B: Threat-to-Recommendation Traceability

| Threat ID | Recommendation | Priority |
|---|---|---|
| ATS-1.1 | (None needed; fail-closed by compiler) | -- |
| ATS-1.2 | R-05 (error classification) | P1 |
| ATS-1.3 | R-05 (error classification) | P1 |
| ATS-1.4 | R-01 (audit trail discipline) | P0 |
| ATS-2.1 | R-14 (cargo-deny) | P3 |
| ATS-2.2 | R-14 (cargo-deny) | P3 |
| ATS-2.3 | (None needed; fail-closed by compiler) | -- |
| ATS-3.1 | **R-01 (remediate ambient authority)** | **P0** |
| ATS-3.2 | **R-02 (eliminate zero sentinel)** | **P0** |
| ATS-3.3 | R-02 (eliminate zero sentinel) | P0 |
| ATS-3.4 | **R-01 (remediate ambient authority)** | **P0** |
| ATS-4.1 | R-08 (placeholder status) | P2 |
| ATS-4.2 | **R-03 (authenticated symbols)** | **P1** |
| ATS-4.3 | **R-04 (proof binding; partial hash-link mitigation landed 2026-02-15)** | **P1** |
| ATS-4.4 | R-07 (cap decode proofs) | P2 |
| ATS-4.5 | R-08 (placeholder status) | P2 |
| ATS-5.1 | R-06 (cancellation checkpoints) | P2 |
| ATS-5.2 | R-13 (budget enforcement) | P3 |
| ATS-5.3 | R-06 (cancellation checkpoints) | P2 |
| ATS-5.4 | (Deferred to obligation model design) | P3 |
| ATS-6.1 | R-11 (supervision tree) | P3 |
| ATS-6.2 | R-05 (error classification) | P1 |
| ATS-6.3 | R-11 (supervision tree) | P3 |
