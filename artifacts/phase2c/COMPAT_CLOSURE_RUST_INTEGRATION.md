# COMPAT_CLOSURE Rust Integration Plan + Module Boundary Skeleton

Bead: `bd-2gi.29.4` [COMPAT-CLOSURE-D]
Subsystem: Drop-in compatibility closure execution architecture (contracts, validation, migration guarantees, and closure evidence)
Source anchors:
- `artifacts/phase2c/COMPAT_CLOSURE_ANCHOR_MAP.md` (bd-2gi.29.1)
- `artifacts/phase2c/COMPAT_CLOSURE_CONTRACT_TABLE.md` (bd-2gi.29.2)
- `artifacts/phase2c/COMPAT_CLOSURE_THREAT_MODEL.md` (bd-2gi.29.3)

---

## 1. Summary

This plan translates COMPAT-CLOSURE contract and threat artifacts into concrete
Rust integration seams and an execution sequence for beads `29.5` through `29.9`.
The objective is to keep closure evidence reproducible and fail-closed while
preventing cross-crate semantic drift.

Non-negotiable constraints:

- strict/hardened mode-split semantics remain explicit and testable,
- closure claims are evidence-backed only,
- sidecar/decode-proof durability is first-class in final evidence,
- migration guarantees must reference deterministic replay artifacts.

---

## 2. Current State (Implemented Substrate)

### 2.1 Contract registry already codified

`COMPAT_CLOSURE_CONTRACT_TABLE.md` defines machine-checkable rows `CC-001` through
`CC-020` spanning dtype/null, alignment, runtime policy, differential taxonomy,
packet gates, durability artifacts, oracle dispatch, and closure boundaries.

### 2.2 Threat surface now codified

`COMPAT_CLOSURE_THREAT_MODEL.md` defines:

- `CTS-*` threat surfaces,
- `CE-*` strict/hardened compatibility envelope,
- `DC-*` fail-closed controls + verification mapping,
- EV and expected-loss decision card for closure gating.

### 2.3 Operational evidence path already exists

Core evidence producers/consumers already in place:

- packet parity/gate/mismatch/durability artifacts in `fp-conformance`,
- policy and decision evidence in `fp-runtime`,
- operator-facing final evidence summaries in `fp-frankentui`.

This bead does not replace those paths; it formalizes integration ownership and
sequencing so downstream closure beads execute without architecture churn.

---

## 3. Target Integration Skeleton

The compat-closure implementation is cross-crate by design. The skeleton below
shows ownership boundaries, not a mandatory immediate refactor.

```text
crates/fp-conformance/src/
  compat_closure/
    mod.rs                    # closure orchestration facade
    contracts.rs              # CC-* contract evaluators and check registry
    drift.rs                  # differential drift aggregation + mismatch corpus shaping
    durability.rs             # sidecar/decode-proof integrity aggregation
    migration.rs              # migration manifest/checkpoint projection
    attestation.rs            # signed pass-summary payload assembly

crates/fp-runtime/src/
  compat/
    policy_bridge.rs          # strict/hardened boundary projection for closure checks
    decision_audit.rs         # decision-evidence projection helpers for closure scope

crates/fp-frankentui/src/
  closure/
    summary.rs                # closure evidence snapshots + risk-note rendering
    replay.rs                 # deterministic replay navigation/indexing hooks

artifacts/phase2c/
  COMPAT_CLOSURE_*.md         # human-auditable contract/threat/integration/evidence docs
```

### 3.1 Boundary rationale

| Layer | Owns | Must not own |
|---|---|---|
| `fp-conformance::compat_closure::*` | closure checks, differential/drift/durability aggregation, migration attestable outputs | runtime decision mutation, UI rendering |
| `fp-runtime::compat::*` | strict/hardened policy projection helpers, decision audit extraction | artifact IO, gate orchestration |
| `fp-frankentui::closure::*` | operator summaries and replay affordances | closure gate decisions, claim signing logic |
| `artifacts/phase2c/*` | auditable specs and evidence linkage | source-of-truth execution logic |

---

## 4. Contract-to-Module Mapping

| Contract/Threat class | Authoritative source | Integration owner |
|---|---|---|
| `CC-001..CC-009` semantic/policy contracts | contract table sections 2-4 | `fp-runtime` + `fp-frame` + `fp-types` tests and projections |
| `CC-010..CC-015` differential/gate/durability | contract table + threat `CTS-2/CTS-3` | `fp-conformance::compat_closure::{drift,durability}` |
| `CC-016..CC-017` oracle fail-closed behavior | contract table + threat `CTS-2` | `fp-conformance` oracle adapter + closure checks |
| `CC-018..CC-020` replay/closure boundaries | contract table + threat `CTS-4/CTS-5` | `fp-conformance::compat_closure::{migration,attestation}` + `fp-frankentui::closure` |
| `CE-*` compatibility envelope | threat model section 4 | shared assertions across runtime/conformance/UI surfaces |
| `DC-*` defensive controls | threat model section 5 | CI + tests + artifacts, with fail-closed release gates |

---

## 5. Risk and Invariant Ownership

### 5.1 High-risk ownership table

| Risk ID | Description | Owner | Required guard |
|---|---|---|---|
| R-CC-01 | strict/hardened boundary collapse | `fp-runtime` policy projections | mode-split unit/property tests must fail on collapse |
| R-CC-02 | closure over-claim with unresolved parity gaps | compat-closure orchestrator | claim builder must reject unresolved rows |
| R-CC-03 | decode-proof/sidecar mismatch accepted | durability aggregation | packet integrity check mandatory for closure evidence |
| R-CC-04 | oracle fallback provenance ambiguity | differential harness | provenance fields required in all closure reports |
| R-CC-05 | replay non-determinism | e2e/replay layer | trace/replay key determinism checks |

### 5.2 Non-negotiable invariants for downstream beads

- INV-CC-STRICT-FAIL-CLOSED: strict-mode unknown/unsafe paths always reject.
- INV-CC-HARDENED-BOUNDED: hardened repairs are bounded, explicit, and allowlisted.
- INV-CC-DURABILITY-T5: every closure packet has valid sidecar/decode pairing.
- INV-CC-CLAIM-SCOPE: closure claim cannot include unresolved parity families.
- INV-CC-REPLAY-DETERMINISM: every failure claim has deterministic replay metadata.

---

## 6. Execution Sequence (Risk-Minimizing)

### Phase 0 (completed prerequisites)

- `29.1` anchor map complete.
- `29.2` contract table complete.
- `29.3` threat model complete.

### Phase 1 (`29.4`, this bead)

1. Freeze integration ownership and module seams.
2. Map contract/threat rows to explicit implementation owners.
3. Define fail-closed invariants required for closure sign-off.

### Phase 2 (`29.5`)

1. Implement unit/property matrix around `CC-*` + `CE-*` boundaries.
2. Enforce deterministic structured logging for closure-critical paths.
3. Ensure strict/hardened divergence is explicit and bounded in tests.

### Phase 3 (`29.6` + `29.7`)

1. Differential/adversarial validation expansion (`CTS-1..CTS-3` stressors).
2. E2E replay/forensics matrix for closure scenarios and migration drills.
3. Add deterministic failure reproducer artifacts for every critical mismatch.

### Phase 4 (`29.8`)

1. Profile-first optimization loop with one-lever-at-a-time changes.
2. Preserve behavior-isomorphism across strict/hardened closure paths.
3. Record p50/p95/p99 + memory deltas tied to closure workloads.

### Phase 5 (`29.9`)

1. Emit final evidence pack with conformance + risk-note + benchmark delta.
2. Enforce durability artifact completeness (sidecar/scrub/decode proofs).
3. Publish attested pass summary linking claim IDs to CI gate outcomes.

---

## 7. EV and Decision Card (Architecture Choice)

### 7.1 Alternatives

- **A:** Keep closure logic ad hoc across existing modules.
- **B:** Centralize via compat-closure seams with explicit ownership (chosen).

### 7.2 EV gate

`EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction)`

- Impact = 4.4
- Confidence = 0.82
- Reuse = 4.1
- Effort = 2.3
- AdoptionFriction = 1.2

`EV = (4.4 * 0.82 * 4.1) / (2.3 * 1.2) = 5.35` -> passes `EV >= 2.0`.

### 7.3 Expected-loss framing

States:

- `S1`: closure evidence complete and semantically faithful,
- `S2`: semantic drift hidden by fragmented ownership,
- `S3`: durability/provenance evidence incomplete.

Actions:

- `A1`: ad hoc continuation,
- `A2`: enforce seam ownership and fail-closed gates (chosen).

Given current evidence, `A2` minimizes expected loss by reducing scope ambiguity
and making closure blockers machine-checkable earlier in the pipeline.

---

## 8. Delivery Checklist for `bd-2gi.29.4`

- [x] Integration seams and ownership defined across conformance/runtime/UI layers.
- [x] Contract/threat rows mapped to concrete implementation owners.
- [x] Fail-closed invariants for closure sign-off explicitly listed.
- [x] Downstream `29.5..29.9` execution sequence defined with risk-minimizing order.
- [x] EV gate and expected-loss rationale recorded for architecture choice.

---

## 9. Changelog

- **bd-2gi.29.4 (2026-02-15):** Added COMPAT-CLOSURE Rust integration plan with
  module boundary skeleton, ownership map, invariant ownership, phased execution
  sequence for `29.5..29.9`, and EV-gated architecture decision.
