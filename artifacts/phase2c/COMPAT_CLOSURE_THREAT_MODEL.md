# COMPAT_CLOSURE Security + Compatibility Threat Model

Bead: `bd-2gi.29.3` [COMPAT-CLOSURE-C]
Subsystem: Drop-in compatibility closure for strict/hardened semantics, oracle differential gates, migration guarantees, and durable closure evidence
Source anchors: `artifacts/phase2c/COMPAT_CLOSURE_ANCHOR_MAP.md` (bd-2gi.29.1), `artifacts/phase2c/COMPAT_CLOSURE_CONTRACT_TABLE.md` (bd-2gi.29.2)

Doctrine: **fail closed on unknown or under-evidenced paths**. Any ambiguity in compatibility claims, drift provenance, or artifact durability must block closure rather than permit optimistic inference.

---

## 1. Summary

This threat model covers security and compatibility failure modes that can invalidate
COMPAT-CLOSURE claims for FrankenPandas packet evidence. Scope includes:

- strict/hardened behavioral envelopes,
- differential/oracle trust boundaries,
- artifact durability and replay chain integrity,
- release-gate misuse and downgrade pathways,
- migration-claim overreach against incomplete parity surfaces.

Out of scope:

- authentication/authorization (local operator workflow),
- network threat surfaces not used by current packet workflows,
- non-phase2c API families that are explicitly marked non-closed in `FEATURE_PARITY.md`.

---

## 2. Threat Surface Enumeration

Surfaces are numbered `CTS-*` (Compat Threat Surface).

### CTS-1: Semantic Drift Surface (Strict/Hardened)

| Property | Value |
|---|---|
| Entry points | `RuntimePolicy`, `Series::binary_op_with_policy`, join admission, unknown-feature decisions |
| Input class | Duplicate labels, null/NaN-heavy payloads, dtype coercion boundaries, unknown features |
| Fail-closed behavior | Strict rejects unknown/unsafe behavior; hardened permits bounded repairs with evidence |

Threats:

- **CTS-1.1 Mode boundary collapse:** strict and hardened produce identical outcomes when strict should reject (silent policy regression).
- **CTS-1.2 Duplicate-index downgrade:** duplicate labels are admitted without decision evidence, violating strict-mode fail-closed semantics.
- **CTS-1.3 Null/NaN coercion drift:** null propagation or NaN sentinel handling diverges from oracle due optimization or coercion shortcuts.
- **CTS-1.4 DType promotion drift:** promotion table changes produce silently different output dtypes for mixed arithmetic.

### CTS-2: Differential/Oracle Provenance Surface

| Property | Value |
|---|---|
| Entry points | `run_differential_suite`, oracle adapter dispatch, mismatch corpus + replay key emission |
| Input class | Fixture payloads, live legacy oracle responses, fallback-mode outputs |
| Fail-closed behavior | Unsupported ops/schema kinds and strict legacy import failures return hard errors |

Threats:

- **CTS-2.1 Oracle fallback provenance drift:** fallback-enabled runs are interpreted as strict closure evidence without explicit provenance marking.
- **CTS-2.2 Mismatch taxonomy erosion:** drift records collapse to coarse classes, masking critical compatibility regressions.
- **CTS-2.3 Replay non-determinism:** failure artifacts lack deterministic trace/replay fields, preventing reproducible audit.

### CTS-3: Durability + Decode-Proof Surface

| Property | Value |
|---|---|
| Entry points | Packet artifact writer, `verify_packet_sidecar_integrity`, decode-proof artifacts |
| Input class | `parity_report.json`, sidecar manifest, scrub report, decode-proof artifact |
| Fail-closed behavior | Missing/mismatched sidecar/decode proofs fail integrity checks and closure evidence |

Threats:

- **CTS-3.1 Sidecar/decode mismatch acceptance:** decode proof hashes are not paired to sidecar envelope hashes.
- **CTS-3.2 Source-hash drift:** sidecar source hash does not match parity report payload hash.
- **CTS-3.3 Scrub false-positive:** scrub metadata claims healthy artifact despite invalid packet set.
- **CTS-3.4 Decode-proof replay:** stale decode-proof artifact replayed across packet contexts.

### CTS-4: Closure Claim and Migration Surface

| Property | Value |
|---|---|
| Entry points | parity matrix, closure docs, migration manifests, operator sign-off |
| Input class | feature-matrix status, packet evidence registry, compatibility narratives |
| Fail-closed behavior | Any un-evidenced API family blocks closure claim and migration attestation |

Threats:

- **CTS-4.1 Over-claim attack:** closure documentation claims full compatibility while known gaps remain `in_progress` or `parity_gap`.
- **CTS-4.2 Migration downgrade ambiguity:** migration guarantees omit strict/hardened behavior boundaries, causing operational misinterpretation.
- **CTS-4.3 Evidence-link rot:** claim statements are not linked to stable artifact IDs/replay commands.

### CTS-5: Gate and Operator Workflow Surface

| Property | Value |
|---|---|
| Entry points | `fp-conformance-cli --require-green`, gate scripts, FRANKENTUI operator summaries |
| Input class | command-line flags, packet filters, artifact roots |
| Fail-closed behavior | gate/report mismatch or non-green packet set must return non-zero |

Threats:

- **CTS-5.1 Non-blocking release path:** release workflow bypasses `--require-green` and ships drifted behavior.
- **CTS-5.2 Packet-filter abuse:** selective packet runs are presented as total-closure evidence.
- **CTS-5.3 Evidence summarizer blind spots:** operator dashboards omit sidecar/decode integrity failures from final risk notes.

### CTS-6: Dependency and Build Surface

| Property | Value |
|---|---|
| Entry points | workspace dependency graph, feature gating, lockfile updates |
| Input class | crate updates, transitive features, CI build matrix |
| Fail-closed behavior | incompatible schema/feature drift fails build or CI gates |

Threats:

- **CTS-6.1 Transitive dependency drift:** serializer or parser behavior changes without corresponding contract updates.
- **CTS-6.2 Feature-gate bleed:** optional compatibility/security code paths become implicitly enabled/disabled by workspace unification.
- **CTS-6.3 Lockfile provenance ambiguity:** build evidence cannot be tied to concrete dependency state.

---

## 3. Threat Matrix

| ID | Surface | Description | Likelihood | Impact | Risk | Fail-Closed Control | Testable Evidence Hook |
|---|---|---|---|---|---|---|---|
| CTS-1.1 | Semantic drift | strict/hardened boundary collapse | M | H | HIGH | enforce explicit mode-branch assertions | mode-split unit/property tests over `CC-008/CC-009` |
| CTS-1.2 | Semantic drift | duplicate-index acceptance in strict mode | M | H | HIGH | strict rejects duplicates by contract | `Series::binary_op_with_policy` strict duplicate tests |
| CTS-1.3 | Semantic drift | null/NaN propagation regression | M | H | HIGH | drift-critical mismatch classification + gate fail | differential drift corpus with null/NaN adversarial fixtures |
| CTS-1.4 | Semantic drift | dtype-promotion regression | M | M | MEDIUM | `common_dtype` table as explicit contract | unit/property checks for promotion matrix |
| CTS-2.1 | Oracle | fallback provenance mis-labeled as strict closure evidence | M | H | HIGH | strict-legacy import fail unless fallback explicitly enabled | oracle-mode audit fields in differential logs |
| CTS-2.2 | Oracle | mismatch taxonomy erosion hides severity | M | H | HIGH | critical/non-critical/informational taxonomy invariants | drift summary + mismatch class regression tests |
| CTS-2.3 | Oracle | replay non-determinism | L | H | MEDIUM | deterministic `trace_id` + `replay_key` + replay command | E2E replay log contract assertions |
| CTS-3.1 | Durability | decode proof hash mismatch accepted | M | H | HIGH | `verify_packet_sidecar_integrity` rejects unpaired hashes | sidecar/decode mismatch regression tests |
| CTS-3.2 | Durability | sidecar source-hash mismatch | L | H | MEDIUM | sidecar hash verification against parity payload | packet-level integrity check outputs |
| CTS-3.3 | Durability | scrub false-positive accepted | L | M | LOW | scrub status and invalid packet count validation | sidecar integrity report fields |
| CTS-3.4 | Durability | decode-proof replay across packets | L | H | MEDIUM | packet_id + proof-hash binding checks | decode-proof integrity test fixtures |
| CTS-4.1 | Closure claim | over-claim against open parity gaps | M | H | HIGH | closure claim blocked unless all scoped rows evidence-backed | feature-matrix + packet evidence reconciliation step |
| CTS-4.2 | Migration | strict/hardened envelope omitted from migration guarantee | M | M | MEDIUM | explicit migration envelope table with mode behavior | migration checklist gate in release docs |
| CTS-4.3 | Closure claim | evidence-link rot (no artifact/replay links) | M | M | MEDIUM | claim records include artifact IDs + replay commands | final evidence pack schema checks |
| CTS-5.1 | Gate workflow | release bypasses `--require-green` | L | H | MEDIUM | release checklist requires gate command non-zero on failure | CI gate script assertions |
| CTS-5.2 | Gate workflow | packet filter scope presented as total closure | M | H | HIGH | require scope metadata in closure reports | packet scope field in signed pass summary |
| CTS-5.3 | Operator workflow | final evidence summary hides integrity failures | M | M | MEDIUM | final evidence pack includes risk notes from sidecar checks | FRANKENTUI final evidence tests + render checks |
| CTS-6.1 | Dependencies | serializer/parser drift invalidates contracts | M | M | MEDIUM | workspace check + clippy + contract doc updates coupled | CI compile/lint gates + contract changelog |
| CTS-6.2 | Dependencies | feature-gate bleed modifies behavior silently | L | M | LOW | explicit feature matrix and compile-gated tests | feature-enabled/disabled build checks |
| CTS-6.3 | Dependencies | lockfile provenance missing from evidence | M | M | MEDIUM | reproducibility ledger must include lockfile pointer | final evidence bundle reproducibility section |

---

## 4. Compatibility Envelope (Strict vs Hardened)

| Envelope ID | Contract Surface | Strict Mode | Hardened Mode | Fail-Closed Trigger |
|---|---|---|---|---|
| CE-001 | Unknown feature handling | hard reject | Bayesian action (allow/repair/reject) with evidence | unknown feature observed without decision record |
| CE-002 | Duplicate labels in arithmetic | reject (`DuplicateIndexUnsupported`) | permit bounded repair path with evidence | strict mode emits non-reject outcome |
| CE-003 | Join admission over cap | no cap override path | forced repair above cap | hardened cap exceeded without repair decision |
| CE-004 | Differential drift severity | critical drift blocks gate | critical drift blocks gate; non-critical may remain budgeted | critical drift does not fail gate |
| CE-005 | Oracle unsupported operation | hard error | hard error | unsupported op returns synthetic success |
| CE-006 | Oracle strict-legacy import | hard fail by default | optional fallback only when explicitly enabled | fallback used without explicit operator enablement |
| CE-007 | Durability artifacts | sidecar + scrub + decode proof required | same requirement | any packet missing required durability artifacts |
| CE-008 | Closure claim scope | only evidence-backed green families in scope | same requirement | claim includes known open parity gaps |

Compatibility claim rule:

1. All `CE-*` rows must pass for scoped families.
2. Any fail-closed trigger blocks closure and migration sign-off.
3. Hardened-mode allowances never expand API shape; they only alter bounded recovery behavior with explicit evidence.

---

## 5. Defensive Controls and Verification Mapping

| Control ID | Control | Verification Command / Artifact | Pass Condition |
|---|---|---|---|
| DC-001 | Enforce packet gate closure | `cargo run -p fp-conformance -- --write-artifacts --write-drift-history --require-green` | non-green packets cause non-zero exit |
| DC-002 | Differential drift classification integrity | `cargo test -p fp-conformance -- --nocapture` (drift taxonomy tests) | critical/non-critical/informational classification stable |
| DC-003 | Sidecar/decode-proof pairing | `verify_packet_sidecar_integrity()` and packet integrity tests | mismatched proof hashes fail integrity |
| DC-004 | Final evidence risk-note visibility | FRANKENTUI final-evidence snapshot tests | decode-proof mismatch appears in packet/global risk notes |
| DC-005 | Strict/hardened contract boundary | mode-split policy/unit/property tests | strict fail-closed overrides remain intact |
| DC-006 | Scope-safe closure claim | parity matrix + packet evidence reconciliation | no unresolved feature family included in closure claim |
| DC-007 | Reproducibility ledger integrity | final evidence docs include lockfile pointer + replay commands + artifact links | independent reviewer can replay claim path deterministically |

---

## 6. EV Gate and Decision Card

### 6.1 Prioritized mitigation lever

Candidate lever: unify closure-threat enforcement around packet-level integrity + risk-note projection (semantic + durability + provenance).

`EV = (Impact * Confidence * Reuse) / (Effort * AdoptionFriction)`

- Impact = 4.3 (directly blocks false closure claims)
- Confidence = 0.86 (existing conformance + FRANKENTUI surfaces already expose needed signals)
- Reuse = 4.0 (applies across COMPAT-CLOSURE downstream beads)
- Effort = 2.1 (docs + validation hooks + regression tests)
- AdoptionFriction = 1.2

`EV = (4.3 * 0.86 * 4.0) / (2.1 * 1.2) = 5.87` -> passes `EV >= 2.0`.

### 6.2 Expected-loss decision card

States:

- `S1`: true compatibility closure (all scoped contracts satisfied)
- `S2`: latent semantic drift remains
- `S3`: durability/provenance evidence corrupted or incomplete

Actions:

- `A1`: approve closure claim
- `A2`: block release and demand remediation
- `A3`: ship hardened-only advisory with strict-mode hold

Loss matrix (`L(a,s)`, lower is better):

| Action \ State | S1 | S2 | S3 |
|---|---:|---:|---:|
| A1 approve | 0.5 | 9.0 | 10.0 |
| A2 block | 3.0 | 1.0 | 1.0 |
| A3 hardened advisory | 1.5 | 4.0 | 6.0 |

Current evidence-weighted belief example (post-29.3):

- `P(S1|e)=0.58`
- `P(S2|e)=0.27`
- `P(S3|e)=0.15`

Expected loss:

- `E[L|A1]=0.5*0.58 + 9*0.27 + 10*0.15 = 4.22`
- `E[L|A2]=3*0.58 + 1*0.27 + 1*0.15 = 2.16`
- `E[L|A3]=1.5*0.58 + 4*0.27 + 6*0.15 = 2.85`

Decision: **A2 (block until evidence complete)** remains lowest-loss unless durability + semantic drift probabilities drop materially.

---

## 7. Open Risk Notes

1. Full drop-in compatibility closure remains blocked by explicit parity gaps (`loc/iloc`, nanops matrix, merge/concat closure, broader API families).
2. Oracle fallback must remain explicitly tagged in evidence to avoid provenance dilution.
3. Migration guarantees require deterministic claim-to-artifact linkage; narrative-only sign-off is insufficient.
4. Packet-filtered runs must be scope-tagged and cannot substitute for total closure evidence.

---

## 8. Changelog

- **bd-2gi.29.3 (2026-02-15):** Added COMPAT-CLOSURE threat model with explicit high-impact abuse/drift classes, strict/hardened compatibility envelope, fail-closed controls, EV gate, and expected-loss decision card tied to closure evidence discipline.
