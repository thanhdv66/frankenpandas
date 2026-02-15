# DOC-PASS-08: Security/Compatibility Edge Cases and Undefined Zones

**Bead:** `bd-2gi.23.9`  
**Date:** 2026-02-15  
**Status:** Complete  
**Scope:** enumerate security-sensitive and compatibility-sensitive edge cases, flag undefined/ambiguous zones, and anchor strict/hardened rationale to source.

---

## 1. Doctrine and Boundaries

FrankenPandas explicitly adopts a fail-closed doctrine for unknown/unsafe paths, with strict vs hardened mode splits used as the primary behavior bifurcation:

- doctrine anchor: `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:5`
- strict/hardened split in runtime policy: `crates/fp-runtime/src/lib.rs:158`, `crates/fp-runtime/src/lib.rs:167`
- strict fail-closed unknown-feature override: `crates/fp-runtime/src/lib.rs:201`
- hardened join-cap repair override: `crates/fp-runtime/src/lib.rs:245`

This pass does not redefine existing threat models; it consolidates edge cases into a source-anchored operator/developer ledger.

---

## 2. Security/Compatibility Edge-Case Matrix

| Edge ID | Edge case | Security / compatibility impact | Strict behavior | Hardened behavior | Mitigation / note anchors |
|---|---|---|---|---|---|
| `EC-01` | CSV ingestion has no explicit row-size/file-size budget in parser loops | potential OOM/DoS on huge streams | currently same parser path | currently same parser path | `crates/fp-io/src/lib.rs:79`, `crates/fp-io/src/lib.rs:175`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:20` |
| `EC-02` | CSV bool parsing is Rust-`bool` parser dependent (`true`/`false`) and can diverge from pandas casing acceptance | parity drift on mixed-case inputs | rejects incompatible casing via parse fallback to `Utf8` | same default behavior | `crates/fp-io/src/lib.rs:133`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:24` |
| `EC-03` | Float/int coercion boundaries (`lossy`, overflow, precision loss) | silent compatibility drift risk for numeric extremes | hard error on invalid casts | same cast semantics | `crates/fp-types/src/lib.rs:118`, `crates/fp-types/src/lib.rs:188`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:47` |
| `EC-04` | NaN semantic equality intentionally diverges from IEEE strictness (`NaN == NaN` in semantic checks) | compatibility-vs-IEEE tension | preserved for pandas-like behavior | preserved | `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:50`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:58` |
| `EC-05` | Runtime priors/LLRs/tie-breaking are hardcoded; equal expected-loss ties default to `Allow` | policy quality and explainability risk | unknown features still forced `Reject` | tie behavior remains unless cap override triggers `Repair` | `crates/fp-runtime/src/lib.rs:299`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:123`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:127` |
| `EC-06` | Join explosion risk: estimator exists but join kernel has no direct cardinality-enforcement hook | OOM risk under duplicate-key cartesian amplification | no strict join cap by default (`None`) | cap can force `Repair` at policy admission layer | `crates/fp-runtime/src/lib.rs:162`, `crates/fp-runtime/src/lib.rs:245`, `crates/fp-join/src/lib.rs:99`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:151` |
| `EC-07` | GroupBy dense-path boundary cliff at fixed span 65,536 | perf/behavior cliff at threshold, potential path drift | same algorithmic boundary | same algorithmic boundary + budget-based arena fallback | `crates/fp-groupby/src/lib.rs:297`, `crates/fp-groupby/src/lib.rs:336`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:177` |
| `EC-08` | Oracle adapter unsupported operation path is fail-closed | explicit compatibility boundary (narrow live-oracle surface) | hard fail in strict-legacy mode | optional fallback if explicitly enabled | `crates/fp-conformance/oracle/pandas_oracle.py:57`, `crates/fp-conformance/oracle/pandas_oracle.py:290`, `crates/fp-conformance/src/bin/fp-conformance-cli.rs:42` |
| `EC-09` | Differential drift gate budgets can fail hard on strict/hardened budgets | release-gate blocking and compatibility envelope enforcement | strict budget violations block | hardened divergence budget violations also block | `crates/fp-conformance/src/lib.rs:914`, `crates/fp-conformance/src/lib.rs:926`, `crates/fp-conformance/src/lib.rs:620` |
| `EC-10` | Drift history and forensic logs are append-only but currently omit some replay IDs | auditability partial for per-case reconstruction | same logging shape | same logging shape | `crates/fp-conformance/src/lib.rs:440`, `crates/fp-conformance/src/lib.rs:652`, `crates/fp-conformance/src/lib.rs:3329` |
| `EC-11` | Decode-proof replay risk (missing explicit artifact binding/nonce) | integrity-attestation spoof risk across artifacts | should reject in strict posture | hardened may tolerate with warning depending on future policy | `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:230`, `artifacts/phase2c/ASUPERSYNC_THREAT_MODEL.md:114` |
| `EC-12` | Runtime timestamp uses ambient wall clock with zero-sentinel fallback on skew | non-deterministic traces + ambiguous `ts=0` values | recommended strict abort path documented in threat model | currently proceeds with fallback | `crates/fp-runtime/src/lib.rs:267`, `crates/fp-runtime/src/lib.rs:311`, `artifacts/phase2c/ASUPERSYNC_THREAT_MODEL.md:78`, `artifacts/phase2c/ASUPERSYNC_THREAT_MODEL.md:82` |

---

## 3. Undefined / Ambiguous Zones Ledger

| Zone ID | Undefined / ambiguous zone | Why ambiguous | Current observed behavior | Risk class | Resolution direction |
|---|---|---|---|---|---|
| `UZ-01` | Decision tie-breaking when expected losses are equal | tie policy not externally documented as contract | decision engine defaults to `Allow` unless a lower loss is found | compatibility drift in edge priors | codify tie rule as explicit contract row + test |
| `UZ-02` | Clock-skew timestamp fallback semantics | zero sentinel is overloaded (`valid epoch` vs `error fallback`) | `ts_unix_ms` can become `0` via `unwrap_or_default` | forensic ambiguity / determinism loss | replace sentinel with explicit error channel |
| `UZ-03` | Strict-mode join cardinality hard-stop policy | strict has no cap, unlike hardened cap path | strict may admit huge joins if Bayesian output is not reject | OOM/DoS in worst cases | evaluate strict defense-in-depth cap |
| `UZ-04` | CSV bool casing parity with pandas | parser behavior tied to Rust bool parse and fallback chain | non-lowercase booleans may become `Utf8` | compatibility edge drift | add explicit casing normalization policy or error contract |
| `UZ-05` | NaN semantic-equality contract scope | pandas parity objective conflicts with IEEE expectations | semantic equality treats missing NaN forms as equal | cross-tool expectation mismatch | retain behavior but document as intentional compatibility mode rule |
| `UZ-06` | Live oracle provenance under fallback flag | fallback may import non-legacy pandas | fallback is opt-in via CLI flag | oracle trust envelope widening | keep opt-in; emit explicit provenance marker in reports |
| `UZ-07` | Differential replay key granularity | records lack dedicated `replay_key` field | replay derives from `case_id`/packet context | replay tooling friction | add explicit `replay_key` in differential artifacts |
| `UZ-08` | E2E case latency semantics | `CaseEnd.elapsed_us` set to `0` in retrospective emission path | no per-case measured latency for timeline | reduced forensic/perf confidence | capture real per-case elapsed times in run loop |

Anchors: `crates/fp-runtime/src/lib.rs:299`, `crates/fp-runtime/src/lib.rs:311`, `crates/fp-runtime/src/lib.rs:162`, `crates/fp-io/src/lib.rs:133`, `crates/fp-conformance/oracle/pandas_oracle.py:57`, `crates/fp-conformance/src/lib.rs:3535`, `crates/fp-conformance/src/lib.rs:3540`.

---

## 4. Hardened-Mode Rationale (Source-Anchored)

Hardened-mode divergences are intentional guardrails, not accidental drift:

1. Unknown-feature handling is no longer force-reject by default; strict keeps fail-closed override.
2. Join-cap boundary in hardened mode forces `Repair` when estimate exceeds cap.
3. Gate evaluation tracks strict and hardened failure counters separately with independent thresholds.
4. Drift allowlist categories are part of hardened gate configuration surface.
5. Evidence ledger remains append-only in both modes, preserving forensic continuity.

Anchors:

- `crates/fp-runtime/src/lib.rs:169`
- `crates/fp-runtime/src/lib.rs:245`
- `crates/fp-conformance/src/lib.rs:392`
- `crates/fp-conformance/src/lib.rs:932`
- `crates/fp-runtime/src/lib.rs:126`

---

## 5. Mitigation Priority Map (from Existing Threat Ledgers)

| Priority band | Representative open risk items | Anchor |
|---|---|---|
| `P0` | join-cardinality OOM risk, CSV-row-limit gap, RaptorQ verification strictness | `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:283`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:284`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:285` |
| `P1` | oracle path/provenance risk, float accumulation and precision drift, authenticated-symbol gaps | `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:286`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:287`, `artifacts/phase2c/ASUPERSYNC_THREAT_MODEL.md:678` |
| `P2` | decode-proof replay binding, hash collision DoS, bounded proof-vector limits | `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:289`, `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:290`, `artifacts/phase2c/ASUPERSYNC_THREAT_MODEL.md:680` |
| `P3` | prior calibration justification and fallback governance hardening | `artifacts/phase2c/SECURITY_COMPATIBILITY_THREAT_MATRIX.md:291`, `artifacts/phase2c/ASUPERSYNC_THREAT_MODEL.md:683` |

---

## 6. Acceptance Criteria Check

| Requirement | Status | Evidence |
|---|---|---|
| Security/compat edge cases enumerated with mitigation notes | complete | Sections 2 and 5 |
| Undefined zones explicitly flagged | complete | Section 3 |
| Hardened-mode rationale source-anchored | complete | Section 4 |

