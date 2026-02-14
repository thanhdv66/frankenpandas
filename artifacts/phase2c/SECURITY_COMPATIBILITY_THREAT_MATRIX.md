# Security/Compatibility Threat Matrix

FrankenPandas Phase-2C foundation artifact per bead **bd-2gi.2**.

Doctrine: frankenlibc/frankenfs-style fail-closed on unknown unsafe paths, explicit compatibility envelopes, and adversarial input classes.

---

## Threat Surface Inventory

### TS-1: Data Ingestion (fp-io)

| Property | Value |
|---|---|
| Entry point | `read_csv_str()` at fp-io/src/lib.rs |
| Input class | Untrusted CSV data |
| Fail-closed behavior | None currently |

**Threats:**
- **TS-1.1** Unbounded row iteration: no row count limit, infinite CSV streams cause OOM
- **TS-1.2** Unbounded field length: Utf8 scalars allocated without size cap
- **TS-1.3** Type inference order bias: `parse_scalar()` tries i64, f64, bool, utf8 in sequence; "1.5e100" becomes Float64 silently, differing from pandas
- **TS-1.4** Boolean case sensitivity: "True" rejected (pandas accepts case-insensitive)
- **TS-1.5** CSV injection: field values not escaped in write round-trip

**Compatibility envelope:**
- Strict mode: reject inputs exceeding row/field size limits (not yet enforced)
- Hardened mode: accept with bounded allocation + diagnostic log

**Required mitigations:**
- [ ] Configurable row count limit (default: 10M)
- [ ] Per-field byte length limit (default: 1MB)
- [ ] Case-insensitive boolean parsing for pandas parity
- [ ] Document type inference order as explicit contract

---

### TS-2: Type Coercion Paths (fp-types)

| Property | Value |
|---|---|
| Entry point | `cast_scalar_owned()`, `common_dtype()` |
| Input class | Internal scalar values from any source |
| Fail-closed behavior | TypeError returned on invalid casts |

**Threats:**
- **TS-2.1** i64-to-f64 precision loss: values > 2^53 lose precision silently
- **TS-2.2** Float-to-bool tolerance gap: 1.9999 accepted (rounds to true), 2.0 rejected
- **TS-2.3** Float-to-int overflow: platform-dependent behavior for OOB values
- **TS-2.4** NaN equality violation: `semantic_eq` treats NaN == NaN (violates IEEE 754, matches pandas)
- **TS-2.5** Null/NaN conflation: `Null(NaN)` and `Float64(NaN)` treated as semantically equal

**Compatibility envelope:**
- Strict mode: reject all lossy coercions with TypeError
- Hardened mode: permit lossy coercions with evidence record in ledger

**Required mitigations:**
- [ ] Document NaN==NaN as intentional pandas-compatible deviation from IEEE 754
- [ ] Add precision-loss detection for i64-to-f64 (optional warning in evidence ledger)
- [ ] Bound float-to-int cast behavior (explicit range check before platform cast)

---

### TS-3: Index Alignment (fp-index)

| Property | Value |
|---|---|
| Entry point | `align_union()`, `position_map_first()`, `has_duplicates()` |
| Input class | User-supplied or computed index labels |
| Fail-closed behavior | None; alignment always succeeds |

**Threats:**
- **TS-3.1** Hash collision DoS: crafted IndexLabel::Utf8 values force O(n^2) hash chains
- **TS-3.2** Large index allocation: align_union pre-allocates left.len() + right.len()
- **TS-3.3** Linear position search: `position()` is O(n) per call; quadratic if called in loop
- **TS-3.4** Empty index edge case: valid but creates redundant allocations

**Compatibility envelope:**
- Strict mode: reject indices exceeding configurable label count limit
- Hardened mode: accept with bounded allocation + fallback to sorted-merge alignment

**Required mitigations:**
- [ ] Rust standard HashMap uses SipHash (DoS-resistant); document this as sufficient
- [ ] Add index label count limit in hardened mode (default: 100M)
- [ ] Replace `position()` calls in hot paths with position_map_first() lookups

---

### TS-4: Null/NaN Propagation (fp-columnar)

| Property | Value |
|---|---|
| Entry point | `Column::binary_numeric()` |
| Input class | Internal column data with validity masks |
| Fail-closed behavior | Missing values propagate deterministically |

**Threats:**
- **TS-4.1** Division by zero produces IEEE 754 Inf (pandas raises ZeroDivisionError for integers)
- **TS-4.2** NaN/Null distinction breaks PartialEq transitivity
- **TS-4.3** Subnormal float results may silently underflow to zero
- **TS-4.4** Rounding error accumulation in chained operations (no error bound tracking)

**Compatibility envelope:**
- Strict mode: division by zero returns error (not Inf) for integer-typed columns
- Hardened mode: division by zero returns NaN with evidence record

**Required mitigations:**
- [ ] Add division-by-zero check before IEEE 754 division for integer-dtype columns
- [ ] Document NaN propagation rules as explicit contract
- [ ] Track error bounds for chained operations (deferred to optimization bead)

---

### TS-5: Runtime Policy (fp-runtime)

| Property | Value |
|---|---|
| Entry point | `decide_unknown_feature()`, `decide_join_admission()` |
| Input class | Internal policy decisions |
| Fail-closed behavior | Strict mode rejects unknowns |

**Threats:**
- **TS-5.1** Bayesian priors (0.25, 0.6) are unjustified magic numbers
- **TS-5.2** Log-likelihood ratios are hardcoded without sensitivity analysis
- **TS-5.3** Mode-based overrides bypass Bayesian decision (strict always rejects unknowns)
- **TS-5.4** Numerical instability: log-odds computation can overflow for extreme priors
- **TS-5.5** Tie-breaking: equal expected losses default to Allow (undocumented)
- **TS-5.6** Strict mode has no join cardinality cap (hardened_join_row_cap = None)

**Compatibility envelope:**
- Strict mode: fail-closed on unknown features; no join cap (relies on correctness)
- Hardened mode: evidence-driven decisions; configurable join cap

**Required mitigations:**
- [ ] Document priors and log-likelihoods as provisional; add calibration test bead
- [ ] Add numerical stability guard (clamp log_odds to [-50, 50])
- [ ] Document tie-breaking behavior explicitly
- [ ] Consider adding strict-mode join cap as defense-in-depth

---

### TS-6: Join Cardinality (fp-join)

| Property | Value |
|---|---|
| Entry point | `join_series()` |
| Input class | Two Series with potentially duplicated index labels |
| Fail-closed behavior | None; Cartesian product always computed |

**Threats:**
- **TS-6.1** Cartesian explosion: all-duplicate indices cause O(n*m) output rows
- **TS-6.2** No actual cardinality check in join_series (only pre-flight estimation in runtime policy)
- **TS-6.3** Memory exhaustion DoS via crafted all-duplicate indices
- **TS-6.4** Estimation is pessimistic (n*m) but doesn't account for selectivity

**Compatibility envelope:**
- Strict mode: compute full Cartesian product (pandas behavior)
- Hardened mode: enforce cardinality cap; truncate with evidence record if exceeded

**Required mitigations:**
- [ ] Add inline cardinality counter in join_series with configurable limit
- [ ] Return error (not OOM) when cardinality limit exceeded
- [ ] Wire hardened_join_row_cap into join_series as actual enforcement point

---

### TS-7: GroupBy Accumulation (fp-groupby)

| Property | Value |
|---|---|
| Entry point | `groupby_sum()` |
| Input class | Series with key column and values column |
| Fail-closed behavior | None; float accumulation always proceeds |

**Threats:**
- **TS-7.1** Float accumulation errors: standard += accumulates O(n*epsilon) error per bucket
- **TS-7.2** Dense path boundary (65536) is arbitrary; performance cliff at boundary
- **TS-7.3** Path state drift: one out-of-range key switches entire groupby from dense to generic path
- **TS-7.4** NaN aliasing: multiple NaN sources (0/0, sqrt(-1)) hash to same bucket via to_bits()
- **TS-7.5** Silent allocation failure fallback: OOM on dense path silently falls back to generic

**Compatibility envelope:**
- Strict mode: exact pandas-matching accumulation order (first-seen key order preserved)
- Hardened mode: compensated accumulation (Kahan) with error bounds

**Required mitigations:**
- [ ] Document float accumulation error bounds for current implementation
- [ ] Add optional Kahan summation path (deferred to AG-08 bead)
- [ ] Document DENSE_INT_KEY_RANGE_LIMIT as explicit contract
- [ ] Document NaN aliasing behavior as intentional (matches pandas groupby NaN handling)

---

### TS-8: Conformance Harness Trust (fp-conformance)

| Property | Value |
|---|---|
| Entry point | `load_fixtures()`, `capture_live_oracle_expected()` |
| Input class | Fixture files and oracle subprocess output |
| Fail-closed behavior | Serde validation on deserialization |

**Threats:**
- **TS-8.1** Fixture file OOM: no file size check before reading
- **TS-8.2** Symlink attacks: recursive directory traversal without path canonicalization
- **TS-8.3** Oracle subprocess: user-controlled script path, no sandboxing
- **TS-8.4** Oracle trust: subprocess output accepted as ground truth without secondary validation
- **TS-8.5** Fixture tampering: expected values embedded in fixture files; no integrity checking

**Compatibility envelope:**
- Strict mode: fixture files must validate against frozen schemas (bd-2gi.3)
- Hardened mode: oracle output cross-validated against fixture-embedded expectations when both available

**Required mitigations:**
- [ ] Add file size limit for fixture loading (default: 10MB per file)
- [ ] Canonicalize fixture paths before traversal
- [ ] Document oracle subprocess trust boundary explicitly
- [ ] Add fixture integrity checking via embedded hash (deferred)

---

### TS-9: RaptorQ Integrity (fp-conformance + fp-runtime)

| Property | Value |
|---|---|
| Entry point | `generate_raptorq_sidecar()`, `verify_raptorq_sidecar()` |
| Input class | Sidecar artifacts and decode proof records |
| Fail-closed behavior | Verification marks status as "failed" but does not abort |

**Threats:**
- **TS-9.1** Decode proof replay: no packet_id or nonce in DecodeProof; cross-packet reuse possible
- **TS-9.2** Verification failure not fatal: sidecar marked "failed" but processing continues
- **TS-9.3** Unbounded decode_proofs Vec: no count limit on proof records
- **TS-9.4** Source hash trusted before verification: intermediate writes assume hash is correct
- **TS-9.5** No cryptographic signature: hash alone insufficient against active adversary
- **TS-9.6** Hex string DoS: unbounded hex payload decoding can cause OOM

**Compatibility envelope:**
- Strict mode: verification failure must abort (not continue with "failed" status)
- Hardened mode: verification failure logs evidence and continues with degraded confidence

**Required mitigations:**
- [ ] Make verification failure abort in strict mode
- [ ] Add packet_id to DecodeProof to prevent cross-packet replay
- [ ] Add decode_proofs count limit (default: 1000)
- [ ] Document that SHA-256 hash provides integrity but not authenticity

---

## Adversarial Input Classes

| Class | Description | Affected Surfaces | Test Strategy |
|---|---|---|---|
| ADV-1 | Infinite/huge CSV | TS-1 | Fuzz with >1M rows, >1MB fields |
| ADV-2 | Pathological hash collisions | TS-3 | Generate IndexLabel values with colliding hashes |
| ADV-3 | All-duplicate join keys | TS-6 | Left/right indices with same repeated label |
| ADV-4 | NaN/Inf arithmetic chains | TS-4, TS-7 | Chain operations producing NaN/Inf intermediate values |
| ADV-5 | Malformed fixture files | TS-8 | Invalid JSON, huge arrays, missing fields |
| ADV-6 | Tampered RaptorQ sidecars | TS-9 | Modified hashes, replayed proofs |
| ADV-7 | Extreme type coercion | TS-2 | i64::MAX, f64::MAX, subnormal floats, -0.0 |
| ADV-8 | Compromised oracle | TS-8 | Oracle returning incorrect expected values |

---

## Mode-Split Compatibility Matrix

| Behavior | Strict Mode | Hardened Mode |
|---|---|---|
| Unknown features | Reject (fail-closed) | Evidence-driven (Allow/Reject/Repair) |
| Join cardinality | Unbounded (pandas-compatible) | Capped (configurable) |
| Division by zero | IEEE 754 Inf (currently) | IEEE 754 Inf (currently) |
| Type coercion errors | TypeError returned | TypeError returned |
| Duplicate indices | Rejected in Series.add_with_policy | Repaired with evidence record |
| Fixture validation | Schema-enforced | Schema-enforced |
| RaptorQ verification failure | Should abort (not yet enforced) | Should log and continue |
| NaN handling | NaN==NaN (pandas-compatible) | NaN==NaN (pandas-compatible) |

---

## Risk Priority Matrix

| Risk ID | Surface | Severity | Likelihood | Impact | Priority |
|---|---|---|---|---|---|
| GAP-26 | Join cardinality | CRITICAL | HIGH | OOM/DoS | P0 |
| GAP-3 | CSV row limit | CRITICAL | HIGH | OOM/DoS | P0 |
| GAP-41 | RaptorQ verification | HIGH | MEDIUM | Silent corruption | P0 |
| GAP-38 | Oracle path injection | HIGH | LOW | Code execution | P1 |
| GAP-31 | Float accumulation | MEDIUM | HIGH | Incorrect results | P1 |
| GAP-7 | Precision loss | MEDIUM | HIGH | Silent errors | P1 |
| GAP-43 | Proof replay | MEDIUM | LOW | Integrity bypass | P2 |
| GAP-11 | Hash DoS | MEDIUM | LOW | Performance DoS | P2 |
| GAP-20 | Unjustified priors | LOW | MEDIUM | Suboptimal policy | P3 |
| GAP-5 | Bool case sensitivity | LOW | HIGH | Parity break | P3 |

---

## Drift Gates

These conditions must hold for the threat matrix to remain valid:

1. `#![forbid(unsafe_code)]` remains on all crates (memory safety guarantee)
2. No new data ingestion paths bypass the type coercion lattice
3. Runtime policy mode split (strict/hardened) is the only behavioral bifurcation point
4. Index alignment via `align_union()` remains the sole alignment primitive
5. RaptorQ sidecar pairing rule (T5 from ARTIFACT_TOPOLOGY.md) is enforced

If any gate is violated, this threat matrix must be re-evaluated.
