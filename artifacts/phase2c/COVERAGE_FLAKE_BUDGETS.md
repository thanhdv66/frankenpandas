# Coverage/Flake Budgets + Reliability Gates

Reliability SLOs for FrankenPandas unit, property, and E2E test suites.
Frozen per bead **bd-2gi.22**.

---

## 1. Coverage Floors

### 1.1 Line Coverage Targets

| Crate | Floor | Target | Measurement |
|---|---|---|---|
| fp-types | 90% | 95% | `cargo llvm-cov --package fp-types` |
| fp-columnar | 85% | 92% | `cargo llvm-cov --package fp-columnar` |
| fp-index | 85% | 92% | `cargo llvm-cov --package fp-index` |
| fp-frame | 80% | 90% | `cargo llvm-cov --package fp-frame` |
| fp-expr | 75% | 85% | `cargo llvm-cov --package fp-expr` |
| fp-groupby | 80% | 90% | `cargo llvm-cov --package fp-groupby` |
| fp-join | 80% | 90% | `cargo llvm-cov --package fp-join` |
| fp-io | 80% | 90% | `cargo llvm-cov --package fp-io` |
| fp-runtime | 75% | 85% | `cargo llvm-cov --package fp-runtime` |
| fp-conformance | 60% | 75% | `cargo llvm-cov --package fp-conformance` |

**Floor** = hard gate; CI must fail below this.
**Target** = soft goal; CI warns between floor and target.

### 1.2 Branch Coverage

Branch coverage is tracked but not gated in Phase-2C. Target: 70% workspace-wide.

### 1.3 Conformance Fixture Coverage

Every packet (FP-P2C-NNN) must have:
- Minimum 3 fixture cases per operation
- At least 1 strict-mode and 1 hardened-mode case per operation
- At least 1 edge case (null propagation, empty input, or boundary value)

### 1.4 Property Test Coverage

Every public function that accepts user-controlled input must have at least one property test in `proptest_properties.rs`. Property tests cover:
- Return type invariants (length consistency, containment)
- No-panic guarantees for valid inputs
- Round-trip properties (serialize/deserialize, read/write)
- Algebraic properties (commutativity, associativity where applicable)

---

## 2. Flake Ceilings

### 2.1 Flake Budget

| Suite | Max Flake Rate | Measurement Window |
|---|---|---|
| Unit tests | 0% (zero tolerance) | Per run |
| Property tests (proptest) | 0.1% (1 in 1000 runs) | Rolling 100 runs |
| E2E / conformance | 0% (zero tolerance) | Per run |
| Integration (live oracle) | 1% (oracle subprocess) | Rolling 50 runs |

### 2.2 Flake Detection

A test is considered flaky if it produces different outcomes on the same code revision. Detection method:

```bash
# Run test suite N times and check for non-determinism
for i in $(seq 1 10); do cargo test --workspace 2>&1 | tail -1; done
```

If any run produces a different result, the test is flaky and must be:
1. Marked `#[ignore]` with comment `// FLAKY: <ticket>`
2. Filed as a bead with label `flaky`
3. Fixed within 48 hours or removed

### 2.3 Property Test Seed Stability

Property tests must produce deterministic results for a given seed. Use `ProptestConfig::with_source(...)` when debugging:

```rust
proptest!(ProptestConfig::with_cases(500), |(input in strategy())| {
    // deterministic for any seed
});
```

---

## 3. Runtime Budget Guardrails

### 3.1 Test Suite Time Budgets

| Suite | Budget (debug) | Budget (release) | Gate |
|---|---|---|---|
| `cargo test --workspace` (all) | 120s | 60s | Hard |
| `cargo test -p fp-conformance` | 30s | 15s | Hard |
| Property tests (proptest) | 60s | 30s | Soft |
| Single unit test | 5s | 2s | Hard |
| Single property test | 10s | 5s | Hard |

If a test exceeds its budget, it must be:
1. Profiled to identify the bottleneck
2. Optimized or split into smaller tests
3. Documented with justification if the budget cannot be met

### 3.2 Memory Budgets

| Suite | Max RSS | Gate |
|---|---|---|
| Full workspace test | 2 GB | Soft |
| Single test | 256 MB | Hard |
| Property test (per case) | 64 MB | Hard |

### 3.3 Fixture Size Budgets

| Fixture Type | Max Size | Gate |
|---|---|---|
| Single fixture JSON | 100 KB | Hard |
| Fixture manifest | 1 MB | Hard |
| Parity report | 10 MB | Soft |
| RaptorQ sidecar | 50 MB | Soft |

---

## 4. Reliability Gate Definitions

### Gate G1: Compilation Gate

```bash
cargo check --all-targets
cargo fmt --check
```

Pass criteria: zero errors, zero formatting violations.

### Gate G2: Lint Gate

```bash
cargo clippy --workspace --all-targets
```

Pass criteria: zero warnings.

### Gate G3: Unit Test Gate

```bash
cargo test --workspace --lib
```

Pass criteria: all tests pass, no flakes, within time budget.

### Gate G4: Property Test Gate

```bash
cargo test --workspace --test proptest_properties
```

Pass criteria: all properties hold for configured case counts, within time budget.

### Gate G5: Integration Test Gate

```bash
cargo test --workspace --test smoke
cargo test --workspace --test '*' -- --ignored  # if any gated tests
```

Pass criteria: all integration tests pass.

### Gate G6: Conformance Gate

```bash
cargo test -p fp-conformance -- --nocapture
```

Pass criteria: all packet fixtures pass, parity gates green, no critical drift.

### Gate G7: Coverage Gate

```bash
cargo llvm-cov --workspace --fail-under-lines <floor>
```

Pass criteria: line coverage >= floor for each crate.

### Gate G8: E2E Gate

```rust
use fp_conformance::{E2eConfig, NoopHooks, run_e2e_suite};
let config = E2eConfig::default_all_phases();
let report = run_e2e_suite(&config, &mut NoopHooks).expect("e2e");
assert!(report.is_green());
```

Pass criteria: E2E report is green (all fixtures pass, all gates pass).

### Gate Ordering

Gates execute in order G1..G8. A failing gate blocks subsequent gates.

```
G1 (compile) -> G2 (lint) -> G3 (unit) -> G4 (property) -> G5 (integration) -> G6 (conformance) -> G7 (coverage) -> G8 (e2e)
```

---

## 5. Regression Policy

### 5.1 No Test Removal Without Replacement

Tests may only be removed if:
- The functionality they test has been removed
- They are replaced by a strictly superior test (higher coverage, better invariants)
- They are documented as obsolete in a bead comment

### 5.2 Coverage Ratchet

Coverage floors only increase, never decrease. If a bead raises coverage above the floor, the floor is bumped to the new value (rounded down to nearest 5%).

### 5.3 Flake Escalation

| Flake Count (rolling 100 runs) | Action |
|---|---|
| 1 | Log and monitor |
| 2 | File bead, investigate root cause |
| 3+ | `#[ignore]` the test, P1 fix bead |

---

## 6. Measurement Commands

### Quick Health Check

```bash
cargo fmt --check && cargo clippy --workspace --all-targets && cargo test --workspace
```

### Full Reliability Audit

```bash
# Gates G1-G6
cargo fmt --check
cargo clippy --workspace --all-targets
cargo test --workspace
cargo test -p fp-conformance -- --nocapture

# Gate G7 (requires cargo-llvm-cov)
cargo llvm-cov --workspace --summary-only

# Gate G8
cargo test -p fp-conformance --lib -- e2e_suite_runs_full_pipeline
```

---

## Changelog

- **bd-2gi.22** (2026-02-14): Initial reliability SLOs. Defines coverage floors per crate, flake ceilings per suite, runtime/memory budgets, 8-gate reliability pipeline (G1..G8), regression policy, and coverage ratchet.
