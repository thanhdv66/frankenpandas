# COMPAT_CLOSURE Optimization + Isomorphism Evidence

Bead: `bd-2gi.29.8` [COMPAT-CLOSURE-H]  
Subsystem: `fp-conformance` compat-closure scenario synthesis  
Primary files:
- `crates/fp-conformance/src/lib.rs`
- `artifacts/phase2c/PERFORMANCE_BASELINES.md`
- `artifacts/perf/ROUND8_ISOMORPHISM_PROOF.md`

---

## 1. Scope

This bead delivers the COMPAT-CLOSURE profile-driven optimization loop for scenario synthesis with explicit baseline/re-baseline and behavior-isomorphism evidence.

Single optimization lever applied:

1. Replace dual per-trace metadata maps (`operation_by_trace` + `mode_by_trace`) with one consolidated trace index keyed by `trace_id`.
2. Preserve deterministic output contract by keeping final step sorting and report aggregation behavior unchanged.

---

## 2. Optimization Implementation

### 2.1 New internal shape

- Added `CompatClosureTraceMetadata { operation, mode }`.
- Added optimized builder path:
  - `build_compat_closure_e2e_scenario_report_optimized_with_stats(...)`
- Added baseline proof path (test-only):
  - `build_compat_closure_e2e_scenario_report_baseline_with_stats(...)`
- Public API stays stable:
  - `build_compat_closure_e2e_scenario_report(...)` now routes through the optimized builder.

### 2.2 Determinism preservation

- Shared post-processing path (`finalize_compat_closure_e2e_scenario_report`) performs stable sort + summary counts.
- Fault-injection step synthesis remains unchanged via shared helper (`append_fault_injection_steps`).

---

## 3. Isomorphism + Profiling Tests

Added tests in `crates/fp-conformance/src/lib.rs`:

1. `compat_closure_e2e_scenario_optimized_path_is_isomorphic_to_baseline`
   - Verifies full report equality (`optimized == baseline`) on amplified forensic workload.
   - Verifies optimization actually reduces metadata index nodes and lookup steps.

2. `compat_closure_e2e_scenario_profile_snapshot_reports_index_delta`
   - Runs 64 repeated baseline/optimized builds.
   - Captures `p50/p95/p99` latency for both paths.
   - Captures metadata index-node and lookup-step totals as allocation/workload proxies.

---

## 4. Measured Snapshot (Remote via `rch`)

Command:

```bash
rch exec -- cargo test -p fp-conformance --lib \
  compat_closure_e2e_scenario_profile_snapshot_reports_index_delta -- --nocapture
```

Observed output (2026-02-15):

- Baseline `p50/p95/p99` (ns): `4167358 / 5477594 / 9182879`
- Optimized `p50/p95/p99` (ns): `3874091 / 6674277 / 7587292`
- Trace metadata index nodes (64 iterations, amplified workload): `65792 -> 32896`
- Trace metadata lookup steps (64 iterations, amplified workload): `65792 -> 32896`

Interpretation:

- Optimization lever reduced metadata index and lookup work by 50% for the stress path.
- Latency distribution remains in the same operating band with lower p50 and p99 in this snapshot while preserving exact output equality.

---

## 5. Replay + Gate Commands

Isomorphism proof:

```bash
rch exec -- cargo test -p fp-conformance --lib \
  compat_closure_e2e_scenario_optimized_path_is_isomorphic_to_baseline -- --nocapture
```

Scenario test surface:

```bash
rch exec -- cargo test -p fp-conformance --lib compat_closure_e2e_scenario -- --nocapture
```

Packet artifact path consistency check:

```bash
rch exec -- cargo run -p fp-conformance --bin fp-conformance-cli -- \
  --packet-id FP-P2C-001 --write-artifacts --require-green --write-drift-history
```

Workspace gates:

```bash
rch exec -- cargo check --workspace --all-targets
rch exec -- cargo clippy --workspace --all-targets -- -D warnings
rch exec -- cargo fmt --check
```

---

## 6. Evidence Chain

1. Code-level optimization + baseline comparator in `crates/fp-conformance/src/lib.rs`.
2. Quantified before/after metrics in `artifacts/phase2c/PERFORMANCE_BASELINES.md` (Round 8 entry).
3. Isomorphism proof narrative in `artifacts/perf/ROUND8_ISOMORPHISM_PROOF.md`.
4. Existing conformance artifact pipeline unchanged (`parity_report`, RaptorQ sidecar, decode proof, gate result, mismatch corpus, drift history).
