# ASUPERSYNC Unit/Property Test + Structured Logging Evidence

Bead: `bd-2gi.27.5` [ASUPERSYNC-E]  
Subsystem: `fp-runtime` ASUPERSYNC decision/test layer  
Primary file: `crates/fp-runtime/src/lib.rs`

---

## 1. Scope

This evidence note documents:

1. Unit and property-style invariants added for ASUPERSYNC decision behavior.
2. Deterministic structured test-log schema and required fields.
3. Reproducible commands for local/CI replay.
4. Coverage floor linkage and enforcement intent.

The goal is to prove behavioral stability for nominal, edge, and adversarial cases while preserving strict/hardened mode boundaries.

---

## 2. Test Inventory

### 2.1 Structured Log Contract Tests

| Test | Purpose | Class |
|---|---|---|
| `asupersync_structured_log_contains_required_fields` | Verifies log payload includes `packet_id`, `case_id`, `mode`, `seed`, `trace_id`, `assertion_path`, `result`, `replay_cmd` | unit |
| `asupersync_structured_log_is_deterministic_for_same_inputs` | Verifies same inputs produce byte-identical JSON payloads | property-style determinism |

### 2.2 Strict/Hardened Invariant Tests

| Test | Purpose | Class |
|---|---|---|
| `asupersync_property_strict_unknown_feature_always_rejects` | For 128 seeds, strict-mode unknown feature decisions always reject and produce valid structured logs | property |
| `asupersync_property_hardened_over_cap_forces_repair` | For 256 seeds and mixed row counts, hardened mode forces `Repair` when `rows > cap` | property |
| `asupersync_property_decision_metrics_are_finite_and_bounded` | For 128 seeds, posterior remains in `[0,1]` and expected-loss metrics are finite | property |
| `asupersync_adversarial_extreme_join_estimate_remains_repair_and_loggable` | Adversarial `usize::MAX` join estimate remains bounded, auditable, and repair-directed | adversarial |

### 2.3 Existing Baseline Tests Preserved

The following pre-existing tests continue to validate ASUPERSYNC-adjacent behavior:

- `strict_mode_fails_closed_for_unknown_features`
- `hardened_mode_repairs_large_join_estimates`
- `decision_card_is_renderable_for_ftui_consumers`
- existing AG-09 conformal calibration suite

---

## 3. Structured Log Schema (Deterministic)

Schema used in test helpers:

```json
{
  "packet_id": "ASUPERSYNC-E",
  "case_id": "<test_case_name>",
  "mode": "strict|hardened",
  "seed": 12345,
  "trace_id": "ASUPERSYNC-E:<case_id>:<seed_hex>",
  "assertion_path": "ASUPERSYNC-E/<assertion_name>",
  "result": "pass|fail|check",
  "replay_cmd": "cargo test -p fp-runtime -- <case_id> --nocapture"
}
```

Determinism constraints:

1. `trace_id` is derived by pure formatting from `packet_id`, `case_id`, and `seed`.
2. `replay_cmd` is derived by pure formatting from `case_id`.
3. Serializing the same log struct with the same inputs must produce identical JSON bytes.

---

## 4. Reproducible Commands

### 4.1 Targeted Replay (single test)

```bash
cargo test -p fp-runtime -- asupersync_property_hardened_over_cap_forces_repair --nocapture
```

### 4.2 Bead Surface Replay (all ASUPERSYNC-E additions)

```bash
cargo test -p fp-runtime -- asupersync_ --nocapture
```

### 4.3 Full crate replay

```bash
cargo test -p fp-runtime -- --nocapture
```

---

## 5. Coverage + Reliability Linkage

Coverage/flake policy source:

- `artifacts/phase2c/COVERAGE_FLAKE_BUDGETS.md`

Relevant floor:

- `fp-runtime` line coverage floor: `75%` (target `85%`).

ASUPERSYNC-E tests increase coverage in:

1. strict fail-closed override paths
2. hardened join-cap override paths
3. decision metric sanity boundaries
4. deterministic logging helpers for replay-grade observability

---

## 6. Invariant Checklist

| Invariant | Status | Proof Surface |
|---|---|---|
| Strict unknown features are fail-closed | enforced | `asupersync_property_strict_unknown_feature_always_rejects` |
| Hardened over-cap join admission repairs | enforced | `asupersync_property_hardened_over_cap_forces_repair` |
| Decision metrics remain finite/bounded | enforced | `asupersync_property_decision_metrics_are_finite_and_bounded` |
| Extreme adversarial row estimate remains bounded | enforced | `asupersync_adversarial_extreme_join_estimate_remains_repair_and_loggable` |
| Structured logs are schema-complete | enforced | `asupersync_structured_log_contains_required_fields` |
| Structured logs are deterministic | enforced | `asupersync_structured_log_is_deterministic_for_same_inputs` |

---

## 7. Known Gaps and Next Steps

1. Feature-gated direct `outcome_to_action` variant mapping tests under `asupersync` feature are still a follow-up.
2. Closed (2026-02-15): cross-crate differential/E2E logging parity for ASUPERSYNC is now wired through `fp-conformance` forensic case events (`scenario_id`, `trace_id`, `step_id`, `seed`, `assertion_path`, `result`, `replay_cmd`, `replay_key`, `mismatch_class`) and replay-oriented failure digests.
3. Feature-on CI wiring remains constrained by upstream `asupersync` dependency compilation blockers; continue using default-path conformance gates plus targeted feature-path smoke once dependency is fixed.
4. Closed (2026-02-15): sidecar/decode-proof evidence binding now enforces proof-hash pairing (`parity_report.decode_proof.json` hashes must exist in sidecar envelope proofs with `sha256:` prefix) via `verify_packet_sidecar_integrity()` and regression test `sidecar_integrity_fails_when_decode_proof_hash_mismatches_sidecar`.
