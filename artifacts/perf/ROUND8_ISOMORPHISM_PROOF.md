# ROUND8 Isomorphism Proof

Round: 8  
Bead: `bd-2gi.29.8` [COMPAT-CLOSURE-H]  
Target: `build_compat_closure_e2e_scenario_report`

## Change

Single optimization lever:

- Replace two trace metadata indices (`BTreeMap<trace_id, FixtureOperation>` and `BTreeMap<trace_id, RuntimeMode>`) with one consolidated index:
  - `HashMap<trace_id, CompatClosureTraceMetadata { operation, mode }>`.

No API behavior changes were introduced.

## Behavior Checks

1. **Ordering**: scenario steps are still sorted by `(scenario_id, step_id, mode)`.
2. **Tie-breaking**: same deterministic mode slug ordering (`strict` before `hardened`) retained.
3. **Null/semantic behavior**: defaults remain unchanged for missing metadata (`mode=strict`, `command_or_api=unknown_operation`).
4. **Failure-injection synthesis**: unchanged via shared helper path.

## Validation Artifacts

Isomorphism equality check:

```bash
rch exec -- cargo test -p fp-conformance --lib \
  compat_closure_e2e_scenario_optimized_path_is_isomorphic_to_baseline -- --nocapture
```

Profile snapshot (baseline vs optimized):

```bash
rch exec -- cargo test -p fp-conformance --lib \
  compat_closure_e2e_scenario_profile_snapshot_reports_index_delta -- --nocapture
```

Broader scenario test surface:

```bash
rch exec -- cargo test -p fp-conformance --lib compat_closure_e2e_scenario -- --nocapture
```

## Observed Result (2026-02-15)

From profile snapshot command:

- Baseline `p50/p95/p99` (ns): `4167358 / 5477594 / 9182879`
- Optimized `p50/p95/p99` (ns): `3874091 / 6674277 / 7587292`
- Trace metadata index nodes: `65792 -> 32896`
- Trace metadata lookup steps: `65792 -> 32896`

From isomorphism command:

- Output equality: `optimized == baseline` (pass)
- No semantic drift detected in scenario-step payloads.

## Conclusion

ROUND8 preserves compat-closure behavior while reducing metadata indexing and lookup workload under amplified scenario synthesis. This satisfies the round-level requirement: baseline -> one lever -> isomorphism proof -> re-baseline.
