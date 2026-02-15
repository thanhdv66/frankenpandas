# COMPAT_CLOSURE Contract Table + Strict/Hardened Policy Matrix

Bead: `bd-2gi.29.2` [COMPAT-CLOSURE-B]  
Subsystem: Drop-in compatibility closure contract surface for alignment, dtype/null semantics, join/groupby contracts, differential gates, and migration evidence  
Source anchors: `artifacts/phase2c/COMPAT_CLOSURE_ANCHOR_MAP.md` (bd-2gi.29.1), `artifacts/phase2c/COMPAT_CLOSURE_THREAT_MODEL.md` (bd-2gi.29.3)

---

## 1. Summary Contract (Tabular)

| Field | Contract |
|---|---|
| packet_id | `COMPAT-CLOSURE-B` |
| input_contract | Typed scalar/index payloads, packet fixtures, parity gate configs, runtime policy (`strict`/`hardened`), oracle mode (`fixture`/`live`) |
| output_contract | Deterministic operation outputs + machine-readable parity/gate artifacts + replayable forensics + drift history ledger |
| error_contract | Fail-closed behavior on unknown operations/schema mismatches/gate mismatches; strict duplicate-index rejection; explicit mismatch corpora |
| null_contract | Deterministic null/NaN propagation and dtype-aware missing sentinels (`Null` vs `NaN`) |
| strict_mode_policy | Maximize behavioral parity, fail closed on unknown features, no behavior-altering repairs |
| hardened_mode_policy | Preserve API contract with bounded defensive repairs and explicit evidence logging |
| excluded_scope | Any unimplemented pandas family (e.g., full `loc/iloc`, full nanops matrix, full DataFrame merge/concat closure, datetime/window/MultiIndex) is excluded from closure claim unless green evidence is present |
| oracle_tests | Packet differential suites + live-oracle adapter ops + property tests + E2E failure-forensics replay hooks |
| performance_sentinels | Compatibility-sensitive hotpaths must preserve semantics while meeting profile budget expectations (alignment/join/groupby) |
| compatibility_risks | Silent drift from optimization, weak provenance in live-oracle fallback mode, closure over-claim risk when feature matrix gaps remain |
| raptorq_artifacts | Per-packet sidecar + scrub + decode proof + drift history hash chain required for durable closure claims |

---

## 2. Machine-Checkable Contract Registry

Legend:
- `strict_expected`: required strict-mode result
- `hardened_expected`: required hardened-mode result
- `evidence`: artifact or code anchor proving the contract

| Contract ID | Domain | Input Contract | Output Contract | Error Contract | strict_expected | hardened_expected | Evidence |
|---|---|---|---|---|---|---|---|
| `CC-001` | DType promotion | Pair of dtypes in `{null,bool,int64,float64,utf8}` | Common dtype from promotion table | Unsupported pair -> `IncompatibleDtypes` | Same as hardened | Same as strict | `crates/fp-types/src/lib.rs:130-144` |
| `CC-002` | Scalar missingness | `Scalar` value | `is_missing`/`is_nan` deterministic flags | Non-numeric to numeric cast -> typed error | Same as hardened | Same as strict | `crates/fp-types/src/lib.rs:47-109` |
| `CC-003` | Alignment union | Two indexes | Union index + left/right first positions | Invalid vector lengths -> `InvalidAlignmentVectors` | Same as hardened | Same as strict | `crates/fp-index/src/lib.rs:519-556` |
| `CC-004` | Series arithmetic alignment | Two series, op in `{+,-,*,/}` | Reindexed arithmetic output on union index | Duplicate labels may reject under strict | Reject duplicates | Allow with evidence record (policy-mediated) | `crates/fp-frame/src/lib.rs:107-154` |
| `CC-005` | Numeric null propagation | Two aligned numeric columns | Missing operands propagate missing output; NaN preserved | Length mismatch error | Same as hardened | Same as strict | `crates/fp-columnar/src/lib.rs:667-725` |
| `CC-006` | Join semantics | `JoinType` + left/right series | Deterministic row/index/value materialization | Invalid downstream coercions bubble as errors | Same as hardened | Same as strict (except admission policy use-site) | `crates/fp-join/src/lib.rs:58-257` |
| `CC-007` | GroupBy semantics | key/value series + `dropna` | First-seen key order + sum aggregation + dropna contract | Type/coercion errors propagate | Same as hardened | Same as strict | `crates/fp-groupby/src/lib.rs:58-239` |
| `CC-008` | Unknown feature decision | Subject + detail + ledger | Decision action + appended decision record | N/A | Force `Reject` | Bayesian argmin (`Allow`/`Repair`/`Reject`) | `crates/fp-runtime/src/lib.rs:172-204` |
| `CC-009` | Join admission decision | Estimated rows + optional cap + ledger | Decision action + appended decision record | N/A | Bayesian argmin (no cap override) | Force `Repair` when `estimated_rows > cap` | `crates/fp-runtime/src/lib.rs:206-249` |
| `CC-010` | Differential taxonomy | Actual vs expected results | Drift records classified by category+severity | N/A | Same as hardened | Same as strict | `crates/fp-conformance/src/lib.rs:265-367` |
| `CC-011` | Parity greenness | Packet report | `green` iff `failed == 0 && fixture_count > 0` | N/A | Same as hardened | Same as strict | `crates/fp-conformance/src/lib.rs:380-385` |
| `CC-012` | Packet gate evaluation | Report + `parity_gate.yaml` | `PacketGateResult` with pass/reasons | Config mismatch/budget violation -> fail reason | Same thresholds as config strict section | Same thresholds as config hardened section | `crates/fp-conformance/src/lib.rs:846-940` |
| `CC-013` | Gate enforcement CLI | Reports + `--require-green` | Non-zero exit on any parity/gate failure | Aggregated failure error payload | Same as hardened | Same as strict | `crates/fp-conformance/src/lib.rs:620-650`, `crates/fp-conformance/src/bin/fp-conformance-cli.rs:75-77` |
| `CC-014` | Packet artifact bundle | Packet report | parity/gate/mismatch/sidecar/decode files | Serialization/IO failure -> hard error | Same as hardened | Same as strict | `crates/fp-conformance/src/lib.rs:774-844` |
| `CC-015` | Drift history ledger | Reports list | Append-only JSONL entries with report hash | IO/serialization failure -> hard error | Same as hardened | Same as strict | `crates/fp-conformance/src/lib.rs:652-690` |
| `CC-016` | Live oracle operation dispatch | JSON request with op | Normalized expected output payload | Unsupported op/kind -> `OracleError` + non-zero | Same as hardened | Same as strict (modulo fallback switch) | `crates/fp-conformance/oracle/pandas_oracle.py:70-97`, `crates/fp-conformance/oracle/pandas_oracle.py:276-319` |
| `CC-017` | Strict legacy oracle import | `--strict-legacy` + legacy root | Legacy pandas import required | Import failure hard error unless fallback explicitly allowed | Fail closed by default | Same unless operator enables fallback | `crates/fp-conformance/oracle/pandas_oracle.py:57-68` |
| `CC-018` | Failure forensics replay | E2E result with failures | Deterministic digest + replay command + artifact path | Missing artifact path tolerated with partial digest | Same as hardened | Same as strict | `crates/fp-conformance/src/lib.rs:3782-3837`, `artifacts/phase2c/FAILURE_FORENSICS_UX.md:19-33` |
| `CC-019` | Scenario-grounded operator workflows | Scenario corpus | Canonical multi-step journeys mapped to packets | Out-of-scope scenarios explicitly excluded | Same as hardened | Same as strict | `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:10-35`, `artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:333-391` |
| `CC-020` | Parity closure claim boundary | Feature parity matrix + packet evidence | Closure claim only for evidence-backed green families | Over-claim is contract violation | Same as hardened | Same as strict | `FEATURE_PARITY.md:11-47` |

---

## 3. Strict/Hardened Policy Matrix

### 3.1 Runtime Policy Boundary Table

| Policy Surface | Strict | Hardened | Boundary Type |
|---|---|---|---|
| Unknown features | Force reject (`fail_closed_unknown_features=true`) | Bayesian argmin (repair/allow possible) | Behavior-affecting mode split |
| Join admission over cap | No cap override path | Forced repair when above configured cap | Behavior-affecting mode split |
| Decision ledger append | Always append decision record | Always append decision record | Shared invariant |
| Decision equation | Bayesian expected-loss argmin | Bayesian expected-loss argmin + cap override | Shared core + hardened override |
| `RuntimePolicy::default()` | strict | N/A | Initialization invariant |

### 3.2 Series Arithmetic Boundary Table

| Surface | Strict | Hardened | Boundary Type |
|---|---|---|---|
| Duplicate labels detected before arithmetic | Reject operation (`DuplicateIndexUnsupported`) | Continue with policy evidence | Behavior-affecting mode split |
| Union alignment and reindex | Required | Required | Shared invariant |
| Null/NaN propagation | Deterministic propagation | Deterministic propagation | Shared invariant |

### 3.3 Gate/Conformance Boundary Table

| Surface | Strict | Hardened | Boundary Type |
|---|---|---|---|
| Gate budget axis | `strict_failed` and strict drift budgets | `hardened_failed` and divergence budget | Behavior-affecting mode split |
| Packet artifact emission | Required | Required | Shared invariant |
| Drift history append | Required | Required | Shared invariant |
| CLI `--require-green` behavior | Hard fail on strict gate/report failure | Hard fail on hardened gate/report failure | Shared release gate invariant |

### 3.4 Oracle Boundary Table

| Surface | Strict | Hardened | Boundary Type |
|---|---|---|---|
| Unsupported operation/kind | Hard `OracleError` | Hard `OracleError` | Shared invariant |
| Legacy import failure | Hard fail by default in strict-legacy mode | May permit fallback only when explicitly allowed | Operator-configurable boundary |

---

## 4. Fail-Closed Rules (Explicit)

| Rule ID | Rule | Enforced By | Evidence |
|---|---|---|---|
| `FC-001` | Unknown incompatible features must fail closed in strict mode | `RuntimePolicy::strict` + `decide_unknown_feature` override | `crates/fp-runtime/src/lib.rs:155-200` |
| `FC-002` | Unsupported oracle operation must fail request | Oracle `dispatch` default error path | `crates/fp-conformance/oracle/pandas_oracle.py:276-290` |
| `FC-003` | Unsupported oracle schema kind must fail request | `label_from_json`/`scalar_from_json` | `crates/fp-conformance/oracle/pandas_oracle.py:70-97` |
| `FC-004` | Gate/report mismatch must fail packet gate | `evaluate_parity_gate` mismatch checks | `crates/fp-conformance/src/lib.rs:887-901` |
| `FC-005` | Non-green parity/gate set must fail enforcement mode | `enforce_packet_gates` + CLI `--require-green` | `crates/fp-conformance/src/lib.rs:620-650`, `crates/fp-conformance/src/bin/fp-conformance-cli.rs:75-77` |
| `FC-006` | Packet artifact write failures must fail run | `write_packet_artifacts` returns `HarnessError` | `crates/fp-conformance/src/lib.rs:762-844` |
| `FC-007` | Drift-history append failure must fail append step | `append_phase2c_drift_history` error propagation | `crates/fp-conformance/src/lib.rs:652-690` |
| `FC-008` | Strict legacy import failure must hard fail unless fallback explicitly enabled | oracle import path | `crates/fp-conformance/oracle/pandas_oracle.py:57-68` |
| `FC-009` | Closure claim cannot include unresolved feature families without green evidence | parity matrix boundary | `FEATURE_PARITY.md:11-47` |

---

## 5. Contract-to-Evidence Mapping

### 5.1 Artifact Evidence Required Per Contract Row

| Contract IDs | Required Artifacts | Why |
|---|---|---|
| `CC-010` to `CC-015` | `parity_report.json`, `parity_gate_result.json`, `parity_mismatch_corpus.json` | Prove differential and gate outcomes |
| `CC-014` to `CC-015` | `parity_report.raptorq.json`, `parity_report.decode_proof.json`, `drift_history.jsonl` | Prove durability + replay continuity |
| `CC-016` to `CC-017` | Live-oracle run logs + adapter stderr/stdout | Prove strict/fallback oracle behavior |
| `CC-018` | Failure digest output + replay command execution trace | Prove one-command reproducibility |
| `CC-019` | Scenario mapping coverage report | Prove user workflow surfaces are represented |
| `CC-020` | Updated parity matrix + packet evidence ledger | Prevent closure over-claim |

### 5.2 Minimum Verification Loop

1. Produce packet reports and artifacts in enforcement mode.  
2. Verify gate results for strict and hardened budgets.  
3. Confirm mismatch corpus and forensics replay for any failure.  
4. Append and hash drift history entries.  
5. Reconcile parity matrix statuses against generated evidence set.

---

## 6. Machine-Readable Check Strategy

Use this deterministic checklist for downstream automation:

| Check ID | Predicate | Pass Condition |
|---|---|---|
| `CHK-001` | `CC-001..CC-007` code anchors exist | All referenced files and symbols resolve |
| `CHK-002` | strict/hardened split contracts (`CC-008`,`CC-009`) | Mode-dependent branches and overrides present |
| `CHK-003` | differential taxonomy contracts (`CC-010`,`CC-011`) | All required drift enums/struct fields present |
| `CHK-004` | gate contracts (`CC-012`,`CC-013`) | `evaluate_parity_gate` + `--require-green` pathways present |
| `CHK-005` | artifact durability contracts (`CC-014`,`CC-015`) | Sidecar + decode + mismatch + drift outputs emitted |
| `CHK-006` | oracle fail-closed contracts (`CC-016`,`CC-017`) | Unsupported operation and strict import failures non-zero |
| `CHK-007` | replay contract (`CC-018`) | Failure digest includes replay command format |
| `CHK-008` | scenario mapping contract (`CC-019`) | Scenario corpus includes packet mapping table |
| `CHK-009` | closure boundary contract (`CC-020`) | No closure claim for rows marked `in_progress`/`parity_gap` in feature matrix |

---

## 7. Open Compatibility Gaps (Explicitly Non-Closed)

These remain out of closure claim until green evidence is attached:

1. Full `loc/iloc` parity (`FEATURE_PARITY.md:15`).
2. Full DataFrame merge/concat closure (`FEATURE_PARITY.md:17`).
3. Full nanops matrix (`FEATURE_PARITY.md:18`).
4. Full CSV parser/formatter parity matrix (`FEATURE_PARITY.md:19`).
5. Scenario-declared out-of-scope families: datetime, window, MultiIndex (`artifacts/phase2c/USER_WORKFLOW_SCENARIOS.md:378-391`).

---

## 8. Downstream Hand-Off (for `bd-2gi.29.3` and `bd-2gi.29.4`)

1. `bd-2gi.29.3` should consume `FC-*` and `CC-*` rows to build threat model entries with explicit attack/failure pathways.
2. `bd-2gi.29.4` should translate `CC-*` rows into module boundaries and trait contracts, preserving strict/hardened split semantics and evidence hooks.
3. Any new compatibility surface must add:
   - one `CC-*` row
   - strict/hardened expected outcomes
   - fail-closed classification (`FC-*` or explicit `N/A`)
   - evidence mapping.
4. `bd-2gi.29.3` output is now materialized in `artifacts/phase2c/COMPAT_CLOSURE_THREAT_MODEL.md`; downstream beads should treat it as the authoritative threat/risk register for closure sign-off.
5. `bd-2gi.29.4` output is now materialized in `artifacts/phase2c/COMPAT_CLOSURE_RUST_INTEGRATION.md`; downstream beads should treat it as the authoritative module-boundary and sequencing plan for `29.5..29.9`.
