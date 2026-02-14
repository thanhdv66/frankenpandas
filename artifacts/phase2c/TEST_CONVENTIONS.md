# Unit/Property Test Conventions + Structured Log Contract

Canonical test style guide and structured logging schema for FrankenPandas.
Frozen per bead **bd-2gi.5**.

---

## 1. Test Organization

### 1.1 Unit Tests

Unit tests live inside each crate's `src/lib.rs` as `#[cfg(test)] mod tests { ... }`.

```rust
#[cfg(test)]
mod tests {
    use super::{PublicType, public_fn};
    use fp_types::{NullKind, Scalar};  // external test deps

    #[test]
    fn operation_scenario_expected_outcome() {
        // Arrange
        let input = ...;
        // Act
        let result = public_fn(input).expect("descriptive context");
        // Assert
        assert_eq!(result.len(), 3);
    }
}
```

Group related tests with a section comment:

```rust
// === PackedBitvec ValidityMask Tests ===
```

### 1.2 Integration Tests

Integration tests live in `crates/<crate>/tests/<name>.rs`. The primary integration test crate is `fp-conformance`.

### 1.3 Property Tests

Property tests live in `crates/fp-conformance/tests/proptest_properties.rs` using the `proptest` crate (v1.10+). All property tests that span multiple crates are centralized here.

Crate-local property tests may live in `crates/<crate>/tests/proptest_<topic>.rs` if they only exercise that crate's API.

---

## 2. Naming Conventions

### 2.1 Unit Test Names

Pattern: `<operation>_<scenario_or_input>_<expected_behavior>`

All names are `snake_case`. The name should read as a sentence fragment: "operation does X when Y".

| Pattern | Example |
|---|---|
| `<op>_<behavior>` | `reindex_injects_missing_values` |
| `<subject>_<condition>_<outcome>` | `strict_mode_rejects_duplicate_indices` |
| `<prefix>_<component>_<property>` | `validity_mask_boundary_65_elements` |
| `test_csv_<scenario>` | `test_csv_quoted_fields` |

Prefixes for bead-specific test batches use the bead's AG/P2C identifier:

```
// === AG-07-T: CSV Parser Optimization Tests ===
#[test]
fn test_csv_vec_based_column_order() { ... }
```

### 2.2 Property Test Names

Pattern: `prop_<component>_<invariant>`

Property names state the invariant being checked, not the scenario:

| Good | Bad |
|---|---|
| `prop_align_union_contains_all_left_labels` | `prop_test_alignment` |
| `prop_series_add_self_doubles_values` | `prop_add_works` |
| `prop_scalar_semantic_eq_reflexive` | `prop_eq` |

### 2.3 Case IDs (Conformance Fixtures)

Pattern: `<operation>_<scenario>_<mode>`

```
series_add_alignment_union_strict
index_has_duplicates_hardened
groupby_sum_dropna_false_strict
```

### 2.4 Packet IDs

Pattern: `FP-P2C-NNN` where NNN is zero-padded.

---

## 3. Assertion Patterns

### 3.1 Standard Assertions

```rust
// Exact equality
assert_eq!(result, expected);
assert_eq!(col.values(), &[Scalar::Int64(1), Scalar::Int64(2)]);

// Inequality
assert_ne!(a, b);

// Boolean conditions
assert!(mask.get(0));
assert!(!value.is_missing(), "concrete value should not be missing");

// Pattern matching on errors
let err = op().expect_err("must fail");
assert!(matches!(err, FrameError::DuplicateIndexUnsupported));
```

### 3.2 Semantic Equality

Use `Column::semantic_eq()` for value comparison that treats NaN == NaN:

```rust
assert!(c1.semantic_eq(&c2), "column {name} not semantically equal");
```

### 3.3 Property Test Assertions

Use `prop_assert!` macros (not `assert!`) inside `proptest!` blocks:

```rust
prop_assert!(result.index().len() >= left_len, "union must contain left");
prop_assert_eq!(out.values().len(), out.index().len());
```

### 3.4 Result Handling in Tests

```rust
// Expected success: use .expect() with context
let frame = read_csv_str(input).expect("parse valid CSV");

// Expected failure: use .expect_err()
let err = op().expect_err("strict mode should reject");

// Soft failure in properties (skip instead of fail):
if index.has_duplicates() {
    return Ok(());  // skip: duplicates are out of scope for this property
}
```

### 3.5 Evidence Ledger Verification

When testing runtime policy decisions, verify ledger records:

```rust
let mut ledger = EvidenceLedger::new();
let result = op_with_policy(&input, &RuntimePolicy::strict(), &mut ledger)?;
assert_eq!(ledger.records().len(), 1);
assert_eq!(ledger.records()[0].mode, RuntimeMode::Strict);
```

---

## 4. Property Test Guidelines

### 4.1 Case Counts

| Operation complexity | Cases | Rationale |
|---|---|---|
| Scalar/index ops | 500 | Fast, high coverage |
| Series arithmetic | 200 | Moderate cost per case |
| Join/groupby | 200 | Heavy allocation |
| Serialization round-trip | 200 | I/O bound |

Configure at the `proptest!` block level:

```rust
#![proptest_config(ProptestConfig::with_cases(500))]
proptest! {
    fn prop_scalar_reflexive(s in arb_numeric_scalar()) { ... }
}
```

### 4.2 Strategy Generators

Canonical generators live at the top of `proptest_properties.rs`. Reuse them rather than creating ad-hoc strategies.

| Generator | Produces | Distribution |
|---|---|---|
| `arb_numeric_scalar()` | `Scalar` | 40% Int64, 40% Float64, 10% Null, 10% NaN |
| `arb_index_label()` | `IndexLabel` | 50% Int64(0..100), 50% Utf8("[a-e]{1,3}") |
| `arb_index_labels(len)` | `Vec<IndexLabel>` | len labels with possible duplicates |
| `arb_index(len)` | `Index` | wraps arb_index_labels |
| `arb_numeric_values(len)` | `Vec<Scalar>` | len numeric scalars |
| `arb_numeric_series(name, len)` | `Series` | full series with index |
| `arb_series_pair(max)` | `(Series, Series)` | independent lengths 1..=max |
| `arb_index_pair(max)` | `(Index, Index)` | independent lengths 1..=max |
| `arb_groupby_pair(max)` | `(Series, Series)` | small label space for grouping |

When adding a new strategy, place it alongside existing ones and follow the `arb_` prefix convention.

### 4.3 Property Categories

Every property test should fit one of these categories:

| Category | Pattern | Example |
|---|---|---|
| **Reflexivity** | `f(x, x) == identity` | `prop_scalar_semantic_eq_reflexive` |
| **Symmetry** | `f(a, b) == f(b, a)` | `prop_bitvec_and_commutative` |
| **Round-trip** | `decode(encode(x)) == x` | `prop_scalar_json_round_trip` |
| **Containment** | `output contains input` | `prop_align_union_contains_all_left_labels` |
| **Subset** | `output subset of input` | `prop_inner_join_subset_of_left_join` |
| **Consistency** | `len(a) == len(b)` | `prop_series_add_index_values_length_match` |
| **No-panic** | operation completes | `prop_series_add_hardened_no_panic` |
| **Determinism** | `f(x) == f(x)` | `prop_has_duplicates_is_deterministic` |
| **Bound** | `output.len() <= input.len()` | `prop_groupby_sum_groups_bounded_by_input` |

---

## 5. Structured Log Contract

### 5.1 Test Log Format

All tests that emit diagnostic output use `eprintln!` with this structured format:

```
[TEST] <test_name> | <key>=<value> [key=value ...] | <PASS|FAIL>
```

Fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `test_name` | string | yes | Function name (snake_case) |
| `rows` | integer | if applicable | Row count in test data |
| `cols` | integer | if applicable | Column count in test data |
| `parse_ok` | boolean | if applicable | Whether parsing succeeded |
| `dtype_per_col` | array | optional | Inferred dtype per column |
| `golden_match` | boolean | if applicable | Whether golden hash matched |
| `elapsed_ms` | integer | optional | Wall-clock time in milliseconds |
| `seed` | string | optional | Proptest seed (for reproducibility) |

Example outputs:

```
[TEST] test_csv_vec_based_column_order | rows=2 cols=3 parse_ok=true | PASS
[TEST] test_csv_mixed_dtypes | rows=3 cols=5 parse_ok=true | dtype_per_col=[int64,float64,utf8,bool,null] | PASS
[TEST] test_csv_golden_output | golden_match=true | PASS
```

### 5.2 Conformance Harness Log Format

The conformance harness emits structured JSONL to `artifacts/phase2c/drift_history.jsonl`. Each entry follows `drift_history_entry.schema.json`:

```json
{
  "packet_id": "FP-P2C-001",
  "run_ts": "2026-02-14T08:00:00Z",
  "suite": "smoke",
  "oracle_present": true,
  "fixture_count": 5,
  "passed": 5,
  "failed": 0,
  "is_green": true
}
```

### 5.3 Evidence Artifact Log Format

When a test or harness produces evidence artifacts, reference them by path relative to repo root:

```
[ARTIFACT] parity_report | path=artifacts/phase2c/FP-P2C-001/parity_report.json | sha256=<hash>
[ARTIFACT] raptorq_sidecar | path=artifacts/phase2c/FP-P2C-001/parity_report.raptorq.json | blocks=42
```

### 5.4 Fixture ID Convention

Every conformance test case has a globally unique identifier:

```
<packet_id>::<case_id>
```

Example: `FP-P2C-001::series_add_alignment_union_strict`

This ID appears in:
- Fixture JSON (`packet_id` + `case_id` fields)
- Parity report (`results[].case_id`)
- Mismatch corpus entries
- Drift history (aggregated by `packet_id`)
- Structured test logs

---

## 6. Test Data Conventions

### 6.1 Scalar Representation in Fixtures

Scalars use the canonical `shared_definitions.schema.json` format:

```json
{ "kind": "int64", "value": "42" }
{ "kind": "float64", "value": "3.14" }
{ "kind": "utf8", "value": "hello" }
{ "kind": "null", "value": "null" }
{ "kind": "null", "value": "nan" }
```

### 6.2 Index Labels in Fixtures

```json
{ "kind": "int64", "value": "1" }
{ "kind": "utf8", "value": "foo" }
```

### 6.3 Runtime Mode in Fixtures

Every fixture specifies `"mode": "strict"` or `"mode": "hardened"`.

Tests that exercise both modes should have paired fixtures:
```
series_add_alignment_union_strict
series_add_alignment_union_hardened
```

---

## 7. Required Checks Before Closing a Bead

Every bead that touches code must pass all four gates:

```bash
cargo fmt --check
cargo check --all-targets
cargo clippy --workspace --all-targets
cargo test --workspace
```

For conformance-specific beads, also run:

```bash
cargo test -p fp-conformance -- --nocapture
```

Zero clippy warnings. Zero test failures. No regressions in existing tests.

---

## 8. Test File Dependency Rules

### 8.1 Dev-Dependencies

| Dependency | Where | Purpose |
|---|---|---|
| `proptest = "1.10.0"` | fp-conformance | Property-based testing |
| `serde_json` (workspace) | Any crate needing serde tests | Serialization round-trips |

Add `serde_json` as a dev-dependency only when a crate needs serialization tests. Do not add `proptest` to individual crates; centralize property tests in `fp-conformance/tests/proptest_properties.rs`.

### 8.2 Cross-Crate Test Imports

Integration tests in `fp-conformance` can import any workspace crate. Unit tests in a crate should only test that crate's public API.

---

## 9. Proptest Seed Reproducibility

When a property test fails, proptest prints the minimal failing seed. To reproduce:

```bash
PROPTEST_SEED="<seed>" cargo test -p fp-conformance --test proptest_properties -- <test_name>
```

Record the seed in the bead comment when filing a regression.

---

## 10. Environment Variables for Test Control

| Variable | Default | Purpose |
|---|---|---|
| `PROPTEST_CASES` | (per-block) | Override case count |
| `PROPTEST_SEED` | random | Reproduce specific failure |
| `FP_ORACLE_PYTHON` | `python3` | Python binary for live oracle |
| `FP_CONFORMANCE_ROOT` | auto-detected | Override repo root for fixtures |

---

## Changelog

- **bd-2gi.5** (2026-02-14): Initial conventions document. Covers naming, assertions, property test guidelines, structured log schema, fixture ID conventions, required checks, and environment variables.
