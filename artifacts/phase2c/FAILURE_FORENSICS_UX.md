# Failure Forensics UX + Artifact Index

Human-first failure diagnostics design for FrankenPandas conformance testing.
Frozen per bead **bd-2gi.21**.

---

## 1. Design Principles

1. **Concise first**: Show the minimum information needed to understand what failed
2. **Actionable second**: Every failure includes a replay command
3. **Deep third**: Full details available in linked artifacts
4. **Deterministic**: Artifact IDs are hash-based, not timestamp-based

---

## 2. Failure Digest Format

When a conformance test fails, the system emits a `FailureDigest` with this display format:

```
FAIL FP-P2C-001::series_add_alignment_union_strict [SeriesAdd/Strict]
  Class:    value_critical
  ReplayKey: FP-P2C-001/series_add_alignment_union_strict/strict
  Trace:    FP-P2C-001:series_add_alignment_union_strict:strict
  Mismatch: expected Int64(10), got Float64(10.0) at index[0]
  Replay:   cargo test -p fp-conformance -- series_add_alignment_union_strict --nocapture
  Artifact: artifacts/phase2c/FP-P2C-001/parity_mismatch_corpus.json
```

Fields:
- **FAIL line**: `{packet_id}::{case_id} [{operation}/{mode}]`
- **Class**: deterministic mismatch taxonomy alias (`<category>_<level>`)
- **ReplayKey**: deterministic replay identity (`{packet_id}/{case_id}/{mode}`)
- **Trace**: deterministic trace identity (`{packet_id}:{case_id}:{mode}`)
- **Mismatch**: First 200 characters of the mismatch description
- **Replay**: Exact cargo command to reproduce the failure
- **Artifact**: Path to the mismatch corpus for full details

---

## 3. Forensics Report Format

The `FailureForensicsReport` provides a summary of all failures in an E2E run:

### All-Green Output
```
ALL GREEN: 25/25 fixtures passed
```

### Failure Output
```
FAILURES: 2/25 fixtures failed

  1. FAIL FP-P2C-001::case_a [SeriesAdd/Strict]
       Mismatch: value drift at index[0]
       Replay:   cargo test -p fp-conformance -- case_a --nocapture

  2. FAIL FP-P2C-003::case_b [SeriesAdd/Hardened]
       Mismatch: shape mismatch: expected len=3, got len=2
       Replay:   cargo test -p fp-conformance -- case_b --nocapture
       Artifact: artifacts/phase2c/FP-P2C-003/parity_mismatch_corpus.json

GATE FAILURES:
  - FP-P2C-001: strict_failed > 0
```

---

## 4. Artifact Index

### 4.1 Deterministic Artifact IDs

Every artifact is identified by an `ArtifactId`:

```
{packet_id}:{artifact_kind}@{short_hash}
```

Example: `FP-P2C-001:parity_report@a1b2c3d4`

The short hash is the first 8 hex characters of SHA-256 over `{packet_id}:{artifact_kind}:{run_ts_unix_ms}`.

### 4.2 Artifact Lookup

Given an artifact ID, the file path is deterministic:

```
artifacts/phase2c/{packet_id}/{artifact_file}
```

Mapping:
| artifact_kind | File |
|---|---|
| `parity_report` | `parity_report.json` |
| `raptorq_sidecar` | `parity_report.raptorq.json` |
| `decode_proof` | `parity_report.decode_proof.json` |
| `gate_result` | `parity_gate_result.json` |
| `mismatch_corpus` | `parity_mismatch_corpus.json` |

### 4.3 Forensic Log Cross-Reference

Every artifact write is recorded in the forensic log with an `ArtifactWritten` event:

```json
{"ts_unix_ms": 1707900000000, "event": {"kind": "artifact_written", "packet_id": "FP-P2C-001", "artifact_kind": "parity_report", "path": "artifacts/phase2c/FP-P2C-001/parity_report.json"}}
```

---

## 5. One-Command Replay

### 5.1 Replay a Specific Fixture

```bash
cargo test -p fp-conformance -- {case_id} --nocapture
```

### 5.2 Replay All Fixtures for a Packet

```bash
cargo test -p fp-conformance -- {packet_id} --nocapture
```

### 5.3 Replay with Full E2E Pipeline

```rust
let config = E2eConfig {
    options: SuiteOptions {
        packet_filter: Some("FP-P2C-001".to_owned()),
        ..Default::default()
    },
    ..E2eConfig::default_all_phases()
};
let report = run_e2e_suite(&config, &mut NoopHooks)?;
let forensics = build_failure_forensics(&report);
eprintln!("{forensics}");
```

### 5.4 Replay from Forensic Log

```bash
# Find the failing case in the forensic log
grep case_end forensic.jsonl | jq 'select(.event.status == "fail")'
# Copy the case_id and replay
cargo test -p fp-conformance -- {case_id} --nocapture
```

---

## 6. Rust API

### Types

```rust
pub struct ArtifactId { packet_id, artifact_kind, run_ts_unix_ms }
pub struct FailureDigest { packet_id, case_id, operation, mode, mismatch_class, replay_key, trace_id, mismatch_summary, replay_command, artifact_path }
pub struct FailureForensicsReport { run_ts_unix_ms, total_fixtures, total_passed, total_failed, failures, gate_failures }
```

### Functions

```rust
pub fn build_failure_forensics(e2e: &E2eReport) -> FailureForensicsReport
```

### Display Traits

All types implement `std::fmt::Display` for human-readable output.
`FailureForensicsReport` and `FailureDigest` implement `Serialize`/`Deserialize` for JSON export.

---

## Changelog

- **bd-2gi.21** (2026-02-14): Initial failure forensics UX design. Defines concise failure digest format, forensics report display, deterministic artifact indexing, one-command replay patterns, and Rust API types.
