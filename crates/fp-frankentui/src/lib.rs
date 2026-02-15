#![forbid(unsafe_code)]

use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use fp_conformance::{
    DecodeProofArtifact, DecodeProofStatus, PacketDriftHistoryEntry, PacketGateResult,
    PacketParityReport,
};
use fp_runtime::{ConformalGuard, EvidenceLedger, RuntimeMode, RuntimePolicy, decision_to_card};
use serde::Deserialize;
use serde::de::DeserializeOwned;
use thiserror::Error;

const PHASE2C_DIR: &str = "artifacts/phase2c";
const DRIFT_HISTORY_FILE: &str = "drift_history.jsonl";

#[derive(Debug, Error)]
pub enum FtuiError {
    #[error("artifact root is not accessible: {path}")]
    ArtifactRootMissing { path: PathBuf },
    #[error("invalid packet id: {packet_id}")]
    InvalidPacketId { packet_id: String },
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArtifactIssueKind {
    MissingFile,
    ParseError,
    IoError,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactIssue {
    pub path: PathBuf,
    pub kind: ArtifactIssueKind,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PacketSnapshot {
    pub packet_id: String,
    pub parity_report: Option<PacketParityReport>,
    pub gate_result: Option<PacketGateResult>,
    pub decode_status: Option<DecodeProofStatus>,
    pub mismatch_count: Option<usize>,
    pub issues: Vec<ArtifactIssue>,
}

impl PacketSnapshot {
    #[must_use]
    pub fn is_green(&self) -> bool {
        self.parity_report
            .as_ref()
            .is_some_and(PacketParityReport::is_green)
            && self.gate_result.as_ref().is_some_and(|gate| gate.pass)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DriftHistorySnapshot {
    pub entries: Vec<PacketDriftHistoryEntry>,
    pub malformed_lines: usize,
    pub missing: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PolicySnapshot {
    pub mode: RuntimeMode,
    pub fail_closed_unknown_features: bool,
    pub hardened_join_row_cap: Option<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConformalSnapshot {
    pub calibrated: bool,
    pub calibration_count: usize,
    pub quantile_threshold: Option<f64>,
    pub empirical_coverage: f64,
    pub coverage_alert: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionCardSnapshot {
    pub mode: RuntimeMode,
    pub card: String,
    pub evidence_terms_total: usize,
    pub evidence_terms_shown: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DecisionDashboardSnapshot {
    pub total_records: usize,
    pub strict_records: usize,
    pub hardened_records: usize,
    pub policy: PolicySnapshot,
    pub cards: Vec<DecisionCardSnapshot>,
    pub evidence_term_cap: usize,
}

pub trait FtuiDataSource {
    fn discover_packet_ids(&self) -> Result<Vec<String>, FtuiError>;
    fn load_packet_snapshot(&self, packet_id: &str) -> Result<PacketSnapshot, FtuiError>;
    fn load_drift_history(&self) -> Result<DriftHistorySnapshot, FtuiError>;
}

#[derive(Debug, Clone)]
pub struct FsFtuiDataSource {
    phase2c_root: PathBuf,
}

impl FsFtuiDataSource {
    #[must_use]
    pub fn from_repo_root(repo_root: impl AsRef<Path>) -> Self {
        Self {
            phase2c_root: repo_root.as_ref().join(PHASE2C_DIR),
        }
    }

    #[must_use]
    pub fn from_phase2c_root(phase2c_root: impl Into<PathBuf>) -> Self {
        Self {
            phase2c_root: phase2c_root.into(),
        }
    }

    #[must_use]
    pub fn phase2c_root(&self) -> &Path {
        &self.phase2c_root
    }

    fn ensure_root_exists(&self) -> Result<(), FtuiError> {
        if self.phase2c_root.is_dir() {
            Ok(())
        } else {
            Err(FtuiError::ArtifactRootMissing {
                path: self.phase2c_root.clone(),
            })
        }
    }
}

impl FtuiDataSource for FsFtuiDataSource {
    fn discover_packet_ids(&self) -> Result<Vec<String>, FtuiError> {
        self.ensure_root_exists()?;
        let mut packet_ids = Vec::new();
        let entries = fs::read_dir(&self.phase2c_root).map_err(|source| FtuiError::Io {
            path: self.phase2c_root.clone(),
            source,
        })?;
        for entry in entries {
            let entry = entry.map_err(|source| FtuiError::Io {
                path: self.phase2c_root.clone(),
                source,
            })?;
            if !entry
                .file_type()
                .map_err(|source| FtuiError::Io {
                    path: entry.path(),
                    source,
                })?
                .is_dir()
            {
                continue;
            }
            let name = entry.file_name();
            let packet_id = name.to_string_lossy();
            if is_valid_packet_id(packet_id.as_ref()) {
                packet_ids.push(packet_id.into_owned());
            }
        }
        packet_ids.sort();
        Ok(packet_ids)
    }

    fn load_packet_snapshot(&self, packet_id: &str) -> Result<PacketSnapshot, FtuiError> {
        self.ensure_root_exists()?;
        if !is_valid_packet_id(packet_id) {
            return Err(FtuiError::InvalidPacketId {
                packet_id: packet_id.to_owned(),
            });
        }

        let packet_root = self.phase2c_root.join(packet_id);
        let mut issues = Vec::new();

        let parity_report: Option<PacketParityReport> =
            read_json_optional(&packet_root.join("parity_report.json"), &mut issues);
        let gate_result: Option<PacketGateResult> =
            read_json_optional(&packet_root.join("parity_gate_result.json"), &mut issues);
        let decode_status = read_json_optional::<DecodeProofArtifact>(
            &packet_root.join("parity_report.decode_proof.json"),
            &mut issues,
        )
        .map(|artifact| artifact.status);
        let mismatch_count = read_json_optional::<MismatchCorpus>(
            &packet_root.join("parity_mismatch_corpus.json"),
            &mut issues,
        )
        .map(|corpus| corpus.mismatch_count);

        Ok(PacketSnapshot {
            packet_id: packet_id.to_owned(),
            parity_report,
            gate_result,
            decode_status,
            mismatch_count,
            issues,
        })
    }

    fn load_drift_history(&self) -> Result<DriftHistorySnapshot, FtuiError> {
        self.ensure_root_exists()?;
        let path = self.phase2c_root.join(DRIFT_HISTORY_FILE);
        let file = match File::open(&path) {
            Ok(file) => file,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => {
                return Ok(DriftHistorySnapshot {
                    entries: Vec::new(),
                    malformed_lines: 0,
                    missing: true,
                });
            }
            Err(source) => return Err(FtuiError::Io { path, source }),
        };

        let mut entries = Vec::new();
        let mut malformed_lines = 0;

        for line in BufReader::new(file).lines() {
            let line = line.map_err(|source| FtuiError::Io {
                path: path.clone(),
                source,
            })?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<PacketDriftHistoryEntry>(&line) {
                Ok(entry) => entries.push(entry),
                Err(_) => malformed_lines += 1,
            }
        }

        Ok(DriftHistorySnapshot {
            entries,
            malformed_lines,
            missing: false,
        })
    }
}

#[derive(Debug, Deserialize)]
struct MismatchCorpus {
    mismatch_count: usize,
}

fn read_json_optional<T: DeserializeOwned>(
    path: &Path,
    issues: &mut Vec<ArtifactIssue>,
) -> Option<T> {
    match fs::read_to_string(path) {
        Ok(content) => match serde_json::from_str::<T>(&content) {
            Ok(parsed) => Some(parsed),
            Err(error) => {
                issues.push(ArtifactIssue {
                    path: path.to_path_buf(),
                    kind: ArtifactIssueKind::ParseError,
                    detail: error.to_string(),
                });
                None
            }
        },
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            issues.push(ArtifactIssue {
                path: path.to_path_buf(),
                kind: ArtifactIssueKind::MissingFile,
                detail: "file not found".to_owned(),
            });
            None
        }
        Err(error) => {
            issues.push(ArtifactIssue {
                path: path.to_path_buf(),
                kind: ArtifactIssueKind::IoError,
                detail: error.to_string(),
            });
            None
        }
    }
}

#[must_use]
pub fn is_valid_packet_id(packet_id: &str) -> bool {
    let Some(rest) = packet_id.strip_prefix("FP-P2C-") else {
        return false;
    };
    rest.len() == 3 && rest.as_bytes().iter().all(u8::is_ascii_digit)
}

#[must_use]
pub fn summarize_conformal_guard(guard: &ConformalGuard) -> ConformalSnapshot {
    ConformalSnapshot {
        calibrated: guard.is_calibrated(),
        calibration_count: guard.calibration_count(),
        quantile_threshold: guard.conformal_quantile(),
        empirical_coverage: guard.empirical_coverage(),
        coverage_alert: guard.coverage_alert(),
    }
}

#[must_use]
pub fn summarize_decision_dashboard(
    ledger: &EvidenceLedger,
    policy: &RuntimePolicy,
    evidence_term_cap: usize,
) -> DecisionDashboardSnapshot {
    let mut strict_records = 0;
    let mut hardened_records = 0;
    let mut cards = Vec::with_capacity(ledger.records().len());

    for record in ledger.records() {
        match record.mode {
            RuntimeMode::Strict => strict_records += 1,
            RuntimeMode::Hardened => hardened_records += 1,
        }
        let evidence_terms_total = record.evidence.len();
        cards.push(DecisionCardSnapshot {
            mode: record.mode,
            card: decision_to_card(record).render_plain(),
            evidence_terms_total,
            evidence_terms_shown: evidence_terms_total.min(evidence_term_cap),
        });
    }

    DecisionDashboardSnapshot {
        total_records: strict_records + hardened_records,
        strict_records,
        hardened_records,
        policy: PolicySnapshot {
            mode: policy.mode,
            fail_closed_unknown_features: policy.fail_closed_unknown_features,
            hardened_join_row_cap: policy.hardened_join_row_cap,
        },
        cards,
        evidence_term_cap,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArtifactIssueKind, FsFtuiDataSource, FtuiDataSource, is_valid_packet_id,
        summarize_decision_dashboard,
    };
    use fp_conformance::PacketDriftHistoryEntry;
    use fp_runtime::{EvidenceLedger, RuntimePolicy};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn packet_id_validation_accepts_expected_shape() {
        assert!(is_valid_packet_id("FP-P2C-001"));
        assert!(is_valid_packet_id("FP-P2C-999"));
        assert!(!is_valid_packet_id("FP-P2C-12"));
        assert!(!is_valid_packet_id("FP-P2C-1000"));
        assert!(!is_valid_packet_id("fp-p2c-001"));
        assert!(!is_valid_packet_id("../FP-P2C-001"));
    }

    #[test]
    fn discover_packet_ids_filters_non_packet_entries() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("artifacts/phase2c");
        fs::create_dir_all(root.join("FP-P2C-001")).expect("packet dir");
        fs::create_dir_all(root.join("scratch")).expect("scratch dir");
        fs::write(root.join("drift_history.jsonl"), "").expect("history file");
        let source = FsFtuiDataSource::from_repo_root(dir.path());
        let ids = source.discover_packet_ids().expect("discover ids");
        assert_eq!(ids, vec!["FP-P2C-001".to_owned()]);
    }

    #[test]
    fn load_drift_history_handles_missing_file() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("artifacts/phase2c");
        fs::create_dir_all(&root).expect("phase2c root");
        let source = FsFtuiDataSource::from_repo_root(dir.path());
        let history = source.load_drift_history().expect("load history");
        assert!(history.missing);
        assert_eq!(history.malformed_lines, 0);
        assert!(history.entries.is_empty());
    }

    #[test]
    fn load_drift_history_skips_malformed_lines() {
        let dir = tempdir().expect("tempdir");
        let root = dir.path().join("artifacts/phase2c");
        fs::create_dir_all(&root).expect("phase2c root");
        let entry = PacketDriftHistoryEntry {
            ts_unix_ms: 1,
            packet_id: "FP-P2C-001".to_owned(),
            suite: "phase2c_packets".to_owned(),
            fixture_count: 2,
            passed: 2,
            failed: 0,
            strict_failed: 0,
            hardened_failed: 0,
            gate_pass: true,
            report_hash: "sha256:abc".to_owned(),
        };
        let line = serde_json::to_string(&entry).expect("serialize entry");
        let history_path = root.join("drift_history.jsonl");
        fs::write(&history_path, format!("{line}\nnot-json\n{{\"partial\":"))
            .expect("write history");

        let source = FsFtuiDataSource::from_repo_root(dir.path());
        let history = source.load_drift_history().expect("load history");
        assert!(!history.missing);
        assert_eq!(history.entries.len(), 1);
        assert_eq!(history.malformed_lines, 2);
    }

    #[test]
    fn packet_snapshot_collects_missing_and_parse_issues() {
        let dir = tempdir().expect("tempdir");
        let packet_root = dir.path().join("artifacts/phase2c/FP-P2C-001");
        fs::create_dir_all(&packet_root).expect("packet root");
        fs::write(packet_root.join("parity_report.json"), "{\"bad\":").expect("bad json");
        fs::write(
            packet_root.join("parity_mismatch_corpus.json"),
            "{\"packet_id\":\"FP-P2C-001\",\"mismatch_count\":7,\"mismatches\":[]}",
        )
        .expect("mismatch corpus");
        fs::write(
            packet_root.join("parity_report.decode_proof.json"),
            "{\"packet_id\":\"FP-P2C-001\",\"decode_proofs\":[],\"status\":\"recovered\"}",
        )
        .expect("decode proof");

        let source = FsFtuiDataSource::from_repo_root(dir.path());
        let snapshot = source
            .load_packet_snapshot("FP-P2C-001")
            .expect("packet snapshot");

        assert!(snapshot.parity_report.is_none());
        assert!(snapshot.gate_result.is_none());
        assert_eq!(snapshot.mismatch_count, Some(7));
        assert!(snapshot.decode_status.is_some());
        assert!(
            snapshot
                .issues
                .iter()
                .any(|i| i.kind == ArtifactIssueKind::ParseError),
            "expected parse issue"
        );
        assert!(
            snapshot
                .issues
                .iter()
                .any(|i| i.kind == ArtifactIssueKind::MissingFile),
            "expected missing-file issue"
        );
    }

    #[test]
    fn decision_summary_counts_modes_and_caps_evidence() {
        let mut ledger = EvidenceLedger::new();
        let strict = RuntimePolicy::strict();
        let hardened = RuntimePolicy::hardened(Some(8));
        strict.decide_unknown_feature("feature_x", "unsupported", &mut ledger);
        hardened.decide_join_admission(42, &mut ledger);

        let summary = summarize_decision_dashboard(&ledger, &hardened, 1);
        assert_eq!(summary.total_records, 2);
        assert_eq!(summary.strict_records, 1);
        assert_eq!(summary.hardened_records, 1);
        assert!(
            summary
                .cards
                .iter()
                .all(|card| card.evidence_terms_shown == 1)
        );
        assert_eq!(summary.policy.hardened_join_row_cap, Some(8));
    }
}
