// Week 33: Artifact Registry, Provenance & Reproducible Runs
//
// This module provides a versioned, queryable record system for all important
// MedLang operations (evidence runs, surrogate training/eval, RL training/eval).
// Enables full reproducibility and audit trails for scientific computing.

pub mod logging;
pub mod storage;

pub use storage::Registry;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::fmt;
use uuid::Uuid;

// =============================================================================
// Core ID Types
// =============================================================================

/// Unique identifier for a run (operation execution)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct RunId(pub Uuid);

impl RunId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_string(s: &str) -> anyhow::Result<Self> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

impl fmt::Display for RunId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Unique identifier for an artifact (output file/model)
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ArtifactId(pub Uuid);

impl ArtifactId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    pub fn from_string(s: &str) -> anyhow::Result<Self> {
        Ok(Self(Uuid::parse_str(s)?))
    }
}

impl fmt::Display for ArtifactId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// =============================================================================
// Run Classification
// =============================================================================

/// Classification of run types for filtering and organization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum RunKind {
    /// Mechanistic QSP simulation
    EvidenceMechanistic,

    /// Surrogate-based simulation
    EvidenceSurrogate,

    /// Hybrid mechanistic/surrogate run with adaptive strategy
    EvidenceHybrid,

    /// Surrogate model training
    SurrogateTrain,

    /// Surrogate model evaluation/qualification
    SurrogateEval,

    /// RL policy training
    RLTrain,

    /// RL policy evaluation
    RLEval,
}

impl RunKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RunKind::EvidenceMechanistic => "EvidenceMechanistic",
            RunKind::EvidenceSurrogate => "EvidenceSurrogate",
            RunKind::EvidenceHybrid => "EvidenceHybrid",
            RunKind::SurrogateTrain => "SurrogateTrain",
            RunKind::SurrogateEval => "SurrogateEval",
            RunKind::RLTrain => "RLTrain",
            RunKind::RLEval => "RLEval",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "EvidenceMechanistic" => Some(RunKind::EvidenceMechanistic),
            "EvidenceSurrogate" => Some(RunKind::EvidenceSurrogate),
            "EvidenceHybrid" => Some(RunKind::EvidenceHybrid),
            "SurrogateTrain" => Some(RunKind::SurrogateTrain),
            "SurrogateEval" => Some(RunKind::SurrogateEval),
            "RLTrain" => Some(RunKind::RLTrain),
            "RLEval" => Some(RunKind::RLEval),
            _ => None,
        }
    }
}

// =============================================================================
// Artifact Classification
// =============================================================================

/// Classification of artifacts for organization
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ArtifactKind {
    /// Evidence/simulation result file
    EvidenceResult,

    /// Trained surrogate model
    SurrogateModel,

    /// Surrogate evaluation report
    SurrogateEvalReport,

    /// Trained RL policy
    RLPolicy,

    /// RL training report
    RLTrainReport,

    /// RL evaluation report
    RLEvalReport,
}

// =============================================================================
// Record Types
// =============================================================================

/// Metadata about an artifact (file, model, policy, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    /// Unique identifier
    pub id: ArtifactId,

    /// Classification
    pub kind: ArtifactKind,

    /// File path (relative to project or absolute)
    pub path: Option<String>,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Free-form metadata (JSON)
    pub metadata: serde_json::Value,
}

impl ArtifactRecord {
    pub fn new(kind: ArtifactKind, path: Option<String>) -> Self {
        Self {
            id: ArtifactId::new(),
            kind,
            path,
            created_at: Utc::now(),
            metadata: serde_json::json!({}),
        }
    }

    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Complete record of a single run (operation execution)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Unique identifier
    pub id: RunId,

    /// Classification
    pub kind: RunKind,

    /// Start timestamp
    pub started_at: DateTime<Utc>,

    /// End timestamp
    pub finished_at: DateTime<Utc>,

    // =============================================================================
    // Context Information
    // =============================================================================
    /// Project root directory
    pub project_root: String,

    /// Module path (e.g., "med.oncology.evidence")
    pub module_path: Option<String>,

    /// Evidence program name (e.g., "OncologyEvidence")
    pub evidence_program: Option<String>,

    /// Git commit hash (if available)
    pub git_commit: Option<String>,

    /// Whether git working directory was dirty
    pub git_dirty: bool,

    // =============================================================================
    // Run Data
    // =============================================================================
    /// Configuration as JSON (backend, hyperparameters, etc.)
    pub config: serde_json::Value,

    /// Summary metrics as JSON (RMSE, rewards, violations, etc.)
    pub metrics: serde_json::Value,

    /// Artifacts produced by this run
    pub artifacts: Vec<ArtifactId>,
}

impl RunRecord {
    pub fn new(kind: RunKind, project_root: String) -> Self {
        let now = Utc::now();
        Self {
            id: RunId::new(),
            kind,
            started_at: now,
            finished_at: now,
            project_root,
            module_path: None,
            evidence_program: None,
            git_commit: None,
            git_dirty: false,
            config: serde_json::json!({}),
            metrics: serde_json::json!({}),
            artifacts: Vec::new(),
        }
    }

    pub fn with_module_path(mut self, path: String) -> Self {
        self.module_path = Some(path);
        self
    }

    pub fn with_evidence_program(mut self, ev: String) -> Self {
        self.evidence_program = Some(ev);
        self
    }

    pub fn with_git_info(mut self, commit: Option<String>, dirty: bool) -> Self {
        self.git_commit = commit;
        self.git_dirty = dirty;
        self
    }

    pub fn with_config(mut self, config: serde_json::Value) -> Self {
        self.config = config;
        self
    }

    pub fn with_metrics(mut self, metrics: serde_json::Value) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_artifacts(mut self, artifacts: Vec<ArtifactId>) -> Self {
        self.artifacts = artifacts;
        self
    }

    pub fn set_finished(&mut self) {
        self.finished_at = Utc::now();
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Get the default registry root directory (~/.medlang/registry)
pub fn default_registry_root() -> anyhow::Result<std::path::PathBuf> {
    let home = dirs::home_dir().ok_or_else(|| anyhow::anyhow!("no HOME directory found"))?;
    Ok(home.join(".medlang/registry"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_id_creation() {
        let id1 = RunId::new();
        let id2 = RunId::new();
        assert_ne!(id1, id2, "RunIds should be unique");
    }

    #[test]
    fn test_run_id_from_string() {
        let id = RunId::new();
        let s = id.to_string();
        let parsed = RunId::from_string(&s).unwrap();
        assert_eq!(id, parsed);
    }

    #[test]
    fn test_artifact_id_creation() {
        let id1 = ArtifactId::new();
        let id2 = ArtifactId::new();
        assert_ne!(id1, id2, "ArtifactIds should be unique");
    }

    #[test]
    fn test_run_kind_string_conversion() {
        assert_eq!(RunKind::EvidenceHybrid.as_str(), "EvidenceHybrid");
        assert_eq!(
            RunKind::from_str("EvidenceHybrid"),
            Some(RunKind::EvidenceHybrid)
        );
        assert_eq!(RunKind::from_str("Invalid"), None);
    }

    #[test]
    fn test_run_record_builder() {
        let record = RunRecord::new(RunKind::RLTrain, "/path/to/project".to_string())
            .with_module_path("med.rl".to_string())
            .with_evidence_program("TestEvidence".to_string())
            .with_config(serde_json::json!({"n_episodes": 1000}))
            .with_metrics(serde_json::json!({"avg_reward": 5.0}));

        assert_eq!(record.kind, RunKind::RLTrain);
        assert_eq!(record.module_path, Some("med.rl".to_string()));
        assert_eq!(record.evidence_program, Some("TestEvidence".to_string()));
        assert_eq!(record.config["n_episodes"], 1000);
        assert_eq!(record.metrics["avg_reward"], 5.0);
    }

    #[test]
    fn test_artifact_record_builder() {
        let artifact = ArtifactRecord::new(
            ArtifactKind::RLPolicy,
            Some("/path/to/policy.json".to_string()),
        )
        .with_metadata(serde_json::json!({"n_actions": 5}));

        assert_eq!(artifact.path, Some("/path/to/policy.json".to_string()));
        assert_eq!(artifact.metadata["n_actions"], 5);
    }
}
