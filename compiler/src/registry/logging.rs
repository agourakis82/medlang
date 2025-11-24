// Week 33: RunLogger - Helper for logging operations
//
// Simplifies logging of runs and artifacts with context management.

use super::storage::Registry;
use super::{ArtifactId, ArtifactRecord, RunId, RunKind, RunRecord};
use chrono::Utc;

/// Helper for logging runs with consistent context
#[derive(Debug, Clone)]
pub struct RunLogger {
    registry: Registry,
    project_root: String,
    module_path: Option<String>,
    evidence_program: Option<String>,
    git_commit: Option<String>,
    git_dirty: bool,
}

impl RunLogger {
    /// Create a new RunLogger for a project
    pub fn new_for_project(
        project_root: &str,
        module_path: Option<&str>,
        evidence_program: Option<&str>,
    ) -> anyhow::Result<Self> {
        let registry = Registry::new_default()?;
        let (commit, dirty) = detect_git_state(project_root);

        Ok(Self {
            registry,
            project_root: project_root.to_string(),
            module_path: module_path.map(|s| s.to_string()),
            evidence_program: evidence_program.map(|s| s.to_string()),
            git_commit: commit,
            git_dirty: dirty,
        })
    }

    /// Create a RunLogger with a custom registry (useful for testing)
    pub fn with_registry(
        registry: Registry,
        project_root: &str,
        module_path: Option<&str>,
        evidence_program: Option<&str>,
    ) -> Self {
        let (commit, dirty) = detect_git_state(project_root);

        Self {
            registry,
            project_root: project_root.to_string(),
            module_path: module_path.map(|s| s.to_string()),
            evidence_program: evidence_program.map(|s| s.to_string()),
            git_commit: commit,
            git_dirty: dirty,
        }
    }

    /// Log a run with artifacts
    pub fn log_run_with_artifacts(
        &self,
        kind: RunKind,
        config: serde_json::Value,
        metrics: serde_json::Value,
        mut artifacts: Vec<ArtifactRecord>,
    ) -> anyhow::Result<RunId> {
        let run_id = RunId::new();
        let now = Utc::now();

        // Assign IDs and log artifacts
        let mut artifact_ids = Vec::new();
        for art in &mut artifacts {
            art.id = ArtifactId::new();
            art.created_at = now;
            self.registry.log_artifact(art)?;
            artifact_ids.push(art.id);
        }

        // Create and log run record
        let run = RunRecord {
            id: run_id,
            kind,
            started_at: now,
            finished_at: now,
            project_root: self.project_root.clone(),
            module_path: self.module_path.clone(),
            evidence_program: self.evidence_program.clone(),
            git_commit: self.git_commit.clone(),
            git_dirty: self.git_dirty,
            config,
            metrics,
            artifacts: artifact_ids,
        };

        self.registry.log_run(&run)?;

        Ok(run_id)
    }

    /// Log a run without artifacts
    pub fn log_run(
        &self,
        kind: RunKind,
        config: serde_json::Value,
        metrics: serde_json::Value,
    ) -> anyhow::Result<RunId> {
        self.log_run_with_artifacts(kind, config, metrics, Vec::new())
    }

    /// Get the underlying registry
    pub fn registry(&self) -> &Registry {
        &self.registry
    }
}

// =============================================================================
// Git State Detection
// =============================================================================

/// Detect git commit and dirty state (simplified for Week 33)
fn detect_git_state(_project_root: &str) -> (Option<String>, bool) {
    // Week 33: Simple stub implementation
    // Future: Use `git2` crate or shell out to `git rev-parse HEAD`
    (None, false)
}

// =============================================================================
// Specialized Logging Helpers
// =============================================================================

/// Log a mechanistic or surrogate evidence run
pub fn log_evidence_run(
    logger: &RunLogger,
    backend: crate::ml::BackendKind,
    metrics: serde_json::Value,
    result_file: &str,
) -> anyhow::Result<RunId> {
    use super::{ArtifactKind, ArtifactRecord};

    let kind = match backend {
        crate::ml::BackendKind::Mechanistic => RunKind::EvidenceMechanistic,
        crate::ml::BackendKind::Surrogate => RunKind::EvidenceSurrogate,
        crate::ml::BackendKind::Hybrid => RunKind::EvidenceHybrid,
    };

    let config = serde_json::json!({
        "backend": format!("{:?}", backend)
    });

    let artifact = ArtifactRecord::new(ArtifactKind::EvidenceResult, Some(result_file.to_string()));

    logger.log_run_with_artifacts(kind, config, metrics, vec![artifact])
}

/// Log a surrogate training run
pub fn log_surrogate_train(
    logger: &RunLogger,
    train_config: &crate::ml::SurrogateTrainConfig,
    model_file: &str,
) -> anyhow::Result<RunId> {
    use super::{ArtifactKind, ArtifactRecord};

    let config = serde_json::to_value(train_config)?;
    let metrics = serde_json::json!({}); // Training metrics could be added here

    let artifact = ArtifactRecord::new(ArtifactKind::SurrogateModel, Some(model_file.to_string()));

    logger.log_run_with_artifacts(RunKind::SurrogateTrain, config, metrics, vec![artifact])
}

/// Log a surrogate evaluation run
pub fn log_surrogate_eval(
    logger: &RunLogger,
    eval_config: serde_json::Value,
    report: serde_json::Value,
    report_file: &str,
) -> anyhow::Result<RunId> {
    use super::{ArtifactKind, ArtifactRecord};

    let artifact = ArtifactRecord::new(
        ArtifactKind::SurrogateEvalReport,
        Some(report_file.to_string()),
    );

    logger.log_run_with_artifacts(RunKind::SurrogateEval, eval_config, report, vec![artifact])
}

/// Log an RL policy training run
pub fn log_rl_train(
    logger: &RunLogger,
    train_config: &crate::rl::RLTrainConfig,
    train_report: serde_json::Value,
    policy_file: &str,
    report_file: Option<&str>,
) -> anyhow::Result<RunId> {
    use super::{ArtifactKind, ArtifactRecord};

    let config = serde_json::to_value(train_config)?;

    let mut artifacts = vec![ArtifactRecord::new(
        ArtifactKind::RLPolicy,
        Some(policy_file.to_string()),
    )];

    if let Some(report_path) = report_file {
        artifacts.push(ArtifactRecord::new(
            ArtifactKind::RLTrainReport,
            Some(report_path.to_string()),
        ));
    }

    logger.log_run_with_artifacts(RunKind::RLTrain, config, train_report, artifacts)
}

/// Log an RL policy evaluation run
pub fn log_rl_eval(
    logger: &RunLogger,
    eval_config: serde_json::Value,
    eval_report: serde_json::Value,
    report_file: &str,
) -> anyhow::Result<RunId> {
    use super::{ArtifactKind, ArtifactRecord};

    let artifact = ArtifactRecord::new(ArtifactKind::RLEvalReport, Some(report_file.to_string()));

    logger.log_run_with_artifacts(RunKind::RLEval, eval_config, eval_report, vec![artifact])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::RunKind;
    use tempfile::TempDir;

    fn create_test_logger() -> (TempDir, RunLogger) {
        let temp_dir = TempDir::new().unwrap();
        let registry = Registry::new(temp_dir.path().to_path_buf()).unwrap();
        let logger = RunLogger::with_registry(
            registry,
            "/test/project",
            Some("med.test"),
            Some("TestEvidence"),
        );
        (temp_dir, logger)
    }

    #[test]
    fn test_log_run_without_artifacts() {
        let (_temp, logger) = create_test_logger();

        let config = serde_json::json!({"test": "config"});
        let metrics = serde_json::json!({"test": "metrics"});

        let run_id = logger.log_run(RunKind::RLTrain, config, metrics).unwrap();

        // Verify it was logged
        let run = logger.registry().find_run(run_id).unwrap();
        assert!(run.is_some());
        let run = run.unwrap();
        assert_eq!(run.kind, RunKind::RLTrain);
        assert_eq!(run.artifacts.len(), 0);
    }

    #[test]
    fn test_log_run_with_artifacts() {
        let (_temp, logger) = create_test_logger();

        let config = serde_json::json!({"test": "config"});
        let metrics = serde_json::json!({"test": "metrics"});

        let artifact = ArtifactRecord::new(
            super::super::ArtifactKind::RLPolicy,
            Some("/test/policy.json".to_string()),
        );

        let run_id = logger
            .log_run_with_artifacts(RunKind::RLTrain, config, metrics, vec![artifact])
            .unwrap();

        // Verify run was logged
        let run = logger.registry().find_run(run_id).unwrap().unwrap();
        assert_eq!(run.kind, RunKind::RLTrain);
        assert_eq!(run.artifacts.len(), 1);

        // Verify artifact was logged
        let artifact_id = run.artifacts[0];
        let artifact = logger.registry().find_artifact(artifact_id).unwrap();
        assert!(artifact.is_some());
    }

    #[test]
    fn test_logger_preserves_context() {
        let (_temp, logger) = create_test_logger();

        let config = serde_json::json!({});
        let metrics = serde_json::json!({});

        let run_id = logger.log_run(RunKind::RLTrain, config, metrics).unwrap();

        let run = logger.registry().find_run(run_id).unwrap().unwrap();
        assert_eq!(run.project_root, "/test/project");
        assert_eq!(run.module_path, Some("med.test".to_string()));
        assert_eq!(run.evidence_program, Some("TestEvidence".to_string()));
    }

    #[test]
    fn test_log_evidence_run() {
        let (_temp, logger) = create_test_logger();

        let metrics = serde_json::json!({"ORR": 0.45});
        let run_id = log_evidence_run(
            &logger,
            crate::ml::BackendKind::Mechanistic,
            metrics,
            "/test/result.json",
        )
        .unwrap();

        let run = logger.registry().find_run(run_id).unwrap().unwrap();
        assert_eq!(run.kind, RunKind::EvidenceMechanistic);
        assert_eq!(run.artifacts.len(), 1);
    }

    #[test]
    fn test_log_rl_train() {
        let (_temp, logger) = create_test_logger();

        let train_cfg = crate::rl::RLTrainConfig {
            n_episodes: 1000,
            max_steps_per_episode: 10,
            gamma: 0.95,
            alpha: 0.1,
            eps_start: 1.0,
            eps_end: 0.05,
        };

        let train_report = serde_json::json!({
            "avg_reward": 5.0,
            "final_epsilon": 0.05
        });

        let run_id = log_rl_train(
            &logger,
            &train_cfg,
            train_report,
            "/test/policy.json",
            Some("/test/report.json"),
        )
        .unwrap();

        let run = logger.registry().find_run(run_id).unwrap().unwrap();
        assert_eq!(run.kind, RunKind::RLTrain);
        assert_eq!(run.artifacts.len(), 2); // Policy + report
    }
}
