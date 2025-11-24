// Week 33: JSONL-based Registry Storage
//
// Append-only, line-delimited JSON storage for runs and artifacts.
// Simple, debuggable, and suitable for local single-user workflows.

use super::{ArtifactId, ArtifactRecord, RunId, RunRecord};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Registry storage backed by JSONL files
#[derive(Debug, Clone)]
pub struct Registry {
    root: PathBuf,
    runs_path: PathBuf,
    artifacts_path: PathBuf,
}

impl Registry {
    /// Create a new registry with the default location (~/.medlang/registry)
    pub fn new_default() -> anyhow::Result<Self> {
        let root = super::default_registry_root()?;
        Self::new(root)
    }

    /// Create a new registry at a specific path (useful for testing)
    pub fn new(root: PathBuf) -> anyhow::Result<Self> {
        std::fs::create_dir_all(&root)?;
        let runs_path = root.join("runs.jsonl");
        let artifacts_path = root.join("artifacts.jsonl");
        Ok(Self {
            root,
            runs_path,
            artifacts_path,
        })
    }

    /// Get the registry root directory
    pub fn root(&self) -> &Path {
        &self.root
    }

    // =============================================================================
    // Write Operations (Append-only)
    // =============================================================================

    /// Log a run record (append to runs.jsonl)
    pub fn log_run(&self, run: &RunRecord) -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.runs_path)?;

        let line = serde_json::to_string(run)?;
        writeln!(file, "{}", line)?;
        file.flush()?;

        Ok(())
    }

    /// Log an artifact record (append to artifacts.jsonl)
    pub fn log_artifact(&self, artifact: &ArtifactRecord) -> anyhow::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.artifacts_path)?;

        let line = serde_json::to_string(artifact)?;
        writeln!(file, "{}", line)?;
        file.flush()?;

        Ok(())
    }

    // =============================================================================
    // Read Operations
    // =============================================================================

    /// Load all run records from the registry
    pub fn load_runs(&self) -> anyhow::Result<Vec<RunRecord>> {
        if !self.runs_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.runs_path)?;
        let reader = BufReader::new(file);

        let mut runs = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            if line.trim().is_empty() {
                continue; // Skip empty lines
            }

            match serde_json::from_str::<RunRecord>(&line) {
                Ok(run) => runs.push(run),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to parse run record at line {}: {}",
                        line_num + 1,
                        e
                    );
                    // Continue parsing other lines
                }
            }
        }

        Ok(runs)
    }

    /// Load all artifact records from the registry
    pub fn load_artifacts(&self) -> anyhow::Result<Vec<ArtifactRecord>> {
        if !self.artifacts_path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.artifacts_path)?;
        let reader = BufReader::new(file);

        let mut artifacts = Vec::new();
        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<ArtifactRecord>(&line) {
                Ok(artifact) => artifacts.push(artifact),
                Err(e) => {
                    eprintln!(
                        "Warning: Failed to parse artifact record at line {}: {}",
                        line_num + 1,
                        e
                    );
                }
            }
        }

        Ok(artifacts)
    }

    // =============================================================================
    // Query Operations
    // =============================================================================

    /// Find a specific run by ID
    pub fn find_run(&self, id: RunId) -> anyhow::Result<Option<RunRecord>> {
        let runs = self.load_runs()?;
        Ok(runs.into_iter().find(|r| r.id == id))
    }

    /// Find a specific artifact by ID
    pub fn find_artifact(&self, id: ArtifactId) -> anyhow::Result<Option<ArtifactRecord>> {
        let artifacts = self.load_artifacts()?;
        Ok(artifacts.into_iter().find(|a| a.id == id))
    }

    /// Get all runs of a specific kind
    pub fn find_runs_by_kind(&self, kind: super::RunKind) -> anyhow::Result<Vec<RunRecord>> {
        let runs = self.load_runs()?;
        Ok(runs.into_iter().filter(|r| r.kind == kind).collect())
    }

    /// Get all artifacts of a specific kind
    pub fn find_artifacts_by_kind(
        &self,
        kind: super::ArtifactKind,
    ) -> anyhow::Result<Vec<ArtifactRecord>> {
        let artifacts = self.load_artifacts()?;
        Ok(artifacts.into_iter().filter(|a| a.kind == kind).collect())
    }

    /// Get the most recent N runs
    pub fn recent_runs(&self, n: usize) -> anyhow::Result<Vec<RunRecord>> {
        let mut runs = self.load_runs()?;
        runs.sort_by(|a, b| b.started_at.cmp(&a.started_at)); // Reverse chronological
        runs.truncate(n);
        Ok(runs)
    }

    // =============================================================================
    // Maintenance Operations
    // =============================================================================

    /// Count total runs in registry
    pub fn count_runs(&self) -> anyhow::Result<usize> {
        Ok(self.load_runs()?.len())
    }

    /// Count total artifacts in registry
    pub fn count_artifacts(&self) -> anyhow::Result<usize> {
        Ok(self.load_artifacts()?.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{ArtifactKind, RunKind, RunRecord};
    use tempfile::TempDir;

    fn create_test_registry() -> (TempDir, Registry) {
        let temp_dir = TempDir::new().unwrap();
        let registry = Registry::new(temp_dir.path().to_path_buf()).unwrap();
        (temp_dir, registry)
    }

    #[test]
    fn test_registry_creation() {
        let (_temp, registry) = create_test_registry();
        assert!(registry.root().exists());
    }

    #[test]
    fn test_log_and_load_runs() {
        let (_temp, registry) = create_test_registry();

        // Create and log a run
        let run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string())
            .with_module_path("med.rl".to_string())
            .with_config(serde_json::json!({"n_episodes": 1000}));

        registry.log_run(&run).unwrap();

        // Load and verify
        let loaded_runs = registry.load_runs().unwrap();
        assert_eq!(loaded_runs.len(), 1);
        assert_eq!(loaded_runs[0].id, run.id);
        assert_eq!(loaded_runs[0].kind, RunKind::RLTrain);
    }

    #[test]
    fn test_log_and_load_artifacts() {
        let (_temp, registry) = create_test_registry();

        // Create and log an artifact
        let artifact = super::super::ArtifactRecord::new(
            ArtifactKind::RLPolicy,
            Some("/path/to/policy.json".to_string()),
        );

        registry.log_artifact(&artifact).unwrap();

        // Load and verify
        let loaded_artifacts = registry.load_artifacts().unwrap();
        assert_eq!(loaded_artifacts.len(), 1);
        assert_eq!(loaded_artifacts[0].id, artifact.id);
        assert_eq!(loaded_artifacts[0].kind, ArtifactKind::RLPolicy);
    }

    #[test]
    fn test_find_run_by_id() {
        let (_temp, registry) = create_test_registry();

        let run1 = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
        let run2 = RunRecord::new(RunKind::SurrogateEval, "/test/project".to_string());

        registry.log_run(&run1).unwrap();
        registry.log_run(&run2).unwrap();

        // Find specific run
        let found = registry.find_run(run1.id).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, run1.id);

        // Non-existent run
        let not_found = registry.find_run(super::RunId::new()).unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_find_runs_by_kind() {
        let (_temp, registry) = create_test_registry();

        let run1 = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
        let run2 = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
        let run3 = RunRecord::new(RunKind::SurrogateEval, "/test/project".to_string());

        registry.log_run(&run1).unwrap();
        registry.log_run(&run2).unwrap();
        registry.log_run(&run3).unwrap();

        let rl_runs = registry.find_runs_by_kind(RunKind::RLTrain).unwrap();
        assert_eq!(rl_runs.len(), 2);

        let eval_runs = registry.find_runs_by_kind(RunKind::SurrogateEval).unwrap();
        assert_eq!(eval_runs.len(), 1);
    }

    #[test]
    fn test_recent_runs() {
        let (_temp, registry) = create_test_registry();

        // Log 5 runs
        for _ in 0..5 {
            let run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
            registry.log_run(&run).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(10)); // Ensure different timestamps
        }

        // Get 3 most recent
        let recent = registry.recent_runs(3).unwrap();
        assert_eq!(recent.len(), 3);

        // Verify they're in reverse chronological order
        for i in 0..recent.len() - 1 {
            assert!(recent[i].started_at >= recent[i + 1].started_at);
        }
    }

    #[test]
    fn test_count_operations() {
        let (_temp, registry) = create_test_registry();

        assert_eq!(registry.count_runs().unwrap(), 0);
        assert_eq!(registry.count_artifacts().unwrap(), 0);

        // Add some records
        let run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
        registry.log_run(&run).unwrap();

        let artifact = super::super::ArtifactRecord::new(ArtifactKind::RLPolicy, None);
        registry.log_artifact(&artifact).unwrap();

        assert_eq!(registry.count_runs().unwrap(), 1);
        assert_eq!(registry.count_artifacts().unwrap(), 1);
    }

    #[test]
    fn test_empty_registry() {
        let (_temp, registry) = create_test_registry();

        // Operations on empty registry should not fail
        assert_eq!(registry.load_runs().unwrap().len(), 0);
        assert_eq!(registry.load_artifacts().unwrap().len(), 0);
        assert!(registry.find_run(RunId::new()).unwrap().is_none());
    }

    #[test]
    fn test_malformed_lines_are_skipped() {
        let (_temp, registry) = create_test_registry();

        // Manually write a malformed line
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(registry.runs_path.clone())
            .unwrap();
        writeln!(file, "{{invalid json}}").unwrap();

        // Log a valid run
        let run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
        registry.log_run(&run).unwrap();

        // Should load the valid run and skip the malformed one
        let runs = registry.load_runs().unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].id, run.id);
    }
}
