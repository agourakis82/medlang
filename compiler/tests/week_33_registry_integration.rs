// Week 33: Registry Integration Tests
//
// Tests for the Artifact Registry, Provenance & Reproducible Runs system.
// Verifies round-trip storage, querying, and end-to-end workflows.

use medlangc::registry::{
    logging::{log_evidence_run, log_surrogate_eval},
    storage::Registry,
    ArtifactId, ArtifactKind, ArtifactRecord, RunId, RunKind, RunLogger, RunRecord,
};
use tempfile::TempDir;

// =============================================================================
// Helpers
// =============================================================================

fn create_test_registry() -> (TempDir, Registry) {
    let temp_dir = TempDir::new().unwrap();
    let registry = Registry::new(temp_dir.path().to_path_buf()).unwrap();
    (temp_dir, registry)
}

fn create_test_logger(temp_dir: &TempDir) -> RunLogger {
    let registry = Registry::new(temp_dir.path().to_path_buf()).unwrap();
    RunLogger::with_registry(
        registry,
        "/test/project",
        Some("med.test.module"),
        Some("TestEvidence"),
    )
}

// =============================================================================
// Basic Storage Round-Trip Tests
// =============================================================================

#[test]
fn test_run_record_round_trip() {
    let (_temp, registry) = create_test_registry();

    // Create and log a run
    let original_run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string())
        .with_module_path("med.rl.training".to_string())
        .with_evidence_program("RLTrainer".to_string())
        .with_config(serde_json::json!({
            "n_episodes": 1000,
            "gamma": 0.95,
            "alpha": 0.1,
        }))
        .with_metrics(serde_json::json!({
            "avg_reward": 5.2,
            "final_epsilon": 0.05,
        }));

    let original_id = original_run.id;
    registry.log_run(&original_run).unwrap();

    // Load and verify
    let loaded_run = registry.find_run(original_id).unwrap();
    assert!(loaded_run.is_some());

    let loaded = loaded_run.unwrap();
    assert_eq!(loaded.id, original_run.id);
    assert_eq!(loaded.kind, RunKind::RLTrain);
    assert_eq!(loaded.module_path, Some("med.rl.training".to_string()));
    assert_eq!(loaded.config["n_episodes"], 1000);
    assert_eq!(loaded.metrics["avg_reward"], 5.2);
}

#[test]
fn test_artifact_record_round_trip() {
    let (_temp, registry) = create_test_registry();

    // Create and log an artifact
    let original_artifact = ArtifactRecord::new(
        ArtifactKind::RLPolicy,
        Some("/output/policy.json".to_string()),
    )
    .with_metadata(serde_json::json!({
        "n_states": 100,
        "n_actions": 5,
        "training_episodes": 1000,
    }));

    let original_id = original_artifact.id;
    registry.log_artifact(&original_artifact).unwrap();

    // Load and verify
    let loaded_artifact = registry.find_artifact(original_id).unwrap();
    assert!(loaded_artifact.is_some());

    let loaded = loaded_artifact.unwrap();
    assert_eq!(loaded.id, original_artifact.id);
    assert_eq!(loaded.kind, ArtifactKind::RLPolicy);
    assert_eq!(loaded.path, Some("/output/policy.json".to_string()));
    assert_eq!(loaded.metadata["n_states"], 100);
}

// =============================================================================
// RunLogger Tests
// =============================================================================

#[test]
fn test_run_logger_preserves_context() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    // Log a run without artifacts
    let config = serde_json::json!({"learning_rate": 0.01});
    let metrics = serde_json::json!({"accuracy": 0.92});
    let run_id = logger.log_run(RunKind::RLTrain, config, metrics).unwrap();

    // Verify context was preserved
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    assert_eq!(run.project_root, "/test/project");
    assert_eq!(run.module_path, Some("med.test.module".to_string()));
    assert_eq!(run.evidence_program, Some("TestEvidence".to_string()));
    assert_eq!(run.kind, RunKind::RLTrain);
}

#[test]
fn test_run_logger_with_artifacts() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    let artifact1 = ArtifactRecord::new(ArtifactKind::RLPolicy, Some("/policy.json".into()));
    let artifact2 = ArtifactRecord::new(ArtifactKind::RLTrainReport, Some("/report.json".into()));

    let config = serde_json::json!({"test": "config"});
    let metrics = serde_json::json!({"test": "metrics"});

    let run_id = logger
        .log_run_with_artifacts(
            RunKind::RLTrain,
            config,
            metrics,
            vec![artifact1, artifact2],
        )
        .unwrap();

    // Verify run and artifacts were logged
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    assert_eq!(run.artifacts.len(), 2);

    // Verify artifacts can be found
    for artifact_id in &run.artifacts {
        let artifact = logger.registry().find_artifact(*artifact_id).unwrap();
        assert!(artifact.is_some());
    }
}

// =============================================================================
// Specialized Logging Helper Tests
// =============================================================================

#[test]
fn test_log_evidence_run_helper() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    let metrics = serde_json::json!({
        "ORR": 0.35,
        "DLT_rate": 0.15,
        "n_subjects": 30,
    });

    let run_id = log_evidence_run(
        &logger,
        medlangc::ml::BackendKind::Hybrid,
        metrics,
        "/output/evidence_result.json",
    )
    .unwrap();

    // Verify run was logged correctly
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    assert_eq!(run.kind, RunKind::EvidenceHybrid);
    assert_eq!(run.artifacts.len(), 1);

    // Verify artifact
    let artifact = logger
        .registry()
        .find_artifact(run.artifacts[0])
        .unwrap()
        .unwrap();
    assert_eq!(artifact.kind, ArtifactKind::EvidenceResult);
    assert_eq!(
        artifact.path,
        Some("/output/evidence_result.json".to_string())
    );
}

#[test]
fn test_log_surrogate_eval_helper() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    let eval_config = serde_json::json!({
        "n_test_points": 1000,
        "validation_strategy": "cross_validation",
    });

    let report = serde_json::json!({
        "rmse": 0.089,
        "mae": 0.067,
        "r_squared": 0.923,
        "qualification_passed": true,
    });

    let run_id =
        log_surrogate_eval(&logger, eval_config, report, "/output/eval_report.json").unwrap();

    // Verify run was logged
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    assert_eq!(run.kind, RunKind::SurrogateEval);
    assert_eq!(run.metrics["rmse"], 0.089);
    assert_eq!(run.metrics["qualification_passed"], true);
}

#[test]
fn test_log_rl_train_via_logger() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    // Simulate RL training run
    let train_report = serde_json::json!({
        "avg_reward": 5.2,
        "final_epsilon": 0.05,
        "convergence_episode": 450,
    });

    let train_config = serde_json::json!({
        "n_episodes": 1000,
        "max_steps_per_episode": 10,
        "gamma": 0.95,
        "alpha": 0.1,
    });

    // Log as artifacts
    let policy_artifact = ArtifactRecord::new(
        ArtifactKind::RLPolicy,
        Some("/output/policy.json".to_string()),
    );
    let report_artifact = ArtifactRecord::new(
        ArtifactKind::RLTrainReport,
        Some("/output/train_report.json".to_string()),
    );

    let run_id = logger
        .log_run_with_artifacts(
            RunKind::RLTrain,
            train_config,
            train_report,
            vec![policy_artifact, report_artifact],
        )
        .unwrap();

    // Verify run and artifacts
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    assert_eq!(run.kind, RunKind::RLTrain);
    assert_eq!(run.artifacts.len(), 2); // Policy + report

    // Verify artifact kinds
    let policy_artifact = logger
        .registry()
        .find_artifact(run.artifacts[0])
        .unwrap()
        .unwrap();
    assert_eq!(policy_artifact.kind, ArtifactKind::RLPolicy);

    let report_artifact = logger
        .registry()
        .find_artifact(run.artifacts[1])
        .unwrap()
        .unwrap();
    assert_eq!(report_artifact.kind, ArtifactKind::RLTrainReport);
}

// =============================================================================
// Query and Filtering Tests
// =============================================================================

#[test]
fn test_find_runs_by_kind() {
    let (_temp, registry) = create_test_registry();

    // Log runs of different kinds
    let run1 = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
    let run2 = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
    let run3 = RunRecord::new(RunKind::SurrogateEval, "/test/project".to_string());
    let run4 = RunRecord::new(RunKind::EvidenceHybrid, "/test/project".to_string());

    registry.log_run(&run1).unwrap();
    registry.log_run(&run2).unwrap();
    registry.log_run(&run3).unwrap();
    registry.log_run(&run4).unwrap();

    // Query by kind
    let rl_runs = registry.find_runs_by_kind(RunKind::RLTrain).unwrap();
    assert_eq!(rl_runs.len(), 2);

    let eval_runs = registry.find_runs_by_kind(RunKind::SurrogateEval).unwrap();
    assert_eq!(eval_runs.len(), 1);

    let hybrid_runs = registry.find_runs_by_kind(RunKind::EvidenceHybrid).unwrap();
    assert_eq!(hybrid_runs.len(), 1);

    // Non-existent kind
    let empty_runs = registry.find_runs_by_kind(RunKind::RLEval).unwrap();
    assert_eq!(empty_runs.len(), 0);
}

#[test]
fn test_recent_runs_sorting() {
    let (_temp, registry) = create_test_registry();

    // Log 5 runs with small delays to ensure different timestamps
    for i in 0..5 {
        let run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
        registry.log_run(&run).unwrap();
        if i < 4 {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
    }

    // Get 3 most recent
    let recent = registry.recent_runs(3).unwrap();
    assert_eq!(recent.len(), 3);

    // Verify reverse chronological order
    for i in 0..recent.len() - 1 {
        assert!(recent[i].started_at >= recent[i + 1].started_at);
    }
}

#[test]
fn test_find_artifacts_by_kind() {
    let (_temp, registry) = create_test_registry();

    // Log artifacts of different kinds
    let art1 = ArtifactRecord::new(ArtifactKind::RLPolicy, None);
    let art2 = ArtifactRecord::new(ArtifactKind::RLPolicy, None);
    let art3 = ArtifactRecord::new(ArtifactKind::SurrogateModel, None);
    let art4 = ArtifactRecord::new(ArtifactKind::EvidenceResult, None);

    registry.log_artifact(&art1).unwrap();
    registry.log_artifact(&art2).unwrap();
    registry.log_artifact(&art3).unwrap();
    registry.log_artifact(&art4).unwrap();

    // Query by kind
    let policies = registry
        .find_artifacts_by_kind(ArtifactKind::RLPolicy)
        .unwrap();
    assert_eq!(policies.len(), 2);

    let models = registry
        .find_artifacts_by_kind(ArtifactKind::SurrogateModel)
        .unwrap();
    assert_eq!(models.len(), 1);

    let results = registry
        .find_artifacts_by_kind(ArtifactKind::EvidenceResult)
        .unwrap();
    assert_eq!(results.len(), 1);
}

// =============================================================================
// Statistics Tests
// =============================================================================

#[test]
fn test_registry_counts() {
    let (_temp, registry) = create_test_registry();

    assert_eq!(registry.count_runs().unwrap(), 0);
    assert_eq!(registry.count_artifacts().unwrap(), 0);

    // Add runs
    for _ in 0..3 {
        let run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
        registry.log_run(&run).unwrap();
    }

    assert_eq!(registry.count_runs().unwrap(), 3);

    // Add artifacts
    for _ in 0..2 {
        let artifact = ArtifactRecord::new(ArtifactKind::RLPolicy, None);
        registry.log_artifact(&artifact).unwrap();
    }

    assert_eq!(registry.count_artifacts().unwrap(), 2);
}

// =============================================================================
// End-to-End Workflow Tests
// =============================================================================

#[test]
fn test_full_hybrid_evidence_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    // Simulate a hybrid evidence run
    let metrics = serde_json::json!({
        "rmse": 0.125,
        "mae": 0.089,
        "mech_violations": 2,
        "surr_violations": 0,
        "execution_time_ms": 2500,
    });

    let config = serde_json::json!({
        "backend": "Hybrid",
        "hybrid_strategy": "adaptive",
        "mech_threshold": 0.1,
    });

    // Log with artifacts
    let artifact = ArtifactRecord::new(
        ArtifactKind::EvidenceResult,
        Some("/output/hybrid_result.json".to_string()),
    )
    .with_metadata(serde_json::json!({
        "n_subjects": 100,
        "n_timepoints": 24,
    }));

    let run_id = logger
        .log_run_with_artifacts(RunKind::EvidenceHybrid, config, metrics, vec![artifact])
        .unwrap();

    // Verify complete record
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    assert_eq!(run.kind, RunKind::EvidenceHybrid);
    assert_eq!(run.module_path, Some("med.test.module".to_string()));
    assert_eq!(run.metrics["rmse"], 0.125);
    assert_eq!(run.config["backend"], "Hybrid");
    assert_eq!(run.artifacts.len(), 1);

    // Verify artifact
    let artifact = logger
        .registry()
        .find_artifact(run.artifacts[0])
        .unwrap()
        .unwrap();
    assert_eq!(
        artifact.path,
        Some("/output/hybrid_result.json".to_string())
    );
    assert_eq!(artifact.metadata["n_subjects"], 100);
}

#[test]
fn test_reproducibility_export_workflow() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    // Initial run
    let original_config = serde_json::json!({
        "algorithm": "Q-Learning",
        "learning_rate": 0.01,
        "gamma": 0.95,
        "epsilon_decay": 0.995,
    });

    let metrics = serde_json::json!({
        "avg_reward": 5.2,
        "episodes_to_converge": 450,
    });

    let run_id = logger
        .log_run(RunKind::RLTrain, original_config.clone(), metrics)
        .unwrap();

    // Export the config for later reproduction
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    let exported_config = run.config.clone();

    // Verify exported config matches original
    assert_eq!(exported_config, original_config);
    assert_eq!(exported_config["learning_rate"], 0.01);
    assert_eq!(exported_config["gamma"], 0.95);
}

#[test]
fn test_multiple_runs_querying() {
    let temp_dir = TempDir::new().unwrap();
    let registry = Registry::new(temp_dir.path().to_path_buf()).unwrap();

    // Log multiple runs for different operations
    let mut run_ids = Vec::new();

    // 2 evidence runs
    for i in 0..2 {
        let run = RunRecord::new(RunKind::EvidenceMechanistic, "/test/project".to_string())
            .with_config(serde_json::json!({"seed": i}));
        registry.log_run(&run).unwrap();
        run_ids.push((RunKind::EvidenceMechanistic, run.id));
    }

    // 3 surrogate eval runs
    for i in 0..3 {
        let run = RunRecord::new(RunKind::SurrogateEval, "/test/project".to_string())
            .with_config(serde_json::json!({"eval_set": i}));
        registry.log_run(&run).unwrap();
        run_ids.push((RunKind::SurrogateEval, run.id));
    }

    // 1 RL training run
    let rl_run = RunRecord::new(RunKind::RLTrain, "/test/project".to_string());
    registry.log_run(&rl_run).unwrap();
    run_ids.push((RunKind::RLTrain, rl_run.id));

    // Verify queries
    let all_runs = registry.load_runs().unwrap();
    assert_eq!(all_runs.len(), 6);

    let mech_runs = registry
        .find_runs_by_kind(RunKind::EvidenceMechanistic)
        .unwrap();
    assert_eq!(mech_runs.len(), 2);

    let eval_runs = registry.find_runs_by_kind(RunKind::SurrogateEval).unwrap();
    assert_eq!(eval_runs.len(), 3);

    let rl_runs = registry.find_runs_by_kind(RunKind::RLTrain).unwrap();
    assert_eq!(rl_runs.len(), 1);

    // Verify each run can be found individually
    for (_, run_id) in run_ids {
        let found = registry.find_run(run_id).unwrap();
        assert!(found.is_some());
    }
}

#[test]
fn test_metrics_preservation() {
    let temp_dir = TempDir::new().unwrap();
    let logger = create_test_logger(&temp_dir);

    // Log with detailed metrics
    let complex_metrics = serde_json::json!({
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.88,
        "f1_score": 0.90,
        "confusion_matrix": {
            "TP": 450,
            "TN": 2000,
            "FP": 50,
            "FN": 50,
        },
        "confidence_intervals": {
            "accuracy": [0.93, 0.97],
            "f1": [0.88, 0.92],
        },
    });

    let run_id = logger
        .log_run(
            RunKind::SurrogateEval,
            serde_json::json!({}),
            complex_metrics.clone(),
        )
        .unwrap();

    // Verify exact metrics preservation
    let run = logger.registry().find_run(run_id).unwrap().unwrap();
    assert_eq!(run.metrics, complex_metrics);
    assert_eq!(run.metrics["accuracy"], 0.95);
    assert_eq!(run.metrics["confusion_matrix"]["TP"], 450);
    assert_eq!(run.metrics["confidence_intervals"]["accuracy"][0], 0.93);
}
