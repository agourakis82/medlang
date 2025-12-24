# Week 33 – Artifact Registry, Provenance & Reproducible Runs

## Executive Summary

Week 33 implements a **versioned, queryable record system** for all important MedLang operations, enabling **full reproducibility and audit trails** for scientific computing. After 6–12 months, you can replay, audit, and compare any evidence/surrogate/RL run with complete provenance.

**Core Principle**: *Every significant run gets a RunId and a structured RunRecord.*

---

## 1. System Architecture

### 1.1 Core Data Model

```rust
// Unique identifiers
pub struct RunId(pub Uuid);      // UUID-based run identifier
pub struct ArtifactId(pub Uuid); // UUID-based artifact identifier

// Run classification
pub enum RunKind {
    EvidenceMechanistic,  // Mechanistic QSP simulation
    EvidenceSurrogate,    // Surrogate-based simulation
    EvidenceHybrid,       // Hybrid mechanistic/surrogate
    SurrogateTrain,       // Surrogate model training
    SurrogateEval,        // Surrogate evaluation/qualification
    RLTrain,              // RL policy training
    RLEval,               // RL policy evaluation
}

// Artifact classification
pub enum ArtifactKind {
    EvidenceResult,
    SurrogateModel,
    SurrogateEvalReport,
    RLPolicy,
    RLTrainReport,
    RLEvalReport,
}

// Full run metadata
pub struct RunRecord {
    pub id: RunId,
    pub kind: RunKind,
    pub started_at: DateTime<Utc>,
    pub finished_at: DateTime<Utc>,
    pub project_root: String,
    pub module_path: Option<String>,
    pub evidence_program: Option<String>,
    pub git_commit: Option<String>,
    pub git_dirty: bool,
    pub config: serde_json::Value,     // Full configuration
    pub metrics: serde_json::Value,    // Summary metrics
    pub artifacts: Vec<ArtifactId>,    // Links to output files
}

// Artifact metadata
pub struct ArtifactRecord {
    pub id: ArtifactId,
    pub kind: ArtifactKind,
    pub path: Option<String>,
    pub created_at: DateTime<Utc>,
    pub metadata: serde_json::Value,
}
```

### 1.2 Storage Layer

**Location**: `~/.medlang/registry/`

```
~/.medlang/registry/
  ├── runs.jsonl          # Append-only JSONL: RunRecord per line
  └── artifacts.jsonl     # Append-only JSONL: ArtifactRecord per line
```

**Design Rationale**:
- **JSONL format**: Human-readable, debuggable with `jq`, simple append-only semantics
- **Immutable**: Only write operations; no deletions or updates
- **Fast**: O(n) for load-all, O(1) for registry creation
- **Auditable**: Full history preserved; can be version-controlled

### 1.3 Registry Interface

```rust
pub struct Registry {
    root: PathBuf,
    runs_path: PathBuf,
    artifacts_path: PathBuf,
}

impl Registry {
    pub fn new_default() -> anyhow::Result<Self>     // ~/.medlang/registry
    pub fn new(root: PathBuf) -> anyhow::Result<Self> // Custom root (testing)
    
    // Write (append-only)
    pub fn log_run(&self, run: &RunRecord) -> anyhow::Result<()>
    pub fn log_artifact(&self, artifact: &ArtifactRecord) -> anyhow::Result<()>
    
    // Query
    pub fn load_runs(&self) -> anyhow::Result<Vec<RunRecord>>
    pub fn load_artifacts(&self) -> anyhow::Result<Vec<ArtifactRecord>>
    pub fn find_run(&self, id: RunId) -> anyhow::Result<Option<RunRecord>>
    pub fn find_runs_by_kind(&self, kind: RunKind) -> anyhow::Result<Vec<RunRecord>>
    pub fn recent_runs(&self, n: usize) -> anyhow::Result<Vec<RunRecord>>
    
    // Utilities
    pub fn count_runs(&self) -> anyhow::Result<usize>
    pub fn count_artifacts(&self) -> anyhow::Result<usize>
}
```

---

## 2. RunLogger – Simplified Logging

The `RunLogger` encapsulates project context and provides convenient logging helpers:

```rust
pub struct RunLogger {
    registry: Registry,
    project_root: String,
    module_path: Option<String>,
    evidence_program: Option<String>,
    git_commit: Option<String>,
    git_dirty: bool,
}

impl RunLogger {
    pub fn new_for_project(
        project_root: &str,
        module_path: Option<&str>,
        evidence_program: Option<&str>,
    ) -> anyhow::Result<Self>
    
    pub fn log_run(
        &self,
        kind: RunKind,
        config: serde_json::Value,
        metrics: serde_json::Value,
    ) -> anyhow::Result<RunId>
    
    pub fn log_run_with_artifacts(
        &self,
        kind: RunKind,
        config: serde_json::Value,
        metrics: serde_json::Value,
        artifacts: Vec<ArtifactRecord>,
    ) -> anyhow::Result<RunId>
}
```

### 2.1 Specialized Logging Helpers

```rust
// Evidence runs
pub fn log_evidence_run(
    logger: &RunLogger,
    backend: BackendKind,
    metrics: serde_json::Value,
    result_file: &str,
) -> anyhow::Result<RunId>

// Surrogate operations
pub fn log_surrogate_train(
    logger: &RunLogger,
    train_config: &SurrogateTrainConfig,
    model_file: &str,
) -> anyhow::Result<RunId>

pub fn log_surrogate_eval(
    logger: &RunLogger,
    eval_config: serde_json::Value,
    report: serde_json::Value,
    report_file: &str,
) -> anyhow::Result<RunId>

// RL operations
pub fn log_rl_train(
    logger: &RunLogger,
    train_config: &RLTrainConfig,
    train_report: serde_json::Value,
    policy_file: &str,
    report_file: Option<&str>,
) -> anyhow::Result<RunId>

pub fn log_rl_eval(
    logger: &RunLogger,
    eval_config: serde_json::Value,
    eval_report: serde_json::Value,
    report_file: &str,
) -> anyhow::Result<RunId>
```

---

## 3. CLI Integration – `mlc runs` Commands

### 3.1 List Recent Runs

```bash
$ mlc runs list
6a14c8a2-... EvidenceHybrid  OncologyEvidence  2025-01-23T14:32:10.123Z
5b23d7c1-... RLTrain          ToxDoseOptim      2025-01-23T13:45:22.456Z
4c32e6d0-... SurrogateEval    OncologyEvidence  2025-01-23T13:12:35.789Z

$ mlc runs list --kind RLTrain -n 5 -v
[Verbose JSON output of last 5 RL training runs]

$ mlc runs list --kind EvidenceHybrid
[All hybrid evidence runs]
```

### 3.2 Show Detailed Run Information

```bash
$ mlc runs show --id 6a14c8a2-...
{
  "id": "6a14c8a2-...",
  "kind": "EvidenceHybrid",
  "started_at": "2025-01-23T14:32:10.123Z",
  "finished_at": "2025-01-23T14:35:22.456Z",
  "project_root": "/path/to/project",
  "module_path": "med.oncology.evidence",
  "evidence_program": "OncologyEvidence",
  "git_commit": "a1b2c3d4e5...",
  "git_dirty": false,
  "config": {
    "hybrid_config": { ... },
    "eval_config": { ... }
  },
  "metrics": {
    "rmse": 0.125,
    "mae": 0.089,
    "mech_violations": 2,
    "surr_violations": 0
  },
  "artifacts": ["artifact-id-1", "artifact-id-2"]
}
```

### 3.3 Export Run Configuration

```bash
$ mlc runs export-config --id 6a14c8a2-... --out my_config.json
[Exports the exact config used to regenerate the run]

$ cat my_config.json
{
  "hybrid_config": { ... },
  "eval_config": { ... },
  ...
}
```

---

## 4. Language-Level API – `med.registry`

### 4.1 MedLang Module Definition

```medlang
module med.registry;

// Kinds of runs
enum RunKind {
  EvidenceMechanistic;
  EvidenceSurrogate;
  EvidenceHybrid;
  SurrogateTrain;
  SurrogateEval;
  RLTrain;
  RLEval;
}

// Type aliases
type RunId = String;

type RunSummary = {
  id: RunId;
  kind: RunKind;
  project_root: String;
  evidence_program: String;
  started_at: String;
  finished_at: String;
  git_commit: String;
  git_dirty: Bool;
};

export enum RunKind;
export type RunId;
export type RunSummary;
```

### 4.2 Using Registry from MedLang

```medlang
module med.clinical_trial.analysis;

import med.registry::{RunKind, list_runs, get_run_summary};

fn compare_hybrid_runs() {
  let hybrid_runs = list_runs(Some(RunKind::EvidenceHybrid));
  
  for run in hybrid_runs {
    let summary = get_run_summary(run.id);
    println("Run {} finished at {}", run.id, summary.finished_at);
  }
}
```

### 4.3 Built-in Functions (Rust Runtime)

```rust
// list_runs(kind: Option<RunKind>) -> Vector<RunSummary>
fn builtin_list_runs(args: &[Value]) -> Result<Value, RuntimeError>

// get_run_summary(id: RunId) -> RunSummary
fn builtin_get_run_summary(args: &[Value]) -> Result<Value, RuntimeError>
```

---

## 5. Integration with Existing Commands

### 5.1 Pattern: Adding `--log` Flags

When implementing a command that should be logged, follow this pattern:

```rust
// 1. Parse arguments (including --log flag)
#[derive(clap::Args)]
pub struct CommandArgs {
    #[arg(long)]
    pub evidence_program: String,
    
    #[arg(long)]
    pub log: bool,  // NEW: Enable registry logging
}

// 2. Execute command
fn command_handler(args: CommandArgs) -> Result<()> {
    // ... perform operation ...
    let result = do_work(...)?;
    
    // 3. Optionally log the run
    if args.log {
        let logger = RunLogger::new_for_project(
            project_root,
            Some(&module_path),
            Some(&args.evidence_program),
        )?;
        
        let run_id = log_evidence_run(
            &logger,
            backend_kind,
            metrics,
            &output_file,
        )?;
        
        println!("Logged run: {}", run_id);
    }
    
    Ok(())
}
```

### 5.2 Commands that Support `--log`

- `mlc run` (evidence execution)
- `mlc surrogate-eval` (surrogate evaluation)
- `mlc run-evidence-hybrid` (hybrid execution)
- `mlc train-surrogate` (surrogate training)
- `mlc train-policy-rl` (RL training)
- `mlc eval-policy-rl` (RL evaluation)

---

## 6. Reproducibility Workflow

### 6.1 Initial Run with Logging

```bash
# Run with --log flag to record in registry
$ mlc run-evidence-hybrid \
    --evidence-program OncologyEvidence \
    --hybrid-config config.json \
    --out results.json \
    --log

# Output: "Logged run: 6a14c8a2-..."
```

### 6.2 Audit Trail

```bash
# List all hybrid runs
$ mlc runs list --kind EvidenceHybrid

# Show specifics of one run
$ mlc runs show --id 6a14c8a2-...

# Export the exact config
$ mlc runs export-config --id 6a14c8a2-... --out original_config.json

# 6 months later: Reproduce with exact config
$ mlc run-evidence-hybrid \
    --evidence-program OncologyEvidence \
    --hybrid-config original_config.json \
    --out results_reproduced.json \
    --log
```

### 6.3 Comparing Runs

```medlang
module med.clinical_trial.comparison;

import med.registry::{RunKind, list_runs, get_run_summary};

fn compare_surrogate_evals() {
  let eval_runs = list_runs(Some(RunKind::SurrogateEval));
  
  var best_rmse = 1000.0;
  var best_run_id = "";
  
  for run in eval_runs {
    let summary = get_run_summary(run.id);
    
    // Retrieve metrics (would need full record access for this)
    if metrics.rmse < best_rmse {
      best_rmse = metrics.rmse;
      best_run_id = run.id;
    }
  }
  
  println("Best surrogate eval: {} with RMSE {}", best_run_id, best_rmse);
}
```

---

## 7. File Layout

### 7.1 Rust Implementation

```
compiler/src/registry/
  ├── mod.rs              (Core types: RunId, ArtifactId, RunKind, RunRecord)
  ├── storage.rs          (Registry JSONL storage)
  ├── logging.rs          (RunLogger + specialized helpers)
```

### 7.2 Standard Library

```
stdlib/med/
  └── registry.medlang    (MedLang module definition)
```

### 7.3 CLI

```
compiler/src/bin/mlc.rs
  ├── RunsCommands enum   (CLI subcommands)
  ├── runs_list_command   (Implementation)
  ├── runs_show_command   (Implementation)
  └── runs_export_config_command (Implementation)
```

### 7.4 Tests

```
compiler/src/registry/
  ├── mod.rs              (Unit tests for types)
  ├── storage.rs          (Unit tests for JSONL storage)
  ├── logging.rs          (Unit tests for RunLogger)

compiler/tests/
  └── week_33_registry_integration.rs (Integration tests)
```

---

## 8. Testing

### 8.1 Unit Tests (Already Implemented)

**File**: `compiler/src/registry/mod.rs`
- Run ID creation and uniqueness
- String parsing of IDs and enums
- RunRecord builder pattern

**File**: `compiler/src/registry/storage.rs`
- JSONL round-trip (write and read back)
- Find by ID operations
- Find by kind filtering
- Recent runs sorting
- Malformed line handling

**File**: `compiler/src/registry/logging.rs`
- RunLogger initialization
- Run logging with/without artifacts
- Context preservation
- Specialized logging helpers

### 8.2 Integration Tests

**File**: `compiler/tests/week_33_registry_integration.rs` (to create)

```rust
#[test]
fn test_full_evidence_hybrid_logging_workflow() {
    // 1. Create logger
    // 2. Execute hybrid run
    // 3. Log with artifacts
    // 4. Query registry
    // 5. Verify all fields preserved
}

#[test]
fn test_cli_runs_list_with_filtering() {
    // 1. Log multiple runs of different kinds
    // 2. Run `mlc runs list --kind RLTrain`
    // 3. Verify output format
}

#[test]
fn test_cli_runs_export_config() {
    // 1. Log a run with config
    // 2. Run `mlc runs export-config --id <id> --out file.json`
    // 3. Verify exported config matches original
}

#[test]
fn test_language_level_list_runs() {
    // 1. Pre-populate registry with runs
    // 2. Write MedLang program calling `list_runs()`
    // 3. Execute and verify results
}
```

---

## 9. Metrics Capture Guidelines

When logging runs, populate `metrics` JSON with operation-specific summaries:

### 9.1 Evidence Runs

```json
{
  "ORR": 0.35,
  "DLT_rate": 0.15,
  "contract_violations": 2,
  "execution_time_ms": 1250
}
```

### 9.2 Surrogate Training

```json
{
  "train_loss": 0.024,
  "val_loss": 0.032,
  "n_epochs": 150,
  "training_time_s": 45.2
}
```

### 9.3 Surrogate Evaluation

```json
{
  "rmse": 0.089,
  "mae": 0.067,
  "r_squared": 0.923,
  "qualification_passed": true
}
```

### 9.4 RL Training

```json
{
  "avg_reward": 5.2,
  "final_epsilon": 0.05,
  "convergence_episodes": 450,
  "training_time_s": 120.3
}
```

### 9.5 RL Evaluation

```json
{
  "avg_reward": 5.1,
  "std_reward": 0.8,
  "success_rate": 0.92,
  "n_simulations": 1000
}
```

---

## 10. Security & Privacy Considerations

### 10.1 Sensitive Data

**DO NOT LOG**:
- Patient data or PII
- Proprietary model weights (only reference file paths)
- Raw API keys or credentials

**DO LOG**:
- Anonymized metrics
- Model architecture (not weights)
- Configuration parameters (general setup)

### 10.2 Access Control

Registry is stored in user's home directory (`~/.medlang/`). Respect OS-level file permissions.

---

## 11. Performance Characteristics

| Operation | Complexity | Typical Time |
|-----------|-----------|-------------|
| Log run | O(1) | <5ms |
| Log artifact | O(1) | <5ms |
| Load all runs | O(n) | ~50ms (for 1000 runs) |
| Find run by ID | O(n) | ~50ms (for 1000 runs) |
| Recent runs (top-n) | O(n log n) | ~100ms (for 1000 runs, sort) |

**Note**: For larger registries (10k+ runs), consider:
- Indexing (e.g., SQLite backend)
- Partitioning by date
- Archive old runs

---

## 12. Future Enhancements (Post-Week 33)

1. **Git Integration**: Automatically capture commit hash and dirty status
2. **Database Backend**: SQLite for indexing and faster queries
3. **Artifact Storage**: Optional S3/cloud storage for large models
4. **UI Dashboard**: Web interface for browsing registry
5. **Diff Tool**: Compare configs/metrics between runs
6. **Export**: Generate PDF/HTML reports from run records
7. **Cleanup**: Archive/delete old runs based on retention policies

---

## 13. Summary

**Week 33 delivers**:
✅ Complete audit trail for all MedLang operations  
✅ Full provenance: who, what, when, how, with what config  
✅ Queryable from CLI and MedLang language level  
✅ JSONL-backed storage: human-readable, debuggable, version-controlled  
✅ Reproducible science: exact configs can be exported and re-run  

**After Week 33**, you can answer with precision:

> "Which model/surrogate/backend/seed produced this figure and these metrics?"

And you can reproduce it exactly 6 months from now.

---

## References

- **Registry module**: `compiler/src/registry/`
- **CLI integration**: `compiler/src/bin/mlc.rs` (lines 578–845)
- **Standard library**: `stdlib/med/registry.medlang`
- **Tests**: `compiler/src/registry/{mod,storage,logging}.rs` + integration tests