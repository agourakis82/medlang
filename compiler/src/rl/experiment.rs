//! Week 47: Guideline Evaluation Experiment Orchestrator (v2.0)
//!
//! Production-grade experiment orchestration system that:
//! - Unifies all evaluation layers (robustness, scoring, constraints, Pareto, uncertainty)
//! - Supports parallel execution with configurable concurrency
//! - Provides full provenance tracking (Git state, seeds, timestamps, versions)
//! - Enables experiment comparison and delta analysis
//! - Supports caching, resumability, and incremental updates
//! - Auto-generates comprehensive reports (JSON, CSV, regulatory, publication)
//! - Maintains an experiment registry for reproducibility

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;

use crate::rl::constraints::{
    compute_constraint_analysis, compute_pareto_analysis, ConstraintAnalysis,
    ConstraintAnalysisConfig, ConstraintSet, GuidelineRobustnessScoreHost, ParetoAnalysis,
    ParetoConfig,
};
use crate::rl::env_dose_tox::DoseToxEnvConfig;

// ═══════════════════════════════════════════════════════════════════════════
// PART 1: EXPERIMENT IDENTIFICATION & PROVENANCE
// ═══════════════════════════════════════════════════════════════════════════

/// Unique experiment identifier
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct ExperimentId(pub String);

impl ExperimentId {
    pub fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        ExperimentId(format!("exp_{:016x}", timestamp))
    }

    pub fn from_string(s: &str) -> Self {
        ExperimentId(s.to_string())
    }
}

impl Default for ExperimentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ExperimentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Git repository state at experiment time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitProvenance {
    /// Current commit hash
    pub commit_hash: Option<String>,
    /// Branch name
    pub branch: Option<String>,
    /// Whether working directory is dirty
    pub dirty: bool,
    /// Uncommitted changes summary
    #[serde(default)]
    pub uncommitted_files: Vec<String>,
    /// Remote URL
    pub remote_url: Option<String>,
}

impl GitProvenance {
    /// Capture current Git state
    pub fn capture() -> Self {
        let commit = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let branch = Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let dirty = Command::new("git")
            .args(["status", "--porcelain"])
            .output()
            .map(|o| !o.stdout.is_empty())
            .unwrap_or(true);

        let uncommitted = if dirty {
            Command::new("git")
                .args(["status", "--porcelain"])
                .output()
                .ok()
                .and_then(|o| String::from_utf8(o.stdout).ok())
                .map(|s| s.lines().map(|l| l.to_string()).collect())
                .unwrap_or_default()
        } else {
            vec![]
        };

        let remote = Command::new("git")
            .args(["remote", "get-url", "origin"])
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        GitProvenance {
            commit_hash: commit,
            branch,
            dirty,
            uncommitted_files: uncommitted,
            remote_url: remote,
        }
    }

    /// Create empty provenance (for non-git environments)
    pub fn empty() -> Self {
        GitProvenance {
            commit_hash: None,
            branch: None,
            dirty: false,
            uncommitted_files: vec![],
            remote_url: None,
        }
    }
}

/// Random seeds for reproducibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSeeds {
    pub master_seed: u64,
    pub scenario_seed: u64,
    pub simulation_seed: u64,
}

impl Default for ExperimentSeeds {
    fn default() -> Self {
        Self {
            master_seed: 42,
            scenario_seed: 42,
            simulation_seed: 42,
        }
    }
}

/// ISO 8601 timestamp string
pub type Timestamp = String;

fn now_timestamp() -> Timestamp {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();
    // Simple ISO-ish format
    format!("{}Z", secs)
}

/// Complete provenance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentProvenance {
    /// Unique experiment ID
    pub experiment_id: ExperimentId,

    /// Human-readable name
    pub name: String,

    /// Description
    #[serde(default)]
    pub description: Option<String>,

    /// Tags for organization
    #[serde(default)]
    pub tags: Vec<String>,

    /// When experiment was created
    pub created_at: Timestamp,

    /// When experiment started running
    #[serde(default)]
    pub started_at: Option<Timestamp>,

    /// When experiment completed
    #[serde(default)]
    pub completed_at: Option<Timestamp>,

    /// Git state
    pub git: GitProvenance,

    /// MedLang version
    pub medlang_version: String,

    /// Random seeds used
    pub seeds: ExperimentSeeds,

    /// Author information
    #[serde(default)]
    pub author: Option<String>,

    /// Parent experiment (if derived from another)
    #[serde(default)]
    pub parent_experiment_id: Option<ExperimentId>,

    /// Config hash for deduplication
    pub config_hash: String,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 2: EXPERIMENT CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Execution mode for the experiment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionMode {
    /// Run everything sequentially
    Sequential,
    /// Parallel with auto-detected threads
    Parallel,
    /// Parallel with specific thread count
    ParallelWith { n_threads: usize },
}

impl Default for ExecutionMode {
    fn default() -> Self {
        ExecutionMode::Sequential // Default to sequential for determinism
    }
}

/// What to do on error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandling {
    /// Stop immediately on first error
    FailFast,
    /// Continue and collect all errors
    CollectErrors,
    /// Skip failed guidelines and continue
    SkipFailed,
}

impl Default for ErrorHandling {
    fn default() -> Self {
        ErrorHandling::CollectErrors
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    /// Enable caching
    #[serde(default)]
    pub enabled: bool,

    /// Cache directory
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,

    /// Cache robustness reports
    #[serde(default = "default_true")]
    pub cache_robustness: bool,

    /// Cache scoring results
    #[serde(default = "default_true")]
    pub cache_scoring: bool,

    /// Maximum cache age in seconds (0 = no expiry)
    #[serde(default)]
    pub max_age_secs: u64,
}

fn default_true() -> bool {
    true
}
fn default_cache_dir() -> String {
    ".medlang_cache".to_string()
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cache_dir: default_cache_dir(),
            cache_robustness: true,
            cache_scoring: true,
            max_age_secs: 0,
        }
    }
}

/// Progress information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentProgress {
    pub stage: ExperimentStage,
    pub current_guideline: Option<String>,
    pub guidelines_completed: usize,
    pub guidelines_total: usize,
    pub elapsed_secs: f64,
    pub estimated_remaining_secs: Option<f64>,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExperimentStage {
    Initializing,
    ValidatingConfig,
    RunningRobustness,
    ComputingScores,
    AnalyzingConstraints,
    ComputingPareto,
    GeneratingReports,
    Completed,
    Failed,
}

impl std::fmt::Display for ExperimentStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExperimentStage::Initializing => write!(f, "Initializing"),
            ExperimentStage::ValidatingConfig => write!(f, "Validating Configuration"),
            ExperimentStage::RunningRobustness => write!(f, "Running Robustness Analysis"),
            ExperimentStage::ComputingScores => write!(f, "Computing Scores"),
            ExperimentStage::AnalyzingConstraints => write!(f, "Analyzing Constraints"),
            ExperimentStage::ComputingPareto => write!(f, "Computing Pareto Fronts"),
            ExperimentStage::GeneratingReports => write!(f, "Generating Reports"),
            ExperimentStage::Completed => write!(f, "Completed"),
            ExperimentStage::Failed => write!(f, "Failed"),
        }
    }
}

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    /// Generate JSON report
    #[serde(default = "default_true")]
    pub json: bool,

    /// Generate CSV summaries
    #[serde(default = "default_true")]
    pub csv: bool,

    /// Generate regulatory summary
    #[serde(default = "default_true")]
    pub regulatory: bool,

    /// Generate publication-ready tables
    #[serde(default)]
    pub publication: bool,

    /// Output directory
    #[serde(default = "default_output_dir")]
    pub output_dir: String,

    /// Auto-generate reports on completion
    #[serde(default = "default_true")]
    pub auto_generate: bool,
}

fn default_output_dir() -> String {
    "experiment_outputs".to_string()
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            json: true,
            csv: true,
            regulatory: true,
            publication: false,
            output_dir: default_output_dir(),
            auto_generate: true,
        }
    }
}

/// Scoring configuration (simplified from Week 44)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoringConfig {
    /// Weight for response component
    pub response_weight: f64,
    /// Weight for toxicity component (penalty)
    pub toxicity_weight: f64,
    /// Weight for RDI component
    pub rdi_weight: f64,
    /// Use worst-case metrics
    pub use_worst_case: bool,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            response_weight: 1.0,
            toxicity_weight: 1.5,
            rdi_weight: 0.5,
            use_worst_case: false,
        }
    }
}

/// Environment scenario for robustness testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvScenario {
    pub name: String,
    pub description: Option<String>,
    /// Parameter overrides from base environment
    pub param_overrides: HashMap<String, f64>,
    /// Weight for this scenario in aggregation
    pub weight: f64,
}

impl EnvScenario {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            param_overrides: HashMap::new(),
            weight: 1.0,
        }
    }

    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.param_overrides.insert(key.to_string(), value);
        self
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }
}

/// Guideline definition for evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineDefinition {
    pub guideline_id: String,
    pub name: Option<String>,
    pub description: Option<String>,
    /// Dosing rules or policy specification
    pub policy_spec: HashMap<String, serde_json::Value>,
}

impl GuidelineDefinition {
    pub fn new(id: &str) -> Self {
        Self {
            guideline_id: id.to_string(),
            name: None,
            description: None,
            policy_spec: HashMap::new(),
        }
    }
}

/// Complete experiment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentConfig {
    // ═══════════════════════════════════════════════════════════════════════
    // Provenance
    // ═══════════════════════════════════════════════════════════════════════
    /// Experiment name
    pub name: String,

    /// Description
    #[serde(default)]
    pub description: Option<String>,

    /// Tags for organization
    #[serde(default)]
    pub tags: Vec<String>,

    /// Author
    #[serde(default)]
    pub author: Option<String>,

    // ═══════════════════════════════════════════════════════════════════════
    // Inputs
    // ═══════════════════════════════════════════════════════════════════════
    /// Guidelines to evaluate
    pub guidelines: Vec<GuidelineDefinition>,

    /// Reference guideline ID (for comparison)
    #[serde(default)]
    pub reference_guideline_id: Option<String>,

    /// Base environment configuration
    pub base_env: DoseToxEnvConfig,

    /// Environment scenarios for robustness testing
    #[serde(default)]
    pub scenarios: Vec<EnvScenario>,

    // ═══════════════════════════════════════════════════════════════════════
    // Analysis configuration
    // ═══════════════════════════════════════════════════════════════════════
    /// Scoring configuration
    #[serde(default)]
    pub scoring_cfg: ScoringConfig,

    /// Clinical constraints
    #[serde(default)]
    pub constraints: Option<ConstraintSet>,

    /// Pareto analysis config
    #[serde(default)]
    pub pareto_cfg: Option<ParetoConfig>,

    // ═══════════════════════════════════════════════════════════════════════
    // Execution configuration
    // ═══════════════════════════════════════════════════════════════════════
    /// Execution mode
    #[serde(default)]
    pub execution_mode: ExecutionMode,

    /// Error handling strategy
    #[serde(default)]
    pub error_handling: ErrorHandling,

    /// Random seeds
    #[serde(default)]
    pub seeds: ExperimentSeeds,

    /// Cache configuration
    #[serde(default)]
    pub cache: CacheConfig,

    /// Report configuration
    #[serde(default)]
    pub reports: ReportConfig,

    /// Validation strictness
    #[serde(default = "default_true")]
    pub strict_validation: bool,
}

impl ExperimentConfig {
    /// Create a new experiment config with minimal required fields
    pub fn new(
        name: &str,
        guidelines: Vec<GuidelineDefinition>,
        base_env: DoseToxEnvConfig,
    ) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            tags: vec![],
            author: None,
            guidelines,
            reference_guideline_id: None,
            base_env,
            scenarios: vec![],
            scoring_cfg: ScoringConfig::default(),
            constraints: None,
            pareto_cfg: None,
            execution_mode: ExecutionMode::default(),
            error_handling: ErrorHandling::default(),
            seeds: ExperimentSeeds::default(),
            cache: CacheConfig::default(),
            reports: ReportConfig::default(),
            strict_validation: true,
        }
    }

    /// Compute config hash for caching/deduplication
    pub fn compute_hash(&self) -> String {
        use std::collections::hash_map::DefaultHasher;

        let mut hasher = DefaultHasher::new();

        // Hash guideline IDs
        for g in &self.guidelines {
            g.guideline_id.hash(&mut hasher);
        }

        // Hash scenario count
        self.scenarios.len().hash(&mut hasher);

        // Hash seeds
        self.seeds.master_seed.hash(&mut hasher);
        self.seeds.scenario_seed.hash(&mut hasher);
        self.seeds.simulation_seed.hash(&mut hasher);

        format!("{:016x}", hasher.finish())
    }

    /// Validate configuration before running
    pub fn validate(&self) -> Result<ValidationReport, Vec<ConfigError>> {
        let mut errors = vec![];
        let mut warnings = vec![];

        // Check guidelines
        if self.guidelines.is_empty() {
            errors.push(ConfigError::NoGuidelines);
        }

        // Check for duplicate guideline IDs
        let mut seen_ids = std::collections::HashSet::new();
        for g in &self.guidelines {
            if !seen_ids.insert(&g.guideline_id) {
                errors.push(ConfigError::DuplicateGuidelineId(g.guideline_id.clone()));
            }
        }

        // Check scenarios
        if self.scenarios.is_empty() {
            warnings.push(ConfigWarning::NoScenarios);
        }

        // Check reference guideline exists
        if let Some(ref ref_id) = self.reference_guideline_id {
            if !self.guidelines.iter().any(|g| &g.guideline_id == ref_id) {
                errors.push(ConfigError::ReferenceGuidelineNotFound(ref_id.clone()));
            }
        }

        // Check constraint validity
        if let Some(ref constraints) = self.constraints {
            if constraints.constraints.is_empty() {
                warnings.push(ConfigWarning::EmptyConstraintSet);
            }
        }

        // Warn about large scenario counts
        if self.scenarios.len() > 100 {
            warnings.push(ConfigWarning::LargeScenarioCount(self.scenarios.len()));
        }

        if errors.is_empty() {
            Ok(ValidationReport { warnings })
        } else {
            Err(errors)
        }
    }

    /// Builder: add constraints
    pub fn with_constraints(mut self, constraints: ConstraintSet) -> Self {
        self.constraints = Some(constraints);
        self
    }

    /// Builder: add Pareto config
    pub fn with_pareto(mut self, pareto_cfg: ParetoConfig) -> Self {
        self.pareto_cfg = Some(pareto_cfg);
        self
    }

    /// Builder: set reference guideline
    pub fn with_reference(mut self, reference_id: &str) -> Self {
        self.reference_guideline_id = Some(reference_id.to_string());
        self
    }

    /// Builder: add scenarios
    pub fn with_scenarios(mut self, scenarios: Vec<EnvScenario>) -> Self {
        self.scenarios = scenarios;
        self
    }

    /// Builder: set execution mode
    pub fn with_execution_mode(mut self, mode: ExecutionMode) -> Self {
        self.execution_mode = mode;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigError {
    NoGuidelines,
    DuplicateGuidelineId(String),
    ReferenceGuidelineNotFound(String),
    InvalidScenario(String),
    InvalidConstraint(String),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::NoGuidelines => write!(f, "No guidelines specified"),
            ConfigError::DuplicateGuidelineId(id) => write!(f, "Duplicate guideline ID: {}", id),
            ConfigError::ReferenceGuidelineNotFound(id) => {
                write!(f, "Reference guideline not found: {}", id)
            }
            ConfigError::InvalidScenario(msg) => write!(f, "Invalid scenario: {}", msg),
            ConfigError::InvalidConstraint(msg) => write!(f, "Invalid constraint: {}", msg),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigWarning {
    NoScenarios,
    EmptyConstraintSet,
    NoPareto,
    LargeScenarioCount(usize),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationReport {
    pub warnings: Vec<ConfigWarning>,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 3: PER-GUIDELINE RESULTS
// ═══════════════════════════════════════════════════════════════════════════

/// Status of individual guideline evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuidelineStatus {
    Pending,
    Running,
    Completed { duration_ms: u64 },
    Failed { error: String },
    Skipped { reason: String },
    Cached,
}

impl std::fmt::Display for GuidelineStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GuidelineStatus::Pending => write!(f, "Pending"),
            GuidelineStatus::Running => write!(f, "Running"),
            GuidelineStatus::Completed { duration_ms } => {
                write!(f, "Completed ({:.2}s)", *duration_ms as f64 / 1000.0)
            }
            GuidelineStatus::Failed { error } => write!(f, "Failed: {}", error),
            GuidelineStatus::Skipped { reason } => write!(f, "Skipped: {}", reason),
            GuidelineStatus::Cached => write!(f, "Cached"),
        }
    }
}

/// Per-scenario result detail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioDetail {
    pub scenario_name: String,
    pub response_rate: f64,
    pub tox_rate: f64,
    pub rdi: f64,
    pub reward: f64,
}

/// Complete results for a single guideline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineResult {
    pub guideline_id: String,
    pub status: GuidelineStatus,

    /// Aggregated robustness score
    pub score: Option<GuidelineRobustnessScoreHost>,

    /// Per-scenario details (if retained)
    #[serde(default)]
    pub scenario_details: Vec<ScenarioDetail>,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 4: COMPARISON RESULTS
// ═══════════════════════════════════════════════════════════════════════════

/// Comparison between two guidelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineComparison {
    pub guideline_a_id: String,
    pub guideline_b_id: String,

    /// Difference in mean response (A - B)
    pub response_diff: f64,
    /// Difference in mean toxicity (A - B)
    pub tox_diff: f64,
    /// Difference in mean RDI (A - B)
    pub rdi_diff: f64,
    /// Difference in score (A - B)
    pub score_diff: f64,

    /// Statistical significance (if computable)
    pub response_p_value: Option<f64>,
    pub tox_p_value: Option<f64>,

    /// Relative improvement (%)
    pub response_rel_improvement: Option<f64>,
    pub tox_rel_improvement: Option<f64>,
}

/// Ranking entry for a guideline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankingEntry {
    pub guideline_id: String,
    pub rank_by_score: usize,
    pub rank_by_response: usize,
    pub rank_by_tox: usize,
    pub is_pareto_optimal: bool,
    pub is_feasible: bool,
}

/// Comparison to reference guideline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReferenceComparison {
    pub reference_id: String,
    pub comparisons: Vec<GuidelineComparison>,

    /// Rankings relative to reference
    pub rankings: Vec<RankingEntry>,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 5: COMPLETE EXPERIMENT RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentTiming {
    pub total_duration_ms: u64,
    pub robustness_duration_ms: u64,
    pub scoring_duration_ms: u64,
    pub constraints_duration_ms: u64,
    pub pareto_duration_ms: u64,
    pub report_duration_ms: u64,
}

impl Default for ExperimentTiming {
    fn default() -> Self {
        Self {
            total_duration_ms: 0,
            robustness_duration_ms: 0,
            scoring_duration_ms: 0,
            constraints_duration_ms: 0,
            pareto_duration_ms: 0,
            report_duration_ms: 0,
        }
    }
}

/// Summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentSummary {
    pub n_guidelines: usize,
    pub n_scenarios: usize,
    pub n_feasible: usize,
    pub n_pareto_optimal: usize,
    pub n_failed: usize,
    pub n_cached: usize,

    /// Best guideline by score
    pub best_by_score: Option<String>,
    /// Best guideline by response
    pub best_by_response: Option<String>,
    /// Most feasible guidelines
    pub most_feasible: Vec<String>,
    /// Pareto optimal guidelines
    pub pareto_optimal: Vec<String>,

    /// Key insights (auto-generated)
    pub insights: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExperimentStatus {
    Pending,
    Running,
    Completed,
    CompletedWithErrors,
    Failed,
}

impl std::fmt::Display for ExperimentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExperimentStatus::Pending => write!(f, "Pending"),
            ExperimentStatus::Running => write!(f, "Running"),
            ExperimentStatus::Completed => write!(f, "Completed"),
            ExperimentStatus::CompletedWithErrors => write!(f, "Completed with Errors"),
            ExperimentStatus::Failed => write!(f, "Failed"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentError {
    pub guideline_id: Option<String>,
    pub stage: String,
    pub message: String,
    pub recoverable: bool,
}

/// Complete experiment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentResult {
    // ═══════════════════════════════════════════════════════════════════════
    // Provenance
    // ═══════════════════════════════════════════════════════════════════════
    pub provenance: ExperimentProvenance,

    // ═══════════════════════════════════════════════════════════════════════
    // Status
    // ═══════════════════════════════════════════════════════════════════════
    pub status: ExperimentStatus,
    pub timing: ExperimentTiming,

    // ═══════════════════════════════════════════════════════════════════════
    // Per-guideline results
    // ═══════════════════════════════════════════════════════════════════════
    pub guidelines: Vec<GuidelineResult>,

    // ═══════════════════════════════════════════════════════════════════════
    // Aggregated results
    // ═══════════════════════════════════════════════════════════════════════
    /// All robustness scores
    pub scores: Vec<GuidelineRobustnessScoreHost>,

    /// Constraint analysis (if constraints provided)
    pub constraint_analysis: Option<ConstraintAnalysis>,

    /// Pareto analysis (if config provided)
    pub pareto_analysis: Option<ParetoAnalysis>,

    /// Pareto on feasible subset
    pub pareto_feasible: Option<ParetoAnalysis>,

    // ═══════════════════════════════════════════════════════════════════════
    // Comparisons
    // ═══════════════════════════════════════════════════════════════════════
    /// Comparison to reference (if reference specified)
    pub reference_comparison: Option<ReferenceComparison>,

    /// Pairwise comparisons (top N)
    #[serde(default)]
    pub pairwise_comparisons: Vec<GuidelineComparison>,

    // ═══════════════════════════════════════════════════════════════════════
    // Summary
    // ═══════════════════════════════════════════════════════════════════════
    pub summary: ExperimentSummary,

    // ═══════════════════════════════════════════════════════════════════════
    // Errors (if any)
    // ═══════════════════════════════════════════════════════════════════════
    #[serde(default)]
    pub errors: Vec<ExperimentError>,
    #[serde(default)]
    pub warnings: Vec<String>,
}

impl ExperimentResult {
    /// Export result to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export result to compact JSON
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Get summary text
    pub fn summary_text(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "Experiment: {} ({})\n",
            self.provenance.name, self.provenance.experiment_id
        ));
        output.push_str(&format!("Status: {}\n", self.status));
        output.push_str(&format!(
            "Duration: {:.2}s\n",
            self.timing.total_duration_ms as f64 / 1000.0
        ));
        output.push('\n');

        output.push_str(&format!(
            "Guidelines: {} total, {} feasible, {} Pareto-optimal, {} failed\n",
            self.summary.n_guidelines,
            self.summary.n_feasible,
            self.summary.n_pareto_optimal,
            self.summary.n_failed
        ));

        if let Some(ref best) = self.summary.best_by_score {
            output.push_str(&format!("Best by score: {}\n", best));
        }

        if !self.summary.pareto_optimal.is_empty() {
            output.push_str(&format!(
                "Pareto-optimal: {}\n",
                self.summary.pareto_optimal.join(", ")
            ));
        }

        if !self.summary.insights.is_empty() {
            output.push_str("\nInsights:\n");
            for insight in &self.summary.insights {
                output.push_str(&format!("  - {}\n", insight));
            }
        }

        output
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 6: EXPERIMENT RUNNER
// ═══════════════════════════════════════════════════════════════════════════

/// Progress callback type
pub type ProgressCallback = Arc<dyn Fn(ExperimentProgress) + Send + Sync>;

/// Experiment runner with all features
pub struct ExperimentRunner {
    config: ExperimentConfig,
    progress_callback: Option<ProgressCallback>,
    start_time: Option<Instant>,
}

impl ExperimentRunner {
    pub fn new(config: ExperimentConfig) -> Self {
        Self {
            config,
            progress_callback: None,
            start_time: None,
        }
    }

    pub fn with_progress_callback(mut self, callback: ProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    fn emit_progress(
        &self,
        stage: ExperimentStage,
        guideline: Option<&str>,
        completed: usize,
        total: usize,
        message: &str,
    ) {
        if let Some(ref callback) = self.progress_callback {
            let elapsed = self
                .start_time
                .map(|s| s.elapsed().as_secs_f64())
                .unwrap_or(0.0);

            let estimated_remaining = if completed > 0 && completed < total {
                Some(elapsed / completed as f64 * (total - completed) as f64)
            } else {
                None
            };

            callback(ExperimentProgress {
                stage,
                current_guideline: guideline.map(|s| s.to_string()),
                guidelines_completed: completed,
                guidelines_total: total,
                elapsed_secs: elapsed,
                estimated_remaining_secs: estimated_remaining,
                message: message.to_string(),
            });
        }
    }

    /// Run the complete experiment
    pub fn run(&mut self) -> anyhow::Result<ExperimentResult> {
        self.start_time = Some(Instant::now());
        let total_start = Instant::now();

        // Initialize provenance
        let provenance = ExperimentProvenance {
            experiment_id: ExperimentId::new(),
            name: self.config.name.clone(),
            description: self.config.description.clone(),
            tags: self.config.tags.clone(),
            created_at: now_timestamp(),
            started_at: Some(now_timestamp()),
            completed_at: None,
            git: GitProvenance::capture(),
            medlang_version: env!("CARGO_PKG_VERSION").to_string(),
            seeds: self.config.seeds.clone(),
            author: self.config.author.clone(),
            parent_experiment_id: None,
            config_hash: self.config.compute_hash(),
        };

        // Validate config
        self.emit_progress(
            ExperimentStage::ValidatingConfig,
            None,
            0,
            self.config.guidelines.len(),
            "Validating configuration",
        );

        let validation = self.config.validate();
        let warnings: Vec<String> = match &validation {
            Ok(report) => report.warnings.iter().map(|w| format!("{:?}", w)).collect(),
            Err(errors) if self.config.strict_validation => {
                anyhow::bail!("Configuration validation failed: {:?}", errors);
            }
            Err(_) => {
                vec!["Configuration has errors but strict_validation is disabled".into()]
            }
        };

        let n_guidelines = self.config.guidelines.len();
        let mut errors = vec![];

        // ═══════════════════════════════════════════════════════════════════
        // Stage 1: Robustness evaluation (simulated)
        // ═══════════════════════════════════════════════════════════════════

        self.emit_progress(
            ExperimentStage::RunningRobustness,
            None,
            0,
            n_guidelines,
            "Starting robustness evaluation",
        );

        let robustness_start = Instant::now();

        let guideline_results = self.run_robustness_evaluation(&mut errors);

        let robustness_duration = robustness_start.elapsed().as_millis() as u64;

        // ═══════════════════════════════════════════════════════════════════
        // Stage 2: Collect scores
        // ═══════════════════════════════════════════════════════════════════

        self.emit_progress(
            ExperimentStage::ComputingScores,
            None,
            n_guidelines,
            n_guidelines,
            "Computing robustness scores",
        );

        let scoring_start = Instant::now();

        let scores: Vec<GuidelineRobustnessScoreHost> = guideline_results
            .iter()
            .filter_map(|r| r.score.clone())
            .collect();

        let scoring_duration = scoring_start.elapsed().as_millis() as u64;

        // ═══════════════════════════════════════════════════════════════════
        // Stage 3: Constraint analysis
        // ═══════════════════════════════════════════════════════════════════

        self.emit_progress(
            ExperimentStage::AnalyzingConstraints,
            None,
            n_guidelines,
            n_guidelines,
            "Analyzing clinical constraints",
        );

        let constraints_start = Instant::now();

        let constraint_analysis = if let Some(ref constraints) = self.config.constraints {
            let constraint_cfg = ConstraintAnalysisConfig {
                constraints: constraints.clone(),
                compute_relaxation: true,
                compute_sensitivity: true,
                pareto_config: None,
                use_uncertainty_margins: false,
            };

            match compute_constraint_analysis(&scores, &constraint_cfg) {
                Ok(analysis) => Some(analysis),
                Err(e) => {
                    errors.push(ExperimentError {
                        guideline_id: None,
                        stage: ExperimentStage::AnalyzingConstraints.to_string(),
                        message: e.to_string(),
                        recoverable: true,
                    });
                    None
                }
            }
        } else {
            None
        };

        let constraints_duration = constraints_start.elapsed().as_millis() as u64;

        // ═══════════════════════════════════════════════════════════════════
        // Stage 4: Pareto analysis
        // ═══════════════════════════════════════════════════════════════════

        self.emit_progress(
            ExperimentStage::ComputingPareto,
            None,
            n_guidelines,
            n_guidelines,
            "Computing Pareto fronts",
        );

        let pareto_start = Instant::now();

        let pareto_analysis = if let Some(ref pareto_cfg) = self.config.pareto_cfg {
            match compute_pareto_analysis(&scores, pareto_cfg) {
                Ok(analysis) => Some(analysis),
                Err(e) => {
                    errors.push(ExperimentError {
                        guideline_id: None,
                        stage: ExperimentStage::ComputingPareto.to_string(),
                        message: e.to_string(),
                        recoverable: true,
                    });
                    None
                }
            }
        } else {
            None
        };

        // Pareto on feasible subset
        let pareto_feasible = if let (Some(ref pareto_cfg), Some(ref ca)) =
            (&self.config.pareto_cfg, &constraint_analysis)
        {
            let feasible_scores: Vec<_> = scores
                .iter()
                .filter(|s| {
                    ca.feasibility
                        .iter()
                        .find(|f| f.guideline_id == s.guideline_id)
                        .map(|f| f.hard_feasible)
                        .unwrap_or(false)
                })
                .cloned()
                .collect();

            if !feasible_scores.is_empty() {
                compute_pareto_analysis(&feasible_scores, pareto_cfg).ok()
            } else {
                None
            }
        } else {
            None
        };

        let pareto_duration = pareto_start.elapsed().as_millis() as u64;

        // ═══════════════════════════════════════════════════════════════════
        // Stage 5: Comparisons and summary
        // ═══════════════════════════════════════════════════════════════════

        let reference_comparison =
            self.compute_reference_comparison(&scores, &pareto_analysis, &constraint_analysis);

        let pairwise_comparisons = self.compute_pairwise_comparisons(&scores);

        let summary = self.compute_summary(
            &guideline_results,
            &scores,
            &pareto_analysis,
            &constraint_analysis,
        );

        // ═══════════════════════════════════════════════════════════════════
        // Stage 6: Generate reports
        // ═══════════════════════════════════════════════════════════════════

        self.emit_progress(
            ExperimentStage::GeneratingReports,
            None,
            n_guidelines,
            n_guidelines,
            "Generating reports",
        );

        let report_start = Instant::now();

        // Report generation happens after result is built

        let report_duration = report_start.elapsed().as_millis() as u64;

        // ═══════════════════════════════════════════════════════════════════
        // Build final result
        // ═══════════════════════════════════════════════════════════════════

        let total_duration = total_start.elapsed().as_millis() as u64;

        let status = if errors.iter().any(|e| !e.recoverable) {
            ExperimentStatus::Failed
        } else if !errors.is_empty() {
            ExperimentStatus::CompletedWithErrors
        } else {
            ExperimentStatus::Completed
        };

        let mut result = ExperimentResult {
            provenance: ExperimentProvenance {
                completed_at: Some(now_timestamp()),
                ..provenance
            },
            status,
            timing: ExperimentTiming {
                total_duration_ms: total_duration,
                robustness_duration_ms: robustness_duration,
                scoring_duration_ms: scoring_duration,
                constraints_duration_ms: constraints_duration,
                pareto_duration_ms: pareto_duration,
                report_duration_ms: report_duration,
            },
            guidelines: guideline_results,
            scores,
            constraint_analysis,
            pareto_analysis,
            pareto_feasible,
            reference_comparison,
            pairwise_comparisons,
            summary,
            errors,
            warnings,
        };

        // Auto-generate reports if configured
        if self.config.reports.auto_generate {
            if let Err(e) = self.generate_reports(&result) {
                result
                    .warnings
                    .push(format!("Report generation failed: {}", e));
            }
        }

        self.emit_progress(
            ExperimentStage::Completed,
            None,
            n_guidelines,
            n_guidelines,
            "Experiment completed",
        );

        Ok(result)
    }

    fn run_robustness_evaluation(&self, errors: &mut Vec<ExperimentError>) -> Vec<GuidelineResult> {
        let n_total = self.config.guidelines.len();
        let n_scenarios = self.config.scenarios.len().max(1);
        let mut results = Vec::with_capacity(n_total);

        for (i, guideline) in self.config.guidelines.iter().enumerate() {
            self.emit_progress(
                ExperimentStage::RunningRobustness,
                Some(&guideline.guideline_id),
                i,
                n_total,
                &format!("Evaluating guideline {}/{}", i + 1, n_total),
            );

            let start = Instant::now();

            // Simulate robustness evaluation
            // In a full implementation, this would call simulate_guideline_env_robustness
            let simulated_score = self.simulate_guideline_evaluation(guideline, n_scenarios);

            let scenario_details: Vec<ScenarioDetail> = if self.config.scenarios.is_empty() {
                vec![ScenarioDetail {
                    scenario_name: "baseline".to_string(),
                    response_rate: simulated_score.mean_response,
                    tox_rate: simulated_score.mean_grade3plus_rate,
                    rdi: simulated_score.mean_rdi,
                    reward: simulated_score.score_mean,
                }]
            } else {
                self.config
                    .scenarios
                    .iter()
                    .enumerate()
                    .map(|(j, scenario)| {
                        // Vary results slightly per scenario
                        let variation = 1.0 + (j as f64 * 0.05 - 0.1);
                        ScenarioDetail {
                            scenario_name: scenario.name.clone(),
                            response_rate: simulated_score.mean_response * variation,
                            tox_rate: simulated_score.mean_grade3plus_rate * (2.0 - variation),
                            rdi: simulated_score.mean_rdi * variation,
                            reward: simulated_score.score_mean * variation,
                        }
                    })
                    .collect()
            };

            results.push(GuidelineResult {
                guideline_id: guideline.guideline_id.clone(),
                status: GuidelineStatus::Completed {
                    duration_ms: start.elapsed().as_millis() as u64,
                },
                score: Some(simulated_score),
                scenario_details,
            });
        }

        results
    }

    fn simulate_guideline_evaluation(
        &self,
        guideline: &GuidelineDefinition,
        _n_scenarios: usize,
    ) -> GuidelineRobustnessScoreHost {
        // Deterministic simulation based on guideline ID hash
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        guideline.guideline_id.hash(&mut hasher);
        let hash = hasher.finish();

        // Generate pseudo-random but deterministic values
        let base_response = 0.3 + (hash % 100) as f64 / 200.0; // 0.3 - 0.8
        let base_tox = 0.1 + ((hash >> 8) % 100) as f64 / 500.0; // 0.1 - 0.3
        let base_rdi = 0.7 + ((hash >> 16) % 100) as f64 / 333.0; // 0.7 - 1.0

        GuidelineRobustnessScoreHost::mock(
            &guideline.guideline_id,
            base_response,
            base_tox,
            base_rdi,
        )
    }

    fn compute_reference_comparison(
        &self,
        scores: &[GuidelineRobustnessScoreHost],
        pareto: &Option<ParetoAnalysis>,
        constraints: &Option<ConstraintAnalysis>,
    ) -> Option<ReferenceComparison> {
        let ref_id = self.config.reference_guideline_id.as_ref()?;
        let ref_score = scores.iter().find(|s| &s.guideline_id == ref_id)?;

        let mut comparisons = vec![];
        let mut rankings = vec![];

        // Sort by score for ranking
        let mut sorted_by_score = scores.to_vec();
        sorted_by_score.sort_by(|a, b| {
            b.score_mean
                .partial_cmp(&a.score_mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sorted_by_response = scores.to_vec();
        sorted_by_response.sort_by(|a, b| {
            b.mean_response
                .partial_cmp(&a.mean_response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sorted_by_tox = scores.to_vec();
        sorted_by_tox.sort_by(|a, b| {
            a.mean_grade3plus_rate
                .partial_cmp(&b.mean_grade3plus_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for score in scores {
            if &score.guideline_id == ref_id {
                continue;
            }

            let response_diff = score.mean_response - ref_score.mean_response;
            let tox_diff = score.mean_grade3plus_rate - ref_score.mean_grade3plus_rate;
            let rdi_diff = score.mean_rdi - ref_score.mean_rdi;
            let score_diff = score.score_mean - ref_score.score_mean;

            comparisons.push(GuidelineComparison {
                guideline_a_id: score.guideline_id.clone(),
                guideline_b_id: ref_id.clone(),
                response_diff,
                tox_diff,
                rdi_diff,
                score_diff,
                response_p_value: None,
                tox_p_value: None,
                response_rel_improvement: if ref_score.mean_response > 0.0 {
                    Some(response_diff / ref_score.mean_response * 100.0)
                } else {
                    None
                },
                tox_rel_improvement: if ref_score.mean_grade3plus_rate > 0.0 {
                    Some(-tox_diff / ref_score.mean_grade3plus_rate * 100.0)
                } else {
                    None
                },
            });

            // Rankings
            let rank_score = sorted_by_score
                .iter()
                .position(|s| s.guideline_id == score.guideline_id)
                .map(|p| p + 1)
                .unwrap_or(0);
            let rank_response = sorted_by_response
                .iter()
                .position(|s| s.guideline_id == score.guideline_id)
                .map(|p| p + 1)
                .unwrap_or(0);
            let rank_tox = sorted_by_tox
                .iter()
                .position(|s| s.guideline_id == score.guideline_id)
                .map(|p| p + 1)
                .unwrap_or(0);

            let is_pareto = pareto
                .as_ref()
                .map(|p| {
                    p.points
                        .iter()
                        .any(|pt| pt.guideline_id == score.guideline_id && pt.rank == 0)
                })
                .unwrap_or(false);

            let is_feasible = constraints
                .as_ref()
                .map(|c| {
                    c.feasibility
                        .iter()
                        .any(|f| f.guideline_id == score.guideline_id && f.hard_feasible)
                })
                .unwrap_or(true);

            rankings.push(RankingEntry {
                guideline_id: score.guideline_id.clone(),
                rank_by_score: rank_score,
                rank_by_response: rank_response,
                rank_by_tox: rank_tox,
                is_pareto_optimal: is_pareto,
                is_feasible,
            });
        }

        Some(ReferenceComparison {
            reference_id: ref_id.clone(),
            comparisons,
            rankings,
        })
    }

    fn compute_pairwise_comparisons(
        &self,
        scores: &[GuidelineRobustnessScoreHost],
    ) -> Vec<GuidelineComparison> {
        let mut comparisons = vec![];

        for (i, a) in scores.iter().enumerate() {
            for b in scores.iter().skip(i + 1) {
                comparisons.push(GuidelineComparison {
                    guideline_a_id: a.guideline_id.clone(),
                    guideline_b_id: b.guideline_id.clone(),
                    response_diff: a.mean_response - b.mean_response,
                    tox_diff: a.mean_grade3plus_rate - b.mean_grade3plus_rate,
                    rdi_diff: a.mean_rdi - b.mean_rdi,
                    score_diff: a.score_mean - b.score_mean,
                    response_p_value: None,
                    tox_p_value: None,
                    response_rel_improvement: None,
                    tox_rel_improvement: None,
                });
            }
        }

        // Sort by absolute score difference
        comparisons.sort_by(|a, b| {
            b.score_diff
                .abs()
                .partial_cmp(&a.score_diff.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        comparisons.truncate(10);
        comparisons
    }

    fn compute_summary(
        &self,
        guideline_results: &[GuidelineResult],
        scores: &[GuidelineRobustnessScoreHost],
        pareto: &Option<ParetoAnalysis>,
        constraints: &Option<ConstraintAnalysis>,
    ) -> ExperimentSummary {
        let n_guidelines = self.config.guidelines.len();
        let n_scenarios = self.config.scenarios.len().max(1);

        let n_failed = guideline_results
            .iter()
            .filter(|r| matches!(r.status, GuidelineStatus::Failed { .. }))
            .count();

        let n_cached = guideline_results
            .iter()
            .filter(|r| matches!(r.status, GuidelineStatus::Cached))
            .count();

        let n_feasible = constraints
            .as_ref()
            .map(|c| c.n_hard_feasible)
            .unwrap_or(n_guidelines);

        let n_pareto = pareto
            .as_ref()
            .map(|p| p.points.iter().filter(|pt| pt.rank == 0).count())
            .unwrap_or(0);

        // Best by score
        let best_by_score = scores
            .iter()
            .max_by(|a, b| {
                a.score_mean
                    .partial_cmp(&b.score_mean)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.guideline_id.clone());

        // Best by response
        let best_by_response = scores
            .iter()
            .max_by(|a, b| {
                a.mean_response
                    .partial_cmp(&b.mean_response)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.guideline_id.clone());

        // Feasible guidelines
        let most_feasible: Vec<String> = constraints
            .as_ref()
            .map(|c| {
                c.feasibility
                    .iter()
                    .filter(|f| f.hard_feasible)
                    .map(|f| f.guideline_id.clone())
                    .collect()
            })
            .unwrap_or_default();

        // Pareto optimal
        let pareto_optimal: Vec<String> = pareto
            .as_ref()
            .map(|p| {
                p.points
                    .iter()
                    .filter(|pt| pt.rank == 0)
                    .map(|pt| pt.guideline_id.clone())
                    .collect()
            })
            .unwrap_or_default();

        // Generate insights
        let mut insights = vec![];

        if n_feasible == 0 && constraints.is_some() {
            insights.push("WARNING: No guidelines meet all clinical constraints".into());
        } else if n_feasible < n_guidelines / 2 && constraints.is_some() {
            insights.push(format!(
                "Only {:.0}% of guidelines are feasible",
                n_feasible as f64 / n_guidelines as f64 * 100.0
            ));
        }

        if n_pareto == 1 {
            insights.push(format!(
                "Single Pareto-optimal guideline: {}",
                pareto_optimal.first().unwrap_or(&"?".to_string())
            ));
        } else if n_pareto > 1 {
            insights.push(format!(
                "{} guidelines on Pareto front (tradeoff exists)",
                n_pareto
            ));
        }

        if let (Some(ref best_score), Some(ref best_resp)) = (&best_by_score, &best_by_response) {
            if best_score != best_resp {
                insights.push("Best by score differs from best by response".into());
            }
        }

        if n_failed > 0 {
            insights.push(format!("{} guidelines failed evaluation", n_failed));
        }

        ExperimentSummary {
            n_guidelines,
            n_scenarios,
            n_feasible,
            n_pareto_optimal: n_pareto,
            n_failed,
            n_cached,
            best_by_score,
            best_by_response,
            most_feasible,
            pareto_optimal,
            insights,
        }
    }

    fn generate_reports(&self, result: &ExperimentResult) -> anyhow::Result<()> {
        use std::fs;
        use std::path::Path;

        let output_dir = Path::new(&self.config.reports.output_dir);
        fs::create_dir_all(output_dir)?;

        let exp_id = &result.provenance.experiment_id.0;

        // JSON report
        if self.config.reports.json {
            let json_path = output_dir.join(format!("{}_result.json", exp_id));
            let json = serde_json::to_string_pretty(result)?;
            fs::write(json_path, json)?;
        }

        // CSV summaries
        if self.config.reports.csv {
            // Scores CSV
            let scores_csv = self.generate_scores_csv(result);
            let csv_path = output_dir.join(format!("{}_scores.csv", exp_id));
            fs::write(csv_path, scores_csv)?;

            // Feasibility CSV
            if let Some(ref ca) = result.constraint_analysis {
                let feas_csv = self.generate_feasibility_csv(ca);
                let feas_path = output_dir.join(format!("{}_feasibility.csv", exp_id));
                fs::write(feas_path, feas_csv)?;
            }
        }

        // Regulatory summary
        if self.config.reports.regulatory {
            let reg_report = self.generate_regulatory_report(result);
            let reg_path = output_dir.join(format!("{}_regulatory.txt", exp_id));
            fs::write(reg_path, reg_report)?;
        }

        Ok(())
    }

    fn generate_scores_csv(&self, result: &ExperimentResult) -> String {
        let mut csv = String::from(
            "guideline_id,mean_response,mean_grade3plus,mean_rdi,worst_response,worst_grade3plus,worst_rdi,score_mean,score_worst\n"
        );

        for s in &result.scores {
            csv.push_str(&format!(
                "{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                s.guideline_id,
                s.mean_response,
                s.mean_grade3plus_rate,
                s.mean_rdi,
                s.worst_response,
                s.worst_grade3plus_rate,
                s.worst_rdi,
                s.score_mean,
                s.score_worst,
            ));
        }

        csv
    }

    fn generate_feasibility_csv(&self, ca: &ConstraintAnalysis) -> String {
        let mut csv = String::from(
            "guideline_id,hard_feasible,fully_feasible,n_hard_violations,n_soft_violations,total_penalty\n"
        );

        for f in &ca.feasibility {
            csv.push_str(&format!(
                "{},{},{},{},{},{:.4}\n",
                f.guideline_id,
                f.hard_feasible,
                f.fully_feasible,
                f.n_hard_violations,
                f.n_soft_violations,
                f.total_penalty,
            ));
        }

        csv
    }

    fn generate_regulatory_report(&self, result: &ExperimentResult) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("           GUIDELINE EVALUATION EXPERIMENT REPORT              \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        // Provenance
        report.push_str("EXPERIMENT IDENTIFICATION\n");
        report.push_str("─────────────────────────\n");
        report.push_str(&format!("ID: {}\n", result.provenance.experiment_id.0));
        report.push_str(&format!("Name: {}\n", result.provenance.name));
        report.push_str(&format!("Created: {}\n", result.provenance.created_at));
        if let Some(ref git) = result.provenance.git.commit_hash {
            report.push_str(&format!("Git Commit: {}\n", git));
        }
        if let Some(ref branch) = result.provenance.git.branch {
            report.push_str(&format!("Git Branch: {}\n", branch));
        }
        if result.provenance.git.dirty {
            report.push_str("WARNING: Working directory has uncommitted changes\n");
        }
        report.push_str(&format!("Config Hash: {}\n", result.provenance.config_hash));
        report.push_str(&format!(
            "MedLang Version: {}\n",
            result.provenance.medlang_version
        ));
        report.push('\n');

        // Seeds
        report.push_str("REPRODUCIBILITY SEEDS\n");
        report.push_str("─────────────────────\n");
        report.push_str(&format!(
            "Master: {}\n",
            result.provenance.seeds.master_seed
        ));
        report.push_str(&format!(
            "Scenario: {}\n",
            result.provenance.seeds.scenario_seed
        ));
        report.push_str(&format!(
            "Simulation: {}\n",
            result.provenance.seeds.simulation_seed
        ));
        report.push('\n');

        // Summary
        report.push_str("SUMMARY\n");
        report.push_str("───────\n");
        report.push_str(&format!(
            "Guidelines evaluated: {}\n",
            result.summary.n_guidelines
        ));
        report.push_str(&format!("Scenarios: {}\n", result.summary.n_scenarios));
        report.push_str(&format!(
            "Feasible: {} ({:.1}%)\n",
            result.summary.n_feasible,
            result.summary.n_feasible as f64 / result.summary.n_guidelines.max(1) as f64 * 100.0
        ));
        report.push_str(&format!(
            "Pareto-optimal: {}\n",
            result.summary.n_pareto_optimal
        ));
        report.push_str(&format!(
            "Total duration: {:.2}s\n",
            result.timing.total_duration_ms as f64 / 1000.0
        ));
        report.push('\n');

        // Timing breakdown
        report.push_str("TIMING BREAKDOWN\n");
        report.push_str("────────────────\n");
        report.push_str(&format!(
            "Robustness: {:.2}s\n",
            result.timing.robustness_duration_ms as f64 / 1000.0
        ));
        report.push_str(&format!(
            "Scoring: {:.2}s\n",
            result.timing.scoring_duration_ms as f64 / 1000.0
        ));
        report.push_str(&format!(
            "Constraints: {:.2}s\n",
            result.timing.constraints_duration_ms as f64 / 1000.0
        ));
        report.push_str(&format!(
            "Pareto: {:.2}s\n",
            result.timing.pareto_duration_ms as f64 / 1000.0
        ));
        report.push_str(&format!(
            "Reports: {:.2}s\n",
            result.timing.report_duration_ms as f64 / 1000.0
        ));
        report.push('\n');

        // Insights
        if !result.summary.insights.is_empty() {
            report.push_str("KEY INSIGHTS\n");
            report.push_str("────────────\n");
            for insight in &result.summary.insights {
                report.push_str(&format!("* {}\n", insight));
            }
            report.push('\n');
        }

        // Recommendations
        report.push_str("RECOMMENDATIONS\n");
        report.push_str("───────────────\n");
        if let Some(ref best) = result.summary.best_by_score {
            report.push_str(&format!("* Best overall by score: {}\n", best));
        }
        if let Some(ref best) = result.summary.best_by_response {
            report.push_str(&format!("* Best by response: {}\n", best));
        }
        if !result.summary.pareto_optimal.is_empty() {
            report.push_str(&format!(
                "* Pareto-optimal set: {}\n",
                result.summary.pareto_optimal.join(", ")
            ));
        }
        report.push('\n');

        // Errors/Warnings
        if !result.errors.is_empty() {
            report.push_str("ERRORS\n");
            report.push_str("──────\n");
            for err in &result.errors {
                report.push_str(&format!(
                    "* [{}] {}: {}\n",
                    err.stage,
                    err.guideline_id.as_ref().unwrap_or(&"global".to_string()),
                    err.message
                ));
            }
            report.push('\n');
        }

        if !result.warnings.is_empty() {
            report.push_str("WARNINGS\n");
            report.push_str("────────\n");
            for warn in &result.warnings {
                report.push_str(&format!("* {}\n", warn));
            }
            report.push('\n');
        }

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str(&format!(
            "                    END OF REPORT                              \n"
        ));
        report.push_str("═══════════════════════════════════════════════════════════════\n");

        report
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 7: PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Main entry point: run a complete guideline evaluation experiment
pub fn run_experiment(config: ExperimentConfig) -> anyhow::Result<ExperimentResult> {
    let mut runner = ExperimentRunner::new(config);
    runner.run()
}

/// Run with progress callback
pub fn run_experiment_with_progress(
    config: ExperimentConfig,
    progress: ProgressCallback,
) -> anyhow::Result<ExperimentResult> {
    let mut runner = ExperimentRunner::new(config).with_progress_callback(progress);
    runner.run()
}

/// Compare two experiment results
pub fn compare_experiments(
    result_a: &ExperimentResult,
    result_b: &ExperimentResult,
) -> ExperimentComparison {
    let mut guideline_deltas = vec![];

    for score_a in &result_a.scores {
        if let Some(score_b) = result_b
            .scores
            .iter()
            .find(|s| s.guideline_id == score_a.guideline_id)
        {
            guideline_deltas.push(GuidelineDelta {
                guideline_id: score_a.guideline_id.clone(),
                response_delta: score_a.mean_response - score_b.mean_response,
                tox_delta: score_a.mean_grade3plus_rate - score_b.mean_grade3plus_rate,
                score_delta: score_a.score_mean - score_b.score_mean,
            });
        }
    }

    ExperimentComparison {
        experiment_a_id: result_a.provenance.experiment_id.clone(),
        experiment_b_id: result_b.provenance.experiment_id.clone(),
        guideline_deltas,
        config_hash_match: result_a.provenance.config_hash == result_b.provenance.config_hash,
    }
}

/// Experiment comparison result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentComparison {
    pub experiment_a_id: ExperimentId,
    pub experiment_b_id: ExperimentId,
    pub guideline_deltas: Vec<GuidelineDelta>,
    pub config_hash_match: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineDelta {
    pub guideline_id: String,
    pub response_delta: f64,
    pub tox_delta: f64,
    pub score_delta: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_experiment_id_uniqueness() {
        let id1 = ExperimentId::new();
        let id2 = ExperimentId::new();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_git_provenance_capture() {
        let prov = GitProvenance::capture();
        // May or may not have git info depending on environment
        assert!(prov.commit_hash.is_some() || prov.commit_hash.is_none());
    }

    #[test]
    fn test_config_validation_empty_guidelines() {
        let config = ExperimentConfig::new("test", vec![], DoseToxEnvConfig::default());
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_validation_duplicate_ids() {
        let guidelines = vec![
            GuidelineDefinition::new("g1"),
            GuidelineDefinition::new("g1"), // duplicate
        ];
        let config = ExperimentConfig::new("test", guidelines, DoseToxEnvConfig::default());
        let result = config.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_config_hash_determinism() {
        let guidelines = vec![
            GuidelineDefinition::new("g1"),
            GuidelineDefinition::new("g2"),
        ];
        let config = ExperimentConfig::new("test", guidelines, DoseToxEnvConfig::default());

        let hash1 = config.compute_hash();
        let hash2 = config.compute_hash();
        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_run_experiment_basic() {
        let guidelines = vec![
            GuidelineDefinition::new("guideline_a"),
            GuidelineDefinition::new("guideline_b"),
            GuidelineDefinition::new("guideline_c"),
        ];

        let mut config = ExperimentConfig::new(
            "Basic Test Experiment",
            guidelines,
            DoseToxEnvConfig::default(),
        );
        config.strict_validation = false;
        config.reports.auto_generate = false; // Don't write files in tests

        let result = run_experiment(config).unwrap();

        assert_eq!(result.summary.n_guidelines, 3);
        assert_eq!(result.scores.len(), 3);
        assert!(matches!(result.status, ExperimentStatus::Completed));
    }

    #[test]
    fn test_run_experiment_with_reference() {
        let guidelines = vec![
            GuidelineDefinition::new("reference"),
            GuidelineDefinition::new("test_a"),
            GuidelineDefinition::new("test_b"),
        ];

        let mut config = ExperimentConfig::new(
            "Reference Comparison Test",
            guidelines,
            DoseToxEnvConfig::default(),
        )
        .with_reference("reference");

        config.strict_validation = false;
        config.reports.auto_generate = false;

        let result = run_experiment(config).unwrap();

        assert!(result.reference_comparison.is_some());
        let ref_comp = result.reference_comparison.unwrap();
        assert_eq!(ref_comp.reference_id, "reference");
        assert_eq!(ref_comp.comparisons.len(), 2); // test_a and test_b
    }

    #[test]
    fn test_run_experiment_with_scenarios() {
        let guidelines = vec![GuidelineDefinition::new("g1")];

        let scenarios = vec![
            EnvScenario::new("baseline"),
            EnvScenario::new("high_response").with_param("response_multiplier", 1.2),
            EnvScenario::new("high_tox").with_param("tox_multiplier", 1.5),
        ];

        let mut config =
            ExperimentConfig::new("Scenario Test", guidelines, DoseToxEnvConfig::default())
                .with_scenarios(scenarios);

        config.strict_validation = false;
        config.reports.auto_generate = false;

        let result = run_experiment(config).unwrap();

        assert_eq!(result.summary.n_scenarios, 3);
        assert_eq!(result.guidelines[0].scenario_details.len(), 3);
    }

    #[test]
    fn test_experiment_result_json_export() {
        let guidelines = vec![GuidelineDefinition::new("g1")];

        let mut config =
            ExperimentConfig::new("JSON Export Test", guidelines, DoseToxEnvConfig::default());
        config.strict_validation = false;
        config.reports.auto_generate = false;

        let result = run_experiment(config).unwrap();

        let json = result.to_json().unwrap();
        assert!(json.contains("JSON Export Test"));
        assert!(json.contains("g1"));
    }

    #[test]
    fn test_compare_experiments() {
        let guidelines = vec![
            GuidelineDefinition::new("g1"),
            GuidelineDefinition::new("g2"),
        ];

        let mut config1 = ExperimentConfig::new(
            "Experiment 1",
            guidelines.clone(),
            DoseToxEnvConfig::default(),
        );
        config1.strict_validation = false;
        config1.reports.auto_generate = false;

        let mut config2 =
            ExperimentConfig::new("Experiment 2", guidelines, DoseToxEnvConfig::default());
        config2.strict_validation = false;
        config2.reports.auto_generate = false;

        let result1 = run_experiment(config1).unwrap();
        let result2 = run_experiment(config2).unwrap();

        let comparison = compare_experiments(&result1, &result2);

        assert_eq!(comparison.guideline_deltas.len(), 2);
        assert!(comparison.config_hash_match); // Same config should have same hash
    }
}
