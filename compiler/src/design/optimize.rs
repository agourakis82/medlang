//! Utility-Based Optimal Trial Design Engine
//!
//! Systematically searches over design space (N, margins, toxicity thresholds)
//! to find designs maximizing a utility function that balances:
//! - Probability of success (PoS)
//! - Efficacy benefit (e.g., ORR improvement)
//! - Safety risk (e.g., DLT rate)
//! - Sample size / cost

use crate::design::{DecisionEvalResult, DesignSummary, PosteriorDraw};
use serde::{Deserialize, Serialize};

// =============================================================================
// Design Candidate & Metrics
// =============================================================================

/// A single design point in the optimization search space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignCandidate {
    /// Sample size per arm
    pub n_per_arm: usize,

    /// Minimal efficacy difference vs control (e.g., ORR margin)
    pub orr_margin: f64,

    /// Maximum acceptable toxicity rate (e.g., DLT threshold)
    pub dlt_threshold: Option<f64>,
}

impl DesignCandidate {
    pub fn new(n_per_arm: usize, orr_margin: f64, dlt_threshold: Option<f64>) -> Self {
        Self {
            n_per_arm,
            orr_margin,
            dlt_threshold,
        }
    }

    /// Create grid of candidates from parameter ranges
    pub fn grid(
        n_per_arm_values: &[usize],
        orr_margin_values: &[f64],
        dlt_threshold_values: &[f64],
    ) -> Vec<Self> {
        let mut candidates = Vec::new();
        for &n in n_per_arm_values {
            for &margin in orr_margin_values {
                for &dlt in dlt_threshold_values {
                    candidates.push(Self::new(n, margin, Some(dlt)));
                }
            }
        }
        candidates
    }
}

/// Summary metrics for evaluating a design candidate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignEvalMetrics {
    /// Probability of success for primary decision
    pub pos: f64,

    /// Efficacy benefit: mean(Test) - mean(Control)
    pub eff_benefit: f64,

    /// Toxicity risk: mean DLT rate or equivalent
    pub tox_risk: Option<f64>,

    /// Total sample size across all arms
    pub sample_size_total: usize,
}

// =============================================================================
// Objective / Utility Configuration
// =============================================================================

/// Objective function configuration for design optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveConfig {
    /// Weight for probability of success (PoS)
    pub w_pos: f64,

    /// Weight for efficacy benefit (higher is better)
    pub w_eff: f64,

    /// Penalty weight for toxicity (higher tox = lower utility)
    pub w_tox: f64,

    /// Penalty weight for sample size / cost
    pub w_size: f64,

    /// Reference sample size for scaling (normalization)
    pub n_ref: usize,
}

impl Default for ObjectiveConfig {
    fn default() -> Self {
        Self {
            w_pos: 1.0,
            w_eff: 0.5,
            w_tox: 1.0,
            w_size: 0.1,
            n_ref: 100,
        }
    }
}

impl ObjectiveConfig {
    /// Create balanced objective (equal weights)
    pub fn balanced() -> Self {
        Self {
            w_pos: 1.0,
            w_eff: 1.0,
            w_tox: 1.0,
            w_size: 0.2,
            n_ref: 100,
        }
    }

    /// Create efficacy-focused objective
    pub fn efficacy_focused() -> Self {
        Self {
            w_pos: 1.0,
            w_eff: 2.0,
            w_tox: 0.5,
            w_size: 0.1,
            n_ref: 100,
        }
    }

    /// Create safety-focused objective
    pub fn safety_focused() -> Self {
        Self {
            w_pos: 1.0,
            w_eff: 0.5,
            w_tox: 2.0,
            w_size: 0.1,
            n_ref: 100,
        }
    }

    /// Create cost-conscious objective
    pub fn cost_conscious() -> Self {
        Self {
            w_pos: 1.0,
            w_eff: 0.5,
            w_tox: 1.0,
            w_size: 0.5,
            n_ref: 100,
        }
    }
}

/// Compute scalar utility from metrics and objective
pub fn compute_utility(metrics: &DesignEvalMetrics, objective: &ObjectiveConfig) -> f64 {
    let pos_term = objective.w_pos * metrics.pos;
    let eff_term = objective.w_eff * metrics.eff_benefit;
    let tox_term = objective.w_tox * metrics.tox_risk.unwrap_or(0.0);
    let size_penalty =
        objective.w_size * (metrics.sample_size_total as f64 / objective.n_ref as f64);

    pos_term + eff_term - tox_term - size_penalty
}

// =============================================================================
// Design Utility Result
// =============================================================================

/// Complete result for a single design candidate evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignUtilityResult {
    pub candidate: DesignCandidate,
    pub metrics: DesignEvalMetrics,
    pub utility: f64,
    pub design_summary: DesignSummary,
}

impl DesignUtilityResult {
    /// Check if this design is dominated by another (lower utility)
    pub fn is_dominated_by(&self, other: &Self) -> bool {
        // Dominated if other has:
        // - Higher PoS
        // - Higher efficacy
        // - Lower toxicity
        // - Smaller size
        // And at least one is strictly better
        let pos_better = other.metrics.pos >= self.metrics.pos;
        let eff_better = other.metrics.eff_benefit >= self.metrics.eff_benefit;
        let tox_better =
            other.metrics.tox_risk.unwrap_or(0.0) <= self.metrics.tox_risk.unwrap_or(0.0);
        let size_better = other.metrics.sample_size_total <= self.metrics.sample_size_total;

        let strictly_better = other.metrics.pos > self.metrics.pos
            || other.metrics.eff_benefit > self.metrics.eff_benefit
            || other.metrics.tox_risk.unwrap_or(0.0) < self.metrics.tox_risk.unwrap_or(0.0)
            || other.metrics.sample_size_total < self.metrics.sample_size_total;

        pos_better && eff_better && tox_better && size_better && strictly_better
    }
}

// =============================================================================
// Optimization Results
// =============================================================================

/// Complete optimization report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationReport {
    pub protocol_name: String,
    pub population_model_name: String,
    pub objective: ObjectiveConfig,
    pub results: Vec<DesignUtilityResult>,
    pub optimal_design: DesignUtilityResult,
    pub pareto_frontier: Vec<DesignUtilityResult>,
}

impl OptimizationReport {
    /// Find top N designs by utility
    pub fn top_n(&self, n: usize) -> &[DesignUtilityResult] {
        &self.results[..n.min(self.results.len())]
    }

    /// Get utility range across all designs
    pub fn utility_range(&self) -> (f64, f64) {
        let utilities: Vec<f64> = self.results.iter().map(|r| r.utility).collect();
        let min = utilities
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        let max = utilities
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or(0.0);
        (min, max)
    }
}

// =============================================================================
// Metric Extraction
// =============================================================================

/// Extract efficacy and toxicity metrics from design summary (Week 21 scaffold)
///
/// In a full implementation, this would parse actual endpoint results from
/// DesignSummary. For Week 21, we use simplified placeholder logic.
pub fn extract_eff_tox_metrics(summary: &DesignSummary) -> (f64, Option<f64>) {
    // Scaffold: Synthetic metrics based on N per arm
    // In real implementation, would extract from summary.decision_results and endpoint data

    // Simulate efficacy benefit increasing with sample size (more precision)
    let eff_benefit = 0.10 + 0.10 * (summary.n_per_arm as f64 / 150.0).min(1.0);

    // Simulate toxicity risk (constant for scaffold)
    let tox_risk = Some(0.15);

    (eff_benefit, tox_risk)
}

// =============================================================================
// Candidate Evaluation
// =============================================================================

/// Evaluate a single design candidate (Week 21 scaffold)
///
/// In a full implementation, this would:
/// 1. Configure IRProtocol with candidate's margins/thresholds
/// 2. Run design evaluation via evaluate_design_pos
/// 3. Extract metrics from results
///
/// For Week 21, we use simplified synthetic evaluation.
pub fn evaluate_candidate(
    protocol_name: &str,
    _population_model_name: &str,
    candidate: &DesignCandidate,
    objective: &ObjectiveConfig,
    _posterior_draws: Option<&[PosteriorDraw]>,
) -> DesignUtilityResult {
    // Scaffold: Synthetic PoS model
    // Better (larger) N → higher PoS
    // Tighter margin → lower PoS
    let n_factor = (candidate.n_per_arm as f64 / 150.0).min(1.0);
    let margin_factor = (1.0 - candidate.orr_margin / 0.20).max(0.3);
    let synthetic_pos = 0.5 + 0.4 * n_factor * margin_factor;

    // Scaffold: Create synthetic design summary
    let design_summary = DesignSummary {
        design_label: format!(
            "N={},M={:.2},DLT={:.2}",
            candidate.n_per_arm,
            candidate.orr_margin,
            candidate.dlt_threshold.unwrap_or(0.0)
        ),
        n_per_arm: candidate.n_per_arm,
        decision_results: vec![DecisionEvalResult {
            decision_name: "Primary_GoNoGo".to_string(),
            endpoint_name: "ORR".to_string(),
            arm_left: "Control".to_string(),
            arm_right: "Treatment".to_string(),
            margin: candidate.orr_margin,
            prob_threshold: 0.80,
            pos: synthetic_pos,
        }],
    };

    // Extract metrics
    let (eff_benefit, tox_risk) = extract_eff_tox_metrics(&design_summary);

    // Assume 2 arms for now
    let sample_size_total = candidate.n_per_arm * 2;

    let metrics = DesignEvalMetrics {
        pos: synthetic_pos,
        eff_benefit,
        tox_risk,
        sample_size_total,
    };

    // Compute utility
    let utility = compute_utility(&metrics, objective);

    DesignUtilityResult {
        candidate: candidate.clone(),
        metrics,
        utility,
        design_summary,
    }
}

// =============================================================================
// Grid Search Optimizer
// =============================================================================

/// Optimize design over a grid of candidates
pub fn optimize_design_over_grid(
    protocol_name: &str,
    population_model_name: &str,
    candidates: &[DesignCandidate],
    objective: &ObjectiveConfig,
    posterior_draws: Option<&[PosteriorDraw]>,
) -> OptimizationReport {
    let mut results = Vec::with_capacity(candidates.len());

    for candidate in candidates {
        let result = evaluate_candidate(
            protocol_name,
            population_model_name,
            candidate,
            objective,
            posterior_draws,
        );
        results.push(result);
    }

    // Sort by utility (descending)
    results.sort_by(|a, b| {
        b.utility
            .partial_cmp(&a.utility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Extract optimal design (highest utility)
    let optimal_design = results[0].clone();

    // Compute Pareto frontier (non-dominated designs)
    let pareto_frontier = compute_pareto_frontier(&results);

    OptimizationReport {
        protocol_name: protocol_name.to_string(),
        population_model_name: population_model_name.to_string(),
        objective: objective.clone(),
        results,
        optimal_design,
        pareto_frontier,
    }
}

/// Compute Pareto frontier (non-dominated designs)
fn compute_pareto_frontier(results: &[DesignUtilityResult]) -> Vec<DesignUtilityResult> {
    let mut frontier = Vec::new();

    for result in results {
        let is_dominated = results.iter().any(|other| {
            if std::ptr::eq(result, other) {
                return false;
            }
            result.is_dominated_by(other)
        });

        if !is_dominated {
            frontier.push(result.clone());
        }
    }

    frontier
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_design_candidate_creation() {
        let candidate = DesignCandidate::new(120, 0.15, Some(0.25));
        assert_eq!(candidate.n_per_arm, 120);
        assert_eq!(candidate.orr_margin, 0.15);
        assert_eq!(candidate.dlt_threshold, Some(0.25));
    }

    #[test]
    fn test_design_candidate_grid() {
        let candidates = DesignCandidate::grid(&[80, 120], &[0.10, 0.15], &[0.20, 0.25]);

        assert_eq!(candidates.len(), 8); // 2 * 2 * 2
        assert_eq!(candidates[0].n_per_arm, 80);
        assert_eq!(candidates[0].orr_margin, 0.10);
    }

    #[test]
    fn test_utility_computation() {
        let metrics = DesignEvalMetrics {
            pos: 0.80,
            eff_benefit: 0.20,
            tox_risk: Some(0.15),
            sample_size_total: 240,
        };

        let objective = ObjectiveConfig::default();
        let utility = compute_utility(&metrics, &objective);

        // Expected: 1.0*0.8 + 0.5*0.2 - 1.0*0.15 - 0.1*(240/100)
        // = 0.8 + 0.1 - 0.15 - 0.24 = 0.51
        assert!((utility - 0.51).abs() < 0.01);
    }

    #[test]
    fn test_objective_presets() {
        let balanced = ObjectiveConfig::balanced();
        assert_eq!(balanced.w_pos, 1.0);
        assert_eq!(balanced.w_eff, 1.0);

        let eff_focused = ObjectiveConfig::efficacy_focused();
        assert_eq!(eff_focused.w_eff, 2.0);

        let safety_focused = ObjectiveConfig::safety_focused();
        assert_eq!(safety_focused.w_tox, 2.0);

        let cost_conscious = ObjectiveConfig::cost_conscious();
        assert_eq!(cost_conscious.w_size, 0.5);
    }

    #[test]
    fn test_optimization_ranking() {
        let candidates = vec![
            DesignCandidate::new(80, 0.15, Some(0.25)),
            DesignCandidate::new(120, 0.15, Some(0.25)),
            DesignCandidate::new(150, 0.15, Some(0.25)),
        ];

        let objective = ObjectiveConfig::default();

        let report =
            optimize_design_over_grid("TestProtocol", "TestModel", &candidates, &objective, None);

        // Results should be sorted by utility
        assert!(report.results.len() == 3);
        assert!(report.results[0].utility >= report.results[1].utility);
        assert!(report.results[1].utility >= report.results[2].utility);

        // Optimal design should match first result
        assert_eq!(
            report.optimal_design.candidate.n_per_arm,
            report.results[0].candidate.n_per_arm
        );
    }

    #[test]
    fn test_pareto_frontier() {
        let candidates = vec![
            DesignCandidate::new(80, 0.15, Some(0.25)),
            DesignCandidate::new(120, 0.15, Some(0.25)),
            DesignCandidate::new(150, 0.10, Some(0.20)),
        ];

        let objective = ObjectiveConfig::default();

        let report =
            optimize_design_over_grid("TestProtocol", "TestModel", &candidates, &objective, None);

        // Should have at least one non-dominated design
        assert!(!report.pareto_frontier.is_empty());
    }

    #[test]
    fn test_utility_range() {
        let candidates = vec![
            DesignCandidate::new(80, 0.15, Some(0.25)),
            DesignCandidate::new(150, 0.15, Some(0.25)),
        ];

        let objective = ObjectiveConfig::default();

        let report =
            optimize_design_over_grid("TestProtocol", "TestModel", &candidates, &objective, None);

        let (min, max) = report.utility_range();
        assert!(min <= max);
        assert!(min > 0.0); // Sanity check
    }

    #[test]
    fn test_top_n_selection() {
        let candidates = vec![
            DesignCandidate::new(80, 0.15, Some(0.25)),
            DesignCandidate::new(100, 0.15, Some(0.25)),
            DesignCandidate::new(120, 0.15, Some(0.25)),
            DesignCandidate::new(150, 0.15, Some(0.25)),
        ];

        let objective = ObjectiveConfig::default();

        let report =
            optimize_design_over_grid("TestProtocol", "TestModel", &candidates, &objective, None);

        let top_2 = report.top_n(2);
        assert_eq!(top_2.len(), 2);
        assert!(top_2[0].utility >= top_2[1].utility);
    }
}
