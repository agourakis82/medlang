//! Bayesian decision layer for trial design evaluation
//!
//! This module provides tools for:
//! - Evaluating decision rules on protocol endpoints
//! - Computing predictive probability of success (PoS) for future trials
//! - Comparing alternative trial designs
//! - Quantum trust-driven sensitivity analysis
//! - Utility-based optimal design selection

pub mod optimize;
pub mod quantum_sensitivity;

use serde::{Deserialize, Serialize};

pub use optimize::{
    compute_utility, evaluate_candidate, optimize_design_over_grid, DesignCandidate,
    DesignEvalMetrics, DesignUtilityResult, ObjectiveConfig, OptimizationReport,
};
pub use quantum_sensitivity::{
    default_scenarios_from_scores, DesignConfigInfo, QuantumDesignScenario,
    QuantumDesignSensitivityReport, QuantumDesignSensitivityResult,
};

/// Configuration for design evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignConfig {
    /// Future trial sample size per arm
    pub n_per_arm: usize,
    /// Number of posterior draws to use for prediction
    pub n_draws: usize,
}

impl Default for DesignConfig {
    fn default() -> Self {
        DesignConfig {
            n_per_arm: 100,
            n_draws: 500,
        }
    }
}

/// Result of evaluating a single decision rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEvalResult {
    pub decision_name: String,
    pub endpoint_name: String,
    pub arm_left: String,
    pub arm_right: String,
    pub margin: f64,
    pub prob_threshold: f64,
    /// Predictive probability of success
    pub pos: f64,
}

/// Summary of design evaluation across all decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignSummary {
    pub design_label: String,
    pub n_per_arm: usize,
    pub decision_results: Vec<DecisionEvalResult>,
}

/// Simple stub for posterior draws (will be replaced with actual implementation)
#[derive(Debug, Clone)]
pub struct PosteriorDraw {
    pub params: std::collections::HashMap<String, f64>,
}

/// Evaluate a single design configuration
///
/// This is a scaffold implementation that demonstrates the structure.
/// Full implementation requires:
/// 1. Posterior draw sampling infrastructure
/// 2. Virtual trial simulation engine per draw
/// 3. Endpoint computation on simulated data
pub fn evaluate_design_pos(
    protocol_name: &str,
    design_cfg: &DesignConfig,
    _posterior_draws: &[PosteriorDraw],
) -> DesignSummary {
    // Scaffold implementation - returns placeholder results
    DesignSummary {
        design_label: format!("N_per_arm={}", design_cfg.n_per_arm),
        n_per_arm: design_cfg.n_per_arm,
        decision_results: vec![DecisionEvalResult {
            decision_name: "Placeholder_Decision".to_string(),
            endpoint_name: "ORR".to_string(),
            arm_left: "ArmA".to_string(),
            arm_right: "ArmB".to_string(),
            margin: 0.10,
            prob_threshold: 0.80,
            pos: 0.0, // Placeholder - would be computed from simulations
        }],
    }
}

/// Evaluate multiple design configurations (grid search)
pub fn evaluate_design_grid(
    protocol_name: &str,
    n_per_arm_values: &[usize],
    n_draws: usize,
    _posterior_draws: &[PosteriorDraw],
) -> Vec<DesignSummary> {
    n_per_arm_values
        .iter()
        .map(|&n_per_arm| {
            let cfg = DesignConfig { n_per_arm, n_draws };
            evaluate_design_pos(protocol_name, &cfg, _posterior_draws)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_design_config_default() {
        let cfg = DesignConfig::default();
        assert_eq!(cfg.n_per_arm, 100);
        assert_eq!(cfg.n_draws, 500);
    }

    #[test]
    fn test_evaluate_design_pos_scaffold() {
        let cfg = DesignConfig {
            n_per_arm: 150,
            n_draws: 100,
        };
        let draws = vec![];

        let summary = evaluate_design_pos("TestProtocol", &cfg, &draws);

        assert_eq!(summary.n_per_arm, 150);
        assert_eq!(summary.design_label, "N_per_arm=150");
        assert!(!summary.decision_results.is_empty());
    }

    #[test]
    fn test_evaluate_design_grid() {
        let n_values = vec![50, 100, 150];
        let draws = vec![];

        let summaries = evaluate_design_grid("TestProtocol", &n_values, 100, &draws);

        assert_eq!(summaries.len(), 3);
        assert_eq!(summaries[0].n_per_arm, 50);
        assert_eq!(summaries[1].n_per_arm, 100);
        assert_eq!(summaries[2].n_per_arm, 150);
    }
}
