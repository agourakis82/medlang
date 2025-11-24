//! Quantum Trust-Driven Design Sensitivity Analysis
//!
//! Evaluates trial design PoS under different quantum prior trust scenarios,
//! enabling robustness assessment when QM priors may be unreliable.

use crate::design::DesignSummary;
use crate::diagnostics::quantum_trust::{QuantumTrustLevel, QuantumTrustScore};
use serde::{Deserialize, Serialize};

// =============================================================================
// Design Sensitivity Scenarios
// =============================================================================

/// A design scenario parameterized by quantum trust assumptions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDesignScenario {
    /// Scenario label (e.g., "QM_high_trust", "QM_mixed_trust")
    pub label: String,

    /// Human-readable description
    pub description: String,

    /// QM parameters considered trusted: (system, param)
    pub included_params: Vec<(String, String)>,

    /// QM parameters considered broken: (system, param)
    pub excluded_params: Vec<(String, String)>,
}

impl QuantumDesignScenario {
    /// Create a scenario that trusts only high-trust parameters
    pub fn high_trust_only(scores: &[QuantumTrustScore]) -> Self {
        let (included, excluded) =
            partition_by_trust(scores, |t| matches!(t, QuantumTrustLevel::High));

        Self {
            label: "QM_high_trust_only".to_string(),
            description: "Trust only HIGH-rated QM priors; treat others as uncertain".to_string(),
            included_params: included,
            excluded_params: excluded,
        }
    }

    /// Create a scenario that trusts high and moderate parameters
    pub fn high_moderate_trust(scores: &[QuantumTrustScore]) -> Self {
        let (included, excluded) = partition_by_trust(scores, |t| {
            matches!(t, QuantumTrustLevel::High | QuantumTrustLevel::Moderate)
        });

        Self {
            label: "QM_high_moderate_trust".to_string(),
            description: "Trust HIGH and MODERATE QM priors; exclude LOW and BROKEN".to_string(),
            included_params: included,
            excluded_params: excluded,
        }
    }

    /// Create a scenario that excludes only broken parameters
    pub fn exclude_broken_only(scores: &[QuantumTrustScore]) -> Self {
        let (included, excluded) =
            partition_by_trust(scores, |t| !matches!(t, QuantumTrustLevel::Broken));

        Self {
            label: "QM_exclude_broken".to_string(),
            description: "Exclude only BROKEN QM priors; trust all others".to_string(),
            included_params: included,
            excluded_params: excluded,
        }
    }

    /// Create a pessimistic scenario (only high trust)
    pub fn pessimistic(scores: &[QuantumTrustScore]) -> Self {
        Self::high_trust_only(scores)
    }

    /// Create an optimistic scenario (exclude only broken)
    pub fn optimistic(scores: &[QuantumTrustScore]) -> Self {
        Self::exclude_broken_only(scores)
    }

    /// Create a baseline scenario (use all QM priors as-is)
    pub fn baseline(scores: &[QuantumTrustScore]) -> Self {
        let all_params: Vec<(String, String)> = scores
            .iter()
            .map(|s| (s.system_name.clone(), s.param_name.clone()))
            .collect();

        Self {
            label: "QM_baseline".to_string(),
            description: "Use all QM priors as originally specified (no trust adjustment)"
                .to_string(),
            included_params: all_params,
            excluded_params: Vec::new(),
        }
    }
}

/// Helper: partition scores by trust criterion
fn partition_by_trust<F>(
    scores: &[QuantumTrustScore],
    should_include: F,
) -> (Vec<(String, String)>, Vec<(String, String)>)
where
    F: Fn(&QuantumTrustLevel) -> bool,
{
    let mut included = Vec::new();
    let mut excluded = Vec::new();

    for s in scores {
        let param_id = (s.system_name.clone(), s.param_name.clone());
        if should_include(&s.trust) {
            included.push(param_id);
        } else {
            excluded.push(param_id);
        }
    }

    (included, excluded)
}

/// Generate default sensitivity scenarios from trust scores
pub fn default_scenarios_from_scores(scores: &[QuantumTrustScore]) -> Vec<QuantumDesignScenario> {
    vec![
        QuantumDesignScenario::baseline(scores),
        QuantumDesignScenario::high_moderate_trust(scores),
        QuantumDesignScenario::high_trust_only(scores),
        QuantumDesignScenario::exclude_broken_only(scores),
    ]
}

// =============================================================================
// Sensitivity Results
// =============================================================================

/// Result of evaluating design PoS under a quantum trust scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDesignSensitivityResult {
    pub scenario: QuantumDesignScenario,
    pub design_summary: DesignSummary,
}

impl QuantumDesignSensitivityResult {
    pub fn new(scenario: QuantumDesignScenario, design_summary: DesignSummary) -> Self {
        Self {
            scenario,
            design_summary,
        }
    }
}

/// Complete sensitivity analysis report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumDesignSensitivityReport {
    pub protocol_name: String,
    pub population_model: String,
    pub fit_dir: String,
    pub design_config: DesignConfigInfo,
    pub results: Vec<QuantumDesignSensitivityResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignConfigInfo {
    pub n_per_arm: usize,
    pub n_draws: usize,
}

impl QuantumDesignSensitivityReport {
    /// Find result by scenario label
    pub fn find_scenario(&self, label: &str) -> Option<&QuantumDesignSensitivityResult> {
        self.results.iter().find(|r| r.scenario.label == label)
    }

    /// Get PoS range across all scenarios for a given decision
    pub fn pos_range(&self, decision_name: &str) -> Option<(f64, f64)> {
        let mut min_pos = f64::MAX;
        let mut max_pos = f64::MIN;
        let mut found = false;

        for result in &self.results {
            for decision_result in &result.design_summary.decision_results {
                if decision_result.decision_name == decision_name {
                    min_pos = min_pos.min(decision_result.pos);
                    max_pos = max_pos.max(decision_result.pos);
                    found = true;
                }
            }
        }

        if found {
            Some((min_pos, max_pos))
        } else {
            None
        }
    }

    /// Calculate PoS sensitivity (max - min) for a decision
    pub fn pos_sensitivity(&self, decision_name: &str) -> Option<f64> {
        self.pos_range(decision_name).map(|(min, max)| max - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::DecisionEvalResult;

    fn make_score(system: &str, param: &str, trust: QuantumTrustLevel) -> QuantumTrustScore {
        QuantumTrustScore {
            system_name: system.to_string(),
            param_name: param.to_string(),
            trust,
            z_prior: 0.0,
            overlap_95: 0.0,
            kl_prior_to_post: 0.0,
        }
    }

    fn make_design_summary(pos: f64) -> DesignSummary {
        DesignSummary {
            design_label: "Test".to_string(),
            n_per_arm: 100,
            decision_results: vec![DecisionEvalResult {
                decision_name: "Decision1".to_string(),
                endpoint_name: "ORR".to_string(),
                arm_left: "ArmA".to_string(),
                arm_right: "ArmB".to_string(),
                margin: 0.15,
                prob_threshold: 0.8,
                pos,
            }],
        }
    }

    #[test]
    fn test_scenario_high_trust_only() {
        let scores = vec![
            make_score("SYS1", "P1", QuantumTrustLevel::High),
            make_score("SYS1", "P2", QuantumTrustLevel::Moderate),
            make_score("SYS1", "P3", QuantumTrustLevel::Low),
            make_score("SYS2", "P4", QuantumTrustLevel::Broken),
        ];

        let scenario = QuantumDesignScenario::high_trust_only(&scores);

        assert_eq!(scenario.label, "QM_high_trust_only");
        assert_eq!(scenario.included_params.len(), 1); // Only P1
        assert_eq!(scenario.excluded_params.len(), 3); // P2, P3, P4
        assert_eq!(scenario.included_params[0].1, "P1");
    }

    #[test]
    fn test_scenario_high_moderate_trust() {
        let scores = vec![
            make_score("SYS1", "P1", QuantumTrustLevel::High),
            make_score("SYS1", "P2", QuantumTrustLevel::Moderate),
            make_score("SYS1", "P3", QuantumTrustLevel::Low),
            make_score("SYS2", "P4", QuantumTrustLevel::Broken),
        ];

        let scenario = QuantumDesignScenario::high_moderate_trust(&scores);

        assert_eq!(scenario.label, "QM_high_moderate_trust");
        assert_eq!(scenario.included_params.len(), 2); // P1, P2
        assert_eq!(scenario.excluded_params.len(), 2); // P3, P4
    }

    #[test]
    fn test_scenario_exclude_broken_only() {
        let scores = vec![
            make_score("SYS1", "P1", QuantumTrustLevel::High),
            make_score("SYS1", "P2", QuantumTrustLevel::Moderate),
            make_score("SYS1", "P3", QuantumTrustLevel::Low),
            make_score("SYS2", "P4", QuantumTrustLevel::Broken),
        ];

        let scenario = QuantumDesignScenario::exclude_broken_only(&scores);

        assert_eq!(scenario.label, "QM_exclude_broken");
        assert_eq!(scenario.included_params.len(), 3); // P1, P2, P3
        assert_eq!(scenario.excluded_params.len(), 1); // P4
    }

    #[test]
    fn test_default_scenarios_generation() {
        let scores = vec![
            make_score("SYS1", "P1", QuantumTrustLevel::High),
            make_score("SYS1", "P2", QuantumTrustLevel::Broken),
        ];

        let scenarios = default_scenarios_from_scores(&scores);

        assert_eq!(scenarios.len(), 4);
        assert_eq!(scenarios[0].label, "QM_baseline");
        assert_eq!(scenarios[1].label, "QM_high_moderate_trust");
        assert_eq!(scenarios[2].label, "QM_high_trust_only");
        assert_eq!(scenarios[3].label, "QM_exclude_broken");
    }

    #[test]
    fn test_sensitivity_result_construction() {
        let scenario = QuantumDesignScenario::baseline(&vec![]);
        let design = make_design_summary(0.85);

        let result = QuantumDesignSensitivityResult::new(scenario, design);

        assert_eq!(result.scenario.label, "QM_baseline");
        assert_eq!(result.design_summary.decision_results[0].pos, 0.85);
    }

    #[test]
    fn test_sensitivity_report_pos_range() {
        let report = QuantumDesignSensitivityReport {
            protocol_name: "TEST".to_string(),
            population_model: "MODEL".to_string(),
            fit_dir: "fit".to_string(),
            design_config: DesignConfigInfo {
                n_per_arm: 100,
                n_draws: 500,
            },
            results: vec![
                QuantumDesignSensitivityResult::new(
                    QuantumDesignScenario::baseline(&vec![]),
                    make_design_summary(0.90),
                ),
                QuantumDesignSensitivityResult::new(
                    QuantumDesignScenario::pessimistic(&vec![]),
                    make_design_summary(0.75),
                ),
                QuantumDesignSensitivityResult::new(
                    QuantumDesignScenario::optimistic(&vec![]),
                    make_design_summary(0.85),
                ),
            ],
        };

        let (min, max) = report.pos_range("Decision1").unwrap();
        assert!((min - 0.75).abs() < 0.01);
        assert!((max - 0.90).abs() < 0.01);

        let sensitivity = report.pos_sensitivity("Decision1").unwrap();
        assert!((sensitivity - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_find_scenario() {
        let report = QuantumDesignSensitivityReport {
            protocol_name: "TEST".to_string(),
            population_model: "MODEL".to_string(),
            fit_dir: "fit".to_string(),
            design_config: DesignConfigInfo {
                n_per_arm: 100,
                n_draws: 500,
            },
            results: vec![QuantumDesignSensitivityResult::new(
                QuantumDesignScenario::baseline(&vec![]),
                make_design_summary(0.85),
            )],
        };

        let found = report.find_scenario("QM_baseline");
        assert!(found.is_some());
        assert_eq!(found.unwrap().scenario.label, "QM_baseline");

        let not_found = report.find_scenario("NonExistent");
        assert!(not_found.is_none());
    }
}
