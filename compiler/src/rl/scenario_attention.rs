// Week 44: Scenario-Attention in Robustness Scoring
//
// Extends robustness evaluation with attention mechanisms to identify which
// scenarios dominate guideline performance variance. Uses information-theoretic
// measures and empirical sensitivity analysis to guide policy distillation.

use crate::rl::dose_guideline_ir::DoseGuidelineIRHost;
use crate::rl::dose_guideline_outcomes::DoseGuidelineOutcomeSummary;
use crate::rl::dose_guideline_robustness::EnvScenario;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Scenario Attention Metrics
// =============================================================================

/// Attention weight for a single scenario
///
/// Higher weight → scenario contributes more to guideline outcome variance
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct ScenarioAttention {
    /// Scenario name or index
    pub scenario_id: usize,

    /// Variance contribution: Var(outcome | scenario) relative to total variance
    /// Range: [0.0, 1.0], sums to 1.0 across all scenarios
    pub variance_contribution: f64,

    /// Sensitivity: d(outcome)/d(parameter) - max absolute partial derivative
    pub sensitivity: f64,

    /// Divergence: KL divergence of outcome distribution from baseline
    pub divergence: f64,

    /// Composite attention: weighted combination of variance, sensitivity, divergence
    pub attention_weight: f64,
}

impl ScenarioAttention {
    /// Create new scenario attention with default weights
    pub fn new(
        scenario_id: usize,
        variance_contribution: f64,
        sensitivity: f64,
        divergence: f64,
    ) -> Self {
        // Composite: 50% variance + 30% sensitivity + 20% divergence
        let attention_weight =
            0.5 * variance_contribution + 0.3 * sensitivity + 0.2 * divergence.ln().abs().min(1.0);

        Self {
            scenario_id,
            variance_contribution,
            sensitivity,
            divergence,
            attention_weight,
        }
    }

    /// Create with custom weights
    pub fn with_custom_weights(
        scenario_id: usize,
        variance_contribution: f64,
        sensitivity: f64,
        divergence: f64,
        var_weight: f64,
        sen_weight: f64,
        div_weight: f64,
    ) -> Self {
        let total = var_weight + sen_weight + div_weight;
        let attention_weight = (var_weight * variance_contribution
            + sen_weight * sensitivity
            + div_weight * divergence.ln().abs().min(1.0))
            / total;

        Self {
            scenario_id,
            variance_contribution,
            sensitivity,
            divergence,
            attention_weight,
        }
    }
}

// =============================================================================
// Attention Report
// =============================================================================

/// Complete scenario-attention analysis for a guideline robustness evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioAttentionReport {
    /// Guideline name
    pub guideline_name: String,

    /// Number of scenarios evaluated
    pub n_scenarios: usize,

    /// Attention for each scenario (sorted by attention_weight descending)
    pub scenario_attentions: Vec<ScenarioAttention>,

    /// Entropy of attention distribution (0 = one dominant scenario, high = uniform)
    pub attention_entropy: f64,

    /// Principal scenario(s): attention > mean + 1 std dev
    pub principal_scenarios: Vec<usize>,

    /// Secondary scenarios: attention between mean and mean + 1 std dev
    pub secondary_scenarios: Vec<usize>,

    /// Marginal scenarios: attention < mean - 0.5 * std dev
    pub marginal_scenarios: Vec<usize>,

    /// Explanation: which parameters/mechanisms drive attention distribution
    pub explanation: String,
}

impl ScenarioAttentionReport {
    /// Create report from attention scores
    pub fn from_attentions(
        guideline_name: String,
        scenario_names: &[String],
        mut attentions: Vec<ScenarioAttention>,
    ) -> Self {
        // Sort by attention weight descending
        attentions.sort_by(|a, b| b.attention_weight.partial_cmp(&a.attention_weight).unwrap());

        let n_scenarios = attentions.len();

        // Compute entropy
        let attention_entropy = compute_entropy(&attentions);

        // Categorize scenarios
        let mean_attention =
            attentions.iter().map(|a| a.attention_weight).sum::<f64>() / n_scenarios as f64;
        let variance = attentions
            .iter()
            .map(|a| (a.attention_weight - mean_attention).powi(2))
            .sum::<f64>()
            / n_scenarios as f64;
        let std_dev = variance.sqrt();

        let mut principal_scenarios = Vec::new();
        let mut secondary_scenarios = Vec::new();
        let mut marginal_scenarios = Vec::new();

        for att in &attentions {
            if att.attention_weight > mean_attention + std_dev {
                principal_scenarios.push(att.scenario_id);
            } else if att.attention_weight > mean_attention - 0.5 * std_dev {
                secondary_scenarios.push(att.scenario_id);
            } else {
                marginal_scenarios.push(att.scenario_id);
            }
        }

        // Generate explanation
        let explanation = if !principal_scenarios.is_empty() {
            let principal_names: Vec<&String> = principal_scenarios
                .iter()
                .map(|&i| &scenario_names[i])
                .collect();
            format!(
                "Principal scenarios dominate guideline performance: {:?}. \
                 These scenarios (entropy={:.3}) drive {} of outcome variance.",
                principal_names,
                attention_entropy,
                (attentions[0].variance_contribution * 100.0).round() as i32
            )
        } else {
            format!(
                "No clear principal scenario. Guideline performance is sensitive to {} scenarios equally.",
                n_scenarios
            )
        };

        Self {
            guideline_name,
            n_scenarios,
            scenario_attentions: attentions,
            attention_entropy,
            principal_scenarios,
            secondary_scenarios,
            marginal_scenarios,
            explanation,
        }
    }

    /// Print report to stderr
    pub fn print_report(&self) {
        eprintln!("\n═══════════════════════════════════════════════════════════════");
        eprintln!("Scenario-Attention Analysis: {}", self.guideline_name);
        eprintln!("═══════════════════════════════════════════════════════════════");
        eprintln!();

        eprintln!("Attention Entropy: {:.4}", self.attention_entropy);
        eprintln!(
            "  (0.0 = single dominant scenario, {:.2} = uniform)",
            (self.n_scenarios as f64).log2()
        );
        eprintln!();

        eprintln!("Scenario Rankings:");
        for (i, att) in self.scenario_attentions.iter().take(5).enumerate() {
            let rank_type = if self.principal_scenarios.contains(&att.scenario_id) {
                "PRINCIPAL"
            } else if self.secondary_scenarios.contains(&att.scenario_id) {
                "SECONDARY"
            } else {
                "MARGINAL"
            };
            eprintln!(
                "  [{}] Scenario {} - Attention: {:.4} {} [Var: {:.3}, Sen: {:.3}, Div: {:.3}]",
                i + 1,
                att.scenario_id,
                att.attention_weight,
                rank_type,
                att.variance_contribution,
                att.sensitivity,
                att.divergence
            );
        }
        if self.scenario_attentions.len() > 5 {
            eprintln!(
                "  ... and {} more scenarios",
                self.scenario_attentions.len() - 5
            );
        }
        eprintln!();

        eprintln!("Analysis:");
        eprintln!("  {}", self.explanation);
        eprintln!();
    }
}

// =============================================================================
// Attention Computation
// =============================================================================

/// Compute attention scores from robustness evaluation results
///
/// Input: outcomes from each scenario in robustness sweep
/// Output: scenario attention weights (sum to 1.0)
pub fn compute_scenario_attention(
    baseline_outcome: &DoseGuidelineOutcomeSummary,
    scenario_outcomes: &[(&EnvScenario, DoseGuidelineOutcomeSummary)],
) -> Result<Vec<ScenarioAttention>> {
    let mut attentions = Vec::new();

    // Compute baseline metrics
    let baseline_orr = baseline_outcome.response_rate;
    let baseline_tox_rate = baseline_outcome.grade3plus_rate;
    let baseline_rdi = baseline_outcome.mean_rdi;

    // Collect outcomes for variance computation
    let mut orr_values = vec![baseline_orr];
    let mut tox_values = vec![baseline_tox_rate];
    let mut rdi_values = vec![baseline_rdi];

    for (_, outcome) in scenario_outcomes {
        orr_values.push(outcome.response_rate);
        tox_values.push(outcome.grade3plus_rate);
        rdi_values.push(outcome.mean_rdi);
    }

    // Compute total variance across all outcomes
    let orr_var = compute_variance(&orr_values);
    let tox_var = compute_variance(&tox_values);
    let rdi_var = compute_variance(&rdi_values);
    let total_var = orr_var + tox_var + rdi_var;

    // For each scenario, compute attention
    for (idx, (scenario, outcome)) in scenario_outcomes.iter().enumerate() {
        // Variance contribution: how much does this scenario variance differ from baseline?
        let orr_delta = (outcome.response_rate - baseline_orr).abs();
        let tox_delta = (outcome.grade3plus_rate - baseline_tox_rate).abs();
        let rdi_delta = (outcome.mean_rdi - baseline_rdi).abs();

        let variance_contribution = if total_var > 0.0 {
            (orr_delta * orr_var + tox_delta * tox_var + rdi_delta * rdi_var) / total_var
        } else {
            1.0 / scenario_outcomes.len() as f64
        };

        // Sensitivity: max relative change
        let sensitivity = {
            let orr_rel = if baseline_orr > 0.001 {
                orr_delta / baseline_orr
            } else {
                orr_delta
            };
            let tox_rel = if baseline_tox_rate > 0.001 {
                tox_delta / baseline_tox_rate
            } else {
                tox_delta
            };
            let rdi_rel = if baseline_rdi > 0.001 {
                rdi_delta / baseline_rdi
            } else {
                rdi_delta
            };
            orr_rel.abs().max(tox_rel.abs()).max(rdi_rel.abs())
        };

        // Divergence: Euclidean distance in 3D outcome space
        let divergence = {
            let d_orr = (outcome.response_rate - baseline_orr).powi(2);
            let d_tox = (outcome.grade3plus_rate - baseline_tox_rate).powi(2);
            let d_rdi = (outcome.mean_rdi - baseline_rdi).powi(2);
            (d_orr + d_tox + d_rdi).sqrt()
        };

        let att = ScenarioAttention::new(idx, variance_contribution, sensitivity, divergence);
        attentions.push(att);
    }

    // Normalize attention weights to sum to 1.0
    let total_weight: f64 = attentions.iter().map(|a| a.attention_weight).sum();
    for att in &mut attentions {
        att.attention_weight /= total_weight;
    }

    Ok(attentions)
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute variance of a list of values
fn compute_variance(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
}

/// Compute Shannon entropy of a probability distribution
fn compute_entropy(attentions: &[ScenarioAttention]) -> f64 {
    attentions
        .iter()
        .filter(|a| a.attention_weight > 1e-10)
        .map(|a| {
            let p = a.attention_weight;
            -p * p.log2()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scenario_attention_creation() {
        let att = ScenarioAttention::new(0, 0.6, 0.3, 0.5);
        assert!(att.attention_weight > 0.0);
        assert!(att.variance_contribution == 0.6);
    }

    #[test]
    fn test_attention_entropy() {
        // Single scenario (entropy should be 0)
        let att_single = vec![ScenarioAttention::with_custom_weights(0, 1.0, 0.5, 0.5, 1.0, 0.0, 0.0)];
        let ent = compute_entropy(&att_single);
        assert!(ent < 0.01); // Close to 0

        // Uniform distribution (entropy should be log2(n))
        let n = 4;
        let att_uniform: Vec<_> = (0..n)
            .map(|i| {
                let mut att = ScenarioAttention::new(i, 0.25, 0.5, 0.5);
                att.attention_weight = 0.25;
                att
            })
            .collect();
        let ent = compute_entropy(&att_uniform);
        assert!((ent - 2.0).abs() < 0.1); // log2(4) = 2
    }

    #[test]
    fn test_scenario_categorization() {
        let attentions = vec![
            ScenarioAttention {
                scenario_id: 0,
                variance_contribution: 0.7,
                sensitivity: 0.8,
                divergence: 0.9,
                attention_weight: 0.6,
            },
            ScenarioAttention {
                scenario_id: 1,
                variance_contribution: 0.2,
                sensitivity: 0.3,
                divergence: 0.4,
                attention_weight: 0.3,
            },
            ScenarioAttention {
                scenario_id: 2,
                variance_contribution: 0.1,
                sensitivity: 0.1,
                divergence: 0.1,
                attention_weight: 0.1,
            },
        ];

        let report = ScenarioAttentionReport::from_attentions(
            "Test".to_string(),
            &[
                "Scenario 0".to_string(),
                "Scenario 1".to_string(),
                "Scenario 2".to_string(),
            ],
            attentions,
        );

        assert_eq!(report.n_scenarios, 3);
        assert!(!report.principal_scenarios.is_empty());
    }
}
