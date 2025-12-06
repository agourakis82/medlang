// Week 42: Env-parameter Robustness Analysis for Dose Guidelines
//
// Evaluates how guideline outcomes change when perturbing DoseToxEnv parameters.
// This provides sensitivity analysis without requiring QM/PK infrastructure.

use crate::rl::dose_guideline_ir::DoseGuidelineIRHost;
use crate::rl::dose_guideline_outcomes::{
    simulate_dose_guideline_outcomes, DoseGuidelineOutcomeConfig, DoseGuidelineOutcomeSummary,
};
use crate::rl::env_dose_tox::DoseToxEnvConfig;
use anyhow::Result;
use serde::{Deserialize, Serialize};

// =============================================================================
// Scenario Definition
// =============================================================================

/// An env-parameter scenario: name + overrides relative to a base config.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvScenario {
    /// Scenario name (e.g., "tox-light", "tox-heavy", "high-efficacy")
    pub name: String,

    /// Optional override for number of treatment cycles
    #[serde(default)]
    pub n_cycles: Option<usize>,

    /// Optional override for response/efficacy reward weight
    #[serde(default)]
    pub reward_response_weight: Option<f64>,

    /// Optional override for toxicity penalty weight
    #[serde(default)]
    pub reward_tox_penalty: Option<f64>,

    /// Optional override for contract violation penalty
    #[serde(default)]
    pub contract_penalty: Option<f64>,

    /// Optional dose scaling factor (multiplies all dose levels)
    #[serde(default)]
    pub dose_scale: Option<f64>,

    /// Optional seed override for reproducibility
    #[serde(default)]
    pub seed: Option<u64>,
}

impl EnvScenario {
    /// Create a new scenario with just a name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            n_cycles: None,
            reward_response_weight: None,
            reward_tox_penalty: None,
            contract_penalty: None,
            dose_scale: None,
            seed: None,
        }
    }

    /// Builder: set n_cycles override
    pub fn with_n_cycles(mut self, n: usize) -> Self {
        self.n_cycles = Some(n);
        self
    }

    /// Builder: set reward_response_weight override
    pub fn with_response_weight(mut self, w: f64) -> Self {
        self.reward_response_weight = Some(w);
        self
    }

    /// Builder: set reward_tox_penalty override
    pub fn with_tox_penalty(mut self, w: f64) -> Self {
        self.reward_tox_penalty = Some(w);
        self
    }

    /// Builder: set contract_penalty override
    pub fn with_contract_penalty(mut self, p: f64) -> Self {
        self.contract_penalty = Some(p);
        self
    }

    /// Builder: set dose_scale factor
    pub fn with_dose_scale(mut self, s: f64) -> Self {
        self.dose_scale = Some(s);
        self
    }

    /// Builder: set seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

// =============================================================================
// Per-Scenario Summary
// =============================================================================

/// Outcome summary for a single env scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineEnvScenarioSummary {
    /// Name of the scenario
    pub scenario_name: String,

    /// The modified env config used for this scenario
    pub env_config: DoseToxEnvConfig,

    /// Outcome summary from simulation
    pub outcome: DoseGuidelineOutcomeSummary,
}

// =============================================================================
// Robustness Report
// =============================================================================

/// Robustness report for one guideline across multiple env scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineEnvRobustnessReport {
    /// Name of the guideline
    pub guideline_name: String,

    /// Base env config (before scenario overrides)
    pub base_env: DoseToxEnvConfig,

    /// Outcome simulation config
    pub outcome_config: DoseGuidelineOutcomeConfig,

    /// Results for base scenario (no overrides)
    pub base_outcome: DoseGuidelineOutcomeSummary,

    /// Results for each modified scenario
    pub scenarios: Vec<GuidelineEnvScenarioSummary>,
}

impl GuidelineEnvRobustnessReport {
    /// Compute robustness metrics across scenarios
    pub fn robustness_summary(&self) -> RobustnessSummary {
        if self.scenarios.is_empty() {
            return RobustnessSummary {
                response_rate_range: (
                    self.base_outcome.response_rate,
                    self.base_outcome.response_rate,
                ),
                grade3plus_rate_range: (
                    self.base_outcome.grade3plus_rate,
                    self.base_outcome.grade3plus_rate,
                ),
                contract_violation_range: (
                    self.base_outcome.contract_violation_rate,
                    self.base_outcome.contract_violation_rate,
                ),
                mean_rdi_range: (self.base_outcome.mean_rdi, self.base_outcome.mean_rdi),
                worst_scenario: None,
                best_scenario: None,
            };
        }

        let all_outcomes: Vec<_> = std::iter::once(&self.base_outcome)
            .chain(self.scenarios.iter().map(|s| &s.outcome))
            .collect();

        let response_rates: Vec<f64> = all_outcomes.iter().map(|o| o.response_rate).collect();
        let g3_rates: Vec<f64> = all_outcomes.iter().map(|o| o.grade3plus_rate).collect();
        let cv_rates: Vec<f64> = all_outcomes
            .iter()
            .map(|o| o.contract_violation_rate)
            .collect();
        let rdi_rates: Vec<f64> = all_outcomes.iter().map(|o| o.mean_rdi).collect();

        let min_max = |v: &[f64]| {
            let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (min, max)
        };

        // Find worst scenario by composite score (low response + high tox)
        let worst_idx = self
            .scenarios
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let score_a = a.outcome.response_rate - a.outcome.grade3plus_rate;
                let score_b = b.outcome.response_rate - b.outcome.grade3plus_rate;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        let best_idx = self
            .scenarios
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                let score_a = a.outcome.response_rate - a.outcome.grade3plus_rate;
                let score_b = b.outcome.response_rate - b.outcome.grade3plus_rate;
                score_a
                    .partial_cmp(&score_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i);

        RobustnessSummary {
            response_rate_range: min_max(&response_rates),
            grade3plus_rate_range: min_max(&g3_rates),
            contract_violation_range: min_max(&cv_rates),
            mean_rdi_range: min_max(&rdi_rates),
            worst_scenario: worst_idx.map(|i| self.scenarios[i].scenario_name.clone()),
            best_scenario: best_idx.map(|i| self.scenarios[i].scenario_name.clone()),
        }
    }
}

/// Summary statistics across all scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessSummary {
    /// (min, max) response rate across scenarios
    pub response_rate_range: (f64, f64),

    /// (min, max) grade 3+ toxicity rate
    pub grade3plus_rate_range: (f64, f64),

    /// (min, max) contract violation rate
    pub contract_violation_range: (f64, f64),

    /// (min, max) mean RDI
    pub mean_rdi_range: (f64, f64),

    /// Scenario with worst composite outcome
    pub worst_scenario: Option<String>,

    /// Scenario with best composite outcome
    pub best_scenario: Option<String>,
}

// =============================================================================
// Core Functions
// =============================================================================

/// Apply scenario overrides to a base env config.
pub fn apply_env_scenario(base: &DoseToxEnvConfig, scen: &EnvScenario) -> DoseToxEnvConfig {
    let mut cfg = base.clone();

    if let Some(nc) = scen.n_cycles {
        cfg.n_cycles = nc;
    }

    if let Some(wr) = scen.reward_response_weight {
        cfg.reward_response_weight = wr;
    }

    if let Some(wt) = scen.reward_tox_penalty {
        cfg.reward_tox_penalty = wt;
    }

    if let Some(cp) = scen.contract_penalty {
        cfg.contract_penalty = cp;
    }

    if let Some(scale) = scen.dose_scale {
        if scale > 0.0 {
            cfg.dose_levels_mg = cfg.dose_levels_mg.iter().map(|d| d * scale).collect();
        }
    }

    if let Some(seed) = scen.seed {
        cfg.seed = Some(seed);
    }

    cfg
}

/// Simulate a guideline across multiple env-parameter scenarios.
///
/// Returns a robustness report containing:
/// - Base outcome (no overrides)
/// - Per-scenario outcomes
/// - Aggregate robustness metrics
pub fn simulate_guideline_env_robustness(
    base_env: &DoseToxEnvConfig,
    guideline: &DoseGuidelineIRHost,
    outcome_cfg: &DoseGuidelineOutcomeConfig,
    scenarios: &[EnvScenario],
) -> Result<GuidelineEnvRobustnessReport> {
    // First, simulate base scenario
    let base_outcome = simulate_dose_guideline_outcomes(base_env, guideline, outcome_cfg)?;

    // Then simulate each modified scenario
    let mut scen_summaries = Vec::with_capacity(scenarios.len());

    for scen in scenarios {
        let env_cfg = apply_env_scenario(base_env, scen);

        let outcome = simulate_dose_guideline_outcomes(&env_cfg, guideline, outcome_cfg)?;

        scen_summaries.push(GuidelineEnvScenarioSummary {
            scenario_name: scen.name.clone(),
            env_config: env_cfg,
            outcome,
        });
    }

    Ok(GuidelineEnvRobustnessReport {
        guideline_name: guideline.name.clone(),
        base_env: base_env.clone(),
        outcome_config: outcome_cfg.clone(),
        base_outcome,
        scenarios: scen_summaries,
    })
}

/// Create a default set of robustness scenarios for sensitivity analysis.
///
/// Returns scenarios that test:
/// - Toxicity penalty variations (0.5x, 2x)
/// - Response weight variations (0.5x, 2x)
/// - Contract penalty variations (0.5x, 2x)
/// - Dose scaling (0.8x, 1.2x)
pub fn default_robustness_scenarios() -> Vec<EnvScenario> {
    vec![
        EnvScenario::new("tox-light").with_tox_penalty(1.0),
        EnvScenario::new("tox-heavy").with_tox_penalty(4.0),
        EnvScenario::new("efficacy-light").with_response_weight(0.5),
        EnvScenario::new("efficacy-heavy").with_response_weight(2.0),
        EnvScenario::new("contract-light").with_contract_penalty(5.0),
        EnvScenario::new("contract-heavy").with_contract_penalty(20.0),
        EnvScenario::new("dose-reduced").with_dose_scale(0.8),
        EnvScenario::new("dose-increased").with_dose_scale(1.2),
        EnvScenario::new("short-treatment").with_n_cycles(4),
        EnvScenario::new("long-treatment").with_n_cycles(8),
    ]
}

/// Pretty-print a robustness report to string.
pub fn format_robustness_report(report: &GuidelineEnvRobustnessReport) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "=== Env-Parameter Robustness Report: {} ===\n\n",
        report.guideline_name
    ));

    out.push_str("Base Environment:\n");
    out.push_str(&format!("  n_cycles: {}\n", report.base_env.n_cycles));
    out.push_str(&format!(
        "  reward_response_weight: {:.2}\n",
        report.base_env.reward_response_weight
    ));
    out.push_str(&format!(
        "  reward_tox_penalty: {:.2}\n",
        report.base_env.reward_tox_penalty
    ));
    out.push_str(&format!(
        "  contract_penalty: {:.2}\n",
        report.base_env.contract_penalty
    ));
    out.push_str(&format!(
        "  dose_levels: {:?}\n\n",
        report.base_env.dose_levels_mg
    ));

    out.push_str("Outcome Config:\n");
    out.push_str(&format!(
        "  n_episodes: {}\n",
        report.outcome_config.n_episodes
    ));
    out.push_str(&format!(
        "  response_threshold: {:.2}\n\n",
        report.outcome_config.response_tumour_ratio_threshold
    ));

    // Base outcome
    out.push_str("Base Outcome:\n");
    out.push_str(&format_outcome(&report.base_outcome, "  "));
    out.push('\n');

    // Scenario outcomes
    out.push_str("Scenario Outcomes:\n");
    out.push_str(&format!(
        "{:<20} {:>10} {:>10} {:>10} {:>10}\n",
        "Scenario", "Response", "G3+ Tox", "Contract", "RDI"
    ));
    out.push_str(&format!("{}\n", "-".repeat(64)));

    out.push_str(&format!(
        "{:<20} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.2}\n",
        "base",
        report.base_outcome.response_rate * 100.0,
        report.base_outcome.grade3plus_rate * 100.0,
        report.base_outcome.contract_violation_rate * 100.0,
        report.base_outcome.mean_rdi
    ));

    for scen in &report.scenarios {
        out.push_str(&format!(
            "{:<20} {:>9.1}% {:>9.1}% {:>9.1}% {:>9.2}\n",
            scen.scenario_name,
            scen.outcome.response_rate * 100.0,
            scen.outcome.grade3plus_rate * 100.0,
            scen.outcome.contract_violation_rate * 100.0,
            scen.outcome.mean_rdi
        ));
    }
    out.push('\n');

    // Robustness summary
    let summary = report.robustness_summary();
    out.push_str("Robustness Summary:\n");
    out.push_str(&format!(
        "  Response rate range: {:.1}% - {:.1}%\n",
        summary.response_rate_range.0 * 100.0,
        summary.response_rate_range.1 * 100.0
    ));
    out.push_str(&format!(
        "  G3+ toxicity range:  {:.1}% - {:.1}%\n",
        summary.grade3plus_rate_range.0 * 100.0,
        summary.grade3plus_rate_range.1 * 100.0
    ));
    out.push_str(&format!(
        "  Contract viol range: {:.1}% - {:.1}%\n",
        summary.contract_violation_range.0 * 100.0,
        summary.contract_violation_range.1 * 100.0
    ));
    out.push_str(&format!(
        "  Mean RDI range:      {:.2} - {:.2}\n",
        summary.mean_rdi_range.0, summary.mean_rdi_range.1
    ));

    if let Some(worst) = &summary.worst_scenario {
        out.push_str(&format!("  Worst scenario: {}\n", worst));
    }
    if let Some(best) = &summary.best_scenario {
        out.push_str(&format!("  Best scenario:  {}\n", best));
    }

    out
}

fn format_outcome(o: &DoseGuidelineOutcomeSummary, indent: &str) -> String {
    format!(
        "{}n_episodes: {}\n{}response_rate: {:.1}%\n{}grade3plus_rate: {:.1}%\n{}grade4plus_rate: {:.1}%\n{}contract_violation_rate: {:.1}%\n{}mean_rdi: {:.2}\n",
        indent, o.n_episodes,
        indent, o.response_rate * 100.0,
        indent, o.grade3plus_rate * 100.0,
        indent, o.grade4plus_rate * 100.0,
        indent, o.contract_violation_rate * 100.0,
        indent, o.mean_rdi
    )
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::dose_guideline_ir::{AtomicConditionIR, ComparisonOpIR, DoseRuleIR};

    fn make_simple_guideline() -> DoseGuidelineIRHost {
        // Simple guideline: if ANC >= 0.5 give 200mg, else give 100mg
        DoseGuidelineIRHost {
            name: "test-guideline".to_string(),
            description: "Test guideline for robustness".to_string(),
            feature_names: vec!["ANC".to_string()],
            dose_levels_mg: vec![0.0, 50.0, 100.0, 200.0, 300.0],
            rules: vec![
                DoseRuleIR {
                    conditions: vec![AtomicConditionIR {
                        feature: "ANC".to_string(),
                        op: ComparisonOpIR::GE,
                        threshold: 0.5,
                    }],
                    action_index: 3, // 200mg
                    action_dose_mg: 200.0,
                },
                DoseRuleIR {
                    conditions: vec![AtomicConditionIR {
                        feature: "ANC".to_string(),
                        op: ComparisonOpIR::LT,
                        threshold: 0.5,
                    }],
                    action_index: 2, // 100mg
                    action_dose_mg: 100.0,
                },
            ],
        }
    }

    #[test]
    fn test_env_scenario_builder() {
        let scen = EnvScenario::new("test")
            .with_n_cycles(8)
            .with_tox_penalty(3.0)
            .with_dose_scale(0.9);

        assert_eq!(scen.name, "test");
        assert_eq!(scen.n_cycles, Some(8));
        assert_eq!(scen.reward_tox_penalty, Some(3.0));
        assert_eq!(scen.dose_scale, Some(0.9));
        assert!(scen.reward_response_weight.is_none());
    }

    #[test]
    fn test_apply_env_scenario_n_cycles() {
        let base = DoseToxEnvConfig::default();
        let scen = EnvScenario::new("test").with_n_cycles(10);

        let modified = apply_env_scenario(&base, &scen);

        assert_eq!(modified.n_cycles, 10);
        assert_eq!(modified.reward_response_weight, base.reward_response_weight);
    }

    #[test]
    fn test_apply_env_scenario_weights() {
        let base = DoseToxEnvConfig::default();
        let scen = EnvScenario::new("test")
            .with_response_weight(0.5)
            .with_tox_penalty(4.0);

        let modified = apply_env_scenario(&base, &scen);

        assert_eq!(modified.reward_response_weight, 0.5);
        assert_eq!(modified.reward_tox_penalty, 4.0);
    }

    #[test]
    fn test_apply_env_scenario_dose_scale() {
        let base = DoseToxEnvConfig {
            dose_levels_mg: vec![0.0, 100.0, 200.0],
            ..DoseToxEnvConfig::default()
        };
        let scen = EnvScenario::new("test").with_dose_scale(0.5);

        let modified = apply_env_scenario(&base, &scen);

        assert_eq!(modified.dose_levels_mg, vec![0.0, 50.0, 100.0]);
    }

    #[test]
    fn test_apply_env_scenario_no_overrides() {
        let base = DoseToxEnvConfig::default();
        let scen = EnvScenario::new("no-changes");

        let modified = apply_env_scenario(&base, &scen);

        assert_eq!(modified.n_cycles, base.n_cycles);
        assert_eq!(modified.reward_response_weight, base.reward_response_weight);
        assert_eq!(modified.reward_tox_penalty, base.reward_tox_penalty);
        assert_eq!(modified.dose_levels_mg, base.dose_levels_mg);
    }

    #[test]
    fn test_default_robustness_scenarios() {
        let scenarios = default_robustness_scenarios();

        assert_eq!(scenarios.len(), 10);

        // Check some specific scenarios exist
        let names: Vec<_> = scenarios.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"tox-light"));
        assert!(names.contains(&"tox-heavy"));
        assert!(names.contains(&"dose-reduced"));
        assert!(names.contains(&"long-treatment"));
    }

    #[test]
    fn test_simulate_robustness_basic() {
        let base_env = DoseToxEnvConfig {
            seed: Some(42),
            n_cycles: 4,
            ..DoseToxEnvConfig::default()
        };

        let guideline = make_simple_guideline();

        let outcome_cfg = DoseGuidelineOutcomeConfig {
            n_episodes: 10,
            response_tumour_ratio_threshold: 0.8,
            grade3_threshold: 3,
            grade4_threshold: 4,
        };

        let scenarios = vec![
            EnvScenario::new("tox-light").with_tox_penalty(1.0),
            EnvScenario::new("tox-heavy").with_tox_penalty(4.0),
        ];

        let report =
            simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios)
                .unwrap();

        assert_eq!(report.guideline_name, "test-guideline");
        assert_eq!(report.scenarios.len(), 2);
        assert_eq!(report.scenarios[0].scenario_name, "tox-light");
        assert_eq!(report.scenarios[1].scenario_name, "tox-heavy");

        // All outcomes should have correct n_episodes
        assert_eq!(report.base_outcome.n_episodes, 10);
        for scen in &report.scenarios {
            assert_eq!(scen.outcome.n_episodes, 10);
        }
    }

    #[test]
    fn test_robustness_summary() {
        let base_env = DoseToxEnvConfig {
            seed: Some(42),
            n_cycles: 3,
            ..DoseToxEnvConfig::default()
        };

        let guideline = make_simple_guideline();

        let outcome_cfg = DoseGuidelineOutcomeConfig {
            n_episodes: 5,
            response_tumour_ratio_threshold: 0.8,
            grade3_threshold: 3,
            grade4_threshold: 4,
        };

        let scenarios = vec![
            EnvScenario::new("s1").with_tox_penalty(1.0),
            EnvScenario::new("s2").with_tox_penalty(3.0),
        ];

        let report =
            simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios)
                .unwrap();

        let summary = report.robustness_summary();

        // Ranges should be valid (min <= max)
        assert!(summary.response_rate_range.0 <= summary.response_rate_range.1);
        assert!(summary.grade3plus_rate_range.0 <= summary.grade3plus_rate_range.1);
        assert!(summary.mean_rdi_range.0 <= summary.mean_rdi_range.1);

        // Should identify best/worst
        assert!(summary.worst_scenario.is_some() || summary.best_scenario.is_some());
    }

    #[test]
    fn test_format_robustness_report() {
        let base_env = DoseToxEnvConfig {
            seed: Some(42),
            n_cycles: 3,
            ..DoseToxEnvConfig::default()
        };

        let guideline = make_simple_guideline();

        let outcome_cfg = DoseGuidelineOutcomeConfig {
            n_episodes: 5,
            response_tumour_ratio_threshold: 0.8,
            grade3_threshold: 3,
            grade4_threshold: 4,
        };

        let scenarios = vec![EnvScenario::new("test-scen").with_tox_penalty(2.0)];

        let report =
            simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios)
                .unwrap();

        let formatted = format_robustness_report(&report);

        // Check key sections are present
        assert!(formatted.contains("Env-Parameter Robustness Report"));
        assert!(formatted.contains("test-guideline"));
        assert!(formatted.contains("Base Environment"));
        assert!(formatted.contains("Scenario Outcomes"));
        assert!(formatted.contains("test-scen"));
        assert!(formatted.contains("Robustness Summary"));
    }

    #[test]
    fn test_empty_scenarios() {
        let base_env = DoseToxEnvConfig {
            seed: Some(42),
            n_cycles: 3,
            ..DoseToxEnvConfig::default()
        };

        let guideline = make_simple_guideline();

        let outcome_cfg = DoseGuidelineOutcomeConfig {
            n_episodes: 5,
            response_tumour_ratio_threshold: 0.8,
            grade3_threshold: 3,
            grade4_threshold: 4,
        };

        let scenarios: Vec<EnvScenario> = vec![];

        let report =
            simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios)
                .unwrap();

        assert!(report.scenarios.is_empty());

        // Summary should still work
        let summary = report.robustness_summary();
        assert_eq!(summary.response_rate_range.0, summary.response_rate_range.1);
        assert!(summary.worst_scenario.is_none());
    }
}
