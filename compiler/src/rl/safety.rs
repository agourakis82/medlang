// Week 35: RL Policy Safety Analysis
//
// Provides safety analysis for RL policies by running them over many episodes
// and aggregating safety-critical metrics (contract violations, toxicity events,
// dose limits, etc.)

use crate::rl::core::RLEnv;
use crate::rl::train::RLPolicyHandle;
use serde::{Deserialize, Serialize};

/// Kinds of safety violations tracked at RL level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SafetyViolationKind {
    /// Underlying contract system fired
    ContractViolation,
    /// Severe toxicity event (e.g., grade 4)
    SevereToxicity,
    /// Action exceeds configured bounds
    DoseOutOfRange,
    /// Dose change above configured limit
    DoseChangeTooLarge,
    /// Violation of eligibility or hard guideline rule
    GuidelineViolation,
}

/// Individual safety violation recorded during policy evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyViolation {
    pub kind: SafetyViolationKind,
    pub episode: usize,
    pub step: usize,
    pub message: String,
}

/// Configuration for policy safety analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySafetyConfig {
    /// Number of episodes (virtual patients) to simulate
    pub n_episodes: usize,

    /// Maximum steps per episode (cycles)
    pub max_steps_per_episode: usize,

    /// Hard limits for dose safety (optional; env may enforce more)
    pub max_dose_mg: Option<f64>,

    /// Maximum allowed dose change per step (mg)
    pub max_delta_dose_mg: Option<f64>,

    /// Thresholds for toxicity: allow at most this many severe events
    pub max_severe_toxicity_episodes: Option<usize>,

    /// Maximum total contract violations allowed
    pub max_total_contract_violations: Option<usize>,

    /// Optional: gate episodes using a static guideline (eligibility)
    pub use_guideline_gate: bool,

    /// Name of @guideline function to call if gate enabled
    pub guideline_name: Option<String>,

    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl PolicySafetyConfig {
    /// Create a default safety config
    pub fn default() -> Self {
        Self {
            n_episodes: 100,
            max_steps_per_episode: 10,
            max_dose_mg: None,
            max_delta_dose_mg: None,
            max_severe_toxicity_episodes: None,
            max_total_contract_violations: None,
            use_guideline_gate: false,
            guideline_name: None,
            seed: None,
        }
    }
}

/// Aggregated safety metrics for a policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicySafetyReport {
    /// Total number of episodes requested
    pub n_episodes: usize,

    /// How many episodes actually ran (some may be skipped by guideline gate)
    pub n_episodes_evaluated: usize,

    // Aggregate metrics
    pub total_contract_violations: usize,
    pub total_severe_toxicity_events: usize,
    pub total_dose_out_of_range: usize,
    pub total_dose_change_too_large: usize,
    pub total_guideline_violations: usize,

    // Episode-level statistics
    pub episodes_with_severe_toxicity: usize,
    pub episodes_with_any_violation: usize,

    // Sanity: average reward if available
    pub avg_reward: f64,

    // Pass/fail relative to thresholds in PolicySafetyConfig
    pub safety_pass: bool,

    // Sampled violations (for debugging, capped at 100)
    pub sample_violations: Vec<SafetyViolation>,
}

impl PolicySafetyReport {
    /// Create an empty report
    pub fn new(n_episodes: usize) -> Self {
        Self {
            n_episodes,
            n_episodes_evaluated: 0,
            total_contract_violations: 0,
            total_severe_toxicity_events: 0,
            total_dose_out_of_range: 0,
            total_dose_change_too_large: 0,
            total_guideline_violations: 0,
            episodes_with_severe_toxicity: 0,
            episodes_with_any_violation: 0,
            avg_reward: 0.0,
            safety_pass: true,
            sample_violations: Vec::new(),
        }
    }

    /// Check if safety thresholds are met
    pub fn check_safety_pass(&mut self, cfg: &PolicySafetyConfig) {
        let mut pass = true;

        if let Some(max_severe) = cfg.max_severe_toxicity_episodes {
            if self.episodes_with_severe_toxicity > max_severe {
                pass = false;
            }
        }

        if let Some(max_contract) = cfg.max_total_contract_violations {
            if self.total_contract_violations > max_contract {
                pass = false;
            }
        }

        self.safety_pass = pass;
    }
}

/// Helper to add a violation to the sample list (capped at 100)
fn push_sample_violation(
    buf: &mut Vec<SafetyViolation>,
    kind: SafetyViolationKind,
    episode: usize,
    step: usize,
    msg: String,
) {
    if buf.len() < 100 {
        buf.push(SafetyViolation {
            kind,
            episode,
            step,
            message: msg,
        });
    }
}

/// Run policy safety analysis
///
/// This function:
/// 1. Simulates the policy over n_episodes
/// 2. Aggregates safety violations (toxicity, contracts, dose limits)
/// 3. Compares against thresholds
/// 4. Returns a comprehensive PolicySafetyReport
pub fn check_policy_safety(
    env: &mut dyn RLEnv,
    policy: &RLPolicyHandle,
    cfg: &PolicySafetyConfig,
) -> anyhow::Result<PolicySafetyReport> {
    let mut report = PolicySafetyReport::new(cfg.n_episodes);
    let mut total_reward = 0.0;

    // Run episodes
    for ep in 0..cfg.n_episodes {
        // Optional: guideline gate (stub for v0.1)
        if cfg.use_guideline_gate {
            if let Some(ref _gname) = cfg.guideline_name {
                // TODO: Implement guideline eligibility check
                // For v0.1, we skip this and assume all patients are eligible
            }
        }

        // Reset environment
        let mut state = env.reset()?;
        let mut ep_reward = 0.0;
        let mut ep_severe_tox = false;
        let mut ep_any_violation = false;

        // Run episode
        for step in 0..cfg.max_steps_per_episode {
            // Greedy policy action
            let action = policy.select_action_greedy(&state)?;

            // Take step
            let step_res = env.step(action)?;
            let info = &step_res.info;
            ep_reward += step_res.reward;

            // Check dose limits
            if let Some(dose) = info.dose_mg {
                if let Some(max_dose) = cfg.max_dose_mg {
                    if dose > max_dose {
                        report.total_dose_out_of_range += 1;
                        ep_any_violation = true;
                        push_sample_violation(
                            &mut report.sample_violations,
                            SafetyViolationKind::DoseOutOfRange,
                            ep,
                            step,
                            format!("Dose {:.1} mg > max_dose {:.1} mg", dose, max_dose),
                        );
                    }
                }
            }

            // Check delta dose
            if let (Some(dose), Some(prev)) = (info.dose_mg, info.prev_dose_mg) {
                if let Some(max_delta) = cfg.max_delta_dose_mg {
                    let delta = (dose - prev).abs();
                    if delta > max_delta {
                        report.total_dose_change_too_large += 1;
                        ep_any_violation = true;
                        push_sample_violation(
                            &mut report.sample_violations,
                            SafetyViolationKind::DoseChangeTooLarge,
                            ep,
                            step,
                            format!("Δdose {:.1} mg > max_delta {:.1} mg", delta, max_delta),
                        );
                    }
                }
            }

            // Check contract violations
            if info.contract_violations > 0 {
                report.total_contract_violations += info.contract_violations;
                ep_any_violation = true;
                push_sample_violation(
                    &mut report.sample_violations,
                    SafetyViolationKind::ContractViolation,
                    ep,
                    step,
                    format!("{} contract violations in step", info.contract_violations),
                );
            }

            // Check severe toxicity (grade 4+)
            if let Some(grade) = info.toxicity_grade {
                if grade >= 4 {
                    report.total_severe_toxicity_events += 1;
                    ep_severe_tox = true;
                    ep_any_violation = true;
                    push_sample_violation(
                        &mut report.sample_violations,
                        SafetyViolationKind::SevereToxicity,
                        ep,
                        step,
                        format!("Toxicity grade {} ≥ 4", grade),
                    );
                }
            }

            state = step_res.next_state;

            if step_res.done {
                break;
            }
        }

        // Update episode-level counters
        if ep_severe_tox {
            report.episodes_with_severe_toxicity += 1;
        }
        if ep_any_violation {
            report.episodes_with_any_violation += 1;
        }

        report.n_episodes_evaluated += 1;
        total_reward += ep_reward;
    }

    // Compute average reward
    if report.n_episodes_evaluated > 0 {
        report.avg_reward = total_reward / report.n_episodes_evaluated as f64;
    }

    // Check safety pass/fail
    report.check_safety_pass(cfg);

    Ok(report)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safety_violation_creation() {
        let violation = SafetyViolation {
            kind: SafetyViolationKind::SevereToxicity,
            episode: 5,
            step: 3,
            message: "Grade 4 neutropenia".to_string(),
        };

        assert_eq!(violation.kind, SafetyViolationKind::SevereToxicity);
        assert_eq!(violation.episode, 5);
        assert_eq!(violation.step, 3);
    }

    #[test]
    fn test_policy_safety_config_default() {
        let cfg = PolicySafetyConfig::default();
        assert_eq!(cfg.n_episodes, 100);
        assert_eq!(cfg.max_steps_per_episode, 10);
        assert_eq!(cfg.max_dose_mg, None);
        assert!(!cfg.use_guideline_gate);
    }

    #[test]
    fn test_safety_report_creation() {
        let report = PolicySafetyReport::new(50);
        assert_eq!(report.n_episodes, 50);
        assert_eq!(report.n_episodes_evaluated, 0);
        assert!(report.safety_pass);
        assert_eq!(report.total_contract_violations, 0);
    }

    #[test]
    fn test_safety_pass_threshold_severe_tox() {
        let mut report = PolicySafetyReport::new(100);
        report.episodes_with_severe_toxicity = 15;

        let cfg = PolicySafetyConfig {
            max_severe_toxicity_episodes: Some(10),
            ..PolicySafetyConfig::default()
        };

        report.check_safety_pass(&cfg);
        assert!(!report.safety_pass);
    }

    #[test]
    fn test_safety_pass_threshold_contracts() {
        let mut report = PolicySafetyReport::new(100);
        report.total_contract_violations = 75;

        let cfg = PolicySafetyConfig {
            max_total_contract_violations: Some(50),
            ..PolicySafetyConfig::default()
        };

        report.check_safety_pass(&cfg);
        assert!(!report.safety_pass);
    }

    #[test]
    fn test_push_sample_violation_cap() {
        let mut violations = Vec::new();

        // Add 150 violations
        for i in 0..150 {
            push_sample_violation(
                &mut violations,
                SafetyViolationKind::ContractViolation,
                i,
                0,
                format!("Violation {}", i),
            );
        }

        // Should be capped at 100
        assert_eq!(violations.len(), 100);
    }
}
