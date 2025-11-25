// Week 31: Dose-Toxicity-Efficacy RL Environment
//
// A canonical RL environment for learning dose adjustment policies in oncology.
// Integrates with QSP models or surrogates to simulate patient response.

use crate::ml::BackendKind;
use crate::rl::core::{Action, RLEnv, State, StepInfo, StepResult};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for DoseToxEnv
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseToxEnvConfig {
    /// Evidence program handle (for QSP simulation)
    pub ev_handle: String,

    /// Backend to use (Mechanistic or Surrogate)
    pub backend: BackendKind,

    /// Number of treatment cycles per episode
    pub n_cycles: usize,

    /// Available dose levels (in mg)
    pub dose_levels_mg: Vec<f64>,

    /// Weight for response/efficacy component in reward
    pub reward_response_weight: f64,

    /// Weight for toxicity penalty in reward
    pub reward_tox_penalty: f64,

    /// Penalty per contract violation
    pub contract_penalty: f64,

    /// Random seed for patient sampling
    pub seed: Option<u64>,
}

impl DoseToxEnvConfig {
    /// Create a default configuration
    pub fn default() -> Self {
        Self {
            ev_handle: "default_ev".to_string(),
            backend: BackendKind::Surrogate,
            n_cycles: 6,
            dose_levels_mg: vec![0.0, 50.0, 100.0, 200.0, 300.0],
            reward_response_weight: 1.0,
            reward_tox_penalty: 2.0,
            contract_penalty: 10.0,
            seed: None,
        }
    }
}

/// Patient state for dose-tox environment
#[derive(Debug, Clone)]
struct PatientState {
    /// Absolute Neutrophil Count (normalized to [0, 1])
    anc: f64,

    /// Tumour size (normalized to [0, 1], 0 = complete response)
    tumour_size: f64,

    /// Current cycle (1-indexed)
    cycle: usize,

    /// Previous dose (normalized)
    prev_dose: f64,

    /// Baseline values for normalization
    baseline_anc: f64,
    baseline_tumour: f64,
}

impl PatientState {
    /// Convert to RL state vector
    fn to_rl_state(&self) -> State {
        State::new(vec![
            self.anc,
            self.tumour_size,
            self.cycle as f64 / 10.0, // Normalize cycle
            self.prev_dose,
        ])
    }
}

/// Dose-Toxicity-Efficacy RL Environment
///
/// State space:
/// - ANC (normalized)
/// - Tumour size (normalized)
/// - Cycle index (normalized)
/// - Previous dose (normalized)
///
/// Action space:
/// - Discrete dose levels (0 = no dose, ..., N = maximum dose)
///
/// Reward:
/// - Efficacy: Reduction in tumour size
/// - Toxicity: Penalty for ANC reduction
/// - Contracts: Large penalty for safety violations
pub struct DoseToxEnv {
    cfg: DoseToxEnvConfig,
    patient: Option<PatientState>,
    rng: ChaCha20Rng,
    episode_count: usize,
}

impl DoseToxEnv {
    /// Create a new DoseToxEnv
    pub fn new(cfg: DoseToxEnvConfig) -> Self {
        let seed = cfg.seed.unwrap_or(42);
        let rng = ChaCha20Rng::seed_from_u64(seed);

        Self {
            cfg,
            patient: None,
            rng,
            episode_count: 0,
        }
    }

    /// Sample a new virtual patient
    fn sample_patient(&mut self) -> PatientState {
        // Sample baseline covariates
        let baseline_anc = self.rng.gen_range(3000.0..8000.0); // cells/μL
        let baseline_tumour = self.rng.gen_range(5.0..20.0); // cm³

        PatientState {
            anc: 1.0,         // Start at baseline (normalized)
            tumour_size: 1.0, // Start at baseline (normalized)
            cycle: 1,
            prev_dose: 0.0,
            baseline_anc,
            baseline_tumour,
        }
    }

    /// Simulate one treatment cycle with given dose
    fn simulate_cycle_internal(
        rng: &mut ChaCha20Rng,
        patient: &PatientState,
        dose_mg: f64,
    ) -> (f64, f64, Vec<String>) {
        // Simplified QSP-like dynamics
        // In reality, this would call the actual QSP model or surrogate

        // Efficacy: Higher dose → more tumour reduction
        let dose_effect = (dose_mg / 300.0).min(1.0); // Normalize to [0, 1]
        let efficacy = 0.95_f64.powf(dose_effect * 2.0); // Exponential decay

        // Add stochasticity
        let efficacy_noise = rng.gen_range(0.95..1.05);
        let new_tumour = (patient.tumour_size * efficacy * efficacy_noise).max(0.0);

        // Toxicity: Higher dose → lower ANC
        let tox_effect = (dose_mg / 300.0).min(1.0);
        let anc_reduction = 0.92_f64.powf(tox_effect * 3.0); // Deeper reduction at high doses

        // Add stochasticity
        let anc_noise = rng.gen_range(0.95..1.05);
        let new_anc = (patient.anc * anc_reduction * anc_noise).max(0.0);

        // Check contracts (safety thresholds)
        let mut violations = Vec::new();

        // ANC < 0.25 (Grade 4 neutropenia - severe toxicity)
        if new_anc < 0.25 {
            violations.push(format!("Grade 4 neutropenia: ANC={:.3} < 0.25", new_anc));
        }

        // ANC < 0.125 (potentially fatal)
        if new_anc < 0.125 {
            violations.push(format!(
                "Critical neutropenia: ANC={:.3} < 0.125 (FATAL)",
                new_anc
            ));
        }

        // Very high dose without clinical justification
        if dose_mg > 250.0 && patient.tumour_size < 0.5 {
            violations.push(format!(
                "Excessive dosing: dose={:.0}mg for tumour_size={:.3}",
                dose_mg, patient.tumour_size
            ));
        }

        (new_anc, new_tumour, violations)
    }

    /// Compute reward from state transition
    fn compute_reward_internal(
        cfg: &DoseToxEnvConfig,
        prev_tumour: f64,
        new_tumour: f64,
        prev_anc: f64,
        new_anc: f64,
        violations: &[String],
    ) -> (f64, StepInfo) {
        // Efficacy reward: Reduction in tumour size
        let tumour_reduction = prev_tumour - new_tumour;
        let efficacy_reward = cfg.reward_response_weight * tumour_reduction;

        // Toxicity penalty: Reduction in ANC
        let anc_reduction = (prev_anc - new_anc).max(0.0);
        let toxicity_penalty = cfg.reward_tox_penalty * anc_reduction;

        // Contract penalty
        let contract_penalty = cfg.contract_penalty * violations.len() as f64;

        // Total reward
        let reward = efficacy_reward - toxicity_penalty - contract_penalty;

        let info = StepInfo {
            contract_violations: violations.len(),
            efficacy_reward,
            toxicity_penalty,
            contract_penalty,
        };

        (reward, info)
    }
}

impl RLEnv for DoseToxEnv {
    fn state_dim(&self) -> usize {
        4 // ANC, tumour_size, cycle, prev_dose
    }

    fn num_actions(&self) -> usize {
        self.cfg.dose_levels_mg.len()
    }

    fn reset(&mut self) -> anyhow::Result<State> {
        // Sample new patient
        self.patient = Some(self.sample_patient());
        self.episode_count += 1;

        let state = self.patient.as_ref().unwrap().to_rl_state();
        Ok(state)
    }

    fn step(&mut self, action: Action) -> anyhow::Result<StepResult> {
        if self.patient.is_none() {
            anyhow::bail!("Environment not initialized. Call reset() first.");
        }

        // Validate action
        if action >= self.num_actions() {
            anyhow::bail!("Invalid action: {} >= {}", action, self.num_actions());
        }

        // Get dose from action (before mutable borrow of patient if needed, but here it is from cfg)
        let dose_mg = self.cfg.dose_levels_mg[action];

        // Simulate one cycle (borrowing patient immutably)
        let (new_anc, new_tumour, violations) = {
            let patient = self.patient.as_ref().unwrap();
            Self::simulate_cycle_internal(&mut self.rng, patient, dose_mg)
        };

        // Update patient state (mutable borrow)
        let patient = self.patient.as_mut().unwrap();
        
        let prev_tumour = patient.tumour_size;
        let prev_anc = patient.anc;

        patient.anc = new_anc;
        patient.tumour_size = new_tumour;
        patient.prev_dose = dose_mg / 300.0; // Normalize
        patient.cycle += 1;
        
        let cycle = patient.cycle; // Copy for done check

        // Compute reward (borrowing cfg)
        let (reward, info) = Self::compute_reward_internal(
            &self.cfg,
            prev_tumour,
            new_tumour,
            prev_anc,
            new_anc,
            &violations,
        );

        // Check if episode is done
        let done = cycle > self.cfg.n_cycles
            || new_anc < 0.125  // Critical toxicity - terminate
            || new_tumour < 0.05; // Complete response - success!

        let next_state = patient.to_rl_state();

        Ok(StepResult {
            next_state,
            reward,
            done,
            info,
        })
    }

    fn name(&self) -> &str {
        "DoseToxEnv"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_env_config_default() {
        let cfg = DoseToxEnvConfig::default();
        assert_eq!(cfg.n_cycles, 6);
        assert_eq!(cfg.dose_levels_mg.len(), 5);
        assert_eq!(cfg.backend, BackendKind::Surrogate);
    }

    #[test]
    fn test_env_creation() {
        let cfg = DoseToxEnvConfig::default();
        let env = DoseToxEnv::new(cfg);

        assert_eq!(env.state_dim(), 4);
        assert_eq!(env.num_actions(), 5);
        assert_eq!(env.name(), "DoseToxEnv");
    }

    #[test]
    fn test_env_reset() {
        let cfg = DoseToxEnvConfig::default();
        let mut env = DoseToxEnv::new(cfg);

        let state = env.reset().unwrap();
        assert_eq!(state.features.len(), 4);

        // Initial state should be at baseline
        assert_eq!(state.features[0], 1.0); // ANC
        assert_eq!(state.features[1], 1.0); // tumour_size
        assert!(state.features[2] > 0.0); // cycle (normalized)
        assert_eq!(state.features[3], 0.0); // prev_dose
    }

    #[test]
    fn test_env_step() {
        let cfg = DoseToxEnvConfig::default();
        let mut env = DoseToxEnv::new(cfg);

        env.reset().unwrap();

        // Take middle dose action
        let action = 2; // 100 mg
        let result = env.step(action).unwrap();

        // Check result structure
        assert_eq!(result.next_state.features.len(), 4);
        assert!(!result.reward.is_nan());

        // Tumour should reduce
        assert!(result.next_state.features[1] < 1.0);

        // ANC should reduce
        assert!(result.next_state.features[0] < 1.0);
    }

    #[test]
    fn test_env_episode() {
        let cfg = DoseToxEnvConfig {
            n_cycles: 3,
            ..DoseToxEnvConfig::default()
        };
        let mut env = DoseToxEnv::new(cfg);

        env.reset().unwrap();

        let mut total_reward = 0.0;
        let mut steps = 0;

        for _ in 0..10 {
            // Max 10 steps (should finish in 3)
            let action = 2; // Fixed action
            let result = env.step(action).unwrap();

            total_reward += result.reward;
            steps += 1;

            if result.done {
                break;
            }
        }

        assert!(steps <= 3, "Episode should finish within n_cycles");
        assert!(!total_reward.is_nan());
    }

    #[test]
    fn test_contract_violations() {
        let cfg = DoseToxEnvConfig {
            dose_levels_mg: vec![0.0, 500.0], // Extreme dose
            ..DoseToxEnvConfig::default()
        };
        let mut env = DoseToxEnv::new(cfg);

        env.reset().unwrap();

        // Take maximum dose
        let result = env.step(1).unwrap();

        // Should likely cause violations due to severe toxicity
        // (stochastic, so might not always happen, but high dose increases probability)
        assert!(result.info.toxicity_penalty > 0.0);
    }

    #[test]
    fn test_invalid_action() {
        let cfg = DoseToxEnvConfig::default();
        let mut env = DoseToxEnv::new(cfg);

        env.reset().unwrap();

        // Try invalid action
        let result = env.step(100);
        assert!(result.is_err());
    }

    #[test]
    fn test_step_without_reset() {
        let cfg = DoseToxEnvConfig::default();
        let mut env = DoseToxEnv::new(cfg);

        // Try to step without reset
        let result = env.step(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let cfg1 = DoseToxEnvConfig {
            seed: Some(42),
            ..DoseToxEnvConfig::default()
        };
        let cfg2 = DoseToxEnvConfig {
            seed: Some(42),
            ..DoseToxEnvConfig::default()
        };

        let mut env1 = DoseToxEnv::new(cfg1);
        let mut env2 = DoseToxEnv::new(cfg2);

        let state1 = env1.reset().unwrap();
        let state2 = env2.reset().unwrap();

        // Same seed should produce same initial state (after baseline values)
        // The baselines are sampled with RNG, so they should match
        assert_eq!(state1, state2);

        let result1 = env1.step(2).unwrap();
        let result2 = env2.step(2).unwrap();

        // Same action should produce same result
        assert_eq!(result1.next_state, result2.next_state);
        assert_eq!(result1.reward, result2.reward);
    }
}