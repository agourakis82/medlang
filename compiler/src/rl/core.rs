// Week 31: Core RL Abstractions
//
// Defines the fundamental reinforcement learning types and the RLEnv trait
// for QSP-based RL environments.

use serde::{Deserialize, Serialize};

/// Discrete action index: 0..(n_actions-1)
pub type Action = usize;

/// State representation at runtime
///
/// Uses a simple vector representation that can be discretized or
/// used directly with function approximation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct State {
    /// Feature vector representing the current state
    ///
    /// For DoseToxEnv:
    /// - features[0]: ANC (normalized)
    /// - features[1]: Tumour size (normalized)
    /// - features[2]: Cycle index (normalized)
    /// - features[3]: Previous dose (normalized)
    pub features: Vec<f64>,
}

impl State {
    /// Create a new state from feature vector
    pub fn new(features: Vec<f64>) -> Self {
        Self { features }
    }

    /// Get number of features
    pub fn dim(&self) -> usize {
        self.features.len()
    }
}

/// Result of taking a step in an environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    /// Next state after taking the action
    pub next_state: State,

    /// Reward received for this transition
    pub reward: f64,

    /// Whether the episode is done
    pub done: bool,

    /// Additional information about the step
    pub info: StepInfo,
}

/// Additional information about an environment step
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StepInfo {
    /// Number of contract violations in this step
    pub contract_violations: usize,

    /// Efficacy component of reward (for logging/analysis)
    pub efficacy_reward: f64,

    /// Toxicity penalty component (for logging/analysis)
    pub toxicity_penalty: f64,

    /// Contract penalty component (for logging/analysis)
    pub contract_penalty: f64,
}

/// Reinforcement Learning Environment trait
///
/// This trait defines the interface that all RL environments must implement.
/// Environments wrap QSP models or surrogates and provide a Markov Decision Process
/// interface for policy learning.
pub trait RLEnv {
    /// Get the dimensionality of the state vector
    fn state_dim(&self) -> usize;

    /// Get the number of discrete actions
    fn num_actions(&self) -> usize;

    /// Reset to the start of a new episode
    ///
    /// This samples a new virtual patient, initializes state, and returns
    /// the initial state.
    fn reset(&mut self) -> anyhow::Result<State>;

    /// Advance one step with the given action
    ///
    /// Takes an action, simulates the environment dynamics (potentially using
    /// a QSP model or surrogate), and returns the next state, reward, and
    /// whether the episode is done.
    fn step(&mut self, action: Action) -> anyhow::Result<StepResult>;

    /// Get a human-readable name for the environment
    fn name(&self) -> &str {
        "RLEnv"
    }
}

/// Episode trajectory for logging and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    /// States visited in order
    pub states: Vec<State>,

    /// Actions taken
    pub actions: Vec<Action>,

    /// Rewards received
    pub rewards: Vec<f64>,

    /// Total return (sum of rewards)
    pub total_return: f64,

    /// Total contract violations
    pub total_violations: usize,

    /// Number of steps
    pub length: usize,
}

impl Episode {
    /// Create a new empty episode
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
            total_return: 0.0,
            total_violations: 0,
            length: 0,
        }
    }

    /// Add a step to the episode
    pub fn add_step(&mut self, state: State, action: Action, reward: f64, violations: usize) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
        self.total_return += reward;
        self.total_violations += violations;
        self.length += 1;
    }

    /// Get average reward per step
    pub fn avg_reward(&self) -> f64 {
        if self.length == 0 {
            0.0
        } else {
            self.total_return / self.length as f64
        }
    }
}

impl Default for Episode {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_state_creation() {
        let state = State::new(vec![0.5, 1.0, 0.3, 0.8]);
        assert_eq!(state.dim(), 4);
        assert_eq!(state.features[0], 0.5);
    }

    #[test]
    fn test_step_result() {
        let step = StepResult {
            next_state: State::new(vec![0.6, 0.9, 0.4, 0.8]),
            reward: 1.5,
            done: false,
            info: StepInfo {
                contract_violations: 0,
                efficacy_reward: 2.0,
                toxicity_penalty: 0.5,
                contract_penalty: 0.0,
            },
        };

        assert_eq!(step.reward, 1.5);
        assert!(!step.done);
        assert_eq!(step.info.contract_violations, 0);
    }

    #[test]
    fn test_episode_tracking() {
        let mut episode = Episode::new();

        episode.add_step(State::new(vec![1.0]), 0, 1.0, 0);
        episode.add_step(State::new(vec![2.0]), 1, 2.0, 0);
        episode.add_step(State::new(vec![3.0]), 0, -1.0, 1);

        assert_eq!(episode.length, 3);
        assert_eq!(episode.total_return, 2.0);
        assert_eq!(episode.total_violations, 1);
        assert_eq!(episode.avg_reward(), 2.0 / 3.0);
    }

    #[test]
    fn test_step_info_default() {
        let info = StepInfo::default();
        assert_eq!(info.contract_violations, 0);
        assert_eq!(info.efficacy_reward, 0.0);
        assert_eq!(info.toxicity_penalty, 0.0);
        assert_eq!(info.contract_penalty, 0.0);
    }
}
