// Week 31-32: Q-Learning Trainer for Tabular RL
//
// Implements Q-learning algorithm for discrete action spaces with
// state discretization for continuous state spaces.

use crate::rl::core::{Action, Episode, RLEnv, State, StepResult};
use crate::rl::discretizer::{BoxDiscretizer, StateDiscretizer};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Configuration for RL training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLTrainConfig {
    /// Number of episodes to train
    pub n_episodes: usize,

    /// Maximum steps per episode (safety limit)
    pub max_steps_per_episode: usize,

    /// Discount factor γ ∈ [0, 1]
    pub gamma: f64,

    /// Learning rate α ∈ (0, 1]
    pub alpha: f64,

    /// Initial exploration rate
    pub eps_start: f64,

    /// Final exploration rate
    pub eps_end: f64,
}

impl RLTrainConfig {
    /// Create a default configuration suitable for quick experiments
    pub fn default_quick() -> Self {
        Self {
            n_episodes: 100,
            max_steps_per_episode: 50,
            gamma: 0.95,
            alpha: 0.1,
            eps_start: 0.5,
            eps_end: 0.05,
        }
    }

    /// Create a production configuration for serious training
    pub fn default_production() -> Self {
        Self {
            n_episodes: 1000,
            max_steps_per_episode: 100,
            gamma: 0.99,
            alpha: 0.05,
            eps_start: 0.3,
            eps_end: 0.01,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.n_episodes == 0 {
            return Err("n_episodes must be > 0".to_string());
        }
        if self.max_steps_per_episode == 0 {
            return Err("max_steps_per_episode must be > 0".to_string());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.alpha) {
            return Err("alpha must be in (0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.eps_start) {
            return Err("eps_start must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.eps_end) {
            return Err("eps_end must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Training report with metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLTrainReport {
    /// Number of episodes trained
    pub n_episodes: usize,

    /// Average reward per episode
    pub avg_reward: f64,

    /// Final epsilon value
    pub final_epsilon: f64,

    /// Average episode length
    pub avg_episode_length: f64,

    /// Total training steps
    pub total_steps: usize,
}

/// Policy handle that can be serialized and reused
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLPolicyHandle {
    /// Number of discrete states
    pub n_states: usize,

    /// Number of actions
    pub n_actions: usize,

    /// Q-values: flattened [state_index * n_actions + action]
    pub q_values: Vec<f64>,

    /// Discretizer metadata for state conversion
    pub bins_per_dim: Vec<usize>,
    pub min_vals: Vec<f64>,
    pub max_vals: Vec<f64>,
}

impl RLPolicyHandle {
    /// Get best action for a state (greedy policy)
    pub fn best_action(&self, state_index: usize) -> Action {
        let mut best_action = 0;
        let mut best_q = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let q = self.q_values[state_index * self.n_actions + action];
            if q > best_q {
                best_q = q;
                best_action = action;
            }
        }

        best_action
    }

    /// Get Q-value for state-action pair
    pub fn q_value(&self, state_index: usize, action: Action) -> f64 {
        self.q_values[state_index * self.n_actions + action]
    }
}

/// Q-table agent for tabular Q-learning
pub struct QTableAgent {
    /// Q-table: flattened [state * n_actions + action]
    q: Vec<f64>,

    /// Current epsilon (exploration rate)
    pub epsilon: f64,

    /// Training configuration
    cfg: RLTrainConfig,

    /// Number of states
    n_states: usize,

    /// Number of actions
    n_actions: usize,
}

impl QTableAgent {
    /// Create a new Q-table agent
    pub fn new(n_states: usize, n_actions: usize, cfg: RLTrainConfig) -> Self {
        Self {
            q: vec![0.0; n_states * n_actions],
            epsilon: cfg.eps_start,
            cfg,
            n_states,
            n_actions,
        }
    }

    /// Get index into Q-table
    fn q_index(&self, state_idx: usize, action: Action) -> usize {
        state_idx * self.n_actions + action
    }

    /// Select action using ε-greedy policy
    pub fn select_action<D: StateDiscretizer>(
        &self,
        state: &State,
        disc: &D,
        rng: &mut impl Rng,
    ) -> Action {
        // Exploration: random action
        if rng.gen::<f64>() < self.epsilon {
            return rng.gen_range(0..self.n_actions);
        }

        // Exploitation: best action according to Q-values
        let state_idx = disc.state_index(state);
        self.best_action(state_idx)
    }

    /// Get best action for a state (greedy)
    fn best_action(&self, state_idx: usize) -> Action {
        let mut best_action = 0;
        let mut best_q = f64::NEG_INFINITY;

        for action in 0..self.n_actions {
            let q = self.q[self.q_index(state_idx, action)];
            if q > best_q {
                best_q = q;
                best_action = action;
            }
        }

        best_action
    }

    /// Update Q-value using Q-learning rule
    pub fn update<D: StateDiscretizer>(
        &mut self,
        state: &State,
        action: Action,
        reward: f64,
        next_state: &State,
        disc: &D,
    ) {
        let state_idx = disc.state_index(state);
        let next_state_idx = disc.state_index(next_state);

        // Find max Q-value for next state
        let mut max_next_q = f64::NEG_INFINITY;
        for next_action in 0..self.n_actions {
            let q = self.q[self.q_index(next_state_idx, next_action)];
            if q > max_next_q {
                max_next_q = q;
            }
        }

        // Q-learning update: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        let q_idx = self.q_index(state_idx, action);
        let current_q = self.q[q_idx];
        let target = reward + self.cfg.gamma * max_next_q;
        self.q[q_idx] = current_q + self.cfg.alpha * (target - current_q);
    }

    /// Update epsilon (linear decay)
    pub fn update_epsilon(&mut self, episode_idx: usize) {
        let progress = episode_idx as f64 / self.cfg.n_episodes.max(1) as f64;
        self.epsilon = self.cfg.eps_start + progress * (self.cfg.eps_end - self.cfg.eps_start);
    }

    /// Extract policy handle
    pub fn freeze_policy(&self, disc: &BoxDiscretizer) -> RLPolicyHandle {
        RLPolicyHandle {
            n_states: self.n_states,
            n_actions: self.n_actions,
            q_values: self.q.clone(),
            bins_per_dim: disc.bins_per_dim.clone(),
            min_vals: disc.min_vals.clone(),
            max_vals: disc.max_vals.clone(),
        }
    }
}

/// Train a Q-learning agent on an environment
pub fn train_q_learning<E: RLEnv>(
    env: &mut E,
    disc: &BoxDiscretizer,
    cfg: &RLTrainConfig,
    rng: &mut impl Rng,
) -> anyhow::Result<(RLTrainReport, RLPolicyHandle)> {
    // Validate config
    cfg.validate()
        .map_err(|e| anyhow::anyhow!("Invalid config: {}", e))?;

    let n_states = disc.num_states();
    let n_actions = env.num_actions();
    let mut agent = QTableAgent::new(n_states, n_actions, cfg.clone());

    let mut total_reward = 0.0;
    let mut total_steps = 0usize;

    for episode_idx in 0..cfg.n_episodes {
        let mut state = env.reset()?;
        let mut episode_reward = 0.0;
        let mut episode_steps = 0;

        for _step in 0..cfg.max_steps_per_episode {
            // Select action
            let action = agent.select_action(&state, disc, rng);

            // Take step
            let step_result = env.step(action)?;

            // Update Q-values
            agent.update(
                &state,
                action,
                step_result.reward,
                &step_result.next_state,
                disc,
            );

            // Update metrics
            episode_reward += step_result.reward;
            episode_steps += 1;
            total_steps += 1;

            // Move to next state
            state = step_result.next_state;

            if step_result.done {
                break;
            }
        }

        total_reward += episode_reward;
        agent.update_epsilon(episode_idx);
    }

    let avg_reward = total_reward / cfg.n_episodes as f64;
    let avg_episode_length = total_steps as f64 / cfg.n_episodes as f64;

    let policy = agent.freeze_policy(disc);

    let report = RLTrainReport {
        n_episodes: cfg.n_episodes,
        avg_reward,
        final_epsilon: agent.epsilon,
        avg_episode_length,
        total_steps,
    };

    Ok((report, policy))
}

/// Evaluate a learned policy
pub fn evaluate_policy<E: RLEnv>(
    env: &mut E,
    policy: &RLPolicyHandle,
    disc: &BoxDiscretizer,
    n_episodes: usize,
) -> anyhow::Result<PolicyEvalReport> {
    let mut total_reward = 0.0;
    let mut total_violations = 0usize;
    let mut total_steps = 0usize;

    for _ in 0..n_episodes {
        let mut state = env.reset()?;
        let mut episode_reward = 0.0;
        let mut episode_violations = 0;

        for _ in 0..1000 {
            // Safety limit
            // Greedy action selection
            let state_idx = disc.state_index(&state);
            let action = policy.best_action(state_idx);

            let step_result = env.step(action)?;

            episode_reward += step_result.reward;
            episode_violations += step_result.info.contract_violations;
            total_steps += 1;

            state = step_result.next_state;

            if step_result.done {
                break;
            }
        }

        total_reward += episode_reward;
        total_violations += episode_violations;
    }

    Ok(PolicyEvalReport {
        n_episodes,
        avg_reward: total_reward / n_episodes as f64,
        avg_contract_violations: total_violations as f64 / n_episodes as f64,
        avg_episode_length: total_steps as f64 / n_episodes as f64,
    })
}

/// Policy evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEvalReport {
    pub n_episodes: usize,
    pub avg_reward: f64,
    pub avg_contract_violations: f64,
    pub avg_episode_length: f64,
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::core::RLEnv;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;

    // Simple toy environment for testing
    struct ToyEnv {
        state: i32,
        max_steps: usize,
        current_step: usize,
    }

    impl RLEnv for ToyEnv {
        fn state_dim(&self) -> usize {
            1
        }

        fn num_actions(&self) -> usize {
            2
        }

        fn reset(&mut self) -> anyhow::Result<State> {
            self.state = 0;
            self.current_step = 0;
            Ok(State::new(vec![self.state as f64]))
        }

        fn step(&mut self, action: Action) -> anyhow::Result<StepResult> {
            self.current_step += 1;

            // Action 1 is always better (+1 reward), action 0 gives 0 reward
            let reward = if action == 1 { 1.0 } else { 0.0 };

            let done = self.current_step >= self.max_steps;

            Ok(StepResult {
                next_state: State::new(vec![self.state as f64]),
                reward,
                done,
                info: Default::default(),
            })
        }
    }

    #[test]
    fn test_config_validation() {
        let valid = RLTrainConfig::default_quick();
        assert!(valid.validate().is_ok());

        let invalid = RLTrainConfig {
            n_episodes: 0,
            ..RLTrainConfig::default_quick()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_q_table_agent_creation() {
        let cfg = RLTrainConfig::default_quick();
        let agent = QTableAgent::new(10, 5, cfg);

        assert_eq!(agent.q.len(), 50);
        assert_eq!(agent.n_states, 10);
        assert_eq!(agent.n_actions, 5);
    }

    #[test]
    fn test_epsilon_decay() {
        let cfg = RLTrainConfig {
            n_episodes: 100,
            eps_start: 1.0,
            eps_end: 0.0,
            ..RLTrainConfig::default_quick()
        };
        let mut agent = QTableAgent::new(10, 2, cfg);

        assert_eq!(agent.epsilon, 1.0);

        agent.update_epsilon(50);
        assert!((agent.epsilon - 0.5).abs() < 0.01);

        agent.update_epsilon(99);
        assert!(agent.epsilon < 0.01);
    }

    #[test]
    fn test_toy_env_learning() {
        let mut env = ToyEnv {
            state: 0,
            max_steps: 5,
            current_step: 0,
        };

        let disc = BoxDiscretizer::new(vec![2], vec![0.0], vec![1.0]);

        let cfg = RLTrainConfig {
            n_episodes: 100,
            max_steps_per_episode: 10,
            gamma: 0.9,
            alpha: 0.5,
            eps_start: 1.0,
            eps_end: 0.0,
        };

        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let (report, policy) = train_q_learning(&mut env, &disc, &cfg, &mut rng).unwrap();

        // After training, agent should learn to always take action 1
        assert!(report.avg_reward > 0.5);

        // Check that Q-values prefer action 1
        let state_idx = 0;
        let q0 = policy.q_value(state_idx, 0);
        let q1 = policy.q_value(state_idx, 1);
        assert!(q1 > q0);
    }

    #[test]
    fn test_policy_handle_best_action() {
        let policy = RLPolicyHandle {
            n_states: 2,
            n_actions: 3,
            q_values: vec![
                1.0, 2.0, 0.5, // State 0: action 1 is best
                0.1, 0.2, 3.0, // State 1: action 2 is best
            ],
            bins_per_dim: vec![2],
            min_vals: vec![0.0],
            max_vals: vec![1.0],
        };

        assert_eq!(policy.best_action(0), 1);
        assert_eq!(policy.best_action(1), 2);
    }

    #[test]
    fn test_train_report_metrics() {
        let mut env = ToyEnv {
            state: 0,
            max_steps: 3,
            current_step: 0,
        };

        let disc = BoxDiscretizer::new(vec![2], vec![0.0], vec![1.0]);
        let cfg = RLTrainConfig::default_quick();
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        let (report, _) = train_q_learning(&mut env, &disc, &cfg, &mut rng).unwrap();

        assert_eq!(report.n_episodes, cfg.n_episodes);
        assert!(report.avg_reward >= 0.0);
        assert!(report.avg_episode_length > 0.0);
        assert!(report.avg_episode_length <= cfg.max_steps_per_episode as f64);
    }
}
