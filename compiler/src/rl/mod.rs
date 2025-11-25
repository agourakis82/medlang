// Week 31: Reinforcement Learning for QSP-based Policy Learning
//
// This module provides RL infrastructure for learning dosing and scheduling
// policies on top of QSP models and surrogates, with contract-aware safety.

pub mod core;
pub mod discretizer;
pub mod env_dose_tox;
pub mod train;

pub use core::{Action, Episode, RLEnv, RlAction, RlObservation, State, StepInfo, StepResult};
pub use discretizer::{BoxDiscretizer, StateDiscretizer};
pub use env_dose_tox::{DoseToxEnv, DoseToxEnvConfig};
pub use train::{
    evaluate_policy, train_q_learning, PolicyEvalReport, RLPolicyHandle, RLTrainConfig,
    RLTrainReport,
};
