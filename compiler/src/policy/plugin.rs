//! Policy Plugin Abstraction
//!
//! Provides a common interface for any policy (MedLang expression-based, RL, external)
//! that can select actions given observations. This allows RL-trained policies to be
//! integrated into MedLang as first-class clinical policies.

use crate::rl::{RlAction, RlObservation};

/// Common interface for any policy that can select an action given an observation.
///
/// This trait enables:
/// - MedLang expression-based policies
/// - External RL policies (Python, ONNX, etc.)
/// - Hybrid policies combining multiple strategies
///
/// All policies are evaluated using the same clinical endpoints (ORR, PFS, DLT),
/// allowing direct comparison between hand-crafted and learned policies.
pub trait PolicyPlugin: Send + Sync {
    /// A short identifier for this policy type.
    ///
    /// Examples: "medlang_expr", "external_json", "onnx"
    fn kind(&self) -> &'static str;

    /// Given an observation, produce an action (e.g., dose adjustment factor).
    ///
    /// The policy should be stateless with respect to the RL environment -
    /// all necessary state should be contained in the observation.
    fn select_action(&mut self, obs: &RlObservation) -> RlAction;

    /// Optional: Return a human-readable name for this policy instance.
    fn name(&self) -> &str {
        "unnamed_policy"
    }

    /// Optional: Reset any internal state (useful for stateful policies).
    ///
    /// Called before each episode begins.
    fn reset(&mut self) {
        // Default: no-op for stateless policies
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trivial policy that always returns dose_factor = 1.0
    struct ConstantPolicy {
        dose_factor: f64,
    }

    impl PolicyPlugin for ConstantPolicy {
        fn kind(&self) -> &'static str {
            "constant"
        }

        fn select_action(&mut self, _obs: &RlObservation) -> RlAction {
            RlAction {
                dose_factor: self.dose_factor,
            }
        }

        fn name(&self) -> &str {
            "constant_1.0"
        }
    }

    #[test]
    fn test_constant_policy() {
        let mut policy = ConstantPolicy { dose_factor: 0.8 };

        let obs = RlObservation {
            tumour: 50.0,
            anc: Some(1.5),
            cycle: 0,
            time_days: 0.0,
            cumulative_dose_mg: 0.0,
            latent_state: vec![],
        };

        let action = policy.select_action(&obs);
        assert_eq!(action.dose_factor, 0.8);
        assert_eq!(policy.kind(), "constant");
        assert_eq!(policy.name(), "constant_1.0");
    }

    #[test]
    fn test_policy_reset() {
        let mut policy = ConstantPolicy { dose_factor: 1.0 };
        policy.reset(); // Should not panic
    }
}
