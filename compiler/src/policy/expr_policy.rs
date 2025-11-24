//! MedLang Expression-Based Policy
//!
//! Implements PolicyPlugin for policies defined as MedLang expressions.
//! These are hand-crafted, interpretable policies that use clinical variables
//! (tumor size, ANC, cycle, cumulative dose) to compute dose adjustments.

use crate::policy::plugin::PolicyPlugin;
use crate::rl::{RlAction, RlObservation};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A policy defined by a MedLang expression that computes dose_factor.
///
/// Variables available in the expression:
/// - Tumour: current tumor volume (mm³)
/// - ANC: absolute neutrophil count (if available)
/// - Cycle: current treatment cycle (0-indexed)
/// - Time: days since treatment start
/// - CumDose: cumulative dose administered (mg)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExprPolicy {
    pub name: String,
    pub expression: String,
}

impl PolicyPlugin for ExprPolicy {
    fn kind(&self) -> &'static str {
        "medlang_expr"
    }

    fn select_action(&mut self, obs: &RlObservation) -> RlAction {
        // Build variable environment from observation
        let env = build_env_from_observation(obs);

        // Evaluate expression (scaffold implementation)
        let dose_factor = eval_dose_expression(&self.expression, &env);

        RlAction { dose_factor }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Build variable environment from RL observation.
///
/// Maps observation fields to variables that can be used in MedLang expressions.
fn build_env_from_observation(obs: &RlObservation) -> HashMap<String, f64> {
    let mut env = HashMap::new();

    env.insert("Tumour".to_string(), obs.tumour);
    env.insert("Cycle".to_string(), obs.cycle as f64);
    env.insert("Time".to_string(), obs.time_days);
    env.insert("CumDose".to_string(), obs.cumulative_dose_mg);

    if let Some(anc) = obs.anc {
        env.insert("ANC".to_string(), anc);
    }

    env
}

/// Evaluate a dose expression given a variable environment.
///
/// Week 22 scaffold: Simple expression evaluation for common patterns.
/// Full implementation would integrate with MedLang IR expression evaluator.
fn eval_dose_expression(expr: &str, env: &HashMap<String, f64>) -> f64 {
    let expr = expr.trim();

    // Handle constant expressions
    if let Ok(val) = expr.parse::<f64>() {
        return val.max(0.0).min(2.0); // Clamp to reasonable range
    }

    // Handle simple variable lookups
    if let Some(&val) = env.get(expr) {
        return val.max(0.0).min(2.0);
    }

    // Common patterns for Week 22 scaffold
    if expr.contains("if") && expr.contains("then") && expr.contains("else") {
        return eval_conditional(expr, env);
    }

    // Default: full dose
    1.0
}

/// Evaluate simple conditional expressions.
///
/// Supports patterns like: "if ANC < 1.0 then 0.5 else 1.0"
fn eval_conditional(expr: &str, env: &HashMap<String, f64>) -> f64 {
    // Very simple parser for Week 22 scaffold
    // Full implementation would use proper MedLang expression AST

    if let Some(if_pos) = expr.find("if") {
        if let Some(then_pos) = expr.find("then") {
            if let Some(else_pos) = expr.find("else") {
                let condition = expr[if_pos + 2..then_pos].trim();
                let then_val = expr[then_pos + 4..else_pos].trim();
                let else_val = expr[else_pos + 4..].trim();

                let cond_result = eval_condition(condition, env);

                if cond_result {
                    eval_dose_expression(then_val, env)
                } else {
                    eval_dose_expression(else_val, env)
                }
            } else {
                1.0
            }
        } else {
            1.0
        }
    } else {
        1.0
    }
}

/// Evaluate simple comparison conditions.
///
/// Supports: "VAR < VALUE", "VAR > VALUE", "VAR <= VALUE", "VAR >= VALUE"
fn eval_condition(cond: &str, env: &HashMap<String, f64>) -> bool {
    let cond = cond.trim();

    for op in &["<=", ">=", "<", ">", "=="] {
        if let Some(pos) = cond.find(op) {
            let left = cond[..pos].trim();
            let right = cond[pos + op.len()..].trim();

            let left_val = env.get(left).copied().unwrap_or(0.0);
            let right_val = right.parse::<f64>().unwrap_or(0.0);

            return match *op {
                "<" => left_val < right_val,
                ">" => left_val > right_val,
                "<=" => left_val <= right_val,
                ">=" => left_val >= right_val,
                "==" => (left_val - right_val).abs() < 1e-6,
                _ => false,
            };
        }
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_policy() {
        let mut policy = ExprPolicy {
            name: "full_dose".to_string(),
            expression: "1.0".to_string(),
        };

        let obs = RlObservation {
            tumour: 50.0,
            anc: Some(1.5),
            cycle: 0,
            time_days: 0.0,
            cumulative_dose_mg: 0.0,
            latent_state: vec![],
        };

        let action = policy.select_action(&obs);
        assert_eq!(action.dose_factor, 1.0);
    }

    #[test]
    fn test_variable_policy() {
        let mut env = HashMap::new();
        env.insert("Tumour".to_string(), 75.0);

        let dose = eval_dose_expression("Tumour", &env);
        assert_eq!(dose, 2.0); // Clamped to max
    }

    #[test]
    fn test_conditional_policy_anc_low() {
        let mut policy = ExprPolicy {
            name: "anc_based".to_string(),
            expression: "if ANC < 1.0 then 0.5 else 1.0".to_string(),
        };

        let obs = RlObservation {
            tumour: 50.0,
            anc: Some(0.8), // Low ANC
            cycle: 2,
            time_days: 42.0,
            cumulative_dose_mg: 600.0,
            latent_state: vec![],
        };

        let action = policy.select_action(&obs);
        assert_eq!(action.dose_factor, 0.5);
    }

    #[test]
    fn test_conditional_policy_anc_normal() {
        let mut policy = ExprPolicy {
            name: "anc_based".to_string(),
            expression: "if ANC < 1.0 then 0.5 else 1.0".to_string(),
        };

        let obs = RlObservation {
            tumour: 50.0,
            anc: Some(1.5), // Normal ANC
            cycle: 2,
            time_days: 42.0,
            cumulative_dose_mg: 600.0,
            latent_state: vec![],
        };

        let action = policy.select_action(&obs);
        assert_eq!(action.dose_factor, 1.0);
    }

    #[test]
    fn test_build_env() {
        let obs = RlObservation {
            tumour: 50.0,
            anc: Some(1.5),
            cycle: 3,
            time_days: 63.0,
            cumulative_dose_mg: 900.0,
            latent_state: vec![],
        };

        let env = build_env_from_observation(&obs);

        assert_eq!(env.get("Tumour"), Some(&50.0));
        assert_eq!(env.get("ANC"), Some(&1.5));
        assert_eq!(env.get("Cycle"), Some(&3.0));
        assert_eq!(env.get("Time"), Some(&63.0));
        assert_eq!(env.get("CumDose"), Some(&900.0));
    }

    #[test]
    fn test_eval_condition() {
        let mut env = HashMap::new();
        env.insert("ANC".to_string(), 0.8);

        assert!(eval_condition("ANC < 1.0", &env));
        assert!(!eval_condition("ANC > 1.0", &env));
        assert!(eval_condition("ANC <= 0.8", &env));
        assert!(!eval_condition("ANC >= 1.0", &env));
    }
}
