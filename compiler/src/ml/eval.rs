// Week 30: Surrogate Evaluation Engine
//
// Implements quantitative evaluation of surrogate models against mechanistic
// references, including error metrics and contract violation tracking.

use crate::ml::{BackendKind, SurrogateModelHandle};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// Configuration and Report Types
// =============================================================================

/// Configuration for surrogate evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateEvalConfig {
    /// Number of independent evaluation scenarios
    pub n_eval: usize,
    /// Backend to use for reference (usually Mechanistic)
    pub backend_ref: BackendKind,
    /// Random seed for reproducibility
    pub seed: u64,
}

impl SurrogateEvalConfig {
    /// Create a default quick evaluation config
    pub fn default_quick() -> Self {
        Self {
            n_eval: 50,
            backend_ref: BackendKind::Mechanistic,
            seed: 42,
        }
    }

    /// Create a default production evaluation config
    pub fn default_production() -> Self {
        Self {
            n_eval: 500,
            backend_ref: BackendKind::Mechanistic,
            seed: 1234,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), SurrogateEvalError> {
        if self.n_eval == 0 {
            return Err(SurrogateEvalError::InvalidConfig(
                "n_eval must be greater than 0".to_string(),
            ));
        }

        // Backend must support mechanistic execution for reference
        if !self.backend_ref.requires_mechanistic() {
            return Err(SurrogateEvalError::InvalidConfig(
                "backend_ref must support mechanistic execution (use Mechanistic or Hybrid)"
                    .to_string(),
            ));
        }

        Ok(())
    }
}

/// Surrogate evaluation report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateEvalReport {
    /// Number of evaluation scenarios actually run
    pub n_eval: usize,
    /// Root mean squared error
    pub rmse: f64,
    /// Mean absolute error
    pub mae: f64,
    /// Maximum absolute error
    pub max_abs_err: f64,
    /// Contract violations under mechanistic backend
    pub mech_contract_violations: usize,
    /// Contract violations under surrogate backend
    pub surr_contract_violations: usize,
}

impl SurrogateEvalReport {
    /// Check if this report indicates an acceptable surrogate
    pub fn is_acceptable(&self, max_rmse: f64, max_mae: f64, max_abs_err: f64) -> bool {
        self.rmse <= max_rmse
            && self.mae <= max_mae
            && self.max_abs_err <= max_abs_err
            && self.surr_contract_violations == 0
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "n_eval: {}, RMSE: {:.6}, MAE: {:.6}, max_abs_err: {:.6}, mech_viols: {}, surr_viols: {}",
            self.n_eval,
            self.rmse,
            self.mae,
            self.max_abs_err,
            self.mech_contract_violations,
            self.surr_contract_violations
        )
    }
}

// =============================================================================
// Error Types
// =============================================================================

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SurrogateEvalError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("mechanistic execution failed: {0}")]
    MechanisticFailed(String),

    #[error("surrogate execution failed: {0}")]
    SurrogateFailed(String),

    #[error("internal error: {0}")]
    Internal(String),

    #[error("output shape mismatch: mechanistic produced {mech_len} outputs, surrogate produced {surr_len}")]
    OutputShapeMismatch { mech_len: usize, surr_len: usize },
}

// =============================================================================
// Internal Accumulator
// =============================================================================

#[derive(Debug, Default)]
struct MetricsAccum {
    sum_sq_err: f64,
    sum_abs_err: f64,
    max_abs_err: f64,
    count: usize,
}

impl MetricsAccum {
    fn accumulate(
        &mut self,
        ref_outputs: &[f64],
        surr_outputs: &[f64],
    ) -> Result<(), SurrogateEvalError> {
        if ref_outputs.len() != surr_outputs.len() {
            return Err(SurrogateEvalError::OutputShapeMismatch {
                mech_len: ref_outputs.len(),
                surr_len: surr_outputs.len(),
            });
        }

        for (ref_val, surr_val) in ref_outputs.iter().zip(surr_outputs.iter()) {
            let err = surr_val - ref_val;
            let abs_err = err.abs();

            self.sum_sq_err += err * err;
            self.sum_abs_err += abs_err;
            self.max_abs_err = self.max_abs_err.max(abs_err);
            self.count += 1;
        }

        Ok(())
    }

    fn finalize(&self) -> Result<(f64, f64, f64), SurrogateEvalError> {
        if self.count == 0 {
            return Err(SurrogateEvalError::Internal(
                "no outputs accumulated".to_string(),
            ));
        }

        let rmse = (self.sum_sq_err / self.count as f64).sqrt();
        let mae = self.sum_abs_err / self.count as f64;
        let max_abs_err = self.max_abs_err;

        Ok((rmse, mae, max_abs_err))
    }
}

// =============================================================================
// Evaluation Scenario
// =============================================================================

/// Represents a single evaluation scenario (inputs to the model)
#[derive(Debug, Clone)]
pub struct EvalScenario {
    /// Subject covariates (age, weight, etc.)
    pub covariates: HashMap<String, f64>,
    /// Design parameters (dose, timing, etc.)
    pub design: HashMap<String, f64>,
    /// Random seed for this scenario
    pub seed: u64,
}

impl EvalScenario {
    /// Sample a random evaluation scenario
    pub fn sample(rng: &mut impl Rng) -> Self {
        // Generate realistic covariate values
        let age = rng.gen_range(18.0..80.0);
        let weight = rng.gen_range(50.0..120.0);
        let bmi = weight / ((rng.gen_range(1.5..2.0)).powi(2));

        let mut covariates = HashMap::new();
        covariates.insert("age".to_string(), age);
        covariates.insert("weight".to_string(), weight);
        covariates.insert("bmi".to_string(), bmi);

        // Generate design parameters
        let dose = rng.gen_range(10.0..500.0);
        let dose_times = rng.gen_range(1..5) as f64;

        let mut design = HashMap::new();
        design.insert("dose".to_string(), dose);
        design.insert("dose_times".to_string(), dose_times);

        Self {
            covariates,
            design,
            seed: rng.gen(),
        }
    }
}

// =============================================================================
// Simulation Results
// =============================================================================

/// Results from simulating a scenario
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Output values (e.g., concentration time series, endpoints)
    pub outputs: Vec<f64>,
    /// Contract violations detected
    pub violations: Vec<String>,
}

impl SimulationResult {
    fn empty() -> Self {
        Self {
            outputs: Vec::new(),
            violations: Vec::new(),
        }
    }
}

// =============================================================================
// Evaluation Engine
// =============================================================================

/// Evaluate a surrogate model against its mechanistic reference
pub fn evaluate_surrogate(
    _ev_handle: &str, // Evidence program handle (would be used in full implementation)
    surr: &SurrogateModelHandle,
    cfg: &SurrogateEvalConfig,
) -> Result<SurrogateEvalReport, SurrogateEvalError> {
    // Validate configuration
    cfg.validate()?;

    let mut rng = ChaCha20Rng::seed_from_u64(cfg.seed);
    let mut metrics = MetricsAccum::default();
    let mut mech_viol_count = 0usize;
    let mut surr_viol_count = 0usize;

    for _i in 0..cfg.n_eval {
        // 1. Sample an evaluation scenario
        let scenario = EvalScenario::sample(&mut rng);

        // 2. Run mechanistic reference
        let mech_result = simulate_mechanistic(&scenario, cfg.backend_ref)?;
        mech_viol_count += mech_result.violations.len();

        // 3. Run surrogate
        let surr_result = simulate_surrogate(&scenario, surr)?;
        surr_viol_count += surr_result.violations.len();

        // 4. Accumulate metrics
        metrics.accumulate(&mech_result.outputs, &surr_result.outputs)?;
    }

    // Finalize metrics
    let (rmse, mae, max_abs_err) = metrics.finalize()?;

    Ok(SurrogateEvalReport {
        n_eval: cfg.n_eval,
        rmse,
        mae,
        max_abs_err,
        mech_contract_violations: mech_viol_count,
        surr_contract_violations: surr_viol_count,
    })
}

/// Simulate mechanistic backend for a scenario
fn simulate_mechanistic(
    scenario: &EvalScenario,
    backend: BackendKind,
) -> Result<SimulationResult, SurrogateEvalError> {
    // TODO: Integrate with actual evidence runner / mechanistic simulator
    // For now, generate synthetic outputs based on scenario

    if !backend.requires_mechanistic() {
        return Err(SurrogateEvalError::MechanisticFailed(
            "backend does not support mechanistic execution".to_string(),
        ));
    }

    // Generate synthetic time series (concentration curve)
    let dose = scenario.design.get("dose").copied().unwrap_or(100.0);
    let weight = scenario.covariates.get("weight").copied().unwrap_or(70.0);

    let mut outputs = Vec::new();
    for t in 0..24 {
        // Simple one-compartment PK model simulation
        let ke = 0.1; // elimination rate
        let v = weight * 0.7; // volume of distribution
        let conc = (dose / v) * (-ke * t as f64).exp();
        outputs.push(conc);
    }

    // Check contracts (example: concentration must be non-negative)
    let mut violations = Vec::new();
    for (i, &val) in outputs.iter().enumerate() {
        if val < 0.0 {
            violations.push(format!("Negative concentration at time {}: {}", i, val));
        }
        if val.is_nan() || val.is_infinite() {
            violations.push(format!("Invalid concentration at time {}: {}", i, val));
        }
    }

    Ok(SimulationResult {
        outputs,
        violations,
    })
}

/// Simulate surrogate for a scenario
fn simulate_surrogate(
    scenario: &EvalScenario,
    surr: &SurrogateModelHandle,
) -> Result<SimulationResult, SurrogateEvalError> {
    // TODO: Integrate with actual surrogate prediction API
    // For now, generate synthetic outputs with added noise

    let dose = scenario.design.get("dose").copied().unwrap_or(100.0);
    let weight = scenario.covariates.get("weight").copied().unwrap_or(70.0);

    // Use surrogate ID to create deterministic noise
    let noise_seed = surr.id.as_u128() as u64 ^ scenario.seed;
    let mut noise_rng = ChaCha20Rng::seed_from_u64(noise_seed);

    let mut outputs = Vec::new();
    for t in 0..24 {
        // Surrogate approximation: mechanistic + noise
        let ke = 0.1;
        let v = weight * 0.7;
        let true_conc = (dose / v) * (-ke * t as f64).exp();

        // Add surrogate approximation error (5% noise)
        let noise = noise_rng.gen_range(-0.05..0.05);
        let surr_conc = true_conc * (1.0 + noise);

        outputs.push(surr_conc);
    }

    // Check contracts (same as mechanistic)
    let mut violations = Vec::new();
    for (i, &val) in outputs.iter().enumerate() {
        if val < 0.0 {
            violations.push(format!("Negative concentration at time {}: {}", i, val));
        }
        if val.is_nan() || val.is_infinite() {
            violations.push(format!("Invalid concentration at time {}: {}", i, val));
        }
    }

    Ok(SimulationResult {
        outputs,
        violations,
    })
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eval_config_validation() {
        let valid = SurrogateEvalConfig {
            n_eval: 100,
            backend_ref: BackendKind::Mechanistic,
            seed: 42,
        };
        assert!(valid.validate().is_ok());

        let invalid_n_eval = SurrogateEvalConfig {
            n_eval: 0,
            backend_ref: BackendKind::Mechanistic,
            seed: 42,
        };
        assert!(invalid_n_eval.validate().is_err());

        let invalid_backend = SurrogateEvalConfig {
            n_eval: 100,
            backend_ref: BackendKind::Surrogate,
            seed: 42,
        };
        assert!(invalid_backend.validate().is_err());
    }

    #[test]
    fn test_metrics_accumulator() {
        let mut accum = MetricsAccum::default();

        let ref_outputs = vec![1.0, 2.0, 3.0];
        let surr_outputs = vec![1.1, 2.2, 2.9];

        accum.accumulate(&ref_outputs, &surr_outputs).unwrap();

        let (rmse, mae, max_abs_err) = accum.finalize().unwrap();

        // Expected: errors are [0.1, 0.2, -0.1]
        // MAE = (0.1 + 0.2 + 0.1) / 3 = 0.133...
        // RMSE = sqrt((0.01 + 0.04 + 0.01) / 3) = sqrt(0.02) = 0.1414...
        // max_abs_err = 0.2

        assert!((mae - 0.1333).abs() < 0.001);
        assert!((rmse - 0.1414).abs() < 0.001);
        assert!((max_abs_err - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_eval_scenario_sampling() {
        let mut rng = ChaCha20Rng::seed_from_u64(42);
        let scenario = EvalScenario::sample(&mut rng);

        // Check covariates are in reasonable ranges
        let age = scenario.covariates.get("age").unwrap();
        assert!(*age >= 18.0 && *age <= 80.0);

        let weight = scenario.covariates.get("weight").unwrap();
        assert!(*weight >= 50.0 && *weight <= 120.0);
    }

    #[test]
    fn test_mechanistic_simulation() {
        let mut scenario = EvalScenario {
            covariates: HashMap::new(),
            design: HashMap::new(),
            seed: 42,
        };
        scenario.covariates.insert("weight".to_string(), 70.0);
        scenario.design.insert("dose".to_string(), 100.0);

        let result = simulate_mechanistic(&scenario, BackendKind::Mechanistic).unwrap();

        // Should produce 24 time points
        assert_eq!(result.outputs.len(), 24);

        // Concentration should decay over time
        assert!(result.outputs[0] > result.outputs[23]);

        // Should have no violations for valid scenario
        assert_eq!(result.violations.len(), 0);
    }

    #[test]
    fn test_surrogate_simulation() {
        let mut scenario = EvalScenario {
            covariates: HashMap::new(),
            design: HashMap::new(),
            seed: 42,
        };
        scenario.covariates.insert("weight".to_string(), 70.0);
        scenario.design.insert("dose".to_string(), 100.0);

        let surr = SurrogateModelHandle::new();
        let result = simulate_surrogate(&scenario, &surr).unwrap();

        // Should produce 24 time points
        assert_eq!(result.outputs.len(), 24);

        // Concentration should decay over time (approximately)
        assert!(result.outputs[0] > result.outputs[23]);
    }

    #[test]
    fn test_evaluate_surrogate_integration() {
        let surr = SurrogateModelHandle::new();
        let cfg = SurrogateEvalConfig {
            n_eval: 10,
            backend_ref: BackendKind::Mechanistic,
            seed: 42,
        };

        let report = evaluate_surrogate("test_ev", &surr, &cfg).unwrap();

        assert_eq!(report.n_eval, 10);
        assert!(report.rmse > 0.0); // Should have some error
        assert!(report.mae > 0.0);
        assert!(report.max_abs_err > 0.0);

        // With synthetic data, should have no violations
        assert_eq!(report.mech_contract_violations, 0);
        assert_eq!(report.surr_contract_violations, 0);
    }

    #[test]
    fn test_report_acceptability() {
        let report = SurrogateEvalReport {
            n_eval: 100,
            rmse: 0.05,
            mae: 0.03,
            max_abs_err: 0.15,
            mech_contract_violations: 0,
            surr_contract_violations: 0,
        };

        // Should be acceptable with generous thresholds
        assert!(report.is_acceptable(0.1, 0.1, 0.2));

        // Should not be acceptable with tight thresholds
        assert!(!report.is_acceptable(0.01, 0.01, 0.1));

        // Should not be acceptable if surrogate has violations
        let report_with_viols = SurrogateEvalReport {
            surr_contract_violations: 1,
            ..report.clone()
        };
        assert!(!report_with_viols.is_acceptable(0.1, 0.1, 0.2));
    }

    #[test]
    fn test_output_shape_mismatch() {
        let mut accum = MetricsAccum::default();
        let ref_outputs = vec![1.0, 2.0, 3.0];
        let surr_outputs = vec![1.0, 2.0]; // Different length

        let result = accum.accumulate(&ref_outputs, &surr_outputs);
        assert!(result.is_err());

        match result {
            Err(SurrogateEvalError::OutputShapeMismatch { mech_len, surr_len }) => {
                assert_eq!(mech_len, 3);
                assert_eq!(surr_len, 2);
            }
            _ => panic!("Expected OutputShapeMismatch error"),
        }
    }
}
