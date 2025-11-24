//! Simulation-Based Calibration (SBC) for validating Bayesian inference
//!
//! SBC tests whether the posterior inference procedure is correctly calibrated
//! by checking if the rank statistic follows a uniform distribution.
//!
//! Reference: Talts et al. (2018) "Validating Bayesian Inference Algorithms with Simulation-Based Calibration"

use crate::diagnostics::DiagnosticsError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for running SBC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbcConfig {
    /// Number of SBC replications to run
    pub n_sims: usize,
    /// Number of posterior draws per replication
    pub n_draws_per_sim: usize,
    /// Parameters to track (empty = track all)
    pub params_to_track: Vec<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

/// Results from a single SBC replication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbcReplication {
    /// Replication index
    pub sim_id: usize,
    /// True parameter values used to generate data
    pub true_params: HashMap<String, f64>,
    /// Rank of true value in posterior draws (one per parameter)
    pub ranks: HashMap<String, usize>,
    /// Whether the fit succeeded
    pub success: bool,
    /// Optional error message if fit failed
    pub error_message: Option<String>,
}

/// Complete SBC results across all replications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbcResult {
    /// Configuration used
    pub config: SbcConfig,
    /// Results from each replication
    pub replications: Vec<SbcReplication>,
    /// Rank histograms (parameter name -> counts per bin)
    pub rank_histograms: HashMap<String, Vec<usize>>,
    /// Number of successful fits
    pub n_success: usize,
    /// Number of failed fits
    pub n_failed: usize,
    /// SBC quality assessment
    pub quality: SbcQuality,
}

/// Quality assessment for SBC results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SbcQuality {
    /// Whether rank distributions appear uniform (chi-square test)
    pub ranks_uniform: HashMap<String, bool>,
    /// Chi-square statistics for each parameter
    pub chi_square_stats: HashMap<String, f64>,
    /// P-values for uniformity tests
    pub p_values: HashMap<String, f64>,
    /// Overall pass/fail assessment
    pub overall_pass: bool,
    /// Diagnostic messages
    pub messages: Vec<String>,
}

impl Default for SbcConfig {
    fn default() -> Self {
        SbcConfig {
            n_sims: 100,
            n_draws_per_sim: 1000,
            params_to_track: Vec::new(),
            seed: None,
        }
    }
}

/// Run Simulation-Based Calibration
///
/// This is a scaffold function that defines the SBC workflow structure.
/// The actual implementation requires:
/// 1. A way to sample from the prior
/// 2. A way to simulate data given parameters
/// 3. A way to fit the model and get posterior draws
///
/// # Arguments
/// * `config` - SBC configuration
///
/// # Returns
/// SBC results including rank histograms and quality assessment
pub fn run_sbc(config: SbcConfig) -> Result<SbcResult, DiagnosticsError> {
    // This is a scaffold - actual implementation would:
    // 1. Loop over n_sims replications
    // 2. For each replication:
    //    a. Sample true parameters from prior
    //    b. Simulate data using true parameters
    //    c. Fit model to simulated data
    //    d. Compute rank of true parameter in posterior
    // 3. Build rank histograms
    // 4. Perform uniformity tests

    Err(DiagnosticsError::IoError(
        "SBC scaffold not yet implemented - requires integration with prior sampling and model fitting".to_string()
    ))
}

/// Compute rank of true value in posterior draws
///
/// # Arguments
/// * `true_value` - The true parameter value
/// * `posterior_draws` - Vector of posterior draws for this parameter
///
/// # Returns
/// Rank of the true value (0 to n_draws inclusive)
pub fn compute_rank(true_value: f64, posterior_draws: &[f64]) -> usize {
    // Count how many posterior draws are less than the true value
    posterior_draws.iter().filter(|&&x| x < true_value).count()
}

/// Build rank histogram from SBC replications
///
/// # Arguments
/// * `ranks` - Vector of rank statistics for a parameter
/// * `n_bins` - Number of bins for the histogram (typically n_draws + 1)
///
/// # Returns
/// Histogram counts for each bin
pub fn build_rank_histogram(ranks: &[usize], n_bins: usize) -> Vec<usize> {
    let mut histogram = vec![0; n_bins];
    for &rank in ranks {
        if rank < n_bins {
            histogram[rank] += 1;
        }
    }
    histogram
}

/// Test uniformity of rank histogram using chi-square test
///
/// # Arguments
/// * `histogram` - Rank histogram counts
///
/// # Returns
/// (chi_square_statistic, p_value, is_uniform)
pub fn test_uniformity(histogram: &[usize]) -> (f64, f64, bool) {
    let n_bins = histogram.len();
    let total_count: usize = histogram.iter().sum();

    if total_count == 0 {
        return (0.0, 1.0, true);
    }

    let expected = total_count as f64 / n_bins as f64;

    // Compute chi-square statistic
    let chi_square: f64 = histogram
        .iter()
        .map(|&observed| {
            let diff = observed as f64 - expected;
            (diff * diff) / expected
        })
        .sum();

    // Degrees of freedom = n_bins - 1
    let df = n_bins - 1;

    // Compute p-value using chi-square distribution
    // For now, use a simple approximation: reject if chi_square > critical value
    // Critical values for common confidence levels (df varies):
    // This is a simplified check - proper implementation would use a chi-square CDF
    let critical_value = approximate_chi_square_critical(df, 0.05);
    let is_uniform = chi_square < critical_value;

    // Approximate p-value (simplified)
    let p_value = if chi_square < critical_value {
        0.1
    } else {
        0.01
    };

    (chi_square, p_value, is_uniform)
}

/// Approximate chi-square critical value for given df and alpha
/// This is a simplified approximation - a real implementation would use proper chi-square quantiles
fn approximate_chi_square_critical(df: usize, _alpha: f64) -> f64 {
    // Rough approximation: critical value ≈ df + 2*sqrt(2*df) for alpha=0.05
    df as f64 + 2.0 * (2.0 * df as f64).sqrt()
}

/// Analyze SBC results and generate quality assessment
///
/// # Arguments
/// * `replications` - Vector of SBC replication results
/// * `config` - SBC configuration
///
/// # Returns
/// Complete SBC result with rank histograms and quality assessment
pub fn analyze_sbc_results(
    replications: Vec<SbcReplication>,
    config: SbcConfig,
) -> Result<SbcResult, DiagnosticsError> {
    if replications.is_empty() {
        return Err(DiagnosticsError::IoError(
            "No SBC replications provided".to_string(),
        ));
    }

    // Count successes and failures
    let n_success = replications.iter().filter(|r| r.success).count();
    let n_failed = replications.len() - n_success;

    // Collect all parameter names
    let param_names: Vec<String> = if let Some(first_rep) = replications.first() {
        first_rep.ranks.keys().cloned().collect()
    } else {
        Vec::new()
    };

    let n_bins = config.n_draws_per_sim + 1;
    let mut rank_histograms = HashMap::new();
    let mut ranks_uniform = HashMap::new();
    let mut chi_square_stats = HashMap::new();
    let mut p_values = HashMap::new();
    let mut messages = Vec::new();

    // Build rank histogram for each parameter
    for param_name in &param_names {
        let ranks: Vec<usize> = replications
            .iter()
            .filter(|r| r.success)
            .filter_map(|r| r.ranks.get(param_name).copied())
            .collect();

        if ranks.is_empty() {
            messages.push(format!(
                "No successful ranks for parameter '{}'",
                param_name
            ));
            continue;
        }

        let histogram = build_rank_histogram(&ranks, n_bins);
        let (chi_sq, p_val, is_uniform) = test_uniformity(&histogram);

        rank_histograms.insert(param_name.clone(), histogram);
        ranks_uniform.insert(param_name.clone(), is_uniform);
        chi_square_stats.insert(param_name.clone(), chi_sq);
        p_values.insert(param_name.clone(), p_val);

        if !is_uniform {
            messages.push(format!(
                "Parameter '{}' shows non-uniform ranks (χ²={:.2}, p={:.3})",
                param_name, chi_sq, p_val
            ));
        }
    }

    // Overall pass = all parameters have uniform ranks AND sufficient success rate
    let success_rate = n_success as f64 / replications.len() as f64;
    let all_uniform = ranks_uniform.values().all(|&u| u);
    let sufficient_success = success_rate >= 0.80; // At least 80% success rate

    let overall_pass = all_uniform && sufficient_success;

    if !sufficient_success {
        messages.push(format!(
            "Low success rate: {}/{} ({:.1}%) fits succeeded",
            n_success,
            replications.len(),
            success_rate * 100.0
        ));
    }

    if overall_pass {
        messages.push("SBC validation passed: ranks appear uniform".to_string());
    }

    let quality = SbcQuality {
        ranks_uniform,
        chi_square_stats,
        p_values,
        overall_pass,
        messages,
    };

    Ok(SbcResult {
        config,
        replications,
        rank_histograms,
        n_success,
        n_failed,
        quality,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_rank() {
        let draws = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // True value less than all draws
        assert_eq!(compute_rank(0.5, &draws), 0);

        // True value in the middle
        assert_eq!(compute_rank(3.5, &draws), 3);

        // True value greater than all draws
        assert_eq!(compute_rank(10.0, &draws), 5);
    }

    #[test]
    fn test_build_rank_histogram() {
        // Perfectly uniform ranks
        let ranks = vec![0, 1, 2, 3, 4, 0, 1, 2, 3, 4];
        let histogram = build_rank_histogram(&ranks, 5);

        assert_eq!(histogram, vec![2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_uniformity_perfect() {
        // Perfectly uniform histogram
        let histogram = vec![20, 20, 20, 20, 20];
        let (chi_sq, _p_val, is_uniform) = test_uniformity(&histogram);

        assert!(
            chi_sq < 0.01,
            "Chi-square should be near 0 for perfect uniformity"
        );
        assert!(is_uniform, "Should pass uniformity test");
    }

    #[test]
    fn test_uniformity_non_uniform() {
        // Highly non-uniform histogram
        let histogram = vec![50, 10, 10, 10, 10];
        let (chi_sq, _p_val, is_uniform) = test_uniformity(&histogram);

        assert!(
            chi_sq > 10.0,
            "Chi-square should be large for non-uniform data"
        );
        assert!(!is_uniform, "Should fail uniformity test");
    }

    #[test]
    fn test_analyze_sbc_results() {
        let config = SbcConfig {
            n_sims: 3,
            n_draws_per_sim: 10,
            params_to_track: vec!["alpha".to_string()],
            seed: Some(42),
        };

        let replications = vec![
            SbcReplication {
                sim_id: 0,
                true_params: [("alpha".to_string(), 1.5)].iter().cloned().collect(),
                ranks: [("alpha".to_string(), 3)].iter().cloned().collect(),
                success: true,
                error_message: None,
            },
            SbcReplication {
                sim_id: 1,
                true_params: [("alpha".to_string(), 1.8)].iter().cloned().collect(),
                ranks: [("alpha".to_string(), 7)].iter().cloned().collect(),
                success: true,
                error_message: None,
            },
            SbcReplication {
                sim_id: 2,
                true_params: [("alpha".to_string(), 1.2)].iter().cloned().collect(),
                ranks: [("alpha".to_string(), 5)].iter().cloned().collect(),
                success: true,
                error_message: None,
            },
        ];

        let result = analyze_sbc_results(replications, config).unwrap();

        assert_eq!(result.n_success, 3);
        assert_eq!(result.n_failed, 0);
        assert!(result.rank_histograms.contains_key("alpha"));
        assert_eq!(result.quality.overall_pass, true);
    }

    #[test]
    fn test_sbc_config_default() {
        let config = SbcConfig::default();
        assert_eq!(config.n_sims, 100);
        assert_eq!(config.n_draws_per_sim, 1000);
        assert!(config.params_to_track.is_empty());
        assert!(config.seed.is_none());
    }
}
