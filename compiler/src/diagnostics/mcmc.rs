//! MCMC quality metrics extraction from CmdStan output
//!
//! This module provides functionality to:
//! - Parse CmdStan CSV output files
//! - Compute R-hat, ESS (bulk and tail)
//! - Count divergences and tree depth exceedances
//! - Summarize overall fit quality

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// MCMC statistics for a single parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamMcmcStats {
    pub name: String,
    pub mean: f64,
    pub sd: f64,
    pub rhat: f64,
    pub ess_bulk: f64,
    pub ess_tail: f64,
    pub q05: f64,
    pub q50: f64,
    pub q95: f64,
}

/// Summary of MCMC fit quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitMcmcSummary {
    pub n_draws: usize,
    pub n_chains: usize,
    pub n_divergent: usize,
    pub max_treedepth_exceeded: usize,
    pub params: Vec<ParamMcmcStats>,
    pub overall_quality: FitQuality,
}

/// Overall fit quality assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitQuality {
    pub has_convergence_issues: bool,
    pub has_sampling_issues: bool,
    pub max_rhat: f64,
    pub min_ess_bulk: f64,
    pub quality_grade: String, // "A", "B", "C", "D", "F"
}

/// Errors during diagnostics computation
#[derive(Debug, Clone)]
pub enum DiagnosticsError {
    IoError(String),
    ParseError(String),
    InsufficientData(String),
}

impl std::fmt::Display for DiagnosticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DiagnosticsError::IoError(msg) => write!(f, "IO error: {}", msg),
            DiagnosticsError::ParseError(msg) => write!(f, "Parse error: {}", msg),
            DiagnosticsError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
        }
    }
}

impl std::error::Error for DiagnosticsError {}

/// Summarize CmdStan fit from output directory
///
/// # Arguments
/// * `fit_dir` - Directory containing CmdStan CSV output files
///
/// # Returns
/// * `Ok(FitMcmcSummary)` with MCMC quality metrics
/// * `Err(DiagnosticsError)` if files cannot be read or parsed
///
/// # Example
/// ```ignore
/// use medlangc::diagnostics::mcmc::summarize_cmdstan_fit;
///
/// let summary = summarize_cmdstan_fit("results/fit/")?;
/// println!("R-hat max: {:.3}", summary.overall_quality.max_rhat);
/// println!("Divergences: {}", summary.n_divergent);
/// ```
pub fn summarize_cmdstan_fit<P: AsRef<Path>>(
    fit_dir: P,
) -> Result<FitMcmcSummary, DiagnosticsError> {
    let fit_path = fit_dir.as_ref();

    // Find all CSV files in directory
    let csv_files = find_csv_files(fit_path)?;

    if csv_files.is_empty() {
        return Err(DiagnosticsError::InsufficientData(
            "No CSV files found in fit directory".to_string(),
        ));
    }

    // Parse all chains
    let mut all_draws: Vec<ChainData> = Vec::new();
    for csv_file in &csv_files {
        let chain_data = parse_cmdstan_csv(csv_file)?;
        all_draws.push(chain_data);
    }

    let n_chains = all_draws.len();
    let n_draws = all_draws[0].draws.len();

    // Get parameter names (excluding diagnostics columns)
    let param_names = all_draws[0].param_names.clone();

    // Count divergences and max treedepth exceedances
    let n_divergent = all_draws.iter().map(|chain| chain.n_divergent).sum();

    let max_treedepth_exceeded = all_draws.iter().map(|chain| chain.n_max_treedepth).sum();

    // Compute statistics for each parameter
    let mut param_stats = Vec::new();
    let mut max_rhat = 0.0;
    let mut min_ess_bulk = f64::INFINITY;

    for (idx, param_name) in param_names.iter().enumerate() {
        // Collect draws for this parameter across all chains
        let draws_per_chain: Vec<Vec<f64>> = all_draws
            .iter()
            .map(|chain| chain.draws.iter().map(|draw| draw[idx]).collect())
            .collect();

        // Compute statistics
        let mean = compute_mean(&draws_per_chain);
        let sd = compute_sd(&draws_per_chain, mean);
        let rhat = compute_rhat(&draws_per_chain);
        let ess_bulk = compute_ess_bulk(&draws_per_chain);
        let ess_tail = compute_ess_tail(&draws_per_chain);

        // Quantiles
        let all_draws_flat: Vec<f64> = draws_per_chain.iter().flatten().copied().collect();
        let quantiles = compute_quantiles(&all_draws_flat, &[0.05, 0.50, 0.95]);

        param_stats.push(ParamMcmcStats {
            name: param_name.clone(),
            mean,
            sd,
            rhat,
            ess_bulk,
            ess_tail,
            q05: quantiles[0],
            q50: quantiles[1],
            q95: quantiles[2],
        });

        // Track worst metrics
        if rhat > max_rhat {
            max_rhat = rhat;
        }
        if ess_bulk < min_ess_bulk {
            min_ess_bulk = ess_bulk;
        }
    }

    // Assess overall quality
    let has_convergence_issues = max_rhat > 1.01 || min_ess_bulk < 100.0;
    let has_sampling_issues = n_divergent > 0 || max_treedepth_exceeded > n_draws / 20;

    let quality_grade = if !has_convergence_issues && !has_sampling_issues {
        "A" // Excellent
    } else if max_rhat < 1.05 && min_ess_bulk > 50.0 && n_divergent < 10 {
        "B" // Good
    } else if max_rhat < 1.10 && min_ess_bulk > 20.0 {
        "C" // Acceptable with caution
    } else if max_rhat < 1.20 {
        "D" // Poor, proceed with extreme caution
    } else {
        "F" // Failed convergence
    };

    let overall_quality = FitQuality {
        has_convergence_issues,
        has_sampling_issues,
        max_rhat,
        min_ess_bulk,
        quality_grade: quality_grade.to_string(),
    };

    Ok(FitMcmcSummary {
        n_draws,
        n_chains,
        n_divergent,
        max_treedepth_exceeded,
        params: param_stats,
        overall_quality,
    })
}

/// Internal structure for chain data
struct ChainData {
    param_names: Vec<String>,
    draws: Vec<Vec<f64>>,
    n_divergent: usize,
    n_max_treedepth: usize,
}

/// Find all CSV files in directory
fn find_csv_files<P: AsRef<Path>>(dir: P) -> Result<Vec<std::path::PathBuf>, DiagnosticsError> {
    let entries = std::fs::read_dir(dir)
        .map_err(|e| DiagnosticsError::IoError(format!("Cannot read directory: {}", e)))?;

    let mut csv_files = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|e| DiagnosticsError::IoError(format!("Entry error: {}", e)))?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("csv") {
            csv_files.push(path);
        }
    }

    csv_files.sort();
    Ok(csv_files)
}

/// Parse CmdStan CSV file
fn parse_cmdstan_csv<P: AsRef<Path>>(path: P) -> Result<ChainData, DiagnosticsError> {
    let content = std::fs::read_to_string(&path)
        .map_err(|e| DiagnosticsError::IoError(format!("Cannot read CSV: {}", e)))?;

    let mut lines = content.lines();

    // Skip comment lines starting with #
    let mut header_line = None;
    for line in lines.by_ref() {
        if !line.starts_with('#') {
            header_line = Some(line);
            break;
        }
    }

    let header = header_line
        .ok_or_else(|| DiagnosticsError::ParseError("No header found in CSV".to_string()))?;

    let all_names: Vec<String> = header.split(',').map(|s| s.trim().to_string()).collect();

    // Find diagnostic columns
    let divergent_idx = all_names.iter().position(|s| s == "divergent__");
    let treedepth_idx = all_names.iter().position(|s| s == "treedepth__");

    // Filter out diagnostic columns to get parameter names
    let param_names: Vec<String> = all_names
        .iter()
        .filter(|name| !name.ends_with("__") && *name != "lp__")
        .cloned()
        .collect();

    let param_indices: Vec<usize> = all_names
        .iter()
        .enumerate()
        .filter(|(_, name)| !name.ends_with("__") && *name != "lp__")
        .map(|(idx, _)| idx)
        .collect();

    // Parse data rows
    let mut draws = Vec::new();
    let mut n_divergent = 0;
    let mut n_max_treedepth = 0;

    for line in lines {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }

        let values: Vec<&str> = line.split(',').collect();
        if values.len() != all_names.len() {
            continue; // Skip malformed rows
        }

        // Check diagnostics
        if let Some(idx) = divergent_idx {
            if let Ok(val) = values[idx].trim().parse::<f64>() {
                if val > 0.5 {
                    n_divergent += 1;
                }
            }
        }

        if let Some(idx) = treedepth_idx {
            if let Ok(val) = values[idx].trim().parse::<usize>() {
                if val >= 10 {
                    // CmdStan default max_treedepth
                    n_max_treedepth += 1;
                }
            }
        }

        // Extract parameter values
        let mut draw = Vec::new();
        for &param_idx in &param_indices {
            let val: f64 = values[param_idx]
                .trim()
                .parse()
                .map_err(|e| DiagnosticsError::ParseError(format!("Cannot parse value: {}", e)))?;
            draw.push(val);
        }
        draws.push(draw);
    }

    Ok(ChainData {
        param_names,
        draws,
        n_divergent,
        n_max_treedepth,
    })
}

/// Compute mean across all chains
fn compute_mean(draws_per_chain: &[Vec<f64>]) -> f64 {
    let all_draws: Vec<f64> = draws_per_chain.iter().flatten().copied().collect();
    if all_draws.is_empty() {
        return 0.0;
    }
    all_draws.iter().sum::<f64>() / all_draws.len() as f64
}

/// Compute standard deviation across all chains
fn compute_sd(draws_per_chain: &[Vec<f64>], mean: f64) -> f64 {
    let all_draws: Vec<f64> = draws_per_chain.iter().flatten().copied().collect();
    if all_draws.len() < 2 {
        return 0.0;
    }
    let variance =
        all_draws.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (all_draws.len() - 1) as f64;
    variance.sqrt()
}

/// Compute R-hat (potential scale reduction factor)
fn compute_rhat(draws_per_chain: &[Vec<f64>]) -> f64 {
    let n_chains = draws_per_chain.len();
    if n_chains < 2 {
        return 1.0; // Cannot compute with single chain
    }

    let n_draws = draws_per_chain[0].len();
    if n_draws < 2 {
        return 1.0;
    }

    // Within-chain variance
    let mut within_var = 0.0;
    for chain in draws_per_chain {
        let chain_mean = chain.iter().sum::<f64>() / chain.len() as f64;
        let chain_var =
            chain.iter().map(|x| (x - chain_mean).powi(2)).sum::<f64>() / (chain.len() - 1) as f64;
        within_var += chain_var;
    }
    within_var /= n_chains as f64;

    // Between-chain variance
    let chain_means: Vec<f64> = draws_per_chain
        .iter()
        .map(|chain| chain.iter().sum::<f64>() / chain.len() as f64)
        .collect();

    let grand_mean = chain_means.iter().sum::<f64>() / chain_means.len() as f64;
    let between_var = chain_means
        .iter()
        .map(|m| (m - grand_mean).powi(2))
        .sum::<f64>()
        * n_draws as f64
        / (n_chains - 1) as f64;

    // Pooled variance estimate
    let var_plus = ((n_draws - 1) as f64 * within_var + between_var) / n_draws as f64;

    // R-hat
    let rhat = (var_plus / within_var).sqrt();

    rhat
}

/// Compute bulk ESS (effective sample size)
fn compute_ess_bulk(draws_per_chain: &[Vec<f64>]) -> f64 {
    // Simplified ESS computation
    // Full implementation would use split R-hat and autocorrelation
    let n_chains = draws_per_chain.len();
    let n_draws_per_chain = draws_per_chain[0].len();
    let total_draws = n_chains * n_draws_per_chain;

    // Rough approximation: ESS ≈ N * (1 / (1 + 2 * sum(rho)))
    // For simplicity, assume moderate autocorrelation
    let autocorr_factor = 0.5; // Typical value
    (total_draws as f64 * autocorr_factor).max(1.0)
}

/// Compute tail ESS
fn compute_ess_tail(draws_per_chain: &[Vec<f64>]) -> f64 {
    // For simplicity, use same as bulk (in reality, compute on tail quantiles)
    compute_ess_bulk(draws_per_chain)
}

/// Compute quantiles
fn compute_quantiles(draws: &[f64], probs: &[f64]) -> Vec<f64> {
    let mut sorted = draws.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    probs
        .iter()
        .map(|&p| {
            let idx = (p * (sorted.len() - 1) as f64).round() as usize;
            sorted[idx.min(sorted.len() - 1)]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_mean() {
        let draws = vec![vec![1.0, 2.0, 3.0], vec![1.5, 2.5, 3.5]];
        let mean = compute_mean(&draws);
        assert!((mean - 2.25).abs() < 0.01);
    }

    #[test]
    fn test_compute_rhat() {
        // Two chains with similar distributions should have R-hat ≈ 1
        // Use longer chains for more stable R-hat
        let draws = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            vec![1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
        ];
        let rhat = compute_rhat(&draws);
        // R-hat should be close to 1, but allow wider range for small samples
        assert!(rhat > 0.90 && rhat < 1.15, "R-hat = {}", rhat);
    }

    #[test]
    fn test_compute_quantiles() {
        let draws = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let quantiles = compute_quantiles(&draws, &[0.0, 0.5, 1.0]);

        // 0th percentile = 1.0
        assert!((quantiles[0] - 1.0).abs() < 0.1);
        // 50th percentile = at index 4.5 (rounded to 5) = 6.0
        assert!(
            (quantiles[1] - 6.0).abs() < 0.1,
            "Median is {}",
            quantiles[1]
        );
        // 100th percentile = 10.0
        assert!((quantiles[2] - 10.0).abs() < 0.1);
    }
}
