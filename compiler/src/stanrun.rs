//! Stan execution and MCMC diagnostics
//!
//! This module provides functionality to:
//! - Detect cmdstan installation
//! - Execute Stan models with data
//! - Parse MCMC output
//! - Compute basic diagnostics (Rhat, ESS)

use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Configuration for Stan MCMC sampling
#[derive(Debug, Clone)]
pub struct StanConfig {
    /// Number of chains to run
    pub num_chains: usize,
    /// Number of warmup iterations per chain
    pub num_warmup: usize,
    /// Number of sampling iterations per chain
    pub num_samples: usize,
    /// Random seed for reproducibility
    pub seed: Option<u32>,
    /// Adapt delta (target acceptance rate)
    pub adapt_delta: f64,
    /// Maximum tree depth
    pub max_treedepth: usize,
}

impl Default for StanConfig {
    fn default() -> Self {
        Self {
            num_chains: 4,
            num_warmup: 1000,
            num_samples: 1000,
            seed: None,
            adapt_delta: 0.8,
            max_treedepth: 10,
        }
    }
}

/// Result of Stan MCMC sampling
#[derive(Debug)]
pub struct StanResult {
    /// Path to output directory
    pub output_dir: PathBuf,
    /// Paths to chain CSV files
    pub chain_files: Vec<PathBuf>,
    /// Parameter samples (parameter name -> chain samples)
    pub samples: HashMap<String, Vec<Vec<f64>>>,
    /// Diagnostics for each parameter
    pub diagnostics: HashMap<String, ParamDiagnostics>,
}

/// Diagnostics for a single parameter
#[derive(Debug, Clone)]
pub struct ParamDiagnostics {
    /// Parameter name
    pub name: String,
    /// Rhat statistic (should be < 1.01 for convergence)
    pub rhat: f64,
    /// Effective sample size
    pub ess_bulk: f64,
    /// Tail effective sample size
    pub ess_tail: f64,
    /// Mean across all chains
    pub mean: f64,
    /// Standard deviation across all chains
    pub sd: f64,
    /// Quantiles [5%, 50%, 95%]
    pub quantiles: [f64; 3],
}

/// Detect cmdstan installation
pub fn detect_cmdstan() -> Result<PathBuf> {
    // Try CMDSTAN environment variable
    if let Ok(path) = std::env::var("CMDSTAN") {
        let cmdstan_path = PathBuf::from(&path);
        if cmdstan_path.exists() {
            return Ok(cmdstan_path);
        }
    }

    // Try common installation locations
    let home = std::env::var("HOME").context("HOME not set")?;
    let common_paths = vec![
        format!("{}/.cmdstan", home),
        format!("{}/cmdstan", home),
        "/usr/local/cmdstan".to_string(),
        "/opt/cmdstan".to_string(),
    ];

    for path in common_paths {
        let cmdstan_path = PathBuf::from(&path);
        if cmdstan_path.exists() {
            // Find the most recent version
            if let Ok(entries) = fs::read_dir(&cmdstan_path) {
                let mut versions: Vec<_> = entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.path().is_dir())
                    .collect();
                versions.sort_by_key(|e| e.path());
                if let Some(latest) = versions.last() {
                    return Ok(latest.path());
                }
            }
        }
    }

    bail!("cmdstan not found. Please set CMDSTAN environment variable or install cmdstan")
}

/// Compile a Stan model to executable
pub fn compile_stan_model(stan_file: &Path, cmdstan_path: &Path) -> Result<PathBuf> {
    let model_name = stan_file
        .file_stem()
        .context("Invalid Stan file name")?
        .to_string_lossy();

    let exe_path = stan_file.with_extension("");

    // Check if already compiled and up-to-date
    if exe_path.exists() {
        let stan_modified = fs::metadata(stan_file)?.modified()?;
        let exe_modified = fs::metadata(&exe_path)?.modified()?;
        if exe_modified > stan_modified {
            return Ok(exe_path);
        }
    }

    // Use cmdstan's make system
    let _make_path = cmdstan_path.join("make").join("standalone");

    eprintln!("Compiling Stan model: {}", model_name);

    let output = Command::new("make")
        .current_dir(cmdstan_path)
        .arg(exe_path.to_string_lossy().to_string())
        .output()
        .context("Failed to execute make")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Stan compilation failed:\n{}", stderr);
    }

    eprintln!("✓ Compilation successful");
    Ok(exe_path)
}

/// Run Stan MCMC sampling
pub fn run_stan_mcmc(
    exe_path: &Path,
    data_file: &Path,
    output_dir: &Path,
    config: &StanConfig,
) -> Result<StanResult> {
    // Create output directory
    fs::create_dir_all(output_dir).context("Failed to create output directory")?;

    let mut chain_files = Vec::new();

    eprintln!("Running MCMC sampling with {} chains...", config.num_chains);

    // Run each chain
    for chain_id in 1..=config.num_chains {
        let output_file = output_dir.join(format!("output_{}.csv", chain_id));
        chain_files.push(output_file.clone());

        eprintln!("  Chain {}/{}...", chain_id, config.num_chains);

        let mut cmd = Command::new(exe_path);
        cmd.arg("sample")
            .arg("num_warmup")
            .arg(config.num_warmup.to_string())
            .arg("num_samples")
            .arg(config.num_samples.to_string())
            .arg("adapt")
            .arg("delta")
            .arg(config.adapt_delta.to_string())
            .arg("algorithm=hmc")
            .arg("engine=nuts")
            .arg("max_depth")
            .arg(config.max_treedepth.to_string())
            .arg("data")
            .arg(format!("file={}", data_file.display()))
            .arg("output")
            .arg(format!("file={}", output_file.display()))
            .arg("id")
            .arg(chain_id.to_string());

        if let Some(seed) = config.seed {
            cmd.arg("random")
                .arg("seed")
                .arg((seed + chain_id as u32).to_string());
        }

        let output = cmd.output().context("Failed to execute Stan model")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("Chain {} failed:\n{}", chain_id, stderr);
        }
    }

    eprintln!("✓ MCMC sampling complete");

    // Parse output files
    let (samples, diagnostics) = parse_stan_output(&chain_files)?;

    Ok(StanResult {
        output_dir: output_dir.to_path_buf(),
        chain_files,
        samples,
        diagnostics,
    })
}

/// Parse Stan CSV output files
fn parse_stan_output(
    chain_files: &[PathBuf],
) -> Result<(
    HashMap<String, Vec<Vec<f64>>>,
    HashMap<String, ParamDiagnostics>,
)> {
    let mut all_samples: HashMap<String, Vec<Vec<f64>>> = HashMap::new();
    let mut param_names: Vec<String> = Vec::new();

    // Parse each chain
    for (chain_idx, chain_file) in chain_files.iter().enumerate() {
        let content = fs::read_to_string(chain_file).context("Failed to read chain output")?;

        let mut lines = content.lines();

        // Skip comment lines (start with #)
        while let Some(line) = lines.next() {
            if !line.starts_with('#') {
                // This is the header line with parameter names
                if chain_idx == 0 {
                    param_names = line.split(',').map(|s| s.to_string()).collect();
                    // Initialize sample storage
                    for name in &param_names {
                        all_samples.insert(name.clone(), Vec::new());
                    }
                }
                break;
            }
        }

        // Parse data lines
        for line in lines {
            if line.trim().is_empty() {
                continue;
            }

            let values: Vec<f64> = line
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            if values.len() != param_names.len() {
                continue; // Skip malformed lines
            }

            for (i, value) in values.iter().enumerate() {
                if let Some(samples) = all_samples.get_mut(&param_names[i]) {
                    if samples.len() <= chain_idx {
                        samples.resize(chain_idx + 1, Vec::new());
                    }
                    samples[chain_idx].push(*value);
                }
            }
        }
    }

    // Compute diagnostics
    let mut diagnostics = HashMap::new();

    for (param_name, chain_samples) in &all_samples {
        // Skip diagnostic columns (lp__, accept_stat__, etc.)
        if param_name.ends_with("__") {
            continue;
        }

        let diag = compute_diagnostics(param_name, chain_samples)?;
        diagnostics.insert(param_name.clone(), diag);
    }

    Ok((all_samples, diagnostics))
}

/// Compute diagnostics for a parameter
fn compute_diagnostics(name: &str, chain_samples: &[Vec<f64>]) -> Result<ParamDiagnostics> {
    if chain_samples.is_empty() {
        bail!("No samples for parameter {}", name);
    }

    let num_chains = chain_samples.len();
    let num_samples = chain_samples[0].len();

    // Flatten all samples
    let mut all_samples: Vec<f64> = chain_samples
        .iter()
        .flat_map(|c| c.iter().copied())
        .collect();
    all_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Compute mean and variance for each chain
    let chain_means: Vec<f64> = chain_samples
        .iter()
        .map(|samples| samples.iter().sum::<f64>() / samples.len() as f64)
        .collect();

    let chain_vars: Vec<f64> = chain_samples
        .iter()
        .zip(&chain_means)
        .map(|(samples, mean)| {
            let sum_sq: f64 = samples.iter().map(|x| (x - mean).powi(2)).sum();
            sum_sq / (samples.len() - 1) as f64
        })
        .collect();

    // Between-chain variance (B)
    let grand_mean = chain_means.iter().sum::<f64>() / num_chains as f64;
    let b = num_samples as f64
        * chain_means
            .iter()
            .map(|m| (m - grand_mean).powi(2))
            .sum::<f64>()
        / (num_chains - 1) as f64;

    // Within-chain variance (W)
    let w = chain_vars.iter().sum::<f64>() / num_chains as f64;

    // Rhat (Gelman-Rubin statistic)
    let var_plus = ((num_samples - 1) as f64 * w + b) / num_samples as f64;
    let rhat = (var_plus / w).sqrt();

    // Effective sample size (simplified)
    let ess_bulk = (num_chains * num_samples) as f64 / rhat.powi(2);
    let ess_tail = ess_bulk * 0.8; // Rough approximation

    // Overall statistics
    let mean = grand_mean;
    let sd = var_plus.sqrt();

    // Quantiles
    let q5_idx = (all_samples.len() as f64 * 0.05) as usize;
    let q50_idx = (all_samples.len() as f64 * 0.50) as usize;
    let q95_idx = (all_samples.len() as f64 * 0.95) as usize;

    let quantiles = [
        all_samples[q5_idx],
        all_samples[q50_idx],
        all_samples[q95_idx],
    ];

    Ok(ParamDiagnostics {
        name: name.to_string(),
        rhat,
        ess_bulk,
        ess_tail,
        mean,
        sd,
        quantiles,
    })
}

/// Print diagnostics summary
pub fn print_diagnostics(result: &StanResult) {
    println!("\n{}", "=".repeat(80));
    println!("MCMC Diagnostics Summary");
    println!("{}", "=".repeat(80));

    println!("\nOutput directory: {}", result.output_dir.display());
    println!("Number of chains: {}", result.chain_files.len());

    // Collect and sort parameters
    let mut params: Vec<_> = result.diagnostics.values().collect();
    params.sort_by(|a, b| a.name.cmp(&b.name));

    println!(
        "\n{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Parameter", "Mean", "SD", "5%", "50%", "95%", "Rhat", "ESS"
    );
    println!("{}", "-".repeat(100));

    for diag in params {
        println!(
            "{:<20} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.3} {:>10.0}",
            diag.name,
            diag.mean,
            diag.sd,
            diag.quantiles[0],
            diag.quantiles[1],
            diag.quantiles[2],
            diag.rhat,
            diag.ess_bulk
        );
    }

    println!("\n{}", "=".repeat(80));

    // Check for convergence issues
    let mut warnings = Vec::new();
    for diag in result.diagnostics.values() {
        if diag.rhat > 1.01 {
            warnings.push(format!(
                "⚠ {} has Rhat = {:.3} (should be < 1.01)",
                diag.name, diag.rhat
            ));
        }
        if diag.ess_bulk < 400.0 {
            warnings.push(format!(
                "⚠ {} has low ESS = {:.0} (should be > 400)",
                diag.name, diag.ess_bulk
            ));
        }
    }

    if warnings.is_empty() {
        println!("✓ All parameters converged successfully");
    } else {
        println!("\nConvergence Warnings:");
        for warning in warnings {
            println!("{}", warning);
        }
    }

    println!("{}", "=".repeat(80));
}
