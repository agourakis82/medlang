//! Quantum Prior-Posterior Diagnostics (Week 18 stub)
//!
//! Compares quantum-derived priors with fitted posteriors to assess agreement.
//! This is a minimal stub to support Week 19's quantum trust layer.

use serde::{Deserialize, Serialize};

/// Quantum prior information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPriorInfo {
    pub mu_prior: f64,
    pub sigma_prior: f64,
    pub scale: String, // "log" or "identity"
}

/// Posterior information from MCMC fit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPosteriorInfo {
    pub mu_post: f64,
    pub sigma_post: f64,
    pub n_draws: usize,
}

/// Prior-posterior comparison metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPriorPosteriorComparison {
    pub system_name: String,
    pub param_name: String,
    pub prior_info: QuantumPriorInfo,
    pub posterior_info: QuantumPosteriorInfo,

    /// Z-score: (μ_post - μ_prior) / σ_prior
    pub z_prior: f64,

    /// Overlap of 95% credible intervals
    pub overlap_95: f64,

    /// KL divergence from prior to posterior
    pub kl_prior_to_post: f64,
}

impl QuantumPriorPosteriorComparison {
    /// Create a new comparison (stub implementation)
    pub fn new(
        system_name: String,
        param_name: String,
        prior_info: QuantumPriorInfo,
        posterior_info: QuantumPosteriorInfo,
    ) -> Self {
        // Calculate z-score
        let z_prior = if prior_info.sigma_prior > 0.0 {
            (posterior_info.mu_post - prior_info.mu_prior) / prior_info.sigma_prior
        } else {
            0.0
        };

        // Approximate overlap (simplified Gaussian overlap)
        let overlap_95 = calculate_overlap(&prior_info, &posterior_info);

        // Approximate KL divergence (Gaussian)
        let kl_prior_to_post = calculate_kl(&prior_info, &posterior_info);

        Self {
            system_name,
            param_name,
            prior_info,
            posterior_info,
            z_prior,
            overlap_95,
            kl_prior_to_post,
        }
    }
}

/// Calculate approximate overlap of 95% CIs (simplified)
fn calculate_overlap(prior: &QuantumPriorInfo, post: &QuantumPosteriorInfo) -> f64 {
    // 95% CI: μ ± 1.96σ
    let prior_low = prior.mu_prior - 1.96 * prior.sigma_prior;
    let prior_high = prior.mu_prior + 1.96 * prior.sigma_prior;
    let post_low = post.mu_post - 1.96 * post.sigma_post;
    let post_high = post.mu_post + 1.96 * post.sigma_post;

    // Calculate overlap
    let overlap_low = prior_low.max(post_low);
    let overlap_high = prior_high.min(post_high);

    if overlap_high <= overlap_low {
        return 0.0;
    }

    let overlap_width = overlap_high - overlap_low;
    let prior_width = prior_high - prior_low;
    let post_width = post_high - post_low;

    // Overlap as fraction of smaller interval
    let min_width = prior_width.min(post_width);
    if min_width > 0.0 {
        (overlap_width / min_width).min(1.0)
    } else {
        0.0
    }
}

/// Calculate KL divergence from prior to posterior (Gaussian approximation)
fn calculate_kl(prior: &QuantumPriorInfo, post: &QuantumPosteriorInfo) -> f64 {
    // KL(P||Q) = log(σ_Q/σ_P) + (σ_P² + (μ_P - μ_Q)²)/(2σ_Q²) - 1/2
    let sigma_p = prior.sigma_prior;
    let sigma_q = post.sigma_post;
    let mu_p = prior.mu_prior;
    let mu_q = post.mu_post;

    if sigma_p <= 0.0 || sigma_q <= 0.0 {
        return 0.0;
    }

    let term1 = (sigma_q / sigma_p).ln();
    let term2 = (sigma_p.powi(2) + (mu_p - mu_q).powi(2)) / (2.0 * sigma_q.powi(2));
    let kl = term1 + term2 - 0.5;

    kl.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_comparison_perfect_agreement() {
        let prior = QuantumPriorInfo {
            mu_prior: 0.0,
            sigma_prior: 1.0,
            scale: "log".to_string(),
        };

        let post = QuantumPosteriorInfo {
            mu_post: 0.0,
            sigma_post: 1.0,
            n_draws: 1000,
        };

        let comp = QuantumPriorPosteriorComparison::new(
            "SYS".to_string(),
            "PARAM".to_string(),
            prior,
            post,
        );

        assert!((comp.z_prior).abs() < 0.01);
        assert!(comp.overlap_95 > 0.99);
        assert!(comp.kl_prior_to_post < 0.01);
    }

    #[test]
    fn test_quantum_comparison_shift() {
        let prior = QuantumPriorInfo {
            mu_prior: 0.0,
            sigma_prior: 1.0,
            scale: "log".to_string(),
        };

        let post = QuantumPosteriorInfo {
            mu_post: 2.0, // 2 SD shift
            sigma_post: 1.0,
            n_draws: 1000,
        };

        let comp = QuantumPriorPosteriorComparison::new(
            "SYS".to_string(),
            "PARAM".to_string(),
            prior,
            post,
        );

        assert!((comp.z_prior - 2.0).abs() < 0.01);
        assert!(comp.overlap_95 < 0.5); // Significant shift reduces overlap
        assert!(comp.kl_prior_to_post > 0.1); // Non-zero KL
    }

    #[test]
    fn test_quantum_comparison_no_overlap() {
        let prior = QuantumPriorInfo {
            mu_prior: 0.0,
            sigma_prior: 0.5,
            scale: "log".to_string(),
        };

        let post = QuantumPosteriorInfo {
            mu_post: 10.0, // Very far
            sigma_post: 0.5,
            n_draws: 1000,
        };

        let comp = QuantumPriorPosteriorComparison::new(
            "SYS".to_string(),
            "PARAM".to_string(),
            prior,
            post,
        );

        assert!(comp.z_prior.abs() > 10.0);
        assert!(comp.overlap_95 < 0.01);
        assert!(comp.kl_prior_to_post > 1.0);
    }
}
