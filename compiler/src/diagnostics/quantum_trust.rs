//! Quantum Trust Scoring & Adaptive Prior Weighting
//!
//! Maps quantum prior-posterior diagnostics into per-parameter trust scores,
//! enabling adaptive model configuration and design sensitivity analysis.

use crate::diagnostics::quantum::QuantumPriorPosteriorComparison;
use serde::{Deserialize, Serialize};

// =============================================================================
// Trust Level Classification
// =============================================================================

/// Trust level for a quantum-derived prior based on prior-posterior agreement
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumTrustLevel {
    /// Excellent agreement: z < 1.0, overlap > 0.8, KL < 0.05
    High,

    /// Good agreement: z < 2.0, overlap > 0.6, KL < 0.2
    Moderate,

    /// Weak agreement: z < 3.5, overlap > 0.3
    Low,

    /// No agreement: z >= 3.5 or overlap <= 0.3
    Broken,
}

impl std::fmt::Display for QuantumTrustLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantumTrustLevel::High => write!(f, "High"),
            QuantumTrustLevel::Moderate => write!(f, "Moderate"),
            QuantumTrustLevel::Low => write!(f, "Low"),
            QuantumTrustLevel::Broken => write!(f, "Broken"),
        }
    }
}

/// Per-parameter quantum trust score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTrustScore {
    pub system_name: String,
    pub param_name: String,
    pub trust: QuantumTrustLevel,
    pub z_prior: f64,
    pub overlap_95: f64,
    pub kl_prior_to_post: f64,
}

/// Classify trust level based on prior-posterior comparison metrics
pub fn classify_trust(c: &QuantumPriorPosteriorComparison) -> QuantumTrustScore {
    let z = c.z_prior.abs();
    let overlap = c.overlap_95;
    let kl = c.kl_prior_to_post;

    let trust = if z < 1.0 && overlap > 0.8 && kl < 0.05 {
        QuantumTrustLevel::High
    } else if z < 2.0 && overlap > 0.6 && kl < 0.2 {
        QuantumTrustLevel::Moderate
    } else if z < 3.5 && overlap > 0.3 {
        QuantumTrustLevel::Low
    } else {
        QuantumTrustLevel::Broken
    };

    QuantumTrustScore {
        system_name: c.system_name.clone(),
        param_name: c.param_name.clone(),
        trust,
        z_prior: c.z_prior,
        overlap_95: c.overlap_95,
        kl_prior_to_post: c.kl_prior_to_post,
    }
}

/// Classify trust for all quantum prior-posterior comparisons
pub fn classify_all(comps: &[QuantumPriorPosteriorComparison]) -> Vec<QuantumTrustScore> {
    comps.iter().map(classify_trust).collect()
}

// =============================================================================
// Prior Inflation Policy
// =============================================================================

/// Policy for inflating prior SDs based on trust level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriorInflationPolicy {
    /// Factor for high trust parameters (default: 1.0 = no inflation)
    pub high_sd_factor: f64,

    /// Factor for moderate trust parameters (default: 2.0)
    pub moderate_sd_factor: f64,

    /// Factor for low trust parameters (default: 4.0)
    pub low_sd_factor: f64,

    /// Factor for broken trust parameters (default: 8.0)
    pub broken_sd_factor: f64,
}

impl Default for PriorInflationPolicy {
    fn default() -> Self {
        Self {
            high_sd_factor: 1.0,
            moderate_sd_factor: 2.0,
            low_sd_factor: 4.0,
            broken_sd_factor: 8.0,
        }
    }
}

impl PriorInflationPolicy {
    /// Get inflation factor for a given trust level
    pub fn inflation_factor(&self, trust: &QuantumTrustLevel) -> f64 {
        match trust {
            QuantumTrustLevel::High => self.high_sd_factor,
            QuantumTrustLevel::Moderate => self.moderate_sd_factor,
            QuantumTrustLevel::Low => self.low_sd_factor,
            QuantumTrustLevel::Broken => self.broken_sd_factor,
        }
    }
}

// =============================================================================
// Prior Configuration Kinds
// =============================================================================

/// Kind of quantum prior configuration for sensitivity analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantumPriorConfigKind {
    /// Original QM priors as specified
    Original,

    /// Relaxed priors based on inflation policy
    Relaxed,

    /// Weak/generic priors (ignore QM)
    Weak,
}

impl std::fmt::Display for QuantumPriorConfigKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantumPriorConfigKind::Original => write!(f, "Original"),
            QuantumPriorConfigKind::Relaxed => write!(f, "Relaxed"),
            QuantumPriorConfigKind::Weak => write!(f, "Weak"),
        }
    }
}

/// Summary of quantum prior configuration for a sensitivity scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumPriorConfigSummary {
    pub kind: QuantumPriorConfigKind,
    pub description: String,

    /// Per-parameter SD inflation factors: (system, param, factor)
    pub per_param_sd_factor: Vec<(String, String, f64)>,
}

impl QuantumPriorConfigSummary {
    /// Create original configuration (no inflation)
    pub fn original() -> Self {
        Self {
            kind: QuantumPriorConfigKind::Original,
            description: "Original QM priors as specified".to_string(),
            per_param_sd_factor: Vec::new(),
        }
    }

    /// Create relaxed configuration based on trust scores
    pub fn relaxed(scores: &[QuantumTrustScore], policy: &PriorInflationPolicy) -> Self {
        let per_param_sd_factor = scores
            .iter()
            .map(|s| {
                let factor = policy.inflation_factor(&s.trust);
                (s.system_name.clone(), s.param_name.clone(), factor)
            })
            .collect();

        Self {
            kind: QuantumPriorConfigKind::Relaxed,
            description: "QM priors with trust-based SD inflation".to_string(),
            per_param_sd_factor,
        }
    }

    /// Create weak configuration (generic broad priors)
    pub fn weak() -> Self {
        Self {
            kind: QuantumPriorConfigKind::Weak,
            description: "Generic weakly-informative priors (QM ignored)".to_string(),
            per_param_sd_factor: Vec::new(),
        }
    }
}

// =============================================================================
// Quantum Trust Report
// =============================================================================

/// Complete quantum trust report for a population model + fit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTrustReport {
    pub population_model: String,
    pub fit_dir: String,
    pub scores: Vec<QuantumTrustScore>,
    pub policy: PriorInflationPolicy,

    /// Summary statistics
    pub summary: QuantumTrustSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumTrustSummary {
    pub total_params: usize,
    pub high_trust: usize,
    pub moderate_trust: usize,
    pub low_trust: usize,
    pub broken_trust: usize,
    pub overall_trust_rate: f64, // (high + moderate) / total
}

impl QuantumTrustSummary {
    pub fn from_scores(scores: &[QuantumTrustScore]) -> Self {
        let total = scores.len();
        let mut high = 0;
        let mut moderate = 0;
        let mut low = 0;
        let mut broken = 0;

        for s in scores {
            match s.trust {
                QuantumTrustLevel::High => high += 1,
                QuantumTrustLevel::Moderate => moderate += 1,
                QuantumTrustLevel::Low => low += 1,
                QuantumTrustLevel::Broken => broken += 1,
            }
        }

        let overall_trust_rate = if total > 0 {
            (high + moderate) as f64 / total as f64
        } else {
            0.0
        };

        Self {
            total_params: total,
            high_trust: high,
            moderate_trust: moderate,
            low_trust: low,
            broken_trust: broken,
            overall_trust_rate,
        }
    }
}

/// Build complete quantum trust report
pub fn build_trust_report(
    population_model_name: &str,
    fit_dir: &str,
    comps: &[QuantumPriorPosteriorComparison],
    policy: &PriorInflationPolicy,
) -> QuantumTrustReport {
    let scores = classify_all(comps);
    let summary = QuantumTrustSummary::from_scores(&scores);

    QuantumTrustReport {
        population_model: population_model_name.to_string(),
        fit_dir: fit_dir.to_string(),
        scores,
        policy: policy.clone(),
        summary,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::diagnostics::quantum::{QuantumPosteriorInfo, QuantumPriorInfo};

    fn make_comparison(
        system: &str,
        param: &str,
        z: f64,
        overlap: f64,
        kl: f64,
    ) -> QuantumPriorPosteriorComparison {
        QuantumPriorPosteriorComparison {
            system_name: system.to_string(),
            param_name: param.to_string(),
            prior_info: QuantumPriorInfo {
                mu_prior: 0.0,
                sigma_prior: 1.0,
                scale: "log".to_string(),
            },
            posterior_info: QuantumPosteriorInfo {
                mu_post: z,
                sigma_post: 1.0,
                n_draws: 1000,
            },
            z_prior: z,
            overlap_95: overlap,
            kl_prior_to_post: kl,
        }
    }

    #[test]
    fn test_trust_classification_high() {
        let comp = make_comparison("SYS", "PARAM", 0.5, 0.9, 0.02);
        let score = classify_trust(&comp);
        assert_eq!(score.trust, QuantumTrustLevel::High);
    }

    #[test]
    fn test_trust_classification_moderate() {
        let comp = make_comparison("SYS", "PARAM", 1.5, 0.7, 0.1);
        let score = classify_trust(&comp);
        assert_eq!(score.trust, QuantumTrustLevel::Moderate);
    }

    #[test]
    fn test_trust_classification_low() {
        let comp = make_comparison("SYS", "PARAM", 2.5, 0.5, 0.3);
        let score = classify_trust(&comp);
        assert_eq!(score.trust, QuantumTrustLevel::Low);
    }

    #[test]
    fn test_trust_classification_broken() {
        let comp = make_comparison("SYS", "PARAM", 4.0, 0.2, 0.8);
        let score = classify_trust(&comp);
        assert_eq!(score.trust, QuantumTrustLevel::Broken);
    }

    #[test]
    fn test_classify_all() {
        let comps = vec![
            make_comparison("SYS1", "P1", 0.5, 0.9, 0.02),
            make_comparison("SYS1", "P2", 1.5, 0.7, 0.1),
            make_comparison("SYS2", "P3", 2.5, 0.5, 0.3),
            make_comparison("SYS2", "P4", 4.0, 0.2, 0.8),
        ];

        let scores = classify_all(&comps);

        assert_eq!(scores.len(), 4);
        assert_eq!(scores[0].trust, QuantumTrustLevel::High);
        assert_eq!(scores[1].trust, QuantumTrustLevel::Moderate);
        assert_eq!(scores[2].trust, QuantumTrustLevel::Low);
        assert_eq!(scores[3].trust, QuantumTrustLevel::Broken);
    }

    #[test]
    fn test_inflation_policy_default() {
        let policy = PriorInflationPolicy::default();

        assert_eq!(policy.inflation_factor(&QuantumTrustLevel::High), 1.0);
        assert_eq!(policy.inflation_factor(&QuantumTrustLevel::Moderate), 2.0);
        assert_eq!(policy.inflation_factor(&QuantumTrustLevel::Low), 4.0);
        assert_eq!(policy.inflation_factor(&QuantumTrustLevel::Broken), 8.0);
    }

    #[test]
    fn test_prior_config_summary_relaxed() {
        let scores = vec![
            QuantumTrustScore {
                system_name: "SYS1".to_string(),
                param_name: "P1".to_string(),
                trust: QuantumTrustLevel::High,
                z_prior: 0.5,
                overlap_95: 0.9,
                kl_prior_to_post: 0.02,
            },
            QuantumTrustScore {
                system_name: "SYS1".to_string(),
                param_name: "P2".to_string(),
                trust: QuantumTrustLevel::Broken,
                z_prior: 4.0,
                overlap_95: 0.2,
                kl_prior_to_post: 0.8,
            },
        ];

        let policy = PriorInflationPolicy::default();
        let config = QuantumPriorConfigSummary::relaxed(&scores, &policy);

        assert_eq!(config.kind, QuantumPriorConfigKind::Relaxed);
        assert_eq!(config.per_param_sd_factor.len(), 2);
        assert_eq!(config.per_param_sd_factor[0].2, 1.0); // High = 1.0
        assert_eq!(config.per_param_sd_factor[1].2, 8.0); // Broken = 8.0
    }

    #[test]
    fn test_trust_summary() {
        let scores = vec![
            QuantumTrustScore {
                system_name: "S1".to_string(),
                param_name: "P1".to_string(),
                trust: QuantumTrustLevel::High,
                z_prior: 0.5,
                overlap_95: 0.9,
                kl_prior_to_post: 0.02,
            },
            QuantumTrustScore {
                system_name: "S1".to_string(),
                param_name: "P2".to_string(),
                trust: QuantumTrustLevel::High,
                z_prior: 0.6,
                overlap_95: 0.85,
                kl_prior_to_post: 0.03,
            },
            QuantumTrustScore {
                system_name: "S1".to_string(),
                param_name: "P3".to_string(),
                trust: QuantumTrustLevel::Moderate,
                z_prior: 1.5,
                overlap_95: 0.7,
                kl_prior_to_post: 0.1,
            },
            QuantumTrustScore {
                system_name: "S2".to_string(),
                param_name: "P4".to_string(),
                trust: QuantumTrustLevel::Broken,
                z_prior: 4.0,
                overlap_95: 0.2,
                kl_prior_to_post: 0.8,
            },
        ];

        let summary = QuantumTrustSummary::from_scores(&scores);

        assert_eq!(summary.total_params, 4);
        assert_eq!(summary.high_trust, 2);
        assert_eq!(summary.moderate_trust, 1);
        assert_eq!(summary.low_trust, 0);
        assert_eq!(summary.broken_trust, 1);
        assert!((summary.overall_trust_rate - 0.75).abs() < 0.01); // (2+1)/4 = 0.75
    }

    #[test]
    fn test_build_trust_report() {
        let comps = vec![
            make_comparison("SYS1", "P1", 0.5, 0.9, 0.02),
            make_comparison("SYS1", "P2", 4.0, 0.2, 0.8),
        ];

        let policy = PriorInflationPolicy::default();
        let report = build_trust_report("PopModel", "fit_dir", &comps, &policy);

        assert_eq!(report.population_model, "PopModel");
        assert_eq!(report.fit_dir, "fit_dir");
        assert_eq!(report.scores.len(), 2);
        assert_eq!(report.summary.total_params, 2);
        assert_eq!(report.summary.high_trust, 1);
        assert_eq!(report.summary.broken_trust, 1);
    }
}
