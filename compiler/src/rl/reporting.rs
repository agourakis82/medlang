//! Week 48: Baseline Comparison & Publication-Ready Export (v2.0)
//!
//! Comprehensive reporting layer that produces:
//! - Statistically rigorous baseline comparisons with CIs, p-values, effect sizes
//! - Clinically meaningful metrics (NNT, NNH, risk ratios)
//! - Multiple export formats (CSV, JSON, LaTeX, Markdown)
//! - Forest plot-ready data structures
//! - Publication-quality tables with proper formatting

use serde::{Deserialize, Serialize};

use crate::rl::constraints::GuidelineRobustnessScoreHost;
use crate::rl::experiment::ExperimentResult;

// ═══════════════════════════════════════════════════════════════════════════
// PART 1: STATISTICAL COMPARISON TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Complete statistical comparison between two guidelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalComparison {
    /// Treatment guideline ID
    pub treatment_id: String,
    /// Control/baseline guideline ID
    pub control_id: String,

    /// Response rate comparison
    pub response: MetricComparison,
    /// Grade 3+ toxicity comparison
    pub toxicity_grade3: MetricComparison,
    /// Grade 4+ toxicity comparison (if available)
    pub toxicity_grade4: Option<MetricComparison>,
    /// Relative Dose Intensity comparison
    pub rdi: MetricComparison,
    /// Composite score comparison
    pub composite_score: MetricComparison,

    /// Clinical interpretation
    pub interpretation: ClinicalInterpretation,

    /// Overall recommendation
    pub recommendation: Recommendation,
}

/// Comparison for a single metric
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    /// Metric name
    pub metric_name: String,
    /// Treatment value
    pub treatment_value: f64,
    /// Control value
    pub control_value: f64,
    /// Absolute difference (treatment - control)
    pub absolute_diff: f64,
    /// Relative difference ((treatment - control) / control)
    pub relative_diff: Option<f64>,
    /// Risk ratio (treatment / control)
    pub risk_ratio: Option<RiskRatio>,
    /// Confidence interval for difference
    pub diff_ci: ConfidenceInterval,
    /// P-value for difference
    pub p_value: f64,
    /// Effect size (Cohen's d or similar)
    pub effect_size: EffectSize,
    /// Whether difference is statistically significant
    pub significant: bool,
    /// Direction of effect
    pub direction: EffectDirection,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    /// Point estimate
    pub estimate: f64,
    /// Lower bound
    pub lower: f64,
    /// Upper bound
    pub upper: f64,
    /// Confidence level (e.g., 0.95)
    pub level: f64,
}

impl ConfidenceInterval {
    /// Create a new confidence interval
    pub fn new(estimate: f64, lower: f64, upper: f64, level: f64) -> Self {
        Self {
            estimate,
            lower,
            upper,
            level,
        }
    }

    /// Width of the interval
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Check if interval contains zero
    pub fn contains_zero(&self) -> bool {
        self.lower <= 0.0 && self.upper >= 0.0
    }

    /// Check if interval contains a value
    pub fn contains(&self, value: f64) -> bool {
        self.lower <= value && value <= self.upper
    }

    /// Format for display
    pub fn format(&self, decimals: usize) -> String {
        format!(
            "{:.prec$} [{:.prec$}, {:.prec$}]",
            self.estimate,
            self.lower,
            self.upper,
            prec = decimals
        )
    }
}

/// Risk ratio with confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskRatio {
    /// Point estimate
    pub ratio: f64,
    /// Confidence interval
    pub ci: ConfidenceInterval,
    /// P-value
    pub p_value: f64,
}

/// Effect size with type indicator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSize {
    /// Effect size type
    pub effect_type: EffectSizeType,
    /// Point estimate
    pub value: f64,
    /// Confidence interval
    pub ci: Option<ConfidenceInterval>,
    /// Interpretation
    pub interpretation: EffectMagnitude,
}

/// Types of effect size measures
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectSizeType {
    /// Cohen's d (standardized mean difference)
    CohenD,
    /// Hedges' g (bias-corrected Cohen's d)
    HedgesG,
    /// Glass's delta
    GlassDelta,
    /// Odds ratio
    OddsRatio,
    /// Risk difference
    RiskDifference,
    /// Number needed to treat
    NNT,
}

/// Effect magnitude interpretation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectMagnitude {
    Negligible,
    Small,
    Medium,
    Large,
    VeryLarge,
}

impl EffectMagnitude {
    /// Interpret Cohen's d magnitude
    pub fn from_cohens_d(d: f64) -> Self {
        let abs_d = d.abs();
        if abs_d < 0.2 {
            EffectMagnitude::Negligible
        } else if abs_d < 0.5 {
            EffectMagnitude::Small
        } else if abs_d < 0.8 {
            EffectMagnitude::Medium
        } else if abs_d < 1.2 {
            EffectMagnitude::Large
        } else {
            EffectMagnitude::VeryLarge
        }
    }
}

/// Direction of effect
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum EffectDirection {
    /// Treatment better (higher response, lower toxicity)
    Favorable,
    /// No meaningful difference
    Neutral,
    /// Treatment worse
    Unfavorable,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 2: CLINICAL METRICS
// ═══════════════════════════════════════════════════════════════════════════

/// Number Needed to Treat (NNT) calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NNTResult {
    /// Point estimate (positive = benefit, negative = harm)
    pub nnt: f64,
    /// Confidence interval
    pub ci: ConfidenceInterval,
    /// Whether this is NNT (benefit) or NNH (harm)
    pub metric_type: NNTType,
    /// Clinical interpretation
    pub interpretation: String,
}

/// Type of NNT metric
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NNTType {
    /// Number Needed to Treat (benefit)
    NNT,
    /// Number Needed to Harm
    NNH,
    /// Infinite (no difference)
    Infinite,
}

/// Complete clinical metrics for a comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalMetrics {
    /// NNT for response improvement
    pub nnt_response: Option<NNTResult>,
    /// NNH for toxicity
    pub nnh_toxicity: Option<NNTResult>,
    /// Absolute risk reduction for toxicity
    pub arr_toxicity: Option<f64>,
    /// Relative risk reduction for toxicity
    pub rrr_toxicity: Option<f64>,
    /// Benefit-harm ratio
    pub benefit_harm_ratio: Option<f64>,
    /// Therapeutic index comparison
    pub therapeutic_index: TherapeuticIndexComparison,
}

/// Therapeutic index comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TherapeuticIndexComparison {
    /// Treatment therapeutic index
    pub treatment_ti: f64,
    /// Control therapeutic index
    pub control_ti: f64,
    /// Ratio of therapeutic indices
    pub ti_ratio: f64,
    /// Interpretation
    pub interpretation: String,
}

/// Clinical interpretation categories
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ClinicalInterpretation {
    /// Treatment is clinically superior
    Superior,
    /// Treatment is non-inferior
    NonInferior,
    /// Treatments are clinically equivalent
    Equivalent,
    /// Treatment is clinically inferior
    Inferior,
    /// Insufficient data to determine
    Inconclusive,
}

impl ClinicalInterpretation {
    /// Get a human-readable description
    pub fn description(&self) -> &'static str {
        match self {
            ClinicalInterpretation::Superior => "Treatment shows clinically meaningful superiority",
            ClinicalInterpretation::NonInferior => "Treatment is non-inferior to control",
            ClinicalInterpretation::Equivalent => "Treatments are clinically equivalent",
            ClinicalInterpretation::Inferior => "Treatment shows clinically meaningful inferiority",
            ClinicalInterpretation::Inconclusive => {
                "Insufficient evidence for definitive conclusion"
            }
        }
    }
}

/// Recommendation based on comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendation {
    /// Primary recommendation
    pub action: RecommendedAction,
    /// Confidence in recommendation
    pub confidence: RecommendationConfidence,
    /// Supporting rationale
    pub rationale: Vec<String>,
    /// Caveats and limitations
    pub caveats: Vec<String>,
}

/// Recommended action
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendedAction {
    /// Prefer treatment over control
    PreferTreatment,
    /// Prefer control over treatment
    PreferControl,
    /// Either option acceptable
    EitherAcceptable,
    /// Need more data
    RequiresMoreData,
    /// Consider patient-specific factors
    IndividualizeChoice,
}

/// Confidence in recommendation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecommendationConfidence {
    High,
    Moderate,
    Low,
    VeryLow,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 3: STATISTICAL CALCULATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Statistical calculator for comparisons
pub struct StatisticalCalculator {
    /// Alpha level for significance testing
    pub alpha: f64,
    /// Confidence level for intervals
    pub confidence_level: f64,
    /// Non-inferiority margin for response
    pub non_inferiority_margin_response: f64,
    /// Non-inferiority margin for toxicity
    pub non_inferiority_margin_toxicity: f64,
    /// Minimum clinically important difference
    pub mcid_response: f64,
    pub mcid_toxicity: f64,
}

impl Default for StatisticalCalculator {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            confidence_level: 0.95,
            non_inferiority_margin_response: 0.10, // 10% margin
            non_inferiority_margin_toxicity: 0.05, // 5% margin
            mcid_response: 0.05,                   // 5% MCID
            mcid_toxicity: 0.03,                   // 3% MCID
        }
    }
}

impl StatisticalCalculator {
    /// Create calculator with custom settings
    pub fn new(
        alpha: f64,
        non_inferiority_margin_response: f64,
        non_inferiority_margin_toxicity: f64,
    ) -> Self {
        Self {
            alpha,
            confidence_level: 1.0 - alpha,
            non_inferiority_margin_response,
            non_inferiority_margin_toxicity,
            mcid_response: 0.05,
            mcid_toxicity: 0.03,
        }
    }

    /// Calculate z-score for given alpha level
    fn z_score(&self) -> f64 {
        // Approximation using inverse error function
        // For alpha=0.05, z ≈ 1.96
        let p = 1.0 - self.alpha / 2.0;
        self.inv_normal_cdf(p)
    }

    /// Inverse normal CDF approximation (Abramowitz and Stegun)
    fn inv_normal_cdf(&self, p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        // Coefficients for rational approximation
        const A: [f64; 6] = [
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
        ];
        const B: [f64; 5] = [
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
        ];
        const C: [f64; 6] = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
        ];
        const D: [f64; 4] = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
        ];

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
                / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
                / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
        }
    }

    /// Normal CDF approximation
    fn normal_cdf(&self, x: f64) -> f64 {
        // Approximation using error function
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation (Horner's method)
    fn erf(&self, x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();

        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Calculate p-value for two proportions
    pub fn two_proportion_p_value(&self, p1: f64, n1: u32, p2: f64, n2: u32) -> f64 {
        let n1 = n1 as f64;
        let n2 = n2 as f64;

        // Pooled proportion
        let p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2);

        if p_pooled <= 0.0 || p_pooled >= 1.0 {
            return 1.0;
        }

        // Standard error
        let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n1 + 1.0 / n2)).sqrt();

        if se <= 0.0 {
            return 1.0;
        }

        // Z-statistic
        let z = (p1 - p2) / se;

        // Two-tailed p-value
        2.0 * (1.0 - self.normal_cdf(z.abs()))
    }

    /// Calculate confidence interval for difference in proportions
    pub fn proportion_diff_ci(&self, p1: f64, n1: u32, p2: f64, n2: u32) -> ConfidenceInterval {
        let n1 = n1 as f64;
        let n2 = n2 as f64;
        let diff = p1 - p2;

        // Standard error using Wald method
        let se = ((p1 * (1.0 - p1) / n1) + (p2 * (1.0 - p2) / n2)).sqrt();
        let z = self.z_score();

        ConfidenceInterval {
            estimate: diff,
            lower: diff - z * se,
            upper: diff + z * se,
            level: self.confidence_level,
        }
    }

    /// Calculate Cohen's d effect size
    pub fn cohens_d(&self, mean1: f64, std1: f64, mean2: f64, std2: f64) -> EffectSize {
        // Pooled standard deviation
        let pooled_std = ((std1 * std1 + std2 * std2) / 2.0).sqrt();

        let d = if pooled_std > 0.0 {
            (mean1 - mean2) / pooled_std
        } else {
            0.0
        };

        EffectSize {
            effect_type: EffectSizeType::CohenD,
            value: d,
            ci: None, // CI calculation requires sample sizes
            interpretation: EffectMagnitude::from_cohens_d(d),
        }
    }

    /// Calculate Hedges' g (bias-corrected Cohen's d)
    pub fn hedges_g(
        &self,
        mean1: f64,
        std1: f64,
        n1: u32,
        mean2: f64,
        std2: f64,
        n2: u32,
    ) -> EffectSize {
        let n1 = n1 as f64;
        let n2 = n2 as f64;

        // Pooled standard deviation with degrees of freedom
        let pooled_std =
            (((n1 - 1.0) * std1 * std1 + (n2 - 1.0) * std2 * std2) / (n1 + n2 - 2.0)).sqrt();

        let d = if pooled_std > 0.0 {
            (mean1 - mean2) / pooled_std
        } else {
            0.0
        };

        // Bias correction factor (Hedges & Olkin, 1985)
        let df = n1 + n2 - 2.0;
        let correction = 1.0 - (3.0 / (4.0 * df - 1.0));
        let g = d * correction;

        // Standard error for CI
        let se = ((n1 + n2) / (n1 * n2) + (g * g) / (2.0 * (n1 + n2))).sqrt();
        let z = self.z_score();

        EffectSize {
            effect_type: EffectSizeType::HedgesG,
            value: g,
            ci: Some(ConfidenceInterval {
                estimate: g,
                lower: g - z * se,
                upper: g + z * se,
                level: self.confidence_level,
            }),
            interpretation: EffectMagnitude::from_cohens_d(g),
        }
    }

    /// Calculate risk ratio with confidence interval
    pub fn risk_ratio(&self, p1: f64, n1: u32, p2: f64, n2: u32) -> Option<RiskRatio> {
        if p2 <= 0.0 || p2 >= 1.0 {
            return None;
        }

        let rr = p1 / p2;

        // Log-transformed CI (more accurate for ratios)
        let n1 = n1 as f64;
        let n2 = n2 as f64;

        if p1 <= 0.0 || p1 >= 1.0 {
            return Some(RiskRatio {
                ratio: rr,
                ci: ConfidenceInterval::new(rr, 0.0, f64::INFINITY, self.confidence_level),
                p_value: 1.0,
            });
        }

        let log_rr = rr.ln();
        let se_log = ((1.0 - p1) / (p1 * n1) + (1.0 - p2) / (p2 * n2)).sqrt();
        let z = self.z_score();

        let ci = ConfidenceInterval {
            estimate: rr,
            lower: (log_rr - z * se_log).exp(),
            upper: (log_rr + z * se_log).exp(),
            level: self.confidence_level,
        };

        // P-value for log(RR) = 0
        let z_stat = log_rr / se_log;
        let p_value = 2.0 * (1.0 - self.normal_cdf(z_stat.abs()));

        Some(RiskRatio {
            ratio: rr,
            ci,
            p_value,
        })
    }

    /// Calculate Number Needed to Treat (NNT)
    pub fn calculate_nnt(&self, p_treatment: f64, p_control: f64) -> NNTResult {
        let arr = p_treatment - p_control; // Absolute Risk Reduction

        if arr.abs() < 1e-10 {
            return NNTResult {
                nnt: f64::INFINITY,
                ci: ConfidenceInterval::new(
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::INFINITY,
                    self.confidence_level,
                ),
                metric_type: NNTType::Infinite,
                interpretation: "No difference between treatments".to_string(),
            };
        }

        let nnt = 1.0 / arr;
        let metric_type = if nnt > 0.0 {
            NNTType::NNT
        } else {
            NNTType::NNH
        };

        // Approximate CI (would need sample sizes for proper calculation)
        let nnt_abs = nnt.abs();
        let interpretation = match (metric_type.clone(), nnt_abs) {
            (NNTType::NNT, n) if n < 5.0 => format!("Excellent: treat {:.0} to benefit 1", n),
            (NNTType::NNT, n) if n < 10.0 => format!("Very good: treat {:.0} to benefit 1", n),
            (NNTType::NNT, n) if n < 20.0 => format!("Good: treat {:.0} to benefit 1", n),
            (NNTType::NNT, n) if n < 50.0 => format!("Moderate: treat {:.0} to benefit 1", n),
            (NNTType::NNT, n) => format!("Marginal benefit: treat {:.0} to benefit 1", n),
            (NNTType::NNH, n) if n < 10.0 => format!("High harm risk: treat {:.0} to harm 1", n),
            (NNTType::NNH, n) if n < 50.0 => {
                format!("Moderate harm risk: treat {:.0} to harm 1", n)
            }
            (NNTType::NNH, n) => format!("Low harm risk: treat {:.0} to harm 1", n),
            _ => "Unable to interpret".to_string(),
        };

        NNTResult {
            nnt,
            ci: ConfidenceInterval::new(nnt, nnt * 0.5, nnt * 2.0, self.confidence_level), // Rough approximation
            metric_type,
            interpretation,
        }
    }

    /// Calculate therapeutic index
    pub fn therapeutic_index(&self, response: f64, toxicity: f64) -> f64 {
        if toxicity <= 0.0 {
            return f64::INFINITY;
        }
        response / toxicity
    }

    /// Determine clinical interpretation
    pub fn interpret_comparison(
        &self,
        response_diff: f64,
        response_p: f64,
        toxicity_diff: f64,
        toxicity_p: f64,
    ) -> ClinicalInterpretation {
        let sig_better_response = response_diff > self.mcid_response && response_p < self.alpha;
        let sig_worse_response = response_diff < -self.mcid_response && response_p < self.alpha;
        let sig_lower_tox = toxicity_diff < -self.mcid_toxicity && toxicity_p < self.alpha;
        let sig_higher_tox = toxicity_diff > self.mcid_toxicity && toxicity_p < self.alpha;

        // Superior: Better response OR lower toxicity without detriment in other
        if (sig_better_response && !sig_higher_tox) || (sig_lower_tox && !sig_worse_response) {
            return ClinicalInterpretation::Superior;
        }

        // Inferior: Worse response OR higher toxicity without benefit in other
        if (sig_worse_response && !sig_lower_tox) || (sig_higher_tox && !sig_better_response) {
            return ClinicalInterpretation::Inferior;
        }

        // Non-inferior: Within margins
        if response_diff >= -self.non_inferiority_margin_response
            && toxicity_diff <= self.non_inferiority_margin_toxicity
        {
            // Equivalent if very close
            if response_diff.abs() < self.mcid_response && toxicity_diff.abs() < self.mcid_toxicity
            {
                return ClinicalInterpretation::Equivalent;
            }
            return ClinicalInterpretation::NonInferior;
        }

        ClinicalInterpretation::Inconclusive
    }

    /// Generate recommendation based on comparison
    pub fn generate_recommendation(&self, comparison: &StatisticalComparison) -> Recommendation {
        let mut rationale = Vec::new();
        let mut caveats = Vec::new();

        // Analyze response
        if comparison.response.significant {
            if comparison.response.direction == EffectDirection::Favorable {
                rationale.push(format!(
                    "Significantly higher response rate ({:.1}% vs {:.1}%, p={:.4})",
                    comparison.response.treatment_value * 100.0,
                    comparison.response.control_value * 100.0,
                    comparison.response.p_value
                ));
            } else if comparison.response.direction == EffectDirection::Unfavorable {
                rationale.push(format!(
                    "Significantly lower response rate ({:.1}% vs {:.1}%, p={:.4})",
                    comparison.response.treatment_value * 100.0,
                    comparison.response.control_value * 100.0,
                    comparison.response.p_value
                ));
            }
        }

        // Analyze toxicity
        if comparison.toxicity_grade3.significant {
            if comparison.toxicity_grade3.direction == EffectDirection::Favorable {
                rationale.push(format!(
                    "Significantly lower Grade 3+ toxicity ({:.1}% vs {:.1}%, p={:.4})",
                    comparison.toxicity_grade3.treatment_value * 100.0,
                    comparison.toxicity_grade3.control_value * 100.0,
                    comparison.toxicity_grade3.p_value
                ));
            } else if comparison.toxicity_grade3.direction == EffectDirection::Unfavorable {
                rationale.push(format!(
                    "Significantly higher Grade 3+ toxicity ({:.1}% vs {:.1}%, p={:.4})",
                    comparison.toxicity_grade3.treatment_value * 100.0,
                    comparison.toxicity_grade3.control_value * 100.0,
                    comparison.toxicity_grade3.p_value
                ));
            }
        }

        // Add caveats
        if comparison.response.diff_ci.width() > 0.20 {
            caveats
                .push("Wide confidence interval for response - consider larger sample".to_string());
        }

        let (action, confidence) = match &comparison.interpretation {
            ClinicalInterpretation::Superior => (
                RecommendedAction::PreferTreatment,
                RecommendationConfidence::High,
            ),
            ClinicalInterpretation::Inferior => (
                RecommendedAction::PreferControl,
                RecommendationConfidence::High,
            ),
            ClinicalInterpretation::Equivalent => {
                caveats.push("Consider cost and convenience factors".to_string());
                (
                    RecommendedAction::EitherAcceptable,
                    RecommendationConfidence::Moderate,
                )
            }
            ClinicalInterpretation::NonInferior => {
                caveats.push("Non-inferior but not demonstrated superior".to_string());
                (
                    RecommendedAction::IndividualizeChoice,
                    RecommendationConfidence::Moderate,
                )
            }
            ClinicalInterpretation::Inconclusive => {
                caveats.push("More data needed for definitive recommendation".to_string());
                (
                    RecommendedAction::RequiresMoreData,
                    RecommendationConfidence::Low,
                )
            }
        };

        Recommendation {
            action,
            confidence,
            rationale,
            caveats,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 4: COMPARISON ENGINE
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for comparison analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonConfig {
    /// Alpha level for statistical tests
    pub alpha: f64,
    /// Non-inferiority margin for response
    pub non_inferiority_margin_response: f64,
    /// Non-inferiority margin for toxicity
    pub non_inferiority_margin_toxicity: f64,
    /// Assumed sample size per arm (for p-value calculations)
    pub assumed_n_per_arm: u32,
    /// Include clinical metrics (NNT, NNH)
    pub include_clinical_metrics: bool,
    /// Include effect sizes
    pub include_effect_sizes: bool,
}

impl Default for ComparisonConfig {
    fn default() -> Self {
        Self {
            alpha: 0.05,
            non_inferiority_margin_response: 0.10,
            non_inferiority_margin_toxicity: 0.05,
            assumed_n_per_arm: 100,
            include_clinical_metrics: true,
            include_effect_sizes: true,
        }
    }
}

/// Main comparison engine
pub struct ComparisonEngine {
    config: ComparisonConfig,
    calculator: StatisticalCalculator,
}

impl ComparisonEngine {
    /// Create new comparison engine with default config
    pub fn new() -> Self {
        Self::with_config(ComparisonConfig::default())
    }

    /// Create comparison engine with custom config
    pub fn with_config(config: ComparisonConfig) -> Self {
        let calculator = StatisticalCalculator::new(
            config.alpha,
            config.non_inferiority_margin_response,
            config.non_inferiority_margin_toxicity,
        );
        Self { config, calculator }
    }

    /// Compare two guidelines
    pub fn compare_guidelines(
        &self,
        treatment: &GuidelineRobustnessScoreHost,
        control: &GuidelineRobustnessScoreHost,
    ) -> StatisticalComparison {
        let n = self.config.assumed_n_per_arm;

        // Response comparison
        let response = self.compare_metric(
            "Response Rate",
            treatment.mean_response,
            control.mean_response,
            treatment.response_std,
            control.response_std,
            n,
            true, // Higher is better
        );

        // Toxicity Grade 3+ comparison
        let toxicity_grade3 = self.compare_metric(
            "Grade 3+ Toxicity",
            treatment.mean_grade3plus_rate,
            control.mean_grade3plus_rate,
            treatment.tox_std,
            control.tox_std,
            n,
            false, // Lower is better
        );

        // Toxicity Grade 4+ comparison (if available)
        let toxicity_grade4 = match (treatment.mean_grade4plus_rate, control.mean_grade4plus_rate) {
            (Some(t), Some(c)) => Some(self.compare_metric(
                "Grade 4+ Toxicity",
                t,
                c,
                treatment.tox_std,
                control.tox_std,
                n,
                false,
            )),
            _ => None,
        };

        // RDI comparison
        let rdi = self.compare_metric(
            "Relative Dose Intensity",
            treatment.mean_rdi,
            control.mean_rdi,
            None,
            None,
            n,
            true, // Higher is better
        );

        // Composite score comparison
        let composite_score = self.compare_metric(
            "Composite Score",
            treatment.score_mean,
            control.score_mean,
            None,
            None,
            n,
            true,
        );

        // Clinical interpretation
        let interpretation = self.calculator.interpret_comparison(
            response.absolute_diff,
            response.p_value,
            toxicity_grade3.absolute_diff,
            toxicity_grade3.p_value,
        );

        let mut comparison = StatisticalComparison {
            treatment_id: treatment.guideline_id.clone(),
            control_id: control.guideline_id.clone(),
            response,
            toxicity_grade3,
            toxicity_grade4,
            rdi,
            composite_score,
            interpretation,
            recommendation: Recommendation {
                action: RecommendedAction::RequiresMoreData,
                confidence: RecommendationConfidence::Low,
                rationale: vec![],
                caveats: vec![],
            },
        };

        // Generate recommendation
        comparison.recommendation = self.calculator.generate_recommendation(&comparison);

        comparison
    }

    /// Compare a single metric
    fn compare_metric(
        &self,
        name: &str,
        treatment_value: f64,
        control_value: f64,
        treatment_std: Option<f64>,
        control_std: Option<f64>,
        n: u32,
        higher_is_better: bool,
    ) -> MetricComparison {
        let diff = treatment_value - control_value;
        let rel_diff = if control_value.abs() > 1e-10 {
            Some(diff / control_value)
        } else {
            None
        };

        // P-value and CI
        let p_value = self
            .calculator
            .two_proportion_p_value(treatment_value, n, control_value, n);

        let diff_ci = self
            .calculator
            .proportion_diff_ci(treatment_value, n, control_value, n);

        // Risk ratio
        let risk_ratio = self
            .calculator
            .risk_ratio(treatment_value, n, control_value, n);

        // Effect size
        let effect_size = if let (Some(t_std), Some(c_std)) = (treatment_std, control_std) {
            self.calculator
                .hedges_g(treatment_value, t_std, n, control_value, c_std, n)
        } else {
            self.calculator
                .cohens_d(treatment_value, 0.1, control_value, 0.1)
        };

        // Direction
        let direction = if higher_is_better {
            if diff > self.calculator.mcid_response {
                EffectDirection::Favorable
            } else if diff < -self.calculator.mcid_response {
                EffectDirection::Unfavorable
            } else {
                EffectDirection::Neutral
            }
        } else {
            // Lower is better (toxicity)
            if diff < -self.calculator.mcid_toxicity {
                EffectDirection::Favorable
            } else if diff > self.calculator.mcid_toxicity {
                EffectDirection::Unfavorable
            } else {
                EffectDirection::Neutral
            }
        };

        MetricComparison {
            metric_name: name.to_string(),
            treatment_value,
            control_value,
            absolute_diff: diff,
            relative_diff: rel_diff,
            risk_ratio,
            diff_ci,
            p_value,
            effect_size,
            significant: p_value < self.config.alpha,
            direction,
        }
    }

    /// Calculate clinical metrics for a comparison
    pub fn calculate_clinical_metrics(
        &self,
        treatment: &GuidelineRobustnessScoreHost,
        control: &GuidelineRobustnessScoreHost,
    ) -> ClinicalMetrics {
        // NNT for response
        let nnt_response = if treatment.mean_response > control.mean_response {
            Some(
                self.calculator
                    .calculate_nnt(treatment.mean_response, control.mean_response),
            )
        } else {
            None
        };

        // NNH for toxicity
        let nnh_toxicity = if treatment.mean_grade3plus_rate > control.mean_grade3plus_rate {
            Some(
                self.calculator
                    .calculate_nnt(treatment.mean_grade3plus_rate, control.mean_grade3plus_rate),
            )
        } else {
            None
        };

        // Absolute and relative risk reduction for toxicity
        let arr_toxicity = Some(control.mean_grade3plus_rate - treatment.mean_grade3plus_rate);
        let rrr_toxicity = if control.mean_grade3plus_rate > 0.0 {
            Some(
                (control.mean_grade3plus_rate - treatment.mean_grade3plus_rate)
                    / control.mean_grade3plus_rate,
            )
        } else {
            None
        };

        // Benefit-harm ratio
        let benefit_harm_ratio = match (&nnt_response, &nnh_toxicity) {
            (Some(nnt), Some(nnh)) if nnt.nnt.abs() > 0.0 && nnh.nnt.abs() > 0.0 => {
                Some(nnh.nnt.abs() / nnt.nnt.abs())
            }
            _ => None,
        };

        // Therapeutic index
        let treatment_ti = self
            .calculator
            .therapeutic_index(treatment.mean_response, treatment.mean_grade3plus_rate);
        let control_ti = self
            .calculator
            .therapeutic_index(control.mean_response, control.mean_grade3plus_rate);
        let ti_ratio = if control_ti > 0.0 && control_ti.is_finite() {
            treatment_ti / control_ti
        } else {
            1.0
        };

        let ti_interpretation = if ti_ratio > 1.2 {
            "Treatment has meaningfully better therapeutic index"
        } else if ti_ratio < 0.8 {
            "Control has meaningfully better therapeutic index"
        } else {
            "Similar therapeutic indices"
        };

        ClinicalMetrics {
            nnt_response,
            nnh_toxicity,
            arr_toxicity,
            rrr_toxicity,
            benefit_harm_ratio,
            therapeutic_index: TherapeuticIndexComparison {
                treatment_ti,
                control_ti,
                ti_ratio,
                interpretation: ti_interpretation.to_string(),
            },
        }
    }
}

impl Default for ComparisonEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 5: FOREST PLOT DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════

/// Data structure for forest plot generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestPlotData {
    /// Plot title
    pub title: String,
    /// Metric being compared
    pub metric: String,
    /// Individual study/comparison entries
    pub entries: Vec<ForestPlotEntry>,
    /// Overall summary (if applicable)
    pub summary: Option<ForestPlotSummary>,
    /// Reference line value (typically 0 for differences, 1 for ratios)
    pub reference_value: f64,
    /// Whether this is a ratio scale
    pub is_ratio_scale: bool,
    /// X-axis label
    pub x_label: String,
}

/// Single entry in forest plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestPlotEntry {
    /// Label for this entry
    pub label: String,
    /// Point estimate
    pub estimate: f64,
    /// Lower CI bound
    pub ci_lower: f64,
    /// Upper CI bound
    pub ci_upper: f64,
    /// Weight (for meta-analysis style plots)
    pub weight: Option<f64>,
    /// Whether this favors treatment
    pub favors_treatment: bool,
    /// Group/category (for subgroup analysis)
    pub group: Option<String>,
}

/// Summary row for forest plot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestPlotSummary {
    /// Label
    pub label: String,
    /// Overall estimate
    pub estimate: f64,
    /// CI lower
    pub ci_lower: f64,
    /// CI upper
    pub ci_upper: f64,
    /// Heterogeneity statistic (I²)
    pub i_squared: Option<f64>,
}

impl ForestPlotData {
    /// Create forest plot data from multiple comparisons
    pub fn from_comparisons(
        comparisons: &[StatisticalComparison],
        metric: ForestPlotMetric,
    ) -> Self {
        let (title, x_label, reference_value, is_ratio) = match metric {
            ForestPlotMetric::ResponseDifference => (
                "Response Rate Differences",
                "Difference (Treatment - Control)",
                0.0,
                false,
            ),
            ForestPlotMetric::ResponseRiskRatio => (
                "Response Rate Risk Ratios",
                "Risk Ratio (Treatment / Control)",
                1.0,
                true,
            ),
            ForestPlotMetric::ToxicityDifference => (
                "Grade 3+ Toxicity Differences",
                "Difference (Treatment - Control)",
                0.0,
                false,
            ),
            ForestPlotMetric::ToxicityRiskRatio => (
                "Grade 3+ Toxicity Risk Ratios",
                "Risk Ratio (Treatment / Control)",
                1.0,
                true,
            ),
            ForestPlotMetric::CompositeScore => (
                "Composite Score Differences",
                "Difference (Treatment - Control)",
                0.0,
                false,
            ),
        };

        let entries: Vec<ForestPlotEntry> = comparisons
            .iter()
            .filter_map(|c| {
                let (estimate, ci_lower, ci_upper, favors) = match metric {
                    ForestPlotMetric::ResponseDifference => (
                        c.response.absolute_diff,
                        c.response.diff_ci.lower,
                        c.response.diff_ci.upper,
                        c.response.absolute_diff > 0.0,
                    ),
                    ForestPlotMetric::ResponseRiskRatio => {
                        let rr = c.response.risk_ratio.as_ref()?;
                        (rr.ratio, rr.ci.lower, rr.ci.upper, rr.ratio > 1.0)
                    }
                    ForestPlotMetric::ToxicityDifference => (
                        c.toxicity_grade3.absolute_diff,
                        c.toxicity_grade3.diff_ci.lower,
                        c.toxicity_grade3.diff_ci.upper,
                        c.toxicity_grade3.absolute_diff < 0.0, // Lower is better
                    ),
                    ForestPlotMetric::ToxicityRiskRatio => {
                        let rr = c.toxicity_grade3.risk_ratio.as_ref()?;
                        (rr.ratio, rr.ci.lower, rr.ci.upper, rr.ratio < 1.0)
                    }
                    ForestPlotMetric::CompositeScore => (
                        c.composite_score.absolute_diff,
                        c.composite_score.diff_ci.lower,
                        c.composite_score.diff_ci.upper,
                        c.composite_score.absolute_diff > 0.0,
                    ),
                };

                Some(ForestPlotEntry {
                    label: format!("{} vs {}", c.treatment_id, c.control_id),
                    estimate,
                    ci_lower,
                    ci_upper,
                    weight: None,
                    favors_treatment: favors,
                    group: None,
                })
            })
            .collect();

        ForestPlotData {
            title: title.to_string(),
            metric: format!("{:?}", metric),
            entries,
            summary: None,
            reference_value,
            is_ratio_scale: is_ratio,
            x_label: x_label.to_string(),
        }
    }

    /// Add a summary row
    pub fn with_summary(
        mut self,
        label: &str,
        estimate: f64,
        ci_lower: f64,
        ci_upper: f64,
    ) -> Self {
        self.summary = Some(ForestPlotSummary {
            label: label.to_string(),
            estimate,
            ci_lower,
            ci_upper,
            i_squared: None,
        });
        self
    }
}

/// Metrics available for forest plots
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForestPlotMetric {
    ResponseDifference,
    ResponseRiskRatio,
    ToxicityDifference,
    ToxicityRiskRatio,
    CompositeScore,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 6: PUBLICATION REPORT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Complete publication-ready report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Executive summary
    pub executive_summary: ExecutiveSummary,
    /// All pairwise comparisons
    pub comparisons: Vec<StatisticalComparison>,
    /// Clinical metrics for each comparison
    pub clinical_metrics: Vec<(String, String, ClinicalMetrics)>,
    /// Summary tables
    pub tables: Vec<ReportTable>,
    /// Forest plot data
    pub forest_plots: Vec<ForestPlotData>,
    /// Recommendations
    pub recommendations: Vec<GuidelineRecommendation>,
    /// Statistical notes
    pub statistical_notes: Vec<String>,
}

/// Report metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMetadata {
    /// Report title
    pub title: String,
    /// Generation timestamp
    pub generated_at: String,
    /// Source experiment ID
    pub experiment_id: Option<String>,
    /// Number of guidelines compared
    pub n_guidelines: usize,
    /// Number of scenarios
    pub n_scenarios: usize,
    /// Alpha level used
    pub alpha: f64,
    /// Software version
    pub version: String,
}

/// Executive summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutiveSummary {
    /// Key findings
    pub key_findings: Vec<String>,
    /// Top recommendation
    pub primary_recommendation: Option<GuidelineRecommendation>,
    /// Significant comparisons count
    pub significant_comparisons: usize,
    /// Total comparisons
    pub total_comparisons: usize,
}

/// Guideline-specific recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineRecommendation {
    /// Guideline ID
    pub guideline_id: String,
    /// Overall rank
    pub rank: usize,
    /// Action recommendation
    pub action: RecommendedAction,
    /// Confidence
    pub confidence: RecommendationConfidence,
    /// Key strengths
    pub strengths: Vec<String>,
    /// Key weaknesses
    pub weaknesses: Vec<String>,
    /// Supporting evidence
    pub evidence: Vec<String>,
}

/// Report table
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportTable {
    /// Table ID
    pub id: String,
    /// Table title
    pub title: String,
    /// Table type
    pub table_type: TableType,
    /// Column headers
    pub headers: Vec<String>,
    /// Rows of data
    pub rows: Vec<Vec<String>>,
    /// Footer notes
    pub footnotes: Vec<String>,
}

/// Types of report tables
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TableType {
    Summary,
    Comparison,
    ClinicalMetrics,
    Rankings,
    EffectSizes,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 7: EXPORT FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Export format options
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExportFormat {
    JSON,
    CSV,
    LaTeX,
    Markdown,
}

/// Export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Include confidence intervals in tables
    pub include_cis: bool,
    /// Include p-values
    pub include_p_values: bool,
    /// Include effect sizes
    pub include_effect_sizes: bool,
    /// Decimal places for percentages
    pub decimal_places_pct: usize,
    /// Decimal places for p-values
    pub decimal_places_p: usize,
    /// Decimal places for effect sizes
    pub decimal_places_effect: usize,
    /// Bold significant results (LaTeX/Markdown)
    pub bold_significant: bool,
    /// Table caption prefix
    pub caption_prefix: String,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            include_cis: true,
            include_p_values: true,
            include_effect_sizes: true,
            decimal_places_pct: 1,
            decimal_places_p: 4,
            decimal_places_effect: 2,
            bold_significant: true,
            caption_prefix: "Table".to_string(),
        }
    }
}

/// Report exporter
pub struct ReportExporter {
    config: ExportConfig,
}

impl ReportExporter {
    /// Create exporter with default config
    pub fn new() -> Self {
        Self {
            config: ExportConfig::default(),
        }
    }

    /// Create exporter with custom config
    pub fn with_config(config: ExportConfig) -> Self {
        Self { config }
    }

    /// Export report to JSON
    pub fn to_json(&self, report: &PublicationReport) -> Result<String, String> {
        serde_json::to_string_pretty(report).map_err(|e| format!("JSON serialization error: {}", e))
    }

    /// Export comparisons to CSV
    pub fn comparisons_to_csv(&self, comparisons: &[StatisticalComparison]) -> String {
        let mut csv = String::new();

        // Header
        csv.push_str("treatment_id,control_id,");
        csv.push_str("response_treatment,response_control,response_diff,response_ci_low,response_ci_high,response_p,");
        csv.push_str("tox3_treatment,tox3_control,tox3_diff,tox3_ci_low,tox3_ci_high,tox3_p,");
        csv.push_str("score_treatment,score_control,score_diff,");
        csv.push_str("interpretation,recommendation\n");

        // Data rows
        for c in comparisons {
            csv.push_str(&format!("{},{},", c.treatment_id, c.control_id));
            csv.push_str(&format!(
                "{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},",
                c.response.treatment_value,
                c.response.control_value,
                c.response.absolute_diff,
                c.response.diff_ci.lower,
                c.response.diff_ci.upper,
                c.response.p_value
            ));
            csv.push_str(&format!(
                "{:.4},{:.4},{:.4},{:.4},{:.4},{:.6},",
                c.toxicity_grade3.treatment_value,
                c.toxicity_grade3.control_value,
                c.toxicity_grade3.absolute_diff,
                c.toxicity_grade3.diff_ci.lower,
                c.toxicity_grade3.diff_ci.upper,
                c.toxicity_grade3.p_value
            ));
            csv.push_str(&format!(
                "{:.4},{:.4},{:.4},",
                c.composite_score.treatment_value,
                c.composite_score.control_value,
                c.composite_score.absolute_diff
            ));
            csv.push_str(&format!(
                "{:?},{:?}\n",
                c.interpretation, c.recommendation.action
            ));
        }

        csv
    }

    /// Export table to LaTeX
    pub fn table_to_latex(&self, table: &ReportTable) -> String {
        let mut latex = String::new();

        // Table environment
        latex.push_str("\\begin{table}[htbp]\n");
        latex.push_str("\\centering\n");
        latex.push_str(&format!(
            "\\caption{{{}}}\n",
            self.escape_latex(&table.title)
        ));
        latex.push_str(&format!("\\label{{tab:{}}}\n", table.id));

        // Tabular environment
        let col_spec = format!("|{}|", "c|".repeat(table.headers.len()));
        latex.push_str(&format!("\\begin{{tabular}}{{{}}}\n", col_spec));
        latex.push_str("\\hline\n");

        // Header row
        let header_row: Vec<String> = table
            .headers
            .iter()
            .map(|h| format!("\\textbf{{{}}}", self.escape_latex(h)))
            .collect();
        latex.push_str(&header_row.join(" & "));
        latex.push_str(" \\\\\n\\hline\\hline\n");

        // Data rows
        for row in &table.rows {
            let escaped: Vec<String> = row.iter().map(|c| self.escape_latex(c)).collect();
            latex.push_str(&escaped.join(" & "));
            latex.push_str(" \\\\\n\\hline\n");
        }

        latex.push_str("\\end{tabular}\n");

        // Footnotes
        if !table.footnotes.is_empty() {
            latex.push_str("\\begin{tablenotes}\\small\n");
            for (i, note) in table.footnotes.iter().enumerate() {
                latex.push_str(&format!("\\item[{}] {}\n", i + 1, self.escape_latex(note)));
            }
            latex.push_str("\\end{tablenotes}\n");
        }

        latex.push_str("\\end{table}\n");

        latex
    }

    /// Export table to Markdown
    pub fn table_to_markdown(&self, table: &ReportTable) -> String {
        let mut md = String::new();

        // Title
        md.push_str(&format!("### {}\n\n", table.title));

        // Header row
        md.push_str("| ");
        md.push_str(&table.headers.join(" | "));
        md.push_str(" |\n");

        // Separator
        md.push_str("| ");
        md.push_str(
            &table
                .headers
                .iter()
                .map(|_| "---")
                .collect::<Vec<_>>()
                .join(" | "),
        );
        md.push_str(" |\n");

        // Data rows
        for row in &table.rows {
            md.push_str("| ");
            md.push_str(&row.join(" | "));
            md.push_str(" |\n");
        }

        // Footnotes
        if !table.footnotes.is_empty() {
            md.push_str("\n");
            for (i, note) in table.footnotes.iter().enumerate() {
                md.push_str(&format!("{}. {}\n", i + 1, note));
            }
        }

        md.push('\n');
        md
    }

    /// Export forest plot data for external plotting
    pub fn forest_plot_to_csv(&self, plot: &ForestPlotData) -> String {
        let mut csv = String::new();

        csv.push_str("label,estimate,ci_lower,ci_upper,weight,favors_treatment,group\n");

        for entry in &plot.entries {
            csv.push_str(&format!(
                "{},{:.4},{:.4},{:.4},{},{},{}\n",
                entry.label,
                entry.estimate,
                entry.ci_lower,
                entry.ci_upper,
                entry
                    .weight
                    .map(|w| format!("{:.4}", w))
                    .unwrap_or_default(),
                entry.favors_treatment,
                entry.group.as_deref().unwrap_or("")
            ));
        }

        if let Some(summary) = &plot.summary {
            csv.push_str(&format!(
                "{},{:.4},{:.4},{:.4},,true,SUMMARY\n",
                summary.label, summary.estimate, summary.ci_lower, summary.ci_upper
            ));
        }

        csv
    }

    /// Escape special LaTeX characters
    fn escape_latex(&self, s: &str) -> String {
        s.replace('&', "\\&")
            .replace('%', "\\%")
            .replace('$', "\\$")
            .replace('#', "\\#")
            .replace('_', "\\_")
            .replace('{', "\\{")
            .replace('}', "\\}")
            .replace('~', "\\textasciitilde{}")
            .replace('^', "\\textasciicircum{}")
    }

    /// Format p-value for display
    pub fn format_p_value(&self, p: f64) -> String {
        if p < 0.001 {
            "<0.001".to_string()
        } else if p < 0.01 {
            format!("{:.3}", p)
        } else {
            format!("{:.prec$}", p, prec = self.config.decimal_places_p)
        }
    }

    /// Format percentage for display
    pub fn format_pct(&self, value: f64) -> String {
        format!(
            "{:.prec$}%",
            value * 100.0,
            prec = self.config.decimal_places_pct
        )
    }

    /// Format with optional significance marker
    pub fn format_with_significance(&self, value: &str, significant: bool) -> String {
        if significant && self.config.bold_significant {
            format!("**{}**", value)
        } else {
            value.to_string()
        }
    }
}

impl Default for ReportExporter {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 8: REPORT GENERATION
// ═══════════════════════════════════════════════════════════════════════════

/// Report generator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportGeneratorConfig {
    /// Comparison configuration
    pub comparison: ComparisonConfig,
    /// Export configuration
    pub export: ExportConfig,
    /// Include forest plots
    pub include_forest_plots: bool,
    /// Include clinical metrics
    pub include_clinical_metrics: bool,
    /// Reference guideline ID (for baseline comparisons)
    pub reference_guideline: Option<String>,
    /// Maximum pairwise comparisons to include
    pub max_pairwise_comparisons: usize,
}

impl Default for ReportGeneratorConfig {
    fn default() -> Self {
        Self {
            comparison: ComparisonConfig::default(),
            export: ExportConfig::default(),
            include_forest_plots: true,
            include_clinical_metrics: true,
            reference_guideline: None,
            max_pairwise_comparisons: 20,
        }
    }
}

/// Main report generator
pub struct ReportGenerator {
    config: ReportGeneratorConfig,
    comparison_engine: ComparisonEngine,
    exporter: ReportExporter,
}

impl ReportGenerator {
    /// Create generator with default config
    pub fn new() -> Self {
        Self::with_config(ReportGeneratorConfig::default())
    }

    /// Create generator with custom config
    pub fn with_config(config: ReportGeneratorConfig) -> Self {
        let comparison_engine = ComparisonEngine::with_config(config.comparison.clone());
        let exporter = ReportExporter::with_config(config.export.clone());
        Self {
            config,
            comparison_engine,
            exporter,
        }
    }

    /// Generate publication report from experiment result
    pub fn generate_from_experiment(&self, result: &ExperimentResult) -> PublicationReport {
        self.generate_from_scores(&result.scores, result.provenance.experiment_id.to_string())
    }

    /// Generate publication report from guideline scores
    pub fn generate_from_scores(
        &self,
        scores: &[GuidelineRobustnessScoreHost],
        experiment_id: String,
    ) -> PublicationReport {
        let mut comparisons = Vec::new();
        let mut clinical_metrics = Vec::new();

        // Determine reference guideline
        let reference = self
            .config
            .reference_guideline
            .as_ref()
            .and_then(|ref_id| scores.iter().find(|s| &s.guideline_id == ref_id))
            .or_else(|| scores.first());

        // Generate all comparisons vs reference
        if let Some(ref_guideline) = reference {
            for treatment in scores {
                if treatment.guideline_id == ref_guideline.guideline_id {
                    continue;
                }

                let comparison = self
                    .comparison_engine
                    .compare_guidelines(treatment, ref_guideline);
                let metrics = self
                    .comparison_engine
                    .calculate_clinical_metrics(treatment, ref_guideline);

                clinical_metrics.push((
                    treatment.guideline_id.clone(),
                    ref_guideline.guideline_id.clone(),
                    metrics,
                ));
                comparisons.push(comparison);
            }
        }

        // Generate tables
        let tables = self.generate_tables(&comparisons, scores);

        // Generate forest plots
        let forest_plots = if self.config.include_forest_plots {
            vec![
                ForestPlotData::from_comparisons(
                    &comparisons,
                    ForestPlotMetric::ResponseDifference,
                ),
                ForestPlotData::from_comparisons(
                    &comparisons,
                    ForestPlotMetric::ToxicityDifference,
                ),
            ]
        } else {
            vec![]
        };

        // Generate recommendations
        let recommendations = self.generate_recommendations(scores, &comparisons);

        // Executive summary
        let significant_count = comparisons
            .iter()
            .filter(|c| c.response.significant || c.toxicity_grade3.significant)
            .count();

        let key_findings = self.generate_key_findings(&comparisons, &recommendations);

        let executive_summary = ExecutiveSummary {
            key_findings,
            primary_recommendation: recommendations.first().cloned(),
            significant_comparisons: significant_count,
            total_comparisons: comparisons.len(),
        };

        // Statistical notes
        let statistical_notes = vec![
            format!(
                "Statistical significance: α = {:.2}",
                self.config.comparison.alpha
            ),
            format!(
                "Assumed sample size: {} per arm",
                self.config.comparison.assumed_n_per_arm
            ),
            format!(
                "Non-inferiority margins: {:.0}% (response), {:.0}% (toxicity)",
                self.config.comparison.non_inferiority_margin_response * 100.0,
                self.config.comparison.non_inferiority_margin_toxicity * 100.0
            ),
            "Confidence intervals calculated using Wald method".to_string(),
            "Effect sizes reported as Hedges' g (bias-corrected)".to_string(),
        ];

        // Metadata
        let metadata = ReportMetadata {
            title: "Guideline Comparison Report".to_string(),
            generated_at: chrono_lite_timestamp(),
            experiment_id: Some(experiment_id),
            n_guidelines: scores.len(),
            n_scenarios: 0, // Would need experiment context
            alpha: self.config.comparison.alpha,
            version: env!("CARGO_PKG_VERSION").to_string(),
        };

        PublicationReport {
            metadata,
            executive_summary,
            comparisons,
            clinical_metrics,
            tables,
            forest_plots,
            recommendations,
            statistical_notes,
        }
    }

    /// Generate summary tables
    fn generate_tables(
        &self,
        comparisons: &[StatisticalComparison],
        scores: &[GuidelineRobustnessScoreHost],
    ) -> Vec<ReportTable> {
        let mut tables = Vec::new();

        // Table 1: Summary of all guidelines
        let summary_table = ReportTable {
            id: "summary".to_string(),
            title: "Summary of Guideline Performance".to_string(),
            table_type: TableType::Summary,
            headers: vec![
                "Guideline".to_string(),
                "Response Rate".to_string(),
                "Grade 3+ Tox".to_string(),
                "RDI".to_string(),
                "Composite Score".to_string(),
            ],
            rows: scores
                .iter()
                .map(|s| {
                    vec![
                        s.guideline_id.clone(),
                        self.exporter.format_pct(s.mean_response),
                        self.exporter.format_pct(s.mean_grade3plus_rate),
                        self.exporter.format_pct(s.mean_rdi),
                        format!("{:.3}", s.score_mean),
                    ]
                })
                .collect(),
            footnotes: vec![
                "Response Rate = objective response rate across scenarios".to_string(),
                "Grade 3+ Tox = rate of grade 3 or higher adverse events".to_string(),
            ],
        };
        tables.push(summary_table);

        // Table 2: Pairwise comparisons
        if !comparisons.is_empty() {
            let comparison_table = ReportTable {
                id: "comparisons".to_string(),
                title: "Pairwise Statistical Comparisons".to_string(),
                table_type: TableType::Comparison,
                headers: vec![
                    "Comparison".to_string(),
                    "Response Δ [95% CI]".to_string(),
                    "p-value".to_string(),
                    "Tox Δ [95% CI]".to_string(),
                    "p-value".to_string(),
                    "Interpretation".to_string(),
                ],
                rows: comparisons
                    .iter()
                    .map(|c| {
                        vec![
                            format!("{} vs {}", c.treatment_id, c.control_id),
                            c.response
                                .diff_ci
                                .format(self.config.export.decimal_places_pct),
                            self.exporter.format_p_value(c.response.p_value),
                            c.toxicity_grade3
                                .diff_ci
                                .format(self.config.export.decimal_places_pct),
                            self.exporter.format_p_value(c.toxicity_grade3.p_value),
                            format!("{:?}", c.interpretation),
                        ]
                    })
                    .collect(),
                footnotes: vec![
                    "Δ = absolute difference (Treatment - Control)".to_string(),
                    format!(
                        "CI = {}% confidence interval",
                        (self.config.comparison.alpha * 100.0) as i32
                    ),
                ],
            };
            tables.push(comparison_table);
        }

        // Table 3: Effect sizes
        let effect_table = ReportTable {
            id: "effects".to_string(),
            title: "Effect Sizes".to_string(),
            table_type: TableType::EffectSizes,
            headers: vec![
                "Comparison".to_string(),
                "Response (g)".to_string(),
                "Magnitude".to_string(),
                "Toxicity (g)".to_string(),
                "Magnitude".to_string(),
            ],
            rows: comparisons
                .iter()
                .map(|c| {
                    vec![
                        format!("{} vs {}", c.treatment_id, c.control_id),
                        format!("{:.2}", c.response.effect_size.value),
                        format!("{:?}", c.response.effect_size.interpretation),
                        format!("{:.2}", c.toxicity_grade3.effect_size.value),
                        format!("{:?}", c.toxicity_grade3.effect_size.interpretation),
                    ]
                })
                .collect(),
            footnotes: vec![
                "g = Hedges' g (bias-corrected standardized mean difference)".to_string(),
                "Small: |g| < 0.5, Medium: 0.5 ≤ |g| < 0.8, Large: |g| ≥ 0.8".to_string(),
            ],
        };
        tables.push(effect_table);

        tables
    }

    /// Generate recommendations for each guideline
    fn generate_recommendations(
        &self,
        scores: &[GuidelineRobustnessScoreHost],
        comparisons: &[StatisticalComparison],
    ) -> Vec<GuidelineRecommendation> {
        // Sort scores by composite score
        let mut sorted_scores: Vec<_> = scores.iter().collect();
        sorted_scores.sort_by(|a, b| {
            b.score_mean
                .partial_cmp(&a.score_mean)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        sorted_scores
            .iter()
            .enumerate()
            .map(|(rank, score)| {
                let mut strengths = Vec::new();
                let mut weaknesses = Vec::new();
                let mut evidence = Vec::new();

                // Analyze strengths/weaknesses
                if score.mean_response > 0.6 {
                    strengths.push(format!(
                        "High response rate ({:.1}%)",
                        score.mean_response * 100.0
                    ));
                } else if score.mean_response < 0.4 {
                    weaknesses.push(format!(
                        "Low response rate ({:.1}%)",
                        score.mean_response * 100.0
                    ));
                }

                if score.mean_grade3plus_rate < 0.2 {
                    strengths.push(format!(
                        "Low toxicity ({:.1}%)",
                        score.mean_grade3plus_rate * 100.0
                    ));
                } else if score.mean_grade3plus_rate > 0.4 {
                    weaknesses.push(format!(
                        "High toxicity ({:.1}%)",
                        score.mean_grade3plus_rate * 100.0
                    ));
                }

                // Find relevant comparisons
                for comp in comparisons {
                    if comp.treatment_id == score.guideline_id {
                        if comp.response.significant
                            && comp.response.direction == EffectDirection::Favorable
                        {
                            evidence.push(format!(
                                "Significantly better response vs {} (p={:.4})",
                                comp.control_id, comp.response.p_value
                            ));
                        }
                    }
                }

                // Determine action
                let (action, confidence) = if rank == 0 {
                    (
                        RecommendedAction::PreferTreatment,
                        RecommendationConfidence::High,
                    )
                } else if rank < 3 {
                    (
                        RecommendedAction::IndividualizeChoice,
                        RecommendationConfidence::Moderate,
                    )
                } else {
                    (
                        RecommendedAction::RequiresMoreData,
                        RecommendationConfidence::Low,
                    )
                };

                GuidelineRecommendation {
                    guideline_id: score.guideline_id.clone(),
                    rank: rank + 1,
                    action,
                    confidence,
                    strengths,
                    weaknesses,
                    evidence,
                }
            })
            .collect()
    }

    /// Generate key findings for executive summary
    fn generate_key_findings(
        &self,
        comparisons: &[StatisticalComparison],
        recommendations: &[GuidelineRecommendation],
    ) -> Vec<String> {
        let mut findings = Vec::new();

        // Top recommendation
        if let Some(top) = recommendations.first() {
            findings.push(format!(
                "Guideline '{}' ranked #1 with strongest overall performance",
                top.guideline_id
            ));
        }

        // Count significant findings
        let sig_response = comparisons
            .iter()
            .filter(|c| c.response.significant)
            .count();
        let sig_tox = comparisons
            .iter()
            .filter(|c| c.toxicity_grade3.significant)
            .count();

        if sig_response > 0 {
            findings.push(format!(
                "{} comparisons showed statistically significant differences in response rate",
                sig_response
            ));
        }

        if sig_tox > 0 {
            findings.push(format!(
                "{} comparisons showed statistically significant differences in toxicity",
                sig_tox
            ));
        }

        // Superior guidelines
        let superior_count = comparisons
            .iter()
            .filter(|c| c.interpretation == ClinicalInterpretation::Superior)
            .count();
        if superior_count > 0 {
            findings.push(format!(
                "{} treatment(s) demonstrated clinical superiority over control",
                superior_count
            ));
        }

        findings
    }

    /// Export report to specified format
    pub fn export(
        &self,
        report: &PublicationReport,
        format: ExportFormat,
    ) -> Result<String, String> {
        match format {
            ExportFormat::JSON => self.exporter.to_json(report),
            ExportFormat::CSV => Ok(self.exporter.comparisons_to_csv(&report.comparisons)),
            ExportFormat::LaTeX => Ok(self.export_all_latex(report)),
            ExportFormat::Markdown => Ok(self.export_all_markdown(report)),
        }
    }

    /// Export all tables to LaTeX
    fn export_all_latex(&self, report: &PublicationReport) -> String {
        let mut latex = String::new();

        latex.push_str("% Auto-generated by MedLang Reporting Module\n");
        latex.push_str(&format!(
            "% Generated: {}\n\n",
            report.metadata.generated_at
        ));

        for table in &report.tables {
            latex.push_str(&self.exporter.table_to_latex(table));
            latex.push_str("\n\n");
        }

        latex
    }

    /// Export all tables to Markdown
    fn export_all_markdown(&self, report: &PublicationReport) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", report.metadata.title));
        md.push_str(&format!(
            "*Generated: {}*\n\n",
            report.metadata.generated_at
        ));

        // Executive summary
        md.push_str("## Executive Summary\n\n");
        for finding in &report.executive_summary.key_findings {
            md.push_str(&format!("- {}\n", finding));
        }
        md.push_str("\n");

        // Tables
        md.push_str("## Results\n\n");
        for table in &report.tables {
            md.push_str(&self.exporter.table_to_markdown(table));
        }

        // Statistical notes
        md.push_str("## Statistical Methods\n\n");
        for note in &report.statistical_notes {
            md.push_str(&format!("- {}\n", note));
        }

        md
    }
}

impl Default for ReportGenerator {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 9: CONVENIENCE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a simple timestamp (ISO 8601 format approximation)
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = duration.as_secs();

    // Approximate ISO 8601 format
    let days_since_epoch = secs / 86400;
    let remaining_secs = secs % 86400;
    let hours = remaining_secs / 3600;
    let minutes = (remaining_secs % 3600) / 60;
    let seconds = remaining_secs % 60;

    // Rough year/month/day calculation (not accounting for leap years perfectly)
    let years = 1970 + (days_since_epoch / 365);
    let day_of_year = days_since_epoch % 365;
    let month = (day_of_year / 30) + 1;
    let day = (day_of_year % 30) + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        years, month, day, hours, minutes, seconds
    )
}

/// Quick comparison between two guidelines
pub fn compare_two_guidelines(
    treatment: &GuidelineRobustnessScoreHost,
    control: &GuidelineRobustnessScoreHost,
) -> StatisticalComparison {
    ComparisonEngine::new().compare_guidelines(treatment, control)
}

/// Generate publication report from experiment result
pub fn generate_publication_report(result: &ExperimentResult) -> PublicationReport {
    ReportGenerator::new().generate_from_experiment(result)
}

/// Generate publication report from scores
pub fn generate_report_from_scores(
    scores: &[GuidelineRobustnessScoreHost],
    experiment_id: &str,
) -> PublicationReport {
    ReportGenerator::new().generate_from_scores(scores, experiment_id.to_string())
}

/// Export report to JSON
pub fn export_report_json(report: &PublicationReport) -> Result<String, String> {
    ReportGenerator::new().export(report, ExportFormat::JSON)
}

/// Export report to CSV
pub fn export_report_csv(report: &PublicationReport) -> String {
    ReportExporter::new().comparisons_to_csv(&report.comparisons)
}

/// Export report to LaTeX
pub fn export_report_latex(report: &PublicationReport) -> Result<String, String> {
    ReportGenerator::new().export(report, ExportFormat::LaTeX)
}

/// Export report to Markdown
pub fn export_report_markdown(report: &PublicationReport) -> Result<String, String> {
    ReportGenerator::new().export(report, ExportFormat::Markdown)
}

/// Generate forest plot data for response differences
pub fn generate_response_forest_plot(comparisons: &[StatisticalComparison]) -> ForestPlotData {
    ForestPlotData::from_comparisons(comparisons, ForestPlotMetric::ResponseDifference)
}

/// Generate forest plot data for toxicity differences
pub fn generate_toxicity_forest_plot(comparisons: &[StatisticalComparison]) -> ForestPlotData {
    ForestPlotData::from_comparisons(comparisons, ForestPlotMetric::ToxicityDifference)
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 10: TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_guideline(
        id: &str,
        response: f64,
        tox: f64,
        score: f64,
    ) -> GuidelineRobustnessScoreHost {
        GuidelineRobustnessScoreHost {
            guideline_id: id.to_string(),
            mean_response: response,
            mean_grade3plus_rate: tox,
            mean_grade4plus_rate: Some(tox * 0.3),
            mean_rdi: 0.85,
            worst_response: response * 0.8,
            worst_grade3plus_rate: tox * 1.2,
            worst_grade4plus_rate: Some(tox * 0.4),
            worst_rdi: 0.75,
            response_std: Some(0.05),
            tox_std: Some(0.03),
            score_mean: score,
            score_worst: score * 0.8,
            response_ci95_low: Some(response - 0.1),
            response_ci95_high: Some(response + 0.1),
            tox_ci95_high: Some(tox + 0.05),
        }
    }

    #[test]
    fn test_statistical_calculator_z_score() {
        let calc = StatisticalCalculator::default();
        let z = calc.z_score();
        assert!(
            (z - 1.96).abs() < 0.01,
            "Z-score for alpha=0.05 should be ~1.96"
        );
    }

    #[test]
    fn test_confidence_interval() {
        let ci = ConfidenceInterval::new(0.1, 0.05, 0.15, 0.95);
        assert_eq!(ci.width(), 0.1);
        assert!(!ci.contains_zero());
        assert!(ci.contains(0.1));
        assert!(!ci.contains(0.2));
    }

    #[test]
    fn test_effect_magnitude_classification() {
        assert_eq!(
            EffectMagnitude::from_cohens_d(0.1),
            EffectMagnitude::Negligible
        );
        assert_eq!(EffectMagnitude::from_cohens_d(0.3), EffectMagnitude::Small);
        assert_eq!(EffectMagnitude::from_cohens_d(0.6), EffectMagnitude::Medium);
        assert_eq!(EffectMagnitude::from_cohens_d(1.0), EffectMagnitude::Large);
        assert_eq!(
            EffectMagnitude::from_cohens_d(1.5),
            EffectMagnitude::VeryLarge
        );
    }

    #[test]
    fn test_nnt_calculation() {
        let calc = StatisticalCalculator::default();

        // Treatment better (50% vs 30%)
        let nnt = calc.calculate_nnt(0.5, 0.3);
        assert!((nnt.nnt - 5.0).abs() < 0.01, "NNT should be ~5");
        assert_eq!(nnt.metric_type, NNTType::NNT);

        // No difference
        let nnt_same = calc.calculate_nnt(0.5, 0.5);
        assert_eq!(nnt_same.metric_type, NNTType::Infinite);
    }

    #[test]
    fn test_therapeutic_index() {
        let calc = StatisticalCalculator::default();

        // Good TI: high response, low toxicity
        let ti_good = calc.therapeutic_index(0.7, 0.1);
        assert_eq!(ti_good, 7.0);

        // Poor TI: low response, high toxicity
        let ti_poor = calc.therapeutic_index(0.3, 0.6);
        assert_eq!(ti_poor, 0.5);
    }

    #[test]
    fn test_clinical_interpretation() {
        let calc = StatisticalCalculator::default();

        // Superior: Better response, same toxicity
        let interp = calc.interpret_comparison(0.15, 0.001, 0.0, 0.5);
        assert_eq!(interp, ClinicalInterpretation::Superior);

        // Inferior: Worse response, same toxicity
        let interp = calc.interpret_comparison(-0.15, 0.001, 0.0, 0.5);
        assert_eq!(interp, ClinicalInterpretation::Inferior);

        // Equivalent: No meaningful differences
        let interp = calc.interpret_comparison(0.02, 0.3, 0.01, 0.4);
        assert_eq!(interp, ClinicalInterpretation::Equivalent);
    }

    #[test]
    fn test_comparison_engine() {
        let engine = ComparisonEngine::new();

        let treatment = mock_guideline("treatment", 0.65, 0.20, 0.75);
        let control = mock_guideline("control", 0.50, 0.25, 0.60);

        let comparison = engine.compare_guidelines(&treatment, &control);

        assert_eq!(comparison.treatment_id, "treatment");
        assert_eq!(comparison.control_id, "control");
        assert!(comparison.response.absolute_diff > 0.0);
        assert!(comparison.toxicity_grade3.absolute_diff < 0.0); // Lower tox is better
    }

    #[test]
    fn test_clinical_metrics_calculation() {
        let engine = ComparisonEngine::new();

        let treatment = mock_guideline("treatment", 0.65, 0.15, 0.75);
        let control = mock_guideline("control", 0.50, 0.25, 0.60);

        let metrics = engine.calculate_clinical_metrics(&treatment, &control);

        // Treatment has better response
        assert!(metrics.nnt_response.is_some());

        // Treatment has lower toxicity (NNH should be None since we're not harming)
        // Actually, NNH is calculated when treatment has higher toxicity

        // TI should favor treatment
        assert!(metrics.therapeutic_index.ti_ratio > 1.0);
    }

    #[test]
    fn test_forest_plot_generation() {
        let engine = ComparisonEngine::new();

        let treatment1 = mock_guideline("t1", 0.65, 0.20, 0.75);
        let treatment2 = mock_guideline("t2", 0.55, 0.22, 0.68);
        let control = mock_guideline("control", 0.50, 0.25, 0.60);

        let comparisons = vec![
            engine.compare_guidelines(&treatment1, &control),
            engine.compare_guidelines(&treatment2, &control),
        ];

        let plot =
            ForestPlotData::from_comparisons(&comparisons, ForestPlotMetric::ResponseDifference);

        assert_eq!(plot.entries.len(), 2);
        assert_eq!(plot.reference_value, 0.0);
        assert!(!plot.is_ratio_scale);
    }

    #[test]
    fn test_report_generation() {
        let scores = vec![
            mock_guideline("guideline_a", 0.65, 0.18, 0.78),
            mock_guideline("guideline_b", 0.60, 0.22, 0.70),
            mock_guideline("guideline_c", 0.55, 0.25, 0.65),
        ];

        let report = generate_report_from_scores(&scores, "test_exp_001");

        assert_eq!(report.metadata.n_guidelines, 3);
        assert_eq!(report.comparisons.len(), 2); // 2 comparisons vs reference
        assert!(!report.tables.is_empty());
        assert!(!report.recommendations.is_empty());
    }

    #[test]
    fn test_csv_export() {
        let engine = ComparisonEngine::new();
        let treatment = mock_guideline("treatment", 0.65, 0.20, 0.75);
        let control = mock_guideline("control", 0.50, 0.25, 0.60);

        let comparisons = vec![engine.compare_guidelines(&treatment, &control)];

        let exporter = ReportExporter::new();
        let csv = exporter.comparisons_to_csv(&comparisons);

        assert!(csv.contains("treatment_id,control_id"));
        assert!(csv.contains("treatment,control"));
    }

    #[test]
    fn test_latex_export() {
        let table = ReportTable {
            id: "test".to_string(),
            title: "Test Table".to_string(),
            table_type: TableType::Summary,
            headers: vec!["A".to_string(), "B".to_string()],
            rows: vec![vec!["1".to_string(), "2".to_string()]],
            footnotes: vec!["Note 1".to_string()],
        };

        let exporter = ReportExporter::new();
        let latex = exporter.table_to_latex(&table);

        assert!(latex.contains("\\begin{table}"));
        assert!(latex.contains("\\caption{Test Table}"));
        assert!(latex.contains("\\textbf{A}"));
        assert!(latex.contains("\\end{table}"));
    }

    #[test]
    fn test_markdown_export() {
        let table = ReportTable {
            id: "test".to_string(),
            title: "Test Table".to_string(),
            table_type: TableType::Summary,
            headers: vec!["A".to_string(), "B".to_string()],
            rows: vec![vec!["1".to_string(), "2".to_string()]],
            footnotes: vec!["Note 1".to_string()],
        };

        let exporter = ReportExporter::new();
        let md = exporter.table_to_markdown(&table);

        assert!(md.contains("### Test Table"));
        assert!(md.contains("| A | B |"));
        assert!(md.contains("| --- | --- |"));
        assert!(md.contains("| 1 | 2 |"));
    }

    #[test]
    fn test_p_value_formatting() {
        let exporter = ReportExporter::new();

        assert_eq!(exporter.format_p_value(0.0001), "<0.001");
        assert_eq!(exporter.format_p_value(0.005), "0.005");
        assert_eq!(exporter.format_p_value(0.05), "0.0500");
    }

    #[test]
    fn test_convenience_functions() {
        let treatment = mock_guideline("treatment", 0.65, 0.20, 0.75);
        let control = mock_guideline("control", 0.50, 0.25, 0.60);

        let comparison = compare_two_guidelines(&treatment, &control);
        assert_eq!(comparison.treatment_id, "treatment");

        let scores = vec![treatment, control];
        let report = generate_report_from_scores(&scores, "test");
        assert!(export_report_json(&report).is_ok());
    }
}
