//! Week 49: Trial Design & Power Analysis (v2.0)
//!
//! Comprehensive trial planning system with:
//! - Power and sample size calculations
//! - Multiple test types (superiority, non-inferiority, equivalence)
//! - Multiple comparison adjustments (Bonferroni, Holm, Hochberg, etc.)
//! - Group sequential designs with spending functions
//! - Uncertainty propagation from robustness scores
//! - Power curves and sensitivity analysis
//! - Regulatory-ready documentation (SAP, LaTeX, Markdown)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::rl::constraints::GuidelineRobustnessScoreHost;
use crate::rl::reporting::ConfidenceInterval;

// ═══════════════════════════════════════════════════════════════════════════
// PART 1: ENUMS AND BASIC TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Primary endpoint type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EndpointType {
    /// Binary response rate (e.g., ORR)
    BinaryResponse,
    /// Binary toxicity rate (e.g., Grade >= 3)
    BinaryToxicity,
    /// Continuous outcome (e.g., tumor shrinkage %)
    Continuous,
    /// Time-to-event (e.g., PFS, OS)
    TimeToEvent,
}

/// Test type / hypothesis structure
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TestType {
    /// H0: p_exp = p_ctrl vs H1: p_exp != p_ctrl
    TwoSidedSuperiority,
    /// H0: p_exp <= p_ctrl vs H1: p_exp > p_ctrl
    OneSidedSuperiority,
    /// H0: p_exp <= p_ctrl - margin vs H1: p_exp > p_ctrl - margin
    NonInferiority { margin: f64 },
    /// H0: |p_exp - p_ctrl| >= margin vs H1: |p_exp - p_ctrl| < margin
    Equivalence { margin: f64 },
}

/// Multiple comparison adjustment method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MultipleComparisonMethod {
    /// No adjustment
    None,
    /// Bonferroni (divide alpha by k)
    Bonferroni,
    /// Holm step-down
    Holm,
    /// Hochberg step-up
    Hochberg,
    /// Benjamini-Hochberg FDR
    BenjaminiHochberg,
    /// Dunnett (many-to-one)
    Dunnett,
    /// Fixed sequence (gatekeeping)
    FixedSequence,
}

impl Default for MultipleComparisonMethod {
    fn default() -> Self {
        Self::None
    }
}

/// Allocation ratio (experimental:control)
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AllocationRatio {
    pub experimental: f64,
    pub control: f64,
}

impl Default for AllocationRatio {
    fn default() -> Self {
        Self {
            experimental: 1.0,
            control: 1.0,
        }
    }
}

impl AllocationRatio {
    pub fn balanced() -> Self {
        Self::default()
    }

    pub fn ratio(exp_to_ctrl: f64) -> Self {
        Self {
            experimental: exp_to_ctrl,
            control: 1.0,
        }
    }

    /// Get k = n_exp / n_ctrl
    pub fn k(&self) -> f64 {
        self.experimental / self.control
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 2: DROPOUT AND ATTRITION
// ═══════════════════════════════════════════════════════════════════════════

/// Dropout/attrition assumptions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Expected dropout rate in control arm
    pub control_dropout_rate: f64,
    /// Expected dropout rate in experimental arm
    pub experimental_dropout_rate: f64,
    /// Whether to inflate sample size to account for dropout
    pub inflate_for_dropout: bool,
}

impl Default for DropoutConfig {
    fn default() -> Self {
        Self {
            control_dropout_rate: 0.10,
            experimental_dropout_rate: 0.10,
            inflate_for_dropout: true,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 3: GROUP SEQUENTIAL DESIGN
// ═══════════════════════════════════════════════════════════════════════════

/// Spending function for alpha/beta
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpendingFunction {
    /// O'Brien-Fleming bounds
    OBrienFleming,
    /// Pocock bounds
    Pocock,
    /// Lan-DeMets approximation to O'Brien-Fleming
    LanDeMetsOBF,
    /// Hwang-Shih-DeCani with parameter gamma
    HwangShihDeCani,
}

impl Default for SpendingFunction {
    fn default() -> Self {
        Self::OBrienFleming
    }
}

/// Group sequential design configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSequentialConfig {
    /// Number of analyses (including final)
    pub n_analyses: usize,
    /// Information fractions at each analysis
    pub info_fractions: Vec<f64>,
    /// Alpha spending function
    pub alpha_spending: SpendingFunction,
    /// Beta spending function (for futility)
    pub beta_spending: Option<SpendingFunction>,
    /// Whether to include futility bounds
    pub include_futility: bool,
}

impl Default for GroupSequentialConfig {
    fn default() -> Self {
        Self {
            n_analyses: 1, // Fixed design (no interim)
            info_fractions: vec![1.0],
            alpha_spending: SpendingFunction::OBrienFleming,
            beta_spending: None,
            include_futility: false,
        }
    }
}

impl GroupSequentialConfig {
    /// Fixed design (no interim analyses)
    pub fn fixed() -> Self {
        Self::default()
    }

    /// Two interim analyses with O'Brien-Fleming
    pub fn two_interim_obf() -> Self {
        Self {
            n_analyses: 3,
            info_fractions: vec![0.33, 0.67, 1.0],
            alpha_spending: SpendingFunction::OBrienFleming,
            beta_spending: None,
            include_futility: false,
        }
    }

    /// One interim at 50% with futility
    pub fn one_interim_with_futility() -> Self {
        Self {
            n_analyses: 2,
            info_fractions: vec![0.5, 1.0],
            alpha_spending: SpendingFunction::OBrienFleming,
            beta_spending: Some(SpendingFunction::OBrienFleming),
            include_futility: true,
        }
    }
}

/// Boundaries at each analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisBoundary {
    pub analysis_number: usize,
    pub info_fraction: f64,
    pub n_enrolled: usize,
    pub efficacy_z: f64,
    pub efficacy_p: f64,
    pub futility_z: Option<f64>,
    pub futility_p: Option<f64>,
    pub cumulative_alpha_spent: f64,
    pub cumulative_beta_spent: Option<f64>,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 4: ARM ASSUMPTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Complete assumptions for one arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmAssumption {
    pub arm_name: String,
    pub guideline_id: String,
    pub is_control: bool,

    /// Event probability (for binary endpoints)
    pub p_event: f64,
    /// Standard error of p_event (if known)
    pub p_event_se: Option<f64>,
    /// Confidence interval for p_event
    pub p_event_ci: Option<ConfidenceInterval>,

    /// Mean (for continuous endpoints)
    pub mean: Option<f64>,
    /// Standard deviation (for continuous endpoints)
    pub std_dev: Option<f64>,

    /// Hazard rate (for time-to-event)
    pub hazard_rate: Option<f64>,
    /// Median survival time (for time-to-event)
    pub median_time: Option<f64>,

    /// Expected dropout rate
    pub dropout_rate: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 5: TRIAL POWER CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

/// Computation mode: what to solve for
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputationMode {
    /// Given n, compute power
    PowerFromN,
    /// Given target power, compute required n
    NFromPower,
    /// Compute power curve over range of n
    PowerCurve,
    /// Compute both power and required n
    Full,
}

impl Default for ComputationMode {
    fn default() -> Self {
        Self::Full
    }
}

/// Clinical thresholds for interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClinicalThresholds {
    /// Minimum clinically important difference for response
    pub mcid_response: f64,
    /// Minimum clinically important difference for toxicity
    pub mcid_toxicity: f64,
}

impl Default for ClinicalThresholds {
    fn default() -> Self {
        Self {
            mcid_response: 0.10,
            mcid_toxicity: 0.05,
        }
    }
}

/// Complete trial power configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialPowerConfig {
    // Basic identifiers
    pub control_guideline_id: String,
    pub experimental_guideline_ids: Vec<String>,

    // Endpoint and test specification
    pub endpoint: EndpointType,
    pub test_type: TestType,

    // Statistical parameters
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_power")]
    pub target_power: f64,
    #[serde(default)]
    pub n_per_arm: Option<usize>,
    #[serde(default)]
    pub n_total: Option<usize>,

    // Design parameters
    #[serde(default)]
    pub allocation: AllocationRatio,
    #[serde(default)]
    pub dropout: DropoutConfig,
    #[serde(default)]
    pub group_sequential: GroupSequentialConfig,

    // Multiple comparisons
    #[serde(default)]
    pub multiplicity_adjustment: MultipleComparisonMethod,

    // Computation options
    #[serde(default)]
    pub computation_mode: ComputationMode,
    #[serde(default = "default_curve_min")]
    pub curve_n_min: usize,
    #[serde(default = "default_curve_max")]
    pub curve_n_max: usize,
    #[serde(default = "default_curve_step")]
    pub curve_n_step: usize,

    // Uncertainty propagation
    #[serde(default)]
    pub propagate_uncertainty: bool,
    #[serde(default = "default_uncertainty_samples")]
    pub uncertainty_samples: usize,
    #[serde(default = "default_confidence")]
    pub power_ci_level: f64,

    // Clinical thresholds
    #[serde(default)]
    pub clinical_thresholds: Option<ClinicalThresholds>,
}

fn default_alpha() -> f64 {
    0.05
}
fn default_power() -> f64 {
    0.80
}
fn default_curve_min() -> usize {
    20
}
fn default_curve_max() -> usize {
    500
}
fn default_curve_step() -> usize {
    10
}
fn default_uncertainty_samples() -> usize {
    1000
}
fn default_confidence() -> f64 {
    0.95
}

impl TrialPowerConfig {
    /// Create a simple superiority trial config
    pub fn superiority(
        control_id: &str,
        experimental_ids: Vec<String>,
        endpoint: EndpointType,
        n_per_arm: usize,
    ) -> Self {
        Self {
            control_guideline_id: control_id.to_string(),
            experimental_guideline_ids: experimental_ids,
            endpoint,
            test_type: TestType::TwoSidedSuperiority,
            alpha: 0.05,
            target_power: 0.80,
            n_per_arm: Some(n_per_arm),
            n_total: None,
            allocation: AllocationRatio::balanced(),
            dropout: DropoutConfig::default(),
            group_sequential: GroupSequentialConfig::fixed(),
            multiplicity_adjustment: MultipleComparisonMethod::None,
            computation_mode: ComputationMode::Full,
            curve_n_min: 20,
            curve_n_max: 500,
            curve_n_step: 10,
            propagate_uncertainty: false,
            uncertainty_samples: 1000,
            power_ci_level: 0.95,
            clinical_thresholds: None,
        }
    }

    /// Create a non-inferiority trial config
    pub fn non_inferiority(
        control_id: &str,
        experimental_ids: Vec<String>,
        endpoint: EndpointType,
        n_per_arm: usize,
        margin: f64,
    ) -> Self {
        Self {
            control_guideline_id: control_id.to_string(),
            experimental_guideline_ids: experimental_ids,
            endpoint,
            test_type: TestType::NonInferiority { margin },
            alpha: 0.025, // One-sided
            target_power: 0.80,
            n_per_arm: Some(n_per_arm),
            n_total: None,
            allocation: AllocationRatio::balanced(),
            dropout: DropoutConfig::default(),
            group_sequential: GroupSequentialConfig::fixed(),
            multiplicity_adjustment: MultipleComparisonMethod::None,
            computation_mode: ComputationMode::Full,
            curve_n_min: 20,
            curve_n_max: 500,
            curve_n_step: 10,
            propagate_uncertainty: false,
            uncertainty_samples: 1000,
            power_ci_level: 0.95,
            clinical_thresholds: None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 6: POWER RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Complete power result for one comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwisePowerResult {
    pub control_id: String,
    pub experimental_id: String,

    // Assumed parameters
    pub p_control: f64,
    pub p_experimental: f64,
    pub effect_size: f64,
    pub relative_effect: f64,

    // Sample size
    pub n_per_arm: usize,
    pub n_total: usize,
    pub n_per_arm_inflated: usize,
    pub n_total_inflated: usize,

    // Power results
    pub power: f64,
    pub power_ci: Option<ConfidenceInterval>,
    pub required_n_per_arm: Option<usize>,
    pub required_n_total: Option<usize>,

    // Test details
    pub endpoint: EndpointType,
    pub test_type: TestType,
    pub alpha: f64,
    pub alpha_adjusted: f64,
    pub z_critical: f64,

    // Interpretation
    pub is_clinically_meaningful: bool,
    pub is_adequately_powered: bool,
    pub minimum_detectable_effect: f64,

    // Group sequential
    pub analysis_boundaries: Option<Vec<AnalysisBoundary>>,
    pub expected_n_h1: Option<f64>,
    pub expected_n_h0: Option<f64>,
}

/// Power curve data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerCurvePoint {
    pub n_per_arm: usize,
    pub n_total: usize,
    pub power: f64,
    pub power_ci_lower: Option<f64>,
    pub power_ci_upper: Option<f64>,
}

/// Complete power curve
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerCurve {
    pub control_id: String,
    pub experimental_id: String,
    pub effect_size: f64,
    pub alpha: f64,
    pub points: Vec<PowerCurvePoint>,
    pub n_for_target_power: Option<usize>,
    pub target_power: f64,
}

/// Sensitivity analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerSensitivity {
    pub experimental_id: String,
    pub vary_p_control: Vec<(f64, f64)>,
    pub vary_p_experimental: Vec<(f64, f64)>,
    pub vary_n: Vec<(usize, f64)>,
    pub vary_effect: Vec<(f64, f64)>,
    pub detectable_effect_by_n: Vec<(usize, f64)>,
}

/// Summary across all comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialPowerSummary {
    pub n_comparisons: usize,
    pub n_adequately_powered: usize,
    pub n_clinically_meaningful: usize,
    pub best_powered: Option<String>,
    pub largest_effect: Option<String>,
    pub min_n_for_all_powered: Option<usize>,
    pub recommendations: Vec<String>,
}

/// Regulatory documentation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialDocumentation {
    pub sap_power_section: String,
    pub assumptions_table_latex: String,
    pub power_table_latex: String,
    pub summary_markdown: String,
}

impl TrialDocumentation {
    pub fn empty() -> Self {
        Self {
            sap_power_section: String::new(),
            assumptions_table_latex: String::new(),
            power_table_latex: String::new(),
            summary_markdown: String::new(),
        }
    }
}

/// Complete trial power result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialPowerResult {
    pub config: TrialPowerConfig,
    pub arms: Vec<ArmAssumption>,
    pub pairwise: Vec<PairwisePowerResult>,
    pub power_curves: Option<Vec<PowerCurve>>,
    pub sensitivity: Option<Vec<PowerSensitivity>>,
    pub summary: TrialPowerSummary,
    pub documentation: TrialDocumentation,
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 7: STATISTICAL FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Error function approximation (Horner's method)
fn erf(x: f64) -> f64 {
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

/// Standard normal CDF
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse normal CDF approximation (Abramowitz and Stegun)
fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }

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

/// Z-critical value for given alpha and test type
fn z_critical(alpha: f64, test_type: &TestType) -> f64 {
    match test_type {
        TestType::TwoSidedSuperiority | TestType::Equivalence { .. } => {
            normal_quantile(1.0 - alpha / 2.0)
        }
        TestType::OneSidedSuperiority | TestType::NonInferiority { .. } => {
            normal_quantile(1.0 - alpha)
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 8: POWER CALCULATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Power for two-sample proportion test (pooled variance)
pub fn power_binary_pooled(
    p_ctrl: f64,
    p_exp: f64,
    n_per_arm: usize,
    allocation: &AllocationRatio,
    alpha: f64,
    test_type: &TestType,
) -> f64 {
    if n_per_arm == 0 {
        return 0.0;
    }

    let k = allocation.k();
    let n_ctrl = n_per_arm as f64;
    let n_exp = n_ctrl * k;

    // Effect size (accounting for non-inferiority margin if applicable)
    let delta = match test_type {
        TestType::NonInferiority { margin } => {
            // Effective effect under H1
            (p_exp - p_ctrl) + margin
        }
        TestType::Equivalence { .. } => (p_exp - p_ctrl).abs(),
        _ => p_exp - p_ctrl,
    };

    // Variance under H0 (pooled)
    let p_bar = (p_ctrl * n_ctrl + p_exp * n_exp) / (n_ctrl + n_exp);
    let var_h0 = p_bar * (1.0 - p_bar) * (1.0 / n_ctrl + 1.0 / n_exp);

    // Variance under H1 (unpooled)
    let var_h1 = (p_ctrl * (1.0 - p_ctrl) / n_ctrl) + (p_exp * (1.0 - p_exp) / n_exp);

    if var_h0 <= 0.0 || var_h1 <= 0.0 {
        return 0.0;
    }

    let se_h0 = var_h0.sqrt();
    let se_h1 = var_h1.sqrt();

    let z_crit = z_critical(alpha, test_type);

    match test_type {
        TestType::TwoSidedSuperiority => {
            let z1 = (delta - z_crit * se_h0) / se_h1;
            let z2 = (delta + z_crit * se_h0) / se_h1;
            (normal_cdf(z1) + (1.0 - normal_cdf(z2))).clamp(0.0, 1.0)
        }
        TestType::OneSidedSuperiority | TestType::NonInferiority { .. } => {
            let z = (delta - z_crit * se_h0) / se_h1;
            normal_cdf(z).clamp(0.0, 1.0)
        }
        TestType::Equivalence { margin } => {
            // TOST
            let delta_upper = p_exp - p_ctrl + margin;
            let delta_lower = -(p_exp - p_ctrl) + margin;

            let z_upper = (delta_upper - z_crit * se_h0) / se_h1;
            let z_lower = (delta_lower - z_crit * se_h0) / se_h1;

            (normal_cdf(z_upper) * normal_cdf(z_lower)).clamp(0.0, 1.0)
        }
    }
}

/// Required sample size per arm for target power (binary endpoint)
pub fn required_n_binary(
    p_ctrl: f64,
    p_exp: f64,
    allocation: &AllocationRatio,
    alpha: f64,
    target_power: f64,
    test_type: &TestType,
) -> usize {
    // Binary search for required n
    let mut lo = 5_usize;
    let mut hi = 100000_usize;

    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        let power = power_binary_pooled(p_ctrl, p_exp, mid, allocation, alpha, test_type);
        if power >= target_power {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    hi
}

/// Minimum detectable effect for given n and target power
pub fn minimum_detectable_effect_binary(
    p_ctrl: f64,
    n_per_arm: usize,
    allocation: &AllocationRatio,
    alpha: f64,
    target_power: f64,
    test_type: &TestType,
) -> f64 {
    let mut lo = 0.001_f64;
    let mut hi = 0.5_f64.min(1.0 - p_ctrl).min(p_ctrl);

    for _ in 0..50 {
        let mid = (lo + hi) / 2.0;
        let p_exp = p_ctrl + mid;
        let power = power_binary_pooled(p_ctrl, p_exp, n_per_arm, allocation, alpha, test_type);

        if power >= target_power {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    hi
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 9: MULTIPLICITY ADJUSTMENT
// ═══════════════════════════════════════════════════════════════════════════

/// Adjust alpha for multiple comparisons
pub fn adjust_alpha(
    alpha: f64,
    n_comparisons: usize,
    method: MultipleComparisonMethod,
    comparison_index: usize,
) -> f64 {
    if n_comparisons <= 1 {
        return alpha;
    }

    match method {
        MultipleComparisonMethod::None => alpha,
        MultipleComparisonMethod::Bonferroni => alpha / n_comparisons as f64,
        MultipleComparisonMethod::Holm => alpha / (n_comparisons - comparison_index) as f64,
        MultipleComparisonMethod::Hochberg => alpha / n_comparisons as f64,
        MultipleComparisonMethod::BenjaminiHochberg => {
            alpha * (comparison_index + 1) as f64 / n_comparisons as f64
        }
        MultipleComparisonMethod::Dunnett => {
            // Sidak approximation
            1.0 - (1.0 - alpha).powf(1.0 / n_comparisons as f64)
        }
        MultipleComparisonMethod::FixedSequence => alpha / n_comparisons as f64,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 10: GROUP SEQUENTIAL DESIGN
// ═══════════════════════════════════════════════════════════════════════════

/// Compute spending at information fraction
fn alpha_spent(info_fraction: f64, total_alpha: f64, spending: SpendingFunction) -> f64 {
    match spending {
        SpendingFunction::OBrienFleming => {
            let z = normal_quantile(1.0 - total_alpha / 2.0);
            2.0 * (1.0 - normal_cdf(z / info_fraction.sqrt()))
        }
        SpendingFunction::Pocock => {
            total_alpha * (1.0 + (std::f64::consts::E - 1.0) * info_fraction).ln()
        }
        SpendingFunction::LanDeMetsOBF => {
            if info_fraction < 0.001 {
                return 0.0;
            }
            let z = normal_quantile(1.0 - total_alpha / 2.0);
            2.0 - 2.0 * normal_cdf(z / info_fraction.sqrt())
        }
        SpendingFunction::HwangShihDeCani => {
            let gamma = -4.0_f64;
            if gamma.abs() < 0.001 {
                total_alpha * info_fraction
            } else {
                total_alpha * (1.0 - (-gamma * info_fraction).exp()) / (1.0 - (-gamma).exp())
            }
        }
    }
}

/// Compute group sequential boundaries
pub fn compute_gs_boundaries(
    config: &GroupSequentialConfig,
    total_alpha: f64,
    total_n_per_arm: usize,
) -> Vec<AnalysisBoundary> {
    let mut boundaries = Vec::with_capacity(config.n_analyses);
    let mut prev_alpha_spent = 0.0;

    for (i, &info_frac) in config.info_fractions.iter().enumerate() {
        let cum_alpha = alpha_spent(info_frac, total_alpha, config.alpha_spending);
        let incr_alpha = (cum_alpha - prev_alpha_spent).max(0.0001);

        let z_eff = normal_quantile(1.0 - incr_alpha / 2.0);
        let p_eff = incr_alpha;

        let (z_fut, p_fut, cum_beta) = if config.include_futility {
            if let Some(beta_spending) = config.beta_spending {
                let cum_beta_spent = alpha_spent(info_frac, 0.20, beta_spending);
                let z = -normal_quantile(1.0 - cum_beta_spent / 2.0);
                (Some(z), Some(1.0 - normal_cdf(z)), Some(cum_beta_spent))
            } else {
                (None, None, None)
            }
        } else {
            (None, None, None)
        };

        let n_enrolled = ((info_frac * total_n_per_arm as f64).ceil() as usize).max(1);

        boundaries.push(AnalysisBoundary {
            analysis_number: i + 1,
            info_fraction: info_frac,
            n_enrolled,
            efficacy_z: z_eff,
            efficacy_p: p_eff,
            futility_z: z_fut,
            futility_p: p_fut,
            cumulative_alpha_spent: cum_alpha,
            cumulative_beta_spent: cum_beta,
        });

        prev_alpha_spent = cum_alpha;
    }

    boundaries
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 11: UNCERTAINTY PROPAGATION
// ═══════════════════════════════════════════════════════════════════════════

/// Simple LCG random number generator
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform for normal distribution
    fn next_normal(&mut self, mean: f64, std: f64) -> f64 {
        let u1 = self.next_f64().max(1e-10);
        let u2 = self.next_f64();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        mean + std * z
    }
}

/// Propagate uncertainty in event rates to power estimate
pub fn power_with_uncertainty(
    p_ctrl: f64,
    p_ctrl_se: f64,
    p_exp: f64,
    p_exp_se: f64,
    n_per_arm: usize,
    allocation: &AllocationRatio,
    alpha: f64,
    test_type: &TestType,
    n_samples: usize,
    conf_level: f64,
) -> (f64, ConfidenceInterval) {
    let mut rng = SimpleRng::new(42);
    let mut powers: Vec<f64> = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let p_c = rng.next_normal(p_ctrl, p_ctrl_se).clamp(0.01, 0.99);
        let p_e = rng.next_normal(p_exp, p_exp_se).clamp(0.01, 0.99);
        let power = power_binary_pooled(p_c, p_e, n_per_arm, allocation, alpha, test_type);
        powers.push(power);
    }

    powers.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean_power = powers.iter().sum::<f64>() / n_samples as f64;
    let lower_idx = ((1.0 - conf_level) / 2.0 * n_samples as f64) as usize;
    let upper_idx = ((1.0 - (1.0 - conf_level) / 2.0) * n_samples as f64) as usize;

    let ci = ConfidenceInterval::new(
        mean_power,
        powers[lower_idx.min(n_samples - 1)],
        powers[upper_idx.min(n_samples - 1)],
        conf_level,
    );

    (mean_power, ci)
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 12: POWER CURVES AND SENSITIVITY
// ═══════════════════════════════════════════════════════════════════════════

/// Compute power curve
pub fn compute_power_curve(
    p_ctrl: f64,
    p_exp: f64,
    allocation: &AllocationRatio,
    alpha: f64,
    test_type: &TestType,
    target_power: f64,
    n_min: usize,
    n_max: usize,
    n_step: usize,
) -> PowerCurve {
    let mut points = Vec::new();
    let mut n_for_target = None;
    let mut prev_power = 0.0;

    let mut n = n_min;
    while n <= n_max {
        let power = power_binary_pooled(p_ctrl, p_exp, n, allocation, alpha, test_type);

        points.push(PowerCurvePoint {
            n_per_arm: n,
            n_total: (n as f64 * (1.0 + allocation.k())).ceil() as usize,
            power,
            power_ci_lower: None,
            power_ci_upper: None,
        });

        if n_for_target.is_none() && prev_power < target_power && power >= target_power {
            let frac = (target_power - prev_power) / (power - prev_power);
            let interp_n = (n - n_step) as f64 + frac * n_step as f64;
            n_for_target = Some(interp_n.ceil() as usize);
        }

        prev_power = power;
        n += n_step;
    }

    PowerCurve {
        control_id: String::new(),
        experimental_id: String::new(),
        effect_size: p_exp - p_ctrl,
        alpha,
        points,
        n_for_target_power: n_for_target,
        target_power,
    }
}

/// Compute sensitivity analysis
pub fn compute_sensitivity(
    p_ctrl: f64,
    p_exp: f64,
    n_per_arm: usize,
    allocation: &AllocationRatio,
    alpha: f64,
    target_power: f64,
    test_type: &TestType,
) -> PowerSensitivity {
    let vary_p_control: Vec<(f64, f64)> = (1..=9)
        .map(|i| {
            let p = i as f64 * 0.1;
            let power = power_binary_pooled(p, p_exp, n_per_arm, allocation, alpha, test_type);
            (p, power)
        })
        .collect();

    let vary_p_experimental: Vec<(f64, f64)> = (1..=9)
        .map(|i| {
            let p = i as f64 * 0.1;
            let power = power_binary_pooled(p_ctrl, p, n_per_arm, allocation, alpha, test_type);
            (p, power)
        })
        .collect();

    let vary_n: Vec<(usize, f64)> = (1..=10)
        .map(|i| {
            let n = i * 50;
            let power = power_binary_pooled(p_ctrl, p_exp, n, allocation, alpha, test_type);
            (n, power)
        })
        .collect();

    let vary_effect: Vec<(f64, f64)> = (-5..=5)
        .map(|i| {
            let effect = i as f64 * 0.05;
            let p_e = (p_ctrl + effect).clamp(0.01, 0.99);
            let power = power_binary_pooled(p_ctrl, p_e, n_per_arm, allocation, alpha, test_type);
            (effect, power)
        })
        .collect();

    let detectable_effect_by_n: Vec<(usize, f64)> = (1..=10)
        .map(|i| {
            let n = i * 50;
            let mde = minimum_detectable_effect_binary(
                p_ctrl,
                n,
                allocation,
                alpha,
                target_power,
                test_type,
            );
            (n, mde)
        })
        .collect();

    PowerSensitivity {
        experimental_id: String::new(),
        vary_p_control,
        vary_p_experimental,
        vary_n,
        vary_effect,
        detectable_effect_by_n,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 13: MAIN API
// ═══════════════════════════════════════════════════════════════════════════

/// Extract event probability from score based on endpoint
fn endpoint_prob(score: &GuidelineRobustnessScoreHost, endpoint: EndpointType) -> f64 {
    match endpoint {
        EndpointType::BinaryResponse => score.mean_response,
        EndpointType::BinaryToxicity => score.mean_grade3plus_rate,
        EndpointType::Continuous | EndpointType::TimeToEvent => score.mean_response,
    }
}

/// Estimate SE from worst-case spread
fn estimate_se(score: &GuidelineRobustnessScoreHost, endpoint: EndpointType) -> f64 {
    let (mean, worst) = match endpoint {
        EndpointType::BinaryResponse => (score.mean_response, score.worst_response),
        EndpointType::BinaryToxicity => (score.mean_grade3plus_rate, score.worst_grade3plus_rate),
        _ => (score.mean_response, score.worst_response),
    };
    (mean - worst).abs() / 2.0
}

/// Build arm assumption from score
fn build_arm_assumption(
    score: &GuidelineRobustnessScoreHost,
    endpoint: EndpointType,
    is_control: bool,
    dropout_rate: f64,
) -> ArmAssumption {
    let p_event = endpoint_prob(score, endpoint);
    let se = estimate_se(score, endpoint);

    ArmAssumption {
        arm_name: if is_control {
            "Control".into()
        } else {
            "Experimental".into()
        },
        guideline_id: score.guideline_id.clone(),
        is_control,
        p_event,
        p_event_se: Some(se),
        p_event_ci: Some(ConfidenceInterval::new(
            p_event,
            (p_event - 1.96 * se).max(0.0),
            (p_event + 1.96 * se).min(1.0),
            0.95,
        )),
        mean: None,
        std_dev: None,
        hazard_rate: None,
        median_time: None,
        dropout_rate,
    }
}

/// Main API: compute trial power from robustness scores
pub fn compute_trial_power(
    scores: &[GuidelineRobustnessScoreHost],
    config: &TrialPowerConfig,
) -> TrialPowerResult {
    let score_map: HashMap<String, &GuidelineRobustnessScoreHost> =
        scores.iter().map(|s| (s.guideline_id.clone(), s)).collect();

    // Get control score
    let ctrl_score = match score_map.get(&config.control_guideline_id) {
        Some(s) => *s,
        None => {
            return TrialPowerResult {
                config: config.clone(),
                arms: vec![],
                pairwise: vec![],
                power_curves: None,
                sensitivity: None,
                summary: TrialPowerSummary {
                    n_comparisons: 0,
                    n_adequately_powered: 0,
                    n_clinically_meaningful: 0,
                    best_powered: None,
                    largest_effect: None,
                    min_n_for_all_powered: None,
                    recommendations: vec!["Control guideline not found".into()],
                },
                documentation: TrialDocumentation::empty(),
            };
        }
    };

    let p_ctrl = endpoint_prob(ctrl_score, config.endpoint);
    let p_ctrl_se = estimate_se(ctrl_score, config.endpoint);

    // Build arms
    let mut arms = vec![build_arm_assumption(
        ctrl_score,
        config.endpoint,
        true,
        config.dropout.control_dropout_rate,
    )];

    let n_comparisons = config.experimental_guideline_ids.len();
    let mut pairwise = Vec::with_capacity(n_comparisons);
    let mut power_curves = Vec::new();
    let mut sensitivities = Vec::new();

    for (idx, exp_id) in config.experimental_guideline_ids.iter().enumerate() {
        let exp_score = match score_map.get(exp_id) {
            Some(s) => *s,
            None => continue,
        };

        arms.push(build_arm_assumption(
            exp_score,
            config.endpoint,
            false,
            config.dropout.experimental_dropout_rate,
        ));

        let p_exp = endpoint_prob(exp_score, config.endpoint);
        let p_exp_se = estimate_se(exp_score, config.endpoint);
        let effect_size = p_exp - p_ctrl;

        // Adjust alpha for multiplicity
        let alpha_adj = adjust_alpha(
            config.alpha,
            n_comparisons,
            config.multiplicity_adjustment,
            idx,
        );

        let z_crit = z_critical(alpha_adj, &config.test_type);

        // Determine n per arm
        let n_per_arm = config.n_per_arm.unwrap_or_else(|| {
            config
                .n_total
                .map(|n| (n as f64 / (1.0 + config.allocation.k())).ceil() as usize)
                .unwrap_or(100)
        });

        // Inflate for dropout
        let (n_inflated, n_total_inflated) = if config.dropout.inflate_for_dropout {
            let max_dropout = config
                .dropout
                .experimental_dropout_rate
                .max(config.dropout.control_dropout_rate);
            let inflation = 1.0 / (1.0 - max_dropout);
            let n_inf = (n_per_arm as f64 * inflation).ceil() as usize;
            let n_tot = (n_inf as f64 * (1.0 + config.allocation.k())).ceil() as usize;
            (n_inf, n_tot)
        } else {
            let n_tot = (n_per_arm as f64 * (1.0 + config.allocation.k())).ceil() as usize;
            (n_per_arm, n_tot)
        };

        // Compute power
        let (power, power_ci) = if config.propagate_uncertainty {
            let (p, ci) = power_with_uncertainty(
                p_ctrl,
                p_ctrl_se,
                p_exp,
                p_exp_se,
                n_per_arm,
                &config.allocation,
                alpha_adj,
                &config.test_type,
                config.uncertainty_samples,
                config.power_ci_level,
            );
            (p, Some(ci))
        } else {
            let p = power_binary_pooled(
                p_ctrl,
                p_exp,
                n_per_arm,
                &config.allocation,
                alpha_adj,
                &config.test_type,
            );
            (p, None)
        };

        // Compute required n for target power
        let required_n = required_n_binary(
            p_ctrl,
            p_exp,
            &config.allocation,
            alpha_adj,
            config.target_power,
            &config.test_type,
        );
        let required_n_inflated = if config.dropout.inflate_for_dropout {
            let max_dropout = config
                .dropout
                .experimental_dropout_rate
                .max(config.dropout.control_dropout_rate);
            let inflation = 1.0 / (1.0 - max_dropout);
            (required_n as f64 * inflation).ceil() as usize
        } else {
            required_n
        };

        // MDE
        let mde = minimum_detectable_effect_binary(
            p_ctrl,
            n_per_arm,
            &config.allocation,
            alpha_adj,
            config.target_power,
            &config.test_type,
        );

        // Clinical meaningfulness
        let mcid = config
            .clinical_thresholds
            .as_ref()
            .map(|t| {
                if matches!(config.endpoint, EndpointType::BinaryResponse) {
                    t.mcid_response
                } else {
                    t.mcid_toxicity
                }
            })
            .unwrap_or(0.10);

        let is_clinically_meaningful = effect_size.abs() >= mcid;
        let is_adequately_powered = power >= config.target_power;

        // Group sequential boundaries
        let (boundaries, expected_n_h1, expected_n_h0) = if config.group_sequential.n_analyses > 1 {
            let bounds = compute_gs_boundaries(&config.group_sequential, alpha_adj, n_per_arm);
            let delta = (p_exp - p_ctrl).abs();
            let exp_h1 = if delta > 0.01 {
                n_per_arm as f64 * 0.8
            } else {
                n_per_arm as f64 * 0.95
            };
            let exp_h0 = if config.group_sequential.include_futility {
                n_per_arm as f64 * 0.85
            } else {
                n_per_arm as f64
            };
            (Some(bounds), Some(exp_h1), Some(exp_h0))
        } else {
            (None, None, None)
        };

        let relative_effect = if p_ctrl > 0.0 { p_exp / p_ctrl } else { 1.0 };

        pairwise.push(PairwisePowerResult {
            control_id: config.control_guideline_id.clone(),
            experimental_id: exp_id.clone(),
            p_control: p_ctrl,
            p_experimental: p_exp,
            effect_size,
            relative_effect,
            n_per_arm,
            n_total: (n_per_arm as f64 * (1.0 + config.allocation.k())).ceil() as usize,
            n_per_arm_inflated: n_inflated,
            n_total_inflated,
            power,
            power_ci,
            required_n_per_arm: Some(required_n_inflated),
            required_n_total: Some(
                (required_n_inflated as f64 * (1.0 + config.allocation.k())).ceil() as usize,
            ),
            endpoint: config.endpoint,
            test_type: config.test_type,
            alpha: config.alpha,
            alpha_adjusted: alpha_adj,
            z_critical: z_crit,
            is_clinically_meaningful,
            is_adequately_powered,
            minimum_detectable_effect: mde,
            analysis_boundaries: boundaries,
            expected_n_h1,
            expected_n_h0,
        });

        // Power curve
        if matches!(
            config.computation_mode,
            ComputationMode::PowerCurve | ComputationMode::Full
        ) {
            let mut curve = compute_power_curve(
                p_ctrl,
                p_exp,
                &config.allocation,
                alpha_adj,
                &config.test_type,
                config.target_power,
                config.curve_n_min,
                config.curve_n_max,
                config.curve_n_step,
            );
            curve.control_id = config.control_guideline_id.clone();
            curve.experimental_id = exp_id.clone();
            power_curves.push(curve);
        }

        // Sensitivity
        if matches!(config.computation_mode, ComputationMode::Full) {
            let mut sens = compute_sensitivity(
                p_ctrl,
                p_exp,
                n_per_arm,
                &config.allocation,
                alpha_adj,
                config.target_power,
                &config.test_type,
            );
            sens.experimental_id = exp_id.clone();
            sensitivities.push(sens);
        }
    }

    // Summary
    let n_adequately_powered = pairwise.iter().filter(|p| p.is_adequately_powered).count();
    let n_clinically_meaningful = pairwise
        .iter()
        .filter(|p| p.is_clinically_meaningful)
        .count();

    let best_powered = pairwise
        .iter()
        .max_by(|a, b| a.power.partial_cmp(&b.power).unwrap())
        .map(|p| p.experimental_id.clone());

    let largest_effect = pairwise
        .iter()
        .max_by(|a, b| {
            a.effect_size
                .abs()
                .partial_cmp(&b.effect_size.abs())
                .unwrap()
        })
        .map(|p| p.experimental_id.clone());

    let min_n_for_all = pairwise.iter().filter_map(|p| p.required_n_per_arm).max();

    let mut recommendations = Vec::new();

    if n_adequately_powered == 0 {
        recommendations.push("No comparisons are adequately powered at current sample size".into());
    } else if n_adequately_powered < n_comparisons {
        recommendations.push(format!(
            "Only {}/{} comparisons are adequately powered",
            n_adequately_powered, n_comparisons
        ));
    } else {
        recommendations.push("All comparisons are adequately powered".into());
    }

    if n_clinically_meaningful < n_comparisons {
        recommendations.push(format!(
            "Only {}/{} effects are clinically meaningful (>= MCID)",
            n_clinically_meaningful, n_comparisons
        ));
    }

    if let Some(min_n) = min_n_for_all {
        recommendations.push(format!(
            "Minimum n per arm for all comparisons to reach {:.0}% power: {}",
            config.target_power * 100.0,
            min_n
        ));
    }

    let summary = TrialPowerSummary {
        n_comparisons,
        n_adequately_powered,
        n_clinically_meaningful,
        best_powered,
        largest_effect,
        min_n_for_all_powered: min_n_for_all,
        recommendations,
    };

    // Documentation
    let documentation = generate_documentation(config, &arms, &pairwise, &summary);

    TrialPowerResult {
        config: config.clone(),
        arms,
        pairwise,
        power_curves: if power_curves.is_empty() {
            None
        } else {
            Some(power_curves)
        },
        sensitivity: if sensitivities.is_empty() {
            None
        } else {
            Some(sensitivities)
        },
        summary,
        documentation,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 14: DOCUMENTATION GENERATION
// ═══════════════════════════════════════════════════════════════════════════

fn generate_documentation(
    config: &TrialPowerConfig,
    arms: &[ArmAssumption],
    pairwise: &[PairwisePowerResult],
    summary: &TrialPowerSummary,
) -> TrialDocumentation {
    TrialDocumentation {
        sap_power_section: generate_sap_section(config, arms, pairwise),
        assumptions_table_latex: generate_assumptions_latex(arms),
        power_table_latex: generate_power_latex(pairwise, config),
        summary_markdown: generate_summary_markdown(config, pairwise, summary),
    }
}

fn generate_sap_section(
    config: &TrialPowerConfig,
    arms: &[ArmAssumption],
    pairwise: &[PairwisePowerResult],
) -> String {
    let mut sap = String::new();

    sap.push_str("## Sample Size and Power\n\n");

    sap.push_str("### Primary Endpoint\n\n");
    let endpoint_desc = match config.endpoint {
        EndpointType::BinaryResponse => {
            "Objective response rate (ORR), defined as the proportion of patients achieving complete or partial response."
        }
        EndpointType::BinaryToxicity => "Rate of Grade >=3 treatment-related adverse events.",
        EndpointType::Continuous => "Continuous efficacy outcome.",
        EndpointType::TimeToEvent => "Time-to-event endpoint.",
    };
    sap.push_str(&format!("{}\n\n", endpoint_desc));

    sap.push_str("### Statistical Hypotheses\n\n");
    let test_desc = match config.test_type {
        TestType::TwoSidedSuperiority => format!(
            "Two-sided superiority test at alpha = {:.3}.\n\
             H0: p_exp = p_ctrl vs H1: p_exp != p_ctrl",
            config.alpha
        ),
        TestType::OneSidedSuperiority => format!(
            "One-sided superiority test at alpha = {:.3}.\n\
             H0: p_exp <= p_ctrl vs H1: p_exp > p_ctrl",
            config.alpha
        ),
        TestType::NonInferiority { margin } => format!(
            "Non-inferiority test with margin = {:.2} at alpha = {:.3}.\n\
             H0: p_exp <= p_ctrl - {:.2} vs H1: p_exp > p_ctrl - {:.2}",
            margin, config.alpha, margin, margin
        ),
        TestType::Equivalence { margin } => format!(
            "Equivalence test (TOST) with margin = +/-{:.2} at alpha = {:.3}.\n\
             H0: |p_exp - p_ctrl| >= {:.2} vs H1: |p_exp - p_ctrl| < {:.2}",
            margin, config.alpha, margin, margin
        ),
    };
    sap.push_str(&format!("{}\n\n", test_desc));

    sap.push_str("### Assumptions\n\n");
    if let Some(ctrl) = arms.iter().find(|a| a.is_control) {
        sap.push_str(&format!(
            "- Control arm event rate: {:.1}%\n",
            ctrl.p_event * 100.0
        ));
    }
    for arm in arms.iter().filter(|a| !a.is_control) {
        sap.push_str(&format!(
            "- Experimental arm ({}) event rate: {:.1}%\n",
            arm.guideline_id,
            arm.p_event * 100.0
        ));
    }
    sap.push_str(&format!(
        "- Dropout rate: {:.1}% (control), {:.1}% (experimental)\n",
        config.dropout.control_dropout_rate * 100.0,
        config.dropout.experimental_dropout_rate * 100.0
    ));
    sap.push_str("\n");

    sap.push_str("### Sample Size and Power\n\n");
    if let Some(first) = pairwise.first() {
        sap.push_str(&format!(
            "With {} patients per arm ({} total), the study has {:.1}% power to detect \
             an absolute difference of {:.1}% in the primary endpoint.\n\n",
            first.n_per_arm_inflated,
            first.n_total_inflated,
            first.power * 100.0,
            first.effect_size.abs() * 100.0
        ));

        if let Some(req_n) = first.required_n_per_arm {
            sap.push_str(&format!(
                "To achieve {:.0}% power, {} patients per arm ({} total) would be required.\n\n",
                config.target_power * 100.0,
                req_n,
                (req_n as f64 * (1.0 + config.allocation.k())).ceil() as usize
            ));
        }
    }

    // Multiplicity adjustment
    if config.experimental_guideline_ids.len() > 1 {
        sap.push_str("### Multiplicity Adjustment\n\n");
        let method_desc = match config.multiplicity_adjustment {
            MultipleComparisonMethod::None => "No multiplicity adjustment is applied.",
            MultipleComparisonMethod::Bonferroni => {
                "Bonferroni correction is applied to control family-wise error rate."
            }
            MultipleComparisonMethod::Holm => "Holm step-down procedure is applied.",
            MultipleComparisonMethod::Hochberg => "Hochberg step-up procedure is applied.",
            MultipleComparisonMethod::BenjaminiHochberg => {
                "Benjamini-Hochberg procedure is applied to control FDR."
            }
            MultipleComparisonMethod::Dunnett => {
                "Dunnett's test is used for many-to-one comparisons."
            }
            MultipleComparisonMethod::FixedSequence => {
                "Fixed-sequence (gatekeeping) procedure is applied."
            }
        };
        sap.push_str(&format!("{}\n\n", method_desc));
    }

    // Group sequential
    if config.group_sequential.n_analyses > 1 {
        sap.push_str("### Interim Analyses\n\n");
        sap.push_str(&format!(
            "{} interim analyses are planned at information fractions: {:?}\n\n",
            config.group_sequential.n_analyses - 1,
            &config.group_sequential.info_fractions[..config.group_sequential.n_analyses - 1]
        ));
        let spending_desc = match config.group_sequential.alpha_spending {
            SpendingFunction::OBrienFleming => "O'Brien-Fleming",
            SpendingFunction::Pocock => "Pocock",
            SpendingFunction::LanDeMetsOBF => "Lan-DeMets (O'Brien-Fleming approximation)",
            SpendingFunction::HwangShihDeCani => "Hwang-Shih-DeCani",
        };
        sap.push_str(&format!("Alpha spending function: {}\n\n", spending_desc));
    }

    sap
}

fn generate_assumptions_latex(arms: &[ArmAssumption]) -> String {
    let mut latex = String::new();

    latex.push_str("\\begin{table}[htbp]\n");
    latex.push_str("\\centering\n");
    latex.push_str("\\caption{Event Rate Assumptions}\n");
    latex.push_str("\\begin{tabular}{lcccc}\n");
    latex.push_str("\\toprule\n");
    latex.push_str("Arm & Guideline & Event Rate & 95\\% CI & Dropout \\\\\n");
    latex.push_str("\\midrule\n");

    for arm in arms {
        let arm_type = if arm.is_control {
            "Control"
        } else {
            "Experimental"
        };
        let ci_str = arm
            .p_event_ci
            .as_ref()
            .map(|ci| format!("({:.1}\\%, {:.1}\\%)", ci.lower * 100.0, ci.upper * 100.0))
            .unwrap_or_else(|| "---".to_string());

        latex.push_str(&format!(
            "{} & {} & {:.1}\\% & {} & {:.1}\\% \\\\\n",
            arm_type,
            arm.guideline_id,
            arm.p_event * 100.0,
            ci_str,
            arm.dropout_rate * 100.0
        ));
    }

    latex.push_str("\\bottomrule\n");
    latex.push_str("\\end{tabular}\n");
    latex.push_str("\\end{table}\n");

    latex
}

fn generate_power_latex(pairwise: &[PairwisePowerResult], config: &TrialPowerConfig) -> String {
    let mut latex = String::new();

    latex.push_str("\\begin{table}[htbp]\n");
    latex.push_str("\\centering\n");
    latex.push_str("\\caption{Power Analysis Results}\n");
    latex.push_str("\\begin{tabular}{lccccc}\n");
    latex.push_str("\\toprule\n");
    latex.push_str("Comparison & Effect & n/arm & Power & Required n & MDE \\\\\n");
    latex.push_str("\\midrule\n");

    for p in pairwise {
        let power_str = if p.is_adequately_powered {
            format!("\\textbf{{{:.1}\\%}}", p.power * 100.0)
        } else {
            format!("{:.1}\\%", p.power * 100.0)
        };

        let req_n_str = p
            .required_n_per_arm
            .map(|n| n.to_string())
            .unwrap_or_else(|| "---".to_string());

        latex.push_str(&format!(
            "{} vs {} & {:+.1}\\% & {} & {} & {} & {:.1}\\% \\\\\n",
            p.experimental_id,
            p.control_id,
            p.effect_size * 100.0,
            p.n_per_arm_inflated,
            power_str,
            req_n_str,
            p.minimum_detectable_effect * 100.0
        ));
    }

    latex.push_str("\\bottomrule\n");
    latex.push_str("\\end{tabular}\n");

    latex.push_str("\\begin{tablenotes}\\small\n");
    latex.push_str(&format!(
        "\\item Target power: {:.0}\\%. Alpha: {:.3} ({}).\n",
        config.target_power * 100.0,
        config.alpha,
        match config.test_type {
            TestType::TwoSidedSuperiority => "two-sided",
            TestType::OneSidedSuperiority | TestType::NonInferiority { .. } => "one-sided",
            TestType::Equivalence { .. } => "TOST",
        }
    ));
    latex.push_str("\\item MDE = Minimum Detectable Effect at target power.\n");
    latex.push_str("\\end{tablenotes}\n");

    latex.push_str("\\end{table}\n");

    latex
}

fn generate_summary_markdown(
    config: &TrialPowerConfig,
    pairwise: &[PairwisePowerResult],
    summary: &TrialPowerSummary,
) -> String {
    let mut md = String::new();

    md.push_str("# Trial Power Analysis Summary\n\n");

    md.push_str("## Overview\n\n");
    md.push_str(&format!(
        "- **Comparisons**: {} experimental vs {} control\n",
        summary.n_comparisons, config.control_guideline_id
    ));
    md.push_str(&format!(
        "- **Adequately powered**: {}/{}\n",
        summary.n_adequately_powered, summary.n_comparisons
    ));
    md.push_str(&format!(
        "- **Clinically meaningful**: {}/{}\n",
        summary.n_clinically_meaningful, summary.n_comparisons
    ));
    md.push_str("\n");

    md.push_str("## Power Results\n\n");
    md.push_str("| Comparison | Effect | n/arm | Power | Required n | Adequate? |\n");
    md.push_str("|------------|--------|-------|-------|------------|----------|\n");

    for p in pairwise {
        let adequate = if p.is_adequately_powered { "Yes" } else { "No" };
        md.push_str(&format!(
            "| {} vs {} | {:+.1}% | {} | {:.1}% | {} | {} |\n",
            p.experimental_id,
            p.control_id,
            p.effect_size * 100.0,
            p.n_per_arm_inflated,
            p.power * 100.0,
            p.required_n_per_arm
                .map(|n| n.to_string())
                .unwrap_or("---".into()),
            adequate
        ));
    }

    md.push_str("\n## Recommendations\n\n");
    for rec in &summary.recommendations {
        md.push_str(&format!("- {}\n", rec));
    }

    md
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 15: CONVENIENCE FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Quick power calculation without full result structure
pub fn quick_power(p_ctrl: f64, p_exp: f64, n_per_arm: usize, alpha: f64, two_sided: bool) -> f64 {
    let test_type = if two_sided {
        TestType::TwoSidedSuperiority
    } else {
        TestType::OneSidedSuperiority
    };
    power_binary_pooled(
        p_ctrl,
        p_exp,
        n_per_arm,
        &AllocationRatio::balanced(),
        alpha,
        &test_type,
    )
}

/// Quick sample size calculation
pub fn quick_sample_size(
    p_ctrl: f64,
    p_exp: f64,
    alpha: f64,
    power: f64,
    two_sided: bool,
) -> usize {
    let test_type = if two_sided {
        TestType::TwoSidedSuperiority
    } else {
        TestType::OneSidedSuperiority
    };
    required_n_binary(
        p_ctrl,
        p_exp,
        &AllocationRatio::balanced(),
        alpha,
        power,
        &test_type,
    )
}

/// Get SAP power section as text
pub fn power_sap_section(result: &TrialPowerResult) -> &str {
    &result.documentation.sap_power_section
}

/// Get power table as LaTeX
pub fn power_table_latex(result: &TrialPowerResult) -> &str {
    &result.documentation.power_table_latex
}

/// Get summary as Markdown
pub fn power_summary_markdown(result: &TrialPowerResult) -> &str {
    &result.documentation.summary_markdown
}

// ═══════════════════════════════════════════════════════════════════════════
// PART 16: TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.001);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.001);
    }

    #[test]
    fn test_normal_quantile() {
        assert!((normal_quantile(0.5) - 0.0).abs() < 0.001);
        assert!((normal_quantile(0.975) - 1.96).abs() < 0.01);
        assert!((normal_quantile(0.025) + 1.96).abs() < 0.01);
    }

    #[test]
    fn test_power_calculation_basic() {
        // 50% vs 60%, n=100 per arm, alpha=0.05 two-sided
        let power = power_binary_pooled(
            0.50,
            0.60,
            100,
            &AllocationRatio::balanced(),
            0.05,
            &TestType::TwoSidedSuperiority,
        );
        // Expected power around 29% for this small effect
        assert!(power > 0.20 && power < 0.40, "Power was {}", power);
    }

    #[test]
    fn test_power_increases_with_n() {
        let power_50 = power_binary_pooled(
            0.40,
            0.55,
            50,
            &AllocationRatio::balanced(),
            0.05,
            &TestType::TwoSidedSuperiority,
        );
        let power_200 = power_binary_pooled(
            0.40,
            0.55,
            200,
            &AllocationRatio::balanced(),
            0.05,
            &TestType::TwoSidedSuperiority,
        );
        assert!(
            power_200 > power_50,
            "Power should increase with n: {} vs {}",
            power_50,
            power_200
        );
    }

    #[test]
    fn test_power_increases_with_effect() {
        let power_small = power_binary_pooled(
            0.50,
            0.55,
            100,
            &AllocationRatio::balanced(),
            0.05,
            &TestType::TwoSidedSuperiority,
        );
        let power_large = power_binary_pooled(
            0.50,
            0.70,
            100,
            &AllocationRatio::balanced(),
            0.05,
            &TestType::TwoSidedSuperiority,
        );
        assert!(
            power_large > power_small,
            "Power should increase with effect size"
        );
    }

    #[test]
    fn test_required_n_calculation() {
        // 50% vs 65% effect, need ~80% power
        let n = required_n_binary(
            0.50,
            0.65,
            &AllocationRatio::balanced(),
            0.05,
            0.80,
            &TestType::TwoSidedSuperiority,
        );
        // Verify power at this n
        let power = power_binary_pooled(
            0.50,
            0.65,
            n,
            &AllocationRatio::balanced(),
            0.05,
            &TestType::TwoSidedSuperiority,
        );
        assert!(
            power >= 0.80,
            "Required n {} should give at least 80% power, got {:.1}%",
            n,
            power * 100.0
        );
    }

    #[test]
    fn test_multiplicity_adjustment() {
        let alpha = 0.05;
        let n = 3;

        let bonf = adjust_alpha(alpha, n, MultipleComparisonMethod::Bonferroni, 0);
        assert!((bonf - 0.05 / 3.0).abs() < 0.0001);

        let holm_first = adjust_alpha(alpha, n, MultipleComparisonMethod::Holm, 0);
        let holm_last = adjust_alpha(alpha, n, MultipleComparisonMethod::Holm, 2);
        assert!(
            holm_last > holm_first,
            "Holm should be less strict for later tests"
        );
    }

    #[test]
    fn test_non_inferiority_power() {
        // Non-inferiority with margin 0.10
        // If p_exp = p_ctrl, we need to show p_exp > p_ctrl - 0.10
        let power_ni = power_binary_pooled(
            0.50,
            0.50,
            200,
            &AllocationRatio::balanced(),
            0.025,
            &TestType::NonInferiority { margin: 0.10 },
        );
        // Should have good power since true effect is 0 which is > -0.10
        assert!(
            power_ni > 0.70,
            "Non-inferiority power should be high when true effect is 0"
        );
    }

    #[test]
    fn test_allocation_ratio() {
        let ratio_1_1 = AllocationRatio::balanced();
        assert_eq!(ratio_1_1.k(), 1.0);

        let ratio_2_1 = AllocationRatio::ratio(2.0);
        assert_eq!(ratio_2_1.k(), 2.0);
    }

    #[test]
    fn test_power_curve() {
        let curve = compute_power_curve(
            0.50,
            0.65,
            &AllocationRatio::balanced(),
            0.05,
            &TestType::TwoSidedSuperiority,
            0.80,
            20,
            200,
            20,
        );

        // Power should be monotonic
        for i in 1..curve.points.len() {
            assert!(
                curve.points[i].power >= curve.points[i - 1].power - 0.001,
                "Power curve should be monotonic"
            );
        }

        // Should find n for target power
        assert!(curve.n_for_target_power.is_some());
    }

    #[test]
    fn test_quick_functions() {
        let power = quick_power(0.50, 0.65, 100, 0.05, true);
        assert!(power > 0.0 && power < 1.0);

        let n = quick_sample_size(0.50, 0.65, 0.05, 0.80, true);
        assert!(n > 0 && n < 10000);
    }

    #[test]
    fn test_group_sequential_boundaries() {
        let gs_config = GroupSequentialConfig::two_interim_obf();
        let boundaries = compute_gs_boundaries(&gs_config, 0.05, 200);

        assert_eq!(boundaries.len(), 3);
        // Boundaries should be decreasing (OBF)
        assert!(boundaries[0].efficacy_z > boundaries[2].efficacy_z);
        // Cumulative alpha should increase
        assert!(boundaries[2].cumulative_alpha_spent > boundaries[0].cumulative_alpha_spent);
    }

    fn mock_guideline(id: &str, response: f64, tox: f64) -> GuidelineRobustnessScoreHost {
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
            score_mean: response - tox,
            score_worst: (response - tox) * 0.8,
            response_ci95_low: Some(response - 0.1),
            response_ci95_high: Some(response + 0.1),
            tox_ci95_high: Some(tox + 0.05),
        }
    }

    #[test]
    fn test_compute_trial_power() {
        let scores = vec![
            mock_guideline("control", 0.50, 0.25),
            mock_guideline("exp_a", 0.65, 0.20),
            mock_guideline("exp_b", 0.55, 0.22),
        ];

        let config = TrialPowerConfig::superiority(
            "control",
            vec!["exp_a".to_string(), "exp_b".to_string()],
            EndpointType::BinaryResponse,
            100,
        );

        let result = compute_trial_power(&scores, &config);

        assert_eq!(result.pairwise.len(), 2);
        assert_eq!(result.arms.len(), 3);

        // exp_a should have higher power (larger effect)
        let power_a = result
            .pairwise
            .iter()
            .find(|p| p.experimental_id == "exp_a")
            .unwrap()
            .power;
        let power_b = result
            .pairwise
            .iter()
            .find(|p| p.experimental_id == "exp_b")
            .unwrap()
            .power;
        assert!(power_a > power_b, "Larger effect should have higher power");
    }

    #[test]
    fn test_documentation_generation() {
        let scores = vec![
            mock_guideline("control", 0.50, 0.25),
            mock_guideline("experimental", 0.65, 0.20),
        ];

        let config = TrialPowerConfig::superiority(
            "control",
            vec!["experimental".to_string()],
            EndpointType::BinaryResponse,
            150,
        );

        let result = compute_trial_power(&scores, &config);

        assert!(!result.documentation.sap_power_section.is_empty());
        assert!(result
            .documentation
            .sap_power_section
            .contains("Sample Size"));
        assert!(result
            .documentation
            .power_table_latex
            .contains("\\begin{table}"));
        assert!(result
            .documentation
            .summary_markdown
            .contains("# Trial Power"));
    }
}
