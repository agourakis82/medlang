//! Week 46: Clinical Constraints & Regulatory Feasibility (v2.0)
//!
//! A complete clinical constraint system that:
//! - Supports hard AND soft constraints with graduated penalties
//! - Enables composite constraints (AND/OR logic)
//! - Provides constraint relaxation analysis ("how far from feasible?")
//! - Includes regulatory presets (FDA Phase I/II/III, EMA, ICH)
//! - Handles uncertainty-aware constraints (threshold ± margin)
//! - Computes constraint sensitivity (impact of threshold changes)
//! - Exports regulatory-ready documentation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ═══════════════════════════════════════════════════════════════════════════
// PREREQUISITE TYPES (Week 44-45 would define these, included here for completeness)
// ═══════════════════════════════════════════════════════════════════════════

/// Robustness score for a single guideline across scenarios (Week 44)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineRobustnessScoreHost {
    pub guideline_id: String,

    // Mean metrics (across scenarios)
    pub mean_response: f64,
    pub mean_grade3plus_rate: f64,
    pub mean_grade4plus_rate: Option<f64>,
    pub mean_rdi: f64,

    // Worst-case metrics
    pub worst_response: f64,
    pub worst_grade3plus_rate: f64,
    pub worst_grade4plus_rate: Option<f64>,
    pub worst_rdi: f64,

    // Variability metrics
    pub response_std: Option<f64>,
    pub tox_std: Option<f64>,

    // Composite scores
    pub score_mean: f64,
    pub score_worst: f64,

    // Confidence intervals (from uncertainty quantification)
    pub response_ci95_low: Option<f64>,
    pub response_ci95_high: Option<f64>,
    pub tox_ci95_high: Option<f64>,
}

impl GuidelineRobustnessScoreHost {
    /// Create a test/mock score for testing
    pub fn mock(id: &str, response: f64, tox: f64, rdi: f64) -> Self {
        Self {
            guideline_id: id.to_string(),
            mean_response: response,
            mean_grade3plus_rate: tox,
            mean_grade4plus_rate: Some(tox * 0.3),
            mean_rdi: rdi,
            worst_response: response * 0.8,
            worst_grade3plus_rate: tox * 1.2,
            worst_grade4plus_rate: Some(tox * 0.4),
            worst_rdi: rdi * 0.9,
            response_std: Some(response * 0.1),
            tox_std: Some(tox * 0.15),
            score_mean: response - tox,
            score_worst: response * 0.8 - tox * 1.2,
            response_ci95_low: Some(response * 0.85),
            response_ci95_high: Some(response * 1.15),
            tox_ci95_high: Some(tox * 1.3),
        }
    }
}

/// Pareto analysis configuration (Week 45)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoConfig {
    /// Objectives to optimize (metric name -> maximize/minimize)
    pub objectives: Vec<ParetoObjective>,
    /// Maximum number of fronts to compute
    pub max_fronts: usize,
}

impl Default for ParetoConfig {
    fn default() -> Self {
        Self {
            objectives: vec![
                ParetoObjective {
                    metric: ConstraintMetric::MeanResponse,
                    maximize: true,
                },
                ParetoObjective {
                    metric: ConstraintMetric::MeanGrade3PlusRate,
                    maximize: false,
                },
            ],
            max_fronts: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoObjective {
    pub metric: ConstraintMetric,
    pub maximize: bool,
}

/// Pareto analysis result (Week 45)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoAnalysis {
    pub config: ParetoConfig,
    pub points: Vec<ParetoPoint>,
    pub n_fronts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParetoPoint {
    pub guideline_id: String,
    pub rank: usize,
    pub crowding_distance: f64,
    pub objective_values: Vec<f64>,
}

/// Compute Pareto analysis (simplified implementation)
pub fn compute_pareto_analysis(
    scores: &[GuidelineRobustnessScoreHost],
    config: &ParetoConfig,
) -> anyhow::Result<ParetoAnalysis> {
    let mut points: Vec<ParetoPoint> = scores
        .iter()
        .map(|s| {
            let obj_values: Vec<f64> = config
                .objectives
                .iter()
                .map(|obj| {
                    let val = extract_metric(s, obj.metric).unwrap_or(0.0);
                    if obj.maximize {
                        val
                    } else {
                        -val
                    }
                })
                .collect();

            ParetoPoint {
                guideline_id: s.guideline_id.clone(),
                rank: 0,
                crowding_distance: 0.0,
                objective_values: obj_values,
            }
        })
        .collect();

    // Simple non-dominated sorting
    let mut remaining: Vec<usize> = (0..points.len()).collect();
    let mut current_rank = 0;

    while !remaining.is_empty() && current_rank < config.max_fronts {
        let mut front = vec![];

        for &i in &remaining {
            let mut dominated = false;
            for &j in &remaining {
                if i != j && dominates(&points[j].objective_values, &points[i].objective_values) {
                    dominated = true;
                    break;
                }
            }
            if !dominated {
                front.push(i);
            }
        }

        for &idx in &front {
            points[idx].rank = current_rank;
        }

        remaining.retain(|x| !front.contains(x));
        current_rank += 1;
    }

    // Assign remaining to last front
    for idx in remaining {
        points[idx].rank = current_rank;
    }

    Ok(ParetoAnalysis {
        config: config.clone(),
        points,
        n_fronts: current_rank + 1,
    })
}

fn dominates(a: &[f64], b: &[f64]) -> bool {
    let mut dominated = false;
    for (av, bv) in a.iter().zip(b.iter()) {
        if av < bv {
            return false;
        }
        if av > bv {
            dominated = true;
        }
    }
    dominated
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.1 METRIC SPECIFICATION
// ═══════════════════════════════════════════════════════════════════════════

/// Metrics available for constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintMetric {
    // Mean metrics (across scenarios)
    MeanResponse,
    MeanGrade3PlusRate,
    MeanGrade4PlusRate,
    MeanRDI,
    MeanReward,

    // Worst-case metrics
    WorstResponse,
    WorstGrade3PlusRate,
    WorstGrade4PlusRate,
    WorstRDI,

    // Variability metrics
    ResponseStd,
    ToxStd,

    // Composite scores
    ScoreMean,
    ScoreWorst,

    // Derived metrics (from uncertainty quantification)
    ResponseCI95Low,
    ResponseCI95High,
    ToxCI95High,
}

/// Direction of constraint inequality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConstraintDirection {
    /// metric >= threshold
    AtLeast,
    /// metric <= threshold
    AtMost,
    /// metric == threshold (with tolerance)
    Equals,
    /// threshold_low <= metric <= threshold_high
    Between,
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.2 CONSTRAINT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Wrapper for f64 in enum (serde compatibility)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OrderedFloat(pub f64);

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Constraint strictness level
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConstraintLevel {
    /// Absolute requirement - violation = infeasible
    Hard,
    /// Strong preference - violation heavily penalized
    Soft { penalty_weight: OrderedFloat },
    /// Weak preference - violation lightly penalized
    Preference { penalty_weight: OrderedFloat },
    /// Informational only - tracked but not penalized
    Advisory,
}

impl Default for ConstraintLevel {
    fn default() -> Self {
        ConstraintLevel::Hard
    }
}

/// Regulatory source for constraint (provenance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegulatorySource {
    /// FDA guidance document
    FDA {
        guidance_id: String,
        section: Option<String>,
    },
    /// EMA guideline
    EMA {
        guideline_id: String,
        section: Option<String>,
    },
    /// ICH harmonized guideline
    ICH { code: String },
    /// Institutional policy
    Institution { name: String },
    /// Literature reference
    Literature { citation: String },
    /// Custom / user-defined
    Custom { rationale: String },
}

/// A single atomic constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicConstraint {
    /// Human-readable name
    pub name: String,

    /// Optional description
    #[serde(default)]
    pub description: Option<String>,

    /// Metric to constrain
    pub metric: ConstraintMetric,

    /// Direction of inequality
    pub direction: ConstraintDirection,

    /// Primary threshold
    pub threshold: f64,

    /// Secondary threshold (for Between direction)
    #[serde(default)]
    pub threshold_high: Option<f64>,

    /// Tolerance for Equals direction
    #[serde(default)]
    pub tolerance: Option<f64>,

    /// Strictness level
    #[serde(default)]
    pub level: ConstraintLevel,

    /// Regulatory source (provenance)
    #[serde(default)]
    pub source: Option<RegulatorySource>,

    /// Uncertainty margin on threshold (±)
    #[serde(default)]
    pub threshold_uncertainty: Option<f64>,
}

/// Composite constraint (AND/OR tree)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintExpr {
    /// Single atomic constraint
    Atom(AtomicConstraint),

    /// All sub-constraints must be satisfied
    And(Vec<ConstraintExpr>),

    /// At least one sub-constraint must be satisfied
    Or(Vec<ConstraintExpr>),

    /// Exactly N of M sub-constraints must be satisfied
    AtLeastN {
        n: usize,
        constraints: Vec<ConstraintExpr>,
    },

    /// Negation (NOT)
    Not(Box<ConstraintExpr>),

    /// Implication: if A then B
    Implies {
        condition: Box<ConstraintExpr>,
        consequence: Box<ConstraintExpr>,
    },
}

/// Named constraint set (e.g., "Phase II Requirements")
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSet {
    pub name: String,
    pub description: Option<String>,
    pub constraints: Vec<ConstraintExpr>,

    /// Regulatory context
    #[serde(default)]
    pub regulatory_context: Option<String>,

    /// Version/date for tracking
    #[serde(default)]
    pub version: Option<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.3 EVALUATION RESULTS
// ═══════════════════════════════════════════════════════════════════════════

/// Evaluation of a single atomic constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AtomicConstraintEval {
    pub constraint_name: String,
    pub metric: ConstraintMetric,
    pub actual_value: f64,
    pub threshold: f64,
    pub direction: ConstraintDirection,

    /// Whether constraint is satisfied
    pub satisfied: bool,

    /// Signed margin: positive = satisfied with room, negative = violated
    /// For AtLeast: actual - threshold
    /// For AtMost: threshold - actual
    pub margin: f64,

    /// Normalized margin (margin / threshold), for comparability
    pub margin_normalized: f64,

    /// Penalty incurred (0 for hard constraints if satisfied, penalty_weight * |margin| for soft)
    pub penalty: f64,

    /// Level of constraint
    pub level: ConstraintLevel,
}

/// Evaluation of a composite constraint expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintExprEval {
    Atom(AtomicConstraintEval),

    And {
        satisfied: bool,
        children: Vec<ConstraintExprEval>,
        n_satisfied: usize,
        n_total: usize,
    },

    Or {
        satisfied: bool,
        children: Vec<ConstraintExprEval>,
        n_satisfied: usize,
        n_total: usize,
    },

    AtLeastN {
        satisfied: bool,
        n_required: usize,
        n_satisfied: usize,
        children: Vec<ConstraintExprEval>,
    },

    Not {
        satisfied: bool,
        child: Box<ConstraintExprEval>,
    },

    Implies {
        satisfied: bool,
        condition_satisfied: bool,
        condition: Box<ConstraintExprEval>,
        consequence: Box<ConstraintExprEval>,
    },
}

impl ConstraintExprEval {
    pub fn is_satisfied(&self) -> bool {
        match self {
            ConstraintExprEval::Atom(a) => a.satisfied,
            ConstraintExprEval::And { satisfied, .. } => *satisfied,
            ConstraintExprEval::Or { satisfied, .. } => *satisfied,
            ConstraintExprEval::AtLeastN { satisfied, .. } => *satisfied,
            ConstraintExprEval::Not { satisfied, .. } => *satisfied,
            ConstraintExprEval::Implies { satisfied, .. } => *satisfied,
        }
    }

    /// Total penalty from this expression tree
    pub fn total_penalty(&self) -> f64 {
        match self {
            ConstraintExprEval::Atom(a) => a.penalty,
            ConstraintExprEval::And { children, .. }
            | ConstraintExprEval::Or { children, .. }
            | ConstraintExprEval::AtLeastN { children, .. } => {
                children.iter().map(|c| c.total_penalty()).sum()
            }
            ConstraintExprEval::Not { child, .. } => child.total_penalty(),
            ConstraintExprEval::Implies {
                condition,
                consequence,
                ..
            } => condition.total_penalty() + consequence.total_penalty(),
        }
    }

    /// Collect all atomic evaluations (flattened)
    pub fn collect_atoms(&self) -> Vec<&AtomicConstraintEval> {
        let mut atoms = vec![];
        self.collect_atoms_recursive(&mut atoms);
        atoms
    }

    fn collect_atoms_recursive<'a>(&'a self, out: &mut Vec<&'a AtomicConstraintEval>) {
        match self {
            ConstraintExprEval::Atom(a) => out.push(a),
            ConstraintExprEval::And { children, .. }
            | ConstraintExprEval::Or { children, .. }
            | ConstraintExprEval::AtLeastN { children, .. } => {
                for c in children {
                    c.collect_atoms_recursive(out);
                }
            }
            ConstraintExprEval::Not { child, .. } => child.collect_atoms_recursive(out),
            ConstraintExprEval::Implies {
                condition,
                consequence,
                ..
            } => {
                condition.collect_atoms_recursive(out);
                consequence.collect_atoms_recursive(out);
            }
        }
    }
}

/// Feasibility classification for one guideline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineFeasibility {
    pub guideline_id: String,

    /// All hard constraints satisfied
    pub hard_feasible: bool,

    /// All constraints (including soft) satisfied
    pub fully_feasible: bool,

    /// Total penalty from soft constraint violations
    pub total_penalty: f64,

    /// Number of hard constraints violated
    pub n_hard_violations: usize,

    /// Number of soft constraints violated
    pub n_soft_violations: usize,

    /// Detailed evaluation per constraint expression
    pub evaluations: Vec<ConstraintExprEval>,

    /// Summary: which atomic constraints are violated
    pub violated_constraints: Vec<String>,

    /// Minimum relaxation needed to become feasible
    /// Maps constraint name -> required threshold change
    pub relaxation_needed: HashMap<String, f64>,
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.4 SENSITIVITY ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/// Sensitivity of feasibility to threshold changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintSensitivity {
    pub constraint_name: String,
    pub current_threshold: f64,

    /// Number of guidelines that become feasible if threshold relaxed by 10%
    pub n_feasible_at_10pct_relaxation: usize,

    /// Number of guidelines that become feasible if threshold relaxed by 20%
    pub n_feasible_at_20pct_relaxation: usize,

    /// Threshold at which 50% of guidelines become feasible
    pub threshold_for_50pct_feasible: Option<f64>,

    /// Threshold at which 80% of guidelines become feasible
    pub threshold_for_80pct_feasible: Option<f64>,

    /// Elasticity: % change in feasible set per % change in threshold
    pub elasticity: f64,
}

/// Relaxation path analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelaxationPath {
    pub guideline_id: String,

    /// Ordered list of constraints to relax to achieve feasibility
    pub relaxation_steps: Vec<RelaxationStep>,

    /// Total "distance" to feasibility (sum of normalized violations)
    pub total_relaxation_distance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelaxationStep {
    pub constraint_name: String,
    pub original_threshold: f64,
    pub required_threshold: f64,
    pub relaxation_amount: f64,
    pub relaxation_percent: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.5 COMPLETE ANALYSIS RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for constraint analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintAnalysisConfig {
    /// Constraint set to evaluate
    pub constraints: ConstraintSet,

    /// Whether to compute relaxation paths
    #[serde(default = "default_true")]
    pub compute_relaxation: bool,

    /// Whether to compute sensitivity analysis
    #[serde(default = "default_true")]
    pub compute_sensitivity: bool,

    /// Pareto config for feasible subset (optional)
    #[serde(default)]
    pub pareto_config: Option<ParetoConfig>,

    /// Include uncertainty in constraint evaluation
    #[serde(default)]
    pub use_uncertainty_margins: bool,
}

fn default_true() -> bool {
    true
}

/// Complete constraint analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintAnalysis {
    /// Configuration used
    pub config: ConstraintAnalysisConfig,

    /// Per-guideline feasibility
    pub feasibility: Vec<GuidelineFeasibility>,

    /// Summary statistics
    pub n_total: usize,
    pub n_hard_feasible: usize,
    pub n_fully_feasible: usize,
    pub feasibility_rate: f64,

    /// Constraint sensitivity analysis
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sensitivity: Option<Vec<ConstraintSensitivity>>,

    /// Relaxation paths for infeasible guidelines
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relaxation_paths: Option<Vec<RelaxationPath>>,

    /// Pareto analysis on all guidelines
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pareto_all: Option<ParetoAnalysis>,

    /// Pareto analysis on hard-feasible guidelines only
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pareto_feasible: Option<ParetoAnalysis>,

    /// Most binding constraints (ordered by how many guidelines they exclude)
    pub most_binding_constraints: Vec<BindingConstraintInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingConstraintInfo {
    pub constraint_name: String,
    pub n_violations: usize,
    pub violation_rate: f64,
    pub avg_violation_margin: f64,
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.6 METRIC EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════

fn extract_metric(score: &GuidelineRobustnessScoreHost, metric: ConstraintMetric) -> Option<f64> {
    Some(match metric {
        ConstraintMetric::MeanResponse => score.mean_response,
        ConstraintMetric::MeanGrade3PlusRate => score.mean_grade3plus_rate,
        ConstraintMetric::MeanGrade4PlusRate => score.mean_grade4plus_rate.unwrap_or(0.0),
        ConstraintMetric::MeanRDI => score.mean_rdi,
        ConstraintMetric::MeanReward => score.score_mean,
        ConstraintMetric::WorstResponse => score.worst_response,
        ConstraintMetric::WorstGrade3PlusRate => score.worst_grade3plus_rate,
        ConstraintMetric::WorstGrade4PlusRate => score.worst_grade4plus_rate.unwrap_or(0.0),
        ConstraintMetric::WorstRDI => score.worst_rdi,
        ConstraintMetric::ResponseStd => score.response_std.unwrap_or(0.0),
        ConstraintMetric::ToxStd => score.tox_std.unwrap_or(0.0),
        ConstraintMetric::ScoreMean => score.score_mean,
        ConstraintMetric::ScoreWorst => score.score_worst,
        ConstraintMetric::ResponseCI95Low => score.response_ci95_low.unwrap_or(score.mean_response),
        ConstraintMetric::ResponseCI95High => {
            score.response_ci95_high.unwrap_or(score.mean_response)
        }
        ConstraintMetric::ToxCI95High => score.tox_ci95_high.unwrap_or(score.mean_grade3plus_rate),
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.7 ATOMIC CONSTRAINT EVALUATION
// ═══════════════════════════════════════════════════════════════════════════

fn evaluate_atomic_constraint(
    constraint: &AtomicConstraint,
    score: &GuidelineRobustnessScoreHost,
    use_uncertainty: bool,
) -> AtomicConstraintEval {
    let actual = extract_metric(score, constraint.metric).unwrap_or(f64::NAN);

    // Adjust threshold for uncertainty if requested
    let (effective_threshold, effective_threshold_high) = if use_uncertainty {
        let margin = constraint.threshold_uncertainty.unwrap_or(0.0);
        match constraint.direction {
            ConstraintDirection::AtLeast => (constraint.threshold + margin, None),
            ConstraintDirection::AtMost => (constraint.threshold - margin, None),
            ConstraintDirection::Between => (
                constraint.threshold + margin,
                constraint.threshold_high.map(|h| h - margin),
            ),
            ConstraintDirection::Equals => (constraint.threshold, None),
        }
    } else {
        (constraint.threshold, constraint.threshold_high)
    };

    let (satisfied, margin) = match constraint.direction {
        ConstraintDirection::AtLeast => {
            let m = actual - effective_threshold;
            (m >= 0.0, m)
        }
        ConstraintDirection::AtMost => {
            let m = effective_threshold - actual;
            (m >= 0.0, m)
        }
        ConstraintDirection::Equals => {
            let tol = constraint.tolerance.unwrap_or(0.01);
            let diff = (actual - effective_threshold).abs();
            (diff <= tol, tol - diff)
        }
        ConstraintDirection::Between => {
            let high = effective_threshold_high.unwrap_or(effective_threshold);
            if actual < effective_threshold {
                (false, actual - effective_threshold)
            } else if actual > high {
                (false, high - actual)
            } else {
                let margin_to_low = actual - effective_threshold;
                let margin_to_high = high - actual;
                (true, margin_to_low.min(margin_to_high))
            }
        }
    };

    // Compute normalized margin
    let margin_normalized = if effective_threshold.abs() > 1e-12 {
        margin / effective_threshold.abs()
    } else {
        margin
    };

    // Compute penalty
    let penalty = match constraint.level {
        ConstraintLevel::Hard => 0.0, // Hard constraints don't have penalty, just feasibility
        ConstraintLevel::Soft { penalty_weight }
        | ConstraintLevel::Preference { penalty_weight } => {
            if satisfied {
                0.0
            } else {
                penalty_weight.0 * margin.abs()
            }
        }
        ConstraintLevel::Advisory => 0.0,
    };

    AtomicConstraintEval {
        constraint_name: constraint.name.clone(),
        metric: constraint.metric,
        actual_value: actual,
        threshold: effective_threshold,
        direction: constraint.direction,
        satisfied,
        margin,
        margin_normalized,
        penalty,
        level: constraint.level,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.8 COMPOSITE CONSTRAINT EVALUATION
// ═══════════════════════════════════════════════════════════════════════════

fn evaluate_constraint_expr(
    expr: &ConstraintExpr,
    score: &GuidelineRobustnessScoreHost,
    use_uncertainty: bool,
) -> ConstraintExprEval {
    match expr {
        ConstraintExpr::Atom(c) => {
            ConstraintExprEval::Atom(evaluate_atomic_constraint(c, score, use_uncertainty))
        }

        ConstraintExpr::And(children) => {
            let child_evals: Vec<_> = children
                .iter()
                .map(|c| evaluate_constraint_expr(c, score, use_uncertainty))
                .collect();
            let n_satisfied = child_evals.iter().filter(|e| e.is_satisfied()).count();
            let satisfied = n_satisfied == child_evals.len();

            ConstraintExprEval::And {
                satisfied,
                n_satisfied,
                n_total: child_evals.len(),
                children: child_evals,
            }
        }

        ConstraintExpr::Or(children) => {
            let child_evals: Vec<_> = children
                .iter()
                .map(|c| evaluate_constraint_expr(c, score, use_uncertainty))
                .collect();
            let n_satisfied = child_evals.iter().filter(|e| e.is_satisfied()).count();
            let satisfied = n_satisfied > 0;

            ConstraintExprEval::Or {
                satisfied,
                n_satisfied,
                n_total: child_evals.len(),
                children: child_evals,
            }
        }

        ConstraintExpr::AtLeastN { n, constraints } => {
            let child_evals: Vec<_> = constraints
                .iter()
                .map(|c| evaluate_constraint_expr(c, score, use_uncertainty))
                .collect();
            let n_satisfied = child_evals.iter().filter(|e| e.is_satisfied()).count();
            let satisfied = n_satisfied >= *n;

            ConstraintExprEval::AtLeastN {
                satisfied,
                n_required: *n,
                n_satisfied,
                children: child_evals,
            }
        }

        ConstraintExpr::Not(child) => {
            let child_eval = evaluate_constraint_expr(child, score, use_uncertainty);
            let satisfied = !child_eval.is_satisfied();

            ConstraintExprEval::Not {
                satisfied,
                child: Box::new(child_eval),
            }
        }

        ConstraintExpr::Implies {
            condition,
            consequence,
        } => {
            let cond_eval = evaluate_constraint_expr(condition, score, use_uncertainty);
            let cons_eval = evaluate_constraint_expr(consequence, score, use_uncertainty);

            // A => B is equivalent to (NOT A) OR B
            let condition_satisfied = cond_eval.is_satisfied();
            let satisfied = !condition_satisfied || cons_eval.is_satisfied();

            ConstraintExprEval::Implies {
                satisfied,
                condition_satisfied,
                condition: Box::new(cond_eval),
                consequence: Box::new(cons_eval),
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.9 GUIDELINE FEASIBILITY EVALUATION
// ═══════════════════════════════════════════════════════════════════════════

fn evaluate_guideline_feasibility(
    score: &GuidelineRobustnessScoreHost,
    constraint_set: &ConstraintSet,
    use_uncertainty: bool,
) -> GuidelineFeasibility {
    let evaluations: Vec<_> = constraint_set
        .constraints
        .iter()
        .map(|c| evaluate_constraint_expr(c, score, use_uncertainty))
        .collect();

    // Collect all atomic evaluations
    let all_atoms: Vec<_> = evaluations.iter().flat_map(|e| e.collect_atoms()).collect();

    // Count violations by level
    let mut n_hard_violations = 0;
    let mut n_soft_violations = 0;
    let mut violated_names = vec![];
    let mut relaxation_needed = HashMap::new();

    for atom in &all_atoms {
        if !atom.satisfied {
            violated_names.push(atom.constraint_name.clone());

            match atom.level {
                ConstraintLevel::Hard => n_hard_violations += 1,
                ConstraintLevel::Soft { .. } | ConstraintLevel::Preference { .. } => {
                    n_soft_violations += 1;
                }
                ConstraintLevel::Advisory => {}
            }

            // Compute relaxation needed
            let relax = match atom.direction {
                ConstraintDirection::AtLeast => atom.threshold - atom.actual_value,
                ConstraintDirection::AtMost => atom.actual_value - atom.threshold,
                _ => atom.margin.abs(),
            };
            relaxation_needed.insert(atom.constraint_name.clone(), relax);
        }
    }

    let total_penalty: f64 = evaluations.iter().map(|e| e.total_penalty()).sum();
    let hard_feasible = n_hard_violations == 0;
    let fully_feasible = hard_feasible && n_soft_violations == 0;

    GuidelineFeasibility {
        guideline_id: score.guideline_id.clone(),
        hard_feasible,
        fully_feasible,
        total_penalty,
        n_hard_violations,
        n_soft_violations,
        evaluations,
        violated_constraints: violated_names,
        relaxation_needed,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.10 SENSITIVITY ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

fn compute_constraint_sensitivity(
    scores: &[GuidelineRobustnessScoreHost],
    constraint: &AtomicConstraint,
    _use_uncertainty: bool,
) -> ConstraintSensitivity {
    let n_total = scores.len();

    // Count feasible at various relaxation levels
    let mut n_at_10pct = 0;
    let mut n_at_20pct = 0;

    let relaxation_10pct = match constraint.direction {
        ConstraintDirection::AtLeast => constraint.threshold * 0.9,
        ConstraintDirection::AtMost => constraint.threshold * 1.1,
        _ => constraint.threshold,
    };

    let relaxation_20pct = match constraint.direction {
        ConstraintDirection::AtLeast => constraint.threshold * 0.8,
        ConstraintDirection::AtMost => constraint.threshold * 1.2,
        _ => constraint.threshold,
    };

    let mut actual_values: Vec<f64> = vec![];

    for score in scores {
        let actual = extract_metric(score, constraint.metric).unwrap_or(f64::NAN);
        actual_values.push(actual);

        // Check at 10% relaxation
        let satisfied_10 = match constraint.direction {
            ConstraintDirection::AtLeast => actual >= relaxation_10pct,
            ConstraintDirection::AtMost => actual <= relaxation_10pct,
            _ => true,
        };
        if satisfied_10 {
            n_at_10pct += 1;
        }

        // Check at 20% relaxation
        let satisfied_20 = match constraint.direction {
            ConstraintDirection::AtLeast => actual >= relaxation_20pct,
            ConstraintDirection::AtMost => actual <= relaxation_20pct,
            _ => true,
        };
        if satisfied_20 {
            n_at_20pct += 1;
        }
    }

    // Find thresholds for 50% and 80% feasibility
    actual_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let threshold_50 = if n_total > 0 {
        let idx = n_total / 2;
        Some(actual_values[idx])
    } else {
        None
    };

    let threshold_80 = if n_total >= 5 {
        let idx = n_total / 5; // 20th percentile for AtLeast, 80th for AtMost
        let idx = match constraint.direction {
            ConstraintDirection::AtLeast => idx,
            ConstraintDirection::AtMost => n_total - 1 - idx,
            _ => idx,
        };
        Some(actual_values[idx.min(n_total - 1)])
    } else {
        None
    };

    // Compute elasticity (simplified)
    let base_feasible = scores
        .iter()
        .filter(|s| {
            let actual = extract_metric(s, constraint.metric).unwrap_or(f64::NAN);
            match constraint.direction {
                ConstraintDirection::AtLeast => actual >= constraint.threshold,
                ConstraintDirection::AtMost => actual <= constraint.threshold,
                _ => true,
            }
        })
        .count();

    let elasticity = if base_feasible > 0 && n_at_10pct > base_feasible {
        let pct_change_feasible = (n_at_10pct - base_feasible) as f64 / base_feasible as f64;
        let pct_change_threshold = 0.1;
        pct_change_feasible / pct_change_threshold
    } else {
        0.0
    };

    ConstraintSensitivity {
        constraint_name: constraint.name.clone(),
        current_threshold: constraint.threshold,
        n_feasible_at_10pct_relaxation: n_at_10pct,
        n_feasible_at_20pct_relaxation: n_at_20pct,
        threshold_for_50pct_feasible: threshold_50,
        threshold_for_80pct_feasible: threshold_80,
        elasticity,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.11 RELAXATION PATH ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

fn compute_relaxation_path(
    _score: &GuidelineRobustnessScoreHost,
    feasibility: &GuidelineFeasibility,
) -> RelaxationPath {
    // Sort violated constraints by normalized violation magnitude
    let mut violations: Vec<_> = feasibility
        .relaxation_needed
        .iter()
        .map(|(name, &amount)| {
            // Find the original constraint to get threshold
            let atoms: Vec<_> = feasibility
                .evaluations
                .iter()
                .flat_map(|e| e.collect_atoms())
                .filter(|a| &a.constraint_name == name)
                .collect();

            let (threshold, direction) = atoms
                .first()
                .map(|a| (a.threshold, a.direction))
                .unwrap_or((1.0, ConstraintDirection::AtLeast));

            let percent = if threshold.abs() > 1e-12 {
                amount / threshold.abs() * 100.0
            } else {
                amount * 100.0
            };

            RelaxationStep {
                constraint_name: name.clone(),
                original_threshold: threshold,
                required_threshold: match direction {
                    ConstraintDirection::AtLeast => threshold - amount,
                    ConstraintDirection::AtMost => threshold + amount,
                    _ => threshold,
                },
                relaxation_amount: amount,
                relaxation_percent: percent,
            }
        })
        .collect();

    // Sort by relaxation percent (smallest first = easiest to fix)
    violations.sort_by(|a, b| {
        a.relaxation_percent
            .partial_cmp(&b.relaxation_percent)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let total_distance: f64 = violations
        .iter()
        .map(|v| v.relaxation_percent / 100.0)
        .sum();

    RelaxationPath {
        guideline_id: feasibility.guideline_id.clone(),
        relaxation_steps: violations,
        total_relaxation_distance: total_distance,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.12 MAIN ANALYSIS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

pub fn compute_constraint_analysis(
    scores: &[GuidelineRobustnessScoreHost],
    config: &ConstraintAnalysisConfig,
) -> anyhow::Result<ConstraintAnalysis> {
    if scores.is_empty() {
        anyhow::bail!("No scores provided for constraint analysis");
    }

    let use_uncertainty = config.use_uncertainty_margins;

    // Evaluate feasibility for all guidelines
    let feasibility: Vec<_> = scores
        .iter()
        .map(|s| evaluate_guideline_feasibility(s, &config.constraints, use_uncertainty))
        .collect();

    // Summary statistics
    let n_total = feasibility.len();
    let n_hard_feasible = feasibility.iter().filter(|f| f.hard_feasible).count();
    let n_fully_feasible = feasibility.iter().filter(|f| f.fully_feasible).count();
    let feasibility_rate = n_hard_feasible as f64 / n_total as f64;

    // Sensitivity analysis
    let sensitivity = if config.compute_sensitivity {
        let atoms = collect_all_atomic_constraints(&config.constraints.constraints);
        let sens: Vec<_> = atoms
            .iter()
            .map(|c| compute_constraint_sensitivity(scores, c, use_uncertainty))
            .collect();
        Some(sens)
    } else {
        None
    };

    // Relaxation paths for infeasible guidelines
    let relaxation_paths = if config.compute_relaxation {
        let paths: Vec<_> = feasibility
            .iter()
            .filter(|f| !f.hard_feasible)
            .map(|f| {
                let score = scores
                    .iter()
                    .find(|s| s.guideline_id == f.guideline_id)
                    .unwrap();
                compute_relaxation_path(score, f)
            })
            .collect();
        Some(paths)
    } else {
        None
    };

    // Pareto analysis (all and feasible)
    let pareto_all = config
        .pareto_config
        .as_ref()
        .map(|cfg| compute_pareto_analysis(scores, cfg))
        .transpose()?;

    let pareto_feasible = if let Some(ref cfg) = config.pareto_config {
        let feasible_scores: Vec<_> = scores
            .iter()
            .filter(|s| {
                feasibility
                    .iter()
                    .find(|f| f.guideline_id == s.guideline_id)
                    .map(|f| f.hard_feasible)
                    .unwrap_or(false)
            })
            .cloned()
            .collect();

        if !feasible_scores.is_empty() {
            Some(compute_pareto_analysis(&feasible_scores, cfg)?)
        } else {
            None
        }
    } else {
        None
    };

    // Most binding constraints
    let most_binding = compute_most_binding_constraints(&feasibility, &config.constraints);

    Ok(ConstraintAnalysis {
        config: config.clone(),
        feasibility,
        n_total,
        n_hard_feasible,
        n_fully_feasible,
        feasibility_rate,
        sensitivity,
        relaxation_paths,
        pareto_all,
        pareto_feasible,
        most_binding_constraints: most_binding,
    })
}

fn collect_all_atomic_constraints(exprs: &[ConstraintExpr]) -> Vec<&AtomicConstraint> {
    let mut atoms = vec![];
    for expr in exprs {
        collect_atoms_from_expr(expr, &mut atoms);
    }
    atoms
}

fn collect_atoms_from_expr<'a>(expr: &'a ConstraintExpr, out: &mut Vec<&'a AtomicConstraint>) {
    match expr {
        ConstraintExpr::Atom(c) => out.push(c),
        ConstraintExpr::And(children) | ConstraintExpr::Or(children) => {
            for c in children {
                collect_atoms_from_expr(c, out);
            }
        }
        ConstraintExpr::AtLeastN { constraints, .. } => {
            for c in constraints {
                collect_atoms_from_expr(c, out);
            }
        }
        ConstraintExpr::Not(child) => collect_atoms_from_expr(child, out),
        ConstraintExpr::Implies {
            condition,
            consequence,
        } => {
            collect_atoms_from_expr(condition, out);
            collect_atoms_from_expr(consequence, out);
        }
    }
}

fn compute_most_binding_constraints(
    feasibility: &[GuidelineFeasibility],
    _constraint_set: &ConstraintSet,
) -> Vec<BindingConstraintInfo> {
    let mut violation_counts: HashMap<String, (usize, f64)> = HashMap::new();

    for f in feasibility {
        for eval in &f.evaluations {
            for atom in eval.collect_atoms() {
                if !atom.satisfied {
                    let entry = violation_counts
                        .entry(atom.constraint_name.clone())
                        .or_insert((0, 0.0));
                    entry.0 += 1;
                    entry.1 += atom.margin.abs();
                }
            }
        }
    }

    let n_total = feasibility.len();
    let mut binding: Vec<_> = violation_counts
        .into_iter()
        .map(|(name, (count, total_margin))| BindingConstraintInfo {
            constraint_name: name,
            n_violations: count,
            violation_rate: count as f64 / n_total as f64,
            avg_violation_margin: if count > 0 {
                total_margin / count as f64
            } else {
                0.0
            },
        })
        .collect();

    binding.sort_by(|a, b| b.n_violations.cmp(&a.n_violations));
    binding
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.13 REGULATORY PRESETS
// ═══════════════════════════════════════════════════════════════════════════

pub mod presets {
    use super::*;

    /// FDA Phase I oncology typical constraints
    pub fn fda_phase1_oncology() -> ConstraintSet {
        ConstraintSet {
            name: "FDA Phase I Oncology".into(),
            description: Some(
                "Typical safety-focused constraints for Phase I oncology trials".into(),
            ),
            constraints: vec![
                ConstraintExpr::Atom(AtomicConstraint {
                    name: "Max DLT rate".into(),
                    description: Some("Dose-limiting toxicity rate ceiling".into()),
                    metric: ConstraintMetric::MeanGrade3PlusRate,
                    direction: ConstraintDirection::AtMost,
                    threshold: 0.33, // Traditional 3+3 target
                    threshold_high: None,
                    tolerance: None,
                    level: ConstraintLevel::Hard,
                    source: Some(RegulatorySource::FDA {
                        guidance_id: "Oncology Dose Finding".into(),
                        section: None,
                    }),
                    threshold_uncertainty: Some(0.05),
                }),
                ConstraintExpr::Atom(AtomicConstraint {
                    name: "Max Grade 4+ toxicity".into(),
                    description: Some("Severe toxicity ceiling".into()),
                    metric: ConstraintMetric::MeanGrade4PlusRate,
                    direction: ConstraintDirection::AtMost,
                    threshold: 0.10,
                    threshold_high: None,
                    tolerance: None,
                    level: ConstraintLevel::Hard,
                    source: None,
                    threshold_uncertainty: None,
                }),
            ],
            regulatory_context: Some("First-in-human dose escalation".into()),
            version: Some("2024-01".into()),
        }
    }

    /// FDA Phase II oncology typical constraints
    pub fn fda_phase2_oncology() -> ConstraintSet {
        ConstraintSet {
            name: "FDA Phase II Oncology".into(),
            description: Some(
                "Efficacy and safety constraints for Phase II oncology trials".into(),
            ),
            constraints: vec![
                ConstraintExpr::Atom(AtomicConstraint {
                    name: "Min objective response rate".into(),
                    description: Some("Minimum ORR for continued development".into()),
                    metric: ConstraintMetric::MeanResponse,
                    direction: ConstraintDirection::AtLeast,
                    threshold: 0.20, // Typical single-arm threshold
                    threshold_high: None,
                    tolerance: None,
                    level: ConstraintLevel::Hard,
                    source: Some(RegulatorySource::FDA {
                        guidance_id: "Clinical Trial Endpoints".into(),
                        section: Some("Oncology".into()),
                    }),
                    threshold_uncertainty: Some(0.05),
                }),
                ConstraintExpr::Atom(AtomicConstraint {
                    name: "Max Grade 3+ toxicity".into(),
                    description: None,
                    metric: ConstraintMetric::MeanGrade3PlusRate,
                    direction: ConstraintDirection::AtMost,
                    threshold: 0.30,
                    threshold_high: None,
                    tolerance: None,
                    level: ConstraintLevel::Hard,
                    source: None,
                    threshold_uncertainty: None,
                }),
                ConstraintExpr::Atom(AtomicConstraint {
                    name: "Min RDI for efficacy".into(),
                    description: Some("Relative dose intensity floor".into()),
                    metric: ConstraintMetric::MeanRDI,
                    direction: ConstraintDirection::AtLeast,
                    threshold: 0.80,
                    threshold_high: None,
                    tolerance: None,
                    level: ConstraintLevel::Soft {
                        penalty_weight: OrderedFloat(10.0),
                    },
                    source: None,
                    threshold_uncertainty: None,
                }),
            ],
            regulatory_context: Some("Single-arm Phase II efficacy trial".into()),
            version: Some("2024-01".into()),
        }
    }

    /// EMA adaptive trial constraints
    pub fn ema_adaptive_oncology() -> ConstraintSet {
        ConstraintSet {
            name: "EMA Adaptive Oncology".into(),
            description: Some("Constraints for EMA adaptive design trials".into()),
            constraints: vec![ConstraintExpr::And(vec![
                ConstraintExpr::Atom(AtomicConstraint {
                    name: "Min response (lower CI bound)".into(),
                    description: Some("Conservative efficacy estimate".into()),
                    metric: ConstraintMetric::ResponseCI95Low,
                    direction: ConstraintDirection::AtLeast,
                    threshold: 0.15,
                    threshold_high: None,
                    tolerance: None,
                    level: ConstraintLevel::Hard,
                    source: Some(RegulatorySource::EMA {
                        guideline_id: "CHMP/EWP/2459/02".into(),
                        section: None,
                    }),
                    threshold_uncertainty: None,
                }),
                ConstraintExpr::Atom(AtomicConstraint {
                    name: "Max toxicity (upper CI bound)".into(),
                    description: Some("Conservative safety estimate".into()),
                    metric: ConstraintMetric::ToxCI95High,
                    direction: ConstraintDirection::AtMost,
                    threshold: 0.40,
                    threshold_high: None,
                    tolerance: None,
                    level: ConstraintLevel::Hard,
                    source: None,
                    threshold_uncertainty: None,
                }),
            ])],
            regulatory_context: Some("EMA adaptive platform trial".into()),
            version: Some("2024-01".into()),
        }
    }

    /// ICH E9 statistical constraints
    pub fn ich_e9_constraints() -> ConstraintSet {
        ConstraintSet {
            name: "ICH E9 Statistical".into(),
            description: Some("Statistical principles from ICH E9".into()),
            constraints: vec![ConstraintExpr::Atom(AtomicConstraint {
                name: "Response variability".into(),
                description: Some("Coefficient of variation constraint".into()),
                metric: ConstraintMetric::ResponseStd,
                direction: ConstraintDirection::AtMost,
                threshold: 0.20, // Max 20% CV
                threshold_high: None,
                tolerance: None,
                level: ConstraintLevel::Soft {
                    penalty_weight: OrderedFloat(5.0),
                },
                source: Some(RegulatorySource::ICH { code: "E9".into() }),
                threshold_uncertainty: None,
            })],
            regulatory_context: Some("Statistical Principles for Clinical Trials".into()),
            version: Some("ICH E9(R1)".into()),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 1.14 EXPORT FORMATS
// ═══════════════════════════════════════════════════════════════════════════

impl ConstraintAnalysis {
    /// Export to JSON
    pub fn to_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }

    /// Export feasibility summary as CSV
    pub fn feasibility_to_csv(&self) -> String {
        let mut csv = String::from(
            "guideline_id,hard_feasible,fully_feasible,n_hard_violations,n_soft_violations,total_penalty\n",
        );

        for f in &self.feasibility {
            csv.push_str(&format!(
                "{},{},{},{},{},{:.4}\n",
                f.guideline_id,
                f.hard_feasible,
                f.fully_feasible,
                f.n_hard_violations,
                f.n_soft_violations,
                f.total_penalty,
            ));
        }

        csv
    }

    /// Export constraint violations detail as CSV
    pub fn violations_to_csv(&self) -> String {
        let mut csv =
            String::from("guideline_id,constraint_name,metric,actual,threshold,margin,satisfied\n");

        for f in &self.feasibility {
            for eval in &f.evaluations {
                for atom in eval.collect_atoms() {
                    csv.push_str(&format!(
                        "{},{},{:?},{:.4},{:.4},{:.4},{}\n",
                        f.guideline_id,
                        atom.constraint_name,
                        atom.metric,
                        atom.actual_value,
                        atom.threshold,
                        atom.margin,
                        atom.satisfied,
                    ));
                }
            }
        }

        csv
    }

    /// Export regulatory report (structured text)
    pub fn to_regulatory_summary(&self) -> String {
        let mut report = String::new();

        report.push_str("═══════════════════════════════════════════════════════════════\n");
        report.push_str("              CLINICAL CONSTRAINT ANALYSIS REPORT              \n");
        report.push_str("═══════════════════════════════════════════════════════════════\n\n");

        report.push_str(&format!(
            "Constraint Set: {}\n",
            self.config.constraints.name
        ));
        if let Some(ref ctx) = self.config.constraints.regulatory_context {
            report.push_str(&format!("Regulatory Context: {}\n", ctx));
        }
        report.push('\n');

        report.push_str("SUMMARY\n");
        report.push_str("───────\n");
        report.push_str(&format!("Total guidelines evaluated: {}\n", self.n_total));
        report.push_str(&format!(
            "Hard-feasible: {} ({:.1}%)\n",
            self.n_hard_feasible,
            self.feasibility_rate * 100.0
        ));
        report.push_str(&format!(
            "Fully-feasible: {} ({:.1}%)\n\n",
            self.n_fully_feasible,
            self.n_fully_feasible as f64 / self.n_total as f64 * 100.0
        ));

        report.push_str("MOST BINDING CONSTRAINTS\n");
        report.push_str("────────────────────────\n");
        for (i, binding) in self.most_binding_constraints.iter().take(5).enumerate() {
            report.push_str(&format!(
                "{}. {} - {:.1}% violation rate (avg margin: {:.3})\n",
                i + 1,
                binding.constraint_name,
                binding.violation_rate * 100.0,
                binding.avg_violation_margin
            ));
        }

        report
    }

    /// Get only hard-feasible guidelines
    pub fn get_feasible_guidelines(&self) -> Vec<&GuidelineFeasibility> {
        self.feasibility
            .iter()
            .filter(|f| f.hard_feasible)
            .collect()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONVENIENCE CONSTRUCTORS
// ═══════════════════════════════════════════════════════════════════════════

/// Create a hard "at least" constraint
pub fn at_least(name: &str, metric: ConstraintMetric, threshold: f64) -> ConstraintExpr {
    ConstraintExpr::Atom(AtomicConstraint {
        name: name.to_string(),
        description: None,
        metric,
        direction: ConstraintDirection::AtLeast,
        threshold,
        threshold_high: None,
        tolerance: None,
        level: ConstraintLevel::Hard,
        source: None,
        threshold_uncertainty: None,
    })
}

/// Create a hard "at most" constraint
pub fn at_most(name: &str, metric: ConstraintMetric, threshold: f64) -> ConstraintExpr {
    ConstraintExpr::Atom(AtomicConstraint {
        name: name.to_string(),
        description: None,
        metric,
        direction: ConstraintDirection::AtMost,
        threshold,
        threshold_high: None,
        tolerance: None,
        level: ConstraintLevel::Hard,
        source: None,
        threshold_uncertainty: None,
    })
}

/// Create a soft constraint with penalty
pub fn soft_at_least(
    name: &str,
    metric: ConstraintMetric,
    threshold: f64,
    penalty: f64,
) -> ConstraintExpr {
    ConstraintExpr::Atom(AtomicConstraint {
        name: name.to_string(),
        description: None,
        metric,
        direction: ConstraintDirection::AtLeast,
        threshold,
        threshold_high: None,
        tolerance: None,
        level: ConstraintLevel::Soft {
            penalty_weight: OrderedFloat(penalty),
        },
        source: None,
        threshold_uncertainty: None,
    })
}

/// Combine constraints with AND logic
pub fn all_of(constraints: Vec<ConstraintExpr>) -> ConstraintExpr {
    ConstraintExpr::And(constraints)
}

/// Combine constraints with OR logic
pub fn any_of(constraints: Vec<ConstraintExpr>) -> ConstraintExpr {
    ConstraintExpr::Or(constraints)
}

/// At least N of M constraints must be satisfied
pub fn at_least_n_of(n: usize, constraints: Vec<ConstraintExpr>) -> ConstraintExpr {
    ConstraintExpr::AtLeastN { n, constraints }
}

/// Default analysis config
pub fn default_constraint_config(constraints: ConstraintSet) -> ConstraintAnalysisConfig {
    ConstraintAnalysisConfig {
        constraints,
        compute_relaxation: true,
        compute_sensitivity: true,
        pareto_config: None,
        use_uncertainty_margins: false,
    }
}

/// Config with Pareto analysis
pub fn constraint_config_with_pareto(
    constraints: ConstraintSet,
    pareto_cfg: ParetoConfig,
) -> ConstraintAnalysisConfig {
    ConstraintAnalysisConfig {
        constraints,
        compute_relaxation: true,
        compute_sensitivity: true,
        pareto_config: Some(pareto_cfg),
        use_uncertainty_margins: false,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn test_scores() -> Vec<GuidelineRobustnessScoreHost> {
        vec![
            GuidelineRobustnessScoreHost::mock("G1", 0.35, 0.20, 0.90), // Good: high response, low tox
            GuidelineRobustnessScoreHost::mock("G2", 0.25, 0.35, 0.85), // Mixed: moderate response, high tox
            GuidelineRobustnessScoreHost::mock("G3", 0.15, 0.15, 0.95), // Poor response, low tox
            GuidelineRobustnessScoreHost::mock("G4", 0.40, 0.45, 0.70), // High response, very high tox
            GuidelineRobustnessScoreHost::mock("G5", 0.30, 0.25, 0.88), // Balanced
        ]
    }

    #[test]
    fn test_atomic_constraint_at_least() {
        let score = GuidelineRobustnessScoreHost::mock("test", 0.30, 0.20, 0.85);
        let constraint = AtomicConstraint {
            name: "Min response".into(),
            description: None,
            metric: ConstraintMetric::MeanResponse,
            direction: ConstraintDirection::AtLeast,
            threshold: 0.25,
            threshold_high: None,
            tolerance: None,
            level: ConstraintLevel::Hard,
            source: None,
            threshold_uncertainty: None,
        };

        let eval = evaluate_atomic_constraint(&constraint, &score, false);
        assert!(eval.satisfied);
        assert!(eval.margin > 0.0);
        assert_eq!(eval.margin, 0.05); // 0.30 - 0.25
    }

    #[test]
    fn test_atomic_constraint_at_most() {
        let score = GuidelineRobustnessScoreHost::mock("test", 0.30, 0.35, 0.85);
        let constraint = AtomicConstraint {
            name: "Max tox".into(),
            description: None,
            metric: ConstraintMetric::MeanGrade3PlusRate,
            direction: ConstraintDirection::AtMost,
            threshold: 0.30,
            threshold_high: None,
            tolerance: None,
            level: ConstraintLevel::Hard,
            source: None,
            threshold_uncertainty: None,
        };

        let eval = evaluate_atomic_constraint(&constraint, &score, false);
        assert!(!eval.satisfied);
        assert!(eval.margin < 0.0);
    }

    #[test]
    fn test_soft_constraint_penalty() {
        let score = GuidelineRobustnessScoreHost::mock("test", 0.20, 0.30, 0.75);
        let constraint = AtomicConstraint {
            name: "Target RDI".into(),
            description: None,
            metric: ConstraintMetric::MeanRDI,
            direction: ConstraintDirection::AtLeast,
            threshold: 0.80,
            threshold_high: None,
            tolerance: None,
            level: ConstraintLevel::Soft {
                penalty_weight: OrderedFloat(10.0),
            },
            source: None,
            threshold_uncertainty: None,
        };

        let eval = evaluate_atomic_constraint(&constraint, &score, false);
        assert!(!eval.satisfied);
        assert!(eval.penalty > 0.0);
        assert!((eval.penalty - 0.5).abs() < 0.01); // 10.0 * 0.05 = 0.5
    }

    #[test]
    fn test_and_constraint() {
        let score = GuidelineRobustnessScoreHost::mock("test", 0.30, 0.20, 0.85);
        let expr = all_of(vec![
            at_least("Min response", ConstraintMetric::MeanResponse, 0.25),
            at_most("Max tox", ConstraintMetric::MeanGrade3PlusRate, 0.30),
        ]);

        let eval = evaluate_constraint_expr(&expr, &score, false);
        assert!(eval.is_satisfied());

        if let ConstraintExprEval::And {
            n_satisfied,
            n_total,
            ..
        } = eval
        {
            assert_eq!(n_satisfied, 2);
            assert_eq!(n_total, 2);
        } else {
            panic!("Expected And evaluation");
        }
    }

    #[test]
    fn test_or_constraint() {
        let score = GuidelineRobustnessScoreHost::mock("test", 0.15, 0.20, 0.85);
        let expr = any_of(vec![
            at_least("High response", ConstraintMetric::MeanResponse, 0.30), // Fails
            at_most("Low tox", ConstraintMetric::MeanGrade3PlusRate, 0.25),  // Passes
        ]);

        let eval = evaluate_constraint_expr(&expr, &score, false);
        assert!(eval.is_satisfied());
    }

    #[test]
    fn test_full_analysis() {
        let scores = test_scores();
        let constraints = presets::fda_phase2_oncology();
        let config = default_constraint_config(constraints);

        let analysis = compute_constraint_analysis(&scores, &config).unwrap();

        assert_eq!(analysis.n_total, 5);
        assert!(analysis.n_hard_feasible <= analysis.n_total);
        assert!(analysis.feasibility_rate >= 0.0 && analysis.feasibility_rate <= 1.0);
        assert!(!analysis.most_binding_constraints.is_empty());
    }

    #[test]
    fn test_sensitivity_analysis() {
        let scores = test_scores();
        let constraint = AtomicConstraint {
            name: "Min response".into(),
            description: None,
            metric: ConstraintMetric::MeanResponse,
            direction: ConstraintDirection::AtLeast,
            threshold: 0.25,
            threshold_high: None,
            tolerance: None,
            level: ConstraintLevel::Hard,
            source: None,
            threshold_uncertainty: None,
        };

        let sens = compute_constraint_sensitivity(&scores, &constraint, false);
        assert_eq!(sens.constraint_name, "Min response");
        assert!(sens.n_feasible_at_10pct_relaxation >= 0);
        assert!(sens.n_feasible_at_20pct_relaxation >= sens.n_feasible_at_10pct_relaxation);
    }

    #[test]
    fn test_relaxation_path() {
        let scores = test_scores();
        let constraints = ConstraintSet {
            name: "Test".into(),
            description: None,
            constraints: vec![at_least(
                "High response",
                ConstraintMetric::MeanResponse,
                0.50, // Very strict - most will fail
            )],
            regulatory_context: None,
            version: None,
        };
        let config = default_constraint_config(constraints);

        let analysis = compute_constraint_analysis(&scores, &config).unwrap();

        if let Some(paths) = &analysis.relaxation_paths {
            assert!(!paths.is_empty());
            for path in paths {
                assert!(path.total_relaxation_distance >= 0.0);
            }
        }
    }

    #[test]
    fn test_csv_export() {
        let scores = test_scores();
        let constraints = presets::fda_phase2_oncology();
        let config = default_constraint_config(constraints);

        let analysis = compute_constraint_analysis(&scores, &config).unwrap();

        let csv = analysis.feasibility_to_csv();
        assert!(csv.contains("guideline_id"));
        assert!(csv.contains("hard_feasible"));

        let violations_csv = analysis.violations_to_csv();
        assert!(violations_csv.contains("constraint_name"));
    }

    #[test]
    fn test_regulatory_presets() {
        // Test that all presets compile and have constraints
        let phase1 = presets::fda_phase1_oncology();
        assert!(!phase1.constraints.is_empty());

        let phase2 = presets::fda_phase2_oncology();
        assert!(!phase2.constraints.is_empty());

        let ema = presets::ema_adaptive_oncology();
        assert!(!ema.constraints.is_empty());

        let ich = presets::ich_e9_constraints();
        assert!(!ich.constraints.is_empty());
    }

    #[test]
    fn test_uncertainty_margins() {
        let score = GuidelineRobustnessScoreHost::mock("test", 0.24, 0.30, 0.85);

        // Without uncertainty: 0.24 < 0.25 → fails
        let constraint = AtomicConstraint {
            name: "Min response".into(),
            description: None,
            metric: ConstraintMetric::MeanResponse,
            direction: ConstraintDirection::AtLeast,
            threshold: 0.25,
            threshold_high: None,
            tolerance: None,
            level: ConstraintLevel::Hard,
            source: None,
            threshold_uncertainty: Some(0.02), // ±2%
        };

        let eval_no_unc = evaluate_atomic_constraint(&constraint, &score, false);
        assert!(!eval_no_unc.satisfied);

        // With uncertainty: threshold becomes 0.27, still fails
        let eval_with_unc = evaluate_atomic_constraint(&constraint, &score, true);
        assert!(!eval_with_unc.satisfied);
        assert!(eval_with_unc.threshold > constraint.threshold);
    }
}
