// Week 38: Dose Guideline Comparison
//
// Compares two dose guidelines (e.g., RL-derived vs standard-of-care) by
// evaluating them over a grid of feature values and computing agreement metrics.

use serde::{Deserialize, Serialize};

use crate::rl::dose_guideline_ir::DoseGuidelineIRHost;

// =============================================================================
// Grid Configuration
// =============================================================================

/// Grid configuration for evaluating guidelines in feature space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseGuidelineGridConfig {
    pub anc_grid: Vec<f64>,
    pub tumour_ratio_grid: Vec<f64>,
    pub prev_dose_grid: Vec<f64>,
    pub cycle_grid: Vec<f64>,
}

impl DoseGuidelineGridConfig {
    /// Create a default coarse grid for quick comparison
    pub fn coarse() -> Self {
        Self {
            anc_grid: vec![0.2, 0.5, 0.8, 1.2],
            tumour_ratio_grid: vec![0.6, 0.8, 1.0, 1.2],
            prev_dose_grid: vec![50.0, 100.0, 200.0],
            cycle_grid: vec![1.0, 2.0, 3.0],
        }
    }

    /// Create a fine grid for detailed comparison
    pub fn fine() -> Self {
        Self {
            anc_grid: vec![0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
            tumour_ratio_grid: vec![0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
            prev_dose_grid: vec![0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0],
            cycle_grid: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    }

    /// Get total number of grid points
    pub fn total_points(&self) -> usize {
        self.anc_grid.len()
            * self.tumour_ratio_grid.len()
            * self.prev_dose_grid.len()
            * self.cycle_grid.len()
    }
}

// =============================================================================
// Difference Summary
// =============================================================================

/// Summary of differences between two dose guidelines
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineDiffSummary {
    /// Total number of grid points evaluated
    pub total_points: usize,

    /// Number of points where guidelines disagree
    pub disagree_points: usize,

    /// Fraction of points where guidelines disagree (0.0 to 1.0)
    pub disagree_fraction: f64,

    /// Fraction where RL recommends higher dose than baseline
    pub rl_more_aggressive_fraction: f64,

    /// Fraction where RL recommends lower dose than baseline
    pub rl_more_conservative_fraction: f64,

    /// Mean absolute dose difference (mg) across all points
    pub mean_dose_difference_mg: f64,

    /// Maximum dose difference (mg) observed
    pub max_dose_difference_mg: f64,
}

impl GuidelineDiffSummary {
    /// Check if guidelines are substantially similar (< 10% disagreement)
    pub fn is_similar(&self) -> bool {
        self.disagree_fraction < 0.1
    }

    /// Check if RL is substantially more aggressive (> 30% higher doses)
    pub fn is_rl_more_aggressive(&self) -> bool {
        self.rl_more_aggressive_fraction > 0.3
    }

    /// Check if RL is substantially more conservative (> 30% lower doses)
    pub fn is_rl_more_conservative(&self) -> bool {
        self.rl_more_conservative_fraction > 0.3
    }
}

// =============================================================================
// Point-wise Difference Record
// =============================================================================

/// Direction of difference between RL and baseline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffDirection {
    /// Doses are identical (within epsilon)
    Same,
    /// RL recommends higher dose than baseline
    RlMoreAggressive,
    /// RL recommends lower dose than baseline
    RlMoreConservative,
}

/// A single point where guidelines differ
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffPoint {
    pub anc: f64,
    pub tumour_ratio: f64,
    pub prev_dose: f64,
    pub cycle: f64,
    pub rl_dose_mg: f64,
    pub baseline_dose_mg: f64,
    pub difference_mg: f64,
    pub direction: DiffDirection,
}

/// Detailed comparison report with point-wise differences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuidelineDiffReport {
    pub summary: GuidelineDiffSummary,
    pub diff_points: Vec<DiffPoint>,
}

// =============================================================================
// Evaluation Logic
// =============================================================================

/// Evaluate a guideline at a single point in feature space
fn eval_guideline(
    gl: &DoseGuidelineIRHost,
    anc: f64,
    tumour_ratio: f64,
    prev_dose: f64,
    cycle: f64,
) -> f64 {
    let features = vec![
        ("ANC".to_string(), anc),
        ("anc".to_string(), anc),
        ("tumour_ratio".to_string(), tumour_ratio),
        ("tumor_ratio".to_string(), tumour_ratio),
        ("prev_dose".to_string(), prev_dose),
        ("previous_dose".to_string(), prev_dose),
        ("cycle".to_string(), cycle),
        ("cycle_index".to_string(), cycle),
    ];

    // Find first matching rule
    gl.evaluate(&features).unwrap_or(0.0)
}

// =============================================================================
// Default Grid Generation
// =============================================================================

/// Build a default grid based on the dose levels from both guidelines
pub fn default_grid_for_guidelines(
    rl: &DoseGuidelineIRHost,
    baseline: &DoseGuidelineIRHost,
) -> DoseGuidelineGridConfig {
    // ANC: rough range, e.g. 0.1 to 4.0 (x10^9/L)
    let anc_grid = vec![0.1, 0.5, 1.0, 2.0, 4.0];

    // Tumour ratio: 0.5 (shrinkage) to 2.0 (growth)
    let tumour_ratio_grid = vec![0.5, 1.0, 1.5, 2.0];

    // Previous dose: union of both guidelines' dose levels, deduplicated
    let mut all_doses = rl.dose_levels_mg.clone();
    all_doses.extend(baseline.dose_levels_mg.iter().copied());
    all_doses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_doses.dedup();
    let prev_dose_grid = if all_doses.is_empty() {
        vec![0.0, 50.0, 100.0, 200.0]
    } else {
        all_doses
    };

    // Cycle index: small set of representative cycles
    let cycle_grid = vec![1.0, 2.0, 4.0, 6.0];

    DoseGuidelineGridConfig {
        anc_grid,
        tumour_ratio_grid,
        prev_dose_grid,
        cycle_grid,
    }
}

// =============================================================================
// Comparison Functions
// =============================================================================

/// Compare two dose guidelines on a grid, returning a summary
pub fn compare_dose_guidelines_on_grid(
    rl: &DoseGuidelineIRHost,
    baseline: &DoseGuidelineIRHost,
    grid: &DoseGuidelineGridConfig,
) -> GuidelineDiffSummary {
    let mut total = 0usize;
    let mut disagree = 0usize;
    let mut rl_more_aggr = 0usize;
    let mut rl_more_cons = 0usize;
    let mut dose_diff_sum: f64 = 0.0;
    let mut max_diff: f64 = 0.0;

    for anc in &grid.anc_grid {
        for tr in &grid.tumour_ratio_grid {
            for prev in &grid.prev_dose_grid {
                for cyc in &grid.cycle_grid {
                    let rl_d = eval_guideline(rl, *anc, *tr, *prev, *cyc);
                    let base_d = eval_guideline(baseline, *anc, *tr, *prev, *cyc);

                    let diff = (rl_d - base_d).abs();
                    dose_diff_sum += diff;
                    max_diff = max_diff.max(diff);

                    total += 1;
                    if diff > f64::EPSILON {
                        disagree += 1;
                        if rl_d > base_d {
                            rl_more_aggr += 1;
                        } else if rl_d < base_d {
                            rl_more_cons += 1;
                        }
                    }
                }
            }
        }
    }

    let total_f = total as f64;
    let disagree_f = disagree as f64;
    let rl_more_aggr_f = rl_more_aggr as f64;
    let rl_more_cons_f = rl_more_cons as f64;

    GuidelineDiffSummary {
        total_points: total,
        disagree_points: disagree,
        disagree_fraction: if total > 0 { disagree_f / total_f } else { 0.0 },
        rl_more_aggressive_fraction: if total > 0 {
            rl_more_aggr_f / total_f
        } else {
            0.0
        },
        rl_more_conservative_fraction: if total > 0 {
            rl_more_cons_f / total_f
        } else {
            0.0
        },
        mean_dose_difference_mg: if total > 0 {
            dose_diff_sum / total_f
        } else {
            0.0
        },
        max_dose_difference_mg: max_diff,
    }
}

/// Compare two dose guidelines on a grid, returning detailed point-wise differences
pub fn compare_dose_guidelines_detailed(
    rl: &DoseGuidelineIRHost,
    baseline: &DoseGuidelineIRHost,
    grid: &DoseGuidelineGridConfig,
) -> GuidelineDiffReport {
    let mut total = 0usize;
    let mut disagree = 0usize;
    let mut rl_more_aggr = 0usize;
    let mut rl_more_cons = 0usize;
    let mut dose_diff_sum: f64 = 0.0;
    let mut max_diff: f64 = 0.0;
    let mut diff_points = Vec::new();

    for anc in &grid.anc_grid {
        for tr in &grid.tumour_ratio_grid {
            for prev in &grid.prev_dose_grid {
                for cyc in &grid.cycle_grid {
                    let rl_d = eval_guideline(rl, *anc, *tr, *prev, *cyc);
                    let base_d = eval_guideline(baseline, *anc, *tr, *prev, *cyc);

                    let diff = (rl_d - base_d).abs();
                    dose_diff_sum += diff;
                    max_diff = max_diff.max(diff);

                    total += 1;
                    if diff > f64::EPSILON {
                        disagree += 1;
                        if rl_d > base_d {
                            rl_more_aggr += 1;
                        } else if rl_d < base_d {
                            rl_more_cons += 1;
                        }

                        // Record this difference point
                        let direction = if rl_d > base_d {
                            DiffDirection::RlMoreAggressive
                        } else {
                            DiffDirection::RlMoreConservative
                        };

                        diff_points.push(DiffPoint {
                            anc: *anc,
                            tumour_ratio: *tr,
                            prev_dose: *prev,
                            cycle: *cyc,
                            rl_dose_mg: rl_d,
                            baseline_dose_mg: base_d,
                            difference_mg: rl_d - base_d,
                            direction,
                        });
                    }
                }
            }
        }
    }

    let total_f = total as f64;
    let disagree_f = disagree as f64;
    let rl_more_aggr_f = rl_more_aggr as f64;
    let rl_more_cons_f = rl_more_cons as f64;

    let summary = GuidelineDiffSummary {
        total_points: total,
        disagree_points: disagree,
        disagree_fraction: if total > 0 { disagree_f / total_f } else { 0.0 },
        rl_more_aggressive_fraction: if total > 0 {
            rl_more_aggr_f / total_f
        } else {
            0.0
        },
        rl_more_conservative_fraction: if total > 0 {
            rl_more_cons_f / total_f
        } else {
            0.0
        },
        mean_dose_difference_mg: if total > 0 {
            dose_diff_sum / total_f
        } else {
            0.0
        },
        max_dose_difference_mg: max_diff,
    };

    GuidelineDiffReport {
        summary,
        diff_points,
    }
}

// =============================================================================
// CSV Export
// =============================================================================

/// Export all grid points to CSV format for plotting
pub fn diff_points_to_csv(
    rl: &DoseGuidelineIRHost,
    baseline: &DoseGuidelineIRHost,
    grid: &DoseGuidelineGridConfig,
) -> String {
    let mut out = String::new();
    out.push_str("anc,tumour_ratio,prev_dose,cycle,rl_dose,baseline_dose,delta,direction\n");

    for anc in &grid.anc_grid {
        for tr in &grid.tumour_ratio_grid {
            for prev in &grid.prev_dose_grid {
                for cyc in &grid.cycle_grid {
                    let rl_dose = eval_guideline(rl, *anc, *tr, *prev, *cyc);
                    let base_dose = eval_guideline(baseline, *anc, *tr, *prev, *cyc);
                    let delta = rl_dose - base_dose;

                    let direction_str = if delta.abs() < f64::EPSILON {
                        "Same"
                    } else if delta > 0.0 {
                        "RlMoreAggressive"
                    } else {
                        "RlMoreConservative"
                    };

                    out.push_str(&format!(
                        "{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{}\n",
                        anc, tr, prev, cyc, rl_dose, base_dose, delta, direction_str
                    ));
                }
            }
        }
    }

    out
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::dose_guideline_ir::{AtomicConditionIR, ComparisonOpIR, DoseRuleIR};

    #[test]
    fn test_grid_config_coarse() {
        let grid = DoseGuidelineGridConfig::coarse();
        assert_eq!(grid.anc_grid.len(), 4);
        assert_eq!(grid.total_points(), 4 * 4 * 3 * 3); // 144 points
    }

    #[test]
    fn test_grid_config_fine() {
        let grid = DoseGuidelineGridConfig::fine();
        assert_eq!(grid.anc_grid.len(), 8);
        assert!(grid.total_points() > 1000);
    }

    #[test]
    fn test_eval_guideline() {
        let mut gl = DoseGuidelineIRHost::new(
            "Test".to_string(),
            "Test".to_string(),
            vec!["ANC".to_string()],
            vec![0.0, 100.0],
        );

        // If ANC < 0.5 then dose 0, else dose 100
        gl.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::LT,
                0.5,
            )],
            0,
            0.0,
        ));

        gl.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::GE,
                0.5,
            )],
            1,
            100.0,
        ));

        let dose_low = eval_guideline(&gl, 0.3, 1.0, 100.0, 1.0);
        assert_eq!(dose_low, 0.0);

        let dose_high = eval_guideline(&gl, 0.7, 1.0, 100.0, 1.0);
        assert_eq!(dose_high, 100.0);
    }

    #[test]
    fn test_compare_identical_guidelines() {
        let mut gl = DoseGuidelineIRHost::new(
            "Test".to_string(),
            "Test".to_string(),
            vec!["ANC".to_string()],
            vec![0.0, 100.0],
        );

        gl.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::LT,
                0.5,
            )],
            0,
            0.0,
        ));

        gl.add_rule(DoseRuleIR::new(vec![], 1, 100.0)); // Default rule

        let grid = DoseGuidelineGridConfig::coarse();
        let summary = compare_dose_guidelines_on_grid(&gl, &gl, &grid);

        assert_eq!(summary.total_points, grid.total_points());
        assert_eq!(summary.disagree_points, 0);
        assert_eq!(summary.disagree_fraction, 0.0);
        assert!(summary.is_similar());
    }

    #[test]
    fn test_compare_different_guidelines() {
        // RL guideline: conservative (always hold if ANC < 1.0)
        let mut rl = DoseGuidelineIRHost::new(
            "RL".to_string(),
            "RL".to_string(),
            vec!["ANC".to_string()],
            vec![0.0, 100.0],
        );

        rl.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::LT,
                1.0,
            )],
            0,
            0.0,
        ));

        rl.add_rule(DoseRuleIR::new(vec![], 1, 100.0));

        // Baseline: aggressive (always give 100 mg)
        let mut baseline = DoseGuidelineIRHost::new(
            "Baseline".to_string(),
            "Baseline".to_string(),
            vec!["ANC".to_string()],
            vec![100.0],
        );

        baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0));

        let grid = DoseGuidelineGridConfig::coarse();
        let summary = compare_dose_guidelines_on_grid(&rl, &baseline, &grid);

        assert!(summary.disagree_points > 0);
        assert!(summary.disagree_fraction > 0.0);
        assert!(summary.rl_more_conservative_fraction > 0.0);
        assert!(!summary.is_similar());
    }

    #[test]
    fn test_compare_detailed() {
        let mut rl = DoseGuidelineIRHost::new(
            "RL".to_string(),
            "RL".to_string(),
            vec!["ANC".to_string()],
            vec![0.0, 100.0],
        );

        rl.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::LT,
                0.5,
            )],
            0,
            0.0,
        ));

        rl.add_rule(DoseRuleIR::new(vec![], 1, 100.0));

        let mut baseline = DoseGuidelineIRHost::new(
            "Baseline".to_string(),
            "Baseline".to_string(),
            vec!["ANC".to_string()],
            vec![50.0, 100.0],
        );

        baseline.add_rule(DoseRuleIR::new(
            vec![AtomicConditionIR::new(
                "ANC".to_string(),
                ComparisonOpIR::LT,
                0.5,
            )],
            0,
            50.0,
        ));

        baseline.add_rule(DoseRuleIR::new(vec![], 1, 100.0));

        let grid = DoseGuidelineGridConfig::coarse();
        let report = compare_dose_guidelines_detailed(&rl, &baseline, &grid);

        assert_eq!(report.summary.total_points, grid.total_points());
        assert!(report.diff_points.len() > 0);
        assert!(report.summary.rl_more_conservative_fraction > 0.0);

        // Check that diff points are recorded correctly
        for dp in &report.diff_points {
            assert_ne!(dp.rl_dose_mg, dp.baseline_dose_mg);
            assert_eq!(dp.difference_mg, dp.rl_dose_mg - dp.baseline_dose_mg);
        }
    }

    #[test]
    fn test_summary_predicates() {
        let similar = GuidelineDiffSummary {
            total_points: 100,
            disagree_points: 5,
            disagree_fraction: 0.05,
            rl_more_aggressive_fraction: 0.03,
            rl_more_conservative_fraction: 0.02,
            mean_dose_difference_mg: 10.0,
            max_dose_difference_mg: 50.0,
        };

        assert!(similar.is_similar());
        assert!(!similar.is_rl_more_aggressive());
        assert!(!similar.is_rl_more_conservative());

        let aggressive = GuidelineDiffSummary {
            total_points: 100,
            disagree_points: 50,
            disagree_fraction: 0.5,
            rl_more_aggressive_fraction: 0.4,
            rl_more_conservative_fraction: 0.1,
            mean_dose_difference_mg: 50.0,
            max_dose_difference_mg: 100.0,
        };

        assert!(!aggressive.is_similar());
        assert!(aggressive.is_rl_more_aggressive());
        assert!(!aggressive.is_rl_more_conservative());
    }

    #[test]
    fn test_mean_and_max_dose_difference() {
        let mut rl = DoseGuidelineIRHost::new(
            "RL".to_string(),
            "RL".to_string(),
            vec!["ANC".to_string()],
            vec![0.0, 50.0, 100.0],
        );

        rl.add_rule(DoseRuleIR::new(vec![], 2, 100.0)); // Always 100 mg

        let mut baseline = DoseGuidelineIRHost::new(
            "Baseline".to_string(),
            "Baseline".to_string(),
            vec!["ANC".to_string()],
            vec![200.0],
        );

        baseline.add_rule(DoseRuleIR::new(vec![], 0, 200.0)); // Always 200 mg

        let grid = DoseGuidelineGridConfig::coarse();
        let summary = compare_dose_guidelines_on_grid(&rl, &baseline, &grid);

        assert_eq!(summary.mean_dose_difference_mg, 100.0);
        assert_eq!(summary.max_dose_difference_mg, 100.0);
        assert_eq!(summary.disagree_fraction, 1.0); // All points differ
    }
}
