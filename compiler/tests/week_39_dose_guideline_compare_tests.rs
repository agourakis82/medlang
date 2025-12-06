// Week 39: Dose Guideline Comparison Tests
//
// Tests the comparison functionality added in Week 39:
// 1. DiffDirection enum
// 2. Default grid generation from dose levels
// 3. CSV export of full grid
// 4. Enhanced diff reporting

use medlangc::rl::{
    compare_dose_guidelines_on_grid, default_grid_for_guidelines, diff_points_to_csv,
    AtomicConditionIR, ComparisonOpIR, DiffDirection, DoseGuidelineGridConfig, DoseGuidelineIRHost,
    DoseRuleIR,
};

// =============================================================================
// Test 1: DiffDirection Classification
// =============================================================================

#[test]
fn test_diff_direction_classification() {
    // Create two guidelines with known differences
    let mut rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL guideline".to_string(),
        vec!["ANC".to_string()],
        vec![0.0, 50.0, 100.0],
    );

    // RL: if ANC < 0.5 hold, else 50 mg
    rl.add_rule(DoseRuleIR::new(
        vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::LT,
            0.5,
        )],
        0,
        0.0,
    ));
    rl.add_rule(DoseRuleIR::new(vec![], 1, 50.0));

    let mut baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline guideline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );

    // Baseline: always 100 mg
    baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0));

    // Use simple grid
    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.3, 0.7],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let report = medlangc::rl::compare_dose_guidelines_detailed(&rl, &baseline, &grid);

    // At ANC=0.3: RL=0, baseline=100 → RlMoreConservative
    // At ANC=0.7: RL=50, baseline=100 → RlMoreConservative
    assert_eq!(report.diff_points.len(), 2);

    for dp in &report.diff_points {
        assert_eq!(dp.direction, DiffDirection::RlMoreConservative);
    }
}

#[test]
fn test_diff_direction_aggressive() {
    // RL gives higher doses than baseline
    let mut rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![200.0],
    );
    rl.add_rule(DoseRuleIR::new(vec![], 0, 200.0));

    let mut baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );
    baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0));

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.5],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let report = medlangc::rl::compare_dose_guidelines_detailed(&rl, &baseline, &grid);

    assert_eq!(report.diff_points.len(), 1);
    assert_eq!(
        report.diff_points[0].direction,
        DiffDirection::RlMoreAggressive
    );
    assert_eq!(report.diff_points[0].difference_mg, 100.0);
}

#[test]
fn test_diff_direction_same() {
    // Identical guidelines
    let mut gl = DoseGuidelineIRHost::new(
        "Test".to_string(),
        "Test".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );
    gl.add_rule(DoseRuleIR::new(vec![], 0, 100.0));

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.5, 1.0],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let report = medlangc::rl::compare_dose_guidelines_detailed(&gl, &gl, &grid);

    // No differences, so diff_points should be empty
    assert_eq!(report.diff_points.len(), 0);
    assert_eq!(report.summary.disagree_points, 0);
    assert_eq!(report.summary.disagree_fraction, 0.0);
}

// =============================================================================
// Test 2: Default Grid Generation
// =============================================================================

#[test]
fn test_default_grid_generation() {
    let rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![0.0, 50.0, 100.0, 200.0],
    );

    let baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![75.0, 150.0, 300.0],
    );

    let grid = default_grid_for_guidelines(&rl, &baseline);

    // Check that ANC and tumour_ratio grids are present
    assert!(!grid.anc_grid.is_empty());
    assert!(!grid.tumour_ratio_grid.is_empty());
    assert!(!grid.cycle_grid.is_empty());

    // Check that prev_dose_grid is the union of both dose levels
    // Union: [0.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0]
    assert_eq!(grid.prev_dose_grid.len(), 7);
    assert!(grid.prev_dose_grid.contains(&0.0));
    assert!(grid.prev_dose_grid.contains(&50.0));
    assert!(grid.prev_dose_grid.contains(&75.0));
    assert!(grid.prev_dose_grid.contains(&100.0));
    assert!(grid.prev_dose_grid.contains(&150.0));
    assert!(grid.prev_dose_grid.contains(&200.0));
    assert!(grid.prev_dose_grid.contains(&300.0));

    // Should be sorted and deduplicated
    for i in 1..grid.prev_dose_grid.len() {
        assert!(grid.prev_dose_grid[i] > grid.prev_dose_grid[i - 1]);
    }
}

#[test]
fn test_default_grid_empty_doses() {
    let rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![],
    );

    let baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![],
    );

    let grid = default_grid_for_guidelines(&rl, &baseline);

    // Should provide default dose grid even if both are empty
    assert!(!grid.prev_dose_grid.is_empty());
}

#[test]
fn test_default_grid_deduplication() {
    // Both guidelines have overlapping dose levels
    let rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![0.0, 100.0, 200.0],
    );

    let baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0, 200.0, 300.0],
    );

    let grid = default_grid_for_guidelines(&rl, &baseline);

    // Union: [0.0, 100.0, 200.0, 300.0] (no duplicates)
    assert_eq!(grid.prev_dose_grid.len(), 4);
    assert!(grid.prev_dose_grid.contains(&0.0));
    assert!(grid.prev_dose_grid.contains(&100.0));
    assert!(grid.prev_dose_grid.contains(&200.0));
    assert!(grid.prev_dose_grid.contains(&300.0));
}

// =============================================================================
// Test 3: CSV Export
// =============================================================================

#[test]
fn test_csv_export_format() {
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
        vec![100.0],
    );
    baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0));

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.3, 0.7],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let csv = diff_points_to_csv(&rl, &baseline, &grid);

    // Check header
    assert!(
        csv.starts_with("anc,tumour_ratio,prev_dose,cycle,rl_dose,baseline_dose,delta,direction\n")
    );

    // Count lines (header + 2 data rows)
    let line_count = csv.lines().count();
    assert_eq!(line_count, 3); // 1 header + 2 data

    // Check that data rows contain expected values
    let lines: Vec<&str> = csv.lines().collect();
    assert!(lines[1].contains("0.300000")); // ANC = 0.3
    assert!(lines[2].contains("0.700000")); // ANC = 0.7

    // Check direction labels
    assert!(lines[1].contains("RlMoreConservative"));
    assert!(lines[2].contains("RlMoreConservative"));
}

#[test]
fn test_csv_export_grid_size() {
    let rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );

    let baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.5, 1.0, 1.5],
        tumour_ratio_grid: vec![0.8, 1.0],
        prev_dose_grid: vec![100.0, 200.0],
        cycle_grid: vec![1.0, 2.0],
    };

    let csv = diff_points_to_csv(&rl, &baseline, &grid);

    // Expected lines: 1 header + (3 * 2 * 2 * 2) = 1 + 24 = 25
    let line_count = csv.lines().count();
    assert_eq!(line_count, 25);
}

#[test]
fn test_csv_export_same_direction() {
    // Identical guidelines should show "Same" direction
    let mut gl = DoseGuidelineIRHost::new(
        "Test".to_string(),
        "Test".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );
    gl.add_rule(DoseRuleIR::new(vec![], 0, 100.0));

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.5],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let csv = diff_points_to_csv(&gl, &gl, &grid);

    let lines: Vec<&str> = csv.lines().collect();
    assert!(lines[1].contains("Same"));
    assert!(lines[1].contains("0.000000")); // delta = 0
}

// =============================================================================
// Test 4: Integration - Grid Config JSON Serialization
// =============================================================================

#[test]
fn test_grid_config_serialization() {
    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.1, 0.5, 1.0, 2.0],
        tumour_ratio_grid: vec![0.5, 1.0, 1.5],
        prev_dose_grid: vec![0.0, 50.0, 100.0],
        cycle_grid: vec![1.0, 2.0, 3.0],
    };

    // Serialize to JSON
    let json = serde_json::to_string_pretty(&grid).expect("Failed to serialize");
    assert!(json.contains("anc_grid"));
    assert!(json.contains("tumour_ratio_grid"));
    assert!(json.contains("prev_dose_grid"));
    assert!(json.contains("cycle_grid"));

    // Deserialize back
    let deserialized: DoseGuidelineGridConfig =
        serde_json::from_str(&json).expect("Failed to deserialize");

    assert_eq!(deserialized.anc_grid, grid.anc_grid);
    assert_eq!(deserialized.tumour_ratio_grid, grid.tumour_ratio_grid);
    assert_eq!(deserialized.prev_dose_grid, grid.prev_dose_grid);
    assert_eq!(deserialized.cycle_grid, grid.cycle_grid);
}

// =============================================================================
// Test 5: Summary Metrics
// =============================================================================

#[test]
fn test_summary_metrics_mean_abs_diff() {
    let mut rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![50.0],
    );
    rl.add_rule(DoseRuleIR::new(vec![], 0, 50.0)); // Always 50 mg

    let mut baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );
    baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0)); // Always 100 mg

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.5, 1.0],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let summary = compare_dose_guidelines_on_grid(&rl, &baseline, &grid);

    // All points differ by 50 mg
    assert_eq!(summary.total_points, 2);
    assert_eq!(summary.disagree_points, 2);
    assert_eq!(summary.mean_dose_difference_mg, 50.0);
    assert_eq!(summary.max_dose_difference_mg, 50.0);
}

#[test]
fn test_summary_metrics_max_diff() {
    let mut rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![0.0, 200.0],
    );

    // RL: if ANC < 0.5 hold (0 mg), else 200 mg
    rl.add_rule(DoseRuleIR::new(
        vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::LT,
            0.5,
        )],
        0,
        0.0,
    ));
    rl.add_rule(DoseRuleIR::new(vec![], 1, 200.0));

    let mut baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );
    baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0)); // Always 100 mg

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.3, 0.7],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let summary = compare_dose_guidelines_on_grid(&rl, &baseline, &grid);

    // At ANC=0.3: diff = |0 - 100| = 100
    // At ANC=0.7: diff = |200 - 100| = 100
    assert_eq!(summary.mean_dose_difference_mg, 100.0);
    assert_eq!(summary.max_dose_difference_mg, 100.0);
}

// =============================================================================
// Test 6: Edge Cases
// =============================================================================

#[test]
fn test_single_point_grid() {
    let mut rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![50.0],
    );
    rl.add_rule(DoseRuleIR::new(vec![], 0, 50.0));

    let mut baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );
    baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0));

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![1.0],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let summary = compare_dose_guidelines_on_grid(&rl, &baseline, &grid);

    assert_eq!(summary.total_points, 1);
    assert_eq!(summary.disagree_points, 1);
    assert_eq!(summary.disagree_fraction, 1.0);
}

#[test]
fn test_large_grid() {
    let rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );

    let baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );

    let grid = DoseGuidelineGridConfig::fine();

    let summary = compare_dose_guidelines_on_grid(&rl, &baseline, &grid);

    // Fine grid: 8 * 9 * 7 * 6 = 3024 points
    assert_eq!(summary.total_points, grid.total_points());
    assert!(summary.total_points > 1000);
}

#[test]
fn test_mixed_differences() {
    let mut rl = DoseGuidelineIRHost::new(
        "RL".to_string(),
        "RL".to_string(),
        vec!["ANC".to_string()],
        vec![0.0, 50.0, 150.0],
    );

    // RL:
    // if ANC <= 0.4 → 0 mg (conservative)
    // if 0.4 < ANC <= 0.8 → 50 mg (conservative)
    // if ANC > 0.8 → 150 mg (aggressive)
    rl.add_rule(DoseRuleIR::new(
        vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::LE,
            0.4,
        )],
        0,
        0.0,
    ));
    rl.add_rule(DoseRuleIR::new(
        vec![
            AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::GT, 0.4),
            AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::LE, 0.8),
        ],
        1,
        50.0,
    ));
    rl.add_rule(DoseRuleIR::new(vec![], 2, 150.0));

    let mut baseline = DoseGuidelineIRHost::new(
        "Baseline".to_string(),
        "Baseline".to_string(),
        vec!["ANC".to_string()],
        vec![100.0],
    );
    baseline.add_rule(DoseRuleIR::new(vec![], 0, 100.0)); // Always 100 mg

    let grid = DoseGuidelineGridConfig {
        anc_grid: vec![0.3, 0.6, 1.0],
        tumour_ratio_grid: vec![1.0],
        prev_dose_grid: vec![100.0],
        cycle_grid: vec![1.0],
    };

    let summary = compare_dose_guidelines_on_grid(&rl, &baseline, &grid);

    assert_eq!(summary.total_points, 3);
    assert_eq!(summary.disagree_points, 3);

    // ANC=0.3: 0 vs 100 → conservative
    // ANC=0.6: 50 vs 100 → conservative
    // ANC=1.0: 150 vs 100 → aggressive
    assert_eq!(summary.rl_more_conservative_fraction, 2.0 / 3.0);
    assert_eq!(summary.rl_more_aggressive_fraction, 1.0 / 3.0);
}
