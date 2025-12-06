// Week 38: Dose Guideline IR and Bridge Integration Tests
//
// Tests the complete pipeline:
// 1. DistilledPolicyTree → DoseGuidelineIRHost
// 2. DoseGuidelineIRHost → GuidelineArtifact
// 3. Guideline comparison functionality
// 4. Pretty-printing and serialization

use medlangc::guideline::ir::{
    DoseActionKind, GuidelineAction, GuidelineExpr, GuidelineMeta, GuidelineValueRef,
};
use medlangc::rl::{
    compare_dose_guidelines_on_grid, dose_guideline_to_guideline_artifact,
    guideline_from_distilled_tree, pretty_print_dose_guideline, AtomicConditionIR, ComparisonOpIR,
    DistilledPolicyTree, DoseGuidelineGridConfig, DoseGuidelineIRHost, DoseRuleIR, TreeNode,
};

// =============================================================================
// Test 1: DistilledPolicyTree → DoseGuidelineIRHost
// =============================================================================

#[test]
fn test_distilled_tree_to_dose_guideline_simple() {
    use medlangc::rl::distill::DistillFeature;

    // Create a simple tree: if ANC <= 0.5 then action 0 (hold), else action 1 (100 mg)
    let tree = DistilledPolicyTree {
        n_actions: 2,
        features: vec![DistillFeature {
            name: "ANC".to_string(),
            index: 0,
            min: 0.0,
            max: 2.0,
        }],
        root: TreeNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf { action: 0 }),
            right: Box::new(TreeNode::Leaf { action: 1 }),
        },
    };

    let dose_levels = vec![0.0, 100.0];
    let guideline = guideline_from_distilled_tree(
        &tree,
        &dose_levels,
        "Simple Test Guideline".to_string(),
        "Test guideline with single split".to_string(),
    );

    // Verify structure
    assert_eq!(guideline.name, "Simple Test Guideline");
    assert_eq!(guideline.rules.len(), 2);
    assert_eq!(guideline.feature_names, vec!["ANC"]);
    assert_eq!(guideline.dose_levels_mg, vec![0.0, 100.0]);

    // Verify first rule: ANC <= 0.5 → 0 mg
    assert_eq!(guideline.rules[0].action_dose_mg, 0.0);
    assert_eq!(guideline.rules[0].action_index, 0);
    assert_eq!(guideline.rules[0].conditions.len(), 1);
    assert_eq!(guideline.rules[0].conditions[0].feature, "ANC");
    assert_eq!(guideline.rules[0].conditions[0].op, ComparisonOpIR::LE);
    assert_eq!(guideline.rules[0].conditions[0].threshold, 0.5);

    // Verify second rule: ANC > 0.5 → 100 mg
    assert_eq!(guideline.rules[1].action_dose_mg, 100.0);
    assert_eq!(guideline.rules[1].action_index, 1);
    assert_eq!(guideline.rules[1].conditions.len(), 1);
    assert_eq!(guideline.rules[1].conditions[0].feature, "ANC");
    assert_eq!(guideline.rules[1].conditions[0].op, ComparisonOpIR::GT);
    assert_eq!(guideline.rules[1].conditions[0].threshold, 0.5);
}

#[test]
fn test_distilled_tree_to_dose_guideline_multi_split() {
    use medlangc::rl::distill::DistillFeature;

    // Create a tree with two features:
    // if ANC <= 0.5:
    //   action 0 (hold)
    // else if tumour_ratio <= 0.8:
    //   action 1 (100 mg)
    // else:
    //   action 2 (200 mg)
    let tree = DistilledPolicyTree {
        n_actions: 3,
        features: vec![
            DistillFeature {
                name: "ANC".to_string(),
                index: 0,
                min: 0.0,
                max: 2.0,
            },
            DistillFeature {
                name: "tumour_ratio".to_string(),
                index: 1,
                min: 0.0,
                max: 2.0,
            },
        ],
        root: TreeNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf { action: 0 }),
            right: Box::new(TreeNode::Split {
                feature_index: 1,
                threshold: 0.8,
                left: Box::new(TreeNode::Leaf { action: 1 }),
                right: Box::new(TreeNode::Leaf { action: 2 }),
            }),
        },
    };

    let dose_levels = vec![0.0, 100.0, 200.0];
    let guideline = guideline_from_distilled_tree(
        &tree,
        &dose_levels,
        "Multi-split Test".to_string(),
        "Test with multiple splits".to_string(),
    );

    // Should have 3 rules (one for each leaf)
    assert_eq!(guideline.rules.len(), 3);
    assert_eq!(guideline.feature_names, vec!["ANC", "tumour_ratio"]);

    // Rule 0: ANC <= 0.5 → 0 mg
    assert_eq!(guideline.rules[0].action_dose_mg, 0.0);

    // Rule 1: ANC > 0.5 AND tumour_ratio <= 0.8 → 100 mg
    assert_eq!(guideline.rules[1].action_dose_mg, 100.0);
    assert_eq!(guideline.rules[1].conditions.len(), 2);

    // Rule 2: ANC > 0.5 AND tumour_ratio > 0.8 → 200 mg
    assert_eq!(guideline.rules[2].action_dose_mg, 200.0);
    assert_eq!(guideline.rules[2].conditions.len(), 2);
}

// =============================================================================
// Test 2: DoseGuidelineIRHost → GuidelineArtifact Bridge
// =============================================================================

#[test]
fn test_dose_guideline_to_guideline_artifact_single_rule() {
    let mut dg = DoseGuidelineIRHost::new(
        "Test Guideline".to_string(),
        "Test description".to_string(),
        vec!["ANC".to_string()],
        vec![0.0, 100.0],
    );

    // Add a single rule: if ANC <= 1.0 then dose 100 mg
    dg.add_rule(DoseRuleIR::new(
        vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::LE,
            1.0,
        )],
        1,
        100.0,
    ));

    // Create metadata
    let meta = GuidelineMeta {
        id: "test-001".to_string(),
        version: "1.0.0".to_string(),
        title: "Test Guideline".to_string(),
        description: "Test guideline for unit testing".to_string(),
        population: "Test population".to_string(),
        line_of_therapy: None,
        regimen_name: None,
        tumor_type: None,
    };

    // Convert to GuidelineArtifact
    let artifact = dose_guideline_to_guideline_artifact(&dg, meta);

    // Verify metadata
    assert_eq!(artifact.meta.id, "test-001");
    assert_eq!(artifact.meta.title, "Test Guideline");

    // Verify rules
    assert_eq!(artifact.rules.len(), 1);
    let rule = &artifact.rules[0];

    // Check condition
    match &rule.condition {
        GuidelineExpr::Compare { lhs, op, rhs } => {
            assert!(matches!(lhs, GuidelineValueRef::Anc));
            assert_eq!(*op, medlangc::guideline::ir::CmpOp::Le);
            assert_eq!(*rhs, 1.0);
        }
        _ => panic!("Expected Compare expression"),
    }

    // Check action
    match &rule.action {
        GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(dose)) => {
            assert_eq!(*dose, 100.0);
        }
        _ => panic!("Expected SetAbsoluteDoseMg action"),
    }

    // Check description
    assert!(rule.description.is_some());
    assert!(rule.description.as_ref().unwrap().contains("100 mg"));
}

#[test]
fn test_dose_guideline_to_guideline_artifact_multiple_rules() {
    let mut dg = DoseGuidelineIRHost::new(
        "Multi-rule".to_string(),
        "Multiple rules".to_string(),
        vec!["ANC".to_string(), "cycle".to_string()],
        vec![0.0, 50.0, 100.0],
    );

    // Rule 1: if ANC < 0.5 then hold
    dg.add_rule(DoseRuleIR::new(
        vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::LT,
            0.5,
        )],
        0,
        0.0,
    ));

    // Rule 2: if ANC >= 0.5 AND cycle > 2 then dose 100 mg
    dg.add_rule(DoseRuleIR::new(
        vec![
            AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::GE, 0.5),
            AtomicConditionIR::new("cycle".to_string(), ComparisonOpIR::GT, 2.0),
        ],
        2,
        100.0,
    ));

    let meta = GuidelineMeta {
        id: "multi-001".to_string(),
        version: "1.0.0".to_string(),
        title: "Multi-rule Test".to_string(),
        description: "Testing multiple rules".to_string(),
        population: "Test".to_string(),
        line_of_therapy: None,
        regimen_name: None,
        tumor_type: None,
    };

    let artifact = dose_guideline_to_guideline_artifact(&dg, meta);

    assert_eq!(artifact.rules.len(), 2);

    // First rule should be HoldDose
    assert!(matches!(
        artifact.rules[0].action,
        GuidelineAction::DoseAction(DoseActionKind::HoldDose)
    ));

    // Second rule should be SetAbsoluteDoseMg(100.0) with AND condition
    assert!(matches!(
        artifact.rules[1].action,
        GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(100.0))
    ));

    match &artifact.rules[1].condition {
        GuidelineExpr::And(exprs) => {
            assert_eq!(exprs.len(), 2);
        }
        _ => panic!("Expected And expression for second rule"),
    }
}

// =============================================================================
// Test 3: End-to-End Pipeline
// =============================================================================

#[test]
fn test_end_to_end_tree_to_guideline_artifact() {
    use medlangc::rl::distill::DistillFeature;

    // Step 1: Create distilled tree
    let tree = DistilledPolicyTree {
        n_actions: 2,
        features: vec![DistillFeature {
            name: "ANC".to_string(),
            index: 0,
            min: 0.0,
            max: 2.0,
        }],
        root: TreeNode::Split {
            feature_index: 0,
            threshold: 0.5,
            left: Box::new(TreeNode::Leaf { action: 0 }),
            right: Box::new(TreeNode::Leaf { action: 1 }),
        },
    };

    let dose_levels = vec![0.0, 100.0];

    // Step 2: Convert to DoseGuidelineIRHost
    let dg = guideline_from_distilled_tree(
        &tree,
        &dose_levels,
        "E2E Test".to_string(),
        "End-to-end test".to_string(),
    );

    assert_eq!(dg.rules.len(), 2);

    // Step 3: Convert to GuidelineArtifact
    let meta = GuidelineMeta {
        id: "e2e-001".to_string(),
        version: "1.0.0".to_string(),
        title: "E2E Test".to_string(),
        description: "End-to-end test".to_string(),
        population: "Test".to_string(),
        line_of_therapy: None,
        regimen_name: None,
        tumor_type: None,
    };

    let artifact = dose_guideline_to_guideline_artifact(&dg, meta);

    assert_eq!(artifact.rules.len(), 2);
    assert_eq!(artifact.meta.id, "e2e-001");

    // Verify serialization round-trip
    let json = serde_json::to_string_pretty(&artifact).expect("Failed to serialize");
    assert!(json.contains("E2E Test"));
    assert!(json.contains("rules"));

    let deserialized: medlangc::guideline::ir::GuidelineArtifact =
        serde_json::from_str(&json).expect("Failed to deserialize");
    assert_eq!(deserialized.rules.len(), 2);
}

// =============================================================================
// Test 4: Guideline Comparison
// =============================================================================

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
fn test_compare_different_guidelines_conservative_vs_aggressive() {
    // RL guideline: conservative (hold if ANC < 1.0)
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

// =============================================================================
// Test 5: Pretty Printing
// =============================================================================

#[test]
fn test_pretty_print_dose_guideline() {
    let mut gl = DoseGuidelineIRHost::new(
        "NSCLC Phase 2".to_string(),
        "Dose adjustment guideline".to_string(),
        vec!["ANC".to_string(), "tumour_ratio".to_string()],
        vec![0.0, 100.0, 200.0],
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

    gl.add_rule(DoseRuleIR::new(
        vec![
            AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::GE, 0.5),
            AtomicConditionIR::new("tumour_ratio".to_string(), ComparisonOpIR::GT, 0.8),
        ],
        2,
        200.0,
    ));

    let output = pretty_print_dose_guideline(&gl);

    // Verify content
    assert!(output.contains("NSCLC Phase 2"));
    assert!(output.contains("Rule #1"));
    assert!(output.contains("Rule #2"));
    assert!(output.contains("ANC < 0.5"));
    assert!(output.contains("tumour_ratio > 0.8"));
    assert!(output.contains("0 mg"));
    assert!(output.contains("200 mg"));
    assert!(output.contains("Features: ANC, tumour_ratio"));
}

// =============================================================================
// Test 6: Guideline Evaluation
// =============================================================================

#[test]
fn test_guideline_evaluation() {
    let mut gl = DoseGuidelineIRHost::new(
        "Test".to_string(),
        "Test guideline".to_string(),
        vec!["ANC".to_string(), "cycle".to_string()],
        vec![0.0, 50.0, 100.0],
    );

    // Rule 1: if ANC < 0.5 then dose 50 mg
    gl.add_rule(DoseRuleIR::new(
        vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::LT,
            0.5,
        )],
        1,
        50.0,
    ));

    // Rule 2: if ANC >= 0.5 then dose 100 mg
    gl.add_rule(DoseRuleIR::new(
        vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::GE,
            0.5,
        )],
        2,
        100.0,
    ));

    // Test low ANC
    let features_low = vec![("ANC".to_string(), 0.3), ("cycle".to_string(), 1.0)];
    assert_eq!(gl.evaluate(&features_low), Some(50.0));

    // Test high ANC
    let features_high = vec![("ANC".to_string(), 0.7), ("cycle".to_string(), 1.0)];
    assert_eq!(gl.evaluate(&features_high), Some(100.0));

    // Test boundary
    let features_boundary = vec![("ANC".to_string(), 0.5), ("cycle".to_string(), 1.0)];
    assert_eq!(gl.evaluate(&features_boundary), Some(100.0));
}
