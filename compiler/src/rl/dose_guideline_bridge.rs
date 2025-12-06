// Week 38: Bridge from DoseGuidelineIR to GuidelineArtifact
//
// Converts the simpler DoseGuidelineIR format into the general-purpose
// GuidelineArtifact format that can be exported to CQL and other formats.

use crate::guideline::ir::{
    CmpOp, DoseActionKind, GuidelineAction, GuidelineArtifact, GuidelineExpr, GuidelineMeta,
    GuidelineRule, GuidelineValueRef,
};
use crate::rl::dose_guideline_ir::{
    AtomicConditionIR, ComparisonOpIR, DoseGuidelineIRHost, DoseRuleIR,
};

// =============================================================================
// Feature Name Mapping
// =============================================================================

/// Map feature name from DoseGuidelineIR to GuidelineValueRef.
/// For now we support canonical DoseTox features; unknown names can be rejected
/// or mapped to a generic variant if present.
fn value_ref_from_feature_name(name: &str) -> Option<GuidelineValueRef> {
    match name {
        "ANC" | "anc" => Some(GuidelineValueRef::Anc),
        "tumour_ratio" | "tumor_ratio" => Some(GuidelineValueRef::TumourRatio),
        "prev_dose" | "previous_dose" => Some(GuidelineValueRef::PrevDose),
        "cycle" | "cycle_index" => Some(GuidelineValueRef::CycleIndex),
        _ => {
            // For v0, be strict and return None for unknown features
            // In the future, we could use GuidelineValueRef::Lab(name.to_string())
            None
        }
    }
}

/// Map comparison operator from DoseGuidelineIR to GuidelineIR
fn cmp_op_from_comparison(op: ComparisonOpIR) -> CmpOp {
    match op {
        ComparisonOpIR::LT => CmpOp::Lt,
        ComparisonOpIR::LE => CmpOp::Le,
        ComparisonOpIR::GT => CmpOp::Gt,
        ComparisonOpIR::GE => CmpOp::Ge,
    }
}

// =============================================================================
// Condition Translation
// =============================================================================

/// Convert atomic conditions from a DoseRuleIR into a GuidelineExpr.
/// All conditions are combined with AND logic.
fn expr_from_conditions(conds: &[AtomicConditionIR]) -> GuidelineExpr {
    if conds.is_empty() {
        return GuidelineExpr::True;
    }

    let mut exprs = Vec::new();
    for c in conds {
        if let Some(lhs) = value_ref_from_feature_name(&c.feature) {
            exprs.push(GuidelineExpr::Compare {
                lhs,
                op: cmp_op_from_comparison(c.op),
                rhs: c.threshold,
            });
        } else {
            // Unknown feature name; for v0 we skip this condition
            // This means the rule will be looser than intended, but won't crash
            eprintln!(
                "Warning: Unknown feature '{}' in dose guideline, skipping condition",
                c.feature
            );
        }
    }

    if exprs.is_empty() {
        GuidelineExpr::True
    } else if exprs.len() == 1 {
        exprs.into_iter().next().unwrap()
    } else {
        GuidelineExpr::And(exprs)
    }
}

// =============================================================================
// Action Translation
// =============================================================================

/// Convert a DoseRuleIR's action into a GuidelineAction
fn action_from_rule(rule: &DoseRuleIR) -> GuidelineAction {
    if rule.action_dose_mg <= 0.0 {
        GuidelineAction::DoseAction(DoseActionKind::HoldDose)
    } else {
        GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(rule.action_dose_mg))
    }
}

// =============================================================================
// Main Bridge Function
// =============================================================================

/// Convert a DoseGuidelineIRHost into a general GuidelineArtifact using given GuidelineMeta.
///
/// This is the main bridge function that transforms the RL-derived dose rules
/// into the generic guideline format suitable for CQL export and clinical use.
pub fn dose_guideline_to_guideline_artifact(
    dg: &DoseGuidelineIRHost,
    meta: GuidelineMeta,
) -> GuidelineArtifact {
    let mut rules = Vec::new();

    for (i, r) in dg.rules.iter().enumerate() {
        let condition = expr_from_conditions(&r.conditions);
        let action = action_from_rule(r);
        let desc = Some(format!(
            "RL-derived dose rule #{}: action_index = {}, dose = {} mg",
            i + 1,
            r.action_index,
            r.action_dose_mg
        ));

        rules.push(GuidelineRule {
            condition,
            action,
            description: desc,
            priority: Some((dg.rules.len() - i) as u32), // Reverse order priority
        });
    }

    GuidelineArtifact { meta, rules }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::dose_guideline_ir::DoseGuidelineIRHost;

    #[test]
    fn test_value_ref_from_feature_name() {
        assert!(matches!(
            value_ref_from_feature_name("ANC"),
            Some(GuidelineValueRef::Anc)
        ));
        assert!(matches!(
            value_ref_from_feature_name("anc"),
            Some(GuidelineValueRef::Anc)
        ));
        assert!(matches!(
            value_ref_from_feature_name("tumour_ratio"),
            Some(GuidelineValueRef::TumourRatio)
        ));
        assert!(matches!(
            value_ref_from_feature_name("prev_dose"),
            Some(GuidelineValueRef::PrevDose)
        ));
        assert!(matches!(
            value_ref_from_feature_name("cycle"),
            Some(GuidelineValueRef::CycleIndex)
        ));
        assert!(value_ref_from_feature_name("unknown").is_none());
    }

    #[test]
    fn test_cmp_op_from_comparison() {
        assert_eq!(cmp_op_from_comparison(ComparisonOpIR::LT), CmpOp::Lt);
        assert_eq!(cmp_op_from_comparison(ComparisonOpIR::LE), CmpOp::Le);
        assert_eq!(cmp_op_from_comparison(ComparisonOpIR::GT), CmpOp::Gt);
        assert_eq!(cmp_op_from_comparison(ComparisonOpIR::GE), CmpOp::Ge);
    }

    #[test]
    fn test_expr_from_conditions_empty() {
        let conds = vec![];
        let expr = expr_from_conditions(&conds);
        assert!(matches!(expr, GuidelineExpr::True));
    }

    #[test]
    fn test_expr_from_conditions_single() {
        let conds = vec![AtomicConditionIR::new(
            "ANC".to_string(),
            ComparisonOpIR::LE,
            1.0,
        )];

        let expr = expr_from_conditions(&conds);

        match expr {
            GuidelineExpr::Compare { lhs, op, rhs } => {
                assert!(matches!(lhs, GuidelineValueRef::Anc));
                assert_eq!(op, CmpOp::Le);
                assert_eq!(rhs, 1.0);
            }
            _ => panic!("Expected Compare expression"),
        }
    }

    #[test]
    fn test_expr_from_conditions_multiple() {
        let conds = vec![
            AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::GE, 0.5),
            AtomicConditionIR::new("tumour_ratio".to_string(), ComparisonOpIR::GT, 0.8),
        ];

        let expr = expr_from_conditions(&conds);

        match expr {
            GuidelineExpr::And(exprs) => {
                assert_eq!(exprs.len(), 2);
            }
            _ => panic!("Expected And expression"),
        }
    }

    #[test]
    fn test_action_from_rule_hold() {
        let rule = DoseRuleIR::new(vec![], 0, 0.0);
        let action = action_from_rule(&rule);

        assert!(matches!(
            action,
            GuidelineAction::DoseAction(DoseActionKind::HoldDose)
        ));
    }

    #[test]
    fn test_action_from_rule_set_dose() {
        let rule = DoseRuleIR::new(vec![], 1, 100.0);
        let action = action_from_rule(&rule);

        match action {
            GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(dose)) => {
                assert_eq!(dose, 100.0);
            }
            _ => panic!("Expected SetAbsoluteDoseMg"),
        }
    }

    #[test]
    fn test_dose_guideline_to_guideline_artifact() {
        // Create a simple DoseGuidelineIRHost
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

        // Verify
        assert_eq!(artifact.meta.id, "test-001");
        assert_eq!(artifact.rules.len(), 1);

        let rule = &artifact.rules[0];
        assert!(rule.description.is_some());
        assert!(matches!(
            rule.action,
            GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(100.0))
        ));

        match &rule.condition {
            GuidelineExpr::Compare { lhs, op, rhs } => {
                assert!(matches!(lhs, GuidelineValueRef::Anc));
                assert_eq!(*op, CmpOp::Le);
                assert_eq!(*rhs, 1.0);
            }
            _ => panic!("Expected Compare expression"),
        }
    }

    #[test]
    fn test_bridge_with_multiple_rules() {
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

        // Second rule should be SetAbsoluteDoseMg(100.0)
        assert!(matches!(
            artifact.rules[1].action,
            GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(100.0))
        ));
    }
}
