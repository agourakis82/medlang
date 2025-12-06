// Week 37: Distilled Policy to Guideline Translation
//
// Converts decision trees from distilled RL policies into clinical
// guideline artifacts that can be exported to CQL or other formats.

use crate::guideline::ir::{
    CmpOp, DoseActionKind, GuidelineAction, GuidelineArtifact, GuidelineExpr, GuidelineMeta,
    GuidelineRule, GuidelineValueRef,
};
use crate::rl::distill::{DistilledPolicyTree, TreeNode};
use crate::rl::env_dose_tox::DoseToxFeatureKind;

// =============================================================================
// Path Extraction
// =============================================================================

/// A constraint on a single feature from a tree path
#[derive(Debug, Clone)]
struct PathConstraint {
    feature: DoseToxFeatureKind,
    lower: Option<f64>, // exclusive: x > lower
    upper: Option<f64>, // inclusive: x <= upper
}

/// A rule extracted from a tree leaf: conditions → action
#[derive(Debug, Clone)]
struct LeafRule {
    constraints: Vec<PathConstraint>,
    action_index: usize,
}

/// Extract all root-to-leaf paths from a decision tree
pub fn extract_tree_paths(tree: &DistilledPolicyTree) -> Vec<LeafRule> {
    let mut leaves = Vec::new();
    let constraints = Vec::new();
    extract_paths_recursive(&tree.root, &constraints, &mut leaves);
    leaves
}

/// Recursively extract paths from a tree node
fn extract_paths_recursive(node: &TreeNode, prefix: &[PathConstraint], out: &mut Vec<LeafRule>) {
    match node {
        TreeNode::Leaf { action } => {
            out.push(LeafRule {
                constraints: prefix.to_vec(),
                action_index: *action,
            });
        }
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            let feature_kind = feature_kind_for_index(*feature_index);

            // Left branch: x <= threshold
            let mut left_constraints = prefix.to_vec();
            add_upper_bound(&mut left_constraints, feature_kind, *threshold);
            extract_paths_recursive(left, &left_constraints, out);

            // Right branch: x > threshold
            let mut right_constraints = prefix.to_vec();
            add_lower_bound(&mut right_constraints, feature_kind, *threshold);
            extract_paths_recursive(right, &right_constraints, out);
        }
    }
}

/// Map feature index to feature kind
fn feature_kind_for_index(idx: usize) -> DoseToxFeatureKind {
    DoseToxFeatureKind::from_index(idx).unwrap_or(DoseToxFeatureKind::Anc)
}

/// Add an upper bound constraint (x <= threshold)
fn add_upper_bound(
    constraints: &mut Vec<PathConstraint>,
    feat: DoseToxFeatureKind,
    threshold: f64,
) {
    if let Some(pc) = constraints.iter_mut().find(|c| c.feature == feat) {
        // Merge with existing constraint
        match pc.upper {
            None => pc.upper = Some(threshold),
            Some(prev) => pc.upper = Some(prev.min(threshold)),
        }
    } else {
        // New constraint
        constraints.push(PathConstraint {
            feature: feat,
            lower: None,
            upper: Some(threshold),
        });
    }
}

/// Add a lower bound constraint (x > threshold)
fn add_lower_bound(
    constraints: &mut Vec<PathConstraint>,
    feat: DoseToxFeatureKind,
    threshold: f64,
) {
    if let Some(pc) = constraints.iter_mut().find(|c| c.feature == feat) {
        // Merge with existing constraint
        match pc.lower {
            None => pc.lower = Some(threshold),
            Some(prev) => pc.lower = Some(prev.max(threshold)),
        }
    } else {
        // New constraint
        constraints.push(PathConstraint {
            feature: feat,
            lower: Some(threshold),
            upper: None,
        });
    }
}

// =============================================================================
// Constraint to Expression Conversion
// =============================================================================

/// Convert path constraints to a guideline expression
fn constraints_to_expr(constraints: &[PathConstraint]) -> GuidelineExpr {
    if constraints.is_empty() {
        return GuidelineExpr::True;
    }

    let mut exprs = Vec::new();

    for c in constraints {
        let value_ref = match c.feature {
            DoseToxFeatureKind::Anc => GuidelineValueRef::Anc,
            DoseToxFeatureKind::TumourRatio => GuidelineValueRef::TumourRatio,
            DoseToxFeatureKind::PrevDose => GuidelineValueRef::PrevDose,
            DoseToxFeatureKind::CycleIndex => GuidelineValueRef::CycleIndex,
        };

        // Lower bound: x > lower
        if let Some(lower) = c.lower {
            exprs.push(GuidelineExpr::Compare {
                lhs: value_ref.clone(),
                op: CmpOp::Gt,
                rhs: lower,
            });
        }

        // Upper bound: x <= upper
        if let Some(upper) = c.upper {
            exprs.push(GuidelineExpr::Compare {
                lhs: value_ref,
                op: CmpOp::Le,
                rhs: upper,
            });
        }
    }

    GuidelineExpr::and(exprs).simplify()
}

// =============================================================================
// Dose Mapping
// =============================================================================

/// Map dose to guideline action
fn dose_to_action(dose_mg: f64) -> GuidelineAction {
    if dose_mg <= 0.0 {
        GuidelineAction::DoseAction(DoseActionKind::HoldDose)
    } else {
        GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(dose_mg))
    }
}

/// Create a description for a rule
fn create_rule_description(
    action_index: usize,
    dose_mg: f64,
    constraints: &[PathConstraint],
) -> String {
    let mut desc = format!(
        "RL-derived rule (action {}, dose {} mg)",
        action_index, dose_mg
    );

    if !constraints.is_empty() {
        desc.push_str(": ");
        let mut cond_parts = Vec::new();
        for c in constraints {
            let name = c.feature.name();
            match (c.lower, c.upper) {
                (Some(lower), Some(upper)) => {
                    cond_parts.push(format!("{} < {} <= {}", lower, name, upper));
                }
                (Some(lower), None) => {
                    cond_parts.push(format!("{} > {}", name, lower));
                }
                (None, Some(upper)) => {
                    cond_parts.push(format!("{} <= {}", name, upper));
                }
                (None, None) => {}
            }
        }
        desc.push_str(&cond_parts.join(" AND "));
    }

    desc
}

// =============================================================================
// Main Translation Function
// =============================================================================

/// Convert a distilled policy tree to a guideline artifact
pub fn distilled_policy_to_guideline_artifact(
    tree: &DistilledPolicyTree,
    dose_levels: &[f64],
    meta: GuidelineMeta,
) -> GuidelineArtifact {
    // Extract all leaf rules
    let leaf_rules = extract_tree_paths(tree);

    // Convert to guideline rules
    let mut rules = Vec::new();
    for (idx, leaf) in leaf_rules.iter().enumerate() {
        // Get dose for this action
        if let Some(&dose_mg) = dose_levels.get(leaf.action_index) {
            let condition = constraints_to_expr(&leaf.constraints);
            let action = dose_to_action(dose_mg);
            let description = Some(create_rule_description(
                leaf.action_index,
                dose_mg,
                &leaf.constraints,
            ));

            rules.push(GuidelineRule {
                condition,
                action,
                description,
                priority: Some((leaf_rules.len() - idx) as u32), // Reverse priority
            });
        }
    }

    GuidelineArtifact { meta, rules }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rl::distill::DistillFeature;

    #[test]
    fn test_simple_tree_extraction() {
        // Create a simple tree: if ANC <= 0.5 then action 0, else action 1
        let tree = DistilledPolicyTree {
            n_actions: 2,
            features: vec![DistillFeature {
                name: "ANC".to_string(),
                index: 0,
                min: 0.0,
                max: 1.0,
            }],
            root: TreeNode::Split {
                feature_index: 0,
                threshold: 0.5,
                left: Box::new(TreeNode::Leaf { action: 0 }),
                right: Box::new(TreeNode::Leaf { action: 1 }),
            },
        };

        let leaves = extract_tree_paths(&tree);
        assert_eq!(leaves.len(), 2);

        // First leaf: ANC <= 0.5 -> action 0
        assert_eq!(leaves[0].action_index, 0);
        assert_eq!(leaves[0].constraints.len(), 1);
        assert_eq!(leaves[0].constraints[0].upper, Some(0.5));

        // Second leaf: ANC > 0.5 -> action 1
        assert_eq!(leaves[1].action_index, 1);
        assert_eq!(leaves[1].constraints.len(), 1);
        assert_eq!(leaves[1].constraints[0].lower, Some(0.5));
    }

    #[test]
    fn test_constraints_to_expr() {
        let constraints = vec![PathConstraint {
            feature: DoseToxFeatureKind::Anc,
            lower: None,
            upper: Some(0.5),
        }];

        let expr = constraints_to_expr(&constraints);

        match expr {
            GuidelineExpr::Compare { lhs, op, rhs } => {
                assert!(matches!(lhs, GuidelineValueRef::Anc));
                assert_eq!(op, CmpOp::Le);
                assert_eq!(rhs, 0.5);
            }
            _ => panic!("Expected Compare expression"),
        }
    }

    #[test]
    fn test_dose_to_action() {
        let hold = dose_to_action(0.0);
        assert!(matches!(
            hold,
            GuidelineAction::DoseAction(DoseActionKind::HoldDose)
        ));

        let set = dose_to_action(100.0);
        match set {
            GuidelineAction::DoseAction(DoseActionKind::SetAbsoluteDoseMg(dose)) => {
                assert_eq!(dose, 100.0);
            }
            _ => panic!("Expected SetAbsoluteDoseMg"),
        }
    }

    #[test]
    fn test_interval_merging() {
        // Create constraints: 0.3 < ANC <= 0.7
        let mut constraints = vec![PathConstraint {
            feature: DoseToxFeatureKind::Anc,
            lower: Some(0.3),
            upper: Some(0.7),
        }];

        // Add tighter upper bound: ANC <= 0.6
        add_upper_bound(&mut constraints, DoseToxFeatureKind::Anc, 0.6);

        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].lower, Some(0.3));
        assert_eq!(constraints[0].upper, Some(0.6)); // Tighter bound wins
    }

    #[test]
    fn test_guideline_artifact_creation() {
        let tree = DistilledPolicyTree {
            n_actions: 2,
            features: vec![],
            root: TreeNode::Split {
                feature_index: 0,
                threshold: 0.5,
                left: Box::new(TreeNode::Leaf { action: 0 }),
                right: Box::new(TreeNode::Leaf { action: 1 }),
            },
        };

        let meta = GuidelineMeta {
            id: "test".to_string(),
            version: "1.0".to_string(),
            title: "Test".to_string(),
            description: "Test".to_string(),
            population: "Test".to_string(),
            line_of_therapy: None,
            regimen_name: None,
            tumor_type: None,
        };

        let dose_levels = vec![0.0, 100.0];
        let artifact = distilled_policy_to_guideline_artifact(&tree, &dose_levels, meta);

        assert_eq!(artifact.rules.len(), 2);
        assert!(artifact.rules[0].description.is_some());
    }

    #[test]
    fn test_create_rule_description() {
        let constraints = vec![PathConstraint {
            feature: DoseToxFeatureKind::Anc,
            lower: Some(0.3),
            upper: Some(0.7),
        }];

        let desc = create_rule_description(1, 100.0, &constraints);
        assert!(desc.contains("action 1"));
        assert!(desc.contains("100 mg"));
        assert!(desc.contains("ANC"));
    }
}
