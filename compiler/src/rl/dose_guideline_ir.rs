// Week 38: Dose Guideline Intermediate Representation
//
// A simpler, more focused IR for dose adjustment guidelines extracted from
// RL policies. This sits between DistilledPolicyTree and GuidelineArtifact,
// providing an explicit IF/THEN dose rule representation.

use serde::{Deserialize, Serialize};

use crate::rl::distill::{DistilledPolicyTree, TreeNode};
use crate::rl::env_dose_tox::DoseToxFeatureKind;

/// Comparison operators for atomic conditions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOpIR {
    LT, // <
    LE, // <=
    GT, // >
    GE, // >=
}

impl ComparisonOpIR {
    pub fn symbol(&self) -> &'static str {
        match self {
            ComparisonOpIR::LT => "<",
            ComparisonOpIR::LE => "<=",
            ComparisonOpIR::GT => ">",
            ComparisonOpIR::GE => ">=",
        }
    }

    pub fn apply(&self, lhs: f64, rhs: f64) -> bool {
        match self {
            ComparisonOpIR::LT => lhs < rhs,
            ComparisonOpIR::LE => lhs <= rhs,
            ComparisonOpIR::GT => lhs > rhs,
            ComparisonOpIR::GE => lhs >= rhs,
        }
    }
}

/// An atomic condition on a single feature
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AtomicConditionIR {
    /// Feature name (e.g., "ANC", "tumour_ratio", "prev_dose", "cycle")
    pub feature: String,

    /// Comparison operator
    pub op: ComparisonOpIR,

    /// Threshold value
    pub threshold: f64,
}

impl AtomicConditionIR {
    pub fn new(feature: String, op: ComparisonOpIR, threshold: f64) -> Self {
        Self {
            feature,
            op,
            threshold,
        }
    }

    /// Check if this condition is satisfied given a feature value
    pub fn evaluate(&self, value: f64) -> bool {
        self.op.apply(value, self.threshold)
    }

    /// Pretty-print this condition
    pub fn to_string(&self) -> String {
        format!("{} {} {:.1}", self.feature, self.op.symbol(), self.threshold)
    }
}

/// A single dose rule: IF conditions THEN dose action
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoseRuleIR {
    /// Conjunction of atomic conditions (all must be satisfied)
    pub conditions: Vec<AtomicConditionIR>,

    /// Action index (index into dose_levels array)
    pub action_index: usize,

    /// Actual dose amount in mg (derived from dose_levels[action_index])
    pub action_dose_mg: f64,
}

impl DoseRuleIR {
    pub fn new(
        conditions: Vec<AtomicConditionIR>,
        action_index: usize,
        action_dose_mg: f64,
    ) -> Self {
        Self {
            conditions,
            action_index,
            action_dose_mg,
        }
    }

    /// Check if all conditions are satisfied
    pub fn matches(&self, features: &[(String, f64)]) -> bool {
        for cond in &self.conditions {
            let value = features
                .iter()
                .find(|(name, _)| name == &cond.feature)
                .map(|(_, v)| *v);

            match value {
                Some(v) => {
                    if !cond.evaluate(v) {
                        return false;
                    }
                }
                None => return false, // Missing feature
            }
        }
        true
    }
}

/// Complete dose guideline with explicit IF/THEN rules
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoseGuidelineIRHost {
    /// Guideline name
    pub name: String,

    /// Description
    pub description: String,

    /// Feature names in order
    pub feature_names: Vec<String>,

    /// Dose levels in mg (indexed by action_index in rules)
    pub dose_levels_mg: Vec<f64>,

    /// List of dose rules (evaluated in order, first match wins)
    pub rules: Vec<DoseRuleIR>,
}

impl DoseGuidelineIRHost {
    pub fn new(
        name: String,
        description: String,
        feature_names: Vec<String>,
        dose_levels_mg: Vec<f64>,
    ) -> Self {
        Self {
            name,
            description,
            feature_names,
            dose_levels_mg,
            rules: Vec::new(),
        }
    }

    pub fn add_rule(&mut self, rule: DoseRuleIR) {
        self.rules.push(rule);
    }

    /// Find the first matching rule and return its dose
    pub fn evaluate(&self, features: &[(String, f64)]) -> Option<f64> {
        for rule in &self.rules {
            if rule.matches(features) {
                return Some(rule.action_dose_mg);
            }
        }
        None // No matching rule
    }

    /// Get dose level for a given action index
    pub fn dose_for_action(&self, action_index: usize) -> Option<f64> {
        self.dose_levels_mg.get(action_index).copied()
    }
}

// =============================================================================
// Pretty Printing
// =============================================================================

/// Pretty-print a dose guideline to a human-readable string
pub fn pretty_print_dose_guideline(gl: &DoseGuidelineIRHost) -> String {
    let mut output = String::new();

    output.push_str(&format!("Dose Guideline: {}\n", gl.name));
    output.push_str(&format!("Description: {}\n", gl.description));
    output.push_str(&format!("Features: {}\n", gl.feature_names.join(", ")));
    output.push_str(&format!("Dose Levels: {:?} mg\n", gl.dose_levels_mg));
    output.push_str(&format!("\nRules ({} total):\n", gl.rules.len()));
    output.push_str("=".repeat(80).as_str());
    output.push('\n');

    for (i, rule) in gl.rules.iter().enumerate() {
        output.push_str(&format!("\nRule #{}:\n", i + 1));

        if rule.conditions.is_empty() {
            output.push_str("  IF: (always)\n");
        } else {
            output.push_str("  IF:\n");
            for cond in &rule.conditions {
                output.push_str(&format!("    - {}\n", cond.to_string()));
            }
        }

        output.push_str(&format!(
            "  THEN: Set dose to {} mg (action index {})\n",
            rule.action_dose_mg, rule.action_index
        ));
    }

    output.push('\n');
    output.push_str("=".repeat(80).as_str());
    output.push('\n');

    output
}

// =============================================================================
// Transformation from DistilledPolicyTree
// =============================================================================

/// A constraint on a single feature from a tree path
#[derive(Debug, Clone)]
struct PathConstraint {
    feature: String,
    lower: Option<f64>, // exclusive: x > lower
    upper: Option<f64>, // inclusive: x <= upper
}

/// Extract all root-to-leaf paths from a decision tree and convert to dose rules
fn extract_tree_rules(tree: &DistilledPolicyTree, dose_levels: &[f64]) -> Vec<DoseRuleIR> {
    let mut rules = Vec::new();
    let constraints = Vec::new();
    extract_rules_recursive(&tree.root, &constraints, dose_levels, &mut rules);
    rules
}

/// Recursively extract rules from a tree node
fn extract_rules_recursive(
    node: &TreeNode,
    prefix: &[PathConstraint],
    dose_levels: &[f64],
    out: &mut Vec<DoseRuleIR>,
) {
    match node {
        TreeNode::Leaf { action } => {
            // Convert path constraints to atomic conditions
            let conditions = constraints_to_atomic_conditions(prefix);

            // Get dose for this action
            if let Some(&dose_mg) = dose_levels.get(*action) {
                out.push(DoseRuleIR::new(conditions, *action, dose_mg));
            }
        }
        TreeNode::Split {
            feature_index,
            threshold,
            left,
            right,
        } => {
            let feature_name = feature_name_for_index(*feature_index);

            // Left branch: x <= threshold
            let mut left_constraints = prefix.to_vec();
            add_upper_bound(&mut left_constraints, feature_name.clone(), *threshold);
            extract_rules_recursive(left, &left_constraints, dose_levels, out);

            // Right branch: x > threshold
            let mut right_constraints = prefix.to_vec();
            add_lower_bound(&mut right_constraints, feature_name, *threshold);
            extract_rules_recursive(right, &right_constraints, dose_levels, out);
        }
    }
}

/// Map feature index to feature name
fn feature_name_for_index(idx: usize) -> String {
    DoseToxFeatureKind::from_index(idx)
        .map(|k| k.name().to_string())
        .unwrap_or_else(|| format!("feature_{}", idx))
}

/// Add an upper bound constraint (x <= threshold)
fn add_upper_bound(constraints: &mut Vec<PathConstraint>, feature: String, threshold: f64) {
    if let Some(pc) = constraints.iter_mut().find(|c| c.feature == feature) {
        // Merge with existing constraint
        match pc.upper {
            None => pc.upper = Some(threshold),
            Some(prev) => pc.upper = Some(prev.min(threshold)),
        }
    } else {
        // New constraint
        constraints.push(PathConstraint {
            feature,
            lower: None,
            upper: Some(threshold),
        });
    }
}

/// Add a lower bound constraint (x > threshold)
fn add_lower_bound(constraints: &mut Vec<PathConstraint>, feature: String, threshold: f64) {
    if let Some(pc) = constraints.iter_mut().find(|c| c.feature == feature) {
        // Merge with existing constraint
        match pc.lower {
            None => pc.lower = Some(threshold),
            Some(prev) => pc.lower = Some(prev.max(threshold)),
        }
    } else {
        // New constraint
        constraints.push(PathConstraint {
            feature,
            lower: Some(threshold),
            upper: None,
        });
    }
}

/// Convert path constraints to atomic conditions
fn constraints_to_atomic_conditions(constraints: &[PathConstraint]) -> Vec<AtomicConditionIR> {
    let mut conditions = Vec::new();

    for c in constraints {
        // Lower bound: x > lower
        if let Some(lower) = c.lower {
            conditions.push(AtomicConditionIR::new(
                c.feature.clone(),
                ComparisonOpIR::GT,
                lower,
            ));
        }

        // Upper bound: x <= upper
        if let Some(upper) = c.upper {
            conditions.push(AtomicConditionIR::new(
                c.feature.clone(),
                ComparisonOpIR::LE,
                upper,
            ));
        }
    }

    conditions
}

/// Transform a DistilledPolicyTree into a DoseGuidelineIRHost
///
/// This is the main entry point for converting RL policies into dose guidelines.
pub fn guideline_from_distilled_tree(
    tree: &DistilledPolicyTree,
    dose_levels: &[f64],
    name: String,
    description: String,
) -> DoseGuidelineIRHost {
    // Extract feature names from tree
    let feature_names: Vec<String> = tree.features.iter().map(|f| f.name.clone()).collect();

    // Create guideline with dose levels
    let mut guideline =
        DoseGuidelineIRHost::new(name, description, feature_names, dose_levels.to_vec());

    // Extract and add all rules
    let rules = extract_tree_rules(tree, dose_levels);
    for rule in rules {
        guideline.add_rule(rule);
    }

    guideline
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_op() {
        assert!(ComparisonOpIR::LT.apply(1.0, 2.0));
        assert!(!ComparisonOpIR::LT.apply(2.0, 1.0));

        assert!(ComparisonOpIR::LE.apply(1.0, 1.0));
        assert!(ComparisonOpIR::LE.apply(1.0, 2.0));

        assert!(ComparisonOpIR::GT.apply(2.0, 1.0));
        assert!(!ComparisonOpIR::GT.apply(1.0, 2.0));

        assert!(ComparisonOpIR::GE.apply(1.0, 1.0));
        assert!(ComparisonOpIR::GE.apply(2.0, 1.0));
    }

    #[test]
    fn test_atomic_condition() {
        let cond = AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::LT, 0.5);

        assert!(cond.evaluate(0.3));
        assert!(!cond.evaluate(0.7));
        assert!(!cond.evaluate(0.5));
    }

    #[test]
    fn test_atomic_condition_to_string() {
        let cond = AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::LE, 1.0);
        assert_eq!(cond.to_string(), "ANC <= 1.0");
    }

    #[test]
    fn test_dose_rule_matching() {
        let rule = DoseRuleIR::new(
            vec![
                AtomicConditionIR::new("ANC".to_string(), ComparisonOpIR::LT, 0.5),
                AtomicConditionIR::new("cycle".to_string(), ComparisonOpIR::GE, 2.0),
            ],
            0,
            50.0,
        );

        // Both conditions satisfied
        let features = vec![("ANC".to_string(), 0.3), ("cycle".to_string(), 3.0)];
        assert!(rule.matches(&features));

        // First condition fails
        let features = vec![("ANC".to_string(), 0.7), ("cycle".to_string(), 3.0)];
        assert!(!rule.matches(&features));

        // Second condition fails
        let features = vec![("ANC".to_string(), 0.3), ("cycle".to_string(), 1.0)];
        assert!(!rule.matches(&features));

        // Missing feature
        let features = vec![("ANC".to_string(), 0.3)];
        assert!(!rule.matches(&features));
    }

    #[test]
    fn test_dose_guideline_evaluation() {
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

        // Test evaluation
        let features_low = vec![("ANC".to_string(), 0.3), ("cycle".to_string(), 1.0)];
        assert_eq!(gl.evaluate(&features_low), Some(50.0));

        let features_high = vec![("ANC".to_string(), 0.7), ("cycle".to_string(), 1.0)];
        assert_eq!(gl.evaluate(&features_high), Some(100.0));
    }

    #[test]
    fn test_dose_guideline_no_match() {
        let gl = DoseGuidelineIRHost::new(
            "Test".to_string(),
            "Test guideline".to_string(),
            vec!["ANC".to_string()],
            vec![0.0, 100.0],
        );

        // No rules, so no match
        let features = vec![("ANC".to_string(), 0.5)];
        assert_eq!(gl.evaluate(&features), None);
    }

    #[test]
    fn test_dose_for_action() {
        let gl = DoseGuidelineIRHost::new(
            "Test".to_string(),
            "Test".to_string(),
            vec!["ANC".to_string()],
            vec![0.0, 50.0, 100.0],
        );

        assert_eq!(gl.dose_for_action(0), Some(0.0));
        assert_eq!(gl.dose_for_action(1), Some(50.0));
        assert_eq!(gl.dose_for_action(2), Some(100.0));
        assert_eq!(gl.dose_for_action(3), None);
    }

    #[test]
    fn test_pretty_print() {
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

        assert!(output.contains("NSCLC Phase 2"));
        assert!(output.contains("Rule #1"));
        assert!(output.contains("Rule #2"));
        assert!(output.contains("ANC < 0.5"));
        assert!(output.contains("tumour_ratio > 0.8"));
        assert!(output.contains("0 mg"));
        assert!(output.contains("200 mg"));
    }

    #[test]
    fn test_empty_conditions_rule() {
        let rule = DoseRuleIR::new(vec![], 1, 100.0);

        // Empty conditions should match any feature set
        let features = vec![("ANC".to_string(), 0.5)];
        assert!(rule.matches(&features));

        let empty_features = vec![];
        assert!(rule.matches(&empty_features));
    }

    #[test]
    fn test_guideline_from_distilled_tree() {
        use crate::rl::distill::DistillFeature;

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

        let dose_levels = vec![0.0, 100.0];
        let guideline = guideline_from_distilled_tree(
            &tree,
            &dose_levels,
            "Test Guideline".to_string(),
            "Test description".to_string(),
        );

        assert_eq!(guideline.name, "Test Guideline");
        assert_eq!(guideline.rules.len(), 2);
        assert_eq!(guideline.feature_names, vec!["ANC"]);
        assert_eq!(guideline.dose_levels_mg, vec![0.0, 100.0]);

        // First rule: ANC <= 0.5 -> dose 0 mg
        assert_eq!(guideline.rules[0].action_dose_mg, 0.0);
        assert_eq!(guideline.rules[0].conditions.len(), 1);
        assert_eq!(guideline.rules[0].conditions[0].feature, "ANC");
        assert_eq!(guideline.rules[0].conditions[0].op, ComparisonOpIR::LE);
        assert_eq!(guideline.rules[0].conditions[0].threshold, 0.5);

        // Second rule: ANC > 0.5 -> dose 100 mg
        assert_eq!(guideline.rules[1].action_dose_mg, 100.0);
        assert_eq!(guideline.rules[1].conditions.len(), 1);
        assert_eq!(guideline.rules[1].conditions[0].feature, "ANC");
        assert_eq!(guideline.rules[1].conditions[0].op, ComparisonOpIR::GT);
        assert_eq!(guideline.rules[1].conditions[0].threshold, 0.5);
    }

    #[test]
    fn test_interval_merging() {
        // Create constraints: 0.3 < ANC <= 0.7
        let mut constraints = vec![PathConstraint {
            feature: "ANC".to_string(),
            lower: Some(0.3),
            upper: Some(0.7),
        }];

        // Add tighter upper bound: ANC <= 0.6
        add_upper_bound(&mut constraints, "ANC".to_string(), 0.6);

        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].lower, Some(0.3));
        assert_eq!(constraints[0].upper, Some(0.6)); // Tighter bound wins
    }

    #[test]
    fn test_constraints_to_atomic_conditions() {
        let constraints = vec![PathConstraint {
            feature: "ANC".to_string(),
            lower: Some(0.3),
            upper: Some(0.7),
        }];

        let conditions = constraints_to_atomic_conditions(&constraints);

        assert_eq!(conditions.len(), 2);

        // Should have: ANC > 0.3 AND ANC <= 0.7
        assert_eq!(conditions[0].feature, "ANC");
        assert_eq!(conditions[0].op, ComparisonOpIR::GT);
        assert_eq!(conditions[0].threshold, 0.3);

        assert_eq!(conditions[1].feature, "ANC");
        assert_eq!(conditions[1].op, ComparisonOpIR::LE);
        assert_eq!(conditions[1].threshold, 0.7);
    }
}
