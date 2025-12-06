// Phase V2: Verification Condition Generator
//
// Generates proof obligations from MedLang code

use super::translator::SMTFormula;
use crate::ir::{IRBinaryOp, IRExpr, IRUnaryOp};
use anyhow::Result;
use std::collections::HashMap;

/// Source location for error reporting
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocation {
    pub file: String,
    pub line: usize,
    pub column: usize,
}

/// Verification condition (proof obligation)
#[derive(Debug, Clone)]
pub struct VerificationCondition {
    /// Assumptions (context/preconditions)
    pub assumptions: Vec<SMTFormula>,
    /// Goal to prove
    pub goal: SMTFormula,
    /// Source location
    pub location: Option<SourceLocation>,
    /// Human-readable description
    pub description: String,
}

impl VerificationCondition {
    pub fn new(assumptions: Vec<SMTFormula>, goal: SMTFormula, description: String) -> Self {
        VerificationCondition {
            assumptions,
            goal,
            location: None,
            description,
        }
    }

    pub fn with_location(mut self, location: SourceLocation) -> Self {
        self.location = Some(location);
        self
    }
}

/// Verification condition generator
pub struct VCGenerator {
    /// Current assumptions in scope
    assumptions: Vec<SMTFormula>,
    /// Generated VCs
    vcs: Vec<VerificationCondition>,
}

impl VCGenerator {
    pub fn new() -> Self {
        VCGenerator {
            assumptions: Vec::new(),
            vcs: Vec::new(),
        }
    }

    /// Add an assumption to the current context
    pub fn assume(&mut self, formula: SMTFormula) {
        self.assumptions.push(formula);
    }

    /// Generate a verification condition
    pub fn generate_vc(&mut self, goal: SMTFormula, description: String) {
        let vc = VerificationCondition::new(self.assumptions.clone(), goal, description);
        self.vcs.push(vc);
    }

    /// Get all generated VCs
    pub fn get_vcs(&self) -> &[VerificationCondition] {
        &self.vcs
    }

    /// Clear all VCs and assumptions
    pub fn reset(&mut self) {
        self.assumptions.clear();
        self.vcs.clear();
    }

    /// Generate division safety VC
    ///
    /// For an expression `a / b`, generates VC: b ≠ 0
    pub fn generate_division_safety(&mut self, denominator_var: &str, description: Option<String>) {
        use super::translator::{ComparisonOp, SMTExpr};

        let goal = SMTFormula::Comparison {
            lhs: SMTExpr::Var(denominator_var.to_string()),
            op: ComparisonOp::Ne,
            rhs: SMTExpr::RealLit(0.0),
        };

        let desc =
            description.unwrap_or_else(|| format!("Division safety: {} ≠ 0", denominator_var));

        self.generate_vc(goal, desc);
    }

    /// Generate range safety VC
    ///
    /// For a variable with range constraint, generates VC that value stays in range
    pub fn generate_range_safety(
        &mut self,
        var: &str,
        lower: f64,
        upper: f64,
        description: Option<String>,
    ) {
        use super::translator::{ComparisonOp, LogicalOp, SMTExpr};

        let lower_bound = SMTFormula::Comparison {
            lhs: SMTExpr::Var(var.to_string()),
            op: ComparisonOp::Ge,
            rhs: SMTExpr::RealLit(lower),
        };

        let upper_bound = SMTFormula::Comparison {
            lhs: SMTExpr::Var(var.to_string()),
            op: ComparisonOp::Le,
            rhs: SMTExpr::RealLit(upper),
        };

        let goal = SMTFormula::Logical {
            op: LogicalOp::And,
            operands: vec![lower_bound, upper_bound],
        };

        let desc = description
            .unwrap_or_else(|| format!("Range safety: {} in [{}, {}]", var, lower, upper));

        self.generate_vc(goal, desc);
    }

    /// Generate non-negativity VC
    ///
    /// For a variable that must be non-negative (e.g., clearance, volume)
    pub fn generate_non_negativity(
        &mut self,
        var: &str,
        strict: bool,
        description: Option<String>,
    ) {
        use super::translator::{ComparisonOp, SMTExpr};

        let op = if strict {
            ComparisonOp::Gt
        } else {
            ComparisonOp::Ge
        };

        let goal = SMTFormula::Comparison {
            lhs: SMTExpr::Var(var.to_string()),
            op,
            rhs: SMTExpr::RealLit(0.0),
        };

        let desc = description.unwrap_or_else(|| {
            if strict {
                format!("Non-negativity (strict): {} > 0", var)
            } else {
                format!("Non-negativity: {} >= 0", var)
            }
        });

        self.generate_vc(goal, desc);
    }

    /// Analyze an IR expression and generate necessary VCs
    pub fn analyze_expr(
        &mut self,
        expr: &IRExpr,
        var_constraints: &HashMap<String, Vec<SMTFormula>>,
    ) -> Result<()> {
        match expr {
            IRExpr::Binary(op, lhs, rhs) => {
                // Recursively analyze subexpressions
                self.analyze_expr(lhs, var_constraints)?;
                self.analyze_expr(rhs, var_constraints)?;

                // Check for division
                if matches!(op, IRBinaryOp::Div) {
                    // If RHS is a variable, check it's not zero
                    if let IRExpr::Var(var_name) = rhs.as_ref() {
                        // Check if we have a constraint proving var > 0 or var < 0
                        if !self.has_nonzero_constraint(var_name, var_constraints) {
                            self.generate_division_safety(
                                var_name,
                                Some(format!("Division by {} is safe", var_name)),
                            );
                        }
                    }
                }
            }

            IRExpr::Unary(_op, expr) => {
                self.analyze_expr(expr, var_constraints)?;
            }

            IRExpr::Call(name, args) => {
                for arg in args {
                    self.analyze_expr(arg, var_constraints)?;
                }
            }

            // Literals, variables, and array indexing don't generate VCs
            IRExpr::Literal(_) | IRExpr::Var(_) | IRExpr::Index(_, _) => {}
        }

        Ok(())
    }

    /// Check if we have a constraint proving variable is nonzero
    fn has_nonzero_constraint(
        &self,
        var_name: &str,
        var_constraints: &HashMap<String, Vec<SMTFormula>>,
    ) -> bool {
        use super::translator::{ComparisonOp, SMTExpr};

        if let Some(constraints) = var_constraints.get(var_name) {
            for constraint in constraints {
                // Check if constraint is x > 0 or x < 0
                if let SMTFormula::Comparison { lhs, op, rhs } = constraint {
                    if let SMTExpr::Var(var) = lhs {
                        if var == var_name {
                            if matches!(op, ComparisonOp::Gt | ComparisonOp::Lt) {
                                if let SMTExpr::RealLit(val) = rhs {
                                    if *val == 0.0 {
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        false
    }
}

impl Default for VCGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::phase_v1::LogicalOp;

    #[test]
    fn test_division_safety_vc() {
        let mut gen = VCGenerator::new();
        gen.generate_division_safety("V", None);

        let vcs = gen.get_vcs();
        assert_eq!(vcs.len(), 1);
        assert!(vcs[0].description.contains("Division safety"));
        assert!(vcs[0].description.contains("V"));
    }

    #[test]
    fn test_range_safety_vc() {
        let mut gen = VCGenerator::new();
        gen.generate_range_safety("WT", 30.0, 200.0, None);

        let vcs = gen.get_vcs();
        assert_eq!(vcs.len(), 1);

        // Check goal is an AND of two comparisons
        if let SMTFormula::Logical { op, operands } = &vcs[0].goal {
            assert_eq!(*op, LogicalOp::And);
            assert_eq!(operands.len(), 2);
        } else {
            panic!("Expected logical AND in goal");
        }
    }

    #[test]
    fn test_non_negativity_vc() {
        let mut gen = VCGenerator::new();

        // Strict non-negativity
        gen.generate_non_negativity("CL", true, None);
        let vcs = gen.get_vcs();
        assert_eq!(vcs.len(), 1);
        assert!(vcs[0].description.contains("strict"));

        // Reset and test non-strict
        gen.reset();
        gen.generate_non_negativity("V", false, None);
        let vcs = gen.get_vcs();
        assert_eq!(vcs.len(), 1);
        assert!(!vcs[0].description.contains("strict"));
    }

    #[test]
    fn test_assumptions() {
        use super::super::translator::{ComparisonOp, SMTExpr};

        let mut gen = VCGenerator::new();

        // Add assumption
        let assumption = SMTFormula::Comparison {
            lhs: SMTExpr::Var("x".to_string()),
            op: ComparisonOp::Gt,
            rhs: SMTExpr::RealLit(0.0),
        };
        gen.assume(assumption.clone());

        // Generate VC
        let goal = SMTFormula::Comparison {
            lhs: SMTExpr::Var("x".to_string()),
            op: ComparisonOp::Ge,
            rhs: SMTExpr::RealLit(0.0),
        };
        gen.generate_vc(goal, "Test VC".to_string());

        let vcs = gen.get_vcs();
        assert_eq!(vcs.len(), 1);
        assert_eq!(vcs[0].assumptions.len(), 1);
        assert_eq!(vcs[0].assumptions[0], assumption);
    }
}
