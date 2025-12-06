// Phase V2: SMT-LIB Translator
//
// Translates MedLang refinement constraints to SMT-LIB formulas for Z3

use crate::ast::phase_v1::{ConstraintExpr, ConstraintLiteral};
use anyhow::{anyhow, Result};
use std::fmt;

// Re-export for other SMT modules
pub use crate::ast::phase_v1::{ComparisonOp, LogicalOp};

/// SMT sort (type) representation
#[derive(Debug, Clone, PartialEq)]
pub enum SMTSort {
    Real,           // Floating-point numbers
    Int,            // Integers
    Bool,           // Boolean values
    Custom(String), // Custom sorts (e.g., unit types)
}

impl fmt::Display for SMTSort {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SMTSort::Real => write!(f, "Real"),
            SMTSort::Int => write!(f, "Int"),
            SMTSort::Bool => write!(f, "Bool"),
            SMTSort::Custom(name) => write!(f, "{}", name),
        }
    }
}

/// SMT expression representation
#[derive(Debug, Clone, PartialEq)]
pub enum SMTExpr {
    Var(String),
    RealLit(f64),
    IntLit(i64),
    BoolLit(bool),
    /// Binary operation: (op lhs rhs)
    BinOp {
        op: String,
        lhs: Box<SMTExpr>,
        rhs: Box<SMTExpr>,
    },
    /// Unary operation: (op expr)
    UnOp {
        op: String,
        expr: Box<SMTExpr>,
    },
}

impl fmt::Display for SMTExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SMTExpr::Var(name) => write!(f, "{}", name),
            SMTExpr::RealLit(val) => write!(f, "{}", val),
            SMTExpr::IntLit(val) => write!(f, "{}", val),
            SMTExpr::BoolLit(val) => write!(f, "{}", val),
            SMTExpr::BinOp { op, lhs, rhs } => {
                write!(f, "({} {} {})", op, lhs, rhs)
            }
            SMTExpr::UnOp { op, expr } => {
                write!(f, "({} {})", op, expr)
            }
        }
    }
}

/// SMT formula (boolean expression)
#[derive(Debug, Clone, PartialEq)]
pub enum SMTFormula {
    /// Boolean literal
    Bool(bool),

    /// Comparison: lhs op rhs
    Comparison {
        lhs: SMTExpr,
        op: ComparisonOp,
        rhs: SMTExpr,
    },

    /// Logical operation
    Logical {
        op: LogicalOp,
        operands: Vec<SMTFormula>,
    },

    /// Negation
    Not(Box<SMTFormula>),

    /// Quantified formula: forall/exists vars. body
    Quantified {
        quantifier: Quantifier,
        vars: Vec<(String, SMTSort)>,
        body: Box<SMTFormula>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Quantifier {
    Forall,
    Exists,
}

impl fmt::Display for SMTFormula {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SMTFormula::Bool(b) => write!(f, "{}", b),
            SMTFormula::Comparison { lhs, op, rhs } => {
                let smt_op = match op {
                    ComparisonOp::Lt => "<",
                    ComparisonOp::Le => "<=",
                    ComparisonOp::Gt => ">",
                    ComparisonOp::Ge => ">=",
                    ComparisonOp::Eq => "=",
                    ComparisonOp::Ne => "distinct",
                };
                write!(f, "({} {} {})", smt_op, lhs, rhs)
            }
            SMTFormula::Logical { op, operands } => {
                let smt_op = match op {
                    LogicalOp::And => "and",
                    LogicalOp::Or => "or",
                };
                write!(f, "({}", smt_op)?;
                for operand in operands {
                    write!(f, " {}", operand)?;
                }
                write!(f, ")")
            }
            SMTFormula::Not(inner) => {
                write!(f, "(not {})", inner)
            }
            SMTFormula::Quantified {
                quantifier,
                vars,
                body,
            } => {
                let q_str = match quantifier {
                    Quantifier::Forall => "forall",
                    Quantifier::Exists => "exists",
                };
                write!(f, "({} (", q_str)?;
                for (i, (var, sort)) in vars.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "({} {})", var, sort)?;
                }
                write!(f, ") {})", body)
            }
        }
    }
}

/// SMT translator for MedLang constraints
pub struct SMTTranslator;

impl SMTTranslator {
    /// Translate a MedLang constraint expression to SMT formula
    pub fn translate_constraint(constraint: &ConstraintExpr) -> Result<SMTFormula> {
        match constraint {
            ConstraintExpr::Comparison { var, op, value } => {
                let lhs = SMTExpr::Var(var.clone());
                let rhs = Self::translate_literal(value)?;
                Ok(SMTFormula::Comparison { lhs, op: *op, rhs })
            }

            ConstraintExpr::Range { var, lower, upper } => {
                // Translate: var in [lower, upper] to (and (>= var lower) (<= var upper))
                let var_expr = SMTExpr::Var(var.clone());
                let lower_expr = Self::translate_literal(lower)?;
                let upper_expr = Self::translate_literal(upper)?;

                let lower_bound = SMTFormula::Comparison {
                    lhs: var_expr.clone(),
                    op: ComparisonOp::Ge,
                    rhs: lower_expr,
                };

                let upper_bound = SMTFormula::Comparison {
                    lhs: var_expr,
                    op: ComparisonOp::Le,
                    rhs: upper_expr,
                };

                Ok(SMTFormula::Logical {
                    op: LogicalOp::And,
                    operands: vec![lower_bound, upper_bound],
                })
            }

            ConstraintExpr::Binary { left, op, right } => {
                let left_formula = Self::translate_constraint(left)?;
                let right_formula = Self::translate_constraint(right)?;

                Ok(SMTFormula::Logical {
                    op: *op,
                    operands: vec![left_formula, right_formula],
                })
            }
        }
    }

    /// Translate a constraint literal to SMT expression
    fn translate_literal(literal: &ConstraintLiteral) -> Result<SMTExpr> {
        match literal {
            ConstraintLiteral::Float(val) => Ok(SMTExpr::RealLit(*val)),
            ConstraintLiteral::Int(val) => Ok(SMTExpr::IntLit(*val)),
            ConstraintLiteral::UnitValue(val, _unit) => {
                // For now, just use the numeric value
                // TODO: Proper unit handling in SMT
                Ok(SMTExpr::RealLit(*val))
            }
        }
    }

    /// Infer SMT sort from constraint literal
    pub fn infer_sort(literal: &ConstraintLiteral) -> SMTSort {
        match literal {
            ConstraintLiteral::Float(_) => SMTSort::Real,
            ConstraintLiteral::Int(_) => SMTSort::Int,
            ConstraintLiteral::UnitValue(_, unit) => {
                // Custom sort for unit types
                SMTSort::Custom(format!("Unit_{}", unit))
            }
        }
    }

    /// Generate SMT-LIB declaration for a variable
    pub fn declare_var(name: &str, sort: &SMTSort) -> String {
        format!("(declare-const {} {})", name, sort)
    }

    /// Generate SMT-LIB assertion
    pub fn assert_formula(formula: &SMTFormula) -> String {
        format!("(assert {})", formula)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translate_simple_comparison() {
        let constraint = ConstraintExpr::Comparison {
            var: "CL".to_string(),
            op: ComparisonOp::Gt,
            value: ConstraintLiteral::Float(0.0),
        };

        let formula = SMTTranslator::translate_constraint(&constraint).unwrap();
        assert_eq!(formula.to_string(), "(> CL 0)");
    }

    #[test]
    fn test_translate_range() {
        let constraint = ConstraintExpr::Range {
            var: "WT".to_string(),
            lower: ConstraintLiteral::Float(30.0),
            upper: ConstraintLiteral::Float(200.0),
        };

        let formula = SMTTranslator::translate_constraint(&constraint).unwrap();
        let expected = "(and (>= WT 30) (<= WT 200))";
        assert_eq!(formula.to_string(), expected);
    }

    #[test]
    fn test_translate_binary() {
        let left = ConstraintExpr::Comparison {
            var: "CL".to_string(),
            op: ComparisonOp::Gt,
            value: ConstraintLiteral::Float(0.0),
        };

        let right = ConstraintExpr::Comparison {
            var: "V".to_string(),
            op: ComparisonOp::Gt,
            value: ConstraintLiteral::Float(0.0),
        };

        let constraint = ConstraintExpr::Binary {
            left: Box::new(left),
            op: LogicalOp::And,
            right: Box::new(right),
        };

        let formula = SMTTranslator::translate_constraint(&constraint).unwrap();
        let expected = "(and (> CL 0) (> V 0))";
        assert_eq!(formula.to_string(), expected);
    }

    #[test]
    fn test_infer_sort() {
        assert_eq!(
            SMTTranslator::infer_sort(&ConstraintLiteral::Float(1.0)),
            SMTSort::Real
        );
        assert_eq!(
            SMTTranslator::infer_sort(&ConstraintLiteral::Int(42)),
            SMTSort::Int
        );
    }

    #[test]
    fn test_declare_var() {
        let decl = SMTTranslator::declare_var("CL", &SMTSort::Real);
        assert_eq!(decl, "(declare-const CL Real)");
    }

    #[test]
    fn test_assert_formula() {
        let formula = SMTFormula::Comparison {
            lhs: SMTExpr::Var("x".to_string()),
            op: ComparisonOp::Gt,
            rhs: SMTExpr::RealLit(0.0),
        };
        let assertion = SMTTranslator::assert_formula(&formula);
        assert_eq!(assertion, "(assert (> x 0))");
    }
}
