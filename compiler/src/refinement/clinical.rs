//! Refinement Types for MedLang
//!
//! This module implements refinement types with SMT-checkable constraints,
//! inspired by Demetrios. Refinement types add logical predicates to base types,
//! enabling compile-time verification of safety properties.
//!
//! ## Examples
//!
//! ```medlang
//! param CL : Clearance where CL > 0.0_L_per_h
//! param AGE : f64 where AGE >= 0.0 && AGE <= 120.0
//! obs C_plasma : ConcMass = A / V where V > 0.0_L
//! ```
//!
//! ## Safety Properties
//!
//! 1. **Physiological Bounds**: Age, weight, clearance must be positive
//! 2. **Division Safety**: Denominators proven non-zero
//! 3. **Domain Constraints**: Logarithm inputs must be positive
//! 4. **Clinical Safety**: Dose ranges, concentration limits
//!
//! ## SMT Integration (Future)
//!
//! Currently implements syntactic constraint checking.
//! Phase V2 will integrate Z3 SMT solver for proof checking.

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// =============================================================================
// Constraint Expressions
// =============================================================================

/// Refinement constraint (logical predicate)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Constraint {
    /// Boolean literal
    Bool(bool),

    /// Variable reference
    Var(String),

    /// Comparison: var op constant
    Comparison {
        var: String,
        op: ComparisonOp,
        value: ConstraintValue,
    },

    /// Binary logical operation
    Binary {
        left: Box<Constraint>,
        op: LogicalOp,
        right: Box<Constraint>,
    },

    /// Unary logical operation
    Unary {
        op: UnaryOp,
        inner: Box<Constraint>,
    },

    /// Range constraint: lower <= var <= upper
    Range {
        var: String,
        lower: ConstraintValue,
        upper: ConstraintValue,
        lower_inclusive: bool,
        upper_inclusive: bool,
    },
}

/// Comparison operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Eq,  // ==
    Ne,  // !=
    Lt,  // <
    Le,  // <=
    Gt,  // >
    Ge,  // >=
}

impl fmt::Display for ComparisonOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ComparisonOp::Eq => write!(f, "=="),
            ComparisonOp::Ne => write!(f, "!="),
            ComparisonOp::Lt => write!(f, "<"),
            ComparisonOp::Le => write!(f, "<="),
            ComparisonOp::Gt => write!(f, ">"),
            ComparisonOp::Ge => write!(f, ">="),
        }
    }
}

/// Logical operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicalOp {
    And, // &&
    Or,  // ||
}

impl fmt::Display for LogicalOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LogicalOp::And => write!(f, "&&"),
            LogicalOp::Or => write!(f, "||"),
        }
    }
}

/// Unary logical operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Not, // !
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOp::Not => write!(f, "!"),
        }
    }
}

/// Values in constraints (literals)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
}

impl fmt::Display for ConstraintValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConstraintValue::Float(v) => write!(f, "{}", v),
            ConstraintValue::Int(v) => write!(f, "{}", v),
            ConstraintValue::Bool(v) => write!(f, "{}", v),
            ConstraintValue::String(v) => write!(f, "\"{}\"", v),
        }
    }
}

impl fmt::Display for Constraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Constraint::Bool(b) => write!(f, "{}", b),
            Constraint::Var(v) => write!(f, "{}", v),
            Constraint::Comparison { var, op, value } => {
                write!(f, "{} {} {}", var, op, value)
            }
            Constraint::Binary { left, op, right } => {
                write!(f, "({} {} {})", left, op, right)
            }
            Constraint::Unary { op, inner } => {
                write!(f, "{}{}", op, inner)
            }
            Constraint::Range {
                var,
                lower,
                upper,
                lower_inclusive,
                upper_inclusive,
            } => {
                let left = if *lower_inclusive { "<=" } else { "<" };
                let right = if *upper_inclusive { "<=" } else { "<" };
                write!(f, "{} {} {} {} {}", lower, left, var, right, upper)
            }
        }
    }
}

// =============================================================================
// Refinement Type Definition
// =============================================================================

/// Refinement type: base type + constraint
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementType {
    /// Base type (e.g., "Clearance", "f64", "Mass")
    pub base_type: String,

    /// Refinement constraint (where clause)
    pub constraint: Option<Constraint>,
}

impl RefinementType {
    pub fn new(base_type: String) -> Self {
        Self {
            base_type,
            constraint: None,
        }
    }

    pub fn with_constraint(base_type: String, constraint: Constraint) -> Self {
        Self {
            base_type,
            constraint: Some(constraint),
        }
    }

    pub fn is_refined(&self) -> bool {
        self.constraint.is_some()
    }
}

impl fmt::Display for RefinementType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.constraint {
            Some(c) => write!(f, "{} where {}", self.base_type, c),
            None => write!(f, "{}", self.base_type),
        }
    }
}

// =============================================================================
// Common Refinement Type Constructors
// =============================================================================

impl Constraint {
    /// Create positive constraint: var > 0
    pub fn positive(var: String) -> Self {
        Constraint::Comparison {
            var,
            op: ComparisonOp::Gt,
            value: ConstraintValue::Float(0.0),
        }
    }

    /// Create non-negative constraint: var >= 0
    pub fn non_negative(var: String) -> Self {
        Constraint::Comparison {
            var,
            op: ComparisonOp::Ge,
            value: ConstraintValue::Float(0.0),
        }
    }

    /// Create non-zero constraint: var != 0
    pub fn non_zero(var: String) -> Self {
        Constraint::Comparison {
            var,
            op: ComparisonOp::Ne,
            value: ConstraintValue::Float(0.0),
        }
    }

    /// Create range constraint: lower <= var <= upper
    pub fn range_inclusive(var: String, lower: f64, upper: f64) -> Self {
        Constraint::Range {
            var,
            lower: ConstraintValue::Float(lower),
            upper: ConstraintValue::Float(upper),
            lower_inclusive: true,
            upper_inclusive: true,
        }
    }

    /// Combine two constraints with AND
    pub fn and(self, other: Constraint) -> Self {
        Constraint::Binary {
            left: Box::new(self),
            op: LogicalOp::And,
            right: Box::new(other),
        }
    }

    /// Combine two constraints with OR
    pub fn or(self, other: Constraint) -> Self {
        Constraint::Binary {
            left: Box::new(self),
            op: LogicalOp::Or,
            right: Box::new(other),
        }
    }
}

// =============================================================================
// Common Clinical Refinement Types
// =============================================================================

pub struct ClinicalRefinements;

impl ClinicalRefinements {
    /// Positive clearance: CL > 0
    pub fn positive_clearance(var_name: String) -> RefinementType {
        RefinementType::with_constraint("Clearance".to_string(), Constraint::positive(var_name))
    }

    /// Positive volume: V > 0
    pub fn positive_volume(var_name: String) -> RefinementType {
        RefinementType::with_constraint("Volume".to_string(), Constraint::positive(var_name))
    }

    /// Positive rate constant: K > 0
    pub fn positive_rate(var_name: String) -> RefinementType {
        RefinementType::with_constraint("RateConst".to_string(), Constraint::positive(var_name))
    }

    /// Age in typical human range: 0 <= age <= 120
    pub fn human_age(var_name: String) -> RefinementType {
        RefinementType::with_constraint(
            "f64".to_string(),
            Constraint::range_inclusive(var_name, 0.0, 120.0),
        )
    }

    /// Body weight in reasonable range: 0.5 kg (premature infant) to 300 kg
    pub fn body_weight(var_name: String) -> RefinementType {
        RefinementType::with_constraint(
            "Mass".to_string(),
            Constraint::range_inclusive(var_name, 0.5, 300.0),
        )
    }

    /// Proportion/probability: 0 <= p <= 1
    pub fn proportion(var_name: String) -> RefinementType {
        RefinementType::with_constraint(
            "f64".to_string(),
            Constraint::range_inclusive(var_name, 0.0, 1.0),
        )
    }

    /// Positive dose amount: dose > 0
    pub fn positive_dose(var_name: String) -> RefinementType {
        RefinementType::with_constraint("DoseMass".to_string(), Constraint::positive(var_name))
    }

    /// Creatinine clearance: 0 < CrCL <= 300 mL/min (typical renal function range)
    pub fn creatinine_clearance(var_name: String) -> RefinementType {
        let positive = Constraint::positive(var_name.clone());
        let upper_bound = Constraint::Comparison {
            var: var_name,
            op: ComparisonOp::Le,
            value: ConstraintValue::Float(300.0),
        };
        RefinementType::with_constraint(
            "Quantity<Volume/Time>".to_string(),
            positive.and(upper_bound),
        )
    }
}

// =============================================================================
// Constraint Checker (Syntactic)
// =============================================================================

/// Constraint checker (currently syntactic, SMT integration in Phase V2)
pub struct ConstraintChecker {
    /// Variable values for checking (runtime or symbolic)
    var_values: std::collections::HashMap<String, f64>,
}

impl ConstraintChecker {
    pub fn new() -> Self {
        Self {
            var_values: std::collections::HashMap::new(),
        }
    }

    /// Set a variable value for checking
    pub fn set_var(&mut self, name: String, value: f64) {
        self.var_values.insert(name, value);
    }

    /// Evaluate constraint (syntactic check)
    pub fn check(&self, constraint: &Constraint) -> Result<bool, RefinementError> {
        match constraint {
            Constraint::Bool(b) => Ok(*b),

            Constraint::Var(v) => {
                self.var_values
                    .get(v)
                    .map(|&val| val != 0.0)
                    .ok_or_else(|| RefinementError::UndefinedVariable {
                        var: v.clone(),
                    })
            }

            Constraint::Comparison { var, op, value } => {
                let var_val = self.var_values.get(var).ok_or_else(|| {
                    RefinementError::UndefinedVariable { var: var.clone() }
                })?;

                let const_val = match value {
                    ConstraintValue::Float(f) => *f,
                    ConstraintValue::Int(i) => *i as f64,
                    _ => {
                        return Err(RefinementError::TypeMismatch {
                            expected: "numeric".to_string(),
                            found: format!("{:?}", value),
                        })
                    }
                };

                let result = match op {
                    ComparisonOp::Eq => (var_val - const_val).abs() < 1e-10,
                    ComparisonOp::Ne => (var_val - const_val).abs() >= 1e-10,
                    ComparisonOp::Lt => var_val < &const_val,
                    ComparisonOp::Le => var_val <= &const_val,
                    ComparisonOp::Gt => var_val > &const_val,
                    ComparisonOp::Ge => var_val >= &const_val,
                };

                Ok(result)
            }

            Constraint::Binary { left, op, right } => {
                let left_val = self.check(left)?;
                let right_val = self.check(right)?;

                let result = match op {
                    LogicalOp::And => left_val && right_val,
                    LogicalOp::Or => left_val || right_val,
                };

                Ok(result)
            }

            Constraint::Unary { op, inner } => {
                let inner_val = self.check(inner)?;
                match op {
                    UnaryOp::Not => Ok(!inner_val),
                }
            }

            Constraint::Range {
                var,
                lower,
                upper,
                lower_inclusive,
                upper_inclusive,
            } => {
                let var_val = self.var_values.get(var).ok_or_else(|| {
                    RefinementError::UndefinedVariable { var: var.clone() }
                })?;

                let lower_val = match lower {
                    ConstraintValue::Float(f) => *f,
                    ConstraintValue::Int(i) => *i as f64,
                    _ => return Err(RefinementError::TypeMismatch {
                        expected: "numeric".to_string(),
                        found: format!("{:?}", lower),
                    }),
                };

                let upper_val = match upper {
                    ConstraintValue::Float(f) => *f,
                    ConstraintValue::Int(i) => *i as f64,
                    _ => return Err(RefinementError::TypeMismatch {
                        expected: "numeric".to_string(),
                        found: format!("{:?}", upper),
                    }),
                };

                let lower_ok = if *lower_inclusive {
                    var_val >= &lower_val
                } else {
                    var_val > &lower_val
                };

                let upper_ok = if *upper_inclusive {
                    var_val <= &upper_val
                } else {
                    var_val < &upper_val
                };

                Ok(lower_ok && upper_ok)
            }
        }
    }

    /// Verify refinement type
    pub fn verify_refinement(&self, rty: &RefinementType) -> Result<bool, RefinementError> {
        match &rty.constraint {
            Some(c) => self.check(c),
            None => Ok(true), // No constraint = always satisfied
        }
    }
}

impl Default for ConstraintChecker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Refinement Errors
// =============================================================================

#[derive(Debug, Error, Clone, PartialEq)]
pub enum RefinementError {
    #[error("Undefined variable in constraint: {var}")]
    UndefinedVariable { var: String },

    #[error("Type mismatch in constraint: expected {expected}, found {found}")]
    TypeMismatch { expected: String, found: String },

    #[error("Constraint violation: {constraint} is false for {context}")]
    ConstraintViolation {
        constraint: String,
        context: String,
    },

    #[error("SMT solver error: {message}")]
    SMTError { message: String },

    #[error("Invalid constraint: {reason}")]
    InvalidConstraint { reason: String },
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_positive_constraint() {
        let c = Constraint::positive("CL".to_string());
        assert_eq!(format!("{}", c), "CL > 0");
    }

    #[test]
    fn test_range_constraint() {
        let c = Constraint::range_inclusive("AGE".to_string(), 0.0, 120.0);
        assert_eq!(format!("{}", c), "0 <= AGE <= 120");
    }

    #[test]
    fn test_refinement_type_display() {
        let rty = ClinicalRefinements::positive_clearance("CL".to_string());
        assert_eq!(format!("{}", rty), "Clearance where CL > 0");
    }

    #[test]
    fn test_constraint_checker_simple() {
        let mut checker = ConstraintChecker::new();
        checker.set_var("CL".to_string(), 10.0);

        let constraint = Constraint::positive("CL".to_string());
        let result = checker.check(&constraint).unwrap();
        assert!(result);
    }

    #[test]
    fn test_constraint_checker_violation() {
        let mut checker = ConstraintChecker::new();
        checker.set_var("CL".to_string(), -5.0);

        let constraint = Constraint::positive("CL".to_string());
        let result = checker.check(&constraint).unwrap();
        assert!(!result);
    }

    #[test]
    fn test_constraint_checker_range() {
        let mut checker = ConstraintChecker::new();
        checker.set_var("AGE".to_string(), 35.0);

        let constraint = Constraint::range_inclusive("AGE".to_string(), 0.0, 120.0);
        let result = checker.check(&constraint).unwrap();
        assert!(result);
    }

    #[test]
    fn test_constraint_and() {
        let mut checker = ConstraintChecker::new();
        checker.set_var("X".to_string(), 5.0);

        let c1 = Constraint::positive("X".to_string());
        let c2 = Constraint::Comparison {
            var: "X".to_string(),
            op: ComparisonOp::Lt,
            value: ConstraintValue::Float(10.0),
        };
        let combined = c1.and(c2);

        let result = checker.check(&combined).unwrap();
        assert!(result);
    }

    #[test]
    fn test_clinical_refinements() {
        let mut checker = ConstraintChecker::new();

        checker.set_var("WT".to_string(), 70.0);
        let wt_type = ClinicalRefinements::body_weight("WT".to_string());
        assert!(checker.verify_refinement(&wt_type).unwrap());

        checker.set_var("AGE".to_string(), 35.0);
        let age_type = ClinicalRefinements::human_age("AGE".to_string());
        assert!(checker.verify_refinement(&age_type).unwrap());
    }
}
