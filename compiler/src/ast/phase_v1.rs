//! Phase V1 AST Extensions
//!
//! This module contains AST node extensions for Phase V1 features:
//! - Effect annotations
//! - Epistemic types (Knowledge<T>)
//! - Refinement type constraints

use crate::effects::{Effect, EffectSet};
use crate::refinement::clinical::Constraint;
use serde::{Deserialize, Serialize};

// =============================================================================
// Effect Annotations
// =============================================================================

/// Effect annotation for declarations (e.g., `with Prob, IO`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EffectAnnotationAst {
    pub effects: Vec<Effect>,
}

impl EffectAnnotationAst {
    pub fn new(effects: Vec<Effect>) -> Self {
        Self { effects }
    }

    pub fn to_effect_set(&self) -> EffectSet {
        EffectSet::from_vec(self.effects.clone())
    }
}

// =============================================================================
// Epistemic Types
// =============================================================================

/// Epistemic type wrapper in AST (e.g., `Knowledge<Mass>`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicTypeAst {
    /// Inner type (e.g., "Mass", "f64")
    pub inner_type: String,

    /// Optional minimum confidence constraint
    pub min_confidence: Option<f64>,
}

/// Epistemic value initializer in AST
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EpistemicValueAst {
    pub value_expr: Box<super::Expr>,
    pub confidence: Option<f64>,
    pub provenance: Option<ProvenanceAst>,
}

/// Provenance specification in AST
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProvenanceAst {
    Measurement {
        source: String,
        timestamp: Option<String>,
        subject_id: Option<String>,
    },
    Computed {
        operation: String,
    },
    Imputed {
        method: String,
    },
    Estimated {
        model: String,
        method: String,
    },
    Literature {
        citation: String,
    },
    Synthetic {
        generator: String,
        seed: Option<u64>,
    },
}

// =============================================================================
// Refinement Type Constraints
// =============================================================================

/// Refinement constraint in AST (e.g., `where CL > 0.0`)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefinementConstraintAst {
    pub constraint: ConstraintExpr,
}

/// Constraint expression in AST
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintExpr {
    /// Comparison: var op literal
    Comparison {
        var: String,
        op: ComparisonOp,
        value: ConstraintLiteral,
    },

    /// Binary logical operation
    Binary {
        left: Box<ConstraintExpr>,
        op: LogicalOp,
        right: Box<ConstraintExpr>,
    },

    /// Range constraint: lower <= var <= upper
    Range {
        var: String,
        lower: ConstraintLiteral,
        upper: ConstraintLiteral,
    },
}

impl ConstraintExpr {
    /// Convert AST constraint to runtime Constraint
    pub fn to_constraint(&self) -> Constraint {
        match self {
            ConstraintExpr::Comparison { var, op, value } => Constraint::Comparison {
                var: var.clone(),
                op: op.to_runtime_op(),
                value: value.to_constraint_value(),
            },

            ConstraintExpr::Binary { left, op, right } => Constraint::Binary {
                left: Box::new(left.to_constraint()),
                op: op.to_runtime_op(),
                right: Box::new(right.to_constraint()),
            },

            ConstraintExpr::Range { var, lower, upper } => Constraint::Range {
                var: var.clone(),
                lower: lower.to_constraint_value(),
                upper: upper.to_constraint_value(),
                lower_inclusive: true,
                upper_inclusive: true,
            },
        }
    }
}

/// Comparison operators in constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOp {
    Eq, // ==
    Ne, // !=
    Lt, // <
    Le, // <=
    Gt, // >
    Ge, // >=
}

impl ComparisonOp {
    pub fn to_runtime_op(&self) -> crate::refinement::clinical::ComparisonOp {
        match self {
            ComparisonOp::Eq => crate::refinement::clinical::ComparisonOp::Eq,
            ComparisonOp::Ne => crate::refinement::clinical::ComparisonOp::Ne,
            ComparisonOp::Lt => crate::refinement::clinical::ComparisonOp::Lt,
            ComparisonOp::Le => crate::refinement::clinical::ComparisonOp::Le,
            ComparisonOp::Gt => crate::refinement::clinical::ComparisonOp::Gt,
            ComparisonOp::Ge => crate::refinement::clinical::ComparisonOp::Ge,
        }
    }
}

/// Logical operators in constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogicalOp {
    And, // &&
    Or,  // ||
}

impl LogicalOp {
    pub fn to_runtime_op(&self) -> crate::refinement::clinical::LogicalOp {
        match self {
            LogicalOp::And => crate::refinement::clinical::LogicalOp::And,
            LogicalOp::Or => crate::refinement::clinical::LogicalOp::Or,
        }
    }
}

/// Literal values in constraints
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ConstraintLiteral {
    Float(f64),
    Int(i64),
    UnitValue(f64, String), // Value with unit, e.g., 0.5_kg
}

impl ConstraintLiteral {
    pub fn to_constraint_value(&self) -> crate::refinement::clinical::ConstraintValue {
        match self {
            ConstraintLiteral::Float(f) => crate::refinement::clinical::ConstraintValue::Float(*f),
            ConstraintLiteral::Int(i) => crate::refinement::clinical::ConstraintValue::Int(*i),
            ConstraintLiteral::UnitValue(f, _unit) => {
                // For now, just use the numeric value
                // TODO: Unit conversion in Phase V2
                crate::refinement::clinical::ConstraintValue::Float(*f)
            }
        }
    }
}

// =============================================================================
// Extended Type Expression
// =============================================================================

/// Type expression with Phase V1 extensions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExprV1 {
    /// Simple type name (existing)
    Simple(String),

    /// Epistemic type: Knowledge<T>
    Epistemic(EpistemicTypeAst),

    /// Type with refinement constraint
    Refined {
        base_type: String,
        constraint: RefinementConstraintAst,
    },

    /// Epistemic type with refinement
    EpistemicRefined {
        inner_type: String,
        inner_constraint: Option<RefinementConstraintAst>,
        min_confidence: Option<f64>,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_annotation_creation() {
        let ann = EffectAnnotationAst::new(vec![Effect::Prob, Effect::IO]);
        assert_eq!(ann.effects.len(), 2);
        assert!(ann.effects.contains(&Effect::Prob));
        assert!(ann.effects.contains(&Effect::IO));
    }

    #[test]
    fn test_constraint_expr_comparison() {
        let expr = ConstraintExpr::Comparison {
            var: "CL".to_string(),
            op: ComparisonOp::Gt,
            value: ConstraintLiteral::Float(0.0),
        };

        let runtime = expr.to_constraint();
        // Verify it converts successfully
        assert!(matches!(runtime, Constraint::Comparison { .. }));
    }

    #[test]
    fn test_constraint_expr_range() {
        let expr = ConstraintExpr::Range {
            var: "AGE".to_string(),
            lower: ConstraintLiteral::Float(0.0),
            upper: ConstraintLiteral::Float(120.0),
        };

        let runtime = expr.to_constraint();
        assert!(matches!(runtime, Constraint::Range { .. }));
    }

    #[test]
    fn test_epistemic_type_ast() {
        let etype = EpistemicTypeAst {
            inner_type: "Mass".to_string(),
            min_confidence: Some(0.85),
        };

        assert_eq!(etype.inner_type, "Mass");
        assert_eq!(etype.min_confidence, Some(0.85));
    }
}
