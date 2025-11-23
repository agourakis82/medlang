//! Abstract Syntax Tree (AST) definitions for MedLang.
//!
//! This module defines the AST node types for the MedLang V0 subset.
//! Each node corresponds to a grammar construct from medlang_d_minimal_grammar_v0.md

use serde::{Deserialize, Serialize};

/// Source location for error reporting
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub line: usize,
    pub column: usize,
    pub length: usize,
}

impl Span {
    pub fn new(line: usize, column: usize, length: usize) -> Self {
        Self {
            line,
            column,
            length,
        }
    }
}

/// Top-level program: collection of declarations
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Program {
    pub declarations: Vec<Declaration>,
}

/// Top-level declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Declaration {
    Model(ModelDef),
    Population(PopulationDef),
    Measure(MeasureDef),
    Timeline(TimelineDef),
    Cohort(CohortDef),
}

// =============================================================================
// Model Definition
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelDef {
    pub name: String,
    pub items: Vec<ModelItem>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelItem {
    State(StateDecl),
    Param(ParamDecl),
    ODE(ODEEquation),
    Observable(ObservableDecl),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StateDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParamDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ODEEquation {
    pub state_name: String, // e.g., "A_gut" in "dA_gut/dt"
    pub rhs: Expr,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObservableDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub expr: Expr,
    pub span: Option<Span>,
}

// =============================================================================
// Population Definition
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PopulationDef {
    pub name: String,
    pub items: Vec<PopulationItem>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PopulationItem {
    ModelRef(String),               // Reference to model
    Param(ParamDecl),               // Population parameter
    Input(InputDecl),               // Covariate
    RandomEffect(RandomEffectDecl), // IIV
    BindParams(BindParamsBlock),    // Individual parameter mapping
    UseMeasure(UseMeasureStmt),     // Error model binding
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RandomEffectDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub dist: DistributionExpr,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BindParamsBlock {
    pub param_name: String, // e.g., "patient"
    pub statements: Vec<Statement>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UseMeasureStmt {
    pub measure_name: String,
    pub observable_name: QualifiedName,
    pub span: Option<Span>,
}

// =============================================================================
// Measure Definition
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeasureDef {
    pub name: String,
    pub items: Vec<MeasureItem>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MeasureItem {
    Pred(TypeExpr),
    Obs(TypeExpr),
    Param(ParamDecl),
    LogLikelihood(Expr),
}

// =============================================================================
// Timeline Definition
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimelineDef {
    pub name: String,
    pub events: Vec<Event>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Event {
    Dose(DoseEvent),
    Observe(ObserveEvent),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DoseEvent {
    pub time: Expr,
    pub amount: Expr,
    pub target: QualifiedName,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObserveEvent {
    pub time: Expr,
    pub target: QualifiedName,
    pub span: Option<Span>,
}

// =============================================================================
// Cohort Definition
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CohortDef {
    pub name: String,
    pub items: Vec<CohortItem>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CohortItem {
    Population(String),
    Timeline(String),
    DataFile(String),
}

// =============================================================================
// Type Expressions
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeExpr {
    /// Simple type (e.g., f64)
    Simple(String),

    /// Built-in unit type (e.g., Mass, Volume, Time)
    Unit(UnitType),

    /// Generic quantity type: Quantity<unit, scalar>
    Quantity(UnitExpr, Box<TypeExpr>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnitType {
    Mass,
    Volume,
    Time,
    DoseMass,
    ConcMass,
    Clearance,
    RateConst,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnitExpr {
    /// Named unit (e.g., kg, L, h)
    Named(String),

    /// Product of units
    Product(Box<UnitExpr>, Box<UnitExpr>),

    /// Quotient of units
    Quotient(Box<UnitExpr>, Box<UnitExpr>),
}

// =============================================================================
// Distribution Expressions
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DistributionExpr {
    Normal { mu: Expr, sigma: Expr },
    LogNormal { mu: Expr, sigma: Expr },
    Uniform { min: Expr, max: Expr },
}

// =============================================================================
// Expressions
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ExprKind {
    /// Literal value
    Literal(Literal),

    /// Variable reference
    Ident(String),

    /// Qualified name (e.g., model.CL, patient.WT)
    QualifiedName(QualifiedName),

    /// Unary operation
    Unary(UnaryOp, Box<Expr>),

    /// Binary operation
    Binary(BinaryOp, Box<Expr>, Box<Expr>),

    /// Function call
    Call(String, Vec<Argument>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Literal {
    /// Floating point number
    Float(f64),

    /// Floating point with unit (e.g., 100.0_mg)
    UnitFloat { value: f64, unit: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg, // -
    Pos, // +
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add, // +
    Sub, // -
    Mul, // *
    Div, // /
    Pow, // ^
    Lt,  // <
    Gt,  // >
    Eq,  // ==
    Ne,  // !=
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Argument {
    pub name: Option<String>, // None for positional, Some for named
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualifiedName {
    pub parts: Vec<String>,
}

impl QualifiedName {
    pub fn new(parts: Vec<String>) -> Self {
        Self { parts }
    }

    pub fn simple(name: String) -> Self {
        Self { parts: vec![name] }
    }
}

// =============================================================================
// Statements
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    /// Let binding: let x = expr
    Let {
        name: String,
        value: Expr,
        span: Option<Span>,
    },

    /// Assignment: qualified_name = expr
    Assign {
        target: QualifiedName,
        value: Expr,
        span: Option<Span>,
    },

    /// Expression statement
    Expr(Expr),
}

// =============================================================================
// Helper implementations
// =============================================================================

impl Expr {
    pub fn literal(value: f64) -> Self {
        Self {
            kind: ExprKind::Literal(Literal::Float(value)),
            span: None,
        }
    }

    pub fn unit_literal(value: f64, unit: String) -> Self {
        Self {
            kind: ExprKind::Literal(Literal::UnitFloat { value, unit }),
            span: None,
        }
    }

    pub fn ident(name: String) -> Self {
        Self {
            kind: ExprKind::Ident(name),
            span: None,
        }
    }

    pub fn qualified(parts: Vec<String>) -> Self {
        Self {
            kind: ExprKind::QualifiedName(QualifiedName::new(parts)),
            span: None,
        }
    }

    pub fn binary(op: BinaryOp, left: Expr, right: Expr) -> Self {
        Self {
            kind: ExprKind::Binary(op, Box::new(left), Box::new(right)),
            span: None,
        }
    }

    pub fn call(name: String, args: Vec<Argument>) -> Self {
        Self {
            kind: ExprKind::Call(name, args),
            span: None,
        }
    }
}

impl std::fmt::Display for QualifiedName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.parts.join("."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_constructors() {
        let e1 = Expr::literal(100.0);
        assert!(matches!(e1.kind, ExprKind::Literal(Literal::Float(100.0))));

        let e2 = Expr::unit_literal(100.0, "mg".to_string());
        match e2.kind {
            ExprKind::Literal(Literal::UnitFloat { value, unit }) => {
                assert_eq!(value, 100.0);
                assert_eq!(unit, "mg");
            }
            _ => panic!("Wrong expr kind"),
        }
    }

    #[test]
    fn test_qualified_name() {
        let qname = QualifiedName::new(vec!["model".to_string(), "CL".to_string()]);
        assert_eq!(qname.to_string(), "model.CL");
    }
}
