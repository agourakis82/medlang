//! Abstract Syntax Tree (AST) definitions for MedLang.
//!
//! This module defines the AST node types for the MedLang V0 subset.
//! Each node corresponds to a grammar construct from medlang_d_minimal_grammar_v0.md

use serde::{Deserialize, Serialize};

/// Type alias for identifiers
pub type Ident = String;

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

pub mod evidence;
pub mod module;

pub use evidence::{
    DesignDecl, DesignGridSpec, EvidenceBody, EvidenceProgram, HierarchyDecl, HierarchyKind,
    MapPriorDecl, TrialDecl,
};

pub use module::{ExportDecl, ExportKind, ImportDecl, ImportItems, ModuleDecl, ModulePath};

/// Top-level declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Declaration {
    Model(ModelDef),
    Population(PopulationDef),
    Measure(MeasureDef),
    Timeline(TimelineDef),
    Cohort(CohortDef),
    Protocol(ProtocolDef),     // Week 8: L₂ Clinical Trial DSL
    Evidence(EvidenceProgram), // Week 24: L₃ Evidence Programs
}

impl Declaration {
    /// Get the name of this declaration
    pub fn name(&self) -> &str {
        match self {
            Declaration::Model(m) => &m.name,
            Declaration::Population(p) => &p.name,
            Declaration::Measure(m) => &m.name,
            Declaration::Timeline(t) => &t.name,
            Declaration::Cohort(c) => &c.name,
            Declaration::Protocol(p) => &p.name,
            Declaration::Evidence(e) => e.name.as_str(),
        }
    }
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
    Input(InputDecl), // Input from another model (e.g., input C_plasma : ConcMass)
    ODE(ODEEquation),
    Observable(ObservableDecl),
    Let(LetBinding), // Let bindings for intermediate values (e.g., let E_drug = ...)
    Submodel(SubmodelDecl), // Submodel declaration
    Connect(ConnectionDecl), // Connection between models
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LetBinding {
    pub name: String,
    pub ty: Option<TypeExpr>, // Optional type annotation
    pub expr: Expr,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SubmodelDecl {
    pub name: String,       // e.g., "PK"
    pub model_type: String, // e.g., "PK_OneCompOral"
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectionDecl {
    pub from_model: String, // e.g., "PK"
    pub from_field: String, // e.g., "C_plasma"
    pub to_model: String,   // e.g., "QSP"
    pub to_field: String,   // e.g., "C_plasma"
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
    TumourVolume, // For QSP models (mm³ or cm³)
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

// =============================================================================
// Protocol Definition (L₂ Clinical Trial DSL - Week 8)
// =============================================================================

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProtocolDef {
    pub name: String,
    pub population_model_name: String,
    pub arms: Vec<ArmDef>,
    pub visits: Vec<VisitDef>,
    pub inclusion: Option<InclusionDef>,
    pub endpoints: Vec<EndpointDef>,
    pub decisions: Vec<DecisionDef>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArmDef {
    pub name: String,
    pub label: String,
    pub dose_mg: f64,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisitDef {
    pub name: String,
    pub time_days: f64,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InclusionDef {
    pub clauses: Vec<InclusionClause>,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InclusionClause {
    AgeBetween { min_years: u32, max_years: u32 },
    ECOGIn { allowed: Vec<u8> },
    BaselineTumourGe { volume_cm3: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EndpointDef {
    pub name: String,
    pub kind: EndpointKind,
    pub spec: EndpointSpec,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EndpointKind {
    Binary,
    TimeToEvent,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EndpointSpec {
    ResponseRate {
        observable: String,
        shrink_fraction: f64,
        window_start_days: f64,
        window_end_days: f64,
    },
    TimeToProgression {
        observable: String,
        increase_fraction: f64,
        window_start_days: f64,
        window_end_days: f64,
        ref_baseline: bool, // if true, use baseline; if false, use nadir (best response)
    },
}

/// Decision rule for trial design evaluation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DecisionDef {
    pub name: String,
    pub endpoint_name: String,
    pub arm_left: String,
    pub arm_right: String,
    pub margin: f64,
    pub prob_threshold: f64,
    pub direction: DecisionDirection,
    pub span: Option<Span>,
}

/// Direction of comparison for decision rule
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecisionDirection {
    LeftBetter,  // arm_left > arm_right by margin
    RightBetter, // arm_right > arm_left by margin
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
