// Week 26: Core Language (L₀) AST with Type Annotations
//
// This module defines the AST for MedLang's host coordination language (L₀).
// L₀ is a small, statically typed language for orchestrating the heavy L₁-L₃ DSLs.

use serde::{Deserialize, Serialize};

pub type Ident = String;

// =============================================================================
// Type Annotations
// =============================================================================

/// Type annotations for L₀
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TypeAnn {
    // Primitive types
    Int,
    Float,
    Bool,
    String,
    Unit,

    // Records: { field: Type, ... }
    Record(Vec<(Ident, TypeAnn)>),

    // Function types: (T1, T2, ...) -> Tr
    FnType {
        params: Vec<TypeAnn>,
        ret: Box<TypeAnn>,
    },

    // Domain types (L₁-L₃)
    Model,
    Protocol,
    Policy,
    EvidenceProgram,

    // Result types from domain execution
    EvidenceResult,
    SimulationResult,
    FitResult,

    // Week 29: AI/ML types
    SurrogateModel,

    // Week 31-32: Reinforcement Learning types
    RLPolicy, // Handle to a trained RL policy (Q-table, neural policy, etc.)
}

impl TypeAnn {
    pub fn as_str(&self) -> &'static str {
        match self {
            TypeAnn::Int => "Int",
            TypeAnn::Float => "Float",
            TypeAnn::Bool => "Bool",
            TypeAnn::String => "String",
            TypeAnn::Unit => "Unit",
            TypeAnn::Record(_) => "Record",
            TypeAnn::FnType { .. } => "Function",
            TypeAnn::Model => "Model",
            TypeAnn::Protocol => "Protocol",
            TypeAnn::Policy => "Policy",
            TypeAnn::EvidenceProgram => "EvidenceProgram",
            TypeAnn::EvidenceResult => "EvidenceResult",
            TypeAnn::SimulationResult => "SimulationResult",
            TypeAnn::FitResult => "FitResult",
            TypeAnn::SurrogateModel => "SurrogateModel",
            TypeAnn::RLPolicy => "RLPolicy",
        }
    }
}

// =============================================================================
// Function Definitions
// =============================================================================

/// Function parameter with type annotation
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Param {
    pub name: Ident,
    pub ty: Option<TypeAnn>, // Required for Week 26
}

/// Function definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FnDef {
    pub name: Ident,
    pub params: Vec<Param>,
    pub ret_type: Option<TypeAnn>, // Required for public functions
    pub contract: Option<crate::ast::FnContract>, // Week 28: function contracts
    pub body: Block,
}

impl FnDef {
    pub fn new(name: Ident, params: Vec<Param>, ret_type: Option<TypeAnn>, body: Block) -> Self {
        Self {
            name,
            params,
            ret_type,
            contract: None, // Week 28: no contract by default
            body,
        }
    }

    pub fn with_contract(mut self, contract: crate::ast::FnContract) -> Self {
        self.contract = Some(contract);
        self
    }
}

// =============================================================================
// Statements and Blocks
// =============================================================================

/// Block of statements
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Block {
    pub stmts: Vec<Stmt>,
}

impl Block {
    pub fn new(stmts: Vec<Stmt>) -> Self {
        Self { stmts }
    }
}

/// Statement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Stmt {
    /// Let binding: let x: Type = expr;
    Let(LetDecl),

    /// Assert statement: assert condition, "message"; (Week 28)
    Assert(AssertStmt),

    /// Expression statement
    Expr(Expr),
}

/// Let declaration
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LetDecl {
    pub name: Ident,
    pub ty: Option<TypeAnn>, // Optional type annotation
    pub expr: Expr,
}

/// Assert statement for L₀
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssertStmt {
    /// The boolean condition to check
    pub condition: Expr,

    /// Optional failure message
    pub message: Option<String>,

    /// Source location
    pub span: Option<crate::ast::Span>,
}

// =============================================================================
// Expressions
// =============================================================================

/// Expression
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Expr {
    /// Integer literal
    IntLiteral(i64),

    /// Float literal
    FloatLiteral(f64),

    /// Boolean literal
    BoolLiteral(bool),

    /// String literal
    StringLiteral(String),

    /// Variable reference
    Var(Ident),

    /// Record literal: { field: expr, ... }
    Record(Vec<(Ident, Expr)>),

    /// Field access: expr.field
    FieldAccess { target: Box<Expr>, field: Ident },

    /// Function call: callee(args)
    Call { callee: Box<Expr>, args: Vec<Expr> },

    /// If expression: if cond { then_branch } else { else_branch }
    If {
        cond: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },

    /// Block expression: { stmts }
    BlockExpr(Block),

    /// Enum variant constructor: Enum::Variant
    EnumVariant(Ident, Ident),

    /// Match expression
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
}

/// Match arm: pattern => expr
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchArm {
    pub pattern: MatchPattern,
    pub body: Expr,
}

impl MatchArm {
    pub fn new(pattern: MatchPattern, body: Expr) -> Self {
        Self { pattern, body }
    }
}

/// Pattern for matching
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MatchPattern {
    /// Enum variant pattern: Enum::Variant
    Variant {
        enum_name: Ident,
        variant_name: Ident,
    },
    /// Wildcard pattern: _
    Wildcard,
}

impl MatchPattern {
    pub fn variant(enum_name: Ident, variant_name: Ident) -> Self {
        Self::Variant { enum_name, variant_name }
    }
    pub fn wildcard() -> Self {
        Self::Wildcard
    }
}

impl Expr {
    pub fn int(value: i64) -> Self {
        Expr::IntLiteral(value)
    }

    pub fn float(value: f64) -> Self {
        Expr::FloatLiteral(value)
    }

    pub fn bool_val(value: bool) -> Self {
        Expr::BoolLiteral(value)
    }

    pub fn string(value: String) -> Self {
        Expr::StringLiteral(value)
    }

    pub fn var(name: Ident) -> Self {
        Expr::Var(name)
    }

    pub fn call(callee: Expr, args: Vec<Expr>) -> Self {
        Expr::Call {
            callee: Box::new(callee),
            args,
        }
    }

    pub fn if_expr(cond: Expr, then_branch: Expr, else_branch: Expr) -> Self {
        Expr::If {
            cond: Box::new(cond),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_ann_display() {
        assert_eq!(TypeAnn::Int.as_str(), "Int");
        assert_eq!(TypeAnn::EvidenceProgram.as_str(), "EvidenceProgram");
        assert_eq!(TypeAnn::Model.as_str(), "Model");
    }

    #[test]
    fn test_fn_def_creation() {
        let fn_def = FnDef::new(
            "test".to_string(),
            vec![Param {
                name: "x".to_string(),
                ty: Some(TypeAnn::Int),
            }],
            Some(TypeAnn::Int),
            Block::new(vec![Stmt::Expr(Expr::var("x".to_string()))]),
        );

        assert_eq!(fn_def.name, "test");
        assert_eq!(fn_def.params.len(), 1);
        assert_eq!(fn_def.ret_type, Some(TypeAnn::Int));
    }

    #[test]
    fn test_expr_constructors() {
        let e1 = Expr::int(42);
        assert_eq!(e1, Expr::IntLiteral(42));

        let e2 = Expr::string("hello".to_string());
        assert_eq!(e2, Expr::StringLiteral("hello".to_string()));

        let e3 = Expr::var("x".to_string());
        assert_eq!(e3, Expr::Var("x".to_string()));
    }

    #[test]
    fn test_call_expr() {
        let call = Expr::call(
            Expr::var("run_evidence".to_string()),
            vec![
                Expr::var("OncologyEvidence".to_string()),
                Expr::string("surrogate".to_string()),
            ],
        );

        match call {
            Expr::Call { callee, args } => {
                assert_eq!(*callee, Expr::Var("run_evidence".to_string()));
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call expression"),
        }
    }

    #[test]
    fn test_if_expr() {
        let if_expr = Expr::if_expr(Expr::bool_val(true), Expr::int(1), Expr::int(2));

        match if_expr {
            Expr::If {
                cond,
                then_branch,
                else_branch,
            } => {
                assert_eq!(*cond, Expr::BoolLiteral(true));
                assert_eq!(*then_branch, Expr::IntLiteral(1));
                assert_eq!(*else_branch, Expr::IntLiteral(2));
            }
            _ => panic!("Expected If expression"),
        }
    }

    #[test]
    fn test_let_decl() {
        let let_decl = LetDecl {
            name: "ev".to_string(),
            ty: Some(TypeAnn::EvidenceProgram),
            expr: Expr::var("OncologyEvidence".to_string()),
        };

        assert_eq!(let_decl.name, "ev");
        assert_eq!(let_decl.ty, Some(TypeAnn::EvidenceProgram));
    }

    #[test]
    fn test_record_expr() {
        let record = Expr::Record(vec![
            ("name".to_string(), Expr::string("test".to_string())),
            ("value".to_string(), Expr::int(42)),
        ]);

        match record {
            Expr::Record(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].0, "name");
                assert_eq!(fields[1].0, "value");
            }
            _ => panic!("Expected Record expression"),
        }
    }
}
