//! Intermediate Representation (IR) for MedLang
//!
//! The IR is a simplified, canonicalized form that's easier to translate to backend
//! languages (Stan, Julia). It resolves all names, flattens scopes, and makes
//! control flow and data dependencies explicit.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Complete IR program ready for code generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRProgram {
    /// The compiled model (structural + population)
    pub model: IRModel,

    /// Measurement/error models (one per observable)
    pub measures: Vec<IRMeasure>,

    /// Data specification
    pub data_spec: IRDataSpec,

    /// External scalar constants (e.g., from quantum stubs)
    #[serde(default)]
    pub externals: Vec<IRExternalScalar>,
}

/// IR for the complete model (structural + population)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRModel {
    pub name: String,

    /// State variables with their dimensions
    pub states: Vec<IRStateVar>,

    /// Parameters (both structural and population-level)
    pub params: Vec<IRParam>,

    /// Covariates (inputs from data)
    pub inputs: Vec<IRInput>,

    /// Random effects
    pub random_effects: Vec<IRRandomEffect>,

    /// Intermediate values (let bindings)
    pub intermediates: Vec<IRIntermediate>,

    /// ODE system
    pub odes: Vec<IRODEEquation>,

    /// Observable expressions
    pub observables: Vec<IRObservable>,

    /// Individual parameter transformations
    pub individual_params: Vec<IRIndividualParam>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRStateVar {
    pub name: String,
    pub dimension: String, // e.g., "Mass", "Volume"
    pub initial_value: Option<IRExpr>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRParam {
    pub name: String,
    pub dimension: String,
    pub kind: ParamKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamKind {
    /// Fixed parameter (structural model)
    Fixed,

    /// Population mean (to be estimated)
    PopulationMean,

    /// Population variance/SD parameter
    PopulationVariance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRInput {
    pub name: String,
    pub dimension: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRRandomEffect {
    pub name: String,
    pub distribution: IRDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRDistribution {
    Normal { mu: IRExpr, sigma: IRExpr },
    LogNormal { mu: IRExpr, sigma: IRExpr },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRIntermediate {
    pub name: String,
    pub dimension: Option<String>, // Optional dimension annotation
    pub expr: IRExpr,
}

/// External scalar constant (e.g., from quantum stub)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRExternalScalar {
    /// Name of the constant (e.g., "Kd_QM", "Kp_tumor_QM")
    pub name: String,

    /// Numerical value
    pub value: f64,

    /// Source/provenance (e.g., "qm_stub:LIG001:EGFR")
    pub source: String,

    /// Optional dimension for type checking
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimension: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRODEEquation {
    pub state_var: String,
    pub rhs: IRExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRObservable {
    pub name: String,
    pub dimension: String,
    pub expr: IRExpr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRIndividualParam {
    /// Target parameter in the model (e.g., "CL", "V", "Ka")
    pub param_name: String,

    /// Expression computing the individual value
    pub expr: IRExpr,
}

/// IR for measurement/error model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRMeasure {
    pub name: String,

    /// Observable being measured
    pub observable_ref: String,

    /// Error model parameters
    pub params: Vec<IRParam>,

    /// Log-likelihood expression
    pub log_likelihood: IRExpr,
}

/// Data specification for the model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRDataSpec {
    /// Number of subjects
    pub n_subjects: String, // Variable name in generated code

    /// Number of observations per subject
    pub n_obs: String,

    /// Column mappings
    pub columns: HashMap<String, String>,
}

/// Simplified expression tree
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IRExpr {
    /// Literal floating-point value
    Literal(f64),

    /// Variable reference (fully qualified)
    Var(String),

    /// Array indexing: arr[idx]
    Index(Box<IRExpr>, Box<IRExpr>),

    /// Unary operation
    Unary(IRUnaryOp, Box<IRExpr>),

    /// Binary operation
    Binary(IRBinaryOp, Box<IRExpr>, Box<IRExpr>),

    /// Function call
    Call(String, Vec<IRExpr>),
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IRUnaryOp {
    Neg,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum IRBinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
}

impl IRExpr {
    pub fn literal(val: f64) -> Self {
        IRExpr::Literal(val)
    }

    pub fn var(name: impl Into<String>) -> Self {
        IRExpr::Var(name.into())
    }

    pub fn binary(op: IRBinaryOp, left: IRExpr, right: IRExpr) -> Self {
        IRExpr::Binary(op, Box::new(left), Box::new(right))
    }

    pub fn call(name: impl Into<String>, args: Vec<IRExpr>) -> Self {
        IRExpr::Call(name.into(), args)
    }

    pub fn neg(expr: IRExpr) -> Self {
        IRExpr::Unary(IRUnaryOp::Neg, Box::new(expr))
    }
}

pub mod evidence;
pub mod module;
pub mod surrogate;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_expr_construction() {
        let expr = IRExpr::binary(IRBinaryOp::Mul, IRExpr::var("Ka"), IRExpr::var("A_gut"));

        match expr {
            IRExpr::Binary(IRBinaryOp::Mul, _, _) => {}
            _ => panic!("Expected binary multiplication"),
        }
    }

    #[test]
    fn test_ir_model_creation() {
        let model = IRModel {
            name: "TestModel".to_string(),
            states: vec![IRStateVar {
                name: "A".to_string(),
                dimension: "Mass".to_string(),
                initial_value: None,
            }],
            params: vec![IRParam {
                name: "K".to_string(),
                dimension: "RateConst".to_string(),
                kind: ParamKind::Fixed,
            }],
            inputs: vec![],
            random_effects: vec![],
            intermediates: vec![],
            odes: vec![IRODEEquation {
                state_var: "A".to_string(),
                rhs: IRExpr::binary(
                    IRBinaryOp::Mul,
                    IRExpr::neg(IRExpr::var("K")),
                    IRExpr::var("A"),
                ),
            }],
            observables: vec![],
            individual_params: vec![],
        };

        assert_eq!(model.name, "TestModel");
        assert_eq!(model.states.len(), 1);
        assert_eq!(model.odes.len(), 1);
    }

    #[test]
    fn test_ir_serialization() {
        let expr = IRExpr::literal(3.14);
        let json = serde_json::to_string(&expr).unwrap();
        let decoded: IRExpr = serde_json::from_str(&json).unwrap();

        match decoded {
            IRExpr::Literal(val) => assert!((val - 3.14).abs() < 1e-10),
            _ => panic!("Expected literal"),
        }
    }
}
