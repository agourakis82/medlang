// Week 29: Runtime Values for L₀
//
// Defines runtime value representation and errors for L₀ execution.

use crate::ml::{BackendKind, SurrogateModelHandle};
use crate::rl::RLPolicyHandle; // Week 31-32: RL policy handles
use crate::types::core_lang::CoreType;
use std::collections::HashMap;

/// Runtime value representation for L₀
#[derive(Debug, Clone, PartialEq)]
pub enum RuntimeValue {
    // Primitives
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Unit,

    // Compound types
    Record(HashMap<String, RuntimeValue>),
    Function {
        name: String,
        // Function closures would be more complex in a full implementation
    },

    // Domain handles
    Model {
        name: String,
        handle: String, // Opaque handle to compiled model
    },
    Protocol {
        name: String,
        handle: String,
    },
    EvidenceProgram {
        name: String,
        handle: String,
    },
    Policy {
        name: String,
        handle: String,
    },

    // Week 29: ML/Surrogate types
    SurrogateModel(SurrogateModelHandle),
    BackendKind(BackendKind),

    // Week 31-32: RL types
    RLPolicy(RLPolicyHandle),

    // Result types
    EvidenceResult {
        posterior_samples: Vec<Vec<f64>>,
        diagnostics: HashMap<String, String>,
    },
    SimulationResult {
        trajectories: Vec<Vec<f64>>,
        summary: HashMap<String, f64>,
    },
    FitResult {
        parameters: HashMap<String, f64>,
        diagnostics: HashMap<String, String>,
    },
}

impl RuntimeValue {
    /// Get the runtime type of this value
    pub fn runtime_type(&self) -> String {
        match self {
            RuntimeValue::Int(_) => "Int".to_string(),
            RuntimeValue::Float(_) => "Float".to_string(),
            RuntimeValue::Bool(_) => "Bool".to_string(),
            RuntimeValue::String(_) => "String".to_string(),
            RuntimeValue::Unit => "Unit".to_string(),
            RuntimeValue::Record(_) => "Record".to_string(),
            RuntimeValue::Function { .. } => "Function".to_string(),
            RuntimeValue::Model { .. } => "Model".to_string(),
            RuntimeValue::Protocol { .. } => "Protocol".to_string(),
            RuntimeValue::EvidenceProgram { .. } => "EvidenceProgram".to_string(),
            RuntimeValue::Policy { .. } => "Policy".to_string(),
            RuntimeValue::SurrogateModel(_) => "SurrogateModel".to_string(),
            RuntimeValue::BackendKind(_) => "BackendKind".to_string(),
            RuntimeValue::RLPolicy(_) => "RLPolicy".to_string(),
            RuntimeValue::EvidenceResult { .. } => "EvidenceResult".to_string(),
            RuntimeValue::SimulationResult { .. } => "SimulationResult".to_string(),
            RuntimeValue::FitResult { .. } => "FitResult".to_string(),
        }
    }

    /// Check if this value matches the given type
    pub fn has_type(&self, ty: &CoreType) -> bool {
        match (self, ty) {
            (RuntimeValue::Int(_), CoreType::Int) => true,
            (RuntimeValue::Float(_), CoreType::Float) => true,
            (RuntimeValue::Bool(_), CoreType::Bool) => true,
            (RuntimeValue::String(_), CoreType::String) => true,
            (RuntimeValue::Unit, CoreType::Unit) => true,
            (RuntimeValue::Model { .. }, CoreType::Model) => true,
            (RuntimeValue::Protocol { .. }, CoreType::Protocol) => true,
            (RuntimeValue::EvidenceProgram { .. }, CoreType::EvidenceProgram) => true,
            (RuntimeValue::Policy { .. }, CoreType::Policy) => true,
            (RuntimeValue::SurrogateModel(_), CoreType::SurrogateModel) => true,
            (RuntimeValue::RLPolicy(_), CoreType::RLPolicy) => true,
            (RuntimeValue::EvidenceResult { .. }, CoreType::EvidenceResult) => true,
            (RuntimeValue::SimulationResult { .. }, CoreType::SimulationResult) => true,
            (RuntimeValue::FitResult { .. }, CoreType::FitResult) => true,
            // TODO: Add Record and Function type checking
            _ => false,
        }
    }

    /// Extract BackendKind from a value
    pub fn as_backend_kind(&self) -> Result<BackendKind, RuntimeError> {
        match self {
            RuntimeValue::BackendKind(bk) => Ok(*bk),
            RuntimeValue::String(s) => {
                BackendKind::from_variant_name(s).map_err(|e| RuntimeError::TypeError {
                    expected: "BackendKind".to_string(),
                    found: format!("String(\"{}\")", s),
                    message: e.to_string(),
                })
            }
            _ => Err(RuntimeError::TypeError {
                expected: "BackendKind".to_string(),
                found: self.runtime_type(),
                message: "Cannot convert to BackendKind".to_string(),
            }),
        }
    }

    /// Extract SurrogateModelHandle from a value
    pub fn as_surrogate_model(&self) -> Result<&SurrogateModelHandle, RuntimeError> {
        match self {
            RuntimeValue::SurrogateModel(handle) => Ok(handle),
            _ => Err(RuntimeError::TypeError {
                expected: "SurrogateModel".to_string(),
                found: self.runtime_type(),
                message: "Expected SurrogateModel value".to_string(),
            }),
        }
    }
}

/// Runtime errors for L₀ execution
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RuntimeError {
    #[error("type error: expected {expected}, found {found}: {message}")]
    TypeError {
        expected: String,
        found: String,
        message: String,
    },

    #[error("arity mismatch: function `{fn_name}` expects {expected} arguments, got {found}")]
    ArityMismatch {
        fn_name: String,
        expected: usize,
        found: usize,
    },

    #[error("unknown function: {0}")]
    UnknownFunction(String),

    #[error("unknown variable: {0}")]
    UnknownVariable(String),

    #[error("field not found: {0}")]
    FieldNotFound(String),

    #[error("surrogate error: {0}")]
    SurrogateError(String),

    #[error("backend error: {0}")]
    BackendError(String),

    #[error("evidence execution failed: {0}")]
    EvidenceError(String),

    #[error("io error: {0}")]
    IoError(String),

    #[error("RL error: {0}")]
    RLError(String),

    #[error("custom error: {0}")]
    Custom(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_value_types() {
        assert_eq!(RuntimeValue::Int(42).runtime_type(), "Int");
        assert_eq!(
            RuntimeValue::String("test".to_string()).runtime_type(),
            "String"
        );
        assert_eq!(
            RuntimeValue::SurrogateModel(SurrogateModelHandle::new()).runtime_type(),
            "SurrogateModel"
        );
    }

    #[test]
    fn test_type_checking() {
        let val = RuntimeValue::Int(42);
        assert!(val.has_type(&CoreType::Int));
        assert!(!val.has_type(&CoreType::String));
    }

    #[test]
    fn test_backend_kind_extraction() {
        let val = RuntimeValue::BackendKind(BackendKind::Mechanistic);
        assert_eq!(val.as_backend_kind().unwrap(), BackendKind::Mechanistic);

        let val_str = RuntimeValue::String("Surrogate".to_string());
        assert_eq!(val_str.as_backend_kind().unwrap(), BackendKind::Surrogate);
    }

    #[test]
    fn test_surrogate_model_extraction() {
        let handle = SurrogateModelHandle::new();
        let val = RuntimeValue::SurrogateModel(handle.clone());
        assert_eq!(val.as_surrogate_model().unwrap(), &handle);

        let wrong_val = RuntimeValue::Int(42);
        assert!(wrong_val.as_surrogate_model().is_err());
    }
}
