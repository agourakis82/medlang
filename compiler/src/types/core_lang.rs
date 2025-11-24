// Week 26: Core Type System for L₀
//
// Defines the type representation and conversion from AST type annotations.

use crate::ast::core_lang::TypeAnn;
use std::collections::HashMap;

/// Core types for L₀ (the host coordination language)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CoreType {
    // Primitive types
    Int,
    Float,
    Bool,
    String,
    Unit,

    // Composite types
    Record(HashMap<String, CoreType>),

    Function {
        params: Vec<CoreType>,
        ret: Box<CoreType>,
    },

    // Enum types (Week 27) - discrete clinical states
    Enum(String), // Enum name: Response, ToxicityGrade, etc.

    // Domain types (L₁-L₃) - opaque at L₀ level but mapped to handles at runtime
    Model,
    Protocol,
    Policy,
    EvidenceProgram,

    // Result types from domain execution
    EvidenceResult,
    SimulationResult,
    FitResult,

    // Week 29: AI/ML types - first-class surrogates
    SurrogateModel, // Handle to a trained neural/surrogate model

    // Week 31-32: Reinforcement Learning types
    RLPolicy, // Handle to a trained RL policy (Q-table, neural policy, etc.)
}

impl CoreType {
    pub fn as_str(&self) -> String {
        match self {
            CoreType::Int => "Int".to_string(),
            CoreType::Float => "Float".to_string(),
            CoreType::Bool => "Bool".to_string(),
            CoreType::String => "String".to_string(),
            CoreType::Unit => "Unit".to_string(),
            CoreType::Record(fields) => {
                let field_strs: Vec<String> = fields
                    .iter()
                    .map(|(k, v)| format!("{}: {}", k, v.as_str()))
                    .collect();
                format!("{{ {} }}", field_strs.join(", "))
            }
            CoreType::Function { params, ret } => {
                let param_strs: Vec<String> = params.iter().map(|p| p.as_str()).collect();
                format!("({}) -> {}", param_strs.join(", "), ret.as_str())
            }
            CoreType::Enum(name) => name.clone(),
            CoreType::Model => "Model".to_string(),
            CoreType::Protocol => "Protocol".to_string(),
            CoreType::Policy => "Policy".to_string(),
            CoreType::EvidenceProgram => "EvidenceProgram".to_string(),
            CoreType::EvidenceResult => "EvidenceResult".to_string(),
            CoreType::SimulationResult => "SimulationResult".to_string(),
            CoreType::FitResult => "FitResult".to_string(),
            CoreType::SurrogateModel => "SurrogateModel".to_string(),
            CoreType::RLPolicy => "RLPolicy".to_string(),
        }
    }

    /// Check if this is a domain type (Model, Protocol, etc.)
    pub fn is_domain_type(&self) -> bool {
        matches!(
            self,
            CoreType::Model
                | CoreType::Protocol
                | CoreType::Policy
                | CoreType::EvidenceProgram
                | CoreType::EvidenceResult
                | CoreType::SimulationResult
                | CoreType::FitResult
                | CoreType::SurrogateModel
                | CoreType::RLPolicy
        )
    }
}

/// Typed function signature
#[derive(Debug, Clone, PartialEq)]
pub struct TypedFnSig {
    pub params: Vec<CoreType>,
    pub ret: CoreType,
}

impl TypedFnSig {
    pub fn new(params: Vec<CoreType>, ret: CoreType) -> Self {
        Self { params, ret }
    }
}

/// Convert AST type annotation to CoreType
pub fn resolve_type_ann(ann: &TypeAnn) -> CoreType {
    match ann {
        TypeAnn::Int => CoreType::Int,
        TypeAnn::Float => CoreType::Float,
        TypeAnn::Bool => CoreType::Bool,
        TypeAnn::String => CoreType::String,
        TypeAnn::Unit => CoreType::Unit,
        TypeAnn::Record(fields) => {
            let mut map = HashMap::new();
            for (id, t) in fields {
                map.insert(id.clone(), resolve_type_ann(t));
            }
            CoreType::Record(map)
        }
        TypeAnn::FnType { params, ret } => CoreType::Function {
            params: params.iter().map(resolve_type_ann).collect(),
            ret: Box::new(resolve_type_ann(ret)),
        },
        TypeAnn::Model => CoreType::Model,
        TypeAnn::Protocol => CoreType::Protocol,
        TypeAnn::Policy => CoreType::Policy,
        TypeAnn::EvidenceProgram => CoreType::EvidenceProgram,
        TypeAnn::EvidenceResult => CoreType::EvidenceResult,
        TypeAnn::SimulationResult => CoreType::SimulationResult,
        TypeAnn::FitResult => CoreType::FitResult,
        TypeAnn::SurrogateModel => CoreType::SurrogateModel,
        TypeAnn::RLPolicy => CoreType::RLPolicy,
    }
}

// =============================================================================
// Week 30: Surrogate Evaluation Record Type Helpers
// =============================================================================

/// Build the type for SurrogateEvalConfig record
pub fn build_surrogate_eval_cfg_type() -> CoreType {
    let mut fields = HashMap::new();
    fields.insert("n_eval".to_string(), CoreType::Int);
    fields.insert(
        "backend_ref".to_string(),
        CoreType::Enum("BackendKind".to_string()),
    );
    fields.insert("seed".to_string(), CoreType::Int);
    CoreType::Record(fields)
}

/// Build the type for SurrogateEvalReport record
pub fn build_surrogate_eval_report_type() -> CoreType {
    let mut fields = HashMap::new();
    fields.insert("n_eval".to_string(), CoreType::Int);
    fields.insert("rmse".to_string(), CoreType::Float);
    fields.insert("mae".to_string(), CoreType::Float);
    fields.insert("max_abs_err".to_string(), CoreType::Float);
    fields.insert("mech_contract_violations".to_string(), CoreType::Int);
    fields.insert("surr_contract_violations".to_string(), CoreType::Int);
    CoreType::Record(fields)
}

/// Build the type for SurrogateThresholds record
pub fn build_surrogate_thresholds_type() -> CoreType {
    let mut fields = HashMap::new();
    fields.insert("max_rmse".to_string(), CoreType::Float);
    fields.insert("max_mae".to_string(), CoreType::Float);
    fields.insert("max_abs_err".to_string(), CoreType::Float);
    CoreType::Record(fields)
}

// =============================================================================
// Week 31-32: RL Record Type Helpers
// =============================================================================

/// Build the type for RLEnvConfig record
pub fn build_rl_env_config_type() -> CoreType {
    let mut fields = HashMap::new();
    fields.insert("evidence_program".to_string(), CoreType::EvidenceProgram);
    fields.insert(
        "backend".to_string(),
        CoreType::Enum("BackendKind".to_string()),
    );
    fields.insert("n_cycles".to_string(), CoreType::Int);
    // dose_levels: Vector<Float> - represented as Record for now
    fields.insert("w_response".to_string(), CoreType::Float);
    fields.insert("w_tox".to_string(), CoreType::Float);
    fields.insert("contract_penalty".to_string(), CoreType::Float);
    CoreType::Record(fields)
}

/// Build the type for RLTrainConfig record
pub fn build_rl_train_config_type() -> CoreType {
    let mut fields = HashMap::new();
    fields.insert("n_episodes".to_string(), CoreType::Int);
    fields.insert("max_steps_per_episode".to_string(), CoreType::Int);
    fields.insert("gamma".to_string(), CoreType::Float);
    fields.insert("alpha".to_string(), CoreType::Float);
    fields.insert("eps_start".to_string(), CoreType::Float);
    fields.insert("eps_end".to_string(), CoreType::Float);
    CoreType::Record(fields)
}

/// Build the type for RLTrainReport record
pub fn build_rl_train_report_type() -> CoreType {
    let mut fields = HashMap::new();
    fields.insert("n_episodes".to_string(), CoreType::Int);
    fields.insert("avg_reward".to_string(), CoreType::Float);
    fields.insert("final_epsilon".to_string(), CoreType::Float);
    fields.insert("avg_episode_length".to_string(), CoreType::Float);
    fields.insert("total_steps".to_string(), CoreType::Int);
    CoreType::Record(fields)
}

/// Build the type for PolicyEvalReport record
pub fn build_policy_eval_report_type() -> CoreType {
    let mut fields = HashMap::new();
    fields.insert("n_episodes".to_string(), CoreType::Int);
    fields.insert("avg_reward".to_string(), CoreType::Float);
    fields.insert("avg_contract_violations".to_string(), CoreType::Float);
    fields.insert("avg_episode_length".to_string(), CoreType::Float);
    CoreType::Record(fields)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_type_display() {
        assert_eq!(CoreType::Int.as_str(), "Int");
        assert_eq!(CoreType::EvidenceProgram.as_str(), "EvidenceProgram");

        let fn_type = CoreType::Function {
            params: vec![CoreType::EvidenceProgram, CoreType::String],
            ret: Box::new(CoreType::EvidenceResult),
        };
        assert_eq!(
            fn_type.as_str(),
            "(EvidenceProgram, String) -> EvidenceResult"
        );
    }

    #[test]
    fn test_is_domain_type() {
        assert!(CoreType::Model.is_domain_type());
        assert!(CoreType::EvidenceProgram.is_domain_type());
        assert!(CoreType::EvidenceResult.is_domain_type());
        assert!(!CoreType::Int.is_domain_type());
        assert!(!CoreType::String.is_domain_type());
    }

    #[test]
    fn test_resolve_type_ann() {
        let ann = TypeAnn::EvidenceProgram;
        let ty = resolve_type_ann(&ann);
        assert_eq!(ty, CoreType::EvidenceProgram);

        let fn_ann = TypeAnn::FnType {
            params: vec![TypeAnn::Int, TypeAnn::String],
            ret: Box::new(TypeAnn::Bool),
        };
        let fn_ty = resolve_type_ann(&fn_ann);
        assert_eq!(
            fn_ty,
            CoreType::Function {
                params: vec![CoreType::Int, CoreType::String],
                ret: Box::new(CoreType::Bool),
            }
        );
    }

    #[test]
    fn test_record_type_conversion() {
        let ann = TypeAnn::Record(vec![
            ("name".to_string(), TypeAnn::String),
            ("age".to_string(), TypeAnn::Int),
        ]);
        let ty = resolve_type_ann(&ann);

        match ty {
            CoreType::Record(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields.get("name"), Some(&CoreType::String));
                assert_eq!(fields.get("age"), Some(&CoreType::Int));
            }
            _ => panic!("Expected Record type"),
        }
    }

    #[test]
    fn test_typed_fn_sig() {
        let sig = TypedFnSig::new(
            vec![CoreType::EvidenceProgram, CoreType::String],
            CoreType::EvidenceResult,
        );

        assert_eq!(sig.params.len(), 2);
        assert_eq!(sig.params[0], CoreType::EvidenceProgram);
        assert_eq!(sig.ret, CoreType::EvidenceResult);
    }
}
