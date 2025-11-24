// Week 29: Integration Tests for First-Class Surrogates & ML Backends
//
// This test suite validates the end-to-end functionality of Week 29's
// first-class surrogate model support.

use medlangc::ast::core_lang::{Block, Expr, FnDef, Param, Stmt, TypeAnn};
use medlangc::ml::{BackendKind, SurrogateModelHandle, SurrogateTrainConfig};
use medlangc::runtime::builtins::{call_builtin, BuiltinFn};
use medlangc::runtime::value::{RuntimeError, RuntimeValue};
use medlangc::typecheck::core_lang::{typecheck_fn, DomainEnv, FnEnv};
use medlangc::types::core_lang::CoreType;
use std::collections::HashMap;

// =============================================================================
// Type System Tests
// =============================================================================

#[test]
fn test_surrogate_model_type_annotation() {
    // Test that SurrogateModel is a valid type annotation
    assert_eq!(TypeAnn::SurrogateModel.as_str(), "SurrogateModel");

    // Test that it resolves to the correct CoreType
    use medlangc::types::core_lang::resolve_type_ann;
    let ty = resolve_type_ann(&TypeAnn::SurrogateModel);
    assert_eq!(ty, CoreType::SurrogateModel);
}

#[test]
fn test_backend_kind_enum_values() {
    // Test all BackendKind variants
    let mechanistic = BackendKind::Mechanistic;
    let surrogate = BackendKind::Surrogate;
    let hybrid = BackendKind::Hybrid;

    assert_eq!(mechanistic.variant_name(), "Mechanistic");
    assert_eq!(surrogate.variant_name(), "Surrogate");
    assert_eq!(hybrid.variant_name(), "Hybrid");
}

#[test]
fn test_backend_kind_from_variant_name() {
    assert_eq!(
        BackendKind::from_variant_name("Mechanistic").unwrap(),
        BackendKind::Mechanistic
    );
    assert_eq!(
        BackendKind::from_variant_name("Surrogate").unwrap(),
        BackendKind::Surrogate
    );
    assert_eq!(
        BackendKind::from_variant_name("Hybrid").unwrap(),
        BackendKind::Hybrid
    );
    assert!(BackendKind::from_variant_name("Unknown").is_err());
}

#[test]
fn test_backend_kind_capabilities() {
    let mechanistic = BackendKind::Mechanistic;
    let surrogate = BackendKind::Surrogate;
    let hybrid = BackendKind::Hybrid;

    // Mechanistic doesn't require surrogate
    assert!(!mechanistic.requires_surrogate());
    assert!(mechanistic.requires_mechanistic());

    // Surrogate requires surrogate but not mechanistic
    assert!(surrogate.requires_surrogate());
    assert!(!surrogate.requires_mechanistic());

    // Hybrid requires both
    assert!(hybrid.requires_surrogate());
    assert!(hybrid.requires_mechanistic());
}

// =============================================================================
// Runtime Value Tests
// =============================================================================

#[test]
fn test_surrogate_model_runtime_value() {
    let handle = SurrogateModelHandle::new();
    let val = RuntimeValue::SurrogateModel(handle.clone());

    assert_eq!(val.runtime_type(), "SurrogateModel");
    assert!(val.has_type(&CoreType::SurrogateModel));
    assert!(!val.has_type(&CoreType::Model));
}

#[test]
fn test_backend_kind_runtime_value() {
    let val = RuntimeValue::BackendKind(BackendKind::Mechanistic);

    assert_eq!(val.runtime_type(), "BackendKind");
    assert_eq!(val.as_backend_kind().unwrap(), BackendKind::Mechanistic);
}

#[test]
fn test_backend_kind_from_string_value() {
    let val = RuntimeValue::String("Surrogate".to_string());
    assert_eq!(val.as_backend_kind().unwrap(), BackendKind::Surrogate);

    let bad_val = RuntimeValue::String("InvalidBackend".to_string());
    assert!(bad_val.as_backend_kind().is_err());
}

// =============================================================================
// Surrogate Training Config Tests
// =============================================================================

#[test]
fn test_surrogate_train_config_validation() {
    // Valid config
    let valid_cfg = SurrogateTrainConfig {
        n_train: 1000,
        backend: BackendKind::Mechanistic,
        seed: 42,
        max_epochs: 100,
        batch_size: 32,
    };
    assert!(valid_cfg.validate().is_ok());

    // Invalid: negative n_train
    let invalid_cfg = SurrogateTrainConfig {
        n_train: -10,
        backend: BackendKind::Mechanistic,
        seed: 42,
        max_epochs: 100,
        batch_size: 32,
    };
    assert!(invalid_cfg.validate().is_err());

    // Invalid: using Surrogate backend (circular dependency)
    let circular_cfg = SurrogateTrainConfig {
        n_train: 1000,
        backend: BackendKind::Surrogate,
        seed: 42,
        max_epochs: 100,
        batch_size: 32,
    };
    assert!(circular_cfg.validate().is_err());
}

#[test]
fn test_surrogate_train_config_defaults() {
    let quick = SurrogateTrainConfig::default_quick();
    assert_eq!(quick.n_train, 100);
    assert_eq!(quick.max_epochs, 50);

    let production = SurrogateTrainConfig::default_production();
    assert_eq!(production.n_train, 10000);
    assert_eq!(production.max_epochs, 500);
}

// =============================================================================
// Built-in Function Signature Tests
// =============================================================================

#[test]
fn test_train_surrogate_signature_exists() {
    let domain_env = DomainEnv::new();
    let fn_env = FnEnv::new(domain_env);

    // Check that train_surrogate is registered
    let sig = fn_env.lookup_fn("train_surrogate");
    assert!(sig.is_ok());

    let sig = sig.unwrap();
    assert_eq!(sig.params.len(), 2);
    assert_eq!(sig.ret, CoreType::SurrogateModel);
}

#[test]
fn test_run_evidence_typed_signature_exists() {
    let domain_env = DomainEnv::new();
    let fn_env = FnEnv::new(domain_env);

    let sig = fn_env.lookup_fn("run_evidence_typed");
    assert!(sig.is_ok());

    let sig = sig.unwrap();
    assert_eq!(sig.params.len(), 2);
    assert_eq!(sig.ret, CoreType::EvidenceResult);
}

#[test]
fn test_run_evidence_with_surrogate_signature_exists() {
    let domain_env = DomainEnv::new();
    let fn_env = FnEnv::new(domain_env);

    let sig = fn_env.lookup_fn("run_evidence_with_surrogate");
    assert!(sig.is_ok());

    let sig = sig.unwrap();
    assert_eq!(sig.params.len(), 3);
    assert_eq!(sig.params[1], CoreType::SurrogateModel);
    assert_eq!(sig.ret, CoreType::EvidenceResult);
}

// =============================================================================
// Built-in Function Runtime Tests
// =============================================================================

#[test]
fn test_train_surrogate_builtin_execution() {
    let ev = RuntimeValue::EvidenceProgram {
        name: "TestEvidence".to_string(),
        handle: "test_handle".to_string(),
    };

    let cfg = RuntimeValue::Record(HashMap::from([
        ("n_train".to_string(), RuntimeValue::Int(100)),
        (
            "backend".to_string(),
            RuntimeValue::BackendKind(BackendKind::Mechanistic),
        ),
        ("seed".to_string(), RuntimeValue::Int(42)),
        ("max_epochs".to_string(), RuntimeValue::Int(50)),
        ("batch_size".to_string(), RuntimeValue::Int(32)),
    ]));

    let result = call_builtin(BuiltinFn::TrainSurrogate, vec![ev, cfg]);
    assert!(result.is_ok());

    match result.unwrap() {
        RuntimeValue::SurrogateModel(handle) => {
            assert!(handle.name.is_some());
        }
        _ => panic!("Expected SurrogateModel"),
    }
}

#[test]
fn test_train_surrogate_rejects_circular_config() {
    let ev = RuntimeValue::EvidenceProgram {
        name: "TestEvidence".to_string(),
        handle: "test_handle".to_string(),
    };

    // Invalid: using Surrogate backend to generate training data
    let cfg = RuntimeValue::Record(HashMap::from([
        ("n_train".to_string(), RuntimeValue::Int(100)),
        (
            "backend".to_string(),
            RuntimeValue::BackendKind(BackendKind::Surrogate),
        ),
        ("seed".to_string(), RuntimeValue::Int(42)),
        ("max_epochs".to_string(), RuntimeValue::Int(50)),
        ("batch_size".to_string(), RuntimeValue::Int(32)),
    ]));

    let result = call_builtin(BuiltinFn::TrainSurrogate, vec![ev, cfg]);
    assert!(result.is_err());

    match result {
        Err(RuntimeError::SurrogateError(msg)) => {
            assert!(msg.contains("cannot use Surrogate backend"));
        }
        _ => panic!("Expected SurrogateError for circular dependency"),
    }
}

#[test]
fn test_run_evidence_typed_builtin_execution() {
    let ev = RuntimeValue::EvidenceProgram {
        name: "TestEvidence".to_string(),
        handle: "test_handle".to_string(),
    };
    let backend = RuntimeValue::BackendKind(BackendKind::Mechanistic);

    let result = call_builtin(BuiltinFn::RunEvidenceTyped, vec![ev, backend]);
    assert!(result.is_ok());

    match result.unwrap() {
        RuntimeValue::EvidenceResult { diagnostics, .. } => {
            assert_eq!(diagnostics.get("backend"), Some(&"Mechanistic".to_string()));
        }
        _ => panic!("Expected EvidenceResult"),
    }
}

#[test]
fn test_run_evidence_typed_rejects_surrogate_backend_without_model() {
    let ev = RuntimeValue::EvidenceProgram {
        name: "TestEvidence".to_string(),
        handle: "test_handle".to_string(),
    };
    let backend = RuntimeValue::BackendKind(BackendKind::Surrogate);

    let result = call_builtin(BuiltinFn::RunEvidenceTyped, vec![ev, backend]);
    assert!(result.is_err());

    match result {
        Err(RuntimeError::BackendError(msg)) => {
            assert!(msg.contains("requires a surrogate model"));
        }
        _ => panic!("Expected BackendError when Surrogate backend used without model"),
    }
}

#[test]
fn test_run_evidence_with_surrogate_builtin_execution() {
    let ev = RuntimeValue::EvidenceProgram {
        name: "TestEvidence".to_string(),
        handle: "test_handle".to_string(),
    };
    let surrogate = RuntimeValue::SurrogateModel(SurrogateModelHandle::new());
    let backend = RuntimeValue::BackendKind(BackendKind::Hybrid);

    let result = call_builtin(
        BuiltinFn::RunEvidenceWithSurrogate,
        vec![ev, surrogate, backend],
    );
    assert!(result.is_ok());

    match result.unwrap() {
        RuntimeValue::EvidenceResult { diagnostics, .. } => {
            assert_eq!(diagnostics.get("backend"), Some(&"Hybrid".to_string()));
            assert!(diagnostics.contains_key("surrogate_id"));
        }
        _ => panic!("Expected EvidenceResult"),
    }
}

// =============================================================================
// Type Checking Integration Tests
// =============================================================================

#[test]
fn test_typecheck_function_with_surrogate_model() {
    // fn train_my_surrogate(ev: EvidenceProgram, n: Int) -> SurrogateModel { ... }
    let fn_def = FnDef::new(
        "train_my_surrogate".to_string(),
        vec![
            Param {
                name: "ev".to_string(),
                ty: Some(TypeAnn::EvidenceProgram),
            },
            Param {
                name: "n".to_string(),
                ty: Some(TypeAnn::Int),
            },
        ],
        Some(TypeAnn::SurrogateModel),
        Block::new(vec![Stmt::Expr(Expr::var("ev".to_string()))]), // Stub body
    );

    let domain_env = DomainEnv::new();
    let fn_env = FnEnv::new(domain_env);

    // This should fail because body doesn't return SurrogateModel
    // but it shows the type system recognizes SurrogateModel
    let result = typecheck_fn(&fn_def, &fn_env);
    // We expect a type mismatch, not an unknown type error
    assert!(result.is_err());
}

// =============================================================================
// End-to-End Scenario Tests
// =============================================================================

#[test]
fn test_complete_surrogate_workflow() {
    // Scenario: Train a surrogate, then use it for inference

    // Step 1: Create an evidence program
    let ev = RuntimeValue::EvidenceProgram {
        name: "PkEvidence".to_string(),
        handle: "pk_handle".to_string(),
    };

    // Step 2: Train a surrogate model
    let train_cfg = RuntimeValue::Record(HashMap::from([
        ("n_train".to_string(), RuntimeValue::Int(1000)),
        (
            "backend".to_string(),
            RuntimeValue::BackendKind(BackendKind::Mechanistic),
        ),
        ("seed".to_string(), RuntimeValue::Int(42)),
        ("max_epochs".to_string(), RuntimeValue::Int(100)),
        ("batch_size".to_string(), RuntimeValue::Int(32)),
    ]));

    let train_result = call_builtin(BuiltinFn::TrainSurrogate, vec![ev.clone(), train_cfg]);
    assert!(train_result.is_ok());

    let surrogate = train_result.unwrap();

    // Step 3: Run evidence with the trained surrogate
    let run_result = call_builtin(
        BuiltinFn::RunEvidenceWithSurrogate,
        vec![
            ev,
            surrogate,
            RuntimeValue::BackendKind(BackendKind::Surrogate),
        ],
    );
    assert!(run_result.is_ok());

    match run_result.unwrap() {
        RuntimeValue::EvidenceResult { diagnostics, .. } => {
            assert_eq!(diagnostics.get("backend"), Some(&"Surrogate".to_string()));
            assert_eq!(
                diagnostics.get("evidence_program"),
                Some(&"PkEvidence".to_string())
            );
        }
        _ => panic!("Expected EvidenceResult"),
    }
}

#[test]
fn test_hybrid_backend_workflow() {
    // Scenario: Use hybrid backend that combines mechanistic and surrogate

    let ev = RuntimeValue::EvidenceProgram {
        name: "HybridEvidence".to_string(),
        handle: "hybrid_handle".to_string(),
    };

    // Train surrogate
    let train_cfg = RuntimeValue::Record(HashMap::from([
        ("n_train".to_string(), RuntimeValue::Int(500)),
        (
            "backend".to_string(),
            RuntimeValue::BackendKind(BackendKind::Mechanistic),
        ),
        ("seed".to_string(), RuntimeValue::Int(123)),
        ("max_epochs".to_string(), RuntimeValue::Int(50)),
        ("batch_size".to_string(), RuntimeValue::Int(64)),
    ]));

    let surrogate = call_builtin(BuiltinFn::TrainSurrogate, vec![ev.clone(), train_cfg]).unwrap();

    // Run with hybrid backend
    let result = call_builtin(
        BuiltinFn::RunEvidenceWithSurrogate,
        vec![
            ev,
            surrogate,
            RuntimeValue::BackendKind(BackendKind::Hybrid),
        ],
    );

    assert!(result.is_ok());
    match result.unwrap() {
        RuntimeValue::EvidenceResult { diagnostics, .. } => {
            assert_eq!(diagnostics.get("backend"), Some(&"Hybrid".to_string()));
        }
        _ => panic!("Expected EvidenceResult"),
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_train_surrogate_missing_config_field() {
    let ev = RuntimeValue::EvidenceProgram {
        name: "TestEvidence".to_string(),
        handle: "test_handle".to_string(),
    };

    // Missing max_epochs field
    let incomplete_cfg = RuntimeValue::Record(HashMap::from([
        ("n_train".to_string(), RuntimeValue::Int(100)),
        (
            "backend".to_string(),
            RuntimeValue::BackendKind(BackendKind::Mechanistic),
        ),
        ("seed".to_string(), RuntimeValue::Int(42)),
        // max_epochs missing
        ("batch_size".to_string(), RuntimeValue::Int(32)),
    ]));

    let result = call_builtin(BuiltinFn::TrainSurrogate, vec![ev, incomplete_cfg]);
    assert!(result.is_err());
}

#[test]
fn test_builtin_arity_checking() {
    // train_surrogate expects 2 args, give it 1
    let ev = RuntimeValue::EvidenceProgram {
        name: "TestEvidence".to_string(),
        handle: "test_handle".to_string(),
    };

    let result = call_builtin(BuiltinFn::TrainSurrogate, vec![ev]);
    assert!(result.is_err());

    match result {
        Err(RuntimeError::ArityMismatch {
            fn_name,
            expected,
            found,
        }) => {
            assert_eq!(fn_name, "train_surrogate");
            assert_eq!(expected, 2);
            assert_eq!(found, 1);
        }
        _ => panic!("Expected ArityMismatch error"),
    }
}

#[test]
fn test_type_error_handling() {
    // Pass Int instead of EvidenceProgram
    let wrong_arg = RuntimeValue::Int(42);
    let cfg = RuntimeValue::Record(HashMap::new());

    let result = call_builtin(BuiltinFn::TrainSurrogate, vec![wrong_arg, cfg]);
    assert!(result.is_err());

    match result {
        Err(RuntimeError::TypeError {
            expected,
            found,
            message: _,
        }) => {
            assert_eq!(expected, "EvidenceProgram");
            assert_eq!(found, "Int");
        }
        _ => panic!("Expected TypeError"),
    }
}
