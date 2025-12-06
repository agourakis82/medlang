//! Integration tests for Phase V1 features
//!
//! Tests the full pipeline for:
//! - Effect system (syntax → parsing → type checking)
//! - Epistemic computing (Knowledge types)
//! - Refinement types (constraints)

use medlangc::ast::phase_v1::*;
use medlangc::effects::{Effect, EffectChecker};
use medlangc::lexer::tokenize;
use medlangc::parser_v1::{effect_annotation, epistemic_type, refinement_constraint};
use medlangc::typeck::TypeContext;
use medlangc::typeck_v1::V1TypeChecker;

// =============================================================================
// Effect System Integration Tests
// =============================================================================

#[test]
fn test_effect_system_pure_annotation() {
    let source = "with Pure";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, annotation) =
        effect_annotation(&tokens).expect("Parser should succeed for 'with Pure'");

    assert!(remaining.is_empty(), "Should consume all tokens");
    assert_eq!(annotation.effects.len(), 1);
    assert_eq!(annotation.effects[0], Effect::Pure);

    // Type check the annotation
    let mut checker = V1TypeChecker::new(TypeContext::new());
    let result = checker.check_effect_annotation("test_fn", &annotation);
    assert!(result.is_ok(), "Type checker should accept Pure effect");
}

#[test]
fn test_effect_system_multiple_effects() {
    let source = "with Prob | IO";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, annotation) =
        effect_annotation(&tokens).expect("Parser should succeed for 'with Prob | IO'");

    assert!(remaining.is_empty());
    assert_eq!(annotation.effects.len(), 2);
    assert!(annotation.effects.contains(&Effect::Prob));
    assert!(annotation.effects.contains(&Effect::IO));

    // Type check
    let mut checker = V1TypeChecker::new(TypeContext::new());
    let result = checker.check_effect_annotation("mcmc_fn", &annotation);
    assert!(result.is_ok());
}

#[test]
fn test_effect_system_gpu_effect() {
    let source = "with GPU";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, annotation) =
        effect_annotation(&tokens).expect("Parser should succeed for 'with GPU'");

    assert!(remaining.is_empty());
    assert_eq!(annotation.effects.len(), 1);
    assert_eq!(annotation.effects[0], Effect::GPU);
}

#[test]
fn test_effect_system_all_effects() {
    let source = "with Pure | Prob | IO | GPU";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, annotation) =
        effect_annotation(&tokens).expect("Parser should succeed for all effects combined");

    assert!(remaining.is_empty());
    assert_eq!(annotation.effects.len(), 4);
    assert!(annotation.effects.contains(&Effect::Pure));
    assert!(annotation.effects.contains(&Effect::Prob));
    assert!(annotation.effects.contains(&Effect::IO));
    assert!(annotation.effects.contains(&Effect::GPU));
}

#[test]
fn test_effect_subsumption_checking() {
    let mut checker = EffectChecker::new();

    // Register a Pure function
    let pure_ann = EffectAnnotationAst::new(vec![Effect::Pure]);
    let mut v1_checker = V1TypeChecker::new(TypeContext::new());
    v1_checker
        .check_effect_annotation("pure_fn", &pure_ann)
        .unwrap();

    // Register a Prob function
    let prob_ann = EffectAnnotationAst::new(vec![Effect::Prob]);
    v1_checker
        .check_effect_annotation("prob_fn", &prob_ann)
        .unwrap();

    // Pure can call Pure (should succeed)
    let result = v1_checker.check_call_effects("pure_fn", "pure_fn");
    assert!(result.is_ok(), "Pure should be able to call Pure");

    // Prob can call Pure (should succeed - subsumption)
    let result = v1_checker.check_call_effects("prob_fn", "pure_fn");
    assert!(result.is_ok(), "Prob should be able to call Pure");
}

// =============================================================================
// Epistemic Computing Integration Tests
// =============================================================================

#[test]
fn test_epistemic_type_simple() {
    let source = "Knowledge<f64>";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, etype) =
        epistemic_type(&tokens).expect("Parser should succeed for Knowledge<f64>");

    assert!(remaining.is_empty());
    assert_eq!(etype.inner_type, "f64");
    assert_eq!(etype.min_confidence, None);

    // Type check
    let mut checker = V1TypeChecker::new(TypeContext::new());
    let result = checker.check_epistemic_type("measured_value", &etype);
    assert!(result.is_ok());
}

#[test]
fn test_epistemic_type_with_confidence() {
    let source = "Knowledge<ConcMass>(0.85)";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, etype) =
        epistemic_type(&tokens).expect("Parser should succeed for Knowledge<ConcMass>(0.85)");

    assert!(remaining.is_empty());
    assert_eq!(etype.inner_type, "ConcMass");
    assert_eq!(etype.min_confidence, Some(0.85));

    // Type check
    let mut checker = V1TypeChecker::new(TypeContext::new());
    let result = checker.check_epistemic_type("plasma_conc", &etype);
    assert!(result.is_ok());
}

#[test]
fn test_epistemic_type_medical_types() {
    // Test with various medical unit types
    let test_cases = vec![
        ("Knowledge<Mass>", "Mass"),
        ("Knowledge<Volume>", "Volume"),
        ("Knowledge<Clearance>", "Clearance"),
        ("Knowledge<Time>", "Time"),
    ];

    for (source, expected_type) in test_cases {
        let tokens = tokenize(source).expect("Lexer should succeed");
        let (remaining, etype) =
            epistemic_type(&tokens).expect(&format!("Parser should succeed for {}", source));

        assert!(remaining.is_empty());
        assert_eq!(etype.inner_type, expected_type);
    }
}

#[test]
fn test_epistemic_confidence_validation() {
    let mut checker = V1TypeChecker::new(TypeContext::new());

    // Register epistemic type with min confidence 0.8
    let etype = EpistemicTypeAst {
        inner_type: "f64".to_string(),
        min_confidence: Some(0.8),
    };
    checker
        .check_epistemic_type("test_var", &etype)
        .expect("Registration should succeed");

    // Test confidence validation
    assert!(
        checker.check_confidence("test_var", 0.9).is_ok(),
        "Should accept confidence >= min"
    );
    assert!(
        checker.check_confidence("test_var", 0.85).is_ok(),
        "Should accept confidence >= min"
    );
    assert!(
        checker.check_confidence("test_var", 0.7).is_err(),
        "Should reject confidence < min"
    );
}

#[test]
fn test_epistemic_confidence_bounds() {
    let mut checker = V1TypeChecker::new(TypeContext::new());

    // Invalid: confidence > 1.0
    let etype_invalid_high = EpistemicTypeAst {
        inner_type: "f64".to_string(),
        min_confidence: Some(1.5),
    };
    let result = checker.check_epistemic_type("invalid_high", &etype_invalid_high);
    assert!(result.is_err(), "Should reject min_confidence > 1.0");

    // Valid: confidence = 0.0
    let etype_valid_zero = EpistemicTypeAst {
        inner_type: "f64".to_string(),
        min_confidence: Some(0.0),
    };
    let result = checker.check_epistemic_type("valid_zero", &etype_valid_zero);
    assert!(result.is_ok(), "Should accept min_confidence = 0.0");

    // Valid: confidence = 1.0
    let etype_valid_one = EpistemicTypeAst {
        inner_type: "f64".to_string(),
        min_confidence: Some(1.0),
    };
    let result = checker.check_epistemic_type("valid_one", &etype_valid_one);
    assert!(result.is_ok(), "Should accept min_confidence = 1.0");
}

// =============================================================================
// Refinement Type Integration Tests
// =============================================================================

#[test]
fn test_refinement_comparison_constraint() {
    let source = "where CL > 0.0";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, constraint) =
        refinement_constraint(&tokens).expect("Parser should succeed for 'where CL > 0.0'");

    assert!(remaining.is_empty());

    match &constraint.constraint {
        ConstraintExpr::Comparison { var, op, value } => {
            assert_eq!(var, "CL");
            assert_eq!(*op, ComparisonOp::Gt);
            match value {
                ConstraintLiteral::Float(v) => assert_eq!(*v, 0.0),
                _ => panic!("Expected Float literal"),
            }
        }
        _ => panic!("Expected Comparison constraint"),
    }
}

#[test]
fn test_refinement_range_constraint() {
    let source = "where AGE in 18.0..120.0";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, constraint) =
        refinement_constraint(&tokens).expect("Parser should succeed for range constraint");

    assert!(remaining.is_empty());

    match &constraint.constraint {
        ConstraintExpr::Range { var, lower, upper } => {
            assert_eq!(var, "AGE");
            match lower {
                ConstraintLiteral::Float(v) => assert_eq!(*v, 18.0),
                _ => panic!("Expected Float literal for lower bound"),
            }
            match upper {
                ConstraintLiteral::Float(v) => assert_eq!(*v, 120.0),
                _ => panic!("Expected Float literal for upper bound"),
            }
        }
        _ => panic!("Expected Range constraint"),
    }
}

#[test]
fn test_refinement_all_comparison_operators() {
    let test_cases = vec![
        ("where X > 0.0", ComparisonOp::Gt),
        ("where X < 100.0", ComparisonOp::Lt),
        ("where X >= 0.0", ComparisonOp::Ge),
        ("where X <= 100.0", ComparisonOp::Le),
        ("where X == 50.0", ComparisonOp::Eq),
        ("where X != 0.0", ComparisonOp::Ne),
    ];

    for (source, expected_op) in test_cases {
        let tokens = tokenize(source).expect("Lexer should succeed");
        let (remaining, constraint) =
            refinement_constraint(&tokens).expect(&format!("Parser should succeed for {}", source));

        assert!(remaining.is_empty());
        match &constraint.constraint {
            ConstraintExpr::Comparison { op, .. } => {
                assert_eq!(*op, expected_op, "Operator mismatch for {}", source);
            }
            _ => panic!("Expected Comparison constraint for {}", source),
        }
    }
}

#[test]
fn test_refinement_unit_literal_constraint() {
    let source = "where DOSE > 0.0_mg";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (remaining, constraint) =
        refinement_constraint(&tokens).expect("Parser should succeed for unit literal constraint");

    assert!(remaining.is_empty());

    match &constraint.constraint {
        ConstraintExpr::Comparison { var, value, .. } => {
            assert_eq!(var, "DOSE");
            match value {
                ConstraintLiteral::UnitValue(val, unit) => {
                    assert_eq!(*val, 0.0);
                    assert_eq!(unit, "mg");
                }
                _ => panic!("Expected UnitValue literal"),
            }
        }
        _ => panic!("Expected Comparison constraint"),
    }
}

#[test]
fn test_refinement_constraint_to_runtime() {
    let source = "where CL > 0.0";
    let tokens = tokenize(source).expect("Lexer should succeed");
    let (_, constraint) = refinement_constraint(&tokens).unwrap();

    // Convert AST constraint to runtime constraint
    let runtime_constraint = constraint.constraint.to_constraint();

    // Verify conversion
    use medlangc::refinement::clinical::Constraint;
    match runtime_constraint {
        Constraint::Comparison { var, .. } => {
            assert_eq!(var, "CL");
        }
        _ => panic!("Expected Comparison constraint after conversion"),
    }
}

// =============================================================================
// Combined Integration Tests
// =============================================================================

#[test]
fn test_combined_epistemic_and_refinement() {
    // Test a parameter with both epistemic type and refinement
    // e.g., param CL : Knowledge<Clearance>(0.8) where CL > 0.0

    let mut checker = V1TypeChecker::new(TypeContext::new());

    // First, register the epistemic type
    let etype = EpistemicTypeAst {
        inner_type: "Clearance".to_string(),
        min_confidence: Some(0.8),
    };
    checker
        .check_epistemic_type("CL", &etype)
        .expect("Epistemic type registration should succeed");

    // Then, add the refinement constraint
    let constraint_source = "where CL > 0.0";
    let tokens = tokenize(constraint_source).unwrap();
    let (_, constraint) = refinement_constraint(&tokens).unwrap();

    let result = checker.check_refinement_constraint("CL", &constraint);
    assert!(
        result.is_ok(),
        "Should accept refinement on epistemic variable"
    );
}

#[test]
fn test_multiple_v1_features_combined() {
    let mut checker = V1TypeChecker::new(TypeContext::new());

    // Register a function with effects
    let effect_ann = EffectAnnotationAst::new(vec![Effect::Prob, Effect::IO]);
    checker
        .check_effect_annotation("mcmc_sample", &effect_ann)
        .expect("Effect registration should succeed");

    // Register an epistemic parameter
    let etype = EpistemicTypeAst {
        inner_type: "f64".to_string(),
        min_confidence: Some(0.85),
    };
    checker
        .check_epistemic_type("posterior_mean", &etype)
        .expect("Epistemic type registration should succeed");

    // Add refinement constraint
    let constraint_source = "where posterior_mean > 0.0";
    let tokens = tokenize(constraint_source).unwrap();
    let (_, constraint) = refinement_constraint(&tokens).unwrap();
    checker
        .check_refinement_constraint("posterior_mean", &constraint)
        .expect("Refinement registration should succeed");

    // Verify all features are registered
    assert!(checker.effect_checker.get_effects("mcmc_sample").is_some());
    assert!(checker.epistemic_types.contains_key("posterior_mean"));
    assert!(checker
        .refinement_constraints
        .contains_key("posterior_mean"));
}

// =============================================================================
// Clinical Use Case Tests
// =============================================================================

#[test]
fn test_clinical_clearance_safety() {
    // Test the common clinical pattern: positive clearance constraint
    let source = "where CL > 0.0";
    let tokens = tokenize(source).unwrap();
    let (_, constraint) = refinement_constraint(&tokens).unwrap();

    let mut checker = V1TypeChecker::new(TypeContext::new());

    // This should succeed (safety constraint for division by zero prevention)
    let result = checker.check_refinement_constraint("CL", &constraint);
    assert!(
        result.is_ok(),
        "Positive clearance constraint should be valid"
    );
}

#[test]
fn test_clinical_age_range() {
    // Test physiological age constraint
    let source = "where AGE in 0.0..120.0";
    let tokens = tokenize(source).unwrap();
    let (_, constraint) = refinement_constraint(&tokens).unwrap();

    let mut checker = V1TypeChecker::new(TypeContext::new());
    let result = checker.check_refinement_constraint("AGE", &constraint);
    assert!(result.is_ok(), "Age range constraint should be valid");
}

#[test]
fn test_clinical_measured_concentration() {
    // Test epistemic type for measured plasma concentration
    let source = "Knowledge<ConcMass>(0.75)";
    let tokens = tokenize(source).unwrap();
    let (_, etype) = epistemic_type(&tokens).unwrap();

    let mut checker = V1TypeChecker::new(TypeContext::new());
    let result = checker.check_epistemic_type("C_plasma_measured", &etype);
    assert!(
        result.is_ok(),
        "Measured concentration with confidence should be valid"
    );

    // Verify confidence validation works
    assert!(checker.check_confidence("C_plasma_measured", 0.80).is_ok());
    assert!(checker.check_confidence("C_plasma_measured", 0.70).is_err());
}

#[test]
fn test_clinical_mcmc_effects() {
    // Test effect annotation for MCMC sampling function
    let source = "with Prob | IO";
    let tokens = tokenize(source).unwrap();
    let (_, annotation) = effect_annotation(&tokens).unwrap();

    let mut checker = V1TypeChecker::new(TypeContext::new());
    let result = checker.check_effect_annotation("run_mcmc_sampling", &annotation);
    assert!(
        result.is_ok(),
        "MCMC function should have Prob and IO effects"
    );
}
