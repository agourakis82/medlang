//! Type checking extensions for Phase V1 features.
//!
//! This module integrates:
//! - Effect system type checking
//! - Epistemic type validation
//! - Refinement constraint checking

use crate::ast::phase_v1::*;
use crate::effects::{Effect, EffectAnnotation, EffectChecker, EffectSet};
use crate::epistemic::{Knowledge, Provenance};
use crate::refinement::clinical::{Constraint, ConstraintValue};
use crate::typeck::{InferredType, TypeContext, TypeError};
use std::collections::HashMap;
use thiserror::Error;

// =============================================================================
// Phase V1 Type Errors
// =============================================================================

#[derive(Debug, Error, Clone, PartialEq)]
pub enum V1TypeError {
    #[error("Effect mismatch: {context} requires {required:?}, but has {actual:?}")]
    EffectMismatch {
        context: String,
        required: EffectSet,
        actual: EffectSet,
    },

    #[error("Confidence too low: {context} requires minimum {required}, got {actual}")]
    InsufficientConfidence {
        context: String,
        required: f64,
        actual: f64,
    },

    #[error("Refinement constraint violated: {constraint}")]
    ConstraintViolation { constraint: String },

    #[error("Epistemic type mismatch: expected Knowledge<{expected}>, found Knowledge<{found}>")]
    EpistemicTypeMismatch { expected: String, found: String },

    #[error("Cannot unwrap Knowledge type without confidence check")]
    UnsafeKnowledgeUnwrap,

    #[error("Base type error: {0}")]
    BaseTypeError(#[from] TypeError),
}

// =============================================================================
// Phase V1 Type Checker
// =============================================================================

/// Extended type checker with Phase V1 features
pub struct V1TypeChecker {
    /// Base type context from V0
    pub base_ctx: TypeContext,

    /// Effect checker for tracking computational effects
    pub effect_checker: EffectChecker,

    /// Epistemic type registry (variable -> Knowledge metadata)
    pub epistemic_types: HashMap<String, EpistemicMetadata>,

    /// Refinement constraint registry
    pub refinement_constraints: HashMap<String, Vec<Constraint>>,
}

/// Metadata for epistemic types
#[derive(Debug, Clone)]
pub struct EpistemicMetadata {
    pub inner_type: InferredType,
    pub min_confidence: Option<f64>,
}

impl V1TypeChecker {
    /// Create a new V1 type checker with base context
    pub fn new(base_ctx: TypeContext) -> Self {
        Self {
            base_ctx,
            effect_checker: EffectChecker::new(),
            epistemic_types: HashMap::new(),
            refinement_constraints: HashMap::new(),
        }
    }

    /// Check effect annotation validity
    pub fn check_effect_annotation(
        &mut self,
        name: &str,
        annotation: &EffectAnnotationAst,
    ) -> Result<(), V1TypeError> {
        let effect_set = annotation.to_effect_set();

        // Register the effect annotation
        let effect_ann = EffectAnnotation {
            effects: effect_set.clone(),
            explicit: true, // User-provided annotation
        };

        self.effect_checker.register(name.to_string(), effect_ann);

        Ok(())
    }

    /// Check epistemic type declaration
    pub fn check_epistemic_type(
        &mut self,
        var_name: &str,
        etype: &EpistemicTypeAst,
    ) -> Result<(), V1TypeError> {
        // Parse the inner type
        let inner_type = self.parse_type_name(&etype.inner_type)?;

        // Validate confidence constraint if present
        if let Some(min_conf) = etype.min_confidence {
            if !(0.0..=1.0).contains(&min_conf) {
                return Err(V1TypeError::InsufficientConfidence {
                    context: format!("Variable {}", var_name),
                    required: 0.0,
                    actual: min_conf,
                });
            }
        }

        // Register epistemic metadata
        self.epistemic_types.insert(
            var_name.to_string(),
            EpistemicMetadata {
                inner_type,
                min_confidence: etype.min_confidence,
            },
        );

        Ok(())
    }

    /// Check refinement constraint
    pub fn check_refinement_constraint(
        &mut self,
        var_name: &str,
        constraint: &RefinementConstraintAst,
    ) -> Result<(), V1TypeError> {
        // Convert AST constraint to runtime constraint
        let runtime_constraint = constraint.constraint.to_constraint();

        // Basic validation: check that referenced variables exist
        self.validate_constraint_vars(&runtime_constraint)?;

        // Register the constraint
        self.refinement_constraints
            .entry(var_name.to_string())
            .or_insert_with(Vec::new)
            .push(runtime_constraint);

        Ok(())
    }

    /// Validate that all variables in a constraint are defined
    fn validate_constraint_vars(&self, constraint: &Constraint) -> Result<(), V1TypeError> {
        match constraint {
            Constraint::Var(name) => {
                if self.base_ctx.lookup(name).is_none() && !self.epistemic_types.contains_key(name)
                {
                    return Err(V1TypeError::BaseTypeError(TypeError::UndefinedVariable(
                        name.clone(),
                    )));
                }
                Ok(())
            }
            Constraint::Comparison { var, .. } => {
                if self.base_ctx.lookup(var).is_none() && !self.epistemic_types.contains_key(var) {
                    return Err(V1TypeError::BaseTypeError(TypeError::UndefinedVariable(
                        var.clone(),
                    )));
                }
                Ok(())
            }
            Constraint::Binary { left, right, .. } => {
                self.validate_constraint_vars(left)?;
                self.validate_constraint_vars(right)?;
                Ok(())
            }
            Constraint::Range { var, .. } => {
                if self.base_ctx.lookup(var).is_none() && !self.epistemic_types.contains_key(var) {
                    return Err(V1TypeError::BaseTypeError(TypeError::UndefinedVariable(
                        var.clone(),
                    )));
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Parse a type name string to InferredType
    fn parse_type_name(&self, type_name: &str) -> Result<InferredType, V1TypeError> {
        match type_name {
            "f64" | "Float" => Ok(InferredType::Float),
            "bool" | "Bool" => Ok(InferredType::Bool),
            "Mass" => Ok(InferredType::Quantity(crate::typeck::UnitDimension::mass())),
            "Volume" => Ok(InferredType::Quantity(
                crate::typeck::UnitDimension::volume(),
            )),
            "Time" => Ok(InferredType::Quantity(crate::typeck::UnitDimension::time())),
            "Clearance" => Ok(InferredType::Quantity(
                crate::typeck::UnitDimension::clearance(),
            )),
            "ConcMass" => Ok(InferredType::Quantity(
                crate::typeck::UnitDimension::conc_mass(),
            )),
            "RateConst" => Ok(InferredType::Quantity(
                crate::typeck::UnitDimension::rate_const(),
            )),
            _ => Ok(InferredType::Unknown),
        }
    }

    /// Check that a function call respects effect annotations
    pub fn check_call_effects(&self, caller: &str, callee: &str) -> Result<(), V1TypeError> {
        if let Err(msg) = self.effect_checker.check_call(caller, callee) {
            // Extract effect sets for error message
            let caller_effects = self
                .effect_checker
                .get_effects(caller)
                .map(|a| a.effects.clone())
                .unwrap_or_else(EffectSet::pure);

            let callee_effects = self
                .effect_checker
                .get_effects(callee)
                .map(|a| a.effects.clone())
                .unwrap_or_else(EffectSet::pure);

            return Err(V1TypeError::EffectMismatch {
                context: format!("{} calling {}", caller, callee),
                required: callee_effects,
                actual: caller_effects,
            });
        }

        Ok(())
    }

    /// Validate that a Knowledge value meets minimum confidence requirements
    pub fn check_confidence(
        &self,
        var_name: &str,
        actual_confidence: f64,
    ) -> Result<(), V1TypeError> {
        if let Some(metadata) = self.epistemic_types.get(var_name) {
            if let Some(min_conf) = metadata.min_confidence {
                if actual_confidence < min_conf {
                    return Err(V1TypeError::InsufficientConfidence {
                        context: var_name.to_string(),
                        required: min_conf,
                        actual: actual_confidence,
                    });
                }
            }
        }

        Ok(())
    }

    // =========================================================================
    // Phase V2: SMT Verification Integration
    // =========================================================================

    /// Verify all refinement constraints using SMT solver (Phase V2)
    ///
    /// This method translates refinement constraints to SMT formulas and uses Z3
    /// to prove their validity or find counterexamples.
    ///
    /// Requires compilation with `--features smt-verification` for full verification.
    /// Without the feature, warnings are printed but no errors are raised.
    pub fn verify_constraints_with_smt(&mut self) -> Result<(), V1TypeError> {
        use crate::smt::vc_gen::VCGenerator;
        use crate::smt::{SMTContext, SMTTranslator, Z3Result};

        let mut smt_ctx = SMTContext::new();
        let mut vc_gen = VCGenerator::new();

        // Translate all refinement constraints to SMT
        for (var_name, constraints) in &self.refinement_constraints {
            for constraint in constraints {
                // Convert clinical constraint to constraint expression
                if let Some(constraint_expr) = self.constraint_to_expr(constraint) {
                    match SMTTranslator::translate_constraint(&constraint_expr) {
                        Ok(smt_formula) => {
                            smt_ctx.add_assertion(var_name, smt_formula.clone());
                            vc_gen.assume(smt_formula);
                        }
                        Err(e) => {
                            eprintln!(
                                "⚠ Warning: Could not translate constraint for {}: {}",
                                var_name, e
                            );
                        }
                    }
                }
            }
        }

        // Generate verification conditions for division safety, etc.
        let vcs = vc_gen.get_vcs();

        // Check each VC
        for vc in vcs {
            match smt_ctx.check(vc) {
                Z3Result::Proven => {
                    #[cfg(feature = "smt-verification")]
                    println!("✓ Verified: {}", vc.description);
                }
                Z3Result::Counterexample(model) => {
                    #[cfg(feature = "smt-verification")]
                    {
                        let mut msg =
                            format!("Cannot prove: {}\n\nCounterexample found:", vc.description);
                        for (var, value) in &model.assignments {
                            msg.push_str(&format!("\n  {} = {:?}", var, value));
                        }
                        return Err(V1TypeError::ConstraintViolation { constraint: msg });
                    }

                    #[cfg(not(feature = "smt-verification"))]
                    {
                        eprintln!("⚠ Warning: Cannot verify: {}", vc.description);
                        eprintln!(
                            "  (Compile with --features smt-verification for full verification)"
                        );
                    }
                }
                Z3Result::Unknown => {
                    eprintln!("⚠ Warning: Could not verify: {}", vc.description);
                }
            }
        }

        Ok(())
    }

    /// Convert clinical Constraint to ConstraintExpr for SMT translation
    fn constraint_to_expr(&self, constraint: &Constraint) -> Option<ConstraintExpr> {
        use crate::ast::phase_v1::{ConstraintLiteral, LogicalOp as AstLogicalOp};
        use crate::refinement::clinical::{
            ComparisonOp as ClinicalCompOp, LogicalOp as ClinicalLogOp,
        };

        match constraint {
            Constraint::Comparison { var, op, value } => {
                // Convert clinical ComparisonOp to AST ComparisonOp
                let ast_op = match op {
                    ClinicalCompOp::Eq => crate::ast::phase_v1::ComparisonOp::Eq,
                    ClinicalCompOp::Ne => crate::ast::phase_v1::ComparisonOp::Ne,
                    ClinicalCompOp::Lt => crate::ast::phase_v1::ComparisonOp::Lt,
                    ClinicalCompOp::Le => crate::ast::phase_v1::ComparisonOp::Le,
                    ClinicalCompOp::Gt => crate::ast::phase_v1::ComparisonOp::Gt,
                    ClinicalCompOp::Ge => crate::ast::phase_v1::ComparisonOp::Ge,
                };

                Some(ConstraintExpr::Comparison {
                    var: var.clone(),
                    op: ast_op,
                    value: self.constraint_value_to_literal(value),
                })
            }

            Constraint::Binary { left, op, right } => {
                let left_expr = self.constraint_to_expr(left)?;
                let right_expr = self.constraint_to_expr(right)?;

                let ast_op = match op {
                    ClinicalLogOp::And => AstLogicalOp::And,
                    ClinicalLogOp::Or => AstLogicalOp::Or,
                };

                Some(ConstraintExpr::Binary {
                    left: Box::new(left_expr),
                    op: ast_op,
                    right: Box::new(right_expr),
                })
            }

            // Bool and Var constraints don't translate directly to ConstraintExpr
            _ => None,
        }
    }

    /// Convert ConstraintValue to ConstraintLiteral
    fn constraint_value_to_literal(&self, value: &ConstraintValue) -> ConstraintLiteral {
        match value {
            ConstraintValue::Float(f) => ConstraintLiteral::Float(*f),
            ConstraintValue::Int(i) => ConstraintLiteral::Int(*i),
            // Bool and String don't have direct equivalents in ConstraintLiteral
            _ => ConstraintLiteral::Float(0.0), // Fallback
        }
    }
}

// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_effect_annotation_registration() {
        let mut checker = V1TypeChecker::new(TypeContext::new());

        let ann = EffectAnnotationAst::new(vec![Effect::Prob, Effect::IO]);
        let result = checker.check_effect_annotation("test_fn", &ann);

        assert!(result.is_ok());
        assert!(checker.effect_checker.get_effects("test_fn").is_some());
    }

    #[test]
    fn test_epistemic_type_registration() {
        let mut checker = V1TypeChecker::new(TypeContext::new());

        let etype = EpistemicTypeAst {
            inner_type: "f64".to_string(),
            min_confidence: Some(0.8),
        };

        let result = checker.check_epistemic_type("test_var", &etype);
        assert!(result.is_ok());
        assert!(checker.epistemic_types.contains_key("test_var"));
    }

    #[test]
    fn test_confidence_validation() {
        let mut checker = V1TypeChecker::new(TypeContext::new());

        let etype = EpistemicTypeAst {
            inner_type: "f64".to_string(),
            min_confidence: Some(0.8),
        };

        checker.check_epistemic_type("test_var", &etype).unwrap();

        // Should pass with sufficient confidence
        assert!(checker.check_confidence("test_var", 0.9).is_ok());

        // Should fail with insufficient confidence
        assert!(checker.check_confidence("test_var", 0.7).is_err());
    }

    #[test]
    fn test_parse_type_names() {
        let checker = V1TypeChecker::new(TypeContext::new());

        assert!(matches!(
            checker.parse_type_name("f64"),
            Ok(InferredType::Float)
        ));
        assert!(matches!(
            checker.parse_type_name("Mass"),
            Ok(InferredType::Quantity(_))
        ));
        assert!(matches!(
            checker.parse_type_name("bool"),
            Ok(InferredType::Bool)
        ));
    }
}
