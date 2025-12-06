//! Refinement Type Checker
//!
//! Main interface for checking refinement types in MedLang programs.
//! Integrates with the existing type checker to add refinement verification.
//!
//! # Usage
//!
//! ```text
//! let checker = RefinementChecker::new();
//! let result = checker.check_function(&func);
//! match result {
//!     Ok(()) => println!("Refinements verified!"),
//!     Err(errors) => for e in errors { eprintln!("{}", e); }
//! }
//! ```

use std::collections::HashMap;

use super::constraint::{Constraint, ConstraintGenerator, ConstraintSet};
use super::error::{Counterexample, RefinementError, RefinementErrorKind};
use super::smt::{SmtResult, SmtSolver};
use super::subtype::{SubtypeChecker, SubtypeResult};
use super::syntax::{BaseTypeRef, Predicate, RefinedVar, RefinementExpr, RefinementType};
use crate::ast::Span;

/// Create a dummy span for errors when no real span is available
fn dummy_span() -> Span {
    Span::new(0, 0, 0)
}

/// Environment for refinement type checking
#[derive(Clone, Debug, Default)]
pub struct RefinementEnv {
    /// Variable -> Refinement type bindings
    bindings: HashMap<String, RefinementType>,
    /// Function signatures with refinements
    functions: HashMap<String, FunctionRefinement>,
    /// Type aliases with refinements
    type_aliases: HashMap<String, RefinementType>,
    /// Current path conditions (from branches)
    path_conditions: Vec<Predicate>,
}

/// Refinement signature for a function
#[derive(Clone, Debug)]
pub struct FunctionRefinement {
    /// Parameter types with refinements
    pub params: Vec<(String, RefinementType)>,
    /// Return type with refinement
    pub return_type: RefinementType,
    /// Preconditions (requires clauses)
    pub requires: Vec<Predicate>,
    /// Postconditions (ensures clauses)
    pub ensures: Vec<Predicate>,
}

impl RefinementEnv {
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a variable to a refinement type
    pub fn bind(&mut self, name: impl Into<String>, ty: RefinementType) {
        self.bindings.insert(name.into(), ty);
    }

    /// Look up a variable's refinement type
    pub fn lookup(&self, name: &str) -> Option<&RefinementType> {
        self.bindings.get(name)
    }

    /// Register a function signature
    pub fn register_function(&mut self, name: impl Into<String>, sig: FunctionRefinement) {
        self.functions.insert(name.into(), sig);
    }

    /// Look up a function signature
    pub fn lookup_function(&self, name: &str) -> Option<&FunctionRefinement> {
        self.functions.get(name)
    }

    /// Register a type alias
    pub fn register_alias(&mut self, name: impl Into<String>, ty: RefinementType) {
        self.type_aliases.insert(name.into(), ty);
    }

    /// Resolve a type alias
    pub fn resolve_alias(&self, name: &str) -> Option<&RefinementType> {
        self.type_aliases.get(name)
    }

    /// Push a path condition (entering a branch)
    pub fn push_condition(&mut self, cond: Predicate) {
        self.path_conditions.push(cond);
    }

    /// Pop a path condition
    pub fn pop_condition(&mut self) {
        self.path_conditions.pop();
    }

    /// Get current path conditions
    pub fn path_conditions(&self) -> &[Predicate] {
        &self.path_conditions
    }

    /// Get all bindings as predicates (for SMT context)
    pub fn bindings_as_predicates(&self) -> Vec<Predicate> {
        let mut preds = Vec::new();
        for (name, rty) in &self.bindings {
            let pred = rty
                .predicate
                .substitute(&rty.binder.name, &RefinementExpr::var(name));
            preds.push(pred);
        }
        preds.extend(self.path_conditions.clone());
        preds
    }

    /// Enter a new scope
    pub fn enter_scope(&self) -> Self {
        self.clone()
    }
}

/// Main refinement type checker
pub struct RefinementChecker {
    /// SMT solver
    solver: SmtSolver,
    /// Subtype checker
    subtype_checker: SubtypeChecker,
    /// Current environment
    env: RefinementEnv,
    /// Collected errors
    errors: Vec<RefinementError>,
    /// Verification statistics
    stats: VerificationStats,
}

/// Verification statistics
#[derive(Clone, Debug, Default)]
pub struct VerificationStats {
    pub constraints_generated: usize,
    pub constraints_verified: usize,
    pub constraints_failed: usize,
    pub solver_calls: usize,
    pub cache_hits: usize,
    pub total_time_ms: u64,
}

impl RefinementChecker {
    pub fn new() -> Self {
        Self {
            solver: SmtSolver::new(),
            subtype_checker: SubtypeChecker::new(),
            env: RefinementEnv::new(),
            errors: Vec::new(),
            stats: VerificationStats::default(),
        }
    }

    pub fn with_solver(mut self, solver: SmtSolver) -> Self {
        self.solver = solver;
        self
    }

    pub fn with_env(mut self, env: RefinementEnv) -> Self {
        self.env = env;
        self
    }

    /// Get the environment
    pub fn env(&self) -> &RefinementEnv {
        &self.env
    }

    /// Get mutable environment
    pub fn env_mut(&mut self) -> &mut RefinementEnv {
        &mut self.env
    }

    /// Get collected errors
    pub fn errors(&self) -> &[RefinementError] {
        &self.errors
    }

    /// Get statistics
    pub fn stats(&self) -> &VerificationStats {
        &self.stats
    }

    /// Clear errors
    pub fn clear_errors(&mut self) {
        self.errors.clear();
    }

    /// Check that a value has a refinement type
    pub fn check_value(
        &mut self,
        value: &RefinementExpr,
        expected: &RefinementType,
        span: Option<Span>,
    ) -> bool {
        self.stats.constraints_generated += 1;

        // Build predicate: expected.predicate[binder := value]
        let pred = expected.predicate.substitute(&expected.binder.name, value);

        // Add context from environment
        let context = self.env.bindings_as_predicates();
        let full_pred = if context.is_empty() {
            pred
        } else {
            let ctx = context.into_iter().reduce(Predicate::and).unwrap();
            Predicate::implies(ctx, pred)
        };

        self.stats.solver_calls += 1;
        match self.solver.check_valid(&full_pred) {
            SmtResult::Unsat => {
                self.stats.constraints_verified += 1;
                true
            }
            SmtResult::Sat(model) => {
                self.stats.constraints_failed += 1;
                self.errors.push(
                    RefinementError::new(RefinementErrorKind::PredicateUnsatisfied {
                        expected: expected.predicate.clone(),
                        counterexample: model.map(|m| m.to_counterexample()),
                    })
                    .with_span(span.clone().unwrap_or_else(dummy_span)),
                );
                false
            }
            SmtResult::Unknown(reason) => {
                // Treat unknown as potential error
                self.errors.push(
                    RefinementError::new(RefinementErrorKind::SolverTimeout {
                        constraint: format!("{}", expected.predicate),
                        timeout_ms: 5000,
                    })
                    .with_note(reason),
                );
                false
            }
            SmtResult::Error(err) => {
                self.errors
                    .push(RefinementError::new(RefinementErrorKind::SolverError(err)));
                false
            }
        }
    }

    /// Check subtype relation
    pub fn check_subtype(
        &mut self,
        sub: &RefinementType,
        sup: &RefinementType,
        span: Option<Span>,
    ) -> bool {
        self.stats.constraints_generated += 1;

        let context = self.env.bindings_as_predicates();
        let result = self.subtype_checker.check_with_context(sub, sup, &context);

        match result {
            SubtypeResult::Valid => {
                self.stats.constraints_verified += 1;
                true
            }
            SubtypeResult::Invalid {
                counterexample,
                reason,
            } => {
                self.stats.constraints_failed += 1;
                self.errors.push(
                    RefinementError::subtype_failed(sub.clone(), sup.clone(), counterexample, None)
                        .with_note(reason),
                );
                false
            }
            SubtypeResult::Unknown(reason) => {
                self.errors.push(
                    RefinementError::new(RefinementErrorKind::SolverTimeout {
                        constraint: format!("{} <: {}", sub, sup),
                        timeout_ms: 5000,
                    })
                    .with_note(reason),
                );
                false
            }
        }
    }

    /// Check function call arguments satisfy parameter refinements
    pub fn check_call(
        &mut self,
        func_name: &str,
        args: &[(String, RefinementExpr)],
        span: Option<Span>,
    ) -> Option<RefinementType> {
        let func_sig = match self.env.lookup_function(func_name) {
            Some(f) => f.clone(),
            None => return None, // No refinement info for this function
        };

        // Check each argument against its parameter type
        for ((param_name, param_ty), (_, arg_expr)) in func_sig.params.iter().zip(args.iter()) {
            self.stats.constraints_generated += 1;

            if !self.check_value(arg_expr, param_ty, span.clone()) {
                self.errors.push(RefinementError::precondition_failed(
                    func_name,
                    param_name,
                    param_ty.predicate.clone(),
                    None,
                    span.clone(),
                ));
            }
        }

        // Check requires clauses
        for req in &func_sig.requires {
            // Substitute parameters with arguments
            let mut pred = req.clone();
            for ((param_name, _), (_, arg_expr)) in func_sig.params.iter().zip(args.iter()) {
                pred = pred.substitute(param_name, arg_expr);
            }

            self.stats.constraints_generated += 1;
            let context = self.env.bindings_as_predicates();
            let full_pred = if context.is_empty() {
                pred.clone()
            } else {
                let ctx = context.into_iter().reduce(Predicate::and).unwrap();
                Predicate::implies(ctx, pred.clone())
            };

            self.stats.solver_calls += 1;
            if let SmtResult::Sat(model) = self.solver.check_valid(&full_pred) {
                self.stats.constraints_failed += 1;
                self.errors.push(RefinementError::precondition_failed(
                    func_name,
                    "requires",
                    pred,
                    model.map(|m| m.to_counterexample()),
                    span.clone(),
                ));
            }
        }

        // Compute return type by substituting parameters
        let mut return_type = func_sig.return_type.clone();
        for ((param_name, _), (_, arg_expr)) in func_sig.params.iter().zip(args.iter()) {
            return_type = return_type.substitute(param_name, arg_expr);
        }

        Some(return_type)
    }

    /// Check let binding
    pub fn check_let(
        &mut self,
        var_name: &str,
        annotation: Option<&RefinementType>,
        init: &RefinementExpr,
        span: Option<Span>,
    ) {
        if let Some(expected_ty) = annotation {
            // Check initializer satisfies the annotation
            self.check_value(init, expected_ty, span.clone());

            // Bind with the annotated type (with value substituted)
            let bound_ty = expected_ty.substitute(&expected_ty.binder.name, init);
            self.env.bind(var_name, bound_ty);
        }
    }

    /// Check if-then-else
    pub fn check_conditional(
        &mut self,
        condition: &Predicate,
        then_type: &RefinementType,
        else_type: &RefinementType,
        expected: &RefinementType,
        span: Option<Span>,
    ) {
        // Check then branch with condition assumed
        self.env.push_condition(condition.clone());
        self.check_subtype(then_type, expected, span.clone());
        self.env.pop_condition();

        // Check else branch with negation of condition assumed
        self.env.push_condition(Predicate::not(condition.clone()));
        self.check_subtype(else_type, expected, span);
        self.env.pop_condition();
    }

    /// Check array bounds
    pub fn check_bounds(
        &mut self,
        index: &RefinementExpr,
        array_len: &RefinementExpr,
        span: Option<Span>,
    ) -> bool {
        self.stats.constraints_generated += 1;

        let in_bounds = Predicate::and(
            Predicate::ge(index.clone(), RefinementExpr::Int(0)),
            Predicate::lt(index.clone(), array_len.clone()),
        );

        let context = self.env.bindings_as_predicates();
        let full_pred = if context.is_empty() {
            in_bounds.clone()
        } else {
            let ctx = context.into_iter().reduce(Predicate::and).unwrap();
            Predicate::implies(ctx, in_bounds.clone())
        };

        self.stats.solver_calls += 1;
        match self.solver.check_valid(&full_pred) {
            SmtResult::Unsat => {
                self.stats.constraints_verified += 1;
                true
            }
            SmtResult::Sat(model) => {
                self.stats.constraints_failed += 1;
                self.errors.push(RefinementError::bounds_check_failed(
                    format!("{}", index),
                    Some(format!("{}", array_len)),
                    model.map(|m| m.to_counterexample()),
                    span.clone(),
                ));
                false
            }
            _ => false,
        }
    }

    /// Check division by non-zero
    pub fn check_division(&mut self, divisor: &RefinementExpr, span: Option<Span>) -> bool {
        self.stats.constraints_generated += 1;

        let non_zero = Predicate::Compare {
            left: Box::new(divisor.clone()),
            op: super::syntax::CompareOp::Ne,
            right: Box::new(RefinementExpr::Int(0)),
        };

        let context = self.env.bindings_as_predicates();
        let full_pred = if context.is_empty() {
            non_zero.clone()
        } else {
            let ctx = context.into_iter().reduce(Predicate::and).unwrap();
            Predicate::implies(ctx, non_zero.clone())
        };

        self.stats.solver_calls += 1;
        match self.solver.check_valid(&full_pred) {
            SmtResult::Unsat => {
                self.stats.constraints_verified += 1;
                true
            }
            SmtResult::Sat(model) => {
                self.stats.constraints_failed += 1;
                self.errors.push(
                    RefinementError::new(RefinementErrorKind::DivisionByZeroPossible {
                        divisor_expr: format!("{}", divisor),
                        counterexample: model.map(|m| m.to_counterexample()),
                    })
                    .with_span(span.clone().unwrap_or_else(dummy_span)),
                );
                false
            }
            _ => false,
        }
    }

    /// Verify dose is within safe range
    pub fn check_dose_range(
        &mut self,
        computed_dose: &RefinementExpr,
        drug_name: Option<&str>,
        min_mg: f64,
        max_mg: f64,
        span: Option<Span>,
    ) -> bool {
        self.stats.constraints_generated += 1;

        let in_range = Predicate::and(
            Predicate::ge(computed_dose.clone(), RefinementExpr::Float(min_mg)),
            Predicate::le(computed_dose.clone(), RefinementExpr::Float(max_mg)),
        );

        let context = self.env.bindings_as_predicates();
        let full_pred = if context.is_empty() {
            in_range.clone()
        } else {
            let ctx = context.into_iter().reduce(Predicate::and).unwrap();
            Predicate::implies(ctx, in_range.clone())
        };

        self.stats.solver_calls += 1;
        match self.solver.check_valid(&full_pred) {
            SmtResult::Unsat => {
                self.stats.constraints_verified += 1;
                true
            }
            SmtResult::Sat(model) => {
                self.stats.constraints_failed += 1;
                self.errors.push(RefinementError::dose_range_violation(
                    drug_name.map(String::from),
                    format!("{}", computed_dose),
                    Some(format!("{} mg", min_mg)),
                    Some(format!("{} mg", max_mg)),
                    model.map(|m| m.to_counterexample()),
                    span.clone(),
                ));
                false
            }
            _ => false,
        }
    }

    /// Verify all constraints in a constraint set
    pub fn verify_constraints(&mut self, constraints: &ConstraintSet) -> bool {
        let mut all_valid = true;

        for constraint in constraints.constraints() {
            self.stats.constraints_generated += 1;
            let pred = constraint.to_predicate();

            self.stats.solver_calls += 1;
            match self.solver.check_valid(&pred) {
                SmtResult::Unsat => {
                    self.stats.constraints_verified += 1;
                }
                SmtResult::Sat(model) => {
                    self.stats.constraints_failed += 1;
                    all_valid = false;

                    let mut err = RefinementError::new(RefinementErrorKind::PredicateUnsatisfied {
                        expected: pred,
                        counterexample: model.map(|m| m.to_counterexample()),
                    });
                    if let Some(ref desc) = constraint.description {
                        err = err.with_note(desc);
                    }
                    self.errors.push(err);
                }
                SmtResult::Unknown(reason) => {
                    all_valid = false;
                    self.errors.push(
                        RefinementError::new(RefinementErrorKind::SolverTimeout {
                            constraint: constraint
                                .description
                                .clone()
                                .unwrap_or_else(|| "unknown".to_string()),
                            timeout_ms: 5000,
                        })
                        .with_note(reason),
                    );
                }
                SmtResult::Error(err) => {
                    all_valid = false;
                    self.errors
                        .push(RefinementError::new(RefinementErrorKind::SolverError(err)));
                }
            }
        }

        all_valid
    }

    /// Has any errors been recorded
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Finish checking and return errors
    pub fn finish(self) -> Result<VerificationStats, Vec<RefinementError>> {
        if self.errors.is_empty() {
            Ok(self.stats)
        } else {
            Err(self.errors)
        }
    }
}

impl Default for RefinementChecker {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Pre-built environments for medical computing
// ============================================================================

impl RefinementEnv {
    /// Create environment with common medical type aliases
    pub fn with_medical_types() -> Self {
        let mut env = Self::new();

        // Common medical refinement types
        env.register_alias("PositiveWeight", RefinementType::valid_weight("weight"));
        env.register_alias("ValidCrCl", RefinementType::valid_crcl("crcl"));
        env.register_alias("ValidAge", RefinementType::valid_age("age"));

        // Common dose ranges
        env.register_alias(
            "SafeMetforminDose",
            RefinementType::safe_dose_mg("dose", 500.0, 2000.0),
        );

        env
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_refinement_env() {
        let mut env = RefinementEnv::new();
        let ty = RefinementType::positive("x", BaseTypeRef::Named("Int".to_string()));
        env.bind("myvar", ty);

        assert!(env.lookup("myvar").is_some());
        assert!(env.lookup("unknown").is_none());
    }

    #[test]
    fn test_checker_creation() {
        let checker = RefinementChecker::new();
        assert!(checker.errors().is_empty());
    }

    #[test]
    fn test_medical_env() {
        let env = RefinementEnv::with_medical_types();
        assert!(env.resolve_alias("PositiveWeight").is_some());
        assert!(env.resolve_alias("ValidCrCl").is_some());
    }

    #[test]
    fn test_path_conditions() {
        let mut env = RefinementEnv::new();
        env.push_condition(Predicate::positive("x"));
        env.push_condition(Predicate::lt(
            RefinementExpr::var("x"),
            RefinementExpr::Int(10),
        ));

        assert_eq!(env.path_conditions().len(), 2);

        env.pop_condition();
        assert_eq!(env.path_conditions().len(), 1);
    }

    #[test]
    fn test_function_registration() {
        let mut env = RefinementEnv::new();

        let sig = FunctionRefinement {
            params: vec![("weight".to_string(), RefinementType::valid_weight("weight"))],
            return_type: RefinementType::positive("dose", BaseTypeRef::Named("mg".to_string())),
            requires: vec![],
            ensures: vec![],
        };

        env.register_function("calculate_dose", sig);
        assert!(env.lookup_function("calculate_dose").is_some());
    }

    // Integration test with SMT solver
    #[test]
    fn test_check_value_positive() {
        let mut checker = RefinementChecker::new();

        if !checker.solver.is_available() {
            println!("Z3 not available, skipping integration test");
            return;
        }

        // Check that 5 is positive
        let ty = RefinementType::positive("x", BaseTypeRef::Named("Int".to_string()));
        let value = RefinementExpr::Int(5);

        let result = checker.check_value(&value, &ty, None);
        assert!(result, "5 should be positive");
        assert!(!checker.has_errors());
    }

    #[test]
    fn test_check_value_negative_fails() {
        let mut checker = RefinementChecker::new();

        if !checker.solver.is_available() {
            println!("Z3 not available, skipping integration test");
            return;
        }

        // Check that -3 is NOT positive
        let ty = RefinementType::positive("x", BaseTypeRef::Named("Int".to_string()));
        let value = RefinementExpr::Int(-3);

        let result = checker.check_value(&value, &ty, None);
        assert!(!result, "-3 should not be positive");
        assert!(checker.has_errors());
    }
}
