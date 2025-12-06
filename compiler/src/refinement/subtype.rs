//! Subtyping for Refinement Types
//!
//! Implements subtype checking for refined types using SMT solving.
//! For { x: T | P } <: { y: T | Q }, we check ∀x. P(x) ⟹ Q[y := x]

use std::collections::HashMap;

use super::error::Counterexample;
use super::smt::{SmtResult, SmtSolver};
use super::syntax::{Predicate, RefinementExpr, RefinementType};

/// Result of a subtyping check
#[derive(Clone, Debug)]
pub enum SubtypeResult {
    /// Subtyping holds
    Valid,
    /// Subtyping fails with counterexample and reason
    Invalid {
        counterexample: Option<Counterexample>,
        reason: String,
    },
    /// Solver couldn't determine
    Unknown(String),
}

impl SubtypeResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, SubtypeResult::Valid)
    }

    pub fn is_invalid(&self) -> bool {
        matches!(self, SubtypeResult::Invalid { .. })
    }
}

/// Aggregated results from multiple checks
#[derive(Clone, Debug, Default)]
pub struct CheckResults {
    pub valid: usize,
    pub invalid: usize,
    pub unknown: usize,
}

impl CheckResults {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, result: &SubtypeResult) {
        match result {
            SubtypeResult::Valid => self.valid += 1,
            SubtypeResult::Invalid { .. } => self.invalid += 1,
            SubtypeResult::Unknown(_) => self.unknown += 1,
        }
    }

    pub fn total(&self) -> usize {
        self.valid + self.invalid + self.unknown
    }

    pub fn all_valid(&self) -> bool {
        self.invalid == 0 && self.unknown == 0
    }
}

/// Subtype checker using SMT solving
pub struct SubtypeChecker {
    solver: SmtSolver,
    cache: HashMap<(String, String), SubtypeResult>,
    use_cache: bool,
}

impl SubtypeChecker {
    pub fn new() -> Self {
        Self {
            solver: SmtSolver::new(),
            cache: HashMap::new(),
            use_cache: true,
        }
    }

    pub fn with_solver(solver: SmtSolver) -> Self {
        Self {
            solver,
            cache: HashMap::new(),
            use_cache: true,
        }
    }

    pub fn disable_cache(&mut self) {
        self.use_cache = false;
        self.cache.clear();
    }

    /// Check if sub <: sup
    /// { x: T | P } <: { y: T | Q } iff ∀x. P(x) ⟹ Q[y := x]
    pub fn check(&mut self, sub: &RefinementType, sup: &RefinementType) -> SubtypeResult {
        self.check_with_context(sub, sup, &[])
    }

    /// Check subtyping with additional context predicates
    pub fn check_with_context(
        &mut self,
        sub: &RefinementType,
        sup: &RefinementType,
        context: &[Predicate],
    ) -> SubtypeResult {
        // Check cache
        if self.use_cache {
            let key = (format!("{:?}", sub), format!("{:?}", sup));
            if let Some(result) = self.cache.get(&key) {
                return result.clone();
            }
        }

        // Substitute supertype's binder with subtype's binder
        let sup_pred = sup
            .predicate
            .substitute(&sup.binder.name, &RefinementExpr::Var(sub.binder.clone()));

        // Build implication: (context ∧ sub_pred) ⟹ sup_pred
        let sub_pred = sub.predicate.clone();

        let antecedent = if context.is_empty() {
            sub_pred
        } else {
            let ctx = context
                .iter()
                .cloned()
                .reduce(Predicate::and)
                .unwrap_or(Predicate::Bool(true));
            Predicate::and(ctx, sub_pred)
        };

        let implication = Predicate::implies(antecedent, sup_pred);

        // Check validity: ∀x. implication
        // Equivalently, check if ¬implication is unsatisfiable
        let result = self.solver.check_valid(&implication);

        let subtype_result = match result {
            SmtResult::Unsat => {
                // Negation is unsat means implication is valid
                SubtypeResult::Valid
            }
            SmtResult::Sat(model) => {
                // Found counterexample
                SubtypeResult::Invalid {
                    counterexample: model.map(|m| m.to_counterexample()),
                    reason: format!(
                        "Refinement {} does not imply {}",
                        sub.predicate, sup.predicate
                    ),
                }
            }
            SmtResult::Unknown(reason) => SubtypeResult::Unknown(reason),
            SmtResult::Error(err) => SubtypeResult::Unknown(format!("Solver error: {}", err)),
        };

        // Cache result
        if self.use_cache {
            let key = (format!("{:?}", sub), format!("{:?}", sup));
            self.cache.insert(key, subtype_result.clone());
        }

        subtype_result
    }

    /// Check if a refinement is satisfiable (has at least one valid value)
    pub fn is_satisfiable(&mut self, ty: &RefinementType) -> bool {
        match self.solver.check_sat(&ty.predicate) {
            SmtResult::Sat(_) => true,
            _ => false,
        }
    }

    /// Check if a refinement is trivially true
    pub fn is_trivial(&self, ty: &RefinementType) -> bool {
        matches!(ty.predicate, Predicate::Bool(true))
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

impl Default for SubtypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::refinement::syntax::{BaseTypeRef, RefinedVar};

    fn make_positive_int() -> RefinementType {
        RefinementType::new(
            RefinedVar::new("x"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::positive("x"),
        )
    }

    fn make_non_negative_int() -> RefinementType {
        RefinementType::new(
            RefinedVar::new("x"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::non_negative("x"),
        )
    }

    #[test]
    fn test_subtype_checker_creation() {
        let checker = SubtypeChecker::new();
        assert!(checker.use_cache);
    }

    #[test]
    fn test_check_results() {
        let mut results = CheckResults::new();
        assert_eq!(results.total(), 0);

        results.add(&SubtypeResult::Valid);
        results.add(&SubtypeResult::Valid);
        results.add(&SubtypeResult::Invalid {
            counterexample: None,
            reason: "test".to_string(),
        });

        assert_eq!(results.valid, 2);
        assert_eq!(results.invalid, 1);
        assert_eq!(results.total(), 3);
        assert!(!results.all_valid());
    }

    #[test]
    fn test_trivial_refinement() {
        let checker = SubtypeChecker::new();
        let trivial =
            RefinementType::trivial(RefinedVar::new("x"), BaseTypeRef::Named("Int".to_string()));
        assert!(checker.is_trivial(&trivial));
    }
}
