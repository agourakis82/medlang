// Phase V2: SMT Verification Module
//
// This module provides Z3-based SMT verification for refinement type constraints.
// It translates MedLang constraints to SMT-LIB format and uses Z3 to prove
// or find counterexamples for verification conditions.

pub mod solver;
pub mod translator;
pub mod vc_gen;

pub use solver::{Z3Model, Z3Result, Z3Solver};
pub use translator::{SMTExpr, SMTFormula, SMTSort, SMTTranslator};
pub use vc_gen::{VCGenerator, VerificationCondition};

use std::collections::HashMap;

/// SMT Context containing the Z3 solver and all assertions
pub struct SMTContext {
    pub solver: Z3Solver,
    pub variables: HashMap<String, SMTSort>,
}

impl SMTContext {
    /// Create a new SMT context with Z3 solver
    pub fn new() -> Self {
        SMTContext {
            solver: Z3Solver::new(),
            variables: HashMap::new(),
        }
    }

    /// Declare a variable with its SMT sort
    pub fn declare_var(&mut self, name: String, sort: SMTSort) {
        self.variables.insert(name, sort);
    }

    /// Add an assertion (assumption) to the solver
    pub fn add_assertion(&mut self, name: &str, formula: SMTFormula) {
        self.solver.assert(formula);
    }

    /// Check a verification condition
    pub fn check(&mut self, vc: &VerificationCondition) -> Z3Result {
        self.solver.check_vc(vc)
    }

    /// Reset the solver (clear all assertions)
    pub fn reset(&mut self) {
        self.solver.reset();
    }
}

impl Default for SMTContext {
    fn default() -> Self {
        Self::new()
    }
}
