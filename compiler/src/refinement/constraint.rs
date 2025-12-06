//! Constraint generation and management for refinement type checking
//!
//! This module handles the generation of SMT constraints from refinement
//! predicates and manages constraint sets during type checking.

use super::syntax::{ArithOp, CompareOp, Predicate, RefinementExpr};
use std::collections::{HashMap, HashSet};

/// The kind/category of a constraint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstraintKind {
    /// Subtype checking constraint (P => Q)
    Subtype,
    /// Range bound constraint (min <= x <= max)
    RangeBound,
    /// Positivity constraint (x > 0)
    Positivity,
    /// Non-negativity constraint (x >= 0)
    NonNegativity,
    /// Division safety (divisor != 0)
    DivisionSafety,
    /// Square root safety (arg >= 0)
    SqrtSafety,
    /// Logarithm safety (arg > 0)
    LogSafety,
    /// Custom/general constraint
    Custom,
}

/// A constraint to be checked by the SMT solver
#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    /// The predicate representing this constraint
    pub predicate: Predicate,

    /// The kind of constraint
    pub kind: ConstraintKind,

    /// Source location for error reporting
    pub source: Option<ConstraintSource>,

    /// Human-readable description
    pub description: Option<String>,
}

/// Information about where a constraint originated
#[derive(Debug, Clone, PartialEq)]
pub struct ConstraintSource {
    /// File path
    pub file: Option<String>,

    /// Line number
    pub line: Option<usize>,

    /// Column number
    pub column: Option<usize>,

    /// The expression that generated this constraint
    pub expr_text: Option<String>,
}

impl Constraint {
    /// Create a new constraint from a predicate
    pub fn new(predicate: Predicate) -> Self {
        Self {
            predicate,
            kind: ConstraintKind::Custom,
            source: None,
            description: None,
        }
    }

    /// Create a new constraint with a specific kind
    pub fn with_kind(predicate: Predicate, kind: ConstraintKind) -> Self {
        Self {
            predicate,
            kind,
            source: None,
            description: None,
        }
    }

    /// Create a constraint from a predicate with variable binding
    pub fn from_predicate(pred: &Predicate, var: &str) -> Self {
        // Replace the refinement variable with the actual variable name
        let bound_pred = pred.substitute(
            &pred.free_vars().iter().next().cloned().unwrap_or_default(),
            &Predicate::var(var),
        );
        Self::new(bound_pred)
    }

    /// Add source location information
    pub fn with_source(mut self, source: ConstraintSource) -> Self {
        self.source = Some(source);
        self
    }

    /// Add a description
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Get all variables referenced in this constraint
    pub fn variables(&self) -> HashSet<String> {
        self.predicate.free_vars()
    }

    /// Get the predicate for this constraint
    pub fn to_predicate(&self) -> Predicate {
        self.predicate.clone()
    }
}

/// A set of constraints accumulated during type checking
#[derive(Debug, Clone, Default)]
pub struct ConstraintSet {
    /// The constraints in this set
    constraints: Vec<Constraint>,

    /// Variable types (for SMT sort assignment)
    var_types: HashMap<String, VarSort>,
}

/// SMT sort for a variable
#[derive(Debug, Clone, PartialEq)]
pub enum VarSort {
    Real,
    Int,
    Bool,
}

impl ConstraintSet {
    /// Create an empty constraint set
    pub fn new() -> Self {
        Self {
            constraints: Vec::new(),
            var_types: HashMap::new(),
        }
    }

    /// Add a constraint to the set
    pub fn add(&mut self, constraint: Constraint) {
        self.constraints.push(constraint);
    }

    /// Add a predicate directly as a constraint
    pub fn add_predicate(&mut self, pred: Predicate) {
        self.constraints.push(Constraint::new(pred));
    }

    /// Declare a variable with its sort
    pub fn declare_var(&mut self, name: impl Into<String>, sort: VarSort) {
        self.var_types.insert(name.into(), sort);
    }

    /// Get the sort of a variable (defaults to Real)
    pub fn var_sort(&self, name: &str) -> VarSort {
        self.var_types.get(name).cloned().unwrap_or(VarSort::Real)
    }

    /// Get all constraints
    pub fn constraints(&self) -> &[Constraint] {
        &self.constraints
    }

    /// Get all variables in the constraint set
    pub fn all_variables(&self) -> HashSet<String> {
        let mut vars = HashSet::new();
        for constraint in &self.constraints {
            vars.extend(constraint.variables());
        }
        vars
    }

    /// Convert the constraint set to a single predicate (conjunction of all)
    pub fn to_predicate(&self) -> Predicate {
        if self.constraints.is_empty() {
            return Predicate::bool_literal(true);
        }

        let mut result = self.constraints[0].predicate.clone();
        for constraint in &self.constraints[1..] {
            result = Predicate::and(result, constraint.predicate.clone());
        }
        result
    }

    /// Check if the constraint set is empty
    pub fn is_empty(&self) -> bool {
        self.constraints.is_empty()
    }

    /// Get the number of constraints
    pub fn len(&self) -> usize {
        self.constraints.len()
    }

    /// Merge another constraint set into this one
    pub fn merge(&mut self, other: ConstraintSet) {
        self.constraints.extend(other.constraints);
        self.var_types.extend(other.var_types);
    }

    /// Create a new scope (for let bindings, etc.)
    pub fn push_scope(&self) -> Self {
        self.clone()
    }
}

/// Generator for constraints from MedLang expressions
#[derive(Debug)]
pub struct ConstraintGenerator {
    /// Current constraint set being built
    constraints: ConstraintSet,

    /// Counter for generating fresh variable names
    fresh_counter: usize,

    /// Stack of scopes for variable bindings
    scopes: Vec<HashMap<String, Predicate>>,
}

impl ConstraintGenerator {
    /// Create a new constraint generator
    pub fn new() -> Self {
        Self {
            constraints: ConstraintSet::new(),
            fresh_counter: 0,
            scopes: vec![HashMap::new()],
        }
    }

    /// Generate a fresh variable name
    pub fn fresh_var(&mut self, prefix: &str) -> String {
        let name = format!("{}_{}", prefix, self.fresh_counter);
        self.fresh_counter += 1;
        name
    }

    /// Enter a new scope
    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    /// Exit the current scope
    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    /// Bind a variable in the current scope
    pub fn bind(&mut self, name: impl Into<String>, value: Predicate) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.into(), value);
        }
    }

    /// Look up a variable in all scopes
    pub fn lookup(&self, name: &str) -> Option<&Predicate> {
        for scope in self.scopes.iter().rev() {
            if let Some(pred) = scope.get(name) {
                return Some(pred);
            }
        }
        None
    }

    /// Add a constraint that a value satisfies a refinement
    pub fn require_refinement(
        &mut self,
        value: &RefinementExpr,
        refinement: &Predicate,
        var_name: &str,
    ) {
        // Substitute the refinement variable with the actual value
        let instantiated = refinement.substitute(var_name, value);
        self.constraints.add_predicate(instantiated);
    }

    /// Add a constraint for subtyping (sub <: sup)
    pub fn require_subtype(&mut self, sub_pred: &Predicate, sup_pred: &Predicate, var_name: &str) {
        // For subtyping, we need: sub_pred => sup_pred
        // The SMT check will verify this implication holds
        let implication = Predicate::implies(sub_pred.clone(), sup_pred.clone());
        self.constraints.add(
            Constraint::new(implication)
                .with_description(format!("Subtype check: {} ⊆ {}", sub_pred, sup_pred)),
        );
    }

    /// Add an assumption (e.g., from a function parameter)
    pub fn assume(&mut self, pred: Predicate) {
        self.constraints.add_predicate(pred);
    }

    /// Declare a variable with a sort
    pub fn declare(&mut self, name: impl Into<String>, sort: VarSort) {
        self.constraints.declare_var(name, sort);
    }

    /// Get the accumulated constraints
    pub fn into_constraints(self) -> ConstraintSet {
        self.constraints
    }

    /// Get a reference to the current constraints
    pub fn constraints(&self) -> &ConstraintSet {
        &self.constraints
    }

    // =========================================================================
    // Constraint generation for specific patterns
    // =========================================================================

    /// Generate constraints for a range check (min <= x <= max)
    pub fn in_range(&mut self, var: &str, min: f64, max: f64) {
        let v = Predicate::var(var);
        let constraint = Predicate::and(
            Predicate::ge(v.clone(), Predicate::float(min)),
            Predicate::le(v, Predicate::float(max)),
        );
        self.constraints.add_predicate(constraint);
    }

    /// Generate constraints for positivity (x > 0)
    pub fn positive(&mut self, var: &str) {
        let constraint = Predicate::gt(Predicate::var(var), Predicate::float(0.0));
        self.constraints.add_predicate(constraint);
    }

    /// Generate constraints for non-negativity (x >= 0)
    pub fn non_negative(&mut self, var: &str) {
        let constraint = Predicate::ge(Predicate::var(var), Predicate::float(0.0));
        self.constraints.add_predicate(constraint);
    }

    /// Generate constraints for a probability (0 <= x <= 1)
    pub fn probability(&mut self, var: &str) {
        self.in_range(var, 0.0, 1.0);
    }

    /// Generate division safety constraint (divisor != 0)
    pub fn div_safe(&mut self, divisor: &RefinementExpr) {
        let constraint = Predicate::ne(divisor.clone(), Predicate::float(0.0));
        self.constraints.add(
            Constraint::with_kind(constraint, ConstraintKind::DivisionSafety)
                .with_description("Division by zero check".to_string()),
        );
    }

    /// Generate sqrt safety constraint (argument >= 0)
    pub fn sqrt_safe(&mut self, arg: &RefinementExpr) {
        let constraint = Predicate::ge(arg.clone(), Predicate::float(0.0));
        self.constraints.add(
            Constraint::with_kind(constraint, ConstraintKind::SqrtSafety)
                .with_description("Square root of non-negative check".to_string()),
        );
    }

    /// Generate log safety constraint (argument > 0)
    pub fn log_safe(&mut self, arg: &RefinementExpr) {
        let constraint = Predicate::gt(arg.clone(), Predicate::float(0.0));
        self.constraints.add(
            Constraint::with_kind(constraint, ConstraintKind::LogSafety)
                .with_description("Logarithm of positive check".to_string()),
        );
    }
}

impl Default for ConstraintGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Weakest precondition calculation for refinement type checking
pub struct WeakestPrecondition;

impl WeakestPrecondition {
    /// Calculate wp(stmt, Q) - the weakest precondition for Q to hold after stmt
    pub fn wp_assign(var: &str, expr: &RefinementExpr, postcondition: &Predicate) -> Predicate {
        // wp(x := e, Q) = Q[e/x]
        postcondition.substitute(var, expr)
    }

    /// wp for sequential composition
    pub fn wp_seq(wp1: Predicate, wp2: Predicate) -> Predicate {
        Predicate::and(wp1, wp2)
    }

    /// wp for conditional
    pub fn wp_if(cond: &Predicate, wp_then: Predicate, wp_else: Predicate) -> Predicate {
        // wp(if c then s1 else s2, Q) = (c => wp(s1,Q)) && (!c => wp(s2,Q))
        Predicate::and(
            Predicate::implies(cond.clone(), wp_then),
            Predicate::implies(Predicate::not(cond.clone()), wp_else),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constraint_set_to_predicate() {
        let mut cs = ConstraintSet::new();
        cs.add_predicate(Predicate::gt(Predicate::var("x"), Predicate::float(0.0)));
        cs.add_predicate(Predicate::lt(Predicate::var("x"), Predicate::float(10.0)));

        let pred = cs.to_predicate();
        let s = pred.to_string();
        assert!(s.contains("&&"));
        assert!(s.contains("x"));
    }

    #[test]
    fn test_constraint_generator_fresh_var() {
        let mut gen = ConstraintGenerator::new();
        let v1 = gen.fresh_var("tmp");
        let v2 = gen.fresh_var("tmp");
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_constraint_generator_scope() {
        let mut gen = ConstraintGenerator::new();
        // Use Bool predicates for testing scope binding
        gen.bind("x", Predicate::Bool(true));

        gen.push_scope();
        gen.bind("x", Predicate::Bool(false)); // Shadow outer x
        assert_eq!(gen.lookup("x"), Some(&Predicate::Bool(false)));
        gen.pop_scope();

        assert_eq!(gen.lookup("x"), Some(&Predicate::Bool(true)));
    }

    #[test]
    fn test_weakest_precondition_assign() {
        // wp(x := 5, x > 0) = 5 > 0 = true
        let postcond = Predicate::gt(Predicate::var("x"), Predicate::float(0.0));
        let expr = RefinementExpr::Float(5.0);
        let wp = WeakestPrecondition::wp_assign("x", &expr, &postcond);

        // After substitution, we should have 5 > 0
        // The result should be a comparison predicate
        match wp {
            Predicate::Compare {
                op: CompareOp::Gt,
                ref left,
                ref right,
            } => {
                assert!(matches!(**left, RefinementExpr::Float(5.0)));
                assert!(matches!(**right, RefinementExpr::Float(0.0)));
            }
            _ => panic!("Wrong predicate structure after wp: {:?}", wp),
        }
    }
}
