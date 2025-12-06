// Week 52: Parametric Polymorphism - Unification
//
// This module implements type unification, the core algorithm for
// type inference with polymorphism.
//
// Unification finds the most general substitution that makes two types equal.

use super::types::{Subst, Type, TypeVar, TypeVarId};
use std::collections::HashSet;
use thiserror::Error;

// =============================================================================
// Unification Errors
// =============================================================================

#[derive(Debug, Clone, Error, PartialEq)]
pub enum UnifyError {
    #[error("cannot unify {0} with {1}")]
    CannotUnify(String, String),

    #[error("infinite type: {var} occurs in {ty}")]
    InfiniteType { var: String, ty: String },

    #[error("arity mismatch: expected {expected}, found {found}")]
    ArityMismatch { expected: usize, found: usize },

    #[error("field mismatch: record types have different fields")]
    FieldMismatch {
        missing: Vec<String>,
        extra: Vec<String>,
    },

    #[error("constructor mismatch: {expected} vs {found}")]
    ConstructorMismatch { expected: String, found: String },
}

// =============================================================================
// Unification Algorithm
// =============================================================================

/// Unify two types, returning a substitution that makes them equal
///
/// The unification algorithm follows Robinson's algorithm:
/// 1. If both types are identical, return empty substitution
/// 2. If one is a type variable, bind it (with occurs check)
/// 3. If both are the same type constructor, unify arguments
/// 4. Otherwise, fail
pub fn unify(t1: &Type, t2: &Type) -> Result<Subst, UnifyError> {
    match (t1, t2) {
        // Identical types
        (Type::Int, Type::Int)
        | (Type::Float, Type::Float)
        | (Type::Bool, Type::Bool)
        | (Type::String, Type::String)
        | (Type::Unit, Type::Unit)
        | (Type::Model, Type::Model)
        | (Type::Protocol, Type::Protocol)
        | (Type::Policy, Type::Policy)
        | (Type::EvidenceProgram, Type::EvidenceProgram)
        | (Type::EvidenceResult, Type::EvidenceResult)
        | (Type::SimulationResult, Type::SimulationResult)
        | (Type::FitResult, Type::FitResult)
        | (Type::SurrogateModel, Type::SurrogateModel)
        | (Type::RLPolicy, Type::RLPolicy)
        | (Type::Dual, Type::Dual)
        | (Type::DualVec, Type::DualVec)
        | (Type::DualRec, Type::DualRec)
        | (Type::Error, Type::Error) => Ok(Subst::new()),

        // Type variable on left: bind it
        (Type::Var(tv), other) => unify_var(tv, other),

        // Type variable on right: bind it
        (other, Type::Var(tv)) => unify_var(tv, other),

        // Function types: unify params and return type
        (
            Type::Function {
                params: p1,
                ret: r1,
            },
            Type::Function {
                params: p2,
                ret: r2,
            },
        ) => {
            if p1.len() != p2.len() {
                return Err(UnifyError::ArityMismatch {
                    expected: p1.len(),
                    found: p2.len(),
                });
            }

            let mut subst = Subst::new();

            // Unify parameters
            for (a, b) in p1.iter().zip(p2.iter()) {
                let s = unify(&subst.apply(a), &subst.apply(b))?;
                subst.extend(&s);
            }

            // Unify return types
            let s = unify(&subst.apply(r1), &subst.apply(r2))?;
            subst.extend(&s);

            Ok(subst)
        }

        // Record types: unify field by field
        (Type::Record(f1), Type::Record(f2)) => {
            let keys1: HashSet<_> = f1.keys().collect();
            let keys2: HashSet<_> = f2.keys().collect();

            let missing: Vec<_> = keys1.difference(&keys2).map(|s| (*s).clone()).collect();
            let extra: Vec<_> = keys2.difference(&keys1).map(|s| (*s).clone()).collect();

            if !missing.is_empty() || !extra.is_empty() {
                return Err(UnifyError::FieldMismatch { missing, extra });
            }

            let mut subst = Subst::new();
            for key in keys1 {
                let t1 = &f1[key];
                let t2 = &f2[key];
                let s = unify(&subst.apply(t1), &subst.apply(t2))?;
                subst.extend(&s);
            }

            Ok(subst)
        }

        // List types
        (Type::List(e1), Type::List(e2)) => unify(e1, e2),

        // Option types
        (Type::Option(i1), Type::Option(i2)) => unify(i1, i2),

        // Result types
        (Type::Result { ok: o1, err: e1 }, Type::Result { ok: o2, err: e2 }) => {
            let s1 = unify(o1, o2)?;
            let s2 = unify(&s1.apply(e1), &s1.apply(e2))?;
            Ok(s1.compose(&s2))
        }

        // Tuple types
        (Type::Tuple(elems1), Type::Tuple(elems2)) => {
            if elems1.len() != elems2.len() {
                return Err(UnifyError::ArityMismatch {
                    expected: elems1.len(),
                    found: elems2.len(),
                });
            }

            let mut subst = Subst::new();
            for (e1, e2) in elems1.iter().zip(elems2.iter()) {
                let s = unify(&subst.apply(e1), &subst.apply(e2))?;
                subst.extend(&s);
            }

            Ok(subst)
        }

        // Named types (must be exact match)
        (Type::Named(n1), Type::Named(n2)) => {
            if n1 == n2 {
                Ok(Subst::new())
            } else {
                Err(UnifyError::CannotUnify(n1.clone(), n2.clone()))
            }
        }

        // Generic type applications
        (
            Type::App {
                constructor: c1,
                args: a1,
            },
            Type::App {
                constructor: c2,
                args: a2,
            },
        ) => {
            if c1 != c2 {
                return Err(UnifyError::ConstructorMismatch {
                    expected: c1.clone(),
                    found: c2.clone(),
                });
            }

            if a1.len() != a2.len() {
                return Err(UnifyError::ArityMismatch {
                    expected: a1.len(),
                    found: a2.len(),
                });
            }

            let mut subst = Subst::new();
            for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                let s = unify(&subst.apply(arg1), &subst.apply(arg2))?;
                subst.extend(&s);
            }

            Ok(subst)
        }

        // Cannot unify
        (t1, t2) => Err(UnifyError::CannotUnify(t1.to_string(), t2.to_string())),
    }
}

/// Unify a type variable with a type
fn unify_var(tv: &TypeVar, ty: &Type) -> Result<Subst, UnifyError> {
    // If the type is the same variable, nothing to do
    if let Type::Var(other_tv) = ty {
        if tv.id == other_tv.id {
            return Ok(Subst::new());
        }
    }

    // Occurs check: ensure tv doesn't appear in ty (would create infinite type)
    if occurs_in(&tv.id, ty) {
        return Err(UnifyError::InfiniteType {
            var: tv.display_name(),
            ty: ty.to_string(),
        });
    }

    // Bind the variable
    Ok(Subst::singleton(tv.id, ty.clone()))
}

/// Check if a type variable occurs in a type (for infinite type detection)
fn occurs_in(var: &TypeVarId, ty: &Type) -> bool {
    match ty {
        Type::Var(tv) => tv.id == *var,
        Type::Function { params, ret } => {
            params.iter().any(|p| occurs_in(var, p)) || occurs_in(var, ret)
        }
        Type::Record(fields) => fields.values().any(|t| occurs_in(var, t)),
        Type::List(elem) => occurs_in(var, elem),
        Type::Option(inner) => occurs_in(var, inner),
        Type::Result { ok, err } => occurs_in(var, ok) || occurs_in(var, err),
        Type::Tuple(elems) => elems.iter().any(|e| occurs_in(var, e)),
        Type::App { args, .. } => args.iter().any(|a| occurs_in(var, a)),
        _ => false,
    }
}

// =============================================================================
// Unification of Multiple Constraints
// =============================================================================

/// A constraint is a pair of types that should be unified
#[derive(Debug, Clone)]
pub struct Constraint {
    pub left: Type,
    pub right: Type,
}

impl Constraint {
    pub fn new(left: Type, right: Type) -> Self {
        Self { left, right }
    }
}

/// Solve a set of constraints, returning a substitution that satisfies all of them
pub fn solve_constraints(constraints: &[Constraint]) -> Result<Subst, UnifyError> {
    let mut subst = Subst::new();

    for constraint in constraints {
        let left = subst.apply(&constraint.left);
        let right = subst.apply(&constraint.right);
        let s = unify(&left, &right)?;
        subst.extend(&s);
    }

    Ok(subst)
}

// =============================================================================
// Type Matching (One-way Unification)
// =============================================================================

/// Match a pattern type against a concrete type (one-way unification)
///
/// Unlike full unification, matching only binds variables in the pattern,
/// not in the target type. This is useful for instantiation.
pub fn match_types(pattern: &Type, target: &Type) -> Result<Subst, UnifyError> {
    match (pattern, target) {
        // Type variable in pattern: bind it
        (Type::Var(tv), _) => {
            if occurs_in(&tv.id, target) {
                Err(UnifyError::InfiniteType {
                    var: tv.display_name(),
                    ty: target.to_string(),
                })
            } else {
                Ok(Subst::singleton(tv.id, target.clone()))
            }
        }

        // Both are identical ground types
        (t1, t2) if !t1.has_type_vars() && t1 == t2 => Ok(Subst::new()),

        // Function types
        (
            Type::Function {
                params: p1,
                ret: r1,
            },
            Type::Function {
                params: p2,
                ret: r2,
            },
        ) => {
            if p1.len() != p2.len() {
                return Err(UnifyError::ArityMismatch {
                    expected: p1.len(),
                    found: p2.len(),
                });
            }

            let mut subst = Subst::new();
            for (a, b) in p1.iter().zip(p2.iter()) {
                let s = match_types(&subst.apply(a), b)?;
                subst.extend(&s);
            }

            let s = match_types(&subst.apply(r1), r2)?;
            subst.extend(&s);

            Ok(subst)
        }

        // List types
        (Type::List(e1), Type::List(e2)) => match_types(e1, e2),

        // Option types
        (Type::Option(i1), Type::Option(i2)) => match_types(i1, i2),

        // Result types
        (Type::Result { ok: o1, err: e1 }, Type::Result { ok: o2, err: e2 }) => {
            let s1 = match_types(o1, o2)?;
            let s2 = match_types(&s1.apply(e1), e2)?;
            Ok(s1.compose(&s2))
        }

        // Tuple types
        (Type::Tuple(elems1), Type::Tuple(elems2)) => {
            if elems1.len() != elems2.len() {
                return Err(UnifyError::ArityMismatch {
                    expected: elems1.len(),
                    found: elems2.len(),
                });
            }

            let mut subst = Subst::new();
            for (e1, e2) in elems1.iter().zip(elems2.iter()) {
                let s = match_types(&subst.apply(e1), e2)?;
                subst.extend(&s);
            }

            Ok(subst)
        }

        // Record types
        (Type::Record(f1), Type::Record(f2)) => {
            let keys1: HashSet<_> = f1.keys().collect();
            let keys2: HashSet<_> = f2.keys().collect();

            if keys1 != keys2 {
                let missing: Vec<_> = keys1.difference(&keys2).map(|s| (*s).clone()).collect();
                let extra: Vec<_> = keys2.difference(&keys1).map(|s| (*s).clone()).collect();
                return Err(UnifyError::FieldMismatch { missing, extra });
            }

            let mut subst = Subst::new();
            for key in keys1 {
                let s = match_types(&subst.apply(&f1[key]), &f2[key])?;
                subst.extend(&s);
            }

            Ok(subst)
        }

        // Generic applications
        (
            Type::App {
                constructor: c1,
                args: a1,
            },
            Type::App {
                constructor: c2,
                args: a2,
            },
        ) => {
            if c1 != c2 {
                return Err(UnifyError::ConstructorMismatch {
                    expected: c1.clone(),
                    found: c2.clone(),
                });
            }

            if a1.len() != a2.len() {
                return Err(UnifyError::ArityMismatch {
                    expected: a1.len(),
                    found: a2.len(),
                });
            }

            let mut subst = Subst::new();
            for (arg1, arg2) in a1.iter().zip(a2.iter()) {
                let s = match_types(&subst.apply(arg1), arg2)?;
                subst.extend(&s);
            }

            Ok(subst)
        }

        // Cannot match
        (t1, t2) => Err(UnifyError::CannotUnify(t1.to_string(), t2.to_string())),
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unify_identical() {
        let s = unify(&Type::Int, &Type::Int).unwrap();
        assert!(s.is_empty());
    }

    #[test]
    fn test_unify_var_to_concrete() {
        let var = Type::var(0);
        let s = unify(&var, &Type::Int).unwrap();

        assert_eq!(s.apply(&var), Type::Int);
    }

    #[test]
    fn test_unify_concrete_to_var() {
        let var = Type::var(0);
        let s = unify(&Type::String, &var).unwrap();

        assert_eq!(s.apply(&var), Type::String);
    }

    #[test]
    fn test_unify_function_types() {
        // (T, Int) -> T  unified with  (String, Int) -> String
        let t1 = Type::function(vec![Type::var(0), Type::Int], Type::var(0));
        let t2 = Type::function(vec![Type::String, Type::Int], Type::String);

        let s = unify(&t1, &t2).unwrap();
        assert_eq!(s.apply(&Type::var(0)), Type::String);
    }

    #[test]
    fn test_unify_two_vars() {
        // T unified with U
        let t1 = Type::var(0);
        let t2 = Type::var(1);

        let s = unify(&t1, &t2).unwrap();
        // One should be mapped to the other
        let applied1 = s.apply(&t1);
        let applied2 = s.apply(&t2);
        assert_eq!(applied1, applied2);
    }

    #[test]
    fn test_unify_list() {
        let t1 = Type::list(Type::var(0));
        let t2 = Type::list(Type::Int);

        let s = unify(&t1, &t2).unwrap();
        assert_eq!(s.apply(&Type::var(0)), Type::Int);
    }

    #[test]
    fn test_unify_failure() {
        let result = unify(&Type::Int, &Type::String);
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_arity_mismatch() {
        let t1 = Type::function(vec![Type::Int], Type::Bool);
        let t2 = Type::function(vec![Type::Int, Type::String], Type::Bool);

        let result = unify(&t1, &t2);
        assert!(matches!(result, Err(UnifyError::ArityMismatch { .. })));
    }

    #[test]
    fn test_occurs_check() {
        // T unified with [T] would create infinite type
        let t1 = Type::var(0);
        let t2 = Type::list(Type::var(0));

        let result = unify(&t1, &t2);
        assert!(matches!(result, Err(UnifyError::InfiniteType { .. })));
    }

    #[test]
    fn test_solve_constraints() {
        let constraints = vec![
            Constraint::new(Type::var(0), Type::Int),
            Constraint::new(Type::var(1), Type::list(Type::var(0))),
        ];

        let s = solve_constraints(&constraints).unwrap();

        assert_eq!(s.apply(&Type::var(0)), Type::Int);
        assert_eq!(s.apply(&Type::var(1)), Type::list(Type::Int));
    }

    #[test]
    fn test_match_types() {
        // Pattern: (T, T) -> T
        // Target: (Int, Int) -> Int
        let pattern = Type::function(vec![Type::var(0), Type::var(0)], Type::var(0));
        let target = Type::function(vec![Type::Int, Type::Int], Type::Int);

        let s = match_types(&pattern, &target).unwrap();
        assert_eq!(s.apply(&Type::var(0)), Type::Int);
    }

    #[test]
    fn test_match_types_failure() {
        // Pattern: (T, T) -> T
        // Target: (Int, String) -> Int  -- T can't be both Int and String
        let pattern = Type::function(vec![Type::var(0), Type::var(0)], Type::var(0));
        let target = Type::function(vec![Type::Int, Type::String], Type::Int);

        let result = match_types(&pattern, &target);
        assert!(result.is_err());
    }

    #[test]
    fn test_unify_record() {
        use std::collections::HashMap;

        let mut f1 = HashMap::new();
        f1.insert("x".to_string(), Type::var(0));
        f1.insert("y".to_string(), Type::Int);

        let mut f2 = HashMap::new();
        f2.insert("x".to_string(), Type::Float);
        f2.insert("y".to_string(), Type::Int);

        let t1 = Type::Record(f1);
        let t2 = Type::Record(f2);

        let s = unify(&t1, &t2).unwrap();
        assert_eq!(s.apply(&Type::var(0)), Type::Float);
    }

    #[test]
    fn test_unify_result_type() {
        let t1 = Type::result(Type::var(0), Type::var(1));
        let t2 = Type::result(Type::Int, Type::String);

        let s = unify(&t1, &t2).unwrap();
        assert_eq!(s.apply(&Type::var(0)), Type::Int);
        assert_eq!(s.apply(&Type::var(1)), Type::String);
    }

    #[test]
    fn test_unify_app() {
        let t1 = Type::app("Vec", vec![Type::var(0)]);
        let t2 = Type::app("Vec", vec![Type::Int]);

        let s = unify(&t1, &t2).unwrap();
        assert_eq!(s.apply(&Type::var(0)), Type::Int);
    }

    #[test]
    fn test_unify_app_constructor_mismatch() {
        let t1 = Type::app("Vec", vec![Type::Int]);
        let t2 = Type::app("Set", vec![Type::Int]);

        let result = unify(&t1, &t2);
        assert!(matches!(
            result,
            Err(UnifyError::ConstructorMismatch { .. })
        ));
    }
}
