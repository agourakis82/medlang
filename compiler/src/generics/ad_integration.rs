// Week 52: Generics-AD Integration
//
// This module bridges the parametric polymorphism system with automatic
// differentiation, enabling generic differentiable functions.
//
// ## Key Concepts
//
// 1. **Differentiable Bound**: Types that support automatic differentiation
//    - Float: Standard floating-point numbers
//    - Dual: Forward-mode dual numbers (primal + tangent)
//    - DualVec: Vector of dual numbers for multi-variable gradients
//    - DualRec: Named parameter gradients
//
// 2. **Generic AD Functions**: Functions like `grad`, `jacobian`, `hessian`
//    that work over any Differentiable type
//
// 3. **Monomorphization for AD**: Specializing generic functions for
//    specific numeric types (Float vs Dual)

use super::infer::TypeEnv;
use super::types::{PolyType, Subst, Type, TypeBound, TypeParam, TypeVarGen, TypeVarId};
use std::collections::HashMap;

// =============================================================================
// Differentiable Type Recognition
// =============================================================================

/// Check if a type is differentiable (supports AD operations)
pub fn is_differentiable(ty: &Type) -> bool {
    matches!(ty, Type::Float | Type::Dual | Type::DualVec | Type::DualRec)
}

/// Check if a type satisfies the Num bound
pub fn satisfies_num(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Int | Type::Float | Type::Dual | Type::DualVec | Type::DualRec
    )
}

/// Check if a type satisfies a given bound
pub fn satisfies_bound(ty: &Type, bound: &TypeBound) -> bool {
    match bound {
        TypeBound::Num => satisfies_num(ty),
        TypeBound::Ord => matches!(ty, Type::Int | Type::Float | Type::String),
        TypeBound::Eq => true, // All types support equality
        TypeBound::Differentiable => is_differentiable(ty),
        TypeBound::Copy => !matches!(ty, Type::Function { .. }), // Functions aren't Copy
        TypeBound::Trait(name) => {
            // Custom trait bounds - check known implementations
            match name.as_str() {
                "Debug" | "Clone" => true,
                "Default" => matches!(
                    ty,
                    Type::Int
                        | Type::Float
                        | Type::Bool
                        | Type::String
                        | Type::Unit
                        | Type::List(_)
                ),
                _ => false,
            }
        }
    }
}

/// Check if all bounds are satisfied
pub fn check_bounds(ty: &Type, bounds: &[TypeBound]) -> Result<(), BoundError> {
    for bound in bounds {
        if !satisfies_bound(ty, bound) {
            return Err(BoundError::NotSatisfied {
                ty: ty.to_string(),
                bound: bound.to_string(),
            });
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum BoundError {
    NotSatisfied { ty: String, bound: String },
}

impl std::fmt::Display for BoundError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BoundError::NotSatisfied { ty, bound } => {
                write!(f, "type {} does not satisfy bound {}", ty, bound)
            }
        }
    }
}

// =============================================================================
// AD-Specific Generic Functions
// =============================================================================

/// Generate AD-specific polymorphic function signatures
pub fn ad_poly_signatures(gen: &mut TypeVarGen) -> HashMap<String, PolyType> {
    let mut sigs = HashMap::new();

    // grad<T: Differentiable>(f: (T) -> T, x: T) -> T
    // Computes gradient of f at x
    let t_grad = gen.fresh();
    sigs.insert(
        "grad".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_grad.id.0).with_bound(TypeBound::Differentiable)],
            Type::function(
                vec![
                    Type::function(vec![Type::Var(t_grad.clone())], Type::Var(t_grad.clone())),
                    Type::Var(t_grad.clone()),
                ],
                Type::Var(t_grad),
            ),
        ),
    );

    // jacobian<T: Differentiable>(f: ([T]) -> [T], x: [T]) -> [[T]]
    // Computes Jacobian matrix
    let t_jac = gen.fresh();
    sigs.insert(
        "jacobian".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_jac.id.0).with_bound(TypeBound::Differentiable)],
            Type::function(
                vec![
                    Type::function(
                        vec![Type::list(Type::Var(t_jac.clone()))],
                        Type::list(Type::Var(t_jac.clone())),
                    ),
                    Type::list(Type::Var(t_jac.clone())),
                ],
                Type::list(Type::list(Type::Var(t_jac))),
            ),
        ),
    );

    // hessian<T: Differentiable>(f: ([T]) -> T, x: [T]) -> [[T]]
    // Computes Hessian matrix (second derivatives)
    let t_hess = gen.fresh();
    sigs.insert(
        "hessian".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_hess.id.0).with_bound(TypeBound::Differentiable)],
            Type::function(
                vec![
                    Type::function(
                        vec![Type::list(Type::Var(t_hess.clone()))],
                        Type::Var(t_hess.clone()),
                    ),
                    Type::list(Type::Var(t_hess.clone())),
                ],
                Type::list(Type::list(Type::Var(t_hess))),
            ),
        ),
    );

    // directional_deriv<T: Differentiable>(f: ([T]) -> T, x: [T], v: [T]) -> T
    // Computes directional derivative
    let t_dir = gen.fresh();
    sigs.insert(
        "directional_deriv".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_dir.id.0).with_bound(TypeBound::Differentiable)],
            Type::function(
                vec![
                    Type::function(
                        vec![Type::list(Type::Var(t_dir.clone()))],
                        Type::Var(t_dir.clone()),
                    ),
                    Type::list(Type::Var(t_dir.clone())),
                    Type::list(Type::Var(t_dir.clone())),
                ],
                Type::Var(t_dir),
            ),
        ),
    );

    // dual<T: Num>(primal: T, tangent: T) -> Dual
    // Creates a dual number from primal and tangent
    let t_dual = gen.fresh();
    sigs.insert(
        "dual".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_dual.id.0).with_bound(TypeBound::Num)],
            Type::function(
                vec![Type::Var(t_dual.clone()), Type::Var(t_dual)],
                Type::Dual,
            ),
        ),
    );

    // primal<T: Differentiable>(x: T) -> Float
    // Extracts the primal value
    let t_prim = gen.fresh();
    sigs.insert(
        "primal".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_prim.id.0).with_bound(TypeBound::Differentiable)],
            Type::function(vec![Type::Var(t_prim)], Type::Float),
        ),
    );

    // tangent<T: Differentiable>(x: T) -> Float
    // Extracts the tangent (derivative) value
    let t_tan = gen.fresh();
    sigs.insert(
        "tangent".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_tan.id.0).with_bound(TypeBound::Differentiable)],
            Type::function(vec![Type::Var(t_tan)], Type::Float),
        ),
    );

    // lift<T: Num>(x: Float) -> T
    // Lifts a float to any numeric type (constant lifting)
    let t_lift = gen.fresh();
    sigs.insert(
        "lift".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_lift.id.0).with_bound(TypeBound::Num)],
            Type::function(vec![Type::Float], Type::Var(t_lift)),
        ),
    );

    // sensitivity<T: Differentiable>(f: (T) -> T, x: T, delta: Float) -> T
    // Computes sensitivity analysis
    let t_sens = gen.fresh();
    sigs.insert(
        "sensitivity".to_string(),
        PolyType::new(
            vec![TypeParam::new("T", t_sens.id.0).with_bound(TypeBound::Differentiable)],
            Type::function(
                vec![
                    Type::function(vec![Type::Var(t_sens.clone())], Type::Var(t_sens.clone())),
                    Type::Var(t_sens.clone()),
                    Type::Float,
                ],
                Type::Var(t_sens),
            ),
        ),
    );

    sigs
}

// =============================================================================
// AD Type Promotion
// =============================================================================

/// Determine the promoted type for AD operations
/// When mixing Float and Dual, promote to Dual
pub fn promote_for_ad(t1: &Type, t2: &Type) -> Type {
    match (t1, t2) {
        (Type::Dual, _) | (_, Type::Dual) => Type::Dual,
        (Type::DualVec, _) | (_, Type::DualVec) => Type::DualVec,
        (Type::DualRec, _) | (_, Type::DualRec) => Type::DualRec,
        (Type::Float, _) | (_, Type::Float) => Type::Float,
        (Type::Int, Type::Int) => Type::Int,
        _ => t1.clone(),
    }
}

/// Get the numeric type hierarchy level (for promotion)
pub fn numeric_level(ty: &Type) -> Option<u8> {
    match ty {
        Type::Int => Some(0),
        Type::Float => Some(1),
        Type::Dual => Some(2),
        Type::DualVec => Some(3),
        Type::DualRec => Some(4),
        _ => None,
    }
}

/// Check if t1 can be promoted to t2
pub fn can_promote(from: &Type, to: &Type) -> bool {
    match (numeric_level(from), numeric_level(to)) {
        (Some(l1), Some(l2)) => l1 <= l2,
        _ => false,
    }
}

// =============================================================================
// Generic AD Function Instantiation
// =============================================================================

/// Specialize a generic AD function for a concrete type
pub fn specialize_ad_function(
    fn_name: &str,
    type_args: &[Type],
) -> Result<SpecializedAdFn, SpecializeError> {
    // Validate type arguments satisfy bounds
    match fn_name {
        "grad" | "jacobian" | "hessian" | "directional_deriv" | "sensitivity" => {
            if type_args.len() != 1 {
                return Err(SpecializeError::WrongArgCount {
                    expected: 1,
                    got: type_args.len(),
                });
            }
            if !is_differentiable(&type_args[0]) {
                return Err(SpecializeError::BoundNotSatisfied {
                    ty: type_args[0].to_string(),
                    bound: "Differentiable".to_string(),
                });
            }
        }
        "dual" | "lift" => {
            if type_args.len() != 1 {
                return Err(SpecializeError::WrongArgCount {
                    expected: 1,
                    got: type_args.len(),
                });
            }
            if !satisfies_num(&type_args[0]) {
                return Err(SpecializeError::BoundNotSatisfied {
                    ty: type_args[0].to_string(),
                    bound: "Num".to_string(),
                });
            }
        }
        "primal" | "tangent" => {
            if type_args.len() != 1 {
                return Err(SpecializeError::WrongArgCount {
                    expected: 1,
                    got: type_args.len(),
                });
            }
            if !is_differentiable(&type_args[0]) {
                return Err(SpecializeError::BoundNotSatisfied {
                    ty: type_args[0].to_string(),
                    bound: "Differentiable".to_string(),
                });
            }
        }
        _ => {
            return Err(SpecializeError::UnknownFunction(fn_name.to_string()));
        }
    }

    Ok(SpecializedAdFn {
        original_name: fn_name.to_string(),
        mangled_name: mangle_ad_fn(fn_name, type_args),
        type_args: type_args.to_vec(),
    })
}

fn mangle_ad_fn(name: &str, type_args: &[Type]) -> String {
    let args_str: Vec<String> = type_args.iter().map(|t| mangle_type(t)).collect();
    format!("{}${}", name, args_str.join("$"))
}

fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::Int => "Int".to_string(),
        Type::Float => "Float".to_string(),
        Type::Dual => "Dual".to_string(),
        Type::DualVec => "DualVec".to_string(),
        Type::DualRec => "DualRec".to_string(),
        Type::Bool => "Bool".to_string(),
        Type::String => "String".to_string(),
        Type::List(elem) => format!("List_{}", mangle_type(elem)),
        _ => "Unknown".to_string(),
    }
}

#[derive(Debug, Clone)]
pub struct SpecializedAdFn {
    pub original_name: String,
    pub mangled_name: String,
    pub type_args: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SpecializeError {
    WrongArgCount { expected: usize, got: usize },
    BoundNotSatisfied { ty: String, bound: String },
    UnknownFunction(String),
}

impl std::fmt::Display for SpecializeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpecializeError::WrongArgCount { expected, got } => {
                write!(
                    f,
                    "wrong number of type arguments: expected {}, got {}",
                    expected, got
                )
            }
            SpecializeError::BoundNotSatisfied { ty, bound } => {
                write!(f, "type {} does not satisfy bound {}", ty, bound)
            }
            SpecializeError::UnknownFunction(name) => {
                write!(f, "unknown AD function: {}", name)
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_differentiable() {
        assert!(is_differentiable(&Type::Float));
        assert!(is_differentiable(&Type::Dual));
        assert!(is_differentiable(&Type::DualVec));
        assert!(is_differentiable(&Type::DualRec));
        assert!(!is_differentiable(&Type::Int));
        assert!(!is_differentiable(&Type::Bool));
        assert!(!is_differentiable(&Type::String));
    }

    #[test]
    fn test_satisfies_num() {
        assert!(satisfies_num(&Type::Int));
        assert!(satisfies_num(&Type::Float));
        assert!(satisfies_num(&Type::Dual));
        assert!(!satisfies_num(&Type::Bool));
        assert!(!satisfies_num(&Type::String));
    }

    #[test]
    fn test_satisfies_bound() {
        assert!(satisfies_bound(&Type::Float, &TypeBound::Num));
        assert!(satisfies_bound(&Type::Float, &TypeBound::Differentiable));
        assert!(satisfies_bound(&Type::Float, &TypeBound::Ord));
        assert!(satisfies_bound(&Type::Int, &TypeBound::Num));
        assert!(!satisfies_bound(&Type::Int, &TypeBound::Differentiable));
    }

    #[test]
    fn test_check_bounds() {
        let bounds = vec![TypeBound::Num, TypeBound::Differentiable];
        assert!(check_bounds(&Type::Float, &bounds).is_ok());
        assert!(check_bounds(&Type::Dual, &bounds).is_ok());
        assert!(check_bounds(&Type::Int, &bounds).is_err());
    }

    #[test]
    fn test_ad_poly_signatures() {
        let mut gen = TypeVarGen::new();
        let sigs = ad_poly_signatures(&mut gen);

        assert!(sigs.contains_key("grad"));
        assert!(sigs.contains_key("jacobian"));
        assert!(sigs.contains_key("hessian"));
        assert!(sigs.contains_key("primal"));
        assert!(sigs.contains_key("tangent"));

        // Check grad signature: (T -> T, T) -> T
        let grad_sig = sigs.get("grad").unwrap();
        assert_eq!(grad_sig.arity(), 1);
        assert!(grad_sig.type_params[0]
            .bounds
            .contains(&TypeBound::Differentiable));
    }

    #[test]
    fn test_promote_for_ad() {
        assert_eq!(promote_for_ad(&Type::Float, &Type::Float), Type::Float);
        assert_eq!(promote_for_ad(&Type::Float, &Type::Dual), Type::Dual);
        assert_eq!(promote_for_ad(&Type::Dual, &Type::Float), Type::Dual);
        assert_eq!(promote_for_ad(&Type::Int, &Type::Float), Type::Float);
        assert_eq!(promote_for_ad(&Type::Int, &Type::Int), Type::Int);
    }

    #[test]
    fn test_can_promote() {
        assert!(can_promote(&Type::Int, &Type::Float));
        assert!(can_promote(&Type::Float, &Type::Dual));
        assert!(can_promote(&Type::Int, &Type::Dual));
        assert!(!can_promote(&Type::Dual, &Type::Float));
        assert!(!can_promote(&Type::Float, &Type::Int));
    }

    #[test]
    fn test_specialize_ad_function() {
        // Valid specialization
        let result = specialize_ad_function("grad", &[Type::Float]);
        assert!(result.is_ok());
        let spec = result.unwrap();
        assert_eq!(spec.mangled_name, "grad$Float");

        // Dual specialization
        let result = specialize_ad_function("grad", &[Type::Dual]);
        assert!(result.is_ok());
        let spec = result.unwrap();
        assert_eq!(spec.mangled_name, "grad$Dual");

        // Invalid: Int is not Differentiable
        let result = specialize_ad_function("grad", &[Type::Int]);
        assert!(result.is_err());
        match result {
            Err(SpecializeError::BoundNotSatisfied { ty, bound }) => {
                assert_eq!(ty, "Int");
                assert_eq!(bound, "Differentiable");
            }
            _ => panic!("Expected BoundNotSatisfied error"),
        }

        // Invalid: wrong arg count
        let result = specialize_ad_function("grad", &[Type::Float, Type::Float]);
        assert!(result.is_err());
        match result {
            Err(SpecializeError::WrongArgCount { expected, got }) => {
                assert_eq!(expected, 1);
                assert_eq!(got, 2);
            }
            _ => panic!("Expected WrongArgCount error"),
        }
    }

    #[test]
    fn test_numeric_level() {
        assert_eq!(numeric_level(&Type::Int), Some(0));
        assert_eq!(numeric_level(&Type::Float), Some(1));
        assert_eq!(numeric_level(&Type::Dual), Some(2));
        assert_eq!(numeric_level(&Type::DualVec), Some(3));
        assert_eq!(numeric_level(&Type::Bool), None);
    }
}
