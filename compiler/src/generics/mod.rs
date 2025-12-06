// Week 52: Parametric Polymorphism (Generics)
//
// This module implements parametric polymorphism for MedLang, enabling
// generic functions and types that can work with multiple concrete types.
//
// ## Architecture
//
// The generics system consists of several key components:
//
// 1. **Type System (`types.rs`)**
//    - `Type`: Extended type representation with type variables
//    - `TypeVar`: Type variable with unique ID and optional name
//    - `PolyType`: Polymorphic types (type schemes) like ∀T. T -> T
//    - `Subst`: Substitution maps for type instantiation
//    - `TypeParam`: Type parameter declarations with optional bounds
//
// 2. **Unification (`unify.rs`)**
//    - Robinson's unification algorithm for type inference
//    - Occurs check for infinite type detection
//    - Constraint solving for multiple type equations
//    - One-way matching for instantiation
//
// 3. **Type Inference (`infer.rs`)**
//    - Hindley-Milner style type inference
//    - Instantiation: PolyType -> Type with fresh variables
//    - Generalization: Type -> PolyType (let-polymorphism)
//    - Type environment management
//    - Built-in polymorphic function signatures
//
// 4. **Monomorphization (`mono.rs`)**
//    - Specialization of generic functions to concrete types
//    - Name mangling for specialized instances
//    - Instantiation tracking and caching
//    - Worklist algorithm for transitive specialization
//
// 5. **AST Extensions (`ast.rs`)**
//    - `TypeExprAst`: Syntactic type expressions
//    - `TypeParamAst`: Type parameter declarations
//    - `GenericFnDecl`: Generic function declarations
//    - `GenericExpr`: Expressions with type arguments
//
// 6. **Type Checking (`check.rs`)**
//    - Type checker for generic code
//    - Function signature validation
//    - Type argument inference and checking
//    - Bound satisfaction checking
//
// ## Usage Example
//
// ```text
// // MedLang source
// fn identity<T>(x: T) -> T {
//     x
// }
//
// fn map<T, U>(f: (T) -> U, xs: [T]) -> [U] {
//     // implementation
// }
//
// // Usage
// let x = identity<Int>(42)
// let ys = map<Float, String>(toString, [1.0, 2.0, 3.0])
// ```
//
// ## Integration with AD
//
// The generics system integrates with automatic differentiation:
//
// ```text
// fn grad<T: Differentiable>(f: (T) -> T, x: T) -> T {
//     // AD gradient computation
// }
// ```
//
// The `Differentiable` bound ensures that `T` supports differentiation,
// which is satisfied by `Float`, `Dual`, `DualVec`, etc.

pub mod ad_integration;
pub mod ast;
pub mod check;
pub mod infer;
pub mod mono;
pub mod types;
pub mod unify;

// Re-exports for convenience
pub use ast::{
    BinaryOpAst, GenericBlock, GenericExpr, GenericFnDecl, GenericParam, GenericStmt, TypeBoundAst,
    TypeExprAst, TypeParamAst, UnaryOpAst,
};
pub use check::{CheckError, GenericTypeChecker};
pub use infer::{
    builtin_poly_signatures, generalize, generalize_with_params, instantiate, instantiate_with,
    InferContext, InferError, TypeEnv,
};
pub use mono::{MonoCollector, MonoContext, MonoError, MonoFunction, MonoKey};
pub use types::{PolyType, Subst, Type, TypeBound, TypeParam, TypeVar, TypeVarGen, TypeVarId};
pub use unify::{match_types, solve_constraints, unify, Constraint, UnifyError};

// AD integration exports
pub use ad_integration::{
    ad_poly_signatures, can_promote, check_bounds, is_differentiable, numeric_level,
    promote_for_ad, satisfies_bound, satisfies_num, specialize_ad_function, BoundError,
    SpecializeError, SpecializedAdFn,
};

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a new type checker with default settings
pub fn new_checker() -> GenericTypeChecker {
    GenericTypeChecker::new()
}

/// Create a new monomorphization context
pub fn new_mono_context() -> MonoContext {
    MonoContext::new()
}

/// Create a new type variable generator
pub fn new_var_gen() -> TypeVarGen {
    TypeVarGen::new()
}

/// Quick instantiate: create a Type from a PolyType
pub fn quick_instantiate(poly: &PolyType) -> Type {
    let mut ctx = InferContext::new();
    instantiate(&mut ctx, poly)
}

// =============================================================================
// Prelude: Common Generic Functions
// =============================================================================

/// Get the standard library generic function signatures
pub fn stdlib_generics() -> std::collections::HashMap<String, PolyType> {
    let mut gen = TypeVarGen::new();
    builtin_poly_signatures(&mut gen)
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_full_workflow() {
        // 1. Create a type checker
        let mut checker = GenericTypeChecker::new();

        // 2. Define a generic function: fn swap<T, U>(a: T, b: U) -> (U, T)
        let fn_decl = GenericFnDecl::new("swap")
            .with_type_params(vec![TypeParamAst::new("T"), TypeParamAst::new("U")])
            .with_params(vec![
                GenericParam::typed("a", TypeExprAst::var("T")),
                GenericParam::typed("b", TypeExprAst::var("U")),
            ])
            .with_ret_type(TypeExprAst::Tuple(vec![
                TypeExprAst::var("U"),
                TypeExprAst::var("T"),
            ]))
            .with_body(GenericBlock::new(vec![GenericStmt::Expr(
                GenericExpr::Tuple(vec![GenericExpr::var("b"), GenericExpr::var("a")]),
            )]));

        let poly = checker.check_fn_decl(&fn_decl).unwrap();
        assert_eq!(poly.arity(), 2);

        // 3. Call the function with concrete types
        let call = GenericExpr::call_generic(
            GenericExpr::var("swap"),
            vec![TypeExprAst::simple("Int"), TypeExprAst::simple("String")],
            vec![GenericExpr::int(42), GenericExpr::string("hello")],
        );

        let result_type = checker
            .check_expr(&call, &std::collections::HashMap::new())
            .unwrap();

        assert_eq!(result_type, Type::tuple(vec![Type::String, Type::Int]));

        // 4. Check that monomorphization was collected
        let instances = checker.mono_collector().get("swap").unwrap();
        assert!(instances.contains(&vec![Type::Int, Type::String]));
    }

    #[test]
    fn test_monomorphization_workflow() {
        // 1. Create mono context
        let mut mono_ctx = MonoContext::new();

        // 2. Register a generic function
        mono_ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        // 3. Request specializations
        let name1 = mono_ctx.request_mono("id", vec![Type::Int]).unwrap();
        let name2 = mono_ctx.request_mono("id", vec![Type::Float]).unwrap();
        let name3 = mono_ctx
            .request_mono("id", vec![Type::list(Type::String)])
            .unwrap();

        assert_eq!(name1, "id$$Int");
        assert_eq!(name2, "id$$Float");
        assert_eq!(name3, "id$$List_String");

        // 4. Process all pending
        let processed = mono_ctx.process_all();
        assert_eq!(processed.len(), 3);

        // 5. Verify instances
        assert_eq!(mono_ctx.total_instances(), 3);

        let int_instance = mono_ctx.get_by_mangled_name("id$$Int").unwrap();
        assert_eq!(int_instance.original_name, "id");

        match &int_instance.mono_type {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(**ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_unification_in_inference() {
        let mut ctx = InferContext::new();

        // Create type variables
        let t1 = ctx.fresh_type();
        let t2 = ctx.fresh_type();

        // Add constraints: T1 = [T2], T2 = Int
        ctx.add_constraint(t1.clone(), Type::list(t2.clone()));
        ctx.add_constraint(t2.clone(), Type::Int);

        // Solve
        let subst = ctx.solve().unwrap();

        // Check results
        assert_eq!(subst.apply(&t2), Type::Int);
        assert_eq!(subst.apply(&t1), Type::list(Type::Int));
    }

    #[test]
    fn test_generalization_and_instantiation() {
        let env = TypeEnv::new();
        let mut ctx = InferContext::new();

        // Type: T0 -> T0 (with T0 not in environment)
        let ty = Type::function(vec![Type::var(0)], Type::var(0));

        // Generalize
        let poly = generalize(&env, &ty);
        assert!(!poly.is_monomorphic());

        // Instantiate twice - should get different fresh variables each time
        let inst1 = instantiate(&mut ctx, &poly);
        let inst2 = instantiate(&mut ctx, &poly);

        // Both should be function types
        match (&inst1, &inst2) {
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
                // Each instance should have the same structure but different variables
                assert_eq!(p1[0], *r1.as_ref());
                assert_eq!(p2[0], *r2.as_ref());
                // But the variables should be different between instances
                assert_ne!(p1[0], p2[0]);
            }
            _ => panic!("Expected function types"),
        }
    }

    #[test]
    fn test_higher_order_generic() {
        let mut checker = GenericTypeChecker::new();

        // fn apply<T, U>(f: (T) -> U, x: T) -> U { f(x) }
        let fn_decl = GenericFnDecl::new("apply")
            .with_type_params(vec![TypeParamAst::new("T"), TypeParamAst::new("U")])
            .with_params(vec![
                GenericParam::typed(
                    "f",
                    TypeExprAst::function(vec![TypeExprAst::var("T")], TypeExprAst::var("U")),
                ),
                GenericParam::typed("x", TypeExprAst::var("T")),
            ])
            .with_ret_type(TypeExprAst::var("U"))
            .with_body(GenericBlock::new(vec![GenericStmt::Expr(
                GenericExpr::call(GenericExpr::var("f"), vec![GenericExpr::var("x")]),
            )]));

        let poly = checker.check_fn_decl(&fn_decl).unwrap();
        assert_eq!(poly.arity(), 2);

        // Call: apply<Int, String>(intToString, 42)
        let call = GenericExpr::call_generic(
            GenericExpr::var("apply"),
            vec![TypeExprAst::simple("Int"), TypeExprAst::simple("String")],
            vec![
                GenericExpr::Lambda {
                    params: vec![GenericParam::typed("n", TypeExprAst::simple("Int"))],
                    ret_type: Some(TypeExprAst::simple("String")),
                    body: Box::new(GenericExpr::string("converted")),
                },
                GenericExpr::int(42),
            ],
        );

        let result_type = checker
            .check_expr(&call, &std::collections::HashMap::new())
            .unwrap();

        assert_eq!(result_type, Type::String);
    }

    #[test]
    fn test_nested_generic_types() {
        let mut checker = GenericTypeChecker::new();
        let type_param_map = std::collections::HashMap::new();

        // [[Int]]
        let nested_list = GenericExpr::List(vec![GenericExpr::List(vec![
            GenericExpr::int(1),
            GenericExpr::int(2),
        ])]);

        let ty = checker.check_expr(&nested_list, &type_param_map).unwrap();
        assert_eq!(ty, Type::list(Type::list(Type::Int)));
    }
}
