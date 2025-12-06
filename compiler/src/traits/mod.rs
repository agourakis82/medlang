// Week 53: Trait System (Typeclasses)
//
// This module implements Rust/Haskell-style traits for MedLang, enabling:
// - Bounded polymorphism: fn sum<T: Numeric>(xs: [T]) -> T
// - Ad-hoc polymorphism: impl Numeric<Float> { ... }
// - Trait inheritance: trait Real<T>: Numeric<T> + Fractional<T>
// - Integration with AD: impl Numeric<Dual> enables differentiable generic code
//
// ## Architecture
//
// The trait system consists of several key components:
//
// 1. **AST (`ast.rs`)**
//    - `TraitDecl`: Trait declarations with methods and super-traits
//    - `TraitImpl`: Trait implementations for concrete types
//    - `TraitRef`: References to traits with type arguments
//    - `TraitMethod`: Method signatures within traits
//
// 2. **Type System (`types.rs`)**
//    - `TraitDeclTy`: Type-level representation of traits
//    - `TraitImplTy`: Type-level representation of implementations
//    - `TraitImplIndex`: Efficient lookup structure for impl resolution
//    - `TraitResolver`: Resolves trait method calls to concrete functions
//
// 3. **Type Checking (`check.rs`)**
//    - Validates trait declarations and implementations
//    - Checks bounded generics satisfy constraints
//    - Resolves trait method calls
//
// 4. **Lowering (`lower.rs`)**
//    - Converts trait impls to plain functions
//    - Rewrites trait method calls to direct function calls
//    - Erases all trait-related constructs before codegen
//
// ## Compilation Pipeline
//
// ```text
// Source with traits
//         ↓ parse
// AST with TraitDecl, TraitImpl, TraitMethodCall
//         ↓ collect_traits
// TypeEnv with trait registry
//         ↓ type_check
// Validated AST with resolved trait bounds
//         ↓ lower_trait_impls
// AST with generated functions (Numeric_Float_add, etc.)
//         ↓ lower_trait_calls
// AST with plain function calls (no TraitMethodCall)
//         ↓ monomorphize
// Monomorphic IR ready for codegen
// ```
//
// ## Example
//
// ```medlang
// trait Numeric<T> {
//     fn zero() -> T;
//     fn add(x: T, y: T) -> T;
// }
//
// impl Numeric<Float> {
//     fn zero() -> Float { 0.0 }
//     fn add(x: Float, y: Float) -> Float { x + y }
// }
//
// fn sum<T: Numeric>(xs: [T]) -> T {
//     fold(Numeric::add, Numeric::zero(), xs)
// }
// ```
//
// After lowering:
// ```text
// fn Numeric_Float_zero() -> Float { 0.0 }
// fn Numeric_Float_add(x: Float, y: Float) -> Float { x + y }
// fn sum_Float(xs: [Float]) -> Float {
//     fold(Numeric_Float_add, Numeric_Float_zero(), xs)
// }
// ```

pub mod ast;
pub mod check;
pub mod lower;
pub mod types;

// Re-exports
pub use ast::{
    AssociatedType, TraitDecl, TraitImpl, TraitMethod, TraitRef, TypeBinding, WhereClause,
};
pub use check::{TraitCheckError, TraitChecker};
pub use lower::{lower_single_impl, lower_trait_calls, lower_trait_impls, GeneratedFn, LowerError};
pub use types::{
    AssocTypeTy, MethodSig, ResolvedMethod, TraitDeclTy, TraitImplIndex, TraitImplTy, TraitRefTy,
    TraitResolver, TypeParamTy, WhereClauseTy,
};

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a new trait checker
pub fn new_trait_checker() -> TraitChecker {
    TraitChecker::new()
}

/// Create a simple trait reference (no type args)
pub fn trait_ref(name: &str) -> TraitRef {
    TraitRef::simple(name)
}

/// Create a trait reference with type arguments
pub fn trait_ref_with_args(name: &str, args: Vec<String>) -> TraitRef {
    TraitRef::with_type_args(name, args)
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_simple_trait_decl() {
        let trait_decl = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"))
            .with_method(
                TraitMethod::new("add")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            );

        assert_eq!(trait_decl.name, "Numeric");
        assert_eq!(trait_decl.type_params.len(), 1);
        assert_eq!(trait_decl.methods.len(), 2);
    }

    #[test]
    fn test_trait_impl() {
        let impl_ = TraitImpl::new(TraitRef::with_type_args(
            "Numeric",
            vec!["Float".to_string()],
        ))
        .with_method_body("zero", vec![], "Float", "0.0")
        .with_method_body(
            "add",
            vec![("x", "Float"), ("y", "Float")],
            "Float",
            "x + y",
        );

        assert_eq!(impl_.trait_ref.trait_name, "Numeric");
        assert_eq!(impl_.methods.len(), 2);
    }

    #[test]
    fn test_trait_ref_display() {
        let simple = TraitRef::simple("Eq");
        assert_eq!(simple.to_string(), "Eq");

        let with_args = TraitRef::with_type_args("Numeric", vec!["Float".to_string()]);
        assert_eq!(with_args.to_string(), "Numeric<Float>");
    }

    #[test]
    fn test_trait_decl_ty() {
        let mut decl = TraitDeclTy::new("Ord");
        decl.add_type_param("T");
        decl.add_method(
            "lt",
            MethodSig {
                type_params: vec![],
                param_names: vec!["x".to_string(), "y".to_string()],
                param_types: vec!["T".to_string(), "T".to_string()],
                ret_type: "Bool".to_string(),
            },
        );

        assert_eq!(decl.name, "Ord");
        assert!(decl.methods.contains_key("lt"));
    }

    #[test]
    fn test_trait_impl_index() {
        let mut index = TraitImplIndex::new();

        let impl1 = TraitImplTy {
            trait_ref: TraitRefTy::new("Numeric", vec!["Float".to_string()]),
            type_params: vec![],
            where_clauses: vec![],
            methods: std::collections::HashMap::new(),
            type_bindings: std::collections::HashMap::new(),
        };

        let impl2 = TraitImplTy {
            trait_ref: TraitRefTy::new("Numeric", vec!["Int".to_string()]),
            type_params: vec![],
            where_clauses: vec![],
            methods: std::collections::HashMap::new(),
            type_bindings: std::collections::HashMap::new(),
        };

        index.add(impl1);
        index.add(impl2);

        assert_eq!(index.find_for_trait("Numeric").len(), 2);
        assert!(index.find_impl("Numeric", &["Float".to_string()]).is_some());
        assert!(index.find_impl("Numeric", &["Int".to_string()]).is_some());
        assert!(index
            .find_impl("Numeric", &["String".to_string()])
            .is_none());
    }

    #[test]
    fn test_mangle_trait_method() {
        let symbol = lower::mangle_trait_method_symbol("Numeric", &["Float".to_string()], "add");
        assert_eq!(symbol, "Numeric_Float_add");

        let symbol2 =
            lower::mangle_trait_method_symbol("Ord", &["Vector_Int".to_string()], "compare");
        assert_eq!(symbol2, "Ord_Vector_Int_compare");
    }

    #[test]
    fn test_trait_with_super_traits() {
        let trait_decl = TraitDecl::new("Real")
            .with_type_param("T")
            .with_super_trait(TraitRef::with_type_args("Numeric", vec!["T".to_string()]))
            .with_super_trait(TraitRef::with_type_args(
                "Fractional",
                vec!["T".to_string()],
            ))
            .with_method(
                TraitMethod::new("sqrt")
                    .with_param("x", "T")
                    .with_ret_type("T"),
            );

        assert_eq!(trait_decl.super_traits.len(), 2);
        assert_eq!(trait_decl.super_traits[0].trait_name, "Numeric");
        assert_eq!(trait_decl.super_traits[1].trait_name, "Fractional");
    }

    #[test]
    fn test_generic_impl_with_bounds() {
        // impl<T: Ord> Sortable<Vector<T>>
        let impl_ = TraitImpl::new(TraitRef::with_type_args(
            "Sortable",
            vec!["Vector<T>".to_string()],
        ))
        .with_type_param_bounded("T", vec![TraitRef::simple("Ord")]);

        assert_eq!(impl_.type_params.len(), 1);
        assert_eq!(impl_.type_params[0].bounds.len(), 1);
    }
}
