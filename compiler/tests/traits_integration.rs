// Week 53: Traits Integration Tests
//
// Comprehensive tests for the trait system (typeclasses).

use medlangc::traits::*;
use std::collections::HashMap;

// =============================================================================
// Trait Declaration Tests
// =============================================================================

mod trait_decl {
    use super::*;

    #[test]
    fn test_simple_trait() {
        let mut checker = TraitChecker::new();

        let decl = TraitDecl::new("Showable").with_type_param("T").with_method(
            TraitMethod::new("show")
                .with_param("x", "T")
                .with_ret_type("String"),
        );

        let result = checker.check_trait_decl(&decl);
        assert!(result.is_ok());
        assert!(checker.has_trait("Showable"));
    }

    #[test]
    fn test_numeric_trait() {
        let mut checker = TraitChecker::new();

        let decl = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"))
            .with_method(TraitMethod::new("one").with_ret_type("T"))
            .with_method(
                TraitMethod::new("add")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            )
            .with_method(
                TraitMethod::new("sub")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            )
            .with_method(
                TraitMethod::new("mul")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            )
            .with_method(
                TraitMethod::new("neg")
                    .with_param("x", "T")
                    .with_ret_type("T"),
            );

        assert!(checker.check_trait_decl(&decl).is_ok());

        let trait_ty = checker.get_trait("Numeric").unwrap();
        assert_eq!(trait_ty.methods.len(), 6);
        assert!(trait_ty.methods.contains_key("zero"));
        assert!(trait_ty.methods.contains_key("add"));
    }

    #[test]
    fn test_trait_with_inheritance() {
        let mut checker = TraitChecker::new();

        // First declare Eq
        let eq_decl = TraitDecl::new("Eq").with_type_param("T").with_method(
            TraitMethod::new("eq")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("Bool"),
        );
        checker.check_trait_decl(&eq_decl).unwrap();

        // Then declare Ord extending Eq
        let ord_decl = TraitDecl::new("Ord")
            .with_type_param("T")
            .with_super_trait(TraitRef::with_single_arg("Eq", "T"))
            .with_method(
                TraitMethod::new("lt")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("Bool"),
            );
        checker.check_trait_decl(&ord_decl).unwrap();

        let ord_ty = checker.get_trait("Ord").unwrap();
        assert!(ord_ty.has_super_trait("Eq"));
    }

    #[test]
    fn test_duplicate_trait_error() {
        let mut checker = TraitChecker::new();

        let decl1 = TraitDecl::new("Test").with_type_param("T");
        let decl2 = TraitDecl::new("Test").with_type_param("U");

        assert!(checker.check_trait_decl(&decl1).is_ok());
        assert!(checker.check_trait_decl(&decl2).is_err());
    }

    #[test]
    fn test_unknown_super_trait_error() {
        let mut checker = TraitChecker::new();

        let decl = TraitDecl::new("Derived")
            .with_type_param("T")
            .with_super_trait(TraitRef::simple("NonExistent"));

        assert!(checker.check_trait_decl(&decl).is_err());
    }
}

// =============================================================================
// Trait Implementation Tests
// =============================================================================

mod trait_impl {
    use super::*;

    fn setup_numeric_trait() -> TraitChecker {
        let mut checker = TraitChecker::new();

        let decl = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"))
            .with_method(
                TraitMethod::new("add")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            );

        checker.check_trait_decl(&decl).unwrap();
        checker
    }

    #[test]
    fn test_simple_impl() {
        let mut checker = setup_numeric_trait();

        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );

        assert!(checker.check_trait_impl(&impl_).is_ok());
    }

    #[test]
    fn test_impl_for_int() {
        let mut checker = setup_numeric_trait();

        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Int"))
            .with_method_body("zero", vec![], "Int", "0")
            .with_method_body("add", vec![("x", "Int"), ("y", "Int")], "Int", "x + y");

        assert!(checker.check_trait_impl(&impl_).is_ok());
    }

    #[test]
    fn test_multiple_impls() {
        let mut checker = setup_numeric_trait();

        let impl_float = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );

        let impl_int = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Int"))
            .with_method_body("zero", vec![], "Int", "0")
            .with_method_body("add", vec![("x", "Int"), ("y", "Int")], "Int", "x + y");

        assert!(checker.check_trait_impl(&impl_float).is_ok());
        assert!(checker.check_trait_impl(&impl_int).is_ok());

        // Both impls should be registered
        let impls = checker.all_impls().find_for_trait("Numeric");
        assert_eq!(impls.len(), 2);
    }

    #[test]
    fn test_missing_method_error() {
        let mut checker = setup_numeric_trait();

        // Missing "add" method
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float")).with_method_body(
            "zero",
            vec![],
            "Float",
            "0.0",
        );

        let result = checker.check_trait_impl(&impl_);
        assert!(result.is_err());
    }

    #[test]
    fn test_extra_method_error() {
        let mut checker = setup_numeric_trait();

        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            )
            .with_method_body("extra", vec![], "Float", "1.0"); // Not in trait

        let result = checker.check_trait_impl(&impl_);
        assert!(result.is_err());
    }

    #[test]
    fn test_signature_mismatch_error() {
        let mut checker = setup_numeric_trait();

        // Wrong return type for "add"
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body("add", vec![("x", "Float"), ("y", "Float")], "Int", "0"); // Wrong return

        let result = checker.check_trait_impl(&impl_);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_impl_error() {
        let mut checker = setup_numeric_trait();

        let impl1 = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );

        let impl2 = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );

        assert!(checker.check_trait_impl(&impl1).is_ok());
        assert!(checker.check_trait_impl(&impl2).is_err()); // Duplicate
    }
}

// =============================================================================
// Trait Method Resolution Tests
// =============================================================================

mod resolution {
    use super::*;

    fn setup_with_impl() -> TraitChecker {
        let mut checker = TraitChecker::new();

        // Declare trait
        let decl = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"))
            .with_method(
                TraitMethod::new("add")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            );
        checker.check_trait_decl(&decl).unwrap();

        // Add impl
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );
        checker.check_trait_impl(&impl_).unwrap();

        checker
    }

    #[test]
    fn test_resolve_method() {
        let checker = setup_with_impl();
        let resolver = checker.resolver();

        let result = resolver.resolve_method("Numeric", "add", &["Float".to_string()]);
        assert!(result.is_ok());

        let resolved = result.unwrap();
        assert_eq!(resolved.symbol, "Numeric_Float_add");
        assert_eq!(resolved.signature.arity(), 2);
    }

    #[test]
    fn test_resolve_nullary_method() {
        let checker = setup_with_impl();
        let resolver = checker.resolver();

        let result = resolver.resolve_method("Numeric", "zero", &["Float".to_string()]);
        assert!(result.is_ok());

        let resolved = result.unwrap();
        assert_eq!(resolved.symbol, "Numeric_Float_zero");
        assert_eq!(resolved.signature.arity(), 0);
    }

    #[test]
    fn test_resolve_unknown_trait() {
        let checker = setup_with_impl();
        let resolver = checker.resolver();

        let result = resolver.resolve_method("Unknown", "method", &["Float".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_unknown_method() {
        let checker = setup_with_impl();
        let resolver = checker.resolver();

        let result = resolver.resolve_method("Numeric", "unknown", &["Float".to_string()]);
        assert!(result.is_err());
    }

    #[test]
    fn test_resolve_missing_impl() {
        let checker = setup_with_impl();
        let resolver = checker.resolver();

        // No impl for Int
        let result = resolver.resolve_method("Numeric", "add", &["Int".to_string()]);
        assert!(result.is_err());
    }
}

// =============================================================================
// Trait Lowering Tests
// =============================================================================

mod lowering {
    use super::*;

    #[test]
    fn test_lower_impl_to_functions() {
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0")
            .with_method_body(
                "add",
                vec![("x", "Float"), ("y", "Float")],
                "Float",
                "x + y",
            );

        let generated = lower_single_impl(&impl_);
        assert_eq!(generated.len(), 2);

        // Check zero
        let zero = generated.iter().find(|f| f.method_name == "zero").unwrap();
        assert_eq!(zero.name, "Numeric_Float_zero");
        assert!(zero.param_names.is_empty());

        // Check add
        let add = generated.iter().find(|f| f.method_name == "add").unwrap();
        assert_eq!(add.name, "Numeric_Float_add");
        assert_eq!(add.param_names, vec!["x", "y"]);
    }

    #[test]
    fn test_rewrite_trait_call() {
        let mut checker = TraitChecker::new();

        // Setup
        let decl = TraitDecl::new("Numeric").with_type_param("T").with_method(
            TraitMethod::new("add")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("T"),
        );
        checker.check_trait_decl(&decl).unwrap();

        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float")).with_method_body(
            "add",
            vec![("x", "Float"), ("y", "Float")],
            "Float",
            "x + y",
        );
        checker.check_trait_impl(&impl_).unwrap();

        // Rewrite a trait call
        let call = lower::TraitMethodCall {
            trait_name: "Numeric".to_string(),
            method_name: "add".to_string(),
            type_args: vec!["Float".to_string()],
            args: vec!["a".to_string(), "b".to_string()],
        };

        let result = lower::rewrite_trait_call(&call, checker.all_traits(), checker.all_impls());
        assert!(result.is_ok());

        let rewritten = result.unwrap();
        assert_eq!(rewritten.fn_name, "Numeric_Float_add");
    }

    #[test]
    fn test_mangle_symbols() {
        assert_eq!(
            lower::mangle_trait_method_symbol("Numeric", &["Float".to_string()], "add"),
            "Numeric_Float_add"
        );

        assert_eq!(
            lower::mangle_trait_method_symbol("Eq", &["String".to_string()], "eq"),
            "Eq_String_eq"
        );

        assert_eq!(
            lower::mangle_trait_method_symbol("Container", &["Vector<Int>".to_string()], "push"),
            "Container_Vector_Int_push"
        );
    }
}

// =============================================================================
// Bound Checking Tests
// =============================================================================

mod bounds {
    use super::*;

    fn setup_with_bounds() -> TraitChecker {
        let mut checker = TraitChecker::new();

        // Declare Numeric
        let numeric = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"));
        checker.check_trait_decl(&numeric).unwrap();

        // Declare Ord
        let ord = TraitDecl::new("Ord").with_type_param("T").with_method(
            TraitMethod::new("lt")
                .with_param("x", "T")
                .with_param("y", "T")
                .with_ret_type("Bool"),
        );
        checker.check_trait_decl(&ord).unwrap();

        // Impls for Float
        let numeric_float = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Float"))
            .with_method_body("zero", vec![], "Float", "0.0");
        checker.check_trait_impl(&numeric_float).unwrap();

        let ord_float = TraitImpl::new(TraitRef::with_single_arg("Ord", "Float")).with_method_body(
            "lt",
            vec![("x", "Float"), ("y", "Float")],
            "Bool",
            "x < y",
        );
        checker.check_trait_impl(&ord_float).unwrap();

        checker
    }

    #[test]
    fn test_single_bound_satisfied() {
        let checker = setup_with_bounds();

        let result = checker.check_bounds("Float", &[TraitRef::simple("Numeric")]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multiple_bounds_satisfied() {
        let checker = setup_with_bounds();

        let bounds = vec![TraitRef::simple("Numeric"), TraitRef::simple("Ord")];
        let result = checker.check_bounds("Float", &bounds);
        assert!(result.is_ok());
    }

    #[test]
    fn test_bound_not_satisfied() {
        let checker = setup_with_bounds();

        // Int has no impls
        let result = checker.check_bounds("Int", &[TraitRef::simple("Numeric")]);
        assert!(result.is_err());
    }

    #[test]
    fn test_partial_bounds_satisfied() {
        let checker = setup_with_bounds();

        // Add Numeric impl for Int but not Ord
        let mut checker = checker;
        let numeric_int = TraitImpl::new(TraitRef::with_single_arg("Numeric", "Int"))
            .with_method_body("zero", vec![], "Int", "0");
        checker.check_trait_impl(&numeric_int).unwrap();

        // Int satisfies Numeric
        assert!(checker
            .check_bounds("Int", &[TraitRef::simple("Numeric")])
            .is_ok());

        // But not Ord
        assert!(checker
            .check_bounds("Int", &[TraitRef::simple("Ord")])
            .is_err());

        // And not both
        let bounds = vec![TraitRef::simple("Numeric"), TraitRef::simple("Ord")];
        assert!(checker.check_bounds("Int", &bounds).is_err());
    }
}

// =============================================================================
// Complex Scenarios Tests
// =============================================================================

mod complex {
    use super::*;

    #[test]
    fn test_full_numeric_hierarchy() {
        let mut checker = TraitChecker::new();

        // Numeric
        let numeric = TraitDecl::new("Numeric")
            .with_type_param("T")
            .with_method(TraitMethod::new("zero").with_ret_type("T"))
            .with_method(TraitMethod::new("one").with_ret_type("T"))
            .with_method(
                TraitMethod::new("add")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            );
        checker.check_trait_decl(&numeric).unwrap();

        // Fractional extends Numeric
        let fractional = TraitDecl::new("Fractional")
            .with_type_param("T")
            .with_super_trait(TraitRef::with_single_arg("Numeric", "T"))
            .with_method(
                TraitMethod::new("div")
                    .with_param("x", "T")
                    .with_param("y", "T")
                    .with_ret_type("T"),
            );
        checker.check_trait_decl(&fractional).unwrap();

        // Real extends Fractional
        let real = TraitDecl::new("Real")
            .with_type_param("T")
            .with_super_trait(TraitRef::with_single_arg("Fractional", "T"))
            .with_method(
                TraitMethod::new("sqrt")
                    .with_param("x", "T")
                    .with_ret_type("T"),
            )
            .with_method(
                TraitMethod::new("exp")
                    .with_param("x", "T")
                    .with_ret_type("T"),
            )
            .with_method(
                TraitMethod::new("log")
                    .with_param("x", "T")
                    .with_ret_type("T"),
            );
        checker.check_trait_decl(&real).unwrap();

        // Verify hierarchy
        let fractional_ty = checker.get_trait("Fractional").unwrap();
        assert!(fractional_ty.has_super_trait("Numeric"));

        let real_ty = checker.get_trait("Real").unwrap();
        assert!(real_ty.has_super_trait("Fractional"));
    }

    #[test]
    fn test_diffable_trait() {
        let mut checker = TraitChecker::new();

        let diffable = TraitDecl::new("Diffable")
            .with_type_param("T")
            .with_method(
                TraitMethod::new("var")
                    .with_param("x", "T")
                    .with_ret_type("Dual"),
            )
            .with_method(
                TraitMethod::new("const_")
                    .with_param("x", "T")
                    .with_ret_type("Dual"),
            )
            .with_method(
                TraitMethod::new("primal")
                    .with_param("d", "Dual")
                    .with_ret_type("T"),
            )
            .with_method(
                TraitMethod::new("tangent")
                    .with_param("d", "Dual")
                    .with_ret_type("T"),
            );
        checker.check_trait_decl(&diffable).unwrap();

        // Impl for Float
        let impl_ = TraitImpl::new(TraitRef::with_single_arg("Diffable", "Float"))
            .with_method_body("var", vec![("x", "Float")], "Dual", "__builtin_dual_var(x)")
            .with_method_body(
                "const_",
                vec![("x", "Float")],
                "Dual",
                "__builtin_dual_const(x)",
            )
            .with_method_body(
                "primal",
                vec![("d", "Dual")],
                "Float",
                "__builtin_dual_primal(d)",
            )
            .with_method_body(
                "tangent",
                vec![("d", "Dual")],
                "Float",
                "__builtin_dual_tangent(d)",
            );
        checker.check_trait_impl(&impl_).unwrap();

        // Verify resolution
        let resolver = checker.resolver();
        let result = resolver.resolve_method("Diffable", "var", &["Float".to_string()]);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().symbol, "Diffable_Float_var");
    }
}
