// Week 52: Generics Integration Tests
//
// Comprehensive tests for the parametric polymorphism system.

use medlangc::generics::*;
use std::collections::HashMap;

// =============================================================================
// Type System Tests
// =============================================================================

mod type_system {
    use super::*;

    #[test]
    fn test_type_var_creation() {
        let mut gen = TypeVarGen::new();

        let v1 = gen.fresh();
        let v2 = gen.fresh_named("T");
        let v3 = gen.fresh();

        // TypeVarGen starts at FRESH_ID_OFFSET (1000) to avoid collision with user type params
        let start = v1.id.0;
        assert_eq!(v2.id.0, start + 1);
        assert_eq!(v3.id.0, start + 2);
        assert_eq!(v2.name, Some("T".to_string()));
    }

    #[test]
    fn test_type_equality() {
        assert_eq!(Type::Int, Type::Int);
        assert_ne!(Type::Int, Type::Float);

        let list1 = Type::list(Type::Int);
        let list2 = Type::list(Type::Int);
        let list3 = Type::list(Type::String);

        assert_eq!(list1, list2);
        assert_ne!(list1, list3);
    }

    #[test]
    fn test_function_type() {
        let fn_type = Type::function(vec![Type::Int, Type::String], Type::Bool);

        match fn_type {
            Type::Function { params, ret } => {
                assert_eq!(params.len(), 2);
                assert_eq!(params[0], Type::Int);
                assert_eq!(params[1], Type::String);
                assert_eq!(*ret, Type::Bool);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_has_type_vars() {
        assert!(!Type::Int.has_type_vars());
        assert!(Type::var(0).has_type_vars());
        assert!(Type::list(Type::var(0)).has_type_vars());
        assert!(!Type::list(Type::Int).has_type_vars());
    }

    #[test]
    fn test_free_type_vars() {
        let ty = Type::function(vec![Type::var(0), Type::var(1)], Type::var(0));
        let vars = ty.free_type_vars();

        assert_eq!(vars.len(), 2);
        assert!(vars.contains(&TypeVarId(0)));
        assert!(vars.contains(&TypeVarId(1)));
    }

    #[test]
    fn test_substitution_apply() {
        let mut subst = Subst::new();
        subst.insert(TypeVarId(0), Type::Int);
        subst.insert(TypeVarId(1), Type::String);

        let original = Type::function(vec![Type::var(0)], Type::var(1));
        let applied = subst.apply(&original);

        match applied {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(*ret, Type::String);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_substitution_compose() {
        // s1: T0 -> Int
        let mut s1 = Subst::new();
        s1.insert(TypeVarId(0), Type::Int);

        // s2: T1 -> [T0]
        let mut s2 = Subst::new();
        s2.insert(TypeVarId(1), Type::list(Type::var(0)));

        // Compose: T1 -> [Int], T0 -> Int
        let composed = s1.compose(&s2);

        assert_eq!(composed.apply(&Type::var(0)), Type::Int);
        assert_eq!(composed.apply(&Type::var(1)), Type::list(Type::Int));
    }

    #[test]
    fn test_polytype() {
        let params = vec![TypeParam::new("T", 0), TypeParam::new("U", 1)];
        let ty = Type::function(vec![Type::var(0), Type::var(1)], Type::var(0));
        let poly = PolyType::new(params, ty);

        assert_eq!(poly.arity(), 2);
        assert!(!poly.is_monomorphic());
    }

    #[test]
    fn test_polytype_mono() {
        let mono = PolyType::mono(Type::Int);
        assert!(mono.is_monomorphic());
        assert_eq!(mono.arity(), 0);
    }
}

// =============================================================================
// Unification Tests
// =============================================================================

mod unification {
    use super::*;

    #[test]
    fn test_unify_identical() {
        assert!(unify(&Type::Int, &Type::Int).is_ok());
        assert!(unify(&Type::Float, &Type::Float).is_ok());
        assert!(unify(&Type::Bool, &Type::Bool).is_ok());
    }

    #[test]
    fn test_unify_different() {
        assert!(unify(&Type::Int, &Type::Float).is_err());
        assert!(unify(&Type::Int, &Type::String).is_err());
    }

    #[test]
    fn test_unify_var_to_concrete() {
        let subst = unify(&Type::var(0), &Type::Int).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Int);
    }

    #[test]
    fn test_unify_concrete_to_var() {
        let subst = unify(&Type::Float, &Type::var(0)).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Float);
    }

    #[test]
    fn test_unify_two_vars() {
        let subst = unify(&Type::var(0), &Type::var(1)).unwrap();
        let applied0 = subst.apply(&Type::var(0));
        let applied1 = subst.apply(&Type::var(1));
        assert_eq!(applied0, applied1);
    }

    #[test]
    fn test_unify_functions() {
        let f1 = Type::function(vec![Type::var(0)], Type::var(0));
        let f2 = Type::function(vec![Type::Int], Type::Int);

        let subst = unify(&f1, &f2).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Int);
    }

    #[test]
    fn test_unify_function_arity_mismatch() {
        let f1 = Type::function(vec![Type::Int], Type::Bool);
        let f2 = Type::function(vec![Type::Int, Type::Int], Type::Bool);

        assert!(matches!(
            unify(&f1, &f2),
            Err(UnifyError::ArityMismatch { .. })
        ));
    }

    #[test]
    fn test_unify_lists() {
        let l1 = Type::list(Type::var(0));
        let l2 = Type::list(Type::String);

        let subst = unify(&l1, &l2).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::String);
    }

    #[test]
    fn test_unify_nested() {
        // [[T]] with [[Int]]
        let t1 = Type::list(Type::list(Type::var(0)));
        let t2 = Type::list(Type::list(Type::Int));

        let subst = unify(&t1, &t2).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Int);
    }

    #[test]
    fn test_occurs_check() {
        // T unified with [T] would create infinite type
        let t1 = Type::var(0);
        let t2 = Type::list(Type::var(0));

        assert!(matches!(
            unify(&t1, &t2),
            Err(UnifyError::InfiniteType { .. })
        ));
    }

    #[test]
    fn test_unify_records() {
        let mut f1 = HashMap::new();
        f1.insert("x".to_string(), Type::var(0));
        f1.insert("y".to_string(), Type::Int);

        let mut f2 = HashMap::new();
        f2.insert("x".to_string(), Type::Float);
        f2.insert("y".to_string(), Type::Int);

        let subst = unify(&Type::Record(f1), &Type::Record(f2)).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Float);
    }

    #[test]
    fn test_unify_option() {
        let o1 = Type::option(Type::var(0));
        let o2 = Type::option(Type::Bool);

        let subst = unify(&o1, &o2).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Bool);
    }

    #[test]
    fn test_unify_result() {
        let r1 = Type::result(Type::var(0), Type::var(1));
        let r2 = Type::result(Type::Int, Type::String);

        let subst = unify(&r1, &r2).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Int);
        assert_eq!(subst.apply(&Type::var(1)), Type::String);
    }

    #[test]
    fn test_solve_constraints() {
        let constraints = vec![
            Constraint::new(Type::var(0), Type::Int),
            Constraint::new(Type::var(1), Type::list(Type::var(0))),
            Constraint::new(
                Type::var(2),
                Type::function(vec![Type::var(0)], Type::var(1)),
            ),
        ];

        let subst = solve_constraints(&constraints).unwrap();

        assert_eq!(subst.apply(&Type::var(0)), Type::Int);
        assert_eq!(subst.apply(&Type::var(1)), Type::list(Type::Int));

        let fn_type = subst.apply(&Type::var(2));
        match fn_type {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(*ret, Type::list(Type::Int));
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_match_types() {
        // Pattern: (T, T) -> T
        // Target: (Int, Int) -> Int
        let pattern = Type::function(vec![Type::var(0), Type::var(0)], Type::var(0));
        let target = Type::function(vec![Type::Int, Type::Int], Type::Int);

        let subst = match_types(&pattern, &target).unwrap();
        assert_eq!(subst.apply(&Type::var(0)), Type::Int);
    }

    #[test]
    fn test_match_types_failure() {
        // Pattern: (T, T) -> T can't match (Int, String) -> Int
        let pattern = Type::function(vec![Type::var(0), Type::var(0)], Type::var(0));
        let target = Type::function(vec![Type::Int, Type::String], Type::Int);

        assert!(match_types(&pattern, &target).is_err());
    }
}

// =============================================================================
// Type Inference Tests
// =============================================================================

mod inference {
    use super::*;

    #[test]
    fn test_instantiate_mono() {
        let mut ctx = InferContext::new();
        let poly = PolyType::mono(Type::Int);

        let inst = instantiate(&mut ctx, &poly);
        assert_eq!(inst, Type::Int);
    }

    #[test]
    fn test_instantiate_poly() {
        let mut ctx = InferContext::new();

        // ∀T. T -> T
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0)],
            Type::function(vec![Type::var(0)], Type::var(0)),
        );

        let inst = instantiate(&mut ctx, &poly);

        // Should be a function with fresh type variables
        match inst {
            Type::Function { params, ret } => {
                assert_eq!(params.len(), 1);
                assert_eq!(params[0], *ret);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_instantiate_with() {
        // ∀T, U. (T, U) -> T
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0), TypeParam::new("U", 1)],
            Type::function(vec![Type::var(0), Type::var(1)], Type::var(0)),
        );

        let inst = instantiate_with(&poly, &[Type::Int, Type::String]).unwrap();

        match inst {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(params[1], Type::String);
                assert_eq!(*ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_generalize_in_empty_env() {
        let env = TypeEnv::new();
        let ty = Type::function(vec![Type::var(0)], Type::var(0));

        let poly = generalize(&env, &ty);

        assert!(!poly.is_monomorphic());
        assert_eq!(poly.type_params.len(), 1);
    }

    #[test]
    fn test_generalize_with_env_vars() {
        let mut env = TypeEnv::new();
        env.extend("x".to_string(), PolyType::mono(Type::var(0)));

        // Type: T0 -> T1 (T0 is in env, T1 is not)
        let ty = Type::function(vec![Type::var(0)], Type::var(1));

        let poly = generalize(&env, &ty);

        // Only T1 should be generalized
        assert_eq!(poly.type_params.len(), 1);
    }

    #[test]
    fn test_infer_context_constraints() {
        let mut ctx = InferContext::new();

        let t1 = ctx.fresh_type();
        let t2 = ctx.fresh_type();

        ctx.add_constraint(t1.clone(), Type::Int);
        ctx.add_constraint(t2.clone(), Type::list(t1.clone()));

        let subst = ctx.solve().unwrap();

        assert_eq!(subst.apply(&t1), Type::Int);
        assert_eq!(subst.apply(&t2), Type::list(Type::Int));
    }

    #[test]
    fn test_type_env_operations() {
        let mut env = TypeEnv::new();

        env.extend("x".to_string(), PolyType::mono(Type::Int));
        env.extend("y".to_string(), PolyType::mono(Type::String));

        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_some());
        assert!(env.lookup("z").is_none());

        env.remove("x");
        assert!(env.lookup("x").is_none());
    }

    #[test]
    fn test_builtin_signatures() {
        let mut gen = TypeVarGen::new();
        let builtins = builtin_poly_signatures(&mut gen);

        // Check identity
        let identity = builtins.get("identity").unwrap();
        assert_eq!(identity.arity(), 1);

        // Check map
        let map = builtins.get("map").unwrap();
        assert_eq!(map.arity(), 2);

        // Check fold
        let fold = builtins.get("fold").unwrap();
        assert_eq!(fold.arity(), 2);

        // Check filter
        let filter = builtins.get("filter").unwrap();
        assert_eq!(filter.arity(), 1);
    }
}

// =============================================================================
// Monomorphization Tests
// =============================================================================

mod monomorphization {
    use super::*;

    #[test]
    fn test_mono_key_mangled_name() {
        let key = MonoKey::new("map", vec![Type::Int, Type::String]);
        assert_eq!(key.mangled_name(), "map$$Int$String");
    }

    #[test]
    fn test_mono_key_no_args() {
        let key = MonoKey::new("foo", vec![]);
        assert_eq!(key.mangled_name(), "foo");
    }

    #[test]
    fn test_mono_key_complex_types() {
        let key = MonoKey::new("f", vec![Type::list(Type::Int), Type::option(Type::String)]);
        assert_eq!(key.mangled_name(), "f$$List_Int$Option_String");
    }

    #[test]
    fn test_mono_function() {
        let poly = PolyType::new(
            vec![TypeParam::new("T", 0)],
            Type::function(vec![Type::var(0)], Type::var(0)),
        );

        let mono = MonoFunction::new("id".to_string(), vec![Type::Int], &poly).unwrap();

        assert_eq!(mono.original_name, "id");
        assert_eq!(mono.mangled_name, "id$$Int");

        match &mono.mono_type {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(**ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_mono_context() {
        let mut ctx = MonoContext::new();

        ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        let name1 = ctx.request_mono("id", vec![Type::Int]).unwrap();
        let name2 = ctx.request_mono("id", vec![Type::Float]).unwrap();
        let name3 = ctx.request_mono("id", vec![Type::Int]).unwrap(); // Duplicate

        assert_eq!(name1, "id$$Int");
        assert_eq!(name2, "id$$Float");
        assert_eq!(name3, name1);

        assert_eq!(ctx.instance_count("id"), 2);
        assert_eq!(ctx.total_instances(), 2);
    }

    #[test]
    fn test_mono_context_process() {
        let mut ctx = MonoContext::new();

        ctx.register_generic(
            "f".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        ctx.request_mono("f", vec![Type::Int]).unwrap();
        ctx.request_mono("f", vec![Type::Bool]).unwrap();

        assert!(ctx.has_pending());

        let processed = ctx.process_all();
        assert_eq!(processed.len(), 2);
        assert!(!ctx.has_pending());
    }

    #[test]
    fn test_mono_context_limit() {
        let mut ctx = MonoContext::new().with_max_instances(2);

        ctx.register_generic(
            "f".to_string(),
            PolyType::new(vec![TypeParam::new("T", 0)], Type::var(0)),
        );

        ctx.request_mono("f", vec![Type::Int]).unwrap();
        ctx.request_mono("f", vec![Type::Float]).unwrap();

        let result = ctx.request_mono("f", vec![Type::String]);
        assert!(matches!(result, Err(MonoError::LimitExceeded(_))));
    }

    #[test]
    fn test_mono_collector() {
        let mut collector = MonoCollector::new();

        collector.record("map", vec![Type::Int, Type::String]);
        collector.record("map", vec![Type::Float, Type::Bool]);
        collector.record("map", vec![Type::Int, Type::String]); // Duplicate

        let instances = collector.get("map").unwrap();
        assert_eq!(instances.len(), 2);
    }

    #[test]
    fn test_mono_collector_apply() {
        let mut collector = MonoCollector::new();
        collector.record("id", vec![Type::Int]);
        collector.record("id", vec![Type::String]);

        let mut ctx = MonoContext::new();
        ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        collector.apply_to(&mut ctx).unwrap();

        assert_eq!(ctx.instance_count("id"), 2);
    }
}

// =============================================================================
// AST Tests
// =============================================================================

mod ast_tests {
    use super::*;

    #[test]
    fn test_type_expr_display() {
        assert_eq!(TypeExprAst::simple("Int").to_string(), "Int");
        assert_eq!(TypeExprAst::var("T").to_string(), "T");
        assert_eq!(
            TypeExprAst::list(TypeExprAst::simple("Int")).to_string(),
            "[Int]"
        );
        assert_eq!(
            TypeExprAst::function(vec![TypeExprAst::var("T")], TypeExprAst::var("U")).to_string(),
            "(T) -> U"
        );
    }

    #[test]
    fn test_generic_fn_decl() {
        let fn_decl = GenericFnDecl::new("swap")
            .with_type_params(vec![TypeParamAst::new("T"), TypeParamAst::new("U")])
            .with_params(vec![
                GenericParam::typed("a", TypeExprAst::var("T")),
                GenericParam::typed("b", TypeExprAst::var("U")),
            ])
            .with_ret_type(TypeExprAst::Tuple(vec![
                TypeExprAst::var("U"),
                TypeExprAst::var("T"),
            ]));

        assert!(fn_decl.is_generic());
        assert_eq!(fn_decl.type_arity(), 2);
    }

    #[test]
    fn test_type_param_with_bounds() {
        let param = TypeParamAst::new("T")
            .with_bound(TypeBoundAst::Num)
            .with_bound(TypeBoundAst::Ord);

        assert_eq!(param.bounds.len(), 2);
        assert_eq!(param.to_string(), "T: Num + Ord");
    }

    #[test]
    fn test_generic_expr_call() {
        let call = GenericExpr::call_generic(
            GenericExpr::var("map"),
            vec![TypeExprAst::simple("Int"), TypeExprAst::simple("String")],
            vec![GenericExpr::var("f"), GenericExpr::var("xs")],
        );

        match call {
            GenericExpr::Call {
                type_args, args, ..
            } => {
                assert_eq!(type_args.len(), 2);
                assert_eq!(args.len(), 2);
            }
            _ => panic!("Expected Call"),
        }
    }

    #[test]
    fn test_generic_lambda() {
        let lambda = GenericExpr::lambda(
            vec![GenericParam::typed("x", TypeExprAst::var("T"))],
            GenericExpr::var("x"),
        );

        match lambda {
            GenericExpr::Lambda { params, .. } => {
                assert_eq!(params.len(), 1);
                assert_eq!(params[0].name, "x");
            }
            _ => panic!("Expected Lambda"),
        }
    }
}

// =============================================================================
// Type Checker Tests
// =============================================================================

mod type_checker {
    use super::*;

    #[test]
    fn test_check_literals() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        assert_eq!(
            checker.check_expr(&GenericExpr::int(42), &tpm).unwrap(),
            Type::Int
        );
        assert_eq!(
            checker.check_expr(&GenericExpr::float(3.14), &tpm).unwrap(),
            Type::Float
        );
        assert_eq!(
            checker
                .check_expr(&GenericExpr::bool_val(true), &tpm)
                .unwrap(),
            Type::Bool
        );
        assert_eq!(
            checker
                .check_expr(&GenericExpr::string("hello"), &tpm)
                .unwrap(),
            Type::String
        );
    }

    #[test]
    fn test_check_binary_ops() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        let add = GenericExpr::Binary {
            op: BinaryOpAst::Add,
            left: Box::new(GenericExpr::int(1)),
            right: Box::new(GenericExpr::int(2)),
        };
        assert_eq!(checker.check_expr(&add, &tpm).unwrap(), Type::Int);

        let lt = GenericExpr::Binary {
            op: BinaryOpAst::Lt,
            left: Box::new(GenericExpr::int(1)),
            right: Box::new(GenericExpr::int(2)),
        };
        assert_eq!(checker.check_expr(&lt, &tpm).unwrap(), Type::Bool);
    }

    #[test]
    fn test_check_if_expr() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        let if_expr = GenericExpr::if_expr(
            GenericExpr::bool_val(true),
            GenericExpr::int(1),
            GenericExpr::int(2),
        );

        assert_eq!(checker.check_expr(&if_expr, &tpm).unwrap(), Type::Int);
    }

    #[test]
    fn test_check_if_branch_mismatch() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        let if_expr = GenericExpr::if_expr(
            GenericExpr::bool_val(true),
            GenericExpr::int(1),
            GenericExpr::string("two"),
        );

        assert!(checker.check_expr(&if_expr, &tpm).is_err());
    }

    #[test]
    fn test_check_list() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        let list = GenericExpr::List(vec![
            GenericExpr::int(1),
            GenericExpr::int(2),
            GenericExpr::int(3),
        ]);

        assert_eq!(
            checker.check_expr(&list, &tpm).unwrap(),
            Type::list(Type::Int)
        );
    }

    #[test]
    fn test_check_lambda() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        let lambda = GenericExpr::Lambda {
            params: vec![GenericParam::typed("x", TypeExprAst::simple("Int"))],
            ret_type: None,
            body: Box::new(GenericExpr::var("x")),
        };

        let ty = checker.check_expr(&lambda, &tpm).unwrap();
        match ty {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(*ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }

    #[test]
    fn test_check_generic_fn_decl() {
        let mut checker = GenericTypeChecker::new();

        let fn_decl = GenericFnDecl::new("identity")
            .with_type_params(vec![TypeParamAst::new("T")])
            .with_params(vec![GenericParam::typed("x", TypeExprAst::var("T"))])
            .with_ret_type(TypeExprAst::var("T"))
            .with_body(GenericBlock::new(vec![GenericStmt::Expr(
                GenericExpr::var("x"),
            )]));

        let poly = checker.check_fn_decl(&fn_decl).unwrap();
        assert_eq!(poly.arity(), 1);
        assert!(!poly.is_monomorphic());
    }

    #[test]
    fn test_check_generic_fn_call() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        // Call identity<Int>(42)
        let call = GenericExpr::call_generic(
            GenericExpr::var("identity"),
            vec![TypeExprAst::simple("Int")],
            vec![GenericExpr::int(42)],
        );

        assert_eq!(checker.check_expr(&call, &tpm).unwrap(), Type::Int);
    }

    #[test]
    fn test_check_undefined_var() {
        let mut checker = GenericTypeChecker::new();
        let tpm = HashMap::new();

        let result = checker.check_expr(&GenericExpr::var("undefined"), &tpm);
        assert!(matches!(result, Err(CheckError::UndefinedVar(_))));
    }

    #[test]
    fn test_check_duplicate_type_param() {
        let mut checker = GenericTypeChecker::new();

        let fn_decl = GenericFnDecl::new("foo")
            .with_type_params(vec![TypeParamAst::new("T"), TypeParamAst::new("T")]);

        let result = checker.check_fn_decl(&fn_decl);
        assert!(matches!(result, Err(CheckError::DuplicateTypeParam(_))));
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

mod integration {
    use super::*;

    #[test]
    fn test_full_generic_workflow() {
        // 1. Create checker
        let mut checker = GenericTypeChecker::new();

        // 2. Define generic function: fn swap<T, U>(a: T, b: U) -> (U, T)
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

        // 3. Call with concrete types
        let call = GenericExpr::call_generic(
            GenericExpr::var("swap"),
            vec![TypeExprAst::simple("Int"), TypeExprAst::simple("String")],
            vec![GenericExpr::int(42), GenericExpr::string("hello")],
        );

        let result_type = checker.check_expr(&call, &HashMap::new()).unwrap();
        assert_eq!(result_type, Type::tuple(vec![Type::String, Type::Int]));

        // 4. Check monomorphization
        let instances = checker.mono_collector().get("swap").unwrap();
        assert!(instances.contains(&vec![Type::Int, Type::String]));
    }

    #[test]
    fn test_higher_order_generic() {
        let mut checker = GenericTypeChecker::new();

        // fn apply<T, U>(f: (T) -> U, x: T) -> U
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

        // Call with lambda
        let call = GenericExpr::call_generic(
            GenericExpr::var("apply"),
            vec![TypeExprAst::simple("Int"), TypeExprAst::simple("Bool")],
            vec![
                GenericExpr::Lambda {
                    params: vec![GenericParam::typed("n", TypeExprAst::simple("Int"))],
                    ret_type: Some(TypeExprAst::simple("Bool")),
                    body: Box::new(GenericExpr::bool_val(true)),
                },
                GenericExpr::int(42),
            ],
        );

        let result_type = checker.check_expr(&call, &HashMap::new()).unwrap();
        assert_eq!(result_type, Type::Bool);
    }

    #[test]
    fn test_monomorphization_workflow() {
        let mut mono_ctx = MonoContext::new();

        // Register generic
        mono_ctx.register_generic(
            "id".to_string(),
            PolyType::new(
                vec![TypeParam::new("T", 0)],
                Type::function(vec![Type::var(0)], Type::var(0)),
            ),
        );

        // Request specializations
        let name1 = mono_ctx.request_mono("id", vec![Type::Int]).unwrap();
        let name2 = mono_ctx.request_mono("id", vec![Type::Float]).unwrap();
        let name3 = mono_ctx
            .request_mono("id", vec![Type::list(Type::String)])
            .unwrap();

        assert_eq!(name1, "id$$Int");
        assert_eq!(name2, "id$$Float");
        assert_eq!(name3, "id$$List_String");

        // Process all
        let processed = mono_ctx.process_all();
        assert_eq!(processed.len(), 3);

        // Verify instances
        let int_instance = mono_ctx.get_by_mangled_name("id$$Int").unwrap();
        match &int_instance.mono_type {
            Type::Function { params, ret } => {
                assert_eq!(params[0], Type::Int);
                assert_eq!(**ret, Type::Int);
            }
            _ => panic!("Expected function type"),
        }
    }
}
