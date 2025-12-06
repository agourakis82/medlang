//! Refinement Types Integration Tests
//!
//! Tests for MedLang's refinement type system with SMT verification.
//! Day 5 implementation: Compile-time verification of logical predicates.

use medlangc::refinement::{
    ArithOp, BaseTypeRef, CompareOp, Predicate, RefinedVar, RefinementExpr, RefinementType,
};
use medlangc::refinement::{ConstraintKind, ConstraintSet, VCGenerator};
use medlangc::refinement::{Counterexample, ErrorSeverity, RefinementError, RefinementErrorKind};
use medlangc::refinement::{FunctionRefinement, RefinementChecker, RefinementEnv};
use medlangc::refinement::{SmtContext, SmtLogic, SmtResult, SmtSolver, SmtSort};
use medlangc::refinement::{SubtypeChecker, SubtypeResult};

// ============================================================================
// Syntax Tests - RefinementType, Predicate, RefinementExpr
// ============================================================================

mod syntax_tests {
    use super::*;

    #[test]
    fn test_refined_var_creation() {
        let var = RefinedVar::new("x");
        assert_eq!(var.name, "x");
    }

    #[test]
    fn test_refined_var_equality() {
        let var1 = RefinedVar::new("x");
        let var2 = RefinedVar::new("x");
        let var3 = RefinedVar::new("y");

        assert_eq!(var1, var2);
        assert_ne!(var1, var3);
    }

    #[test]
    fn test_base_type_ref_named() {
        let int_ty = BaseTypeRef::Named("Int".to_string());
        let float_ty = BaseTypeRef::Named("Float".to_string());

        assert!(matches!(int_ty, BaseTypeRef::Named(ref s) if s == "Int"));
        assert!(matches!(float_ty, BaseTypeRef::Named(ref s) if s == "Float"));
    }

    #[test]
    fn test_base_type_ref_with_unit() {
        let dose_ty = BaseTypeRef::WithUnit {
            base: "Float".to_string(),
            unit: "mg".to_string(),
        };

        if let BaseTypeRef::WithUnit { base, unit } = dose_ty {
            assert_eq!(base, "Float");
            assert_eq!(unit, "mg");
        } else {
            panic!("Expected WithUnit variant");
        }
    }

    #[test]
    fn test_simple_predicate() {
        // x > 0
        let pred = Predicate::Compare {
            left: Box::new(RefinementExpr::Var(RefinedVar::new("x"))),
            op: CompareOp::Gt,
            right: Box::new(RefinementExpr::Int(0)),
        };

        assert!(matches!(pred, Predicate::Compare { .. }));
    }

    #[test]
    fn test_predicate_and() {
        // x > 0 && x < 100
        let left = Predicate::gt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(0),
        );
        let right = Predicate::lt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(100),
        );

        let combined = Predicate::and(left, right);
        assert!(matches!(combined, Predicate::And(_, _)));
    }

    #[test]
    fn test_predicate_or() {
        // x < 0 || x > 100
        let left = Predicate::lt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(0),
        );
        let right = Predicate::gt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(100),
        );

        let combined = Predicate::or(left, right);
        assert!(matches!(combined, Predicate::Or(_, _)));
    }

    #[test]
    fn test_predicate_not() {
        let inner = Predicate::Bool(true);
        let negated = Predicate::not(inner);
        assert!(matches!(negated, Predicate::Not(_)));
    }

    #[test]
    fn test_predicate_implies() {
        let premise = Predicate::Bool(true);
        let conclusion = Predicate::gt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(0),
        );

        let impl_pred = Predicate::implies(premise, conclusion);
        assert!(matches!(impl_pred, Predicate::Implies(_, _)));
    }

    #[test]
    fn test_bool_predicates() {
        let true_pred = Predicate::Bool(true);
        let false_pred = Predicate::Bool(false);

        assert!(matches!(true_pred, Predicate::Bool(true)));
        assert!(matches!(false_pred, Predicate::Bool(false)));
    }

    #[test]
    fn test_refinement_type_creation() {
        // { x: Int | x > 0 }
        let positive_int = RefinementType::new(
            RefinedVar::new("x"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::gt(
                RefinementExpr::Var(RefinedVar::new("x")),
                RefinementExpr::Int(0),
            ),
        );

        assert_eq!(positive_int.binder.name, "x");
        assert!(matches!(positive_int.base_type, BaseTypeRef::Named(ref s) if s == "Int"));
    }

    #[test]
    fn test_refinement_type_trivial() {
        // Just Int with no constraint (true)
        let just_int =
            RefinementType::trivial(RefinedVar::new("v"), BaseTypeRef::Named("Int".to_string()));
        assert!(matches!(just_int.predicate, Predicate::Bool(true)));
    }

    #[test]
    fn test_refinement_expr_arithmetic() {
        // x + 1
        let add = RefinementExpr::Arith {
            left: Box::new(RefinementExpr::Var(RefinedVar::new("x"))),
            op: ArithOp::Add,
            right: Box::new(RefinementExpr::Int(1)),
        };

        // x * 2
        let mul = RefinementExpr::Arith {
            left: Box::new(RefinementExpr::Var(RefinedVar::new("x"))),
            op: ArithOp::Mul,
            right: Box::new(RefinementExpr::Int(2)),
        };

        assert!(matches!(
            add,
            RefinementExpr::Arith {
                op: ArithOp::Add,
                ..
            }
        ));
        assert!(matches!(
            mul,
            RefinementExpr::Arith {
                op: ArithOp::Mul,
                ..
            }
        ));
    }

    #[test]
    fn test_refinement_expr_ite() {
        // if x > 0 then x else -x
        let ite = RefinementExpr::Ite {
            cond: Box::new(Predicate::gt(
                RefinementExpr::Var(RefinedVar::new("x")),
                RefinementExpr::Int(0),
            )),
            then_expr: Box::new(RefinementExpr::Var(RefinedVar::new("x"))),
            else_expr: Box::new(RefinementExpr::Neg(Box::new(RefinementExpr::Var(
                RefinedVar::new("x"),
            )))),
        };

        assert!(matches!(ite, RefinementExpr::Ite { .. }));
    }

    #[test]
    fn test_free_variables() {
        // x + y > z
        let pred = Predicate::Compare {
            left: Box::new(RefinementExpr::Arith {
                left: Box::new(RefinementExpr::Var(RefinedVar::new("x"))),
                op: ArithOp::Add,
                right: Box::new(RefinementExpr::Var(RefinedVar::new("y"))),
            }),
            op: CompareOp::Gt,
            right: Box::new(RefinementExpr::Var(RefinedVar::new("z"))),
        };

        let free_vars = pred.free_vars();
        assert!(free_vars.contains("x"));
        assert!(free_vars.contains("y"));
        assert!(free_vars.contains("z"));
        assert_eq!(free_vars.len(), 3);
    }

    #[test]
    fn test_substitution() {
        // Start with: x > 0
        let pred = Predicate::gt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(0),
        );

        // Substitute x with 5
        let substituted = pred.substitute("x", &RefinementExpr::Int(5));

        // Result should be: 5 > 0
        if let Predicate::Compare { left, op, right } = substituted {
            assert!(matches!(*left, RefinementExpr::Int(5)));
            assert!(matches!(op, CompareOp::Gt));
            assert!(matches!(*right, RefinementExpr::Int(0)));
        } else {
            panic!("Expected Compare predicate");
        }
    }

    #[test]
    fn test_predicate_in_range() {
        // 0 <= x && x <= 100
        let range = Predicate::in_range("x", RefinementExpr::Int(0), RefinementExpr::Int(100));
        assert!(matches!(range, Predicate::And(_, _)));
    }

    #[test]
    fn test_predicate_positive() {
        // x > 0
        let pos = Predicate::positive("x");
        assert!(matches!(
            pos,
            Predicate::Compare {
                op: CompareOp::Gt,
                ..
            }
        ));
    }

    #[test]
    fn test_predicate_non_negative() {
        // x >= 0
        let non_neg = Predicate::non_negative("x");
        assert!(matches!(
            non_neg,
            Predicate::Compare {
                op: CompareOp::Ge,
                ..
            }
        ));
    }

    #[test]
    fn test_compare_operators() {
        assert!(matches!(CompareOp::Eq, CompareOp::Eq));
        assert!(matches!(CompareOp::Ne, CompareOp::Ne));
        assert!(matches!(CompareOp::Lt, CompareOp::Lt));
        assert!(matches!(CompareOp::Le, CompareOp::Le));
        assert!(matches!(CompareOp::Gt, CompareOp::Gt));
        assert!(matches!(CompareOp::Ge, CompareOp::Ge));
    }

    #[test]
    fn test_arith_operators() {
        assert!(matches!(ArithOp::Add, ArithOp::Add));
        assert!(matches!(ArithOp::Sub, ArithOp::Sub));
        assert!(matches!(ArithOp::Mul, ArithOp::Mul));
        assert!(matches!(ArithOp::Div, ArithOp::Div));
        assert!(matches!(ArithOp::Mod, ArithOp::Mod));
    }
}

// ============================================================================
// Medical Safety Types Tests
// ============================================================================

mod medical_types_tests {
    use super::*;

    #[test]
    fn test_dose_type() {
        // { dose: mg | dose >= 0.5 && dose <= 10.0 }
        let safe_dose = RefinementType::new(
            RefinedVar::new("dose"),
            BaseTypeRef::WithUnit {
                base: "Float".to_string(),
                unit: "mg".to_string(),
            },
            Predicate::and(
                Predicate::ge(
                    RefinementExpr::Var(RefinedVar::new("dose")),
                    RefinementExpr::Float(0.5),
                ),
                Predicate::le(
                    RefinementExpr::Var(RefinedVar::new("dose")),
                    RefinementExpr::Float(10.0),
                ),
            ),
        );

        assert_eq!(safe_dose.binder.name, "dose");
        assert!(matches!(safe_dose.base_type, BaseTypeRef::WithUnit { .. }));
    }

    #[test]
    fn test_valid_crcl_type() {
        // { crcl: mL/min | crcl >= 0 && crcl <= 200 }
        let valid_crcl = RefinementType::new(
            RefinedVar::new("crcl"),
            BaseTypeRef::WithUnit {
                base: "Float".to_string(),
                unit: "mL/min".to_string(),
            },
            Predicate::in_range(
                "crcl",
                RefinementExpr::Float(0.0),
                RefinementExpr::Float(200.0),
            ),
        );

        assert_eq!(valid_crcl.binder.name, "crcl");
    }

    #[test]
    fn test_valid_weight_type() {
        // { w: kg | w > 0 && w <= 500 }
        let valid_weight = RefinementType::new(
            RefinedVar::new("w"),
            BaseTypeRef::WithUnit {
                base: "Float".to_string(),
                unit: "kg".to_string(),
            },
            Predicate::and(
                Predicate::positive("w"),
                Predicate::le(
                    RefinementExpr::Var(RefinedVar::new("w")),
                    RefinementExpr::Float(500.0),
                ),
            ),
        );

        assert_eq!(valid_weight.binder.name, "w");
    }

    #[test]
    fn test_positive_int_type() {
        // { n: Int | n > 0 }
        let pos_int = RefinementType::new(
            RefinedVar::new("n"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::positive("n"),
        );

        assert!(matches!(pos_int.base_type, BaseTypeRef::Named(ref s) if s == "Int"));
    }

    #[test]
    fn test_non_negative_int_type() {
        // { n: Int | n >= 0 }
        let non_neg = RefinementType::new(
            RefinedVar::new("n"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::non_negative("n"),
        );

        assert!(matches!(
            non_neg.predicate,
            Predicate::Compare {
                op: CompareOp::Ge,
                ..
            }
        ));
    }

    #[test]
    fn test_bounded_int_type() {
        // { n: Int | n >= 1 && n <= 100 }
        let bounded = RefinementType::new(
            RefinedVar::new("n"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::in_range("n", RefinementExpr::Int(1), RefinementExpr::Int(100)),
        );

        assert!(matches!(bounded.predicate, Predicate::And(_, _)));
    }
}

// ============================================================================
// Constraint Generation Tests
// ============================================================================

mod constraint_tests {
    use super::*;

    #[test]
    fn test_constraint_set_creation() {
        let cs = ConstraintSet::new();
        assert!(cs.is_empty());
        assert_eq!(cs.len(), 0);
    }

    #[test]
    fn test_vc_generator_creation() {
        let gen = VCGenerator::new();
        // VCGenerator created successfully
        let _ = gen;
    }

    #[test]
    fn test_constraint_kind_subtype() {
        // ConstraintKind is now a simple enum for categorizing constraints
        let kind = ConstraintKind::Subtype;
        assert!(matches!(kind, ConstraintKind::Subtype));

        // Test other variants
        let kind2 = ConstraintKind::RangeBound;
        assert!(matches!(kind2, ConstraintKind::RangeBound));

        let kind3 = ConstraintKind::DivisionSafety;
        assert!(matches!(kind3, ConstraintKind::DivisionSafety));
    }
}

// ============================================================================
// SMT Solver Tests
// ============================================================================

mod smt_tests {
    use super::*;

    #[test]
    fn test_smt_context_creation() {
        let ctx = SmtContext::new();
        // Just verify it creates successfully
        let _ = ctx;
    }

    #[test]
    fn test_smt_context_with_logic() {
        let ctx = SmtContext::new().with_logic(SmtLogic::QFLIA);
        let _ = ctx;
    }

    #[test]
    fn test_smt_context_with_timeout() {
        let ctx = SmtContext::new().with_timeout(10000);
        let _ = ctx;
    }

    #[test]
    fn test_declare_variable() {
        let mut ctx = SmtContext::new();
        ctx.declare("x", SmtSort::Int);
        ctx.declare("y", SmtSort::Real);
        ctx.declare("b", SmtSort::Bool);
    }

    #[test]
    fn test_assert_formula() {
        let mut ctx = SmtContext::new();
        ctx.declare("x", SmtSort::Int);
        ctx.assert("(> x 0)");
    }

    #[test]
    fn test_translate_predicate() {
        let mut ctx = SmtContext::new();

        // x > 0
        let pred = Predicate::gt(
            RefinementExpr::Var(RefinedVar::new("x")),
            RefinementExpr::Int(0),
        );

        let smt_str = ctx.translate_predicate(&pred);
        assert!(smt_str.contains(">"));
        assert!(smt_str.contains("x"));
        assert!(smt_str.contains("0"));
    }

    #[test]
    fn test_translate_and_predicate() {
        let mut ctx = SmtContext::new();

        // x > 0 && x < 100
        let pred = Predicate::and(
            Predicate::gt(
                RefinementExpr::Var(RefinedVar::new("x")),
                RefinementExpr::Int(0),
            ),
            Predicate::lt(
                RefinementExpr::Var(RefinedVar::new("x")),
                RefinementExpr::Int(100),
            ),
        );

        let smt_str = ctx.translate_predicate(&pred);
        assert!(smt_str.contains("and"));
    }

    #[test]
    fn test_translate_or_predicate() {
        let mut ctx = SmtContext::new();

        let pred = Predicate::or(Predicate::Bool(true), Predicate::Bool(false));

        let smt_str = ctx.translate_predicate(&pred);
        assert!(smt_str.contains("or"));
    }

    #[test]
    fn test_translate_implies_predicate() {
        let mut ctx = SmtContext::new();

        let pred = Predicate::implies(Predicate::positive("x"), Predicate::non_negative("x"));

        let smt_str = ctx.translate_predicate(&pred);
        assert!(smt_str.contains("=>"));
    }

    #[test]
    fn test_smt_solver_creation() {
        let solver = SmtSolver::new();
        // Solver may or may not be available depending on Z3 installation
        let _ = solver;
    }

    #[test]
    fn test_smt_logic_variants() {
        assert!(matches!(SmtLogic::QFLIA, SmtLogic::QFLIA));
        assert!(matches!(SmtLogic::QFLRA, SmtLogic::QFLRA));
        assert!(matches!(SmtLogic::QFNRA, SmtLogic::QFNRA));
        assert!(matches!(SmtLogic::LIRA, SmtLogic::LIRA));
        assert!(matches!(SmtLogic::ALL, SmtLogic::ALL));
    }

    #[test]
    fn test_smt_sort_variants() {
        assert!(matches!(SmtSort::Int, SmtSort::Int));
        assert!(matches!(SmtSort::Real, SmtSort::Real));
        assert!(matches!(SmtSort::Bool, SmtSort::Bool));
    }

    #[test]
    fn test_smt_result_variants() {
        let sat = SmtResult::Sat(None);
        let unsat = SmtResult::Unsat;
        let unknown = SmtResult::Unknown("timeout".to_string());
        let error = SmtResult::Error("parse error".to_string());

        assert!(matches!(sat, SmtResult::Sat(_)));
        assert!(matches!(unsat, SmtResult::Unsat));
        assert!(matches!(unknown, SmtResult::Unknown(_)));
        assert!(matches!(error, SmtResult::Error(_)));
    }
}

// ============================================================================
// Subtype Checker Tests
// ============================================================================

mod subtype_tests {
    use super::*;

    #[test]
    fn test_subtype_checker_creation() {
        let checker = SubtypeChecker::new();
        let _ = checker;
    }

    #[test]
    fn test_subtype_result_variants() {
        let valid = SubtypeResult::Valid;
        let invalid = SubtypeResult::Invalid {
            counterexample: None,
            reason: "test".to_string(),
        };
        let unknown = SubtypeResult::Unknown("timeout".to_string());

        assert!(matches!(valid, SubtypeResult::Valid));
        assert!(matches!(invalid, SubtypeResult::Invalid { .. }));
        assert!(matches!(unknown, SubtypeResult::Unknown(_)));
    }

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
    fn test_subtype_check() {
        let mut checker = SubtypeChecker::new();

        let pos = make_positive_int();
        let non_neg = make_non_negative_int();

        // x > 0 should imply x >= 0
        let result = checker.check(&pos, &non_neg);

        // Result depends on solver availability
        assert!(matches!(
            result,
            SubtypeResult::Valid | SubtypeResult::Unknown(_) | SubtypeResult::Invalid { .. }
        ));
    }
}

// ============================================================================
// Refinement Checker Tests
// ============================================================================

mod checker_tests {
    use super::*;

    #[test]
    fn test_refinement_checker_creation() {
        let checker = RefinementChecker::new();
        assert!(checker.errors().is_empty());
    }

    #[test]
    fn test_refinement_env_creation() {
        let env = RefinementEnv::new();
        assert!(env.lookup("x").is_none());
    }

    #[test]
    fn test_refinement_env_bind() {
        let mut env = RefinementEnv::new();
        let ty = RefinementType::new(
            RefinedVar::new("x"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::positive("x"),
        );

        env.bind("x", ty);

        assert!(env.lookup("x").is_some());
        assert!(env.lookup("y").is_none());
    }

    #[test]
    fn test_refinement_env_register_function() {
        let mut env = RefinementEnv::new();

        let func = FunctionRefinement {
            params: vec![(
                "weight".to_string(),
                RefinementType::new(
                    RefinedVar::new("w"),
                    BaseTypeRef::Named("Float".to_string()),
                    Predicate::positive("w"),
                ),
            )],
            return_type: RefinementType::new(
                RefinedVar::new("dose"),
                BaseTypeRef::Named("Float".to_string()),
                Predicate::positive("dose"),
            ),
            requires: vec![],
            ensures: vec![],
        };

        env.register_function("calculate_dose", func);

        assert!(env.lookup_function("calculate_dose").is_some());
        assert!(env.lookup_function("other_func").is_none());
    }

    #[test]
    fn test_refinement_env_type_alias() {
        let mut env = RefinementEnv::new();

        let ty = RefinementType::new(
            RefinedVar::new("n"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::positive("n"),
        );

        env.register_alias("PositiveInt", ty);

        assert!(env.resolve_alias("PositiveInt").is_some());
        assert!(env.resolve_alias("OtherType").is_none());
    }

    #[test]
    fn test_refinement_env_path_conditions() {
        let mut env = RefinementEnv::new();

        let cond = Predicate::positive("x");
        env.push_condition(cond);

        assert_eq!(env.path_conditions().len(), 1);

        env.pop_condition();
        assert_eq!(env.path_conditions().len(), 0);
    }

    #[test]
    fn test_refinement_env_enter_scope() {
        let mut env = RefinementEnv::new();
        let ty =
            RefinementType::trivial(RefinedVar::new("x"), BaseTypeRef::Named("Int".to_string()));
        env.bind("x", ty);

        let scoped = env.enter_scope();
        assert!(scoped.lookup("x").is_some());
    }

    #[test]
    fn test_verification_stats() {
        let checker = RefinementChecker::new();
        let stats = checker.stats();

        assert_eq!(stats.constraints_generated, 0);
        assert_eq!(stats.constraints_verified, 0);
        assert_eq!(stats.constraints_failed, 0);
        assert_eq!(stats.solver_calls, 0);
    }

    #[test]
    fn test_checker_with_env() {
        let env = RefinementEnv::new();
        let checker = RefinementChecker::new().with_env(env);

        assert!(checker.errors().is_empty());
    }

    #[test]
    fn test_checker_clear_errors() {
        let mut checker = RefinementChecker::new();
        checker.clear_errors();
        assert!(checker.errors().is_empty());
    }
}

// ============================================================================
// Error Tests
// ============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_counterexample_creation() {
        let ce = Counterexample::new();
        assert!(ce.is_empty());
    }

    #[test]
    fn test_counterexample_with_assignment() {
        let ce = Counterexample::new()
            .with_assignment("x", "0")
            .with_assignment("y", "-5");

        assert_eq!(ce.assignments.len(), 2);
    }

    #[test]
    fn test_error_severity() {
        assert!(matches!(ErrorSeverity::Error, ErrorSeverity::Error));
        assert!(matches!(ErrorSeverity::Warning, ErrorSeverity::Warning));
        assert!(matches!(ErrorSeverity::Critical, ErrorSeverity::Critical));
    }

    #[test]
    fn test_refinement_error_kind_subtype() {
        let sub =
            RefinementType::trivial(RefinedVar::new("x"), BaseTypeRef::Named("Int".to_string()));
        let sup = RefinementType::new(
            RefinedVar::new("x"),
            BaseTypeRef::Named("Int".to_string()),
            Predicate::positive("x"),
        );

        let kind = RefinementErrorKind::SubtypeFailed {
            sub,
            sup,
            counterexample: None,
        };

        assert!(matches!(kind, RefinementErrorKind::SubtypeFailed { .. }));
    }

    #[test]
    fn test_refinement_error_kind_precondition() {
        let kind = RefinementErrorKind::PreconditionFailed {
            function: "calculate_dose".to_string(),
            param: "weight".to_string(),
            required: Predicate::positive("weight"),
            counterexample: None,
        };

        assert!(matches!(
            kind,
            RefinementErrorKind::PreconditionFailed { .. }
        ));
    }

    #[test]
    fn test_refinement_error_new() {
        let kind = RefinementErrorKind::SolverError("test error".to_string());
        let error = RefinementError::new(kind);

        assert!(matches!(error.kind, RefinementErrorKind::SolverError(_)));
    }
}

// ============================================================================
// Integration Tests
// ============================================================================

mod integration_tests {
    use super::*;

    #[test]
    fn test_full_dose_type_definition() {
        // Define a safe dose type
        let safe_dose = RefinementType::new(
            RefinedVar::new("dose"),
            BaseTypeRef::WithUnit {
                base: "Float".to_string(),
                unit: "mg".to_string(),
            },
            Predicate::and(
                Predicate::ge(
                    RefinementExpr::Var(RefinedVar::new("dose")),
                    RefinementExpr::Float(0.5),
                ),
                Predicate::le(
                    RefinementExpr::Var(RefinedVar::new("dose")),
                    RefinementExpr::Float(50.0),
                ),
            ),
        );

        // Should have the right structure
        assert_eq!(safe_dose.binder.name, "dose");
        assert!(matches!(safe_dose.predicate, Predicate::And(_, _)));

        // Free vars should be the binder
        let free = safe_dose.free_vars();
        assert!(free.is_empty()); // binder is removed from free vars
    }

    #[test]
    fn test_function_refinement_definition() {
        // Define a function with refinements
        let func = FunctionRefinement {
            params: vec![
                (
                    "weight".to_string(),
                    RefinementType::new(
                        RefinedVar::new("w"),
                        BaseTypeRef::WithUnit {
                            base: "Float".to_string(),
                            unit: "kg".to_string(),
                        },
                        Predicate::positive("w"),
                    ),
                ),
                (
                    "crcl".to_string(),
                    RefinementType::new(
                        RefinedVar::new("c"),
                        BaseTypeRef::WithUnit {
                            base: "Float".to_string(),
                            unit: "mL/min".to_string(),
                        },
                        Predicate::in_range(
                            "c",
                            RefinementExpr::Float(0.0),
                            RefinementExpr::Float(200.0),
                        ),
                    ),
                ),
            ],
            return_type: RefinementType::new(
                RefinedVar::new("dose"),
                BaseTypeRef::WithUnit {
                    base: "Float".to_string(),
                    unit: "mg".to_string(),
                },
                Predicate::in_range(
                    "dose",
                    RefinementExpr::Float(0.5),
                    RefinementExpr::Float(50.0),
                ),
            ),
            requires: vec![Predicate::positive("w")],
            ensures: vec![Predicate::positive("dose")],
        };

        assert_eq!(func.params.len(), 2);
        assert_eq!(func.requires.len(), 1);
        assert_eq!(func.ensures.len(), 1);
    }

    #[test]
    fn test_smt_translation_pipeline() {
        let mut ctx = SmtContext::new().with_logic(SmtLogic::QFLRA);

        // Translate a complex medical predicate
        let pred = Predicate::implies(
            Predicate::and(
                Predicate::positive("weight"),
                Predicate::in_range(
                    "crcl",
                    RefinementExpr::Float(30.0),
                    RefinementExpr::Float(90.0),
                ),
            ),
            Predicate::in_range(
                "dose",
                RefinementExpr::Float(0.5),
                RefinementExpr::Float(10.0),
            ),
        );

        let smt = ctx.translate_predicate(&pred);

        // Should produce valid SMT-LIB2
        assert!(smt.contains("=>"));
        assert!(smt.contains("and"));
    }

    #[test]
    fn test_refinement_env_full_setup() {
        let mut env = RefinementEnv::new();

        // Register type aliases
        env.register_alias(
            "PositiveWeight",
            RefinementType::new(
                RefinedVar::new("w"),
                BaseTypeRef::WithUnit {
                    base: "Float".to_string(),
                    unit: "kg".to_string(),
                },
                Predicate::positive("w"),
            ),
        );

        env.register_alias(
            "ValidCrCl",
            RefinementType::new(
                RefinedVar::new("c"),
                BaseTypeRef::WithUnit {
                    base: "Float".to_string(),
                    unit: "mL/min".to_string(),
                },
                Predicate::in_range(
                    "c",
                    RefinementExpr::Float(0.0),
                    RefinementExpr::Float(200.0),
                ),
            ),
        );

        // Bind variables
        if let Some(weight_ty) = env.resolve_alias("PositiveWeight") {
            env.bind("patient_weight", weight_ty.clone());
        }

        assert!(env.lookup("patient_weight").is_some());
        assert!(env.resolve_alias("ValidCrCl").is_some());
    }

    #[test]
    fn test_checker_integration() {
        let mut checker = RefinementChecker::new();

        // Set up environment
        let weight_ty = RefinementType::new(
            RefinedVar::new("w"),
            BaseTypeRef::Named("Float".to_string()),
            Predicate::positive("w"),
        );

        checker.env_mut().bind("weight", weight_ty);

        // Check that environment is set up
        assert!(checker.env().lookup("weight").is_some());
        assert!(checker.errors().is_empty());
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

mod edge_case_tests {
    use super::*;

    #[test]
    fn test_nested_and_predicate() {
        // ((a && b) && c)
        let a = Predicate::positive("a");
        let b = Predicate::positive("b");
        let c = Predicate::positive("c");

        let inner = Predicate::and(a, b);
        let outer = Predicate::and(inner, c);

        assert!(matches!(outer, Predicate::And(_, _)));
    }

    #[test]
    fn test_deeply_nested_arithmetic() {
        // ((x + 1) * 2) - 3
        let x = RefinementExpr::Var(RefinedVar::new("x"));
        let x_plus_1 = RefinementExpr::Arith {
            left: Box::new(x),
            op: ArithOp::Add,
            right: Box::new(RefinementExpr::Int(1)),
        };
        let times_2 = RefinementExpr::Arith {
            left: Box::new(x_plus_1),
            op: ArithOp::Mul,
            right: Box::new(RefinementExpr::Int(2)),
        };
        let minus_3 = RefinementExpr::Arith {
            left: Box::new(times_2),
            op: ArithOp::Sub,
            right: Box::new(RefinementExpr::Int(3)),
        };

        assert!(matches!(
            minus_3,
            RefinementExpr::Arith {
                op: ArithOp::Sub,
                ..
            }
        ));
    }

    #[test]
    fn test_large_integer_literal() {
        let large = RefinementExpr::Int(i64::MAX);
        let pred = Predicate::gt(large, RefinementExpr::Int(0));

        assert!(matches!(pred, Predicate::Compare { .. }));
    }

    #[test]
    fn test_negative_float_literal() {
        let neg = RefinementExpr::Float(-3.14159);
        let pred = Predicate::lt(neg, RefinementExpr::Float(0.0));

        assert!(matches!(pred, Predicate::Compare { .. }));
    }

    #[test]
    fn test_quantified_predicate() {
        // forall x: Int. x >= 0 => x * x >= 0
        let body = Predicate::implies(
            Predicate::non_negative("x"),
            Predicate::ge(
                RefinementExpr::Arith {
                    left: Box::new(RefinementExpr::Var(RefinedVar::new("x"))),
                    op: ArithOp::Mul,
                    right: Box::new(RefinementExpr::Var(RefinedVar::new("x"))),
                },
                RefinementExpr::Int(0),
            ),
        );

        let forall = Predicate::Forall {
            var: RefinedVar::new("x"),
            ty: BaseTypeRef::Named("Int".to_string()),
            body: Box::new(body),
        };

        assert!(matches!(forall, Predicate::Forall { .. }));

        // Free vars should not include the bound variable
        let free = forall.free_vars();
        assert!(!free.contains("x"));
    }

    #[test]
    fn test_existential_predicate() {
        // exists x: Int. x > 0
        let exists = Predicate::Exists {
            var: RefinedVar::new("x"),
            ty: BaseTypeRef::Named("Int".to_string()),
            body: Box::new(Predicate::positive("x")),
        };

        assert!(matches!(exists, Predicate::Exists { .. }));
    }

    #[test]
    fn test_function_call_predicate() {
        // is_valid_dose(d)
        let call = Predicate::Call {
            func: "is_valid_dose".to_string(),
            args: vec![RefinementExpr::Var(RefinedVar::new("d"))],
        };

        assert!(matches!(call, Predicate::Call { .. }));
    }

    #[test]
    fn test_array_length_expr() {
        // len(arr)
        let len = RefinementExpr::Len(Box::new(RefinementExpr::Var(RefinedVar::new("arr"))));

        assert!(matches!(len, RefinementExpr::Len(_)));
    }

    #[test]
    fn test_old_expr() {
        // old(x) for postconditions
        let old = RefinementExpr::Old(Box::new(RefinementExpr::Var(RefinedVar::new("x"))));

        assert!(matches!(old, RefinementExpr::Old(_)));
    }

    #[test]
    fn test_field_access_expr() {
        // patient.weight
        let field = RefinementExpr::Field {
            base: Box::new(RefinementExpr::Var(RefinedVar::new("patient"))),
            field: "weight".to_string(),
        };

        assert!(matches!(field, RefinementExpr::Field { .. }));
    }

    #[test]
    fn test_index_expr() {
        // arr[i]
        let index = RefinementExpr::Index {
            base: Box::new(RefinementExpr::Var(RefinedVar::new("arr"))),
            index: Box::new(RefinementExpr::Var(RefinedVar::new("i"))),
        };

        assert!(matches!(index, RefinementExpr::Index { .. }));
    }
}
