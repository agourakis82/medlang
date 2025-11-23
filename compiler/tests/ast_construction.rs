//! Test AST construction for simple MedLang programs.
//!
//! This demonstrates building AST nodes programmatically,
//! which will later be produced by the parser.

use medlangc::ast::*;

#[test]
fn test_simple_model_ast() {
    // Construct: model SimpleModel { state A : Mass }
    let model = ModelDef {
        name: "SimpleModel".to_string(),
        items: vec![ModelItem::State(StateDecl {
            name: "A".to_string(),
            ty: TypeExpr::Unit(UnitType::Mass),
            span: None,
        })],
        span: None,
    };

    assert_eq!(model.name, "SimpleModel");
    assert_eq!(model.items.len(), 1);

    match &model.items[0] {
        ModelItem::State(s) => {
            assert_eq!(s.name, "A");
            assert!(matches!(s.ty, TypeExpr::Unit(UnitType::Mass)));
        }
        _ => panic!("Expected state declaration"),
    }
}

#[test]
fn test_ode_equation_ast() {
    // Construct: dA/dt = -Ka * A
    let ode = ODEEquation {
        state_name: "A".to_string(),
        rhs: Expr::binary(
            BinaryOp::Mul,
            Expr::binary(
                BinaryOp::Sub,
                Expr::literal(0.0),
                Expr::ident("Ka".to_string()),
            ),
            Expr::ident("A".to_string()),
        ),
        span: None,
    };

    assert_eq!(ode.state_name, "A");

    // Check structure: Binary(Mul, Binary(Sub, 0, Ka), A)
    match &ode.rhs.kind {
        ExprKind::Binary(BinaryOp::Mul, left, right) => {
            // Left should be: -(Ka), i.e., 0 - Ka
            match &left.kind {
                ExprKind::Binary(BinaryOp::Sub, zero, ka) => {
                    assert!(matches!(zero.kind, ExprKind::Literal(Literal::Float(0.0))));
                    assert!(matches!(ka.kind, ExprKind::Ident(ref name) if name == "Ka"));
                }
                _ => panic!("Expected subtraction"),
            }

            // Right should be: A
            assert!(matches!(right.kind, ExprKind::Ident(ref name) if name == "A"));
        }
        _ => panic!("Expected multiplication"),
    }
}

#[test]
fn test_observable_with_division() {
    // Construct: obs C : ConcMass = A / V
    let obs = ObservableDecl {
        name: "C".to_string(),
        ty: TypeExpr::Unit(UnitType::ConcMass),
        expr: Expr::binary(
            BinaryOp::Div,
            Expr::ident("A".to_string()),
            Expr::ident("V".to_string()),
        ),
        span: None,
    };

    assert_eq!(obs.name, "C");
    assert!(matches!(obs.ty, TypeExpr::Unit(UnitType::ConcMass)));

    match &obs.expr.kind {
        ExprKind::Binary(BinaryOp::Div, num, den) => {
            assert!(matches!(num.kind, ExprKind::Ident(ref n) if n == "A"));
            assert!(matches!(den.kind, ExprKind::Ident(ref n) if n == "V"));
        }
        _ => panic!("Expected division"),
    }
}

#[test]
fn test_dose_event() {
    // Construct: at 0.0_h: dose { amount = 100.0_mg; to = Model.A_gut }
    let dose = DoseEvent {
        time: Expr::unit_literal(0.0, "h".to_string()),
        amount: Expr::unit_literal(100.0, "mg".to_string()),
        target: QualifiedName::new(vec!["Model".to_string(), "A_gut".to_string()]),
        span: None,
    };

    match &dose.time.kind {
        ExprKind::Literal(Literal::UnitFloat { value, unit }) => {
            assert_eq!(*value, 0.0);
            assert_eq!(unit, "h");
        }
        _ => panic!("Expected unit literal"),
    }

    match &dose.amount.kind {
        ExprKind::Literal(Literal::UnitFloat { value, unit }) => {
            assert_eq!(*value, 100.0);
            assert_eq!(unit, "mg");
        }
        _ => panic!("Expected unit literal"),
    }

    assert_eq!(dose.target.to_string(), "Model.A_gut");
}

#[test]
fn test_function_call_with_named_args() {
    // Construct: Normal_logpdf(x = residual, mu = 0.0, sd = sigma_prop)
    let call = Expr::call(
        "Normal_logpdf".to_string(),
        vec![
            Argument {
                name: Some("x".to_string()),
                value: Expr::ident("residual".to_string()),
            },
            Argument {
                name: Some("mu".to_string()),
                value: Expr::literal(0.0),
            },
            Argument {
                name: Some("sd".to_string()),
                value: Expr::ident("sigma_prop".to_string()),
            },
        ],
    );

    match &call.kind {
        ExprKind::Call(name, args) => {
            assert_eq!(name, "Normal_logpdf");
            assert_eq!(args.len(), 3);

            assert_eq!(args[0].name, Some("x".to_string()));
            assert_eq!(args[1].name, Some("mu".to_string()));
            assert_eq!(args[2].name, Some("sd".to_string()));
        }
        _ => panic!("Expected function call"),
    }
}

#[test]
fn test_let_statement() {
    // Construct: let w = patient.WT / 70.0_kg
    let stmt = Statement::Let {
        name: "w".to_string(),
        value: Expr::binary(
            BinaryOp::Div,
            Expr::qualified(vec!["patient".to_string(), "WT".to_string()]),
            Expr::unit_literal(70.0, "kg".to_string()),
        ),
        span: None,
    };

    match stmt {
        Statement::Let { name, value, .. } => {
            assert_eq!(name, "w");

            match &value.kind {
                ExprKind::Binary(BinaryOp::Div, num, den) => {
                    match &num.kind {
                        ExprKind::QualifiedName(qn) => {
                            assert_eq!(qn.to_string(), "patient.WT");
                        }
                        _ => panic!("Expected qualified name"),
                    }

                    assert!(matches!(
                        den.kind,
                        ExprKind::Literal(Literal::UnitFloat { value: 70.0, .. })
                    ));
                }
                _ => panic!("Expected division"),
            }
        }
        _ => panic!("Expected let statement"),
    }
}

#[test]
fn test_random_effect_declaration() {
    // Construct: rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    let rand_eff = RandomEffectDecl {
        name: "eta_CL".to_string(),
        ty: TypeExpr::Simple("f64".to_string()),
        dist: DistributionExpr::Normal {
            mu: Expr::literal(0.0),
            sigma: Expr::ident("omega_CL".to_string()),
        },
        span: None,
    };

    assert_eq!(rand_eff.name, "eta_CL");
    assert!(matches!(rand_eff.ty, TypeExpr::Simple(ref t) if t == "f64"));

    match &rand_eff.dist {
        DistributionExpr::Normal { mu, sigma } => {
            assert!(matches!(mu.kind, ExprKind::Literal(Literal::Float(0.0))));
            assert!(matches!(sigma.kind, ExprKind::Ident(ref n) if n == "omega_CL"));
        }
        _ => panic!("Expected Normal distribution"),
    }
}

#[test]
fn test_complete_program() {
    // Construct a minimal complete program
    let program = Program {
        declarations: vec![
            Declaration::Model(ModelDef {
                name: "Test".to_string(),
                items: vec![ModelItem::State(StateDecl {
                    name: "A".to_string(),
                    ty: TypeExpr::Unit(UnitType::Mass),
                    span: None,
                })],
                span: None,
            }),
            Declaration::Cohort(CohortDef {
                name: "TestCohort".to_string(),
                items: vec![CohortItem::DataFile("data.csv".to_string())],
                span: None,
            }),
        ],
    };

    assert_eq!(program.declarations.len(), 2);

    match &program.declarations[0] {
        Declaration::Model(m) => assert_eq!(m.name, "Test"),
        _ => panic!("Expected model"),
    }

    match &program.declarations[1] {
        Declaration::Cohort(c) => {
            assert_eq!(c.name, "TestCohort");
            assert_eq!(c.items.len(), 1);
        }
        _ => panic!("Expected cohort"),
    }
}
