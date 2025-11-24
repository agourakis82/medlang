use medlangc::{ast::*, lexer::tokenize, parser::parse_program};

#[test]
fn test_parse_canonical_example() {
    // Read the canonical example file
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/examples/one_comp_oral_pk.medlang");
    let source =
        std::fs::read_to_string(&example_path).expect("Failed to read canonical example file");

    // Tokenize
    let tokens = tokenize(&source).expect("Failed to tokenize canonical example");

    // Parse
    let program = parse_program(&tokens).expect("Failed to parse canonical example");

    // Validate structure
    assert!(
        program.declarations.len() >= 4,
        "Expected at least 4 top-level declarations (model, population, measure, timeline)"
    );

    // Count declaration types
    let mut model_count = 0;
    let mut population_count = 0;
    let mut measure_count = 0;
    let mut timeline_count = 0;

    for decl in &program.declarations {
        match decl {
            Declaration::Model(_) => model_count += 1,
            Declaration::Population(_) => population_count += 1,
            Declaration::Measure(_) => measure_count += 1,
            Declaration::Timeline(_) => timeline_count += 1,
            Declaration::Cohort(_) => {}
            Declaration::Protocol(_) => {}
        }
    }

    assert_eq!(model_count, 1, "Expected 1 model definition");
    assert_eq!(population_count, 1, "Expected 1 population definition");
    assert_eq!(measure_count, 1, "Expected 1 measure definition");
    assert_eq!(timeline_count, 1, "Expected 1 timeline definition");

    // Validate model structure
    let model = match &program.declarations[0] {
        Declaration::Model(m) => m,
        _ => panic!("First declaration should be a model"),
    };

    assert_eq!(model.name, "OneCompOral");

    // Count model items
    let mut state_count = 0;
    let mut param_count = 0;
    let mut ode_count = 0;
    let mut obs_count = 0;

    for item in &model.items {
        match item {
            ModelItem::State(_) => state_count += 1,
            ModelItem::Param(_) => param_count += 1,
            ModelItem::ODE(_) => ode_count += 1,
            ModelItem::Observable(_) => obs_count += 1,
            ModelItem::Input(_)
            | ModelItem::Let(_)
            | ModelItem::Submodel(_)
            | ModelItem::Connect(_) => {
                // New constructs for composite models - not in V0 canonical example
            }
        }
    }

    assert_eq!(
        state_count, 2,
        "Expected 2 state variables (A_gut, A_central)"
    );
    assert_eq!(param_count, 3, "Expected 3 parameters (Ka, CL, V)");
    assert_eq!(ode_count, 2, "Expected 2 ODE equations");
    assert_eq!(obs_count, 1, "Expected 1 observable (C_plasma)");

    println!("✅ Successfully parsed canonical example:");
    println!(
        "   - Model: {} with {} states, {} params, {} ODEs, {} observables",
        model.name, state_count, param_count, ode_count, obs_count
    );
    println!("   - {} top-level declarations", program.declarations.len());
}

#[test]
fn test_parse_model_with_types() {
    let source = r#"
model SimpleModel {
    state A : DoseMass
    param K : RateConst
    dA/dt = -K * A
    obs C : ConcMass = A / 50.0
}
    "#;

    let tokens = tokenize(source).unwrap();
    let program = parse_program(&tokens).unwrap();

    assert_eq!(program.declarations.len(), 1);

    let model = match &program.declarations[0] {
        Declaration::Model(m) => m,
        _ => panic!("Expected model"),
    };

    assert_eq!(model.name, "SimpleModel");
    assert_eq!(model.items.len(), 4);

    // Check state has correct type
    match &model.items[0] {
        ModelItem::State(s) => {
            assert_eq!(s.name, "A");
            match &s.ty {
                TypeExpr::Unit(UnitType::DoseMass) => {}
                _ => panic!("Expected DoseMass type"),
            }
        }
        _ => panic!("Expected state"),
    }

    // Check param has correct type
    match &model.items[1] {
        ModelItem::Param(p) => {
            assert_eq!(p.name, "K");
            match &p.ty {
                TypeExpr::Unit(UnitType::RateConst) => {}
                _ => panic!("Expected RateConst type"),
            }
        }
        _ => panic!("Expected param"),
    }
}

#[test]
fn test_parse_population_with_random_effects() {
    let source = r#"
population TestPop {
    model TestModel
    param CL_pop : Clearance
    param omega_CL : f64
    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    bind_params(patient) {
        let CL_ind = CL_pop * exp(eta_CL)
        model.CL = CL_ind
    }
}
    "#;

    let tokens = tokenize(source).unwrap();
    let program = parse_program(&tokens).unwrap();

    assert_eq!(program.declarations.len(), 1);

    let pop = match &program.declarations[0] {
        Declaration::Population(p) => p,
        _ => panic!("Expected population"),
    };

    assert_eq!(pop.name, "TestPop");

    // Count items
    let mut model_ref = false;
    let mut param_count = 0;
    let mut rand_count = 0;
    let mut bind_count = 0;

    for item in &pop.items {
        match item {
            PopulationItem::ModelRef(_) => model_ref = true,
            PopulationItem::Param(_) => param_count += 1,
            PopulationItem::RandomEffect(_) => rand_count += 1,
            PopulationItem::BindParams(_) => bind_count += 1,
            _ => {}
        }
    }

    assert!(model_ref, "Expected model reference");
    assert_eq!(param_count, 2, "Expected 2 params");
    assert_eq!(rand_count, 1, "Expected 1 random effect");
    assert_eq!(bind_count, 1, "Expected 1 bind_params block");
}

#[test]
fn test_parse_timeline_with_events() {
    let source = r#"
timeline SingleDose {
    at 0.0_h:
        dose {
            amount = 100.0_mg
            to = model.A_gut
        }
    at 1.0_h:
        observe measure.plasma
    at 2.0_h:
        observe measure.plasma
    at 4.0_h:
        observe measure.plasma
}
    "#;

    let tokens = tokenize(source).unwrap();
    let program = parse_program(&tokens).unwrap();

    assert_eq!(program.declarations.len(), 1);

    let timeline = match &program.declarations[0] {
        Declaration::Timeline(t) => t,
        _ => panic!("Expected timeline"),
    };

    assert_eq!(timeline.name, "SingleDose");
    assert_eq!(timeline.events.len(), 4);

    // Check first event is dose
    match &timeline.events[0] {
        Event::Dose(d) => {
            // Verify dose has time (unit literal), amount (unit literal), and target
            assert!(matches!(
                d.time.kind,
                ExprKind::Literal(Literal::UnitFloat { .. })
            ));
            assert!(matches!(
                d.amount.kind,
                ExprKind::Literal(Literal::UnitFloat { .. })
            ));
        }
        _ => panic!("Expected dose event"),
    }

    // Check remaining events are observe
    for i in 1..4 {
        match &timeline.events[i] {
            Event::Observe(_) => {}
            _ => panic!("Expected observe event at index {}", i),
        }
    }
}

#[test]
fn test_parse_measure_definition() {
    let source = r#"
measure ProportionalError {
    pred : ConcMass
    obs : ConcMass
    param sigma_prop : f64
    log_likelihood = normal_lpdf(obs, pred, sigma_prop * pred)
}
    "#;

    let tokens = tokenize(source).unwrap();
    let program = parse_program(&tokens).unwrap();

    assert_eq!(program.declarations.len(), 1);

    let measure = match &program.declarations[0] {
        Declaration::Measure(m) => m,
        _ => panic!("Expected measure"),
    };

    assert_eq!(measure.name, "ProportionalError");
    assert_eq!(measure.items.len(), 4);

    // Check items
    let mut pred_count = 0;
    let mut obs_count = 0;
    let mut param_count = 0;
    let mut ll_count = 0;

    for item in &measure.items {
        match item {
            MeasureItem::Pred(_) => pred_count += 1,
            MeasureItem::Obs(_) => obs_count += 1,
            MeasureItem::Param(_) => param_count += 1,
            MeasureItem::LogLikelihood(_) => ll_count += 1,
        }
    }

    assert_eq!(pred_count, 1);
    assert_eq!(obs_count, 1);
    assert_eq!(param_count, 1);
    assert_eq!(ll_count, 1);
}

#[test]
fn test_expression_precedence() {
    let source = r#"
model PrecedenceTest {
    state A : DoseMass
    dA/dt = -1.0 + 2.0 * 3.0
}
    "#;

    let tokens = tokenize(source).unwrap();
    let program = parse_program(&tokens).unwrap();

    let model = match &program.declarations[0] {
        Declaration::Model(m) => m,
        _ => panic!("Expected model"),
    };

    let ode = match &model.items[1] {
        ModelItem::ODE(o) => o,
        _ => panic!("Expected ODE"),
    };

    // Should parse as: -1.0 + (2.0 * 3.0)
    // Which is: Add(Unary(Neg, 1.0), Mul(2.0, 3.0))
    match &ode.rhs.kind {
        ExprKind::Binary(BinaryOp::Add, left, right) => {
            // Left should be unary negation
            assert!(matches!(left.kind, ExprKind::Unary(UnaryOp::Neg, _)));
            // Right should be multiplication
            assert!(matches!(right.kind, ExprKind::Binary(BinaryOp::Mul, _, _)));
        }
        _ => panic!("Expected addition at top level"),
    }
}
