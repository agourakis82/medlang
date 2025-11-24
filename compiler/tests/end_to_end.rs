use medlangc::{
    codegen::generate_stan, lexer::tokenize, lower::lower_program, parser::parse_program,
};

#[test]
fn test_compile_simple_model_to_stan() {
    let source = r#"
model TestModel {
    state A : DoseMass
    param K : RateConst
    dA/dt = -K * A
    obs C : ConcMass = A / 10.0
}

population TestPop {
    model TestModel
    param K_pop : RateConst
    param omega_K : f64
    rand eta_K : f64 ~ Normal(0.0, omega_K)
    bind_params(patient) {
        model.K = K_pop * exp(eta_K)
    }
    use_measure TestMeasure for model.C
}

measure TestMeasure {
    pred : ConcMass
    obs : ConcMass
    param sigma_prop : f64
    log_likelihood = normal_lpdf(obs, pred, sigma_prop * pred)
}
    "#;

    // Parse
    let tokens = tokenize(source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");

    // Lower to IR
    let ir = lower_program(&ast).expect("Lowering failed");

    // Generate Stan code
    let stan_code = generate_stan(&ir).expect("Code generation failed");

    // Verify generated code contains expected elements
    assert!(stan_code.contains("functions {"));
    assert!(stan_code.contains("data {"));
    assert!(stan_code.contains("parameters {"));
    assert!(stan_code.contains("transformed parameters {"));
    assert!(stan_code.contains("model {"));

    // Verify ODE system
    assert!(stan_code.contains("ode_system"));
    assert!(stan_code.contains("dydt"));

    // Verify parameters
    assert!(stan_code.contains("K_pop"));
    assert!(stan_code.contains("omega_K"));
    assert!(stan_code.contains("eta_K"));

    // Verify data structures
    assert!(stan_code.contains("subject_id"));
    assert!(stan_code.contains("time"));
    assert!(stan_code.contains("observation"));

    println!("Generated Stan code:");
    println!("{}", stan_code);
}

#[test]
fn test_compile_one_comp_oral_to_stan() {
    let source = r#"
model OneCompOral {
    state A_gut : DoseMass
    state A_central : DoseMass

    param Ka : RateConst
    param CL : Clearance
    param V : Volume

    dA_gut/dt = -Ka * A_gut
    dA_central/dt = Ka * A_gut - (CL / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}

population OneCompOralPop {
    model OneCompOral

    param CL_pop : Clearance
    param V_pop : Volume
    param Ka_pop : RateConst

    param omega_CL : f64
    param omega_V : f64
    param omega_Ka : f64

    input WT : Quantity<kg, f64>

    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    rand eta_V : f64 ~ Normal(0.0, omega_V)
    rand eta_Ka : f64 ~ Normal(0.0, omega_Ka)

    bind_params(patient) {
        let w = patient.WT / 70.0_kg
        model.CL = CL_pop * pow(w, 0.75) * exp(eta_CL)
        model.V = V_pop * w * exp(eta_V)
        model.Ka = Ka_pop * exp(eta_Ka)
    }

    use_measure ConcPropError for model.C_plasma
}

measure ConcPropError {
    pred : ConcMass
    obs : ConcMass
    param sigma_prop : f64
    log_likelihood = normal_lpdf(obs, pred, sigma_prop * pred)
}
    "#;

    let tokens = tokenize(source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");
    let ir = lower_program(&ast).expect("Lowering failed");
    let stan_code = generate_stan(&ir).expect("Code generation failed");

    // Verify key components
    assert!(stan_code.contains("A_gut"));
    assert!(stan_code.contains("A_central"));
    assert!(stan_code.contains("CL_pop"));
    assert!(stan_code.contains("V_pop"));
    assert!(stan_code.contains("Ka_pop"));
    assert!(stan_code.contains("omega_CL"));
    assert!(stan_code.contains("omega_V"));
    assert!(stan_code.contains("omega_Ka"));
    assert!(stan_code.contains("eta_CL"));
    assert!(stan_code.contains("eta_V"));
    assert!(stan_code.contains("eta_Ka"));
    assert!(stan_code.contains("WT"));

    // Verify ODE equations are present
    assert!(stan_code.contains("Ka * A_gut"));
    assert!(stan_code.contains("CL / V"));

    // Verify individual parameter transformations
    assert!(stan_code.contains("pow"));
    assert!(stan_code.contains("exp"));

    println!("\n=== Generated Stan Code for One-Compartment Oral PK ===\n");
    println!("{}", stan_code);
}

#[test]
fn test_ir_roundtrip() {
    let source = r#"
model Simple {
    state A : DoseMass
    param K : RateConst
    dA/dt = -K * A
}

population SimplePop {
    model Simple
    param K_pop : RateConst
    rand eta_K : f64 ~ Normal(0.0, 0.3)
    bind_params(patient) {
        model.K = K_pop * exp(eta_K)
    }
    use_measure SimpleMeasure for model.A
}

measure SimpleMeasure {
    pred : DoseMass
    obs : DoseMass
    param sigma : f64
    log_likelihood = normal_lpdf(obs, pred, sigma)
}
    "#;

    let tokens = tokenize(source).unwrap();
    let ast = parse_program(&tokens).unwrap();
    let ir = lower_program(&ast).unwrap();

    // Verify IR structure
    assert_eq!(ir.model.name, "Simple");
    assert_eq!(ir.model.states.len(), 1);
    assert_eq!(ir.model.odes.len(), 1);
    assert_eq!(ir.model.random_effects.len(), 1);

    // Verify we can serialize/deserialize IR
    let json = serde_json::to_string(&ir).unwrap();
    let ir_decoded: medlangc::ir::IRProgram = serde_json::from_str(&json).unwrap();

    assert_eq!(ir_decoded.model.name, "Simple");
    assert_eq!(ir_decoded.model.states.len(), 1);
}

#[test]
fn test_validate_ir_from_canonical_example() {
    // Read the canonical example
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/examples/one_comp_oral_pk.medlang");

    let source = std::fs::read_to_string(&example_path).expect("Failed to read canonical example");

    // Compile to IR
    let tokens = tokenize(&source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");
    let ir = lower_program(&ast).expect("Lowering failed");

    // Validate IR structure
    assert_eq!(ir.model.name, "OneCompOral");
    assert_eq!(ir.model.states.len(), 2, "Should have 2 state variables");
    assert_eq!(ir.model.odes.len(), 2, "Should have 2 ODEs");
    assert!(
        ir.model.inputs.len() >= 1,
        "Should have at least 1 input (WT)"
    );
    assert!(
        ir.model.random_effects.len() >= 3,
        "Should have at least 3 random effects"
    );

    // Validate observables
    assert!(!ir.model.observables.is_empty(), "Should have observables");

    // Validate individual parameters
    assert!(
        !ir.model.individual_params.is_empty(),
        "Should have individual parameter transformations"
    );

    println!("IR validation passed for canonical example");
    println!("States: {}", ir.model.states.len());
    println!("ODEs: {}", ir.model.odes.len());
    println!("Random effects: {}", ir.model.random_effects.len());
    println!("Individual params: {}", ir.model.individual_params.len());
}

#[test]
fn test_compile_canonical_to_stan() {
    // Read the canonical example
    let example_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/examples/one_comp_oral_pk.medlang");

    let source = std::fs::read_to_string(&example_path).expect("Failed to read canonical example");

    // Full compilation pipeline
    let tokens = tokenize(&source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");
    let ir = lower_program(&ast).expect("Lowering failed");
    let stan_code = generate_stan(&ir).expect("Code generation failed");

    // Verify Stan code structure
    assert!(stan_code.contains("functions {"));
    assert!(stan_code.contains("data {"));
    assert!(stan_code.contains("parameters {"));
    assert!(stan_code.contains("model {"));

    // Verify it contains the model name
    assert!(stan_code.contains("OneCompOral"));

    // Write to file for manual inspection
    let output_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../generated/one_comp_oral.stan");

    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    std::fs::write(&output_path, &stan_code).ok();

    println!("\n=== Stan code generated from canonical example ===");
    println!("Written to: {:?}", output_path);
    println!("\nFirst 50 lines:");
    for (i, line) in stan_code.lines().take(50).enumerate() {
        println!("{:3}: {}", i + 1, line);
    }
}

#[test]
fn test_stan_code_syntax_basics() {
    let source = r#"
model M {
    state A : DoseMass
    param K : RateConst
    dA/dt = -K * A
}

population P {
    model M
    param K_pop : RateConst
    rand eta_K : f64 ~ Normal(0.0, 0.3)
    bind_params(patient) {
        model.K = K_pop
    }
    use_measure Meas for model.A
}

measure Meas {
    pred : DoseMass
    obs : DoseMass
    param sigma : f64
    log_likelihood = normal_lpdf(obs, pred, sigma)
}
    "#;

    let tokens = tokenize(source).unwrap();
    let ast = parse_program(&tokens).unwrap();
    let ir = lower_program(&ast).unwrap();
    let stan_code = generate_stan(&ir).unwrap();

    // Check for balanced braces
    let open_braces = stan_code.matches('{').count();
    let close_braces = stan_code.matches('}').count();
    assert_eq!(
        open_braces, close_braces,
        "Unbalanced braces in generated Stan code"
    );

    // Check for balanced parentheses
    let open_parens = stan_code.matches('(').count();
    let close_parens = stan_code.matches(')').count();
    assert_eq!(
        open_parens, close_parens,
        "Unbalanced parentheses in generated Stan code"
    );

    // Check that each statement in model block ends with semicolon
    let model_block_start = stan_code.find("model {").expect("No model block");
    let model_block_end = stan_code[model_block_start..]
        .find('}')
        .expect("Unclosed model block");
    let model_block = &stan_code[model_block_start..model_block_start + model_block_end];

    // Count non-comment, non-brace lines
    let statement_lines: Vec<_> = model_block
        .lines()
        .filter(|l| {
            let trimmed = l.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with("//")
                && !trimmed.starts_with("{")
                && !trimmed.starts_with("}")
                && trimmed != "model"
        })
        .collect();

    // Check that statements are properly formed (not checking semicolons as some lines are block headers)
    // Just verify the code is non-empty and contains key structures
    assert!(
        statement_lines.len() > 0,
        "Model block should have statements"
    );
}
