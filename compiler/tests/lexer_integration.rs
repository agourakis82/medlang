//! Integration tests for lexer on realistic MedLang code snippets.

use medlangc::lexer::{tokenize, Token};

#[test]
fn test_tokenize_simple_model() {
    let source = r#"
model OneCompOral {
    state A_gut : DoseMass
    state A_central : DoseMass

    param Ka : RateConst
    param CL : Clearance
    param V  : Volume

    dA_gut/dt = -Ka * A_gut
    dA_central/dt = Ka * A_gut - (CL / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}
"#;

    let result = tokenize(source);
    assert!(result.is_ok(), "Tokenization should succeed");

    let tokens = result.unwrap();
    println!("Tokenized {} tokens", tokens.len());

    // Should start with "model OneCompOral {"
    assert!(matches!(tokens[0].0, Token::Model));
    assert!(matches!(tokens[1].0, Token::Ident(ref s) if s == "OneCompOral"));
    assert!(matches!(tokens[2].0, Token::LBrace));

    // Should have state declarations
    let state_count = tokens
        .iter()
        .filter(|(t, _, _)| matches!(t, Token::State))
        .count();
    assert_eq!(state_count, 2, "Should have 2 state declarations");

    // Should have param declarations
    let param_count = tokens
        .iter()
        .filter(|(t, _, _)| matches!(t, Token::Param))
        .count();
    assert_eq!(param_count, 3, "Should have 3 param declarations");

    // Should have ODE derivatives
    let ode_count = tokens
        .iter()
        .filter(|(t, _, _)| matches!(t, Token::ODEDeriv(_)))
        .count();
    assert_eq!(ode_count, 2, "Should have 2 ODE equations");

    // Should have observable
    let obs_count = tokens
        .iter()
        .filter(|(t, _, _)| matches!(t, Token::Obs))
        .count();
    assert_eq!(obs_count, 1, "Should have 1 observable");
}

#[test]
fn test_tokenize_population_snippet() {
    let source = r#"
population OneCompOralPop {
    model OneCompOral

    param CL_pop : Clearance
    param omega_CL : f64

    input WT : Quantity<kg, f64>

    rand eta_CL : f64 ~ Normal(0.0, omega_CL)

    bind_params(patient) {
        let w = patient.WT / 70.0_kg
        model.CL = CL_pop * pow(w, 0.75) * exp(eta_CL)
    }
}
"#;

    let result = tokenize(source);
    assert!(result.is_ok(), "Tokenization should succeed");

    let tokens = result.unwrap();

    // Check for key tokens
    assert!(tokens
        .iter()
        .any(|(t, _, _)| matches!(t, Token::Population)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Input)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Rand)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Tilde))); // ~
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Normal)));
    assert!(tokens
        .iter()
        .any(|(t, _, _)| matches!(t, Token::BindParams)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Let)));

    // Check unit literal (70.0_kg in the bind_params block)
    // Note: Quantity<kg, f64> has "kg" as an identifier, not a unit literal
    let unit_lit_count = tokens
        .iter()
        .filter(|(t, _, _)| matches!(t, Token::UnitLiteral(ref ul) if ul.unit == "kg"))
        .count();
    assert_eq!(unit_lit_count, 1, "Should have 1 kg unit literal (70.0_kg)");
}

#[test]
fn test_tokenize_measure() {
    let source = r#"
measure ConcPropError {
    pred : ConcMass
    obs  : ConcMass
    param sigma_prop : f64

    log_likelihood = Normal_logpdf(
        x  = (obs / pred) - 1.0,
        mu = 0.0,
        sd = sigma_prop
    )
}
"#;

    let result = tokenize(source);
    assert!(result.is_ok(), "Tokenization should succeed");

    let tokens = result.unwrap();

    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Measure)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Pred)));
    assert!(tokens
        .iter()
        .any(|(t, _, _)| matches!(t, Token::LogLikelihood)));

    // Check for function call identifier
    assert!(tokens
        .iter()
        .any(|(t, _, _)| matches!(t, Token::Ident(ref s) if s == "Normal_logpdf")));
}

#[test]
fn test_tokenize_timeline() {
    let source = r#"
timeline OneCompOralTimeline {
    at 0.0_h:
        dose { amount = 100.0_mg; to = OneCompOral.A_gut }

    at 1.0_h:  observe OneCompOral.C_plasma
    at 2.0_h:  observe OneCompOral.C_plasma
}
"#;

    let result = tokenize(source);
    assert!(result.is_ok(), "Tokenization should succeed");

    let tokens = result.unwrap();

    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Timeline)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::At)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Dose)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Amount)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::To)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Observe)));

    // Count "at" keywords
    let at_count = tokens
        .iter()
        .filter(|(t, _, _)| matches!(t, Token::At))
        .count();
    assert_eq!(at_count, 3, "Should have 3 'at' keywords");
}

#[test]
fn test_tokenize_cohort() {
    let source = r#"
cohort OneCompCohort {
    population OneCompOralPop
    timeline   OneCompOralTimeline
    data_file  "data/onecomp_synth.csv"
}
"#;

    let result = tokenize(source);
    assert!(result.is_ok(), "Tokenization should succeed");

    let tokens = result.unwrap();

    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::Cohort)));
    assert!(tokens.iter().any(|(t, _, _)| matches!(t, Token::DataFile)));

    // Check for string literal
    assert!(tokens
        .iter()
        .any(|(t, _, _)| matches!(t, Token::String(ref s) if s == "data/onecomp_synth.csv")));
}

#[test]
fn test_tokenize_error_recovery() {
    let source = "model Test { @ }"; // @ is not a valid token

    let result = tokenize(source);
    assert!(result.is_err(), "Should fail on invalid character");

    let err = result.unwrap_err();
    assert!(
        err.snippet.contains("@"),
        "Error should reference the invalid character"
    );
}

#[test]
fn test_preserve_span_info() {
    let source = "model Test";
    let tokens = tokenize(source).unwrap();

    assert_eq!(tokens.len(), 2);

    // First token: "model" at position 0-5
    assert_eq!(tokens[0].1, 0); // start
    assert_eq!(tokens[0].2, 5); // end

    // Second token: "Test" at position 6-10
    assert_eq!(tokens[1].1, 6);
    assert_eq!(tokens[1].2, 10);
}
