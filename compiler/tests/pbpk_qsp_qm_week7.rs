/// Week 7 Integration Tests: PBPK-QSP-QM Vertical Pipeline
///
/// Tests the complete multiscale pipeline:
/// QM stub → Kp_tumor → PBPK tumour exposure → QSP tumour dynamics → population inference
use medlangc::*;

#[test]
fn test_pbpk_qsp_simple_compilation() {
    let source = include_str!("../../docs/examples/oncology_pbpk_qsp_simple.medlang");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");

    // Should have 3 declarations: model, population, measure
    assert_eq!(program.declarations.len(), 3);

    // Lower to IR without QM stub
    let ir = lower::lower_program(&program).expect("Lowering failed");

    // Should have 3 states: A_plasma, A_tumor, Tumour
    assert_eq!(
        ir.model.states.len(),
        3,
        "Should have 3 states (PBPK + QSP)"
    );
    assert_eq!(ir.model.states[0].name, "A_plasma");
    assert_eq!(ir.model.states[1].name, "A_tumor");
    assert_eq!(ir.model.states[2].name, "Tumour");

    // Should have 3 ODEs
    assert_eq!(ir.model.odes.len(), 3);

    // Should have 3 observables: C_plasma_obs, C_tumor_obs, TumourVol
    assert_eq!(ir.model.observables.len(), 3);
}

#[test]
fn test_pbpk_qsp_with_qm_stub() {
    let source = include_str!("../../docs/examples/oncology_pbpk_qsp_simple.medlang");
    let stub_json = include_str!("../../data/lig001_egfr_qm.json");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");

    // Load QM stub
    let stub: qm_stub::QuantumStub =
        serde_json::from_str(stub_json).expect("Failed to parse QM stub");

    // Lower with QM stub
    let ir = lower::lower_program_with_qm(&program, Some(&stub)).expect("Lowering failed");

    // Should have 2 external constants: Kd_QM and Kp_tumor_QM
    assert_eq!(ir.externals.len(), 2, "Should have 2 QM constants");

    let kd_ext = ir
        .externals
        .iter()
        .find(|e| e.name == "Kd_QM")
        .expect("Kd_QM not found");
    assert_eq!(kd_ext.value, stub.Kd_M);
    assert_eq!(kd_ext.dimension, Some("ConcMass".to_string()));

    let kp_ext = ir
        .externals
        .iter()
        .find(|e| e.name == "Kp_tumor_QM")
        .expect("Kp_tumor_QM not found");
    let expected_kp = stub.kp_tumor_from_dg().expect("ΔG_part missing");
    assert!((kp_ext.value - expected_kp).abs() < 1e-6);
    assert_eq!(kp_ext.dimension, None); // Kp is dimensionless
}

#[test]
fn test_pbpk_intermediates_in_ode() {
    let source = include_str!("../../docs/examples/oncology_pbpk_qsp_simple.medlang");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");
    let ir = lower::lower_program(&program).expect("Lowering failed");

    // Should have 4 intermediates: C_plasma, C_tumor, C_tumor_vein, E_drug
    assert_eq!(
        ir.model.intermediates.len(),
        4,
        "Should have 4 intermediates"
    );

    let names: Vec<_> = ir
        .model
        .intermediates
        .iter()
        .map(|i| i.name.as_str())
        .collect();
    assert!(names.contains(&"C_plasma"));
    assert!(names.contains(&"C_tumor"));
    assert!(names.contains(&"C_tumor_vein"));
    assert!(names.contains(&"E_drug"));
}

#[test]
fn test_pbpk_qsp_stan_codegen() {
    let source = include_str!("../../docs/examples/oncology_pbpk_qsp_simple.medlang");
    let stub_json = include_str!("../../data/lig001_egfr_qm.json");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");
    let stub: qm_stub::QuantumStub =
        serde_json::from_str(stub_json).expect("Failed to parse QM stub");
    let ir = lower::lower_program_with_qm(&program, Some(&stub)).expect("Lowering failed");

    // Generate Stan code
    let stan_code = codegen::stan::generate_stan(&ir).expect("Stan codegen failed");

    // Verify key components are present
    assert!(
        stan_code.contains("vector ode_system"),
        "Missing ODE function"
    );
    assert!(
        stan_code.contains("real A_plasma = y[1]"),
        "Missing A_plasma state"
    );
    assert!(
        stan_code.contains("real A_tumor = y[2]"),
        "Missing A_tumor state"
    );
    assert!(
        stan_code.contains("real Tumour = y[3]"),
        "Missing Tumour state"
    );

    // Verify intermediates are emitted
    assert!(
        stan_code.contains("real C_plasma ="),
        "Missing C_plasma intermediate"
    );
    assert!(
        stan_code.contains("real C_tumor ="),
        "Missing C_tumor intermediate"
    );
    assert!(
        stan_code.contains("real C_tumor_vein ="),
        "Missing C_tumor_vein intermediate"
    );
    assert!(
        stan_code.contains("real E_drug ="),
        "Missing E_drug intermediate"
    );

    // Verify QM constants in data block
    assert!(
        stan_code.contains("real<lower=0> Kd_QM"),
        "Missing Kd_QM in data block"
    );
    assert!(
        stan_code.contains("real<lower=0> Kp_tumor_QM"),
        "Missing Kp_tumor_QM in data block"
    );

    // Verify PBPK equations
    assert!(stan_code.contains("dydt[1] ="), "Missing dA_plasma/dt");
    assert!(stan_code.contains("dydt[2] ="), "Missing dA_tumor/dt");
    assert!(stan_code.contains("dydt[3] ="), "Missing dTumour/dt");

    // Verify QSP uses C_tumor (not Kp*C_plasma)
    // E_drug is defined using C_tumor: E_drug = (Emax * C_tumor) / (EC50 + C_tumor)
    assert!(
        stan_code.contains("Emax * C_tumor"),
        "E_drug definition should use C_tumor"
    );
    assert!(
        stan_code.contains("E_drug * Tumour"),
        "QSP ODE should use E_drug"
    );
}

#[test]
fn test_allometric_scaling_parameters() {
    let source = include_str!("../../docs/examples/oncology_pbpk_qsp_simple.medlang");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");
    let ir = lower::lower_program(&program).expect("Lowering failed");

    // Should have WT as input (covariate)
    assert!(
        ir.model.inputs.iter().any(|i| i.name == "WT"),
        "Should have WT covariate"
    );

    // Check that individual params use pow(w, ...) for allometric scaling
    let has_allometric = ir
        .model
        .individual_params
        .iter()
        .any(|p| format!("{:?}", p.expr).contains("pow"));
    assert!(has_allometric, "Should have allometric scaling with pow()");
}

#[test]
fn test_qm_informed_parameters() {
    let source = include_str!("../../docs/examples/oncology_pbpk_qsp_simple.medlang");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");
    let ir = lower::lower_program(&program).expect("Lowering failed");

    // Find bind_params expressions
    let kp_param = ir
        .model
        .individual_params
        .iter()
        .find(|p| p.param_name == "Kp_tumor")
        .expect("Kp_tumor binding not found");

    // Should reference Kp_tumor_QM
    assert!(
        format!("{:?}", kp_param.expr).contains("Kp_tumor_QM"),
        "Kp_tumor should use Kp_tumor_QM from QM stub"
    );

    let ec50_param = ir
        .model
        .individual_params
        .iter()
        .find(|p| p.param_name == "EC50")
        .expect("EC50 binding not found");

    // Should reference Kd_QM
    assert!(
        format!("{:?}", ec50_param.expr).contains("Kd_QM"),
        "EC50 should use Kd_QM from QM stub"
    );
}

#[test]
fn test_composite_model_flattening() {
    // Test the basic composite model (without inputs/connections yet)
    let source = include_str!("../../docs/examples/test_composite_minimal.medlang");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");

    // Should have 3 models: PBPK_Simple, QSP_Simple, Composite
    let model_count = program
        .declarations
        .iter()
        .filter(|d| matches!(d, ast::Declaration::Model(_)))
        .count();
    assert_eq!(model_count, 3, "Should have 3 models");

    // Lower the program
    let ir = lower::lower_program(&program).expect("Lowering failed");

    // The composite model should be flattened to have states from both submodels
    assert_eq!(ir.model.states.len(), 2, "Should have 2 flattened states");
}

#[test]
fn test_pbpk_qsp_parameter_count() {
    let source = include_str!("../../docs/examples/oncology_pbpk_qsp_simple.medlang");

    let tokens = lexer::tokenize(source).expect("Tokenization failed");
    let program = parser::parse_program(&tokens).expect("Parsing failed");
    let ir = lower::lower_program(&program).expect("Lowering failed");

    // Count PBPK parameters: CL, Q_tum, V_plasma, V_tumor, Kp_tumor
    // Count QSP parameters: k_grow, T_max, Emax, EC50
    // Plus population parameters and omegas
    assert!(
        ir.model.params.len() >= 9,
        "Should have at least 9 structural parameters"
    );

    // Should have random effects for IIV
    assert!(
        ir.model.random_effects.len() >= 6,
        "Should have at least 6 random effects"
    );
}
