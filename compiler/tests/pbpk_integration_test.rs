use medlangc::codegen::stan::generate_stan;
/// Integration tests for PBPK functionality
///
/// Tests multi-compartment physiologically-based pharmacokinetic models
/// with let binding intermediates and QM integration.
use medlangc::lexer::tokenize;
use medlangc::lower::lower_program_with_qm;
use medlangc::parser::parse_program;
use medlangc::qm_stub::QuantumStub;
use std::fs;

#[test]
fn test_pbpk_2comp_compilation() {
    let source = fs::read_to_string("../docs/examples/pbpk_2comp_simple.medlang")
        .expect("Failed to read 2-comp PBPK example");

    let tokens = tokenize(&source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");
    let ir = lower_program_with_qm(&ast, None).expect("Lowering failed");

    // Verify model structure
    assert_eq!(ir.model.states.len(), 2, "Should have 2 PBPK states");
    assert_eq!(ir.model.odes.len(), 2, "Should have 2 PBPK ODEs");
    assert_eq!(
        ir.model.intermediates.len(),
        3,
        "Should have 3 intermediate values"
    );

    // Verify intermediates
    let intermediate_names: Vec<&str> = ir
        .model
        .intermediates
        .iter()
        .map(|i| i.name.as_str())
        .collect();
    assert!(intermediate_names.contains(&"C_plasma"));
    assert!(intermediate_names.contains(&"C_tissue_unbound"));
    assert!(intermediate_names.contains(&"C_tissue_vein"));
}

#[test]
fn test_pbpk_intermediates_in_stan() {
    let source = fs::read_to_string("../docs/examples/pbpk_2comp_simple.medlang")
        .expect("Failed to read example");

    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();
    let ir = lower_program_with_qm(&ast, None).unwrap();

    let stan_code = generate_stan(&ir).expect("Stan codegen failed");

    // Verify intermediates are declared in ODE function
    assert!(stan_code.contains("// Intermediate values"));
    assert!(stan_code.contains("real C_plasma = "));
    assert!(stan_code.contains("real C_tissue_vein = "));

    // Verify intermediates are used in ODEs
    assert!(stan_code.contains("(C_tissue_vein - C_plasma)"));
    assert!(stan_code.contains("(C_plasma - C_tissue_vein)"));
}

#[test]
fn test_pbpk_5comp_compilation() {
    let source = fs::read_to_string("../docs/examples/pbpk_5comp_qsp_qm.medlang")
        .expect("Failed to read 5-comp PBPK example");

    let tokens = tokenize(&source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");

    // Lower without QM stub first
    let ir_no_qm = lower_program_with_qm(&ast, None).expect("Lowering failed");

    assert_eq!(
        ir_no_qm.model.states.len(),
        6,
        "Should have 6 states (5 PBPK + 1 QSP)"
    );
    assert_eq!(ir_no_qm.model.odes.len(), 6, "Should have 6 ODEs");
    assert_eq!(
        ir_no_qm.model.intermediates.len(),
        9,
        "Should have 9 intermediates (5 tissue + 4 venous)"
    );
    assert_eq!(
        ir_no_qm.externals.len(),
        0,
        "Should have no externals without QM stub"
    );
}

#[test]
fn test_pbpk_5comp_with_qm_stub() {
    let source = fs::read_to_string("../docs/examples/pbpk_5comp_qsp_qm.medlang")
        .expect("Failed to read example");

    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();

    // Load QM stub
    let stub = QuantumStub::load("../data/lig001_egfr_qm.json").expect("Failed to load QM stub");

    // Lower with QM stub
    let ir = lower_program_with_qm(&ast, Some(&stub)).unwrap();

    // Verify QM externals
    assert_eq!(ir.externals.len(), 2, "Should have 2 QM externals");

    let kd_ext = ir
        .externals
        .iter()
        .find(|e| e.name == "Kd_QM")
        .expect("Kd_QM should be present");
    assert!((kd_ext.value - 2.5e-9).abs() < 1e-15);

    let kp_ext = ir
        .externals
        .iter()
        .find(|e| e.name == "Kp_tumor_QM")
        .expect("Kp_tumor_QM should be present");
    assert!((kp_ext.value - 3.664).abs() < 0.001);
}

#[test]
fn test_pbpk_qsp_stan_codegen() {
    let source = fs::read_to_string("../docs/examples/pbpk_5comp_qsp_qm.medlang")
        .expect("Failed to read example");

    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();
    let stub = QuantumStub::load("../data/lig001_egfr_qm.json").unwrap();
    let ir = lower_program_with_qm(&ast, Some(&stub)).unwrap();

    let stan_code = generate_stan(&ir).expect("Stan codegen failed");

    // Verify PBPK intermediates
    assert!(stan_code.contains("real C_plasma = "));
    assert!(stan_code.contains("real C_tumor_tissue = "));
    assert!(stan_code.contains("real C_liver_vein = "));
    assert!(stan_code.contains("real C_kidney_vein = "));
    assert!(stan_code.contains("real C_rest_vein = "));

    // Verify QM constants in data block
    assert!(stan_code.contains("real<lower=0> Kd_QM;"));
    assert!(stan_code.contains("real<lower=0> Kp_tumor_QM;"));

    // Verify QSP dynamics use C_tumor_tissue (not Kp*C_plasma)
    assert!(
        stan_code.contains("C_tumor_tissue") && stan_code.contains("TumourSize"),
        "QSP should use C_tumor_tissue"
    );
}

#[test]
fn test_pbpk_allometric_scaling() {
    let source = fs::read_to_string("../docs/examples/pbpk_5comp_qsp_qm.medlang")
        .expect("Failed to read example");

    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();
    let ir = lower_program_with_qm(&ast, None).unwrap();

    // Verify allometric scaling is present in individual params
    assert!(
        !ir.model.individual_params.is_empty(),
        "Should have individual parameter mappings"
    );

    // Generate Stan and verify scaling code exists
    let stan_code = generate_stan(&ir).unwrap();

    // Should have weight covariate
    assert!(
        stan_code.contains("vector[N] WT"),
        "Should have WT covariate"
    );
}

#[test]
fn test_pbpk_tissue_concentrations() {
    let source = fs::read_to_string("../docs/examples/pbpk_5comp_qsp_qm.medlang")
        .expect("Failed to read example");

    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();
    let ir = lower_program_with_qm(&ast, None).unwrap();

    // Verify we have tissue concentration intermediates
    let intermediate_names: Vec<&str> = ir
        .model
        .intermediates
        .iter()
        .map(|i| i.name.as_str())
        .collect();

    assert!(intermediate_names.contains(&"C_plasma"));
    assert!(intermediate_names.contains(&"C_tumor_tissue"));
    assert!(intermediate_names.contains(&"C_liver_tissue"));
    assert!(intermediate_names.contains(&"C_kidney_tissue"));
    assert!(intermediate_names.contains(&"C_rest_tissue"));

    // Verify venous concentrations
    assert!(intermediate_names.contains(&"C_tumor_vein"));
    assert!(intermediate_names.contains(&"C_liver_vein"));
    assert!(intermediate_names.contains(&"C_kidney_vein"));
    assert!(intermediate_names.contains(&"C_rest_vein"));
}

#[test]
fn test_pbpk_observables() {
    let source = fs::read_to_string("../docs/examples/pbpk_5comp_qsp_qm.medlang")
        .expect("Failed to read example");

    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();
    let ir = lower_program_with_qm(&ast, None).unwrap();

    // Should have 3 observables: C_plasma, C_tumor, TumourVol
    assert_eq!(ir.model.observables.len(), 3);

    let obs_names: Vec<&str> = ir
        .model
        .observables
        .iter()
        .map(|o| o.name.as_str())
        .collect();

    assert!(obs_names.contains(&"C_plasma_obs"));
    assert!(obs_names.contains(&"C_tumor_obs"));
    assert!(obs_names.contains(&"TumourVol"));
}
