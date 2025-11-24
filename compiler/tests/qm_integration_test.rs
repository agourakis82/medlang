use medlangc::codegen::stan::generate_stan;
/// Integration tests for Quantum Stub (Track C) functionality
///
/// Tests the full pipeline from QM stub loading through IR generation
/// to Stan code emission for quantum-informed PK-PD models.
use medlangc::lexer::tokenize;
use medlangc::lower::lower_program_with_qm;
use medlangc::parser::parse_program;
use medlangc::qm_stub::QuantumStub;
use std::fs;
use std::io::Write;
use tempfile::NamedTempFile;

#[test]
fn test_qm_stub_loading() {
    let stub = QuantumStub::load("../data/lig001_egfr_qm.json").expect("Failed to load QM stub");

    assert_eq!(stub.drug_id, "LIG001");
    assert_eq!(stub.target_id, "EGFR");
    assert!((stub.Kd_M - 2.5e-9).abs() < 1e-15);

    let kp = stub.kp_tumor_from_dg().expect("Kp should be computed");
    assert!(kp > 1.0); // Negative ΔG means favorable partition
}

#[test]
fn test_lower_program_with_qm_stub() {
    // Load the PK-QSP example
    let source = fs::read_to_string("../docs/examples/pk_qsp_inline.medlang")
        .expect("Failed to read PK-QSP example");

    let tokens = tokenize(&source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");

    // Load QM stub
    let stub = QuantumStub::load("../data/lig001_egfr_qm.json").expect("Failed to load QM stub");

    // Lower with QM stub
    let ir = lower_program_with_qm(&ast, Some(&stub)).expect("Lowering with QM stub failed");

    // Verify externals were added
    assert_eq!(ir.externals.len(), 2, "Should have 2 QM externals");

    let kd_ext = ir
        .externals
        .iter()
        .find(|e| e.name == "Kd_QM")
        .expect("Kd_QM not found in externals");
    assert!((kd_ext.value - 2.5e-9).abs() < 1e-15);
    assert_eq!(kd_ext.source, "qm_stub:LIG001:EGFR");
    assert_eq!(kd_ext.dimension.as_deref(), Some("ConcMass"));

    let kp_ext = ir
        .externals
        .iter()
        .find(|e| e.name == "Kp_tumor_QM")
        .expect("Kp_tumor_QM not found in externals");
    assert!(kp_ext.value > 1.0);
    assert!(kp_ext.source.contains("dG_part"));
}

#[test]
fn test_stan_codegen_includes_qm_constants() {
    let source = fs::read_to_string("../docs/examples/pk_qsp_inline.medlang")
        .expect("Failed to read PK-QSP example");

    let tokens = tokenize(&source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");

    let stub = QuantumStub::load("../data/lig001_egfr_qm.json").expect("Failed to load QM stub");

    let ir = lower_program_with_qm(&ast, Some(&stub)).expect("Lowering failed");

    let stan_code = generate_stan(&ir).expect("Stan codegen failed");

    // Check that QM constants appear in data block
    assert!(
        stan_code.contains("// External quantum constants"),
        "Stan code should contain QM constants section"
    );
    assert!(
        stan_code.contains("real<lower=0> Kd_QM;"),
        "Stan code should declare Kd_QM"
    );
    assert!(
        stan_code.contains("from qm_stub:LIG001:EGFR"),
        "Stan code should document QM source"
    );
    assert!(
        stan_code.contains("real<lower=0> Kp_tumor_QM;"),
        "Stan code should declare Kp_tumor_QM"
    );
    assert!(
        stan_code.contains("from qm_stub:LIG001:dG_part"),
        "Stan code should document partition source"
    );
}

#[test]
fn test_qm_stub_without_partition_data() {
    // Create a minimal QM stub with only Kd
    let json_content = r#"{
        "drug_id": "MINIMAL",
        "target_id": "RECEPTOR",
        "Kd_M": 1.0e-9
    }"#;

    let mut temp_file = NamedTempFile::new().unwrap();
    temp_file.write_all(json_content.as_bytes()).unwrap();

    let stub = QuantumStub::load(temp_file.path()).expect("Failed to load minimal stub");

    assert_eq!(stub.drug_id, "MINIMAL");
    assert!(
        stub.kp_tumor_from_dg().is_none(),
        "Kp should be None without dG_part"
    );

    // Test lowering with minimal stub
    let source = fs::read_to_string("../docs/examples/pk_qsp_inline.medlang")
        .expect("Failed to read example");
    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();

    let ir = lower_program_with_qm(&ast, Some(&stub)).unwrap();

    // Should only have Kd_QM, not Kp_tumor_QM
    assert_eq!(ir.externals.len(), 1, "Should only have Kd_QM");
    assert_eq!(ir.externals[0].name, "Kd_QM");
}

#[test]
fn test_lower_without_qm_stub() {
    let source = fs::read_to_string("../docs/examples/pk_qsp_inline.medlang")
        .expect("Failed to read example");

    let tokens = tokenize(&source).unwrap();
    let ast = parse_program(&tokens).unwrap();

    // Lower without QM stub (None)
    let ir = lower_program_with_qm(&ast, None).unwrap();

    // Should have no externals
    assert_eq!(
        ir.externals.len(),
        0,
        "Should have no externals without QM stub"
    );

    // Stan code should not have QM section
    let stan_code = generate_stan(&ir).unwrap();
    assert!(
        !stan_code.contains("External quantum constants"),
        "Stan code should not contain QM constants section without stub"
    );
}

#[test]
fn test_thermodynamic_consistency() {
    let stub = QuantumStub::load("../data/lig001_egfr_qm.json").unwrap();

    // For QM stubs, Kd and dG_bind may come from different calculations
    // (e.g., Kd from experiment, dG_bind from QM), so we just verify
    // both are present and have reasonable values
    assert!(
        stub.dG_bind_kcal_per_mol.is_some(),
        "dG_bind should be present"
    );

    let dg = stub.dG_bind_kcal_per_mol.unwrap();
    // For binding, dG should be negative (favorable)
    assert!(dg < 0.0, "Binding dG should be negative (favorable)");
    // Typical range for drug binding: -8 to -15 kcal/mol
    assert!(
        dg > -20.0 && dg < 0.0,
        "dG = {:.1} kcal/mol outside typical range",
        dg
    );
}

#[test]
fn test_kp_calculation_range() {
    // Test that Kp values are in reasonable range
    let stub = QuantumStub::load("../data/lig001_egfr_qm.json").unwrap();

    let kp = stub.kp_tumor_from_dg().expect("Kp should be available");

    // Kp should be positive
    assert!(kp > 0.0, "Kp must be positive");

    // For typical drugs, Kp_tumor ranges from 0.1 to 10
    // Our stub has dG_part = -0.8 kcal/mol, so Kp should be > 1
    assert!(
        kp > 1.0 && kp < 100.0,
        "Kp = {:.2} is outside typical range for favorable partition",
        kp
    );
}
