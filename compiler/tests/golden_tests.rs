use medlangc::codegen::stan::generate_stan;
/// Golden file tests for Stan code generation
///
/// These tests verify that the compiler produces consistent, correct Stan code
/// by comparing against reference "golden" files. This ensures:
/// 1. Regression prevention: Changes don't inadvertently alter output
/// 2. Output quality: Generated code matches expected structure
/// 3. Documentation: Golden files serve as examples of expected output
use medlangc::lexer::tokenize;
use medlangc::lower::lower_program;
use medlangc::parser::parse_program;
use std::fs;
use std::path::Path;

const UPDATE_GOLDEN: bool = false; // Set to true to regenerate golden files

/// Helper to load or create golden file
fn check_golden(test_name: &str, generated: &str) {
    let golden_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden");

    // Ensure golden directory exists
    if !golden_dir.exists() {
        fs::create_dir_all(&golden_dir).expect("Failed to create golden directory");
    }

    let golden_path = golden_dir.join(format!("{}.stan", test_name));

    if UPDATE_GOLDEN || !golden_path.exists() {
        // Write new golden file
        fs::write(&golden_path, generated)
            .expect(&format!("Failed to write golden file: {:?}", golden_path));
        println!("Updated golden file: {:?}", golden_path);
    } else {
        // Compare against existing golden file
        let golden = fs::read_to_string(&golden_path)
            .expect(&format!("Failed to read golden file: {:?}", golden_path));

        if generated.trim() != golden.trim() {
            // Write actual output for comparison
            let actual_path = golden_dir.join(format!("{}.actual.stan", test_name));
            fs::write(&actual_path, generated).expect("Failed to write actual output");

            panic!(
                "\nGolden file mismatch for test '{}'\n\
                 Expected: {:?}\n\
                 Actual:   {:?}\n\n\
                 To update golden files, set UPDATE_GOLDEN = true and re-run tests.\n\
                 To see diff:\n  diff {:?} {:?}\n",
                test_name, golden_path, actual_path, golden_path, actual_path
            );
        }
    }
}

/// Helper to compile MedLang source to Stan
fn compile_to_stan(source: &str) -> String {
    let tokens = tokenize(source).expect("Tokenization failed");
    let ast = parse_program(&tokens).expect("Parsing failed");
    let ir = lower_program(&ast).expect("Lowering failed");
    generate_stan(&ir).expect("Code generation failed")
}

#[test]
fn golden_canonical_example() {
    // The authoritative V0 example - 185 lines of complete MedLang
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");

    let generated = compile_to_stan(&source);
    check_golden("canonical_example", &generated);

    // Verify key structure
    assert!(
        generated.contains("OneCompOral"),
        "Should contain model name"
    );
    assert!(
        generated.contains("functions {"),
        "Should have functions block"
    );
    assert!(generated.contains("ode_system"), "Should have ODE system");
    assert!(
        generated.contains("A_gut"),
        "Should reference gut compartment"
    );
    assert!(
        generated.contains("A_central"),
        "Should reference central compartment"
    );
}

#[test]
fn test_stan_syntax_validity() {
    // Compile canonical example and verify Stan syntax structure
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");
    let stan_code = compile_to_stan(&source);

    // Check for required Stan blocks
    assert!(stan_code.contains("functions {"), "Missing functions block");
    assert!(stan_code.contains("data {"), "Missing data block");
    assert!(
        stan_code.contains("parameters {"),
        "Missing parameters block"
    );
    assert!(
        stan_code.contains("transformed parameters {"),
        "Missing transformed parameters block"
    );
    assert!(stan_code.contains("model {"), "Missing model block");

    // Check for ODE system
    assert!(
        stan_code.contains("vector ode_system("),
        "Missing ODE system function"
    );

    // Check for proper Stan syntax
    assert!(
        stan_code.contains("int<lower=1>"),
        "Missing proper integer constraints"
    );
    assert!(
        stan_code.contains("real<lower=0>"),
        "Missing proper real constraints"
    );

    // Check for population parameters
    assert!(stan_code.contains("CL_pop"), "Missing CL_pop parameter");
    assert!(stan_code.contains("V_pop"), "Missing V_pop parameter");
    assert!(stan_code.contains("Ka_pop"), "Missing Ka_pop parameter");

    // Check for random effects
    assert!(stan_code.contains("omega_CL"), "Missing omega_CL");
    assert!(stan_code.contains("omega_V"), "Missing omega_V");
    assert!(stan_code.contains("omega_Ka"), "Missing omega_Ka");
    assert!(stan_code.contains("eta_CL"), "Missing eta_CL");
    assert!(stan_code.contains("eta_V"), "Missing eta_V");
    assert!(stan_code.contains("eta_Ka"), "Missing eta_Ka");

    // Check for covariate
    assert!(stan_code.contains("WT"), "Missing weight covariate");

    // Check that all blocks are properly closed
    let open_braces = stan_code.matches('{').count();
    let close_braces = stan_code.matches('}').count();
    assert_eq!(
        open_braces, close_braces,
        "Unbalanced braces in generated Stan code"
    );

    // Verify line count is reasonable (canonical example generates ~107 lines)
    let line_count = stan_code.lines().count();
    assert!(
        line_count >= 100 && line_count <= 150,
        "Expected ~107 lines, got {}",
        line_count
    );
}

#[test]
fn test_ode_system_structure() {
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");
    let stan_code = compile_to_stan(&source);

    // Extract ODE system function
    assert!(
        stan_code.contains("vector ode_system("),
        "Missing ODE function"
    );

    // Check ODE parameters
    assert!(stan_code.contains("real Ka"), "ODE missing Ka parameter");
    assert!(stan_code.contains("real CL"), "ODE missing CL parameter");
    assert!(stan_code.contains("real V"), "ODE missing V parameter");

    // Check state unpacking
    assert!(
        stan_code.contains("real A_gut = y[1]"),
        "Missing A_gut unpacking"
    );
    assert!(
        stan_code.contains("real A_central = y[2]"),
        "Missing A_central unpacking"
    );

    // Check derivatives vector
    assert!(
        stan_code.contains("vector[2] dydt"),
        "Missing dydt declaration"
    );
    assert!(
        stan_code.contains("dydt[1]"),
        "Missing gut derivative assignment"
    );
    assert!(
        stan_code.contains("dydt[2]"),
        "Missing central derivative assignment"
    );

    // Check return
    assert!(stan_code.contains("return dydt"), "Missing dydt return");
}

#[test]
fn test_data_block_structure() {
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");
    let stan_code = compile_to_stan(&source);

    // Check data block declarations
    assert!(
        stan_code.contains("int<lower=1> N"),
        "Missing N declaration"
    );
    assert!(
        stan_code.contains("int<lower=1> n_obs"),
        "Missing n_obs declaration"
    );
    assert!(
        stan_code.contains("vector[n_obs] time"),
        "Missing time vector"
    );
    assert!(
        stan_code.contains("vector[n_obs] observation"),
        "Missing observation vector"
    );
    assert!(stan_code.contains("vector[N] WT"), "Missing WT covariate");
    assert!(
        stan_code.contains("real dose_amount"),
        "Missing dose_amount"
    );
    assert!(stan_code.contains("real dose_time"), "Missing dose_time");
}

#[test]
fn test_parameters_block_structure() {
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");
    let stan_code = compile_to_stan(&source);

    // Population parameters with constraints
    assert!(
        stan_code.contains("real<lower=0> CL_pop"),
        "Missing CL_pop with constraint"
    );
    assert!(
        stan_code.contains("real<lower=0> V_pop"),
        "Missing V_pop with constraint"
    );
    assert!(
        stan_code.contains("real<lower=0> Ka_pop"),
        "Missing Ka_pop with constraint"
    );

    // Variability parameters
    assert!(
        stan_code.contains("real<lower=0> omega_CL"),
        "Missing omega_CL"
    );
    assert!(
        stan_code.contains("real<lower=0> omega_V"),
        "Missing omega_V"
    );
    assert!(
        stan_code.contains("real<lower=0> omega_Ka"),
        "Missing omega_Ka"
    );

    // Random effects vectors
    assert!(
        stan_code.contains("vector[N] eta_CL"),
        "Missing eta_CL vector"
    );
    assert!(
        stan_code.contains("vector[N] eta_V"),
        "Missing eta_V vector"
    );
    assert!(
        stan_code.contains("vector[N] eta_Ka"),
        "Missing eta_Ka vector"
    );

    // Error parameter
    assert!(
        stan_code.contains("real<lower=0> sigma_prop"),
        "Missing sigma_prop"
    );
}

#[test]
fn test_transformed_parameters_covariate_model() {
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");
    let stan_code = compile_to_stan(&source);

    // Individual parameter vectors
    assert!(stan_code.contains("vector[N] CL"), "Missing CL vector");
    assert!(stan_code.contains("vector[N] V"), "Missing V vector");
    assert!(stan_code.contains("vector[N] Ka"), "Missing Ka vector");

    // Weight normalization
    assert!(
        stan_code.contains("real w = WT[i] / 70.0"),
        "Missing weight normalization"
    );

    // Allometric scaling for CL (exponent 0.75)
    assert!(
        stan_code.contains("pow(w, 0.75)"),
        "Missing allometric scaling for CL"
    );

    // Random effects in exponential form
    assert!(stan_code.contains("exp(eta_CL[i])"), "Missing exp(eta_CL)");
    assert!(stan_code.contains("exp(eta_V[i])"), "Missing exp(eta_V)");
    assert!(stan_code.contains("exp(eta_Ka[i])"), "Missing exp(eta_Ka)");
}

#[test]
fn test_model_block_priors() {
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");
    let stan_code = compile_to_stan(&source);

    // Population parameter priors
    assert!(
        stan_code.contains("CL_pop ~ lognormal"),
        "Missing CL_pop prior"
    );
    assert!(
        stan_code.contains("V_pop ~ lognormal"),
        "Missing V_pop prior"
    );
    assert!(
        stan_code.contains("Ka_pop ~ lognormal"),
        "Missing Ka_pop prior"
    );

    // Variability priors (uses Cauchy distribution)
    assert!(
        stan_code.contains("omega_CL ~ cauchy"),
        "Missing omega_CL prior"
    );
    assert!(
        stan_code.contains("omega_V ~ cauchy"),
        "Missing omega_V prior"
    );
    assert!(
        stan_code.contains("omega_Ka ~ cauchy"),
        "Missing omega_Ka prior"
    );

    // Random effects priors
    assert!(
        stan_code.contains("eta_CL ~ normal(0, omega_CL)"),
        "Missing eta_CL prior"
    );
    assert!(
        stan_code.contains("eta_V ~ normal(0, omega_V)"),
        "Missing eta_V prior"
    );
    assert!(
        stan_code.contains("eta_Ka ~ normal(0, omega_Ka)"),
        "Missing eta_Ka prior"
    );

    // Note: sigma_prop prior not currently emitted in model block
    // It's defined in parameters block with lower bound constraint
}

#[test]
fn test_output_consistency() {
    // Verify that compiling the same source twice produces identical output
    let source_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/examples/one_comp_oral_pk.medlang");

    let source = fs::read_to_string(&source_path).expect("Failed to read canonical example");

    let output1 = compile_to_stan(&source);
    let output2 = compile_to_stan(&source);

    assert_eq!(output1, output2, "Compiler output should be deterministic");
}

#[test]
fn test_golden_file_existence() {
    // After running with UPDATE_GOLDEN=true, verify golden file exists
    let golden_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/canonical_example.stan");

    if !UPDATE_GOLDEN {
        assert!(
            golden_path.exists(),
            "Golden file should exist: {:?}\nRun tests with UPDATE_GOLDEN=true first",
            golden_path
        );

        // Verify it's not empty
        let content = fs::read_to_string(&golden_path).expect("Failed to read golden file");
        assert!(!content.is_empty(), "Golden file should not be empty");
        assert!(
            content.len() > 1000,
            "Golden file should be substantial (>1KB)"
        );
    }
}
