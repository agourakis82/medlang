//! Test composite model connection resolution (Week 7 fix)

use medlangc::ast::*;
use medlangc::ir::*;
use medlangc::lexer::tokenize;
use medlangc::lower::lower_program;
use medlangc::parser::parse_program;

#[test]
fn test_composite_model_input_substitution() {
    let source = r#"
model PBPK_Simple {
    state A_plasma : DoseMass
    param CL : Clearance
    param V : Volume
    let C_plasma = A_plasma / V
    dA_plasma/dt = -CL * C_plasma
    obs C_plasma_obs : ConcMass = C_plasma
}

model QSP_Simple {
    input C_drug : ConcMass
    state Tumor : TumourVolume
    param k_grow : RateConst
    param Emax : f64
    param EC50 : ConcMass
    let E_drug = Emax * C_drug / (EC50 + C_drug)
    dTumor/dt = k_grow * Tumor - E_drug * Tumor
    obs TumorVol : TumourVolume = Tumor
}

model Composite {
    submodel PK : PBPK_Simple
    submodel PD : QSP_Simple
    connect {
        PD.C_drug = PK.C_plasma_obs
    }
}

population CompositePop {
    model Composite
    param CL_pop : Clearance
    param V_pop : Volume
    param k_grow_pop : RateConst
    param Emax_pop : f64
    param EC50_pop : ConcMass
    bind_params(patient) {
        model.CL = CL_pop
        model.V = V_pop
        model.k_grow = k_grow_pop
        model.Emax = Emax_pop
        model.EC50 = EC50_pop
    }
    use_measure TestMeasure for model.C_plasma_obs
}

measure TestMeasure {
    pred : ConcMass
    obs : ConcMass
    param sigma : f64
    log_likelihood = Normal_logpdf(obs, pred, sigma)
}
"#;

    let tokens = tokenize(source).expect("Tokenization failed");
    let program = parse_program(&tokens).expect("Parsing failed");
    let ir = lower_program(&program).expect("Lowering failed");

    // Verify the IR model was created
    assert_eq!(ir.model.name, "Composite");

    // Verify we have both states from both submodels
    assert_eq!(ir.model.states.len(), 2, "Should have 2 states (A_plasma, Tumor)");
    
    // Verify we have all parameters
    assert!(ir.model.params.len() >= 5, "Should have at least 5 parameters");

    // Verify we have intermediates (C_plasma, E_drug)
    assert_eq!(ir.model.intermediates.len(), 2, "Should have 2 intermediates");

    // Verify we have both ODEs
    assert_eq!(ir.model.odes.len(), 2, "Should have 2 ODEs");

    // Key test: Verify that E_drug intermediate uses C_plasma, not C_drug
    let e_drug_intermediate = ir.model.intermediates
        .iter()
        .find(|i| i.name == "E_drug")
        .expect("E_drug intermediate should exist");

    // The expression should contain references to C_plasma (from the observable resolution)
    // not C_drug (the input variable that should have been substituted)
    let expr_contains_c_plasma = contains_var(&e_drug_intermediate.expr, "C_plasma");
    let expr_contains_c_drug = contains_var(&e_drug_intermediate.expr, "C_drug");
    let expr_contains_c_plasma_obs = contains_var(&e_drug_intermediate.expr, "C_plasma_obs");

    assert!(expr_contains_c_plasma, 
        "E_drug should reference C_plasma (the inlined observable expression)");
    assert!(!expr_contains_c_drug, 
        "E_drug should NOT reference C_drug (the input that was substituted)");
    assert!(!expr_contains_c_plasma_obs, 
        "E_drug should NOT reference C_plasma_obs (should be inlined)");

    println!("✓ Composite model input substitution test passed");
    println!("  E_drug correctly uses C_plasma instead of C_drug");
}

/// Helper function to check if an expression contains a variable reference
fn contains_var(expr: &IRExpr, var_name: &str) -> bool {
    match expr {
        IRExpr::Var(name) => name == var_name,
        IRExpr::Literal(_) => false,
        IRExpr::Unary(_, operand) => contains_var(operand, var_name),
        IRExpr::Binary(_, lhs, rhs) => {
            contains_var(lhs, var_name) || contains_var(rhs, var_name)
        }
        IRExpr::Index(array, index) => {
            contains_var(array, var_name) || contains_var(index, var_name)
        }
        IRExpr::Call(_, args) => {
            args.iter().any(|arg| contains_var(arg, var_name))
        }
    }
}
