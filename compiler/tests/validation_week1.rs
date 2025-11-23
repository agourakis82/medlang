//! Week 1 validation tests
//!
//! Ensures all Week 1 deliverables are correct:
//! 1. Grammar specification exists and is well-formed
//! 2. Canonical example parses (manual check for now)
//! 3. Dataset has correct structure and statistics

use std::fs;
use std::path::Path;

#[test]
fn test_grammar_spec_exists() {
    let path = "../docs/medlang_d_minimal_grammar_v0.md";
    assert!(Path::new(path).exists(), "Grammar spec missing: {}", path);

    let content = fs::read_to_string(path).expect("Failed to read grammar spec");

    // Should contain key sections
    assert!(content.contains("## Lexical Elements"));
    assert!(content.contains("## Model Definition"));
    assert!(content.contains("## Population Definition"));
    assert!(content.contains("## Measure Definition"));
    assert!(content.contains("## Timeline Definition"));
    assert!(content.contains("## Type Expressions"));

    // Should have EBNF notation
    assert!(content.contains("::="));

    println!("✓ Grammar spec is well-formed ({} bytes)", content.len());
}

#[test]
fn test_canonical_example_exists() {
    let path = "../docs/examples/one_comp_oral_pk.medlang";
    assert!(
        Path::new(path).exists(),
        "Canonical example missing: {}",
        path
    );

    let content = fs::read_to_string(path).expect("Failed to read example");

    // Should contain all major constructs
    assert!(content.contains("model OneCompOral"));
    assert!(content.contains("population OneCompOralPop"));
    assert!(content.contains("measure ConcPropError"));
    assert!(content.contains("timeline OneCompOralTimeline"));
    assert!(content.contains("cohort OneCompCohort"));

    // Should have states
    assert!(content.contains("state A_gut"));
    assert!(content.contains("state A_central"));

    // Should have ODEs
    assert!(content.contains("dA_gut/dt"));
    assert!(content.contains("dA_central/dt"));

    // Should have observable
    assert!(content.contains("obs C_plasma"));

    // Should have unit annotations
    assert!(content.contains("DoseMass"));
    assert!(content.contains("ConcMass"));
    assert!(content.contains("Clearance"));

    let lines = content.lines().count();
    println!("✓ Canonical example is well-formed ({} lines)", lines);
}

#[test]
fn test_dataset_structure() {
    let path = "../docs/examples/onecomp_synth.csv";
    assert!(Path::new(path).exists(), "Dataset missing: {}", path);

    let content = fs::read_to_string(path).expect("Failed to read dataset");
    let lines: Vec<&str> = content.lines().collect();

    // Check header
    assert_eq!(lines[0], "ID,TIME,DV,WT,EVID,AMT", "Invalid CSV header");

    // Count rows
    let n_rows = lines.len() - 1; // Exclude header
    println!("Dataset has {} data rows", n_rows);

    // Should have 20 subjects × 7 events = 140 rows
    assert_eq!(n_rows, 140, "Expected 140 data rows");

    // Count dose rows (EVID=1)
    let dose_rows = lines
        .iter()
        .skip(1)
        .filter(|line| line.contains(",1,"))
        .count();
    assert_eq!(dose_rows, 20, "Expected 20 dose rows (1 per subject)");

    // Count observation rows (EVID=0)
    let obs_rows = lines
        .iter()
        .skip(1)
        .filter(|line| line.contains(",0,"))
        .count();
    assert_eq!(
        obs_rows, 120,
        "Expected 120 observation rows (6 per subject)"
    );

    println!(
        "✓ Dataset structure valid: {} dose + {} obs = {} total",
        dose_rows, obs_rows, n_rows
    );
}

#[test]
fn test_dataset_values_reasonable() {
    let path = "../docs/examples/onecomp_synth.csv";
    let content = fs::read_to_string(path).expect("Failed to read dataset");

    let mut weights = Vec::new();
    let mut concentrations = Vec::new();

    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 {
            continue;
        }

        // Parse weight
        if let Ok(wt) = fields[3].parse::<f64>() {
            weights.push(wt);
        }

        // Parse DV (concentration) for observation rows
        if fields[4] == "0" {
            if let Ok(dv) = fields[2].parse::<f64>() {
                concentrations.push(dv);
            }
        }
    }

    // Check weights in reasonable range [50, 90] kg
    let min_wt = weights.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_wt = weights.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!(
        min_wt >= 50.0 && min_wt <= 90.0,
        "Min weight out of range: {}",
        min_wt
    );
    assert!(
        max_wt >= 50.0 && max_wt <= 90.0,
        "Max weight out of range: {}",
        max_wt
    );
    println!("✓ Weights in range [{:.1}, {:.1}] kg", min_wt, max_wt);

    // Check concentrations are positive and reasonable
    let min_conc = concentrations.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_conc = concentrations
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    assert!(min_conc >= 0.0, "Negative concentration: {}", min_conc);
    assert!(
        max_conc < 10.0,
        "Suspiciously high concentration: {}",
        max_conc
    );
    println!(
        "✓ Concentrations in range [{:.3}, {:.3}] mg/L",
        min_conc, max_conc
    );

    // Check mean concentration is reasonable (rough PK check)
    let mean_conc: f64 = concentrations.iter().sum::<f64>() / concentrations.len() as f64;
    assert!(
        mean_conc > 0.1 && mean_conc < 5.0,
        "Mean concentration suspicious: {}",
        mean_conc
    );
    println!("✓ Mean concentration: {:.3} mg/L", mean_conc);
}

#[test]
fn test_datagen_reproducibility() {
    use medlangc::datagen::{generate_dataset, TrueParams};

    let params = TrueParams::default();
    let obs_times = vec![1.0, 2.0, 4.0, 8.0, 12.0, 24.0];

    // Generate with same seed twice
    let data1 = generate_dataset(5, &obs_times, 100.0, &params, 12345);
    let data2 = generate_dataset(5, &obs_times, 100.0, &params, 12345);

    assert_eq!(data1.len(), data2.len());

    for (r1, r2) in data1.iter().zip(data2.iter()) {
        assert_eq!(r1.id, r2.id);
        assert_eq!(r1.time, r2.time);
        assert_eq!(r1.evid, r2.evid);
        assert!((r1.wt - r2.wt).abs() < 1e-10, "Weight mismatch");

        match (r1.dv, r2.dv) {
            (Some(v1), Some(v2)) => assert!((v1 - v2).abs() < 1e-10, "DV mismatch"),
            (None, None) => {}
            _ => panic!("DV presence mismatch"),
        }
    }

    println!("✓ Data generation is deterministic with same seed");
}
