//! Simple protocol simulator
//!
//! Simulates a clinical trial protocol with synthetic subject trajectories
//! and computes endpoints (ORR, PFS) per arm.

use medlangc::ast::*;
use medlangc::endpoints::*;
use medlangc::lexer::tokenize;
use medlangc::parser::parse_program;
use rand::Rng;
use serde_json::json;
use std::fs;
use std::path::PathBuf;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: simulate_protocol <protocol.medlang> <n_subjects_per_arm>");
        eprintln!();
        eprintln!("Example:");
        eprintln!("  simulate_protocol examples/simple_protocol.medlang 100");
        std::process::exit(1);
    }

    let protocol_file = &args[1];
    let n_per_arm: usize = args[2]
        .parse()
        .expect("n_subjects_per_arm must be a number");

    println!("MedLang Protocol Simulator");
    println!("==========================");
    println!();

    // Load and parse protocol
    println!("Loading protocol: {}", protocol_file);
    let source = fs::read_to_string(protocol_file).expect("Failed to read protocol file");

    let tokens = tokenize(&source).expect("Failed to tokenize protocol");

    let program = parse_program(&tokens).expect("Failed to parse protocol");

    // Extract protocol
    let protocol = match program.declarations.first() {
        Some(Declaration::Protocol(p)) => p,
        _ => {
            eprintln!("Error: No protocol found in file");
            std::process::exit(1);
        }
    };

    println!("Protocol: {}", protocol.name);
    println!("  Arms: {}", protocol.arms.len());
    println!("  Visits: {}", protocol.visits.len());
    println!("  Endpoints: {}", protocol.endpoints.len());
    println!();

    // Simulate each arm
    let mut arm_results = Vec::new();

    for arm in &protocol.arms {
        println!(
            "Simulating {} (dose={} mg, N={})...",
            arm.label, arm.dose_mg, n_per_arm
        );

        // Generate synthetic subjects
        let subjects = generate_synthetic_subjects(
            n_per_arm,
            &protocol.visits,
            arm.dose_mg,
            protocol.inclusion.as_ref(),
        );

        // Compute endpoints
        let mut arm_result = json!({
            "arm": arm.name,
            "label": arm.label,
            "dose_mg": arm.dose_mg,
            "n_subjects": n_per_arm,
        });

        let inclusion_criteria: Vec<InclusionClause> = protocol
            .inclusion
            .as_ref()
            .map(|inc| inc.clauses.clone())
            .unwrap_or_default();

        for endpoint in &protocol.endpoints {
            match &endpoint.spec {
                EndpointSpec::ResponseRate { .. } => {
                    let orr_summary =
                        compute_arm_orr(&arm.name, &endpoint.spec, &subjects, &inclusion_criteria);

                    arm_result["endpoints"][&endpoint.name] = json!({
                        "type": "binary",
                        "n_included": orr_summary.n_included,
                        "n_responders": orr_summary.n_responders,
                        "response_rate": orr_summary.response_rate,
                    });

                    println!(
                        "  {}: {:.1}% ({}/{})",
                        endpoint.name,
                        orr_summary.response_rate * 100.0,
                        orr_summary.n_responders,
                        orr_summary.n_included,
                    );
                }

                EndpointSpec::TimeToProgression { .. } => {
                    let pfs_summary =
                        compute_arm_pfs(&arm.name, &endpoint.spec, &subjects, &inclusion_criteria);

                    arm_result["endpoints"][&endpoint.name] = json!({
                        "type": "time_to_event",
                        "n_included": pfs_summary.n_included,
                        "median_days": pfs_summary.median_time,
                        "km_times": pfs_summary.times,
                        "km_surv": pfs_summary.surv,
                        "km_n_risk": pfs_summary.n_risk,
                        "km_n_event": pfs_summary.n_event,
                    });

                    if let Some(median) = pfs_summary.median_time {
                        println!(
                            "  {}: median={:.0} days ({:.1} months)",
                            endpoint.name,
                            median,
                            median / 30.0,
                        );
                    } else {
                        println!("  {}: median not reached", endpoint.name);
                    }
                }
            }
        }

        arm_results.push(arm_result);
        println!();
    }

    // Output results as JSON
    let output = json!({
        "protocol": protocol.name,
        "n_per_arm": n_per_arm,
        "arms": arm_results,
    });

    let output_file = PathBuf::from(protocol_file).with_extension("sim_results.json");

    fs::write(&output_file, serde_json::to_string_pretty(&output).unwrap())
        .expect("Failed to write output file");

    println!("Results saved to: {}", output_file.display());
}

/// Generate synthetic subject trajectories for a trial arm
///
/// This is a placeholder that generates simple exponential decay/growth
/// patterns. In a real implementation, this would:
/// 1. Load the population PBPK+QSP model
/// 2. Solve ODEs for each subject with random parameters
/// 3. Extract tumor volume trajectories
fn generate_synthetic_subjects(
    n: usize,
    visits: &[VisitDef],
    dose_mg: f64,
    inclusion: Option<&InclusionDef>,
) -> Vec<SubjectTrajectory> {
    let mut rng = rand::thread_rng();
    let mut subjects = Vec::new();

    // Extract visit times
    let times: Vec<f64> = visits.iter().map(|v| v.time_days).collect();

    for id in 0..n {
        // Generate random covariates
        let age = rng.gen_range(18..=75);
        let ecog = rng.gen_range(0..=2);
        let weight = rng.gen_range(50.0..100.0);

        // Generate baseline tumor volume
        let baseline = rng.gen_range(50.0..150.0);

        // Generate tumor trajectory based on dose
        // Higher dose → better response (more shrinkage, less progression)
        let response_quality = (dose_mg / 200.0).min(1.0); // Normalize: 200mg = full effect
        let shrinkage_rate = rng.gen_range(0.1..0.6) * response_quality; // 10-60% max shrinkage
        let growth_rate = rng.gen_range(0.0..0.05) * (1.0 - response_quality * 0.5);

        let mut trajectory = Vec::new();
        let mut current = baseline;

        for &t in &times {
            if t == 0.0 {
                trajectory.push(baseline);
            } else {
                // Simple model: shrink first, then potentially grow back
                let shrink_phase = (t / 56.0).min(1.0); // Shrink over first 8 weeks
                let shrunk = baseline * (1.0 - shrinkage_rate * shrink_phase);

                // Then potential regrowth
                if shrink_phase >= 1.0 {
                    let grow_phase = (t - 56.0) / 28.0;
                    current = shrunk * (1.0 + growth_rate * grow_phase);
                } else {
                    current = shrunk;
                }

                // Add some noise
                current *= rng.gen_range(0.95..1.05);
                current = current.max(10.0); // Minimum tumor size

                trajectory.push(current);
            }
        }

        subjects.push(SubjectTrajectory {
            id,
            times_days: times.clone(),
            tumour_vol: trajectory,
            baseline_tumour: baseline,
            covariates: SubjectCovariates {
                age_years: age,
                ecog,
                weight_kg: weight,
            },
        });
    }

    subjects
}
