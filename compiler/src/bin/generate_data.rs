//! Binary to generate synthetic dataset for testing.
//!
//! Usage: cargo run --bin generate_data

use medlangc::datagen::{generate_dataset, write_csv, TrueParams};

fn main() {
    println!("Generating synthetic one-compartment oral PK dataset...");

    let params = TrueParams::default();
    let n_subjects = 20;
    let obs_times = vec![1.0, 2.0, 4.0, 8.0, 12.0, 24.0];
    let dose_amount = 100.0;
    let seed = 42;

    let data = generate_dataset(n_subjects, &obs_times, dose_amount, &params, seed);

    let output_path = "../docs/examples/onecomp_synth.csv";
    write_csv(output_path, &data).expect("Failed to write CSV");

    println!("✓ Generated: {}", output_path);
    println!("  Subjects: {}", n_subjects);
    println!("  Total rows: {}", data.len());
    println!(
        "  Dose rows (EVID=1): {}",
        data.iter().filter(|r| r.evid == 1).count()
    );
    println!(
        "  Observation rows (EVID=0): {}",
        data.iter().filter(|r| r.evid == 0).count()
    );
    println!();
    println!("True population parameters:");
    println!("  CL_pop = {} L/h", params.cl_pop);
    println!("  V_pop = {} L", params.v_pop);
    println!("  Ka_pop = {} 1/h", params.ka_pop);
    println!("  omega_CL = {}", params.omega_cl);
    println!("  omega_V = {}", params.omega_v);
    println!("  omega_Ka = {}", params.omega_ka);
    println!("  sigma_prop = {}", params.sigma_prop);
}
