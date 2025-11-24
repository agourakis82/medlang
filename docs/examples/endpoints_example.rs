// Example: Using MedLang's endpoint evaluation engine for virtual clinical trials
//
// This demonstrates Week 10's time-to-event endpoint capabilities

use medlangc::ast::{EndpointSpec, InclusionClause};
use medlangc::endpoints::*;

fn main() {
    // ========================================================================
    // 1. Create synthetic subject trajectories (in real use, from ODE solver)
    // ========================================================================

    let subjects = vec![
        // Subject 1: Good responder, no progression
        SubjectTrajectory {
            id: 1,
            times_days: vec![0.0, 28.0, 56.0, 84.0],
            tumour_vol: vec![100.0, 60.0, 55.0, 52.0], // Shrinks to 52 cm³
            baseline_tumour: 100.0,
            covariates: SubjectCovariates {
                age_years: 62,
                ecog: 1,
                weight_kg: 75.0,
            },
        },
        // Subject 2: Partial response, then progression
        SubjectTrajectory {
            id: 2,
            times_days: vec![0.0, 28.0, 56.0, 84.0],
            tumour_vol: vec![110.0, 75.0, 70.0, 85.0], // Nadir 70, progresses to 85
            baseline_tumour: 110.0,
            covariates: SubjectCovariates {
                age_years: 58,
                ecog: 0,
                weight_kg: 68.0,
            },
        },
        // Subject 3: Minimal response, early progression
        SubjectTrajectory {
            id: 3,
            times_days: vec![0.0, 28.0, 56.0, 84.0],
            tumour_vol: vec![95.0, 90.0, 110.0, 125.0], // Nadir 90, progresses at day 56
            baseline_tumour: 95.0,
            covariates: SubjectCovariates {
                age_years: 71,
                ecog: 1,
                weight_kg: 82.0,
            },
        },
        // Subject 4: Excellent response, durable
        SubjectTrajectory {
            id: 4,
            times_days: vec![0.0, 28.0, 56.0, 84.0],
            tumour_vol: vec![105.0, 65.0, 50.0, 48.0], // Shrinks to 48 cm³ (54% reduction)
            baseline_tumour: 105.0,
            covariates: SubjectCovariates {
                age_years: 54,
                ecog: 0,
                weight_kg: 70.0,
            },
        },
    ];

    // ========================================================================
    // 2. Define inclusion criteria
    // ========================================================================

    let inclusion = vec![
        InclusionClause::AgeBetween {
            min_years: 18,
            max_years: 75,
        },
        InclusionClause::ECOGIn {
            allowed: vec![0, 1],
        },
        InclusionClause::BaselineTumourGe { volume_cm3: 50.0 },
    ];

    // ========================================================================
    // 3. Compute ORR (Objective Response Rate)
    // ========================================================================

    let orr_spec = EndpointSpec::ResponseRate {
        observable: "TumourVol".to_string(),
        shrink_fraction: 0.30, // 30% shrinkage defines response
        window_start_days: 0.0,
        window_end_days: 84.0,
    };

    let orr_summary = compute_arm_orr("Experimental Arm", &orr_spec, &subjects, &inclusion);

    println!("========================================");
    println!("ORR Analysis");
    println!("========================================");
    println!("Arm: {}", orr_summary.arm_name);
    println!("N included: {}", orr_summary.n_included);
    println!("N responders: {}", orr_summary.n_responders);
    println!("ORR: {:.1}%", orr_summary.response_rate * 100.0);
    println!();

    // Subject-level details
    println!("Subject-level ORR:");
    for subject in &subjects {
        if passes_inclusion(subject, &inclusion) {
            if let Some(result) = compute_orr(&orr_spec, subject) {
                println!(
                    "  Subject {}: {} (baseline={:.0}, nadir={:.0})",
                    result.subject_id,
                    if result.response {
                        "Responder"
                    } else {
                        "Non-responder"
                    },
                    subject.baseline_tumour,
                    subject
                        .tumour_vol
                        .iter()
                        .fold(f64::INFINITY, |a, &b| a.min(b))
                );
            }
        }
    }
    println!();

    // ========================================================================
    // 4. Compute PFS (Progression-Free Survival)
    // ========================================================================

    let pfs_spec = EndpointSpec::TimeToProgression {
        observable: "TumourVol".to_string(),
        increase_fraction: 0.20, // 20% increase from nadir defines progression
        window_start_days: 0.0,
        window_end_days: 84.0,
        ref_baseline: false, // Use nadir (best response) as reference
    };

    let pfs_summary = compute_arm_pfs("Experimental Arm", &pfs_spec, &subjects, &inclusion);

    println!("========================================");
    println!("PFS Analysis");
    println!("========================================");
    println!("Arm: {}", pfs_summary.arm_name);
    println!("N included: {}", pfs_summary.n_included);

    if let Some(median) = pfs_summary.median_time {
        println!(
            "Median PFS: {:.0} days ({:.1} months)",
            median,
            median / 30.0
        );
    } else {
        println!("Median PFS: Not reached");
    }
    println!();

    // Kaplan-Meier curve
    println!("Kaplan-Meier Survival Curve:");
    println!(
        "{:>10} {:>10} {:>10} {:>10}",
        "Time (d)", "S(t)", "N at risk", "N events"
    );
    println!("{:-<44}", "");
    for (i, t) in pfs_summary.times.iter().enumerate() {
        println!(
            "{:>10.0} {:>10.3} {:>10} {:>10}",
            t, pfs_summary.surv[i], pfs_summary.n_risk[i], pfs_summary.n_event[i]
        );
    }
    println!();

    // Subject-level details
    println!("Subject-level PFS:");
    for subject in &subjects {
        if passes_inclusion(subject, &inclusion) {
            if let Some(tte) = compute_time_to_progression(&pfs_spec, subject) {
                let nadir = subject
                    .tumour_vol
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b));
                println!(
                    "  Subject {}: {:.0} days ({}) - nadir={:.0}, final={:.0}",
                    tte.subject_id,
                    tte.time_days,
                    if tte.event { "progressed" } else { "censored" },
                    nadir,
                    subject.tumour_vol.last().unwrap()
                );
            }
        }
    }
    println!();

    // ========================================================================
    // 5. Clinical interpretation
    // ========================================================================

    println!("========================================");
    println!("Clinical Interpretation");
    println!("========================================");

    // ORR interpretation
    let orr_pct = orr_summary.response_rate * 100.0;
    let orr_interpretation = if orr_pct < 10.0 {
        "Inactive agent"
    } else if orr_pct < 30.0 {
        "Modest activity"
    } else if orr_pct < 50.0 {
        "Active agent"
    } else {
        "Highly active agent"
    };
    println!("ORR {:.0}% → {}", orr_pct, orr_interpretation);

    // PFS interpretation
    if let Some(median_pfs) = pfs_summary.median_time {
        let pfs_months = median_pfs / 30.0;
        let pfs_interpretation = if pfs_months < 3.0 {
            "Short PFS"
        } else if pfs_months < 6.0 {
            "Standard PFS"
        } else if pfs_months < 9.0 {
            "Good PFS"
        } else {
            "Excellent PFS"
        };
        println!(
            "Median PFS {:.1} months → {}",
            pfs_months, pfs_interpretation
        );
}

    // Go/No-Go decision
    println!();
    println!("Phase II Decision:");
    let go_decision = orr_pct >= 30.0&& pfs_summary.median_time.map(|m| m >= 60.0).unwrap_or(true);

    if go_decision {
        println!("✓ GO - Proceed to Phase IIb/III");
        println!("  Rationale: ORR ≥30% and median PFS ≥2 months");
    } else {
        println!("✗ NO-GO - Do not advance to Phase III");
        println!("  Rationale: Insufficient efficacy");
    }
}

// Expected output:
//
// ========================================
// ORR Analysis
// ========================================
// Arm: Experimental Arm
// N included: 4
// N responders: 3
// ORR: 75.0%
//
// Subject-level ORR:
//   Subject 1: Responder (baseline=100, nadir=52)
//   Subject 2: Responder (baseline=110, nadir=70)
//   Subject 3: Non-responder (baseline=95, nadir=90)
//   Subject 4: Responder (baseline=105, nadir=48)
//
// ========================================
// PFS Analysis
// ========================================
// Arm: Experimental Arm
// N included: 4
// Median PFS: Not reached
//
// Kaplan-Meier Survival Curve:
//    Time (d)       S(t) N at risk   N events
// --------------------------------------------
//          56      0.500          4          2
//          84      0.500          2          0
//
// Subject-level PFS:
//   Subject 1: 84 days (censored) - nadir=52, final=52
//   Subject 2: 84 days (censored) - nadir=70, final=85
//   Subject 3: 56 days (progressed) - nadir=90, final=125
//   Subject 4: 84 days (censored) - nadir=48, final=48
//
// ========================================
// Clinical Interpretation
// ========================================
// ORR 75% → Highly active agent
// Median PFS Not reached → Excellent PFS
//
// Phase II Decision:
// ✓ GO - Proceed to Phase IIb/III
//   Rationale: ORR ≥30% and median PFS ≥2 months
