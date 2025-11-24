//! Endpoint evaluation for clinical trial protocols
//!
//! This module provides functionality to compute clinical trial endpoints
//! from mechanistic model trajectories. Supports:
//! - Binary endpoints (ORR - Objective Response Rate)
//! - Time-to-event endpoints (PFS/TTP - Progression-Free Survival)

use crate::ast::{EndpointSpec, InclusionClause};
use serde::{Deserialize, Serialize};

/// Subject covariate data for inclusion/exclusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectCovariates {
    pub age_years: u32,
    pub ecog: u8,
    pub weight_kg: f64,
}

/// Time-series trajectory for a single subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubjectTrajectory {
    pub id: usize,
    pub times_days: Vec<f64>,
    pub tumour_vol: Vec<f64>, // Tumour volume at each time point
    pub baseline_tumour: f64, // Baseline (first) tumour volume
    pub covariates: SubjectCovariates,
}

/// Binary endpoint result (e.g., responder or not)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryEndpointResult {
    pub subject_id: usize,
    pub response: bool,
}

/// Time-to-event result for a single subject
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeToEvent {
    pub subject_id: usize,
    pub time_days: f64,
    pub event: bool, // true = event occurred, false = censored
}

/// Arm-level summary for a binary endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmBinarySummary {
    pub arm_name: String,
    pub n_included: usize,
    pub n_responders: usize,
    pub response_rate: f64,
}

/// Kaplan-Meier survival analysis for an arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArmSurvivalSummary {
    pub arm_name: String,
    pub n_included: usize,
    pub times: Vec<f64>,          // Unique event/censoring times
    pub surv: Vec<f64>,           // S(t) - survival function at each time
    pub n_risk: Vec<usize>,       // Number at risk at each time
    pub n_event: Vec<usize>,      // Number of events at each time
    pub median_time: Option<f64>, // Median survival time (first t where S(t) <= 0.5)
}

/// Check if a subject passes inclusion criteria
pub fn passes_inclusion(subject: &SubjectTrajectory, inclusion: &[InclusionClause]) -> bool {
    for clause in inclusion {
        match clause {
            InclusionClause::AgeBetween {
                min_years,
                max_years,
            } => {
                if subject.covariates.age_years < *min_years
                    || subject.covariates.age_years > *max_years
                {
                    return false;
                }
            }
            InclusionClause::ECOGIn { allowed } => {
                if !allowed.contains(&subject.covariates.ecog) {
                    return false;
                }
            }
            InclusionClause::BaselineTumourGe { volume_cm3 } => {
                if subject.baseline_tumour < *volume_cm3 {
                    return false;
                }
            }
        }
    }
    true
}

/// Compute ORR (Objective Response Rate) for a subject
///
/// Response is defined as >= shrink_fraction reduction in tumour volume
/// from baseline, measured within the specified time window.
pub fn compute_orr(
    spec: &EndpointSpec,
    subject: &SubjectTrajectory,
) -> Option<BinaryEndpointResult> {
    if let EndpointSpec::ResponseRate {
        shrink_fraction,
        window_start_days,
        window_end_days,
        ..
    } = spec
    {
        let baseline = subject.baseline_tumour;
        let response_threshold = baseline * (1.0 - shrink_fraction);

        // Check all time points within the window
        for (t, vol) in subject.times_days.iter().zip(subject.tumour_vol.iter()) {
            if *t >= *window_start_days && *t <= *window_end_days {
                if *vol <= response_threshold {
                    return Some(BinaryEndpointResult {
                        subject_id: subject.id,
                        response: true,
                    });
                }
            }
        }

        Some(BinaryEndpointResult {
            subject_id: subject.id,
            response: false,
        })
    } else {
        None
    }
}

/// Compute time to progression (PFS/TTP) for a subject
///
/// Progression is defined as increase_fraction increase above the reference
/// (either baseline or nadir/best response).
pub fn compute_time_to_progression(
    spec: &EndpointSpec,
    subject: &SubjectTrajectory,
) -> Option<TimeToEvent> {
    if let EndpointSpec::TimeToProgression {
        increase_fraction,
        window_start_days,
        window_end_days,
        ref_baseline,
        ..
    } = spec
    {
        // Collect in-window measurements
        let mut in_window_times = Vec::new();
        let mut in_window_vols = Vec::new();

        for (t, vol) in subject.times_days.iter().zip(subject.tumour_vol.iter()) {
            if *t >= *window_start_days && *t <= *window_end_days {
                in_window_times.push(*t);
                in_window_vols.push(*vol);
            }
        }

        if in_window_times.is_empty() {
            return None;
        }

        if *ref_baseline {
            // Reference is baseline (first value)
            let baseline = in_window_vols[0];
            let progression_threshold = baseline * (1.0 + increase_fraction);

            // Find first time where tumour crosses threshold
            for (t, vol) in in_window_times.iter().zip(in_window_vols.iter()) {
                if *vol >= progression_threshold {
                    return Some(TimeToEvent {
                        subject_id: subject.id,
                        time_days: *t,
                        event: true,
                    });
                }
            }
        } else {
            // Reference is nadir (running minimum up to each point)
            let mut running_min = in_window_vols[0];

            for (i, (t, vol)) in in_window_times
                .iter()
                .zip(in_window_vols.iter())
                .enumerate()
            {
                // Update running minimum
                if i > 0 && *vol < running_min {
                    running_min = *vol;
                }

                // Check for progression relative to current nadir
                let progression_threshold = running_min * (1.0 + increase_fraction);
                if *vol >= progression_threshold && i > 0 {
                    // Only count as progression if not at baseline
                    return Some(TimeToEvent {
                        subject_id: subject.id,
                        time_days: *t,
                        event: true,
                    });
                }
            }
        }

        // No progression observed - censored at last observation
        Some(TimeToEvent {
            subject_id: subject.id,
            time_days: *in_window_times.last().unwrap(),
            event: false,
        })
    } else {
        None
    }
}

/// Compute ORR summary for an arm
pub fn compute_arm_orr(
    arm_name: &str,
    spec: &EndpointSpec,
    subjects: &[SubjectTrajectory],
    inclusion: &[InclusionClause],
) -> ArmBinarySummary {
    let mut included_subjects = Vec::new();

    // Filter by inclusion criteria
    for subject in subjects {
        if passes_inclusion(subject, inclusion) {
            included_subjects.push(subject);
        }
    }

    let n_included = included_subjects.len();
    let mut n_responders = 0;

    // Compute response for each included subject
    for subject in &included_subjects {
        if let Some(result) = compute_orr(spec, subject) {
            if result.response {
                n_responders += 1;
            }
        }
    }

    let response_rate = if n_included > 0 {
        n_responders as f64 / n_included as f64
    } else {
        0.0
    };

    ArmBinarySummary {
        arm_name: arm_name.to_string(),
        n_included,
        n_responders,
        response_rate,
    }
}

/// Compute Kaplan-Meier survival curves
///
/// Input: vectors of times and event indicators
/// Output: survival function S(t) at each unique event time
pub fn kaplan_meier(
    times: &[f64],
    events: &[bool],
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
    if times.is_empty() {
        return (vec![], vec![], vec![], vec![]);
    }

    // Create (time, event) pairs and sort by time
    let mut data: Vec<(f64, bool)> = times
        .iter()
        .zip(events.iter())
        .map(|(t, e)| (*t, *e))
        .collect();
    data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // Group by unique times
    let mut unique_times = Vec::new();
    let mut n_events_at_time = Vec::new();
    let mut n_censored_at_time = Vec::new();

    let mut i = 0;
    while i < data.len() {
        let current_time = data[i].0;
        let mut n_event = 0;
        let mut n_censor = 0;

        // Count events and censoring at this time
        while i < data.len() && (data[i].0 - current_time).abs() < 1e-6 {
            if data[i].1 {
                n_event += 1;
            } else {
                n_censor += 1;
            }
            i += 1;
        }

        unique_times.push(current_time);
        n_events_at_time.push(n_event);
        n_censored_at_time.push(n_censor);
    }

    // Compute Kaplan-Meier estimate
    let n_total = data.len();
    let mut surv_probs = Vec::new();
    let mut n_at_risk_vec = Vec::new();
    let mut n_events_vec = Vec::new();

    let mut s_t = 1.0;
    let mut n_at_risk = n_total;

    for (j, &time) in unique_times.iter().enumerate() {
        let d_j = n_events_at_time[j];
        let c_j = n_censored_at_time[j];

        n_at_risk_vec.push(n_at_risk);
        n_events_vec.push(d_j);

        if n_at_risk > 0 && d_j > 0 {
            s_t *= 1.0 - (d_j as f64 / n_at_risk as f64);
        }

        surv_probs.push(s_t);

        // Update at-risk count (events and censoring reduce the risk set)
        n_at_risk = n_at_risk.saturating_sub(d_j + c_j);
    }

    (unique_times, surv_probs, n_at_risk_vec, n_events_vec)
}

/// Compute median survival time from KM curve
fn compute_median_survival(times: &[f64], surv: &[f64]) -> Option<f64> {
    for (t, s) in times.iter().zip(surv.iter()) {
        if *s <= 0.5 {
            return Some(*t);
        }
    }
    None // Median not reached
}

/// Compute PFS/TTP summary for an arm
pub fn compute_arm_pfs(
    arm_name: &str,
    spec: &EndpointSpec,
    subjects: &[SubjectTrajectory],
    inclusion: &[InclusionClause],
) -> ArmSurvivalSummary {
    let mut included_subjects = Vec::new();

    // Filter by inclusion criteria
    for subject in subjects {
        if passes_inclusion(subject, inclusion) {
            included_subjects.push(subject);
        }
    }

    let n_included = included_subjects.len();
    let mut times = Vec::new();
    let mut events = Vec::new();

    // Compute TTE for each included subject
    for subject in &included_subjects {
        if let Some(tte) = compute_time_to_progression(spec, subject) {
            times.push(tte.time_days);
            events.push(tte.event);
        }
    }

    // Compute Kaplan-Meier
    let (km_times, km_surv, km_n_risk, km_n_event) = kaplan_meier(&times, &events);
    let median_time = compute_median_survival(&km_times, &km_surv);

    ArmSurvivalSummary {
        arm_name: arm_name.to_string(),
        n_included,
        times: km_times,
        surv: km_surv,
        n_risk: km_n_risk,
        n_event: km_n_event,
        median_time,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_subject(id: usize, tumour_trajectory: Vec<f64>) -> SubjectTrajectory {
        let times_days: Vec<f64> = (0..tumour_trajectory.len())
            .map(|i| i as f64 * 28.0)
            .collect();
        let baseline_tumour = tumour_trajectory[0];

        SubjectTrajectory {
            id,
            times_days,
            tumour_vol: tumour_trajectory,
            baseline_tumour,
            covariates: SubjectCovariates {
                age_years: 60,
                ecog: 1,
                weight_kg: 70.0,
            },
        }
    }

    #[test]
    fn test_orr_responder() {
        let spec = EndpointSpec::ResponseRate {
            observable: "TumourVol".to_string(),
            shrink_fraction: 0.30,
            window_start_days: 0.0,
            window_end_days: 84.0,
        };

        // Subject with 40% shrinkage (baseline=100, nadir=60)
        let subject = create_test_subject(1, vec![100.0, 80.0, 60.0]);

        let result = compute_orr(&spec, &subject).unwrap();
        assert!(result.response, "Should be responder with 40% shrinkage");
    }

    #[test]
    fn test_orr_non_responder() {
        let spec = EndpointSpec::ResponseRate {
            observable: "TumourVol".to_string(),
            shrink_fraction: 0.30,
            window_start_days: 0.0,
            window_end_days: 84.0,
        };

        // Subject with only 20% shrinkage (baseline=100, nadir=80)
        let subject = create_test_subject(1, vec![100.0, 90.0, 80.0]);

        let result = compute_orr(&spec, &subject).unwrap();
        assert!(
            !result.response,
            "Should be non-responder with only 20% shrinkage"
        );
    }

    #[test]
    fn test_pfs_progression_from_baseline() {
        let spec = EndpointSpec::TimeToProgression {
            observable: "TumourVol".to_string(),
            increase_fraction: 0.20,
            window_start_days: 0.0,
            window_end_days: 84.0,
            ref_baseline: true, // Reference is baseline
        };

        // Baseline=100, progression at day 56 (tumour=120)
        let subject = create_test_subject(1, vec![100.0, 110.0, 120.0]);

        let result = compute_time_to_progression(&spec, &subject).unwrap();
        assert!(result.event, "Should have progression event");
        assert_eq!(result.time_days, 56.0, "Progression at day 56");
    }

    #[test]
    fn test_pfs_progression_from_nadir() {
        let spec = EndpointSpec::TimeToProgression {
            observable: "TumourVol".to_string(),
            increase_fraction: 0.20,
            window_start_days: 0.0,
            window_end_days: 84.0,
            ref_baseline: false, // Reference is nadir (best response)
        };

        // Baseline=100, nadir=60 at day 28, progression at day 56 (tumour=72)
        let subject = create_test_subject(1, vec![100.0, 60.0, 72.0]);

        let result = compute_time_to_progression(&spec, &subject).unwrap();
        assert!(result.event, "Should have progression event");
        assert_eq!(
            result.time_days, 56.0,
            "Progression at day 56 (20% above nadir of 60)"
        );
    }

    #[test]
    fn test_pfs_censored() {
        let spec = EndpointSpec::TimeToProgression {
            observable: "TumourVol".to_string(),
            increase_fraction: 0.20,
            window_start_days: 0.0,
            window_end_days: 84.0,
            ref_baseline: false,
        };

        // No progression observed
        let subject = create_test_subject(1, vec![100.0, 80.0, 70.0]);

        let result = compute_time_to_progression(&spec, &subject).unwrap();
        assert!(!result.event, "Should be censored (no progression)");
        assert_eq!(result.time_days, 56.0, "Censored at last observation");
    }

    #[test]
    fn test_kaplan_meier_simple() {
        // Simple example: 4 subjects
        // Subject 1: event at t=10
        // Subject 2: censored at t=15
        // Subject 3: event at t=20
        // Subject 4: event at t=20
        let times = vec![10.0, 15.0, 20.0, 20.0];
        let events = vec![true, false, true, true];

        let (km_times, km_surv, km_n_risk, km_n_event) = kaplan_meier(&times, &events);

        assert_eq!(km_times.len(), 3, "Should have 3 unique time points");
        assert_eq!(km_times[0], 10.0);
        assert_eq!(km_times[1], 15.0);
        assert_eq!(km_times[2], 20.0);

        // At t=10: 1 event out of 4 at risk → S(10) = 1 * (1 - 1/4) = 0.75
        assert!((km_surv[0] - 0.75).abs() < 0.01, "S(10) should be 0.75");
        assert_eq!(km_n_risk[0], 4, "4 at risk at t=10");
        assert_eq!(km_n_event[0], 1, "1 event at t=10");

        // At t=15: 0 events (1 censored), S(15) = S(10) = 0.75
        assert!((km_surv[1] - 0.75).abs() < 0.01, "S(15) should be 0.75");
        assert_eq!(km_n_risk[1], 3, "3 at risk at t=15");
        assert_eq!(km_n_event[1], 0, "0 events at t=15");

        // At t=20: 2 events out of 2 at risk → S(20) = 0.75 * (1 - 2/2) = 0.0
        assert!((km_surv[2] - 0.0).abs() < 0.01, "S(20) should be 0.0");
        assert_eq!(km_n_risk[2], 2, "2 at risk at t=20");
        assert_eq!(km_n_event[2], 2, "2 events at t=20");
    }

    #[test]
    fn test_inclusion_criteria() {
        let subject = SubjectTrajectory {
            id: 1,
            times_days: vec![0.0],
            tumour_vol: vec![60.0],
            baseline_tumour: 60.0,
            covariates: SubjectCovariates {
                age_years: 55,
                ecog: 1,
                weight_kg: 70.0,
            },
        };

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

        assert!(
            passes_inclusion(&subject, &inclusion),
            "Subject should pass all criteria"
        );

        // Test failure cases
        let too_young = SubjectTrajectory {
            covariates: SubjectCovariates {
                age_years: 17,
                ..subject.covariates.clone()
            },
            ..subject.clone()
        };
        assert!(
            !passes_inclusion(&too_young, &inclusion),
            "Should fail age check"
        );

        let bad_ecog = SubjectTrajectory {
            covariates: SubjectCovariates {
                ecog: 3,
                ..subject.covariates.clone()
            },
            ..subject.clone()
        };
        assert!(
            !passes_inclusion(&bad_ecog, &inclusion),
            "Should fail ECOG check"
        );
    }
}
