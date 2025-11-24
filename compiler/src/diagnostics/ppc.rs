use crate::diagnostics::DiagnosticsError;
use crate::endpoints::SubjectTrajectory;
use serde::{Deserialize, Serialize};

/// Posterior predictive check results for tumor volumes at a specific visit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpcTumourPerVisit {
    pub arm_name: String,
    pub visit_time: f64,
    pub n_obs: usize,
    pub obs_mean: f64,
    pub obs_sd: f64,
    pub pred_mean: f64,
    pub pred_sd: f64,
    pub pred_p05: f64,
    pub pred_p50: f64,
    pub pred_p95: f64,
    pub z_score: f64, // How many SDs is obs_mean from pred_mean?
}

/// Posterior predictive check results for arm-level endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpcEndpoint {
    pub arm_name: String,
    pub endpoint_name: String, // "ORR" or "PFS_12m"
    pub obs_value: f64,
    pub pred_mean: f64,
    pub pred_sd: f64,
    pub pred_p05: f64,
    pub pred_p50: f64,
    pub pred_p95: f64,
    pub in_credible_interval: bool, // Is obs_value in [p05, p95]?
}

/// Complete posterior predictive check report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpcReport {
    pub tumour_per_visit: Vec<PpcTumourPerVisit>,
    pub endpoints: Vec<PpcEndpoint>,
}

/// Compute posterior predictive checks comparing observed data to model predictions
///
/// # Arguments
/// * `arm_name` - Name of the treatment arm being analyzed
/// * `observed_trajectories` - The observed subject trajectories
/// * `predicted_trajectories` - Vector of predicted subject trajectories, one per posterior draw
///
/// # Returns
/// A PpcReport containing trajectory and endpoint comparisons
pub fn posterior_predictive_checks(
    arm_name: &str,
    observed_trajectories: &[SubjectTrajectory],
    predicted_trajectories: &[Vec<SubjectTrajectory>],
) -> Result<PpcReport, DiagnosticsError> {
    if predicted_trajectories.is_empty() {
        return Err(DiagnosticsError::IoError(
            "No predicted trajectories provided".to_string(),
        ));
    }

    let tumour_per_visit =
        compute_tumour_ppc(arm_name, observed_trajectories, predicted_trajectories)?;
    let endpoints = compute_endpoint_ppc(arm_name, observed_trajectories, predicted_trajectories)?;

    Ok(PpcReport {
        tumour_per_visit,
        endpoints,
    })
}

/// Compute PPC for tumor volumes at each visit
fn compute_tumour_ppc(
    arm_name: &str,
    observed: &[SubjectTrajectory],
    predicted: &[Vec<SubjectTrajectory>],
) -> Result<Vec<PpcTumourPerVisit>, DiagnosticsError> {
    let mut results = Vec::new();

    // Collect all unique visit times from observed data (rounded to nearest 0.1)
    let mut visit_times = std::collections::HashSet::new();
    for subj in observed {
        for &time in &subj.times_days {
            let rounded = (time * 10.0).round() / 10.0;
            visit_times.insert(OrderedFloat(rounded));
        }
    }
    let mut visit_times: Vec<f64> = visit_times.into_iter().map(|of| of.0).collect();
    visit_times.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // For each visit time
    for visit_time in visit_times {
        let tolerance = 0.05; // 5% tolerance for time matching

        // Collect observed volumes at this visit
        let obs_volumes: Vec<f64> = observed
            .iter()
            .flat_map(|subj| {
                subj.times_days
                    .iter()
                    .zip(&subj.tumour_vol)
                    .filter(|(t, _)| (**t - visit_time).abs() <= tolerance)
                    .map(|(_, v)| *v)
            })
            .collect();

        if obs_volumes.is_empty() {
            continue;
        }

        let n_obs = obs_volumes.len();
        let obs_mean = obs_volumes.iter().sum::<f64>() / n_obs as f64;
        let obs_sd = if n_obs > 1 {
            let variance = obs_volumes
                .iter()
                .map(|v| (v - obs_mean).powi(2))
                .sum::<f64>()
                / (n_obs - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        // Collect predicted means from each posterior draw
        let mut pred_means = Vec::new();
        for pred_draw in predicted {
            let pred_volumes: Vec<f64> = pred_draw
                .iter()
                .flat_map(|subj| {
                    subj.times_days
                        .iter()
                        .zip(&subj.tumour_vol)
                        .filter(|(t, _)| (**t - visit_time).abs() <= tolerance)
                        .map(|(_, v)| *v)
                })
                .collect();

            if !pred_volumes.is_empty() {
                let pred_mean_this_draw =
                    pred_volumes.iter().sum::<f64>() / pred_volumes.len() as f64;
                pred_means.push(pred_mean_this_draw);
            }
        }

        if pred_means.is_empty() {
            continue;
        }

        // Compute summary statistics of predicted means
        pred_means.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let pred_mean = pred_means.iter().sum::<f64>() / pred_means.len() as f64;
        let pred_sd = if pred_means.len() > 1 {
            let variance = pred_means
                .iter()
                .map(|v| (v - pred_mean).powi(2))
                .sum::<f64>()
                / (pred_means.len() - 1) as f64;
            variance.sqrt()
        } else {
            0.0
        };

        let pred_p05 = compute_quantile(&pred_means, 0.05);
        let pred_p50 = compute_quantile(&pred_means, 0.50);
        let pred_p95 = compute_quantile(&pred_means, 0.95);

        let z_score = if pred_sd > 0.0 {
            (obs_mean - pred_mean) / pred_sd
        } else {
            0.0
        };

        results.push(PpcTumourPerVisit {
            arm_name: arm_name.to_string(),
            visit_time,
            n_obs,
            obs_mean,
            obs_sd,
            pred_mean,
            pred_sd,
            pred_p05,
            pred_p50,
            pred_p95,
            z_score,
        });
    }

    Ok(results)
}

/// Compute PPC for arm-level endpoints (ORR, PFS)
fn compute_endpoint_ppc(
    arm_name: &str,
    observed: &[SubjectTrajectory],
    predicted: &[Vec<SubjectTrajectory>],
) -> Result<Vec<PpcEndpoint>, DiagnosticsError> {
    let mut results = Vec::new();

    // Compute observed endpoints
    let obs_orr = compute_orr(observed);
    let obs_pfs = compute_pfs_12m(observed);

    // Collect predicted endpoints from each posterior draw
    let mut pred_orr_values = Vec::new();
    let mut pred_pfs_values = Vec::new();

    for pred_draw in predicted {
        pred_orr_values.push(compute_orr(pred_draw));
        pred_pfs_values.push(compute_pfs_12m(pred_draw));
    }

    // ORR PPC
    if !pred_orr_values.is_empty() {
        let orr_ppc = compute_endpoint_stats(arm_name, "ORR", obs_orr, &pred_orr_values);
        results.push(orr_ppc);
    }

    // PFS PPC
    if !pred_pfs_values.is_empty() {
        let pfs_ppc = compute_endpoint_stats(arm_name, "PFS_12m", obs_pfs, &pred_pfs_values);
        results.push(pfs_ppc);
    }

    Ok(results)
}

/// Compute endpoint PPC statistics
fn compute_endpoint_stats(
    arm_name: &str,
    endpoint_name: &str,
    obs_value: f64,
    pred_values: &[f64],
) -> PpcEndpoint {
    let mut sorted = pred_values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let pred_mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let pred_sd = if sorted.len() > 1 {
        let variance =
            sorted.iter().map(|v| (v - pred_mean).powi(2)).sum::<f64>() / (sorted.len() - 1) as f64;
        variance.sqrt()
    } else {
        0.0
    };

    let pred_p05 = compute_quantile(&sorted, 0.05);
    let pred_p50 = compute_quantile(&sorted, 0.50);
    let pred_p95 = compute_quantile(&sorted, 0.95);

    let in_credible_interval = obs_value >= pred_p05 && obs_value <= pred_p95;

    PpcEndpoint {
        arm_name: arm_name.to_string(),
        endpoint_name: endpoint_name.to_string(),
        obs_value,
        pred_mean,
        pred_sd,
        pred_p05,
        pred_p50,
        pred_p95,
        in_credible_interval,
    }
}

/// Compute objective response rate (ORR) as proportion with >=30% reduction
fn compute_orr(trajectories: &[SubjectTrajectory]) -> f64 {
    if trajectories.is_empty() {
        return 0.0;
    }

    let responders = trajectories
        .iter()
        .filter(|s| {
            if s.tumour_vol.is_empty() {
                return false;
            }
            let baseline = s.baseline_tumour;
            if baseline <= 0.0 {
                return false;
            }
            let min_volume = s
                .tumour_vol
                .iter()
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            let reduction = (baseline - min_volume) / baseline;
            reduction >= 0.30
        })
        .count();

    responders as f64 / trajectories.len() as f64
}

/// Compute 12-month PFS rate (proportion alive and progression-free at 12 months)
fn compute_pfs_12m(trajectories: &[SubjectTrajectory]) -> f64 {
    if trajectories.is_empty() {
        return 0.0;
    }

    let pfs_at_12m = trajectories
        .iter()
        .filter(|s| {
            // Check if subject was progression-free at 12 months
            if s.tumour_vol.is_empty() {
                return false;
            }

            // Find observation closest to 12 months (360 days)
            let target_time = 360.0;
            let mut closest_idx = 0;
            let mut min_distance = f64::MAX;

            for (i, &time) in s.times_days.iter().enumerate() {
                let distance = (time - target_time).abs();
                if distance < min_distance {
                    min_distance = distance;
                    closest_idx = i;
                }
            }

            // Only consider if observation is within 30 days of 12 months
            if min_distance > 30.0 {
                return false;
            }

            // Check if tumor didn't grow >20% from baseline
            let baseline = s.baseline_tumour;
            if baseline <= 0.0 {
                return false;
            }
            let volume_at_12m = s.tumour_vol[closest_idx];
            let growth = (volume_at_12m - baseline) / baseline;
            growth < 0.20
        })
        .count();

    pfs_at_12m as f64 / trajectories.len() as f64
}

/// Compute quantile from sorted data
fn compute_quantile(sorted_data: &[f64], p: f64) -> f64 {
    if sorted_data.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted_data.len() - 1) as f64).round() as usize;
    sorted_data[idx.min(sorted_data.len() - 1)]
}

/// Helper struct for using f64 in HashSet (required for visit time collection)
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedFloat(f64);

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endpoints::SubjectCovariates;

    fn create_test_trajectory(id: usize, times: Vec<f64>, volumes: Vec<f64>) -> SubjectTrajectory {
        let baseline = volumes[0];
        SubjectTrajectory {
            id,
            times_days: times,
            tumour_vol: volumes,
            baseline_tumour: baseline,
            covariates: SubjectCovariates {
                age_years: 60,
                ecog: 0,
                weight_kg: 70.0,
            },
        }
    }

    #[test]
    fn test_compute_orr() {
        // Create trajectories with 2 responders out of 3
        let trajectories = vec![
            create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 60.0, 50.0]), // 50% reduction -> responder
            create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 80.0, 75.0]), // 25% reduction -> non-responder
            create_test_trajectory(3, vec![0.0, 30.0, 60.0], vec![100.0, 65.0, 60.0]), // 40% reduction -> responder
        ];

        let orr = compute_orr(&trajectories);
        assert!(
            (orr - 0.6667).abs() < 0.01,
            "ORR should be ~66.67%, got {}",
            orr
        );
    }

    #[test]
    fn test_compute_pfs_12m() {
        // Create trajectories where 2 out of 3 are progression-free at month 12
        let trajectories = vec![
            create_test_trajectory(
                1,
                vec![0.0, 90.0, 180.0, 360.0],
                vec![100.0, 105.0, 110.0, 115.0],
            ), // 15% growth -> PFS
            create_test_trajectory(
                2,
                vec![0.0, 90.0, 180.0, 360.0],
                vec![100.0, 110.0, 125.0, 140.0],
            ), // 40% growth -> progressed
            create_test_trajectory(
                3,
                vec![0.0, 90.0, 180.0, 360.0],
                vec![100.0, 95.0, 90.0, 85.0],
            ), // shrinking -> PFS
        ];

        let pfs = compute_pfs_12m(&trajectories);
        assert!(
            (pfs - 0.6667).abs() < 0.01,
            "PFS should be ~66.67%, got {}",
            pfs
        );
    }

    #[test]
    fn test_tumour_ppc() {
        let observed = vec![
            create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 90.0, 80.0]),
            create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 95.0, 85.0]),
        ];

        // Create 3 posterior draws with similar patterns
        let predicted = vec![
            vec![
                create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 88.0, 78.0]),
                create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 92.0, 82.0]),
            ],
            vec![
                create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 91.0, 81.0]),
                create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 96.0, 86.0]),
            ],
            vec![
                create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 89.0, 79.0]),
                create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 94.0, 84.0]),
            ],
        ];

        let result = compute_tumour_ppc("Arm A", &observed, &predicted).unwrap();

        // Should have results for 3 visits (t=0, t=30, t=60)
        assert_eq!(result.len(), 3, "Should have 3 time points");

        // Check baseline (t=0)
        let t0_ppc = result.iter().find(|r| r.visit_time == 0.0).unwrap();
        assert_eq!(t0_ppc.n_obs, 2);
        assert!((t0_ppc.obs_mean - 100.0).abs() < 0.1);
        assert!((t0_ppc.pred_mean - 100.0).abs() < 1.0);
    }

    #[test]
    fn test_endpoint_ppc() {
        let observed = vec![
            create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 60.0, 50.0]), // responder
            create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 80.0, 75.0]), // non-responder
            create_test_trajectory(3, vec![0.0, 30.0, 60.0], vec![100.0, 65.0, 60.0]), // responder
        ];

        let predicted = vec![
            vec![
                create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 62.0, 52.0]),
                create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 82.0, 77.0]),
                create_test_trajectory(3, vec![0.0, 30.0, 60.0], vec![100.0, 67.0, 62.0]),
            ],
            vec![
                create_test_trajectory(1, vec![0.0, 30.0, 60.0], vec![100.0, 58.0, 48.0]),
                create_test_trajectory(2, vec![0.0, 30.0, 60.0], vec![100.0, 78.0, 73.0]),
                create_test_trajectory(3, vec![0.0, 30.0, 60.0], vec![100.0, 63.0, 58.0]),
            ],
        ];

        let result = compute_endpoint_ppc("Arm A", &observed, &predicted).unwrap();

        // Should have ORR and PFS results
        assert_eq!(result.len(), 2);

        let orr_ppc = result.iter().find(|r| r.endpoint_name == "ORR").unwrap();
        assert!(
            (orr_ppc.obs_value - 0.6667).abs() < 0.01,
            "Expected ORR ~66.67%, got {}",
            orr_ppc.obs_value
        );
        assert!(orr_ppc.pred_mean > 0.0);
    }

    #[test]
    fn test_posterior_predictive_checks() {
        let observed = vec![
            create_test_trajectory(1, vec![0.0, 30.0], vec![100.0, 90.0]),
            create_test_trajectory(2, vec![0.0, 30.0], vec![100.0, 95.0]),
        ];

        let predicted = vec![
            vec![
                create_test_trajectory(1, vec![0.0, 30.0], vec![100.0, 88.0]),
                create_test_trajectory(2, vec![0.0, 30.0], vec![100.0, 92.0]),
            ],
            vec![
                create_test_trajectory(1, vec![0.0, 30.0], vec![100.0, 91.0]),
                create_test_trajectory(2, vec![0.0, 30.0], vec![100.0, 96.0]),
            ],
        ];

        let report = posterior_predictive_checks("Arm A", &observed, &predicted).unwrap();

        assert!(!report.tumour_per_visit.is_empty());
        assert!(!report.endpoints.is_empty());
    }
}
