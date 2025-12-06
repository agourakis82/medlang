use crate::rl::core::RLEnv;
use crate::rl::env_dose_tox::DoseToxEnv;
use crate::rl::guideline_eval::GuidelinePolicy;
use crate::rl::{DoseGuidelineIRHost, DoseToxEnvConfig};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Per-episode outcomes collected during simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeOutcome {
    // Baseline features for stratification
    pub baseline_anc: f64,
    pub baseline_tumour_ratio: f64,

    // Best tumour response (lower is better)
    pub best_tumour_ratio: f64,

    // Safety outcomes
    pub any_grade3plus_tox: bool,
    pub any_grade4plus_tox: bool,
    pub any_contract_violation: bool,

    // Dose delivery
    pub total_dose_mg: f64,
    pub full_dose_mg: f64,
    pub n_cycles: usize,
}

/// Configuration for outcome simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseGuidelineOutcomeConfig {
    pub n_episodes: usize,
    pub response_tumour_ratio_threshold: f64,
    pub grade3_threshold: u8,
    pub grade4_threshold: u8,
}

/// Aggregated outcome summary across episodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseGuidelineOutcomeSummary {
    pub n_episodes: usize,
    pub response_rate: f64,
    pub mean_best_tumour_ratio: f64,
    pub grade3plus_rate: f64,
    pub grade4plus_rate: f64,
    pub contract_violation_rate: f64,
    pub mean_rdi: f64,
}

/// Stratification variable kinds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StratVarKind {
    BaselineAnc,
    BaselineTumourRatio,
}

/// A bin over a stratification variable: interval [lower, upper).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratBin {
    pub label: String,
    pub lower: f64,
    pub upper: f64,
}

/// Stratifier configuration: variable + bins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratifierConfig {
    pub var: StratVarKind,
    pub bins: Vec<StratBin>,
}

/// Outcome summary for a single stratum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StratumOutcomeSummary {
    pub label: String,
    pub n_episodes: usize,
    pub response_rate: f64,
    pub mean_best_tumour_ratio: f64,
    pub grade3plus_rate: f64,
    pub grade4plus_rate: f64,
    pub contract_violation_rate: f64,
    pub mean_rdi: f64,
}

/// Stratified outcome report for a guideline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DoseGuidelineStratifiedOutcomeReport {
    pub guideline_name: String,
    pub outcome_config: DoseGuidelineOutcomeConfig,
    pub stratifier: StratifierConfig,
    pub overall: DoseGuidelineOutcomeSummary,
    pub strata: Vec<StratumOutcomeSummary>,
}

/// Simulate a guideline and return per-episode outcomes.
pub fn simulate_dose_guideline_episodes(
    env_cfg: &DoseToxEnvConfig,
    guideline: &DoseGuidelineIRHost,
    cfg: &DoseGuidelineOutcomeConfig,
) -> Result<Vec<EpisodeOutcome>> {
    let policy = GuidelinePolicy {
        guideline: guideline.clone(),
    };

    // Full dose is the maximum non-zero dose level (used for RDI)
    let full_dose_mg = env_cfg
        .dose_levels_mg
        .iter()
        .copied()
        .filter(|d| *d > 0.0)
        .fold(0.0, f64::max);

    let mut episodes = Vec::with_capacity(cfg.n_episodes);
    let mut env = DoseToxEnv::new(env_cfg.clone());

    for _ in 0..cfg.n_episodes {
        let mut state = env.reset()?;

        let baseline_anc = state.features.get(0).copied().unwrap_or(0.0);
        let baseline_tr = state.features.get(1).copied().unwrap_or(1.0);
        let mut best_tumour_ratio = baseline_tr;

        let mut any_g3 = false;
        let mut any_g4 = false;
        let mut any_contract = false;
        let mut total_dose_mg = 0.0;
        let mut n_cycles = 0usize;

        loop {
            let action = policy.act(&state);
            let dose_mg = env_cfg.dose_levels_mg.get(action).copied().unwrap_or(0.0);

            let step_res = env.step(action)?;

            // Update aggregates
            total_dose_mg += dose_mg;
            n_cycles += 1;
            best_tumour_ratio = best_tumour_ratio.min(
                step_res
                    .next_state
                    .features
                    .get(1)
                    .copied()
                    .unwrap_or(best_tumour_ratio),
            );

            if let Some(grade) = step_res.info.toxicity_grade {
                if grade >= cfg.grade3_threshold {
                    any_g3 = true;
                }
                if grade >= cfg.grade4_threshold {
                    any_g4 = true;
                }
            }
            if step_res.info.contract_violations > 0 {
                any_contract = true;
            }

            state = step_res.next_state;

            if step_res.done {
                break;
            }
        }

        episodes.push(EpisodeOutcome {
            baseline_anc,
            baseline_tumour_ratio: baseline_tr,
            best_tumour_ratio,
            any_grade3plus_tox: any_g3,
            any_grade4plus_tox: any_g4,
            any_contract_violation: any_contract,
            total_dose_mg,
            full_dose_mg,
            n_cycles,
        });
    }

    Ok(episodes)
}

/// Aggregate per-episode outcomes into a summary.
pub fn aggregate_outcomes(
    episodes: &[EpisodeOutcome],
    cfg: &DoseGuidelineOutcomeConfig,
) -> DoseGuidelineOutcomeSummary {
    if episodes.is_empty() {
        return DoseGuidelineOutcomeSummary {
            n_episodes: 0,
            response_rate: 0.0,
            mean_best_tumour_ratio: 0.0,
            grade3plus_rate: 0.0,
            grade4plus_rate: 0.0,
            contract_violation_rate: 0.0,
            mean_rdi: 0.0,
        };
    }

    let n = episodes.len();
    let mut response = 0usize;
    let mut sum_best_tr = 0.0;
    let mut g3 = 0usize;
    let mut g4 = 0usize;
    let mut contracts = 0usize;
    let mut rdi_sum = 0.0;

    for ep in episodes {
        if ep.best_tumour_ratio <= cfg.response_tumour_ratio_threshold {
            response += 1;
        }
        sum_best_tr += ep.best_tumour_ratio;
        if ep.any_grade3plus_tox {
            g3 += 1;
        }
        if ep.any_grade4plus_tox {
            g4 += 1;
        }
        if ep.any_contract_violation {
            contracts += 1;
        }

        let denom = ep.full_dose_mg * ep.n_cycles as f64;
        if denom > 0.0 {
            rdi_sum += ep.total_dose_mg / denom;
        }
    }

    DoseGuidelineOutcomeSummary {
        n_episodes: episodes.len(),
        response_rate: response as f64 / n as f64,
        mean_best_tumour_ratio: sum_best_tr / n as f64,
        grade3plus_rate: g3 as f64 / n as f64,
        grade4plus_rate: g4 as f64 / n as f64,
        contract_violation_rate: contracts as f64 / n as f64,
        mean_rdi: rdi_sum / n as f64,
    }
}

/// Simulate outcomes (overall only).
pub fn simulate_dose_guideline_outcomes(
    env_cfg: &DoseToxEnvConfig,
    guideline: &DoseGuidelineIRHost,
    cfg: &DoseGuidelineOutcomeConfig,
) -> Result<DoseGuidelineOutcomeSummary> {
    let episodes = simulate_dose_guideline_episodes(env_cfg, guideline, cfg)?;
    Ok(aggregate_outcomes(&episodes, cfg))
}

fn strat_value(ep: &EpisodeOutcome, var: &StratVarKind) -> f64 {
    match var {
        StratVarKind::BaselineAnc => ep.baseline_anc,
        StratVarKind::BaselineTumourRatio => ep.baseline_tumour_ratio,
    }
}

fn find_bin_index(val: f64, strat: &StratifierConfig) -> Option<usize> {
    for (i, b) in strat.bins.iter().enumerate() {
        if val >= b.lower && val < b.upper {
            return Some(i);
        }
    }
    None
}

fn aggregate_strata(
    episodes: &[EpisodeOutcome],
    cfg: &DoseGuidelineOutcomeConfig,
    strat: &StratifierConfig,
) -> Vec<StratumOutcomeSummary> {
    let mut grouped: Vec<Vec<EpisodeOutcome>> = vec![Vec::new(); strat.bins.len()];

    for ep in episodes {
        if let Some(idx) = find_bin_index(strat_value(ep, &strat.var), strat) {
            grouped[idx].push(ep.clone());
        }
    }

    strat
        .bins
        .iter()
        .zip(grouped.into_iter())
        .map(|(bin, eps)| {
            let summary = if eps.is_empty() {
                DoseGuidelineOutcomeSummary {
                    n_episodes: 0,
                    response_rate: 0.0,
                    mean_best_tumour_ratio: 1.0,
                    grade3plus_rate: 0.0,
                    grade4plus_rate: 0.0,
                    contract_violation_rate: 0.0,
                    mean_rdi: 0.0,
                }
            } else {
                aggregate_outcomes(&eps, cfg)
            };

            StratumOutcomeSummary {
                label: bin.label.clone(),
                n_episodes: summary.n_episodes,
                response_rate: summary.response_rate,
                mean_best_tumour_ratio: summary.mean_best_tumour_ratio,
                grade3plus_rate: summary.grade3plus_rate,
                grade4plus_rate: summary.grade4plus_rate,
                contract_violation_rate: summary.contract_violation_rate,
                mean_rdi: summary.mean_rdi,
            }
        })
        .collect()
}

/// Simulate outcomes with stratification.
pub fn simulate_dose_guideline_outcomes_stratified(
    env_cfg: &DoseToxEnvConfig,
    guideline: &DoseGuidelineIRHost,
    cfg: &DoseGuidelineOutcomeConfig,
    strat: &StratifierConfig,
) -> Result<DoseGuidelineStratifiedOutcomeReport> {
    let episodes = simulate_dose_guideline_episodes(env_cfg, guideline, cfg)?;
    let overall = aggregate_outcomes(&episodes, cfg);
    let strata = aggregate_strata(&episodes, cfg, strat);

    Ok(DoseGuidelineStratifiedOutcomeReport {
        guideline_name: guideline.name.clone(),
        outcome_config: cfg.clone(),
        stratifier: strat.clone(),
        overall,
        strata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_bin_index() {
        let strat = StratifierConfig {
            var: StratVarKind::BaselineAnc,
            bins: vec![
                StratBin {
                    label: "low".into(),
                    lower: 0.0,
                    upper: 1.0,
                },
                StratBin {
                    label: "mid".into(),
                    lower: 1.0,
                    upper: 2.0,
                },
            ],
        };

        assert_eq!(find_bin_index(0.5, &strat), Some(0));
        assert_eq!(find_bin_index(1.5, &strat), Some(1));
        assert_eq!(find_bin_index(2.5, &strat), None);
    }

    #[test]
    fn test_aggregate_outcomes_basic() {
        let cfg = DoseGuidelineOutcomeConfig {
            n_episodes: 2,
            response_tumour_ratio_threshold: 0.8,
            grade3_threshold: 3,
            grade4_threshold: 4,
        };

        let eps = vec![
            EpisodeOutcome {
                baseline_anc: 1.0,
                baseline_tumour_ratio: 1.0,
                best_tumour_ratio: 0.7,
                any_grade3plus_tox: true,
                any_grade4plus_tox: false,
                any_contract_violation: false,
                total_dose_mg: 100.0,
                full_dose_mg: 100.0,
                n_cycles: 1,
            },
            EpisodeOutcome {
                baseline_anc: 1.2,
                baseline_tumour_ratio: 1.1,
                best_tumour_ratio: 0.9,
                any_grade3plus_tox: false,
                any_grade4plus_tox: false,
                any_contract_violation: true,
                total_dose_mg: 50.0,
                full_dose_mg: 100.0,
                n_cycles: 1,
            },
        ];

        let summary = aggregate_outcomes(&eps, &cfg);
        assert_eq!(summary.n_episodes, 2);
        assert!((summary.response_rate - 0.5).abs() < 1e-6);
        assert!((summary.grade3plus_rate - 0.5).abs() < 1e-6);
        assert!((summary.grade4plus_rate - 0.0).abs() < 1e-6);
        assert!((summary.contract_violation_rate - 0.5).abs() < 1e-6);
        assert!((summary.mean_rdi - 0.75).abs() < 1e-6);
    }
}
