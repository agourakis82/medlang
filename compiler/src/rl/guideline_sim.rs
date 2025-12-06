use crate::rl::core::RLEnv;
use crate::rl::env_dose_tox::DoseToxEnv;
use crate::rl::guideline_eval::GuidelinePolicy;
use crate::rl::train::PolicyEvalReport;
use crate::rl::DoseToxEnvConfig;
use anyhow::Result;

/// Simulate a dose guideline as a policy within a DoseToxEnv and collect outcome metrics.
pub fn simulate_dose_guideline_for_dose_tox(
    cfg: DoseToxEnvConfig,
    guideline: &GuidelinePolicy,
    n_episodes: usize,
) -> Result<PolicyEvalReport> {
    let mut env = DoseToxEnv::new(cfg);

    let mut total_reward = 0.0;
    let mut total_contract_violations = 0usize;
    let mut total_steps = 0usize;
    let mut total_response_events = 0usize;
    let mut total_grade3plus_tox = 0usize;
    let mut total_dose_reductions = 0usize;
    let mut total_dose_holds = 0usize;

    for _ep in 0..n_episodes {
        let mut state = env.reset()?;
        let mut ep_steps = 0usize;
        let mut prev_dose_mg = 0.0;

        loop {
            let action = guideline.act(&state);
            let step_res = env.step(action)?;

            total_reward += step_res.reward;
            total_contract_violations += step_res.info.contract_violations;
            ep_steps += 1;
            total_steps += 1;

            if step_res.info.response_event {
                total_response_events += 1;
            }
            if step_res.info.grade3plus_tox_event {
                total_grade3plus_tox += 1;
            }
            if let Some(dose) = step_res.info.dose_mg {
                if dose <= 0.0 {
                    total_dose_holds += 1;
                }
                if dose > 0.0 && dose < prev_dose_mg {
                    total_dose_reductions += 1;
                }
                prev_dose_mg = dose;
            }

            state = step_res.next_state;
            if step_res.done {
                break;
            }
        }
    }

    let avg_reward = if n_episodes > 0 {
        total_reward / n_episodes as f64
    } else {
        0.0
    };

    let avg_contracts = if n_episodes > 0 {
        total_contract_violations as f64 / n_episodes as f64
    } else {
        0.0
    };

    let avg_episode_length = if n_episodes > 0 {
        total_steps as f64 / n_episodes as f64
    } else {
        0.0
    };

    let response_rate = if total_steps > 0 {
        total_response_events as f64 / total_steps as f64
    } else {
        0.0
    };

    let tox_rate = if total_steps > 0 {
        total_grade3plus_tox as f64 / total_steps as f64
    } else {
        0.0
    };

    Ok(PolicyEvalReport {
        n_episodes,
        avg_reward,
        avg_contract_violations: avg_contracts,
        avg_episode_length,
        response_rate,
        tox_grade3plus_rate: tox_rate,
        avg_dose_reductions_per_episode: total_dose_reductions as f64 / n_episodes as f64,
        avg_dose_holds_per_episode: total_dose_holds as f64 / n_episodes as f64,
    })
}
