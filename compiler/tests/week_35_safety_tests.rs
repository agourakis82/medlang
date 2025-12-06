// Week 35: Basic RL Safety Analysis Tests
//
// Tests for PolicySafetyConfig, PolicySafetyReport, and check_policy_safety function

use medlangc::ml::BackendKind;
use medlangc::rl::{
    check_policy_safety, DoseToxEnv, DoseToxEnvConfig, PolicySafetyConfig, PolicySafetyReport,
    RLPolicyHandle, RLTrainConfig, SafetyViolationKind,
};
use medlangc::rl::{train_q_learning, BoxDiscretizer};

#[test]
fn test_policy_safety_config_creation() {
    let cfg = PolicySafetyConfig {
        n_episodes: 50,
        max_steps_per_episode: 10,
        max_dose_mg: Some(300.0),
        max_delta_dose_mg: Some(100.0),
        max_severe_toxicity_episodes: Some(5),
        max_total_contract_violations: Some(20),
        use_guideline_gate: false,
        guideline_name: None,
        seed: Some(12345),
    };

    assert_eq!(cfg.n_episodes, 50);
    assert_eq!(cfg.max_steps_per_episode, 10);
    assert_eq!(cfg.max_dose_mg, Some(300.0));
    assert_eq!(cfg.max_delta_dose_mg, Some(100.0));
}

#[test]
fn test_policy_safety_report_creation() {
    let report = PolicySafetyReport::new(100);

    assert_eq!(report.n_episodes, 100);
    assert_eq!(report.n_episodes_evaluated, 0);
    assert_eq!(report.total_contract_violations, 0);
    assert_eq!(report.total_severe_toxicity_events, 0);
    assert!(report.safety_pass);
    assert!(report.sample_violations.is_empty());
}

#[test]
fn test_safety_pass_threshold_severe_toxicity() {
    let mut report = PolicySafetyReport::new(100);
    report.episodes_with_severe_toxicity = 15;

    let cfg = PolicySafetyConfig {
        n_episodes: 100,
        max_steps_per_episode: 10,
        max_dose_mg: None,
        max_delta_dose_mg: None,
        max_severe_toxicity_episodes: Some(10),
        max_total_contract_violations: None,
        use_guideline_gate: false,
        guideline_name: None,
        seed: None,
    };

    report.check_safety_pass(&cfg);
    assert!(!report.safety_pass);
}

#[test]
fn test_safety_pass_threshold_contracts() {
    let mut report = PolicySafetyReport::new(100);
    report.total_contract_violations = 75;

    let cfg = PolicySafetyConfig {
        n_episodes: 100,
        max_steps_per_episode: 10,
        max_dose_mg: None,
        max_delta_dose_mg: None,
        max_severe_toxicity_episodes: None,
        max_total_contract_violations: Some(50),
        use_guideline_gate: false,
        guideline_name: None,
        seed: None,
    };

    report.check_safety_pass(&cfg);
    assert!(!report.safety_pass);
}

#[test]
fn test_check_policy_safety_basic() {
    // Create a simple environment
    let env_cfg = DoseToxEnvConfig {
        ev_handle: "test_ev".to_string(),
        backend: BackendKind::Surrogate,
        n_cycles: 4,
        dose_levels_mg: vec![0.0, 100.0, 200.0],
        reward_response_weight: 1.0,
        reward_tox_penalty: 2.0,
        contract_penalty: 10.0,
        seed: Some(42),
    };

    let mut env = DoseToxEnv::new(env_cfg.clone());

    // Train a quick policy
    let train_cfg = RLTrainConfig {
        n_episodes: 20,
        max_steps_per_episode: 4,
        gamma: 0.95,
        alpha: 0.1,
        eps_start: 0.5,
        eps_end: 0.1,
    };

    let disc = BoxDiscretizer::new(
        vec![4, 4, 4, 3],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
    );

    let (policy, _report) = train_q_learning(&mut env, &disc, &train_cfg, 42).unwrap();

    // Run safety analysis
    let safety_cfg = PolicySafetyConfig {
        n_episodes: 10,
        max_steps_per_episode: 4,
        max_dose_mg: Some(250.0),
        max_delta_dose_mg: Some(150.0),
        max_severe_toxicity_episodes: Some(5),
        max_total_contract_violations: Some(20),
        use_guideline_gate: false,
        guideline_name: None,
        seed: Some(123),
    };

    let mut test_env = DoseToxEnv::new(env_cfg);
    let report = check_policy_safety(&mut test_env, &policy, &safety_cfg).unwrap();

    // Verify report structure
    assert_eq!(report.n_episodes, 10);
    assert!(report.n_episodes_evaluated > 0);
    assert!(report.n_episodes_evaluated <= 10);

    // Should have some metrics
    println!("Safety Report:");
    println!("  Episodes evaluated: {}", report.n_episodes_evaluated);
    println!("  Avg reward: {:.3}", report.avg_reward);
    println!(
        "  Severe toxicity episodes: {}",
        report.episodes_with_severe_toxicity
    );
    println!(
        "  Contract violations: {}",
        report.total_contract_violations
    );
    println!("  Safety pass: {}", report.safety_pass);
    println!("  Sample violations: {}", report.sample_violations.len());
}

#[test]
fn test_unsafe_policy_detection() {
    // Create environment
    let env_cfg = DoseToxEnvConfig {
        ev_handle: "test_ev".to_string(),
        backend: BackendKind::Surrogate,
        n_cycles: 3,
        dose_levels_mg: vec![0.0, 100.0, 300.0], // High max dose
        reward_response_weight: 1.0,
        reward_tox_penalty: 1.0, // Low toxicity penalty
        contract_penalty: 1.0,   // Low contract penalty
        seed: Some(42),
    };

    let mut env = DoseToxEnv::new(env_cfg.clone());

    // Create an "aggressive" policy (always max dose)
    let n_states = 4 * 4 * 3 * 3;
    let n_actions = 3;
    let mut q_values = vec![0.0; n_states * n_actions];

    // Set Q-values to prefer highest dose (action 2)
    for state in 0..n_states {
        for action in 0..n_actions {
            if action == 2 {
                q_values[state * n_actions + action] = 100.0; // High Q for max dose
            } else {
                q_values[state * n_actions + action] = 0.0;
            }
        }
    }

    let aggressive_policy = RLPolicyHandle {
        n_states,
        n_actions,
        q_values,
        bins_per_dim: vec![4, 4, 3, 3],
        min_vals: vec![0.0, 0.0, 0.0, 0.0],
        max_vals: vec![1.0, 1.0, 1.0, 1.0],
    };

    // Run safety analysis with strict thresholds
    let safety_cfg = PolicySafetyConfig {
        n_episodes: 20,
        max_steps_per_episode: 3,
        max_dose_mg: None,
        max_delta_dose_mg: None,
        max_severe_toxicity_episodes: Some(2),  // Very strict
        max_total_contract_violations: Some(5), // Very strict
        use_guideline_gate: false,
        guideline_name: None,
        seed: Some(123),
    };

    let mut test_env = DoseToxEnv::new(env_cfg);
    let report = check_policy_safety(&mut test_env, &aggressive_policy, &safety_cfg).unwrap();

    println!("\nAggressive Policy Safety Report:");
    println!("  Episodes evaluated: {}", report.n_episodes_evaluated);
    println!(
        "  Severe toxicity episodes: {}",
        report.episodes_with_severe_toxicity
    );
    println!(
        "  Total contract violations: {}",
        report.total_contract_violations
    );
    println!("  Safety pass: {}", report.safety_pass);

    // Aggressive policy should trigger safety violations
    assert!(
        report.episodes_with_severe_toxicity > 0 || report.total_contract_violations > 0,
        "Aggressive policy should trigger some safety events"
    );
}

#[test]
fn test_dose_limit_violations() {
    // Create environment with high doses
    let env_cfg = DoseToxEnvConfig {
        ev_handle: "test_ev".to_string(),
        backend: BackendKind::Surrogate,
        n_cycles: 3,
        dose_levels_mg: vec![0.0, 100.0, 200.0, 400.0], // 400mg exceeds limit
        reward_response_weight: 1.0,
        reward_tox_penalty: 2.0,
        contract_penalty: 10.0,
        seed: Some(42),
    };

    let mut env = DoseToxEnv::new(env_cfg.clone());

    // Create policy that prefers high doses
    let n_states = 4 * 4 * 3 * 4;
    let n_actions = 4;
    let mut q_values = vec![0.0; n_states * n_actions];

    // Prefer action 3 (400mg dose)
    for state in 0..n_states {
        for action in 0..n_actions {
            q_values[state * n_actions + action] = if action == 3 { 50.0 } else { 0.0 };
        }
    }

    let high_dose_policy = RLPolicyHandle {
        n_states,
        n_actions,
        q_values,
        bins_per_dim: vec![4, 4, 3, 4],
        min_vals: vec![0.0, 0.0, 0.0, 0.0],
        max_vals: vec![1.0, 1.0, 1.0, 1.0],
    };

    // Set max_dose_mg to 300 (should catch 400mg doses)
    let safety_cfg = PolicySafetyConfig {
        n_episodes: 10,
        max_steps_per_episode: 3,
        max_dose_mg: Some(300.0),
        max_delta_dose_mg: None,
        max_severe_toxicity_episodes: None,
        max_total_contract_violations: None,
        use_guideline_gate: false,
        guideline_name: None,
        seed: Some(123),
    };

    let mut test_env = DoseToxEnv::new(env_cfg);
    let report = check_policy_safety(&mut test_env, &high_dose_policy, &safety_cfg).unwrap();

    println!("\nDose Limit Test:");
    println!("  Dose out of range: {}", report.total_dose_out_of_range);
    println!(
        "  Episodes with violations: {}",
        report.episodes_with_any_violation
    );

    // Should detect dose limit violations
    assert!(
        report.total_dose_out_of_range > 0,
        "Should detect doses exceeding 300mg limit"
    );
}
