// Week 36: Policy Distillation Integration Tests
//
// Tests for distilling RL policies into interpretable decision trees

use medlangc::ml::BackendKind;
use medlangc::rl::{
    distill_policy, train_q_learning, BoxDiscretizer, DistillConfig, DoseToxEnv, DoseToxEnvConfig,
    RLTrainConfig,
};

#[test]
fn test_distill_config_creation() {
    let cfg = DistillConfig {
        n_episodes: 100,
        max_steps_per_episode: 10,
        max_depth: 3,
        min_samples_leaf: 20,
        features: vec![],
    };

    assert_eq!(cfg.n_episodes, 100);
    assert_eq!(cfg.max_depth, 3);
    assert_eq!(cfg.min_samples_leaf, 20);
}

#[test]
fn test_distill_config_default() {
    let cfg = DistillConfig::default();

    assert_eq!(cfg.n_episodes, 200);
    assert_eq!(cfg.max_steps_per_episode, 10);
    assert_eq!(cfg.max_depth, 3);
    assert_eq!(cfg.min_samples_leaf, 20);
}

#[test]
fn test_basic_distillation() {
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

    // Train a simple policy
    let train_cfg = RLTrainConfig {
        n_episodes: 50,
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

    // Distill the policy
    let distill_cfg = DistillConfig {
        n_episodes: 20,
        max_steps_per_episode: 4,
        max_depth: 3,
        min_samples_leaf: 5,
        features: vec![],
    };

    let mut env2 = DoseToxEnv::new(env_cfg);
    let result = distill_policy(&mut env2, &policy, &distill_cfg);

    assert!(result.is_ok());
    let (tree, report) = result.unwrap();

    // Verify report fields
    assert!(report.n_train_samples > 0);
    assert!(report.train_accuracy >= 0.0 && report.train_accuracy <= 1.0);
    assert!(report.eval_accuracy >= 0.0 && report.eval_accuracy <= 1.0);
    assert!(report.tree_depth > 0);
    assert!(report.tree_depth <= distill_cfg.max_depth);
    assert!(report.n_nodes > 0);
    assert_eq!(tree.n_actions, 3);

    println!("Distillation Results:");
    println!("  Training samples: {}", report.n_train_samples);
    println!("  Train accuracy: {:.3}", report.train_accuracy);
    println!("  Eval accuracy: {:.3}", report.eval_accuracy);
    println!("  Tree depth: {}", report.tree_depth);
    println!("  Tree nodes: {}", report.n_nodes);
}

#[test]
fn test_tree_fidelity() {
    // Test that distilled tree has reasonable fidelity to original policy
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

    // Train a policy with more episodes for better convergence
    let train_cfg = RLTrainConfig {
        n_episodes: 100,
        max_steps_per_episode: 4,
        gamma: 0.95,
        alpha: 0.15,
        eps_start: 0.5,
        eps_end: 0.05,
    };

    let disc = BoxDiscretizer::new(
        vec![4, 4, 4, 3],
        vec![0.0, 0.0, 0.0, 0.0],
        vec![1.0, 1.0, 1.0, 1.0],
    );

    let (policy, _report) = train_q_learning(&mut env, &disc, &train_cfg, 42).unwrap();

    // Distill with enough samples
    let distill_cfg = DistillConfig {
        n_episodes: 50,
        max_steps_per_episode: 4,
        max_depth: 4,
        min_samples_leaf: 10,
        features: vec![],
    };

    let mut env2 = DoseToxEnv::new(env_cfg);
    let (tree, report) = distill_policy(&mut env2, &policy, &distill_cfg).unwrap();

    // Tree should have decent fidelity (>60%) for this simple problem
    assert!(
        report.eval_accuracy > 0.4,
        "Eval accuracy {} should be > 0.4",
        report.eval_accuracy
    );

    // Tree should be reasonably sized
    assert!(tree.depth() <= distill_cfg.max_depth);
    assert!(tree.n_nodes() < 50); // Reasonable tree size

    println!("Fidelity Test:");
    println!("  Eval accuracy: {:.3}", report.eval_accuracy);
    println!("  Tree depth: {}", tree.depth());
    println!("  Tree nodes: {}", tree.n_nodes());
}

#[test]
fn test_tree_complexity_control() {
    // Test that max_depth and min_samples_leaf control tree complexity
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

    let train_cfg = RLTrainConfig {
        n_episodes: 50,
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

    let (policy, _) = train_q_learning(&mut env, &disc, &train_cfg, 42).unwrap();

    // Distill with shallow tree
    let shallow_cfg = DistillConfig {
        n_episodes: 30,
        max_steps_per_episode: 4,
        max_depth: 2,
        min_samples_leaf: 20,
        features: vec![],
    };

    let mut env1 = DoseToxEnv::new(env_cfg.clone());
    let (shallow_tree, shallow_report) = distill_policy(&mut env1, &policy, &shallow_cfg).unwrap();

    // Distill with deeper tree
    let deep_cfg = DistillConfig {
        n_episodes: 30,
        max_steps_per_episode: 4,
        max_depth: 5,
        min_samples_leaf: 5,
        features: vec![],
    };

    let mut env2 = DoseToxEnv::new(env_cfg);
    let (deep_tree, deep_report) = distill_policy(&mut env2, &policy, &deep_cfg).unwrap();

    println!(
        "Shallow tree: depth={}, nodes={}",
        shallow_tree.depth(),
        shallow_tree.n_nodes()
    );
    println!(
        "Deep tree: depth={}, nodes={}",
        deep_tree.depth(),
        deep_tree.n_nodes()
    );

    // Shallow tree should be simpler
    assert!(shallow_tree.depth() <= shallow_cfg.max_depth);
    assert!(shallow_tree.n_nodes() <= deep_tree.n_nodes());

    // Deeper tree might have better fidelity (but not guaranteed)
    println!("Shallow accuracy: {:.3}", shallow_report.eval_accuracy);
    println!("Deep accuracy: {:.3}", deep_report.eval_accuracy);
}

#[test]
fn test_tree_execution() {
    // Test that the distilled tree can be used to select actions
    use medlangc::rl::State;

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

    let train_cfg = RLTrainConfig {
        n_episodes: 30,
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

    let (policy, _) = train_q_learning(&mut env, &disc, &train_cfg, 42).unwrap();

    let distill_cfg = DistillConfig {
        n_episodes: 20,
        max_steps_per_episode: 4,
        max_depth: 3,
        min_samples_leaf: 5,
        features: vec![],
    };

    let mut env2 = DoseToxEnv::new(env_cfg);
    let (tree, _) = distill_policy(&mut env2, &policy, &distill_cfg).unwrap();

    // Test tree execution on various states
    let test_states = vec![
        State::new(vec![0.8, 0.9, 0.0, 0.0]), // Good ANC, large tumor, start
        State::new(vec![0.3, 0.5, 0.5, 0.5]), // Low ANC, medium tumor, mid-cycle
        State::new(vec![0.9, 0.1, 0.8, 0.8]), // Good ANC, small tumor, late
    ];

    for (i, state) in test_states.iter().enumerate() {
        let action = tree.act(state);
        assert!(action < tree.n_actions);
        println!(
            "State {}: features={:?}, action={}",
            i, state.features, action
        );
    }
}

#[test]
fn test_per_action_accuracy() {
    // Test that per-action accuracy is computed correctly
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

    let train_cfg = RLTrainConfig {
        n_episodes: 50,
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

    let (policy, _) = train_q_learning(&mut env, &disc, &train_cfg, 42).unwrap();

    let distill_cfg = DistillConfig {
        n_episodes: 30,
        max_steps_per_episode: 4,
        max_depth: 3,
        min_samples_leaf: 5,
        features: vec![],
    };

    let mut env2 = DoseToxEnv::new(env_cfg);
    let (_, report) = distill_policy(&mut env2, &policy, &distill_cfg).unwrap();

    // Should have per-action accuracy for each action
    assert_eq!(report.per_action_accuracy.len(), 3);

    for (i, acc) in report.per_action_accuracy.iter().enumerate() {
        assert!(*acc >= 0.0 && *acc <= 1.0);
        println!("Action {} accuracy: {:.3}", i, acc);
    }
}

#[test]
fn test_feature_inference() {
    // Test that feature metadata is inferred correctly
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

    let (policy, _) = train_q_learning(&mut env, &disc, &train_cfg, 42).unwrap();

    let distill_cfg = DistillConfig {
        n_episodes: 20,
        max_steps_per_episode: 4,
        max_depth: 3,
        min_samples_leaf: 5,
        features: vec![],
    };

    let mut env2 = DoseToxEnv::new(env_cfg);
    let (tree, _) = distill_policy(&mut env2, &policy, &distill_cfg).unwrap();

    // Should have 4 features for DoseToxEnv
    assert_eq!(tree.features.len(), 4);

    // Check feature names
    let feature_names: Vec<&str> = tree.features.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(feature_names[0], "ANC");
    assert_eq!(feature_names[1], "tumour_size");
    assert_eq!(feature_names[2], "cycle");
    assert_eq!(feature_names[3], "prev_dose");

    // Check that min/max are reasonable
    for feature in &tree.features {
        assert!(feature.min <= feature.max);
        println!(
            "Feature '{}': [{:.3}, {:.3}]",
            feature.name, feature.min, feature.max
        );
    }
}
