// Week 42: Env-Parameter Robustness Tests
//
// Tests for evaluating dose guideline robustness across
// variations in DoseToxEnv parameters (reward weights, penalties, dose scaling, etc.)

use medlangc::rl::{
    apply_env_scenario, default_robustness_scenarios, format_robustness_report,
    simulate_guideline_env_robustness, AtomicConditionIR, ComparisonOpIR, DoseGuidelineIRHost,
    DoseGuidelineOutcomeConfig, DoseRuleIR, DoseToxEnvConfig, EnvScenario,
};

// =============================================================================
// Test Helpers
// =============================================================================

fn make_simple_guideline() -> DoseGuidelineIRHost {
    // Simple guideline: if ANC >= 0.5 give 200mg, else give 100mg
    DoseGuidelineIRHost {
        name: "test-guideline".to_string(),
        description: "Test guideline for robustness analysis".to_string(),
        feature_names: vec!["ANC".to_string()],
        dose_levels_mg: vec![0.0, 50.0, 100.0, 200.0, 300.0],
        rules: vec![
            DoseRuleIR {
                conditions: vec![AtomicConditionIR {
                    feature: "ANC".to_string(),
                    op: ComparisonOpIR::GE,
                    threshold: 0.5,
                }],
                action_index: 3, // 200mg
                action_dose_mg: 200.0,
            },
            DoseRuleIR {
                conditions: vec![AtomicConditionIR {
                    feature: "ANC".to_string(),
                    op: ComparisonOpIR::LT,
                    threshold: 0.5,
                }],
                action_index: 2, // 100mg
                action_dose_mg: 100.0,
            },
        ],
    }
}

fn make_base_env_config() -> DoseToxEnvConfig {
    DoseToxEnvConfig {
        ev_handle: "test_ev".to_string(),
        backend: medlangc::ml::BackendKind::Surrogate,
        n_cycles: 4,
        dose_levels_mg: vec![0.0, 50.0, 100.0, 200.0, 300.0],
        reward_response_weight: 1.0,
        reward_tox_penalty: 2.0,
        contract_penalty: 10.0,
        seed: Some(42),
    }
}

fn make_outcome_config() -> DoseGuidelineOutcomeConfig {
    DoseGuidelineOutcomeConfig {
        n_episodes: 20,
        response_tumour_ratio_threshold: 0.8,
        grade3_threshold: 3,
        grade4_threshold: 4,
    }
}

// =============================================================================
// EnvScenario Tests
// =============================================================================

#[test]
fn test_env_scenario_builder() {
    let scen = EnvScenario::new("test-scenario")
        .with_n_cycles(8)
        .with_tox_penalty(3.0)
        .with_response_weight(0.5)
        .with_dose_scale(0.9)
        .with_contract_penalty(15.0)
        .with_seed(123);

    assert_eq!(scen.name, "test-scenario");
    assert_eq!(scen.n_cycles, Some(8));
    assert_eq!(scen.reward_tox_penalty, Some(3.0));
    assert_eq!(scen.reward_response_weight, Some(0.5));
    assert_eq!(scen.dose_scale, Some(0.9));
    assert_eq!(scen.contract_penalty, Some(15.0));
    assert_eq!(scen.seed, Some(123));
}

#[test]
fn test_env_scenario_partial_override() {
    let scen = EnvScenario::new("partial").with_tox_penalty(4.0);

    assert_eq!(scen.name, "partial");
    assert_eq!(scen.reward_tox_penalty, Some(4.0));
    assert!(scen.n_cycles.is_none());
    assert!(scen.reward_response_weight.is_none());
    assert!(scen.dose_scale.is_none());
}

// =============================================================================
// apply_env_scenario Tests
// =============================================================================

#[test]
fn test_apply_env_scenario_n_cycles() {
    let base = make_base_env_config();
    let scen = EnvScenario::new("test").with_n_cycles(10);

    let modified = apply_env_scenario(&base, &scen);

    assert_eq!(modified.n_cycles, 10);
    // Other fields unchanged
    assert_eq!(modified.reward_response_weight, base.reward_response_weight);
    assert_eq!(modified.reward_tox_penalty, base.reward_tox_penalty);
    assert_eq!(modified.contract_penalty, base.contract_penalty);
    assert_eq!(modified.dose_levels_mg, base.dose_levels_mg);
}

#[test]
fn test_apply_env_scenario_weights() {
    let base = make_base_env_config();
    let scen = EnvScenario::new("test")
        .with_response_weight(0.5)
        .with_tox_penalty(4.0)
        .with_contract_penalty(20.0);

    let modified = apply_env_scenario(&base, &scen);

    assert_eq!(modified.reward_response_weight, 0.5);
    assert_eq!(modified.reward_tox_penalty, 4.0);
    assert_eq!(modified.contract_penalty, 20.0);
    // n_cycles unchanged
    assert_eq!(modified.n_cycles, base.n_cycles);
}

#[test]
fn test_apply_env_scenario_dose_scale() {
    let base = DoseToxEnvConfig {
        dose_levels_mg: vec![0.0, 100.0, 200.0, 300.0],
        ..make_base_env_config()
    };
    let scen = EnvScenario::new("test").with_dose_scale(0.5);

    let modified = apply_env_scenario(&base, &scen);

    assert_eq!(modified.dose_levels_mg, vec![0.0, 50.0, 100.0, 150.0]);
}

#[test]
fn test_apply_env_scenario_dose_scale_increase() {
    let base = DoseToxEnvConfig {
        dose_levels_mg: vec![0.0, 100.0, 200.0],
        ..make_base_env_config()
    };
    let scen = EnvScenario::new("test").with_dose_scale(1.5);

    let modified = apply_env_scenario(&base, &scen);

    assert_eq!(modified.dose_levels_mg, vec![0.0, 150.0, 300.0]);
}

#[test]
fn test_apply_env_scenario_no_overrides() {
    let base = make_base_env_config();
    let scen = EnvScenario::new("no-changes");

    let modified = apply_env_scenario(&base, &scen);

    assert_eq!(modified.n_cycles, base.n_cycles);
    assert_eq!(modified.reward_response_weight, base.reward_response_weight);
    assert_eq!(modified.reward_tox_penalty, base.reward_tox_penalty);
    assert_eq!(modified.contract_penalty, base.contract_penalty);
    assert_eq!(modified.dose_levels_mg, base.dose_levels_mg);
}

#[test]
fn test_apply_env_scenario_seed_override() {
    let base = make_base_env_config();
    let scen = EnvScenario::new("test").with_seed(999);

    let modified = apply_env_scenario(&base, &scen);

    assert_eq!(modified.seed, Some(999));
}

// =============================================================================
// default_robustness_scenarios Tests
// =============================================================================

#[test]
fn test_default_robustness_scenarios_count() {
    let scenarios = default_robustness_scenarios();
    assert_eq!(scenarios.len(), 10);
}

#[test]
fn test_default_robustness_scenarios_names() {
    let scenarios = default_robustness_scenarios();
    let names: Vec<&str> = scenarios.iter().map(|s| s.name.as_str()).collect();

    assert!(names.contains(&"tox-light"));
    assert!(names.contains(&"tox-heavy"));
    assert!(names.contains(&"efficacy-light"));
    assert!(names.contains(&"efficacy-heavy"));
    assert!(names.contains(&"dose-reduced"));
    assert!(names.contains(&"dose-increased"));
    assert!(names.contains(&"short-treatment"));
    assert!(names.contains(&"long-treatment"));
}

#[test]
fn test_default_robustness_scenarios_have_overrides() {
    let scenarios = default_robustness_scenarios();

    // Each scenario should have at least one override
    for scen in &scenarios {
        let has_override = scen.n_cycles.is_some()
            || scen.reward_response_weight.is_some()
            || scen.reward_tox_penalty.is_some()
            || scen.contract_penalty.is_some()
            || scen.dose_scale.is_some();
        assert!(
            has_override,
            "Scenario '{}' should have at least one override",
            scen.name
        );
    }
}

// =============================================================================
// simulate_guideline_env_robustness Tests
// =============================================================================

#[test]
fn test_robustness_basic_two_scenarios() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = make_outcome_config();

    let scenarios = vec![
        EnvScenario::new("tox-light").with_tox_penalty(1.0),
        EnvScenario::new("tox-heavy").with_tox_penalty(4.0),
    ];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    // Check report structure
    assert_eq!(report.guideline_name, "test-guideline");
    assert_eq!(report.scenarios.len(), 2);
    assert_eq!(report.scenarios[0].scenario_name, "tox-light");
    assert_eq!(report.scenarios[1].scenario_name, "tox-heavy");

    // All outcomes should have correct n_episodes
    assert_eq!(report.base_outcome.n_episodes, 20);
    for scen in &report.scenarios {
        assert_eq!(scen.outcome.n_episodes, 20);
    }

    // Scenarios should have modified env configs
    assert_eq!(report.scenarios[0].env_config.reward_tox_penalty, 1.0);
    assert_eq!(report.scenarios[1].env_config.reward_tox_penalty, 4.0);
}

#[test]
fn test_robustness_empty_scenarios() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = make_outcome_config();

    let scenarios: Vec<EnvScenario> = vec![];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    assert!(report.scenarios.is_empty());
    // Base outcome should still be computed
    assert_eq!(report.base_outcome.n_episodes, 20);
}

#[test]
fn test_robustness_with_defaults() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 10, // Smaller for faster test
        ..make_outcome_config()
    };

    let scenarios = default_robustness_scenarios();

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    assert_eq!(report.scenarios.len(), 10);
    for scen in &report.scenarios {
        assert_eq!(scen.outcome.n_episodes, 10);
    }
}

// =============================================================================
// RobustnessSummary Tests
// =============================================================================

#[test]
fn test_robustness_summary_ranges() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 10,
        ..make_outcome_config()
    };

    let scenarios = vec![
        EnvScenario::new("s1").with_tox_penalty(1.0),
        EnvScenario::new("s2").with_tox_penalty(3.0),
    ];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    let summary = report.robustness_summary();

    // Ranges should be valid (min <= max)
    assert!(summary.response_rate_range.0 <= summary.response_rate_range.1);
    assert!(summary.grade3plus_rate_range.0 <= summary.grade3plus_rate_range.1);
    assert!(summary.contract_violation_range.0 <= summary.contract_violation_range.1);
    assert!(summary.mean_rdi_range.0 <= summary.mean_rdi_range.1);

    // Rates should be in [0, 1]
    assert!(summary.response_rate_range.0 >= 0.0 && summary.response_rate_range.1 <= 1.0);
    assert!(summary.grade3plus_rate_range.0 >= 0.0 && summary.grade3plus_rate_range.1 <= 1.0);
}

#[test]
fn test_robustness_summary_best_worst() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 10,
        ..make_outcome_config()
    };

    let scenarios = vec![
        EnvScenario::new("scenario-a").with_tox_penalty(1.0),
        EnvScenario::new("scenario-b").with_tox_penalty(4.0),
    ];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    let summary = report.robustness_summary();

    // Should identify best/worst scenarios
    assert!(summary.worst_scenario.is_some() || summary.best_scenario.is_some());

    if let Some(worst) = &summary.worst_scenario {
        assert!(worst == "scenario-a" || worst == "scenario-b");
    }
    if let Some(best) = &summary.best_scenario {
        assert!(best == "scenario-a" || best == "scenario-b");
    }
}

#[test]
fn test_robustness_summary_empty_scenarios() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 10,
        ..make_outcome_config()
    };

    let scenarios: Vec<EnvScenario> = vec![];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    let summary = report.robustness_summary();

    // With no scenarios, ranges should be the base outcome value
    assert_eq!(summary.response_rate_range.0, summary.response_rate_range.1);
    assert!(summary.worst_scenario.is_none());
    assert!(summary.best_scenario.is_none());
}

// =============================================================================
// format_robustness_report Tests
// =============================================================================

#[test]
fn test_format_robustness_report() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 5,
        ..make_outcome_config()
    };

    let scenarios = vec![EnvScenario::new("test-scen").with_tox_penalty(2.0)];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    let formatted = format_robustness_report(&report);

    // Check key sections are present
    assert!(formatted.contains("Env-Parameter Robustness Report"));
    assert!(formatted.contains("test-guideline"));
    assert!(formatted.contains("Base Environment"));
    assert!(formatted.contains("Scenario Outcomes"));
    assert!(formatted.contains("test-scen"));
    assert!(formatted.contains("Robustness Summary"));
    assert!(formatted.contains("Response rate range"));
}

// =============================================================================
// Scenario Diversity Tests
// =============================================================================

#[test]
fn test_dose_scale_affects_outcomes() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 30,
        ..make_outcome_config()
    };

    let scenarios = vec![
        EnvScenario::new("dose-reduced").with_dose_scale(0.5),
        EnvScenario::new("dose-increased").with_dose_scale(1.5),
    ];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    // Dose scaling should be reflected in env configs
    let reduced = &report.scenarios[0];
    let increased = &report.scenarios[1];

    // Check dose levels are scaled
    assert!(reduced.env_config.dose_levels_mg[1] < base_env.dose_levels_mg[1]);
    assert!(increased.env_config.dose_levels_mg[1] > base_env.dose_levels_mg[1]);
}

#[test]
fn test_n_cycles_affects_outcomes() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 20,
        ..make_outcome_config()
    };

    let scenarios = vec![
        EnvScenario::new("short").with_n_cycles(2),
        EnvScenario::new("long").with_n_cycles(8),
    ];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    // n_cycles should be reflected in env configs
    assert_eq!(report.scenarios[0].env_config.n_cycles, 2);
    assert_eq!(report.scenarios[1].env_config.n_cycles, 8);
}

// =============================================================================
// Serialization Tests
// =============================================================================

#[test]
fn test_env_scenario_serialization() {
    let scen = EnvScenario::new("test")
        .with_n_cycles(6)
        .with_tox_penalty(3.0);

    let json = serde_json::to_string(&scen).unwrap();
    let parsed: EnvScenario = serde_json::from_str(&json).unwrap();

    assert_eq!(parsed.name, "test");
    assert_eq!(parsed.n_cycles, Some(6));
    assert_eq!(parsed.reward_tox_penalty, Some(3.0));
}

#[test]
fn test_robustness_report_serialization() {
    let base_env = make_base_env_config();
    let guideline = make_simple_guideline();
    let outcome_cfg = DoseGuidelineOutcomeConfig {
        n_episodes: 5,
        ..make_outcome_config()
    };

    let scenarios = vec![EnvScenario::new("test").with_tox_penalty(2.0)];

    let report =
        simulate_guideline_env_robustness(&base_env, &guideline, &outcome_cfg, &scenarios).unwrap();

    // Should be serializable
    let json = serde_json::to_string_pretty(&report).unwrap();
    assert!(json.contains("guideline_name"));
    assert!(json.contains("base_env"));
    assert!(json.contains("base_outcome"));
    assert!(json.contains("scenarios"));

    // Should be deserializable
    let parsed: medlangc::rl::GuidelineEnvRobustnessReport = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.guideline_name, report.guideline_name);
    assert_eq!(parsed.scenarios.len(), 1);
}
