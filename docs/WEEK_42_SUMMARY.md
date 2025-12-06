# Week 42: Env-Parameter Robustness Analysis for Dose Guidelines

## Overview

Week 42 introduces **robustness analysis** for dose guidelines by evaluating how guideline outcomes change when perturbing DoseToxEnv parameters. This provides sensitivity analysis to understand how fragile or robust a given guideline is to reasonable changes in the environment's reward structure, treatment duration, and dose scaling.

## Key Concepts

### Why Robustness Analysis?

A dose guideline that performs well under one set of assumptions may fail under slightly different conditions:

- **Toxicity weighting**: If we underestimate toxicity penalties, a guideline may appear safe when it isn't
- **Efficacy weighting**: Different efficacy expectations can change optimal dosing strategies
- **Treatment duration**: Longer or shorter treatment cycles affect cumulative outcomes
- **Dose scaling**: Protocol variations (e.g., dose reductions) impact guideline behavior

Robustness analysis answers: *"How sensitive is this guideline to changes in our modeling assumptions?"*

### EnvScenario

An `EnvScenario` represents a single perturbation of the base environment configuration:

```rust
pub struct EnvScenario {
    pub name: String,                        // Scenario identifier
    pub n_cycles: Option<usize>,             // Override treatment cycles
    pub reward_response_weight: Option<f64>, // Override efficacy weight
    pub reward_tox_penalty: Option<f64>,     // Override toxicity penalty
    pub contract_penalty: Option<f64>,       // Override contract violation penalty
    pub dose_scale: Option<f64>,             // Scale all dose levels
    pub seed: Option<u64>,                   // Override random seed
}
```

### Default Robustness Scenarios

The module provides 10 default scenarios covering common sensitivity axes:

| Scenario | Override |
|----------|----------|
| `tox-light` | reward_tox_penalty = 1.0 |
| `tox-heavy` | reward_tox_penalty = 4.0 |
| `efficacy-light` | reward_response_weight = 0.5 |
| `efficacy-heavy` | reward_response_weight = 2.0 |
| `contract-light` | contract_penalty = 5.0 |
| `contract-heavy` | contract_penalty = 20.0 |
| `dose-reduced` | dose_scale = 0.8 |
| `dose-increased` | dose_scale = 1.2 |
| `short-treatment` | n_cycles = 4 |
| `long-treatment` | n_cycles = 8 |

## Implementation

### Core Module

**File**: `compiler/src/rl/dose_guideline_robustness.rs`

Key functions:

- `apply_env_scenario(base, scenario)` - Apply scenario overrides to a base config
- `simulate_guideline_env_robustness(env_cfg, guideline, outcome_cfg, scenarios)` - Run robustness analysis
- `default_robustness_scenarios()` - Get standard sensitivity scenarios
- `format_robustness_report(report)` - Pretty-print robustness report

### Data Structures

```rust
/// Per-scenario outcome summary
pub struct GuidelineEnvScenarioSummary {
    pub scenario_name: String,
    pub env_config: DoseToxEnvConfig,
    pub outcome: DoseGuidelineOutcomeSummary,
}

/// Full robustness report
pub struct GuidelineEnvRobustnessReport {
    pub guideline_name: String,
    pub base_env: DoseToxEnvConfig,
    pub outcome_config: DoseGuidelineOutcomeConfig,
    pub base_outcome: DoseGuidelineOutcomeSummary,
    pub scenarios: Vec<GuidelineEnvScenarioSummary>,
}

/// Aggregated robustness metrics
pub struct RobustnessSummary {
    pub response_rate_range: (f64, f64),
    pub grade3plus_rate_range: (f64, f64),
    pub contract_violation_range: (f64, f64),
    pub mean_rdi_range: (f64, f64),
    pub worst_scenario: Option<String>,
    pub best_scenario: Option<String>,
}
```

### Runtime Builtin

**Builtin**: `simulate_guideline_env_robustness`

```
simulate_guideline_env_robustness(
    env_cfg: DoseToxEnvConfig,
    guideline: DoseGuidelineIR,
    outcome_cfg: DoseGuidelineOutcomeConfig,
    scenarios: Vector<EnvScenario>
) -> GuidelineEnvRobustnessReport
```

### CLI Command

```bash
mlc guideline-env-robustness \
    --env-config base_env.json \
    --guideline guideline.json \
    --outcome-config outcome_cfg.json \
    [--scenarios scenarios.json] \
    --out robustness_report.json \
    [-v]
```

If `--scenarios` is omitted, the 10 default scenarios are used.

## Usage Examples

### Rust API

```rust
use medlangc::rl::{
    simulate_guideline_env_robustness,
    default_robustness_scenarios,
    format_robustness_report,
    DoseToxEnvConfig,
    DoseGuidelineIRHost,
    DoseGuidelineOutcomeConfig,
    EnvScenario,
};

// Create base config
let base_env = DoseToxEnvConfig {
    n_cycles: 6,
    dose_levels_mg: vec![0.0, 50.0, 100.0, 200.0, 300.0],
    reward_response_weight: 1.0,
    reward_tox_penalty: 2.0,
    contract_penalty: 10.0,
    ..Default::default()
};

let guideline = load_guideline("my_guideline.json")?;

let outcome_cfg = DoseGuidelineOutcomeConfig {
    n_episodes: 100,
    response_tumour_ratio_threshold: 0.7,
    grade3_threshold: 3,
    grade4_threshold: 4,
};

// Use default scenarios
let scenarios = default_robustness_scenarios();

// Run analysis
let report = simulate_guideline_env_robustness(
    &base_env,
    &guideline,
    &outcome_cfg,
    &scenarios,
)?;

// Print formatted report
println!("{}", format_robustness_report(&report));

// Check robustness metrics
let summary = report.robustness_summary();
println!("Response rate range: {:.1}% - {:.1}%",
    summary.response_rate_range.0 * 100.0,
    summary.response_rate_range.1 * 100.0);

if let Some(worst) = &summary.worst_scenario {
    println!("Worst performing scenario: {}", worst);
}
```

### Custom Scenarios

```rust
// Define custom scenarios for specific sensitivity analysis
let scenarios = vec![
    EnvScenario::new("very-toxic")
        .with_tox_penalty(5.0)
        .with_contract_penalty(25.0),
    
    EnvScenario::new("dose-escalation-study")
        .with_dose_scale(1.5)
        .with_n_cycles(8),
    
    EnvScenario::new("conservative")
        .with_response_weight(0.5)
        .with_tox_penalty(3.0),
];

let report = simulate_guideline_env_robustness(
    &base_env,
    &guideline,
    &outcome_cfg,
    &scenarios,
)?;
```

### CLI Usage

```bash
# Using default scenarios
mlc guideline-env-robustness \
    --env-config env.json \
    --guideline my_guideline.json \
    --outcome-config outcome.json \
    --out report.json -v

# Using custom scenarios
cat > scenarios.json << 'EOF'
[
  {"name": "tox-high", "reward_tox_penalty": 5.0},
  {"name": "short-course", "n_cycles": 3},
  {"name": "dose-reduction", "dose_scale": 0.7}
]
EOF

mlc guideline-env-robustness \
    --env-config env.json \
    --guideline my_guideline.json \
    --outcome-config outcome.json \
    --scenarios scenarios.json \
    --out report.json
```

## Output Format

### JSON Report Structure

```json
{
  "guideline_name": "my-guideline",
  "base_env": {
    "n_cycles": 6,
    "reward_response_weight": 1.0,
    "reward_tox_penalty": 2.0,
    "contract_penalty": 10.0,
    "dose_levels_mg": [0.0, 50.0, 100.0, 200.0, 300.0]
  },
  "outcome_config": {
    "n_episodes": 100,
    "response_tumour_ratio_threshold": 0.7,
    "grade3_threshold": 3,
    "grade4_threshold": 4
  },
  "base_outcome": {
    "n_episodes": 100,
    "response_rate": 0.65,
    "mean_best_tumour_ratio": 0.72,
    "grade3plus_rate": 0.15,
    "grade4plus_rate": 0.05,
    "contract_violation_rate": 0.02,
    "mean_rdi": 0.85
  },
  "scenarios": [
    {
      "scenario_name": "tox-light",
      "env_config": { ... },
      "outcome": { ... }
    },
    ...
  ]
}
```

### Pretty-Print Output

```
=== Env-Parameter Robustness Report: my-guideline ===

Base Environment:
  n_cycles: 6
  reward_response_weight: 1.00
  reward_tox_penalty: 2.00
  contract_penalty: 10.00
  dose_levels: [0.0, 50.0, 100.0, 200.0, 300.0]

Scenario Outcomes:
Scenario             Response    G3+ Tox   Contract        RDI
----------------------------------------------------------------
base                    65.0%      15.0%       2.0%       0.85
tox-light               68.0%      18.0%       3.0%       0.88
tox-heavy               60.0%      12.0%       1.0%       0.80
efficacy-light          55.0%      10.0%       1.5%       0.75
efficacy-heavy          72.0%      20.0%       4.0%       0.90
...

Robustness Summary:
  Response rate range: 55.0% - 72.0%
  G3+ toxicity range:  10.0% - 20.0%
  Contract viol range: 1.0% - 4.0%
  Mean RDI range:      0.75 - 0.90
  Worst scenario: efficacy-light
  Best scenario:  efficacy-heavy
```

## Interpreting Results

### Response Rate Range
- **Narrow range** (e.g., 60-65%): Guideline is robust to parameter changes
- **Wide range** (e.g., 40-80%): Guideline is sensitive; investigate scenarios that cause poor performance

### Toxicity Rate Variance
- High variance indicates guideline safety depends heavily on model assumptions
- Consider more conservative thresholds if toxicity varies significantly

### Worst/Best Scenarios
- Identify which parameter changes cause largest outcome swings
- Use for targeted protocol modifications or conservative adjustments

## Files Added/Modified

### New Files
- `compiler/src/rl/dose_guideline_robustness.rs` - Core robustness module
- `compiler/tests/week_42_robustness_tests.rs` - Test suite
- `docs/WEEK_42_SUMMARY.md` - This documentation

### Modified Files
- `compiler/src/rl/mod.rs` - Added robustness module exports
- `compiler/src/runtime/builtins.rs` - Added `SimulateGuidelineEnvRobustness` builtin
- `compiler/src/bin/mlc.rs` - Added `guideline-env-robustness` CLI command

## Dependencies

Week 42 builds on:
- **Week 31-32**: DoseToxEnv and RL core infrastructure
- **Week 38**: DoseGuidelineIRHost and guideline representation
- **Week 41**: DoseGuidelineOutcomeConfig and outcome simulation

## Future Extensions

1. **Covariate robustness**: Test guideline performance across patient subpopulations
2. **Stochastic scenarios**: Monte Carlo sampling of parameter combinations
3. **Visualization**: Generate robustness heatmaps and sensitivity plots
4. **Optimization**: Find parameter regions where guideline is most robust
5. **QM/PK integration**: When Track C is complete, extend to quantum pharmacology parameters
