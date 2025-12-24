# Week 9: Time-to-Event Endpoints & Clinical Interoperability

## Premise

**Week 8** established the L₂ protocol foundation with binary endpoints (ORR).

**Week 9** extends this with:
1. **Time-to-Event Endpoints** - PFS, OS, TTP with full survival analysis
2. **FHIR/CQL Export** - Standard clinical trial format interoperability
3. **Bayesian Power Analysis** - Operating characteristics for trial design

This transforms MedLang from "virtual trial simulator" to "complete clinical development platform."

---

## Part 1: Time-to-Event Endpoints (PFS/OS)

### Motivation

ORR (Week 8) answers: *"Did the tumor shrink?"*

PFS/OS answer: *"How long until progression/death?"*

These require:
- Event time extraction from trajectories
- Censoring handling (trial end, dropout)
- Kaplan-Meier estimation
- Hazard ratios and log-rank tests

### AST Extensions

Add to `EndpointSpec` enum:

```rust
pub enum EndpointSpec {
    ResponseRate { ... }, // Week 8
    
    // Week 9: Time-to-event
    ProgressionFreeTime {
        observable: String,           // "TumourVol"
        progression_threshold: f64,   // e.g., 1.20 (20% increase from nadir)
        baseline_reference: bool,     // true = vs baseline, false = vs nadir
        window_start_days: f64,
        window_end_days: f64,
    },
    
    OverallSurvivalTime {
        death_observable: String,     // "PatientStatus" or similar
        window_start_days: f64,
        window_end_days: f64,
    },
}
```

**MedLang syntax example:**

```medlang
endpoints {
    ORR {
        type = "binary"
        observable = "TumourVol"
        shrink_frac = 0.30
        window = [0.0_d, 84.0_d]
    }
    
    PFS {
        type = "time_to_event"
        observable = "TumourVol"
        event_definition = "progression"
        progression_threshold = 1.20  // 20% increase from nadir
        reference = "nadir"           // vs "baseline"
        window = [0.0_d, 365.0_d]
    }
    
    OS {
        type = "time_to_event"
        observable = "PatientStatus"
        event_definition = "death"
        window = [0.0_d, 730.0_d]
    }
}
```

### Trajectory Data Structure

Extend `SubjectTrajectory`:

```rust
pub struct SubjectTrajectory {
    pub subject_id: usize,
    pub arm: String,
    pub times_days: Vec<f64>,
    pub tumour_volume: Vec<f64>,
    pub baseline_tumour: f64,
    
    // Week 9: Event tracking
    pub alive_status: Vec<bool>,  // false = death
    pub in_study: Vec<bool>,      // false = dropout/censoring
}

pub struct EventTime {
    pub subject_id: usize,
    pub time_days: f64,
    pub event_occurred: bool,  // true = event, false = censored
}
```

### PFS Event Detection

```rust
pub fn detect_progression_events(
    spec: &ProgressionFreeTimeSpec,
    trajectories: &[SubjectTrajectory],
) -> Vec<EventTime> {
    trajectories.iter().map(|traj| {
        let baseline = traj.baseline_tumour;
        
        // Find nadir (minimum tumor volume)
        let nadir_idx = traj.tumour_volume.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        let nadir_vol = traj.tumour_volume[nadir_idx];
        let reference = if spec.baseline_reference { baseline } else { nadir_vol };
        
        // Find first time tumor exceeds threshold
        for (i, (&t, &v)) in traj.times_days.iter()
            .zip(traj.tumour_volume.iter())
            .enumerate()
            .skip(nadir_idx) 
        {
            if t < spec.window_start_days { continue; }
            if t > spec.window_end_days { break; }
            
            if v >= reference * spec.progression_threshold {
                return EventTime {
                    subject_id: traj.subject_id,
                    time_days: t,
                    event_occurred: true,
                };
            }
        }
        
        // No progression observed = censored at last follow-up
        EventTime {
            subject_id: traj.subject_id,
            time_days: *traj.times_days.last().unwrap_or(&spec.window_end_days),
            event_occurred: false,
        }
    }).collect()
}
```

### Kaplan-Meier Estimation

Create `compiler/src/survival.rs`:

```rust
pub struct KaplanMeier {
    pub time_points: Vec<f64>,
    pub survival_prob: Vec<f64>,
    pub n_at_risk: Vec<usize>,
    pub n_events: Vec<usize>,
    pub se: Vec<f64>,  // Greenwood standard error
}

pub fn estimate_kaplan_meier(events: &[EventTime]) -> KaplanMeier {
    // 1. Sort by time
    let mut sorted_events = events.to_vec();
    sorted_events.sort_by(|a, b| a.time_days.partial_cmp(&b.time_days).unwrap());
    
    // 2. Group by unique event times
    let mut time_points = Vec::new();
    let mut n_events = Vec::new();
    let mut n_at_risk = Vec::new();
    
    let total_n = sorted_events.len();
    let mut current_time = 0.0;
    let mut events_at_time = 0;
    let mut n_risk = total_n;
    
    for (i, evt) in sorted_events.iter().enumerate() {
        if evt.time_days > current_time && events_at_time > 0 {
            time_points.push(current_time);
            n_events.push(events_at_time);
            n_at_risk.push(n_risk);
            events_at_time = 0;
        }
        
        current_time = evt.time_days;
        if evt.event_occurred {
            events_at_time += 1;
        }
        n_risk = total_n - i - 1;
    }
    
    // Final time point
    if events_at_time > 0 {
        time_points.push(current_time);
        n_events.push(events_at_time);
        n_at_risk.push(0);
    }
    
    // 3. Compute survival probabilities
    let mut survival_prob = vec![1.0];
    let mut se = vec![0.0];
    let mut s_prev = 1.0;
    let mut var_log_s = 0.0;
    
    for (&d, &n) in n_events.iter().zip(n_at_risk.iter()) {
        if n > 0 {
            s_prev *= (n - d) as f64 / n as f64;
            survival_prob.push(s_prev);
            
            // Greenwood's formula
            var_log_s += d as f64 / (n * (n - d)) as f64;
            se.push(s_prev * var_log_s.sqrt());
        }
    }
    
    KaplanMeier {
        time_points,
        survival_prob,
        n_at_risk,
        n_events,
        se,
    }
}

pub fn median_survival(km: &KaplanMeier) -> Option<f64> {
    // Find first time where S(t) <= 0.5
    for (i, &s) in km.survival_prob.iter().enumerate() {
        if s <= 0.5 {
            return Some(km.time_points[i]);
        }
    }
    None  // Median not reached
}
```

### Log-Rank Test (Arm Comparison)

```rust
pub fn log_rank_test(events_arm1: &[EventTime], events_arm2: &[EventTime]) -> LogRankResult {
    // Combine and sort all events
    let mut all_events: Vec<(f64, usize, bool)> = Vec::new();
    
    for e in events_arm1 {
        all_events.push((e.time_days, 1, e.event_occurred));
    }
    for e in events_arm2 {
        all_events.push((e.time_days, 2, e.event_occurred));
    }
    
    all_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    
    // Compute test statistic
    let mut observed_arm1 = 0.0;
    let mut expected_arm1 = 0.0;
    let mut variance = 0.0;
    
    let mut n1_at_risk = events_arm1.len() as f64;
    let mut n2_at_risk = events_arm2.len() as f64;
    
    for (time, arm, is_event) in all_events {
        if !is_event { continue; }
        
        let n_total = n1_at_risk + n2_at_risk;
        let expected_1 = n1_at_risk / n_total;
        
        if arm == 1 {
            observed_arm1 += 1.0;
        }
        expected_arm1 += expected_1;
        
        // Variance (hypergeometric)
        variance += (n1_at_risk * n2_at_risk) / (n_total * n_total * (n_total - 1.0));
        
        if arm == 1 {
            n1_at_risk -= 1.0;
        } else {
            n2_at_risk -= 1.0;
        }
    }
    
    // Chi-square statistic
    let z = (observed_arm1 - expected_arm1) / variance.sqrt();
    let chi_sq = z * z;
    
    // p-value (chi-square with 1 df)
    let p_value = 1.0 - chi_squared_cdf(chi_sq, 1);
    
    LogRankResult {
        chi_squared: chi_sq,
        p_value,
        z_score: z,
    }
}

pub struct LogRankResult {
    pub chi_squared: f64,
    pub p_value: f64,
    pub z_score: f64,  // + favors arm1, - favors arm2
}
```

### Hazard Ratio (Cox Model)

For Week 9, implement simplified HR:

```rust
pub fn hazard_ratio_simple(
    events_arm1: &[EventTime],
    events_arm2: &[EventTime],
) -> HazardRatio {
    // Simple HR = (events_arm1 / person-time_arm1) / (events_arm2 / person-time_arm2)
    
    let (e1, pt1) = compute_events_and_person_time(events_arm1);
    let (e2, pt2) = compute_events_and_person_time(events_arm2);
    
    let rate1 = e1 / pt1;
    let rate2 = e2 / pt2;
    let hr = rate1 / rate2;
    
    // Log-rank based CI
    let log_hr = hr.ln();
    let se_log_hr = (1.0 / e1 + 1.0 / e2).sqrt();
    let ci_lower = (log_hr - 1.96 * se_log_hr).exp();
    let ci_upper = (log_hr + 1.96 * se_log_hr).exp();
    
    HazardRatio {
        hr,
        ci_95_lower: ci_lower,
        ci_95_upper: ci_upper,
    }
}

fn compute_events_and_person_time(events: &[EventTime]) -> (f64, f64) {
    let n_events = events.iter().filter(|e| e.event_occurred).count() as f64;
    let person_time: f64 = events.iter().map(|e| e.time_days).sum();
    (n_events, person_time)
}

pub struct HazardRatio {
    pub hr: f64,
    pub ci_95_lower: f64,
    pub ci_95_upper: f64,
}
```

### Endpoint Evaluation Integration

Update `compute_endpoint`:

```rust
pub fn compute_endpoint(
    spec: &IREndpointSpec,
    trajectories: &[SubjectTrajectory],
) -> EndpointResult {
    match spec {
        IREndpointSpec::ResponseRate { ... } => {
            let orr = compute_orr(spec, trajectories);
            EndpointResult::Binary { 
                name: "ORR",
                value: orr,
                ci_95: compute_binomial_ci(orr, trajectories.len()),
            }
        },
        
        IREndpointSpec::ProgressionFreeTime { ... } => {
            let events = detect_progression_events(spec, trajectories);
            let km = estimate_kaplan_meier(&events);
            let median = median_survival(&km);
            
            EndpointResult::TimeToEvent {
                name: "PFS",
                median_days: median,
                km_curve: km,
                n_events: events.iter().filter(|e| e.event_occurred).count(),
                n_censored: events.iter().filter(|e| !e.event_occurred).count(),
            }
        },
        
        IREndpointSpec::OverallSurvivalTime { ... } => {
            // Similar to PFS but uses death events
            unimplemented!("OS in full Week 9 implementation")
        },
    }
}

pub enum EndpointResult {
    Binary {
        name: &'static str,
        value: f64,
        ci_95: (f64, f64),
    },
    TimeToEvent {
        name: &'static str,
        median_days: Option<f64>,
        km_curve: KaplanMeier,
        n_events: usize,
        n_censored: usize,
    },
}
```

### CLI Output for PFS

```bash
mlc simulate-protocol oncology_phase2.medlang \
    --n-per-arm 200 \
    --endpoints ORR,PFS \
    --out results.json
```

**Output JSON:**

```json
{
  "protocol": "Oncology_Phase2",
  "arms": [
    {
      "name": "ArmA",
      "label": "200 mg QD",
      "n_included": 180,
      "endpoints": {
        "ORR": {
          "type": "binary",
          "value": 0.35,
          "ci_95": [0.28, 0.42]
        },
        "PFS": {
          "type": "time_to_event",
          "median_days": 156.0,
          "ci_95": [142.0, 178.0],
          "n_events": 98,
          "n_censored": 82,
          "km_12mo": 0.42
        }
      }
    },
    {
      "name": "ArmB",
      "label": "400 mg QD",
      "n_included": 185,
      "endpoints": {
        "ORR": {
          "type": "binary",
          "value": 0.48,
          "ci_95": [0.41, 0.55]
        },
        "PFS": {
          "type": "time_to_event",
          "median_days": 224.0,
          "ci_95": [198.0, 256.0],
          "n_events": 87,
          "n_censored": 98,
          "km_12mo": 0.58
        }
      }
    }
  ],
  "comparisons": {
    "ORR_diff": 0.13,
    "ORR_p_value": 0.012,
    "PFS_HR": 0.68,
    "PFS_HR_ci_95": [0.51, 0.91],
    "PFS_logrank_p": 0.008
  }
}
```

---

## Part 2: FHIR/CQL Export

### Motivation

Clinical trials must interface with:
- Electronic Health Records (FHIR)
- Clinical Quality Language (CQL) for eligibility
- Standard trial registries (ClinicalTrials.gov)

**Week 9 Goal**: Export MedLang protocols to FHIR `PlanDefinition` and CQL inclusion criteria.

### FHIR PlanDefinition Mapping

Create `compiler/src/fhir_export.rs`:

```rust
use serde_json::json;

pub fn protocol_to_fhir_plan_definition(protocol: &IRProtocol) -> serde_json::Value {
    json!({
        "resourceType": "PlanDefinition",
        "id": protocol.name.to_lowercase().replace(" ", "-"),
        "title": protocol.name,
        "type": {
            "coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/plan-definition-type",
                "code": "clinical-protocol"
            }]
        },
        "status": "draft",
        "date": chrono::Utc::now().to_rfc3339(),
        
        // Arms as actions
        "action": protocol.arms.iter().map(|arm| {
            json!({
                "title": arm.label,
                "description": format!("{} - {} mg every {} days", 
                    arm.drug_id, arm.dose_mg, arm.schedule_days),
                "code": [{
                    "coding": [{
                        "system": "http://example.org/drug-codes",
                        "code": arm.drug_id,
                        "display": arm.label
                    }]
                }],
                "timing": {
                    "repeat": {
                        "frequency": 1,
                        "period": arm.schedule_days,
                        "periodUnit": "d"
                    }
                },
                "dosage": [{
                    "doseAndRate": [{
                        "doseQuantity": {
                            "value": arm.dose_mg,
                            "unit": "mg",
                            "system": "http://unitsofmeasure.org",
                            "code": "mg"
                        }
                    }]
                }]
            })
        }).collect::<Vec<_>>(),
        
        // Visits as schedule
        "relatedArtifact": [{
            "type": "documentation",
            "label": "Visit Schedule",
            "document": {
                "title": "Protocol Visits",
                "attachment": {
                    "contentType": "application/json",
                    "data": base64::encode(serde_json::to_string(&protocol.visits).unwrap())
                }
            }
        }]
    })
}
```

### CQL Inclusion Criteria

```rust
pub fn inclusion_to_cql(inclusion: &IRInclusion) -> String {
    let mut cql = String::from("library ProtocolEligibility version '1.0.0'\n\n");
    cql.push_str("using FHIR version '4.0.1'\n\n");
    cql.push_str("context Patient\n\n");
    
    // Age criterion
    if let Some((min_age, max_age)) = inclusion.age_range {
        cql.push_str(&format!(
            "define \"Age In Range\":\n  AgeInYears() >= {} and AgeInYears() <= {}\n\n",
            min_age, max_age
        ));
    }
    
    // ECOG criterion
    if let Some(ref ecog_allowed) = inclusion.ecog_allowed {
        let ecog_list = ecog_allowed.iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        
        cql.push_str(&format!(
            "define \"ECOG Performance Status\":\n  \
            exists([Observation: \"ECOG Performance Status\"] O\n    \
            where O.value in {{{}}}\n  )\n\n",
            ecog_list
        ));
    }
    
    // Baseline tumor volume
    if let Some(min_vol) = inclusion.baseline_tumour_min_cm3 {
        cql.push_str(&format!(
            "define \"Tumor Volume Adequate\":\n  \
            exists([Observation: \"Tumor Volume\"] O\n    \
            where O.value >= {} 'cm3'\n  )\n\n",
            min_vol
        ));
    }
    
    // Combined eligibility
    cql.push_str("define \"Patient Eligible\":\n");
    let mut criteria = Vec::new();
    if inclusion.age_range.is_some() {
        criteria.push("\"Age In Range\"");
    }
    if inclusion.ecog_allowed.is_some() {
        criteria.push("\"ECOG Performance Status\"");
    }
    if inclusion.baseline_tumour_min_cm3.is_some() {
        criteria.push("\"Tumor Volume Adequate\"");
    }
    cql.push_str(&format!("  {}\n", criteria.join(" and\n  ")));
    
    cql
}
```

### CLI Export Commands

```bash
# Export to FHIR
mlc export-fhir oncology_phase2.medlang -o protocol.fhir.json

# Export inclusion to CQL
mlc export-cql oncology_phase2.medlang -o eligibility.cql

# Combined export
mlc export-protocol oncology_phase2.medlang \
    --format fhir,cql \
    --out-dir exports/
```

---

## Part 3: Bayesian Power Analysis

### Motivation

Trial designers need to know:
- "What sample size do I need for 80% power?"
- "What if my EC50 prior is wrong?"
- "How does QM uncertainty affect power?"

**Week 9 Goal**: Operating characteristics via Bayesian simulation.

### Power Analysis Framework

Create `compiler/src/power_analysis.rs`:

```rust
pub struct PowerAnalysisSpec {
    pub protocol: IRProtocol,
    pub null_hypothesis: HypothesisSpec,
    pub alternative_hypothesis: HypothesisSpec,
    pub sample_sizes: Vec<usize>,  // e.g., [50, 100, 150, 200]
    pub n_simulations: usize,       // e.g., 1000
    pub alpha: f64,                 // e.g., 0.05
}

pub enum HypothesisSpec {
    ORR_Difference {
        arm1_rate: f64,
        arm2_rate: f64,
        delta: f64,  // minimum clinically meaningful difference
    },
    PFS_HazardRatio {
        hr: f64,
        hr_threshold: f64,
    },
}

pub struct PowerResult {
    pub sample_size: usize,
    pub power: f64,           // Pr(reject H0 | H1 true)
    pub type_i_error: f64,    // Pr(reject H0 | H0 true)
    pub expected_events: f64,
}

pub fn compute_power_curve(
    spec: &PowerAnalysisSpec,
    qm_stub: Option<&QuantumStub>,
) -> Vec<PowerResult> {
    spec.sample_sizes.iter().map(|&n| {
        let mut n_rejections = 0;
        
        for _ in 0..spec.n_simulations {
            // Simulate trial with n subjects per arm
            let trial_result = simulate_virtual_trial(
                &spec.protocol,
                n,
                qm_stub,
            );
            
            // Test hypothesis
            let reject_h0 = match &spec.alternative_hypothesis {
                HypothesisSpec::ORR_Difference { delta, .. } => {
                    let diff = trial_result.arm2_orr - trial_result.arm1_orr;
                    let p = binomial_test(
                        trial_result.arm1_orr,
                        trial_result.arm2_orr,
                        n,
                        n,
                    );
                    p < spec.alpha && diff.abs() >= *delta
                },
                
                HypothesisSpec::PFS_HazardRatio { hr_threshold, .. } => {
                    let hr = trial_result.pfs_hr;
                    let p = trial_result.pfs_logrank_p;
                    p < spec.alpha && hr <= *hr_threshold
                },
            };
            
            if reject_h0 {
                n_rejections += 1;
            }
        }
        
        PowerResult {
            sample_size: n,
            power: n_rejections as f64 / spec.n_simulations as f64,
            type_i_error: spec.alpha,
            expected_events: 0.0,  // compute from simulations
        }
    }).collect()
}
```

### Uncertainty Quantification

```rust
pub struct BayesianPowerSpec {
    pub protocol: IRProtocol,
    pub qm_stub: QuantumStub,
    pub qm_uncertainty: QMUncertainty,  // Uncertainty in Kd, ΔG
    pub population_prior: PopulationPrior,
    pub n_posterior_draws: usize,       // e.g., 1000
}

pub struct QMUncertainty {
    pub kd_cv: f64,      // Coefficient of variation for Kd
    pub dg_sd: f64,      // SD for ΔG in kcal/mol
}

pub fn bayesian_power_analysis(
    spec: &BayesianPowerSpec,
) -> BayesianPowerResult {
    let mut power_samples = Vec::new();
    
    for _ in 0..spec.n_posterior_draws {
        // 1. Draw from QM uncertainty
        let kd_sample = sample_lognormal(
            spec.qm_stub.Kd_M,
            spec.qm_uncertainty.kd_cv,
        );
        
        let dg_sample = sample_normal(
            spec.qm_stub.dG_part_plasma_tumor_kcal_per_mol.unwrap(),
            spec.qm_uncertainty.dg_sd,
        );
        
        // 2. Draw from population parameter priors
        let pop_params = sample_population_prior(&spec.population_prior);
        
        // 3. Simulate trial with these parameters
        let trial = simulate_with_parameters(
            &spec.protocol,
            kd_sample,
            dg_sample,
            &pop_params,
        );
        
        // 4. Compute power for this parameter set
        let power = compute_single_power(&trial);
        power_samples.push(power);
    }
    
    BayesianPowerResult {
        mean_power: power_samples.iter().sum::<f64>() / power_samples.len() as f64,
        power_95_ci: compute_credible_interval(&power_samples, 0.95),
        power_distribution: power_samples,
    }
}
```

---

## Implementation Priority

### Phase 1: Time-to-Event Core (High Priority)
1. PFS event detection (~3-4 hours)
2. Kaplan-Meier estimation (~4-5 hours)
3. Log-rank test (~2-3 hours)
4. Hazard ratio (~2 hours)
5. Integration with endpoint system (~2-3 hours)

**Total**: 13-17 hours

### Phase 2: FHIR/CQL Export (Medium Priority)
1. FHIR PlanDefinition mapping (~4-5 hours)
2. CQL generation (~3-4 hours)
3. Export CLI commands (~2 hours)
4. Validation against schemas (~2-3 hours)

**Total**: 11-14 hours

### Phase 3: Bayesian Power (Lower Priority)
1. Basic power analysis framework (~5-6 hours)
2. Bayesian uncertainty integration (~4-5 hours)
3. Visualization/reporting (~3-4 hours)

**Total**: 12-15 hours

**Grand Total Week 9**: 36-46 hours

---

## Testing Strategy

### PFS/Survival Tests

```rust
#[test]
fn test_pfs_event_detection() {
    let traj = SubjectTrajectory {
        subject_id: 1,
        times_days: vec![0.0, 28.0, 56.0, 84.0],
        tumour_volume: vec![100.0, 70.0, 60.0, 85.0],  // Progression at day 84
        baseline_tumour: 100.0,
        ..Default::default()
    };
    
    let events = detect_progression_events(&pfs_spec, &[traj]);
    assert_eq!(events[0].time_days, 84.0);
    assert_eq!(events[0].event_occurred, true);
}

#[test]
fn test_kaplan_meier_median() {
    // Create synthetic event times with known median
    let events = vec![
        EventTime { subject_id: 1, time_days: 50.0, event_occurred: true },
        EventTime { subject_id: 2, time_days: 100.0, event_occurred: true },
        EventTime { subject_id: 3, time_days: 150.0, event_occurred: false },
        EventTime { subject_id: 4, time_days: 200.0, event_occurred: true },
    ];
    
    let km = estimate_kaplan_meier(&events);
    let median = median_survival(&km);
    
    assert!(median.is_some());
    assert!((median.unwrap() - 100.0).abs() < 1.0);
}
```

### FHIR Export Tests

```rust
#[test]
fn test_fhir_export_valid_json() {
    let protocol = create_test_protocol();
    let fhir = protocol_to_fhir_plan_definition(&protocol);
    
    assert_eq!(fhir["resourceType"], "PlanDefinition");
    assert!(fhir["action"].is_array());
    assert_eq!(fhir["action"].as_array().unwrap().len(), 2);  // 2 arms
}

#[test]
fn test_cql_generation() {
    let inclusion = IRInclusion {
        age_range: Some((18, 75)),
        ecog_allowed: Some(vec![0, 1]),
        baseline_tumour_min_cm3: Some(50.0),
    };
    
    let cql = inclusion_to_cql(&inclusion);
    
    assert!(cql.contains("AgeInYears() >= 18"));
    assert!(cql.contains("ECOG Performance Status"));
    assert!(cql.contains("Tumor Volume"));
}
```

---

## Documentation Deliverables

1. **Survival Analysis Guide** (`docs/survival_analysis.md`)
   - PFS vs OS definitions
   - Kaplan-Meier interpretation
   - Hazard ratio interpretation

2. **FHIR Export Guide** (`docs/fhir_interop.md`)
   - Protocol → PlanDefinition mapping
   - CQL generation workflow
   - Integration with EHR systems

3. **Power Analysis Guide** (`docs/power_analysis.md`)
   - Sample size calculation
   - Operating characteristics
   - Bayesian sensitivity analysis

4. **Week 9 Summary** (`docs/week9_summary.md`)
   - Complete feature list
   - Example workflows
   - Comparison with clinical trial software

---

## What Week 9 Unlocks

### For Trialists
- Complete endpoint toolkit (ORR + PFS + OS)
- Standard format export (FHIR/CQL)
- Power calculations with mechanistic underpinnings

### For Researchers
- QM uncertainty → clinical outcome uncertainty
- Sensitivity analysis: "How wrong can my Kd be?"
- Mechanistic trial optimization

### For Regulators
- Standards-compliant protocol specifications
- Audit trail from quantum to clinic
- Reproducible virtual trial simulations

---

## Conclusion

Week 9 transforms MedLang from:
- **"Virtual trial simulator"** (Week 8)

To:
- **"Complete clinical development platform"** (Week 9)

With:
1. ✅ Binary endpoints (ORR)
2. ✅ Time-to-event endpoints (PFS/OS)
3. ✅ Survival analysis (KM, log-rank, HR)
4. ✅ FHIR/CQL interoperability
5. ✅ Bayesian power analysis

All while maintaining the **quantum → tissue → patient → population** vertical integration.

**Next (Week 10+)**: Adaptive designs, biomarker endpoints, combination therapy optimization.
