# Week 10: Time-to-Event Endpoints (PFS/TTP)

## Overview

Week 10 extends the L₂ Protocol DSL with **time-to-event (TTE) endpoint** capabilities, enabling virtual clinical trials to compute progression-free survival (PFS) and time-to-progression (TTP) from mechanistic PBPK+QSP tumor trajectories.

This builds on Week 8's foundation (binary ORR endpoints) to support survival analysis with Kaplan-Meier estimation.

## Key Features

### 1. Time-to-Event Endpoint Definition

Extended `EndpointSpec` to support PFS/TTP:

```rust
pub enum EndpointSpec {
    ResponseRate {
        observable: String,
        shrink_fraction: f64,
        window_start_days: f64,
        window_end_days: f64,
    },
    TimeToProgression {
        observable: String,
        increase_fraction: f64,    // e.g., 0.20 for 20% increase
        window_start_days: f64,
        window_end_days: f64,
        ref_baseline: bool,         // NEW: true=baseline, false=nadir reference
    },
}
```

### 2. Progression Definition

Two reference modes for defining progression:

**Option A: Baseline Reference** (`ref_baseline = true`)
- Progression occurs when: `T(t) ≥ (1 + δ) × T_baseline`
- Example: 20% increase from baseline tumor volume
- Use case: Simple, conservative definition

**Option B: Nadir Reference** (`ref_baseline = false`) ✓ Recommended
- Progression occurs when: `T(t) ≥ (1 + δ) × T_nadir(t)`
- Nadir = running minimum (best response up to time t)
- Example: 20% increase from best response
- Use case: More realistic, aligns with RECIST criteria

### 3. Subject-Level TTE Computation

For each subject, compute:

```rust
pub struct TimeToEvent {
    pub subject_id: usize,
    pub time_days: f64,    // Time to event or censoring
    pub event: bool,       // true = progression, false = censored
}
```

**Algorithm** (nadir reference mode):

1. Filter measurements to analysis window `[t_start, t_end]`
2. For each time point `t`:
   - Track running minimum: `nadir(t) = min{T(s) : s ≤ t}`
   - Compute threshold: `threshold(t) = nadir(t) × (1 + δ)`
   - If `T(t) ≥ threshold(t)` and `t > t_0`: return `(t, event=true)`
3. If no progression: return `(t_last, event=false)` (censored)

### 4. Kaplan-Meier Survival Analysis

Arm-level summary using Kaplan-Meier estimator:

```rust
pub struct ArmSurvivalSummary {
    pub arm_name: String,
    pub n_included: usize,
    pub times: Vec<f64>,         // Unique event/censoring times
    pub surv: Vec<f64>,          // S(t) at each time
    pub n_risk: Vec<usize>,      // Number at risk at each time
    pub n_event: Vec<usize>,     // Number of events at each time
    pub median_time: Option<f64>, // Median PFS (first t where S(t) ≤ 0.5)
}
```

**Kaplan-Meier Formula**:

For each unique time `t_j`:
- `d_j` = number of events at `t_j`
- `n_j` = number at risk just before `t_j`
- `S(t_j) = S(t_{j-1}) × (1 - d_j / n_j)`

**Median PFS**: First time where `S(t) ≤ 0.5`

## Implementation

### Module: `compiler/src/endpoints.rs`

Core types and functions:

```rust
// Subject trajectory data
pub struct SubjectTrajectory {
    pub id: usize,
    pub times_days: Vec<f64>,
    pub tumour_vol: Vec<f64>,
    pub baseline_tumour: f64,
    pub covariates: SubjectCovariates,
}

// Compute PFS for a single subject
pub fn compute_time_to_progression(
    spec: &EndpointSpec,
    subject: &SubjectTrajectory,
) -> Option<TimeToEvent>

// Compute arm-level PFS summary with KM
pub fn compute_arm_pfs(
    arm_name: &str,
    spec: &EndpointSpec,
    subjects: &[SubjectTrajectory],
    inclusion: &[InclusionClause],
) -> ArmSurvivalSummary

// Kaplan-Meier estimator
pub fn kaplan_meier(
    times: &[f64],
    events: &[bool],
) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>)
```

### Binary Endpoints (ORR)

Also implemented in the same module:

```rust
// Compute ORR for a single subject
pub fn compute_orr(
    spec: &EndpointSpec,
    subject: &SubjectTrajectory,
) -> Option<BinaryEndpointResult>

// Compute arm-level ORR summary
pub fn compute_arm_orr(
    arm_name: &str,
    spec: &EndpointSpec,
    subjects: &[SubjectTrajectory],
    inclusion: &[InclusionClause],
) -> ArmBinarySummary
```

### Inclusion/Exclusion Criteria

```rust
pub fn passes_inclusion(
    subject: &SubjectTrajectory,
    inclusion: &[InclusionClause],
) -> bool
```

Supports:
- Age ranges: `AgeBetween { min_years, max_years }`
- ECOG status: `ECOGIn { allowed: Vec<u8> }`
- Baseline tumor: `BaselineTumourGe { volume_cm3 }`

## Example Usage

### Protocol Definition

```medlang
protocol Oncology_Phase2 {
    population_model Oncology_PBPK_QSP_Pop
    
    arms {
        ArmA { label = "200 mg QD"; dose = 200.0_mg }
        ArmB { label = "400 mg QD"; dose = 400.0_mg }
    }
    
    visits {
        baseline at 0.0_d
        cycle1   at 28.0_d
        cycle2   at 56.0_d
        cycle3   at 84.0_d
    }
    
    inclusion {
        age between 18_y and 75_y
        ECOG in [0, 1]
        baseline_tumour_volume >= 50.0_cm3
    }
    
    endpoints {
        ORR {
            type          = "binary"
            observable    = TumourVol
            shrink_frac   = 0.30        // 30% shrinkage
            window        = [0.0_d, 84.0_d]
        }
        
        PFS {
            type              = "time_to_event"
            observable        = TumourVol
            progression_frac  = 0.20    // 20% increase from nadir
            ref_baseline      = false   // Use nadir (best response)
            window            = [0.0_d, 84.0_d]
        }
    }
}
```

### Programmatic Usage

```rust
use medlangc::endpoints::*;
use medlangc::ast::EndpointSpec;

// Create PFS endpoint specification
let pfs_spec = EndpointSpec::TimeToProgression {
    observable: "TumourVol".to_string(),
    increase_fraction: 0.20,
    window_start_days: 0.0,
    window_end_days: 84.0,
    ref_baseline: false,  // Use nadir
};

// Compute subject-level TTE
let tte = compute_time_to_progression(&pfs_spec, &subject)?;
println!("Subject {}: PFS = {} days, event = {}", 
    tte.subject_id, tte.time_days, tte.event);

// Compute arm-level summary
let pfs_summary = compute_arm_pfs(
    "Arm A",
    &pfs_spec,
    &subjects,
    &inclusion_criteria,
);

println!("Median PFS: {:?} days", pfs_summary.median_time);
println!("KM curve:");
for (t, s) in pfs_summary.times.iter().zip(pfs_summary.surv.iter()) {
    println!("  S({:.0}) = {:.3}", t, s);
}
```

### Expected Output Format

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
          "n_responders": 63,
          "response_rate": 0.35
        },
        "PFS": {
          "median_days": 62.0,
          "km_times": [0.0, 28.0, 56.0, 84.0],
          "km_surv": [1.0, 0.92, 0.75, 0.60],
          "km_n_risk": [180, 165, 135, 108],
          "km_n_event": [0, 15, 30, 27]
        }
      }
    },
    {
      "name": "ArmB",
      "label": "400 mg QD",
      "n_included": 185,
      "endpoints": {
        "ORR": {
          "n_responders": 92,
          "response_rate": 0.50
        },
        "PFS": {
          "median_days": 78.0,
          "km_times": [0.0, 28.0, 56.0, 84.0],
          "km_surv": [1.0, 0.95, 0.82, 0.68],
          "km_n_risk": [185, 176, 152, 126],
          "km_n_event": [0, 9, 24, 26]
        }
      }
    }
  ]
}
```

## Testing

Comprehensive test suite in `compiler/src/endpoints.rs`:

### PFS/TTP Tests

1. **`test_pfs_progression_from_baseline`**: Validates progression detection using baseline reference
2. **`test_pfs_progression_from_nadir`**: Validates progression detection using nadir reference
3. **`test_pfs_censored`**: Validates censoring when no progression occurs

### ORR Tests

4. **`test_orr_responder`**: Subject with sufficient shrinkage
5. **`test_orr_non_responder`**: Subject with insufficient shrinkage

### Kaplan-Meier Tests

6. **`test_kaplan_meier_simple`**: Validates KM calculation with events and censoring

### Inclusion Criteria Tests

7. **`test_inclusion_criteria`**: Validates filtering by age, ECOG, baseline tumor

### Test Results

```bash
$ cargo test --lib endpoints --release

running 7 tests
test endpoints::tests::test_inclusion_criteria ... ok
test endpoints::tests::test_kaplan_meier_simple ... ok
test endpoints::tests::test_orr_non_responder ... ok
test endpoints::tests::test_orr_responder ... ok
test endpoints::tests::test_pfs_progression_from_baseline ... ok
test endpoints::tests::test_pfs_censored ... ok
test endpoints::tests::test_pfs_progression_from_nadir ... ok

test result: ok. 7 passed; 0 failed; 0 ignored
```

## Clinical Interpretation

### PFS vs ORR

| Endpoint | Type | Measures | Advantages | Disadvantages |
|----------|------|----------|------------|---------------|
| **ORR** | Binary | Best response (shrinkage) | Simple, early readout | Doesn't capture duration |
| **PFS** | Time-to-event | Time to progression | Captures duration, regulatory endpoint | Requires longer follow-up |

### Typical Values (Oncology)

**ORR**:
- Standard of care: 10-30%
- Active new agent: 30-50%
- Highly active agent: >50%

**PFS** (solid tumors):
- Standard of care: 3-6 months
- Modest improvement: +2-3 months
- Substantial benefit: +6+ months

### RECIST Alignment

The nadir reference mode (`ref_baseline = false`) aligns with RECIST v1.1 progressive disease criteria:
- PD = ≥20% increase AND ≥5mm absolute increase from nadir
- MedLang simplified version: ≥20% increase from nadir (no absolute criterion yet)

## Implementation Status

### Completed (Week 10) ✅

- [x] Extended AST with `ref_baseline` field
- [x] Implemented subject-level PFS computation with nadir tracking
- [x] Implemented Kaplan-Meier survival analysis
- [x] Created per-arm survival summary structures
- [x] Added comprehensive test suite (7 tests, all passing)
- [x] Full documentation

### Future Enhancements

**Week 11+ (Not Yet Implemented)**:

1. **Parser Integration**: 
   - Parse `PFS` endpoint blocks from `.medlang` files
   - Currently only programmatic API available

2. **CLI Integration**:
   - `mlc simulate-protocol --endpoints ORR,PFS`
   - JSON output generation

3. **Additional Features**:
   - Overall Survival (OS) endpoint
   - Hazard ratios between arms
   - Log-rank test p-values
   - Confidence intervals for median PFS
   - RECIST absolute size criterion (≥5mm)
   - Competing risks (death before progression)

4. **Visualization**:
   - KM curve plotting
   - Forest plots for hazard ratios
   - Waterfall plots combining ORR + PFS

5. **Stan Integration**:
   - Bayesian survival models
   - Time-varying covariates from PBPK states
   - Cure fraction models

## Mathematical Details

### Kaplan-Meier Estimator

Given times `t₁ < t₂ < ... < t_k` with events and censoring:

1. **Sort** observations by time
2. **Group** by unique times
3. **Compute** at each time `t_j`:
   - `d_j` = # events at `t_j`
   - `c_j` = # censored at `t_j`
   - `n_j` = # at risk before `t_j` = `n_{j-1} - d_{j-1} - c_{j-1}`
   - `S(t_j) = S(t_{j-1}) × (1 - d_j / n_j)`

4. **Median**: `median = min{t : S(t) ≤ 0.5}`

### Running Minimum Algorithm

For nadir reference mode:

```python
nadir = [T[0]]  # Initialize with baseline
for i in range(1, len(T)):
    nadir.append(min(nadir[-1], T[i]))
    
    threshold = nadir[i] * (1 + delta)
    if T[i] >= threshold:
        return (time[i], event=True)
        
return (time[-1], event=False)  # Censored
```

This ensures progression is always measured relative to the best response achieved up to that point.

## Files Modified/Created

### New Files

- **`compiler/src/endpoints.rs`** (420 lines): Complete endpoint evaluation module
  - Subject trajectory types
  - ORR and PFS computation
  - Kaplan-Meier estimator
  - Inclusion criteria filtering
  - Comprehensive test suite

### Modified Files

- **`compiler/src/lib.rs`**: Added `pub mod endpoints;`
- **`compiler/src/ast/mod.rs`**: Added `ref_baseline: bool` to `TimeToProgression` variant

### Documentation

- **`docs/week10_time_to_event_endpoints.md`** (this file): Complete specification and guide

## Summary

Week 10 successfully implements time-to-event endpoint analysis for MedLang's clinical trial DSL. The endpoint evaluation module provides a solid foundation for virtual oncology trials with both binary (ORR) and survival (PFS) endpoints.

The implementation is **production-ready for programmatic use** and requires only parser/CLI integration to enable full `.medlang` file support.

**Next steps**: Integrate with protocol parser (Week 8 completion) and CLI simulator to enable end-to-end virtual trial workflows.
