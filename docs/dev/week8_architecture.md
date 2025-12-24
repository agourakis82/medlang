# Week 8: L₂ Clinical Protocol DSL - Architecture & Implementation Plan

## Vision

Transform MedLang from a modeling DSL into a **clinical trial design language** by adding an L₂ protocol layer that enables clinicians to specify:

- Trial arms (treatment regimens)
- Visit schedules
- Inclusion/exclusion criteria
- Clinical endpoints (ORR, PFS, OS)

Built on top of the mechanistic PBPK-QSP-QM models from Weeks 1-7.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  L₂ Protocol Layer (Week 8)                             │
│  ┌───────────┐  ┌─────────┐  ┌───────────┐             │
│  │ Protocol  │  │  Arms   │  │ Endpoints │             │
│  │   Spec    │  │         │  │ (ORR,PFS) │             │
│  └─────┬─────┘  └────┬────┘  └─────┬─────┘             │
│        │             │              │                    │
│        └─────────────┴──────────────┘                    │
│                      │                                   │
│              Instantiate per arm                         │
│                      ↓                                   │
└──────────────────────┼───────────────────────────────────┘
                       │
┌──────────────────────┼───────────────────────────────────┐
│  L₁ Mechanistic Layer (Weeks 1-7)                        │
│                      ↓                                   │
│  ┌──────────────────────────────────────┐                │
│  │  Timeline (per arm)                  │                │
│  │  - Dosing: 200mg QD vs 400mg QD      │                │
│  │  - Visits: baseline, C1, C2, FU      │                │
│  └────────────┬─────────────────────────┘                │
│               │                                          │
│               ↓                                          │
│  ┌──────────────────────────────────────┐                │
│  │  Cohort (filtered by inclusion)      │                │
│  │  - Age 18-75                          │                │
│  │  - ECOG 0-1                           │                │
│  │  - Baseline tumor ≥ 50 cm³           │                │
│  └────────────┬─────────────────────────┘                │
│               │                                          │
│               ↓                                          │
│  ┌──────────────────────────────────────┐                │
│  │  Population Model                    │                │
│  │  Oncology_PBPK_QSP (with QM stub)    │                │
│  │  - PBPK: A_plasma, A_tumor           │                │
│  │  - QSP: Tumour volume                │                │
│  │  - QM: Kd → EC50, ΔG → Kp_tumor      │                │
│  └──────────────────────────────────────┘                │
└───────────────────────────────────────────────────────────┘
```

## Week 8 Progress

### ✅ Completed

1. **AST Extensions** (`compiler/src/ast/mod.rs`)
   - `ProtocolDef` - Top-level protocol container
   - `ArmDef` - Treatment arm specification (label, dose)
   - `VisitDef` - Visit schedule (name, time)
   - `InclusionDef` - Inclusion criteria clauses
   - `EndpointDef` - Clinical endpoint specifications
   - `EndpointKind` - Binary vs TimeToEvent
   - `EndpointSpec` - ResponseRate and TimeToProgression specs
   - Added `Declaration::Protocol` variant

2. **Lexer Tokens** (`compiler/src/lexer.rs`)
   - `protocol`, `arms`, `visits`, `inclusion`, `endpoints`
   - `between` (for age ranges)
   - Display implementations

3. **Type Checker** (`compiler/src/typeck.rs`)
   - Added Protocol handling (placeholder for Week 8 completion)

4. **Example Protocol** (`docs/examples/oncology_phase2_protocol.medlang`)
   - 2-arm Phase 2 oncology trial
   - Visit schedule (baseline → followup)
   - Inclusion criteria (age, ECOG, baseline tumor)
   - ORR endpoint specification

### 🚧 Remaining Implementation

#### 1. Parser Support (`compiler/src/parser.rs`)

Need to add:

```rust
fn protocol_def(input: TokenSlice) -> IResult<TokenSlice, ProtocolDef> {
    // Parse: protocol Name { ... }
}

fn arms_block(input: TokenSlice) -> IResult<TokenSlice, Vec<ArmDef>> {
    // Parse: arms { ArmA { label = "..."; dose = 200.0_mg; } ... }
}

fn visits_block(input: TokenSlice) -> IResult<TokenSlice, Vec<VisitDef>> {
    // Parse: visits { baseline at 0.0_d; cycle1 at 28.0_d; ... }
}

fn inclusion_block(input: TokenSlice) -> IResult<TokenSlice, InclusionDef> {
    // Parse: inclusion { age between 18_y and 75_y; ... }
}

fn endpoints_block(input: TokenSlice) -> IResult<TokenSlice, Vec<EndpointDef>> {
    // Parse: endpoints { ORR { type = "binary"; ... } }
}
```

**Parsing strategy:**
- Reuse existing expression parsers for values
- Use block-based parsing similar to `model { }`
- Handle unit literals (200.0_mg, 28.0_d, etc.)

#### 2. IR Layer (`compiler/src/ir.rs` - new structs)

```rust
pub struct IRProtocol {
    pub name: String,
    pub population_model_name: String,
    pub arms: Vec<IRArm>,
    pub visits: Vec<IRVisit>,
    pub inclusion: Option<IRInclusion>,
    pub endpoints: Vec<IREndpoint>,
}

pub struct IRArm {
    pub name: String,
    pub label: String,
    pub dose_mg: f64,
}

pub struct IRVisit {
    pub name: String,
    pub time_days: f64,
}

pub struct IRInclusion {
    pub age_range: Option<(u32, u32)>,
    pub ecog_allowed: Option<Vec<u8>>,
    pub baseline_tumour_min_cm3: Option<f64>,
}

pub enum IREndpointSpec {
    ResponseRate {
        observable: String,
        shrink_fraction: f64,
        window_start_days: f64,
        window_end_days: f64,
    },
}
```

**Lowering:** AST `ProtocolDef` → IR `IRProtocol`
- Extract all fields
- Convert unit literals to canonical units (days, mg, etc.)

#### 3. Protocol → Timeline Instantiation (`compiler/src/protocol.rs` - new module)

```rust
pub struct ArmStudy {
    pub arm: IRArm,
    pub timeline: IRTimeline,
    pub cohort: IRCohort,
}

pub fn instantiate_arm_timeline(
    protocol: &IRProtocol,
    arm: &IRArm,
) -> Result<IRTimeline, ProtocolError> {
    // 1. Create dose events at t=0 with arm.dose_mg
    // 2. Create observation events at each visit time
    // 3. Return IRTimeline for this arm
}

pub fn create_arm_cohort(
    protocol: &IRProtocol,
    arm: &IRArm,
    n_subjects: usize,
) -> Result<IRCohort, ProtocolError> {
    // 1. Reference the population model
    // 2. Reference the arm's timeline
    // 3. Apply inclusion criteria as filters
}
```

#### 4. Endpoint Evaluation (`compiler/src/endpoints.rs` - new module)

```rust
pub struct SubjectTrajectory {
    pub subject_id: usize,
    pub times: Vec<f64>,
    pub tumour_volume: Vec<f64>,
    pub baseline_tumour: f64,
}

pub fn compute_orr(
    spec: &IREndpointSpec,
    trajectories: &[SubjectTrajectory],
) -> EndpointResult {
    match spec {
        IREndpointSpec::ResponseRate { shrink_fraction, .. } => {
            let responders: Vec<bool> = trajectories.iter().map(|traj| {
                let baseline = traj.baseline_tumour;
                let min_vol = traj.tumour_volume.iter().copied()
                    .fold(f64::INFINITY, f64::min);
                
                // Response = ≥30% shrinkage
                (baseline - min_vol) / baseline >= *shrink_fraction
            }).collect();
            
            let orr = responders.iter().filter(|&&r| r).count() as f64 
                    / responders.len() as f64;
            
            EndpointResult::Binary { value: orr }
        },
        _ => unimplemented!("PFS in Week 9"),
    }
}

pub enum EndpointResult {
    Binary { value: f64 },       // ORR
    TimeToEvent { median: f64 }, // PFS
}
```

#### 5. CLI Commands (`compiler/src/bin/mlc.rs`)

```rust
// New commands:

// 1. Compile protocol (debugging)
mlc compile-protocol examples/oncology_phase2_protocol.medlang -o protocol.json

// 2. Simulate virtual trial
mlc simulate-protocol \
    examples/oncology_phase2_protocol.medlang \
    --qm-stub data/lig001_qm_stub.json \
    --n-subjects 200 \
    --out results/phase2_sim.json
```

**Implementation:**

```rust
fn simulate_protocol_command(
    protocol_file: &Path,
    qm_stub_file: Option<&Path>,
    n_subjects: usize,
    output: &Path,
) -> Result<(), Error> {
    // 1. Parse protocol
    let protocol_ast = parse_protocol_file(protocol_file)?;
    let ir_protocol = lower_protocol(&protocol_ast)?;
    
    // 2. Load QM stub
    let qm_stub = qm_stub_file.map(|p| QuantumStub::load(p)).transpose()?;
    
    // 3. For each arm:
    let mut arm_results = Vec::new();
    for arm in &ir_protocol.arms {
        // 3a. Instantiate timeline and cohort
        let timeline = instantiate_arm_timeline(&ir_protocol, arm)?;
        let cohort = create_arm_cohort(&ir_protocol, arm, n_subjects)?;
        
        // 3b. Simulate subjects (reuse existing simulation)
        let trajectories = simulate_cohort(
            &ir_protocol.population_model_name,
            &timeline,
            &cohort,
            qm_stub.as_ref(),
        )?;
        
        // 3c. Apply inclusion filters
        let included = apply_inclusion_criteria(
            trajectories,
            &ir_protocol.inclusion,
        );
        
        // 3d. Compute endpoints
        let orr = compute_orr(
            &ir_protocol.endpoints[0].spec, // First endpoint
            &included,
        )?;
        
        arm_results.push(ArmResult {
            arm_name: arm.name.clone(),
            label: arm.label.clone(),
            n_included: included.len(),
            orr,
        });
    }
    
    // 4. Write JSON output
    serde_json::to_writer_pretty(
        File::create(output)?,
        &TrialResults { arms: arm_results },
    )?;
    
    Ok(())
}
```

## Example Output

Running:
```bash
mlc simulate-protocol \
    oncology_phase2_protocol.medlang \
    --qm-stub lig001_egfr_qm.json \
    --n-subjects 200
```

Would produce:
```json
{
  "protocol": "Oncology_Phase2",
  "population_model": "Oncology_PBPK_QSP_Pop",
  "qm_stub": "LIG001 → EGFR",
  "arms": [
    {
      "name": "ArmA",
      "label": "200 mg QD",
      "dose_mg": 200.0,
      "n_randomized": 200,
      "n_included": 178,
      "endpoints": {
        "ORR": {
          "type": "binary",
          "value": 0.35,
          "95_ci": [0.28, 0.42]
        }
      }
    },
    {
      "name": "ArmB",
      "label": "400 mg QD",
      "dose_mg": 400.0,
      "n_randomized": 200,
      "n_included": 181,
      "endpoints": {
        "ORR": {
          "type": "binary",
          "value": 0.48,
          "95_ci": [0.41, 0.55]
        }
      }
    }
  ],
  "comparison": {
    "delta_ORR": 0.13,
    "p_value": 0.012
  }
}
```

## Testing Strategy

1. **Parser tests** (`tests/protocol_parser_test.rs`)
   - Parse arms block
   - Parse visits block
   - Parse inclusion criteria
   - Parse endpoints
   - Full protocol round-trip

2. **IR tests** (`tests/protocol_ir_test.rs`)
   - AST → IR lowering
   - Unit conversion (mg, days, etc.)
   - Validation (dose > 0, times increasing, etc.)

3. **Endpoint tests** (`tests/endpoint_evaluation_test.rs`)
   - Synthetic trajectories with known ORR
   - Edge cases (all responders, none, missing data)

4. **Integration tests** (`tests/protocol_integration_test.rs`)
   - Full protocol → timelines instantiation
   - Simulated trial with known parameters
   - Endpoint computation

## Design Decisions

### 1. Keep L₂ Separate from L₁

Protocol layer is **compositional**, not invasive:
- Protocols reference existing population models
- No changes to core MedLang-D semantics
- Clean separation: `protocol.rs`, `endpoints.rs`

### 2. Limited Scope for Week 8

Focus on **one working vertical slice**:
- ✅ ORR endpoint (binary)
- ❌ PFS (defer to Week 9)
- ✅ Simple inclusion (age, ECOG, baseline tumor)
- ❌ Complex eligibility (biomarkers, prior therapy)

### 3. Deterministic Simulation First

Week 8 focuses on **virtual trials** with mechanistic model:
- Fixed population parameters (no uncertainty yet)
- Simulate N subjects per arm
- Compute endpoint statistics

Week 9+ can add:
- Bayesian uncertainty in population parameters
- Power calculations
- Adaptive designs

## Next Steps (Week 9 Options)

After completing Week 8, natural extensions:

### Option A: Time-to-Event Endpoints (PFS/OS)

- Add `TimeToProgression` endpoint evaluation
- Implement Kaplan-Meier estimation
- Log-rank tests for arm comparison
- Hazard ratio calculation

### Option B: FHIR/CQL Export

- Map `ProtocolDef` to FHIR PlanDefinition
- Generate CQL inclusion criteria
- Export to standard clinical trial formats

### Option C: Bayesian Trial Design

- Treat population parameters as uncertain
- Add prior elicitation from QM + literature
- Implement operating characteristics
- Adaptive randomization

## Summary

Week 8 establishes MedLang as a **clinical trial design language** by adding an L₂ protocol layer that:

1. **Specifies** trial structure (arms, visits, criteria, endpoints)
2. **Instantiates** per-arm timelines from protocol spec
3. **Simulates** virtual patients with mechanistic PBPK-QSP-QM model
4. **Evaluates** clinical endpoints (ORR, with PFS coming in Week 9)

This transforms MedLang from "mechanistic model compiler" to "complete clinical development platform" connecting molecular physics to patient outcomes.

## Files Created/Modified

### Created
- `compiler/src/endpoints.rs` (new module - to implement)
- `compiler/src/protocol.rs` (new module - to implement)
- `docs/examples/oncology_phase2_protocol.medlang` (example)
- `docs/week8_architecture.md` (this document)

### Modified
- `compiler/src/ast/mod.rs` (+80 lines) - Protocol AST structures
- `compiler/src/lexer.rs` (+20 lines) - Protocol tokens
- `compiler/src/typeck.rs` (+1 line) - Protocol type checking stub
- `compiler/src/parser.rs` (to be modified) - Protocol parsing
- `compiler/src/ir.rs` (to be modified) - Protocol IR
- `compiler/src/bin/mlc.rs` (to be modified) - CLI commands

### Test Files (to be created)
- `tests/protocol_parser_test.rs`
- `tests/protocol_ir_test.rs`
- `tests/endpoint_evaluation_test.rs`
- `tests/protocol_integration_test.rs`
