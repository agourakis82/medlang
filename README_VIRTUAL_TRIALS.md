# MedLang Virtual Clinical Trials

## Quick Start

### 1. Define a Protocol

Create a `.medlang` file with your trial protocol:

```medlang
protocol MyPhase2Trial {
    population model Oncology_PBPK_QSP_Pop
    
    arms {
        LowDose { label = "100 mg QD"; dose = 100.0 }
        HighDose { label = "200 mg QD"; dose = 200.0 }
    }
    
    visits {
        baseline at 0.0
        week4 at 28.0
        week8 at 56.0
        week12 at 84.0
    }
    
    inclusion {
        age between 18 and 75
        ECOG in [0, 1]
        baseline_tumour_volume >= 50.0
    }
    
    endpoints {
        ORR {
            type = "binary"
            observable = TumourVol
            shrink_frac = 0.30
            window = [0.0, 84.0]
        }
        
        PFS {
            type = "time_to_event"
            observable = TumourVol
            progression_frac = 0.20
            ref_baseline = false
            window = [0.0, 84.0]
        }
    }
}
```

### 2. Run Virtual Trial

```bash
cd compiler
cargo build --release
./target/release/simulate_protocol ../docs/examples/my_trial.medlang 200
```

### 3. Analyze Results

The simulator outputs:
- Console summary with ORR and median PFS per arm
- JSON file with complete results including Kaplan-Meier curves

Example output:
```
MedLang Protocol Simulator
==========================

Protocol: MyPhase2Trial
  Arms: 2
  Visits: 4
  Endpoints: 2

Simulating 100 mg QD (dose=100 mg, N=200)...
  ORR: 12.3% (17/138)
  PFS: median not reached

Simulating 200 mg QD (dose=200 mg, N=200)...
  ORR: 58.7% (81/138)
  PFS: median not reached
```

## Protocol Syntax

### Arms
```medlang
arms {
    ArmName {
        label = "Human-readable label"
        dose = 100.0  // mg
    }
}
```

### Visits
```medlang
visits {
    visit_name at 0.0    // days
    week4 at 28.0
    week8 at 56.0
}
```

### Inclusion Criteria
```medlang
inclusion {
    age between 18 and 75           // years
    ECOG in [0, 1]                  // performance status
    baseline_tumour_volume >= 50.0  // cm³
}
```

### Endpoints

**Binary Endpoint (ORR)**:
```medlang
ORR {
    type = "binary"
    observable = TumourVol
    shrink_frac = 0.30      // 30% shrinkage = response
    window = [0.0, 84.0]    // analysis window (days)
}
```

**Time-to-Event Endpoint (PFS)**:
```medlang
PFS {
    type = "time_to_event"
    observable = TumourVol
    progression_frac = 0.20       // 20% increase = progression
    ref_baseline = false          // false = nadir reference, true = baseline
    window = [0.0, 84.0]
}
```

## JSON Output Format

```json
{
  "protocol": "MyPhase2Trial",
  "n_per_arm": 200,
  "arms": [
    {
      "arm": "LowDose",
      "label": "100 mg QD",
      "dose_mg": 100.0,
      "endpoints": {
        "ORR": {
          "type": "binary",
          "n_included": 138,
          "n_responders": 17,
          "response_rate": 0.123
        },
        "PFS": {
          "type": "time_to_event",
          "n_included": 138,
          "median_days": null,
          "km_times": [84.0],
          "km_surv": [0.95],
          "km_n_risk": [138],
          "km_n_event": [7]
        }
      }
    }
  ]
}
```

## Clinical Interpretation

### ORR (Objective Response Rate)

**Definition**: Proportion of subjects with ≥30% tumor shrinkage from baseline

**Typical Values** (solid tumors):
- Standard of care: 10-30%
- Active new agent: 30-50%
- Highly active agent: >50%

**Example**:
- Low dose (100mg): 12.3% ORR → Modest activity
- High dose (200mg): 58.7% ORR → Highly active agent

### PFS (Progression-Free Survival)

**Definition**: Time to 20% tumor increase from best response (nadir)

**Typical Values** (solid tumors):
- Standard of care: 3-6 months
- Modest improvement: +2-3 months
- Substantial benefit: +6+ months

**Kaplan-Meier Interpretation**:
- `km_times`: Observation times
- `km_surv`: Survival probability S(t)
- `km_n_risk`: Number at risk
- `km_n_event`: Number of progression events
- `median_days`: Median PFS (first time S(t) ≤ 0.5)

## Implementation Status

### ✅ Implemented
- [x] Protocol DSL parser
- [x] ORR endpoint computation
- [x] PFS endpoint computation
- [x] Kaplan-Meier survival analysis
- [x] Inclusion/exclusion criteria
- [x] Synthetic trajectory generation
- [x] JSON output with KM curves
- [x] Dose-response modeling

### 🔜 Coming Soon
- [ ] Real PBPK+QSP ODE solver integration
- [ ] Overall Survival (OS) endpoint
- [ ] Hazard ratios and log-rank tests
- [ ] Confidence intervals for median PFS
- [ ] FHIR/CQL clinical data export
- [ ] Bayesian power analysis
- [ ] Adaptive trial designs

## Example Workflows

### Dose Selection for Phase IIb

**Scenario**: Select optimal dose from 100mg, 200mg, 400mg

```bash
# Create protocol with 3 arms
cat > dose_selection.medlang << 'EOF'
protocol DoseSelection {
    population model Oncology_Pop
    arms {
        Arm100 { label = "100 mg"; dose = 100.0 }
        Arm200 { label = "200 mg"; dose = 200.0 }
        Arm400 { label = "400 mg"; dose = 400.0 }
    }
    visits { baseline at 0.0; eot at 84.0 }
    endpoints {
        ORR { type = "binary"; observable = TumourVol; shrink_frac = 0.30; window = [0.0, 84.0] }
    }
}
EOF

# Run simulation
simulate_protocol dose_selection.medlang 100

# Decision criteria:
# - ORR ≥30% → Active
# - Highest ORR without plateauing → Optimal dose
```

### Phase IIa Go/No-Go Decision

**Criteria**:
- ORR ≥30% → GO
- Median PFS ≥2 months → GO
- Both criteria met → Proceed to Phase IIb/III

```bash
simulate_protocol phase2a.medlang 200
# Check results:
# - ORR: 58.7% ✓ (>30%)
# - PFS: Not reached ✓ (>2 months)
# Decision: GO to Phase IIb
```

## Technical Details

### Synthetic Data Model

Current implementation uses simple dose-response model:

- **Response quality**: `R = min(dose / 200mg, 1.0)`
- **Shrinkage rate**: `10-60% × R`
- **Growth rate**: `0-5% × (1 - 0.5R)`
- **Heterogeneity**: Random age, ECOG, weight per subject

Future: Replace with full PBPK+QSP ODE solver for mechanistic simulations.

### Endpoint Algorithms

**ORR** (binary):
1. For each subject, find minimum tumor volume in window
2. If min ≤ baseline × (1 - shrink_frac): Responder
3. Compute proportion of responders

**PFS** (time-to-event):
1. Track running minimum (nadir) at each time point
2. Check if current ≥ nadir × (1 + progression_frac)
3. Record first progression time or censor at last visit
4. Compute Kaplan-Meier: S(t) = ∏ (1 - events/at_risk)
5. Find median: first t where S(t) ≤ 0.5

### Files

| File | Purpose |
|------|---------|
| `src/endpoints.rs` | Endpoint evaluation engine |
| `src/parser.rs` | Protocol DSL parser |
| `src/bin/simulate_protocol.rs` | CLI simulator |
| `tests/protocol_parser_test.rs` | Parser tests |
| `docs/week10_time_to_event_endpoints.md` | Technical documentation |

## Testing

```bash
# Run all tests
cargo test --release

# Run protocol tests specifically
cargo test --test protocol_parser_test --release

# Run endpoint tests
cargo test --lib endpoints --release
```

Current test coverage:
- 4 protocol parser tests
- 7 endpoint evaluation tests
- All tests passing ✅

## Support

For questions or issues:
- See `docs/week10_time_to_event_endpoints.md` for technical details
- See `docs/SESSION_COMPLETION_SUMMARY.md` for implementation notes
- Check example protocols in `docs/examples/`

## License

MIT OR Apache-2.0
