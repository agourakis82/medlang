# MedLang Phase V1 Quick Reference

## Effect System

### Effect Types
```medlang
Pure        // No side effects (default)
Prob        // Probabilistic/stochastic operations
IO          // Input/output operations  
GPU         // GPU-accelerated computations
```

### Syntax
```medlang
// Single effect
population MyModel with Prob { ... }

// Multiple effects
population LargeSim with GPU, Prob { ... }

// I/O effect
cohort MyCohort with IO {
    data_file "data.csv"
}
```

### Rules
- Pure functions can only call pure functions
- Effectful functions can call pure functions
- Effect sets are combined with union
- Violations detected at compile time

---

## Epistemic Computing

### Knowledge Wrapper
```medlang
Knowledge<T> {
    value: T           // Actual value
    confidence: f64    // [0.0, 1.0]
    provenance: ...    // Source tracking
}
```

### Provenance Types
```medlang
Measurement { source, timestamp, subject_id }
Computed { operation, inputs }
Imputed { method, original_missing }
Estimated { model, method, iterations }
Literature { citation, population }
Synthetic { generator, seed }
```

### Syntax
```medlang
input DOSE : Knowledge<DoseMass> {
    value = 100.0_mg,
    confidence = 0.95,
    provenance = Measurement {
        source = "pharmacy_record",
        timestamp = "2024-12-06T10:00:00Z",
        subject_id = "SUBJ001"
    }
}
```

### Confidence Propagation

| Operation | Rule |
|-----------|------|
| `a + b`, `a - b`, `a * b`, `a / b` | `min(conf_a, conf_b)` |
| `a ^ b` | `conf_a * conf_b` |
| `exp(a)`, `ln(a)` | `conf_a * 0.95` |
| `mean([a, b, c])` | `(conf_a + conf_b + conf_c) / 3` |
| `min([a, b, c])` | `min(conf_a, conf_b, conf_c)` |

---

## Refinement Types

### Constraint Operators
```medlang
==  !=  <  <=  >  >=     // Comparison
&&  ||                   // Logical
```

### Clinical Constraints
```medlang
// Positive values
param CL : Clearance where CL > 0.0_L_per_h
param V : Volume where V > 0.0_L
param Ka : RateConst where Ka > 0.0_per_h

// Ranges
param AGE : f64 where AGE >= 0.0 && AGE <= 120.0
param WT : Mass where WT >= 0.5_kg && WT <= 300.0_kg
param p : f64 where p >= 0.0 && p <= 1.0

// Range syntax
param AGE : f64 where 0.0 <= AGE <= 120.0

// Complex constraints
param CrCL : Quantity<Volume/Time> 
    where CrCL > 0.0_mL_per_min && CrCL <= 300.0_mL_per_min
```

### Built-in Clinical Refinements

From `ClinicalRefinements` module:
```rust
positive_clearance(var)     // CL > 0
positive_volume(var)        // V > 0  
positive_rate(var)          // Ka > 0
positive_dose(var)          // DOSE > 0
human_age(var)              // 0 ≤ AGE ≤ 120
body_weight(var)            // 0.5 ≤ WT ≤ 300 kg
proportion(var)             // 0 ≤ p ≤ 1
creatinine_clearance(var)   // 0 < CrCL ≤ 300 mL/min
```

---

## Complete Example

```medlang
// Model with safety constraints
model OneCompOral {
    state A_gut : DoseMass
    state A_central : DoseMass
    
    // Refinement types ensure safety
    param Ka : RateConst where Ka > 0.0_per_h
    param CL : Clearance where CL > 0.0_L_per_h
    param V : Volume where V > 0.0_L
    
    dA_gut/dt = -Ka * A_gut
    dA_central/dt = Ka * A_gut - (CL / V) * A_central
    
    // Division safe: V proven > 0
    obs C_plasma : ConcMass = A_central / V
}

// Population with probabilistic effects
population OneCompPop with Prob {
    model OneCompOral
    
    param CL_pop : Clearance where CL_pop > 0.0_L_per_h
    param V_pop : Volume where V_pop > 0.0_L
    param Ka_pop : RateConst where Ka_pop > 0.0_per_h
    
    param omega_CL : f64 where omega_CL >= 0.0
    param omega_V : f64 where omega_V >= 0.0
    
    // Epistemic input with confidence
    input WT : Knowledge<Mass> 
        where WT.value >= 0.5_kg && WT.value <= 300.0_kg
    
    // Probabilistic random effects
    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    rand eta_V : f64 ~ Normal(0.0, omega_V)
    
    bind_params(patient) {
        let w = patient.WT.value / 70.0_kg
        
        model.CL = CL_pop * pow(w, 0.75) * exp(eta_CL)
        model.V = V_pop * w * exp(eta_V)
        model.Ka = Ka_pop
    }
}

// Cohort with I/O effect
cohort OneCompCohort with IO {
    population OneCompPop
    data_file "data/onecomp_synth.csv"
}
```

---

## Testing

```bash
# Test all new features
cargo test effects epistemic clinical

# Test specific module
cargo test effects::
cargo test epistemic::
cargo test refinement::clinical::

# Show test output
cargo test -- --nocapture
```

---

## Compilation

```bash
# Check effects and constraints
mlc check model.medlang

# Verbose output shows effect checking
mlc check model.medlang -v

# Compile with all checks
mlc compile model.medlang --backend stan
```

---

## Error Messages

### Effect Violations
```
Error: Pure function 'calculate_dose' cannot call effectful function 'sample'
  with effects: Prob
  
  Help: Add 'with Prob' annotation to 'calculate_dose'
```

### Confidence Threshold Violations
```
Error: Confidence threshold violation
  Context: dose_adjustment requires confidence >= 0.85
  Actual: 0.72
  
  Help: Repeat measurement or use more reliable data source
```

### Constraint Violations
```
Error: Constraint violation: CL > 0.0
  Context: parameter CL = -5.0
  
  Help: Clearance must be positive (physiological requirement)
```

---

## Best Practices

### 1. Effect Annotations
- Mark all probabilistic models with `with Prob`
- Mark data-loading cohorts with `with IO`
- Mark large simulations with `with GPU`

### 2. Epistemic Values
- Use high confidence (>0.9) for direct measurements
- Use medium confidence (0.7-0.9) for imputed values
- Use low confidence (<0.7) for estimated parameters
- Always track provenance for regulatory compliance

### 3. Refinement Types
- Always constrain clearance, volume, rates to be positive
- Add physiological bounds to age, weight, renal function
- Use range constraints for proportions and probabilities
- Leverage built-in clinical refinements from `ClinicalRefinements`

---

## Module Locations

```
compiler/src/effects.rs              # Effect system
compiler/src/epistemic.rs            # Epistemic computing
compiler/src/refinement/clinical.rs  # Clinical refinements
```

---

## Documentation

- Full guide: `docs/PHASE_V1_ENHANCEMENTS.md`
- Summary: `REAL_IMPROVEMENTS_SUMMARY.md`
- API docs: `cargo doc --open`
