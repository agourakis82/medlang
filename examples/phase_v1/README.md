# Phase V1 Examples

This directory contains example MedLang programs demonstrating the three major Phase V1 features inspired by the Demetrios language:

1. **Effect System** - Track computational side effects (Pure, Prob, IO, GPU)
2. **Epistemic Computing** - Represent uncertainty with `Knowledge<T>` types
3. **Refinement Types** - Add compile-time safety constraints with `where` clauses

## Examples

### 1. `epistemic_pk_model.medlang`

**Focus**: Epistemic Types (`Knowledge<T>`)

Demonstrates how to represent measurements with associated confidence levels in a pharmacokinetic model.

**Key Features**:
- `Knowledge<Mass>(0.95)` - Patient weight with 95% confidence
- `Knowledge<ConcMass>(0.80)` - Plasma concentration with 80% confidence
- `Knowledge<Clearance>(0.70)` - Estimated clearance with 70% confidence
- Automatic confidence propagation through calculations

**Clinical Use Case**: Handling varying measurement quality in population PK analysis.

**Run**:
```bash
mlc compile epistemic_pk_model.medlang
```

---

### 2. `refinement_safety.medlang`

**Focus**: Refinement Types (`where` constraints)

Demonstrates compile-time safety for common clinical constraints.

**Key Features**:
- `where CL > 0.0` - Prevents division by zero
- `where WT in 30.0_kg..200.0_kg` - Physiological bounds
- `where AGE in 0.0..120.0` - Human lifespan constraint
- `where DOSE in 10.0_mg..1000.0_mg` - Therapeutic window

**Clinical Use Case**: Ensuring all parameters are physiologically valid before model execution.

**Run**:
```bash
mlc compile refinement_safety.medlang
```

---

### 3. `effect_system_mcmc.medlang`

**Focus**: Effect System (`with` annotations)

Demonstrates tracking computational side effects in MCMC workflows.

**Key Features**:
- `with Pure` - Deterministic, no side effects
- `with Prob` - Probabilistic sampling
- `with IO` - File I/O operations
- `with GPU` - GPU computation
- Effect subsumption rules (Prob can call Pure)

**Clinical Use Case**: Explicit tracking of non-determinism and I/O in pharmacometric workflows.

**Run**:
```bash
mlc compile effect_system_mcmc.medlang
```

---

### 4. `combined_features.medlang` ⭐

**Focus**: All Three Features Combined

Comprehensive example showing how all Phase V1 features work together in a production-quality population PK model.

**Key Features**:
- **Effects**: `with Pure`, `with Prob`, `with IO`, `with Prob | IO`
- **Epistemic**: `Knowledge<Mass>(0.95)`, `Knowledge<ConcMass>(0.80)`
- **Refinements**: `where CL > 0.0`, `where WT in 30.0_kg..200.0_kg`

**Clinical Use Case**: Full population PK analysis with:
- Safety constraints (refinements)
- Measurement uncertainty (epistemic types)
- Workflow tracking (effect system)

**Run**:
```bash
mlc compile combined_features.medlang
```

---

## Syntax Quick Reference

### Effect Annotations

```medlang
// Pure function (no side effects)
fn calculate_dose(weight: f64) : f64 with Pure { ... }

// Probabilistic function
fn sample_prior() : f64 with Prob { ... }

// I/O function
fn load_data(file: string) : Data with IO { ... }

// Multiple effects
fn run_mcmc(data: Data) : Results with Prob | IO { ... }

// GPU computation
fn matrix_mult(A: Matrix, B: Matrix) : Matrix with GPU { ... }
```

### Epistemic Types

```medlang
// Knowledge type without confidence requirement
param measured_value : Knowledge<f64>

// Knowledge type with minimum confidence
param weight : Knowledge<Mass>(0.95)  // Requires ≥95% confidence

// Knowledge type in expressions
let combined : Knowledge<f64> = knowledge_a + knowledge_b
// Confidence propagates: min(conf_a, conf_b)
```

### Refinement Constraints

```medlang
// Positive value constraint
param CL : Clearance where CL > 0.0

// Range constraint
param AGE : f64 where AGE in 18.0..85.0

// Combined constraints
param V : Volume where V > 0.0 && V < 1000.0_L

// With unit literals
param DOSE : Mass where DOSE > 10.0_mg && DOSE < 500.0_mg
```

### Combined Features

```medlang
// Parameter with epistemic type AND refinement
param WT : Knowledge<Mass>(0.90) where WT in 30.0_kg..200.0_kg

// Function with effects returning epistemic type
fn analyze_data(file: string) : Knowledge<f64> with Prob | IO {
    // Load data (IO)
    let data = load_file(file)
    
    // Sample (Prob)
    let result ~ sample_posterior(data)
    
    // Return epistemic value
    Knowledge { value: result, confidence: 0.85 }
}

// Function with refinement on parameters
fn calculate(x: f64, y: f64) : f64 with Pure 
where x > 0.0 && y > 0.0 {
    x / y  // Safe: both x and y proven > 0
}
```

---

## Benefits of Phase V1 Features

### 1. Safety (Refinement Types)
- **Compile-time prevention** of division by zero
- **Physiological bounds** enforced before execution
- **Therapeutic windows** validated in type system
- **No runtime crashes** from invalid parameters

### 2. Uncertainty Quantification (Epistemic Types)
- **Explicit confidence** for all measurements
- **Automatic propagation** through calculations
- **Quality tracking** from raw data to final predictions
- **Regulatory compliance** with uncertainty documentation

### 3. Reproducibility (Effect System)
- **Deterministic functions** clearly marked (`Pure`)
- **Non-deterministic code** explicitly tracked (`Prob`)
- **I/O operations** visible in function signatures
- **Resource requirements** (GPU) declared upfront

### 4. Clinical Impact
- **Safer models**: Impossible to create physiologically invalid scenarios
- **Better science**: Uncertainty tracked from measurement to conclusion
- **Easier debugging**: Effects tell you what a function can do
- **Regulatory approval**: All assumptions encoded in types

---

## Comparison: V0 vs V1

### V0 (Traditional)
```medlang
param CL : Clearance           // No safety check
input WT : Mass                // No confidence tracking
fn analyze(data) { ... }       // Unknown side effects
```

**Problems**:
- CL could be zero (runtime crash)
- WT quality unknown (could be estimated, measured, or imputed)
- analyze() might do I/O, sampling, or both (can't tell from signature)

### V1 (With Phase V1 Features)
```medlang
param CL : Clearance where CL > 0.0           // Safe: guaranteed > 0
input WT : Knowledge<Mass>(0.95)              // Quality: 95% confidence
fn analyze(data) : Results with Prob | IO {  // Effects: Prob + IO
    ...
}
```

**Benefits**:
- CL > 0 proven at compile time
- WT confidence explicit and propagated
- analyze() effects visible (needs seed for reproducibility, handles I/O errors)

---

## Integration with Existing MedLang Features

Phase V1 features work seamlessly with V0 features:

- **M·L·T Dimensional Analysis**: Refinements can use units (`where CL > 0.0_L_per_h`)
- **NLME Models**: Epistemic types track measurement uncertainty in population models
- **Stan/Julia Backends**: Effects guide code generation (Pure → pure functions, IO → error handling)

---

## Future Enhancements (Phase V2)

- **Z3 SMT Solver**: Prove complex refinements (e.g., `CL / V < threshold`)
- **Effect Inference**: Automatically infer effects from function bodies
- **Confidence Inference**: Automatically compute confidence from provenance
- **Linear Types**: Prevent resource leaks with affine types
- **Dependent Types**: Express relationships between values (e.g., `array[n]` for `n : Nat`)

---

## Testing

Run Phase V1 integration tests:
```bash
cd compiler
cargo test --test phase_v1_integration
```

Run Phase V1 module tests:
```bash
cargo test parser_v1 typeck_v1
```

---

## Documentation

- **`PHASE_V1_INTEGRATION_COMPLETE.md`** - Full integration report
- **`docs/PHASE_V1_ENHANCEMENTS.md`** - Technical specification
- **`docs/QUICK_REFERENCE_V1.md`** - Quick reference card

---

## Contributing

When adding new Phase V1 examples:

1. Focus on clinical use cases
2. Include both syntax examples and explanations
3. Show benefits over V0 approach
4. Add comments explaining compile-time guarantees
5. Test with `mlc compile <file>.medlang`

---

## Questions?

See the main project documentation or open an issue at:
https://github.com/agourakis82/medlang/issues
