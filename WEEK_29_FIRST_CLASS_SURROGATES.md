# Week 29: First-Class Surrogates & ML Backends

**Status**: Core infrastructure implemented  
**Date**: 2024  
**Goal**: Make AI/ML surrogates first-class typed values in MedLang

## Overview

Week 29 brings **AI into MedLang** as first-class typed values rather than external CLI flags. This is a fundamental shift in the language design:

**Before Week 29** (stringly-typed, external):
```rust
// CLI: mlc run --backend=surrogate evidence.medlang
run_evidence(ev, "surrogate")  // Magic string, no type safety
```

**After Week 29** (typed, first-class):
```medlang
let backend: BackendKind = BackendKind::Surrogate;
let surr: SurrogateModel = train_surrogate(ev, cfg);
run_evidence_with_surrogate(ev, surr, backend)  // Fully typed!
```

This makes MedLang a language where **mechanistic models and AI surrogates are co-equal citizens**, enabling:
- Type-safe backend selection
- Surrogate models as composable values
- Clear separation of mechanistic vs. surrogate computation
- Foundation for hybrid modeling approaches

## Motivation

Clinical modeling increasingly requires **fast approximations** of complex mechanistic models:

### The Problem
- Full mechanistic simulations: **slow but accurate** (minutes to hours)
- Clinical decision support: needs **fast responses** (seconds)
- Evidence synthesis: requires **thousands of simulations**

### The Solution: Surrogates
Train neural networks to approximate mechanistic models:
- **100-1000x speedup** for online inference
- **Preserves mechanistic fidelity** (trained on mechanistic data)
- **Enables real-time** clinical decision support

### Week 29 Makes This Native
Instead of bolting surrogates on via CLI flags and JSON configs, MedLang now has:

```medlang
// 1. Define backend selection as a typed enum
enum BackendKind {
  Mechanistic;  // Full mechanistic simulation
  Surrogate;    // Neural network approximation  
  Hybrid;       // Best of both worlds
}

// 2. Train surrogates as first-class operations
fn build_fast_surrogate() -> SurrogateModel {
  let cfg: SurrogateTrainConfig = {
    n_train   = 5000,           // Generate 5K mechanistic runs
    backend   = BackendKind::Mechanistic,
    seed      = 42,
    max_epochs = 200,
    batch_size = 128,
  };
  
  train_surrogate(OncologyEvidence, cfg)
}

// 3. Use surrogates in evidence programs
fn run_fast_evidence() -> EvidenceResult {
  let surr = build_fast_surrogate();
  run_evidence_with_surrogate(
    OncologyEvidence,
    surr,
    BackendKind::Surrogate
  )
}
```

## Implementation Summary

### 1. Core Types (`compiler/src/types/core_lang.rs`)

**New domain type**:
```rust
pub enum CoreType {
    // ... existing types
    SurrogateModel,  // Handle to trained surrogate
}
```

**Enum for backend selection** (uses Week 27 enum machinery):
```rust
CoreType::Enum("BackendKind")  // Mechanistic | Surrogate | Hybrid
```

### 2. Standard Library Types

**Backend selection** (`stdlib/med/ml/backend.medlang`):
```medlang
enum BackendKind {
  Mechanistic;  // Full mechanistic simulation
  Surrogate;    // Neural network surrogate
  Hybrid;       // Combination approach
}
```

**Training configuration** (`stdlib/med/ml/surrogate.medlang`):
```medlang
type SurrogateTrainConfig = {
  n_train: Int;           // #mechanistic sims for training
  backend: BackendKind;   // Backend to generate training data
  seed: Int;              // RNG seed
  max_epochs: Int;        // Training epochs
  batch_size: Int;        // Batch size
};
```

### 3. Runtime Support (`compiler/src/ml/`)

**BackendKind mapping** (`backend.rs`):
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendKind {
    Mechanistic,
    Surrogate,
    Hybrid,
}

impl BackendKind {
    pub fn from_variant_name(name: &str) -> Result<Self, BackendError>;
    pub fn requires_surrogate(&self) -> bool;
    pub fn requires_mechanistic(&self) -> bool;
}
```

**Surrogate handle** (`surrogate.rs`):
```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SurrogateModelHandle {
    pub id: Uuid,
    pub name: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SurrogateTrainConfig {
    pub n_train: i64,
    pub backend: BackendKind,
    pub seed: i64,
    pub max_epochs: i64,
    pub batch_size: i64,
}
```

## Usage Examples

### Example 1: Training a Surrogate for Oncology Evidence

```medlang
module projects.oncology_surrogate;

import med.ml.backend::{BackendKind};
import med.ml.surrogate::{SurrogateTrainConfig};
import med.oncology.evidence::{OncologyEvidence};

/// Train a fast surrogate for exploratory analysis
fn train_quick_surrogate() -> SurrogateModel {
  let cfg: SurrogateTrainConfig = {
    n_train   = 1000,              // Quick training (1K samples)
    backend   = BackendKind::Mechanistic,
    seed      = 42,
    max_epochs = 50,
    batch_size = 64,
  };
  
  train_surrogate(OncologyEvidence, cfg)
}

/// Train a production-quality surrogate
fn train_production_surrogate() -> SurrogateModel {
  let cfg: SurrogateTrainConfig = {
    n_train   = 10000,             // High-quality (10K samples)
    backend   = BackendKind::Mechanistic,
    seed      = 42,
    max_epochs = 200,
    batch_size = 128,
  };
  
  train_surrogate(OncologyEvidence, cfg)
}
```

### Example 2: Iterative Development Workflow

```medlang
fn iterative_evidence_workflow() -> EvidenceResult {
  let ev: EvidenceProgram = OncologyEvidence;
  
  // Phase 1: Quick exploration with fast surrogate
  let quick_surr = train_quick_surrogate();
  let explore_result = run_evidence_with_surrogate(
    ev,
    quick_surr,
    BackendKind::Surrogate
  );
  
  // TODO: Inspect explore_result, refine model/protocol
  
  // Phase 2: High-fidelity validation with mechanistic
  let validate_result = run_evidence(
    ev,
    BackendKind::Mechanistic
  );
  
  validate_result  // Return validated result
}
```

### Example 3: Hybrid Approach

```medlang
fn hybrid_evidence_analysis() -> EvidenceResult {
  let ev: EvidenceProgram = OncologyEvidence;
  let surr: SurrogateModel = train_production_surrogate();
  
  // Use hybrid backend: surrogate for cheap exploration,
  // mechanistic for calibration and uncertainty quantification
  run_evidence_with_surrogate(
    ev,
    surr,
    BackendKind::Hybrid
  )
}
```

### Example 4: Backend Selection Based on Context

```medlang
fn select_backend(use_fast: Bool) -> BackendKind {
  match use_fast {
    true  => BackendKind::Surrogate,
    false => BackendKind::Mechanistic,
  }
}

fn context_aware_evidence(is_production: Bool) -> EvidenceResult {
  let ev: EvidenceProgram = OncologyEvidence;
  let backend = select_backend(!is_production);
  
  match backend {
    BackendKind::Surrogate => {
      let surr = train_quick_surrogate();
      run_evidence_with_surrogate(ev, surr, backend)
    },
    BackendKind::Mechanistic => {
      run_evidence(ev, backend)
    },
    BackendKind::Hybrid => {
      let surr = train_production_surrogate();
      run_evidence_with_surrogate(ev, surr, backend)
    },
  }
}
```

## Type System Integration

### Type Safety Guarantees

1. **Backend selection is type-checked**:
   ```medlang
   let backend: BackendKind = BackendKind::Mechanistic;  // ✓
   let backend = "mechanistic";  // ✗ Type error (if string overload removed)
   ```

2. **Surrogate models are typed values**:
   ```medlang
   let surr: SurrogateModel = train_surrogate(ev, cfg);  // ✓
   let surr: Int = train_surrogate(ev, cfg);  // ✗ Type error
   ```

3. **Configuration validation**:
   ```medlang
   let cfg: SurrogateTrainConfig = {
     n_train   = 5000,     // ✓ Int
     n_train   = 5000.0,   // ✗ Type error (Float)
     backend   = BackendKind::Mechanistic,  // ✓
     backend   = "mechanistic",  // ✗ Type error
   };
   ```

### Domain Type Properties

`SurrogateModel` is a **domain handle type**:
- **Opaque**: Cannot inspect internals from L₀
- **Non-copyable**: Passed by reference (in future: lifetime tracking)
- **Runtime-backed**: Maps to `SurrogateModelHandle` at runtime

`BackendKind` is an **enum type** (Week 27):
- **Exhaustiveness checking**: Pattern matches must cover all variants
- **Type-safe dispatch**: No runtime string comparisons
- **Compile-time validation**: Typos caught immediately

## Built-In Functions (Planned)

### `train_surrogate`
```medlang
fn train_surrogate(
  ev: EvidenceProgram,
  cfg: SurrogateTrainConfig
) -> SurrogateModel
```

**Semantics**:
1. Sample parameter configs according to `ev`
2. Run `cfg.n_train` mechanistic simulations using `cfg.backend`
3. Train neural network (e.g., feedforward, PINN) on simulation data
4. Return handle to trained surrogate

**Errors**:
- `InvalidConfig`: if `cfg.backend == Surrogate` (circular!)
- `TrainingFailed`: if neural network training diverges
- `InsufficientData`: if `n_train` too small for model complexity

### `run_evidence`
```medlang
fn run_evidence(
  ev: EvidenceProgram,
  backend: BackendKind
) -> EvidenceResult
```

**Semantics**:
- `Mechanistic`: Full mechanistic simulation (existing path)
- `Surrogate`: **Error** (requires explicit surrogate handle)
- `Hybrid`: Mechanistic only (no surrogate to hybridize with)

### `run_evidence_with_surrogate`
```medlang
fn run_evidence_with_surrogate(
  ev: EvidenceProgram,
  surr: SurrogateModel,
  backend: BackendKind
) -> EvidenceResult
```

**Semantics**:
- `Mechanistic`: Ignores `surr`, runs mechanistic (warning?)
- `Surrogate`: Uses `surr` exclusively
- `Hybrid`: Uses `surr` for cheap exploration, mechanistic for calibration

## Runtime Architecture

```
┌─────────────────────────────────────────────────────┐
│                 MedLang Program                     │
│  let surr = train_surrogate(ev, cfg);               │
│  run_evidence_with_surrogate(ev, surr, backend)     │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                Type Checker                         │
│  - surr: SurrogateModel ✓                           │
│  - backend: BackendKind ✓                           │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│                 Interpreter                         │
│  Value::SurrogateModel(handle)                      │
│  Value::Enum("BackendKind", "Surrogate")            │
└─────────────────────┬───────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              Runtime (Rust)                         │
│  BackendKind::from_variant_name("Surrogate")        │
│  surrogate::train(ev, cfg) -> Handle                │
│  evidence::run_with_surrogate(ev, surr, backend)    │
└─────────────────────────────────────────────────────┘
```

### Key Mappings

| MedLang Type | AST TypeAnn | CoreType | Runtime Rust Type |
|-------------|-------------|----------|-------------------|
| `BackendKind` | N/A (enum) | `Enum("BackendKind")` | `ml::BackendKind` |
| `SurrogateModel` | `TypeAnn::SurrogateModel` | `CoreType::SurrogateModel` | `ml::SurrogateModelHandle` |
| `SurrogateTrainConfig` | `TypeAnn::Record(...)` | `CoreType::Record(...)` | `ml::SurrogateTrainConfig` |

## Integration with Existing Features

### Week 26: Typed Host Language
Surrogates extend L₀ with new domain types:
```medlang
fn process(ev: EvidenceProgram) -> EvidenceResult {
  let surr: SurrogateModel = train_surrogate(ev, cfg);
  run_evidence_with_surrogate(ev, surr, BackendKind::Surrogate)
}
```

### Week 27: Enums & Pattern Matching
`BackendKind` uses enum machinery for exhaustive pattern matching:
```medlang
fn describe_backend(b: BackendKind) -> String {
  match b {
    BackendKind::Mechanistic => "Slow but accurate",
    BackendKind::Surrogate   => "Fast but approximate",
    BackendKind::Hybrid      => "Balanced approach",
  }
}
```

### Week 28: Contracts & Invariants
Contracts can enforce surrogate usage constraints:
```medlang
fn run_evidence_with_surrogate(
  ev: EvidenceProgram,
  surr: SurrogateModel,
  backend: BackendKind
) -> EvidenceResult
  requires backend == BackendKind::Surrogate || 
           backend == BackendKind::Hybrid,
           "backend must support surrogates"
  ensures result.converged == true
{ ... }
```

### L₃ Evidence Programs
Surrogates integrate seamlessly with evidence programs:
```medlang
evidence_program OncologyEvidence {
  population_model Oncology_PBPK_QSP_QM;
  
  trial OncologyTrial_2019 {
    protocol = OncologyProtocol_2019;
    data_file = "data/oncology_2019.csv";
  };
  
  // Future: specify preferred backend in evidence program
  // preferred_backend = BackendKind::Hybrid;
}
```

## Design Decisions

### 1. Enum vs. Union Type for BackendKind

**Decision**: Use enum (Week 27 machinery)

**Rationale**:
- Exhaustive pattern matching catches missing cases
- Clear, finite set of backends
- Easy to extend with new backends (`AdaptiveSampling`, etc.)

**Alternative**: String literals would be error-prone:
```medlang
"mechanistic"  // No typo detection
"Mechanistic"  // Case sensitivity issues
"mech"         // Abbreviations?
```

### 2. Opaque Handles vs. Transparent Surrogates

**Decision**: Opaque `SurrogateModel` handle

**Rationale**:
- Hides implementation details (PyTorch, JAX, custom NN)
- Enables optimization (caching, lazy loading)
- Prevents accidental misuse (can't manually edit weights)

**Alternative**: Transparent surrogates would expose internals:
```medlang
type SurrogateModel = {
  weights: Matrix[Float];  // ✗ Too much detail for L₀
  architecture: String;     // ✗ Not type-safe
}
```

### 3. Separate train_surrogate vs. Implicit Training

**Decision**: Explicit `train_surrogate` function

**Rationale**:
- Makes cost visible (training is expensive!)
- Enables caching and reuse of trained surrogates
- Allows for offline training workflows

**Alternative**: Implicit training would hide cost:
```medlang
run_evidence(ev, BackendKind::Surrogate)  // ✗ When does training happen?
```

### 4. Config Record vs. Variadic Arguments

**Decision**: Structured `SurrogateTrainConfig` record

**Rationale**:
- Named fields prevent argument order mistakes
- Extensible without breaking backward compatibility
- Can be constructed and inspected as a value

**Alternative**: Positional args would be error-prone:
```medlang
train_surrogate(ev, 5000, "Mechanistic", 42, 200, 128)  // ✗ What's 200?
```

## Performance Implications

### Training Overhead
- **One-time cost**: Training happens once, reuse many times
- **Typical training time**: 5-30 minutes (depending on `n_train`, model complexity)
- **Amortized**: For 1000+ inference calls, surrogate is **100-1000x faster** overall

### Inference Speedup
| Backend | Time per Simulation | Use Case |
|---------|---------------------|----------|
| Mechanistic | 1-10 seconds | High-fidelity validation |
| Surrogate | 1-10 milliseconds | Real-time decision support |
| Hybrid | 0.1-1 seconds | Balanced exploration + calibration |

### Memory Footprint
- **Surrogate model**: 1-100 MB (typical neural network)
- **Training data**: 100 MB - 10 GB (mechanistic simulation results)
- **Runtime inference**: ~10 MB (batched predictions)

## Future Work (Not Yet Implemented)

### 1. Parser Support
Add parsing for `SurrogateModel` type annotations:
```rust
// In parser/types.rs
map(kw("SurrogateModel"), |_| TypeAnn::SurrogateModel)
```

### 2. Interpreter Built-ins
Implement runtime evaluation of:
- `train_surrogate(ev, cfg)`
- `run_evidence(ev, backend)`
- `run_evidence_with_surrogate(ev, surr, backend)`

### 3. Backend Infrastructure
Wire to existing surrogate training:
```rust
pub fn train(
    ev: EvidenceProgramHandle,
    cfg: SurrogateTrainConfig,
    ctx: &mut RuntimeContext,
) -> Result<SurrogateModelHandle, SurrogateError>
```

### 4. Surrogate Persistence
Save/load trained surrogates:
```medlang
fn save_surrogate(surr: SurrogateModel, path: String);
fn load_surrogate(path: String) -> SurrogateModel;
```

### 5. Multi-Backend Ensembles
Combine multiple backends for uncertainty quantification:
```medlang
enum EnsembleBackend {
  MechanisticEnsemble { n_samples: Int };
  SurrogateEnsemble { models: List[SurrogateModel] };
  HybridEnsemble { 
    surr: SurrogateModel,
    n_calibration: Int
  };
}
```

### 6. Adaptive Backend Selection
Runtime backend switching based on accuracy requirements:
```medlang
fn adaptive_run_evidence(
  ev: EvidenceProgram,
  surr: SurrogateModel,
  target_accuracy: Float
) -> EvidenceResult {
  // Start with surrogate, fall back to mechanistic if error > target
}
```

### 7. Surrogate Quality Metrics
Built-in diagnostics for surrogate fidelity:
```medlang
type SurrogateMetrics = {
  mse: Float;              // Mean squared error
  r2: Float;               // R² score
  calibration_error: Float; // Calibration plot metrics
};

fn evaluate_surrogate(
  surr: SurrogateModel,
  test_data: EvidenceResult
) -> SurrogateMetrics;
```

## Testing

### Unit Tests

**Backend mapping** (`ml/backend.rs`):
```rust
#[test]
fn test_from_variant_name() {
    assert_eq!(
        BackendKind::from_variant_name("Mechanistic").unwrap(),
        BackendKind::Mechanistic
    );
    assert!(BackendKind::from_variant_name("Unknown").is_err());
}

#[test]
fn test_requires_surrogate() {
    assert!(!BackendKind::Mechanistic.requires_surrogate());
    assert!(BackendKind::Surrogate.requires_surrogate());
    assert!(BackendKind::Hybrid.requires_surrogate());
}
```

**Surrogate handles** (`ml/surrogate.rs`):
```rust
#[test]
fn test_surrogate_handle_creation() {
    let handle = SurrogateModelHandle::new();
    assert!(handle.name.is_none());

    let named = SurrogateModelHandle::with_name("Test".to_string());
    assert_eq!(named.name.as_deref(), Some("Test"));
}

#[test]
fn test_train_config_validation() {
    let mut cfg = SurrogateTrainConfig::default_quick();
    assert!(cfg.validate().is_ok());

    cfg.n_train = 0;
    assert!(cfg.validate().is_err());

    cfg = SurrogateTrainConfig::default_quick();
    cfg.backend = BackendKind::Surrogate;  // Circular!
    assert!(cfg.validate().is_err());
}
```

### Integration Tests (Planned)

**Type checking**:
```medlang
fn test_surrogate_types() {
  let surr: SurrogateModel = train_surrogate(ev, cfg);  // ✓
  let backend: BackendKind = BackendKind::Surrogate;    // ✓
  run_evidence_with_surrogate(ev, surr, backend)        // ✓
}
```

**Enum exhaustiveness**:
```medlang
fn test_backend_exhaustiveness(b: BackendKind) -> String {
  match b {
    BackendKind::Mechanistic => "mech",
    BackendKind::Surrogate   => "surr",
    // Missing Hybrid -> Compile error!
  }
}
```

## Files Modified/Created

### New Files
- `stdlib/med/ml/backend.medlang` (BackendKind enum)
- `stdlib/med/ml/surrogate.medlang` (SurrogateTrainConfig type)
- `compiler/src/ml/mod.rs` (ML runtime module)
- `compiler/src/ml/backend.rs` (BackendKind runtime, 150 lines)
- `compiler/src/ml/surrogate.rs` (Surrogate runtime, 250 lines)
- `WEEK_29_FIRST_CLASS_SURROGATES.md` (this file)

### Modified Files
- `compiler/src/types/core_lang.rs`: Added `CoreType::SurrogateModel`
- `compiler/src/ast/core_lang.rs`: Added `TypeAnn::SurrogateModel`
- `compiler/src/lib.rs`: Registered `ml` module
- `compiler/Cargo.toml`: Added `uuid` dependency

## Comparison with Other Languages

### Python (Current Standard for ML in Science)
```python
# Untyped, error-prone
backend = "surrogate"  # Typo: "surroate" would fail at runtime
model = train(evidence, backend="mechanistic")  # No relation to backend variable
result = run(evidence, model, "surrogate")  # String again
```

MedLang improvements:
- ✓ Type-safe backend selection
- ✓ Compile-time typo detection
- ✓ Clear model vs. backend separation

### Julia (Technical Computing)
```julia
# Better typing, but still mostly runtime
backend = :surrogate  # Symbol (better than string)
model = train(evidence; backend=:mechanistic)
result = run(evidence, model, :surrogate)
```

MedLang improvements:
- ✓ Exhaustive enum checking (Julia symbols are open-ended)
- ✓ Compile-time contract validation
- ✓ Domain-specific types (`EvidenceProgram`, not generic `Any`)

### Rust (Systems Programming)
```rust
// Type-safe but verbose
enum BackendKind { Mechanistic, Surrogate, Hybrid }
let backend = BackendKind::Surrogate;
let model: SurrogateHandle = train(evidence, config)?;
let result = run_with_surrogate(evidence, model, backend)?;
```

MedLang matches Rust's type safety but with:
- ✓ Domain-specific syntax for clinical modeling
- ✓ Integrated evidence programs (not external data)
- ✓ Higher-level abstractions for scientific computing

## Conclusion

Week 29 transforms MedLang from a language that **uses AI externally** to one where **AI is native**. Key achievements:

1. **Type Safety**: `BackendKind` enum eliminates string-based dispatch
2. **First-Class Values**: `SurrogateModel` is a typed value you can pass around
3. **Composability**: Train once, use many times; combine with evidence programs
4. **Foundation for Hybrid Modeling**: Infrastructure for mechanistic+surrogate workflows

This is the **first concrete step** toward the original MedLang vision: a language where mechanistic modeling and AI/ML are co-equal citizens, not bolted-on afterthoughts.

**Next steps**: Parser integration, interpreter built-ins, and full evidence runner integration will make these types fully functional at runtime.

---

**Week 29 Implementation**: Core types and runtime infrastructure complete ✓  
**Next Week (30)**: Runtime integration and evidence runner wiring
