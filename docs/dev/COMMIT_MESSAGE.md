feat: Phase V1 - Demetrios-Inspired Enhancements (Effect System, Epistemic Computing, Clinical Refinements)

## Summary

This commit adds three major features to MedLang inspired by the Demetrios language,
adapted for medical/clinical computing:

1. **Effect System** - Algebraic effects for tracking side effects (Prob, IO, GPU, Pure)
2. **Epistemic Computing** - Knowledge<T> wrapper for confidence tracking and provenance
3. **Clinical Refinement Types** - Medical-specific constraint predicates

Total: 1,680 LOC + 24 comprehensive tests
Status: ✅ All tests passing, release build successful

## New Files

- `compiler/src/effects.rs` (450 LOC, 8 tests)
  * Effect types: Pure, Prob, IO, GPU
  * EffectSet and EffectChecker for compile-time validation
  * Effect subsumption checking (Pure cannot call Prob, etc.)

- `compiler/src/epistemic.rs` (580 LOC, 10 tests)
  * Knowledge<T> wrapper with value, confidence, provenance
  * Provenance types: Measurement, Computed, Imputed, Estimated, Literature, Synthetic
  * Automatic confidence propagation through arithmetic operations
  * Propagation rules for binary ops, unary ops, aggregations

- `compiler/src/refinement/clinical.rs` (650 LOC, 6 tests)
  * Clinical constraint predicates (positive clearance, age ranges, etc.)
  * Built-in ClinicalRefinements module with common constraints
  * Constraint checker for runtime/compile-time verification

## Modified Files

- `compiler/src/lib.rs`
  * Added `pub mod effects;`
  * Added `pub mod epistemic;`
  * Updated refinement module comment

- `compiler/src/refinement/mod.rs`
  * Added `pub mod clinical;` submodule

## Documentation

- `docs/PHASE_V1_ENHANCEMENTS.md` - Comprehensive 400+ line guide
- `docs/QUICK_REFERENCE_V1.md` - Quick reference card for new features
- `REAL_IMPROVEMENTS_SUMMARY.md` - Executive summary with stats

## Testing

```bash
cargo test effects::     # 8 tests passing
cargo test epistemic::   # 10 tests passing
cargo test clinical::    # 6 tests passing
cargo build --release    # ✅ Success
```

All existing tests (103) continue to pass.

## Clinical Benefits

1. **Reproducibility**: Effect system ensures probabilistic operations are tracked
2. **Uncertainty Quantification**: Epistemic computing propagates measurement confidence
3. **Safety Verification**: Refinement types prevent runtime errors (division by zero, negative parameters)

## Comparison with Demetrios

### Adopted Features
- ✅ Effect system architecture (Prob, IO, GPU)
- ✅ Epistemic computing (Knowledge<T>)
- ✅ Refinement type patterns

### Enhanced for Medical Domain
- ✅ Clinical provenance types (Measurement, Imputed, Estimated)
- ✅ Pharmacometric-specific constraints (positive CL/V, age/weight ranges)
- ✅ Regulatory compliance (FDA/EMA data tracking)

### Maintained MedLang Identity
- ✅ M·L·T dimensional analysis (superior to generic units)
- ✅ NLME population models (no Demetrios equivalent)
- ✅ Clinical timeline DSL
- ✅ Stan/Julia backends

## Example Usage

```medlang
// Effect annotation
population OneCompPop with Prob {
    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
}

// Epistemic value
input DOSE : Knowledge<DoseMass> {
    value = 100.0_mg,
    confidence = 0.95,
    provenance = Measurement { source = "pharmacy", ... }
}

// Refinement type
param CL : Clearance where CL > 0.0_L_per_h
param AGE : f64 where AGE >= 0.0 && AGE <= 120.0
```

## Next Steps (Phase V2)

- Z3 SMT solver integration for full refinement type checking
- JIT compilation (Cranelift) for interactive REPL
- GPU kernel code generation (CUDA/PTX)
- Enhanced LSP with effect/confidence hover info

## References

- Demetrios Language: https://github.com/chiuratto-AI/demetrios
- Effect Systems: Plotkin & Power (2003)
- Refinement Types: Knowles & Flanagan (2010)

---

Co-authored-by: Claude (Anthropic)
Inspired-by: Demetrios Language
