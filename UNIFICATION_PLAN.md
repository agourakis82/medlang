# MedLang → Sounio Unification Plan

## Decision

**MedLang is being unified into Sounio as `stdlib/medlang/`**

MedLang standalone repository will be archived after migration.

---

## Inventory Summary

### MedLang Standalone

- **Compiler**: 184 Rust files, ~95K lines
- **Runtime**: 1 file (placeholder)
- **FFI**: Complete C ABI (846 lines Rust + 466 lines C header)
- **Standard Library**: 
  - `medlang_std/`: 4 files (models, protocols, policies)
  - `stdlib/med/`: 19 files (core, ml, rl modules)
- **Examples**: 22 MedLang examples
- **Documentation**: Comprehensive (spec, guides, examples, dev)

### Key Components to Migrate

| Component | Action | Priority |
|-----------|--------|----------|
| `medlang_std/models/` | Migrate to `sounio/stdlib/medlang/pk/` | High |
| `medlang_std/protocols/` | Migrate to `sounio/stdlib/medlang/dose/` | High |
| `medlang_std/policies/` | Migrate to `sounio/stdlib/medlang/policy/` | High |
| `stdlib/med/core/` | Integrate with Sounio core | Medium |
| `stdlib/med/ml/` | Integrate with Sounio ML | Medium |
| `stdlib/med/rl/` | Integrate with Sounio RL | Medium |
| `docs/examples/` | Convert to `.sio` examples | High |
| `docs/spec/` | Reference documentation | Low |
| `compiler/` | Evaluate for useful features | Low |
| `ffi/` | May be useful for Sounio FFI | Medium |
| `beagle/` | Separate project | N/A |

---

## Target Structure in Sounio

```
sounio/stdlib/medlang/
├── mod.sio              # Main module exports
├── model.sio            # model DSL syntax
├── param.sio            # Parameter types with Knowledge<T>
├── compartment.sio      # Compartment definitions
├── flow.sio             # Flow definitions
├── dose.sio             # Dosing regimens
├── observe.sio          # Observations/endpoints
├── pk/
│   ├── mod.sio
│   ├── one_compartment.sio
│   ├── two_compartment.sio
│   └── three_compartment.sio
├── pd/
│   ├── mod.sio
│   ├── emax.sio
│   ├── sigmoid.sio
│   └── indirect.sio
├── pbpk/
│   ├── mod.sio
│   ├── tissue.sio
│   ├── liver.sio
│   ├── kidney.sio
│   └── brain.sio
├── population/
│   ├── mod.sio
│   ├── mixed_effects.sio
│   ├── covariate.sio
│   └── iiv.sio
├── simulation/
│   ├── mod.sio
│   ├── monte_carlo.sio
│   ├── vpc.sio
│   └── bootstrap.sio
├── estimation/
│   ├── mod.sio
│   ├── foce.sio
│   ├── saem.sio
│   └── bayesian.sio
└── nonmem/
    ├── mod.sio
    ├── parser.sio
    ├── translator.sio
    └── compat.sio
```

---

## Migration Steps

### Phase 1: Preparation ✅
- [x] Inventory MedLang standalone
- [x] Document unification plan
- [x] Create archive notice

### Phase 2: Code Migration
- [ ] Create structure in Sounio
- [ ] Migrate `medlang_std/` models
- [ ] Migrate `medlang_std/` protocols
- [ ] Migrate `medlang_std/` policies
- [ ] Convert examples to `.sio`

### Phase 3: Integration
- [ ] Update Sounio README
- [ ] Update Sounio CHANGELOG
- [ ] Add MedLang to Sounio docs
- [ ] Test compilation

### Phase 4: Archive
- [ ] Update MedLang README with archive notice
- [ ] Archive repository on GitHub
- [ ] Update all links

---

## Key Advantages of Sounio Integration

1. **Native Epistemic Computing**: `Knowledge<T>` types for automatic uncertainty propagation
2. **Unified Language**: No need for separate compiler
3. **Better Type System**: Sounio's refinement types + MedLang's dimensional analysis
4. **Single Ecosystem**: All pharmacometric code in one place

---

## Example: MedLang → Sounio

### Before (MedLang standalone)
```medlang
model OneComp {
    param CL : Clearance
    param V : Volume
    state A_central : DoseMass
    dA_central/dt = -(CL / V) * A_central
    obs C_plasma : ConcMass = A_central / V
}
```

### After (Sounio)
```sounio
import sounio::medlang::*

model OneComp {
    // Uncertainty is automatic with Knowledge<T>
    param CL: Knowledge<L/h> ~ LogNormal(10.0, omega: 0.30)
    param V: Knowledge<L> ~ LogNormal(50.0, omega: 0.25)
    
    compartment Central { volume: V }
    flow Central -> Elimination: CL
    
    observe Cp = Central.concentration
    // Cp automatically carries propagated uncertainty
}
```

---

**Status**: Planning phase complete. Ready for migration.

