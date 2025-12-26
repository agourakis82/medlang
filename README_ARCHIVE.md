# MedLang (Archived)

> ⚠️ **This repository has been archived.**
> 
> MedLang has been unified into the **Sounio Programming Language** as a 
> standard library module.

## New Location

MedLang is now part of Sounio:

- **Repository:** [sounio-lang/sounio](https://github.com/sounio-lang/sounio)
- **Module:** `stdlib/medlang/`
- **Import:** `import sounio::medlang::*`

## Why the Change?

Sounio provides native epistemic computing with `Knowledge<T>` types, making 
uncertainty propagation automatic — essential for pharmacometrics where every 
parameter has confidence intervals.

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

## Historical Code

This repository contains the original Rust-based MedLang prototype developed
during 2024-2025. The code remains available for historical reference but is
no longer maintained.

**Key Statistics:**
- 184 Rust files (~95K lines) in compiler
- Complete FFI crate (C ABI)
- 22+ examples
- Comprehensive documentation

## Migration Guide

| MedLang (old) | Sounio (new) |
|---------------|--------------|
| `medlang run model.med` | `souc run model.sio` |
| `import pk::*` | `import sounio::medlang::pk::*` |
| `.med` files | `.sio` files |
| `param CL : Clearance` | `param CL: Knowledge<L/h> ~ LogNormal(...)` |

## Repository Structure (Historical)

```
medlang/
├── compiler/          # Rust compiler (184 files, ~95K lines)
├── runtime/           # Runtime layer (placeholder)
├── ffi/               # C ABI interface
├── medlang_std/       # Standard library (models, protocols, policies)
├── stdlib/med/        # Advanced modules (core, ml, rl)
├── docs/              # Comprehensive documentation
└── examples/          # 22+ MedLang examples
```

## What Was Migrated

- ✅ Standard PK/PD models → `sounio/stdlib/medlang/pk/`
- ✅ Dosing protocols → `sounio/stdlib/medlang/dose/`
- ✅ Policies → `sounio/stdlib/medlang/policy/`
- ✅ Examples → Converted to `.sio` format
- ✅ Documentation → Integrated into Sounio docs

## Links

- **Sounio Repository**: https://github.com/sounio-lang/sounio
- **Sounio Documentation**: See Sounio repo for MedLang module docs
- **Unification Plan**: [UNIFICATION_PLAN.md](UNIFICATION_PLAN.md)

---

**Sounio — Compute at the Horizon of Certainty** 🏛️🌊

*MedLang é Sounio. Sounio é epistêmico. Farmacologia com incerteza nativa.*

*Author: Demetrios Chiuratto Agourakis*

