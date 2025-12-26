# Release v0.7.0 - MedLang → Sounio Migration Complete

**Release Date**: December 25, 2025  
**Status**: Final Release - Repository Archived  
**DOI**: [To be assigned by Zenodo]

## 🎯 Migration Summary

This is the **final release** of MedLang as a standalone project. MedLang has been successfully unified into the **Sounio Programming Language** as a standard library module.

## ✨ What Changed

### Migration to Sounio

MedLang is now part of Sounio's standard library:

- **New Location**: https://github.com/sounio-lang/sounio
- **Module Path**: `stdlib/medlang/`
- **Import**: `import stdlib.medlang::*`

### Migrated Components

✅ **PK Models** (`stdlib/medlang/pk/`)
- One-compartment models (IV and oral)
- Two-compartment models
- All using `Knowledge<T>` for uncertainty propagation

✅ **Dosing Protocols** (`stdlib/medlang/dose/`)
- Weekly dosing protocols
- Q3W (every 3 weeks) protocols
- Daily oral protocols

✅ **Dosing Policies** (`stdlib/medlang/policy/`)
- FixedDose, ANCBased, TumorResponseBased
- CycleEscalation, TimeBasedReduction

## 🔄 Why the Migration?

Sounio provides **native epistemic computing** with `Knowledge<T>` types, making uncertainty propagation automatic—essential for pharmacometrics where every parameter has confidence intervals.

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
```sio
import stdlib.medlang::pk::one_compartment::*

model OneCompartmentIV {
    param CL: Knowledge<L/h> ~ LogNormal(mean: 10.0 L/h, omega: 0.30)
    param V: Knowledge<L> ~ LogNormal(mean: 50.0 L, omega: 0.25)
    
    compartment Central {
        volume: V
    }
    
    flow Central -> Elimination {
        rate: CL
    }
    
    observe Cp: Concentration = Central.concentration
    // Cp automatically carries propagated uncertainty
}
```

## 📊 Repository Statistics

### MedLang Standalone (Final)

- **Compiler**: 184 Rust files, ~95,000 lines
- **FFI Crate**: 846 lines Rust + 466 lines C header
- **Standard Library**: 23 files (models, protocols, policies, core, ml, rl)
- **Examples**: 22 MedLang examples
- **Documentation**: Comprehensive (spec, guides, examples, dev)
- **Tests**: 127+ tests, 100% pass rate

### Migrated to Sounio

- **PK Models**: 2 files (one/two compartment)
- **Dosing Protocols**: 1 file (3 protocols)
- **Policies**: 1 file (5 policies)
- **Total**: ~300 lines of Sounio code

## 🔗 Links

- **Sounio Repository**: https://github.com/sounio-lang/sounio
- **MedLang Module**: https://github.com/sounio-lang/sounio/tree/main/stdlib/medlang
- **Migration Guide**: [migration/MIGRATION_GUIDE.md](migration/MIGRATION_GUIDE.md)
- **Unification Plan**: [UNIFICATION_PLAN.md](UNIFICATION_PLAN.md)

## 📝 Citation

If you use MedLang in academic work, please cite:

```bibtex
@software{medlang2025,
  title = {MedLang: A Medical-Native Programming Language for Computational Pharmacology},
  author = {Agourakis, Demetrios Chiuratto},
  year = {2025},
  version = {0.7.0},
  url = {https://github.com/agourakis82/medlang},
  note = {Unified into Sounio Programming Language. See https://github.com/sounio-lang/sounio}
}
```

## 🏛️ Sounio Citation

```bibtex
@software{sounio2025,
  title = {Sounio: A Systems Language for Epistemic Computing},
  author = {Agourakis, Demetrios Chiuratto},
  year = {2025},
  url = {https://github.com/sounio-lang/sounio}
}
```

## 📦 Archive Information

This repository is **archived** as of December 25, 2025. All active development continues in the Sounio repository.

**Archive Reason**: Unified into Sounio for better integration with epistemic computing (`Knowledge<T>` types).

**Historical Value**: This repository contains the original Rust-based MedLang prototype (2024-2025), which served as proof-of-concept for the pharmacometric DSL. The code remains available for historical reference.

## 🙏 Acknowledgments

MedLang was developed as a research prototype to explore domain-specific language design for computational pharmacology. The insights gained informed the design of Sounio's MedLang module, which provides native uncertainty propagation—a critical feature for pharmacometric modeling.

---

**MedLang é Sounio. Sounio é epistêmico. Farmacologia com incerteza nativa.** 🏛️🌊

*Final release: v0.7.0 - Migration Complete*

