# MedLang → Sounio Migration Package

This directory contains files prepared for migration to Sounio.

## Structure

```
migration/
├── README.md                    # This file
├── MIGRATION_GUIDE.md          # Detailed migration guide
└── sounio-structure/           # Proposed Sounio structure
    └── stdlib/
        └── medlang/
            ├── mod.sio         # Main module
            ├── pk/             # PK models
            ├── pd/             # PD models (to be created)
            ├── pbpk/           # PBPK models (to be created)
            ├── population/     # Population models (to be created)
            ├── simulation/     # Simulation tools (to be created)
            ├── estimation/     # Estimation methods (to be created)
            ├── nonmem/         # NONMEM compatibility (to be created)
            ├── dose/           # Dosing protocols
            └── policy/         # Dosing policies
```

## What's Included

### ✅ Migrated
- `pk/one_compartment.sio` - One-compartment PK models
- `pk/two_compartment.sio` - Two-compartment PK models
- `dose/mod.sio` - Dosing protocols (Weekly, Q3W, Daily)
- `policy/mod.sio` - Dosing policies
- `mod.sio` - Main module exports

### ⬜ To Be Created
- PD models (Emax, sigmoid, indirect)
- PBPK tissue models
- Population modeling (mixed effects, covariates)
- Simulation tools (Monte Carlo, VPC)
- Estimation methods (FOCE, SAEM, Bayesian)
- NONMEM compatibility layer

## Usage

When Sounio repository is available:

1. Copy `sounio-structure/stdlib/medlang/` to `sounio/stdlib/medlang/`
2. Review and adapt syntax to match Sounio's actual syntax
3. Test compilation: `souc check stdlib/medlang/mod.sio`
4. Add tests
5. Document in Sounio's main documentation

## Notes

- These files use proposed Sounio syntax - may need adjustment
- `Knowledge<T>` is assumed to be available in Sounio
- Compartment/flow syntax is proposed - actual syntax may differ
- All models use epistemic types for automatic uncertainty propagation

## Next Steps

1. Wait for Sounio repository structure
2. Review actual Sounio syntax
3. Adapt these files to match Sounio
4. Migrate remaining components
5. Test and integrate

