# MedLang Standard Library

Week 25: Module System

The MedLang standard library provides reusable, validated components for clinical trial design and analysis.

## Structure

```
medlang_std/
├── models/
│   └── pkpd.med          # Standard PK/PD models (OneCmptIV, TwoCmptIV, OneCmptOral)
├── protocols/
│   └── standard_dose.med # Standard dosing protocols (WeeklyDose, Q3WDose, DailyOral)
└── policies/
    └── simple.med        # Simple interpretable policies (FixedDose, ANCBased, etc.)
```

## Usage

Import models, protocols, or policies into your MedLang programs:

```medlang
// Import specific models
import medlang_std.models.pkpd::{OneCmptIV, TwoCmptIV};

// Import all from a module
import medlang_std.protocols.standard_dose::*;

// Use imported definitions
population OneCmptIV_Pop {
    model OneCmptIV;
    // ...
}
```

## Modules

### `medlang_std.models.pkpd`

Standard pharmacokinetic models:
- **OneCmptIV**: One-compartment IV bolus model
- **TwoCmptIV**: Two-compartment IV model with peripheral distribution
- **OneCmptOral**: One-compartment oral absorption model

### `medlang_std.protocols.standard_dose`

Reusable dosing protocol templates:
- **WeeklyDose**: Weekly dosing with multiple dose levels
- **Q3WDose**: Every-3-weeks dosing (Q3W regimen)
- **DailyOral**: Daily oral dosing protocol

### `medlang_std.policies.simple`

Simple, interpretable dosing policies:
- **FixedDose**: No dose modification (always 100%)
- **ANCBased**: ANC-guided dose reduction
- **TumorResponseBased**: Tumor-guided dose escalation/reduction
- **CycleEscalation**: Gradual dose escalation by cycle
- **TimeBasedReduction**: Time-dependent dose reduction

## Extending the Standard Library

To add new modules:

1. Create a new `.med` file in the appropriate directory
2. Use `module <path> { ... }` syntax
3. Export public definitions with `export` declarations
4. Follow naming conventions (PascalCase for types, snake_case for functions)

## Future Additions

Planned for future weeks:
- QSP models (tumor growth, immune dynamics)
- Adaptive trial designs
- Surrogate endpoint models
- Real-world evidence integration protocols
