# MedLang and Demetrios Integration

## Overview

**MedLang** is an embedded Domain-Specific Language (eDSL) built on top of the **Demetrios (D) language**. This document describes the relationship between MedLang and Demetrios, and how they work together.

## What is Demetrios?

[Demetrios (D)](https://github.com/Chiuratto-AI/demetrios) is a general-purpose programming language that provides:

- Advanced type system with refinement types
- Effect system for tracking computational side effects
- Epistemic computing with `Knowledge<T>` wrapper
- SMT-based verification
- First-class support for surrogates and ML models
- Reinforcement learning primitives

## MedLang as an eDSL

MedLang extends Demetrios with domain-specific constructs for computational pharmacology:

### Medical-Specific Types

```medlang
// MedLang adds medical units and types
state A_central : DoseMass
param CL : Clearance      // L³/T
param V : Volume          // L³
obs C_plasma : ConcMass   // M/L³
```

### Pharmacometric Constructs

```medlang
// Population modeling
population OneCompPop {
    model OneComp
    param CL_pop : Clearance
    param V_pop : Volume
    rand eta_CL : Real
    rand eta_V : Real
}

// ODE systems
dA_central/dt = -(CL / V) * A_central
```

### Clinical Protocols

```medlang
protocol StandardDose {
    dose: 100.0_mg
    route: Oral
    frequency: Daily
}
```

## Compilation Pipeline

```
MedLang Source (.medlang)
  ↓
Demetrios AST (via parser)
  ↓
MedLang IR (CIR/NIR)
  ↓
Backend Code Generation
  ↓
Stan / Julia / etc.
```

## Integration Points

### 1. Type System

MedLang leverages Demetrios' refinement type system for clinical constraints:

```medlang
// Demetrios refinement types
param Age : Int where Age >= 0 && Age <= 150

// MedLang extends with dimensional analysis
param CL : Clearance  // M·L·T dimensional checking
```

### 2. Effect System

MedLang uses Demetrios' effect system to track computational effects:

```medlang
// Demetrios effects: Prob, IO, GPU, Pure
fn simulate() : Prob {
    // Probabilistic computation
}

// MedLang adds medical effects
fn fit_model() : Prob + Clinical {
    // Probabilistic + clinical reasoning
}
```

### 3. Epistemic Computing

MedLang integrates Demetrios' `Knowledge<T>` for uncertainty quantification:

```medlang
// Demetrios epistemic wrapper
let cl_estimate : Knowledge<Clearance> = fit_parameter(data)

// MedLang uses for parameter estimates
param CL : Knowledge<Clearance> = estimate_from_data()
```

### 4. SMT Verification

MedLang uses Demetrios' SMT backend for clinical safety verification:

```medlang
// Demetrios contracts
fn dose_calculation(weight: Real) -> Real
    requires weight > 0.0 && weight < 300.0
    ensures result > 0.0 && result < 1000.0
{
    // MedLang dosing logic
}
```

## Shared Features

Both MedLang and Demetrios share:

- **Refinement Types**: Clinical constraints and dimensional analysis
- **Effect System**: Tracking computational side effects
- **Epistemic Computing**: Uncertainty quantification
- **SMT Verification**: Safety and correctness guarantees
- **ML Integration**: Surrogate models and neural networks
- **RL Support**: Reinforcement learning for optimization

## Development Workflow

1. **Write MedLang code** using medical-specific constructs
2. **Compile to Demetrios IR** (CIR/NIR)
3. **Leverage Demetrios features** (effects, refinements, SMT)
4. **Generate backend code** (Stan, Julia, etc.)

## Repository Structure

```
medlang/                    # MedLang eDSL
├── compiler/              # MedLang → Demetrios IR
├── runtime/               # Medical runtime
└── ffi/                   # C API for integration

demetrios/                 # Demetrios host language
├── compiler/              # Demetrios compiler
├── stdlib/                # Standard library
└── ...
```

## Future Integration

As both projects mature, we plan to:

1. **Tighter Integration**: Direct compilation to Demetrios bytecode
2. **Shared Runtime**: Common runtime for both languages
3. **Unified Type System**: Seamless type checking across languages
4. **Cross-Language Interop**: Call Demetrios functions from MedLang

## References

- **Demetrios Repository**: https://github.com/Chiuratto-AI/demetrios
- **MedLang Repository**: https://github.com/agourakis82/medlang
- **MedLang Manifesto**: [manifesto.md](manifesto.md)
- **MedLang Architecture**: [dev/ARCHITECTURE.md](dev/ARCHITECTURE.md)

