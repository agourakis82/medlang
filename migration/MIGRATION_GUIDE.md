# MedLang → Sounio Migration Guide

This guide explains how to migrate MedLang code to Sounio format.

## Key Differences

### 1. Uncertainty is Native

**MedLang (old)**:
```medlang
param CL : Clearance
```

**Sounio (new)**:
```sounio
param CL: Knowledge<L/h> ~ LogNormal(mean: 10.0 L/h, omega: 0.30)
```

### 2. Compartment-Based Syntax

**MedLang (old)**:
```medlang
state A_central : DoseMass
dA_central/dt = -(CL / V) * A_central
```

**Sounio (new)**:
```sounio
compartment Central {
    volume: V
}
flow Central -> Elimination {
    rate: CL
}
```

### 3. Module Path

**MedLang (old)**:
```medlang
import medlang_std.models.pkpd::*
```

**Sounio (new)**:
```sounio
import sounio::medlang::pk::*
```

## Migration Steps

### Step 1: Convert Parameters

Convert parameters to use `Knowledge<T>`:

```sounio
// Old
param CL : Clearance
param V : Volume

// New
param CL: Knowledge<L/h> ~ LogNormal(mean: 10.0 L/h, omega: 0.30)
param V: Knowledge<L> ~ LogNormal(mean: 50.0 L, omega: 0.25)
```

### Step 2: Convert Compartments

Replace ODE syntax with compartment syntax:

```sounio
// Old
state A_central : DoseMass
dA_central/dt = -(CL / V) * A_central

// New
compartment Central {
    volume: V
}
flow Central -> Elimination {
    rate: CL
}
```

### Step 3: Convert Observables

```sounio
// Old
obs C_plasma : ConcMass = A_central / V

// New
observe Cp: Concentration = Central.concentration
```

### Step 4: Update Imports

```sounio
// Old
import medlang_std.models.pkpd::OneCmptIV

// New
import sounio::medlang::pk::one_compartment::OneCompartmentIV
```

## File Structure

### MedLang Structure
```
medlang/
├── medlang_std/
│   ├── models/
│   ├── protocols/
│   └── policies/
└── examples/
```

### Sounio Structure
```
sounio/stdlib/medlang/
├── pk/
├── pd/
├── pbpk/
├── population/
├── simulation/
├── estimation/
├── dose/
└── policy/
```

## Complete Example

### MedLang (old)
```medlang
model OneCompOral {
    state A_gut : DoseMass
    state A_central : DoseMass
    param Ka : RateConst
    param CL : Clearance
    param V : Volume
    dA_gut/dt = -Ka * A_gut
    dA_central/dt = Ka * A_gut - (CL / V) * A_central
    obs C_plasma : ConcMass = A_central / V
}
```

### Sounio (new)
```sounio
import sounio::medlang::*

model OneCompartmentOral {
    param ka: Knowledge<1/h> ~ LogNormal(mean: 1.0 1/h, omega: 0.40)
    param CL: Knowledge<L/h> ~ LogNormal(mean: 10.0 L/h, omega: 0.30)
    param V: Knowledge<L> ~ LogNormal(mean: 50.0 L, omega: 0.25)
    
    compartment Gut {
        transit_time: 1.0 / ka
    }
    
    compartment Central {
        volume: V
    }
    
    flow Gut -> Central {
        rate: ka
    }
    
    flow Central -> Elimination {
        rate: CL
    }
    
    dose Oral {
        into: Gut
    }
    
    observe Cp: Concentration = Central.concentration
}
```

## Automated Migration

See `migration/convert.sh` for automated conversion scripts (to be created).

## Testing

After migration, test with:

```bash
souc check model.sio
souc run model.sio
```

