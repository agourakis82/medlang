# MedLang-D Grammar v0.4

**Purpose:** Grammar specification for Vertical Slice 0.4 (Saturable Absorption and Depot Models)
**Notation:** EBNF-like syntax
**Status:** V0.4 - Extends V0.3 with non-linear absorption and IM/SC routes
**Reference Implementation:** Darwin PBPK Platform (Julia)

---

## What's New in v0.4

- **Saturable Block:** Michaelis-Menten absorption kinetics
- **Depot Block:** Multi-compartment IM/SC absorption
- **Route Extensions:** Full support for IM and SC administration
- **New Functions:** apparent_ka, mean_absorption_time, dose_proportionality

---

## Saturable (Non-Linear) Absorption (New in v0.4)

### Motivation

Saturable absorption occurs when drugs are transported by capacity-limited
mechanisms:
- Saturable transporters (PEPT1, P-gp, OATP)
- Dose-dependent bioavailability
- Less-than-proportional AUC increase with dose

The Michaelis-Menten model captures this behavior.

### EBNF

    saturable_def ::= 'saturable' '{' saturable_item* '}'
    
    saturable_item ::=
        | 'vmax' '=' expr             // Maximum absorption rate (mg/h)
        | 'km'   '=' expr             // Michaelis constant (mg)
        | 'passive_ka' '=' expr       // Additional passive absorption (1/h)
        | 'f'    '=' expr             // Fraction absorbed (optional)
        | 'fg'   '=' expr             // Gut availability (optional)
        | 'fh'   '=' expr             // Hepatic availability (optional)
        | 'lag'  '=' expr             // Lag time (optional)

### Parameters

| Parameter | Type | Units | Description | Default |
|-----------|------|-------|-------------|---------|
| vmax | Rate | mg/h | Maximum absorption rate | Required |
| km | Amount | mg | Michaelis constant (amount at half-Vmax) | Required |
| passive_ka | RateConst | 1/h | Passive absorption component | 0.0 |
| f | Fraction | - | Fraction available for absorption | 1.0 |
| fg | Fraction | - | Gut availability | 1.0 |
| fh | Fraction | - | Hepatic availability | 1.0 |
| lag | Time | h | Lag time | 0.0 |

### ODE Equations

Michaelis-Menten absorption:

    dA_gut/dt = -(Vmax * A_gut / (Km + A_gut)) - (passive_ka * A_gut)

Limiting behavior:
- At A_gut << Km: rate = (Vmax/Km) * A_gut (first-order, Ka_app = Vmax/Km)
- At A_gut >> Km: rate = Vmax (zero-order, saturated)

### Example

    model GabapentinSaturable {
        route: ORAL
        
        // Saturable PEPT1-mediated absorption
        saturable {
            vmax = 100.0_mg/h      // Maximum rate
            km   = 50.0_mg         // Half-saturation amount
            passive_ka = 0.1_1/h   // Small passive component
            f    = 1.0             // Complete dissolution
            fg   = 1.0             // No gut metabolism
            fh   = 1.0             // No first-pass
        }
        
        state A_gut     : DoseMass
        state A_central : DoseMass
        
        param V  : Volume    = 77.0_L
        param CL : Clearance = 7.5_L/h   // Renal clearance
        
        obs C_plasma : ConcMass = A_central / V
    }

### Dose Proportionality Analysis

For saturable absorption, AUC follows power law:

    AUC = alpha * Dose^beta

Where:
- beta = 1.0: Linear (dose-proportional)
- beta < 1.0: Less-than-proportional (saturable absorption)
- beta > 1.0: Greater-than-proportional (autoinduction, rare)

---

## Depot (IM/SC) Absorption (New in v0.4)

### Motivation

IM and SC administration involves drug release from a depot site:
- Slower than oral absorption
- No first-pass metabolism
- Flip-flop kinetics possible (absorption rate-limiting)
- Complex release profiles (biologics)

### EBNF

    depot_def ::= 'depot' '{' depot_item* '}'
    
    depot_item ::=
        | 'n_depots'   '=' expr       // Number of depot compartments (1-3)
        | 'ka'         '=' expr_list  // Absorption rate constant(s) (1/h)
        | 'fractions'  '=' expr_list  // Fraction of dose in each depot
        | 'f'          '=' expr       // Overall bioavailability (0-1)
        | 'lag'        '=' expr       // Lag time (h)

### Parameters

| Parameter | Type | Units | Description | Default |
|-----------|------|-------|-------------|---------|
| n_depots | Integer | - | Number of depot compartments (1-3) | 1 |
| ka | RateConst[] | 1/h | Absorption rate for each depot | Required |
| fractions | Fraction[] | - | Dose fraction in each depot | [1.0] |
| f | Fraction | - | Overall bioavailability | 0.9 (IM), 0.85 (SC) |
| lag | Time | h | Lag time before absorption | 0.0 |

### Depot Models

**Single depot (simple):**

    dA_depot/dt = -ka * A_depot
    dC_blood/dt = ... + ka * A_depot * F / V_blood

**Dual depot (fast + slow release):**

    dA_depot1/dt = -ka1 * A_depot1      (fast component)
    dA_depot2/dt = -ka2 * A_depot2      (slow component)
    dC_blood/dt = ... + (ka1*A1 + ka2*A2) * F / V_blood

### Route-Specific Defaults

| Route | Typical Ka | Typical F | Typical Lag |
|-------|------------|-----------|-------------|
| IM | 0.3-1.0 1/h | 0.85-0.95 | 0.1h |
| SC | 0.1-0.5 1/h | 0.70-0.90 | 0.2h |
| SC (biologic) | 0.5 + 0.05 1/h | 0.60-0.80 | 0.5h |

### Example: IM Injection

    model DrugIM {
        route: IM
        
        depot {
            n_depots = 1
            ka = [0.5_1/h]        // Moderate absorption
            fractions = [1.0]
            f = 0.90              // 90% bioavailable
            lag = 0.1_h           // Short lag
        }
        
        state A_depot   : DoseMass
        state A_central : DoseMass
        state A_periph  : DoseMass
        
        param V1 : Volume    = 50.0_L
        param V2 : Volume    = 100.0_L
        param CL : Clearance = 10.0_L/h
        param Q  : Clearance = 20.0_L/h
        
        obs C_plasma : ConcMass = A_central / V1
    }

### Example: SC Biologic (Dual Absorption)

    model AdalimumabSC {
        route: SC
        
        // Dual absorption: fast initial + slow sustained
        depot {
            n_depots = 2
            ka = [0.5_1/h, 0.05_1/h]   // Fast and slow
            fractions = [0.3, 0.7]      // 30% fast, 70% slow
            f = 0.70                    // 70% bioavailable
            lag = 0.5_h                 // Delayed start
        }
        
        state A_depot_fast : DoseMass
        state A_depot_slow : DoseMass
        state A_central    : DoseMass
        state A_periph     : DoseMass
        
        // Biologic PK parameters
        param V1  : Volume    = 3.5_L     // Small Vd
        param V2  : Volume    = 2.0_L
        param CL  : Clearance = 0.35_L/h  // Slow clearance
        param Q   : Clearance = 0.5_L/h
        
        obs C_plasma : ConcMass = A_central / V1
    }

---

## Extended Route Types

### EBNF (Updated)

    route_type ::= 'IV' | 'ORAL' | 'IM' | 'SC' | 'INFUSION'

### Route-Block Requirements

| Route | Required Block | First-Pass |
|-------|----------------|------------|
| IV | none (direct to blood) | No |
| ORAL | absorption OR saturable OR transit | Yes (if firstpass defined) |
| IM | depot | No |
| SC | depot | No |
| INFUSION | infusion_rate in dose event | No |

---

## Extended Model Definition

### EBNF (Updated)

    model_def ::= 'model' IDENT '{' model_item* '}'

    model_item ::=
        | state_decl
        | param_decl
        | ode_equation
        | observable_decl
        | route_def           // v0.2
        | absorption_def      // v0.2
        | firstpass_def       // v0.2
        | organ_def           // v0.2
        | clearance_def       // v0.2
        | transit_def         // v0.3
        | ehr_def             // v0.3
        | saturable_def       // v0.4 - NEW
        | depot_def           // v0.4 - NEW

---

## New Built-in Functions (v0.4)

| Function | Signature | Description |
|----------|-----------|-------------|
| apparent_ka(vmax, km) | (f64, f64) -> f64 | Linear approximation Ka = Vmax/Km |
| mean_absorption_time(ka[], f[]) | (Vec, Vec) -> f64 | MAT for depot model |
| dose_proportionality(doses, aucs) | (Vec, Vec) -> f64 | Power exponent beta |
| is_flip_flop(tmax, ka) | (f64, f64) -> bool | Check if absorption rate-limiting |

---

## Validation Rules (Extended)

### Semantic Checks for Saturable Absorption

1. **Saturable constraints:**
   - vmax > 0 required
   - km > 0 required
   - passive_ka >= 0

2. **Saturable + route consistency:**
   - saturable block requires route: ORAL
   - saturable and absorption blocks are mutually exclusive
   - saturable and transit blocks are mutually exclusive

### Semantic Checks for Depot

1. **Depot constraints:**
   - n_depots in [1, 3]
   - length(ka) == n_depots
   - length(fractions) == n_depots
   - sum(fractions) == 1.0
   - f in [0, 1]

2. **Depot + route consistency:**
   - depot block requires route: IM or SC
   - depot block incompatible with absorption/saturable/transit

---

## Reference Implementation

**Repository:** github.com/agourakis82/darwin-pbpk-platform

**Key Files (v0.4):**
- julia-migration/src/DarwinPBPK/ode_solver.jl
  - SaturableAbsorptionParams struct
  - saturable_ode_system!()
  - simulate_saturable()
  - analyze_dose_proportionality()
  - DepotParams struct
  - depot_ode_system!()
  - simulate_depot()

**Validation Results:**
- Saturable: Dose-normalized AUC decreases 0.119 -> 0.108 (100-1200mg)
- Saturable: Tmax 10.18h at high dose vs 0.48h first-order
- IM depot: Tmax 2.58h, flip-flop kinetics detected
- SC biologic: MAT 15.1h with dual absorption
- Route order: Oral Tmax < IM Tmax < SC Tmax confirmed

---

## Changelog

### v0.4 (2024-11)
- Added saturable block (vmax, km, passive_ka)
- Added depot block (n_depots, ka[], fractions[], f, lag)
- Extended route support for IM and SC
- New functions: apparent_ka, mean_absorption_time, dose_proportionality
- Michaelis-Menten ODE for saturable absorption
- Multi-depot ODE for IM/SC with flip-flop detection
- Reference implementation in Darwin PBPK Platform

### v0.3 (2024-11)
- Added transit block (n, ktr, mtt, ka)
- Added ehr block (f_bile, k_bile, f_reabs, k_reabs, t_gb)
- Added meal events in timeline

### v0.2 (2024-11)
- Added route, absorption, firstpass, organ, clearance definitions

### v0.1 (2024-08)
- Initial grammar specification

---

**End of Grammar Specification v0.4**
