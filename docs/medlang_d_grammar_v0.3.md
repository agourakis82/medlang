# MedLang-D Grammar v0.3

**Purpose:** Grammar specification for Vertical Slice 0.3 (Transit Absorption and EHR)
**Notation:** EBNF-like syntax
**Status:** V0.3 - Extends V0.2 with advanced absorption constructs
**Reference Implementation:** Darwin PBPK Platform (Julia)

---

## What's New in v0.3

- **Transit Block:** Multi-compartment absorption model (CAT model)
- **EHR Block:** Enterohepatic recirculation modeling
- **Meal Events:** Gallbladder emptying triggers in timeline
- **New Functions:** mean_transit_time, secondary peak detection

---

## Transit Compartment Absorption (New in v0.3)

### Motivation

Transit compartment models (CAT - Compartmental Absorption and Transit) 
better describe drugs with:
- Delayed absorption (BCS Class III/IV)
- Modified release formulations
- Complex GI transit profiles

The model uses n compartments in series to create a delayed, smoother absorption profile.

### EBNF

    transit_def ::= 'transit' '{' transit_item* '}'
    
    transit_item ::=
        | 'n'   '=' expr              // Number of transit compartments (1-10)
        | 'ktr' '=' expr              // Transit rate constant (1/h)
        | 'mtt' '=' expr              // Mean transit time (h) - alternative to ktr
        | 'ka'  '=' expr              // Final absorption rate constant (1/h)
        | 'f'   '=' expr              // Fraction absorbed (optional)
        | 'fg'  '=' expr              // Gut availability (optional)
        | 'fh'  '=' expr              // Hepatic availability (optional)
        | 'lag' '=' expr              // Initial lag time (optional)

### Parameters

| Parameter | Type | Units | Description | Default |
|-----------|------|-------|-------------|---------|
| n | Integer | - | Number of transit compartments (1-10) | Required |
| ktr | RateConst | 1/h | Transit rate constant | Required* |
| mtt | Time | h | Mean transit time | Required* |
| ka | RateConst | 1/h | Absorption rate constant | Required |
| f | Fraction | - | Fraction absorbed | 1.0 |
| fg | Fraction | - | Gut availability | 1.0 |
| fh | Fraction | - | Hepatic availability | 1.0 |
| lag | Time | h | Initial lag time | 0.0 |

*Either ktr or mtt must be specified, not both.

### Relationship Between ktr and mtt

    ktr = (n + 1) / mtt
    mtt = (n + 1) / ktr

### ODE Equations

For n transit compartments:

    dA_1/dt = -ktr * A_1                          (first transit)
    dA_i/dt = ktr * A_{i-1} - ktr * A_i           (intermediate, i=2..n)
    dA_gut/dt = ktr * A_n - ka * A_gut            (gut lumen)
    dC_blood/dt = ... + ka * A_gut * fg * fh / V_blood

### Example

    model GabapentinTransit {
        route: ORAL
        
        // Transit compartment absorption (BCS III drug)
        transit {
            n   = 4              // 4 transit compartments
            mtt = 1.5_h          // Mean transit time 1.5h
            ka  = 1.2_1/h        // Absorption from gut lumen
            f   = 0.6            // 60% bioavailability (saturable)
        }
        
        state A_transit_1 : DoseMass
        state A_transit_2 : DoseMass
        state A_transit_3 : DoseMass
        state A_transit_4 : DoseMass
        state A_gut       : DoseMass
        state A_central   : DoseMass
        
        param V  : Volume    = 77.0_L
        param CL : Clearance = 7.5_L/h   // Renal only
        
        obs C_plasma : ConcMass = A_central / V
    }

---

## Enterohepatic Recirculation (New in v0.3)

### Motivation

Enterohepatic recirculation (EHR) occurs when drugs are:
1. Excreted from liver into bile
2. Stored in gallbladder
3. Released into gut (triggered by meals)
4. Reabsorbed back into systemic circulation

This creates characteristic secondary peaks in plasma concentration.

### EBNF

    ehr_def ::= 'ehr' '{' ehr_item* '}'
    
    ehr_item ::=
        | 'f_bile'  '=' expr          // Fraction excreted in bile (0-1)
        | 'k_bile'  '=' expr          // Biliary excretion rate constant (1/h)
        | 'f_reabs' '=' expr          // Fraction reabsorbed from gut (0-1)
        | 'k_reabs' '=' expr          // Reabsorption rate constant (1/h)
        | 't_gb'    '=' expr          // Gallbladder emptying delay (h)

### Parameters

| Parameter | Type | Units | Description | Default |
|-----------|------|-------|-------------|---------|
| f_bile | Fraction | - | Fraction of hepatic drug excreted in bile | Required |
| k_bile | RateConst | 1/h | Biliary excretion rate constant | Required |
| f_reabs | Fraction | - | Fraction reabsorbed from gut | Required |
| k_reabs | RateConst | 1/h | Intestinal reabsorption rate | Required |
| t_gb | Time | h | Delay from meal to gallbladder emptying | 1.0 |

### ODE Equations

    dC_liver/dt = ... - k_bile * f_bile * C_liver
    dA_bile/dt = k_bile * f_bile * A_liver - k_empty(t) * A_bile
    dA_gut_reabs/dt = k_empty(t) * A_bile - k_reabs * A_gut_reabs
    dC_blood/dt = ... + k_reabs * f_reabs * A_gut_reabs / V_blood

Where k_empty(t) increases during meal times (Gaussian pulse).

### Example

    model MycophenolateEHR {
        route: ORAL
        
        absorption {
            ka = 2.0_1/h
            f  = 0.94
        }
        
        firstpass {
            fg = 0.90
            fh = 0.85
        }
        
        // Enterohepatic recirculation
        ehr {
            f_bile  = 0.4          // 40% biliary excretion
            k_bile  = 0.3_1/h      // Biliary rate
            f_reabs = 0.9          // 90% reabsorbed
            k_reabs = 1.0_1/h      // Reabsorption rate
            t_gb    = 1.0_h        // GB emptying delay
        }
        
        state A_gut     : DoseMass
        state A_central : DoseMass
        state A_bile    : DoseMass     // Bile/gallbladder
        state A_reabs   : DoseMass     // Gut reabsorption pool
        
        param V  : Volume    = 50.0_L
        param CL : Clearance = 15.0_L/h
        
        obs C_plasma : ConcMass = A_central / V
    }

---

## Meal Events in Timeline (New in v0.3)

### EBNF

    meal_event ::= 'at' time_expr ':' 'meal' meal_spec?
    
    meal_spec ::= '{' meal_field* '}'
    
    meal_field ::=
        | 'type' '=' meal_type
        | 'fat_content' '=' expr
    
    meal_type ::= 'light' | 'standard' | 'high_fat'

### Example

    timeline EHRStudy {
        at 0.0_h:
            dose { amount = 1000.0_mg; to = model.A_gut; route = ORAL }
        
        // Meals trigger gallbladder emptying
        at 4.0_h:  meal { type = standard }
        at 8.0_h:  meal { type = standard }
        at 12.0_h: meal { type = high_fat }
        
        // Observations to capture secondary peaks
        at 0.5_h:  observe model.C_plasma
        at 1.0_h:  observe model.C_plasma
        at 2.0_h:  observe model.C_plasma
        at 4.0_h:  observe model.C_plasma
        at 5.0_h:  observe model.C_plasma
        at 6.0_h:  observe model.C_plasma
        at 8.0_h:  observe model.C_plasma
        at 9.0_h:  observe model.C_plasma
        at 12.0_h: observe model.C_plasma
        at 24.0_h: observe model.C_plasma
    }

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
        | transit_def         // v0.3 - NEW
        | ehr_def             // v0.3 - NEW

---

## New Built-in Functions (v0.3)

| Function | Signature | Description |
|----------|-----------|-------------|
| mean_transit_time(n, ktr) | (Int, f64) -> f64 | Calculate MTT from n and ktr |
| transit_absorption(n, ktr, t) | (Int, f64, f64) -> f64 | Fraction absorbed at time t |
| detect_peaks(conc, time) | (Vec, Vec) -> Vec | Find local maxima in profile |

---

## Validation Rules (Extended)

### Semantic Checks for Transit Absorption

1. **Transit constraints:**
   - n must be integer in range [0, 10]
   - Either ktr or mtt must be specified, not both
   - If mtt specified: ktr = (n + 1) / mtt
   - ka > 0 required

2. **Transit + route consistency:**
   - transit block requires route: ORAL
   - transit and absorption blocks are mutually exclusive

### Semantic Checks for EHR

1. **EHR constraints:**
   - All f_* parameters must be in [0, 1]
   - All k_* parameters must be > 0
   - t_gb >= 0

2. **EHR + absorption consistency:**
   - ehr block requires absorption or transit block
   - ehr adds bile and gut_reabs states automatically

3. **Meal events:**
   - meal events only affect EHR dynamics
   - meal times should be within simulation timespan

---

## Reference Implementation

**Repository:** github.com/agourakis82/darwin-pbpk-platform

**Key Files (v0.3):**
- julia-migration/src/DarwinPBPK/ode_solver.jl
  - TransitParams struct
  - transit_ode_system!()
  - simulate_transit()
  - EHRParams struct
  - ehr_ode_system!()
  - simulate_ehr()

**Validation Results:**
- Transit model: Tmax delay verified (3.15h vs 1.45h for first-order)
- EHR model: Functional with meal-triggered GB emptying
- 4/5 tests passed

---

## Changelog

### v0.3 (2024-11)
- Added transit block (n, ktr, mtt, ka, f, fg, fh, lag)
- Added ehr block (f_bile, k_bile, f_reabs, k_reabs, t_gb)
- Added meal events in timeline
- New functions: mean_transit_time, detect_peaks
- Transit compartment ODE system (up to 10 compartments)
- Enterohepatic recirculation ODE system
- Reference implementation in Darwin PBPK Platform

### v0.2 (2024-11)
- Added route definition (IV, ORAL, IM, SC, INFUSION)
- Added absorption block (ka, f, lag)
- Added firstpass block (fg, fh)
- Added organ definition (volume, flow, kp)
- Added clearance definition (hepatic, renal)

### v0.1 (2024-08)
- Initial grammar specification
- 1-compartment oral PK with NLME
- Basic model, population, measure, timeline, cohort constructs

---

**End of Grammar Specification v0.3**
