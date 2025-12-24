# MedLang-D Grammar v0.2

**Purpose:** Grammar specification for Vertical Slice 0.2 (Oral Absorption & First-Pass Metabolism)
**Notation:** EBNF-like syntax
**Status:** V0.2 - Extends V0.1 with oral PK constructs
**Reference Implementation:** Darwin PBPK Platform (Julia)

---

## What's New in v0.2

- **Route Definition:** Explicit administration route specification (IV, ORAL, IM, SC, INFUSION)
- **Absorption Block:** Ka, F (bioavailability), lag time parameters
- **First-Pass Block:** Fg (gut availability), Fh (hepatic availability)
- **Organ Definition:** Tissue compartment with volume, flow, partition coefficient
- **Clearance Definition:** Hepatic/renal clearance with extraction ratio

---

## Lexical Elements

### Keywords (Extended)

    model population measure timeline cohort
    state param obs rand input
    dA_dt use_measure bind_params at dose observe to

    // New in v0.2
    route absorption firstpass organ clearance
    ka f lag fg fh
    volume flow kp extraction
    IV ORAL IM SC INFUSION
    hepatic renal

### Identifiers

    IDENT ::= [a-zA-Z_][a-zA-Z0-9_]*

### Literals

    FLOAT_LIT ::= [0-9]+ ('.' [0-9]+)? ([eE] [+-]? [0-9]+)?
    UNIT_LIT  ::= FLOAT_LIT '_' UNIT_SUFFIX
    UNIT_SUFFIX ::= 'mg' | 'L' | 'h' | 'kg' | '1/h' | 'L/h' | 'mL/min' | ...

    STRING_LIT ::= '"' [^"]* '"'

### Operators

    + - * / ^ = ~ < > ( ) { } [ ] , ; : . |

### Comments

    // Single-line comment
    /* Multi-line comment */

---

## Top-Level Constructs

### Program (EBNF)

    program ::= declaration*

    declaration ::=
        | model_def
        | population_def
        | measure_def
        | timeline_def
        | cohort_def

---

## Model Definition (Extended)

### EBNF

    model_def ::= 'model' IDENT '{' model_item* '}'

    model_item ::=
        | state_decl
        | param_decl
        | ode_equation
        | observable_decl
        | route_def           // New in v0.2
        | absorption_def      // New in v0.2
        | firstpass_def       // New in v0.2
        | organ_def           // New in v0.2
        | clearance_def       // New in v0.2

    state_decl ::= 'state' IDENT ':' type_expr
    param_decl ::= 'param' IDENT ':' type_expr
    ode_equation ::= 'd' IDENT '/' 'dt' '=' expr
    observable_decl ::= 'obs' IDENT ':' type_expr '=' expr

---

## Route Definition (New in v0.2)

### EBNF

    route_def ::= 'route' ':' route_type
    route_type ::= 'IV' | 'ORAL' | 'IM' | 'SC' | 'INFUSION'

### Route Types

| Route | Description | Default Compartment |
|-------|-------------|---------------------|
| IV | Intravenous bolus | Blood/Central |
| ORAL | Oral administration | Gut lumen (requires absorption block) |
| IM | Intramuscular | Depot compartment |
| SC | Subcutaneous | Depot compartment |
| INFUSION | IV infusion | Blood/Central (rate-controlled) |

### Example

    model OralDrug {
        route: ORAL
        // ... rest of model
    }

---

## Absorption Definition (New in v0.2)

### EBNF

    absorption_def ::= 'absorption' '{' absorption_item* '}'
    absorption_item ::=
        | 'ka' '=' expr               // Absorption rate constant (required)
        | 'f'  '=' expr               // Bioavailability fraction (optional, default 1.0)
        | 'lag' '=' expr              // Lag time (optional, default 0.0)

### Parameters

| Parameter | Type | Units | Description | Default |
|-----------|------|-------|-------------|---------|
| ka | RateConst | 1/h | First-order absorption rate constant | Required |
| f | Fraction | - | Oral bioavailability (0-1) | 1.0 |
| lag | Time | h | Absorption lag time | 0.0 |

### Bioavailability Calculation

When firstpass block is defined:

    F_effective = f * Fg * Fh

Otherwise:

    F_effective = f

### Example

    absorption {
        ka  = 1.5_1/h      // Rapid absorption
        f   = 0.85         // 85% bioavailable
        lag = 0.5_h        // 30-minute lag
    }

---

## First-Pass Metabolism Definition (New in v0.2)

### EBNF

    firstpass_def ::= 'firstpass' '{' firstpass_item* '}'
    firstpass_item ::=
        | 'fg' '=' expr               // Gut availability (required)
        | 'fh' '=' expr               // Hepatic availability (required)

### Parameters

| Parameter | Type | Range | Description |
|-----------|------|-------|-------------|
| fg | Fraction | 0-1 | Fraction escaping gut wall metabolism |
| fh | Fraction | 0-1 | Fraction escaping hepatic first-pass |

### Calculation from Clearance

    Fh = 1 - (CLH / QH)

Where:
- CLH = Hepatic clearance
- QH = Hepatic blood flow (~1.5 L/min or 90 L/h)

### Example

    firstpass {
        fg = 0.95          // Minimal gut metabolism
        fh = 0.70          // 30% hepatic extraction
    }

---

## Organ Definition (New in v0.2)

### EBNF

    organ_def ::= 'organ' IDENT '{' organ_item* '}'
    organ_item ::=
        | 'volume' '=' expr           // Tissue volume
        | 'flow'   '=' expr           // Blood flow to organ
        | 'kp'     '=' expr           // Tissue:plasma partition coefficient

### Parameters

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| volume | Volume | L | Anatomical tissue volume |
| flow | FlowRate | L/h | Blood flow rate to tissue |
| kp | Dimensionless | - | Tissue:plasma partition coefficient |

### Example

    organ liver {
        volume = 1.8_L
        flow   = 90.0_L/h      // Hepatic blood flow
        kp     = 2.5           // Lipophilic drug
    }

    organ muscle {
        volume = 28.0_L
        flow   = 42.0_L/h
        kp     = 1.2
    }

---

## Clearance Definition (New in v0.2)

### EBNF

    clearance_def ::= 'clearance' clearance_type '{' clearance_item* '}'
    clearance_type ::= 'hepatic' | 'renal'
    clearance_item ::=
        | 'cl'         '=' expr       // Clearance value
        | 'extraction' '=' expr       // Extraction ratio (optional)

### Parameters

| Parameter | Type | Units | Description |
|-----------|------|-------|-------------|
| cl | Clearance | L/h | Organ clearance |
| extraction | Fraction | - | Extraction ratio (E = CL / Q) |

### Example

    clearance hepatic {
        cl = 25.0_L/h
        extraction = 0.28      // Calculated: 25 / 90
    }

    clearance renal {
        cl = 5.0_L/h
    }

---

## Complete Oral PBPK Model Example

    // Full 15-compartment PBPK with oral absorption
    model MetforminPBPK {
        route: ORAL
        
        // Absorption kinetics
        absorption {
            ka  = 0.5_1/h      // Slow absorption (BCS III)
            f   = 0.55         // 55% bioavailability
            lag = 0.25_h       // 15-minute lag
        }
        
        // First-pass metabolism
        firstpass {
            fg = 0.98          // Minimal gut metabolism
            fh = 0.56          // Hepatic extraction ~44%
        }
        
        // States (15 compartments)
        state A_gut     : DoseMass    // Gut lumen depot
        state A_blood   : DoseMass    // Blood/plasma
        state A_liver   : DoseMass    // Liver
        state A_kidney  : DoseMass    // Kidney
        state A_muscle  : DoseMass    // Muscle
        state A_adipose : DoseMass    // Adipose
        state A_brain   : DoseMass    // Brain
        state A_heart   : DoseMass    // Heart
        state A_lung    : DoseMass    // Lung
        state A_skin    : DoseMass    // Skin
        state A_bone    : DoseMass    // Bone
        state A_gut_tis : DoseMass    // GI tissue
        state A_spleen  : DoseMass    // Spleen
        state A_rest    : DoseMass    // Rest of body
        
        // Organ definitions
        organ blood {
            volume = 5.0_L
        }
        
        organ liver {
            volume = 1.8_L
            flow   = 90.0_L/h
            kp     = 1.5
        }
        
        organ kidney {
            volume = 0.31_L
            flow   = 72.0_L/h
            kp     = 3.0
        }
        
        // Clearance
        clearance hepatic {
            cl = 15.0_L/h
        }
        
        clearance renal {
            cl = 25.0_L/h      // OCT2-mediated secretion
        }
        
        // ODE system (gut absorption)
        dA_gut/dt = -Ka * A_gut
        
        // Observables
        obs C_plasma : ConcMass = A_blood / V_blood
    }

---

## Simple Oral PK Model Example

    // 2-compartment oral PK (minimal)
    model SimpleOral {
        route: ORAL
        
        absorption {
            ka = 1.0_1/h
            f  = 0.8
        }
        
        state A_gut     : DoseMass
        state A_central : DoseMass
        
        param V  : Volume    = 50.0_L
        param CL : Clearance = 10.0_L/h
        
        dA_gut/dt     = -Ka * A_gut
        dA_central/dt =  Ka * A_gut * F - (CL / V) * A_central
        
        obs C_plasma : ConcMass = A_central / V
    }

---

## Population Definition

### EBNF

    population_def ::= 'population' IDENT '{' population_item* '}'
    population_item ::=
        | 'model' IDENT                    // Reference to model
        | param_decl                       // Population parameter
        | input_decl                       // Covariate
        | random_effect_decl               // IIV
        | bind_params_block                // Individual parameter mapping
        | use_measure_stmt                 // Error model binding

    input_decl ::= 'input' IDENT ':' type_expr
    random_effect_decl ::= 'rand' IDENT ':' type_expr '~' distribution_expr
    distribution_expr ::=
        | 'Normal' '(' expr ',' expr ')'
        | 'LogNormal' '(' expr ',' expr ')'
        | 'Uniform' '(' expr ',' expr ')'
    bind_params_block ::= 'bind_params' '(' IDENT ')' '{' statement* '}'
    use_measure_stmt ::= 'use_measure' IDENT 'for' qualified_name

### Example with Oral Absorption

    population OralPKPop {
        model SimpleOral

        // Population parameters
        param Ka_pop : RateConst = 1.0_1/h
        param F_pop  : Fraction  = 0.8
        param CL_pop : Clearance = 10.0_L/h
        param V_pop  : Volume    = 50.0_L
        
        // IIV on absorption
        param omega_Ka : f64 = 0.3
        param omega_F  : f64 = 0.2
        param omega_CL : f64 = 0.25
        param omega_V  : f64 = 0.20

        // Covariate
        input WT : Quantity<kg, f64>

        // Random effects
        rand eta_Ka : f64 ~ LogNormal(0.0, omega_Ka)
        rand eta_F  : f64 ~ Normal(0.0, omega_F)
        rand eta_CL : f64 ~ Normal(0.0, omega_CL)
        rand eta_V  : f64 ~ Normal(0.0, omega_V)

        // Individual parameter mapping
        bind_params(patient) {
            let w = patient.WT / 70.0_kg
            
            // Absorption parameters with IIV
            model.absorption.Ka = Ka_pop * eta_Ka
            model.absorption.F  = F_pop * (1 + eta_F)  // Bounded IIV
            
            // PK parameters
            model.CL = CL_pop * pow(w, 0.75) * exp(eta_CL)
            model.V  = V_pop * w * exp(eta_V)
        }

        use_measure ConcPropError for model.C_plasma
    }

---

## Measure Definition

### EBNF

    measure_def ::= 'measure' IDENT '{' measure_item* '}'
    measure_item ::=
        | 'pred' ':' type_expr
        | 'obs'  ':' type_expr
        | param_decl
        | loglik_stmt
    loglik_stmt ::= 'log_likelihood' '=' expr

### Example

    measure ConcPropError {
        pred : ConcMass
        obs  : ConcMass
        param sigma_prop : f64

        log_likelihood = Normal_logpdf(
            x  = (obs / pred) - 1.0,
            mu = 0.0,
            sd = sigma_prop
        )
    }

---

## Timeline Definition

### EBNF

    timeline_def ::= 'timeline' IDENT '{' event* '}'
    event ::= dose_event | observe_event
    dose_event ::= 'at' time_expr ':' 'dose' dose_spec
    dose_spec ::= '{' dose_field* '}'
    dose_field ::=
        | 'amount' '=' expr
        | 'to' '=' qualified_name
        | 'route' '=' route_type        // New in v0.2
    observe_event ::= 'at' time_expr ':' 'observe' qualified_name
    time_expr ::= expr  // Must have Time units

### Example

    timeline OralDoseTimeline {
        // Single oral dose
        at 0.0_h:
            dose { 
                amount = 500.0_mg
                to     = SimpleOral.A_gut
                route  = ORAL
            }

        // Rich PK sampling
        at 0.5_h:  observe SimpleOral.C_plasma
        at 1.0_h:  observe SimpleOral.C_plasma
        at 2.0_h:  observe SimpleOral.C_plasma
        at 4.0_h:  observe SimpleOral.C_plasma
        at 8.0_h:  observe SimpleOral.C_plasma
        at 12.0_h: observe SimpleOral.C_plasma
        at 24.0_h: observe SimpleOral.C_plasma
    }

---

## Cohort Definition

### EBNF

    cohort_def ::= 'cohort' IDENT '{' cohort_item* '}'
    cohort_item ::=
        | 'population' IDENT
        | 'timeline'   IDENT
        | 'data_file'  STRING_LIT

### Example

    cohort OralPKStudy {
        population OralPKPop
        timeline   OralDoseTimeline
        data_file  "data/oral_pk_study.csv"
    }

---

## Type Expressions

### EBNF

    type_expr ::=
        | IDENT                                    // Simple type (e.g., f64)
        | unit_type                                // Built-in unit type
        | 'Quantity' '<' unit_expr ',' type_expr '>'  // Generic quantity

    unit_type ::=
        | 'Mass' | 'Volume' | 'Time'               // Base units
        | 'DoseMass' | 'ConcMass'                  // Derived units
        | 'Clearance' | 'RateConst'                // Pharmacokinetic units
        | 'Fraction' | 'FlowRate'                  // New in v0.2
        | 'AUCUnit'                                // New in v0.2

    unit_expr ::=
        | IDENT                                    // Unit name (e.g., kg)
        | unit_expr '*' unit_expr                  // Product
        | unit_expr '/' unit_expr                  // Quotient

### Built-in Unit Types (v0.2)

| Type | Definition | Example |
|------|------------|---------|
| Mass | Base unit | mg, g, kg |
| Volume | Base unit | mL, L |
| Time | Base unit | h, min, day |
| DoseMass | Alias for Mass | 100.0_mg |
| ConcMass | Mass / Volume | 5.0_mg/L |
| Clearance | Volume / Time | 10.0_L/h |
| RateConst | 1 / Time | 1.5_1/h |
| Fraction | Dimensionless [0,1] | 0.85 |
| FlowRate | Volume / Time | 90.0_L/h |
| AUCUnit | Mass * Time / Volume | mg*h/L |

---

## Expressions

### EBNF

    expr ::=
        | literal
        | IDENT
        | qualified_name
        | unary_op expr
        | expr binary_op expr
        | function_call
        | '(' expr ')'

    qualified_name ::= IDENT ('.' IDENT)*
    literal ::= FLOAT_LIT | UNIT_LIT
    unary_op ::= '-' | '+'
    binary_op ::=
        | '+' | '-' | '*' | '/' | '^'   // Arithmetic
        | '<' | '>' | '==' | '!='       // Comparison
    function_call ::= IDENT '(' argument_list? ')'
    argument_list ::= argument (',' argument)*
    argument ::= expr | IDENT '=' expr          // Positional or Named

### Built-in Functions (v0.2)

| Function | Signature | Description |
|----------|-----------|-------------|
| exp(x) | f64 -> f64 | Exponential |
| log(x) | f64 -> f64 | Natural log |
| pow(x, y) | (f64, f64) -> f64 | Power |
| sqrt(x) | f64 -> f64 | Square root |
| Normal_logpdf(...) | Named args | Log-PDF of normal |
| integral(x) | expr -> AUCUnit | Time integral (AUC) |
| max(x, y) | (f64, f64) -> f64 | Maximum |
| min(x, y) | (f64, f64) -> f64 | Minimum |
| clamp(x, lo, hi) | (f64, f64, f64) -> f64 | Bounded value |

---

## Statements

### EBNF

    statement ::= let_stmt | assign_stmt | expr_stmt
    let_stmt ::= 'let' IDENT '=' expr
    assign_stmt ::= qualified_name '=' expr
    expr_stmt ::= expr

---

## Validation Rules (Extended)

### Semantic Checks for Oral Absorption

1. **Route consistency:**
   - If route: ORAL, model MUST have absorption block
   - absorption.ka is required for oral route
   - absorption.f defaults to 1.0 if not specified

2. **First-pass constraints:**
   - firstpass block requires route: ORAL
   - Both fg and fh must be in range [0, 1]
   - If firstpass defined, F_eff = f * fg * fh

3. **Dose target validation:**
   - For route: ORAL, dose target should be gut/depot state
   - For route: IV, dose target should be blood/central state

4. **Organ definition:**
   - volume is required
   - flow required for non-blood organs
   - kp required for tissue distribution

5. **Clearance validation:**
   - hepatic clearance requires liver organ definition
   - renal clearance requires kidney organ definition
   - extraction must be <= 1.0

---

## Reference Implementation

The reference implementation is available in the Darwin PBPK Platform:

**Repository:** github.com/agourakis82/darwin-pbpk-platform

**Key Files:**
- julia-migration/src/DarwinPBPK/medlang/parser.jl - Tokenizer and parser
- julia-migration/src/DarwinPBPK/medlang/transpiler.jl - Transpiler to Julia/PBPKParams
- julia-migration/src/DarwinPBPK/ode_solver.jl - 15-compartment ODE system

**Validation Results (572 drugs):**
- Success rate: 92.1%
- Half-life GMFE: 2.02 (meets FDA threshold <= 2.0)
- Cmax within 2-fold: 24.3% (full ODE integration)

---

## Extensions for Future Versions

**Not in v0.2, but planned:**

- Transit compartment absorption (v0.3)
- Enterohepatic recirculation (v0.3)
- Non-linear (saturable) absorption (v0.4)
- Multi-compartment depot (IM/SC) (v0.4)
- QSP constructs (species, reaction) (v1.0)
- ML submodels (mlmodel, train) (v1.0)
- Track C operators (QM_BindingFreeEnergy, etc.) (v1.0)

---

## Changelog

### v0.2 (2024-11)
- Added route definition (IV, ORAL, IM, SC, INFUSION)
- Added absorption block (ka, f, lag)
- Added firstpass block (fg, fh)
- Added organ definition (volume, flow, kp)
- Added clearance definition (hepatic, renal)
- New types: Fraction, FlowRate, AUCUnit
- New functions: integral, max, min, clamp
- Extended timeline with route specification
- Reference implementation in Darwin PBPK Platform

### v0.1 (2024-08)
- Initial grammar specification
- 1-compartment oral PK with NLME
- Basic model, population, measure, timeline, cohort constructs

---

**End of Grammar Specification v0.2**

*This grammar extends MedLang v0.1 with comprehensive oral absorption modeling. Implementation follows the Julia reference in Darwin PBPK Platform.*
