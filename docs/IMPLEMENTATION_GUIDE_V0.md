# MedLang Implementation Guide — Vertical Slice 0

**Version:** 0.1  
**Status:** Implementation Specification  
**Target:** Minimal viable MedLang-D compiler for 1-compartment oral PK with NLME

---

## Mission Statement

Implement **Vertical Slice 0** of MedLang: a minimal but fully executable, type-safe core of **MedLang-D (Track D)** that demonstrates end-to-end compilation from domain-specific syntax to executable inference code.

**Success Criteria:**
1. Parse a 1-compartment oral PK model with NLME
2. Type-check with full unit safety
3. Lower to internal IR (NIR-lite)
4. Generate executable backend (Stan/Torsten or Julia)
5. Run simulation and compute log-likelihood

**Quality Bar:** Publication-ready (Q1 journal level) code quality, documentation, and rigor.

---

## 1. Context and Scope

### 1.1 What is MedLang?

MedLang is a domain-specific language for computational medicine designed as:
- **Low-level, readable "assembly" for scientific computing**
- **Native pharmacometrics/QSP semantics** (Track D)
- **Quantum pharmacology extension** (Track C) for quantum-derived parameters

### 1.2 Vertical Slice 0 Scope

**IN SCOPE (MUST HAVE):**

1. **Surface Language Subset**
   - `model` with states, parameters, ODEs, observables
   - `measure` for residual error (proportional only)
   - `population` with fixed effects, random effects, covariate mappings
   - `timeline` with dose and observation events
   - `cohort` tying everything together

2. **Unit-Aware Type System**
   - Base units: Mass, Volume, Time
   - Derived: ConcMass = Mass/Volume, Clearance = Volume/Time, RateConst = 1/Time
   - Enforcement: ODE RHS units match dState/dt, exp/log only on dimensionless

3. **Internal IR (NIR-lite)**
   - Numeric IR capturing ODE, parameters, random effects, observations
   - Backend-agnostic, serializable

4. **Backend Code Generation**
   - Target: Stan/Torsten OR Julia (DifferentialEquations + Turing/Pumas)
   - Generate: ODE system, parameter definitions, likelihood

5. **End-to-End Pipeline**
   - Parse → Type-check → IR → Backend → Simulation + Log-likelihood

**OUT OF SCOPE (for V0):**
- Track C quantum operators
- Multi-compartment PBPK, QSP, ML submodels
- General function definitions
- Advanced error models (additive, combined, censored)

---

## 2. Canonical Example: 1-Compartment Oral PK with NLME

### 2.1 Mathematical Formulation

**Individual-level ODEs:**

States:
- $A_g(t)$: amount in gut [Mass, mg]
- $A_c(t)$: amount in central compartment [Mass, mg]

Parameters (individual $i$):
- $CL_i$: clearance [Volume/Time]
- $V_i$: volume [Volume]
- $K_{a,i}$: absorption rate [1/Time]

Dynamics:
$$
\begin{aligned}
\frac{dA_g}{dt} &= -K_{a,i} A_g \\
\frac{dA_c}{dt} &= K_{a,i} A_g - \frac{CL_i}{V_i} A_c
\end{aligned}
$$

Observable:
$$
C_i(t) = \frac{A_c(t)}{V_i} \quad [\text{Mass/Volume}]
$$

**Population model:**
$$
\begin{aligned}
CL_i &= CL_{\text{pop}} \left(\frac{WT_i}{70}\right)^{0.75} \exp(\eta_{CL,i}) \\
V_i &= V_{\text{pop}} \left(\frac{WT_i}{70}\right)^{1.0} \exp(\eta_{V,i}) \\
K_{a,i} &= K_{a,\text{pop}} \exp(\eta_{Ka,i})
\end{aligned}
$$

with $\eta_i \sim \mathcal{N}(0, \Omega)$ (diagonal for V0).

**Observation model (proportional error):**
$$
C_{i,j}^{\text{obs}} = C_{i,j}^{\text{pred}} (1 + \epsilon_{i,j}), \quad \epsilon_{i,j} \sim \mathcal{N}(0, \sigma_{\text{prop}}^2)
$$

### 2.2 Target MedLang-D Syntax

```medlang
model OneCompOral {
    state A_gut     : DoseMass
    state A_central : DoseMass

    param Ka : RateConst      // 1/h
    param CL : Clearance      // L/h
    param V  : Volume         // L

    dA_gut/dt     = -Ka * A_gut
    dA_central/dt =  Ka * A_gut - (CL / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}

measure ConcPropError {
    pred : ConcMass
    obs  : ConcMass
    param sigma_prop : f64   // dimensionless SD

    log_likelihood = Normal_logpdf(
        x  = (obs / pred) - 1.0,
        mu = 0.0,
        sd = sigma_prop
    )
}

population OneCompOralPop {
    model OneCompOral

    // Population parameters
    param CL_pop  : Clearance
    param V_pop   : Volume
    param Ka_pop  : RateConst
    param omega_CL : f64
    param omega_V  : f64
    param omega_Ka : f64

    // Covariate
    input WT : Quantity<kg, f64>

    // Random effects (IIV)
    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    rand eta_V  : f64 ~ Normal(0.0, omega_V)
    rand eta_Ka : f64 ~ Normal(0.0, omega_Ka)

    // Individual parameter mapping
    bind_params(patient) {
        let w     = patient.WT / 70.0_kg     // dimensionless
        let CL_i  = CL_pop * pow(w, 0.75) * exp(eta_CL)
        let V_i   = V_pop  * w              * exp(eta_V)
        let Ka_i  = Ka_pop                  * exp(eta_Ka)

        model.CL = CL_i
        model.V  = V_i
        model.Ka = Ka_i
    }

    use_measure ConcPropError for model.C_plasma
}

timeline OneCompOralTimeline {
    at 0.0_h:
        dose { amount = 100.0_mg; to = OneCompOral.A_gut }

    at 1.0_h:  observe OneCompOral.C_plasma
    at 2.0_h:  observe OneCompOral.C_plasma
    at 4.0_h:  observe OneCompOral.C_plasma
    at 8.0_h:  observe OneCompOral.C_plasma
}

cohort OneCompCohort {
    population OneCompOralPop
    timeline  OneCompOralTimeline
    data_file "data/onecomp_synth.csv"
}
```

---

## 3. Technical Requirements

### 3.1 Implementation Language

**Recommended:** Julia or Rust

**Requirements:**
- Strong typing
- Good numerical support
- Ability to call Stan or generate Julia code
- Clear separation: front-end (parse + typecheck) → IR → back-end (codegen)

### 3.2 AST and Type System

**Required AST Nodes:**

```
ModelDef
├── StateDecl(name, unit)
├── ParamDecl(name, unit)
├── ODEEquation(lhs_state, rhs_expr)
└── ObservableDecl(name, unit, expr)

MeasureDef
├── PredDecl(unit)
├── ObsDecl(unit)
├── ParamDecl(name, type)
└── LogLikelihoodExpr

PopulationDef
├── ModelRef
├── PopParamDecl(name, unit)
├── RandomEffectDecl(name, dist)
├── CovariateDecl(name, unit)
├── BindParamsBlock(assignments)
└── UseMeasure(measure_ref, observable_ref)

TimelineDef
└── Event[] (DoseEvent | ObserveEvent)

CohortDef
├── PopulationRef
├── TimelineRef
└── DataFile(path)
```

**Unit Type System:**

```
Unit ::=
  | Mass
  | Volume
  | Time
  | ConcMass    = Mass / Volume
  | Clearance   = Volume / Time
  | RateConst   = 1 / Time
  | Dimensionless
  | CompoundUnit(numerator[], denominator[])
```

**Type Checking Rules:**
1. `dState/dt` must have unit `Unit(State) / Time`
2. Observable units must match measure expected units
3. `exp(x)`, `log(x)`, `pow(x, y)` require `x` dimensionless (or special handling for `pow`)
4. Arithmetic: `Unit * Unit`, `Unit / Unit`, `Unit + Unit` (only if same unit)

### 3.3 Internal IR (NIR-lite)

**NIR Structure:**

```
NIRModel {
    states: [(name, unit)],
    params: [(name, unit)],
    ode_rhs: [(state_idx, expr_graph)],
    observables: [(name, unit, expr_graph)]
}

NIRPopulation {
    pop_params: [(name, unit)],
    random_effects: [(name, distribution)],
    covariates: [(name, unit)],
    param_mapping: ExprGraph,  // maps (pop_params, covariates, eta) → individual params
    measure: NIRMeasure
}

NIRMeasure {
    observable_ref: String,
    error_params: [(name, type)],
    loglik_expr: ExprGraph
}

NIRTimeline {
    events: [(time, EventType, payload)]
}

ExprGraph ::=
  | Const(value, unit)
  | Var(name, unit)
  | BinOp(op, left, right, result_unit)
  | UnaryOp(op, arg, result_unit)
```

**Design Principles:**
- Serializable (JSON/binary) for debugging
- Backend-agnostic
- Explicit unit annotations for validation

### 3.4 Backend Code Generation

**Target:** Stan/Torsten OR Julia (choose one)

**Required Outputs:**

1. **ODE System**
   ```stan
   // Stan example
   real[] ode_rhs(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
       real dydt[2];
       real Ka = theta[1];
       real CL = theta[2];
       real V  = theta[3];
       
       dydt[1] = -Ka * y[1];
       dydt[2] = Ka * y[1] - (CL/V) * y[2];
       return dydt;
   }
   ```

2. **Data Layout**
   - Individual IDs, times, observations, covariates, dosing events

3. **Parameter Blocks**
   ```stan
   parameters {
       real<lower=0> CL_pop;
       real<lower=0> V_pop;
       real<lower=0> Ka_pop;
       real<lower=0> omega_CL;
       real<lower=0> omega_V;
       real<lower=0> omega_Ka;
       real<lower=0> sigma_prop;
       vector[3] eta[N_subjects];
   }
   ```

4. **Likelihood**
   ```stan
   model {
       for (i in 1:N_subjects) {
           // Individual parameters
           real CL_i = CL_pop * pow(WT[i]/70.0, 0.75) * exp(eta[i][1]);
           // ... solve ODE, compute predictions, likelihood
       }
   }
   ```

---

## 4. Implementation Plan (Step-by-Step)

### Step 1: Grammar and Examples (Week 1)

**Tasks:**
1. Create `docs/medlang_d_minimal_grammar_v0.md`
   - Define complete grammar for V0 subset
   - EBNF or similar formal notation
   - Examples for each construct

2. Create `docs/examples/one_comp_oral_pk.medlang`
   - Full canonical example from §2.2
   - Inline comments explaining each section

3. Create `docs/examples/onecomp_synth.csv`
   - Synthetic dataset (10-20 subjects)
   - Columns: ID, TIME, DV, WT, EVID, AMT

**Deliverables:**
- Grammar specification
- Runnable example file
- Test dataset

### Step 2: AST and Parser (Week 1-2)

**Tasks:**
1. Create `src/ast/`
   - `mod.rs` / `ast.jl`: AST node definitions
   - Clear struct/enum definitions for all constructs
   - Pretty-printing for debugging

2. Create `src/parser/`
   - Hand-rolled recursive descent OR combinator library
   - Parse tokens → AST
   - Informative error messages (line, column, context)

3. Add unit tests
   - Valid parse: `one_comp_oral_pk.medlang`
   - Invalid syntax: intentional errors
   - Edge cases: whitespace, comments, numeric literals

**Deliverables:**
- Working parser
- 10+ unit tests
- Error message examples

### Step 3: Unit System and Type Checker (Week 2)

**Tasks:**
1. Create `src/types/`
   - `unit.rs` / `unit.jl`: Unit enum and operations
   - Unit arithmetic: `*, /, +, -`
   - Unit checking functions

2. Implement type checker
   - Walk AST, assign units to all expressions
   - Validate ODE RHS units
   - Validate observable units vs. measure
   - Check `exp`, `log` only on dimensionless

3. Add unit tests
   - Valid: canonical example
   - Invalid: unit mismatches (CL + Ka, exp(CL), etc.)

**Deliverables:**
- Unit type system
- Type checker with error messages
- 20+ unit tests covering valid/invalid cases

### Step 4: IR Lowering (Week 2-3)

**Tasks:**
1. Create `src/ir/`
   - `nir.rs` / `nir.jl`: NIR data structures
   - Serialization (JSON/Debug)

2. Implement lowering pass
   - AST → NIR for model
   - AST → NIR for population
   - AST → NIR for measure
   - AST → NIR for timeline

3. Add integration tests
   - Parse → Type-check → Lower → Inspect NIR
   - Validate NIR structure matches expected

**Deliverables:**
- NIR definitions
- Lowering compiler pass
- NIR serialization examples

### Step 5: Backend Codegen (Week 3-4)

**Tasks:**
1. Create `src/backend/stan/` OR `src/backend/julia/`
   - Choose one backend for V0
   - Template-based or programmatic code generation

2. Implement code generation
   - ODE RHS function
   - Data layout (event table, covariate matrix)
   - Parameter declarations
   - Likelihood evaluation

3. Add codegen tests
   - Generate Stan/Julia code
   - Compile generated code (validate syntax)
   - Run simple simulation test

**Deliverables:**
- Backend code generator
- Generated `onecomp.stan` or `onecomp.jl`
- Compilation validation

### Step 6: CLI and Integration (Week 4)

**Tasks:**
1. Create `src/main.rs` / `src/main.jl`
   - CLI: `medlangc <input.medlang> --data <data.csv> --backend stan`
   - Pipeline: parse → typecheck → IR → codegen
   - Write output files

2. Add end-to-end tests
   - Full pipeline on canonical example
   - Compare simulation results to analytic solution
   - Validate log-likelihood computation

3. Create `README.md` for usage
   - Installation instructions
   - Example usage
   - Expected outputs

**Deliverables:**
- Working CLI tool
- Integration tests
- User documentation

### Step 7: Validation and Documentation (Week 4-5)

**Tasks:**
1. Numerical validation
   - Simulate with known parameters
   - Compare to analytic 1-comp oral solution
   - Tolerance: < 1e-6 relative error

2. Log-likelihood validation
   - Compute log-likelihood with synthetic data
   - Compare MedLang vs. direct Stan/Julia
   - Tolerance: < 1e-10 absolute difference

3. Documentation
   - API docs (inline comments)
   - Architecture diagram (AST → IR → Backend)
   - Design decisions document

**Deliverables:**
- Validation report
- Full API documentation
- Architecture documentation

---

## 5. Quality Standards

### 5.1 Code Quality

- **Clarity over cleverness:** Readable, well-commented code
- **Separation of concerns:** Clean module boundaries
- **Explicit over implicit:** No magic constants, clear naming
- **Q1 journal level:** Publication-ready code quality

### 5.2 Documentation

- **Mathematical rigor:** Reference equations from specs
- **Design rationale:** Explain non-obvious choices
- **Examples:** Working examples for every feature
- **API docs:** Inline documentation for all public functions

### 5.3 Testing

- **Unit tests:** Every module, every function
- **Integration tests:** Full pipeline end-to-end
- **Validation tests:** Numerical accuracy checks
- **Coverage:** Aim for >80% code coverage

### 5.4 Error Handling

- **Informative messages:** File, line, column, context
- **Suggestions:** Hint at fixes where possible
- **Categories:** Parse errors, type errors, runtime errors
- **No silent failures:** Fail fast and loudly

---

## 6. Success Metrics

**Vertical Slice 0 is complete when:**

1. ✅ Canonical example parses without errors
2. ✅ Type checker catches all intentional unit violations
3. ✅ Generated Stan/Julia code compiles
4. ✅ Simulation matches analytic solution (< 1e-6 error)
5. ✅ Log-likelihood matches direct computation (< 1e-10 error)
6. ✅ Full Bayesian inference runs (optional bonus)
7. ✅ All tests pass
8. ✅ Documentation is complete

**Stretch Goals:**
- Support for 2-compartment model
- Additive error model
- Multi-variate random effects (non-diagonal Omega)
- Julia AND Stan backends

---

## 7. Non-Goals (Explicitly Deferred)

**DO NOT implement in V0:**
- Track C quantum operators
- PBPK/QSP multi-scale models
- ML submodels
- General function definitions
- Advanced error models
- Optimization (focus on correctness first)

**These will be V0.2+.**

---

## 8. References

- **MedLang Core Spec:** `docs/medlang_core_spec_v0.1.md`
- **Track D Spec:** `docs/medlang_pharmacometrics_qsp_spec_v0.1.md`
- **Track C Spec:** `docs/medlang_qm_pharmacology_spec_v0.1.md` (reference only)
- **Stan Manual:** https://mc-stan.org/docs/
- **Julia DiffEq:** https://diffeq.sciml.ai/
- **Turing.jl:** https://turing.ml/

---

## Appendix A: Sample Generated Stan Code

```stan
functions {
    real[] ode_rhs(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
        real dydt[2];
        real Ka = theta[1];
        real CL = theta[2];
        real V  = theta[3];
        
        dydt[1] = -Ka * y[1];                    // dA_gut/dt
        dydt[2] = Ka * y[1] - (CL/V) * y[2];    // dA_central/dt
        return dydt;
    }
}

data {
    int<lower=1> N_subjects;
    int<lower=1> N_obs;
    int<lower=1,upper=N_subjects> subject_id[N_obs];
    real<lower=0> times[N_obs];
    real<lower=0> DV[N_obs];
    real<lower=0> WT[N_subjects];
    // ... dosing data
}

parameters {
    real<lower=0> CL_pop;
    real<lower=0> V_pop;
    real<lower=0> Ka_pop;
    real<lower=0> omega_CL;
    real<lower=0> omega_V;
    real<lower=0> omega_Ka;
    real<lower=0> sigma_prop;
    
    vector[3] eta[N_subjects];
}

model {
    // Priors
    CL_pop ~ lognormal(log(10), 1);
    V_pop ~ lognormal(log(50), 1);
    Ka_pop ~ lognormal(log(1), 1);
    omega_CL ~ halfnormal(1);
    omega_V ~ halfnormal(1);
    omega_Ka ~ halfnormal(1);
    sigma_prop ~ halfnormal(1);
    
    // Random effects
    for (i in 1:N_subjects) {
        eta[i] ~ normal(0, [omega_CL, omega_V, omega_Ka]);
    }
    
    // Likelihood
    for (i in 1:N_subjects) {
        real CL_i = CL_pop * pow(WT[i]/70.0, 0.75) * exp(eta[i][1]);
        real V_i = V_pop * (WT[i]/70.0) * exp(eta[i][2]);
        real Ka_i = Ka_pop * exp(eta[i][3]);
        
        // ... solve ODE, extract predictions
        // ... compute likelihood
    }
}
```

---

**End of Implementation Guide V0**

*This document will be updated as implementation progresses.*
