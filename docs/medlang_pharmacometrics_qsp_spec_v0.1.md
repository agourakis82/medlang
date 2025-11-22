# MedLang Pharmacometrics & QSP Specification — v0.1 (Draft)

**Status:** Draft  
**Scope:** Pharmacokinetics (PK), pharmacodynamics (PD), physiologically based pharmacokinetic models (PBPK), quantitative systems pharmacology (QSP), nonlinear mixed–effects (NLME) population models, and probabilistic/Bayesian formulations on top of the MedLang core calculus.

---

## 1. Introduction and Scope

MedLang is a domain–specific language for computational medicine whose core calculus already defines:

- base scalar and tensor types,
- unit–aware quantities,
- clinical entities (`Patient`, `Cohort`, `Timeline`),
- and probabilistic primitives (`Measure`, `ProbKernel`, `Model`).

This document specifies **Track D**: the pharmacometrics and QSP layer of MedLang.

Track D provides a **formal and executable language** for:

1. **Pharmacokinetics (PK)**  
   Representation of drug absorption, distribution, metabolism, and excretion via:
   - compartmental models (1‑, 2‑, 3‑compartment, transit models),
   - physiologically based pharmacokinetic (PBPK) models at organ scale.

2. **Pharmacodynamics (PD)**  
   Representation of drug effects on biomarkers, clinical endpoints, and disease states via:
   - direct effect models (Emax, sigmoidal Emax, log–linear),
   - indirect response models and turnover models,
   - disease progression models (e.g. tumor growth, viral dynamics).

3. **Quantitative Systems Pharmacology (QSP)**  
   Mechanistic systems linking drug and disease through:
   - medium– to large–scale ODE systems,
   - optional PDE / spatial extensions (e.g. tissue penetration),
   - multi–scale couplings (cell, tissue, organ, organism).

4. **Population Modelling (NLME)**  
   Explicit representation of inter–individual and intra–individual variability via:
   - random effects on structural parameters (nonlinear mixed–effects),
   - covariate models (e.g. body weight, genotype, organ function),
   - residual unexplained variability (RUV) models.

5. **Probabilistic and Bayesian Semantics**  
   Language–level support for:
   - likelihood definitions and observation models,
   - priors on parameters and hyperparameters,
   - population and individual posterior predictive simulation.

6. **Hybrid Mechanistic–ML Models**  
   Embedding machine learning components into mechanistic models, such as:
   - learned sub–models for partition coefficients, clearance, or unknown dynamics,
   - neural ODE / PINN–style formulations constrained by mechanistic structure.

Track D is designed to be:

- **backend–agnostic**: compatible with classical NLME engines and modern probabilistic programming systems;
- **interop–ready**: alignable with NONMEM/Monolix/Stan/Torsten on the modelling side, and with FHIR/CQL on the data side;
- **quantum–ready**: capable of consuming parameters produced by quantum pharmacology operators (Track C) such as `DFT`, `QM_MM`, and `Compile_QM→PK`.

This specification focuses on **semantics and surface syntax**, not on the details of numerical solvers or inference algorithms. Those are treated as interchangeable backends that must respect the contracts defined here.

---

## 2. Conceptual Model

This section introduces the conceptual layers of Track D and their relationship to the MedLang core.

### 2.1 Levels of Description

Track D models operate simultaneously at three conceptual levels:

1. **Individual (deterministic) level**  
   A fixed parameter vector θ defines a deterministic dynamical system:
   ```
   dX(t)/dt = f(X(t), θ, u(t), t)
   ```
   where:
   - X(t) is the state vector (compartment amounts, concentrations, biomarkers),
   - u(t) encodes exogenous inputs (doses, interventions),
   - f is a vector field specified by a MedLang `Model`.

2. **Population (hierarchical) level**  
   Individual parameter vectors θᵢ are treated as draws from population distributions:
   ```
   θᵢ ~ p(θ | φ)
   ```
   where:
   - φ denotes population–level parameters (typical values and variability),
   - the distribution p is represented by a `ProbKernel` attached to the `Model`.

3. **Observation (measurement) level**  
   Observations yᵢⱼ (e.g. concentrations, biomarkers) for individual i at time tᵢⱼ are modeled as:
   ```
   yᵢⱼ ~ p(y | h(Xᵢ(tᵢⱼ), θᵢ), ψ)
   ```
   where:
   - h is a mapping from latent state to predicted observable,
   - ψ aggregates observation–level error parameters,
   - the conditional distribution is represented as a `Measure` over the observable space.

The **MedLang core types** encode these levels as:

- `Model<State, Param>` for f and h,
- `Timeline<Event>` for u(t) and observation times,
- `ProbKernel<Param, Param>` and `ProbKernel<Param, Real>` for population and prior distributions,
- `Measure<Observable>` for observation models.

### 2.2 Deterministic Structural Model

At the structural level, a Track D `Model` specifies:

- a finite set of **state variables** X = (X₁,...,Xₙ),
- a finite set of **structural parameters** θ = (θ₁,...,θₚ),
- a **vector field** f: ℝⁿ × Θ × U × ℝ → ℝⁿ,
- one or more **observation maps** hₖ: ℝⁿ × Θ → ℝ for observables.

In MedLang notation this corresponds to:

```medlang
model PK_PD_Model {
    state  X : StateVector;       // e.g. amounts or concentrations
    param  θ : ParamVector;       // CL, V, Ka, etc.

    // Structural dynamics
    dX_dt = f(X, θ, u, t);

    // One or more observation channels
    obs Conc_plasma = h_plasma(X, θ);
    obs Effect      = h_effect(X, θ);
}
```

The **semantics** of `Model` at this level is that of an initial–value problem (IVP):

- given θ, initial state X₀, and input signal u(t) specified by a `Timeline`,
- the model defines a unique (possibly numerically approximated) trajectory X(t) on a time domain of interest.

Formally, Track D **does not fix** the numerical integration scheme; it only requires that any compliant backend realize a mapping:
```
(f, X₀, θ, u, {tⱼ}) ↦ {X(tⱼ)}ⱼ
```
with well–controlled error.

### 2.3 Population Structure and Random Effects

To express **population variability**, Track D attaches probabilistic structure to parameters.

For an individual i, let θᵢ be decomposed into:

- **fixed–effect component** θᶠⁱˣ (e.g. typical values),
- **random–effect component** ηᵢ (inter–individual deviations),
- **covariate component** g(zᵢ) (deterministic function of covariates zᵢ, such as weight, age, genotype).

A common decomposition is:
```
θᵢ = g(zᵢ; β) ⊙ exp(ηᵢ)
```
with:
```
ηᵢ ~ N(0, Ω)
```
where ⊙ is element–wise multiplication and Ω is a covariance matrix.

In MedLang terms:

- the distribution ηᵢ ~ N(0,Ω) is captured by a `ProbKernel<Unit, ParamVector>`,
- the covariate model g(zᵢ; β) is a deterministic mapping in the `Model`,
- the composition yielding θᵢ is represented in the parameter–building section of the `Model`.

Track D **does not constrain** the form of `ProbKernel` to be Gaussian; log–normal, t‑distributions, or custom distributions are allowed, as long as they are representable in the core probabilistic calculus.

### 2.4 Observation Model and Residual Error

Observation models in pharmacometrics typically combine:

1. A **deterministic prediction** ŷᵢⱼ = h(Xᵢ(tᵢⱼ), θᵢ).
2. A **residual error model**:
   - additive: yᵢⱼ = ŷᵢⱼ + εᵢⱼ,
   - proportional: yᵢⱼ = ŷᵢⱼ(1 + εᵢⱼ),
   - combined, or more complex structures,
   
   where εᵢⱼ are i.i.d. (or structured) random variables.

In MedLang, a `Measure<Observable>` object formalizes:
```
p(yᵢⱼ | ŷᵢⱼ, ψ)
```
where ψ collects error–model parameters (e.g. standard deviations, exponents).

Conceptually:

- `Measure` defines the **likelihood contribution** at each observation point,
- the product over all observations in all individuals yields the joint likelihood used for estimation or Bayesian inference.

---

## 3. Core Track D Constructs in MedLang

This section defines the Track D–specific use of several core MedLang constructs and introduces the additional structure required for pharmacometrics and QSP.

### 3.1 `Model` for PK/PD and QSP

**Intuition.**
A Track D `Model` is a **typed container** for:

- state declarations,
- parameter declarations,
- structural dynamics (ODE/PDE),
- observation channels,
- optional algebraic and derived quantities.

**Signature.**
At the type level, a Track D model is a specialization:
```
Model<State, Param>
```
with:

- `State` typically a tuple or record of `Quantity`–typed fields,
- `Param` a record of scalar or tensor parameters (possibly with units).

**Obligations of a Track D `Model`:**

1. Provide a well–typed right–hand side for each differential state:
   ```
   dXₖ/dt = fₖ(X,θ,u,t)
   ```
   respecting unit consistency at compile time.

2. Optionally define named **observables** as pure functions of `(X, θ)`.

3. Define how exogenous inputs `u(t)` are bound from a `Timeline`.

4. Expose a canonical **parameterization** interface:
   - fixed effects,
   - random–effects structure (references to `ProbKernel`),
   - covariate mappings.

The exact surface syntax is specified in §4; the present section fixes the conceptual contract.

### 3.2 `Timeline` for Dosing and Observation

In Track D, `Timeline<Event>` represents a **piecewise–structured time axis** with:

- dosing events (e.g. oral dose, IV bolus, infusion start/stop),
- sampling events (times at which `Measure`s are evaluated),
- optional intervention events (e.g. covariate changes, co‑medication).

A `Timeline` for individual i can be seen as an ordered multiset:
```
𝒯ᵢ = {(tₖ, eₖ)}ₖ₌₁ᴷⁱ
```
where each eₖ is an event of a well–defined event type.

Track D assumes:

- Dosing events are converted to an **input function** u(t) for the `Model` (e.g. impulses or infusion rates).
- Sampling events define the times {tᵢⱼ} at which observables are evaluated and `Measure`s instantiated.

The typing of event payloads (dose amount, administration route, etc.) must use appropriate `Quantity` types (e.g. `mg`, `mg/kg`, `mg/h`).

### 3.3 `ProbKernel` for Variability and Priors

`ProbKernel<X,Y>` in Track D is used in two main roles:

1. **Population variability (random effects)**  
   Mapping from a unit/context (e.g. an individual index and covariates) to a distribution over parameters, such as:
   ```
   Individual i ↦ θᵢ ~ p(θ | zᵢ, φ)
   ```

2. **Priors and hyperpriors**  
   For Bayesian formulations, `ProbKernel` expresses:
   - priors on population parameters φ,
   - priors on error model parameters ψ,
   - potentially hyperpriors at higher hierarchy levels.

Track D does not commit to a single estimation or inference algorithm; instead, it requires that:

- every `ProbKernel` has a **densities / log–densities** interface available to downstream inference backends,
- compositions of kernels (e.g. hierarchical models) are expressible in the core calculus.

### 3.4 `Measure` for Observation Models

`Measure<Observable>` in Track D associates:

- a **prediction** ŷ (from `Model`),
- error–model parameters ψ,
- and an observed value y,

with a probability law p(y | ŷ, ψ).

Conceptually:

- At the level of a single datum, `Measure` provides a log–likelihood contribution log p(y | ŷ, ψ).
- Over a dataset, the product (or sum of logs) defines the overall likelihood.

Different residual error structures (additive, proportional, combined, censoring) correspond to different `Measure` constructors or configurations.

---

## 4. Outline of Remaining Sections

The following sections are to be filled in subsequent iterations:

- **4. Typing and Unit Semantics in Track D**
  - Units for typical PK/PD quantities (dose, concentration, clearance, volume, time).
  - Dimensional analysis rules for ODE right–hand sides.
  - Type rules for `Timeline` events and `Measure`s.

- **5. Structural Model Patterns**
  - Canonical building blocks:
    - 1‑compartment and 2‑compartment PK.
    - Saturable and time‑varying clearance.
    - Standard PD models (Emax, indirect response).
  - Reusable MedLang templates and suggested naming conventions.

- **6. Population and NLME Semantics**
  - Formal treatment of random effects, covariates, and intra–individual variability.
  - Relation to NONMEM/Monolix/Stan abstractions.
  - Hierarchical model diagrams and their encoding in MedLang.

- **7. Inference Modes and Backend Contracts**
  - Definitions of:
    - simulation–only mode (no parameter learning),
    - frequentist NLME mode (likelihood maximization),
    - Bayesian mode (posterior inference).
  - Required interfaces for inference engines (log–likelihood, log–prior, gradients).

- **8. Hybrid Mechanistic–ML and PINN Integration**
  - Syntax and semantics for embedding learned components (`Model<X,Y>`) into Track D models.
  - Constraints for safe integration (type and unit consistency, monotonicity where needed).
  - Hooks for physics–informed training (PINN‑style losses).

- **9. Worked Examples**
  - **Example 1:** One‑compartment oral PK with log–normal IIV on CL, weight covariate, proportional error. ✓ (See below)
  - **Example 2:** Simple QSP model (drug + biomarker + tumor) with random effects.
  - **Example 3 (optional):** PBPK model with quantum‑derived parameters via `Compile_QM→PK`.

- **10. Implementation Notes and IR Mapping**
  - How Track D constructs are represented in CIR and NIR.
  - Lowering patterns to MLIR (ODE op, log–pdf ops, event handling).
  - Considerations for batched simulation and GPU execution.

---

## 9. Worked Examples

### Example 1: One-Compartment Oral PK with Population Variability

This example demonstrates a complete Track D pharmacometric model: a one-compartment oral absorption PK model with log-normal inter-individual variability on clearance (CL) and volume (V), an allometric body weight covariate on both parameters, and proportional residual error.

#### 9.1.1 Clinical and Mathematical Context

**Clinical scenario:**  
A population pharmacokinetic analysis of an orally administered drug. The study enrolled N subjects with varying body weights. Each subject received one or more oral doses, and plasma concentrations were measured at multiple time points.

**Structural model:**  
The one-compartment model with first-order absorption is described by:

```
dA_depot/dt  = -Ka · A_depot
dA_central/dt = Ka · A_depot - (CL/V) · A_central
```

where:
- A_depot: amount of drug in the absorption depot (mg)
- A_central: amount of drug in the central compartment (mg)
- Ka: absorption rate constant (1/h)
- CL: clearance (L/h)
- V: volume of distribution (L)

The predicted plasma concentration is:
```
C_plasma = A_central / V   (mg/L)
```

**Population model:**  
Individual parameters are related to population typical values and covariates via:

```
CLᵢ = CL_pop · (WTᵢ/70)^0.75 · exp(η_CL,i)
Vᵢ  = V_pop  · (WTᵢ/70)^1.0  · exp(η_V,i)
Kaᵢ = Ka_pop · exp(η_Ka,i)
```

where:
- CL_pop, V_pop, Ka_pop: population typical values
- WTᵢ: body weight of individual i (kg), normalized to 70 kg
- η_CL,i, η_V,i, η_Ka,i: random effects for individual i

The random effects are assumed multivariate normal:
```
[η_CL,i]       [0]   [ω²_CL    ρ·ω_CL·ω_V    0      ]
[η_V,i  ] ~ N( [0] , [ρ·ω_CL·ω_V  ω²_V        0      ] )
[η_Ka,i]       [0]   [0           0          ω²_Ka  ]
```

where ρ is the correlation between CL and V random effects (often positive due to physiological coupling).

**Observation model:**  
Observed concentrations are modeled with proportional error:
```
yᵢⱼ = Ĉᵢ(tᵢⱼ) · (1 + εᵢⱼ)
εᵢⱼ ~ N(0, σ²_prop)
```

where:
- yᵢⱼ: observed concentration for individual i at time tᵢⱼ
- Ĉᵢ(tᵢⱼ): predicted concentration from the ODE solution
- σ_prop: proportional error standard deviation

#### 9.1.2 MedLang Implementation

**Step 1: Define units and types**

```medlang
// Units for this model
unit mg;           // milligrams
unit kg;           // kilograms
unit L;            // liters
unit h;            // hours
unit mg_per_L = mg / L;
unit L_per_h  = L / h;
unit per_h    = 1 / h;

// State type for the ODE system
struct PKState {
    A_depot:   Quantity<mg, f64>,
    A_central: Quantity<mg, f64>
}

// Parameter type for individual parameters
struct PKParams {
    CL: Quantity<L_per_h, f64>,
    V:  Quantity<L, f64>,
    Ka: Quantity<per_h, f64>
}

// Population-level parameters
struct PopParams {
    CL_pop: Quantity<L_per_h, f64>,
    V_pop:  Quantity<L, f64>,
    Ka_pop: Quantity<per_h, f64>,
    
    omega_CL: f64,      // SD of log(CL) random effect
    omega_V:  f64,      // SD of log(V) random effect
    omega_Ka: f64,      // SD of log(Ka) random effect
    rho_CL_V: f64,      // correlation between η_CL and η_V
    
    sigma_prop: f64     // proportional error SD
}

// Random effects (log-scale deviations)
struct RandomEffects {
    eta_CL: f64,
    eta_V:  f64,
    eta_Ka: f64
}

// Individual covariates
struct Covariates {
    weight: Quantity<kg, f64>
}
```

**Step 2: Define the structural model**

```medlang
model OneCptOralPK {
    // State variables
    state X: PKState;
    
    // Individual parameters
    param θ: PKParams;
    
    // Exogenous input (dosing)
    input dose_rate: Quantity<mg_per_h, f64>;
    
    // ODE system
    dX.A_depot/dt = -θ.Ka * X.A_depot;
    
    dX.A_central/dt = θ.Ka * X.A_depot 
                      - (θ.CL / θ.V) * X.A_central
                      + dose_rate;  // handles IV infusions if needed
    
    // Observable: plasma concentration
    obs C_plasma: Quantity<mg_per_L, f64> = X.A_central / θ.V;
    
    // Initial conditions
    init {
        X.A_depot   = 0.0 mg;
        X.A_central = 0.0 mg;
    }
}
```

**Step 3: Define the population model (ProbKernel)**

```medlang
// Function to compute individual parameters from population parameters,
// covariates, and random effects
fn individual_params(
    pop: PopParams,
    cov: Covariates,
    eta: RandomEffects
) -> PKParams {
    // Allometric scaling factors
    let wt_norm = cov.weight / (70.0 kg);
    let size_CL = pow(wt_norm, 0.75);
    let size_V  = pow(wt_norm, 1.0);
    
    // Individual parameters with log-normal variability
    PKParams {
        CL: pop.CL_pop * size_CL * exp(eta.eta_CL),
        V:  pop.V_pop  * size_V  * exp(eta.eta_V),
        Ka: pop.Ka_pop * exp(eta.eta_Ka)
    }
}

// ProbKernel for random effects: multivariate normal with correlation
kernel PopulationVariability(pop: PopParams) -> ProbKernel<Covariates, RandomEffects> {
    // Construct covariance matrix
    let Omega = [
        [pop.omega_CL^2,                      pop.rho_CL_V * pop.omega_CL * pop.omega_V,  0.0],
        [pop.rho_CL_V * pop.omega_CL * pop.omega_V,  pop.omega_V^2,                          0.0],
        [0.0,                                  0.0,                                         pop.omega_Ka^2]
    ];
    
    // Return kernel: for any individual with covariates, 
    // sample eta ~ MVN(0, Omega)
    return |cov: Covariates| -> Measure<RandomEffects> {
        MultivariateNormal(mean: [0.0, 0.0, 0.0], cov: Omega)
    };
}
```

**Step 4: Define the observation model (Measure)**

```medlang
// Proportional error observation model
measure ProportionalError(
    predicted: Quantity<mg_per_L, f64>,
    sigma_prop: f64
) -> Measure<Quantity<mg_per_L, f64>> {
    // Observed = Predicted · (1 + ε), where ε ~ N(0, σ²)
    // Equivalently: Observed ~ N(Predicted, (σ · Predicted)²)
    
    let sd = sigma_prop * predicted;
    Normal(mean: predicted, sd: sd)
}
```

**Step 5: Define dosing timeline**

```medlang
// Example: single oral dose of 100 mg at t=0
fn example_dosing() -> Timeline<DoseEvent> {
    Timeline::new([
        DoseEvent {
            time: 0.0 h,
            amount: 100.0 mg,
            route: Oral,
            compartment: "A_depot"  // targets the depot compartment
        }
    ])
}

// Example: observation times
fn example_sampling() -> Timeline<ObsEvent> {
    Timeline::new([
        ObsEvent { time: 0.5 h },
        ObsEvent { time: 1.0 h },
        ObsEvent { time: 2.0 h },
        ObsEvent { time: 4.0 h },
        ObsEvent { time: 8.0 h },
        ObsEvent { time: 12.0 h },
        ObsEvent { time: 24.0 h }
    ])
}
```

**Step 6: Simulation for a single individual**

```medlang
fn simulate_individual(
    pop: PopParams,
    cov: Covariates,
    eta: RandomEffects,
    dosing: Timeline<DoseEvent>,
    sampling: Timeline<ObsEvent>,
    seed: u64
) -> Vec<Quantity<mg_per_L, f64>> {
    // Compute individual parameters
    let theta = individual_params(pop, cov, eta);
    
    // Instantiate model
    let model = OneCptOralPK { θ: theta };
    
    // Integrate ODE over timeline
    let trajectory = integrate_ode(
        model: model,
        dosing: dosing,
        t_end: 24.0 h,
        method: LSODA
    );
    
    // Extract predictions at sampling times
    let predictions = sampling.times.map(|t| {
        trajectory.at(t).C_plasma
    });
    
    // Add observation error
    random(seed) {
        predictions.map(|pred| {
            sample(ProportionalError(pred, pop.sigma_prop))
        })
    }
}
```

**Step 7: Population simulation (forward)**

```medlang
fn simulate_population(
    pop: PopParams,
    cohort: Cohort<Patient>,
    dosing_schedule: fn(Patient) -> Timeline<DoseEvent>,
    sampling_schedule: fn(Patient) -> Timeline<ObsEvent>,
    seed: u64
) -> Vec<Vec<Quantity<mg_per_L, f64>>> {
    
    let kernel = PopulationVariability(pop);
    
    // For each patient, sample random effects and simulate
    random(seed) {
        par_map(|patient: Patient| {
            // Extract covariates
            let cov = Covariates { weight: patient.weight };
            
            // Sample random effects
            let eta = sample(kernel(cov));
            
            // Simulate this individual
            simulate_individual(
                pop, 
                cov, 
                eta,
                dosing_schedule(patient),
                sampling_schedule(patient),
                derive_seed(seed, patient.id)
            )
        }, cohort)
    }
}
```

**Step 8: Likelihood for parameter estimation**

```medlang
fn log_likelihood(
    pop: PopParams,
    data: CohortData<Concentration>
) -> f64 {
    
    let kernel = PopulationVariability(pop);
    
    // Sum log-likelihood contributions over all individuals
    data.individuals.map(|indiv| {
        let cov = Covariates { weight: indiv.weight };
        
        // Marginalize over random effects (simplified; real impl uses Laplace or importance sampling)
        let eta_dist = kernel(cov);
        
        // For demonstration: evaluate at mode (η = 0) - this is First-Order approximation
        let eta = RandomEffects { eta_CL: 0.0, eta_V: 0.0, eta_Ka: 0.0 };
        let theta = individual_params(pop, cov, eta);
        
        // Integrate model
        let model = OneCptOralPK { θ: theta };
        let traj = integrate_ode(model: model, dosing: indiv.dosing, t_end: indiv.t_max);
        
        // Sum log-likelihood over observations
        indiv.observations.map(|obs| {
            let pred = traj.at(obs.time).C_plasma;
            let meas = ProportionalError(pred, pop.sigma_prop);
            log_pdf(meas, obs.value)
        }).sum()
    }).sum()
}
```

**Step 9: Bayesian inference (sketch)**

```medlang
// Define priors on population parameters
kernel PopPriors() -> Measure<PopParams> {
    PopParams {
        CL_pop:     LogNormal(log(10.0 L/h), 0.5),
        V_pop:      LogNormal(log(50.0 L), 0.5),
        Ka_pop:     LogNormal(log(1.0 /h), 0.5),
        omega_CL:   HalfNormal(0.3),
        omega_V:    HalfNormal(0.3),
        omega_Ka:   HalfNormal(0.5),
        rho_CL_V:   Uniform(-1.0, 1.0),
        sigma_prop: HalfNormal(0.2)
    }
}

// Posterior: p(pop | data) ∝ p(data | pop) · p(pop)
fn posterior_log_density(
    pop: PopParams,
    data: CohortData<Concentration>
) -> f64 {
    let log_prior = log_pdf(PopPriors(), pop);
    let log_lik   = log_likelihood(pop, data);
    log_prior + log_lik
}

// Run MCMC (delegated to backend, e.g., Stan, NumPyro)
fn run_inference(data: CohortData<Concentration>) -> Measure<PopParams> {
    // This is a conceptual interface; actual implementation would:
    // 1. Export model to Stan/NumPyro/etc.
    // 2. Run HMC/NUTS
    // 3. Return posterior as a Measure (empirical distribution from samples)
    
    infer_posterior(
        log_density: |pop| posterior_log_density(pop, data),
        prior: PopPriors(),
        method: HMC(num_samples: 2000, warmup: 1000),
        backend: Stan
    )
}
```

#### 9.1.3 Comparison with Existing Frameworks

**NONMEM equivalent (simplified):**

```
$PROB One-compartment oral PK with IIV and weight covariate

$INPUT ID TIME AMT DV WT EVID MDV

$SUBROUTINES ADVAN2 TRANS2

$PK
TVCL = THETA(1) * (WT/70)**0.75
TVV  = THETA(2) * (WT/70)
TVKA = THETA(3)

CL = TVCL * EXP(ETA(1))
V  = TVV  * EXP(ETA(2))
KA = TVKA * EXP(ETA(3))

$ERROR
IPRED = F
Y = IPRED * (1 + EPS(1))

$THETA
(0, 10)   ; CL_pop
(0, 50)   ; V_pop
(0, 1)    ; Ka_pop

$OMEGA BLOCK(2)
0.09      ; omega_CL^2
0.01 0.09 ; cov(CL,V), omega_V^2

$OMEGA
0.25      ; omega_Ka^2

$SIGMA
0.04      ; sigma_prop^2
```

**Stan/Torsten equivalent (simplified structure):**

```stan
data {
  int<lower=1> N_obs;
  int<lower=1> N_indiv;
  vector[N_obs] time;
  vector[N_obs] dv;
  vector[N_indiv] weight;
  // ... dosing data ...
}

parameters {
  real<lower=0> CL_pop;
  real<lower=0> V_pop;
  real<lower=0> Ka_pop;
  real<lower=0> omega_CL;
  real<lower=0> omega_V;
  real<lower=0> omega_Ka;
  real<lower=-1,upper=1> rho_CL_V;
  real<lower=0> sigma_prop;
  
  matrix[N_indiv, 3] eta_raw;
}

transformed parameters {
  // ... build individual params with allometric scaling and random effects ...
}

model {
  // Priors
  CL_pop ~ lognormal(log(10), 0.5);
  // ... other priors ...
  
  // Random effects
  to_vector(eta_raw) ~ std_normal();
  
  // Likelihood (simplified; real Torsten uses pmx_solve_*)
  for (i in 1:N_obs) {
    dv[i] ~ normal(predicted[i], sigma_prop * predicted[i]);
  }
}
```

**MedLang advantages:**

1. **Type and unit safety**: Compile-time checking prevents mg/L vs mg/mL errors, CL with wrong units, etc.
2. **Explicit structure**: `Model`, `ProbKernel`, `Measure`, `Timeline` make the hierarchical structure transparent.
3. **Composability**: The same `OneCptOralPK` model can be:
   - Simulated forward,
   - Used in likelihood-based estimation,
   - Embedded in Bayesian inference,
   - Extended to PBPK or QSP without rewriting the core.
4. **Backend flexibility**: Same model can target NONMEM, Stan, custom GPU solvers, etc.
5. **Quantum readiness**: Parameters like `CL` could be computed from `Compile_QM→PK(molecule)` in Track C.

#### 9.1.4 Type Checking and Dimensional Analysis

The MedLang compiler performs the following checks on this model:

**ODE right-hand side dimensional consistency:**

```
dX.A_depot/dt has type Quantity<mg/h, f64>
  -θ.Ka * X.A_depot
  = Quantity<per_h, f64> * Quantity<mg, f64>
  = Quantity<mg/h, f64>  ✓

dX.A_central/dt has type Quantity<mg/h, f64>
  θ.Ka * X.A_depot
  = Quantity<per_h, f64> * Quantity<mg, f64>
  = Quantity<mg/h, f64>  ✓
  
  (θ.CL / θ.V) * X.A_central
  = (Quantity<L/h, f64> / Quantity<L, f64>) * Quantity<mg, f64>
  = Quantity<per_h, f64> * Quantity<mg, f64>
  = Quantity<mg/h, f64>  ✓
```

**Observable type:**

```
C_plasma = X.A_central / θ.V
         = Quantity<mg, f64> / Quantity<L, f64>
         = Quantity<mg/L, f64>  ✓
```

**Parameter transformations:**

```
CL = pop.CL_pop * size_CL * exp(eta.eta_CL)
   = Quantity<L/h, f64> * f64 * f64
   = Quantity<L/h, f64>  ✓
```

All operations type-check correctly; any unit mismatch (e.g., adding `mg` and `L`) would be rejected at compile time.

#### 9.1.5 Execution Semantics

**Single individual simulation:**

1. **Parameter computation**: `individual_params` is a pure function evaluated once.
2. **ODE integration**: The `integrate_ode` operation:
   - Constructs the right-hand side function `f(X, t)` from the `model.dX/dt` definitions,
   - Handles dose events from `Timeline` by impulse or infusion injection,
   - Calls a numerical solver (LSODA, CVODE, etc.) to produce trajectory,
   - Returns a `Trajectory<PKState>` object with `.at(t)` interface.
3. **Observation**: For each sampling time, evaluate `trajectory.at(t).C_plasma`.
4. **Error model**: Sample from `ProportionalError(pred, sigma)` using the RNG seeded by `seed`.

**Population simulation:**

1. **Parallel execution**: `par_map` over cohort schedules parallel execution:
   - Each individual's simulation is independent,
   - Can be distributed across CPU cores or GPU with SPMD pattern.
2. **Random effect sampling**: Each individual draws `eta ~ MVN(0, Omega)` once.
3. **Result**: A `Vec<Vec<Concentration>>` (outer: individuals, inner: time points).

**Likelihood computation:**

1. **Marginalization**: True NLME likelihood requires integrating over random effects:
   ```
   L(pop | data) = ∏ᵢ ∫ p(yᵢ | ηᵢ, pop) p(ηᵢ | pop) dηᵢ
   ```
   The example uses First-Order approximation (η = 0); real implementations use:
   - Laplace approximation (FOCE),
   - Importance sampling,
   - Stochastic EM (SAEM),
   - or HMC (Bayesian).

2. **Gradients**: For optimization-based inference, MedLang can compute:
   - `∇_pop log L(pop | data)` via autodiff through the ODE solver (adjoint method).

**Bayesian inference:**

1. **Prior specification**: `PopPriors()` defines a `Measure<PopParams>`.
2. **Posterior**: Proportional to `exp(log_prior + log_lik)`.
3. **Sampling**: Delegated to backend (Stan HMC, NumPyro NUTS, etc.):
   - MedLang exports the model as Stan code or JAX/PyTorch functions,
   - Backend runs MCMC,
   - Samples are imported back as `Measure<PopParams>` (empirical distribution).

---

*This completes Example 1. The remaining examples (QSP model, PBPK with quantum parameters) will be added in subsequent iterations.*

---

## 10. Implementation Notes and IR Mapping

(To be filled: CIR/NIR representation, MLIR lowering, GPU batching considerations)

---

*This specification is a living document and will evolve as Track D matures. Sections 4–10 will be filled iteratively with formal definitions, worked examples, and implementation guidance.*
