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

- **5. Structural Model Patterns** ✓ (See below)
  - Canonical building blocks:
    - 1‑compartment and 2‑compartment PK.
    - Saturable and time‑varying clearance.
    - Standard PD models (Emax, indirect response).
  - Reusable MedLang templates and suggested naming conventions.

- **6. Population and NLME Semantics** ✓ (See below)
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

## 5. Structural Model Patterns

This section defines **canonical structural patterns** for Track D models:

- classical PK compartment models,
- standard PD models,
- indirect response / turnover structures,
- extensible QSP skeletons.

The goal is not to constrain MedLang to these forms, but to:

1. Provide **reference patterns** and naming conventions,
2. Ensure that library templates and examples are **consistent and type‑safe**,
3. Facilitate **interoperability** with NONMEM/Monolix/Stan/Torsten formulations.

All patterns here assume the **typing and unit semantics** of Section 4.

### 5.1. Design Principles and Naming Conventions

Across Track D examples and libraries we adopt:

- **Parameter names:**
  - `CL` – clearance (Volume/Time),
  - `Q` – inter-compartmental clearance (Volume/Time),
  - `V`, `Vc`, `Vp` – volumes (Volume),
  - `Ka` – first-order absorption rate (1/Time),
  - `Ke`, `Kout`, `Kin` – elimination/turnover rates (1/Time),
  - `Emax`, `EC50`, `γ` – PD parameters (effect scale, potency, Hill coefficient).

- **States:**
  - `A_gut`, `A_central`, `A_periph` – amounts (Mass or Amount of Substance),
  - `C_plasma`, `C_effect` – concentrations (Mass/Volume) or dimensionless,
  - `R`, `Biomarker`, `Tumour` – PD/QSP states with domain‑specific units.

- **Observables:**
  - `C_plasma` – canonical plasma concentration observable,
  - `Effect` – dimensionless or unit‑specific PD effect,
  - `Biomarker_X` – explicitly named biomarkers.

These conventions are **non‑binding** at the language level but strongly recommended for clarity and alignment with pharmacometric practice.

---

### 5.2. One‑Compartment PK Models

#### 5.2.1. IV Bolus

**Mathematical structure**

Single compartment, amount state A_c(t):

```
dA_c/dt = -(CL/V) * A_c
```

with:
- A_c [Mass],
- CL [Volume/Time],
- V [Volume],
- concentration observable:
  ```
  C_plasma(t) = A_c(t) / V   [Mass/Volume]
  ```

**Pseudo‑MedLang**

```medlang
model OneCompIV {
    // States
    state A_central : DoseMass    // mg

    // Parameters
    param CL : Clearance          // L/h
    param V  : Volume             // L

    // Structural ODE
    dA_central/dt = -(CL / V) * A_central

    // Observables
    obs C_plasma : ConcMass = A_central / V
}
```

A bolus dose is represented as a `Timeline` event adding to `A_central` at a specific time.

```medlang
timeline OneCompIV_Timeline {
    at 0.0_h:
        dose {
            amount = 100.0_mg
            to     OneCompIV.A_central
        }
}
```

#### 5.2.2. IV Infusion

Now we introduce a constant infusion rate R_in [Mass/Time]:

```
dA_c/dt = R_in - (CL/V) * A_c
```

**Pseudo‑MedLang**

```medlang
model OneCompIVInfusion {
    state A_central : DoseMass       // mg
    param CL        : Clearance      // L/h
    param V         : Volume         // L

    input R_in      : FlowMass       // mg/h (from Timeline)

    dA_central/dt = R_in - (CL / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}
```

`R_in` is bound from `Timeline` infusion events:

```medlang
timeline OneCompIVInfusion_Timeline {
    at 0.0_h:
        start_infusion {
            rate   = 10.0_mg_per_h
            target = OneCompIVInfusion.A_central
        }

    at 8.0_h:
        stop_infusion {
            target = OneCompIVInfusion.A_central
        }
}
```

Semantics: between `start_infusion` and `stop_infusion`, `R_in` is constant; outside, `R_in = 0`.

#### 5.2.3. Oral First‑Order Absorption

Two amounts:
- A_g(t) – amount in gut,
- A_c(t) – amount in central compartment.

Equations:

```
dA_g/dt = -Ka * A_g
dA_c/dt = Ka * A_g - (CL/V) * A_c
```

**Pseudo‑MedLang**

```medlang
model OneCompOral {
    // States
    state A_gut     : DoseMass      // mg
    state A_central : DoseMass      // mg

    // Parameters
    param Ka : RateConst            // 1/h
    param CL : Clearance            // L/h
    param V  : Volume               // L

    // Structural dynamics
    dA_gut/dt     = -Ka * A_gut
    dA_central/dt =  Ka * A_gut - (CL / V) * A_central

    // Observable
    obs C_plasma : ConcMass = A_central / V
}
```

Oral dosing is modeled as bolus additions to `A_gut`:

```medlang
timeline OneCompOral_Timeline {
    at 0.0_h:
        dose {
            route  = Oral
            amount = 100.0_mg
            to     OneCompOral.A_gut
        }
}
```

---

### 5.3. Two‑Compartment PK Models

Two‑compartment models consist of:
- central compartment A_c(t),
- peripheral compartment A_p(t),
- elimination from the central compartment,
- distribution clearance (Q) between compartments.

Equations:

```
dA_c/dt = -(CL/V_c) * A_c - (Q/V_c) * A_c + (Q/V_p) * A_p + input(t)
dA_p/dt = (Q/V_c) * A_c - (Q/V_p) * A_p
```

with:
- A_c, A_p [Mass],
- V_c, V_p [Volume],
- CL, Q [Volume/Time].

**Pseudo‑MedLang**

```medlang
model TwoCompIV {
    // States
    state A_central   : DoseMass     // mg
    state A_peripheral: DoseMass     // mg

    // Parameters
    param CL : Clearance             // L/h
    param Q  : Clearance             // L/h    // inter-compartmental clearance
    param Vc : Volume                // L
    param Vp : Volume                // L

    // Input rate to central (for infusion or other routes)
    input R_in : FlowMass            // mg/h

    // Structural dynamics
    dA_central/dt    =
        R_in
        - (CL / Vc) * A_central
        - (Q  / Vc) * A_central
        + (Q  / Vp) * A_peripheral

    dA_peripheral/dt =
        (Q / Vc) * A_central
        - (Q / Vp) * A_peripheral

    // Observable
    obs C_plasma : ConcMass = A_central / Vc
}
```

Oral 2‑compartment variants are obtained by adding `A_gut` and a `Ka` term as in §5.2.3.

---

### 5.4. Saturable and Time‑Varying Clearance

#### 5.4.1. Michaelis–Menten (Capacity‑Limited) Elimination

In some contexts, elimination is better represented by a **Michaelis–Menten** process:

```
dA_c/dt = -(Vmax * C) / (Km + C)
```

where C = A_c / V, and:
- Vmax [Mass/Time],
- Km [Concentration].

Rewriting in terms of A_c:

```
dA_c/dt = -(Vmax * (A_c/V)) / (Km + (A_c/V))
```

**Pseudo‑MedLang**

```medlang
model OneCompIV_MM {
    state A_central : DoseMass          // mg

    param Vmax : FlowMass               // mg/h
    param Km   : ConcMass               // mg/L
    param V    : Volume                 // L

    obs C_plasma : ConcMass = A_central / V

    dA_central/dt =
        - Vmax * (C_plasma / (Km + C_plasma))
}
```

Typing check:
- `C_plasma / (Km + C_plasma)` is dimensionless,
- `Vmax * (dimensionless)` → `Mass/Time`,
- consistent with `dA_central/dt`.

#### 5.4.2. Time‑Dependent Clearance

Autoinduction or time‑varying clearance can be modeled by a dynamic `CL(t)`.

Simple exponential change:

```
CL(t) = CL_0 * (1 + α * (1 - exp(-k_ind * t)))
```

with:
- `CL_0 : Clearance`,
- `α : Fraction`,
- `k_ind : RateConst`.

**Pseudo‑MedLang**

```medlang
model OneCompIV_TimeVaryingCL {
    state A_central : DoseMass
    param CL0       : Clearance
    param alpha     : f64          // dimensionless
    param k_ind     : RateConst    // 1/h
    param V         : Volume

    fn CL_t(t : Time) : Clearance {
        return CL0 * (1.0 + alpha * (1.0 - exp(-k_ind * t)));
    }

    dA_central/dt = -(CL_t(t) / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}
```

This pattern generalizes to any covariate‑driven or state‑driven time‑varying `CL`.

---

### 5.5. Direct‑Effect PD Models

Direct‑effect models operate on a **driver** (usually drug concentration) and define an effect E(t) with no additional dynamics, or minimal dynamics.

#### 5.5.1. Linear and Log‑Linear Models

Linear:

```
E(t) = E_0 + S * C(t)
```

Log‑linear:

```
E(t) = E_0 + S * log(C(t) + ε)
```

with small ε to avoid log(0).

**Pseudo‑MedLang**

```medlang
model DirectEffectLinear {
    // PK driver provided externally or from another model
    input C_driver : ConcMass         // e.g. plasma concentration

    param E0 : EffectUnit             // baseline effect
    param S  : EffectUnit_per_Conc    // slope

    obs Effect : EffectUnit = E0 + S * C_driver
}
```

Unit example:
- `EffectUnit`: e.g. `mmHg`, `cells_per_uL`, or dimensionless,
- `EffectUnit_per_Conc` = EffectUnit / ConcMass.

#### 5.5.2. Emax and Sigmoid Emax

Emax:

```
E(t) = E_0 + (Emax * C(t)) / (EC50 + C(t))
```

Sigmoid Emax:

```
E(t) = E_0 + (Emax * C(t)^γ) / (EC50^γ + C(t)^γ)
```

**Pseudo‑MedLang**

```medlang
model DirectEffectEmax {
    input C_driver : ConcMass

    param E0    : EffectUnit
    param Emax  : EffectUnit
    param EC50  : ConcMass
    param gamma : f64            // dimensionless

    fn E(C : ConcMass) : EffectUnit {
        let num = Emax * pow(C, gamma)
        let den = pow(EC50, gamma) + pow(C, gamma)
        return E0 + num / den
    }

    obs Effect : EffectUnit = E(C_driver)
}
```

Typing:
- `pow(C, γ)` is only allowed after ensuring `C` is normalized to a dimensionless ratio (e.g. `C/EC50`), or via a library that handles this explicitly; the spec assumes the implementation enforces Section 4 rules on exponentiation of dimensionful quantities.

---

### 5.6. Indirect Response / Turnover Models

Indirect response models represent turnover of a biomarker R(t) under stimulation/inhibition by drug.

Baseline turnover:

```
dR/dt = K_in - K_out * R
```

Drug‑induced inhibition (e.g. of production):

```
dR/dt = K_in * (1 - I(C)) - K_out * R
```

with an inhibitory Emax:

```
I(C) = (Imax * C) / (IC50 + C)
```

**Pseudo‑MedLang**

```medlang
model IndirectResponseInhibition {
    input C_driver : ConcMass       // drug concentration

    // Biomarker state
    state R : BiomarkerUnit

    // Turnover parameters
    param Kin  : Rate_Biomarker     // BiomarkerUnit/h
    param Kout : RateConst          // 1/h

    // Inhibition parameters
    param Imax : f64                // dimensionless, in [0,1]
    param IC50 : ConcMass

    fn I(C : ConcMass) : f64 {
        return (Imax * C) / (IC50 + C)
    }

    dR/dt = Kin * (1.0 - I(C_driver)) - Kout * R

    obs Biomarker : BiomarkerUnit = R
}
```

This pattern generalizes to:
- stimulation of production (replace (1 - I(C)) with (1 + S(C))),
- inhibition/stimulation of loss (`Kout` modulated instead of `Kin`).

---

### 5.7. Effect Compartment Models

Effect compartment models introduce a **biophase delay** between plasma concentration and effect site concentration.

Define:
- Plasma concentration C_p(t),
- Effect site concentration C_e(t).

Dynamics:

```
dC_e/dt = Ke0 * (C_p(t) - C_e(t))
```

with PD driven by C_e(t) via any direct‑effect model (e.g. Emax).

**Pseudo‑MedLang**

```medlang
model EffectCompartment {
    input C_plasma : ConcMass

    state C_effect : ConcMass
    param Ke0      : RateConst     // 1/h

    dC_effect/dt = Ke0 * (C_plasma - C_effect)

    // PD layer (example: Emax)
    param E0    : EffectUnit
    param Emax  : EffectUnit
    param EC50  : ConcMass
    param gamma : f64

    fn E(Ce : ConcMass) : EffectUnit {
        let num = Emax * pow(Ce, gamma)
        let den = pow(EC50, gamma) + pow(Ce, gamma)
        return E0 + num / den
    }

    obs Effect : EffectUnit = E(C_effect)
}
```

This model is typically **composed** with a PK model providing `C_plasma`.

---

### 5.8. QSP Skeleton Pattern

Quantitative Systems Pharmacology often requires a **network of interacting species** (cells, cytokines, receptors) coupled to PK.

Abstract form:

```
dX/dt = F(X(t), C(t), θ_QSP, t)
```

where:
- X(t) – vector of biological states,
- C(t) – drug exposure (concentration/amount) from a PK model,
- θ_QSP – QSP parameters.

**Pseudo‑MedLang skeleton**

```medlang
model QSP_Skeleton {
    // PK driver (e.g. plasma concentration)
    input C_plasma : ConcMass

    // Biological states (examples)
    state Tumour      : Quantity<mm3, f64>
    state Effector    : CellCount           // cells
    state Cytokine    : ConcMass           // mg/L or pg/mL

    // Parameters (illustrative)
    param k_grow   : RateConst
    param k_kill   : RateConst
    param k_prolif : RateConst
    param k_death  : RateConst
    param k_sec    : Rate_Cytokine
    param k_clear  : RateConst

    // Structural dynamics (illustrative forms)
    dTumour/dt =
        k_grow * Tumour * (1.0 - Tumour / Tumour_max)
        - k_kill * Effector * Tumour * f_exposure(C_plasma)

    dEffector/dt =
        k_prolif * g(Cytokine) * Effector
        - k_death * Effector

    dCytokine/dt =
        k_sec * Effector
        - k_clear * Cytokine

    // Observables
    obs TumourVolume  : Quantity<mm3, f64> = Tumour
    obs EffCellCount  : CellCount          = Effector
    obs CytokineLevel : ConcMass           = Cytokine
}
```

Functions `f_exposure` and `g` can be:
- simple saturating functions of `C_plasma` and `Cytokine`,
- or learned components (see Section 8, Hybrid Mechanistic–ML).

This skeleton demonstrates:
- how PK exposure enters QSP dynamics via `input`,
- how QSP states map to observables for comparison with data.

---

### 5.9. Summary

The structural patterns defined in this section serve as:

- **reference implementations** for Track D libraries and examples,
- **testbeds** for the type system and solver integration,
- **anchors** for interoperability with legacy tools (by showing direct analogues to standard NONMEM/Monolix/Stan models).

Subsequent sections (6–10) will build on these patterns to define:
- formal NLME semantics,
- inference modes and backend contracts,
- hybrid ML extensions,
- and complete worked examples ready for implementation and validation.

---

## 6. Population and NLME Semantics

This section formalizes **population** and **nonlinear mixed–effects (NLME)** semantics in Track D and specifies how they are expressed using MedLang core constructs:

- `Model` (structural dynamics),
- `ProbKernel` (random effects, priors),
- `Measure` (observation models),
- `Timeline` (dosing/observation schedules),
- `Patient` / `Cohort` (clinical indexing and covariates).

The objective is to make **population PK/PD/QSP models** first‑class citizens in MedLang, with a clear generative interpretation that can be mapped to:

- classical NLME engines (NONMEM, Monolix),
- probabilistic programming tools (Stan, PyMC, NumPyro).

### 6.1 Informal Overview

A population model in Track D separates three levels:

1. **Structural (individual) level**  
   For a given parameter vector θᵢ and individual event history (doses, covariates, etc.), the `Model` defines a deterministic trajectory Xᵢ(t) and associated predictions ŷᵢⱼ.

2. **Population (between‑subject variability) level**  
   Individual parameters θᵢ are treated as random draws from a population distribution, typically parameterized by fixed effects φ and random effects distribution parameters (e.g. variances, covariances).

3. **Observation (residual error) level**  
   Observed data yᵢⱼ are generated from predictions ŷᵢⱼ via a residual error model (e.g. additive, proportional, combined), described by a `Measure`.

NLME = **Nonlinear mixed–effects**:

- "Nonlinear" because the structural model is usually nonlinear in parameters and states (ODEs).
- "Mixed‑effects" because it combines fixed effects (population typical values, covariate effects) and random effects (inter‑individual / inter‑occasion deviations).

Track D's semantics make this hierarchy explicit and machine‑checkable.

---

### 6.2 Formal Generative Model

Consider a **cohort** of N individuals indexed by i ∈ {1,...,N}. For individual i:

- Let zᵢ be a vector of covariates (e.g. weight, age, sex, genotype).
- Let 𝒯ᵢ be the `Timeline` of dosing and observation events.
- Let θᵢ be the individual parameter vector for the structural `Model`.
- Let yᵢ = {yᵢⱼ}ⱼ₌₁ⁿⁱ be observed data.

A generic Track D population model has the following generative process:

1. **Population parameters**  
   Population‑level parameters φ and error‑model parameters ψ are either:
   - treated as **unknown but fixed** (frequentist NLME),
   - or assigned **priors** and treated as random (Bayesian).

2. **Individual parameters (random effects + covariates)**  
   For each individual i,
   ```
   θᵢ ~ p(θ | zᵢ, φ)
   ```
   typically via:
   ```
   θᵢ = h(zᵢ, φ, ηᵢ),    ηᵢ ~ p(η | φ)
   ```
   where ηᵢ are random effects.

3. **Structural dynamics**  
   Given θᵢ and `Timeline` 𝒯ᵢ, the structural `Model` defines:
   ```
   Xᵢ(t) = solve_IVP(f(·; θᵢ), X₀,ᵢ, uᵢ(t), t)
   ```
   where uᵢ(t) encodes exogenous inputs (dosing) derived from 𝒯ᵢ.

4. **Observation model**  
   For each observation time tᵢⱼ and observation channel k:
   ```
   ŷᵢⱼ = hₖ(Xᵢ(tᵢⱼ), θᵢ)
   yᵢⱼ ~ p(y | ŷᵢⱼ, ψ)
   ```
   where p is defined by a `Measure` (e.g. additive or proportional error).

The **joint distribution** over all data and latent variables can be written as:

```
p({yᵢ}, {θᵢ}, φ, ψ) = p(φ, ψ) ∏ᵢ₌₁ᴺ [p(θᵢ | zᵢ, φ) ∏ⱼ₌₁ⁿⁱ p(yᵢⱼ | ŷᵢⱼ(θᵢ), ψ)]
```

with ŷᵢⱼ(θᵢ) implicitly defined via ODE solves.

This factorization is the **semantic contract** that any inference backend must respect.

---

### 6.3 Covariate Models

Covariate models specify how individual parameters θᵢ depend on individual covariates zᵢ and random effects ηᵢ.

#### 6.3.1 Log‑normal parameterization with covariates

A common parameterization for a parameter θᵢ⁽ᵏ⁾ (e.g. clearance) is:

```
θᵢ⁽ᵏ⁾ = θ_pop⁽ᵏ⁾ · g⁽ᵏ⁾(zᵢ; β⁽ᵏ⁾) · exp(ηᵢ⁽ᵏ⁾),    ηᵢ⁽ᵏ⁾ ~ N(0, ω²ₖ)
```

where:

- θ_pop⁽ᵏ⁾ is a typical value at reference covariates,
- g⁽ᵏ⁾(zᵢ; β⁽ᵏ⁾) is a dimensionless covariate function, e.g.:
  - allometric scaling: g(zᵢ; β) = (WTᵢ / 70)^β,
  - categorical covariates: binary or multi‑level factors.
- ηᵢ⁽ᵏ⁾ is the random effect,
- ω²ₖ is the variance of that effect.

In MedLang, this is expressed by:

- deterministic covariate mappings inside `Model` code,
- `ProbKernel` describing the distribution of ηᵢ.

**Pseudo‑MedLang sketch:**

```medlang
// Population-level parameters
param CL_pop  : Quantity<L/h, f64>   // typical CL for 70 kg
param beta_CL : f64                  // allometric exponent
param omega_CL: f64                  // std dev of log-CL random effect (dimensionless)

// Individual covariate and random effect
input WT      : Quantity<kg, f64>    // body weight
rand  eta_CL  : f64                  // from ProbKernel(0, omega_CL^2)

// Covariate function (dimensionless)
fn g_CL(WT : Quantity<kg, f64>) : f64 {
    return pow(WT / 70.0_kg, beta_CL)
}

// Individual CL_i
let CL_i : Quantity<L/h, f64> =
    CL_pop * g_CL(WT) * exp(eta_CL)
```

Unit rules:

- `g_CL` and `exp(eta_CL)` are dimensionless,
- base unit `CL_pop` carries `Volume/Time`,
- thus `CL_i` has the correct unit.

#### 6.3.2 Other covariate structures

Track D does not limit covariate models to multiplicative log‑normal forms. Other allowed patterns include:

- **Additive covariate models:**
  ```
  θᵢ⁽ᵏ⁾ = θ_pop⁽ᵏ⁾ + β⁽ᵏ⁾ᵀ zᵢ + ηᵢ⁽ᵏ⁾
  ```

- **Categorical factors** (e.g. sex, genotype):
  implemented via indicator functions or one‑hot encodings in `g`.

- **Multi‑parameter covariate strength**:
  correlated covariate effects across parameters are permitted by sharing covariates and random effects or by using multivariate `ProbKernel`s (see 6.4).

MedLang imposes only **type/units correctness** and measurability requirements; statistical form is left to the modeller.

---

### 6.4 Random Effects Types and `ProbKernel` Structure

Population variability is decomposed into:

1. **Inter‑individual variability (IIV)**  
   Differences between subjects:
   ```
   ηᵢ ~ N(0, Ω_IIV)
   ```

2. **Inter‑occasion variability (IOV)** (optional for v0.1)  
   Differences between occasions (e.g. study periods) within the same subject:
   ```
   κᵢ,ₒ ~ N(0, Ω_IOV)
   ```
   contributing to θᵢ,ₒ.

3. **Residual unexplained variability (RUV)**  
   Captured at the observation level via `Measure`.

Track D expresses IIV and IOV via **random‑effect variables** associated with `ProbKernel`s.

#### 6.4.1 Inter‑individual variability (IIV)

For a vector of random effects ηᵢ ∈ ℝᵍ,

```
ηᵢ ~ N(0, Ω)
```

with covariance matrix Ω ∈ ℝᵍˣᵍ.

In MedLang:

```medlang
// Population hyperparameters
param Omega : CovMatrix<q>   // covariance of random effects (dimensionless)

// Random effects for individual i
rand eta : Vector<q, f64> ~ MVNormal(mean = 0, cov = Omega)
```

Then, individual parameters are determined via a function h as in 6.3.

`ProbKernel` encodes the mapping from hyperparameters to the distribution p(η | hyper). At compile time, backends must be able to:

- evaluate log‑densities `log p(eta | Omega)`,
- sample from the kernel if needed (simulation, Bayesian inference).

#### 6.4.2 Inter‑occasion variability (IOV) (optional)

If occasions (e.g. study periods, treatment cycles) are modelled, additional random effects κᵢ,ₒ can be included:

```
θᵢ,ₒ = h(zᵢ, φ, ηᵢ, κᵢ,ₒ)
```

with their own `ProbKernel`. v0.1 of the spec **acknowledges** this pattern but does not mandate a specific syntax; implementations MAY support it as an extension.

#### 6.4.3 Random‑effect units

As per Section 4:

- In **log‑normal parameterizations**, random effects are dimensionless; units live entirely in the base parameter.
- In **additive parameterizations**, random effects must have the same unit as the parameter.

`ProbKernel` typing enforces these choices.

---

### 6.5 Observation-Level Noise (RUV) and `Measure`

Residual unexplained variability (RUV) is modeled via `Measure` objects.

Canonical forms:

1. **Additive error:**
   ```
   yᵢⱼ = ŷᵢⱼ + εᵢⱼ,    εᵢⱼ ~ N(0, σ²_add)
   ```

2. **Proportional error:**
   ```
   yᵢⱼ = ŷᵢⱼ(1 + εᵢⱼ),    εᵢⱼ ~ N(0, σ²_prop)
   ```

3. **Combined error:**
   ```
   yᵢⱼ = ŷᵢⱼ(1 + ε₁,ᵢⱼ) + ε₂,ᵢⱼ
   ```

4. **Other structures:**  
   heteroscedasticity, log‑normal error on positive quantities, censored data, etc.

**MedLang sketch (proportional error):**

```medlang
// Hyperparameters for RUV
param sigma_prop : f64   // dimensionless SD

measure ConcObs {
    pred : Quantity<mg/L, f64>   // model prediction ŷᵢⱼ
    obs  : Quantity<mg/L, f64>   // observed data yᵢⱼ

    // Noise model: y = pred * (1 + eps)
    rand eps : f64 ~ Normal(mean = 0.0, sd = sigma_prop)

    log_likelihood = Normal_logpdf(
        x   = obs / pred - 1.0,
        mu  = 0.0,
        sd  = sigma_prop
    )
}
```

The key requirement is that every `Measure` can expose a **log‑likelihood contribution** for each data point, with unit correctness guaranteed by construction.

---

### 6.6 Mapping to MedLang Core Constructs

A complete NLME model in Track D is represented as a composition of:

- **Structural `Model`**
  - States and parameters,
  - ODE/PDE definitions,
  - observable definitions.

- **Random‑effects `ProbKernel`s**
  - For inter‑individual (and optional inter‑occasion) variability,
  - Potentially multivariate, with covariance structures.

- **Observation `Measure`s**
  - Error models linking predictions to data.

- **`Timeline`s**
  - Dosing and observation schedules per individual.

- **`Cohort<Patient, L, d>`**
  - Collection of patients, each with:
    - covariates,
    - timeline,
    - measurement records.

Conceptually, one can introduce a **population model combinator**:

```medlang
population PopModel {
    model     : Model<State, Param>
    re_kernel : ProbKernel<Unit, RandomEffects>   // IIV (and optionally IOV)
    obs_model : Measure<Observable>               // RUV

    cohort    : Cohort<Patient, ..., ...>         // individuals and data
}
```

The spec does not mandate a concrete surface syntax for this combinator yet, but:

- an implementation MUST offer an abstraction that bundles these pieces,
- inference engines MUST be able to consume this bundle as a unified model.

---

### 6.7 Relation to NONMEM, Monolix, and Stan/Torsten

The Track D semantics are designed so that classical NLME abstractions are **special cases** of the MedLang formulation.

#### 6.7.1 NONMEM mapping

- `Model` (structural ODE + covariates) ↔ `$DES` + `$PK` blocks.
- `ProbKernel` for IIV ↔ `OMEGA` definition and associated `ETA`s.
- `Measure` (RUV) ↔ `$ERROR` with `SIGMA` matrices and `EPS` variables.
- `Timeline` ↔ `$INPUT` / event records (AMT, TIME, RATE, etc.).
- `Cohort` ↔ data file (rows with ID, TIME, DV, etc.).

A simple one‑compartment oral NLME model in NONMEM:

- `$PK` defines `TVCL`, `TVV`, covariate effects, and parameterization from `ETA`.
- `$DES` encodes ODEs for `A_gut`, `A_c`.
- `$ERROR` encodes proportional residual error.

In MedLang, these are unified under the constructs defined above; exporting to NONMEM is largely a matter of syntax translation and choice of approximations (e.g., log‑normal vs normal random effects, censoring conventions).

#### 6.7.2 Monolix mapping

Monolix `.mlxtran` models:

- `DEFI` / `EQUATION` blocks ↔ `Model` ODEs and observation maps,
- `DEFINITION:` sections for parameters / random effects ↔ `ProbKernel` definitions,
- `OBSERVATION:` ↔ `Measure`.

The high‑level structure is analogous to NONMEM; MedLang's explicit generative semantics match the stochastic language underlying Monolix.

#### 6.7.3 Stan / Torsten mapping

In a Stan/Torsten formulation:

- `Model` ODEs ↔ Stan `functions` block ODE system or Torsten ODE/PBPK solvers,
- random effects ↔ hierarchical parameters in Stan `parameters` / `model` blocks,
- `Measure` ↔ Stan likelihood statements (`y ~ normal(...)` etc.),
- `Timeline` ↔ dosing/observation records consumed by Torsten solvers,
- priors ↔ `ProbKernel` with logpdf calls.

A MedLang population model can conceptually be **compiled into** a Stan/Torsten model by:

1. Generating Stan data definitions from `Cohort` and `Timeline`.
2. Generating parameter blocks from `Param` and random effects definitions.
3. Generating ODE solver calls for `Model`.
4. Generating likelihood statements from `Measure` (and priors from `ProbKernel`).

This mapping is not specified in detail here but guides the design of the MedLang IR and backend contracts (Section 7).

---

### 6.8 Frequentist vs Bayesian Modes

Track D defines the **model semantics** independently of the estimation paradigm.

- In **frequentist NLME mode**:
  - φ, ψ, and possibly Ω are estimated as fixed but unknown parameters.
  - Random effects ηᵢ are latent variables integrated out (or approximated) in the likelihood.
  - `ProbKernel` is used both to specify the distribution of ηᵢ and to compute the likelihood contributions p(ηᵢ | φ).

- In **Bayesian mode**:
  - Priors are assigned to φ, ψ, Ω via additional `ProbKernel`s.
  - Inference produces posterior distributions over all unknowns, enabling full uncertainty quantification.
  - Posterior predictive checks are naturally expressible by sampling from the posterior and simulating via `Model` + `Timeline` + `Measure`.

Implementations MAY support one or both modes; the spec requires that:

- all components needed for both (likelihoods, priors, structural model) are explicitly represented,
- switching between modes does not require changing the high‑level model definition, only the **inference configuration**.

---

### 6.9 Summary

Section 6 establishes:

- a **hierarchical generative semantics** for population PK/PD/QSP models,
- a consistent use of `Model`, `ProbKernel`, `Measure`, `Timeline`, and `Cohort` to encode NLME structures,
- compatibility with existing tools (NONMEM, Monolix, Stan/Torsten),
- and flexibility for both frequentist and Bayesian inference.

This foundation enables:

- rigorous **population–level simulation** and virtual trials,
- integration of quantum‑derived parameters (Track C) into population models,
- and the introduction of hybrid mechanistic–ML models in Section 8 without breaking probabilistic coherence.

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
