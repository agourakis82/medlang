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

- **7. Inference Modes and Backend Contracts** (See below)
  - Definitions of:
    - simulation–only mode (no parameter learning),
    - frequentist NLME mode (likelihood maximization),
    - Bayesian mode (posterior inference).
  - Required interfaces for inference engines (log–likelihood, log–prior, gradients).

- **8. Hybrid Mechanistic–ML and PINN Integration** ✓ (See below)
  - Syntax and semantics for embedding learned components (`Model<X,Y>`) into Track D models.
  - Constraints for safe integration (type and unit consistency, monotonicity where needed).
  - Hooks for physics–informed training (PINN‑style losses).

- **9. Worked Examples**
  - **Example 1:** One‑compartment oral PK with log–normal IIV on CL, weight covariate, proportional error. ✓ (See below)
  - **Example 2:** Simple QSP model (drug + biomarker + tumor) with random effects.
  - **Example 3 (optional):** PBPK model with quantum‑derived parameters via `Compile_QM→PK`.

- **10. Implementation Notes and IR Mapping** ✓ (See below)
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

## 7. Inference Modes and Backend Contracts

Track D deliberately separates:

- **Model specification** — what the PK/PD/QSP + population model *means*,
- **Inference** — how parameters and random effects are estimated or sampled,
- **Simulation** — how trajectories and virtual trials are generated.

This section specifies the **inference modes** supported conceptually and the **minimal contracts** that MedLang Track D must expose to inference backends (frequentist NLME engines, probabilistic programming systems, or custom solvers).

### 7.1 Design Goals

Inference design in Track D is guided by the following principles:

1. **Separation of concerns**  
   The same MedLang model should be usable for:
   - forward simulation only,
   - frequentist NLME estimation,
   - Bayesian inference and posterior predictive simulation,
   
   without changing the model definition itself; only inference configuration changes.

2. **Backend‑agnostic semantics**  
   MedLang defines the **probabilistic semantics** (joint density, likelihood, priors) in a backend‑neutral way. Different engines (FOCE, SAEM, HMC, VI, etc.) can be plugged in as long as they implement a simple log‑density and differentiation interface.

3. **Differentiability and AD compatibility**  
   Where possible, structural and probabilistic components are differentiable with respect to parameters, enabling:
   - gradient‑based optimization for ML/MLE/NLME,
   - gradient‑based MCMC (e.g. HMC/NUTS),
   - sensitivity analysis.

4. **Batching and vectorization**  
   Population models must be evaluable in **batched** form across individuals and time points, allowing efficient CPU/GPU execution.

5. **Determinism and reproducibility**  
   Given:
   - a model definition,
   - a dataset,
   - an inference configuration (algorithm, seeds, tolerances),
   
   MedLang should make the resulting inference **reproducible** up to differences introduced by stochastic algorithms and floating‑point rounding.

---

### 7.2 Inference Modes

Track D supports three conceptual inference modes:

1. **Simulation‑only mode**  
   - No parameter estimation; all parameters and random effects are user‑specified or sampled from priors.
   - Used for:
     - scenario analysis,
     - virtual clinical trials,
     - design exploration (dose regimen, covariate distributions).

2. **Frequentist NLME mode** (maximum likelihood / penalized likelihood)  
   - Population parameters (φ, Ω, ψ) are treated as unknown fixed quantities.
   - Estimation seeks:
     ```
     (φ̂, Ω̂, ψ̂) = argmax_{φ,Ω,ψ} log L(φ, Ω, ψ; y)
     ```
   - Random effects ηᵢ are latent variables integrated out or approximated (FO, FOCE, Laplace, SAEM).

3. **Bayesian mode**  
   - Population parameters and random effects are endowed with priors p(φ, Ω, ψ).
   - Inference targets the posterior:
     ```
     p(φ, Ω, ψ, η | y) ∝ p(φ, Ω, ψ) ∏ᵢ p(ηᵢ | Ω) ∏ⱼ p(yᵢⱼ | θᵢ(φ, zᵢ, ηᵢ), ψ)
     ```
   - Usually implemented via MCMC (HMC/NUTS), SMC, or variational inference.

Each mode consumes **the same MedLang representation** of the model; only the **algorithmic backend** and **configuration** differ.

---

### 7.3 Core Backend Contract: Log‑Densities and Simulation

Any backend that performs inference on a Track D population model must be able to:

1. **Evaluate log‑densities** for:
   - **Random effects**:
     ```
     log p(ηᵢ | Ω)
     ```
     via the `ProbKernel` interface.
   - **Observation models**:
     ```
     log p(yᵢⱼ | ŷᵢⱼ, ψ)
     ```
     via the `Measure` interface.

2. **Simulate structural trajectories** for a given individual:
   ```
   Xᵢ(t) = solve(f, X₀,ᵢ, θᵢ, 𝒯ᵢ)
   ```
   using the Track D `Model` and `Timeline`. This may involve:
   - ODE/PDE solvers,
   - handling of dosing and observation events.

3. **Compose log‑likelihoods** across individuals and observations:
   ```
   log p(y | φ, Ω, ψ, η) = ∑ᵢ₌₁ᴺ ∑ⱼ₌₁ⁿⁱ log p(yᵢⱼ | ŷᵢⱼ(θᵢ), ψ)
   ```

4. **(Optional but recommended) Compute gradients** of relevant log‑densities w.r.t. parameters:
   - ∇_{φ,Ω,ψ} log L(φ, Ω, ψ; y),
   - ∇_{φ,Ω,ψ,η} log p(φ, Ω, ψ, η | y) in Bayesian mode.

MedLang itself does not define the **numerical method** for these operations; it defines a **uniform interface** at the IR level (see §10).

---

### 7.4 Simulation‑Only Mode

In pure simulation mode, no fitting is performed. Instead, the population model is used as a **generator** of trajectories and synthetic data.

#### 7.4.1 Individual simulation

Given:
- structural `Model`,
- parameter vector θ,
- `Timeline` 𝒯,

the backend implements:

```text
simulate_individual(Model, θ, Timeline) -> Trajectory
```

where `Trajectory` contains:
- time grid and state trajectories X(t),
- derived observables ŷ(t).

#### 7.4.2 Population simulation (virtual trial)

Given:
- `PopulationModel` (structural model + random–effects + residual error),
- `Cohort` specification with covariate distributions and dosing regimens,

the backend implements:

```text
simulate_population(PopulationModel, CohortSpec, n_subjects) -> SimData
```

Typical steps:

1. For each virtual subject i:
   - Sample covariates zᵢ from the specified distributions,
   - Sample random effects ηᵢ ~ p(η | Ω),
   - Construct θᵢ = g(φ, zᵢ, ηᵢ),
   - Simulate Xᵢ(t) and predictions ŷᵢⱼ,
   - Sample measurement noise and generate synthetic yᵢⱼ.

2. Aggregate into a synthetic dataset `SimData` with a structure analogous to real data.

This mode is crucial for:
- design evaluation (power, exposure–response),
- virtual populations and virtual clinical trials,
- stress‑testing models.

---

### 7.5 Frequentist NLME Mode: Likelihood Engines

In frequentist NLME mode, the backend is a **likelihood engine** operating on Track D models.

#### 7.5.1 Required operations

The backend must be able to:

1. **Build the integrated log‑likelihood**:
   ```
   log L(φ, Ω, ψ; y) = ∑ᵢ₌₁ᴺ log ∫ p(ηᵢ | Ω) ∏ⱼ p(yᵢⱼ | θᵢ(φ, zᵢ, ηᵢ), ψ) dηᵢ
   ```
   using approximations as appropriate (Laplace, FO, FOCE, SAEM, etc.).

2. Optionally compute **gradients and Hessians**:
   - For gradient‑based optimizers: ∇_{φ,Ω,ψ} log L,
   - For approximate standard errors: Hessian or observed Fisher information.

3. Accept **constraints** and **reparameterizations**:
   - positivity constraints (e.g. for variances, volumes, clearances),
   - correlation structures (Cholesky factorization of covariance matrices),
   - transformation between internal and user parameterizations.

#### 7.5.2 Contract at the IR level

At the IR level (NIR), Track D guarantees:

- Random effects distributions are represented as composable `logpdf` ops,
- Structural models are represented as calls to ODE/PDE solvers with differentiable dependence on parameters,
- `Measure`s yield log‑likelihood contributions with correct units.

A frequentist backend may:
- treat random effects as latent variables and approximate the integrals, or
- approximate the full joint mode (e.g. Laplace approx to the posterior), or
- employ SAEM‑like stochastic approximation.

The spec does **not** impose a particular algorithm, only that:
- the backend uses the Track D log‑density structure,
- results correspond to maximizing (an approximation of) the integrated likelihood.

---

### 7.6 Bayesian Mode: Probabilistic Programming Backends

In Bayesian mode, the same Track D model is interpreted as specifying a **joint posterior** over:
- population parameters (φ, Ω, ψ),
- random effects η.

#### 7.6.1 Priors via `ProbKernel`

Priors are attached to:
- fixed effects (e.g. log‑normal or normal priors on `TV_CL`, `TV_V`),
- covariance parameters (e.g. priors on Cholesky factors of Ω),
- residual error parameters (e.g. half‑normal priors on σ parameters).

Example (conceptual):

```medlang
prior TV_CL ~ LogNormal(mean_log = log(CL_ref), sd_log = 0.5)
prior TV_V  ~ LogNormal(mean_log = log(V_ref),  sd_log = 0.5)
prior omega_CL ~ HalfNormal(sd = 1.0)
prior omega_V  ~ HalfNormal(sd = 1.0)
prior sigma_prop ~ HalfNormal(sd = 0.5)
```

At the IR level, these are simply further `ProbKernel`s with `logpdf` definitions.

#### 7.6.2 Backend contract for Bayesian inference

A Bayesian backend (e.g. Stan, PyMC, NumPyro) must:

1. Construct the **joint log‑density**:
   ```
   log p(φ, Ω, ψ, η, y) = log p(φ, Ω, ψ) + ∑ᵢ log p(ηᵢ | Ω) + ∑ᵢⱼ log p(yᵢⱼ | ŷᵢⱼ, ψ)
   ```

2. Provide **sampling** or **approximate inference**:
   - MCMC (e.g. HMC / NUTS),
   - SMC,
   - variational inference (e.g. ADVI, structured VI).

3. Expose **posterior samples** in a way that MedLang can:
   - perform posterior predictive simulation,
   - compute summary statistics (credible intervals, posterior means),
   - support diagnostic tools (e.g. convergence diagnostics, R‑hat, ESS) at a higher layer.

MedLang's IR must be translatable to a probabilistic programming representation (e.g. a Stan program), but the exact compilation path is delegated to the implementation.

---

### 7.7 Configuration of Inference Runs

A Track D **inference configuration** is a metadata object that specifies:

- **Mode:** `SimulationOnly`, `FrequentistNLME`, or `Bayesian`.
- **Backend:** e.g. `Backend.NONMEM_like`, `Backend.Stan`, `Backend.InHouse`.
- **Algorithm details:** learning rates, tolerances, max iterations, etc.
- **Random seeds:** for reproducibility.
- **Output requirements:** which posterior quantities or diagnostics to save.

Conceptually:

```text
InferenceConfig = {
    mode:      InferenceMode,
    backend:   BackendKind,
    algorithm: AlgorithmSettings,
    seed:      int,
    outputs:   OutputSpec
}
```

MedLang itself does not standardize all fields, but requires that:
- the mode is explicit,
- the backend is identified,
- enough configuration is present for reproducible runs.

---

### 7.8 Error Handling, Diagnostics, and Robustness Considerations

Inference backends should surface diagnostics that can be associated back to model elements:

- **Integration failures** (stiff ODE, divergence):
  - Provide time, individual, and parameter context,
  - Allow the user to adjust tolerances or reparameterize.

- **Non‑identifiability / flat likelihood**:
  - Indicate parameters with weak information,
  - Suggest regularization or priors.

- **Pathological random‑effects distributions**:
  - E.g. non‑positive definite covariance estimates,
  - E.g. estimated variances collapsing toward zero.

While the details of diagnostic reporting are implementation‑specific, Track D encourages:
- linking error messages to specific `Model` states, parameters, `ProbKernel`s, and `Measure`s,
- using the MedLang source structure (file names, line numbers) where possible.

---

### 7.9 Summary

Section 7 defines:

- the **inference modes** (simulation‑only, frequentist NLME, Bayesian),
- a **minimal backend contract** in terms of log‑densities, simulation, and differentiation,
- how MedLang Track D models are consumed by:
  - likelihood engines in the frequentist setting,
  - probabilistic programming systems in the Bayesian setting,
- and how inference configurations and diagnostics integrate with the language.

This abstraction allows:
- swapping backends without rewriting models,
- leveraging existing tools (NONMEM‑like engines, Stan, PyMC) via compilation from the MedLang IR,
- and future extensions (e.g. PINN‑based inference, differentiable programming) without changing model semantics.

Subsequent sections (§8–§10) will address hybrid mechanistic–ML integration, worked examples, and explicit IR mappings that make these contracts concrete in an implementation.

---

## 8. Hybrid Mechanistic–ML and PINN Integration

Track D explicitly supports **hybrid models** that combine:

- mechanistic structure (PK/PD/QSP ODE/PDE systems), and
- data–driven components (classical ML, neural networks, Gaussian Processes, PINNs),

while preserving:

- **type and unit safety** (Section 4),
- **probabilistic semantics** (Section 6),
- and compatibility with all inference modes (Section 7).

This section specifies how ML components may be integrated into Track D models, which patterns are allowed, and which constraints must be enforced to maintain interpretability and identifiability.

### 8.1 Objectives and Use–Cases

Hybrid mechanistic–ML is motivated by several recurring needs:

1. **Parameter prediction from high–dimensional features**
   - Predict partition coefficients (Kp), tissue permeabilities, clearance (CL), or bioavailability (F) from:
     - molecular descriptors,
     - omics profiles,
     - imaging or biomarker panels.
   - Example: a GNN or transformer predicting tissue Kp from chemical structure.

2. **Unknown or partially known dynamics**
   - Mechanistic structure is known only partially; certain reaction terms or feedback functions are unknown or too complex.
   - Example: immune–tumour interactions where the form of the killing function is uncertain.

3. **Emulation of expensive submodels**
   - Replace an expensive submodel (e.g. detailed cellular model or spatial PDE) by a learned surrogate (NN or GP) trained on its outputs.

4. **Physics–informed learning (PINNs / universal DEs)**
   - Use differential equation structure as a **soft constraint** during training, combining:
     - data misfit,
     - ODE/PDE residual penalties.

Track D must support these patterns without compromising the clarity of the **mechanistic backbone** or the probabilistic semantics.

---

### 8.2 ML Submodels as Deterministic Components

At the language level, an ML component is treated as a **deterministic function**:

```
f_ML(x; w): 𝒳 → 𝒴
```

where:
- x is an input vector (covariates, states, time, features),
- w is a vector (or structured object) of parameters/weights.

In MedLang, we treat f_ML as a specialized `Model` or function with:

- **well–typed inputs and outputs** (with units),
- parameters w that are part of the overall parameter vector and can be:
  - fixed (pre–trained),
  - fitted (frequentist),
  - assigned priors (Bayesian).

Conceptual pseudo–signature:

```medlang
model MLSubmodel {
    param w : MLParamVector      // NN weights, GP hyperparameters, etc.

    fn forward(x : InputType) : OutputType {
        // ML computation graph (opaque at this level)
    }
}
```

The core `Model` may then call `MLSubmodel.forward` inside either:
- **parameterization code**, or
- **dynamics right–hand side**.

---

### 8.3 Parameter–Level Hybrids (Mechanistic Structure, ML Parameters)

The simplest and most interpretable hybrid pattern is:

> **Mechanistic structure is fixed; some parameters are predicted by ML from features.**

Example: CL, V, Kp predicted from covariates or molecular features.

#### 8.3.1 Example: ML–predicted clearance

Let xᵢ be a feature vector (covariates, molecular descriptors) for individual or compound i. Let an ML model predict a **dimensionless multiplier** g_ML(xᵢ; w), applied to a baseline clearance:

```
CLᵢ = CL_base · g_ML(xᵢ; w)
```

where:
- CL_base has unit `Volume/Time`,
- g_ML is constrained to be positive dimensionless (e.g. via `softplus`).

Pseudo–MedLang:

```medlang
model CL_MLSubmodel {
    param w : MLParamVector   // NN weights

    // xᵢ includes covariates and/or molecular descriptors
    fn g_ML(x : FeatureVec) : f64 {
        // Implementation detail: ML graph with final activation > 0
        return softplus(nn_forward(x, w))   // dimensionless, > 0
    }
}

model PK_With_ML_CL {
    // Baseline parameter
    param CL_base : Clearance    // L/h
    param V       : Volume       // L

    // Features for current entity (individual/compound)
    input x_feat  : FeatureVec

    // ML submodel for CL multiplier
    param w_CL    : MLParamVector

    fn CL_i(x : FeatureVec) : Clearance {
        let mult : f64 = CL_MLSubmodel{w = w_CL}.g_ML(x)
        return CL_base * mult
    }

    state A_central : DoseMass
    dA_central/dt = -(CL_i(x_feat) / V) * A_central

    obs C_plasma : ConcMass = A_central / V
}
```

Notes:
- The ODE structure remains classical.
- Unit correctness is preserved:
  - `CL_base` has `Volume/Time`,
  - `mult` is dimensionless,
  - `CL_i` has `Volume/Time`.

This pattern is recommended for:
- PK parameters (CL, V, Kp),
- PD sensitivities (Emax, EC50) when informed by biomarkers.

---

### 8.4 Dynamics–Level Hybrids (Neural ODEs / Universal Differential Equations)

A more powerful but riskier pattern is:

> **Embed ML terms directly into the dynamical system's right–hand side.**

Given:

```
dX/dt = f_mech(X, t, θ) + f_ML(X, t, u; w)
```

where:
- f_mech is mechanistic,
- f_ML is a data–driven term parameterized by w,
- u(t) exogenous inputs (doses, biomarkers).

#### 8.4.1 Example: unknown feedback in QSP

Consider the QSP skeleton from §5.8 with an unknown feedback function g on cytokine influence:

```
d(Effector)/dt = k_prolif · g(Cytokine; w) · Effector - k_death · Effector
```

We can let g be an ML submodel constrained to positive, saturating behaviour.

Pseudo–MedLang:

```medlang
model G_Cytokine_ML {
    param w : MLParamVector

    fn g(Cyt : ConcMass) : f64 {
        // Convert to dimensionless via reference scale
        let Cyt_norm : f64 = (Cyt / Cyt_ref)
        // ML approximation with bounded output, e.g. sigmoid
        let raw = nn_forward(Cyt_norm, w)      // unconstrained
        return sigmoid(raw)                    // in (0,1)
    }
}

model QSP_With_ML_Feedback {
    // PK driver
    input C_plasma : ConcMass

    // Biological states
    state Tumour   : TumourVolumeUnit
    state Effector : CellCount
    state Cytokine : ConcMass

    param k_prolif : RateConst
    param k_death  : RateConst

    param w_g      : MLParamVector    // parameters for feedback function

    fn g_feedback(Cyt : ConcMass) : f64 {
        return G_Cytokine_ML{w = w_g}.g(Cyt)
    }

    dEffector/dt =
        k_prolif * g_feedback(Cytokine) * Effector
        - k_death * Effector

    // Other dynamics elided (Tumour, Cytokine, etc.)

    obs EffCellCount : CellCount = Effector
}
```

Constraints:
- `g_feedback` is dimensionless, bounded (e.g. [0,1]), ensuring **unit consistency** and some biological plausibility.
- We maintain interpretability: we still know the structure (proportional to Effector, modulated by Cytokine).

#### 8.4.2 Universal Differential Equations

If the entire unknown dynamics are approximated by a neural network:

```
dX/dt = f_NN(X, t; w)
```

this is essentially a **Neural ODE**. Track D allows this formally, but:

- v0.1 spec **recommends** using universal DE patterns only for:
  - isolated subsystems,
  - or exploratory modelling,
- and encourages the presence of mechanistic terms whenever possible.

Such fully data–driven dynamical components are expressed as:

```medlang
model NeuralDynamics {
    param w : MLParamVector

    state X : StateVector   // typed and unit–aware

    dX/dt = NN_rhs(X, t, w) // must pass unit checks
}
```

It is the responsibility of the implementation to ensure the ML graph respects dimensional analysis, via explicit rescaling or unit‑aware layers.

---

### 8.5 PINNs and Physics–Informed Learning

Physics‑informed neural networks (PINNs) augment training objectives with **equation residual penalties**. Conceptually, given:

- data points (tₖ, yₖ),
- collocation points (tᶜ),

we minimize:

```
ℒ(w, θ) = ∑ₖ |yₖ - ŷ(tₖ; w, θ)|² + λ ∑ᶜ |∂ₜX(tᶜ; w, θ) - f(X(tᶜ; w, θ), tᶜ, θ)|²
          ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯   ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
          data misfit                      physics residual
```

In Track D semantics, we interpret PINN training as:

- a **deterministic optimization** problem (simulation‑only mode + custom loss), or
- a **penalized likelihood** / pseudo‑Bayesian model where residual penalties act as additional "pseudodata".

We do **not** fix a unique interpretation; instead we specify:

1. A `PINNLoss` abstraction:

```medlang
struct PINNLoss {
    data_term      : LossTerm   // from Measure & observed data
    physics_term   : LossTerm   // from ODE/PDE residuals at collocation points
    lambda         : f64        // trade-off weight
}
```

2. An **inference configuration** that selects PINN training as an algorithmic option:

```medlang
InferenceConfig {
    mode      = SimulationOnly
    algorithm = Algorithm.PINN_Train    // or Hybrid
}
```

From a probabilistic standpoint, implementations may interpret physics terms as:

- log‑likelihood contributions of "virtual observations" of the ODE residual being near zero,
- or as log–prior terms on trajectories.

The spec only requires that:

- the mechanistic equations are used in the loss,
- unit and type consistency are preserved (residuals have well‑defined units).

---

### 8.6 Typing and Unit Constraints for ML Components

To preserve the guarantees of Section 4, any ML component must satisfy:

1. **Typed inputs/outputs**
   - Inputs and outputs of ML functions must be `Quantity<Unit, _>` or dimensionless `f64`, not untyped scalars.
   - Internally, ML layers may work in normalized, dimensionless space, but wrapping functions must perform explicit scaling.

2. **Dimensional normalization**
   - If a network consumes a dimensionful quantity Q, it must first be converted into a **dimensionless** representation, e.g.:
     ```
     x = Q / Q_ref
     ```
     where Q_ref has the same unit as Q.
   - Activation functions like `exp`, `log`, `sin` must operate on dimensionless inputs.

3. **Output unit guarantees**
   - Outputs must be typed:
     - dimensionless factors (e.g. multipliers),
     - or re–dimensionalized via known reference scales.

Recommended pattern:

```medlang
fn nn_predict_factor(
    x : Quantity<UnitX, f64>,
    w : MLParamVector
) : f64 {
    let x_norm : f64 = (x / X_ref)           // dimensionless
    let raw    : f64 = nn_forward(x_norm, w) // dimensionless
    return activation(raw)                   // e.g. sigmoid, softplus
}

fn nn_predict_quantity(
    x : Quantity<UnitX, f64>,
    w : MLParamVector
) : Quantity<UnitY, f64> {
    let factor : f64 = nn_predict_factor(x, w)  // dimensionless
    return factor * Y_ref                       // has UnitY
}
```

4. **Static checking at boundary**
   - The type system does not introspect inside the ML graph, but it checks:
     - that boundary functions respect units,
     - that the ML outputs are used in unit‑consistent expressions.

---

### 8.7 Interaction with Inference Modes

Hybrid models integrate naturally with the inference modes of Section 7.

#### 8.7.1 Simulation‑only

- ML parameters w are:
  - loaded from pre‑trained weights,
  - or sampled from configured distributions,
- and kept fixed during simulation.

Use cases:
- deploy a pre‑trained ML surrogate inside PBPK/QSP for high‑throughput simulation,
- test model behaviour under fixed learned components.

#### 8.7.2 Frequentist NLME

- ML parameters w may be:
  - held fixed (pre‑trained),
  - fine‑tuned jointly with (φ, ψ),
  - estimated in a two‑stage process (pretrain ML on local data, then calibrate mechanistic model).

Considerations:
- ML parameter spaces are often high‑dimensional; naive joint optimization can be ill‑posed and unstable.
- v0.1 spec **recommends**:
  - restricting ML components to moderate size for joint NLME fitting,
  - or pretraining ML parts separately and only fitting low‑dimensional scaling parameters in NLME mode.

#### 8.7.3 Bayesian

- ML parameters w are given priors (e.g. Gaussian on weights), and included in the posterior:
  ```
  p(φ, ψ, w, η | y)
  ```
- Full Bayesian treatment is computationally heavy; approximations (variational inference, low–rank representations, sparse GPs) are likely needed.

Track D allows this, but recommends:
- when possible, using **structured ML** (e.g. low–dimensional basis expansions, GP emulators with few inducing points),
- or restricting HMC to a subset of parameters while treating ML weights via MAP/VI.

Regardless of mode, MedLang semantics do not change: ML weights are just another component of the parameter vector with possible priors.

---

### 8.8 IR and Implementation Considerations

At the IR (NIR/MLIR) level, ML integration has the following implications:

1. **Separate ML subgraphs**
   - ML submodels are represented as separate compute graphs callable from the main numerical IR.
   - They operate on batched inputs (e.g. `[batch, features]`) for efficiency.

2. **Differentiability**
   - ML subgraphs must be differentiable w.r.t. their parameters w and inputs (where required), enabling:
     - gradient–based NLME optimization,
     - HMC/VI in Bayesian mode,
     - PINN–style joint training.

3. **Device placement**
   - ML subgraphs may run on GPU/TPU, while ODE solvers may run on CPU or GPU.
   - The IR must permit clear device annotations and data movement with minimal overhead.

4. **Caching and memoization**
   - When ML submodels compute time‑invariant quantities (e.g. CL from covariates), results can be cached per individual/compound to avoid redundant computation during iterative inference.

The detailed IR design is out of scope for this document; these points serve as **requirements** for any implementation claiming Track D compliance with ML integration.

---

### 8.9 Summary

Section 8 introduces a controlled integration of **machine learning** into MedLang Track D:

- At the **parameter level**, ML predicts parameters (CL, V, Kp, PD parameters) from high‑dimensional features while mechanistic ODE structure remains intact.
- At the **dynamics level**, ML terms can augment or partially replace right–hand sides, turning models into universal differential equations or Neural ODEs, with clear typing and unit constraints.
- **PINN–style training** is supported conceptually via composite losses that penalize both data misfit and physics residuals.
- All ML components are:
  - typed and unit–checked at the boundary,
  - compatible with simulation‑only, frequentist NLME, and Bayesian inference modes,
  - representable and differentiable at the IR level.

This enables MedLang to serve as a **unifying language** for mechanistic, statistical, and ML models in pharmacometrics and QSP, while preserving rigorous semantics and physical plausibility.

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

This section describes how Track D constructs are **represented and lowered** through MedLang's intermediate representation (IR) layers, ultimately reaching executable numeric/symbolic code and inference backends. It is intended for compiler and backend developers, not for end users.

### 10.1 Overview of IR Layers

MedLang adopts a **multi-level IR strategy** to balance:

- **high-level semantic clarity** (medical/pharmacometric domain),
- **middle-level generality** (numeric and probabilistic operations),
- **low-level optimization** (MLIR/LLVM for hardware).

The canonical flow is:

```
MedLang Surface Syntax (Track D models, timelines, measures)
          ↓
    Clinical IR (CIR)
          ↓
    Numeric / Neural IR (NIR)
          ↓
    MLIR dialects (linalg, affine, scf, arith, func, etc.)
          ↓
    LLVM IR / GPU backends (CUDA, ROCm, etc.)
```

Each layer has distinct responsibilities:

1. **CIR (Clinical IR)**  
   Represents Track D models as typed, hierarchical objects:
   - `cir.model` (structural dynamics, observables, parameters),
   - `cir.timeline` (dosing/observation events),
   - `cir.measure` (observation models, residual error),
   - `cir.prob_kernel` (random effects, priors),
   - `cir.cohort` (patient data, covariates).

   CIR is **still domain-aware** and **unit-typed**.

2. **NIR (Numeric/Neural IR)**  
   Lowers domain-specific constructs to **composable numeric primitives**:
   - ODE/PDE integration ops,
   - log-pdf evaluations for distributions,
   - batched simulation and likelihood aggregation,
   - ML subgraph calls (for hybrid models).

   NIR is **unit-erased** (units have been checked and removed) but **shape-typed** (tensors have static or dynamic shapes).

3. **MLIR / LLVM**  
   Standard compiler infrastructure:
   - loop optimization, vectorization, parallelization,
   - device placement (CPU, GPU, TPU),
   - final code generation.

This section focuses on **CIR → NIR → MLIR** mappings relevant to Track D.

---

### 10.2 CIR Representation of Track D Constructs

#### 10.2.1 `cir.model`

A Track D `Model` is represented in CIR as a **structured operation** with:

- **State schema**: list of state variables with types and units,
- **Parameter schema**: list of parameters with types and units,
- **Dynamics region**: a control-flow graph (CFG) encoding the right-hand side of ODEs/PDEs,
- **Observables region**: pure functions mapping `(State, Param) → Observable`,
- **Input bindings**: how `Timeline` events are mapped to input signals `u(t)`.

Example (conceptual MLIR-like syntax):

```mlir
cir.model @OneCptOralPK
  state_type   = !cir.struct<A_depot: !qty<mg, f64>, A_central: !qty<mg, f64>>
  param_type   = !cir.struct<CL: !qty<L/h, f64>, V: !qty<L, f64>, Ka: !qty</h, f64>>
  input_type   = !cir.struct<dose_rate: !qty<mg/h, f64>>
{
  // Right-hand side function: f(X, θ, u, t) → dX/dt
  cir.dynamics {
    ^entry(%X: !state_type, %theta: !param_type, %u: !input_type, %t: !qty<h, f64>):
      %dA_depot   = cir.mul %theta.Ka, %X.A_depot : (!qty<1/h>, !qty<mg>) -> !qty<mg/h>
      %dA_depot_neg = cir.neg %dA_depot : !qty<mg/h>
      
      %dA_central_1 = %dA_depot : !qty<mg/h>   // absorption term
      %dA_central_2 = cir.div %theta.CL, %theta.V : (!qty<L/h>, !qty<L>) -> !qty<1/h>
      %dA_central_3 = cir.mul %dA_central_2, %X.A_central : (!qty<1/h>, !qty<mg>) -> !qty<mg/h>
      %dA_central_neg = cir.neg %dA_central_3 : !qty<mg/h>
      %dA_central = cir.add %dA_central_1, %dA_central_neg, %u.dose_rate : !qty<mg/h>
      
      cir.yield %dA_depot_neg, %dA_central : !qty<mg/h>, !qty<mg/h>
  }
  
  // Observable: C_plasma = A_central / V
  cir.observable @C_plasma {
    ^entry(%X: !state_type, %theta: !param_type):
      %C = cir.div %X.A_central, %theta.V : (!qty<mg>, !qty<L>) -> !qty<mg/L>
      cir.yield %C : !qty<mg/L>
  }
}
```

At this level:

- **Units are explicit** and type-checked,
- Control flow is simple (no loops/conditionals in this example, but allowed for complex QSP),
- The representation is **backend-agnostic**.

#### 10.2.2 `cir.timeline`

A `Timeline` is represented as:

- A sequence of **event records**, each with:
  - time stamp,
  - event type (dose, observation, covariate change),
  - payload (e.g., dose amount, route, target compartment).

Example:

```mlir
cir.timeline @example_dosing : !cir.timeline<DoseEvent> {
  cir.event @dose1 { time = 0.0 : !qty<h, f64>, amount = 100.0 : !qty<mg, f64>, route = Oral, target = "A_depot" }
  cir.event @dose2 { time = 12.0 : !qty<h, f64>, amount = 50.0 : !qty<mg, f64>, route = Oral, target = "A_depot" }
}

cir.timeline @example_sampling : !cir.timeline<ObsEvent> {
  cir.event @obs1 { time = 0.5 : !qty<h, f64> }
  cir.event @obs2 { time = 1.0 : !qty<h, f64> }
  // ...
}
```

Timelines are **data**, not compute; they will be lowered to control parameters for ODE solvers (discontinuities, output grid).

#### 10.2.3 `cir.prob_kernel`

Random effects and priors are represented as **parameterized probability kernels**:

```mlir
cir.prob_kernel @PopulationVariability
  hyper_type = !cir.struct<omega_CL: f64, omega_V: f64, omega_Ka: f64, rho_CL_V: f64>
  input_type = !cir.struct<weight: !qty<kg, f64>>
  output_type = !cir.struct<eta_CL: f64, eta_V: f64, eta_Ka: f64>
{
  cir.logpdf {
    ^entry(%hyper: !hyper_type, %input: !input_type, %eta: !output_type):
      // Construct covariance matrix
      %Omega = cir.build_cov_matrix %hyper.omega_CL, %hyper.omega_V, %hyper.omega_Ka, %hyper.rho_CL_V
      // log pdf of MVN(0, Omega)
      %logpdf = cir.mvn_logpdf %eta, %Omega
      cir.yield %logpdf : f64
  }
  
  cir.sample {
    ^entry(%hyper: !hyper_type, %input: !input_type, %rng_state: !cir.rng):
      %Omega = cir.build_cov_matrix %hyper.omega_CL, %hyper.omega_V, %hyper.omega_Ka, %hyper.rho_CL_V
      %eta = cir.mvn_sample %Omega, %rng_state
      cir.yield %eta : !output_type
  }
}
```

This allows backends to:

- **evaluate log-densities** (for likelihood/posterior computation),
- **sample** (for forward simulation or MCMC).

#### 10.2.4 `cir.measure`

Observation models are similarly represented with `logpdf` and `sample` regions:

```mlir
cir.measure @ProportionalError
  pred_type = !qty<mg/L, f64>
  obs_type  = !qty<mg/L, f64>
  param_type = !cir.struct<sigma_prop: f64>
{
  cir.logpdf {
    ^entry(%pred: !pred_type, %obs: !obs_type, %param: !param_type):
      %sd = cir.mul %param.sigma_prop, %pred : (f64, !qty<mg/L>) -> !qty<mg/L>
      %logpdf = cir.normal_logpdf %obs, %pred, %sd
      cir.yield %logpdf : f64
  }
  
  cir.sample {
    ^entry(%pred: !pred_type, %param: !param_type, %rng_state: !cir.rng):
      %sd = cir.mul %param.sigma_prop, %pred
      %obs = cir.normal_sample %pred, %sd, %rng_state
      cir.yield %obs : !obs_type
  }
}
```

---

### 10.3 NIR: Numeric Building Blocks

NIR is **unit-erased** and **tensor-oriented**. It provides a small set of **high-level numeric operations** that CIR lowers to.

#### 10.3.1 ODE Integration

CIR dynamics regions are lowered to an **ODE integration operation**:

```mlir
%trajectory = nir.ode_integrate(
    %f       : (tensor<?x?xf64>, tensor<?xf64>, f64) -> tensor<?x?xf64>,  // RHS function
    %X0      : tensor<?x?xf64>,      // initial states [batch, n_state]
    %theta   : tensor<?x?xf64>,      // parameters [batch, n_param]
    %controls: !nir.ode_controls,    // dosing events, discontinuities
    %t_grid  : tensor<?xf64>,        // output time grid
    %solver_cfg: !nir.solver_config  // tolerances, method
) -> tensor<?x?x?xf64>               // [batch, n_time, n_state]
```

Key features:

- **Batched**: operates on multiple individuals/parameter sets in parallel,
- **Differentiable**: supports adjoint/forward sensitivity for gradients,
- **Event handling**: `controls` encodes dose times, infusion start/stop, observation grid.

The RHS function `f` is itself a NIR function built from the CIR dynamics region, with units stripped.

#### 10.3.2 Probability Density Operations

Each `cir.prob_kernel` and `cir.measure` lowers to NIR ops:

```mlir
%logpdf = nir.logpdf_mvn(
    %x    : tensor<?xf64>,     // random variable value
    %mu   : tensor<?xf64>,     // mean
    %Sigma: tensor<?x?xf64>    // covariance
) -> f64

%sample = nir.sample_mvn(
    %mu   : tensor<?xf64>,
    %Sigma: tensor<?x?xf64>,
    %rng  : !nir.rng_state
) -> tensor<?xf64>
```

Similarly for other distributions (Normal, LogNormal, HalfNormal, etc.).

#### 10.3.3 Likelihood Aggregation

Population-level log-likelihood is built from:

1. **Per-individual trajectory simulation**:
   ```mlir
   %traj_i = nir.ode_integrate(...)
   ```

2. **Observable extraction** at observation times:
   ```mlir
   %pred_ij = nir.gather(%traj_i, %obs_times_i)
   ```

3. **Log-likelihood contribution per observation**:
   ```mlir
   %loglik_ij = nir.logpdf_normal(%obs_ij, %pred_ij, %sigma_ij)
   ```

4. **Summing over observations and individuals**:
   ```mlir
   %loglik_total = nir.reduce_sum(%loglik_ij)
   ```

This is expressed as a **batched map-reduce** over the cohort.

---

### 10.4 Lowering Timeline to Numeric Controls

A `cir.timeline` with dose and observation events is lowered to:

1. **Discontinuity grid**: time points where ODE solver must stop/restart (dose times, infusion start/stop),
2. **Impulse injections**: instantaneous additions to state at dose times,
3. **Infusion schedules**: piecewise-constant input function `u(t)`,
4. **Observation grid**: time points where observables are evaluated.

Example lowering (pseudo-code):

```python
def lower_timeline(timeline: cir.Timeline, model: cir.Model) -> nir.ODEControls:
    dose_times = [event.time for event in timeline if event.type == Dose]
    obs_times  = [event.time for event in timeline if event.type == Obs]
    
    # Build piecewise-constant input function u(t)
    u_intervals = []
    for event in sorted(timeline, key=lambda e: e.time):
        if event.type == StartInfusion:
            u_intervals.append((event.time, inf, event.rate, event.target))
        elif event.type == StopInfusion:
            u_intervals.append((event.time, inf, 0.0, event.target))
        elif event.type == Bolus:
            # Impulse: handled separately as state reset
            pass
    
    return nir.ODEControls(
        discontinuities = sorted(set(dose_times)),
        impulses        = [(t, amount, compartment) for t, amount, compartment in bolus_doses],
        input_function  = piecewise_constant(u_intervals),
        output_grid     = sorted(obs_times)
    )
```

This control structure is passed to the ODE integrator, which:

- stops at each discontinuity,
- applies impulses to the state,
- evaluates the input function `u(t)` as needed,
- outputs state at the observation grid.

---

### 10.5 Population Model Batching

For efficient inference, NIR represents **batched operations** over populations:

- States: `tensor<[N_indiv, N_state], f64>`
- Parameters: `tensor<[N_indiv, N_param], f64>`
- Trajectories: `tensor<[N_indiv, N_time, N_state], f64>`

**Random effects** are sampled in batch:

```mlir
%eta_batch = nir.sample_mvn_batch(
    %mu    = constant 0.0 : tensor<3xf64>,
    %Sigma = %Omega : tensor<3x3xf64>,
    %N     = %N_indiv : index,
    %rng   = %rng_state
) -> tensor<?x3xf64>   // [N_indiv, 3]
```

**Individual parameters** are computed in batch via vectorized transformations:

```mlir
%CL_batch = nir.mul(
    %CL_pop,
    nir.pow(%WT_batch / 70.0, 0.75),
    nir.exp(%eta_CL_batch)
) -> tensor<?xf64>
```

**ODE integration** is vectorized:

```mlir
%traj_batch = nir.ode_integrate_batch(
    %f, %X0_batch, %theta_batch, %controls_batch, %t_grid, %solver_cfg
) -> tensor<?x?x?xf64>
```

This batching is critical for:

- **GPU execution**: batch size becomes the parallel dimension,
- **Efficient MCMC**: evaluate likelihood for all individuals in a single kernel launch,
- **Virtual trials**: simulate thousands of individuals concurrently.

---

### 10.6 Backend Mapping

NIR is designed to be **lowered to multiple backends**:

#### 10.6.1 Frequentist NLME Backends

For backends like NONMEM, Monolix, or custom FOCE/SAEM engines:

- **Export CIR model** to backend-specific syntax (e.g., `$PK`, `$DES`, `$ERROR` for NONMEM),
- **Map `cir.prob_kernel`** to `OMEGA` / `ETA` definitions,
- **Map `cir.measure`** to `$ERROR` / `SIGMA` / `EPS`,
- **Map `cir.timeline`** to dosing records and observation grid.

NIR can also **drive custom engines** directly:

- Compile NIR to LLVM or GPU code,
- Implement Laplace approximation or SAEM in the runtime,
- Use NIR `ode_integrate` and `logpdf` ops as building blocks.

#### 10.6.2 Bayesian / PPL Backends

For backends like Stan, PyMC, NumPyro:

- **Export NIR to Stan code**:
  - NIR ODE ops → Stan `ode_rk45` or Torsten `pmx_solve_*`,
  - NIR logpdf ops → Stan `target +=` statements,
  - NIR parameters → Stan `parameters` block,
  - CIR priors → Stan priors in `model` block.

- **Export NIR to JAX/PyTorch**:
  - NIR → JAX `jax.lax.scan` + `diffrax.diffeqsolve`,
  - NIR logpdf → `jax.scipy.stats.*` or `torch.distributions.*`,
  - Use NumPyro/Pyro for MCMC on top.

The MedLang compiler provides **modular exporters** for each target.

#### 10.6.3 Custom GPU/HPC Backends

For high-performance simulation or likelihood evaluation:

- **Lower NIR to MLIR**:
  - `nir.ode_integrate` → custom MLIR dialect for ODE solvers (or library calls),
  - `nir.logpdf_*` → inlined math ops (log, exp, matrix ops),
  - batched ops → `linalg` / `affine` / `scf` dialects with parallel loops.

- **Lower MLIR to GPU**:
  - `scf.parallel` → `gpu.launch`,
  - `linalg.generic` → custom CUDA/ROCm kernels or library calls (cuBLAS, cuSolver, etc.),
  - Final LLVM IR → PTX or GCN assembly.

This path is essential for:

- **Virtual clinical trials** with millions of individuals,
- **Real-time Bayesian updating** in clinical decision support,
- **Large-scale QSP** with 100+ ODEs and complex networks.

---

### 10.7 ML / Hybrid Model Integration in IR

For Track D models with ML components (Section 8):

#### 10.7.1 CIR Representation

ML submodels are represented as **opaque function calls** in CIR:

```mlir
cir.ml_submodel @CL_NN
  input_type  = !cir.tensor<f64, [?,10]>   // [batch, 10 features]
  output_type = !cir.tensor<f64, [?]>      // [batch] dimensionless multiplier
  param_type  = !cir.ml_params<"torch_state_dict", "cl_nn.pt">
{
  cir.call_external @torch_forward
}
```

This defers the actual ML computation to an **external runtime** (PyTorch, JAX, TensorFlow).

#### 10.7.2 NIR Representation

In NIR, ML calls become:

```mlir
%CL_mult = nir.ml_call(
    @CL_NN,
    %features : tensor<?x10xf64>,
    %weights  : !nir.ml_weights
) -> tensor<?xf64>
```

with well-defined:

- **Forward pass** (evaluation),
- **Backward pass** (gradients w.r.t. weights and inputs, for end-to-end differentiation).

#### 10.7.3 Differentiation Through ML and ODE

For PINN-style training or joint parameter estimation:

- NIR must support **automatic differentiation** through:
  - ML subgraph calls,
  - ODE integration (via adjoint method or forward sensitivity).

- The compiler constructs a **reverse-mode AD graph** that:
  - Backpropagates through ML layers,
  - Backpropagates through ODE solver (adjoint state),
  - Computes gradients of loss w.r.t. all parameters (mechanistic + ML).

This is conceptually similar to:

- **JAX**: `jax.grad` through `diffrax.diffeqsolve`,
- **PyTorch**: `torch.autograd` through `torchdiffeq.odeint`.

MedLang's IR is designed to enable **similar flows** but with:

- **explicit unit checking** at CIR level,
- **backend flexibility** (not tied to a single AD framework).

---

### 10.8 Diagnostics and Debug Information

The IR preserves **source location metadata** from the MedLang surface syntax:

- Each CIR/NIR operation is annotated with:
  - source file,
  - line/column,
  - user-visible names (e.g., "CL", "A_central").

This enables:

- **Error reporting**: "ODE integration failed for individual 42 at parameter CL=15.3 L/h (line 23 of model.med)",
- **Profiling**: "90% of runtime spent in ODE solver for Model @OneCptOralPK",
- **Debugging**: step through IR transformations while preserving link to source.

Implementations should emit:

- **DWARF debug info** for compiled code (if lowering to LLVM),
- **Stack traces** linking NIR ops back to CIR and surface syntax.

---

### 10.9 Testing and Validation Strategy

Implementations of Track D must validate correctness at each IR level:

#### 10.9.1 CIR Validation

- **Type checking**: all operations respect unit and type rules,
- **Well-formedness**: models have required regions (dynamics, observables), timelines are sorted, etc.

#### 10.9.2 NIR Validation

- **Shape consistency**: tensor operations have compatible shapes,
- **Numerical correctness**: ODE integrators meet tolerance specs, probability densities normalize (where analytically checkable).

#### 10.9.3 End-to-End Validation

Compare MedLang implementations against **reference solutions**:

1. **NONMEM comparison**:
   - Translate a MedLang model to NONMEM,
   - Run both on the same dataset,
   - Compare parameter estimates (should match within numerical tolerance).

2. **Stan comparison**:
   - Translate to Stan,
   - Compare posterior distributions (KL divergence, Wasserstein distance).

3. **Analytic test cases**:
   - Use models with known closed-form solutions (1-compartment IV bolus),
   - Verify trajectories match analytic formulas.

4. **Stochastic tests**:
   - Run virtual trials with known parameters,
   - Verify that parameter estimates recover true values (bias, MSE, coverage).

A **reference test suite** should be maintained in the MedLang repository, covering:

- canonical PK/PD models,
- NLME fitting scenarios,
- Bayesian inference cases,
- hybrid ML models.

---

### 10.10 Performance Considerations

#### 10.10.1 ODE Solver Choice

Different models require different solvers:

- **Non-stiff**: explicit RK methods (RK45, Dormand-Prince),
- **Stiff**: implicit methods (CVODE, Radau),
- **Large QSP**: sparse Jacobian exploitation, Krylov methods.

NIR `solver_config` allows per-model tuning; backends should:

- **Auto-select** solvers based on model characteristics (Jacobian sparsity, stiffness detection),
- **Expose tuning knobs** for tolerances, max steps, etc.

#### 10.10.2 Batching and Parallelism

For population models with N individuals:

- **CPU**: parallelize over individuals (OpenMP, thread pools),
- **GPU**: batch as many individuals as fit in memory, SIMD/SIMT execution.

Typical GPU kernel structure:

```cuda
__global__ void population_likelihood_kernel(
    float* theta_batch,    // [N, P]
    float* eta_batch,      // [N, Q]
    float* observations,   // [N, T]
    float* output_loglik   // [N]
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Integrate ODE for individual i
        // Compute log-likelihood for individual i
        output_loglik[i] = ...;
    }
}
```

For very large N (millions), use **multi-GPU** strategies with MPI or NCCL.

#### 10.10.3 Gradient Computation

For NLME optimization or HMC:

- **Adjoint method** for ODE gradients (O(1) memory, backward pass),
- **Forward sensitivity** for low-dimensional parameters (parallel gradient computation),
- **Automatic differentiation** for ML components.

Backends should choose method based on:

- number of parameters vs. number of states,
- memory constraints,
- hardware (GPU adjoint solvers are complex but feasible).

---

### 10.11 Summary

Section 10 provides:

- A **map of IR layers** (CIR → NIR → MLIR → LLVM/GPU),
- **Concrete representations** of Track D constructs (`model`, `timeline`, `prob_kernel`, `measure`) in CIR,
- **Lowering patterns** to numeric operations in NIR (ODE integration, probability ops, batched execution),
- **Backend export strategies** for NONMEM, Stan, custom GPU solvers,
- **ML integration** at the IR level (external calls, differentiation),
- **Testing and validation** requirements,
- **Performance considerations** for real-world pharmacometric workloads.

This establishes Track D as:

- **Implementable** with clear compilation paths,
- **Interoperable** with existing tools,
- **High-performance** via modern compiler and hardware optimizations,
- **Extensible** to quantum-derived parameters (Track C) and multi-scale models (future tracks).

---

## 11. Track C → Track D Mapping (Quantum Pharmacology Bridge)

This section formalizes how **Track C (Quantum Extension)** outputs, specified in `medlang_qm_pharmacology_spec_v0.1.md`, are consumed by **Track D (Pharmacometrics/QSP)**.

Track C provides domain-level operators such as:

- `QM_BindingFreeEnergy` → outputs including:
  - `ΔG_bind : Quantity<Energy, f64>`,
  - `k_on : RateConstPerConc`,
  - `k_off : RateConst`.
- `QM_PartitionCoefficient` → outputs including:
  - `ΔG_partition : Quantity<Energy, f64>`,
  - `Kp : f64` (dimensionless partition coefficient).

Track D consumes these outputs as **typed covariates** and **hyperparameters** for PBPK and QSP models. The mapping is defined at the level of **parameter transforms**, respecting all unit rules of Section 4.

### 11.1 Quantum → PBPK: Partition Coefficients and Kp

Given a quantum partition result for a drug:

```medlang
let partition_result = QM_PartitionCoefficient {
    molecule = drug,
    phase_A  = SMD(solvent = "water"),      // plasma
    phase_B  = SMD(solvent = "tumor_tissue"),
    method   = qm_method,
    temperature = 310.0 K
}

let ΔG_partition : Energy = partition_result.ΔG_partition
let Kp_QM        : f64    = partition_result.Kp
```

Track D defines a **tissue partition coefficient** for PBPK models as:

```medlang
fn Kp_tissue(
    ΔG_partition : Quantity<Energy, f64>,
    T            : Quantity<Kelvin, f64>,
    w_ML         : MLParamVector,
    eta_Kp       : f64
) -> f64 {
    // Thermodynamic baseline from QM
    let R      : EnergyPerMolPerK = 8.314e-3 kJ/(mol·K)  // gas constant
    let expo   : f64 = -(ΔG_partition / (R * T))         // dimensionless
    let Kp_QM  : f64 = exp(expo)                         // dimensionless

    // ML correction (dimensionless) and random effect
    let Kp_ML  : f64 = KpT_ML{w = w_ML}.g_Kp(ΔG_partition, other_features)
    let Kp_IIV : f64 = exp(eta_Kp)

    return Kp_QM * Kp_ML * Kp_IIV
}
```

PBPK tumor or organ compartments then use `Kp_tissue` as the tissue:plasma partition:

```medlang
model PBPK_Tumor {
    param Kp_tumor : f64 = Kp_tissue(
        ΔG_partition = partition_result.ΔG_partition,
        T            = 310.0 K,
        w_ML         = w_Kp,
        eta_Kp       = eta_Kp_tumor_i
    )
    ...
}
```

This makes the PBPK structure **quantum-informed** but still allows:

* ML corrections (to capture multiscale effects not handled by QM), and
* inter-individual variability via random effects.

### 11.2 Quantum → PD/QSP: EC50 and k_kill

Given binding data from `QM_BindingFreeEnergy` and `QM_Kinetics`:

```medlang
let binding = QM_BindingFreeEnergy {
    ligand   = drug,
    target   = receptor,
    ...
}

let kinetics = QM_Kinetics {
    ligand   = drug,
    target   = receptor,
    ...
}

let ΔG_bind : Energy          = binding.ΔG_bind
let k_on    : RateConstPerConc = kinetics.k_on
let k_off   : RateConst        = kinetics.k_off
```

Track D defines **quantum-informed PD parameters** as:

#### 11.2.1 EC50 from ΔG_bind

```medlang
fn Kd_from_ΔG(
    ΔG_bind : Quantity<Energy, f64>,
    T       : Quantity<Kelvin, f64>
) -> Quantity<Concentration, f64> {
    let R  : EnergyPerMolPerK = 8.314e-3 kJ/(mol·K)
    let C0 : Concentration    = 1.0 M                 // standard concentration
    
    let exponent : f64 = (ΔG_bind / (R * T))          // dimensionless
    return C0 * exp(exponent)
}

fn EC50_from_Kd(
    Kd          : Quantity<Concentration, f64>,
    alpha_EC50  : f64,     // calibration factor
    eta_EC50    : f64      // random effect
) -> Quantity<Concentration, f64> {
    return alpha_EC50 * Kd * exp(eta_EC50)
}
```

Usage in Track D model:

```medlang
// Population model
param alpha_EC50 : f64              // calibration factor to be estimated
rand  eta_EC50   : f64 ~ Normal(0, omega_EC50^2)

let Kd_QM   = Kd_from_ΔG(ΔG_bind = binding.ΔG_bind, T = 310.0 K)
let EC50_i  = EC50_from_Kd(Kd_QM, alpha_EC50, eta_EC50)

// Use in PD model
model PD_Emax {
    param EC50 : Concentration = EC50_i
    ...
}
```

#### 11.2.2 k_kill from k_on, k_off

A QSP tumor-immune model may define:

```medlang
fn f_QM_kill_scale(
    k_on      : Quantity<RateConstPerConc, f64>,
    k_off     : Quantity<RateConst, f64>,
    k_on_ref  : Quantity<RateConstPerConc, f64>,
    k_off_ref : Quantity<RateConst, f64>,
    beta_on   : f64,
    beta_off  : f64
) -> f64 {
    let ratio_on  : f64 = (k_on / k_on_ref)           // dimensionless
    let ratio_off : f64 = (k_off_ref / k_off)         // dimensionless
    return pow(ratio_on, beta_on) * pow(ratio_off, beta_off)
}

fn k_kill_from_QM(
    k_on         : Quantity<RateConstPerConc, f64>,
    k_off        : Quantity<RateConst, f64>,
    k_kill_base  : Quantity<RateConst, f64>,
    eta_k_kill   : f64
) -> Quantity<RateConst, f64> {
    let f_qm : f64 = f_QM_kill_scale(k_on, k_off, k_on_ref, k_off_ref, beta_on, beta_off)
    return k_kill_base * f_qm * exp(eta_k_kill)
}
```

Binding into QSP:

```medlang
param k_kill_base : RateConst
param beta_on     : f64 = 0.5
param beta_off    : f64 = 0.5
rand  eta_k_kill  : f64 ~ Normal(0, omega_k_kill^2)

let k_kill_i = k_kill_from_QM(kinetics.k_on, kinetics.k_off, k_kill_base, eta_k_kill)

model TumorImmuneQSP {
    param k_kill : RateConst = k_kill_i
    
    dTumor/dt = k_grow * Tumor * (1 - Tumor/T_max) - k_kill * Effector * Tumor
    ...
}
```

Thus, tumor killing rates and EC50 values are **functions of quantum binding mechanics**, with:

* explicit scaling hyperparameters (`alpha_EC50`, `beta_on`, `beta_off`), and
* random effects for inter-individual or inter-compound variability.

### 11.3 Probabilistic Use of QM Outputs

Track D treats Track C outputs **deterministically** by default, but they can inform priors:

```medlang
// Example: EC50 prior centered on quantum prediction
let Kd_QM = Kd_from_ΔG(binding.ΔG_bind, T = 310.0 K)

inference Bayesian_QM_Informed {
    population_model = ...
    
    priors {
        // Prior centered at QM prediction with QM uncertainty
        alpha_EC50 ~ Normal(1.0, 0.5)  // calibration factor
        
        // Or allow ΔG_bind itself to vary around QM prediction
        ΔG_bind_true ~ Normal(
            mean = binding.ΔG_bind,
            sd   = binding.uncertainty
        )
        
        // Use ΔG_bind_true in parameter mapping
        let Kd_i = Kd_from_ΔG(ΔG_bind_true, T)
        ...
    }
}
```

or appear in `Measure`s as pseudo-data. This retains:

* physical grounding from quantum pharmacology, and
* statistical calibration and uncertainty quantification at the population level.

### 11.4 Summary

Section 11 establishes:

- **Formal mappings** from Track C quantum operators to Track D parameters:
  * `ΔG_partition → Kp` (PBPK tissue partitioning)
  * `ΔG_bind → Kd → EC50` (PD potency)
  * `k_on, k_off → k_kill` (QSP kinetics)

- **Unit safety** through all mappings:
  * Thermodynamic exponentials are dimensionless: `exp(ΔG/(R·T))`
  * Output quantities have correct units: `Kd : Concentration`, `EC50 : Concentration`, `k_kill : RateConst`

- **Calibration layers** allow data-driven correction:
  * `alpha_EC50`, `beta_on`, `beta_off` as free parameters
  * Posterior inference reveals agreement/tension between QM and clinical data

- **Uncertainty propagation** in Bayesian mode:
  * QM method uncertainty → parameter priors → PK/PD predictions

This completes the quantum-to-clinical vertical, enabling:

> **Ab initio quantum calculations → PBPK partition → QSP dynamics → Population inference → Clinical outcomes**

with full type safety, unit consistency, and statistical rigor.

---

*This completes the MedLang Pharmacometrics & QSP Specification v0.1 with Track C integration. All sections (1–11) are now complete. The specification provides a rigorous foundation for quantum-informed pharmacometrics with hybrid mechanistic-ML models.*
