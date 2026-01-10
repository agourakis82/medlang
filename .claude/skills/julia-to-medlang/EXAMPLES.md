# Julia-to-MedLang Conversion Examples

Real-world examples converting Julia pharmacokinetic models to MedLang.

---

## Example 1: One-Compartment Oral PK (with Random Effects)

### Julia Source

```julia
using DifferentialEquations, Turing, Distributions

function pk_system!(du, u, p, t)
    ka, CL, V = p
    A_gut, A_central = u

    du[1] = -ka * A_gut
    du[2] = ka * A_gut - (CL/V) * A_central
end

@model function pk_model(y_obs, dose, times, n_subjects)
    # Population priors
    θ_ka ~ LogNormal(log(1.2), 0.3)
    θ_CL ~ LogNormal(log(5.0), 0.5)
    θ_V ~ LogNormal(log(50.0), 0.5)

    # Variability priors
    ω_ka ~ Exponential(1.5)
    ω_CL ~ Exponential(1.5)
    ω_V ~ Exponential(1.5)

    # Observation error
    σ ~ Exponential(1.0)

    # Subject random effects
    η_ka ~ filldist(Normal(0, 1), n_subjects)
    η_CL ~ filldist(Normal(0, 1), n_subjects)
    η_V ~ filldist(Normal(0, 1), n_subjects)

    # Generate predictions
    for i in 1:n_subjects
        # Subject-level parameters
        ka_i = θ_ka * exp(ω_ka * η_ka[i])
        CL_i = θ_CL * exp(ω_CL * η_CL[i])
        V_i = θ_V * exp(ω_V * η_V[i])

        # Solve ODE
        u0 = [dose, 0.0]
        p = [ka_i, CL_i, V_i]
        prob = ODEProblem(pk_system!, u0, (0.0, 24.0), p)
        sol = solve(prob, Tsit5(), saveat=times)

        # Likelihood
        C_pred = [sol(t)[2] / V_i for t in times]
        y_obs[i, :] ~ MvNormal(C_pred, σ^2 * I)
    end
end
```

### MedLang Equivalent

```medlang
// Fixed inputs
fixed DOSE : Mass = 100.0_mg
fixed WT : Mass = 70.0_kg

// Population parameters (means)
param ka_pop : RateConst ~ LogNormal(0.18, 0.3)
param CL_pop : Clearance ~ LogNormal(1.61, 0.5)
param V_pop : Volume ~ LogNormal(3.91, 0.5)

// Population parameters (variances)
param ω_ka : StdDev ~ Exponential(1.5)
param ω_CL : StdDev ~ Exponential(1.5)
param ω_V : StdDev ~ Exponential(1.5)

// Observation error
param σ : StdDev ~ Exponential(1.0)

// Subject random effects
param η_ka : RandomEffect ~ Normal(0, 1)
param η_CL : RandomEffect ~ Normal(0, 1)
param η_V : RandomEffect ~ Normal(0, 1)

// Subject-level parameters
param ka : RateConst = ka_pop * exp(ω_ka * η_ka)
param CL : Clearance = CL_pop * exp(ω_CL * η_CL)
param V : Volume = V_pop * exp(ω_V * η_V)

// ODE system
dA_gut/dt = -ka * A_gut
dA_central/dt = ka * A_gut - (CL / V) * A_central

// Observation model
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central / V + error, σ)
```

### Conversion Notes

1. **Array indexing** (`u[1]`, `u[2]`) → named variables (`A_gut`, `A_central`)
2. **Parameter loop** (Julia for-loop over subjects) → MedLang scalar definitions
3. **Random effects** (`filldist(Normal(...), n)`) → single `RandomEffect` parameter, replicated at inference
4. **Likelihood** (`MvNormal(C_pred, σ²I)`) → scalar `Normal(concentration, σ)`
5. **ODE solving** (explicit in Julia) → implicit in MedLang (backend handles integration)

**Verification**: Compile and test:
```bash
mlc check one_comp_oral.medlang -v
mlc compile one_comp_oral.medlang
mlc generate-data -n 20 -o test_data.csv
mlc run one_comp_oral.stan --data test_data.json
```

---

## Example 2: Two-Compartment Model with Covariate Effects

### Julia Source

```julia
using DifferentialEquations, Turing

function two_comp!(du, u, p, t)
    ka, CL, V1, Q, V2, WT, WT_ref = p
    A_gut, A_c1, A_c2 = u

    # Weight-scaled clearance
    CL_adj = CL * (WT / WT_ref)^0.75
    Q_adj = Q * (WT / WT_ref)^0.75

    du[1] = -ka * A_gut
    du[2] = ka * A_gut - (CL_adj + Q_adj) / V1 * A_c1 + Q_adj / V2 * A_c2
    du[3] = Q_adj / V1 * A_c1 - Q_adj / V2 * A_c2
end

@model function two_comp_model(y_obs, dose, WT, times)
    # Priors
    θ_ka ~ LogNormal(log(1.0), 0.4)
    θ_CL ~ LogNormal(log(3.0), 0.5)
    θ_V1 ~ LogNormal(log(30.0), 0.5)
    θ_Q ~ LogNormal(log(2.0), 0.5)
    θ_V2 ~ LogNormal(log(60.0), 0.5)

    ω_ka ~ Exponential(1.5)
    ω_CL ~ Exponential(1.5)
    ω_V1 ~ Exponential(1.5)

    σ ~ Exponential(1.0)

    # Random effects
    η_ka ~ Normal(0, 1)
    η_CL ~ Normal(0, 1)
    η_V1 ~ Normal(0, 1)

    # Subject parameters
    ka = θ_ka * exp(ω_ka * η_ka)
    CL = θ_CL * exp(ω_CL * η_CL)
    V1 = θ_V1 * exp(ω_V1 * η_V1)
    Q = θ_Q
    V2 = θ_V2
    WT_ref = 70.0

    # ODE
    u0 = [dose, 0.0, 0.0]
    p = [ka, CL, V1, Q, V2, WT, WT_ref]
    prob = ODEProblem(two_comp!, u0, (0.0, 72.0), p)
    sol = solve(prob, Tsit5(), saveat=times)

    # Likelihood
    C_pred = [sol(t)[2] / V1 for t in times]
    y_obs ~ MvNormal(C_pred, σ^2 * I)
end
```

### MedLang Equivalent

```medlang
// Fixed inputs
fixed DOSE : Mass = 250.0_mg
fixed WT : Mass = 85.0_kg
fixed WT_ref : Mass = 70.0_kg

// Population parameters
param ka_pop : RateConst ~ LogNormal(0.0, 0.4)
param CL_pop : Clearance ~ LogNormal(1.10, 0.5)
param V1_pop : Volume ~ LogNormal(3.40, 0.5)
param Q_pop : Clearance ~ LogNormal(0.69, 0.5)
param V2_pop : Volume ~ LogNormal(4.09, 0.5)

// Variability
param ω_ka : StdDev ~ Exponential(1.5)
param ω_CL : StdDev ~ Exponential(1.5)
param ω_V1 : StdDev ~ Exponential(1.5)
param σ : StdDev ~ Exponential(1.0)

// Random effects
param η_ka : RandomEffect ~ Normal(0, 1)
param η_CL : RandomEffect ~ Normal(0, 1)
param η_V1 : RandomEffect ~ Normal(0, 1)

// Subject-level parameters with allometric scaling
param ka : RateConst = ka_pop * exp(ω_ka * η_ka)
param CL : Clearance = CL_pop * (WT / WT_ref)^0.75 * exp(ω_CL * η_CL)
param V1 : Volume = V1_pop * (WT / WT_ref) * exp(ω_V1 * η_V1)
param Q : Clearance = Q_pop * (WT / WT_ref)^0.75
param V2 : Volume = V2_pop * (WT / WT_ref)

// ODE system
dA_gut/dt = -ka * A_gut
dA_central/dt = ka * A_gut - (CL + Q)/V1 * A_central + Q/V2 * A_peripheral
dA_peripheral/dt = Q/V1 * A_central - Q/V2 * A_peripheral

// Observation model
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central / V1 + error, σ)
```

### Conversion Notes

1. **Covariate scaling** embedded in parameter definitions (clean, composable)
2. **WT-dependent transformations** (allometric): `(WT/WT_ref)^0.75`
3. **Q is inter-compartmental clearance** (L³/T), not dimensionless
4. **Three compartments** map to three ODE states: `A_gut`, `A_central`, `A_peripheral`

**Key insight**: MedLang's parameter composition (`param X = ...`) replaces Julia's conditional logic in the ODE function.

---

## Example 3: Indirect Response Pharmacodynamics

### Julia Source

```julia
function pd_system!(du, u, p, t)
    Kin, Kout, IC50, gamma, E_max = p
    A_effect = u[1]

    # Input from PK model (concentration)
    C = get_concentration_from_pk(t)

    # Hill equation
    E = E_max * C^gamma / (IC50^gamma + C^gamma)

    # Indirect response
    du[1] = Kin * (1 - E) - Kout * A_effect
end

@model function pd_model(effect_obs, times, concentration)
    Kin ~ Normal(10.0, 2.0)
    Kout ~ Exponential(0.5)
    IC50 ~ LogNormal(log(2.0), 0.5)
    gamma ~ Normal(2.0, 0.5)
    E_max ~ Uniform(0.0, 1.0)
    σ ~ Exponential(1.0)

    # Solve
    u0 = [Kin / Kout]  # baseline
    p = [Kin, Kout, IC50, gamma, E_max]
    prob = ODEProblem(pd_system!, u0, (0.0, 72.0), p)
    sol = solve(prob, Tsit5(), saveat=times)

    # Likelihood
    effect_pred = [sol(t)[1] for t in times]
    effect_obs ~ MvNormal(effect_pred, σ^2 * I)
end
```

### MedLang Equivalent

```medlang
// PK parameters (from previous example)
// [Assume CL, V, etc. are defined above]

// PD parameters
param Kin : Dimensionless ~ Normal(10.0, 2.0)        // production rate
param Kout : RateConst ~ Exponential(0.5)            // elimination rate
param IC50 : ConcMass ~ LogNormal(0.69, 0.5)         // half-maximal concentration
param gamma : Dimensionless ~ Normal(2.0, 0.5)       // Hill coefficient
param E_max : Dimensionless ~ Uniform(0.0, 1.0)      // max effect
param σ : StdDev ~ Exponential(1.0)

// Hill equation (effect as function of concentration)
// E = E_max * C^gamma / (IC50^gamma + C^gamma)
// where C = A_central / V

// Indirect response ODE
// Effect = A_eff (amount in effect compartment)
// dA_eff/dt = Kin * (1 - E) - Kout * A_eff

param C_eff : ConcMass = A_central / V    // effective concentration
param E : Dimensionless = E_max * C_eff^gamma / (IC50^gamma + C_eff^gamma)

dA_eff/dt = Kin * (1 - E) - Kout * A_eff

// Observation model
error ~ Normal(0, σ)
likelihood effect ~ Normal(A_eff + error, σ)
```

### Conversion Notes

1. **Hill equation** (nonlinear pharmacodynamics) expressed as parameter transformation
2. **Indirect response** modeled as ODE (classical PD structure)
3. **Baseline** (`Kin/Kout`) computed implicitly by solver initial conditions
4. **Composite parameters**: `C_eff` derived from PK compartment `A_central`

---

## Example 4: Mixture Model (Below Detection Limit Handling)

### Julia Source (Advanced)

```julia
# Handling data below detection limit
@model function pk_with_bdl(y_obs, times, BDL, dose)
    # ... [priors as before]

    # Likelihood with BDL
    for i in 1:length(times)
        if y_obs[i] < BDL
            # Probability of being below limit
            y_obs[i] ~ Uniform(0, BDL)
        else
            # Standard likelihood above limit
            y_obs[i] ~ Normal(y_pred[i], σ^2)
        end
    end
end
```

### MedLang Current (V0 Limitation)

**Problem**: MedLang V0 does not support conditional likelihood.

**Solutions**:

#### Option A: Pre-process data (Recommended)

```medlang
// Only include observations above detection limit
// y_obs: pre-filtered to BDL-exceeding values

param σ : StdDev ~ Exponential(1.0)
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central / V + error, σ)
```

Then, in data preprocessing (Julia):
```julia
# Filter observations below BDL
y_filtered = y_obs[y_obs .>= BDL]
times_filtered = times[y_obs .>= BDL]
```

#### Option B: Imputation strategy (Workaround)

```medlang
// Replace BDL values with BDL/2 or use multiple imputation externally
// Not ideal, but works with V0

param σ : StdDev ~ Exponential(1.0)
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central / V + error, σ)
```

**Note**: V1 will support conditional likelihoods and mixture models.

---

## Example 5: Catlab.jl Morphism (Category-Theoretic Composition)

### Julia Source (Catlab-based parameter hierarchy)

```julia
using Catlab, Catlab.CategoricalAlgebra

# Define a category of pharmacokinetic parameters
PharmacokineticParams = @acset_type PharmacokineticParams(FreeSchema) begin
    Param::Ob
    Effect::Ob
    depends::Hom(Effect, Param)
    scales::Data(Float64)
end

# Morphism: raw_params → dimensional_params
struct DimensionalFunctor <: Functor
    domain::PharmacokineticParams    # raw parameters
    codomain::PharmacokineticParams  # dimensional parameters

    function map_param(p::Param)
        # p: arbitrary parameter
        # return: (type, dimension)
        # e.g., CL → (Clearance, [1, 3, -1])  # 1/T * L³
    end

    function map_effect(e::Effect)
        # e: composite effect
        # return: (dimension via morphism)
        # e.g., CL_subject → composition of CL_pop, ω_CL, η_CL
    end
end
```

### MedLang Equivalent (Parameter Composition as Morphism)

```medlang
// Category 1: Raw parameters (abstract layer)
// Objects: {raw_CL, raw_V, raw_eta_CL}
// Morphisms: composition rules

// Category 2: Dimensional parameters (concrete layer)
// Objects: {CL : Clearance, V : Volume, η_CL : RandomEffect}

// Functor F: Category1 → Category2
// Maps: raw_CL → CL : Clearance (dimension L³/T)
//       raw_V → V : Volume (dimension L³)
//       raw_eta_CL → η_CL : RandomEffect (dimensionless)

// Population parameters (source objects)
param CL_pop : Clearance ~ Normal(2.0, 0.5)          // raw_CL_pop
param V_pop : Volume ~ Normal(3.0, 0.5)              // raw_V_pop
param ω_CL : StdDev ~ Exponential(0.5)               // raw_ω_CL
param η_CL : RandomEffect ~ Normal(0, 1)             // raw_η_CL

// Functor morphisms: F(raw_X) → dimensional_X
// F(raw_CL_pop, raw_ω_CL, raw_η_CL) = CL (subject-level clearance)
param CL : Clearance = CL_pop * exp(ω_CL * η_CL)

// Wiring diagram (morphism composition)
// Inputs: {CL_pop, ω_CL, η_CL}
// Process: exp(ω_CL * η_CL) * CL_pop
// Output: CL

// ODE system uses the result
dA_central/dt = -(CL / V_pop) * A_central
```

### Conversion Notes

1. **Catlab Objects** (parameters) → MedLang parameter definitions
2. **Catlab Morphisms** (transformations) → MedLang parameter expressions
3. **Functor composition** (F∘G) → nested parameter definitions
4. **Type preservation** through M·L·T dimensional analysis

**Conceptual bridge**:
- Catlab: explicit category theory and morphism composition
- MedLang: implicit functorial structure via type checking

---

## Example 6: Turnover Model (Chain of Exponential Processes)

### Julia Source

```julia
function turnover_system!(du, u, p, t)
    Kin, Kout, Emax, EC50, gamma = p
    Response = u[1]

    # Input from PK model
    C = get_concentration(t)

    # Efficacy (Hill equation)
    E = Emax * C^gamma / (EC50^gamma + C^gamma)

    # Turnover model: balance between production and loss
    # dR/dt = Kin * (1 - E) - Kout * R
    du[1] = Kin * (1 - E) - Kout * Response
end
```

### MedLang Equivalent

```medlang
// Assume PK part already defined (C = A_central / V)

// Turnover model parameters
param Kin : Dimensionless ~ Normal(100.0, 20.0)      // baseline production
param Kout : RateConst ~ Exponential(0.2)            // elimination/degradation
param Emax : Dimensionless ~ Uniform(0.0, 1.0)       // max effect
param EC50 : ConcMass ~ LogNormal(log(1.0), 0.5)     // half-maximal effect
param gamma : Dimensionless ~ Normal(1.0, 0.3)       // Hill exponent

// Derived: concentration
param C : ConcMass = A_central / V

// Derived: efficacy (Hill equation)
param E : Dimensionless = Emax * C^gamma / (EC50^gamma + C^gamma)

// Turnover ODE
dResponse/dt = Kin * (1 - E) - Kout * Response

// Observation
param σ : StdDev ~ Exponential(0.5)
error ~ Normal(0, σ)
likelihood effect ~ Normal(Response + error, σ)
```

**Key pattern**: Multilayer ODE system (PK → derived concentration → PD → response)

---

## Troubleshooting Conversions

### Problem: "Can't convert Julia conditional ODE"

**Julia**:
```julia
if t < t_dose
    du[1] = 0
else
    du[1] = -ka * u[1]
end
```

**MedLang** (V0 workaround):
```medlang
// Option 1: Assume single dose at t=0 (implicit)
dA_gut/dt = -ka * A_gut

// Option 2: Model as input rate (for continuous dosing)
param dose_rate : Clearance = 10.0   // mg/hour
dA_gut/dt = dose_rate - ka * A_gut
```

### Problem: "Julia uses array slicing; MedLang doesn't"

**Julia**:
```julia
y = sol.u[2]  # 2nd state vector
C = y[1]      # 1st element
```

**MedLang**:
```medlang
// All states are named variables
// A_central (not u[1])
// Concentration derived from state
param C : ConcMass = A_central / V
```

### Problem: "Catlab wiring diagram is complex"

**Strategy**: Map each "box" (stage) to ODE subsystem:

```
Julia Catlab:
[Input] → [Box1: Absorption] → [Box2: Distribution] → [Output]

MedLang:
dA_gut/dt = ...              // Box 1
dA_central/dt = ...          // Box 2
```

Each box becomes one ODE equation (or block of equations).

---

## Summary of Common Patterns

| Julia Pattern | MedLang Translation |
|---------------|---------------------|
| Array indexing `u[i]` | Named state variables |
| Parameter loop `for p in params` | Individual `param` definitions + inference replication |
| Conditional ODE `if t < t0` | Fixed initial conditions or rate parameters |
| Nested function calls | Intermediate parameter definitions |
| Wiring diagrams (Catlab) | Coupled ODE systems + parameter composition |
| Implicit type system | Explicit M·L·T dimensional types |

---

## Next Steps

1. Choose your Julia model
2. Follow the step-by-step conversion in REFERENCE.md
3. Test with `mlc check model.medlang`
4. Verify numerical equivalence on sample data
5. Submit model for inference

For questions or stuck conversions, ask the julia-to-medlang skill!
