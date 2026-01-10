---
name: julia-to-medlang
description: Convert Julia code (DifferentialEquations.jl, Turing.jl, Catlab.jl) to MedLang. Use when porting Julia models to MedLang, analyzing Julia ODE systems, translating Bayesian inference, mapping category-theoretic structures, or validating dimensional consistency. Handles syntax conversion, type mapping, and idiomatic MedLang translation.
allowed-tools: Read, Grep, Glob, WebFetch
---

# Julia-to-MedLang Port Skill

Ports Julia pharmacokinetic/pharmacodynamic models and category-theoretic structures to MedLang with correct semantics and dimensional analysis.

## When to Use This Skill

✓ Converting Julia ODE models to MedLang
✓ Translating DifferentialEquations.jl systems
✓ Porting Turing.jl Bayesian models
✓ Mapping Catlab.jl category structures
✓ Understanding Julia pharmacometric patterns
✓ Validating dimensional consistency in conversions
✓ Converting Catlab morphisms to MedLang transformations

## Core Translation Philosophy

MedLang is a **domain-specific language for medical/pharmaceutical modeling** with:
- Compile-time **dimensional analysis** (M·L·T type system)
- GPU/HPC acceleration via Stan and Julia backends
- **Explicit parameter classification** (Fixed, PopulationMean, PopulationVariance, RandomEffect)

Julia (especially Catlab.jl) emphasizes **composability** and **category theory**. When porting:
1. Extract mathematical semantics (ODEs, distributions, transformations)
2. Map to MedLang's dimensional type system
3. Preserve functional composition where applicable
4. Use MedLang's parameter structure for hierarchical models

## Quick Reference: ODE Translation

### Julia (DifferentialEquations.jl)

```julia
function pk_system!(du, u, p, t)
    CL, V = p
    A_central = u[1]
    du[1] = -(CL/V) * A_central
end
```

### MedLang

```medlang
param CL : Clearance    // L³/T
param V : Volume        // L³

// ODE in d-notation
dA_central/dt = -(CL / V) * A_central
```

**Key difference**: Named variables + dimensional types replace array indexing.

---

## Parameter Classification

When porting Julia parameters to MedLang, classify them:

| Julia Pattern | MedLang Classification | Meaning |
|---------------|----------------------|---------|
| `dose = 100.0` | `fixed DOSE : Mass` | Time-invariant input |
| `CL_pop = 2.5` | `param CL_pop : Clearance` | Population mean (estimated) |
| `ω_CL = 0.3` | `param ω_CL : StdDev` | Inter-individual variability |
| `η ~ N(0,1)` | `param η_CL : RandomEffect` | Subject deviation |
| `CL = CL_pop * exp(ω_CL * η_CL)` | `param CL : Clearance = ...` | Subject-level parameter |

---

## Distribution Mapping

| Julia | MedLang | Syntax |
|-------|---------|--------|
| `Normal(μ, σ)` | Normal prior | `param X : Type ~ Normal(μ, σ)` |
| `Exponential(λ)` | Exponential prior | `param X : Type ~ Exponential(λ)` |
| `LogNormal(μ, σ)` | Log-normal prior | `param X : Type ~ LogNormal(μ, σ)` |
| `Uniform(a, b)` | Uniform prior | `param X : Type ~ Uniform(a, b)` |

**Likelihood** (from Turing.jl):
```julia
y ~ MvNormal(μ, σ²)  // Julia
```

```medlang
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central/V + error, σ)  // MedLang
```

---

## Catlab.jl → MedLang: Category-Theoretic Structures

### Morphisms as Parameter Transformations

**Catlab approach** (functorial mapping):
```julia
struct Morphism
    domain::Object
    codomain::Object
    map::Function
end

# Functor F: Category1 → Category2
F(x::Obj1) = map_to_category2(x)
```

**MedLang approach** (dimensional transformation):
```medlang
// Weight-scaled clearance (covariate morphism)
param WT : Mass
param CL_pop : Clearance
param allometry_exp : Exponent = 0.75

// Morphism: WT → CL (functional composition)
param CL : Clearance = CL_pop * (WT / 70.0_kg)^allometry_exp
```

### Wiring Diagrams as ODE Composition

**Catlab** (compositional structure):
```julia
# Wiring diagram: inputs → processes → outputs
diagram = WiringDiagram([In, In], [Out])
add_box!(diagram, add_box)
add_box!(diagram, multiply_box)
add_wire!(diagram, (input_id, 1) => (add_id, 1))
```

**MedLang** (ODE system composition):
```medlang
// Composition of processes: absorption → distribution → elimination

param ka : RateConst    // absorption rate constant
param CL : Clearance    // elimination clearance
param V : Volume        // distribution volume

dA_gut/dt = -ka * A_gut              // absorption box
dA_central/dt = ka * A_gut - (CL/V) * A_central  // distribution + elimination
```

### Functors as Type Mappings

**Catlab functor**:
```julia
function F(c::Category) -> Category
    # Map objects: F(X), F(Y), ...
    # Map morphisms: F(f), F(g), ...
end
```

**MedLang dimensional functor**:
```medlang
// Implicit functor via M·L·T type system
// F: (arbitrary units) → (M·L·T dimensions)

param CL : Clearance      // F(CL_raw) = Volume/Time
param V : Volume          // F(V_raw) = Length³
param ka : RateConst      // F(ka_raw) = 1/Time

// Type system ensures composability:
// (CL : L³/T) / (V : L³) = 1/T (valid elimination rate)
dA/dt = -(CL/V) * A     // type-checked composition
```

---

## Step-by-Step Porting Workflow

### 1. **Identify Components**

| Julia | Extract |
|-------|---------|
| Function signature | Parameters and their roles |
| `du[i]` assignments | ODE equations (use d-notation) |
| `p` vector | Parameter list with dimensions |
| `u` vector | State variables with units |
| Priors/distributions | Parameter bounds and uncertainty |

### 2. **Extract Dimensions (M·L·T Analysis)**

For each parameter, determine:
- **M** (Mass): Amounts, dosages, weights
- **L** (Length): Volumes, compartments
- **T** (Time): Rates, clearances

**Example**:
```julia
# Julia: p = [CL, V, ka]
CL = 5.0      # How to interpret? L/T? mL/min/kg?
V = 50.0      # How to interpret? mL? L?
ka = 0.8      # How to interpret? 1/hour? 1/min?
```

**MedLang - be explicit**:
```medlang
param CL : Clearance      // L³/T (volume per time)
param V : Volume          // L³ (pure volume)
param ka : RateConst      // 1/T (rate constant)
```

### 3. **Classify Parameters**

```medlang
// Fixed (given data)
fixed DOSE : Mass
fixed WT : Mass

// Population level (estimated)
param CL_pop : Clearance
param V_pop : Volume

// Variability (population variance)
param ω_CL : StdDev
param ω_V : StdDev

// Random effects (subject deviations)
param η_CL : RandomEffect
param η_V : RandomEffect

// Subject level (derived)
param CL : Clearance = CL_pop * exp(ω_CL * η_CL)
param V : Volume = V_pop * exp(ω_V * η_V)
```

### 4. **Convert ODEs**

```julia
# Julia array notation
du[1] = -(CL/V) * u[1]
du[2] = (CL/V) * u[1] - (CL2/V2) * u[2]
```

```medlang
# MedLang d-notation (named variables)
dA_central/dt = -(CL/V) * A_central
dA_peripheral/dt = (CL/V) * A_central - (CL2/V2) * A_peripheral
```

### 5. **Encode Priors & Likelihood**

```julia
# Turing.jl
@model function pk_model(y)
    CL ~ Normal(2.0, 0.5)
    σ ~ Exponential(1)
    μ = predict(CL, ...)
    y ~ MvNormal(μ, σ²)
end
```

```medlang
# MedLang
param CL : Clearance ~ Normal(2.0, 0.5)
param σ : StdDev ~ Exponential(1)
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central/V + error, σ)
```

### 6. **Validate Dimensional Consistency**

MedLang type checker automatically verifies:
```medlang
param CL : Clearance    // L³/T
param V : Volume        // L³

// (L³/T) / L³ = 1/T ✓
dA/dt = -(CL/V) * A     // rate equation - valid!
```

---

## Common Issues & Solutions

### Issue: Index-based Array Access

**Julia**:
```julia
du[1] = -k * u[1]  # What do [1] and [1] mean?
```

**MedLang** (explicit naming):
```medlang
dA_central/dt = -ka * A_central  # clear meaning
```

**Solution**: Create a mapping table showing original indices → variable names.

---

### Issue: Implicit Units

**Julia**:
```julia
CL = 5.0   # 5 what? L/min? mL/h? Subject to user interpretation
```

**MedLang** (dimensional types):
```medlang
param CL : Clearance ~ Normal(2.0, 0.5)  # Clearance type enforces L³/T
```

**Solution**: Audit all parameters, assign M·L·T dimensions, update priors accordingly.

---

### Issue: Functional Composition (Catlab)

**Catlab** (wiring diagrams, morphism composition):
```julia
diagram = @acset Wiring(
  boxes=[Box(f), Box(g)],
  wires=[Wire(box1, port1 => box2, port2)]
)
```

**MedLang** (ODE composition):
```medlang
// Processes are composed via ODE system:
dA1/dt = f(A1, params)      // first "box"
dA2/dt = g(A1, A2, params)  // second "box", wired to A1
```

**Solution**: Reframe morphisms as parameter transformations and ODE couplings.

---

### Issue: Type Abstraction

**Catlab** (abstract categories):
```julia
@abstract_acset_type AbstractWiringDiagram
struct WiringDiagram <: AbstractWiringDiagram end
```

**MedLang** (concrete types):
```medlang
param CL : Clearance        // concrete type, enforced at compile time
param V : Volume            // no abstract supertypes in V0
```

**Solution**: Use MedLang's fixed type set (Mass, Length, Time, derived types); compose via parameter definitions.

---

## Verification Checklist

After porting Julia to MedLang, verify:

- [ ] All ODEs translated to d-notation
- [ ] All parameters have dimensions (M·L·T types)
- [ ] Fixed inputs marked with `fixed`
- [ ] Population parameters marked with `param`
- [ ] Random effects and variability included
- [ ] Priors converted to MedLang `~ Distribution` syntax
- [ ] Likelihood model specified correctly
- [ ] `mlc check model.medlang` passes (syntax & type checking)
- [ ] `mlc compile model.medlang` generates valid Stan/Julia
- [ ] Generated code matches Julia numeric behavior (test on sample data)

---

## Example: One-Compartment Oral PK

### Julia (DifferentialEquations.jl + Turing.jl)

```julia
using DifferentialEquations, Turing, Distributions

function pk_system!(du, u, p, t)
    ka, CL, V = p[1], p[2], p[3]
    A_gut, A_central = u[1], u[2]

    du[1] = -ka * A_gut
    du[2] = ka * A_gut - (CL/V) * A_central
end

@model function pk_model(y_obs, dose)
    # Priors
    ka ~ LogNormal(log(1.2), 0.3)
    CL_pop ~ LogNormal(log(5.0), 0.3)
    V_pop ~ LogNormal(log(50.0), 0.3)
    ω_CL ~ Exponential(2)
    ω_V ~ Exponential(2)
    σ ~ Exponential(1)

    # Random effects
    η_CL ~ MvNormal(zeros(N), I)
    η_V ~ MvNormal(zeros(N), I)

    # Subject parameters
    CL = CL_pop .* exp.(ω_CL .* η_CL)
    V = V_pop .* exp.(ω_V .* η_V)

    # ODE solving
    u0 = [dose, 0.0]
    tspan = (0.0, 24.0)
    p_ode = [ka, CL[1], V[1]]
    prob = ODEProblem(pk_system!, u0, tspan, p_ode)
    sol = solve(prob, Tsit5())

    # Likelihood
    y_pred = sol(times)
    y_obs ~ MvNormal(y_pred ./ V[1], σ^2)
end
```

### MedLang Equivalent

```medlang
// Fixed inputs
fixed DOSE : Mass = 100.0_mg
fixed WT : Mass = 70.0_kg

// Population parameters (means)
param ka_pop : RateConst ~ LogNormal(0.18, 0.3)
param CL_pop : Clearance ~ LogNormal(1.61, 0.3)
param V_pop : Volume ~ LogNormal(3.91, 0.3)

// Inter-individual variability
param ω_CL : StdDev ~ Exponential(2.0)
param ω_V : StdDev ~ Exponential(2.0)
param ω_ka : StdDev ~ Exponential(2.0)

// Random effects (subject deviations)
param η_CL : RandomEffect ~ Normal(0, 1)
param η_V : RandomEffect ~ Normal(0, 1)
param η_ka : RandomEffect ~ Normal(0, 1)

// Subject-level parameters
param ka : RateConst = ka_pop * exp(ω_ka * η_ka)
param CL : Clearance = CL_pop * exp(ω_CL * η_CL)
param V : Volume = V_pop * exp(ω_V * η_V)

// ODE system
dA_gut/dt = -ka * A_gut
dA_central/dt = ka * A_gut - (CL / V) * A_central

// Observation model
param σ : StdDev ~ Exponential(1)
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central / V + error, σ)
```

---

## References

- **MedLang Grammar**: [medlang_d_minimal_grammar_v0.md](../../docs/medlang_d_minimal_grammar_v0.md)
- **MedLang Architecture**: [ARCHITECTURE.md](../../docs/ARCHITECTURE.md)
- **Canonical Example**: [one_comp_oral_pk.medlang](../../docs/examples/one_comp_oral_pk.medlang)
- **Catlab.jl Docs**: https://algebraicjulia.github.io/Catlab.jl/dev/
- **DifferentialEquations.jl**: https://diffeq.sciml.ai/
- **Turing.jl**: https://turinglang.org/
