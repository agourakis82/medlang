# Julia-to-MedLang Conversion Reference

Detailed patterns and advanced mappings for porting Julia code to MedLang.

## Complete Type System Mapping

### MedLang M·L·T Dimensions

```
M (Mass)      - amounts, dosages, weights
L (Length)    - volumes, compartments (L³ notation)
T (Time)      - rates, half-lives
```

### Derived Dimensional Types

| Type | Dimension | Julia Interpretation | MedLang Usage |
|------|-----------|---------------------|---------------|
| `Mass` | M | mg, ng, g | doses, weights, biomarkers |
| `Volume` | L³ | mL, L, dL | compartment volumes |
| `RateConst` | 1/T | h⁻¹, min⁻¹ | ka, ke, first-order rates |
| `Clearance` | L³/T | mL/min, L/h | renal, hepatic, total clearance |
| `ConcMass` | M/L³ | ng/mL, μg/dL | concentrations |
| `Exposure` | M·T/L³ | AUC (ng·h/mL) | area under curve |
| `StdDev` | dimensionless | unitless | variability parameters (ω) |
| `Exponent` | dimensionless | unitless | scaling exponents (e.g., 0.75 for allometry) |

### Creating Custom Dimensions

**Pattern**: Only use predefined types (V0 limitation). For custom dimensions, decompose into M·L·T:

```julia
# Julia: Free-form units
elimination_rate = CL / V  # Could be any dimension
```

```medlang
# MedLang: Must be a defined type
param CL : Clearance      // L³/T - explicit
param V : Volume          // L³ - explicit
// CL/V is automatically 1/T (RateConst) - checked at compile time
dA/dt = -(CL/V) * A       // typechecked composition
```

---

## ODE System Translations

### Single Compartment

**Julia**:
```julia
function one_comp!(du, u, p, t)
    ka, CL, V = p
    A_gut, A_c = u

    du[1] = -ka * A_gut
    du[2] = ka * A_gut - (CL/V) * A_c
end
```

**MedLang**:
```medlang
param ka : RateConst
param CL : Clearance
param V : Volume

dA_gut/dt = -ka * A_gut
dA_central/dt = ka * A_gut - (CL / V) * A_central
```

### Two-Compartment Distribution

**Julia**:
```julia
function two_comp!(du, u, p, t)
    ka, CL, V1, Q, V2 = p
    A_gut, A_c1, A_c2 = u

    du[1] = -ka * A_gut
    du[2] = ka * A_gut - (CL + Q) / V1 * A_c1 + Q / V2 * A_c2
    du[3] = Q / V1 * A_c1 - Q / V2 * A_c2
end
```

**MedLang**:
```medlang
param ka : RateConst
param CL : Clearance
param V_central : Volume
param Q : Clearance        // inter-compartmental clearance
param V_peripheral : Volume

dA_gut/dt = -ka * A_gut
dA_central/dt = ka * A_gut - (CL + Q)/V_central * A_central + Q/V_peripheral * A_peripheral
dA_peripheral/dt = Q/V_central * A_central - Q/V_peripheral * A_peripheral
```

**Note**: `Q` is an inter-compartmental clearance (L³/T), like CL.

### Nonlinear Metabolism

**Julia** (Michaelis-Menten):
```julia
# Michaelis-Menten rate
Vmax = p[1]
Km = p[2]
rate = (Vmax * A) / (Km + A)
du[1] = -rate * A
```

**MedLang** (same function form):
```medlang
param Vmax : Mass / Time      // mg/h - maximum rate
param Km : ConcMass           // ng/mL - Michaelis constant

// Nonlinear ODE
dA/dt = -(Vmax * A) / (Km + A/V)
```

---

## Parameter and Prior Translations

### Population Pharmacokinetics (PopPK)

**Julia** (typical Turing.jl structure):
```julia
@model function pk_model(y, dose)
    # Population level
    θ_ka ~ LogNormal(log(1.2), 0.3)
    θ_CL ~ LogNormal(log(5.0), 0.5)
    θ_V ~ LogNormal(log(50.0), 0.5)

    # Variability
    ω_ka ~ HalfNormal(0.3)
    ω_CL ~ HalfNormal(0.3)
    ω_V ~ HalfNormal(0.3)

    σ ~ HalfNormal(0.1)

    # Subject parameters (vectorized)
    η_ka ~ MvNormal(zeros(N), I)
    η_CL ~ MvNormal(zeros(N), I)
    η_V ~ MvNormal(zeros(N), I)

    ka = θ_ka .* exp.(ω_ka .* η_ka)
    CL = θ_CL .* exp.(ω_CL .* η_CL)
    V = θ_V .* exp.(ω_V .* η_V)

    # Likelihood
    ...
end
```

**MedLang** (hierarchical structure):
```medlang
// Population level (means) - estimated
param ka_pop : RateConst ~ LogNormal(0.18, 0.3)
param CL_pop : Clearance ~ LogNormal(1.61, 0.5)
param V_pop : Volume ~ LogNormal(3.91, 0.5)

// Population level (variances)
param ω_ka : StdDev ~ Exponential(1.0)      // ~ HalfNormal(0.3)
param ω_CL : StdDev ~ Exponential(1.0)      // ~ HalfNormal(0.3)
param ω_V : StdDev ~ Exponential(1.0)       // ~ HalfNormal(0.3)

// Observation error
param σ : StdDev ~ Exponential(10.0)        // ~ HalfNormal(0.1)

// Subject level (random effects) - per subject
param η_ka : RandomEffect ~ Normal(0, 1)
param η_CL : RandomEffect ~ Normal(0, 1)
param η_V : RandomEffect ~ Normal(0, 1)

// Subject parameters (derived)
param ka : RateConst = ka_pop * exp(ω_ka * η_ka)
param CL : Clearance = CL_pop * exp(ω_CL * η_CL)
param V : Volume = V_pop * exp(ω_V * η_V)
```

**Key differences**:
- Julia uses vectorized arrays (`MvNormal(zeros(N), I)`)
- MedLang uses scalar `RandomEffect` with N replicates at inference time
- Julia explicit `HalfNormal`; MedLang uses `Exponential` (equivalent via re-parameterization)

### Prior Equivalences

| Julia | MedLang | Notes |
|-------|---------|-------|
| `Normal(μ, σ)` | `~ Normal(μ, σ)` | Standard normal |
| `LogNormal(μ, σ)` | `~ LogNormal(μ, σ)` | Log scale (good for clearance/volume) |
| `Exponential(λ)` | `~ Exponential(λ)` | Right-skewed, rate parameter |
| `HalfNormal(σ)` | `~ Exponential(2/(σ√π))` | Half-normal ≈ folded exponential |
| `Uniform(a, b)` | `~ Uniform(a, b)` | Equal probability over range |
| `InverseGamma(α, β)` | `~ InverseGamma(α, β)` | For variance priors |
| `MvNormal(μ, Σ)` | Multiple univariate `~ Normal(...)` | Multivariate → independent scalars (V0) |

---

## Covariate Effects

### Allometric Scaling (Weight-Based)

**Julia**:
```julia
# Allometric scaling
WT = patient_data.weight
WT_ref = 70.0  # reference weight in kg
allom_exp = 0.75

CL = CL_pop * (WT / WT_ref)^allom_exp
V = V_pop * (WT / WT_ref)^allom_exp
```

**MedLang**:
```medlang
fixed WT : Mass = 70.0_kg         // subject weight
fixed WT_ref : Mass = 70.0_kg     // reference weight

param CL_pop : Clearance
param allom_exp : Exponent = 0.75

// Covariate-adjusted parameter
param CL : Clearance = CL_pop * (WT / WT_ref)^allom_exp
```

### Linear Covariate Effect

**Julia**:
```julia
# Covariate: AGE
AGE = patient_data.age
age_effect = 0.05  # 5% change per year
CL = CL_pop * (1 + age_effect * (AGE - 40))
```

**MedLang**:
```medlang
fixed AGE : Dimensionless      // could use custom scaling
param CL_pop : Clearance
param age_effect : Dimensionless = 0.05

param CL : Clearance = CL_pop * (1 + age_effect * (AGE - 40.0))
```

---

## Likelihood and Observation Models

### Standard PK Observation

**Julia**:
```julia
# Standard observation model
μ = [sol(t)[2] / V for t in times]  # central compartment concentration
y ~ MvNormal(μ, σ^2)                 # normal likelihood
```

**MedLang**:
```medlang
param σ : StdDev ~ Exponential(1.0)
error ~ Normal(0, σ)
likelihood y ~ Normal(A_central / V + error, σ)
```

### Log-Normal Observation

**Julia**:
```julia
# Log-normal (often better for concentration data)
μ = [sol(t)[2] / V for t in times]
y ~ MvNormal(log.(μ), σ^2)  # on log scale
```

**MedLang** (may need custom implementation):
```medlang
// V0: standard normal on linear scale
// For log-scale: would need LogNormal distribution
param σ : StdDev ~ Exponential(1.0)
error ~ Normal(0, σ)
likelihood y ~ LogNormal(log(A_central/V), σ)  // planned for future versions
```

### Mixture Model (Detection Limit)

**Julia**:
```julia
# Below detection limit (BDL) handling
for i in eachindex(y_obs)
    if y_obs[i] < BDL
        # Probability of being below limit
        y_obs[i] ~ Uniform(0, BDL)
    else
        y_obs[i] ~ Normal(μ[i], σ)
    end
end
```

**MedLang** (not yet supported in V0):
```medlang
// V0 does not support conditional likelihood
// Workaround: pre-process data, only use BDL-exceeding observations
likelihood y ~ Normal(A_central / V, σ)
```

---

## Catlab.jl → MedLang Mappings

### Morphisms (Transformations)

**Catlab**:
```julia
# Morphism: parameter transformation
struct ParameterFunctor <: Functor
    source::Category        # raw parameters
    target::Category        # derived parameters
    map::Dict               # parameter mappings
end
```

**MedLang** (via parameter definitions):
```medlang
// Morphism: raw → derived parameters
param CL_pop : Clearance ~ Normal(2.0, 0.5)   // source
param η_CL : RandomEffect ~ Normal(0, 1)      // source
param ω_CL : StdDev ~ Exponential(0.5)        // source

param CL : Clearance = CL_pop * exp(ω_CL * η_CL)  // morphism (map)
```

**Conceptually**:
- Source category: `{CL_pop, η_CL, ω_CL}` (raw parameters)
- Target category: `{CL}` (derived parameter)
- Morphism: exponential transformation

### Wiring Diagrams (Composition)

**Catlab**:
```julia
# Wiring: compose transformations
diagram = WiringDiagram([In1, In2], [Out])
add_box!(diagram, absorption_transform)
add_box!(diagram, distribution_transform)
add_box!(diagram, elimination_transform)
add_wire!(diagram, (input_id, 1) => (absorption_id, 1))
add_wire!(diagram, (absorption_id, 1) => (distribution_id, 1))
```

**MedLang** (via ODE composition):
```medlang
// Three "boxes": absorption, distribution, elimination
// Wiring: A_gut → A_central → (output)

param ka : RateConst              // absorption rate
param CL : Clearance              // elimination
param V : Volume                  // distribution

dA_gut/dt = -ka * A_gut                                    // absorption box
dA_central/dt = ka * A_gut - (CL/V) * A_central            // distribution + elim box
```

**Conceptually**:
- Box 1: absorption (input: `A_gut`, output: elimination of `A_gut`)
- Box 2: distribution + elimination (input: influx from gut, output: central compartment)
- Wires: connect `A_gut` degradation → `A_central` production

### Functors (Type Mappings)

**Catlab**:
```julia
# Functor between categories
F(obj::SourceCategory) = target_category_object(obj)
F(mor::SourceMorphism) = target_category_morphism(mor)
```

**MedLang** (dimensional analysis functor):
```medlang
// Implicit functor: raw_param → dimensional_type
// F(CL_raw : Unknown) = CL : Clearance (L³/T)
// F(V_raw : Unknown) = V : Volume (L³)

param CL : Clearance   // F(raw_CL) → Clearance
param V : Volume       // F(raw_V) → Volume

// Functorial property: F preserves composition
// If (CL : L³/T) / (V : L³) defined, result is 1/T
dA/dt = -(CL/V) * A    // F(CL/V) = 1/T, valid!
```

### Acsets (Data Structure Mappings)

**Catlab** (ACSets - attributed C-sets):
```julia
# Graph-like structure with attributes
@acset_type SimpleGraph(FreeSchema) <: AbstractGraph
@acset SimpleGraph begin
    V::Int
    E::Int
    src::Tuple(E,V)
    tgt::Tuple(E,V)
    weight::Data(Float64)
end
```

**MedLang** (parameter graph):
```medlang
// Implicit parameter dependency graph
// Nodes: parameters
// Edges: functional dependencies

param WT : Mass                              // node (independent)
param CL_pop : Clearance                    // node (independent)
param allom_exp : Exponent                  // node (independent)

param CL : Clearance = CL_pop * (WT/70)^allom_exp  // edges: CL_pop→CL, WT→CL, allom_exp→CL
```

---

## Operator and Function Mappings

### Arithmetic

| Operation | Julia | MedLang | Notes |
|-----------|-------|---------|-------|
| Addition | `a + b` | `a + b` | Same |
| Subtraction | `a - b` | `a - b` | Same |
| Multiplication | `a * b` | `a * b` | Same |
| Division | `a / b` | `a / b` | Same |
| Power | `a ^ b` | `a ^ b` | Same |
| Element-wise | `.* ./ .^` | N/A (scalars only) | MedLang V0: no arrays |

### Mathematical Functions

| Function | Julia | MedLang | Notes |
|----------|-------|---------|-------|
| Exponential | `exp(x)` | `exp(x)` | Base e |
| Natural log | `log(x)` | `log(x)` | Base e |
| Square root | `sqrt(x)` | `sqrt(x)` | Same |
| Absolute value | `abs(x)` | `abs(x)` | Same |
| Trigonometric | `sin(x), cos(x), tan(x)` | Not in V0 | Consider for future |
| Maximum | `max(a, b)` | `max(a, b)` | Min of two |
| Minimum | `min(a, b)` | `min(a, b)` | Min of two |

### Logical (in Conditional)

| Op | Julia | MedLang | Status |
|-----|-------|---------|--------|
| `==` | `a == b` | `a == b` | Comparison |
| `>`, `<` | `a > b`, `a < b` | N/A | Conditionals not in V0 |
| `&&`, `\|\|` | `a && b`, `a \|\| b` | N/A | Boolean logic not in V0 |

---

## Common Patterns and Idioms

### Conditional Expressions

**Julia**:
```julia
function pk_system!(du, u, p, t)
    # Dose enters at t=0
    if t < 0.01
        external_input = dose / (0.01)  # bolus
    else
        external_input = 0.0
    end

    du[1] = external_input - ka * u[1]
end
```

**MedLang** (V0 limitation):
```medlang
// V0 does not support conditional logic
// Workaround: use init conditions + prior structure
// Assumption: single dose at t=0

// Initial condition (implicitly in Stan/Julia backend)
dA_gut/dt = -ka * A_gut

// Dose is "fixed input" that initializes A_gut
fixed DOSE : Mass
// Backend initialization: A_gut[t=0] = DOSE
```

### Event Handling

**Julia** (discrete events):
```julia
# Event: dose at specific times
callback = PresetTimeCallback([8, 16], affect!)

function affect!(integrator)
    integrator.u[1] += dose  # add dose to gut
end

prob = ODEProblem(..., callback=callback)
```

**MedLang** (V0 limitation):
```medlang
// V0 does not support multiple dose events
// Workaround: model as input rate (for continuous infusions)

param infusion_rate : Clearance   // input rate (L³/T) × concentration
dA_gut/dt = infusion_rate * V - ka * A_gut
```

### Loop Constructs

**Julia** (vectorized operations):
```julia
for i in 1:n_subjects
    CL[i] = CL_pop * exp(ω_CL * η_CL[i])
    V[i] = V_pop * exp(ω_V * η_V[i])
end

# Solve ODE for each subject
for i in 1:n_subjects
    sol[i] = solve(ODEProblem(..., p=[CL[i], V[i]]), ...)
end
```

**MedLang** (scalar parameters, expanded at inference):
```medlang
// Scalar definitions
param CL : Clearance = CL_pop * exp(ω_CL * η_CL)
param V : Volume = V_pop * exp(ω_V * η_V)

dA_gut/dt = -ka * A_gut
dA_central/dt = ka * A_gut - (CL/V) * A_central

// Backend (Stan/Julia) automatically replicates for all subjects
```

---

## Testing and Validation

### Numerical Comparison

After conversion, verify numeric equivalence:

```bash
# Step 1: Run Julia model
julia> sol_julia = solve(prob, Tsit5(), saveat=times)

# Step 2: Compile and run MedLang
$ mlc compile model.medlang
$ mlc generate-data -n 1 -o test_data.csv
$ mlc run model.stan --data test_data.json

# Step 3: Compare predictions
# If E[y_julia] ≈ E[y_medlang] (within rounding), conversion successful
```

### Unit Checking

Verify dimensional consistency:

```bash
$ mlc check model.medlang -v

# Example output:
# Variable A_central: M (mass) ✓
# Variable CL/V: (L³/T) / L³ = 1/T ✓
# ODE dA/dt = -(CL/V)*A:
#   Left: M/T ✓
#   Right: (1/T) * M = M/T ✓
```

### Type Safety

MedLang catches unit mismatches at compile time:

```medlang
// Invalid: incompatible dimensions
param k : RateConst
param V : Volume
dA/dt = k * V * A          // ERROR: (1/T) * (L³) * (M) = M·L³/T (wrong!)
```

```medlang
// Correct
dA/dt = k * A              // (1/T) * (M) = M/T ✓
```

---

## Troubleshooting

### Problem: "Unknown type X"

**Symptom**: `mlc check` fails with "unknown type"

**Cause**: Using a type not defined in MedLang's M·L·T system

**Solution**: Decompose into base or derived types:
```medlang
// ❌ Wrong
param rate : BioClearance

// ✓ Right
param rate : Clearance    // L³/T (use standard type)
```

### Problem: "Dimensional mismatch"

**Symptom**: Type checker rejects ODE

**Cause**: Left and right sides have different dimensions

**Solution**: Check coefficient dimensions:
```medlang
// ❌ Wrong
param k : RateConst       // 1/T
dA/dt = k * V * A         // (1/T) * (L³) * (M) = not M/T!

// ✓ Right
dA/dt = k * A             // (1/T) * (M) = M/T ✓
```

### Problem: "Undefined variable X"

**Symptom**: Compilation fails, "variable X not found"

**Cause**: Using array index instead of named variable

**Solution**: Replace indices with explicit names:
```medlang
// ❌ Wrong (leftover Julia syntax)
dA[1]/dt = -k * A[2]

// ✓ Right (MedLang d-notation)
dA_central/dt = -k * A_peripheral
```

---

## Advanced: Custom Type Extensions (Future)

MedLang V0 has fixed types. V1 will support user-defined dimensions:

```medlang
// Planned for V1:
dimension MyDimension = M^2 * L^-1 * T^3
param custom : MyDimension = 5.0
```

Until then, use existing M·L·T types and compose dimensionally.
