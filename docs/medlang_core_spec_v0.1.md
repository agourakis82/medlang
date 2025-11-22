# MedLang Core Specification — Version 0.1 (Draft)

**Status:** Draft  
**Scope:** Core calculus, type system, and operational semantics for MedLang's low‑level and clinical core.  
**Non‑scope (for this document):** Full quantum-chemistry extension details, full AI model zoo, full Beagle integration. Those will be layered on top of this core.

---

## 1. Design Goals

MedLang is a domain-specific language for computational medicine with the following goals:

1. **Clinical nativeness.**  
   Patients, cohorts, timelines, doses, and protocols are first‑class entities.

2. **Hardware awareness.**  
   The language compiles efficiently to CPU/GPU/HPC backends with explicit device semantics and parallel execution.

3. **Type and dimensional safety.**  
   Units (mg, μg/kg/min, mmHg, etc.) are part of the type system. Well‑typed programs cannot silently mis-handle units.

4. **Mathematical expressiveness.**  
   Tensors, probability measures, kernels, and (later) quantum and fractal operators are primitives, not mere library conventions.

5. **Determinism by default.**  
   Execution is deterministic unless probabilistic or non-deterministic constructs are explicitly used.

6. **Foundation for extensions.**  
   This core serves as the basis for:
   - Clinical DSL (L₂),
   - Quantum pharmacology,
   - AI models (MLP/GNN/PINN),
   - Fractal analysis,
   - Probabilistic programming.

---

## 2. Core Concepts

MedLang core defines:

- **Expressions**: computations that evaluate to values.
- **Values**: fully evaluated entities (numbers, quantities, tensors, patients, etc.).
- **Types**: classify values and expressions.
- **Devices**: abstract over hardware (e.g. CPU, GPU(0)).
- **Layouts**: describe memory layout (e.g. AoS, SoA).
- **Store/State**: runtime memory for values, per device, plus logs and RNG state.

We use:

- Γ for typing context (variable → type).
- Σ for global signatures (structs, models, functions).
- σ for runtime state.

Typing judgment: `Γ ⊢ e : τ`  
Evaluation judgment: `⟨e, σ⟩ → ⟨e', σ'⟩` (small‑step).

---

## 3. Syntax (Informal Core Grammar)

This is an informal abstract syntax; concrete surface syntax can vary slightly.

### 3.1. Identifiers, literals, units

- Variables: `x`, `p`, `cohort1`, `dt`
- Type identifiers: `Patient`, `Cohort`, `Tensor`, `Quantity`
- Units (examples): `kg`, `m`, `s`, `h`, `mg`, `μg`, `mmHg`, etc.
- Numeric literals: `42`, `3.14`, `0.05`
- Literal quantities: `82 kg`, `65 mmHg`, `0.05 μg/kg/min`

### 3.2. Core expression forms

```bnf
e ::= x                              -- variable
    | c                              -- constant (numeric, boolean, quantity literal)
    | λ x : τ . e                    -- function abstraction
    | e1 e2                          -- application
    | let x : τ = e1 in e2           -- binding
    | { ℓ1 = e1, …, ℓn = en }        -- struct literal
    | e.ℓ                            -- field projection
    | if e then e1 else e2           -- conditional
    | e1 ⊕ e2                        -- arithmetic op (+, -, *, /)
    | tensor[e1, …, ek]              -- tensor literal
    | e[i]                           -- tensor indexing (abstract; IR-level)
    | patient{ ℓ1 = e1, … }          -- patient literal
    | cohort[e1, …, en]              -- cohort literal
    | timeline_dense(t0, dt, e)      -- dense timeline
    | timeline_sparse(e)             -- sparse timeline
    | e(t)                           -- timeline access by time
    | par_map(f, e)                  -- parallel map over collection
    | parallel_for x in e do e_body  -- high-level parallel loop (sugared to par_map)
    | simulate(e_protocol, e_cohort, e_horizon, e_dt)
    | measure_op(e)                  -- probabilistic, fractal, or QM operators (core slots)
```

We will later desugar high-level constructs like `parallel for` into `par_map` plus environment handling.

---

## 4. Types

We define the set of types τ by the following grammar.

### 4.1. Base and scalar types

```text
τ ::= i32 | f32 | f64 | bool
```

### 4.2. Units and quantities

We assume a kind `Unit` which forms a free abelian group over base units (kg, m, s, ...). A quantity is a scalar with units:

```text
τ ::= Quantity<u, τ_scalar>
```

where:
- `u` is a unit expression (e.g. `mg`, `μg/kg/min`),
- `τ_scalar ∈ {f32, f64}`.

### 4.3. Devices and layouts

We assume:

```text
Device ::= CPU | GPU(n) | TPU(n) | ...
Layout ::= AoS | SoA
```

These appear as type parameters.

### 4.4. Tensors

```text
τ ::= Tensor<R, τ_elem, Device d, Layout L>
```

- `R ∈ ℕ` is rank.
- `τ_elem` is an element type (e.g. `f32`, `Quantity<mmHg,f32>`).

### 4.5. References (ownership/borrowing)

```text
τ ::= &imm τ'        -- shared immutable reference
    | &mut τ'        -- unique mutable reference
```

Certain types are *resources* (e.g. large buffers/cohorts) and subject to linear/affine rules.

### 4.6. Clinical types

Minimal core:

```text
τ ::= Patient
    | Cohort<Patient, Layout L, Device d>
    | Timeline<τ_elem>
```

`Patient` is a globally defined struct type:

```text
struct Patient {
  id         : PatientId
  birth_date : Date
  sex        : Sex
  weight     : Quantity<kg, f32>
  height     : Quantity<m, f32>
  -- plus domain-specific fields
}
```

For this core spec, we treat `Patient` as an opaque record type with known field types defined in Σ.

### 4.7. Probabilistic types

We introduce measures and kernels:

```text
τ ::= Measure<τ_x>
    | ProbKernel<τ_x, τ_y>
```

Intuition:
- `Measure<X>` = probability distribution over space `X`.
- `ProbKernel<X,Y>` = Markov kernel from X to distributions over Y.

### 4.8. Model type (for AI/ML)

```text
τ ::= Model<τ_x, τ_y>
```

`Model<X,Y>` represents a parameterised operator from X to Y, with associated parameter space and training semantics (formalised later).

### 4.9. Quantum and fractal (stubs in core)

For the core spec we define placeholders:

```text
τ ::= QState(n)      -- finite-dimensional quantum state (ℂ^n), normalized
    | QOp(n)         -- linear operator on ℂ^n
    | QObs(n)        -- observable on ℂ^n
    | FractalDimension
    | MultifractalSpectrum
```

Details of quantum and fractal operators are in separate extension specs; here they are simply types the core can track.

---

## 5. Typing Rules (Selected)

We write the typing judgment as `Γ ⊢ e : τ`.

### 5.1. Variables and constants

```text
(T-Var)
 Γ(x) = τ
────────────
 Γ ⊢ x : τ

(T-Const-F32)
──────────────
 Γ ⊢ n_f : f32

(T-Const-Bool)
───────────────
 Γ ⊢ b : bool
```

### 5.2. Quantities and unit safety

Construction:

```text
(T-Quant)
 Γ ⊢ e : τ_s    τ_s ∈ {f32, f64}
────────────────────────────────
 Γ ⊢ quantity(e, u) : Quantity<u, τ_s>
```

Addition (same unit):

```text
(T-QAdd)
 Γ ⊢ e1 : Quantity<u, τ_s>    Γ ⊢ e2 : Quantity<u, τ_s>
────────────────────────────────────────────────────────
      Γ ⊢ e1 + e2 : Quantity<u, τ_s>
```

Multiplication (units compose):

```text
(T-QMul)
 Γ ⊢ e1 : Quantity<u1, τ_s>    Γ ⊢ e2 : Quantity<u2, τ_s>
──────────────────────────────────────────────────────────
 Γ ⊢ e1 * e2 : Quantity<u1 * u2, τ_s>
```

Where `u1 * u2` is unit multiplication at the type level (adding exponents). Addition between incompatible units does not type‑check.

### 5.3. Functions

```text
(T-Abs)
 Γ, x : τ1 ⊢ e : τ2
───────────────────────
 Γ ⊢ λ x : τ1 . e : τ1 → τ2

(T-App)
 Γ ⊢ e1 : τ1 → τ2    Γ ⊢ e2 : τ1
──────────────────────────────────
        Γ ⊢ e1 e2 : τ2
```

### 5.4. Structs and projections

Given `struct S { ℓ1: τ1, …, ℓn: τn }` in Σ:

```text
(T-Struct-Intro)
 Γ ⊢ e1 : τ1  …  Γ ⊢ en : τn
──────────────────────────────────────
 Γ ⊢ { ℓ1 = e1, …, ℓn = en } : S

(T-Struct-Elim)
 Γ ⊢ e : S    Σ(S)(ℓ) = τ
────────────────────────────
       Γ ⊢ e.ℓ : τ
```

`Patient` is treated as a `struct Patient { ... }` under Σ.

### 5.5. Tensors

We treat tensor operations at a high level:

```text
(T-Tensor-Intro)
 Γ ⊢ e1 : τ   …   Γ ⊢ ek : τ
───────────────────────────────
 Γ ⊢ tensor[e1,…,ek] : Tensor<1, τ, d, L>       -- d,L may have defaults

(T-Tensor-Index)
 Γ ⊢ e : Tensor<R, τ, d, L>    Γ ⊢ i : i32
────────────────────────────────────────────
 Γ ⊢ e[i] : Tensor<R-1, τ, d, L>  (R ≥ 1)
```

More precise shape checking can be modelled in an extended system; core spec abstracts that away.

### 5.6. Clinical types: `Patient`, `Cohort`, `Timeline`

Cohort:

```text
(T-Cohort-Intro)
 Γ ⊢ e1 : Patient  …  Γ ⊢ en : Patient
───────────────────────────────────────────────
 Γ ⊢ cohort[e1,…,en] : Cohort<Patient, L, d>       -- L,d may be inferred
```

Parallel map over cohort:

```text
(T-ParMap)
 Γ ⊢ f : Patient → τ_r
 Γ ⊢ C : Cohort<Patient, L, d>
─────────────────────────────────────────────
 Γ ⊢ par_map(f, C) : Tensor<1, τ_r, d, L_out>
```

(Where `L_out` may be inferred from implementation; in simplest form we can treat the result as a 1D tensor.)

Timeline:

```text
(T-TLDense)
 Γ ⊢ t0 : TimeInstant
 Γ ⊢ dt : Duration
 Γ ⊢ v  : Tensor<1, τ, d, L>
──────────────────────────────────────────────
 Γ ⊢ timeline_dense(t0, dt, v) : Timeline<τ>

(T-TL-At)
 Γ ⊢ tl : Timeline<τ>
 Γ ⊢ t  : TimeInstant
─────────────────────────────
 Γ ⊢ tl(t) : τ
```

### 5.7. Probabilistic types

```text
(T-Measure-Intro)
 Γ ⊢ support : Tensor<1, τ_x, d, L>
 Γ ⊢ weights : Tensor<1, f32, d, L>
─────────────────────────────────────
 Γ ⊢ measure(support, weights) : Measure<τ_x>

(T-ProbKernel-Apply)
 Γ ⊢ k : ProbKernel<τ_x, τ_y>
 Γ ⊢ x : τ_x
────────────────────────────────────────
 Γ ⊢ k(x) : Measure<τ_y>

(T-Pushforward)
 Γ ⊢ k  : ProbKernel<τ_x, τ_y>
 Γ ⊢ μx : Measure<τ_x>
────────────────────────────────────────────
 Γ ⊢ pushforward(k, μx) : Measure<τ_y>
```

### 5.8. Model application

```text
(T-Model-App)
 Γ ⊢ m : Model<τ_x, τ_y>
 Γ ⊢ x : τ_x
────────────────────────────
 Γ ⊢ m(x) : τ_y
```

Training and optimization are not part of the basic typing judgment—they are separate meta‑operations that manipulate models and parameter spaces. We only require that the resulting trained model is still of type `Model<τ_x, τ_y>`.

---

## 6. Operational Semantics (Core)

We define a small-step semantics `⟨e, σ⟩ → ⟨e', σ'⟩`.

### 6.1. Values

We assume a standard set of values v:

- Numeric: `n`, `true`, `false`.
- Quantity values: `q = (n, u)`.
- Functions: `λ x : τ . e`.
- Structs: `{ ℓ1 = v1, …, ℓn = vn }`.
- Patients: `patient{…}` as a struct value.
- Cohorts: `cohort[v1,…,vn]`.
- Tensors: tuples of values, or device‑allocated references in σ.
- Timelines: `timeline_dense(t0, dt, v)` where v is a tensor value, etc.
- Measures: encoded as `measure(support, weights)` values.
- Models: compiled/opaque values with a known `forward` implementation.

### 6.2. Evaluation strategy

MedLang core uses a **call‑by‑value** semantics:

- Evaluate function argument before application.
- Evaluate subexpressions left‑to‑right.

Examples:

```text
(E-App1)
⟨e1, σ⟩ → ⟨e1', σ'⟩
─────────────────────────
⟨e1 e2, σ⟩ → ⟨e1' e2, σ'⟩

(E-App2)
v1 is a value
⟨e2, σ⟩ → ⟨e2', σ'⟩
─────────────────────────
⟨v1 e2, σ⟩ → ⟨v1 e2', σ'⟩

(E-AppAbs)
v2 is a value
────────────────────────────────────
⟨(λ x : τ . e) v2, σ⟩ → ⟨e[x := v2], σ⟩
```

Arithmetic (ignoring units for brevity):

```text
(E-Add)
⟨n1 + n2, σ⟩ → ⟨n, σ⟩      where n = n1 + n2 (numeric addition)
```

With units, evaluation rules must respect unit compatibility (type system guarantees that). Evaluation becomes arithmetic on `(value, unit)` pairs.

### 6.3. Parallel constructs

We model `par_map(f, C)` at a high level as an atomic step from the core's point of view, delegated to the runtime:

```text
(E-ParMap)
v_f is a function value
v_C is a cohort value
σ' = run_parallel_map(v_f, v_C, σ)
─────────────────────────────────────────────
⟨par_map(v_f, v_C), σ⟩ → ⟨v_out, σ'⟩
```

where:
- `run_parallel_map` spawns SPMD-style tasks on appropriate devices,
- `v_out` is a tensor or cohort result,
- σ' includes updated device memory and logs.

A more detailed semantic account could model threads explicitly; for the core we assume atomic parallel combinators.

### 6.4. Timeline evaluation

```text
(E-TL-At-Dense)
tl = timeline_dense(t0, dt, v_tensor)
v_t is a time instant
index = floor((v_t - t0) / dt)
───────────────────────────────────────────────
⟨tl(v_t), σ⟩ → ⟨v_tensor[index], σ⟩
```

Boundary conditions (out-of-range times) are handled via defined conventions (clamp, error, or special value), specified at the language level.

### 6.5. Probabilistic operators and determinism

We conceptually distinguish two modes:

- **Deterministic mode**: probabilistic operators are disabled or return abstract measures without sampling.
- **Sampling mode**: `sample(μ)` draws from μ using RNG in σ.

Example:

```text
(E-Sample)
μ is a measure value
(r, σ') = draw(μ, σ)          -- using RNG in σ
────────────────────────────────────────────
⟨sample(μ), σ⟩ → ⟨r, σ'⟩
```

Determinism is controlled by:
- global seeds,
- recording of random choices when needed (for replay),
- configuration flags on the runtime.

---

## 7. Safety Properties (Stated)

This section states the desired theorems; full proofs are left for future work.

### 7.1. Type safety

**Theorem (Preservation).**  
If `Γ ⊢ e : τ` and `⟨e, σ⟩ → ⟨e', σ'⟩`, then `Γ ⊢ e' : τ`.

**Theorem (Progress).**  
If `Γ ⊢ e : τ` and e is closed (no free variables), then either:
- e is a value, or
- there exist `e', σ'` such that `⟨e, σ⟩ → ⟨e', σ'⟩`.

Together they ensure well‑typed programs do not "get stuck".

### 7.2. Dimensional safety

**Theorem (Dimensional correctness).**  
If `Γ ⊢ e : Quantity<u, τ_s>`, no reduction step can produce a quantity with a different unit `u' ≠ u` unless there is an explicit, well‑typed conversion operator applied.

### 7.3. Race‑freedom (sketch)

For a restricted parallel subset using `par_map`/`parallel_for` over cohorts/tensors, and under the discipline that each iteration writes only to its own index, we expect:

**Proposition (Local race‑freedom).**  
Well‑typed uses of `par_map` that satisfy the SPMD index discipline cannot cause write‑write data races in the shared store.

A formal statement requires:
- a definition of SPMD index discipline,
- a model of shared memory writes,
- and proof that indices do not alias.

---

## 8. Examples

### 8.1. Simple clinical computation

Compute mean MAP over a cohort on GPU:

```medlang
device gpu0

fn mean_MAP(c: Cohort<Patient, SoA, gpu0>) : Quantity<mmHg, f32> {
    let maps = par_map(λ p : Patient . p.MAP(t_now), c)
    return mean(maps)    -- mean is unit-aware
}
```

Typing (informal):
- `Γ ⊢ c : Cohort<Patient, SoA, GPU(0)>`
- `Γ, p:Patient ⊢ p.MAP(t_now) : Quantity<mmHg, f32>`
- `par_map(λp. p.MAP(t_now), c) : Tensor<1, Quantity<mmHg, f32>, GPU(0), SoA>`
- `mean : Tensor<1, Quantity<mmHg,f32>, d, L> → Quantity<mmHg,f32>`
- So `Γ ⊢ mean_MAP(c) : Quantity<mmHg, f32>`.

### 8.2. Connecting to a model

```medlang
model MORTALITY_30D : Model[ClinicalFeatures, Probability] @gpu { ... }

fn risk_for_patient(p: Patient) : Probability {
    let x = features_from_patient(p)
    return MORTALITY_30D(x)
}
```

Typing:
- `MORTALITY_30D : Model<ClinicalFeatures, Probability>`
- `features_from_patient : Patient → ClinicalFeatures`
- Thus `risk_for_patient : Patient → Probability`.

---

## 9. Extension Hooks

The core specification defines:

- common types (`Quantity`, `Tensor`, `Cohort`, `Timeline`, `Model`, `Measure`, `ProbKernel`),
- typing principles,
- and small‑step semantics.

The following are explicit extension hooks:

1. **Quantum extension (MedLang-QM).**  
   Provides concrete typing and semantics for `Molecule`, `ElectronicState`, `QState`, `QOp`, `QObs`, and QM operators (HF, DFT, etc.).

2. **AI extension (MedLang-AI).**  
   Defines concrete model constructors:
   - `MLP`, `Conv2D`, `GNNLayer`, `PINN`, etc.  
     and attaches automatic differentiation and optimization semantics.

3. **Fractal extension (MedLang-FRA).**  
   Specifies semantics and algorithms for `FractalDimension`, `MultifractalSpectrum`, and their operators over timelines and trajectories.

4. **Clinical DSL (MedLang-L₂).**  
   Adds syntax sugar:
   - `protocol`, `simulate`, `inclusion`, `endpoint`, etc.  
     which desugar to the core constructs specified here.

---

This is a **v0.1** core spec: it's intentionally incomplete but structurally coherent.  
Next steps we can do:

- Tighten the formal typing rules (especially for references, devices, and layouts).
- Define a minimal concrete syntax (so we can start parsing).
- Start a matching **IR design document** (CIR/NIR) aligned with this core.
