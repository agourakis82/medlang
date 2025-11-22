# MedLang Core Language Specification v0.1

## 1. Design Goals

MedLang is designed to be:

1. **Medical-native**: First-class support for clinical entities (patients, cohorts, protocols, doses, timelines)
2. **Quantum-aware**: Native operators for quantum chemistry (HF/DFT/QM/MM)
3. **AI-first**: Built-in autodiff, neural networks, and optimization
4. **Hardware-transparent**: Explicit device mapping (CPU/GPU/HPC) without low-level details
5. **Safe**: Type system enforces units, ownership, and determinism
6. **Reproducible**: Explicit control of randomness and approximation

## 2. Core Concepts

### 2.1 First-Class Clinical Types

- **Patient**: Individual with timeline, states, interventions
- **Cohort**: Collections of patients with shared properties
- **Protocol**: Structured treatment/research plans
- **Dose**: Quantities with units and safety constraints
- **Timeline**: Temporal sequences of states and events
- **Endpoint**: Measurable clinical outcomes

### 2.2 First-Class Computational Types

- **Tensor**: Multi-dimensional arrays with device placement
- **Measure**: Probability distributions
- **Fractal**: Objects with fractal dimension
- **QState**: Quantum mechanical states
- **Model**: Parameterized functions (neural nets, etc.)

### 2.3 Execution Model

- **Deterministic by default**: All operations are reproducible
- **Explicit randomness**: Random operations require explicit seeds
- **Device-aware**: Code specifies CPU/GPU placement
- **Safe parallelism**: Ownership system prevents data races

## 3. Syntax Overview

### 3.1 Basic Syntax

```
// Variable binding
let x: Type = expr

// Function definition
fn name(param: Type) -> RetType {
    body
}

// Type annotations
expr : Type

// Device placement
@device(GPU) expr

// Deterministic block
deterministic {
    body
}

// Random block with explicit seed
random(seed: u64) {
    body
}
```

### 3.2 Clinical Syntax

```
// Patient definition
patient p {
    id: PatientID,
    timeline: Timeline<State>,
    metadata: Record
}

// Cohort definition
cohort c = patients.filter(|p| condition(p))

// Protocol definition
protocol treat {
    entry: Criteria,
    steps: [Intervention],
    endpoints: [Endpoint]
}

// Dose with units
let dose = 100.mg / kg
```

### 3.3 Computational Syntax

```
// Tensor with device placement
let t: Tensor<f32, [10, 20]> @device(GPU) = ...

// Probability measure
let mu: Measure<State> = gaussian(mean, cov)

// Quantum state
let psi: QState = molecule.groundState(method: HF)

// Neural network
let model: Model<Input, Output> = MLP([784, 128, 10])
```

## 4. Type System

### 4.1 Base Types

- **Numeric**: `i32`, `i64`, `f32`, `f64`, `complex64`, `complex128`
- **Boolean**: `bool`
- **Unit**: `()` (unit type)
- **String**: `str`

### 4.2 Composite Types

- **Tuple**: `(T1, T2, ..., Tn)`
- **Array**: `[T; n]` (fixed size)
- **Vector**: `Vec<T>` (dynamic size)
- **Record**: `{field1: T1, field2: T2, ...}`
- **Sum**: `enum E { Variant1(T1), Variant2(T2), ... }`

### 4.3 Tensor Types

- **Tensor**: `Tensor<dtype, shape>`
  - `dtype`: element type (`f32`, `f64`, `complex64`, etc.)
  - `shape`: compile-time or runtime shape `[d1, d2, ...]`

### 4.4 Unit Types

- **Dimensional**: Doses, concentrations, etc. with unit checking
- **Syntax**: `Quantity<T, Unit>`
  - Example: `Dose = Quantity<f64, mg/kg>`

### 4.5 Ownership and Borrowing

- **Owned**: `T` (unique ownership)
- **Borrowed**: `&T` (shared reference)
- **Mutable borrow**: `&mut T` (exclusive reference)

Rules:
- At most one mutable borrow at a time
- No mutable borrows while shared borrows exist
- Enforces memory safety and prevents data races

### 4.6 Device Types

- **Device annotation**: `@device(D)` where `D` is `CPU`, `GPU`, or device ID
- Transfers must be explicit: `tensor.to(GPU)`

### 4.7 Clinical Types

- **Patient**: `Patient<S>` where `S` is state type
- **Cohort**: `Cohort<P>` where `P` is patient type
- **Timeline**: `Timeline<T>` where `T` is event/state type
- **Protocol**: `Protocol<I, E>` where `I` is intervention, `E` is endpoint

### 4.8 Quantum Types

- **QState**: `QState<Basis>` — quantum state in given basis
- **Operator**: `Operator<A, B>` — quantum operator from basis A to B
- **Molecule**: `Molecule` — molecular structure

### 4.9 Probabilistic Types

- **Measure**: `Measure<T>` — probability distribution over `T`
- **Random**: `Random<T, Seed>` — random variable with explicit seed

### 4.10 Model Types

- **Model**: `Model<X, Y>` — parameterized function from `X` to `Y`
- **Differentiable**: Models support automatic differentiation

## 5. Typing Rules (Informal)

### 5.1 Basic Rules

```
Γ ⊢ x: T    if (x: T) ∈ Γ

Γ, x: T1 ⊢ e: T2
─────────────────────
Γ ⊢ (fn(x: T1) -> T2 { e }): T1 -> T2

Γ ⊢ f: T1 -> T2    Γ ⊢ e: T1
─────────────────────────────
Γ ⊢ f(e): T2
```

### 5.2 Tensor Rules

```
Γ ⊢ e: Tensor<T, [d1, ..., dn]>    op: binary arithmetic
Γ ⊢ e': Tensor<T, [d1, ..., dn]>
─────────────────────────────────────────────────────────
Γ ⊢ e op e': Tensor<T, [d1, ..., dn]>

(shape broadcasting rules apply where appropriate)
```

### 5.3 Unit Rules

```
Γ ⊢ e1: Quantity<T, U1>    Γ ⊢ e2: Quantity<T, U2>
───────────────────────────────────────────────────
Γ ⊢ e1 + e2: Quantity<T, U1>    [requires U1 = U2]

Γ ⊢ e1: Quantity<T, U1>    Γ ⊢ e2: Quantity<T, U2>
───────────────────────────────────────────────────
Γ ⊢ e1 * e2: Quantity<T, U1 * U2>

Γ ⊢ e1: Quantity<T, U1>    Γ ⊢ e2: Quantity<T, U2>
───────────────────────────────────────────────────
Γ ⊢ e1 / e2: Quantity<T, U1 / U2>
```

### 5.4 Ownership Rules

```
Γ ⊢ x: T    x not borrowed
───────────────────────────
Γ' ⊢ move(x): T    [Γ' = Γ \ {x}]

Γ ⊢ x: T
─────────────────
Γ ⊢ &x: &T

Γ ⊢ x: T    x not borrowed
───────────────────────────
Γ ⊢ &mut x: &mut T
```

### 5.5 Device Rules

```
Γ ⊢ e: Tensor<T, S> @device(D1)
────────────────────────────────────
Γ ⊢ e.to(D2): Tensor<T, S> @device(D2)

Γ ⊢ e1: Tensor<T, S> @device(D)
Γ ⊢ e2: Tensor<T, S> @device(D)
────────────────────────────────────
Γ ⊢ e1 op e2: Tensor<T, S> @device(D)

[Operations on tensors from different devices require explicit transfer]
```

## 6. Operational Semantics (Informal)

### 6.1 Evaluation Contexts

```
E ::= []                    (hole)
    | E + e                 (left add)
    | v + E                 (right add)
    | E(e)                  (function application)
    | let x = E in e        (let binding)
    | ...
```

### 6.2 Reduction Rules

```
(fn(x: T) -> T' { e })(v) ⟶ e[v/x]

let x = v in e ⟶ e[v/x]

v1 + v2 ⟶ v3    where v3 = numeric_add(v1, v2)

deterministic { e } ⟶ e[seed := 0]

random(s) { e } ⟶ e[seed := s]
```

### 6.3 Device Semantics

```
@device(D) { e } ⟶ run_on_device(D, e)

tensor.to(D2) where tensor@D1 ⟶ device_transfer(tensor, D1, D2)
```

### 6.4 Quantum Semantics

```
molecule.groundState(HF) ⟶ run_hf_calculation(molecule)

psi.energy() ⟶ ⟨psi | H | psi⟩

operator(psi) ⟶ apply_operator(operator, psi)
```

## 7. Safety Properties

### 7.1 Type Safety

**Theorem (Type Soundness)**: If `Γ ⊢ e: T` and `e ⟶* v`, then `v: T`.

*(Progress and Preservation theorems to be formalized)*

### 7.2 Memory Safety

**Theorem (No Data Races)**: Well-typed programs with ownership checking do not have data races.

- Ownership system ensures unique mutable access or multiple read-only access
- No concurrent mutable and read-only access to same location

### 7.3 Unit Safety

**Theorem (Dimensional Correctness)**: Operations on quantities with incompatible units are rejected at compile time.

- Addition/subtraction require identical units
- Multiplication/division compose units correctly

### 7.4 Determinism

**Theorem (Reproducibility)**: In `deterministic` blocks, all operations produce identical results across runs.

- No non-deterministic operations allowed
- Random operations require explicit seeds

## 8. Examples

### 8.1 Basic Tensor Computation

```
fn matrix_multiply(a: Tensor<f32, [M, K]>, 
                   b: Tensor<f32, [K, N]>) 
                   -> Tensor<f32, [M, N]> {
    a @ b  // matrix multiplication
}

let x: Tensor<f32, [10, 20]> @device(GPU) = random_normal([10, 20])
let y: Tensor<f32, [20, 5]> @device(GPU) = random_normal([20, 5])
let z = matrix_multiply(x, y)  // result on GPU
```

### 8.2 Clinical Dose Calculation

```
fn calculate_dose(weight: Quantity<f64, kg>, 
                  dose_per_kg: Quantity<f64, mg/kg>) 
                  -> Quantity<f64, mg> {
    weight * dose_per_kg
}

let patient_weight = 70.kg
let dosage = 5.mg / kg
let total_dose = calculate_dose(patient_weight, dosage)  // 350 mg
```

### 8.3 Cohort Analysis

```
cohort icu_patients = all_patients.filter(|p| {
    p.location == ICU && p.severity > 3
})

let outcomes = icu_patients.map(|p| {
    endpoint_reached(p, mortality_28d)
})

let survival_rate = outcomes.filter(|o| !o).count() / outcomes.len()
```

### 8.4 Quantum Chemistry

```
let molecule = Molecule::from_smiles("CCO")  // ethanol
let psi = molecule.groundState(method: HF, basis: "6-31G*")
let energy = psi.energy()  // Hartree-Fock energy

// Transition to DFT
let psi_dft = molecule.groundState(method: DFT(functional: B3LYP))
let energy_dft = psi_dft.energy()
```

### 8.5 Neural Network Training

```
let model: Model<Tensor<f32, [784]>, Tensor<f32, [10]>> = 
    MLP([784, 128, 64, 10])

let optimizer = Adam(lr: 0.001)

deterministic {
    for batch in train_data {
        let (x, y) = batch
        let pred = model(x)
        let loss = cross_entropy(pred, y)
        
        let grads = loss.backward()
        optimizer.step(model, grads)
    }
}
```

### 8.6 PK/PD Model

```
fn one_compartment_pk(dose: Dose, 
                      volume: Volume, 
                      clearance: Clearance,
                      time: Time) -> Concentration {
    let k_el = clearance / volume
    let c_0 = dose / volume
    c_0 * exp(-k_el * time)
}

let patient_dose = 500.mg
let patient_vd = 40.L
let patient_cl = 5.L / hr

let concentration_at_4h = one_compartment_pk(
    patient_dose, patient_vd, patient_cl, 4.hr
)
```

## 9. Extension Hooks

The core specification provides hooks for extensions:

### 9.1 Quantum Extensions

- Additional QM methods (CCSD, MP2, etc.)
- QM/MM hybrid calculations
- Excited states and spectroscopy

### 9.2 AI Extensions

- Additional model architectures (GNN, Transformers, etc.)
- Advanced optimizers (natural gradient, Riemannian)
- Physics-informed neural networks (PINNs)

### 9.3 Fractal Extensions

- Fractal dimension operators
- Multifractal analysis
- Fractal-based optimization

### 9.4 Clinical Extensions

- FHIR integration
- Clinical decision support rules
- Protocol verification and validation

### 9.5 Probabilistic Extensions

- Advanced probability measures (Lévy, Dirichlet processes)
- Bayesian inference primitives
- Causal inference operators

## 10. Future Directions

### 10.1 Formal Verification

- Mechanized proofs in Coq/Lean
- Verified compiler implementation
- Certified clinical protocols

### 10.2 Intermediate Representations

- **CIR** (Clinical IR): High-level clinical reasoning
- **NIR** (Numeric IR): Tensor/array computations
- Lowering to MLIR → LLVM

### 10.3 Runtime System

- Device management and scheduling
- Memory allocation strategies
- Quantum backend integration

### 10.4 Tooling

- IDE support (LSP)
- Debugger with clinical context
- Profiler and optimizer

---

*This specification is a living document and will evolve as the language design matures.*
