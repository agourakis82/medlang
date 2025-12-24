# MedLang Vertical Slice 1 — Real Population Inference Engine

**Prerequisites:** Vertical Slice 0 (MVP-0) must be complete  
**Target:** Full Bayesian/NLME inference with Stan/Turing backend  
**Status:** Ready to implement after V0  
**Estimated effort:** 2-3 weeks

---

## Mission

**Vertical Slice 1** transforms the V0 toy compiler into a **real population inference engine** by:

1. **Upgrading random effects** from simplified treatment to full MVNormal with covariance
2. **Compiling to probabilistic backends** (Stan or Julia+Turing)
3. **Enabling real inference** (Bayesian HMC or frequentist FOCE/SAEM)

The model remains **1-compartment oral PK**, but now with **production-quality population modeling**.

---

## Context: What Changed Since V0?

### V0 (MVP-0) Had:
- ✅ Basic AST + type system with units
- ✅ Parser for minimal MedLang syntax
- ✅ CIR (Clinical IR) internal representation
- ✅ Runtime that can simulate + compute log-likelihood
- ✅ Simplified random effects treatment (diagonal, no full covariance)

### V1 Will Add:
- 🆕 **Explicit random effects structure in IR** (MVNormal with full Ω)
- 🆕 **Probabilistic backend codegen** (Stan or Turing)
- 🆕 **Full Bayesian inference** (MCMC) or NLME (FOCE/SAEM)
- 🆕 **CLI for inference workflows** (not just simulation)

---

## Mathematical Model (Unchanged from V0)

### Structural ODEs (Individual Level)

States:
- $A_{\text{gut}}(t)$ : Mass [mg]
- $A_{\text{central}}(t)$ : Mass [mg]

Parameters (individual $i$):
- $CL_i$ : Clearance [L/h]
- $V_i$ : Volume [L]
- $K_{a,i}$ : Absorption rate [1/h]

Dynamics:
$$
\begin{aligned}
\frac{dA_{\text{gut}}}{dt} &= -K_{a,i} \cdot A_{\text{gut}} \\
\frac{dA_{\text{central}}}{dt} &= K_{a,i} \cdot A_{\text{gut}} - \frac{CL_i}{V_i} \cdot A_{\text{central}}
\end{aligned}
$$

Observable:
$$
C_i(t) = \frac{A_{\text{central}}(t)}{V_i} \quad [\text{mg/L}]
$$

### Population Model (UPGRADED in V1)

Covariate: $WT_i$ [kg]

$$
\begin{aligned}
\log(CL_i) &= \log(CL_{\text{pop}}) + \beta_{CL} \log(WT_i / 70) + \eta_{CL,i} \\
\log(V_i) &= \log(V_{\text{pop}}) + \beta_V \log(WT_i / 70) + \eta_{V,i} \\
\log(K_{a,i}) &= \log(K_{a,\text{pop}}) + \eta_{Ka,i}
\end{aligned}
$$

**Random effects (V1 UPGRADE):**
$$
\boldsymbol{\eta}_i = \begin{bmatrix} \eta_{CL,i} \\ \eta_{V,i} \\ \eta_{Ka,i} \end{bmatrix} \sim \mathcal{N}(\mathbf{0}, \boldsymbol{\Omega})
$$

where $\boldsymbol{\Omega}$ is a **full 3×3 positive-definite covariance matrix** (not diagonal).

**Residual error (proportional):**
$$
C_{\text{obs}}(i,j) = C_{\text{pred}}(i,j) \cdot (1 + \varepsilon_{i,j}), \quad \varepsilon_{i,j} \sim \mathcal{N}(0, \sigma_{\text{prop}}^2)
$$

---

## Scope of Vertical Slice 1

### MUST HAVE

1. **Extend IR to encode full random-effects structure**
   - Represent $\boldsymbol{\eta}_i$ as vector of length 3
   - Represent $\boldsymbol{\Omega}$ as:
     - Full symmetric positive-definite covariance matrix, OR
     - Cholesky factor $L_\Omega$ (numerically stable parameterization)

2. **Add probabilistic backend**
   - Choose **one** of:
     - Stan/Torsten (generate `.stan` model)
     - Julia + Turing (generate Turing `@model`)
   - Generate:
     - Parameters (population, $\Omega$, $\sigma_{\text{prop}}$)
     - Random effects $\boldsymbol{\eta}_i$
     - ODE system
     - Likelihood

3. **Expose CLI for inference**
   ```bash
   medlangc infer examples/one_comp_oral_pk.medlang \
       --data examples/onecomp_synth.csv \
       --backend stan \
       --method mcmc \
       --chains 4 \
       --iter 2000
   ```

### DO NOT NEED

- ❌ Multiple models
- ❌ PBPK or QSP
- ❌ QM or Track C integration
- ❌ Complex error models (stick to proportional)

---

## Technical Tasks

### Task 1: IR Enhancement — Random Effects Structure

**Goal:** Explicitly represent MVNormal random effects in CIR

#### Current CIR (from V0):
```rust
struct CIRPopulation {
    pop_params: Vec<(String, Unit)>,
    random_effects: Vec<(String, Distribution)>,  // Simplified
    param_mapping: ExprGraph,
    measure: CIRMeasure,
}
```

#### V1 CIR (enhanced):
```rust
struct CIRRandomEffects {
    dimension: usize,  // d_eta = 3
    param_names: Vec<String>,  // ["eta_CL", "eta_V", "eta_Ka"]
    covariance: CovarianceStructure,
}

enum CovarianceStructure {
    Diagonal { variances: Vec<String> },  // V0 backward-compat
    Full { omega_matrix: String },         // 3x3 symmetric PD
    Cholesky { L_omega: String, tau: Option<Vec<String>> },  // LKJ + scales
}

struct CIRPopulation {
    pop_params: Vec<(String, Unit)>,
    random_effects: CIRRandomEffects,  // ← UPGRADED
    param_mapping: ExprGraph,
    measure: CIRMeasure,
}
```

**Action Items:**
1. Extend `CIRRandomEffects` struct
2. Update lowering pass (AST → CIR) to populate this structure
3. Add validation: ensure covariance is PD (at runtime or via constraints)

---

### Task 2: Probabilistic Backend — Choose One

#### Option A: Stan Backend

**Generate a complete `.stan` file with:**

##### 1. Functions Block (ODE RHS)
```stan
functions {
    real[] one_comp_oral(real t,
                         real[] A,
                         real[] theta,
                         real[] x_r,
                         int[] x_i) {
        real Ka = theta[1];
        real CL = theta[2];
        real V  = theta[3];

        real dA_gut     = -Ka * A[1];
        real dA_central =  Ka * A[1] - (CL / V) * A[2];

        return { dA_gut, dA_central };
    }
}
```

##### 2. Data Block
```stan
data {
    int<lower=1> N_subjects;
    int<lower=1> N_obs;
    
    int<lower=1,upper=N_subjects> subject_id[N_obs];
    real<lower=0> times[N_obs];
    real<lower=0> DV[N_obs];       // Observed concentrations
    real<lower=0> WT[N_subjects];  // Weights
    
    // Dosing data
    int<lower=1> N_doses;
    int<lower=1,upper=N_subjects> dose_subject_id[N_doses];
    real<lower=0> dose_times[N_doses];
    real<lower=0> dose_amounts[N_doses];
}
```

##### 3. Parameters Block
```stan
parameters {
    // Population parameters
    real<lower=0> CL_pop;
    real<lower=0> V_pop;
    real<lower=0> Ka_pop;
    real beta_CL;
    real beta_V;
    
    // Random effects covariance (Cholesky parameterization)
    cholesky_factor_corr[3] L_Omega;  // Correlation Cholesky
    vector<lower=0>[3] tau;           // Scales
    
    // Residual error
    real<lower=0> sigma_prop;
    
    // Individual random effects
    matrix[N_subjects, 3] eta_std;  // Standard normal
}
```

##### 4. Transformed Parameters Block
```stan
transformed parameters {
    matrix[N_subjects, 3] eta;
    vector[N_subjects] CL;
    vector[N_subjects] V;
    vector[N_subjects] Ka;
    
    // Transform standard normal to MVNormal
    eta = eta_std * diag_pre_multiply(tau, L_Omega)';
    
    // Individual parameters
    for (i in 1:N_subjects) {
        real w = WT[i] / 70.0;
        CL[i] = CL_pop * pow(w, beta_CL) * exp(eta[i, 1]);
        V[i]  = V_pop  * pow(w, beta_V)  * exp(eta[i, 2]);
        Ka[i] = Ka_pop                   * exp(eta[i, 3]);
    }
}
```

##### 5. Model Block
```stan
model {
    // Priors
    CL_pop ~ lognormal(log(10), 1);
    V_pop  ~ lognormal(log(50), 1);
    Ka_pop ~ lognormal(log(1), 1);
    beta_CL ~ normal(0.75, 0.5);
    beta_V  ~ normal(1.0, 0.5);
    
    L_Omega ~ lkj_corr_cholesky(2);
    tau ~ cauchy(0, 0.5);
    sigma_prop ~ cauchy(0, 0.5);
    
    // Random effects (standard normal)
    to_vector(eta_std) ~ normal(0, 1);
    
    // Likelihood
    for (i in 1:N_obs) {
        int subj = subject_id[i];
        real t = times[i];
        
        // Initial conditions (after dose)
        real A0[2];
        // ... handle dosing ...
        
        // Solve ODE
        real theta[3] = { Ka[subj], CL[subj], V[subj] };
        real A[2] = ode_rk45(one_comp_oral, A0, 0, t, theta, ...);
        
        // Predicted concentration
        real C_pred = A[2] / V[subj];
        
        // Likelihood (proportional error)
        real residual = (DV[i] / C_pred) - 1.0;
        residual ~ normal(0, sigma_prop);
    }
}
```

**Action Items:**
1. Implement `backend/stan/codegen.rs` (or `.jl`)
2. Template-based generation or programmatic string building
3. Handle dosing events (bolus to `A_gut`)
4. Validate generated `.stan` compiles with `stanc`

---

#### Option B: Julia + Turing Backend

**Generate a Julia file with Turing `@model`:**

```julia
using DifferentialEquations, Turing, Distributions

function one_comp_oral!(du, u, p, t)
    Ka, CL, V = p
    A_gut, A_central = u
    
    du[1] = -Ka * A_gut
    du[2] = Ka * A_gut - (CL / V) * A_central
end

@model function OneCompOralPop(times, DV, WT, subject_id, N_subjects)
    # Priors
    CL_pop ~ LogNormal(log(10), 1)
    V_pop  ~ LogNormal(log(50), 1)
    Ka_pop ~ LogNormal(log(1), 1)
    beta_CL ~ Normal(0.75, 0.5)
    beta_V  ~ Normal(1.0, 0.5)
    
    # Random effects covariance (Cholesky)
    tau ~ filldist(truncated(Cauchy(0, 0.5), 0, Inf), 3)
    L_Omega ~ LKJCholesky(3, 2.0)
    Omega = Symmetric(L_Omega.L * Diagonal(tau) * L_Omega.L')
    
    # Residual error
    sigma_prop ~ truncated(Cauchy(0, 0.5), 0, Inf)
    
    # Individual random effects
    eta = Matrix{Float64}(undef, N_subjects, 3)
    for i in 1:N_subjects
        eta[i, :] ~ MvNormal(zeros(3), Omega)
    end
    
    # Individual parameters
    CL = similar(WT)
    V  = similar(WT)
    Ka = similar(WT)
    for i in 1:N_subjects
        w = WT[i] / 70.0
        CL[i] = CL_pop * w^beta_CL * exp(eta[i, 1])
        V[i]  = V_pop  * w^beta_V  * exp(eta[i, 2])
        Ka[i] = Ka_pop             * exp(eta[i, 3])
    end
    
    # Likelihood
    for j in 1:length(times)
        i = subject_id[j]
        
        # Solve ODE
        prob = ODEProblem(one_comp_oral!, u0, (0.0, times[j]), [Ka[i], CL[i], V[i]])
        sol = solve(prob, Tsit5())
        
        # Predicted concentration
        C_pred = sol.u[end][2] / V[i]
        
        # Proportional error
        residual = (DV[j] / C_pred) - 1.0
        residual ~ Normal(0, sigma_prop)
    end
end
```

**Action Items:**
1. Implement `backend/julia/codegen.rs` (or `.jl`)
2. Generate dosing logic (reset `u0` at dose times)
3. Validate generated Julia code parses
4. Add simple inference test (NUTS sampling)

---

### Task 3: CLI for Inference Workflows

**Extend CLI with `infer` subcommand:**

```bash
medlangc infer examples/one_comp_oral_pk.medlang \
    --data examples/onecomp_synth.csv \
    --backend stan \
    --method mcmc \
    --chains 4 \
    --iter 2000 \
    --warmup 1000 \
    --output results/onecomp_fit.json
```

**Implementation:**

```rust
// CLI structure
enum Command {
    Compile { ... },     // V0 functionality
    Check { ... },       // V0 functionality
    Infer {              // V1 NEW
        model: PathBuf,
        data: PathBuf,
        backend: Backend,
        method: InferenceMethod,
        chains: usize,
        iter: usize,
        warmup: usize,
        output: PathBuf,
    },
}

enum InferenceMethod {
    MCMC,           // Bayesian (HMC/NUTS)
    MAP,            // Maximum a posteriori
    // Future: FOCE, SAEM
}
```

**Pipeline:**
1. Parse + typecheck `.medlang` file
2. Load dataset (CSV → structured data)
3. Generate backend code (`.stan` or `.jl`)
4. Write backend code to temp file
5. Call backend inference engine:
   - **Stan:** `cmdstan` via `stan-rs` or subprocess
   - **Julia:** Execute generated code via `julia` subprocess
6. Parse inference results (samples, diagnostics)
7. Save to JSON output

**Action Items:**
1. Add `infer` subcommand to CLI
2. Implement data loading (CSV parser)
3. Add backend execution logic
4. Parse results and serialize to JSON

---

### Task 4: Testing and Validation

#### Test 1: IR Integrity
**Goal:** Confirm CIR correctly represents random effects

```rust
#[test]
fn test_random_effects_ir() {
    let ast = parse_file("examples/one_comp_oral_pk.medlang").unwrap();
    let cir = lower_to_cir(ast).unwrap();
    
    assert_eq!(cir.population.random_effects.dimension, 3);
    assert_eq!(cir.population.random_effects.param_names, 
               vec!["eta_CL", "eta_V", "eta_Ka"]);
    
    match &cir.population.random_effects.covariance {
        CovarianceStructure::Cholesky { .. } => {},
        _ => panic!("Expected Cholesky covariance"),
    }
}
```

#### Test 2: Generated Code Compiles
**Goal:** Ensure backend code is syntactically valid

```rust
#[test]
fn test_stan_codegen_compiles() {
    let cir = load_example_cir();
    let stan_code = generate_stan_code(&cir).unwrap();
    
    // Write to temp file
    let temp_file = NamedTempFile::new().unwrap();
    write!(temp_file, "{}", stan_code).unwrap();
    
    // Validate with stanc
    let output = Command::new("stanc")
        .arg(temp_file.path())
        .output()
        .expect("stanc not found");
    
    assert!(output.status.success(), "Stan code failed to compile");
}
```

#### Test 3: Numerical Sanity (Fixed η)
**Goal:** Log-likelihood matches between internal runtime and backend

```rust
#[test]
fn test_loglik_agreement() {
    // Set fixed parameters and η = 0
    let params = FixedParams { 
        CL_pop: 10.0, V_pop: 50.0, Ka_pop: 1.0,
        eta: vec![0.0, 0.0, 0.0],
        sigma_prop: 0.1,
    };
    
    let data = load_synthetic_data();
    
    // Compute log-likelihood via V0 runtime
    let loglik_internal = compute_loglik_internal(&params, &data);
    
    // Compute via generated Stan model (single evaluation)
    let loglik_stan = evaluate_stan_loglik(&params, &data);
    
    assert!((loglik_internal - loglik_stan).abs() < 1e-10);
}
```

#### Test 4: Inference Smoke Test
**Goal:** MCMC runs without errors and produces reasonable posteriors

```rust
#[test]
fn test_inference_smoke() {
    let result = run_inference(
        "examples/one_comp_oral_pk.medlang",
        "examples/onecomp_synth.csv",
        Backend::Stan,
        InferenceMethod::MCMC { chains: 2, iter: 500 },
    ).unwrap();
    
    // Check diagnostics
    assert!(result.rhat["CL_pop"] < 1.1);
    assert!(result.ess["CL_pop"] > 100.0);
    
    // Check posteriors are reasonable
    let cl_mean = result.posterior_mean("CL_pop");
    assert!(cl_mean > 5.0 && cl_mean < 20.0);  // True value ~10
}
```

---

## Success Criteria

**Vertical Slice 1 is complete when:**

- ✅ CIR explicitly represents full MVNormal random effects with covariance
- ✅ Generated Stan/Turing code includes full Ω (Cholesky parameterization)
- ✅ `medlangc infer` command runs end-to-end (parse → codegen → MCMC)
- ✅ MCMC converges on synthetic data (R-hat < 1.1, ESS > 100)
- ✅ Posterior means recover true parameters within 20%
- ✅ All unit tests pass
- ✅ Integration tests pass (full inference workflow)
- ✅ Documentation updated (new CLI commands, examples)

**Stretch Goals:**
- Support for FOCE/SAEM (frequentist NLME)
- Both Stan AND Julia backends
- Posterior predictive checks
- Parameter uncertainty propagation to predictions

---

## Quality Standards

Same as V0:
- **Code quality:** Modular, well-documented, publication-ready
- **Testing:** Unit + integration tests, >80% coverage
- **Error handling:** Informative messages with context
- **Documentation:** API docs, examples, design rationale

---

## Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | IR enhancement, backend codegen | Updated CIR, Stan/Julia templates |
| 2 | CLI inference, integration | `infer` command, full pipeline |
| 3 | Testing, validation, docs | All tests passing, documentation |

**Total:** 2-3 weeks (assumes V0 is complete)

---

## What Happens After V1?

Once V1 is complete, the natural next steps are:

**Vertical Slice 2:** QSP Integration
- Add tumor-immune QSP module (still no quantum)
- Multi-scale PBPK ↔ QSP coupling
- Hybrid mechanistic-ML models

**Vertical Slice 3:** Quantum Integration (Track C)
- Load quantum-derived parameters from JSON (`ΔG_bind`, `ΔG_partition`)
- Map to PK/PD parameters (`Kp`, `EC50`)
- Full QM → PBPK/QSP vertical slice

---

## References

- **MedLang Track D Spec:** `docs/medlang_pharmacometrics_qsp_spec_v0.1.md`
- **V0 Implementation:** `docs/PROMPT_V0_BASIC_COMPILER.md`
- **Stan Manual:** https://mc-stan.org/docs/stan-users-guide/multivariate-hierarchical-priors.html
- **Turing LKJ:** https://turing.ml/dev/tutorials/07-poisson-regression/
- **NLME Theory:** Davidian & Giltinan (1995), "Nonlinear Models for Repeated Measurement Data"

---

**Ready to upgrade to V1?** Ensure V0 is complete, then start with Task 1 (IR enhancement). 🚀
