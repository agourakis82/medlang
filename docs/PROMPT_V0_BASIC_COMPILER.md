# MedLang Vertical Slice 0 — Implementation Prompt

**Target:** Build the foundational MedLang compiler for 1-compartment oral PK with NLME  
**Status:** Ready to implement  
**Estimated effort:** 4-5 weeks (full-time)

---

## Mission

Implement **Vertical Slice 0**: a minimal but fully functional MedLang-D compiler demonstrating:

1. **Domain-specific syntax** for pharmacometric models
2. **Unit-safe type system** with compile-time dimensional analysis
3. **IR-based compilation** (AST → CIR → Backend)
4. **Executable backend** (Stan/Torsten OR Julia)
5. **End-to-end validation** (simulation + log-likelihood)

**Quality bar:** Publication-ready (Q1 journal level) code, documentation, and rigor.

---

## Context: What is MedLang?

MedLang is a **domain-specific language for computational medicine** designed to provide:

- **Type-safe dimensional analysis** for pharmacometric models
- **Multi-scale vertical integration** (quantum → clinical)
- **Native PK/PD/PBPK/QSP semantics**
- **Compile-time correctness guarantees**

This is **Vertical Slice 0**: the minimal core proving the concept works.

**Repository:** https://github.com/[your-org]/medlang  
**Specifications:** See `docs/medlang_core_spec_v0.1.md`, `docs/medlang_pharmacometrics_qsp_spec_v0.1.md`

---

## What You're Building

### Inputs
A `.medlang` file describing a 1-compartment oral PK model with population structure:

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

population OneCompOralPop {
    model OneCompOral

    param CL_pop  : Clearance
    param V_pop   : Volume
    param Ka_pop  : RateConst
    param omega_CL : f64
    param omega_V  : f64
    param omega_Ka : f64

    input WT : Quantity<kg, f64>

    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    rand eta_V  : f64 ~ Normal(0.0, omega_V)
    rand eta_Ka : f64 ~ Normal(0.0, omega_Ka)

    bind_params(patient) {
        let w = patient.WT / 70.0_kg
        model.CL = CL_pop * pow(w, 0.75) * exp(eta_CL)
        model.V  = V_pop * w * exp(eta_V)
        model.Ka = Ka_pop * exp(eta_Ka)
    }

    use_measure ConcPropError for model.C_plasma
}

measure ConcPropError {
    pred : ConcMass
    obs  : ConcMass
    param sigma_prop : f64

    log_likelihood = Normal_logpdf(
        x  = (obs / pred) - 1.0,
        mu = 0.0,
        sd = sigma_prop
    )
}
```

### Outputs

1. **Parsed and type-checked IR** (with unit validation)
2. **Generated Stan or Julia code** implementing the model
3. **Simulation capability** (forward solve ODEs)
4. **Log-likelihood computation** (for parameter estimation)

---

## Technical Architecture

### Compilation Pipeline

```
.medlang file
    ↓
[PARSER] → AST (syntax tree)
    ↓
[TYPE CHECKER] → Typed AST (units validated)
    ↓
[IR LOWERING] → CIR (Clinical IR, unit-typed intermediate representation)
    ↓
[BACKEND CODEGEN] → Stan/Julia code
    ↓
[STAN/JULIA COMPILER] → Executable binary
    ↓
[RUNTIME] → Simulation + Log-likelihood
```

### Module Structure

```
medlang/
├── compiler/
│   ├── src/
│   │   ├── ast/          # AST node definitions
│   │   ├── parser/       # Recursive descent or combinator parser
│   │   ├── types/        # Unit system and type checker
│   │   ├── ir/           # CIR definitions and lowering
│   │   ├── backend/      # Code generation (Stan or Julia)
│   │   └── main.rs/jl    # CLI entry point
│   └── tests/            # Unit and integration tests
├── runtime/              # (Optional) Direct simulation runtime
├── docs/
│   ├── examples/
│   │   ├── one_comp_oral_pk.medlang
│   │   └── onecomp_synth.csv
│   └── IMPLEMENTATION_GUIDE_V0.md  # Detailed implementation spec
└── README.md
```

---

## Implementation Plan (7 Steps)

### Step 1: Grammar + Examples (Week 1)

**Tasks:**
1. Define minimal MedLang grammar (EBNF or similar)
2. Create canonical example: `examples/one_comp_oral_pk.medlang`
3. Generate synthetic dataset: `examples/onecomp_synth.csv` (10-20 subjects)

**Deliverables:**
- `docs/medlang_d_minimal_grammar_v0.md`
- `docs/examples/one_comp_oral_pk.medlang`
- `docs/examples/onecomp_synth.csv`

### Step 2: AST + Parser (Week 1-2)

**Tasks:**
1. Define AST nodes for:
   - `ModelDef` (states, params, ODEs, observables)
   - `PopulationDef` (population params, random effects, bind_params)
   - `MeasureDef` (prediction, observation, log_likelihood)
2. Implement parser (recursive descent or combinator library)
3. Add error handling (informative parse errors with line/column)

**Deliverables:**
- `compiler/src/ast/mod.rs` (or `ast.jl`)
- `compiler/src/parser/mod.rs` (or `parser.jl`)
- 10+ parser unit tests

### Step 3: Type System (Week 2)

**Tasks:**
1. Implement unit type system:
   - Base units: `Mass`, `Volume`, `Time`
   - Derived: `ConcMass = Mass/Volume`, `Clearance = Volume/Time`, `RateConst = 1/Time`
   - Unit arithmetic: `*`, `/`, `+` (only same units)
2. Implement type checker:
   - Walk AST, assign units to all expressions
   - Validate `dState/dt` has correct units
   - Check `exp(x)`, `log(x)` only on dimensionless values
   - Validate observable units match measure expectations

**Deliverables:**
- `compiler/src/types/unit.rs` (or `unit.jl`)
- `compiler/src/types/typechecker.rs` (or `typechecker.jl`)
- 20+ type checking tests (valid + invalid cases)

**Critical test cases:**
- ✅ Valid: `dA_gut/dt = -Ka * A_gut` (RateConst × Mass = Mass/Time)
- ❌ Invalid: `dA_gut/dt = -CL * A_gut` (Clearance × Mass ≠ Mass/Time)
- ❌ Invalid: `exp(CL)` (cannot exponentiate dimensional quantity)

### Step 4: IR Lowering (Week 2-3)

**Tasks:**
1. Define CIR (Clinical IR) structures:
   ```
   CIRModel {
       states: Vec<(String, Unit)>,
       params: Vec<(String, Unit)>,
       ode_rhs: Vec<(StateIdx, ExprGraph)>,
       observables: Vec<(String, Unit, ExprGraph)>
   }
   
   CIRPopulation {
       pop_params: Vec<(String, Unit)>,
       random_effects: Vec<(String, Distribution)>,
       param_mapping: ExprGraph,
       measure: CIRMeasure
   }
   ```

2. Implement lowering pass: AST → CIR
3. Add CIR serialization (JSON for debugging)

**Deliverables:**
- `compiler/src/ir/cir.rs` (or `cir.jl`)
- `compiler/src/ir/lowering.rs` (or `lowering.jl`)
- Integration tests: Parse → Typecheck → Lower → Inspect CIR

### Step 5: Backend Codegen (Week 3-4)

**Choose ONE backend for V0:** Stan/Torsten OR Julia (DifferentialEquations + Turing/Pumas)

**Tasks (Stan example):**
1. Generate `functions` block with ODE RHS:
   ```stan
   functions {
       real[] ode_rhs(real t, real[] y, real[] theta, real[] x_r, int[] x_i) {
           real dydt[2];
           real Ka = theta[1];
           real CL = theta[2];
           real V  = theta[3];
           
           dydt[1] = -Ka * y[1];
           dydt[2] = Ka * y[1] - (CL/V) * y[2];
           return dydt;
       }
   }
   ```

2. Generate `data` block (subject IDs, times, observations, covariates)
3. Generate `parameters` block (population params, random effects, error params)
4. Generate `model` block (priors, random effects, likelihood)

**Deliverables:**
- `compiler/src/backend/stan/codegen.rs` (or `backend/julia/`)
- Generated code compiles successfully (validate with `stanc` or Julia parser)
- Basic simulation test (run generated code with known parameters)

### Step 6: CLI + Integration (Week 4)

**Tasks:**
1. Create CLI tool:
   ```bash
   medlangc compile examples/one_comp_oral_pk.medlang \
       --backend stan \
       --output out/onecomp.stan
   ```

2. Add subcommands:
   - `compile`: Full pipeline (parse → typecheck → IR → codegen)
   - `check`: Type-check only
   - `ir`: Output CIR JSON for inspection

3. Integration tests:
   - Full pipeline on canonical example
   - Compare simulation to analytic solution
   - Validate log-likelihood computation

**Deliverables:**
- `compiler/src/main.rs` (or `main.jl`)
- CLI documentation in README
- End-to-end integration tests

### Step 7: Validation (Week 4-5)

**Numerical Validation:**

1. **Simulation accuracy:**
   - Analytic solution for 1-comp oral: $A_c(t) = \frac{F \cdot D \cdot K_a}{V(K_a - K_e)}(e^{-K_e t} - e^{-K_a t})$
   - Tolerance: < 1e-6 relative error

2. **Log-likelihood accuracy:**
   - Compute log-likelihood with synthetic data
   - Compare MedLang-generated vs. hand-coded Stan/Julia
   - Tolerance: < 1e-10 absolute difference

**Deliverables:**
- Validation report (`docs/VALIDATION_REPORT_V0.md`)
- All numerical tests passing
- Full API documentation

---

## Success Criteria

**Vertical Slice 0 is complete when:**

- ✅ Canonical example (`one_comp_oral_pk.medlang`) parses without errors
- ✅ Type checker catches all intentional unit violations
- ✅ Generated Stan/Julia code compiles successfully
- ✅ Simulation matches analytic solution (< 1e-6 error)
- ✅ Log-likelihood matches direct computation (< 1e-10 error)
- ✅ All unit tests pass (>80% coverage)
- ✅ Integration tests pass (full pipeline)
- ✅ Documentation complete (API docs + examples)

**Stretch goals (optional):**
- Support 2-compartment model
- Additive error model
- Both Stan AND Julia backends

---

## Quality Standards

### Code Quality
- **Clarity over cleverness:** Readable code with clear variable names
- **Separation of concerns:** Clean module boundaries
- **Explicit over implicit:** No magic constants
- **Q1 journal level:** Publication-ready code quality

### Documentation
- **Mathematical rigor:** Reference equations from specs
- **Design rationale:** Explain non-obvious choices
- **Examples:** Working examples for every feature
- **API docs:** Inline documentation for all public functions

### Testing
- **Unit tests:** Every module, every function
- **Integration tests:** Full pipeline end-to-end
- **Validation tests:** Numerical accuracy checks
- **Coverage:** Aim for >80% code coverage

### Error Handling
- **Informative messages:** File, line, column, context
- **Suggestions:** Hint at fixes where possible
- **No silent failures:** Fail fast and loudly

---

## Recommended Tech Stack

**Option A: Rust**
- Parser: `nom` or `pest`
- ODE solving: Call out to Stan or generate Julia
- Strong typing, excellent error messages
- Fast compilation

**Option B: Julia**
- Parser: `ParserCombinator.jl` or hand-rolled
- ODE solving: Native `DifferentialEquations.jl`
- Excellent numerical ecosystem
- Fast prototyping

**Choose based on your team's expertise.**

---

## Getting Started

### Immediate Next Steps

1. **Read the specifications:**
   - `docs/medlang_core_spec_v0.1.md`
   - `docs/medlang_pharmacometrics_qsp_spec_v0.1.md`
   - `docs/IMPLEMENTATION_GUIDE_V0.md`

2. **Set up repository structure:**
   ```bash
   cd medlang
   mkdir -p compiler/src/{ast,parser,types,ir,backend}
   mkdir -p compiler/tests
   mkdir -p docs/examples
   ```

3. **Start with Step 1 (Grammar):**
   - Create `docs/medlang_d_minimal_grammar_v0.md`
   - Write canonical example
   - Generate synthetic dataset

4. **Build incrementally:**
   - Each step builds on the previous
   - Test continuously
   - Document as you go

---

## Non-Goals (Explicitly Deferred to V0.2+)

**DO NOT implement in V0:**
- ❌ Track C quantum operators
- ❌ PBPK/QSP multi-scale models
- ❌ ML submodels
- ❌ General function definitions
- ❌ Advanced error models (additive, combined, censored)
- ❌ Optimization (focus on correctness first)

**V0 is ONLY:**
- ✅ 1-compartment oral PK
- ✅ NLME with diagonal omega
- ✅ Proportional error
- ✅ Single backend (Stan OR Julia)

---

## References

- **MedLang Specs:** `docs/` directory
- **Stan Manual:** https://mc-stan.org/docs/
- **Julia DiffEq:** https://diffeq.sciml.ai/
- **Turing.jl:** https://turing.ml/
- **NLME Background:** Beal & Sheiner (1982), "Estimating Population Kinetics"

---

## Expected Timeline

| Week | Focus | Deliverables |
|------|-------|--------------|
| 1 | Grammar, AST, Parser | Grammar spec, parser with tests |
| 2 | Type system, IR lowering | Type checker, CIR definitions |
| 3 | Backend codegen | Stan/Julia code generation |
| 4 | CLI, integration | Working CLI, integration tests |
| 5 | Validation, docs | Validation report, complete docs |

**Total:** 4-5 weeks full-time

---

## Support

- **Specs:** See `docs/` directory
- **Issues:** GitHub issues for questions/bugs
- **Architecture questions:** Refer to `docs/IMPLEMENTATION_GUIDE_V0.md`

---

## What Happens After V0?

Once V0 is complete and validated, the natural next steps are:

**Vertical Slice 1:** Real population inference
- MVNormal random effects with full covariance
- Compile to probabilistic backend (Stan/Turing)
- Full Bayesian inference capability

**Vertical Slice 2:** QSP integration
- Add tumor-immune QSP module
- Multi-scale PBPK ↔ QSP coupling

**Vertical Slice 3:** Quantum integration
- Track C stub (load quantum-derived parameters from JSON)
- Full QM → Kp/EC50 → PBPK/QSP vertical

But **first**, we build V0.

---

**Ready to build?** Start with Step 1 (Grammar + Examples). Good luck! 🚀
