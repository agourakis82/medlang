# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MedLang** is a medical-native, GPU/HPC-accelerated programming language designed to unify clinical reasoning, quantum pharmacology, AI models, probabilistic kernels, and fractal analysis. The project consists of:

- **compiler/** — MedLang compiler (`mlc`) written in Rust, targeting Stan and Julia backends
- **runtime/** — Device/runtime layer (CPU/GPU, QM backends, fractal kernels) - currently skeleton
- **beagle/** — Reference clinical application and UI - currently skeleton
- **docs/** — Manifesto, formal specifications, and examples

**Current Status**: V0 Complete (Phase C) - Production ready compiler with end-to-end workflow from MedLang source to MCMC inference.

## Building and Testing

### Compiler Build Commands

```bash
# From project root
cd compiler

# Debug build
cargo build

# Release build (faster, optimized)
cargo build --release

# Binary location
# Debug: compiler/target/debug/mlc
# Release: compiler/target/release/mlc
```

### Running Tests

```bash
# All tests (103 tests)
cargo test

# Specific test suites
cargo test --test golden_tests       # Regression tests
cargo test --test end_to_end          # E2E compilation tests
cargo test --test julia_backend_tests # Julia backend
cargo test lexer                      # Lexer unit tests
cargo test parser                     # Parser unit tests

# Show test output
cargo test -- --nocapture

# Run single test by name
cargo test test_one_comp_oral_compilation
```

### Linting

```bash
# Run clippy
cargo clippy

# Format code
cargo fmt

# Check formatting without modifying
cargo fmt -- --check
```

## CLI Commands

The `mlc` compiler provides five main commands:

### 1. Compile MedLang to Backend

```bash
# Compile to Stan (default)
mlc compile model.medlang

# Compile to Julia
mlc compile model.medlang --backend julia

# Verbose output showing all stages
mlc compile model.medlang -v

# Custom output path
mlc compile model.medlang -o output.stan

# Emit IR for debugging
mlc compile model.medlang --emit-ir ir.json
```

### 2. Check Syntax and Types

```bash
# Check without generating code
mlc check model.medlang

# Verbose mode
mlc check model.medlang -v
```

### 3. Generate Test Data

```bash
# Generate synthetic dataset
mlc generate-data -n 20 -o data.csv

# With verbose output
mlc generate-data -n 20 -o data.csv --verbose

# Custom dose and seed
mlc generate-data -n 20 -o data.csv --dose-amount 150.0 --seed 123
```

### 4. Convert Data to Stan Format

```bash
# Convert NONMEM-style CSV to Stan JSON
mlc convert-data data.csv -o data.json

# Verbose mode
mlc convert-data data.csv -o data.json -v
```

### 5. Run MCMC Sampling

```bash
# Run Stan MCMC with default settings
mlc run model.stan --data data.json

# Custom output directory and parameters
mlc run model.stan --data data.json --output results/ \
  --chains 4 --warmup 1000 --samples 1000 --seed 42

# Verbose with diagnostics
mlc run model.stan --data data.json -v
```

**Note**: The `mlc run` command requires cmdstan to be installed. Set `CMDSTAN` environment variable to cmdstan installation path, or it will auto-detect from common locations.

## Architecture and Code Structure

### Compilation Pipeline

The compiler follows a classic multi-stage pipeline:

```
Source (.medlang)
  → Lexer (lexer.rs - Logos DFA)
  → Parser (parser.rs - Nom combinators)
  → Type Checker (typeck.rs - M·L·T dimensional analysis)
  → Lowering (lower.rs - AST → IR)
  → Code Generator (codegen/stan.rs or codegen/julia.rs)
  → Output (.stan or .jl)
```

### Key Modules

| Module | Lines | Purpose |
|--------|-------|---------|
| `ast/mod.rs` | 450 | AST node definitions (Program, Declaration, Expression) |
| `lexer.rs` | 500 | Tokenization with special syntax (unit literals, ODE derivatives) |
| `parser.rs` | 850 | Nom-based parser implementing full V0 grammar |
| `typeck.rs` | 550 | Dimensional analysis type system (Mass·Length·Time) |
| `ir.rs` | 200 | Backend-agnostic intermediate representation |
| `lower.rs` | 350 | AST → IR lowering with name resolution |
| `codegen/stan.rs` | 450 | Stan code generation (ODE systems, NLME structure) |
| `codegen/julia.rs` | 400 | Julia code generation (DifferentialEquations.jl, Turing.jl) |
| `dataload.rs` | 250 | NONMEM CSV → Stan JSON conversion |
| `stanrun.rs` | 400 | Cmdstan integration with MCMC diagnostics |
| `datagen.rs` | 260 | Synthetic data generation (RK4 ODE solver) |
| `bin/mlc.rs` | 460 | Main CLI tool (5 commands) |

### Type System: M·L·T Dimensional Analysis

MedLang uses compile-time dimensional analysis to verify unit consistency:

- **M** (Mass): Amount of substance
- **L** (Length): Spatial extent (Volume = L³)
- **T** (Time): Temporal extent

**Derived Units**:
- `Clearance` = L³/T
- `RateConst` = 1/T
- `ConcMass` = M/L³

**Example Type Checking**:
```medlang
param CL : Clearance  // L³/T
param V  : Volume     // L³

// CL / V → (L³/T) / L³ = 1/T (RateConst) ✓
dA_central/dt = -(CL / V) * A_central
```

The type checker (`typeck.rs`) verifies all expressions have dimensionally consistent types and will fail compilation with clear error messages if units don't match.

### Intermediate Representation (IR)

The IR layer (`ir.rs`) provides:
1. **Backend Independence**: Both Stan and Julia use the same IR
2. **Serialization**: Can export to JSON via `--emit-ir` flag
3. **Explicit Classification**: Parameters are marked as Fixed, PopulationMean, PopulationVariance, or RandomEffect

The IR uses flat scopes with all variables resolved to unique names.

### Code Generation Strategy

**Stan Backend** (`codegen/stan.rs`):
- Generates complete Stan programs with functions, data, parameters, transformed parameters, and model blocks
- ODE systems use Stan's built-in `ode_system` function
- Implements NLME structure with population parameters and random effects
- Handles covariate models (e.g., allometric weight scaling)

**Julia Backend** (`codegen/julia.rs`):
- Uses DifferentialEquations.jl for ODE solving
- Generates Turing.jl `@model` for Bayesian inference
- Context-aware code generation (proper array indexing in loops)
- Julia-specific operator mapping (pow → ^)

## Important Implementation Details

### Golden File Testing

The project uses golden file regression tests (`tests/golden_tests.rs`). To update golden files:

1. Edit `tests/golden_tests.rs` and set `const UPDATE_GOLDEN: bool = true;`
2. Run `cargo test --test golden_tests`
3. **IMPORTANT**: Change back to `false` immediately after updating
4. Commit the updated golden files

### Special Syntax Handling

The lexer recognizes several special syntaxes:

- **Unit literals**: `100.0_mg`, `70.0_kg`, `24.0_h`
- **ODE derivatives**: `dA_gut/dt` extracts variable name `A_gut`
- **Qualified names**: `Normal.lpdf` for distribution calls

### Parameter Classification in IR

When lowering AST to IR, parameters are classified:

- **Fixed**: Time-invariant inputs (WT, DOSE)
- **PopulationMean**: Population-level parameters (CL_pop, V_pop)
- **PopulationVariance**: Inter-individual variability (ω_CL, ω_V)
- **RandomEffect**: Subject-level deviations (η_CL, η_V)

This classification drives backend code generation for proper NLME structure.

### MCMC Diagnostics

The `stanrun.rs` module computes comprehensive diagnostics:

- **Rhat**: Gelman-Rubin convergence statistic (should be < 1.01)
- **ESS bulk**: Effective sample size for bulk of distribution
- **ESS tail**: Effective sample size for tail regions
- **Quantiles**: 5%, 50% (median), 95%

These are automatically displayed after running `mlc run`.

## Common Development Workflows

### Adding a New Test

1. Add test to appropriate file in `tests/` directory
2. Run `cargo test <test_name>` to verify
3. Ensure all existing tests still pass: `cargo test`

### Debugging Compilation Issues

1. Use verbose mode: `mlc compile model.medlang -v`
2. Emit IR for inspection: `mlc compile model.medlang --emit-ir ir.json`
3. Examine IR structure: `cat ir.json | jq`
4. Check specific pipeline stage by running tests for that module

### Testing End-to-End Workflow

```bash
# 1. Compile MedLang to Stan
mlc compile docs/examples/one_comp_oral_pk.medlang -v

# 2. Generate test data
mlc generate-data -n 20 -o test_data.csv -v

# 3. Convert to Stan JSON format
mlc convert-data test_data.csv -o test_data.json -v

# 4. Run MCMC sampling (requires cmdstan)
mlc run one_comp_oral_pk.stan --data test_data.json --output results/ -v
```

### Adding a New Backend

1. Create `compiler/src/codegen/<backend>.rs`
2. Implement `generate_<backend>(ir: &IRProgram) -> Result<String>`
3. Add backend enum variant in `bin/mlc.rs`
4. Add backend-specific tests in `tests/`
5. Update documentation

## Important Files and Examples

- **Canonical Example**: `docs/examples/one_comp_oral_pk.medlang` (185 lines, fully commented)
- **V0 Grammar**: `docs/medlang_d_minimal_grammar_v0.md` (EBNF specification)
- **Architecture Guide**: `docs/ARCHITECTURE.md` (comprehensive technical documentation)
- **Workflow Guide**: `docs/WORKFLOW.md` (end-to-end usage)
- **Status Report**: `STATUS.md` (current progress and roadmap)

## Current Limitations and Future Work

**V0 Scope (Intentional Exclusions)**:
- Single compartment models only (2+ compartments → multi-compartment support exists but limited testing)
- Proportional error model only (additive/combined → future)
- Single dosing event (complex regimens → future)
- No LSP support (→ Phase V1)
- No IR optimization passes (→ Phase V1)

**Phase V1 Roadmap**:
- Multi-compartment models (3+)
- Time-varying covariates
- Complex dosing (infusion, multiple doses)
- Language server protocol (IDE support)
- IR optimization passes
- Visualization (trace plots, VPCs, parameter correlations)

## Dependencies

**Production**:
- `logos` 0.13 - DFA-based lexer generation
- `nom` 7.1 - Parser combinators
- `serde` 1.0 - Serialization (IR to JSON)
- `clap` 4.4 - CLI framework
- `anyhow` 1.0 - Error handling
- `thiserror` 1.0 - Custom error types
- `rand` 0.8 - Random number generation
- `csv` 1.3 - CSV parsing

**Development**:
- `pretty_assertions` 1.4 - Better test output
- `criterion` 0.5 - Benchmarking
- `tempfile` 3.8 - Temporary files for tests

## Performance Characteristics

**Compilation Speed** (185-line canonical example):
- Tokenization: ~500 µs
- Parsing: ~2 ms
- Type Checking: ~1 ms
- Lowering: ~500 µs
- Code Generation: ~1 ms
- **Total**: ~5 ms end-to-end

**Scalability**: Current implementation handles models with 10+ states, 20+ parameters, and unlimited expression depth.

## Project Philosophy

1. **Type Safety First**: Use dimensional analysis to catch unit errors at compile time
2. **Backend Agnostic**: IR layer enables multiple targets without duplicating logic
3. **Research Quality**: Support state-of-the-art pharmacometric workflows
4. **Deterministic**: Compilation and data generation are reproducible with seeds
5. **Testing**: Comprehensive coverage (103 tests, 100% pass rate)
6. **Documentation**: Architecture guides for new contributors

## Notes for Future Development

- The runtime and Beagle directories are currently skeletons - focus is on compiler maturity
- When extending grammar, update both EBNF specification and parser implementation
- All new features should include tests and update golden files if needed
- IR is the extension point - keep it backend-agnostic
- Maintain dimensional analysis correctness when adding new types
