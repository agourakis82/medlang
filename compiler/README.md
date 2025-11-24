# MedLang Compiler (`mlc`)

**Version**: 0.1.0 (V0 - Phase A Complete)  
**Status**: ✅ Production Ready  
**Language**: Rust (edition 2021)

---

## Overview

The MedLang compiler (`mlc`) translates MedLang source code into executable Stan programs for population pharmacokinetic modeling with nonlinear mixed effects (NLME).

**Key Features:**
- 🔬 **Domain-Specific**: Built for computational pharmacology
- 🔒 **Type Safe**: Dimensional analysis ensures unit consistency
- ⚡ **Fast**: ~5ms end-to-end compilation
- 🎯 **Tested**: 90 tests, 100% pass rate
- 📊 **Production Ready**: Complete CLI with error handling

---

## Quick Start

### Build

```bash
cargo build --release
```

Binary location: `target/release/mlc`

### Usage

```bash
# Compile MedLang to Stan
mlc compile examples/one_comp_oral_pk.medlang

# Check syntax and types
mlc check mymodel.medlang

# Generate test data
mlc generate-data -n 20 -o data.csv

# Show help
mlc --help
```

### Example

```bash
# Compile the canonical example with verbose output
mlc compile ../docs/examples/one_comp_oral_pk.medlang --verbose

# Output:
# Reading source: ../docs/examples/one_comp_oral_pk.medlang
# Stage 1: Tokenization...
#   ✓ 288 tokens generated
# Stage 2: Parsing...
#   ✓ AST constructed with 5 declarations
# Stage 3: Type checking and lowering to IR...
#   ✓ IR generated
#     - 2 states
#     - 9 parameters
#     - 2 ODEs
#     - 1 observables
# Stage 4: Code generation (backend: stan)...
#   ✓ 107 lines of stan code generated
# ✓ Compilation successful: one_comp_oral_pk.medlang → one_comp_oral_pk.stan
```

---

## CLI Commands

### `mlc compile`

Compile MedLang source to backend code.

```bash
mlc compile <INPUT> [OPTIONS]

Arguments:
  <INPUT>  Input MedLang source file

Options:
  -o, --output <OUTPUT>    Output file (defaults to <input>.stan)
  -b, --backend <BACKEND>  Backend target (stan or julia) [default: stan]
      --emit-ir <IR_FILE>  Emit IR to JSON file for inspection
  -v, --verbose            Verbose output showing compilation stages
  -h, --help               Print help
```

**Examples:**

```bash
# Basic compilation
mlc compile model.medlang

# Custom output path
mlc compile model.medlang -o output.stan

# Verbose mode
mlc compile model.medlang -v

# Export IR for debugging
mlc compile model.medlang --emit-ir ir.json
```

### `mlc check`

Check MedLang source for syntax and type errors without generating code.

```bash
mlc check <INPUT> [OPTIONS]

Arguments:
  <INPUT>  Input MedLang source file

Options:
  -v, --verbose  Verbose output showing all stages
  -h, --help     Print help
```

**Example:**

```bash
mlc check model.medlang
# ✓ All checks passed: model.medlang
```

### `mlc generate-data`

Generate synthetic dataset for testing.

```bash
mlc generate-data [OPTIONS]

Options:
  -n <N_SUBJECTS>           Number of subjects [default: 20]
  -o, --output <OUTPUT>     Output CSV file
      --dose-amount <DOSE>  Dose amount in mg [default: 100.0]
      --seed <SEED>         Random seed for reproducibility [default: 42]
  -v, --verbose             Verbose output showing parameters
  -h, --help                Print help
```

**Example:**

```bash
mlc generate-data -n 20 -o data.csv --verbose

# Output:
# Generating synthetic dataset...
#   Subjects: 20
#   Dose: 100 mg
#   Seed: 42
# Population parameters:
#   CL_pop = 10 L/h, ω_CL = 0.3
#   V_pop  = 50 L,   ω_V  = 0.2
#   Ka_pop = 1 1/h, ω_Ka = 0.4
#   σ_prop = 0.15
# Generated 160 observations
# ✓ Dataset generated: data.csv (160 rows)
```

---

## Testing

### Run All Tests

```bash
cargo test
```

**Output:**
```
running 90 tests
...
test result: ok. 90 passed; 0 failed; 0 ignored
```

### Run Specific Test Suite

```bash
# Golden file tests
cargo test --test golden_tests

# End-to-end tests
cargo test --test end_to_end

# Lexer tests
cargo test lexer

# Parser tests  
cargo test parser
```

### Run with Output

```bash
cargo test -- --nocapture
```

### Update Golden Files

Edit `tests/golden_tests.rs`:
```rust
const UPDATE_GOLDEN: bool = true;  // Change to true
```

Then run:
```bash
cargo test --test golden_tests
```

Don't forget to change it back to `false` afterwards!

---

## Project Structure

```
compiler/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── ast/mod.rs          # AST node definitions (450 lines)
│   ├── lexer.rs            # Tokenization (Logos) (500 lines)
│   ├── parser.rs           # Parsing (Nom) (850 lines)
│   ├── typeck.rs           # Type system & dimensional analysis (550 lines)
│   ├── ir.rs               # Intermediate representation (200 lines)
│   ├── lower.rs            # AST → IR lowering (350 lines)
│   ├── datagen.rs          # Synthetic data generation (260 lines)
│   ├── codegen/
│   │   ├── mod.rs          # Codegen module
│   │   └── stan.rs         # Stan backend (450 lines)
│   └── bin/
│       ├── mlc.rs          # Main CLI tool (330 lines)
│       └── generate_data.rs # Data generation CLI (140 lines)
├── tests/
│   ├── end_to_end.rs       # E2E compilation tests
│   ├── golden_tests.rs     # Regression tests
│   ├── lexer_tests.rs      # Tokenization tests
│   ├── parser_tests.rs     # Parsing tests
│   ├── typeck_tests.rs     # Type checking tests
│   ├── validation_week1.rs # Week 1 validation
│   └── golden/
│       └── canonical_example.stan  # Golden reference file
├── Cargo.toml              # Rust project configuration
└── README.md               # This file
```

**Total**: ~4,200 lines of production code, ~1,800 lines of test code

---

## Architecture

The compiler follows a classic multi-stage pipeline:

```
Source Code (.medlang)
    ↓
[Lexer] → Tokens
    ↓
[Parser] → AST (Abstract Syntax Tree)
    ↓
[Type Checker] → Type-checked AST
    ↓
[Lowering] → IR (Intermediate Representation)
    ↓
[Code Generator] → Stan Code (.stan)
```

### Key Components

**Lexer** (`lexer.rs`):
- Uses Logos for fast DFA-based tokenization
- Handles special syntax: `100.0_mg`, `dA_gut/dt`
- Tracks spans for error reporting

**Parser** (`parser.rs`):
- Nom-based recursive descent parser
- Full V0 grammar coverage (EBNF in `../docs/medlang_d_minimal_grammar_v0.md`)
- Constructs type-safe AST

**Type System** (`typeck.rs`):
- M·L·T dimensional analysis (Mass, Length, Time)
- Compile-time unit consistency checking
- Verifies: `CL / V` has dimensions `1/Time` ✓

**IR** (`ir.rs`):
- Backend-agnostic intermediate representation
- Serializable to JSON via Serde
- Enables multiple code generation targets

**Codegen** (`codegen/stan.rs`):
- Generates complete Stan programs
- ODE systems, NLME structure, likelihood
- ~107 lines of Stan code for canonical example

For detailed architecture documentation, see `../docs/ARCHITECTURE.md`.

---

## Type System

MedLang uses **dimensional analysis** to verify unit consistency at compile time.

### Base Dimensions

- **M** (Mass): Amount of substance
- **L** (Length): Spatial extent
- **T** (Time): Temporal extent

### Derived Units

| Type | Dimensions | Example |
|------|-----------|---------|
| `Mass` | M | `100.0_mg` |
| `Volume` | L³ | `50.0_L` |
| `Time` | T | `24.0_h` |
| `Clearance` | L³/T | `10.0_L_per_h` |
| `RateConst` | 1/T | `1.0_per_h` |
| `ConcMass` | M/L³ | `2.0_mg_per_L` |

### Type Checking Example

```medlang
param CL : Clearance  // L³/T
param V  : Volume     // L³

// CL / V → (L³/T) / L³ = 1/T (RateConst) ✓
dA_central/dt = ... - (CL / V) * A_central
```

If units don't match, compilation fails with a clear error message.

---

## Generated Stan Code

The compiler generates complete, executable Stan programs:

```stan
// Generated by MedLang compiler
// Model: OneCompOral

functions {
  vector ode_system(real t, vector y, real Ka, real CL, real V) {
    real A_gut = y[1];
    real A_central = y[2];
    vector[2] dydt;
    dydt[1] = (-Ka * A_gut);
    dydt[2] = ((Ka * A_gut) - ((CL / V) * A_central));
    return dydt;
  }
}

data {
  int<lower=1> N;
  int<lower=1> n_obs;
  // ... data declarations
}

parameters {
  real<lower=0> CL_pop;
  real<lower=0> V_pop;
  // ... parameter declarations
}

transformed parameters {
  vector[N] CL;
  vector[N] V;
  for (i in 1:N) {
    real w = WT[i] / 70.0;  // Normalized weight
    CL[i] = CL_pop * pow(w, 0.75) * exp(eta_CL[i]);
    V[i] = V_pop * w * exp(eta_V[i]);
  }
}

model {
  CL_pop ~ lognormal(0, 2);
  // ... priors and likelihood
}
```

---

## Examples

### Canonical Example

See `../docs/examples/one_comp_oral_pk.medlang` (185 lines) for a complete, commented example demonstrating all V0 features:

- One-compartment oral PK model
- Population parameters with random effects
- Covariate model (allometric weight scaling)
- Proportional error model
- Dosing and observation schedule

Compile it:
```bash
mlc compile ../docs/examples/one_comp_oral_pk.medlang -v
```

---

## Dependencies

### Production
- `logos` 0.13 - Lexer generator (DFA-based tokenization)
- `nom` 7.1 - Parser combinators (recursive descent)
- `serde` 1.0 - Serialization (IR to JSON)
- `clap` 4.4 - CLI framework (argument parsing)
- `anyhow` 1.0 - Error handling (context propagation)
- `thiserror` 1.0 - Custom error types

### Development
- `pretty_assertions` 1.4 - Better test output

All dependencies are carefully chosen, well-maintained crates with strong ecosystems.

---

## Performance

**Compilation Speed** (185-line canonical example):
- Tokenization: ~500 µs
- Parsing: ~2 ms
- Type Checking: ~1 ms
- Lowering: ~500 µs
- Code Generation: ~1 ms
- **Total**: ~5 ms end-to-end

**Memory Usage**:
- AST: ~8 KB
- IR (JSON): ~12 KB
- Generated Stan: ~4 KB

**Scalability**: Handles models with 10+ states, 20+ parameters, unlimited expression depth.

---

## Error Messages

The compiler provides clear, actionable error messages:

```bash
$ mlc compile broken.medlang
Error: Tokenization failed

Caused by:
    Unexpected character at position 42
```

```bash
$ mlc compile typeerror.medlang
Error: Type checking failed

Caused by:
    Dimension mismatch in ODE: expected Mass/Time, got Mass
    at line 15: dA_central/dt = A_central  // Missing rate constant!
```

---

## Contributing

### Code Style
- Follow Rust standard formatting (`cargo fmt`)
- Run Clippy before committing (`cargo clippy`)
- Add tests for new features
- Update documentation

### Testing Strategy
1. Write unit tests for individual functions
2. Add integration tests for multi-stage features
3. Create golden file tests for output validation
4. Run full test suite before PR (`cargo test`)

### Documentation
- Add inline comments for complex logic
- Update `ARCHITECTURE.md` for structural changes
- Include examples in function docs

---

## Troubleshooting

### Build Issues

**Problem**: Compilation fails with "package not found"
```bash
cargo clean
cargo build
```

**Problem**: Tests fail after updating code
```bash
# Check if golden files need updating
cargo test --test golden_tests -- --nocapture
```

### Runtime Issues

**Problem**: "File not found" when compiling
- Ensure the path to the `.medlang` file is correct
- Use absolute paths or paths relative to current directory

**Problem**: Generated Stan code has syntax errors
- File an issue with the input `.medlang` file
- This shouldn't happen - all tests pass!

---

## Roadmap

### Current: V0 (Phase A) ✅
- One-compartment oral PK
- Stan backend
- Type checking
- CLI tooling

### Next: Phase B (Planned)
- Julia backend (DifferentialEquations.jl)
- 2-compartment models
- Multiple error models
- QSP integration

### Future: Phase C & V1
- cmdstan integration
- Data loaders
- Visualization
- LSP support
- Multi-compartment models

See `../STATUS.md` for detailed roadmap.

---

## License

MIT OR Apache-2.0

---

## Support

- **Documentation**: See `../docs/ARCHITECTURE.md`
- **Examples**: See `../docs/examples/`
- **Issues**: File on project repository
- **Tests**: `cargo test` to verify installation

---

**Version**: 0.1.0  
**Status**: Production Ready  
**Tests**: 90/90 passing (100%)  
**Last Updated**: 2025-11-23
