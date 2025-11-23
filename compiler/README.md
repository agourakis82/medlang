# MedLang Compiler (V0)

**Status:** Week 1 Complete ✅  
**Target:** Vertical Slice 0 - One-compartment oral PK with NLME

---

## Quick Start

```bash
# Build compiler
cargo build

# Generate synthetic dataset
cargo run --bin generate_data

# Run tests
cargo test
```

---

## Project Structure

```
compiler/
├── src/
│   ├── ast/           # AST node definitions (Week 2)
│   ├── parser/        # Parser implementation (Week 2)
│   ├── types/         # Unit system and type checker (Week 2)
│   ├── ir/            # CIR definitions and lowering (Week 2-3)
│   ├── backend/       # Code generation (Week 3-4)
│   │   ├── stan/      # Stan/Torsten backend
│   │   └── julia/     # Julia backend
│   ├── datagen.rs     # ✅ Synthetic dataset generator
│   ├── lib.rs         # Library root
│   ├── main.rs        # CLI entry point (Week 4)
│   └── bin/
│       └── generate_data.rs  # ✅ Data generation tool
├── tests/             # Integration tests (Week 4-5)
├── Cargo.toml         # ✅ Dependencies configured
└── README.md          # This file
```

---

## Week 1 Progress ✅

### Completed

1. **Grammar Specification**
   - `../docs/medlang_d_minimal_grammar_v0.md`
   - Complete EBNF syntax for V0 subset
   - Parsing and type checking strategies

2. **Canonical Example**
   - `../docs/examples/one_comp_oral_pk.medlang`
   - Full 1-compartment oral PK with NLME
   - Detailed unit checking annotations

3. **Dataset Generator** (Rust)
   - `src/datagen.rs`: RK4 ODE solver + data generation
   - `src/bin/generate_data.rs`: CLI tool
   - Generated: `../docs/examples/onecomp_synth.csv`
   - 20 subjects, 140 total rows (20 dose + 120 observations)
   - No external dependencies (self-contained RNG and ODE solver)

4. **Project Setup**
   - Rust project with Cargo
   - Dependencies: nom, logos, serde, clap
   - Directory structure for compiler modules

### True Population Parameters

Dataset generated with:
- `CL_pop = 10.0 L/h`
- `V_pop = 50.0 L`
- `Ka_pop = 1.0 1/h`
- `omega_CL = 0.3`
- `omega_V = 0.2`
- `omega_Ka = 0.4`
- `sigma_prop = 0.15`

---

## Week 2 Plan (Next)

### Tasks

1. **AST Definitions** (`src/ast/mod.rs`)
   - Struct definitions for all grammar constructs
   - Pretty-printing for debugging
   - Unit tests

2. **Parser Implementation** (`src/parser/mod.rs`)
   - Tokenizer using `logos`
   - Recursive descent parser using `nom`
   - Error handling with source locations
   - 10+ parser tests

3. **Type System** (`src/types/`)
   - Unit type system (Mass, Volume, Time, derived)
   - Type checker with dimensional analysis
   - 20+ type checking tests

### Deliverables

- Working parser that can parse `one_comp_oral_pk.medlang`
- Type checker that validates unit consistency
- Comprehensive test suite

---

## References

- **V0 Implementation Guide:** `../docs/PROMPT_V0_BASIC_COMPILER.md`
- **MedLang Grammar:** `../docs/medlang_d_minimal_grammar_v0.md`
- **Track D Spec:** `../docs/medlang_pharmacometrics_qsp_spec_v0.1.md`
- **Canonical Example:** `../docs/examples/one_comp_oral_pk.medlang`

---

## Development

### Building

```bash
cargo build           # Debug build
cargo build --release # Release build
```

### Testing

```bash
cargo test                    # Run all tests
cargo test datagen           # Run datagen tests only
cargo test --lib             # Library tests only
```

### Code Quality

```bash
cargo fmt                # Format code
cargo clippy             # Lint
cargo doc --open         # Generate and open docs
```

---

## Timeline (5 Weeks)

- **Week 1:** ✅ Grammar, example, dataset generator
- **Week 2:** 🔜 AST + Parser + Type system
- **Week 3-4:** Backend codegen (Stan or Julia)
- **Week 4:** CLI + Integration
- **Week 5:** Validation + Documentation

---

**Current Status:** Ready for Week 2 implementation 🚀
