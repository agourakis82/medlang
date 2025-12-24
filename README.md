# MedLang

**MedLang** is an embedded Domain-Specific Language (eDSL) for the [Demetrios (D) language](https://github.com/Chiuratto-AI/demetrios), designed for computational pharmacology and medical modeling.

MedLang provides a medical-native, GPU/HPC-accelerated programming interface that unifies:

- Clinical reasoning (patients, cohorts, protocols, endpoints)
- Quantum pharmacology and molecular modeling (HF/DFT/post-HF/QM/MM)
- AI models (MLP, GNN, PINN) with native autodiff and optimization
- Probabilistic kernels and measures
- Fractal analysis of physiological and clinical signals

## 🎯 Overview

MedLang is built on top of the **Demetrios (D) language** and extends it with domain-specific constructs for computational medicine. It aims to be a **single coherent eDSL** where:

- **Molecules**, **patients**, **cohorts**, **protocols**, **states of mind**, **probability measures**, **quantum operators**, and **fractal dimensions** are all first-class citizens
- Code that starts at quantum-level pharmacology (HF/DFT/QM/MM) can propagate up to PK/PD, physiological dynamics, AI risk models, and clinical decision support — without leaving the language
- Safety is enforced at the language level:
  - typed units for doses and physiological quantities
  - ownership/borrowing à la Rust to avoid memory bugs
  - deterministic execution by default
  - explicit control of randomness and approximation

## 📁 Project Structure

```
medlang/
├── compiler/          # MedLang compiler (Rust)
│   ├── src/          # Compiler source code
│   └── tests/        # Compiler tests
├── runtime/          # Runtime layer (CPU/GPU, QM backends, fractal kernels)
├── beagle/           # Reference application and IDE/cockpit for MedLang
├── ffi/               # FFI crate (C ABI for external language integration)
│   ├── src/          # FFI source code
│   └── include/       # Generated C headers
├── stdlib/           # MedLang standard library (advanced modules)
├── medlang_std/      # MedLang standard library (models, protocols, policies)
├── docs/             # Complete documentation
│   ├── spec/         # Formal specifications (grammar, core, extensions)
│   ├── guides/       # User guides and tutorials
│   ├── examples/     # MedLang code examples
│   └── dev/          # Development documentation (history, status)
└── examples/         # Usage examples (empty, use docs/examples/)
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/medlang.git
cd medlang

# Build the compiler
cd compiler
cargo build --release

# Binary will be at: target/release/mlc
```

### Basic Usage

```bash
# Compile a MedLang model to Stan
./target/release/mlc compile docs/examples/one_comp_oral_pk.medlang

# Compile to Julia
./target/release/mlc compile docs/examples/one_comp_oral_pk.medlang --backend julia

# Check syntax and types
./target/release/mlc check docs/examples/one_comp_oral_pk.medlang

# Generate synthetic data
./target/release/mlc generate-data -n 20 -o data.csv

# Convert NONMEM CSV data to Stan JSON format
./target/release/mlc convert-data data.csv -o data.json

# Run MCMC with Stan
./target/release/mlc run model.stan --data data.json --output results/
```

## 📚 Documentation

### Formal Specifications
- **[Grammar V0](docs/spec/medlang_d_minimal_grammar_v0.md)** — Basic EBNF grammar
- **[Grammar V1.0](docs/spec/medlang_d_grammar_v1.0.md)** — Complete grammar
- **[Core Specification](docs/spec/medlang_core_spec_v0.1.md)** — Core language specification
- **[Pharmacometrics/QSP](docs/spec/medlang_pharmacometrics_qsp_spec_v0.1.md)** — Pharmacometric extensions
- **[Quantum Pharmacology](docs/spec/medlang_qm_pharmacology_spec_v0.1.md)** — QM extensions

### User Guides
- **[Complete Workflow](docs/guides/WORKFLOW.md)** — End-to-end usage guide
- **[Quick Reference](docs/guides/QUICKREF.md)** — Quick reference
- **[PBPK User Guide](docs/guides/pbpk_user_guide.md)** — PBPK models guide
- **[Trial Analysis](docs/guides/USER_GUIDE_ANALYZE_TRIAL.md)** — Clinical trial data analysis

### Development Documentation
- **[Architecture](docs/dev/ARCHITECTURE.md)** — Compiler architecture
- **[Status](docs/dev/STATUS.md)** — Current project status
- **[Changelog](docs/dev/CHANGELOG.md)** — Change history

### Examples
- **[Complete Examples](docs/examples/)** — Collection of MedLang examples
- **[Canonical Example](docs/examples/one_comp_oral_pk.medlang)** — 1-compartment oral PK model (185 lines, fully commented)

## 🏗️ Project Status

**Current Version**: v0.5.0 (Phase V1 Complete)

### ✅ Implemented Features

- ✅ Complete compilation pipeline (Lexer → Parser → Type Checker → IR → Codegen)
- ✅ Type system with dimensional analysis (M·L·T)
- ✅ Backends: Stan and Julia
- ✅ Multi-compartment model support
- ✅ Synthetic data generation
- ✅ Data loading (NONMEM CSV → Stan JSON)
- ✅ MCMC execution with cmdstan and diagnostics
- ✅ Effect system (Prob, IO, GPU, Pure)
- ✅ Epistemic computing (Knowledge<T>)
- ✅ Clinical refinement types
- ✅ 127+ tests passing (100% pass rate)

### 🚧 In Development

- 🔄 Multi-compartment support (3+)
- 🔄 Time-varying covariates
- 🔄 Complex dosing (infusion, multiple doses)
- 🔄 Language Server Protocol (LSP)
- 🔄 IR optimization passes
- 🔄 Visualization (trace plots, VPCs, correlations)

## 🧪 Testing

```bash
cd compiler

# All tests
cargo test

# Specific tests
cargo test --test golden_tests       # Regression tests
cargo test --test end_to_end          # End-to-end tests
cargo test --test julia_backend_tests # Julia backend

# With output
cargo test -- --nocapture
```

## 📦 Standard Libraries

### `medlang_std/` — Basic Models and Protocols
- **models/** — Standard PK/PD models (OneCmptIV, TwoCmptIV, OneCmptOral)
- **protocols/** — Reusable dosing protocols
- **policies/** — Interpretable dosing policies

### `stdlib/` — Advanced Modules
- **core/** — Core functionality (autodiff, generics, traits)
- **ml/** — Machine learning models
- **rl/** — Reinforcement learning for dose optimization
- **registry/** — Model and protocol registry

## 🔌 Foreign Function Interface (FFI)

MedLang provides a C-compatible FFI for integration with other languages (C, Python, Julia, R, etc.).

### Building the FFI

```bash
cd ffi
cargo build --release
```

This produces `libmedlang.so` (Linux), `libmedlang.dylib` (macOS), or `medlang.dll` (Windows).

### Usage Examples

**C:**
```c
#include "ffi/include/medlang.h"
medlang_ctx_t* ctx = medlang_init();
// ... use FFI functions ...
medlang_free(ctx);
```

**Python (ctypes):**
```python
import ctypes
lib = ctypes.CDLL("./ffi/target/release/libmedlang.so")
ctx = lib.medlang_init()
```

**Julia:**
```julia
using Libdl
lib = Libdl.dlopen("./ffi/target/release/libmedlang.so")
ctx = ccall(Libdl.dlsym(lib, :medlang_init), Ptr{Cvoid}, ())
```

See [ffi/README.md](ffi/README.md) for complete FFI documentation.

## 🔬 Technical Features

### Type System
- **Dimensional Analysis**: Compile-time unit checking (M·L·T)
- **Refinement Types**: Clinical predicates for data validation
- **Effect System**: Computational side-effect tracking

### Backends
- **Stan**: Stan code generation with ODE and NLME integration
- **Julia**: DifferentialEquations.jl + Turing.jl for Bayesian inference

### Compilation Pipeline
```
Source (.medlang)
  → Lexer (lexer.rs - Logos DFA)
  → Parser (parser.rs - Nom combinators)
  → Type Checker (typeck.rs - M·L·T dimensional analysis)
  → Lowering (lower.rs - AST → IR)
  → Code Generator (codegen/stan.rs or codegen/julia.rs)
  → Output (.stan or .jl)
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines

- Read [docs/dev/ARCHITECTURE.md](docs/dev/ARCHITECTURE.md) to understand the architecture
- Run all tests before submitting (`cargo test`)
- Follow standard Rust code style (`cargo fmt`, `cargo clippy`)
- Add tests for new features
- Update documentation as needed

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This project is licensed under the MIT OR Apache-2.0 license — see the LICENSE files for details.

## 🔗 Links

- **Demetrios (D) Language**: [github.com/Chiuratto-AI/demetrios](https://github.com/Chiuratto-AI/demetrios) - The host language for MedLang
- **Manifesto**: [docs/manifesto.md](docs/manifesto.md)
- **Detailed Status**: [docs/dev/STATUS.md](docs/dev/STATUS.md)
- **Architecture**: [docs/dev/ARCHITECTURE.md](docs/dev/ARCHITECTURE.md)
- **Project Structure**: [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 📧 Contact

For questions and discussions, please open an issue on GitHub.

---

**MedLang** — An eDSL for [Demetrios (D)](https://github.com/Chiuratto-AI/demetrios), unifying quantum pharmacology, clinical modeling, and AI in a single coherent language.

See [docs/DEMETRIOS_INTEGRATION.md](docs/DEMETRIOS_INTEGRATION.md) for details on the MedLang-Demetrios relationship.
