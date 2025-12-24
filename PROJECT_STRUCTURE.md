# MedLang Project Structure

This document describes the organizational structure of the MedLang repository.

## 📁 Overview

```
medlang/
├── README.md                 # Main project README
├── CONTRIBUTING.md           # Contributing guide
├── PROJECT_STRUCTURE.md      # This file
├── .gitignore              # Git ignored files
│
├── compiler/                 # MedLang compiler (Rust)
│   ├── Cargo.toml           # Rust project configuration
│   ├── Cargo.lock           # Dependency lock file
│   ├── README.md            # Compiler documentation
│   ├── src/                 # Source code
│   │   ├── main.rs         # Entry point
│   │   ├── lib.rs          # Public library
│   │   ├── lexer.rs        # Tokenization
│   │   ├── parser.rs       # Parsing
│   │   ├── typeck.rs       # Type checking
│   │   ├── lower.rs        # Lowering AST → IR
│   │   ├── ir.rs           # Intermediate representation
│   │   ├── ast/            # AST definitions
│   │   ├── codegen/        # Code generation (Stan, Julia)
│   │   ├── bin/            # Binaries (mlc, generate_data)
│   │   └── ...             # Other modules
│   └── tests/               # Compiler tests
│       ├── golden_tests.rs  # Regression tests
│       └── ...              # Other tests
│
├── runtime/                 # Runtime layer (CPU/GPU, QM backends)
│   └── README.md
│
├── beagle/                  # Reference application and IDE
│   ├── README.md
│   └── README_OLD.md        # Backup
│
├── stdlib/                  # Standard library (advanced modules)
│   └── med/
│       ├── core/            # Core functionality
│       ├── ml/              # Machine learning
│       ├── rl/              # Reinforcement learning
│       └── ...
│
├── medlang_std/             # Standard library (basic models)
│   ├── models/              # Standard PK/PD models
│   ├── protocols/           # Dosing protocols
│   ├── policies/            # Dosing policies
│   └── README.md
│
├── docs/                    # Complete documentation
│   ├── README.md            # Documentation index
│   ├── manifesto.md         # Project manifesto
│   │
│   ├── spec/                # Formal specifications
│   │   ├── README.md
│   │   ├── medlang_d_grammar_v*.md    # Grammars
│   │   ├── medlang_core_spec_v*.md    # Core spec
│   │   ├── medlang_pharmacometrics_qsp_spec_v*.md
│   │   ├── medlang_qm_pharmacology_spec_v*.md
│   │   └── ...
│   │
│   ├── guides/              # User guides
│   │   ├── README.md
│   │   ├── WORKFLOW.md      # Complete workflow
│   │   ├── QUICKREF.md      # Quick reference
│   │   └── ...
│   │
│   ├── examples/            # Code examples
│   │   ├── README.md
│   │   ├── one_comp_oral_pk.medlang  # Canonical example
│   │   ├── phase_v1/        # Phase V1 examples
│   │   └── ...
│   │
│   └── dev/                 # Development documentation
│       ├── README.md
│       ├── ARCHITECTURE.md  # Compiler architecture
│       ├── STATUS.md        # Project status
│       ├── WEEK_*.md        # Weekly reports
│       └── ...
│
└── examples/                # Usage examples (empty, use docs/examples/)
```

## 🎯 Organization by Category

### Source Code
- **`compiler/`** — Main compiler (Rust)
- **`runtime/`** — Runtime and backends
- **`beagle/`** — IDE/reference application

### Libraries
- **`stdlib/`** — Advanced standard library (autodiff, ML, RL)
- **`medlang_std/`** — Basic standard library (models, protocols)

### Documentation
- **`docs/spec/`** — Formal specifications (grammars, specs)
- **`docs/guides/`** — User guides and tutorials
- **`docs/examples/`** — Code examples
- **`docs/dev/`** — Development documentation (history, architecture)

## 📝 Conventions

### Naming
- **Rust files**: `snake_case.rs`
- **MedLang files**: `snake_case.medlang` or `PascalCase.med`
- **Documentation**: `UPPERCASE.md` or `snake_case.md`

### Directory Structure
- Each main directory has a `README.md` explaining its contents
- Specifications are versioned (v0.1, v1.0, etc.)
- Development documentation is organized by week/phase

### Versioning
- Specifications: `medlang_d_grammar_v1.0.md`
- Releases: `RELEASE_v0.5.0.md` (in `docs/dev/`)
- Status: `STATUS.md` (in `docs/dev/`)

## 🔍 Finding Files

### Want to...
- **Use the compiler?** → `compiler/README.md`
- **See examples?** → `docs/examples/`
- **Understand the language?** → `docs/spec/`
- **Learn how to use?** → `docs/guides/`
- **Develop?** → `docs/dev/ARCHITECTURE.md`
- **See status?** → `docs/dev/STATUS.md`

## 🚀 Maintenance

### Adding New Documentation
- **Specification**: `docs/spec/`
- **User guide**: `docs/guides/`
- **Example**: `docs/examples/`
- **Development docs**: `docs/dev/`

### Adding New Code
- **Compiler**: `compiler/src/`
- **Runtime**: `runtime/`
- **Standard library**: `stdlib/` or `medlang_std/`

### Cleanup
- Temporary files: `.gitignore`
- Backups: Removed or moved to `docs/dev/`
- History: `docs/dev/WEEK_*.md`

---

**Last updated**: Complete repository organization as professional eDSL project
