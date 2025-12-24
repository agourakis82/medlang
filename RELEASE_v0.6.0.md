# Release v0.6.0 - Repository Reorganization and FFI Crate

**Release Date**: December 24, 2025

## 🎉 Major Changes

This release represents a major milestone in MedLang's evolution as a professional eDSL project, with complete repository reorganization and a new FFI crate for external language integration.

## ✨ New Features

### FFI Crate (`ffi/`)
- **Complete C ABI Interface**: Full foreign function interface for C, Python, Julia, R integration
- **C Header**: Comprehensive 466-line header file with Doxygen-style documentation
- **FFI Implementation**: 846-line Rust implementation with all core functions
- **Functions Included**:
  - Context management
  - Parsing and compilation
  - Model fitting and simulation
  - ODE solver (RK4)
  - Quantum chemistry (HF, DFT)
  - Fractal analysis (Higuchi FD, DFA)
- **Documentation**: Complete README, examples, and integration guide

### Workspace Structure
- **Cargo Workspace**: Unified workspace for compiler, runtime, and FFI crates
- **Runtime Crate**: Placeholder crate structure for future runtime implementation

### Documentation Organization
- **Professional Structure**: Organized into `spec/`, `guides/`, `examples/`, `dev/`
- **Comprehensive READMEs**: Each directory has its own README
- **Integration Docs**: New `DEMETRIOS_INTEGRATION.md` explaining MedLang as eDSL for Demetrios

## 📁 Repository Changes

### Reorganization
- **Development History**: Moved to `docs/dev/` (55+ files)
- **Specifications**: Consolidated in `docs/spec/` (12 files)
- **User Guides**: Organized in `docs/guides/` (8 files)
- **Examples**: Consolidated in `docs/examples/` (22+ files)

### New Files
- `Cargo.toml` - Workspace configuration
- `CONTRIBUTING.md` - Contribution guidelines
- `PROJECT_STRUCTURE.md` - Project structure documentation
- `CHANGELOG.md` - Changelog following Keep a Changelog format
- `docs/DEMETRIOS_INTEGRATION.md` - MedLang-Demetrios relationship

## 🔧 Improvements

- **Documentation**: All documentation now in English
- **Structure**: Professional eDSL repository organization
- **Cleanup**: Removed temporary files and backups
- **README**: Updated to reflect MedLang as eDSL for Demetrios

## 📊 Statistics

- **128 files changed**: 2,539 insertions, 1,525 deletions
- **FFI Crate**: 846 lines of Rust code, 466 lines of C header
- **Documentation**: 100+ files organized professionally

## 🔗 Links

- **Repository**: https://github.com/agourakis82/medlang
- **Demetrios Language**: https://github.com/Chiuratto-AI/demetrios
- **FFI Documentation**: [ffi/README.md](ffi/README.md)
- **Integration Guide**: [docs/DEMETRIOS_INTEGRATION.md](docs/DEMETRIOS_INTEGRATION.md)

## 🚀 Usage

### Building FFI

```bash
cd ffi
cargo build --release
```

### Using from C

```c
#include "ffi/include/medlang.h"
medlang_ctx_t* ctx = medlang_init();
// ... use FFI functions ...
medlang_free(ctx);
```

### Using from Python

```python
import ctypes
lib = ctypes.CDLL("./ffi/target/release/libmedlang.so")
ctx = lib.medlang_init()
```

See [ffi/EXAMPLES.md](ffi/EXAMPLES.md) for complete examples.

## 📝 Migration Notes

- Development history files moved to `docs/dev/`
- Examples moved to `docs/examples/`
- Specifications moved to `docs/spec/`
- User guides moved to `docs/guides/`

All links in documentation have been updated accordingly.

## 🙏 Acknowledgments

This release represents a significant organizational improvement, making MedLang more accessible and professional for contributors and users.

---

**Full Changelog**: [CHANGELOG.md](CHANGELOG.md)

