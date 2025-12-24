# Changelog

All notable changes to MedLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.6.0] - 2024-12-24

### Added
- **FFI Crate**: Complete C ABI interface for external language integration
  - C header file with comprehensive documentation (`ffi/include/medlang.h`)
  - Full FFI implementation (`ffi/src/lib.rs`) with 846 lines
  - Support for C, Python, Julia, R integration
  - ODE solver, quantum chemistry, and fractal analysis functions
- **Workspace Structure**: Cargo workspace with compiler, runtime, and FFI crates
- **Documentation Organization**: Professional eDSL repository structure
  - Organized docs into `spec/`, `guides/`, `examples/`, and `dev/` directories
  - Created comprehensive README files for each directory
  - Added `DEMETRIOS_INTEGRATION.md` documenting MedLang as eDSL for Demetrios
- **Project Structure**: Added `PROJECT_STRUCTURE.md` and `CONTRIBUTING.md`

### Changed
- **Repository Organization**: Major reorganization as professional eDSL project
  - Moved development history to `docs/dev/`
  - Consolidated specifications in `docs/spec/`
  - Organized user guides in `docs/guides/`
  - Consolidated examples in `docs/examples/`
- **README**: Updated to reflect MedLang as eDSL for Demetrios language
- **Documentation**: All documentation now in English with professional structure

### Fixed
- Cleaned up temporary files and backups
- Removed duplicate documentation files

## [0.5.0] - 2024-11-23

### Added
- Phase V1 features: Effect System, Epistemic Computing, Clinical Refinements
- Phase V2: SMT Verification with Z3
- Complete compilation pipeline
- Stan and Julia backends
- 127+ tests passing

[0.6.0]: https://github.com/agourakis82/medlang/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/agourakis82/medlang/releases/tag/v0.5.0

