# MedLang Documentation

This directory contains all documentation for the MedLang project, organized by category.

**Note**: MedLang is an embedded Domain-Specific Language (eDSL) for the [Demetrios (D) language](https://github.com/Chiuratto-AI/demetrios). See [DEMETRIOS_INTEGRATION.md](DEMETRIOS_INTEGRATION.md) for details.

## 📂 Structure

### `spec/` — Formal Specifications
Technical and formal language specifications:
- Grammars (EBNF)
- Core language specifications
- Extension specifications (pharmacometrics, QM, etc.)
- Data schemas

### `guides/` — User Guides
Practical guides for users:
- Complete end-to-end workflow
- Quick references
- Feature-specific usage guides
- Tutorials

### `examples/` — Code Examples
Collection of MedLang examples:
- PK/PD models
- Clinical trial protocols
- PBPK/QSP models
- ML/RL integration examples

### `dev/` — Development Documentation
Internal documentation for developers:
- Compiler architecture
- Project status
- Development history
- Roadmaps and future plans

## 🚀 Quick Start

1. **New user?** Start with:
   - [Complete Workflow](guides/WORKFLOW.md)
   - [Quick Reference](guides/QUICKREF.md)
   - [Canonical Example](examples/one_comp_oral_pk.medlang)

2. **Want to understand the language?** Read:
   - [Grammar V1.0](spec/medlang_d_grammar_v1.0.md)
   - [Core Specification](spec/medlang_core_spec_v0.1.md)

3. **Want to contribute?** See:
   - [Architecture](dev/ARCHITECTURE.md)
   - [Status](dev/STATUS.md)

## 📚 Complete Index

### Specifications
- [Grammar V0 (Minimal)](spec/medlang_d_minimal_grammar_v0.md)
- [Grammar V0.2](spec/medlang_d_grammar_v0.2.md)
- [Grammar V0.3](spec/medlang_d_grammar_v0.3.md)
- [Grammar V0.4](spec/medlang_d_grammar_v0.4.md)
- [Grammar V1.0](spec/medlang_d_grammar_v1.0.md)
- [Core Specification v0.1](spec/medlang_core_spec_v0.1.md)
- [Pharmacometrics/QSP v0.1](spec/medlang_pharmacometrics_qsp_spec_v0.1.md)
- [Quantum Pharmacology v0.1](spec/medlang_qm_pharmacology_spec_v0.1.md)
- [PBPK Design v0.1](spec/pbpk_design_v0.1.md)
- [Trial Data Schema v0.1](spec/trial_data_schema_v0.1.md)
- [Standards Mappings](spec/STANDARDS_MAPPINGS.md)

### Guides
- [Complete Workflow](guides/WORKFLOW.md)
- [Quick Reference](guides/QUICKREF.md)
- [Quick Reference V1](guides/QUICK_REFERENCE_V1.md)
- [Implementation Guide V0](guides/IMPLEMENTATION_GUIDE_V0.md)
- [PBPK User Guide](guides/pbpk_user_guide.md)
- [Quantum Stub Guide](guides/quantum_stub_guide.md)
- [User Guide: Analyze Trial](guides/USER_GUIDE_ANALYZE_TRIAL.md)

### Development
- [Architecture](dev/ARCHITECTURE.md)
- [Status](dev/STATUS.md)
- [Changelog](dev/CHANGELOG.md) (if exists)
- [Roadmap](dev/NEXT_STEPS_ROADMAP.md)

## 🔍 Finding Documentation

- **Want to learn how to use?** → `guides/`
- **Want to understand the specification?** → `spec/`
- **Want to see examples?** → `examples/`
- **Want to develop?** → `dev/`
- **Want to understand MedLang-Demetrios relationship?** → [DEMETRIOS_INTEGRATION.md](DEMETRIOS_INTEGRATION.md)
