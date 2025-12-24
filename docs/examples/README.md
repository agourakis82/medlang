# MedLang Examples

This directory contains MedLang code examples organized by category.

## 📋 Available Examples

### Basic PK/PD Models

- **[one_comp_oral_pk.medlang](one_comp_oral_pk.medlang)** — 1-compartment oral PK model (canonical example, 185 lines)
- **[two_comp_iv.medlang](two_comp_iv.medlang)** — 2-compartment IV PK model
- **[depot_im_sc.medlang](depot_im_sc.medlang)** — Depot model (IM/SC)

### PBPK Models

- **[pbpk_2comp_simple.medlang](pbpk_2comp_simple.medlang)** — Simple 2-compartment PBPK
- **[pbpk_5comp_qsp_qm.medlang](pbpk_5comp_qsp_qm.medlang)** — 5-compartment PBPK with QSP/QM
- **[oral_absorption_pbpk.medlang](oral_absorption_pbpk.medlang)** — PBPK with oral absorption
- **[enterohepatic_recirculation.medlang](enterohepatic_recirculation.medlang)** — Enterohepatic recirculation

### Advanced Models

- **[saturable_absorption.medlang](saturable_absorption.medlang)** — Saturable absorption
- **[transit_absorption.medlang](transit_absorption.medlang)** — Transit absorption
- **[tmdd_oncology.medlang](tmdd_oncology.medlang)** — TMDD in oncology
- **[oncology_pbpk_qsp_simple.medlang](oncology_pbpk_qsp_simple.medlang)** — Simple oncology PBPK/QSP
- **[oncology_pbpk_qsp_qm_stub.medlang](oncology_pbpk_qsp_qm_stub.medlang)** — Oncology PBPK/QSP/QM

### Trial Protocols

- **[simple_protocol.medlang](simple_protocol.medlang)** — Simple protocol
- **[example_trial_protocol.medlang](example_trial_protocol.medlang)** — Example trial protocol
- **[oncology_phase2_protocol.medlang](oncology_phase2_protocol.medlang)** — Oncology phase 2 protocol

### ML/QSP Integration

- **[ml_hybrid_model.medlang](ml_hybrid_model.medlang)** — ML hybrid model
- **[pk_qsp_inline.medlang](pk_qsp_inline.medlang)** — PK/QSP inline

### Test Examples

- **[test_composite_minimal.medlang](test_composite_minimal.medlang)** — Minimal composite model test
- **[test_multi_measure.medlang](test_multi_measure.medlang)** — Multiple measures test
- **[test_two_models.medlang](test_two_models.medlang)** — Two models test

### Phase V1

- **[phase_v1/](phase_v1/)** — Phase V1 examples (Effect System, Epistemic Computing, Refinements)

### Other

- **[guideline_experiment.medlang](guideline_experiment.medlang)** — Guideline experiment
- **[regulatory_constraint_analysis.medlang](regulatory_constraint_analysis.medlang)** — Regulatory constraint analysis

## 🚀 How to Use

1. **To learn**: Start with `one_comp_oral_pk.medlang` (canonical example)
2. **For your use case**: Look for similar examples in the list above
3. **To test**: Compile examples with `mlc compile example.medlang`

## 📝 Compiling Examples

```bash
# Compile to Stan
mlc compile docs/examples/one_comp_oral_pk.medlang

# Compile to Julia
mlc compile docs/examples/one_comp_oral_pk.medlang --backend julia

# Check syntax
mlc check docs/examples/one_comp_oral_pk.medlang
```

## 🔍 Structure

Examples are organized by functionality. Some examples may have auxiliary files (`.csv`, `.jl`, `.md`) in the same folder.
