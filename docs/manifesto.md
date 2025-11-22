# MedLang / Beagle Manifesto (v0.1)

## Motivation

Modern medicine is fractured across levels:

- Quantum chemistry and molecular pharmacology (HF, DFT, QM/MM, MD…)
- Pharmacokinetics/pharmacodynamics (PK/PD, compartmental models)
- Physiology and biophysics (ODEs, PDEs, PINNs)
- Clinical data and risk models (MLP, GNN, Bayesian, causal)
- Guidelines, protocols and decision support (CQL, Arden, FHIR, etc.)

Each level has its own tools, file formats, and fragile integrations.
There is no single, coherent language in which a clinician-scientist can:

- Describe a molecule at quantum level
- Derive PK parameters from QM calculations
- Embed those in physiological/PK/PD models
- Combine them with real clinical data and AI models
- Express protocols and endpoints
- And run the whole stack deterministically & reproducibly on HPC hardware.

**MedLang** is an attempt to fill this gap.

## Vision

MedLang is a **medical-native, quantum-aware, AI-first systems language**
in which:

- Patients, cohorts, protocols, doses, endpoints, and timelines are
  **first-class types**.
- Tensors, probability measures, fractal dimensions, and quantum states
  are also **first-class types**.
- Safety is enforced at the language level (units, types, ownership).
- Execution is explicitly mapped to CPU/GPU/HPC devices.

**Beagle** is the reference application and cockpit for MedLang:
a place where clinicians interact with MedLang code, not with low-level
numerical libraries.

## Principles

1. **Clinical-native semantics**  
   Code should read like structured clinical reasoning, but compile down
   to strongly-typed, optimized kernels.

2. **Hardware transparency**  
   The mapping to devices (CPU/GPU) is explicit and controllable, without
   leaking CUDA/PTX details into clinical code.

3. **Dimensional and clinical safety**  
   Doses, units and protocol constraints are enforced by the type system.

4. **Quantum pharmacology as a first-class citizen**  
   HF/DFT/post-HF/QM/MM methods are exposed as language operators,
   not opaque external binaries.

5. **AI as a language primitive**  
   Neural networks (MLP, GNN, PINN, etc.) and autodiff are built into
   the language core.

6. **Fractal and probabilistic reasoning**  
   Fractal operators and probabilistic kernels are native, allowing
   complex behavior of signals and distributions to be part of the
   objective landscape.

## Long-term goal

To provide a language and runtime that can:

- Serve as a reference implementation for computational psychiatry,
  computational pharmacology and ICU medicine.
- Underpin Q1-level scientific work with the same rigor as existing
  quantum chemistry packages and ML frameworks, but in a single
  coherent stack.

This repository is the first step: formalizing the **core language**
and **runtime architecture** that everything else will rely on.
