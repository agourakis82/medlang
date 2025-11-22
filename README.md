# MedLang & Beagle Stack

**MedLang** is a medical-native, GPU/HPC-accelerated programming language
designed to unify:

- Clinical reasoning (patients, cohorts, protocols, endpoints)
- Quantum pharmacology and molecular modelling (HF/DFT/post-HF/QM/MM)
- AI models (MLP, GNN, PINN) with native autodiff and optimization
- Probabilistic kernels and measures
- Fractal analysis of physiological and clinical signals

**Beagle** is the reference application and IDE/cockpit for MedLang:
a clinical environment where protocols, simulations and models are written
in MedLang and executed on real hardware (CPU/GPU/HPC) with full auditability.

This repository contains:

- `docs/` — Manifesto and formal specifications (core language, extensions)
- `compiler/` — Frontend, IRs (CIR/NIR), and MLIR/LLVM lowering
- `runtime/` — Device/runtime layer (CPU/GPU, QM backends, fractal kernels)
- `beagle/` — Reference clinical application and UI

> Status: experimental, research-grade. The first goal is to build a
> mathematically sound and clinically meaningful core, then iterate.

## High-level vision

MedLang aims to become a *single coherent language* where:

- **Molecules**, **patients**, **cohorts**, **protocols**, **states of mind**,
  **probability measures**, **quantum operators**, and **fractal dimensions**
  are all first-class citizens.

- Code that starts at quantum-level pharmacology (HF/DFT/QM/MM) can
  propagate up to PK/PD, physiological dynamics, AI risk models, and
  clinical decision support — without leaving the language.

- Safety is enforced at the language level:
  - typed units for doses and physiological quantities
  - ownership/borrowing à la Rust to avoid memory bugs
  - deterministic execution by default
  - explicit control of randomness and approximation

## Roadmap (short version)

1. **Core spec and IRs**  
   - Finalize core calculus, type system, and operational semantics  
   - Design Clinical IR (CIR) and Numeric IR (NIR)

2. **Core compiler & runtime**  
   - Parser + typechecker  
   - NIR → MLIR → LLVM/CPU/GPU lowering  
   - Minimal runtime for tensors, cohorts, timelines

3. **AI & PDE**  
   - Native `Model<X,Y>`, MLP, GNN, PINN  
   - Autodiff and optimization (including Riemannian/natural gradient)

4. **Quantum & fractal**  
   - QM operators (HF/DFT) and QM→PK probabilistic compilation  
   - Fractal operators on timelines and trajectories

5. **Beagle v1**  
   - Clinical cockpit to edit/run MedLang code  
   - Example workflows: ICU vasopressors, PK/PD, psychiatric trajectories

---

Work in progress. Everything here is subject to change as the theory and
implementation converge.
