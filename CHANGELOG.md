# Changelog

All notable changes to MedLang will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-11-27

### Added

#### Grammar Extensions (Track D)
- **Route Definition**: Explicit administration route specification
  - Supported routes: IV, ORAL, IM, SC, INFUSION
  - Syntax: route: ORAL

- **Absorption Block**: First-order oral absorption modeling
  - ka: Absorption rate constant (required)
  - f: Bioavailability fraction (default 1.0)
  - lag: Absorption lag time (default 0.0)

- **First-Pass Block**: Pre-systemic metabolism
  - fg: Gut availability (fraction escaping gut metabolism)
  - fh: Hepatic availability (fraction escaping hepatic first-pass)
  - Effective bioavailability: F_eff = f * fg * fh

- **Organ Definition**: PBPK tissue compartments
  - volume: Anatomical tissue volume
  - flow: Blood flow rate to organ
  - kp: Tissue:plasma partition coefficient

- **Clearance Definition**: Organ-specific elimination
  - Types: hepatic, renal
  - cl: Clearance value
  - extraction: Extraction ratio (optional)

#### New Types
- Fraction: Dimensionless value in [0, 1]
- FlowRate: Volume per time (L/h)
- AUCUnit: Mass * Time / Volume (mg*h/L)

#### New Functions
- integral(x): Time integral for AUC calculation
- max(x, y): Maximum of two values
- min(x, y): Minimum of two values
- clamp(x, lo, hi): Bounded value

#### Examples
- oral_absorption_pbpk.medlang: Metoprolol with CYP2D6 first-pass metabolism

### Reference Implementation
- Darwin PBPK Platform (Julia): github.com/agourakis82/darwin-pbpk-platform
- 15-compartment ODE system with gut lumen depot
- Validated against 572 drugs (GMFE 2.02, Cmax 24.3% within 2-fold)

## [0.1.0] - 2024-08-15

### Added

#### Core Grammar (Track D)
- Model definition with states, parameters, ODEs, observables
- Population definition with random effects and covariate models
- Measure definition for error models
- Timeline definition for dosing and sampling schedules
- Cohort definition linking population, timeline, and data

#### Type System
- Unit types: Mass, Volume, Time, Clearance, RateConst
- Derived types: DoseMass, ConcMass
- Generic Quantity type with unit inference

#### Built-in Functions
- Mathematical: exp, log, pow, sqrt
- Statistical: Normal_logpdf

#### Examples
- one_comp_oral_pk.medlang: Basic 1-compartment oral PK
- two_comp_iv.medlang: 2-compartment IV bolus
- pbpk_2comp_simple.medlang: Simple PBPK model
