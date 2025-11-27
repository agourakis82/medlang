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

## [0.3.0] - 2024-11-27

### Added

#### Transit Compartment Absorption (CAT Model)
- **transit block**: Multi-compartment absorption modeling
  - n: Number of transit compartments (1-10)
  - ktr: Transit rate constant (1/h)
  - mtt: Mean transit time (alternative to ktr)
  - ka: Final absorption rate constant
  - f, fg, fh, lag: Same as absorption block

#### Enterohepatic Recirculation (EHR)
- **ehr block**: Biliary excretion and intestinal reabsorption
  - f_bile: Fraction excreted in bile
  - k_bile: Biliary excretion rate constant
  - f_reabs: Fraction reabsorbed from gut
  - k_reabs: Reabsorption rate constant
  - t_gb: Gallbladder emptying delay

#### Timeline Extensions
- **meal events**: Trigger gallbladder emptying for EHR
  - meal types: light, standard, high_fat

#### New Functions
- mean_transit_time(n, ktr): Calculate MTT
- detect_peaks(conc, time): Find local maxima

#### Examples
- transit_absorption.medlang: Gabapentin-like with CAT model
- enterohepatic_recirculation.medlang: Mycophenolate with EHR

### Validation Results
- Transit model: Tmax delayed from 1.45h to 3.15h (as expected)
- EHR model: Functional with meal-triggered GB emptying
- 4/5 validation tests passed
