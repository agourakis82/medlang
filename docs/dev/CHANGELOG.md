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

## [0.4.0] - 2024-11-27

### Added

#### Non-linear Saturable Absorption (Michaelis-Menten)
- **absorption saturable block**: Transporter-limited uptake modeling
  - vmax: Maximum absorption rate (mg/h)
  - km: Michaelis constant (mg)
  - passive_ka: Passive diffusion rate constant (optional)
  - fa, fg, fh, lag: Same as absorption block
- Dose-dependent bioavailability automatically calculated
- Clinical relevance: BCS Class III drugs (gabapentin, metformin)

#### Multi-Compartment Depot (IM/SC Routes)
- **depot block**: Injectable depot modeling
  - route: Administration route (:IM, :SC)
  - n_depots: Number of absorption compartments (1-3)
  - ka: Vector of absorption rate constants
  - fractions: Vector of dose fractions per depot (sum to 1.0)
  - f: Overall bioavailability
  - lag: Initial lag time
- Supports flip-flop kinetics (ka << kel)
- Clinical relevance: Biologics, LAI antipsychotics, insulin

#### New Grammar Constructs
- Saturable absorption: 
- Depot definition: 
- Route symbols: :IV, :ORAL, :IM, :SC, :INFUSION

#### New Functions
- michaelis_menten(s, vmax, km): MM kinetics
- dose_normalized_auc(auc, dose): For saturation assessment
- flip_flop_indicator(ka, kel): Detect absorption-limited kinetics

#### Examples
- saturable_absorption.medlang: Gabapentin with LAT1 transporter saturation
- depot_im_sc.medlang: Adalimumab dual-depot SC model with flip-flop kinetics

### Validation Results
- Saturable absorption: Tmax delayed at high dose (10.18h vs 0.48h), PASS
- Dose-normalized AUC decreases with dose (0.119 → 0.108), PASS  
- IM depot flip-flop kinetics confirmed (Tmax 2.58h), PASS
- Route ordering verified: Oral Tmax < IM Tmax < SC Tmax, PASS
- 3/5 validation tests passed (core functionality working)

### Reference Implementation
- Darwin PBPK Platform (Julia) v0.4
- ODE state vector extended to 30+ states (organs + gut + transit + bile + depot)
- Full backward compatibility with v0.2/v0.3 models

## [1.0.0] - 2024-11-27

### Added

#### QSP Constructs
- **TMDD (Target-Mediated Drug Disposition)**
  - Full, QSS, and QE approximations
  - Parameters: kon, koff, ksyn, kdeg, kint
  - Derived: Kd, R0, target occupancy
  - Clinical relevance: mAbs, receptor-targeted drugs

- **Tumor Growth-Kill Models**
  - Growth: exponential, logistic, gompertz
  - Kill: emax, linear, simeoni transit
  - Response classification (RECIST-like)
  - Parameters: kg, kmax, kk, emax, ec50, gamma

- **Receptor-Ligand Dynamics**
  - General binding kinetics
  - Receptor recycling and internalization
  - Occupancy calculations

- **Enzyme Turnover**
  - CYP induction (Emax model)
  - Reversible and time-dependent inhibition

#### ML Integration
- **ML Parameter Predictor**
  - Model types: gnn, chemberta, multimodal
  - Predict: Kp, clearance, solubility
  - QSPR fallback methods

- **PK Surrogate Model**
  - Neural network fast prediction
  - Inputs: dose, CL, Vd, ka, F
  - Outputs: Cmax, Tmax, AUC, half-life

- **Uncertainty Quantification**
  - Methods: ensemble, dropout, conformal
  - Prediction intervals with confidence levels

- **Neural ODE Correction**
  - Hybrid mechanistic-ML models
  - Learn residuals from data

#### Track C: Clinical Trial Operators
- **Covariate Effects**
  - Types: power, exponential, categorical, linear
  - Built-in allometric scaling (CL^0.75, V^1.0)

- **Inter-Individual Variability (IIV)**
  - Distributions: exponential, proportional, additive
  - Correlation groups for multivariate

- **Virtual Population Generator**
  - Configurable demographics
  - Distribution types: normal, uniform, lognormal, categorical
  - Automatic IIV sampling

- **Dosing Regimen**
  - Routes: IV, ORAL, SC, IM, INFUSION
  - Interval-based repeated dosing
  - Infusion duration support

- **Trial Design**
  - Multi-arm studies
  - Customizable endpoints
  - Sampling schedule specification

- **Bioequivalence Analysis**
  - Test vs reference comparison
  - 90% CI for geometric mean ratio
  - BE limits (80-125%)

- **Exposure-Response Analysis**
  - Emax model fitting
  - Efficacy correlation

#### New Examples
- tmdd_oncology.medlang: Pembrolizumab NSCLC with TMDD + tumor dynamics
- ml_hybrid_model.medlang: ML-augmented PBPK for drug discovery

### Validation Results
- QSP structs: TMDD, tumor growth-kill functional (PASS)
- ML prediction: Kp, clearance QSPR fallback working (PASS)
- PK surrogate: Analytical solutions verified (PASS)
- Covariate effects: Allometric scaling correct (PASS)
- Trial design: Dosing times, arm specs functional (PASS)
- **5/5 validation tests passed**

### Reference Implementation
- Darwin PBPK Platform (Julia) v1.0
- ODE state vector: 37+ states
- Full backward compatibility with v0.2-v0.4
- Repository: github.com/agourakis82/darwin-pbpk-platform
