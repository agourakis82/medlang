# MedLang-D Grammar Specification v1.0

## Overview

MedLang-D v1.0 introduces comprehensive support for:
- **QSP Constructs**: Target-mediated drug disposition (TMDD), tumor growth-kill models, receptor-ligand dynamics
- **ML Integration**: Neural network parameter prediction, surrogate models, uncertainty quantification
- **Track C Operators**: Clinical trial simulation, virtual populations, covariate models, bioequivalence analysis

This version enables full mechanistic-ML hybrid modeling for drug development.

## Version History

| Version | Features |
|---------|----------|
| v0.2 | Oral absorption, first-pass metabolism |
| v0.3 | Transit compartments, enterohepatic recirculation |
| v0.4 | Saturable absorption, IM/SC depot |
| **v1.0** | **QSP, ML submodels, Track C operators** |

---

## 1. QSP Constructs

### 1.1 Target-Mediated Drug Disposition (TMDD)

For drugs where target binding significantly affects pharmacokinetics.

#### Syntax

    tmdd <name> {
        // Binding kinetics
        kon: <value> 1/nM/h,      // Association rate
        koff: <value> 1/h,        // Dissociation rate
        
        // Target turnover
        ksyn: <value> nM/h,       // Target synthesis rate
        kdeg: <value> 1/h,        // Target degradation rate
        
        // Complex fate
        kint: <value> 1/h,        // Internalization rate
        
        // Optional: approximations
        approximation: <full|qss|qe>  // Default: full
    }

#### Example

    tmdd Pembrolizumab_PD1 {
        kon: 0.5 1/nM/h,
        koff: 0.001 1/h,          // High affinity (Kd = 2 pM)
        ksyn: 0.1 nM/h,
        kdeg: 0.02 1/h,           // R0 = 5 nM
        kint: 0.1 1/h
    }

#### Derived Values

- Kd = koff / kon (equilibrium dissociation constant)
- R0 = ksyn / kdeg (baseline target concentration)
- Target occupancy: TO = L / (L + Kd)

---

### 1.2 Tumor Growth-Kill Model

For oncology QSP applications with Simeoni transit model.

#### Syntax

    tumor <name> {
        // Growth model
        growth: <exponential|logistic|gompertz>,
        kg: <value> 1/day,        // Growth rate
        kmax: <value> mm3,        // Carrying capacity (logistic/gompertz)
        
        // Kill model
        kill: <emax|linear|simeoni>,
        kk: <value> 1/day,        // Kill rate constant
        emax: <value>,            // Maximum effect (0-1)
        ec50: <value> ng/mL,      // Half-maximal concentration
        gamma: <value>,           // Hill coefficient (default: 1)
        
        // Simeoni transit
        n_transit: <int>,         // Number of damage compartments
        ktr: <value> 1/day,       // Transit rate
        
        // Initial condition
        tumor0: <value> mm3       // Initial tumor volume
    }

#### Example

    tumor Colorectal_5FU {
        growth: logistic,
        kg: 0.03 1/day,
        kmax: 10000 mm3,
        
        kill: simeoni,
        kk: 0.15 1/day,
        emax: 0.8,
        ec50: 200 ng/mL,
        
        n_transit: 3,
        ktr: 0.5 1/day,
        
        tumor0: 200 mm3
    }

#### Response Classification (RECIST-like)

- Partial Response: change <= -30%
- Stable Disease: -30% < change < 20%
- Progressive Disease: change >= 20%

---

### 1.3 Receptor-Ligand Dynamics

General receptor binding and signaling.

#### Syntax

    receptor <name> {
        kon: <value> 1/nM/h,
        koff: <value> 1/h,
        rtot: <value> nM,         // Total receptor
        krecycle: <value> 1/h,    // Receptor recycling
        kintern: <value> 1/h      // Internalization
    }

### 1.4 Enzyme Turnover

For CYP induction/inhibition modeling.

#### Syntax

    enzyme <name> {
        ksyn: <value> /h,
        kdeg: <value> 1/h,
        
        // Induction
        emax_ind: <value>,        // Max fold induction
        ec50_ind: <value>,
        
        // Inhibition
        ki: <value>,              // Reversible Ki
        kinact: <value> 1/h       // Inactivation (TDI)
    }

---

## 2. ML Integration

### 2.1 Parameter Predictor

Neural network-based parameter prediction from molecular structure.

#### Syntax

    ml_predictor <name> {
        model_type: <gnn|chemberta|multimodal>,
        model_path: "<path>",     // Optional: trained model
        
        predict: [<param_list>],  // Parameters to predict
        
        // Fallback QSPR
        fallback: <poulin_theil|rodgers_rowland|qspr>
    }

#### Example

    ml_predictor Kp_Predictor {
        model_type: multimodal,
        predict: [kp_liver, kp_adipose, kp_brain],
        fallback: poulin_theil
    }

### 2.2 PK Surrogate Model

Fast neural network replacement for ODE solving.

#### Syntax

    surrogate <name> {
        inputs: [dose, logP, mw, clearance],
        outputs: [cmax, tmax, auc, half_life],
        
        model_path: "<path>",
        trained: <true|false>
    }

### 2.3 Uncertainty Quantification

#### Syntax

    uncertainty <name> {
        method: <ensemble|dropout|conformal>,
        n_samples: <int>,
        confidence: <0.0-1.0>     // Default: 0.95
    }

---

## 3. Track C: Clinical Trial Operators

### 3.1 Covariate Effects

#### Syntax

    covariate <name> on <parameter> {
        type: <power|exponential|categorical|linear>,
        reference: <value>,
        theta: <value>,
        
        // For categorical
        factors: {
            "<category1>": <value>,
            "<category2>": <value>
        }
    }

#### Standard Allometric Scaling

    // Built-in constants
    ALLOMETRIC_CL: power(WT/70, 0.75)
    ALLOMETRIC_V:  power(WT/70, 1.0)

#### Example

    covariate WT on CL {
        type: power,
        reference: 70 kg,
        theta: 0.75
    }
    
    covariate SEX on CL {
        type: categorical,
        factors: {
            "M": 1.0,
            "F": 0.85
        }
    }

---

### 3.2 Inter-Individual Variability (IIV)

#### Syntax

    iiv <parameter> {
        distribution: <exponential|proportional|additive>,
        omega: <value>,           // Standard deviation
        correlation_group: <int>  // For correlated parameters
    }

#### Example

    iiv CL {
        distribution: exponential,
        omega: 0.3                // ~30% CV
    }
    
    iiv V {
        distribution: exponential,
        omega: 0.25,
        correlation_group: 1      // Correlated with CL
    }

---

### 3.3 Virtual Population

#### Syntax

    virtual_population <name> {
        n_subjects: <int>,
        
        demographics {
            WT: normal(70, 15, min=40, max=150),
            AGE: uniform(18, 65),
            SEX: categorical(M=0.5, F=0.5),
            CRCL: normal(100, 25, min=30, max=150)
        }
        
        iiv: [<iiv_spec_list>]
    }

#### Example

    virtual_population HealthyAdults {
        n_subjects: 100,
        
        demographics {
            WT: normal(70, 15),
            AGE: uniform(18, 55),
            SEX: categorical(M=0.5, F=0.5),
            CRCL: normal(110, 20)
        }
        
        iiv: [iiv_CL, iiv_V]
    }

---

### 3.4 Dosing Regimen

#### Syntax

    regimen <name> {
        dose: <value> <unit>,
        route: <IV|ORAL|SC|IM|INFUSION>,
        interval: <value> h,
        n_doses: <int>,
        infusion_duration: <value> h  // For IV infusion
    }

#### Example

    regimen QD_Oral {
        dose: 100 mg,
        route: ORAL,
        interval: 24 h,
        n_doses: 14
    }
    
    regimen IV_Infusion {
        dose: 500 mg,
        route: INFUSION,
        interval: 168 h,          // Weekly
        n_doses: 4,
        infusion_duration: 1 h
    }

---

### 3.5 Trial Design

#### Syntax

    trial <name> {
        arms: [<arm_list>],
        duration: <value> h,
        endpoints: [<endpoint_list>]
    }
    
    arm <name> {
        regimen: <regimen_name>,
        n_subjects: <int>,
        sampling_times: [<time_list>]
    }

#### Example

    trial Phase1_SAD {
        arms: [
            arm Dose1 { regimen: Low_Dose, n_subjects: 8 },
            arm Dose2 { regimen: Mid_Dose, n_subjects: 8 },
            arm Dose3 { regimen: High_Dose, n_subjects: 8 },
            arm Placebo { regimen: Placebo, n_subjects: 6 }
        ],
        duration: 168 h,
        endpoints: [Cmax, AUC, Tmax, half_life]
    }

---

### 3.6 Bioequivalence Analysis

#### Syntax

    bioequivalence <name> {
        test: <arm_name>,
        reference: <arm_name>,
        parameters: [AUC, Cmax],
        alpha: 0.1,               // For 90% CI
        limits: [0.80, 1.25]      // BE acceptance
    }

#### Example

    bioequivalence Generic_vs_Brand {
        test: Generic_Arm,
        reference: Brand_Arm,
        parameters: [AUC, Cmax],
        alpha: 0.1,
        limits: [0.80, 1.25]
    }

---

### 3.7 Exposure-Response Analysis

#### Syntax

    exposure_response <name> {
        exposure: <AUC|Cmax|Cavg|Ctrough>,
        response: <efficacy_endpoint>,
        model: <emax|linear|logistic>,
        
        // Emax model parameters
        e0: <value>,
        emax: <value>,
        ec50: <value>,
        gamma: <value>
    }

---

## 4. Complete Example

    // Oncology PBPK-QSP-ML Model
    model Pembrolizumab_NSCLC {
        
        // Drug properties (ML-predicted)
        drug Pembrolizumab {
            mw: 149000,
            type: "mAb"
        }
        
        ml_predictor tissue_distribution {
            model_type: gnn,
            predict: [kp_tumor, kp_spleen]
        }
        
        // TMDD for PD-1 binding
        tmdd PD1_Binding {
            kon: 0.5 1/nM/h,
            koff: 0.001 1/h,
            ksyn: 0.1 nM/h,
            kdeg: 0.02 1/h,
            kint: 0.1 1/h
        }
        
        // Tumor dynamics
        tumor NSCLC_Response {
            growth: gompertz,
            kg: 0.01 1/day,
            kmax: 50000 mm3,
            
            kill: emax,
            kk: 0.05 1/day,
            emax: 0.7,
            ec50: 10 nM,            // Based on target occupancy
            
            tumor0: 5000 mm3
        }
        
        // Population
        virtual_population NSCLC_Patients {
            n_subjects: 200,
            demographics {
                WT: normal(75, 18),
                AGE: uniform(45, 80),
                ECOG: categorical(0=0.4, 1=0.5, 2=0.1)
            }
        }
        
        // Trial
        regimen Q3W {
            dose: 200 mg,
            route: INFUSION,
            interval: 504 h,        // 3 weeks
            n_doses: 12,
            infusion_duration: 0.5 h
        }
        
        trial Phase2_NSCLC {
            arms: [
                arm Treatment { regimen: Q3W, n_subjects: 100 }
            ],
            duration: 6048 h,       // 36 weeks
            endpoints: [ORR, PFS, OS, target_occupancy]
        }
        
        // Exposure-response
        exposure_response Efficacy {
            exposure: Cavg,
            response: ORR,
            model: emax,
            ec50: 20 ug/mL
        }
    }

---

## 5. Validation Rules

### QSP
- TMDD: 0 < kon, koff, ksyn, kdeg, kint
- Tumor: 0 < kg, kmax, ec50; 0 <= emax <= 1

### ML
- model_type must be one of: gnn, chemberta, multimodal
- confidence must be in (0, 1)

### Track C
- Covariate theta typically in [-2, 2] for power model
- IIV omega typically in [0.1, 1.0]
- n_subjects > 0
- BE limits must satisfy lower < 1 < upper

---

## Reference Implementation

Darwin PBPK Platform (Julia) v1.0
- Repository: github.com/agourakis82/darwin-pbpk-platform
- ODE state vector: 37+ states (organs + gut + transit + bile + depot + TMDD + tumor)
- Full backward compatibility with v0.2-v0.4 models
