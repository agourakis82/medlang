# PBPK User Guide (Week 7)

## Overview

**Physiologically-Based Pharmacokinetic (PBPK)** modeling in MedLang enables mechanistic representation of drug distribution across multiple organs and tissues. Combined with **Quantum Stub integration** (Week 6), MedLang now supports the complete chain:

```
Quantum Mechanics → Tissue Partition → PBPK Distribution → QSP Dynamics
```

## What is PBPK?

PBPK models describe drug movement through the body using:

1. **Anatomical structure**: Explicit organ compartments (liver, kidney, tumor, etc.)
2. **Physiological parameters**: Blood flows, organ volumes
3. **Physicochemical properties**: Tissue:plasma partition coefficients (Kp)

Unlike compartmental PK (which uses empirical compartments), PBPK compartments correspond to **real organs**, making predictions more mechanistic and scalable.

## 5-Compartment PBPK Model

MedLang's canonical PBPK structure includes:

| Compartment | Purpose | Volume (70 kg) | Flow (70 kg) |
|-------------|---------|----------------|--------------|
| **Plasma** | Central blood | 3.0 L | - |
| **Tumor** | Target tissue (QSP) | 0.5 L | 0.5 L/h |
| **Liver** | Metabolism | 1.8 L | 90 L/h |
| **Kidney** | Excretion | 0.3 L | 70 L/h |
| **Rest** | Other tissues | 35 L | 150 L/h |

### Mass Balance ODEs

Each compartment follows:

```
dA_tissue/dt = Q_tissue * (C_plasma - C_tissue_vein)
```

Where:
- `A_tissue`: Amount of drug in tissue [mass]
- `Q_tissue`: Blood flow to tissue [volume/time]
- `C_plasma`: Plasma concentration [concentration]
- `C_tissue_vein`: Venous concentration leaving tissue [concentration]

The venous concentration depends on the partition coefficient:

```
C_tissue_vein = (A_tissue / V_tissue) / Kp_tissue
```

This represents equilibration between tissue and blood.

## MedLang PBPK Syntax

### Complete 5-Compartment Example

```medlang
model PBPK_5Comp {
    // PBPK States
    state A_plasma : DoseMass
    state A_tumor  : DoseMass
    state A_liver  : DoseMass
    state A_kidney : DoseMass
    state A_rest   : DoseMass
    
    // Parameters
    param V_plasma : Volume
    param V_tumor  : Volume
    param V_liver  : Volume
    param V_kidney : Volume
    param V_rest   : Volume
    
    param Q_tumor  : Clearance
    param Q_liver  : Clearance
    param Q_kidney : Clearance
    param Q_rest   : Clearance
    param CL_renal : Clearance
    
    param Kp_tumor  : f64
    param Kp_liver  : f64
    param Kp_kidney : f64
    param Kp_rest   : f64
    
    // Intermediate concentrations
    let C_plasma = A_plasma / V_plasma
    
    let C_tumor_tissue = A_tumor / V_tumor
    let C_tumor_vein   = C_tumor_tissue / Kp_tumor
    
    let C_liver_tissue = A_liver / V_liver
    let C_liver_vein   = C_liver_tissue / Kp_liver
    
    let C_kidney_tissue = A_kidney / V_kidney
    let C_kidney_vein   = C_kidney_tissue / Kp_kidney
    
    let C_rest_tissue = A_rest / V_rest
    let C_rest_vein   = C_rest_tissue / Kp_rest
    
    // PBPK ODEs
    dA_plasma/dt = Q_tumor  * (C_tumor_vein  - C_plasma)
                 + Q_liver  * (C_liver_vein  - C_plasma)
                 + Q_kidney * (C_kidney_vein - C_plasma)
                 + Q_rest   * (C_rest_vein   - C_plasma)
                 - CL_renal * C_plasma
    
    dA_tumor/dt  = Q_tumor  * (C_plasma - C_tumor_vein)
    dA_liver/dt  = Q_liver  * (C_plasma - C_liver_vein)
    dA_kidney/dt = Q_kidney * (C_plasma - C_kidney_vein)
    dA_rest/dt   = Q_rest   * (C_plasma - C_rest_vein)
    
    // Observables
    obs C_plasma_obs : ConcMass = C_plasma
    obs C_tumor_obs  : ConcMass = C_tumor_tissue
}
```

### Key Features

#### 1. Let Bindings for Intermediates

```medlang
let C_plasma = A_plasma / V_plasma
let C_tumor_tissue = A_tumor / V_tumor
let C_tumor_vein = C_tumor_tissue / Kp_tumor
```

These intermediate values:
- Are computed **before** the ODEs
- Can be used in ODE right-hand sides
- Are emitted in the Stan ODE function
- Make equations more readable and avoid redundant calculations

#### 2. Dimensional Types

MedLang enforces dimensional consistency:
- `DoseMass`: Amount of drug in compartment
- `Volume`: Organ volume
- `Clearance`: Blood flow or clearance (L/h)
- `ConcMass`: Drug concentration (mass/volume)
- `f64`: Dimensionless (Kp values)

## Allometric Scaling

Physiological parameters scale with body weight using power laws:

### Scaling Equations

```medlang
population PBPK_Pop {
    // Reference values for 70 kg
    param V_plasma_ref : Volume
    param Q_liver_ref  : Clearance
    
    // Covariate
    input WT : Mass
    
    bind_params(patient) {
        let w = patient.WT / 70.0_kg
        
        // Volumes scale linearly (exponent = 1.0)
        model.V_plasma = V_plasma_ref * w
        model.V_liver  = V_liver_ref  * w
        
        // Flows scale with exponent = 0.75
        model.Q_liver  = Q_liver_ref  * pow(w, 0.75)
        model.CL_renal = CL_renal_ref * pow(w, 0.75)
    }
}
```

### Standard Allometric Exponents

| Parameter Type | Exponent | Rationale |
|----------------|----------|-----------|
| Organ volumes | 1.0 | Linear with body mass |
| Blood flows | 0.75 | Metabolic scaling (Kleiber's law) |
| Clearances | 0.75 | Driven by blood flow |
| Partition coefficients | 0.0 | Physicochemical property (constant) |

## QM-Informed Kp Integration

### The Problem

Traditionally, Kp values are:
- Estimated from data (many parameters)
- Literature values (may not match your compound)
- Predicted empirically (e.g., Rodgers-Rowland equations)

### The Solution: Quantum Partition Coefficients

With a QM stub providing `ΔG_part`, we compute:

```
Kp_tumor = exp(-ΔG_part_plasma_tumor / (R * T))
```

### MedLang Implementation

**QM Stub** (`data/drug_qm.json`):
```json
{
  "drug_id": "LIG001",
  "target_id": "EGFR",
  "Kd_M": 2.5e-9,
  "dG_part_plasma_tumor_kcal_per_mol": -0.8,
  "T_ref_K": 310.0
}
```

**MedLang Model**:
```medlang
population PBPK_QSP_Pop {
    // Kp_tumor is NOT estimated - it comes from QM stub
    param Kp_liver_pop  : f64  // Estimated
    param Kp_kidney_pop : f64  // Estimated
    param Kp_rest_pop   : f64  // Estimated
    
    bind_params(patient) {
        // QM-informed (external constant)
        model.Kp_tumor = Kp_tumor_QM  
        
        // Estimated from data
        model.Kp_liver  = Kp_liver_pop
        model.Kp_kidney = Kp_kidney_pop
        model.Kp_rest   = Kp_rest_pop
    }
}
```

**Compile with QM stub**:
```bash
mlc compile pbpk_model.medlang \
    --qm-stub data/drug_qm.json \
    -v
```

**Output**:
```
  ✓ QM stub loaded: LIG001 targeting EGFR
    Kd = 2.50e-9 M
    Kp_tumor = 3.664
  ✓ IR generated
    - 5 states
    - 20 parameters
    - 5 ODEs
    - 2 observables
    - 2 external QM constants
      • Kd_QM = 2.500e-9
      • Kp_tumor_QM = 3.664e0
```

## PBPK + QSP Integration

### The Key Innovation

**Traditional PK-PD**:
```
C_effect = Kp_tumor * C_plasma
```
- Assumes instantaneous equilibration
- Kp is a simple multiplier
- No dynamics of tissue distribution

**PBPK-QSP**:
```medlang
let C_tumor_tissue = A_tumor / V_tumor

dTumourSize/dt = k_grow * TumourSize * (1.0 - TumourSize / T_max)
               - (Emax * C_tumor_tissue / (EC50 + C_tumor_tissue)) * TumourSize
```

- `C_tumor_tissue` is **dynamically computed** from PBPK
- Accounts for blood flow, partition kinetics
- More realistic for slow-equilibrating tissues

### Complete PBPK-QSP-QM Example

See `docs/examples/pbpk_5comp_qsp_qm.medlang` for a complete working model combining:
- 5-compartment PBPK (plasma, tumor, liver, kidney, rest)
- Tumor QSP dynamics (growth-kill with Emax model)
- QM-informed Kp_tumor (from ΔG_part)
- QM-informed EC50 (from Kd)
- Allometric scaling for all physiological parameters

## Stan Code Generation

### Intermediate Values in ODE Function

MedLang automatically generates:

```stan
functions {
  vector ode_system(real t, vector y, ...) {
    real A_plasma = y[1];
    real A_tumor  = y[2];
    // ... unpack other states
    
    // Intermediate values
    real C_plasma = (A_plasma / V_plasma);
    real C_tumor_tissue = (A_tumor / V_tumor);
    real C_tumor_vein = (C_tumor_tissue / Kp_tumor);
    // ... other intermediates
    
    vector[5] dydt;
    dydt[1] = Q_tumor * (C_tumor_vein - C_plasma) + ...;
    // ... other ODEs
    
    return dydt;
  }
}
```

### QM Constants in Data Block

```stan
data {
  // ... other data
  
  // External quantum constants
  real<lower=0> Kd_QM;        // from qm_stub:LIG001:EGFR
  real<lower=0> Kp_tumor_QM;  // from qm_stub:LIG001:dG_part
}
```

These must be provided in your Stan JSON data file.

## Practical Workflow

### 1. Create PBPK Model

Write your MedLang model with PBPK structure and let bindings.

### 2. Prepare QM Stub

Generate or obtain quantum properties:
```json
{
  "drug_id": "MY_DRUG",
  "target_id": "MY_TARGET",
  "Kd_M": 5.0e-9,
  "dG_part_plasma_tumor_kcal_per_mol": -1.2,
  "T_ref_K": 310.0
}
```

### 3. Compile

```bash
mlc compile my_pbpk_model.medlang \
    --qm-stub data/my_drug_qm.json \
    --emit-ir my_model_ir.json \
    -v
```

### 4. Prepare Stan Data

Your Stan JSON must include:
- Observations (`observation`, `time`, `subject_id`)
- Covariates (`WT`)
- Dosing (`dose_amount`, `dose_time`)
- **QM constants** (`Kd_QM`, `Kp_tumor_QM`)
- ODE solver settings (`rtol`, `atol`, `max_steps`)

Example:
```json
{
  "N": 20,
  "n_obs": 100,
  "subject_id": [...],
  "time": [...],
  "observation": [...],
  "WT": [70, 75, 68, ...],
  "Kd_QM": 5.0e-9,
  "Kp_tumor_QM": 4.123,
  "dose_amount": 100.0,
  "dose_time": 0.0,
  "rtol": 1e-6,
  "atol": 1e-8,
  "max_steps": 10000
}
```

### 5. Run Inference

```bash
mlc run my_pbpk_model.stan \
    --data my_data.json \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    -v
```

## Interpreting Results

### Partition Coefficients

Estimated Kp values tell you about tissue distribution:

- **Kp > 1**: Drug concentrates in tissue (favorable for that tissue)
- **Kp = 1**: Equal partition between plasma and tissue
- **Kp < 1**: Drug excluded from tissue

For tumor targeting, you want **Kp_tumor > 1** for good exposure.

### QM-Informed vs Estimated

Compare Kp_tumor from QM stub to what would be estimated from data alone:
- If similar: QM prediction is accurate
- If QM predicts higher: Barriers not captured by QM (efflux, binding)
- If QM predicts lower: Active transport or other mechanisms

### Tumor Exposure

With PBPK, tumor concentration isn't just `Kp * C_plasma`:
- It has **dynamics** (time to equilibrate)
- It depends on **blood flow** to tumor
- It reflects **tissue binding** (via Kp)

## Advantages of PBPK-QM Integration

1. **Mechanistic Priors**: Use QM-derived Kp as informed prior
2. **Fewer Parameters**: Kp_tumor fixed, not estimated
3. **Predictive**: For new drugs with QM data but no clinical PK
4. **Multi-Scale**: Seamless connection from molecular properties to tissue distribution
5. **Interpretable**: Deviations from QM Kp reveal active transport, efflux, etc.

## Limitations & Future Work

### Current Limitations

- **Static QM**: No uncertainty propagation from QM calculations
- **Well-stirred assumption**: Assumes instant mixing within each organ
- **No metabolism**: Liver compartment doesn't metabolize (yet)
- **No protein binding**: Free vs bound drug not distinguished
- **Manual data prep**: QM constants must be manually added to Stan data

### Future Enhancements (Week 8+)

- Bayesian priors on Kp with QM uncertainty
- Hepatic metabolism (CYP enzymes)
- Plasma and tissue protein binding (fu)
- Permeability-surface area limited distribution
- Active transport (P-gp, BCRP, OATP)
- Multi-drug interactions

## Examples

| Example | Description | States | QM |
|---------|-------------|--------|-----|
| `pbpk_2comp_simple.medlang` | 2-comp PBPK (plasma + tissue) | 2 | No |
| `pbpk_5comp_qsp_qm.medlang` | 5-comp PBPK + tumor QSP + QM | 6 | Yes |

## References

- Grass & Sinko (2002). *Physiologically-based pharmacokinetic simulation modelling*. Advanced Drug Delivery Reviews.
- Jones et al. (2015). *Physiologically based pharmacokinetic modeling in drug discovery and development*. CPT Pharmacometrics Syst Pharmacol.
- Rodgers & Rowland (2006). *Physiologically based pharmacokinetic modelling: predicting the tissue distribution of drugs*. J Pharm Sci.
- Mahmood (2007). *Allometric scaling in pharmacokinetics*. Expert Opin Drug Metab Toxicol.

---

**Status**: Week 7 Complete ✓  
**Next**: Week 8 - Bayesian QM priors or metabolism/binding extensions
