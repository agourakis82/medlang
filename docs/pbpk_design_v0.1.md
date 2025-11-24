# PBPK Design v0.1 - Week 7

## Overview

Extend MedLang with physiologically-based pharmacokinetic (PBPK) modeling capabilities, enabling:
- Multi-tissue drug distribution
- QM-informed tissue partition coefficients (Kp)
- Coupling to QSP tumor dynamics via C_tumor (not Kp·C_plasma)

## 5-Compartment PBPK Structure

### Compartments

1. **Plasma (P)**
   - Central blood compartment
   - Volume: V_plasma
   - State: A_plasma [DoseMass]

2. **Tumor (T)**
   - Target tissue with QSP dynamics
   - Volume: V_tumor
   - State: A_tumor [DoseMass]
   - **Kp_tumor from QM stub** (ΔG_part)

3. **Liver (L)**
   - Metabolic organ
   - Volume: V_liver
   - State: A_liver [DoseMass]
   - Kp_liver: population parameter

4. **Kidney (K)**
   - Excretion organ
   - Volume: V_kidney
   - State: A_kidney [DoseMass]
   - Kp_kidney: population parameter

5. **Rest-of-Body (R)**
   - Lumped peripheral tissues
   - Volume: V_rest
   - State: A_rest [DoseMass]
   - Kp_rest: population parameter

### Mass Balance ODEs

Using **permeability-limited (well-stirred)** distribution:

```
dA_plasma/dt = 
    + Q_tumor  * (C_tumor_vein  - C_plasma)
    + Q_liver  * (C_liver_vein  - C_plasma)
    + Q_kidney * (C_kidney_vein - C_plasma)
    + Q_rest   * (C_rest_vein   - C_plasma)
    - CL_renal * C_plasma
    + Input(t)

dA_tumor/dt = Q_tumor * (C_plasma - C_tumor_vein)

dA_liver/dt = Q_liver * (C_plasma - C_liver_vein)

dA_kidney/dt = Q_kidney * (C_plasma - C_kidney_vein)

dA_rest/dt = Q_rest * (C_plasma - C_rest_vein)
```

Where venous concentrations use partition coefficients:

```
C_plasma = A_plasma / V_plasma

C_tumor_vein  = (A_tumor  / V_tumor)  / Kp_tumor
C_liver_vein  = (A_liver  / V_liver)  / Kp_liver
C_kidney_vein = (A_kidney / V_kidney) / Kp_kidney
C_rest_vein   = (A_rest   / V_rest)   / Kp_rest
```

### Tumor Tissue Concentration

**This is the key innovation:**

```
C_tumor = A_tumor / V_tumor
```

NOT `Kp_tumor * C_plasma`, but the actual tissue concentration from PBPK dynamics.

This C_tumor drives the QSP tumor model:

```
dTumourSize/dt = k_grow * TumourSize * (1 - TumourSize / T_max)
                 - (Emax * C_tumor / (EC50 + C_tumor)) * TumourSize
```

## QM Integration

### From QM Stub to Kp_tumor

The quantum stub provides:
```json
{
  "dG_part_plasma_tumor_kcal_per_mol": -0.8,
  "T_ref_K": 310.0
}
```

Converted to Kp_tumor:
```
Kp_tumor = exp(-ΔG_part / (R * T))
         = exp(-(-0.8) / (0.00198720 * 310))
         = exp(1.297)
         ≈ 3.66
```

### Population Model

```medlang
population PBPK_QSP_Pop {
    // ... other params ...
    
    // Tumor Kp is QM-informed (fixed from stub)
    // Other tissue Kps are estimated
    param Kp_liver : f64
    param Kp_kidney : f64
    param Kp_rest : f64
    
    bind_params(patient) {
        // Kp_tumor comes from externals (set by QM stub)
        model.Kp_tumor = Kp_tumor_QM  // From QM stub
        model.Kp_liver = Kp_liver
        model.Kp_kidney = Kp_kidney
        model.Kp_rest = Kp_rest
    }
}
```

## Allometric Scaling

Organ volumes and blood flows scale with body weight (WT):

### Standard 70 kg Reference Values

| Parameter | 70 kg Value | Units | Allometric Exponent |
|-----------|-------------|-------|---------------------|
| V_plasma  | 3.0         | L     | 1.0                 |
| V_tumor   | 0.5         | L     | 1.0                 |
| V_liver   | 1.8         | L     | 1.0                 |
| V_kidney  | 0.3         | L     | 1.0                 |
| V_rest    | 35.0        | L     | 1.0                 |
| Q_tumor   | 0.5         | L/h   | 0.75                |
| Q_liver   | 90.0        | L/h   | 0.75                |
| Q_kidney  | 70.0        | L/h   | 0.75                |
| Q_rest    | 150.0       | L/h   | 0.75                |
| CL_renal  | 6.0         | L/h   | 0.75                |

### Scaling Equations

```medlang
bind_params(patient) {
    let w = patient.WT / 70.0_kg
    
    // Volumes scale linearly with weight
    model.V_plasma = V_plasma_ref * w
    model.V_tumor  = V_tumor_ref  * w
    model.V_liver  = V_liver_ref  * w
    model.V_kidney = V_kidney_ref * w
    model.V_rest   = V_rest_ref   * w
    
    // Flows scale with exponent 0.75
    model.Q_tumor  = Q_tumor_ref  * pow(w, 0.75)
    model.Q_liver  = Q_liver_ref  * pow(w, 0.75)
    model.Q_kidney = Q_kidney_ref * pow(w, 0.75)
    model.Q_rest   = Q_rest_ref   * pow(w, 0.75)
    model.CL_renal = CL_renal_ref * pow(w, 0.75)
}
```

## Dimensional Types

New types needed:

```medlang
// Already exist:
DoseMass, Volume, ConcMass, Clearance, RateConst

// New for PBPK:
BloodFlow : Volume / Time  // L/h
PartitionCoeff : f64       // Dimensionless ratio
```

## Example MedLang PBPK-QSP Model

```medlang
model PBPK_5Comp_TumorQSP {
    // =========================================================================
    // PBPK States (5 compartments)
    // =========================================================================
    state A_plasma : DoseMass
    state A_tumor  : DoseMass
    state A_liver  : DoseMass
    state A_kidney : DoseMass
    state A_rest   : DoseMass
    
    // QSP State
    state TumourSize : TumourVolume
    
    // =========================================================================
    // PBPK Parameters
    // =========================================================================
    // Volumes
    param V_plasma : Volume
    param V_tumor  : Volume
    param V_liver  : Volume
    param V_kidney : Volume
    param V_rest   : Volume
    
    // Flows
    param Q_tumor  : Clearance  // Reusing Clearance type for L/h
    param Q_liver  : Clearance
    param Q_kidney : Clearance
    param Q_rest   : Clearance
    param CL_renal : Clearance
    
    // Partition coefficients
    param Kp_tumor  : f64
    param Kp_liver  : f64
    param Kp_kidney : f64
    param Kp_rest   : f64
    
    // =========================================================================
    // QSP Parameters
    // =========================================================================
    param k_grow : RateConst
    param T_max  : TumourVolume
    param Emax   : f64
    param EC50   : ConcMass
    
    // =========================================================================
    // Intermediate Concentrations
    // =========================================================================
    let C_plasma = A_plasma / V_plasma
    
    let C_tumor_tissue = A_tumor / V_tumor
    let C_tumor_vein   = C_tumor_tissue / Kp_tumor
    
    let C_liver_tissue = A_liver / V_liver
    let C_liver_vein   = C_liver_tissue / Kp_liver
    
    let C_kidney_tissue = A_kidney / V_kidney
    let C_kidney_vein   = C_kidney_tissue / Kp_kidney
    
    let C_rest_tissue = A_rest / V_rest
    let C_rest_vein   = C_rest_tissue / Kp_rest
    
    // =========================================================================
    // PBPK ODEs
    // =========================================================================
    dA_plasma/dt = Q_tumor  * (C_tumor_vein  - C_plasma)
                 + Q_liver  * (C_liver_vein  - C_plasma)
                 + Q_kidney * (C_kidney_vein - C_plasma)
                 + Q_rest   * (C_rest_vein   - C_plasma)
                 - CL_renal * C_plasma
    
    dA_tumor/dt  = Q_tumor  * (C_plasma - C_tumor_vein)
    dA_liver/dt  = Q_liver  * (C_plasma - C_liver_vein)
    dA_kidney/dt = Q_kidney * (C_plasma - C_kidney_vein)
    dA_rest/dt   = Q_rest   * (C_plasma - C_rest_vein)
    
    // =========================================================================
    // QSP ODE (tumor growth-kill)
    // =========================================================================
    dTumourSize/dt = k_grow * TumourSize * (1.0 - TumourSize / T_max)
                   - (Emax * C_tumor_tissue / (EC50 + C_tumor_tissue)) * TumourSize
    
    // =========================================================================
    // Observables
    // =========================================================================
    obs C_plasma_obs : ConcMass = C_plasma
    obs TumourVol : TumourVolume = TumourSize
}
```

## Implementation Plan

### Phase 1: Intermediate Values (Let Bindings)
- Already implemented in Week 5
- Verify they work in ODE right-hand sides

### Phase 2: PBPK Example Model
- Create canonical 5-comp PBPK-QSP model
- Test compilation with existing infrastructure
- Verify Stan code generation

### Phase 3: QM Integration
- Use Kp_tumor_QM from stub in bind_params
- Generate example with tumor Kp from ΔG_part
- Demonstrate C_tumor ≠ Kp·C_plasma

### Phase 4: Data Generation
- Extend datagen for PBPK observables
- Simulate plasma and tumor concentrations
- Create synthetic dataset

### Phase 5: Documentation
- PBPK user guide
- Allometric scaling reference
- QM-PBPK-QSP integration tutorial

## Success Criteria

✅ 5-compartment PBPK model compiles to Stan
✅ Let bindings work in ODE RHS
✅ Kp_tumor from QM stub flows through to Stan
✅ C_tumor drives QSP dynamics (not Kp·C_plasma)
✅ Allometric scaling in bind_params
✅ All existing tests still pass
✅ PBPK-specific integration tests

## Future Extensions (Week 8+)

- Metabolism in liver (CYP enzymes)
- Protein binding (fu_plasma, fu_tissue)
- Active transport (P-gp, BCRP)
- Bayesian priors on Kp values
- Multi-drug interactions

## References

- Physiologically Based Pharmacokinetic Modeling (Grass & Sinko, 2002)
- PBPK Modeling in Drug Development (Jones et al., 2015)
- Allometric Scaling in Pharmacokinetics (Mahmood, 2007)

---

**Status**: Week 7 Design Complete ✓
**Next**: Implementation of let bindings in ODE RHS
