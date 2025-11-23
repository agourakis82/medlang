# Example: Quantum-Informed PBPK + QSP for a Single Drug–Target Pair

This example shows how Track C (quantum) and Track D (pharmacometrics/QSP) interact end-to-end for:

- one small-molecule drug `LIG`, and  
- one receptor `REC` expressed in tumor tissue.

The flow:

1. Use `QM_BindingFreeEnergy` and `QM_PartitionCoefficient` to compute:
   - binding free energy and Kd for `LIG–REC`,
   - partition free energy and Kp between plasma and tumor.
2. Map these into:
   - PBPK tumor partition coefficient `Kp_tumor`, and  
   - PD parameters `EC50` and `k_kill` for a tumor-immune QSP model.
3. Run a population PK-PD simulation.

---

## 1. Quantum Step (Track C)

We define the quantum systems and run the binding and partition calculations:

```medlang
// =============================
// Molecular structures
// =============================
let ligand = Molecule::from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  // example: caffeine-like
let receptor = Molecule::from_pdb("1A4G", chain = "A")              // example: adenosine receptor

// =============================
// Binding calculation (Track C)
// =============================
let binding = QM_BindingFreeEnergy {
    ligand   = ligand,
    target   = receptor,
    binding_pose = docking_result.best_pose,
    method   = QMMethod {
        theory     = DFT,
        functional = wB97XD,
        basis_set  = Def2_TZVP,
        dispersion = D3
    },
    qmmm_partition = QMMM_System {
        qm_region     = ligand + select_atoms(radius = 5.0 Å around ligand),
        mm_region     = rest_of(receptor),
        boundary      = LinkAtom(H),
        mm_forcefield = AMBER_ff14SB
    },
    temperature = 310.0 K
}

// Outputs
let ΔG_bind_QM : Energy = binding.ΔG_bind        // e.g., -8.5 kcal/mol
let σ_ΔG       : Energy = binding.uncertainty    // e.g., ± 1.2 kcal/mol

// =============================
// Kinetics calculation (Track C)
// =============================
let kinetics = QM_Kinetics {
    ligand        = ligand,
    target        = receptor,
    bound_complex = optimized_complex,
    method        = QMMethod {
        theory     = DFT,
        functional = wB97XD,
        basis_set  = Def2_SVP
    },
    temperature = 310.0 K
}

let k_on_QM  : RateConstPerConc = kinetics.k_on   // e.g., 6.5e5 1/(M·s)
let k_off_QM : RateConst        = kinetics.k_off  // e.g., 0.018 1/s

// =============================
// Partition coefficient (Track C)
// =============================
let partition = QM_PartitionCoefficient {
    molecule = ligand,
    phase_A  = SMD(solvent = "water"),           // plasma
    phase_B  = SMD(solvent = "tumor_tissue"),    // tumor interstitium
    method   = QMMethod {
        theory     = DFT,
        functional = M06_2X,
        basis_set  = Def2_TZVP
    },
    temperature = 310.0 K
}

let ΔG_partition_QM : Energy = partition.ΔG_partition  // e.g., -1.8 kcal/mol
let Kp_QM_direct    : f64    = partition.Kp            // e.g., 0.72
```

---

## 2. PBPK Step (Track D): Plasma + Tumor

We define a minimal PBPK model with plasma and tumor compartments:

```medlang
// =============================
// 2-compartment PBPK model
// =============================
model PBPK_2Comp_Tumor {
    // ---- States (drug amounts) ----
    state A_plasma : DoseMass    // mg
    state A_tumor  : DoseMass    // mg

    // ---- Parameters ----
    param CL      : Clearance    // plasma clearance, L/h
    param Q_t     : Flow         // plasma–tumor blood flow, L/h
    param V_pl    : Volume       // plasma volume, L
    param V_tumor : Volume       // tumor volume, L
    param Kp_t    : f64          // tumor partition coefficient (dimensionless)

    // ---- Concentrations ----
    let C_plasma : ConcMass = A_plasma / V_pl
    let C_tumor  : ConcMass = A_tumor / V_tumor

    // ---- Structural dynamics ----
    dA_plasma/dt =
        - (CL / V_pl) * A_plasma
        - Q_t * (C_plasma - C_tumor / Kp_t)

    dA_tumor/dt =
        Q_t * (C_plasma - C_tumor / Kp_t)

    // ---- Observables ----
    obs C_plasma_obs : ConcMass = C_plasma
    obs C_tumor_obs  : ConcMass = C_tumor
}
```

### 2.1 Quantum → PBPK Parameter Mapping

We map Track C `ΔG_partition` to PBPK `Kp_tumor`:

```medlang
// =============================
// Quantum-informed Kp mapping
// =============================
fn Kp_tumor_from_QM(
    ΔG_partition : Quantity<Energy, f64>,
    T            : Quantity<Kelvin, f64>,
    w_ML         : MLParamVector,
    eta_Kp       : f64
) -> f64 {
    // Thermodynamic baseline
    let R      : EnergyPerMolPerK = 8.314e-3 kJ/(mol·K)
    let expo   : f64 = -(ΔG_partition / (R * T))  // dimensionless
    let Kp_QM  : f64 = exp(expo)                  // dimensionless

    // Optional ML correction (e.g., for microenvironment effects)
    let Kp_ML  : f64 = KpT_ML{w = w_ML}.g_Kp(ΔG_partition)
    
    // Inter-individual variability
    let Kp_IIV : f64 = exp(eta_Kp)

    return Kp_QM * Kp_ML * Kp_IIV
}

// Usage in population model
param w_Kp_ML  : MLParamVector   // ML correction weights
rand  eta_Kp_i : f64 ~ Normal(0, omega_Kp^2)

let Kp_tumor_i = Kp_tumor_from_QM(
    ΔG_partition = partition.ΔG_partition,
    T            = 310.0 K,
    w_ML         = w_Kp_ML,
    eta_Kp       = eta_Kp_i
)

// Bind to PBPK model
PBPK_2Comp_Tumor.Kp_t = Kp_tumor_i
```

---

## 3. QSP Step (Track D): Tumor-Immune PD

We consider a simple tumor-immune QSP model:

```medlang
// =============================
// Tumor-immune QSP model
// =============================
model TumorImmune_QSP {
    // ---- Input from PBPK ----
    input C_tumor : ConcMass    // drug concentration in tumor from PBPK

    // ---- States ----
    state Tumor    : TumorVolume    // mm³
    state Effector : CellDensity    // cells/μL

    // ---- Tumor growth parameters ----
    param k_grow : RateConst        // 1/h
    param T_max  : TumorVolume      // mm³ (carrying capacity)

    // ---- Drug effect parameters ----
    param Emax_kill : f64           // max fractional kill rate (dimensionless)
    param EC50_drug : ConcMass      // drug concentration for half-max effect

    // ---- Immune parameters ----
    param k_prolif  : RateConst     // effector proliferation rate, 1/h
    param k_death   : RateConst     // effector death rate, 1/h

    // ---- Drug effect function ----
    fn E_drug(C : ConcMass) -> f64 {
        // Hill function for drug effect (dimensionless)
        return Emax_kill * C / (EC50_drug + C)
    }

    // ---- Dynamics ----
    dTumor/dt =
        k_grow * Tumor * (1.0 - Tumor / T_max)
        - E_drug(C_tumor) * Effector * Tumor

    dEffector/dt =
        k_prolif * Effector
        - k_death * Effector

    // ---- Observables ----
    obs TumorVolume   : TumorVolume = Tumor
    obs EffectorCount : CellDensity = Effector
}
```

### 3.1 Quantum → QSP Parameter Mapping

We map Track C binding results to QSP parameters:

```medlang
// =============================
// Quantum-informed EC50 mapping
// =============================
fn Kd_from_ΔG(
    ΔG_bind : Quantity<Energy, f64>,
    T       : Quantity<Kelvin, f64>
) -> Quantity<Concentration, f64> {
    let R  : EnergyPerMolPerK = 8.314e-3 kJ/(mol·K)
    let C0 : Concentration    = 1.0 M      // standard concentration
    
    let exponent : f64 = (ΔG_bind / (R * T))  // dimensionless
    return C0 * exp(exponent)
}

fn EC50_from_Kd(
    Kd         : Quantity<Concentration, f64>,
    alpha_EC50 : f64,    // calibration factor
    eta_EC50   : f64     // random effect
) -> Quantity<Concentration, f64> {
    return alpha_EC50 * Kd * exp(eta_EC50)
}

// Population parameters
param alpha_EC50 : f64 = 1.5         // calibration factor (to be estimated)
rand  eta_EC50_i : f64 ~ Normal(0, omega_EC50^2)

// Map quantum ΔG_bind to EC50
let Kd_QM   = Kd_from_ΔG(ΔG_bind = binding.ΔG_bind, T = 310.0 K)
let EC50_i  = EC50_from_Kd(Kd_QM, alpha_EC50, eta_EC50_i)

// Bind to QSP model
TumorImmune_QSP.EC50_drug = EC50_i
```

### 3.2 Optional: k_kill from k_on/k_off

For a more mechanistic model, we can map kinetics to killing rate:

```medlang
fn f_QM_kill_scale(
    k_on      : Quantity<RateConstPerConc, f64>,
    k_off     : Quantity<RateConst, f64>,
    k_on_ref  : Quantity<RateConstPerConc, f64>,
    k_off_ref : Quantity<RateConst, f64>,
    beta_on   : f64,
    beta_off  : f64
) -> f64 {
    let ratio_on  : f64 = (k_on / k_on_ref)
    let ratio_off : f64 = (k_off_ref / k_off)
    return pow(ratio_on, beta_on) * pow(ratio_off, beta_off)
}

param Emax_base : f64 = 0.5
param beta_on   : f64 = 0.5
param beta_off  : f64 = 0.5

let f_scale = f_QM_kill_scale(k_on_QM, k_off_QM, k_on_ref, k_off_ref, beta_on, beta_off)

TumorImmune_QSP.Emax_kill = Emax_base * f_scale
```

---

## 4. Coupled PBPK + QSP Model

We connect the PBPK and QSP models:

```medlang
// =============================
// Coupled system
// =============================
model QM_PBPK_QSP_Coupled {
    // ---- Submodels ----
    submodel PBPK : PBPK_2Comp_Tumor
    submodel QSP  : TumorImmune_QSP

    // ---- Coupling: PBPK → QSP ----
    // Tumor drug concentration from PBPK drives QSP dynamics
    QSP.C_tumor = PBPK.C_tumor_obs

    // ---- Optional: QSP → PBPK ----
    // Tumor volume from QSP affects PBPK compartment size
    PBPK.V_tumor = QSP.Tumor * 1e-6 L/mm³  // convert mm³ to L

    // ---- Observables ----
    obs C_plasma      : ConcMass     = PBPK.C_plasma_obs
    obs C_tumor       : ConcMass     = PBPK.C_tumor_obs
    obs TumorVolume   : TumorVolume  = QSP.TumorVolume
    obs EffectorCount : CellDensity  = QSP.EffectorCount
}
```

---

## 5. Population Model and Simulation

### 5.1 Population Structure

```medlang
// =============================
// Population model with quantum parameters
// =============================
population QM_PBPK_QSP_PopModel {
    structural_model = QM_PBPK_QSP_Coupled

    // ---- Track C constants (precomputed quantum results) ----
    const ΔG_bind_QM      : Energy = -8.5 kcal/mol
    const ΔG_partition_QM : Energy = -1.8 kcal/mol
    const k_on_QM         : RateConstPerConc = 6.5e5 1/(M·s)
    const k_off_QM        : RateConst        = 0.018 1/s

    // ---- Population fixed effects (PK) ----
    hyperparam CL_pop  : Clearance = 10.0 L/h
    hyperparam Q_t_pop : Flow      = 0.5 L/h
    hyperparam V_pl_pop : Volume   = 3.0 L

    // ---- Population fixed effects (QSP) ----
    hyperparam k_grow_pop   : RateConst    = 0.01 /h
    hyperparam T_max_pop    : TumorVolume  = 8000.0 mm³
    hyperparam k_prolif_pop : RateConst    = 0.02 /h
    hyperparam k_death_pop  : RateConst    = 0.01 /h

    // ---- Quantum calibration factors ----
    hyperparam alpha_EC50 : f64 = 1.5   // EC50 = alpha * Kd_QM
    hyperparam alpha_Kp   : f64 = 1.0   // Kp_tumor = alpha * Kp_QM
    hyperparam beta_on    : f64 = 0.5   // kinetics scaling
    hyperparam beta_off   : f64 = 0.5

    // ---- ML correction (optional) ----
    hyperparam w_Kp_ML : MLParamVector

    // ---- Random effects covariance ----
    hyperparam Omega : CovMatrix<4>  // [CL, k_grow, Kp, EC50]

    // ---- Residual errors ----
    hyperparam sigma_C_plasma : f64 = 0.2
    hyperparam sigma_tumor    : TumorVolume = 100.0 mm³

    // ---- Covariates ----
    covariate BW : BodyWeight

    // ---- Random effects ----
    random_effects eta : Vector<4, f64> {
        distribution = MVNormal(mean = zeros(4), cov = Omega)
        labels = ["eta_CL", "eta_k_grow", "eta_Kp", "eta_EC50"]
    }

    // ---- Parameter transformation ----
    transform individual_params(
        BW  : BodyWeight,
        eta : Vector<4, f64>
    ) -> CoupledParams {
        // Allometric scaling for CL
        let w_scaled : f64 = BW / (70.0 kg)
        let CL_i = CL_pop * pow(w_scaled, 0.75) * exp(eta[0])

        // QSP with IIV
        let k_grow_i = k_grow_pop * exp(eta[1])

        // Quantum-derived Kp
        let Kp_tumor_i = Kp_tumor_from_QM(
            ΔG_partition_QM, 310.0 K, w_Kp_ML, eta[2]
        )

        // Quantum-derived EC50
        let Kd_QM   = Kd_from_ΔG(ΔG_bind_QM, 310.0 K)
        let EC50_i  = EC50_from_Kd(Kd_QM, alpha_EC50, eta[3])

        // Quantum-derived Emax
        let Emax_i = Emax_base * f_QM_kill_scale(
            k_on_QM, k_off_QM, k_on_ref, k_off_ref, beta_on, beta_off
        )

        return CoupledParams {
            CL      = CL_i,
            Q_t     = Q_t_pop,
            V_pl    = V_pl_pop,
            Kp_t    = Kp_tumor_i,     // ← QUANTUM
            k_grow  = k_grow_i,
            T_max   = T_max_pop,
            EC50_drug = EC50_i,       // ← QUANTUM
            Emax_kill = Emax_i,       // ← QUANTUM
            k_prolif  = k_prolif_pop,
            k_death   = k_death_pop
        }
    }

    // ---- Observation models ----
    observation_model pk_obs = ProportionalError {
        channel     = QM_PBPK_QSP_Coupled.C_plasma
        error_param = sigma_C_plasma
    }

    observation_model tumor_obs = AdditiveError {
        channel     = QM_PBPK_QSP_Coupled.TumorVolume
        error_param = sigma_tumor
    }
}
```

### 5.2 Simulation

```medlang
// =============================
// Forward simulation (virtual trial)
// =============================
simulation VirtualTrial {
    population_model = QM_PBPK_QSP_PopModel
    
    n_subjects = 100
    
    // Dosing regimen
    timeline per_subject {
        at 0.0_h:
            dose {
                route  = IV
                amount = 500.0_mg
                to     PBPK_2Comp_Tumor.A_plasma
            }
        
        // Observations
        at [0.5, 1, 2, 4, 8, 12, 24]_h: observe C_plasma
        at [0, 7, 14, 21, 28]_days:     observe TumorVolume
    }
    
    // Sample covariates
    covariate_distribution {
        BW ~ LogNormal(log(70.0 kg), 0.2)
    }
}
```

### 5.3 Bayesian Inference

```medlang
// =============================
// Bayesian inference with QM-informed priors
// =============================
inference Bayesian_QM_Informed {
    population_model = QM_PBPK_QSP_PopModel
    cohort           = clinical_trial_data
    
    mode      = Bayesian
    backend   = Backend.Stan
    algorithm = Algorithm.NUTS
    
    priors {
        // PK parameters
        CL_pop ~ LogNormal(log(10.0 L/h), 0.5)
        Q_t_pop ~ LogNormal(log(0.5 L/h), 0.5)
        
        // QSP parameters
        k_grow_pop ~ LogNormal(log(0.01 /h), 0.7)
        k_prolif_pop ~ LogNormal(log(0.02 /h), 0.5)
        
        // Quantum calibration factors (key parameters!)
        alpha_EC50 ~ Normal(1.0, 0.5)  // If posterior ≈ 1.0, QM is accurate
        alpha_Kp   ~ Normal(1.0, 0.3)
        beta_on    ~ Normal(0.5, 0.3)
        beta_off   ~ Normal(0.5, 0.3)
        
        // Random effects
        Omega ~ LKJCov(eta = 2.0, scale_prior = HalfNormal(0.5))
        
        // Residual errors
        sigma_C_plasma ~ HalfNormal(0.3)
        sigma_tumor    ~ HalfNormal(150.0 mm³)
        
        // Optional: QM uncertainty propagation
        ΔG_bind_true ~ Normal(
            mean = ΔG_bind_QM,
            sd   = σ_ΔG  // quantum method uncertainty
        )
    }
}
```

---

## 6. Validation and Interpretation

### 6.1 Quantum Prediction Accuracy

After Bayesian inference, examine the posterior of `alpha_EC50`:

- **If `posterior(alpha_EC50) ≈ 1.0`:** Quantum prediction is accurate; EC50 = Kd_QM.
- **If `posterior(alpha_EC50) > 1.0`:** System-level effects increase EC50 beyond Kd.
- **If `posterior(alpha_EC50) < 1.0`:** Kd overestimates EC50 (less common).

Similarly for `alpha_Kp`, `beta_on`, `beta_off`.

### 6.2 Parameter Recovery Test

1. Generate synthetic data with known quantum values.
2. Fit the model.
3. Check if true `alpha_EC50`, `alpha_Kp` are within 95% credible intervals.

### 6.3 Cross-Validation

Compare model predictions with/without quantum information:

- **Model A:** EC50 and Kp as free parameters (no quantum).
- **Model B:** EC50 and Kp from quantum with calibration (this example).
- **Compare:** WAIC, LOO, posterior predictive checks.

If Model B has better predictive performance, quantum information is valuable.

---

## 7. Summary

This example demonstrates the complete Track C → Track D workflow:

1. **Track C (Quantum):**
   - Compute `ΔG_bind`, `k_on`, `k_off` via QM/MM.
   - Compute `ΔG_partition`, `Kp` via solvation models.

2. **Track D (PBPK):**
   - Map `ΔG_partition → Kp_tumor` with calibration and IIV.
   - Use in 2-compartment PBPK.

3. **Track D (QSP):**
   - Map `ΔG_bind → Kd → EC50` with calibration and IIV.
   - Map `k_on, k_off → Emax_kill` via mechanistic scaling.
   - Use in tumor-immune dynamics.

4. **Track D (Population):**
   - Wrap in population model with random effects.
   - Bayesian inference with QM-informed priors.
   - Posterior reveals QM accuracy via calibration factors.

**Key Advantages:**

✅ **Physical grounding:** Parameters derived from first principles.  
✅ **Data-driven correction:** Calibration factors allow adjustment.  
✅ **Uncertainty quantification:** QM uncertainty propagates through PBPK/QSP to clinical outcomes.  
✅ **Type safety:** All unit conversions explicit and checked.

This workflow is the **core innovation of MedLang**: bridging quantum chemistry and clinical pharmacology with full mathematical rigor.
