# Stress Test 3: PBPK + QSP + ML + Quantum Integration

**Purpose:** Validate the full MedLang vertical by implementing a realistic multi-scale oncology model that integrates:
- **Track C (Quantum):** Binding affinity, partition coefficients, kinetic parameters
- **Track D (PBPK):** Multi-organ pharmacokinetics with quantum-derived Kp
- **Track D (QSP):** Tumor-immune-drug dynamics with ML-augmented killing function
- **Track D (NLME):** Population variability and Bayesian inference

This is the **ultimate stress test** of the MedLang spec: every layer, every interface, every unit-checking rule.

---

## 1. Clinical Scenario

**Drug:** Novel checkpoint inhibitor + chemotherapy combination for solid tumors.

**Objective:** Predict tumor volume trajectories in a heterogeneous patient population, accounting for:
1. **Drug partitioning** into tumor tissue (PBPK with quantum-derived Kp).
2. **Binding affinity** to PD-L1 (quantum ΔG_bind → EC50).
3. **Immune cell activation** dynamics (QSP).
4. **Unknown killing mechanism** (ML-learned function).
5. **Inter-individual variability** in PK, immune response, tumor growth.

**Data:**
- N = 50 patients
- Weekly tumor volume measurements (CT/MRI)
- Biweekly effector cell counts (flow cytometry)
- Dense plasma PK sampling in subset (n=10)

**Challenge:** Can MedLang express this, maintain unit safety, and compile to executable inference code?

---

## 2. Mathematical Formulation

### 2.1 Multi-Compartment PBPK

**Compartments:**
1. **Blood** (central)
2. **Liver** (metabolic clearance)
3. **Kidney** (renal clearance)
4. **Tumor** (site of action)
5. **Rest of body** (lumped peripheral)

**Mass balance for organ compartment k:**
$$
V_k \frac{dC_k}{dt} = Q_k (C_{\text{blood}} - C_k / K_{p,k}) - \text{CL}_k \cdot C_k
$$

where:
- $V_k$ = volume of compartment $k$ [L]
- $Q_k$ = blood flow to compartment $k$ [L/h]
- $C_k$ = drug concentration in compartment $k$ [mg/L]
- $K_{p,k}$ = partition coefficient (dimensionless)
- $\text{CL}_k$ = intrinsic clearance in compartment $k$ [L/h]

**Central blood compartment:**
$$
V_{\text{blood}} \frac{dC_{\text{blood}}}{dt} = \sum_k Q_k \left(\frac{C_k}{K_{p,k}} - C_{\text{blood}}\right) + \text{infusion}(t)
$$

**Quantum input:** $K_{p,\text{tumor}}$ from Track C.

### 2.2 Tumor-Immune QSP

**States:**
- $T(t)$ — tumor volume [mm³]
- $E(t)$ — effector T-cell density in tumor [cells/μL]
- $I(t)$ — PD-L1 inhibition level (dimensionless, [0,1])

**Dynamics:**
$$
\begin{aligned}
\frac{dT}{dt} &= k_{\text{grow}} \, T \left(1 - \frac{T}{T_{\max}}\right) - f_{\text{kill}}(C_{\text{tumor}}, E, I; w_{\text{NN}}) \cdot T \\[6pt]
\frac{dE}{dt} &= k_{\text{prolif}} \, g_{\text{act}}(I) \, E - k_{\text{death}} \, E + k_{\text{recruit}} \, h(T) \\[6pt]
\frac{dI}{dt} &= k_{\text{on}} \, C_{\text{tumor}} \, (1 - I) - k_{\text{off}} \, I
\end{aligned}
$$

where:
- $g_{\text{act}}(I) = \frac{I^{\gamma}}{EC_{50}^{\gamma} + I^{\gamma}}$ — immune activation by checkpoint inhibition
- $h(T) = \frac{T}{T + T_{50}}$ — recruitment proportional to tumor burden
- $f_{\text{kill}}(C, E, I; w)$ — **ML-predicted killing function** (Track D Section 8)

**Quantum inputs:**
- $k_{\text{on}}, k_{\text{off}}$ from Track C (QM_Kinetics)
- $EC_{50}$ from $K_d$ via Track C (QM_BindingFreeEnergy)

### 2.3 Population Model

**PK parameters (log-normal IIV):**
$$
\begin{aligned}
\text{CL}_i &= \text{CL}_{\text{pop}} \left(\frac{\text{WT}_i}{70}\right)^{0.75} \exp(\eta_{\text{CL},i}) \\
V_i &= V_{\text{pop}} \left(\frac{\text{WT}_i}{70}\right)^{1.0} \exp(\eta_{V,i})
\end{aligned}
$$

**QSP parameters (log-normal IIV):**
$$
\begin{aligned}
k_{\text{grow},i} &= k_{\text{grow,pop}} \exp(\eta_{\text{grow},i}) \\
k_{\text{prolif},i} &= k_{\text{prolif,pop}} \exp(\eta_{\text{prolif},i})
\end{aligned}
$$

**Quantum-derived parameters (with calibration):**
$$
\begin{aligned}
K_{p,\text{tumor},i} &= K_{p,\text{QM}} \cdot f_{\text{ML}}(G_i; w_{\text{Kp}}) \cdot \exp(\eta_{K_p,i}) \\
EC_{50,i} &= \alpha_{\text{EC50}} \cdot K_{d,\text{QM}} \cdot \exp(\eta_{\text{EC50},i})
\end{aligned}
$$

where:
- $K_{p,\text{QM}} = \exp(-\Delta G_{\text{partition}} / RT)$ (Track C output)
- $K_{d,\text{QM}} = \exp(\Delta G_{\text{bind}} / RT) \cdot C^{\circ}$ (Track C output)

**Observation models:**
- Tumor volume: $T_{\text{obs},ij} \sim \mathcal{N}(T_i(t_{ij}), \sigma_T^2)$ (additive error)
- Effector count: $E_{\text{obs},ij} \sim \mathcal{N}(E_i(t_{ij}), (\sigma_E \cdot E_i)^2)$ (proportional error)
- Plasma concentration: $C_{\text{obs},ij} \sim \mathcal{N}(C_{\text{blood},i}(t_{ij}), (\sigma_C \cdot C)^2)$ (proportional error)

---

## 3. MedLang Implementation (Track C + Track D)

### 3.1 Track C: Quantum Calculations (Precomputed)

```medlang
// =============================
// Drug and target structures
// =============================
let drug = Molecule::from_smiles("CC(C)NCC(COc1ccc(COCCOC(C)C)cc1)O")  // checkpoint inhibitor
let pdl1 = Molecule::from_pdb("5J89", chain = "A")  // PD-L1 structure

// =============================
// Quantum binding calculation
// =============================
let binding = QM_BindingFreeEnergy {
    ligand   = drug,
    target   = pdl1,
    binding_pose = docking_result.best_pose,
    method   = QMMethod {
        theory     = DFT,
        functional = wB97XD,
        basis_set  = Def2_TZVP,
        dispersion = D3
    },
    qmmm_partition = QMMM_System {
        qm_region     = drug + select_atoms(radius = 5.0 Å around drug),
        mm_region     = rest_of(pdl1),
        boundary      = LinkAtom(H),
        mm_forcefield = AMBER_ff14SB
    },
    temperature = 310.0 K
}

let ΔG_bind_qm : Energy = binding.ΔG_bind        // e.g., -9.3 kcal/mol
let σ_ΔG       : Energy = binding.uncertainty    // e.g., ± 1.5 kcal/mol

// =============================
// Quantum partition coefficient
// =============================
let kp_tumor = QM_PartitionCoefficient {
    molecule = drug,
    phase_A  = SMD(solvent = "water"),           // plasma
    phase_B  = SMD(solvent = "tumor_tissue"),    // tumor interstitium
    method   = QMMethod {
        theory     = DFT,
        functional = M06_2X,
        basis_set  = Def2_TZVP
    },
    temperature = 310.0 K
}

let ΔG_partition_qm : Energy = kp_tumor.ΔG_partition  // e.g., -2.1 kcal/mol
let Kp_tumor_qm     : f64    = kp_tumor.Kp            // e.g., 0.68

// =============================
// Quantum kinetics
// =============================
let kinetics = QM_Kinetics {
    ligand        = drug,
    target        = pdl1,
    bound_complex = optimized_complex,
    method        = QMMethod {
        theory     = DFT,
        functional = wB97XD,
        basis_set  = Def2_SVP
    },
    temperature = 310.0 K
}

let k_on_qm  : RateConstPerConc = kinetics.k_on   // e.g., 8.2e5 1/(M·s)
let k_off_qm : RateConst        = kinetics.k_off  // e.g., 0.015 1/s
```

### 3.2 Track D: PBPK Model

```medlang
// =============================
// Multi-compartment PBPK
// =============================
model PBPK_5Compartment {
    // ---- States (drug amounts in each compartment) ----
    state A_blood : DoseMass    // mg
    state A_liver : DoseMass
    state A_kidney : DoseMass
    state A_tumor : DoseMass
    state A_periph : DoseMass
    
    // ---- Physiological parameters (allometrically scaled) ----
    param BW : BodyWeight       // kg (body weight)
    
    // Volumes (fraction of body weight)
    const fV_blood  : f64 = 0.07
    const fV_liver  : f64 = 0.026
    const fV_kidney : f64 = 0.004
    const fV_periph : f64 = 0.80
    
    let V_blood  : Volume = fV_blood  * BW
    let V_liver  : Volume = fV_liver  * BW
    let V_kidney : Volume = fV_kidney * BW
    let V_periph : Volume = fV_periph * BW
    
    // Tumor volume (state variable from QSP, converted to L)
    input V_tumor : Volume      // from QSP model
    
    // Blood flows (fraction of cardiac output)
    const CO_per_kg : FlowPerWeight = 5.0 L/h/kg  // cardiac output
    let CO : Flow = CO_per_kg * BW
    
    const fQ_liver  : f64 = 0.25
    const fQ_kidney : f64 = 0.19
    const fQ_tumor  : f64 = 0.01   // small fraction
    const fQ_periph : f64 = 0.55
    
    let Q_liver  : Flow = fQ_liver  * CO
    let Q_kidney : Flow = fQ_kidney * CO
    let Q_tumor  : Flow = fQ_tumor  * CO
    let Q_periph : Flow = fQ_periph * CO
    
    // ---- PK parameters ----
    param CL_hepatic : Clearance    // hepatic intrinsic clearance
    param CL_renal   : Clearance    // renal clearance
    
    // ---- Partition coefficients ----
    param Kp_liver  : f64           // liver/plasma
    param Kp_kidney : f64           // kidney/plasma
    param Kp_tumor  : f64           // tumor/plasma (QUANTUM-DERIVED)
    param Kp_periph : f64           // rest of body/plasma
    
    // ---- Concentrations (derived) ----
    let C_blood  : ConcMass = A_blood  / V_blood
    let C_liver  : ConcMass = A_liver  / V_liver
    let C_kidney : ConcMass = A_kidney / V_kidney
    let C_tumor  : ConcMass = A_tumor  / V_tumor
    let C_periph : ConcMass = A_periph / V_periph
    
    // ---- Dynamics (mass balance) ----
    dA_blood/dt =
        Q_liver  * (C_liver  / Kp_liver  - C_blood) +
        Q_kidney * (C_kidney / Kp_kidney - C_blood) +
        Q_tumor  * (C_tumor  / Kp_tumor  - C_blood) +
        Q_periph * (C_periph / Kp_periph - C_blood)
    
    dA_liver/dt =
        Q_liver * (C_blood - C_liver / Kp_liver)
        - CL_hepatic * C_liver
    
    dA_kidney/dt =
        Q_kidney * (C_blood - C_kidney / Kp_kidney)
        - CL_renal * C_kidney
    
    dA_tumor/dt =
        Q_tumor * (C_blood - C_tumor / Kp_tumor)
    
    dA_periph/dt =
        Q_periph * (C_blood - C_periph / Kp_periph)
    
    // ---- Observables ----
    obs C_plasma : ConcMass = C_blood
    obs C_tumor_obs : ConcMass = C_tumor
}
```

### 3.3 Track D: ML-Augmented QSP Model

```medlang
// =============================
// ML submodel: Tumor killing function
// =============================
model TumorKillingNN {
    const C_ref : ConcMass   = 1.0 mg/L
    const E_ref : CellDensity = 100.0 cells_per_uL
    const I_ref : f64        = 1.0
    
    param w : MLParamVector
    param k_kill_base : RateConst  // 1/h
    
    fn predict_kill_rate(
        C_tumor : ConcMass,
        E       : CellDensity,
        I       : f64
    ) -> RateConst {
        // Normalize inputs
        let C_norm : f64 = C_tumor / C_ref
        let E_norm : f64 = E / E_ref
        let I_norm : f64 = I / I_ref
        
        // NN forward: [C, E, I] -> f64
        let raw : f64 = NN_forward([C_norm, E_norm, I_norm], w)
        
        // Ensure positivity
        let multiplier : f64 = softplus(raw)
        
        return k_kill_base * multiplier
    }
}

// =============================
// QSP: Tumor-Immune Dynamics
// =============================
model TumorImmune_QSP {
    // ---- States ----
    state T : TumorVolume        // mm³
    state E : CellDensity        // cells/μL
    state I : f64                // PD-L1 inhibition level (dimensionless, [0,1])
    
    // ---- Inputs from PBPK ----
    input C_tumor : ConcMass     // tumor drug concentration from PBPK
    
    // ---- Tumor growth parameters ----
    param k_grow : RateConst     // 1/h
    param T_max  : TumorVolume   // mm³
    
    // ---- Immune parameters ----
    param k_prolif  : RateConst   // effector proliferation 1/h
    param k_death   : RateConst   // effector death 1/h
    param k_recruit : RecruitRate // recruitment rate
    param T_50      : TumorVolume // tumor size for half-max recruitment
    param EC50_act  : f64         // inhibition level for half-max activation
    param gamma_act : f64         // Hill coefficient
    
    // ---- Target binding kinetics (QUANTUM-DERIVED) ----
    param k_on  : RateConstPerConc  // 1/(M·s)
    param k_off : RateConst         // 1/s
    
    // ---- ML killing submodel ----
    param w_NN         : MLParamVector
    param k_kill_base  : RateConst
    
    let kill_model = TumorKillingNN {
        w = w_NN,
        k_kill_base = k_kill_base
    }
    
    // ---- Helper functions ----
    fn g_activation(I : f64) -> f64 {
        // Immune activation by checkpoint inhibition
        let num = pow(I, gamma_act)
        let den = pow(EC50_act, gamma_act) + num
        return num / den
    }
    
    fn h_recruitment(T : TumorVolume) -> f64 {
        // Recruitment proportional to tumor burden
        return T / (T + T_50)
    }
    
    // ---- Dynamics ----
    dT/dt =
        k_grow * T * (1.0 - T / T_max)
        - kill_model.predict_kill_rate(C_tumor, E, I) * T
    
    dE/dt =
        k_prolif * g_activation(I) * E
        - k_death * E
        + k_recruit * h_recruitment(T)
    
    dI/dt =
        k_on * C_tumor * (1.0 - I)
        - k_off * I
    
    // ---- Observables ----
    obs TumorVolume   : TumorVolume  = T
    obs EffectorCount : CellDensity  = E
    obs Inhibition    : f64          = I
}
```

### 3.4 Track D: Coupled PBPK + QSP Model

```medlang
// =============================
// Coupled system
// =============================
model PBPK_QSP_Coupled {
    // Instantiate submodels
    submodel pbpk : PBPK_5Compartment
    submodel qsp  : TumorImmune_QSP
    
    // ---- Coupling: tumor volume from QSP to PBPK ----
    pbpk.V_tumor = qsp.T * 1e-6 L/mm³  // convert mm³ to L
    
    // ---- Coupling: tumor drug concentration from PBPK to QSP ----
    qsp.C_tumor = pbpk.C_tumor_obs
    
    // ---- All parameters forwarded ----
    // (PBPK params)
    param BW          : BodyWeight
    param CL_hepatic  : Clearance
    param CL_renal    : Clearance
    param Kp_liver    : f64
    param Kp_kidney   : f64
    param Kp_tumor    : f64  // ← QUANTUM-DERIVED
    param Kp_periph   : f64
    
    // (QSP params)
    param k_grow      : RateConst
    param T_max       : TumorVolume
    param k_prolif    : RateConst
    param k_death     : RateConst
    param k_recruit   : RecruitRate
    param T_50        : TumorVolume
    param EC50_act    : f64  // ← From QUANTUM Kd via calibration
    param gamma_act   : f64
    param k_on        : RateConstPerConc  // ← QUANTUM-DERIVED
    param k_off       : RateConst         // ← QUANTUM-DERIVED
    param w_NN        : MLParamVector
    param k_kill_base : RateConst
    
    // ---- Observables ----
    obs C_plasma      : ConcMass     = pbpk.C_plasma
    obs TumorVolume   : TumorVolume  = qsp.TumorVolume
    obs EffectorCount : CellDensity  = qsp.EffectorCount
}
```

### 3.5 Track D: Population Model with Quantum Integration

```medlang
// =============================
// Population model
// =============================
population PBPK_QSP_PopModel {
    structural_model = PBPK_QSP_Coupled
    
    // ---- Population fixed effects ----
    hyperparam CL_hepatic_pop  : Clearance
    hyperparam CL_renal_pop    : Clearance
    hyperparam Kp_liver_pop    : f64 = 1.2
    hyperparam Kp_kidney_pop   : f64 = 0.8
    hyperparam Kp_periph_pop   : f64 = 0.5
    
    hyperparam k_grow_pop      : RateConst
    hyperparam k_prolif_pop    : RateConst
    hyperparam k_death_pop     : RateConst
    hyperparam k_recruit_pop   : RecruitRate
    
    hyperparam T_max_pop       : TumorVolume
    hyperparam T_50_pop        : TumorVolume
    hyperparam gamma_act_pop   : f64 = 2.0
    
    // ---- Quantum-derived parameters (with calibration) ----
    // From Track C precomputed results
    const ΔG_bind_qm       : Energy          = -9.3 kcal/mol
    const ΔG_partition_qm  : Energy          = -2.1 kcal/mol
    const k_on_qm          : RateConstPerConc = 8.2e5 1/(M·s)
    const k_off_qm         : RateConst        = 0.015 1/s
    
    // Calibration factors (to be estimated from data)
    hyperparam alpha_EC50  : f64   // EC50 = alpha * Kd
    hyperparam alpha_Kp    : f64   // Kp_tumor = alpha * Kp_QM * ...
    hyperparam beta_on     : f64   // kinetics scaling exponents
    hyperparam beta_off    : f64
    
    // QM uncertainty (for Bayesian mode)
    hyperparam sigma_ΔG_bind : Energy = 1.5 kcal/mol
    hyperparam sigma_ΔG_part : Energy = 0.8 kcal/mol
    
    // ---- ML submodel (shared weights) ----
    hyperparam w_NN_shared     : MLParamVector
    hyperparam k_kill_base_pop : RateConst
    
    // ---- Random effects covariance ----
    hyperparam Omega : CovMatrix<6>  // [CL, k_grow, k_prolif, k_death, Kp_tumor, EC50]
    
    // ---- Residual error ----
    hyperparam sigma_T : TumorVolume
    hyperparam sigma_E : f64
    hyperparam sigma_C : f64
    
    // ---- Covariates ----
    covariate BW : BodyWeight
    
    // ---- Random effects ----
    random_effects eta : Vector<6, f64> {
        distribution = MVNormal(mean = zeros(6), cov = Omega)
        labels = ["eta_CL", "eta_k_grow", "eta_k_prolif", "eta_k_death", "eta_Kp", "eta_EC50"]
    }
    
    // ---- Parameter transformations ----
    transform individual_params(
        BW  : BodyWeight,
        eta : Vector<6, f64>
    ) -> CoupledParams {
        // Allometric scaling
        let w_scaled : f64 = BW / (70.0 kg)
        
        // PK parameters with IIV
        let CL_hepatic_i = CL_hepatic_pop * pow(w_scaled, 0.75) * exp(eta[0])
        let CL_renal_i   = CL_renal_pop   * pow(w_scaled, 0.75) * exp(eta[0])
        
        // Quantum-derived Kp_tumor
        let Kp_tumor_qm : f64 = Kp_from_ΔG_partition(ΔG_partition_qm, T = 310.0 K)
        let Kp_tumor_i  : f64 = alpha_Kp * Kp_tumor_qm * exp(eta[4])
        
        // Quantum-derived EC50
        let Kd_qm    = Kd_from_ΔG(ΔG_bind_qm, T = 310.0 K)
        let EC50_i   = alpha_EC50 * Kd_qm * exp(eta[5])
        
        // QSP parameters with IIV
        let k_grow_i   = k_grow_pop   * exp(eta[1])
        let k_prolif_i = k_prolif_pop * exp(eta[2])
        let k_death_i  = k_death_pop  * exp(eta[3])
        
        // Kinetics (quantum-derived, with optional scaling)
        let k_on_i  = k_on_qm   // or apply f_QM_kill scaling
        let k_off_i = k_off_qm
        
        return CoupledParams {
            BW          = BW,
            CL_hepatic  = CL_hepatic_i,
            CL_renal    = CL_renal_i,
            Kp_liver    = Kp_liver_pop,
            Kp_kidney   = Kp_kidney_pop,
            Kp_tumor    = Kp_tumor_i,     // ← QUANTUM + IIV
            Kp_periph   = Kp_periph_pop,
            k_grow      = k_grow_i,
            k_prolif    = k_prolif_i,
            k_death     = k_death_i,
            k_recruit   = k_recruit_pop,
            T_max       = T_max_pop,
            T_50        = T_50_pop,
            EC50_act    = EC50_i,         // ← QUANTUM + IIV
            gamma_act   = gamma_act_pop,
            k_on        = k_on_i,         // ← QUANTUM
            k_off       = k_off_i,        // ← QUANTUM
            w_NN        = w_NN_shared,
            k_kill_base = k_kill_base_pop
        }
    }
    
    // ---- Observation models ----
    observation_model tumor_obs = AdditiveError {
        channel     = PBPK_QSP_Coupled.TumorVolume
        error_param = sigma_T
    }
    
    observation_model effector_obs = ProportionalError {
        channel     = PBPK_QSP_Coupled.EffectorCount
        error_param = sigma_E
    }
    
    observation_model pk_obs = ProportionalError {
        channel     = PBPK_QSP_Coupled.C_plasma
        error_param = sigma_C
    }
}
```

### 3.6 Bayesian Inference Configuration

```medlang
// =============================
// Bayesian inference with QM-informed priors
// =============================
inference PBPK_QSP_Bayesian {
    population_model = PBPK_QSP_PopModel
    cohort           = oncology_trial_data
    
    mode      = Bayesian
    backend   = Backend.Stan
    algorithm = Algorithm.NUTS
    
    priors {
        // ---- PK parameters ----
        CL_hepatic_pop ~ LogNormal(log(15.0 L/h), 0.5)
        CL_renal_pop   ~ LogNormal(log(5.0 L/h), 0.5)
        
        // ---- QSP parameters ----
        k_grow_pop   ~ LogNormal(log(0.01 /h), 0.7)
        k_prolif_pop ~ LogNormal(log(0.02 /h), 0.7)
        k_death_pop  ~ LogNormal(log(0.01 /h), 0.5)
        k_recruit_pop ~ LogNormal(log(0.001 /h), 0.5)
        
        T_max_pop ~ LogNormal(log(10000.0 mm³), 0.5)
        T_50_pop  ~ LogNormal(log(1000.0 mm³), 0.5)
        
        // ---- Quantum calibration factors ----
        // These are the key parameters: how much to trust QM vs. data
        alpha_EC50 ~ Normal(1.0, 0.5)   // centered at 1 = trust QM
        alpha_Kp   ~ Normal(1.0, 0.3)
        
        beta_on    ~ Normal(0.5, 0.3)
        beta_off   ~ Normal(0.5, 0.3)
        
        // ---- ML killing submodel ----
        k_kill_base_pop ~ LogNormal(log(0.001 /h), 1.0)
        
        // Prior on NN weights (regularization)
        for w in w_NN_shared {
            w ~ Normal(0.0, 1.0)
        }
        
        // ---- Random effects ----
        Omega ~ LKJCov(eta = 2.0, scale_prior = HalfNormal(0.5))
        
        // ---- Residual errors ----
        sigma_T ~ HalfNormal(100.0 mm³)
        sigma_E ~ HalfNormal(0.3)
        sigma_C ~ HalfNormal(0.2)
        
        // ---- Optional: QM uncertainty propagation ----
        // Allow ΔG_bind to vary around QM prediction
        ΔG_bind_true ~ Normal(
            mean = ΔG_bind_qm,
            sd   = sigma_ΔG_bind
        )
        
        // Use ΔG_bind_true instead of ΔG_bind_qm in Kd calculation
        // This quantifies agreement/tension between QM and clinical data
    }
}
```

---

## 4. What This Tests

### 4.1 Track C → Track D Interface

✅ **Quantum outputs consumed as typed quantities:**
- `ΔG_bind : Energy` → `Kd : Concentration` → `EC50_i : Concentration`
- `ΔG_partition : Energy` → `Kp_tumor : f64`
- `k_on : RateConstPerConc`, `k_off : RateConst` → `dI/dt` kinetics

✅ **Unit safety through thermodynamic mappings:**
- `exp(ΔG / (R·T))` — dimensionless exponent ✓
- `Kd` has units `[Concentration]` ✓
- `Kp` is dimensionless ✓

✅ **Calibration parameters allow data-driven correction:**
- `alpha_EC50`, `alpha_Kp` as free parameters
- Posterior tells us if QM predictions align with clinical outcomes

### 4.2 Multi-Scale Coupling (PBPK ↔ QSP)

✅ **Bidirectional coupling:**
- PBPK → QSP: `C_tumor` (drug concentration in tumor)
- QSP → PBPK: `V_tumor` (tumor volume affects PBPK compartment size)

✅ **Unit consistency at interfaces:**
- `V_tumor : Volume` conversion from `T : TumorVolume` via `mm³ → L`
- `C_tumor : ConcMass` flows from PBPK to QSP

### 4.3 ML Integration (Section 8)

✅ **Dynamics-level hybrid:**
- `f_kill(C, E, I; w_NN)` embedded in `dT/dt`

✅ **Unit safety:**
- NN inputs normalized to dimensionless: `C/C_ref`, `E/E_ref`, `I/I_ref`
- NN output re-dimensionalized: `k_kill_base * softplus(NN_out)` → `[1/h]`

✅ **Shared weights across population:**
- `w_NN_shared` with prior for regularization

### 4.4 Population Modeling (Section 6)

✅ **Hierarchical structure:**
- Fixed effects (population typical values)
- Random effects (IIV via MVNormal)
- Covariates (allometric scaling by body weight)

✅ **Complex covariance structure:**
- `Omega : CovMatrix<6>` allows correlation between PK and PD random effects

### 4.5 Bayesian Inference (Section 7)

✅ **QM-informed priors:**
- `ΔG_bind_true ~ Normal(ΔG_bind_qm, sigma_ΔG_bind)`
- Allows quantification of QM prediction uncertainty

✅ **Calibration inference:**
- `alpha_EC50 ~ Normal(1.0, 0.5)` — if posterior mean ≈ 1.0, QM is accurate
- If posterior mean differs significantly, indicates systematic QM bias

✅ **Full uncertainty propagation:**
- QM uncertainty → parameter priors → PK/PD predictions → clinical endpoints

---

## 5. Execution Semantics and IR Lowering

### 5.1 CIR Representation

**Track C operators become CIR constants:**
```mlir
cir.const @ΔG_bind_qm       : !qty<kcal/mol, f64> = -9.3
cir.const @ΔG_partition_qm  : !qty<kcal/mol, f64> = -2.1
cir.const @k_on_qm          : !qty<1/(M·s), f64>  = 8.2e5
cir.const @k_off_qm         : !qty<1/s, f64>      = 0.015
```

**Population model with quantum mappings:**
```mlir
cir.population_model @PBPK_QSP_PopModel {
    structural_model = @PBPK_QSP_Coupled
    
    hyperparams {
        alpha_EC50 : f64
        alpha_Kp   : f64
        ...
    }
    
    transform @individual_params {
        // Kp from quantum
        %Kp_qm = cir.call @Kp_from_ΔG_partition(@ΔG_partition_qm, 310.0 K)
        %Kp_i  = cir.mul(%alpha_Kp, %Kp_qm) : f64
        %Kp_i_final = cir.mul(%Kp_i, cir.exp(%eta[4])) : f64
        
        // EC50 from quantum
        %Kd_qm = cir.call @Kd_from_ΔG(@ΔG_bind_qm, 310.0 K)
        %EC50_i = cir.mul(%alpha_EC50, %Kd_qm) : !qty<Concentration>
        ...
    }
}
```

### 5.2 NIR Representation

**Coupled ODE system (PBPK + QSP):**
```mlir
// State vector: [A_blood, A_liver, A_kidney, A_tumor, A_periph, T, E, I]
func @coupled_rhs(
    %X     : tensor<8xf64>,
    %theta : tensor<?xf64>,
    %t     : f64
) -> tensor<8xf64> {
    // Extract PBPK states
    %A_blood  = tensor.extract %X[0]
    %A_liver  = tensor.extract %X[1]
    ...
    
    // Extract QSP states
    %T = tensor.extract %X[5]
    %E = tensor.extract %X[6]
    %I = tensor.extract %X[7]
    
    // Coupling: V_tumor from T
    %V_tumor = arith.mulf %T, %mm3_to_L
    
    // PBPK dynamics
    %dA_blood = ... // uses V_tumor
    ...
    
    // Coupling: C_tumor from PBPK
    %C_tumor = arith.divf %A_tumor, %V_tumor
    
    // QSP dynamics (with ML call)
    %C_norm = arith.divf %C_tumor, %C_ref
    %E_norm = arith.divf %E, %E_ref
    %I_norm = arith.divf %I, %I_ref
    %nn_input = tensor.from_elements %C_norm, %E_norm, %I_norm
    
    %raw_kill = nir.ml_call @tumor_kill_nn(%nn_input, %w_NN)
    %mult = math.softplus %raw_kill
    %f_kill = arith.mulf %k_kill_base, %mult
    
    %dT = ... // growth - kill
    %dE = ... // prolif - death + recruit
    %dI = ... // binding kinetics
    
    %dX = tensor.from_elements %dA_blood, ..., %dT, %dE, %dI
    return %dX
}
```

**Batched execution over population:**
```mlir
%traj_batch = nir.ode_integrate_batch(
    @coupled_rhs,
    %X0_batch   : tensor<?x8xf64>,    // [N_patients, 8 states]
    %theta_batch : tensor<?x?xf64>,   // [N_patients, n_params]
    %t_grid,
    %solver_cfg
) -> tensor<?x?x8xf64>  // [N_patients, n_time, 8]
```

---

## 6. Validation Protocol

### 6.1 Synthetic Data Generation

1. **Fix "true" parameters** (including quantum values).
2. **Generate virtual patients** with known IIV.
3. **Simulate PBPK + QSP** forward to get tumor volumes, effector counts, PK.
4. **Add measurement noise** per observation model.

### 6.2 Parameter Recovery

1. **Fit model to synthetic data** (frequentist or Bayesian).
2. **Check:**
   - Estimated population parameters vs. true values
   - Coverage of credible/confidence intervals
   - Bias, MSE, shrinkage

### 6.3 Cross-Backend Validation

1. **Export to Stan:** Use MedLang → Stan compiler.
2. **Export to custom GPU solver:** Use MedLang → NIR → MLIR → CUDA.
3. **Compare:**
   - Posterior means, SDs
   - ESS, R-hat (MCMC diagnostics)
   - Wall-clock time

### 6.4 QM Validation

1. **Run with `alpha_EC50 = 1.0` (fixed)** → trust QM completely.
2. **Run with `alpha_EC50 ~ prior`** → let data correct QM.
3. **Compare posteriors:**
   - If posterior(alpha_EC50) ≈ 1.0 → QM predictions are accurate.
   - If posterior(alpha_EC50) differs → quantify systematic bias.

---

## 7. Expected Outcomes

If the spec is **sound**:

✅ **Type checking passes** — all unit conversions are explicit and correct.  
✅ **CIR lowering succeeds** — quantum mappings, PBPK-QSP coupling, ML calls representable.  
✅ **NIR compiles** — ODE + ML differentiation graph is constructible.  
✅ **Backends execute** — Stan, MLIR/GPU, NONMEM-export all run.  
✅ **Parameter recovery works** — synthetic data → fit → true parameters recovered.  
✅ **QM calibration is interpretable** — `alpha_EC50` posterior tells us QM accuracy.

If the spec has **gaps**:

❌ Unit mismatch in PBPK-QSP coupling → compile error at CIR.  
❌ ML unit safety violation → `exp(dimensionful quantity)` → type error.  
❌ Quantum mapping ambiguity → unclear how `ΔG → Kp` with units.  
❌ Population structure incoherent → random effects don't compose with quantum params.  
❌ Backend export fails → NIR → Stan translation breaks on quantum terms.

---

## 8. Conclusion

This stress test exercises **every major feature** of MedLang:

- **Track C:** Quantum binding, partition, kinetics
- **Track D PBPK:** Multi-organ, allometric scaling, quantum Kp
- **Track D QSP:** Tumor-immune, ML killing, quantum EC50/kinetics
- **Track D NLME:** IIV, covariates, complex Omega
- **Track D Bayesian:** QM-informed priors, calibration inference
- **Section 8 ML:** Dynamics-level NN with unit safety
- **Section 10 IR:** CIR → NIR → MLIR compilation

If this compiles, type-checks, and executes correctly, **the MedLang spec is validated** as a coherent, multi-scale, quantum-to-clinical programming language for computational medicine.

---

*This is the ultimate validation case for MedLang v0.1. Implementation of this example will drive refinement of Sections 4 (typing), 7 (inference), 8 (ML), 10 (IR), and the Track C spec.*
