# MedLang Quantum Pharmacology Specification — Track C v0.1 (Draft)

**Status:** Draft  
**Scope:** Quantum mechanical calculations for drug discovery and pharmacology: electronic structure, binding free energies, partition coefficients, kinetic parameters, and quantum-to-classical parameter mappings for consumption by Track D (Pharmacometrics/QSP).

---

## 1. Introduction and Scope

MedLang **Track C** provides a domain-specific language for **quantum pharmacology**: the use of quantum mechanical (QM) calculations to predict molecular properties, binding affinities, partition coefficients, and kinetic parameters that govern drug action.

Track C is designed to:

1. **Generate physically grounded molecular descriptors** for use in Track D models.
2. **Replace empirical parameters** (EC50, Kd, Kp, CL) with **quantum-derived predictions** where appropriate.
3. **Quantify uncertainty** in QM predictions for propagation through pharmacometric models.
4. **Integrate multi-scale calculations**: pure QM (DFT, wavefunction methods), QM/MM (ligand-protein complexes), continuum solvation, and classical force fields.

### 1.1 What Track C Provides

Track C defines **quantum operators** that map molecular and environmental specifications to **typed physical quantities**:

- `QM_SCF` — self-consistent field (Hartree-Fock, DFT) single-point energy and properties.
- `QM_Optimize` — geometry optimization at QM level.
- `QM_BindingFreeEnergy` — free energy of binding (ligand-protein, ligand-receptor).
- `QM_PartitionCoefficient` — logP, Kp (tissue/plasma partitioning).
- `QM_Kinetics` — k_on, k_off, activation barriers.
- `QM_MM` — hybrid QM/MM for large biomolecular systems.
- `QM_Solvent` — implicit (continuum) or explicit solvation.

Each operator:
- Accepts typed inputs (molecular structures, environments, method specifications).
- Produces typed outputs (energies, gradients, rate constants, partition coefficients).
- Is **backend-agnostic**: can be executed by quantum chemistry engines (Gaussian, ORCA, Psi4, CP2K, Q-Chem, etc.) or ML surrogate models.

### 1.2 Track C ↔ Track D Interface

**Track C outputs** are consumed by **Track D** (pharmacometrics/QSP) as:
- **Covariates** (per-molecule, per-target, per-environment).
- **Parameter priors or constraints** (Bayesian or penalized NLME).
- **Direct parameter values** (when QM predictions are trusted).

Example data flow:
```
Drug molecule (SMILES, 3D structure)
    ↓ Track C
QM_BindingFreeEnergy → ΔG_bind [Energy]
    ↓ Track D
Kd = exp(ΔG_bind / RT) · C⁰ [Concentration]
    ↓ Track D
EC50 = α · Kd + random effects
    ↓ Track D
PD model (Emax, tumor killing, etc.)
```

### 1.3 Design Principles

1. **Physical units from first principles**
   - All QM energies in consistent units (e.g., Hartree, kcal/mol, kJ/mol).
   - Automatic conversion to Track D units (Quantity<Energy, f64>).

2. **Multi-level theory support**
   - Operators accept method specifications (HF, DFT functional, MP2, CCSD(T), etc.).
   - Backends validate supported methods and report errors for unsupported ones.

3. **Uncertainty quantification**
   - QM predictions include **epistemic uncertainty** (method error, basis set incompleteness).
   - Represented as probability distributions for Track D consumption.

4. **Reproducibility and provenance**
   - QM operator invocations record: method, basis set, software version, convergence criteria.
   - Outputs are cacheable and traceable.

5. **Integration with classical MD and docking**
   - QM operators can consume outputs from classical docking (initial poses) or MD (conformational ensembles).
   - Hybrid QM/MM workflows are first-class citizens.

---

## 2. Core Track C Types and Constructs

### 2.1 Molecular Representations

Track C uses **typed molecular objects**:

```medlang
// Molecular structure
struct Molecule {
    atoms     : List<Atom>          // atomic numbers, positions
    charge    : Int                 // total molecular charge
    multiplicity : Int              // spin multiplicity (1 = singlet, 2 = doublet, etc.)
    identifier : MoleculeId         // SMILES, InChI, PDB ID, or custom
}

struct Atom {
    element  : Element              // H, C, N, O, etc.
    position : Quantity<Length, Vector<3, f64>>  // Å or Bohr
    charge   : f64                  // partial charge (optional, for MM region)
}

enum MoleculeId {
    SMILES(String),
    InChI(String),
    PDB(String, ChainId),
    Custom(String)
}
```

### 2.2 Quantum Method Specifications

```medlang
struct QMMethod {
    theory     : QMTheory           // HF, DFT, MP2, CCSD(T), etc.
    basis_set  : BasisSet           // 6-31G*, def2-TZVP, aug-cc-pVTZ, etc.
    functional : Option<Functional> // B3LYP, PBE, ωB97X-D, etc. (for DFT)
    dispersion : Option<Dispersion> // D3, D4, etc.
    solvent    : Option<SolventModel>
}

enum QMTheory {
    HartreeFock,
    DFT,
    MP2,
    CCSD,
    CCSD_T,
    CASSCF(active_space),
    Custom(String)
}

enum BasisSet {
    STO_3G,
    Pople_6_31Gd,         // 6-31G*
    Pople_6_311Gdp,       // 6-311G**
    Def2_SVP,
    Def2_TZVP,
    AugCCPVDZ,
    AugCCPVTZ,
    Custom(String)
}

enum Functional {
    B3LYP,
    PBE,
    PBE0,
    wB97XD,
    M06_2X,
    Custom(String)
}

enum Dispersion {
    D3,
    D4,
    None
}

enum SolventModel {
    PCM(epsilon: f64),    // Polarizable Continuum Model
    COSMO(epsilon: f64),
    SMD(solvent: String), // Universal solvation model
    Explicit(solvent_box: Molecule)
}
```

### 2.3 QM/MM Partitioning

For large systems (e.g., ligand-protein complexes):

```medlang
struct QMMM_System {
    qm_region  : Molecule           // ligand or active site
    mm_region  : Molecule           // protein, membrane, solvent
    boundary   : QM_MM_Boundary     // how to treat QM/MM interface
    mm_forcefield : ForceField      // AMBER, CHARMM, OPLS, etc.
}

enum QM_MM_Boundary {
    LinkAtom(element: Element),     // cap with H or other atom
    FrozenOrbital,
    ElectrostaticEmbedding
}

enum ForceField {
    AMBER_ff14SB,
    CHARMM36,
    OPLS_AA,
    Custom(String)
}
```

---

## 3. Core Track C Operators

### 3.1 `QM_SCF`: Single-Point Energy and Properties

Performs a self-consistent field calculation (HF or DFT) at a fixed geometry.

**Signature:**
```medlang
operator QM_SCF {
    input molecule : Molecule
    input method   : QMMethod
    
    // Computational settings
    param convergence  : f64 = 1e-8       // SCF convergence threshold
    param max_iter     : Int = 100
    
    // Outputs
    obs energy         : Quantity<Energy, f64>           // Electronic energy
    obs dipole_moment  : Quantity<Debye, Vector<3, f64>> // Dipole
    obs homo_energy    : Quantity<Energy, f64>           // HOMO energy
    obs lumo_energy    : Quantity<Energy, f64>           // LUMO energy
    obs mulliken_charges : Vector<N_atoms, f64>          // Atomic charges
    
    // Provenance
    obs backend        : String                          // "Psi4", "ORCA", etc.
    obs wall_time      : Quantity<Second, f64>
}
```

**Example usage:**
```medlang
let aspirin = Molecule::from_smiles("CC(=O)Oc1ccccc1C(=O)O")

let scf_result = QM_SCF {
    molecule = aspirin,
    method   = QMMethod {
        theory     = DFT,
        functional = B3LYP,
        basis_set  = Def2_TZVP,
        dispersion = D3,
        solvent    = PCM(epsilon = 78.4)  // water
    }
}

let E_aspirin : Energy = scf_result.energy  // e.g., -457.234 Hartree
```

### 3.2 `QM_Optimize`: Geometry Optimization

Finds a local minimum on the potential energy surface.

**Signature:**
```medlang
operator QM_Optimize {
    input initial_geometry : Molecule
    input method           : QMMethod
    
    param convergence      : f64 = 1e-6   // gradient norm threshold
    param max_steps        : Int = 200
    
    obs optimized_geometry : Molecule
    obs final_energy       : Quantity<Energy, f64>
    obs gradient_norm      : f64
    obs num_steps          : Int
}
```

**Example:**
```medlang
let opt_result = QM_Optimize {
    initial_geometry = aspirin,
    method = QMMethod {
        theory    = DFT,
        functional = wB97XD,
        basis_set = Def2_SVP
    }
}

let aspirin_opt : Molecule = opt_result.optimized_geometry
```

### 3.3 `QM_BindingFreeEnergy`: Ligand-Target Binding

Computes ΔG_bind for a ligand-protein/receptor complex.

**Method:** QM/MM + free energy perturbation, thermodynamic integration, or surrogate ML model trained on QM data.

**Signature:**
```medlang
operator QM_BindingFreeEnergy {
    input ligand        : Molecule
    input target        : Molecule              // protein, receptor, DNA, etc.
    input binding_pose  : Pose                  // ligand position/orientation in target
    input method        : QMMethod              // for QM region (ligand + nearby residues)
    input qmmm_partition : QMMM_System
    
    param temperature   : Quantity<Kelvin, f64> = 310.0 K
    param pressure      : Quantity<Pressure, f64> = 1.0 atm
    
    // Outputs
    obs ΔG_bind         : Quantity<Energy, f64>      // Free energy of binding
    obs ΔH_bind         : Quantity<Energy, f64>      // Enthalpy (optional)
    obs TΔS_bind        : Quantity<Energy, f64>      // Entropy term (optional)
    obs uncertainty     : Quantity<Energy, f64>      // Estimated method uncertainty
    
    obs provenance      : String
}
```

**Example:**
```medlang
let drug = Molecule::from_smiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")  // caffeine
let target = Molecule::from_pdb("1A4G", chain = "A")              // adenosine receptor

let docking_result = ... // from classical docking, gives initial pose

let binding = QM_BindingFreeEnergy {
    ligand       = drug,
    target       = target,
    binding_pose = docking_result.best_pose,
    method       = QMMethod {
        theory     = DFT,
        functional = wB97XD,
        basis_set  = Def2_SVP
    },
    qmmm_partition = QMMM_System {
        qm_region = select_atoms(radius = 5.0 Å around ligand),
        mm_region = rest_of(target),
        boundary  = LinkAtom(H),
        mm_forcefield = AMBER_ff14SB
    },
    temperature = 310.0 K
}

let ΔG : Energy = binding.ΔG_bind  // e.g., -8.2 kcal/mol
```

### 3.4 `QM_PartitionCoefficient`: Tissue/Plasma Partitioning

Computes Kp (tissue-to-plasma partition coefficient) or logP (octanol-water).

**Method:** QM solvation free energies in two phases.

**Signature:**
```medlang
operator QM_PartitionCoefficient {
    input molecule      : Molecule
    input phase_A       : SolventModel      // e.g., water (plasma)
    input phase_B       : SolventModel      // e.g., octanol, lipid bilayer
    input method        : QMMethod
    
    param temperature   : Quantity<Kelvin, f64> = 310.0 K
    
    // Outputs
    obs ΔG_partition    : Quantity<Energy, f64>   // ΔG(B) - ΔG(A)
    obs log_Kp          : f64                     // log10(Kp) where Kp = exp(-ΔG/RT)
    obs Kp              : f64                     // dimensionless partition coefficient
    obs uncertainty     : f64                     // ± log units
}
```

**Example:**
```medlang
let logP_result = QM_PartitionCoefficient {
    molecule = drug,
    phase_A  = SMD(solvent = "water"),
    phase_B  = SMD(solvent = "octanol"),
    method   = QMMethod {
        theory     = DFT,
        functional = M06_2X,
        basis_set  = Def2_TZVP
    }
}

let logP : f64 = logP_result.log_Kp  // e.g., 2.3
```

### 3.5 `QM_Kinetics`: Rate Constants (k_on, k_off)

Computes association and dissociation rate constants from transition-state theory.

**Method:** Locate transition state, compute ΔG‡, apply Eyring equation.

**Signature:**
```medlang
operator QM_Kinetics {
    input ligand            : Molecule
    input target            : Molecule
    input bound_complex     : Molecule          // optimized complex
    input method            : QMMethod
    
    param temperature       : Quantity<Kelvin, f64> = 310.0 K
    
    // Outputs
    obs ΔG_barrier_on       : Quantity<Energy, f64>     // Association barrier
    obs ΔG_barrier_off      : Quantity<Energy, f64>     // Dissociation barrier
    obs k_on                : Quantity<RateConstPerConc, f64>  // 1/(M·s)
    obs k_off               : Quantity<RateConst, f64>         // 1/s
    obs tau_residence       : Quantity<Second, f64>            // 1/k_off
    obs uncertainty_k_on    : Quantity<RateConstPerConc, f64>
    obs uncertainty_k_off   : Quantity<RateConst, f64>
}
```

**Example:**
```medlang
let kinetics = QM_Kinetics {
    ligand   = drug,
    target   = target,
    bound_complex = optimized_complex,
    method   = QMMethod {
        theory    = DFT,
        functional = wB97XD,
        basis_set = Def2_SVP
    },
    temperature = 310.0 K
}

let k_on  : RateConstPerConc = kinetics.k_on   // e.g., 1.2e6 1/(M·s)
let k_off : RateConst        = kinetics.k_off  // e.g., 0.05 1/s
let Kd : Concentration = k_off / k_on          // e.g., 41.7 nM
```

### 3.6 `QM_MM`: Hybrid QM/MM Simulation

Runs molecular dynamics or Monte Carlo with QM treatment of active region.

**Signature:**
```medlang
operator QM_MM {
    input system        : QMMM_System
    input qm_method     : QMMethod
    input md_settings   : MDSettings
    
    // Outputs
    obs trajectory      : Trajectory<QMMM_System>
    obs avg_energy      : Quantity<Energy, f64>
    obs qm_region_charges : TimeSeries<Vector<N_qm, f64>>
}

struct MDSettings {
    temperature   : Quantity<Kelvin, f64>
    pressure      : Option<Quantity<Pressure, f64>>
    timestep      : Quantity<Second, f64>
    total_time    : Quantity<Second, f64>
    thermostat    : Thermostat
    barostat      : Option<Barostat>
}
```

---

## 4. Track C → Track D Parameter Mappings

This section formalizes how Track C outputs are consumed by Track D.

### 4.1 Binding Affinity → EC50

**Thermodynamic relation:**
$$
K_d = \exp\left(\frac{\Delta G_{\text{bind}}}{RT}\right) \cdot C^\circ
$$

where $C^\circ$ is a standard concentration (e.g., 1 M).

In Track D, EC50 is often proportional to Kd:
$$
\text{EC50}_i = \alpha_{\text{EC50}} \cdot K_{d,i} \cdot \exp(\eta_{\text{EC50},i})
$$

**MedLang mapping function:**
```medlang
fn Kd_from_ΔG(
    ΔG_bind : Quantity<Energy, f64>,
    T       : Quantity<Kelvin, f64>
) -> Quantity<Concentration, f64> {
    let R  : EnergyPerMolPerK = 8.314e-3 kJ/(mol·K)  // gas constant
    let C0 : Concentration    = 1.0 M                // standard concentration
    
    let exponent : f64 = (ΔG_bind / (R * T))         // dimensionless
    return C0 * exp(exponent)
}

fn EC50_from_Kd(
    Kd          : Quantity<Concentration, f64>,
    alpha_EC50  : f64,                              // calibration factor
    eta_EC50    : f64                               // random effect
) -> Quantity<Concentration, f64> {
    return alpha_EC50 * Kd * exp(eta_EC50)
}
```

**Usage in Track D model:**
```medlang
// Track C computation
let binding = QM_BindingFreeEnergy { ligand = drug, target = receptor, ... }
let ΔG_bind = binding.ΔG_bind

// Track D parameter mapping
param alpha_EC50 : f64              // population calibration
rand  eta_EC50   : f64 ~ Normal(0, omega_EC50^2)

let Kd_i   = Kd_from_ΔG(ΔG_bind, T = 310.0 K)
let EC50_i = EC50_from_Kd(Kd_i, alpha_EC50, eta_EC50)

// Use in PD model
model PD_Emax {
    param EC50 : Concentration = EC50_i
    ...
}
```

### 4.2 Partition Free Energy → Kp

**Thermodynamic relation:**
$$
K_p = \exp\left(-\frac{\Delta G_{\text{partition}}}{RT}\right)
$$

where $\Delta G_{\text{partition}} = G_{\text{tissue}} - G_{\text{plasma}}$.

**MedLang mapping:**
```medlang
fn Kp_from_ΔG_partition(
    ΔG_part : Quantity<Energy, f64>,
    T       : Quantity<Kelvin, f64>
) -> f64 {
    let R : EnergyPerMolPerK = 8.314e-3 kJ/(mol·K)
    let exponent : f64 = -(ΔG_part / (R * T))  // dimensionless
    return exp(exponent)                       // dimensionless Kp
}
```

**Usage in PBPK:**
```medlang
// Track C
let kp_qm_result = QM_PartitionCoefficient {
    molecule = drug,
    phase_A  = SMD("plasma"),
    phase_B  = SMD("tumor_tissue"),
    method   = qm_method
}

let ΔG_partition_tumor = kp_qm_result.ΔG_partition

// Track D PBPK parameter
let Kp_tumor_qm : f64 = Kp_from_ΔG_partition(ΔG_partition_tumor, T = 310.0 K)

// Optionally blend with ML correction and IIV
param w_Kp_ML   : MLParamVector
rand  eta_Kp    : f64 ~ Normal(0, omega_Kp^2)

let Kp_tumor_i = Kp_tumor_qm * ML_Kp_correction(drug_features, w_Kp_ML) * exp(eta_Kp)

// Use in PBPK model
model PBPK_Tumor {
    param Kp_tumor : f64 = Kp_tumor_i
    ...
}
```

### 4.3 Kinetics (k_on, k_off) → PD Killing Rate

**Mechanistic hypothesis:**
Tumor killing rate scales with target engagement and residence time.

**Heuristic mapping:**
$$
k_{\text{kill},i} = k_{\text{kill,base}} \cdot f(k_{\text{on},i}, k_{\text{off},i}) \cdot \exp(\eta_{k_{\text{kill}},i})
$$

where:
$$
f(k_{\text{on}}, k_{\text{off}}) = \left(\frac{k_{\text{on}}}{k_{\text{on,ref}}}\right)^{\beta_{\text{on}}} \left(\frac{k_{\text{off,ref}}}{k_{\text{off}}}\right)^{\beta_{\text{off}}}
$$

**MedLang mapping:**
```medlang
fn f_QM_kill(
    k_on      : Quantity<RateConstPerConc, f64>,
    k_off     : Quantity<RateConst, f64>,
    k_on_ref  : Quantity<RateConstPerConc, f64>,
    k_off_ref : Quantity<RateConst, f64>,
    beta_on   : f64,
    beta_off  : f64
) -> f64 {
    let ratio_on  : f64 = (k_on / k_on_ref)          // dimensionless
    let ratio_off : f64 = (k_off_ref / k_off)        // dimensionless
    return pow(ratio_on, beta_on) * pow(ratio_off, beta_off)
}

fn k_kill_from_QM(
    k_on         : Quantity<RateConstPerConc, f64>,
    k_off        : Quantity<RateConst, f64>,
    k_kill_base  : Quantity<RateConst, f64>,
    eta_k_kill   : f64
) -> Quantity<RateConst, f64> {
    let f_qm : f64 = f_QM_kill(k_on, k_off, k_on_ref, k_off_ref, beta_on, beta_off)
    return k_kill_base * f_qm * exp(eta_k_kill)
}
```

**Usage in QSP:**
```medlang
// Track C
let kinetics = QM_Kinetics { ligand = drug, target = immune_checkpoint, ... }

// Track D QSP parameter
param k_kill_base : RateConst
param beta_on     : f64 = 0.5
param beta_off    : f64 = 0.5
rand  eta_k_kill  : f64 ~ Normal(0, omega_k_kill^2)

let k_kill_i = k_kill_from_QM(kinetics.k_on, kinetics.k_off, k_kill_base, eta_k_kill)

// Use in QSP model
model TumorImmuneQSP {
    param k_kill : RateConst = k_kill_i
    
    dTumor/dt = k_grow * Tumor * (1 - Tumor/T_max) - k_kill * Effector * Tumor
    ...
}
```

---

## 5. Uncertainty Quantification and Propagation

### 5.1 Sources of QM Uncertainty

1. **Method error**: DFT functional approximation, basis set incompleteness.
2. **Conformational sampling**: Multiple local minima, entropic effects.
3. **Solvation model**: Continuum vs explicit, parameterization.
4. **QM/MM boundary**: Link atom artifacts, polarization.

### 5.2 Representing Uncertainty in Track C

Each QM operator output can optionally include uncertainty:

```medlang
obs ΔG_bind       : Quantity<Energy, f64>
obs uncertainty   : Quantity<Energy, f64>  // ± kcal/mol
```

This can be interpreted as:
- **Point estimate + error bar** (frequentist).
- **Mean + standard deviation of a Gaussian** (Bayesian).

### 5.3 Propagation to Track D

In Track D, QM-derived parameters can be treated as:

**Option 1: Fixed covariates** (trust QM completely)
```medlang
let Kd_i = Kd_from_ΔG(ΔG_bind_qm, T)
```

**Option 2: Uncertain covariates with priors** (fold QM uncertainty into inference)
```medlang
// Prior on ΔG_bind centered at QM prediction with QM uncertainty
param ΔG_bind ~ Normal(
    mean = ΔG_bind_qm,
    sd   = uncertainty_qm
)

let Kd_i = Kd_from_ΔG(ΔG_bind, T)
```

**Option 3: Calibration layer** (learn systematic bias)
```medlang
param ΔG_bias ~ Normal(0, 2.0 kcal/mol)  // systematic correction

let ΔG_bind_corrected = ΔG_bind_qm + ΔG_bias
let Kd_i = Kd_from_ΔG(ΔG_bind_corrected, T)
```

In Bayesian inference, Track D will fit the calibration and IIV parameters to clinical data, effectively **learning the alignment** between QM predictions and reality.

---

## 6. Implementation and Backend Mapping

### 6.1 Backend Contract

A Track C backend (quantum chemistry engine) must:

1. **Accept typed inputs**: Molecule, QMMethod, convergence settings.
2. **Execute quantum calculation**: SCF, geometry optimization, transition state search, etc.
3. **Return typed outputs**: Energy, gradients, properties, provenance metadata.
4. **Handle failures gracefully**: convergence failure, unsupported method, etc.

### 6.2 Supported Backends (Illustrative)

| Backend | Supported Methods | Notes |
|---------|-------------------|-------|
| **Psi4** | HF, DFT, MP2, CCSD(T) | Open-source, Python API |
| **ORCA** | All major methods, QM/MM | Fast, good parallelization |
| **Gaussian** | Industry standard | Proprietary, widely used |
| **CP2K** | DFT, QM/MM, MD | Excellent for periodic systems |
| **Q-Chem** | Advanced DFT, TDDFT | Strong for excited states |
| **ML Surrogate** | Pretrained GNN/transformer | Fast approximation for screening |

### 6.3 CIR Representation of QM Operators

Track C operators are represented in CIR as:

```mlir
cir.qm_operator @QM_BindingFreeEnergy
  input_type = !cir.struct<
    ligand: !cir.molecule,
    target: !cir.molecule,
    binding_pose: !cir.pose,
    method: !cir.qm_method,
    qmmm_partition: !cir.qmmm_system
  >
  output_type = !cir.struct<
    ΔG_bind: !qty<Energy, f64>,
    ΔH_bind: !qty<Energy, f64>,
    TΔS_bind: !qty<Energy, f64>,
    uncertainty: !qty<Energy, f64>
  >
{
  cir.backend_call @orca_qmmm_fep
}
```

### 6.4 Execution Model

QM operators can be executed:

1. **On-demand**: Compute during inference (expensive, high-fidelity).
2. **Precomputed**: Run QM offline, cache results, load as covariates (fast, inflexible).
3. **Surrogate model**: Train ML model on QM data, use during inference (fast, approximate).

**Precomputed workflow:**
```medlang
// Offline: run QM for all molecules in library
for drug in drug_library {
    let binding = QM_BindingFreeEnergy { ligand = drug, target = target, ... }
    cache(drug.id, binding.ΔG_bind, binding.uncertainty)
}

// Online: load cached results
let ΔG_bind_drug = load_from_cache(drug.id, "ΔG_bind")
```

**Surrogate workflow:**
```medlang
// Train ML surrogate on QM data
train_surrogate(
    data = qm_training_set,
    model = GNN_Surrogate,
    target = "ΔG_bind"
)

// Use surrogate during inference
let ΔG_bind_approx = GNN_Surrogate.predict(drug)
```

---

## 7. Worked Example: Aspirin Binding to COX-2

### 7.1 Problem Statement

Predict the binding affinity and kinetics of aspirin (acetylsalicylic acid) to cyclooxygenase-2 (COX-2), and use these to parameterize a PK/PD model for anti-inflammatory effects.

### 7.2 Track C Workflow

**Step 1: Prepare structures**
```medlang
let aspirin = Molecule::from_smiles("CC(=O)Oc1ccccc1C(=O)O")
let cox2    = Molecule::from_pdb("5KIR", chain = "A")  // COX-2 structure
```

**Step 2: Dock aspirin into COX-2 active site**
```medlang
let docking = ClassicalDocking {
    ligand = aspirin,
    target = cox2,
    binding_site = select_residues([120, 348, 352, 516, 523]),  // active site
    forcefield = AMBER_ff14SB
}

let best_pose = docking.poses[0]  // lowest energy pose
```

**Step 3: QM/MM geometry optimization**
```medlang
let qmmm_system = QMMM_System {
    qm_region = aspirin + select_atoms(radius = 3.0 Å around aspirin),
    mm_region = rest_of(cox2),
    boundary  = LinkAtom(H),
    mm_forcefield = AMBER_ff14SB
}

let optimized = QM_MM_Optimize {
    system = qmmm_system,
    qm_method = QMMethod {
        theory     = DFT,
        functional = wB97XD,
        basis_set  = Def2_SVP,
        dispersion = D3
    }
}
```

**Step 4: Compute binding free energy**
```medlang
let binding = QM_BindingFreeEnergy {
    ligand         = aspirin,
    target         = cox2,
    binding_pose   = optimized.geometry,
    method         = QMMethod { theory = DFT, functional = wB97XD, basis_set = Def2_TZVP },
    qmmm_partition = qmmm_system,
    temperature    = 310.0 K
}

let ΔG_bind : Energy = binding.ΔG_bind  // e.g., -7.8 kcal/mol
let σ_ΔG    : Energy = binding.uncertainty  // e.g., ± 1.2 kcal/mol
```

**Step 5: Compute kinetics**
```medlang
let kinetics = QM_Kinetics {
    ligand        = aspirin,
    target        = cox2,
    bound_complex = optimized.geometry,
    method        = QMMethod { theory = DFT, functional = wB97XD, basis_set = Def2_SVP },
    temperature   = 310.0 K
}

let k_on  = kinetics.k_on   // e.g., 5.3e5 1/(M·s)
let k_off = kinetics.k_off  // e.g., 0.02 1/s
```

### 7.3 Track D Integration

**Map to PD parameters:**
```medlang
// Binding affinity → IC50
let Kd_aspirin = Kd_from_ΔG(ΔG_bind, T = 310.0 K)  // e.g., 23 nM

param alpha_IC50 : f64 = 1.5  // calibration factor
rand  eta_IC50   : f64 ~ Normal(0, 0.2^2)

let IC50_i = alpha_IC50 * Kd_aspirin * exp(eta_IC50)  // e.g., 35 nM

// Kinetics → target occupancy dynamics
param k_synth : RateConst  // COX-2 synthesis rate
param k_deg   : RateConst  // COX-2 degradation rate

model PD_COX2_Inhibition {
    state COX2_free   : Concentration
    state COX2_bound  : Concentration
    
    param IC50 : Concentration = IC50_i
    param k_on_aspirin  : RateConstPerConc = k_on
    param k_off_aspirin : RateConst = k_off
    
    input C_aspirin : Concentration  // from PK model
    
    dCOX2_free/dt = 
        k_synth 
        - k_deg * COX2_free
        - k_on_aspirin * C_aspirin * COX2_free
        + k_off_aspirin * COX2_bound
    
    dCOX2_bound/dt =
        k_on_aspirin * C_aspirin * COX2_free
        - k_off_aspirin * COX2_bound
        - k_deg * COX2_bound
    
    obs inhibition : f64 = COX2_bound / (COX2_free + COX2_bound)
}
```

**Inference:**
```medlang
inference Aspirin_PKPD {
    population_model = Aspirin_PopModel
    cohort           = clinical_trial_data
    
    mode = Bayesian
    
    priors {
        // QM-informed prior on IC50 calibration
        alpha_IC50 ~ Normal(1.0, 0.5)
        
        // Allow QM prediction to be adjusted by data
        ΔG_bias ~ Normal(0.0, 1.0 kcal/mol)
        
        // Standard PK/PD priors
        CL_pop ~ LogNormal(log(10.0 L/h), 0.5)
        ...
    }
}
```

**Result interpretation:**
- If `α_IC50` posterior is close to 1.0 → QM prediction aligns well with clinical data.
- If `ΔG_bias` is significantly non-zero → systematic error in QM method, can be corrected.
- Posterior predictive checks validate the full Track C + Track D pipeline.

---

## 8. Future Extensions

### 8.1 Excited States and Photochemistry

For photosensitizers, photoactivated drugs:
```medlang
operator QM_ExcitedState {
    input molecule : Molecule
    input method   : QMMethod  // TDDFT, CASSCF, EOM-CCSD
    
    obs excitation_energy : Quantity<Energy, f64>
    obs oscillator_strength : f64
    obs excited_state_lifetime : Quantity<Second, f64>
}
```

### 8.2 Reaction Mechanisms and Metabolic Pathways

For ADME predictions:
```medlang
operator QM_ReactionPath {
    input reactants : List<Molecule>
    input products  : List<Molecule>
    input catalyst  : Option<Molecule>  // enzyme, metal center
    
    obs transition_states : List<Molecule>
    obs ΔG_activation     : Quantity<Energy, f64>
    obs reaction_rate     : Quantity<RateConst, f64>
}
```

### 8.3 Multi-Configurational and Strong Correlation

For transition metals, open-shell systems:
```medlang
operator QM_CASSCF {
    input molecule     : Molecule
    input active_space : (n_electrons, n_orbitals)
    
    obs energies       : List<Quantity<Energy, f64>>  // multiple states
    obs wavefunctions  : List<Wavefunction>
}
```

---

## 9. Summary and Design Validation

Track C establishes:

1. **Typed quantum operators** with clear input/output contracts.
2. **Physical units from first principles** (Energy, Concentration, RateConst).
3. **Formal mappings** to Track D parameters (Kd → EC50, ΔG_partition → Kp).
4. **Uncertainty quantification** for Bayesian integration.
5. **Backend-agnostic execution** (Psi4, ORCA, Gaussian, ML surrogates).

**Design coherence:**
- Track C outputs are **covariates** or **parameter priors** for Track D.
- No "magic"—every QM → PD mapping is explicit and physically motivated.
- Calibration parameters (`α_EC50`, `β_on`, etc.) allow **data-driven correction** of QM predictions.

**Next steps:**
- Section 10: IR representation of QM operators (CIR → quantum backend calls).
- Worked Example 2: Multi-target drug with off-target binding (QM for selectivity).
- Worked Example 3: Full PBPK + QSP with quantum-derived Kp and EC50.

---

*This completes the initial draft of Track C v0.1. Future iterations will add formal semantics for wavefunction representations, detailed QM/MM boundary treatments, and integration with quantum hardware for near-term quantum algorithms (VQE, quantum phase estimation).*
