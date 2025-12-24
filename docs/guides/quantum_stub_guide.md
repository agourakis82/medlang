# Quantum Stub Integration Guide (Week 6 - Track C)

## Overview

The **Quantum Stub** feature enables integration of quantum-mechanically derived molecular properties into PK-PD models. This bridges **Track C** (quantum pharmacology) with **Track D** (systems QSP), allowing drug binding affinity (Kd) and tissue partition coefficients (Kp) to inform pharmacodynamic parameters like EC50.

## What is a Quantum Stub?

A quantum stub is a JSON file containing precomputed molecular properties from quantum mechanics calculations:

- **Kd** (dissociation constant): Drug-target binding affinity
- **ΔG_bind**: Binding free energy
- **ΔG_part**: Partition free energy between compartments (e.g., plasma→tumor)
- **T_ref**: Reference temperature for calculations

These values are derived from quantum simulations (ONIOM, DFT, etc.) but stored as simple constants for fast integration.

## JSON Schema (v0.1)

```json
{
  "drug_id": "LIG001",
  "target_id": "EGFR",
  "Kd_M": 2.5e-9,
  "dG_bind_kcal_per_mol": -11.7,
  "dG_part_plasma_tumor_kcal_per_mol": -0.8,
  "T_ref_K": 310.0
}
```

### Required Fields

- `drug_id` (string): Unique drug identifier
- `target_id` (string): Biological target (receptor, enzyme, etc.)
- `Kd_M` (number): Binding dissociation constant in **molar** units (must be > 0)

### Optional Fields

- `dG_bind_kcal_per_mol` (number): Binding free energy in kcal/mol
- `dG_part_plasma_tumor_kcal_per_mol` (number): Partition free energy in kcal/mol
- `T_ref_K` (number): Temperature in Kelvin (default: 310 K = 37°C)

## How It Works

### 1. Stub Loading

The compiler loads the QM stub JSON file when you provide `--qm-stub`:

```bash
mlc compile model.medlang --qm-stub data/drug_qm.json
```

### 2. IR Integration

The lowering pass (`lower_program_with_qm`) extracts quantum properties and adds them as **external constants** to the IR:

- **Kd_QM**: The binding constant from the stub
- **Kp_tumor_QM**: Computed from ΔG_part using: `Kp = exp(-ΔG / (R*T))`

### 3. Stan/Julia Codegen

External constants appear in the `data` block of generated Stan code:

```stan
data {
  // ... other data ...
  
  // External quantum constants
  real<lower=0> Kd_QM;  // from qm_stub:LIG001:EGFR
  real<lower=0> Kp_tumor_QM;  // from qm_stub:LIG001:dG_part
}
```

### 4. Parameter Mapping

In your MedLang model, you can reference these constants when defining individual parameters. For example, to make EC50 quantum-informed:

**Traditional (without QM):**
```medlang
// EC50 is a free population parameter
param EC50_pop : ConcMass
bind_params(patient) {
    model.EC50 = EC50_pop * exp(eta_EC50)
}
```

**Quantum-informed (with QM stub):**
```medlang
// EC50 is derived from quantum Kd with a scaling factor
param alpha_EC50 : f64  // In vivo scaling factor
bind_params(patient) {
    model.EC50 = alpha_EC50 * Kd_QM * exp(eta_EC50)
}
```

Here:
- `Kd_QM` is provided by the stub (fixed constant)
- `alpha_EC50` is estimated from data (accounts for in vivo vs in vitro differences)
- `eta_EC50` is individual variability

## Practical Example: PK-QSP Oncology Model

### 1. Create QM Stub

`data/gefitinib_egfr_qm.json`:
```json
{
  "drug_id": "GEFITINIB",
  "target_id": "EGFR",
  "Kd_M": 3.7e-9,
  "dG_bind_kcal_per_mol": -11.5,
  "dG_part_plasma_tumor_kcal_per_mol": -0.6,
  "T_ref_K": 310.0
}
```

### 2. MedLang Model

`examples/egfr_inhibitor_qsp.medlang`:
```medlang
model EGFR_Inhibitor_QSP {
    // PK compartments
    state A_central : DoseMass
    
    // QSP tumor
    state Tumour : TumourVolume
    
    // PK parameters
    param CL : Clearance
    param V : Volume
    
    // QSP parameters (EC50 will be quantum-informed)
    param EC50 : ConcMass
    param Emax : f64
    param k_grow : RateConst
    param T_max : TumourVolume
    
    // ODEs
    dA_central/dt = -(CL / V) * A_central
    
    dTumour/dt = k_grow * Tumour * (1.0 - Tumour / T_max)
                 - (Emax * (A_central / V) / (EC50 + (A_central / V))) * Tumour
    
    // Observables
    obs C_plasma : ConcMass = A_central / V
    obs TumourVol : TumourVolume = Tumour
}

population EGFR_Inhibitor_Pop {
    model EGFR_Inhibitor_QSP
    
    // PK population parameters
    param CL_pop : Clearance
    param V_pop : Volume
    
    // QSP: alpha_EC50 instead of EC50_pop
    param alpha_EC50 : f64
    param Emax_pop : f64
    param k_grow_pop : RateConst
    param T_max_pop : TumourVolume
    
    // IIV
    param omega_EC50 : f64
    rand eta_EC50 : f64 ~ Normal(0.0, omega_EC50)
    
    bind_params(patient) {
        model.CL = CL_pop
        model.V = V_pop
        
        // Quantum-informed EC50
        model.EC50 = alpha_EC50 * Kd_QM * exp(eta_EC50)
        
        model.Emax = Emax_pop
        model.k_grow = k_grow_pop
        model.T_max = T_max_pop
    }
    
    use_measure TumourError for model.TumourVol
}

measure TumourError {
    pred : TumourVolume
    obs : TumourVolume
    param sigma_tum : f64
    log_likelihood = Normal_logpdf(
        x = log(obs / pred),
        mu = 0.0,
        sd = sigma_tum
    )
}
```

### 3. Compile with QM Stub

```bash
mlc compile examples/egfr_inhibitor_qsp.medlang \
    --qm-stub data/gefitinib_egfr_qm.json \
    -v
```

**Output:**
```
Loading quantum stub: data/gefitinib_egfr_qm.json
  ✓ QM stub loaded: GEFITINIB targeting EGFR
    Kd = 3.70e-9 M
    Kp_tumor = 2.718
Reading source: examples/egfr_inhibitor_qsp.medlang
...
  ✓ IR generated
    - 2 states
    - 10 parameters
    - 2 ODEs
    - 2 observables
    - 2 external QM constants
      • Kd_QM = 3.700e-9
      • Kp_tumor_QM = 2.718e0
...
✓ Compilation successful
```

### 4. Stan Data File

When running MCMC, you must provide the QM constants in your Stan JSON data file:

`data/egfr_inhibitor_data.json`:
```json
{
  "N": 20,
  "n_obs": 100,
  "subject_id": [...],
  "time": [...],
  "observation": [...],
  
  "Kd_QM": 3.7e-9,
  "Kp_tumor_QM": 2.718,
  
  "dose_amount": 250.0,
  "rtol": 1e-6,
  "atol": 1e-8,
  "max_steps": 10000
}
```

**Note:** The CLI doesn't yet automatically inject QM values into Stan data files. You must add them manually or via a script.

## Scientific Interpretation

### EC50 Scaling Factor (alpha_EC50)

The estimated `alpha_EC50` tells you how in vivo potency relates to in vitro binding:

- **alpha_EC50 ≈ 1.0**: In vivo EC50 matches quantum Kd
- **alpha_EC50 > 1.0**: In vivo EC50 higher than Kd (common)
  - Causes: protein binding, active efflux, poor penetration
- **alpha_EC50 < 1.0**: In vivo EC50 lower than Kd (rare)
  - Causes: active metabolites, allosteric effects

### Partition Coefficient (Kp_tumor)

The computed Kp_tumor indicates drug accumulation in tumor:

- **Kp > 1**: Drug concentrates in tumor (favorable for efficacy)
- **Kp = 1**: Equal concentration in plasma and tumor
- **Kp < 1**: Drug excluded from tumor (poor penetration)

For our gefitinib example, Kp_tumor = 2.72 means tumor concentration is ~2.7× plasma concentration.

## Advantages of QM-Informed Modeling

1. **Mechanistic Priors**: Use quantum-derived Kd as an informed prior for EC50
2. **Reduced Parameters**: Estimate alpha_EC50 instead of EC50_pop directly
3. **Cross-Drug Comparison**: Compare alpha values across compounds to identify formulation/delivery issues
4. **Predictive Power**: For new drugs with QM data but no clinical data yet
5. **Multi-Scale Integration**: Seamless connection from atoms to organs

## Limitations

- **Stub is static**: No uncertainty propagation from QM calculations (Week 7+ feature)
- **Single target**: Current implementation assumes one drug-target pair
- **Manual data injection**: QM values must be manually added to Stan data files
- **No DSL syntax**: QM constants referenced implicitly, not declared in MedLang (Week 7+ feature)

## File Locations

- **QM stub schema**: `data/qm_stub_schema_v0.1.json`
- **Example stub**: `data/lig001_egfr_qm.json`
- **Example model**: `docs/examples/pk_qsp_inline.medlang`
- **Module source**: `compiler/src/qm_stub.rs`
- **Integration tests**: `compiler/tests/qm_integration_test.rs`

## CLI Reference

### Compile with QM Stub

```bash
mlc compile <MODEL.medlang> --qm-stub <STUB.json> [OPTIONS]
```

**Arguments:**
- `--qm-stub <PATH>`: Path to quantum stub JSON file
- `-v, --verbose`: Show QM stub loading details
- `--emit-ir <PATH>`: Emit IR with external constants to JSON

**Example:**
```bash
mlc compile model.medlang \
    --qm-stub data/drug_qm.json \
    --emit-ir model_ir.json \
    -v
```

## Future Enhancements (Week 7+)

- [ ] DSL syntax: `quantum { Kd from "stub.json" }`
- [ ] Automatic injection of QM values into Stan data files
- [ ] Uncertainty propagation: treat Kd as uncertain with priors
- [ ] Multi-drug/multi-target stubs
- [ ] PBPK integration: use Kp_tumor in tissue distribution models
- [ ] Real QM calculations: replace stubs with live ONIOM calls

## References

- MedLang QM Extension Spec: `docs/medlang_qm_extension_spec_v0.1.md`
- Track C Overview: (conceptual design documents)
- Week 6 Implementation Plan: (detailed task breakdown)

---

**Status**: Week 6 Complete ✓  
**Next**: Week 7 - PBPK skeleton or DSL integration for QM
