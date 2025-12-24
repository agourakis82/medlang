# MedLang End-to-End Workflow Guide

This guide demonstrates the complete workflow from writing a MedLang model to running MCMC inference.

## Prerequisites

- MedLang compiler (`mlc`) built and installed
- For Stan backend: [cmdstan](https://mc-stan.org/users/interfaces/cmdstan) installed (optional for compilation only)
- For Julia backend: Julia with DifferentialEquations.jl and Turing.jl (optional)

## Complete Workflow Example

### Step 1: Write Your Model

Create a MedLang model file (e.g., `mymodel.medlang`):

```medlang
model OneCompOral {
    state A_gut     : DoseMass
    state A_central : DoseMass
    
    param Ka : RateConst
    param CL : Clearance
    param V  : Volume
    
    dA_gut/dt     = -Ka * A_gut
    dA_central/dt =  Ka * A_gut - (CL / V) * A_central
    
    obs C_plasma : ConcMass = A_central / V
}

population OneCompOralPop {
    model OneCompOral
    
    param CL_pop : Clearance
    param V_pop  : Volume
    param Ka_pop : RateConst
    
    param omega_CL : f64
    param omega_V  : f64
    param omega_Ka : f64
    
    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    rand eta_V  : f64 ~ Normal(0.0, omega_V)
    rand eta_Ka : f64 ~ Normal(0.0, omega_Ka)
    
    input WT : Mass
    
    bind_params(patient) {
        let w = patient.WT / 70.0_kg
        model.CL = CL_pop * pow(w, 0.75) * exp(eta_CL)
        model.V  = V_pop * w * exp(eta_V)
        model.Ka = Ka_pop * exp(eta_Ka)
    }
}

measure ProportionalError {
    param sigma_prop : f64
    
    observe(expected: ConcMass, observed: ConcMass) {
        let sigma = sigma_prop * expected
        observed ~ Normal(expected, sigma)
    }
}
```

### Step 2: Compile to Backend

**For Stan:**
```bash
mlc compile mymodel.medlang -o mymodel.stan -v
```

**For Julia:**
```bash
mlc compile mymodel.medlang --backend julia -o mymodel.jl -v
```

**Output:**
```
Reading source: mymodel.medlang
Stage 1: Tokenization...
  ✓ 288 tokens generated
Stage 2: Parsing...
  ✓ AST constructed with 5 declarations
Stage 3: Type checking and lowering to IR...
  ✓ IR generated
    - 2 states
    - 9 parameters
    - 2 ODEs
    - 1 observables
Stage 4: Code generation (backend: stan)...
  ✓ 107 lines of stan code generated
✓ Compilation successful: mymodel.medlang → mymodel.stan
```

### Step 3: Generate or Prepare Data

**Option A: Generate Synthetic Data**
```bash
mlc generate-data -n 20 -o data.csv --verbose
```

**Output:**
```
Generating synthetic dataset...
  Subjects: 20
  Dose: 100 mg
  Seed: 42
Population parameters:
  CL_pop = 10 L/h, ω_CL = 0.3
  V_pop  = 50 L,   ω_V  = 0.2
  Ka_pop = 1 1/h, ω_Ka = 0.4
  σ_prop = 0.15
Generated 160 observations
✓ Dataset generated: data.csv (160 rows)
```

**Option B: Use Real Data**

Prepare your data in NONMEM-style CSV format:
```csv
ID,TIME,AMT,DV,EVID,WT
1,0.0,100,.,1,70
1,0.5,.,2.5,0,70
1,1.0,.,3.2,0,70
...
```

### Step 4: Convert Data to Stan Format

```bash
mlc convert-data data.csv -o data.json -v
```

**Output:**
```
Loading CSV data: data.csv

Dataset Summary:
- Subjects: 20
- Total records: 160
- Observations: 140
- Dose events: 20
- Covariates: WT
- Time range: 0.00 - 24.00

Converting to Stan JSON format...
✓ Data converted: data.csv → data.json
```

### Step 5: Run MCMC Sampling (Stan)

```bash
mlc run mymodel.stan --data data.json --output results/ --verbose
```

**Options:**
- `--chains 4` - Number of MCMC chains (default: 4)
- `--warmup 1000` - Warmup iterations (default: 1000)
- `--samples 1000` - Sampling iterations (default: 1000)
- `--seed 12345` - Random seed for reproducibility
- `--adapt-delta 0.8` - Target acceptance rate (increase for divergences)
- `--max-treedepth 10` - Maximum tree depth

**Output:**
```
MedLang Stan Runner
Model: mymodel.stan
Data: data.json
Output: results/

Detecting cmdstan installation...
✓ Found cmdstan at: /Users/username/.cmdstan/cmdstan-2.33.1

Compiling Stan model: mymodel
✓ Compilation successful
✓ Model compiled to: mymodel

MCMC Configuration:
  Chains: 4
  Warmup: 1000
  Samples: 1000
  Adapt delta: 0.8
  Max tree depth: 10

Running MCMC sampling with 4 chains...
  Chain 1/4...
  Chain 2/4...
  Chain 3/4...
  Chain 4/4...
✓ MCMC sampling complete

================================================================================
MCMC Diagnostics Summary
================================================================================

Output directory: results/
Number of chains: 4

Parameter            Mean         SD         5%        50%        95%       Rhat        ESS
----------------------------------------------------------------------------------------------------
CL_pop              9.876      0.234      9.512      9.881     10.245      1.001       3850
Ka_pop              0.987      0.156      0.745      0.982      1.234      1.000       3920
V_pop              49.234      1.456     47.123     49.201     51.345      1.002       3780
omega_CL            0.312      0.045      0.245      0.309      0.382      1.001       2890
omega_Ka            0.423      0.078      0.298      0.418      0.551      1.000       3120
omega_V             0.198      0.034      0.145      0.196      0.254      1.001       3450
sigma_prop          0.147      0.012      0.128      0.147      0.166      1.000       4000

================================================================================
✓ All parameters converged successfully
================================================================================

✓ Results saved to: results/
  Chain files: ["output_1.csv", "output_2.csv", "output_3.csv", "output_4.csv"]
```

### Step 6: Analyze Results

The MCMC output includes:
- **Rhat**: Should be < 1.01 for convergence
- **ESS**: Effective sample size (should be > 400)
- **Quantiles**: 5%, 50% (median), 95%
- **Chain CSV files**: Full posterior samples for further analysis

## Quick Reference

### All Commands

```bash
# Compile
mlc compile <input.medlang> [--backend stan|julia] [-o output]

# Check syntax
mlc check <input.medlang>

# Generate test data
mlc generate-data -n <subjects> -o <output.csv>

# Convert data format
mlc convert-data <input.csv> -o <output.json>

# Run MCMC (Stan only)
mlc run <model.stan> --data <data.json> [options]
```

### Typical Workflow

```bash
# 1. Compile model
mlc compile model.medlang -v

# 2. Generate or convert data
mlc generate-data -n 20 -o data.csv
mlc convert-data data.csv -o data.json

# 3. Run inference
mlc run model.stan --data data.json --output results/ -v

# 4. Examine results
cat results/output_1.csv
```

## Troubleshooting

### cmdstan Not Found

If you get "cmdstan not found", either:

1. Install cmdstan: https://mc-stan.org/users/interfaces/cmdstan
2. Set environment variable: `export CMDSTAN=/path/to/cmdstan`

### Divergent Transitions

If you see divergence warnings:
- Increase `--adapt-delta` (e.g., 0.95, 0.99)
- Increase `--warmup` iterations
- Check model specification

### Low ESS

If effective sample size is low:
- Increase `--samples`
- Check for high correlation between parameters
- Consider reparameterization

### High Rhat

If Rhat > 1.01:
- Increase `--warmup` and `--samples`
- Run more `--chains`
- Check for multimodality or identifiability issues

## Next Steps

- Explore Julia backend for faster prototyping
- Add visualization of results
- Implement posterior predictive checks
- Compare models with different error structures
