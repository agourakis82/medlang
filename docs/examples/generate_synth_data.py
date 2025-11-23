#!/usr/bin/env python3
"""
Generate synthetic dataset for one-compartment oral PK model.

True population parameters:
- CL_pop = 10.0 L/h
- V_pop = 50.0 L
- Ka_pop = 1.0 1/h
- omega_CL = 0.3
- omega_V = 0.2
- omega_Ka = 0.4
- sigma_prop = 0.15

Generates data for 20 subjects with:
- Body weights: 50-90 kg (uniform)
- Dose: 100 mg oral at time 0
- Observations: 1, 2, 4, 8, 12, 24 hours post-dose
"""

import numpy as np
import pandas as pd
from scipy.integrate import odeint

# Set random seed for reproducibility
np.random.seed(42)

# ==============================================================================
# True Population Parameters
# ==============================================================================

CL_POP = 10.0   # L/h
V_POP = 50.0    # L
KA_POP = 1.0    # 1/h

OMEGA_CL = 0.3  # SD of log(CL) random effects
OMEGA_V = 0.2   # SD of log(V) random effects
OMEGA_KA = 0.4  # SD of log(Ka) random effects

SIGMA_PROP = 0.15  # Proportional residual error SD

DOSE_AMOUNT = 100.0  # mg
OBS_TIMES = [1.0, 2.0, 4.0, 8.0, 12.0, 24.0]  # hours

N_SUBJECTS = 20

# ==============================================================================
# ODE System: One-Compartment Oral PK
# ==============================================================================

def one_comp_oral_ode(y, t, Ka, CL, V):
    """
    ODE system for one-compartment oral PK.

    States:
        y[0] = A_gut (mg)
        y[1] = A_central (mg)

    Parameters:
        Ka : absorption rate (1/h)
        CL : clearance (L/h)
        V  : volume (L)

    Returns:
        dydt = [dA_gut/dt, dA_central/dt]
    """
    A_gut, A_central = y

    dA_gut = -Ka * A_gut
    dA_central = Ka * A_gut - (CL / V) * A_central

    return [dA_gut, dA_central]

# ==============================================================================
# Solve ODE for Individual
# ==============================================================================

def solve_pk(Ka, CL, V, dose, times):
    """
    Solve PK ODEs for an individual.

    Returns concentrations (mg/L) at observation times.
    """
    # Initial conditions: dose goes into gut compartment
    y0 = [dose, 0.0]

    # Solve ODE system
    # Note: add time 0 to get initial state
    t_solve = np.concatenate([[0.0], times])
    solution = odeint(one_comp_oral_ode, y0, t_solve, args=(Ka, CL, V))

    # Extract A_central at observation times
    A_central = solution[1:, 1]  # Skip t=0

    # Compute concentrations
    C = A_central / V

    return C

# ==============================================================================
# Generate Dataset
# ==============================================================================

def generate_data():
    """Generate complete synthetic dataset."""

    rows = []

    for subject_id in range(1, N_SUBJECTS + 1):
        # Sample body weight (uniform 50-90 kg)
        WT = np.random.uniform(50.0, 90.0)

        # Sample random effects (IIV)
        eta_CL = np.random.normal(0.0, OMEGA_CL)
        eta_V = np.random.normal(0.0, OMEGA_V)
        eta_Ka = np.random.normal(0.0, OMEGA_KA)

        # Compute individual parameters with allometric scaling
        w_norm = WT / 70.0
        CL_i = CL_POP * (w_norm ** 0.75) * np.exp(eta_CL)
        V_i = V_POP * w_norm * np.exp(eta_V)
        Ka_i = KA_POP * np.exp(eta_Ka)

        # Solve PK model
        C_pred = solve_pk(Ka_i, CL_i, V_i, DOSE_AMOUNT, OBS_TIMES)

        # Add dose row (EVID=1)
        rows.append({
            'ID': subject_id,
            'TIME': 0.0,
            'DV': np.nan,
            'WT': WT,
            'EVID': 1,
            'AMT': DOSE_AMOUNT,
        })

        # Add observation rows (EVID=0) with residual error
        for time, c_pred in zip(OBS_TIMES, C_pred):
            # Proportional error: DV = C_pred * (1 + epsilon)
            epsilon = np.random.normal(0.0, SIGMA_PROP)
            DV = c_pred * (1.0 + epsilon)

            # Ensure non-negative (truncate at 0)
            DV = max(0.0, DV)

            rows.append({
                'ID': subject_id,
                'TIME': time,
                'DV': DV,
                'WT': WT,
                'EVID': 0,
                'AMT': np.nan,
            })

    # Convert to DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns
    df = df[['ID', 'TIME', 'DV', 'WT', 'EVID', 'AMT']]

    return df

# ==============================================================================
# Main
# ==============================================================================

if __name__ == '__main__':
    # Generate data
    df = generate_data()

    # Save to CSV
    output_path = 'onecomp_synth.csv'
    df.to_csv(output_path, index=False, float_format='%.6f')

    print(f"Generated synthetic dataset: {output_path}")
    print(f"  Subjects: {N_SUBJECTS}")
    print(f"  Total rows: {len(df)}")
    print(f"  Dose rows (EVID=1): {(df['EVID'] == 1).sum()}")
    print(f"  Observation rows (EVID=0): {(df['EVID'] == 0).sum()}")
    print()
    print("True population parameters:")
    print(f"  CL_pop = {CL_POP} L/h")
    print(f"  V_pop = {V_POP} L")
    print(f"  Ka_pop = {KA_POP} 1/h")
    print(f"  omega_CL = {OMEGA_CL}")
    print(f"  omega_V = {OMEGA_V}")
    print(f"  omega_Ka = {OMEGA_KA}")
    print(f"  sigma_prop = {SIGMA_PROP}")
    print()
    print("First few rows:")
    print(df.head(10))
