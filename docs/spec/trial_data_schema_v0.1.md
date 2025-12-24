# MedLang Trial Data Schema v0.1

**Purpose**: This document specifies the canonical format for observed clinical trial data to be analyzed with MedLang's Protocol DSL and endpoint evaluation engine.

---

## Overview

The MedLang trial data schema enables ingestion of real-world oncology trial observations for:
- Endpoint evaluation (ORR, PFS) on observed data
- Comparison of virtual trial predictions vs. actual outcomes
- Retrospective analysis using the same protocol definitions as prospective simulations

**Design principles**:
1. **Minimal**: Only essential columns required for endpoint evaluation
2. **Flexible**: Supports both long-format (one row per observation) and wide-format options
3. **Clinically aligned**: Matches standard oncology data structures (CDISC ADaM-like)
4. **Observable-agnostic**: Can extend beyond tumor volume to other biomarkers

---

## Required Columns

All trial data files **MUST** include these columns:

| Column   | Type           | Description                                                                 | Example Values        |
|----------|----------------|-----------------------------------------------------------------------------|-----------------------|
| `ID`     | string or int  | Unique subject identifier within the trial                                  | `"S001"`, `101`       |
| `ARM`    | string         | Treatment arm identifier (must match protocol arm names)                    | `"ArmA"`, `"Control"` |
| `TIME`   | numeric (f64)  | Time in **days** relative to baseline (day 0 = randomization/first dose)   | `0.0`, `28.0`, `56.0` |
| `TUMVOL` | numeric (f64)  | Tumor volume or sum of longest diameters (SLD) in mm or mm³                | `100.5`, `75.3`       |

### Column Details

**`ID` (Subject Identifier)**:
- Must be unique within the trial dataset
- Can be alphanumeric string or integer
- Used to group observations into subject trajectories
- Example: `"P001"`, `"P002"` or `101`, `102`

**`ARM` (Treatment Arm)**:
- Must exactly match arm names defined in the protocol's `arms { ... }` block
- Case-sensitive
- Used to assign subjects to protocol-defined treatment arms
- Example: If protocol defines `ArmA { label = "Low Dose"; dose = 100.0 }`, then `ARM` values should be `"ArmA"`

**`TIME` (Observation Time)**:
- **Units**: Days since baseline (day 0)
- Baseline observation should have `TIME = 0.0`
- All subsequent measurements are positive offsets
- Used for windowing endpoint evaluations and survival analysis
- Example: Baseline (0), Week 4 (28), Week 8 (56), Week 12 (84)

**`TUMVOL` (Tumor Volume)**:
- Primary observable for oncology endpoints (ORR, PFS)
- **Units**: Flexible (mm³, mm SLD, percentage of baseline)
  - Must be consistent within a trial
  - Absolute values required for shrinkage/progression calculations
- Corresponds to the `TumourVol` observable in MedLang population models
- Used for RECIST-like response criteria
- Example: Baseline 100 mm³ → Week 8: 65 mm³ (35% shrinkage)

---

## Optional Columns

These columns enhance analysis but are not strictly required:

| Column     | Type          | Description                                                      | Example Values   |
|------------|---------------|------------------------------------------------------------------|------------------|
| `CENS`     | int (0 or 1)  | Censoring indicator for time-to-event endpoints (0=event, 1=censored) | `0`, `1`         |
| `AGE`      | numeric (int) | Patient age in years at baseline                                 | `45`, `62`       |
| `ECOG`     | int (0–4)     | ECOG performance status                                          | `0`, `1`, `2`    |
| `SEX`      | string        | Biological sex                                                   | `"M"`, `"F"`     |
| `OBS_TYPE` | string        | Observable type (if multiple observables tracked)                | `"TUMOUR"`, `"PK"` |

### Optional Column Details

**`CENS` (Censoring Indicator)**:
- Used for explicit censoring in survival analysis
- `0` = Event observed (e.g., progression occurred)
- `1` = Censored (e.g., patient withdrew, study ended before event)
- If absent: MedLang infers censoring automatically
  - Last observation within window = censored if no event detected
  - Events detected via progression criteria in endpoint spec

**`AGE`, `ECOG`, `SEX` (Covariates)**:
- Used for inclusion/exclusion filtering
- Enables covariate-based subgroup analysis (future)
- Maps to `SubjectCovariates` structure in `endpoints.rs`

**`OBS_TYPE` (Observable Type)**:
- Distinguishes different measurement types in multi-observable trials
- Example: `"TUMOUR"` for tumor assessments, `"PK"` for drug concentrations
- Default behavior: If absent, all rows assumed to be tumor volume observations

---

## Data Format: Long vs. Wide

### Long Format (Recommended)

**One row per observation** (subject-time pair):

```csv
ID,ARM,TIME,TUMVOL,AGE,ECOG
S001,ArmA,0.0,120.5,55,1
S001,ArmA,28.0,95.3,55,1
S001,ArmA,56.0,78.1,55,1
S002,ArmA,0.0,105.0,62,0
S002,ArmA,28.0,88.0,62,0
S003,ArmB,0.0,135.2,48,1
S003,ArmB,28.0,70.5,48,1
S003,ArmB,56.0,55.0,48,1
```

**Advantages**:
- Natural representation for time-series data
- Easy to handle irregular visit schedules
- Directly maps to `SubjectTrajectory` structure
- Standard in CDISC ADaM

### Wide Format (Alternative)

**One row per subject**, with time-indexed columns:

```csv
ID,ARM,AGE,ECOG,TUMVOL_0,TUMVOL_28,TUMVOL_56
S001,ArmA,55,1,120.5,95.3,78.1
S002,ArmA,62,0,105.0,88.0,NA
S003,ArmB,48,1,135.2,70.5,55.0
```

**Conversion note**: MedLang loaders will support both formats, but internally convert wide → long for processing.

---

## File Formats

### CSV (Primary Format)

**Recommended encoding**: UTF-8  
**Delimiter**: Comma (`,`)  
**Header**: Required (first row contains column names)  
**Missing values**: `NA`, empty cell, or `-999.0` (configurable)

Example file: `trial_data.csv`

```csv
ID,ARM,TIME,TUMVOL,AGE,ECOG,SEX
P001,ArmA,0.0,100.0,55,1,M
P001,ArmA,28.0,75.0,55,1,M
P001,ArmA,56.0,60.0,55,1,M
P002,ArmA,0.0,120.0,62,0,F
P002,ArmA,28.0,110.0,62,0,F
P003,ArmB,0.0,95.0,48,1,M
P003,ArmB,28.0,50.0,48,1,M
P003,ArmB,56.0,45.0,48,1,M
```

### JSON (Alternative Format)

**Structure**: Array of subject records, each containing trajectory data

```json
{
  "trial_id": "TRIAL-2024-001",
  "subjects": [
    {
      "id": "P001",
      "arm": "ArmA",
      "age": 55,
      "ecog": 1,
      "sex": "M",
      "observations": [
        {"time": 0.0, "tumvol": 100.0},
        {"time": 28.0, "tumvol": 75.0},
        {"time": 56.0, "tumvol": 60.0}
      ]
    },
    {
      "id": "P002",
      "arm": "ArmA",
      "age": 62,
      "ecog": 0,
      "sex": "F",
      "observations": [
        {"time": 0.0, "tumvol": 120.0},
        {"time": 28.0, "tumvol": 110.0}
      ]
    }
  ]
}
```

---

## Data Validation Rules

When loading trial data, MedLang will validate:

1. **Required columns present**: `ID`, `ARM`, `TIME`, `TUMVOL` must exist
2. **No missing required values**: These columns cannot have `NA` or empty cells
3. **ARM matching**: All `ARM` values must match protocol arm names
4. **Time ordering**: Within each subject, `TIME` must be non-decreasing
5. **Baseline existence**: Each subject should have at least one observation at `TIME = 0.0` (tolerance: ±1 day)
6. **Positive tumor volumes**: `TUMVOL > 0.0` (negative values rejected)
7. **Type consistency**: Numeric columns must parse as valid floats/ints

**Validation failures** → Loader returns error with descriptive message

---

## Mapping to MedLang Structures

### TrialDataset → SubjectTrajectory

The trial data loader converts raw CSV/JSON into the internal representation:

```rust
// From trial data schema
pub struct TrialRow {
    pub id: String,
    pub arm: String,
    pub time: f64,
    pub tumvol: f64,
    pub age: Option<u8>,
    pub ecog: Option<u8>,
    pub sex: Option<String>,
    pub cens: Option<bool>,
}

pub struct TrialDataset {
    pub rows: Vec<TrialRow>,
}

// To endpoint evaluation format (existing in endpoints.rs)
pub struct SubjectTrajectory {
    pub id: usize,              // Mapped from string ID
    pub times_days: Vec<f64>,   // Aggregated TIME values
    pub tumour_vol: Vec<f64>,   // Aggregated TUMVOL values
    pub baseline_tumour: f64,   // First TUMVOL value
    pub covariates: SubjectCovariates,
}

pub struct SubjectCovariates {
    pub age: u8,
    pub ecog: u8,
}
```

### Conversion Logic

1. **Group by ID**: Collect all rows for each subject
2. **Sort by TIME**: Ensure chronological ordering
3. **Extract baseline**: First observation (`TIME ≈ 0.0`) → `baseline_tumour`
4. **Build trajectory**: 
   - `times_days` = all `TIME` values
   - `tumour_vol` = all `TUMVOL` values
5. **Extract covariates**: Use values from first row (assumed constant)
6. **Assign numeric ID**: Map string IDs to integer indices for internal use

---

## Example: Complete Trial Data File

**File**: `example_trial.csv`

```csv
ID,ARM,TIME,TUMVOL,AGE,ECOG,SEX
S001,ArmA,0.0,100.0,55,1,M
S001,ArmA,28.0,75.0,55,1,M
S001,ArmA,56.0,60.0,55,1,M
S001,ArmA,84.0,65.0,55,1,M
S002,ArmA,0.0,120.0,62,0,F
S002,ArmA,28.0,110.0,62,0,F
S002,ArmA,56.0,130.0,62,0,F
S003,ArmA,0.0,95.0,48,1,M
S003,ArmA,28.0,85.0,48,1,M
S004,ArmB,0.0,110.0,59,1,F
S004,ArmB,28.0,60.0,59,1,F
S004,ArmB,56.0,45.0,59,1,F
S004,ArmB,84.0,40.0,59,1,F
S005,ArmB,0.0,105.0,51,0,M
S005,ArmB,28.0,80.0,51,0,M
S005,ArmB,56.0,70.0,51,0,M
S006,ArmB,0.0,130.0,67,2,F
S006,ArmB,28.0,120.0,67,2,F
```

**Analysis scenario**:
- 6 subjects total (3 per arm)
- ArmA: Mix of responders and progressors
- ArmB: Mostly responders
- Protocol endpoint: ORR (≥30% shrinkage) and PFS (20% progression from nadir)
- Evaluation window: [0.0, 84.0] days

**Expected outcomes**:
- **S001** (ArmA): ORR ✓ (40% shrinkage), PFS = censored (no progression)
- **S002** (ArmA): ORR ✗ (8% shrinkage), PFS = 56 days (progression)
- **S003** (ArmA): ORR ✗ (10% shrinkage), PFS = censored
- **S004** (ArmB): ORR ✓ (64% shrinkage), PFS = censored
- **S005** (ArmB): ORR ✓ (33% shrinkage), PFS = censored
- **S006** (ArmB): ORR ✗ (8% shrinkage), PFS = censored

---

## Usage with MedLang CLI

Once implemented, the trial data ingestion will be used as:

```bash
# Analyze observed trial data against a protocol
mlc analyze-trial --protocol protocol.medlang --data trial_data.csv --output results.json

# Output: Per-arm ORR, PFS, Kaplan-Meier curves for observed data
```

**Output format**: Same JSON structure as virtual trial results (from `simulate_protocol`), enabling direct comparison:

```json
{
  "protocol": "TestProtocol",
  "data_source": "observed",
  "trial_file": "trial_data.csv",
  "arms": [
    {
      "arm_name": "ArmA",
      "label": "Low Dose",
      "n_subjects": 3,
      "n_included": 3,
      "ORR": {
        "n_responders": 1,
        "rate": 0.333,
        "ci_95": [0.01, 0.91]
      },
      "PFS": {
        "n_events": 1,
        "median_days": null,
        "km_times": [0.0, 28.0, 56.0, 84.0],
        "km_surv": [1.0, 1.0, 0.67, 0.67]
      }
    }
  ]
}
```

---

## Future Extensions (v0.2+)

Potential enhancements to the schema:

1. **Multi-observable support**: Explicit `OBS_TYPE` column for PK, PD, biomarkers
2. **Dose modifications**: `DOSE_ACTUAL` column for per-subject dose adjustments
3. **Adverse events**: Link to safety data (AE grade, SAE flags)
4. **Longitudinal covariates**: Time-varying ECOG, weight, lab values
5. **Metadata header**: Trial ID, study phase, protocol version in JSON wrapper
6. **Wide format loader**: Native support without long-format conversion

---

## References

- **CDISC ADaM**: Analysis Data Model standards for clinical trials
- **RECIST 1.1**: Response Evaluation Criteria In Solid Tumors
- **MedLang endpoints module**: `compiler/src/endpoints.rs` (implementation reference)
- **Protocol DSL spec**: `docs/week10_time_to_event_endpoints.md`

---

## Version History

- **v0.1** (2025-11-23): Initial schema definition
  - Required columns: ID, ARM, TIME, TUMVOL
  - Optional columns: CENS, AGE, ECOG, SEX, OBS_TYPE
  - CSV and JSON format support
  - Long-format primary, wide-format alternative
