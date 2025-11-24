# Standards & Regulatory Interoperability

MedLang enables export to three major regulatory and clinical standards for healthcare interoperability and trial data management:

## Overview

| Standard | Purpose | Use Case | MedLang Component |
|----------|---------|----------|-------------------|
| **FHIR R4** | Healthcare interoperability | Trial protocol & data sharing | `interop::fhir` |
| **CQL** | Clinical endpoint definitions | FHIR-compliant query language | `interop::cql` |
| **CDISC** | Regulatory data standards | FDA/EMA submissions | `interop::cdisc` |

---

## 1. FHIR R4 Export (`export-fhir`)

**FHIR = Fast Healthcare Interoperability Resources**

FHIR is an international standard for exchanging healthcare data using modern web technologies (REST, JSON, XML).

### MedLang → FHIR Mapping

#### ResearchStudy Resource

Converts the overall protocol to a FHIR `ResearchStudy` resource.

| MedLang Element | FHIR ResearchStudy Field | Notes |
|-----------------|-------------------------|-------|
| `protocol.name` | `ResearchStudy.id` | Unique study identifier |
| `protocol.arms` | `ResearchStudy.arm[]` | List of treatment arms |
| `protocol.endpoints` | `ResearchStudy.objective[]` | Study objectives |
| Protocol description | `ResearchStudy.description` | Study summary |

**Example:**
```json
{
  "resourceType": "ResearchStudy",
  "id": "TRIAL001",
  "status": "active",
  "title": "TRIAL001",
  "arm": [
    {
      "name": "ArmA",
      "description": "Control (0mg)"
    },
    {
      "name": "ArmB",
      "description": "Treatment (100mg)"
    }
  ],
  "objective": [
    {
      "name": "ORR",
      "type": {
        "text": "ORR (Objective Response Rate)"
      }
    }
  ]
}
```

#### PlanDefinition Resource

Converts the trial design (visits, arms) to a FHIR `PlanDefinition`.

| MedLang Element | FHIR PlanDefinition Field | Notes |
|-----------------|--------------------------|-------|
| `protocol.visits` | `PlanDefinition.action[]` | Visit schedule as actions |
| `visit.day` | `action.timing.event` | Timing relative to baseline |

**Example:**
```json
{
  "resourceType": "PlanDefinition",
  "id": "TRIAL001_plan",
  "status": "active",
  "action": [
    {
      "title": "Baseline",
      "timing": {
        "event": ["Day 0"]
      }
    },
    {
      "title": "Week12",
      "timing": {
        "event": ["Day 84"]
      }
    }
  ]
}
```

#### Measure Resources

Converts each endpoint to a FHIR `Measure` resource (one per endpoint).

| MedLang Element | FHIR Measure Field | Notes |
|-----------------|-------------------|-------|
| `endpoint.name` | `Measure.title` | Endpoint code |
| `endpoint.kind` | `Measure.group.population[].code` | ORR / PFS classification |
| `endpoint.window_days` | Measure description | Assessment window |

**Example:**
```json
{
  "resourceType": "Measure",
  "id": "TRIAL001_ORR",
  "status": "active",
  "title": "ORR",
  "description": "Endpoint: ORR (Objective Response Rate)",
  "group": [
    {
      "code": {
        "text": "ORR (Objective Response Rate)"
      },
      "description": "Window: 84 days"
    }
  ]
}
```

#### Bundle Resource (Trial Data)

Converts `TrialDataset` to a FHIR `Bundle` with Patient and Observation resources.

| TrialDataset | FHIR Bundle Entry | Notes |
|--------------|-------------------|-------|
| Unique `subject_id` | `Patient` | One Patient per subject |
| Each observation row | `Observation` | Measurement at time point |
| `dv` (dependent variable) | `Observation.value_quantity` | Tumor volume measurement |
| `time_days` | `Observation.effective_date_time` | Measurement date/time |

**Example:**
```json
{
  "resourceType": "Bundle",
  "type": "collection",
  "entry": [
    {
      "resource": {
        "resourceType": "Patient",
        "id": "patient_1",
        "identifier": [{
          "system": "urn:medlang:study:TRIAL001",
          "value": "1"
        }]
      }
    },
    {
      "resource": {
        "resourceType": "Observation",
        "id": "obs_0",
        "code": {
          "text": "Tumor Volume"
        },
        "subject": {
          "reference": "Patient/patient_1"
        },
        "effective_date_time": "Day 0",
        "value_quantity": {
          "value": 100.0,
          "unit": "mm3",
          "code": "mm3"
        }
      }
    }
  ]
}
```

### CLI Usage

```bash
# Export protocol as FHIR resources
mlc export-fhir \
  --protocol protocol.medlang \
  --protocol-name TRIAL001 \
  --output output/fhir

# Export protocol with trial data as Bundle
mlc export-fhir \
  --protocol protocol.medlang \
  --protocol-name TRIAL001 \
  --trial-data trial_data.json \
  --output output/fhir
```

### Output Structure

```
output/fhir/
├── ResearchStudy.json      # Overall trial metadata
├── PlanDefinition.json     # Trial design (visits, arms)
├── Measure_1.json          # Primary endpoint definition
├── Measure_2.json          # Secondary endpoint definition
└── Bundle.json             # Patient + observation data (if provided)
```

### Use Cases

- **ClinicalTrials.gov registration**: Upload protocol resources
- **FHIR server integration**: Load trial data into EHR systems
- **Regulatory submissions**: Include FHIR representation in eCTD
- **Data sharing**: FHIR-compliant interoperability

---

## 2. CQL Export (`export-cql`)

**CQL = Clinical Quality Language**

CQL is a FHIR-compatible language for defining clinical quality measures and endpoint logic.

### MedLang → CQL Mapping

#### ORR Endpoint → CQL Library

Converts a binary response endpoint (≥30% reduction) to CQL.

| MedLang Concept | CQL Definition | Semantics |
|-----------------|----------------|-----------|
| `endpoint.threshold` | `define "Has Response"` | Response threshold logic |
| `endpoint.window_days` | `define "Assessment Window"` | Time window for evaluation |
| Baseline measurement | `define "Baseline Tumor Volume"` | Initial observation |
| Assessment measurement | `define "Assessment Tumor Volume"` | Final observation in window |
| Response calculation | `define "Percent Change"` | (Assessment - Baseline) / Baseline |

**Example CQL for ORR:**
```cql
library TRIAL001_ORR version '1.0.0'

using FHIR version '4.0.1'
include FHIRHelpers version '4.0.1'

context Patient

define "Assessment Window":
  Interval[@2024-01-01, @2024-01-01 + 84 days]

define "Tumor Volume Observations":
  [Observation: "Tumor Volume"] O
    where O.status = 'final'
      and O.effective in "Assessment Window"

define "Baseline Tumor Volume":
  First([Observation: "Tumor Volume"]).value as Quantity

define "Assessment Tumor Volume":
  Last("Tumor Volume Observations").value as Quantity

define "Percent Change":
  if "Baseline Tumor Volume" is not null
  then (("Assessment Tumor Volume".value - "Baseline Tumor Volume".value) / "Baseline Tumor Volume".value)
  else null

define "Has Response":
  "Percent Change" is not null and "Percent Change" <= -0.3

define "Objective Response":
  if "Has Response" then 1 else 0
```

#### PFS Endpoint → CQL Library

Converts a time-to-event endpoint to CQL.

| MedLang Concept | CQL Definition | Semantics |
|-----------------|----------------|-----------|
| Progression criterion | `define "Progression Events"` | ≥20% increase from nadir |
| Event time | `define "Progression Time Days"` | Days to first progression |
| Censoring | `define "Follow Up Days"` | Days to last observation |
| PFS outcome | `define "PFS Outcome"` | (time, event) tuple |

**Example CQL for PFS:**
```cql
library TRIAL001_PFS version '1.0.0'

using FHIR version '4.0.1'
include FHIRHelpers version '4.0.1'

context Patient

define "Progression Events":
  "Tumor Volume Observations" O
    let nadir: Minimum(...),
        percent_change: (...) / nadir.value
    where percent_change >= 0.20
    return { time: O.effective, volume: O.value }

define "Progression Time Days":
  if exists("Progression Events")
  then days between @baseline and First("Progression Events").time
  else null

define "PFS Outcome":
  {
    time_days: Coalesce("Progression Time Days", "Follow Up Days"),
    event: if "Progression Time Days" is not null then 1 else 0
  }
```

### CLI Usage

```bash
# Export endpoints as CQL libraries
mlc export-cql \
  --protocol protocol.medlang \
  --protocol-name TRIAL001 \
  --output output/cql
```

### Output Structure

```
output/cql/
├── TRIAL001_ORR.cql        # ORR endpoint definition
└── TRIAL001_PFS.cql        # PFS endpoint definition
```

### Use Cases

- **FHIR query language**: Execute on FHIR-compliant systems
- **Clinical decision support**: Integrate with EHR workflows
- **Regulatory measure definitions**: FDA NLM submissions
- **Quality measure automation**: Compute outcomes programmatically

---

## 3. CDISC Export (`export-cdisc`)

**CDISC = Clinical Data Interchange Standards Consortium**

CDISC defines standard formats for clinical trial data, required for regulatory submissions to FDA, EMA, PMDA.

### MedLang → CDISC Mapping

#### ADSL (Subject-Level Analysis Dataset)

Subject-level characteristics and enrollment data.

| MedLang Element | ADSL Column | Notes |
|-----------------|-------------|-------|
| `protocol.name` | `STUDYID` | Study identifier |
| `subject_id` | `SUBJID` | Subject identifier |
| `arm` | `ARM` | Treatment arm |
| `dose_mg` | `DOSE_MG` | Assigned dose |
| `wt` (baseline weight) | `WEIGHT_KG` | Subject weight in kg |
| First `dv` observation | `BASELINE_VOL` | Baseline tumor volume |
| Row count per subject | `N_OBS` | Number of observations |
| Last `time_days` | `LAST_OBS_DAY` | Final follow-up day |
| Always "completed" | `STATUS` | Subject status |

**Example ADSL:**
```csv
STUDYID,SUBJID,ARM,DOSE_MG,WEIGHT_KG,BASELINE_VOL,N_OBS,LAST_OBS_DAY,STATUS
TRIAL001,1,ArmA,0,70,100,2,84,completed
TRIAL001,2,ArmB,100,75,110,2,84,completed
TRIAL001,3,ArmB,100,72,105,1,84,completed
```

#### ADTR (Tumor Response Analysis Dataset)

Observation-level tumor response data.

| MedLang Element | ADTR Column | Notes |
|-----------------|-------------|-------|
| `protocol.name` | `STUDYID` | Study identifier |
| `subject_id` | `SUBJID` | Subject identifier |
| `arm` | `ARM` | Treatment arm |
| `time_days` | `TIME_DAY` | Days from baseline |
| `dv` | `TUMOR_VOL` | Tumor volume (mm³) |
| Baseline dv | `BASELINE_VOL` | Subject baseline for % calculation |
| (dv - baseline) / baseline | `PCT_CHANGE` | % change from baseline |
| 1 if PCT_CHANGE ≤ -30% | `RESPONSE` | Response (1=responder, 0=non-responder) |

**Example ADTR:**
```csv
STUDYID,SUBJID,ARM,TIME_DAY,TUMOR_VOL,BASELINE_VOL,PCT_CHANGE,RESPONSE
TRIAL001,1,ArmA,0,100,100,0,0
TRIAL001,1,ArmA,84,95,100,-5,0
TRIAL001,2,ArmB,0,110,110,0,0
TRIAL001,2,ArmB,84,70,110,-36.36,1
TRIAL001,3,ArmB,0,105,105,0,0
TRIAL001,3,ArmB,84,73,105,-30.48,1
```

### Response Classification Rules

- **Responder**: ≥30% reduction from baseline at any visit
- **Non-responder**: <30% reduction or progression
- **Progression**: ≥20% increase from nadir (for PFS)

### CLI Usage

```bash
# Export trial data as CDISC datasets
mlc export-cdisc \
  --trial-data trial_data.json \
  --study-id TRIAL001 \
  --output output/cdisc
```

### Output Structure

```
output/cdisc/
├── adsl.csv               # Subject-level (CSV)
├── adsl.json              # Subject-level (JSON)
├── adtr.csv               # Observation-level (CSV)
└── adtr.json              # Observation-level (JSON)
```

### Use Cases

- **FDA submissions**: Required for eCTD Module 5.3.5
- **EMA submissions**: XPORT/XPT format (convert from CSV)
- **PMDA submissions**: Japanese regulatory submissions
- **SAS analysis**: Standard input for statistical workflows
- **Trial databases**: Direct import into clinical trial systems

---

## Workflow Examples

### Complete Trial Submission Package

```bash
#!/bin/bash

# Define inputs
PROTOCOL="protocol.medlang"
TRIAL_DATA="trial_data.json"
STUDY_ID="TRIAL001"
OUTPUT_DIR="submission"

# Create FHIR resources for protocol & data
mlc export-fhir \
  --protocol $PROTOCOL \
  --protocol-name $STUDY_ID \
  --trial-data $TRIAL_DATA \
  --output $OUTPUT_DIR/fhir

# Create CQL endpoint definitions
mlc export-cql \
  --protocol $PROTOCOL \
  --protocol-name $STUDY_ID \
  --output $OUTPUT_DIR/cql

# Create CDISC datasets for regulatory submission
mlc export-cdisc \
  --trial-data $TRIAL_DATA \
  --study-id $STUDY_ID \
  --output $OUTPUT_DIR/cdisc

# Package for submission
zip -r $STUDY_ID"_submission.zip" $OUTPUT_DIR/
```

### FHIR Server Integration

```bash
# Export protocol to FHIR
mlc export-fhir \
  --protocol protocol.medlang \
  --protocol-name TRIAL001 \
  --trial-data trial_data.json \
  --output trial.fhir

# Load into FHIR server
curl -X POST http://fhir-server/fhir \
  -H "Content-Type: application/fhir+json" \
  -d @trial.fhir/ResearchStudy.json

curl -X POST http://fhir-server/fhir \
  -H "Content-Type: application/fhir+json" \
  -d @trial.fhir/PlanDefinition.json

curl -X POST http://fhir-server/fhir \
  -H "Content-Type: application/fhir+json" \
  -d @trial.fhir/Bundle.json
```

### CQL Endpoint Evaluation

```bash
# Export CQL definitions
mlc export-cql \
  --protocol protocol.medlang \
  --protocol-name TRIAL001 \
  --output trial.cql

# Execute CQL in FHIR system
# (Uses native CQL execution engine)
# Evaluate "TRIAL001_ORR"."Objective Response"
# For each patient in cohort
```

---

## Reference Standards

### FHIR
- **Specification**: http://hl7.org/fhir/R4/
- **Resources Used**:
  - `ResearchStudy`: Clinical trial metadata (FHIR R4)
  - `PlanDefinition`: Study design definition (FHIR R4)
  - `Measure`: Quality measure / endpoint (FHIR R4)
  - `Bundle`: Collection of resources (FHIR R4)
  - `Patient`: Study participant (FHIR R4)
  - `Observation`: Clinical measurement (FHIR R4)

### CQL
- **Specification**: https://hl7.org/fhirpath/
- **Version**: CQL 1.5 (FHIR 4.0.1 compatible)
- **Execution**: FHIR-compliant CQL engine required

### CDISC
- **Organization**: Clinical Data Interchange Standards Consortium
- **Specifications**:
  - SDTM (Study Data Tabulation Model)
  - ADaM (Analysis Data Model)
  - SEND (Standard for Exchange of Non-clinical Data)
- **Regulatory Use**: FDA, EMA, PMDA, Health Canada, others
- **eCTD Module**: Module 5.3.5 (Study Data Tabulation)

---

## Data Flow Diagram

```
MedLang Protocol
    │
    ├─→ FHIR Export ──→ ResearchStudy.json
    │                 └─→ PlanDefinition.json
    │                 └─→ Measure_N.json
    │
    ├─→ CQL Export ───→ TRIAL001_ORR.cql
    │                 └─→ TRIAL001_PFS.cql
    │
    └─→ CDISC Export ─→ adsl.csv
                      └─→ adtr.csv

TrialDataset
    │
    ├─→ FHIR Export ───→ Bundle.json (Patients + Observations)
    │
    └─→ CDISC Export ─→ adsl.csv
                      └─→ adtr.csv
```

---

## Implementation Notes

1. **Endpoint Response Threshold**: Default 30% reduction for ORR (RECIST v1.1)
2. **Progression Threshold**: Default 20% increase from nadir for PFS
3. **FHIR Compliance**: Resources validated against FHIR R4 StructureDefinitions
4. **CQL Context**: Patient-level evaluation (one CQL context per patient)
5. **CDISC Naming**: Follows CDISC-ADaM naming conventions
6. **Baseline Definition**: First observation (time_days = 0) for each subject
7. **Missing Data**: Handled gracefully; censored observations in PFS

---

## Future Extensions

- **XPORT/SAS Format**: Generate XPORT files directly (for FDA submissions)
- **ADaM Extension**: Derived analysis datasets (ADVS, ADAE, etc.)
- **CQL Execution**: Native CQL execution engine
- **LOINC Coding**: Map endpoints to LOINC codes automatically
- **JSON-LD**: Semantic web representation of trial data
