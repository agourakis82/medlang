# User Guide: Analyzing Clinical Trial Data with MedLang

This guide shows you how to analyze observed clinical trial data using MedLang's `analyze-trial` command.

---

## Quick Start

```bash
mlc analyze-trial \
  --protocol my_protocol.medlang \
  --data trial_data.csv \
  --output results.json
```

That's it! MedLang will:
1. Parse your protocol definition
2. Load and validate the trial data
3. Apply inclusion/exclusion criteria
4. Compute all endpoints (ORR, PFS, etc.)
5. Generate JSON results with Kaplan-Meier curves

---

## Step-by-Step Tutorial

### Step 1: Prepare Your Trial Data

Create a CSV file with your observed trial data. **Required columns**:
- `ID` - Subject identifier (string or number)
- `ARM` - Treatment arm name (must match protocol)
- `TIME` - Days since baseline
- `TUMVOL` - Tumor volume measurement

**Optional columns**: `AGE`, `ECOG`, `SEX`, `CENS`

**Example** (`my_trial_data.csv`):
```csv
ID,ARM,TIME,TUMVOL,AGE,ECOG,SEX
S001,TreatmentA,0.0,100.0,55,1,M
S001,TreatmentA,28.0,75.0,55,1,M
S001,TreatmentA,56.0,65.0,55,1,M
S002,TreatmentB,0.0,120.0,62,0,F
S002,TreatmentB,28.0,60.0,62,0,F
S002,TreatmentB,56.0,50.0,62,0,F
```

**Important**:
- Each subject MUST have a baseline observation (TIME = 0 or close to 0)
- TIME values must be in days
- TUMVOL values must be positive
- ARM names must exactly match your protocol definition

### Step 2: Define Your Protocol

Create a protocol file defining your trial structure:

**Example** (`my_protocol.medlang`):
```medlang
protocol MyTrial {
    population model TumorModel
    
    arms {
        TreatmentA { label = "Treatment A"; dose = 100.0 }
        TreatmentB { label = "Treatment B"; dose = 200.0 }
    }
    
    visits {
        baseline at 0.0
        week4 at 28.0
        week8 at 56.0
    }
    
    inclusion {
        age between 18 and 75
        ECOG in [0, 1]
        baseline_tumour_volume >= 50.0
    }
    
    endpoints {
        ORR {
            type = "binary"
            observable = TumourVol
            shrink_frac = 0.30
            window = [0.0, 56.0]
        }
        PFS {
            type = "time_to_event"
            observable = TumourVol
            progression_frac = 0.20
            ref_baseline = false
            window = [0.0, 56.0]
        }
    }
}
```

**Key points**:
- `arms { }` - Define treatment arms (names must match CSV ARM column)
- `visits { }` - Specify observation timepoints
- `inclusion { }` - Set eligibility criteria (optional)
- `endpoints { }` - Define which endpoints to evaluate

### Step 3: Run the Analysis

```bash
mlc analyze-trial \
  --protocol my_protocol.medlang \
  --data my_trial_data.csv \
  --output my_results.json \
  --verbose
```

**Flags**:
- `--protocol` - Path to your protocol definition file
- `--data` - Path to your trial data CSV (or JSON)
- `--output` - Where to save JSON results (optional)
- `--verbose` - Show detailed progress (optional)

### Step 4: View Results

**Console output** (human-readable summary):
```
Trial Analysis Results
======================
Protocol: MyTrial
Data source: my_trial_data.csv

Arm: TreatmentA (Treatment A)
  Subjects: 10 total, 9 included, 1 excluded

  ORR: ORR = 44.4% (4/9 responders)
  PFS: Median = not reached, Events = 2, Censored = 7

Arm: TreatmentB (Treatment B)
  Subjects: 10 total, 10 included, 0 excluded

  ORR: ORR = 70.0% (7/10 responders)
  PFS: Median = not reached, Events = 1, Censored = 9

✓ Results saved to: my_results.json
```

**JSON output** (structured data for downstream analysis):
```json
{
  "protocol_name": "MyTrial",
  "data_source": "my_trial_data.csv",
  "arms": [
    {
      "arm_name": "TreatmentA",
      "label": "Treatment A",
      "n_subjects": 10,
      "n_included": 9,
      "n_excluded": 1,
      "endpoints": {
        "ORR": {
          "type": "Binary",
          "n_responders": 4,
          "response_rate": 0.444
        },
        "PFS": {
          "type": "TimeToEvent",
          "n_events": 2,
          "n_censored": 7,
          "median_days": null,
          "km_times": [28.0, 56.0],
          "km_surv": [1.0, 0.778],
          "km_n_risk": [9, 7],
          "km_n_event": [0, 2]
        }
      }
    }
  ]
}
```

---

## Understanding the Results

### Objective Response Rate (ORR)

**Definition**: Percentage of subjects with ≥30% tumor shrinkage (or custom threshold)

**Output**:
```
ORR: ORR = 44.4% (4/9 responders)
```
- 4 out of 9 subjects achieved ≥30% reduction
- Response rate = 44.4%

**JSON**:
```json
"ORR": {
  "type": "Binary",
  "n_responders": 4,
  "response_rate": 0.444
}
```

### Progression-Free Survival (PFS)

**Definition**: Time until tumor progresses (≥20% increase from nadir or baseline)

**Output**:
```
PFS: Median = 56.0 days, Events = 2, Censored = 7
```
- 2 subjects had progression events
- 7 subjects were censored (no progression observed)
- Median PFS = 56 days (50% of subjects progressed by this time)
- "not reached" means <50% had events

**JSON**:
```json
"PFS": {
  "type": "TimeToEvent",
  "n_events": 2,
  "n_censored": 7,
  "median_days": 56.0,
  "km_times": [28.0, 56.0, 84.0],
  "km_surv": [1.0, 0.778, 0.5],
  "km_n_risk": [9, 7, 4],
  "km_n_event": [0, 2, 2]
}
```

**Kaplan-Meier curve**:
- `km_times`: Observation times
- `km_surv`: Survival probability at each time (S(t))
- `km_n_risk`: Number of subjects at risk
- `km_n_event`: Number of events at each time

### Inclusion/Exclusion

```
Subjects: 10 total, 9 included, 1 excluded
```
- Total: All subjects in the CSV for this arm
- Included: Met all inclusion criteria
- Excluded: Failed one or more criteria (age, ECOG, baseline tumor size)

---

## Common Use Cases

### 1. Compare Multiple Arms

```medlang
arms {
    Control { label = "Placebo"; dose = 0.0 }
    LowDose { label = "100mg"; dose = 100.0 }
    HighDose { label = "200mg"; dose = 200.0 }
}
```

MedLang will automatically analyze all three arms and compare results.

### 2. Test Different Endpoint Thresholds

Try multiple ORR thresholds:
```medlang
endpoints {
    ORR_20 {
        type = "binary"
        observable = TumourVol
        shrink_frac = 0.20  // 20% shrinkage
        window = [0.0, 84.0]
    }
    ORR_30 {
        type = "binary"
        observable = TumourVol
        shrink_frac = 0.30  // 30% shrinkage (standard)
        window = [0.0, 84.0]
    }
}
```

### 3. Evaluate Different Time Windows

Early assessment vs. late assessment:
```medlang
endpoints {
    PFS_early {
        type = "time_to_event"
        observable = TumourVol
        progression_frac = 0.20
        ref_baseline = false
        window = [0.0, 56.0]  // First 8 weeks
    }
    PFS_late {
        type = "time_to_event"
        observable = TumourVol
        progression_frac = 0.20
        ref_baseline = false
        window = [0.0, 168.0]  // Up to 24 weeks
    }
}
```

### 4. Apply Subgroup Filters

Analyze only older patients:
```medlang
inclusion {
    age between 60 and 80
    ECOG in [0, 1]
}
```

Or patients with good performance status:
```medlang
inclusion {
    age between 18 and 75
    ECOG in [0]  // Only ECOG 0
}
```

---

## Troubleshooting

### Error: "ARM mismatch"

```
Error: Protocol arm 'TreatmentA' not found in data. Available arms: ["Treatment_A", "Treatment_B"]
```

**Problem**: ARM names in CSV don't match protocol

**Solution**: Check spelling and capitalization. Update either:
- CSV: Change `Treatment_A` → `TreatmentA`
- Protocol: Change `TreatmentA` → `Treatment_A`

### Error: "No baseline observation"

```
Error: Subject S001: no baseline observation (TIME ≈ 0.0) found
```

**Problem**: Subject missing TIME = 0 row

**Solution**: Ensure every subject has a baseline measurement:
```csv
S001,ArmA,0.0,100.0,55,1,M  ← Must have this row
S001,ArmA,28.0,85.0,55,1,M
```

### Error: "Non-positive tumor volume"

```
Error: Invalid value at row 5, column 'TUMVOL': non-positive tumor volume: 0.0
```

**Problem**: TUMVOL ≤ 0

**Solution**: Replace zero/negative values with small positive numbers or remove the row

### Warning: High exclusion rate

```
Arm: TreatmentA (Treatment A)
  Subjects: 20 total, 5 included, 15 excluded
```

**Problem**: Many subjects don't meet inclusion criteria

**Solution**: Review your inclusion block:
```medlang
inclusion {
    age between 18 and 75  ← Too restrictive?
    ECOG in [0, 1]
    baseline_tumour_volume >= 50.0  ← Threshold too high?
}
```

---

## Advanced Features

### JSON Data Input

Instead of CSV, use JSON format:

```bash
mlc analyze-trial \
  --protocol my_protocol.medlang \
  --data trial_data.json
```

**JSON structure**:
```json
{
  "trial_id": "TRIAL-2024-001",
  "subjects": [
    {
      "id": "S001",
      "arm": "TreatmentA",
      "age": 55,
      "ecog": 1,
      "sex": "M",
      "observations": [
        {"time": 0.0, "tumvol": 100.0},
        {"time": 28.0, "tumvol": 75.0},
        {"time": 56.0, "tumvol": 65.0}
      ]
    }
  ]
}
```

### Programmatic Analysis

Use MedLang as a library:

```rust
use medlangc::data::{TrialDataset, analyze_trial};
use medlangc::lexer::tokenize;
use medlangc::parser::parse_protocol_from_tokens;

// Load data
let dataset = TrialDataset::from_csv("data.csv")?;

// Parse protocol
let protocol_source = std::fs::read_to_string("protocol.medlang")?;
let tokens = tokenize(&protocol_source)?;
let protocol = parse_protocol_from_tokens(&tokens)?;

// Analyze
let results = analyze_trial(&protocol, &dataset, "data.csv")?;

// Extract specific results
for arm in &results.arms {
    if let Some(orr) = arm.endpoints.get("ORR") {
        println!("{}: ORR = {:.1}%", arm.arm_name, orr.response_rate * 100.0);
    }
}
```

---

## Best Practices

### Data Quality

1. **Always include baseline**: Every subject needs TIME = 0
2. **Consistent timing**: Use protocol-aligned visit days (0, 28, 56, etc.)
3. **Complete covariates**: Fill AGE, ECOG for all subjects (required for inclusion filtering)
4. **Handle missing data**: Remove rows with missing TUMVOL or mark as censored

### Protocol Design

1. **Match real data**: Ensure ARM names exactly match your CSV
2. **Realistic windows**: Set endpoint windows based on actual observation times
3. **Appropriate thresholds**: Use standard thresholds (30% for ORR, 20% for PFS)
4. **Document decisions**: Use descriptive labels for arms and endpoints

### Workflow

1. **Start simple**: Test with one arm, one endpoint first
2. **Validate incrementally**: Add complexity after confirming basic analysis works
3. **Compare with manual calculations**: Verify ORR/PFS match expected values
4. **Document results**: Save JSON output with version-controlled protocols

---

## FAQs

**Q: Can I analyze trials with more than 3 arms?**  
A: Yes, define as many arms as needed in the protocol.

**Q: What if my CSV has different column names?**  
A: MedLang accepts both uppercase and lowercase (ID/id, ARM/arm, TIME/time, TUMVOL/tumvol).

**Q: How do I handle missing observations?**  
A: MedLang automatically handles irregular visit schedules. Just include whatever TIME points you have.

**Q: Can I use this for non-oncology trials?**  
A: Currently optimized for tumor response endpoints. Other observables can be added in future versions.

**Q: How do I cite MedLang in publications?**  
A: See the main README for citation information.

---

## Getting Help

- **Documentation**: See `docs/trial_data_schema_v0.1.md` for detailed schema spec
- **Examples**: Check `docs/examples/` for working protocol and data files
- **Issues**: Report bugs at https://github.com/medlang/medlang/issues
- **Verbose mode**: Use `--verbose` flag to see detailed progress and diagnostics

---

**Happy analyzing!** 🎉
