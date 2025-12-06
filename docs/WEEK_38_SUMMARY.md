# Week 38 Implementation Summary

## Goal: DoseGuidelineIR → GuidelineArtifact → CQL Export Pipeline

Week 38 builds the bridge from RL-derived dose rules into the **generic guideline layer**, making RL policies first-class clinical artifacts that can be exported to CQL and compared with standard-of-care guidelines.

---

## What Was Implemented

### 1. DoseGuidelineIR Layer (`dose_guideline_ir.rs`)

A simpler, focused intermediate representation for dose adjustment guidelines with explicit IF/THEN rules:

**Core Types:**
- `ComparisonOpIR`: LT, LE, GT, GE operators
- `AtomicConditionIR`: Simple feature comparisons (e.g., "ANC <= 0.5")
- `DoseRuleIR`: IF conditions THEN dose action
- `DoseGuidelineIRHost`: Complete guideline with rules, features, and dose levels

**Key Functions:**
- `guideline_from_distilled_tree()`: Transform DistilledPolicyTree → DoseGuidelineIRHost
- `pretty_print_dose_guideline()`: Human-readable text output
- `DoseGuidelineIRHost::evaluate()`: Evaluate guideline for given features

**Features:**
- Path extraction from decision trees (root-to-leaf traversal)
- Interval merging for tighter bounds (e.g., x > 0.3 AND x <= 0.7)
- First-match evaluation semantics
- Serializable to JSON via serde

### 2. Bridge Layer (`dose_guideline_bridge.rs`)

Transforms DoseGuidelineIR into general GuidelineArtifact format:

**Main Function:**
```rust
pub fn dose_guideline_to_guideline_artifact(
    dg: &DoseGuidelineIRHost,
    meta: GuidelineMeta,
) -> GuidelineArtifact
```

**Transformations:**
- `AtomicConditionIR` → `GuidelineExpr::Compare`
- Conjunctions → `GuidelineExpr::And`
- Feature names → `GuidelineValueRef` (Anc, TumourRatio, PrevDose, CycleIndex)
- Dose amounts → `DoseActionKind::SetAbsoluteDoseMg` or `HoldDose`
- Auto-generated rule descriptions with dose amounts

### 3. Comparison Framework (`dose_guideline_diff.rs`)

Grid-based comparison for RL vs baseline guidelines:

**Core Types:**
- `DoseGuidelineGridConfig`: Feature space sampling grids (coarse/fine presets)
- `GuidelineDiffSummary`: Aggregate metrics (disagreement, conservative/aggressive ratios)
- `GuidelineDiffReport`: Detailed point-wise differences

**Main Functions:**
```rust
pub fn compare_dose_guidelines_on_grid(
    rl: &DoseGuidelineIRHost,
    baseline: &DoseGuidelineIRHost,
    grid: &DoseGuidelineGridConfig,
) -> GuidelineDiffSummary

pub fn compare_dose_guidelines_detailed(...) -> GuidelineDiffReport
```

**Metrics Computed:**
- Total points evaluated
- Disagreement fraction (0.0 to 1.0)
- RL more aggressive/conservative fractions
- Mean and max dose differences (mg)
- Individual diff points with feature values

**Predicates:**
- `is_similar()`: < 10% disagreement
- `is_rl_more_aggressive()`: > 30% higher doses
- `is_rl_more_conservative()`: > 30% lower doses

### 4. MedLang Standard Library

**New Module:** `stdlib/med/rl/dose_guideline.medlang`

Exports:
- `ComparisonOp` enum (LT, LE, GT, GE)
- `AtomicCondition` type
- `DoseRule` type
- `DoseGuidelineIR` type
- `guideline_from_distilled_policy()` function
- `pretty_print_dose_guideline()` function
- `dose_guideline_to_guideline()` function (bridge to GuidelineArtifact)

### 5. Type Aliases for Backwards Compatibility

Added to `rl/core.rs`:
```rust
pub type RlAction = Action;
pub type RlObservation = State;
```

This resolves import errors in policy modules while allowing gradual migration to new naming.

---

## Complete Pipeline

```
DistilledPolicyTree (from Week 37)
    ↓ [guideline_from_distilled_tree]
DoseGuidelineIRHost (explicit IF/THEN rules)
    ↓ [dose_guideline_to_guideline_artifact + GuidelineMeta]
GuidelineArtifact (generic clinical guideline IR)
    ↓ [CQL exporter - Week 37]
CQL file (FHIR-compatible clinical logic)
```

---

## Example Usage (MedLang)

```medlang
module projects.nsclc_phase2_rl_to_cql;

import med.rl::{RLEnvConfig, train_policy_rl};
import med.rl.explain::{DistillConfig, distill_policy_tree};
import med.rl.dose_guideline::{
  DoseGuidelineIR,
  guideline_from_distilled_policy,
  pretty_print_dose_guideline,
  dose_guideline_to_guideline
};
import med.rl.guideline::{GuidelineMeta, GuidelineArtifact};

fn create_rl_guideline() -> GuidelineArtifact {
  // 1. Configure and train RL policy
  let env_cfg: RLEnvConfig = /* ... */;
  let train_cfg = /* ... */;
  let result = train_policy_rl(env_cfg, train_cfg);
  let policy = result.1;

  // 2. Distill to decision tree
  let distill_cfg: DistillConfig = /* ... */;
  let distill_result = distill_policy_tree(env_cfg, policy, distill_cfg);
  let tree = distill_result.policy;

  // 3. Convert to DoseGuidelineIR
  let dg_ir: DoseGuidelineIR = guideline_from_distilled_policy(
    env_cfg,
    tree,
    "NSCLC Phase 2 RL-derived Guideline",
    "Dose adjustment rules from QSP-RL policy"
  );

  // 4. Pretty-print for review
  let text: String = pretty_print_dose_guideline(dg_ir);
  print(text);

  // 5. Bridge to GuidelineArtifact
  let meta: GuidelineMeta = {
    id = "NSCLC-Phase2-RL-001";
    version = "1.0.0";
    title = "NSCLC Phase 2 RL Dose Guideline";
    description = "RL-derived dose adjustments for ANC/response";
    population = "Adult NSCLC, 2L, ECOG 0-1";
  };

  let artifact: GuidelineArtifact = dose_guideline_to_guideline(dg_ir, meta);
  artifact
}
```

---

## Example Output: Pretty-Printed Guideline

```
Dose Guideline: NSCLC Phase 2 RL-derived Guideline
Description: Dose adjustment rules from QSP-RL policy
Features: ANC, tumour_ratio, prev_dose, cycle
Dose Levels: [0.0, 50.0, 100.0, 200.0, 300.0] mg

Rules (8 total):
================================================================================

Rule #1:
  IF:
    - ANC <= 0.5
    - cycle > 2.0
  THEN: Set dose to 0 mg (action index 0)

Rule #2:
  IF:
    - ANC > 0.5
    - ANC <= 0.8
    - tumour_ratio > 0.7
  THEN: Set dose to 100 mg (action index 2)

Rule #3:
  IF:
    - ANC > 0.8
    - tumour_ratio <= 1.2
  THEN: Set dose to 200 mg (action index 3)
...
```

---

## Example Comparison

```rust
// Create RL-derived guideline
let rl_guideline = guideline_from_distilled_tree(...);

// Create baseline (standard-of-care) guideline
let mut soc_guideline = DoseGuidelineIRHost::new(...);
soc_guideline.add_rule(/* SoC rules */);

// Compare on coarse grid
let grid = DoseGuidelineGridConfig::coarse(); // 4×4×3×3 = 144 points
let summary = compare_dose_guidelines_on_grid(&rl_guideline, &soc_guideline, &grid);

println!("Total points: {}", summary.total_points);
println!("Disagree: {} ({:.1}%)", 
    summary.disagree_points, 
    summary.disagree_fraction * 100.0
);
println!("RL more aggressive: {:.1}%", summary.rl_more_aggressive_fraction * 100.0);
println!("RL more conservative: {:.1}%", summary.rl_more_conservative_fraction * 100.0);
println!("Mean dose diff: {:.1} mg", summary.mean_dose_difference_mg);
println!("Max dose diff: {:.1} mg", summary.max_dose_difference_mg);

if summary.is_similar() {
    println!("✓ RL guideline is substantially similar to SoC");
} else if summary.is_rl_more_conservative() {
    println!("⚠ RL guideline is substantially more conservative");
}
```

---

## Testing

Comprehensive test suite in `tests/week_38_dose_guideline_tests.rs`:

1. **DistilledPolicyTree → DoseGuidelineIR:**
   - Simple single-split trees
   - Multi-feature nested splits
   - Interval merging logic

2. **DoseGuidelineIR → GuidelineArtifact:**
   - Single rule conversion
   - Multiple rules with conjunctions
   - Hold dose vs set dose actions

3. **End-to-End Pipeline:**
   - Tree → DoseGuidelineIR → GuidelineArtifact
   - JSON serialization round-trip

4. **Guideline Comparison:**
   - Identical guidelines (100% agreement)
   - Conservative vs aggressive policies
   - Grid evaluation correctness

5. **Pretty Printing:**
   - Format verification
   - Content completeness

6. **Guideline Evaluation:**
   - First-match semantics
   - Boundary conditions
   - Missing features

**Test Coverage:**
- 503 lines of test code
- 13 test functions
- All core transformations covered

---

## CLI Integration (Planned for Week 39)

**Command:** `mlc rl-dose-guideline-cql`

```bash
# End-to-end: RL policy → distilled tree → guideline IR → CQL
mlc rl-dose-guideline-cql \
  --env-config cfg/env_nsclc_phase2.json \
  --distilled-policy artifacts/distilled_policy_nsclc.json \
  --meta cfg/guideline_meta_nsclc.json \
  --out-guideline-json artifacts/nsclc_phase2_guideline.json \
  --out-cql artifacts/nsclc_phase2_guideline.cql
```

This will be added in the next iteration along with CQL export integration.

---

## Files Modified/Added

**New Files:**
- `compiler/src/rl/dose_guideline_ir.rs` (398 lines)
- `compiler/src/rl/dose_guideline_bridge.rs` (355 lines)
- `compiler/src/rl/dose_guideline_diff.rs` (545 lines)
- `stdlib/med/rl/dose_guideline.medlang` (200 lines)
- `compiler/tests/week_38_dose_guideline_tests.rs` (503 lines)

**Modified Files:**
- `compiler/src/rl/mod.rs`: Added exports for new modules
- `compiler/src/rl/core.rs`: Added RlAction/RlObservation type aliases

**Total New Code:** ~2,000 lines (including tests and documentation)

---

## Key Design Decisions

### 1. Two-Layer IR Design

**DoseGuidelineIR (simpler):**
- Focused on dose adjustment logic
- Explicit IF/THEN structure
- Easy to generate from trees
- Easy to pretty-print

**GuidelineArtifact (general):**
- Supports multiple action types (dose, labs, imaging)
- Rich metadata for clinical context
- CQL export target
- Comparable across sources (RL, human-written, literature)

**Why Both?**
- Separation of concerns: dose logic vs clinical integration
- Simpler debugging and validation of RL-derived rules
- Easier to extend DoseGuidelineIR for RL-specific features

### 2. Grid-Based Comparison

**Why Grid Evaluation?**
- Deterministic and reproducible
- Easy to visualize differences
- Computationally tractable (144-2000 points)
- Comprehensive coverage of feature space

**Alternative Considered:**
- Monte Carlo sampling: less reproducible
- Symbolic rule matching: brittle for complex trees

### 3. Feature Name Mapping

**Current Approach:**
- Canonical names: "ANC", "tumour_ratio", "prev_dose", "cycle"
- Case-insensitive matching in bridge
- Unknown features skipped with warning

**Future Extension:**
- Support `GuidelineValueRef::Lab(name)` for arbitrary labs
- Feature ontology mapping (e.g., LOINC codes)

---

## Known Limitations (v0)

1. **Feature Support:** Only 4 canonical features (ANC, tumour_ratio, prev_dose, cycle)
   - Future: Generic lab value support via GuidelineValueRef::Lab(String)

2. **Comparison Grid:** Fixed feature set assumption
   - Future: Configurable grids per guideline feature space

3. **CLI Integration:** Not yet wired into mlc
   - Week 39: Add `mlc rl-dose-guideline-cql` command

4. **CQL Export:** Bridge exists but CQL generation pending
   - Week 37 has guideline → CQL exporter (needs integration)

5. **Comparison Visualization:** Text-only summary
   - Future: HTML reports, heatmaps, diff tables

---

## Next Steps (Week 39)

1. **CLI Command:** `mlc rl-dose-guideline-cql`
   - Wire end-to-end pipeline from JSON inputs to CQL output
   - Add `mlc rl-compare-guidelines` for side-by-side diffs

2. **CQL Export Integration:**
   - Connect GuidelineArtifact bridge to existing CQL exporter
   - Validate CQL syntax for RL-derived guidelines

3. **Visualization:**
   - HTML diff reports with feature space heatmaps
   - Interactive comparison UI (optional)

4. **Advanced Comparison:**
   - Per-feature disagreement analysis
   - Sensitivity analysis (which features drive differences)
   - Confidence intervals on metrics

5. **Standard-of-Care Baselines:**
   - Encode common NSCLC dosing guidelines
   - Benchmark RL policies against literature

---

## Impact

Week 38 establishes **RL policies as first-class clinical artifacts**:

✅ **Interpretability:** Explicit IF/THEN rules clinicians can review  
✅ **Standardization:** Same GuidelineArtifact format as human guidelines  
✅ **Interoperability:** CQL export path to FHIR/EHR systems  
✅ **Validation:** Grid-based comparison to standard-of-care  
✅ **Traceability:** Full audit trail from RL training to clinical rule  

This infrastructure enables:
- Regulatory submission of RL-derived guidelines
- Clinical trial protocol generation from policies
- Real-time policy comparison in clinical settings
- Continuous learning: update guidelines as policies improve

---

## Architecture Diagram

```
┌─────────────────────────┐
│ DistilledPolicyTree     │ ← Week 37 (decision tree from RL)
│ (root, features, ...)   │
└────────────┬────────────┘
             │ guideline_from_distilled_tree()
             ↓
┌─────────────────────────┐
│ DoseGuidelineIRHost     │ ← Week 38 (explicit dose rules)
│ - rules: Vec<DoseRule>  │
│ - dose_levels_mg        │
│ - feature_names         │
└────────────┬────────────┘
             │ dose_guideline_to_guideline_artifact(meta)
             ↓
┌─────────────────────────┐
│ GuidelineArtifact       │ ← Week 37 (general guideline IR)
│ - meta: GuidelineMeta   │
│ - rules: Vec<...>       │
└────────────┬────────────┘
             │ guideline_to_cql() [Week 37]
             ↓
┌─────────────────────────┐
│ CQL File                │ ← FHIR-compatible clinical logic
│ (text format)           │
└─────────────────────────┘

            Comparison Path (parallel):
┌─────────────────────────┐   ┌─────────────────────────┐
│ RL DoseGuidelineIR      │   │ SoC DoseGuidelineIR     │
└────────────┬────────────┘   └────────────┬────────────┘
             │                             │
             └──────────────┬──────────────┘
                            │ compare_dose_guidelines_on_grid()
                            ↓
                  ┌───────────────────────┐
                  │ GuidelineDiffSummary  │
                  │ - disagree_fraction   │
                  │ - aggressive/conserv  │
                  │ - dose differences    │
                  └───────────────────────┘
```

---

## Conclusion

Week 38 successfully bridges RL-derived policies into the clinical guideline ecosystem. The implementation provides:

- **Clear transformation pipeline:** Tree → DoseGuidelineIR → GuidelineArtifact → CQL
- **Comparison framework:** Quantitative RL vs SoC analysis
- **Clinical readability:** Pretty-printed IF/THEN rules
- **Type safety:** Dimensional consistency via feature name mapping
- **Extensibility:** Foundation for advanced guideline analysis (Week 39+)

The codebase is ready for CLI integration and end-to-end workflow testing in Week 39.