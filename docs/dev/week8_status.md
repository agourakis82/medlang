# Week 8 Status: L₂ Clinical Protocol DSL

## Current Status: Foundation Complete, Parser Implementation In Progress

### ✅ Completed (Ready for Use)

1. **AST Infrastructure** - Complete protocol representation
   - Full type system for protocols, arms, visits, endpoints
   - Inclusion criteria (age, ECOG, baseline tumor)
   - Binary and time-to-event endpoint specs
   - Integrated into existing `Declaration` enum

2. **Lexer Support** - All protocol keywords recognized
   - `protocol`, `arms`, `visits`, `inclusion`, `endpoints`
   - `between` for ranges
   - Reuses existing `at` for visit times

3. **Type System** - Protocol validation hooks ready
   - Placeholder in typeck.rs for full implementation
   - No regression in existing tests

4. **Example Protocol** - Oncology Phase 2 trial specification
   - 2 arms (200mg vs 400mg QD)
   - 4 visits (baseline → 12 weeks)
   - Realistic inclusion criteria
   - ORR endpoint definition

### 🚧 In Progress / Remaining

#### Critical Path Items

1. **Parser Implementation** (3-4 hours)
   - Protocol block parsing
   - Arms, visits, inclusion, endpoints sub-blocks
   - Unit literal handling for doses/times
   - Integration with existing parser infrastructure

2. **IR Layer** (2-3 hours)
   - `IRProtocol` and supporting structures
   - AST → IR lowering
   - Unit conversion and validation

3. **Protocol → Timeline Instantiation** (3-4 hours)
   - Per-arm timeline generation
   - Dose events from arm specification
   - Observation events from visit schedule
   - Cohort creation with inclusion filters

4. **Endpoint Evaluation Engine** (4-5 hours)
   - `SubjectTrajectory` data structure
   - ORR computation (response rate)
   - Trajectory extraction from simulation results
   - Statistical summaries (mean, CI)

5. **CLI Integration** (2-3 hours)
   - `compile-protocol` command
   - `simulate-protocol` command
   - JSON output formatting
   - Integration with existing simulation

6. **Testing** (3-4 hours)
   - Parser tests (arms, visits, endpoints)
   - IR lowering tests
   - Endpoint evaluation tests (synthetic data)
   - End-to-end protocol simulation test

**Total Estimated Effort**: 17-23 hours

### Architecture Decisions Made

1. **L₂ Layer is Compositional**
   - Protocols reference existing population models
   - No changes to L₁ (MedLang-D) semantics
   - Clean module separation

2. **Protocol → Multiple Timelines**
   - Each arm gets its own `IRTimeline`
   - Allows per-arm dosing, observation schedules
   - Reuses existing timeline/cohort infrastructure

3. **Inclusion as Post-Simulation Filter**
   - Simulate all subjects
   - Apply inclusion criteria to trajectories
   - Allows "screen failure" analysis

4. **Endpoints as Pure Functions**
   - Take `Vec<SubjectTrajectory>` → `EndpointResult`
   - No side effects, easy to test
   - Modular: can add PFS, OS, biomarker endpoints later

### What Works Right Now

With the current implementation, you can:

- ✅ Define protocol AST structures programmatically
- ✅ Serialize/deserialize protocols via serde
- ✅ Type-check protocols (basic validation)
- ✅ Compile existing PBPK-QSP-QM models
- ✅ Run simulations with QM stubs

### What Needs Implementation

To make Week 8 fully functional:

- ❌ Parse `.medlang` protocol files
- ❌ Convert protocol → IR
- ❌ Generate timelines from protocol
- ❌ Simulate virtual trials
- ❌ Compute ORR from trajectories
- ❌ CLI commands for protocols

## Recommended Completion Path

### Phase 1: Parser (Priority 1)

Implement protocol parsing to enable:
```medlang
protocol Oncology_Phase2 {
    population_model Oncology_PBPK_QSP_Pop
    arms { ... }
    visits { ... }
    endpoints { ... }
}
```

**Deliverable**: Can parse `oncology_phase2_protocol.medlang` into AST

### Phase 2: IR + Instantiation (Priority 2)

Implement:
- Protocol IR structures
- AST → IR lowering
- Timeline generation per arm

**Deliverable**: `protocol → Vec<(Arm, Timeline)>`

### Phase 3: Endpoint Evaluation (Priority 3)

Implement:
- Trajectory data structure
- ORR computation
- Result formatting

**Deliverable**: Can compute ORR from simulated data

### Phase 4: CLI Integration (Priority 4)

Implement:
- `simulate-protocol` command
- JSON output
- Integration tests

**Deliverable**: End-to-end virtual trial

### Phase 5: Documentation (Priority 5)

Write:
- Protocol DSL guide
- Endpoint specification reference
- Example workflows
- Week 8 summary

## Testing Status

**Current Test Count**: 139 tests (all passing)

**New Tests Needed**:
- Protocol parser: 5-8 tests
- Protocol IR: 3-5 tests
- Endpoint evaluation: 4-6 tests
- Integration: 2-3 tests

**Target Test Count**: 155-165 tests

## Timeline Estimate

### If Completed Incrementally

- **Day 1**: Parser implementation + tests (6-8 hours)
- **Day 2**: IR + instantiation + tests (5-7 hours)
- **Day 3**: Endpoint eval + CLI + tests (6-8 hours)
- **Day 4**: Documentation + examples (3-4 hours)

**Total**: 20-27 hours over 4 days

### If Completed as Sprint

Focused implementation: **2-3 full days**

## Dependencies

Week 8 builds on:
- ✅ Week 1-5: Core MedLang-D (models, populations, measures)
- ✅ Week 6: QM stub integration
- ✅ Week 7: PBPK-QSP-QM vertical integration

Week 8 enables:
- Week 9: PFS/OS time-to-event endpoints
- Week 10: FHIR/CQL export
- Week 11: Adaptive trial designs

## Key Insights from Week 8 Design

1. **Protocol DSL is Higher-Order**
   - Protocols are *specifications* that generate timelines
   - One protocol → N arms → N timelines
   - This is true meta-programming for clinical trials

2. **Endpoints are Functional Transforms**
   - `Trajectory → Bool` (responder?)
   - `Vec<Trajectory> → f64` (ORR)
   - `Vec<Trajectory> → KM_Curve` (PFS, coming in Week 9)

3. **L₂ + L₁ = Complete Clinical Development**
   - L₁: "How does the drug work?" (mechanistic)
   - L₂: "Does the drug work?" (clinical endpoints)
   - Together: Full translational pipeline

4. **Quantum → Clinic in One File**
   ```
   lig001_qm_stub.json (Kd, ΔG)
           ↓
   Oncology_PBPK_QSP (PBPK + QSP + QM)
           ↓
   Oncology_Phase2 (arms, visits, ORR)
           ↓
   Virtual Trial Results (ORR = 35% vs 48%)
   ```

## Current Limitations

1. **No Parser Yet**
   - Must create protocols programmatically
   - Can't parse `.medlang` protocol files

2. **No Simulation Integration**
   - Have the models (PBPK-QSP-QM)
   - Have the protocol spec (AST)
   - Need the connector (instantiation logic)

3. **ORR Not Implemented**
   - Endpoint spec exists in AST/IR
   - Evaluation function not written
   - Need trajectory extraction from ODE solutions

## What This Unlocks

Once Week 8 is complete, users can:

1. **Design Clinical Trials in MedLang**
   ```medlang
   protocol MyTrial {
       population_model MyDrug_PBPK_QSP
       arms { ... }
       endpoints { ORR { ... } }
   }
   ```

2. **Simulate Virtual Trials**
   ```bash
   mlc simulate-protocol my_trial.medlang \
       --n-subjects 200 \
       --out results.json
   ```

3. **Evaluate Endpoints**
   - Automatic ORR calculation
   - Per-arm statistics
   - Treatment comparison

4. **Iterate Designs Rapidly**
   - Change dose: recompile
   - Add arm: recompile
   - Adjust sample size: rerun
   - All in minutes, not months

## Conclusion

Week 8 establishes the **architectural foundation** for MedLang as a clinical trial design language. The core data structures are in place, and the implementation path is clear.

**Current State**: AST + Lexer ready, parser and execution pending  
**Effort to Complete**: 20-25 hours of focused implementation  
**Impact**: Transforms MedLang from research tool to clinical development platform

The foundation is solid. The path forward is clear. Week 8 is ready to go from architecture to implementation.
