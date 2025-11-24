# MedLang Completion Roadmap - Unfinished Features from Weeks 1-9

## Status Assessment

### Week 1-5: L₁ MedLang-D Core
**Status**: ✅ **COMPLETE**
- Lexer, parser, AST ✅
- Type system with dimensional analysis ✅
- IR/NIR lowering ✅
- Stan backend ✅
- Basic Julia backend ✅
- 1-comp and 2-comp PK models ✅
- Population NLME ✅
- Data generation ✅

**Nothing to complete** - Core is solid and tested.

---

### Week 6: QM Stub Integration
**Status**: ✅ **COMPLETE**
- `QuantumStub` struct with Kd and ΔG ✅
- JSON loading ✅
- `ec50_from_kd()` and `kp_tumor_from_dg()` ✅
- `IRExternalScalar` for QM constants ✅
- `lower_program_with_qm()` ✅
- Stan backend emits QM constants in data block ✅
- CLI `--qm-stub` flag ✅
- 7 integration tests ✅

**Nothing to complete** - QM integration is functional.

---

### Week 7: PBPK-QSP-QM Vertical
**Status**: ✅ **MOSTLY COMPLETE** (1 minor gap)

**What Works**:
- 5-compartment PBPK model ✅
- Composite model AST (submodel/connect) ✅
- Intermediates emitted in Stan ODE ✅
- QM-informed Kp_tumor and EC50 ✅
- Allometric scaling ✅
- Working example: `oncology_pbpk_qsp_simple.medlang` ✅
- 8 PBPK-QSP-QM tests ✅

**What's Incomplete**:
❌ **Composite model connection resolution** - Input substitution not implemented

**Issue**: In `lower_composite_model()`, connections are collected but not applied:
```rust
// Current code collects connections:
for conn in &connections {
    let to_key = format!("{}.{}", conn.to_model, conn.to_field);
    if let Some(entry) = input_map.get_mut(&to_key) {
        *entry = (conn.from_model.clone(), conn.from_field.clone());
    }
}

// But never substitutes inputs with connected expressions
// ODEs still reference undefined `C_drug` instead of connected `C_tumor`
```

**Impact**: Composite models parse but generate invalid Stan code (undefined variables).

**Workaround**: Use monolithic models like `oncology_pbpk_qsp_simple.medlang`.

---

### Week 8: Protocol DSL Foundation
**Status**: 🟡 **FOUNDATION ONLY** (5 major gaps)

**What Works**:
- Protocol AST structures (ProtocolDef, ArmDef, VisitDef, etc.) ✅
- Lexer tokens (protocol, arms, visits, endpoints, etc.) ✅
- Type checker integration ✅
- Example protocol file ✅
- Architecture documentation ✅

**What's Incomplete**:
❌ **Protocol parser** - Cannot parse `.medlang` protocol files  
❌ **Protocol IR** - No `IRProtocol` structures  
❌ **Protocol → Timeline instantiation** - Cannot generate per-arm timelines  
❌ **ORR endpoint evaluation** - No trajectory analysis  
❌ **CLI `simulate-protocol` command** - Not implemented  

**Impact**: Can define protocols programmatically in Rust, but cannot parse/execute them.

**Estimated effort**: 20-25 hours

---

### Week 9: Extensions (PFS, FHIR, Power)
**Status**: 📋 **DESIGN ONLY** (not started)

**What Exists**:
- Complete architecture document ✅
- Code examples for all modules ✅

**What's Incomplete**:
❌ **Everything** - All Week 9 features are architectural only:
  - PFS/survival analysis
  - FHIR export
  - Bayesian power analysis

**Impact**: No code written yet, only design.

**Estimated effort**: 36-46 hours

---

## Prioritized Completion Plan

### Priority 1: Complete Week 7 Composite Models (CRITICAL)

**Why**: Blocking issue for clean model composition. Currently have workaround but it's not ideal.

**Tasks**:
1. Implement input expression substitution in `lower_composite_model()`
2. Add prefix/namespace handling for variable resolution
3. Test with `test_composite_minimal.medlang`

**Files to modify**:
- `compiler/src/lower.rs` (~50 lines)

**Effort**: 3-4 hours

**Test**:
```rust
#[test]
fn test_composite_connection_resolution() {
    let source = include_str!("../../docs/examples/test_composite_minimal.medlang");
    let ir = lower_program(&parse_program(&tokenize(source).unwrap()).unwrap()).unwrap();
    
    // Should NOT reference undefined C_drug
    let stan = codegen::stan::generate_stan(&ir).unwrap();
    assert!(!stan.contains("C_drug"), "Should resolve C_drug to C_plasma connection");
}
```

---

### Priority 2: Complete Week 8 Protocol Implementation (HIGH)

**Why**: Needed for clinical-facing features. Foundation is solid, just needs execution.

**Phase 1: Parser (6-8 hours)**

Files to create/modify:
- `compiler/src/parser.rs` (~200 lines)

Tasks:
1. `protocol_def()` - Parse protocol blocks
2. `arms_block()` - Parse arms { ArmA { ... } }
3. `visits_block()` - Parse visits { baseline at 0.0_d; ... }
4. `inclusion_block()` - Parse inclusion criteria
5. `endpoints_block()` - Parse endpoints { ORR { ... } }

**Phase 2: IR Layer (5-7 hours)**

Files to create:
- `compiler/src/ir/protocol.rs` (~250 lines)

Files to modify:
- `compiler/src/ir.rs` - Add `mod protocol;`
- `compiler/src/lower.rs` (~100 lines)

Tasks:
1. Define `IRProtocol`, `IRArm`, `IRVisit`, `IREndpoint` structs
2. Implement `lower_protocol(ast: &ProtocolDef) -> IRProtocol`
3. Unit conversion (mg, days, cm³)
4. Validation (doses > 0, times increasing, etc.)

**Phase 3: Timeline Instantiation (5-6 hours)**

Files to create:
- `compiler/src/protocol.rs` (~300 lines)

Tasks:
1. `instantiate_arm_timeline()` - Generate timeline from arm spec
2. `create_arm_cohort()` - Generate cohort with inclusion filters
3. `apply_inclusion_criteria()` - Filter subjects

**Phase 4: Endpoint Evaluation (6-8 hours)**

Files to create:
- `compiler/src/endpoints.rs` (~200 lines)

Tasks:
1. `SubjectTrajectory` data structure
2. `extract_trajectories()` - From ODE solution
3. `compute_orr()` - Response rate calculation
4. `compute_binomial_ci()` - Confidence intervals

**Phase 5: CLI Integration (3-4 hours)**

Files to modify:
- `compiler/src/bin/mlc.rs` (~150 lines)

Tasks:
1. Add `compile-protocol` subcommand
2. Add `simulate-protocol` subcommand
3. JSON output formatting
4. Integration with existing simulation

**Phase 6: Testing (3-4 hours)**

Files to create:
- `tests/protocol_parser_test.rs` (~150 lines)
- `tests/protocol_ir_test.rs` (~100 lines)
- `tests/endpoint_evaluation_test.rs` (~120 lines)
- `tests/protocol_integration_test.rs` (~80 lines)

Tasks:
1. Parser tests (5-8 tests)
2. IR lowering tests (3-5 tests)
3. Endpoint evaluation tests (4-6 tests)
4. End-to-end integration tests (2-3 tests)

**Total Week 8 effort**: 28-37 hours

---

### Priority 3: Week 9 Implementation (MEDIUM - can defer)

Since this is all new features on top of Week 8, defer until Week 8 is complete.

**Estimated effort**: 36-46 hours (as designed)

---

## Recommended Execution Order

### Sprint 1: Fix Composite Models (1 day)
**Effort**: 3-4 hours  
**Deliverable**: Composite models work correctly  
**Files**: `compiler/src/lower.rs`

### Sprint 2: Protocol Parser + IR (2 days)
**Effort**: 11-15 hours  
**Deliverable**: Can parse and lower protocols to IR  
**Files**: `parser.rs`, `ir/protocol.rs`, `lower.rs`

### Sprint 3: Protocol Execution (2 days)
**Effort**: 11-14 hours  
**Deliverable**: Can instantiate timelines and evaluate ORR  
**Files**: `protocol.rs`, `endpoints.rs`

### Sprint 4: CLI + Testing (1-2 days)
**Effort**: 6-8 hours  
**Deliverable**: Working `mlc simulate-protocol` command  
**Files**: `bin/mlc.rs`, test files

### Sprint 5: Week 9 Extensions (4-5 days)
**Effort**: 36-46 hours  
**Deliverable**: PFS, FHIR export, power analysis  
**Files**: `survival.rs`, `fhir_export.rs`, `power_analysis.rs`

**Total estimated time**: 67-87 hours (9-12 days of focused work)

---

## Current Test Status

```
Running tests:
✅ 55 passed (lib)
✅ 8 passed (codegen)
✅ 6 passed (other)
✅ 9 passed (qm_stub unit)
✅ 11 passed (integration)
✅ 7 passed (qm integration)
✅ 6 passed (validation)
✅ 8 passed (pbpk integration)
✅ 8 passed (pbpk_qsp_qm week7)

Total: 118 tests passing
```

**Target after completion**: 155-165 tests

---

## Detailed Task Breakdown

### Task 1: Fix Composite Model Connection Resolution

**File**: `compiler/src/lower.rs`

**Current issue**:
```rust
// In lower_composite_model():
// Step 3: Apply connections
for conn in &connections {
    let to_key = format!("{}.{}", conn.to_model, conn.to_field);
    if let Some(entry) = input_map.get_mut(&to_key) {
        *entry = (conn.from_model.clone(), conn.from_field.clone());
    }
}

// Step 4: (MISSING) Replace input references in ODEs
// Need to walk IR expressions and substitute variables
```

**Solution**:
```rust
// Add after Step 3:

// Step 4: Substitute inputs in expressions
for ode in &mut all_odes {
    ode.rhs = substitute_inputs(&ode.rhs, &input_map);
}

for intermediate in &mut all_intermediates {
    intermediate.expr = substitute_inputs(&intermediate.expr, &input_map);
}

fn substitute_inputs(
    expr: &IRExpr,
    input_map: &HashMap<String, (String, String)>,
) -> IRExpr {
    match expr {
        IRExpr::Var(name) => {
            // Check if this variable is an input that should be substituted
            for (input_key, (from_model, from_field)) in input_map {
                if input_key.ends_with(&format!(".{}", name)) {
                    // Replace with the connected variable
                    return IRExpr::Var(from_field.clone());
                }
            }
            IRExpr::Var(name.clone())
        },
        IRExpr::BinOp { op, lhs, rhs } => IRExpr::BinOp {
            op: *op,
            lhs: Box::new(substitute_inputs(lhs, input_map)),
            rhs: Box::new(substitute_inputs(rhs, input_map)),
        },
        IRExpr::UnaryOp { op, operand } => IRExpr::UnaryOp {
            op: *op,
            operand: Box::new(substitute_inputs(operand, input_map)),
        },
        IRExpr::FnCall { name, args } => IRExpr::FnCall {
            name: name.clone(),
            args: args.iter().map(|a| substitute_inputs(a, input_map)).collect(),
        },
        _ => expr.clone(),
    }
}
```

**Test**:
```rust
#[test]
fn test_input_substitution() {
    // Composite model: QSP.C_drug = PBPK.C_tumor
    let source = r#"
        model PBPK_Mini {
            state A : DoseMass
            param V : Volume
            let C_tumor = A / V
            dA/dt = -C_tumor
            obs C_obs : ConcMass = C_tumor
        }
        
        model QSP_Mini {
            input C_drug : ConcMass
            state T : TumourVolume
            param k : RateConst
            dT/dt = -k * C_drug * T
        }
        
        model Composite {
            submodel PK : PBPK_Mini
            submodel PD : QSP_Mini
            connect {
                PD.C_drug = PK.C_tumor
            }
        }
        
        population Test {
            model Composite
            param V_pop : Volume
            param k_pop : RateConst
            bind_params(p) {
                model.V = V_pop
                model.k = k_pop
            }
            use_measure M for model.C_obs
        }
        
        measure M {
            pred : ConcMass
            obs : ConcMass
            param sigma : f64
            log_likelihood = Normal_logpdf(obs, pred, sigma)
        }
    "#;
    
    let tokens = tokenize(source).unwrap();
    let program = parse_program(&tokens).unwrap();
    let ir = lower_program(&program).unwrap();
    
    // Check that C_drug was substituted with C_tumor
    let stan = codegen::stan::generate_stan(&ir).unwrap();
    assert!(!stan.contains("C_drug"), "C_drug should be substituted");
    assert!(stan.contains("C_tumor"), "Should reference C_tumor");
}
```

---

### Task 2: Protocol Parser Implementation

**File**: `compiler/src/parser.rs`

**Add to declaration parsing**:
```rust
fn declaration(input: TokenSlice) -> IResult<TokenSlice, Declaration> {
    alt((
        map(model_def, Declaration::Model),
        map(population_def, Declaration::Population),
        map(measure_def, Declaration::Measure),
        map(timeline_def, Declaration::Timeline),
        map(cohort_def, Declaration::Cohort),
        map(protocol_def, Declaration::Protocol),  // NEW
    ))(input)
}

fn protocol_def(input: TokenSlice) -> IResult<TokenSlice, ProtocolDef> {
    let (input, _) = token(Token::Protocol)(input)?;
    let (input, name) = identifier(input)?;
    let (input, _) = token(Token::LBrace)(input)?;
    
    // Parse protocol items
    let (input, population_model_name) = parse_population_model_ref(input)?;
    let (input, arms) = parse_arms_block(input)?;
    let (input, visits) = parse_visits_block(input)?;
    let (input, inclusion) = opt(parse_inclusion_block)(input)?;
    let (input, endpoints) = parse_endpoints_block(input)?;
    
    let (input, _) = token(Token::RBrace)(input)?;
    
    Ok((input, ProtocolDef {
        name,
        population_model_name,
        arms,
        visits,
        inclusion,
        endpoints,
        span: None,
    }))
}

fn parse_population_model_ref(input: TokenSlice) -> IResult<TokenSlice, String> {
    // Parse: population_model ModelName
    let (input, _) = token(Token::Population)(input)?;
    let (input, _) = token(Token::Model)(input)?;
    let (input, model_name) = identifier(input)?;
    Ok((input, model_name))
}

fn parse_arms_block(input: TokenSlice) -> IResult<TokenSlice, Vec<ArmDef>> {
    let (input, _) = token(Token::Arms)(input)?;
    let (input, _) = token(Token::LBrace)(input)?;
    
    let (input, arms) = many1(parse_arm_def)(input)?;
    
    let (input, _) = token(Token::RBrace)(input)?;
    Ok((input, arms))
}

fn parse_arm_def(input: TokenSlice) -> IResult<TokenSlice, ArmDef> {
    // Parse: ArmName { label = "..."; dose = 200.0_mg; }
    let (input, name) = identifier(input)?;
    let (input, _) = token(Token::LBrace)(input)?;
    
    let (input, label) = parse_arm_field(input, "label")?;
    let (input, dose_mg) = parse_arm_field_float(input, "dose")?;
    
    let (input, _) = token(Token::RBrace)(input)?;
    
    Ok((input, ArmDef {
        name,
        label,
        dose_mg,
        span: None,
    }))
}

// Similar functions for visits, inclusion, endpoints...
```

---

## Summary

**What's Actually Incomplete**:

1. ✅ **Weeks 1-6**: Complete and tested
2. 🟡 **Week 7**: 95% complete, missing connection resolution (3-4 hours)
3. 🟡 **Week 8**: Foundation complete, execution missing (28-37 hours)
4. ❌ **Week 9**: Design only, no code (36-46 hours)

**Total Remaining Work**: 67-87 hours (9-12 focused days)

**Recommended Order**:
1. Fix Week 7 composite models (1 day)
2. Complete Week 8 protocol implementation (5 days)
3. Implement Week 9 extensions (5 days)

**Result**: Fully functional quantum-to-clinic platform with no gaps.
