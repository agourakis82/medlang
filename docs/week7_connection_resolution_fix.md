# Week 7: Composite Model Connection Resolution Fix

## Problem Statement

Composite models in MedLang allow combining multiple base models (e.g., PBPK + QSP) using `submodel` declarations and `connect` blocks. However, the connection resolution was incomplete, causing generated Stan code to contain undefined variable references.

### Example Issue

Given this composite model:

```medlang
model PBPK_Simple {
    state A_plasma : DoseMass
    param CL : Clearance
    param V : Volume
    let C_plasma = A_plasma / V
    dA_plasma/dt = -CL * C_plasma
    obs C_plasma_obs : ConcMass = C_plasma
}

model QSP_Simple {
    input C_drug : ConcMass
    state Tumor : TumourVolume
    param k_grow : RateConst
    param Emax : f64
    param EC50 : ConcMass
    let E_drug = Emax * C_drug / (EC50 + C_drug)
    dTumor/dt = k_grow * Tumor - E_drug * Tumor
}

model Composite {
    submodel PK : PBPK_Simple
    submodel PD : QSP_Simple
    connect {
        PD.C_drug = PK.C_plasma_obs
    }
}
```

**Before Fix**: Generated Stan code referenced undefined `C_plasma_obs`:
```stan
real E_drug = ((Emax * C_plasma_obs) / (EC50 + C_plasma_obs));
```

**After Fix**: Generated Stan code correctly uses the inlined expression:
```stan
real E_drug = ((Emax * C_plasma) / (EC50 + C_plasma));
```

## Root Cause

The `lower_composite_model()` function in `compiler/src/lower.rs` was collecting connection information but not applying the substitutions to IR expressions. Specifically:

1. Connections were recorded in `input_map`
2. But input variables in ODEs, intermediates, and observables were never replaced
3. Additionally, when connecting to observables, the observable name was used instead of inlining its expression

## Solution Implementation

### 1. Added Expression Substitution (Step 4)

After collecting connections, added explicit substitution pass for all IR expressions:

```rust
// Step 4: Substitute input references in ODEs and intermediates
for ode in &mut all_odes {
    ode.rhs = substitute_inputs(&ode.rhs, &input_map, &observable_expr_map);
}

for intermediate in &mut all_intermediates {
    intermediate.expr = substitute_inputs(&intermediate.expr, &input_map, &observable_expr_map);
}

for observable in &mut all_observables {
    observable.expr = substitute_inputs(&observable.expr, &input_map, &observable_expr_map);
}
```

### 2. Implemented Observable Resolution

Created `observable_expr_map` to track observable expressions and inline them when referenced:

```rust
let mut observable_expr_map: HashMap<String, IRExpr> = HashMap::new();

// When processing observables:
ModelItem::Observable(obs) => {
    let obs_expr = lower_expr(&obs.expr);
    observable_expr_map.insert(obs.name.clone(), obs_expr.clone());
    all_observables.push(IRObservable {
        name: obs.name.clone(),
        dimension: type_expr_to_dimension(&obs.ty),
        expr: obs_expr,
    });
}
```

### 3. Created Recursive Substitution Function

Implemented `substitute_inputs()` that:
- Walks IR expression trees recursively
- Checks if variables match input declarations
- Resolves observables to their underlying expressions
- Handles all IR expression variants (Var, Literal, Unary, Binary, Index, Call)

```rust
fn substitute_inputs(
    expr: &IRExpr,
    input_map: &HashMap<String, (String, String)>,
    observable_expr_map: &HashMap<String, IRExpr>,
) -> IRExpr {
    match expr {
        IRExpr::Var(name) => {
            for (input_key, (_from_model, from_field)) in input_map {
                if let Some(field_name) = input_key.split('.').nth(1) {
                    if field_name == name {
                        // Check if connected field is an observable
                        if let Some(obs_expr) = observable_expr_map.get(from_field) {
                            return obs_expr.clone(); // Inline observable expression
                        } else {
                            return IRExpr::Var(from_field.clone());
                        }
                    }
                }
            }
            IRExpr::Var(name.clone())
        }
        // ... recursive handling for all other expression types
    }
}
```

## Files Modified

### `compiler/src/lower.rs`

1. **Line ~257**: Added `observable_expr_map` to track observable expressions
2. **Line ~297**: Store observable expressions when processing submodels
3. **Line ~180-237**: Implemented `substitute_inputs()` function with observable resolution
4. **Line ~338-347**: Added Step 4 to apply substitutions to all IR expressions

### `compiler/tests/parser_integration.rs`

- **Line 35**: Added `Declaration::Protocol(_) => {}` to handle Week 8 Protocol variant

### `compiler/tests/composite_model_test.rs` (New file)

- Created comprehensive test to verify input substitution works correctly
- Test validates that `C_drug` is replaced with `C_plasma` (not `C_plasma_obs`)

## Verification

### Test Results

```bash
$ cargo test composite_model_input_substitution --release
test test_composite_model_input_substitution ... ok
✓ Composite model input substitution test passed
  E_drug correctly uses C_plasma instead of C_drug
```

### Generated Code Validation

**Input**: `docs/examples/test_composite_minimal.medlang`

**Output**: `docs/examples/test_composite_minimal.stan`

Key section showing correct substitution:
```stan
// Intermediate values
real C_plasma = (A_plasma / V);
real E_drug = ((Emax * C_plasma) / (EC50 + C_plasma));  // ✓ Uses C_plasma, not C_plasma_obs
```

### All Tests Passing

```bash
$ cargo test --release
test result: ok. 55 passed; 0 failed; 0 ignored; 0 measured
```

## Impact

This fix completes Week 7's composite model architecture, enabling:

1. **Clean Model Composition**: PBPK and QSP models can be properly composed
2. **Correct Code Generation**: Stan and Julia backends generate valid code without undefined variables
3. **Observable Inlining**: When connecting to observables, their expressions are properly inlined
4. **Foundation for Protocol DSL**: Week 8's clinical protocol layer can now build on solid composite models

## Estimated Effort

- **Planned**: 3-4 hours
- **Actual**: ~3 hours (implementation + testing)

## Status

✅ **COMPLETE** - Week 7 composite model connection resolution is now fully functional.

## Next Steps

With Week 7 complete, the remaining incomplete features are:

1. **Week 8 Protocol DSL** (28-37 hours):
   - Parser implementation
   - IR layer
   - Endpoint evaluation
   - CLI integration

2. **Week 9 Extensions** (36-46 hours):
   - PFS/OS survival analysis
   - FHIR/CQL export
   - Bayesian power analysis

Total remaining effort: 64-83 hours
