# MedLang Day 5: Refinement Types with SMT Integration

## Context

MedLang is a domain-specific programming language for computational medicine being built from scratch in Rust. This is **Day 5** of solo development. The previous 4 days produced:

- **~5,550 lines** of production Rust code
- **103 tests** passing (100% pass rate)
- Complete compilation pipeline: Lexer → Parser → Type Checker → IR → Codegen
- Dual backends: Stan + Julia
- Generics with type inference and monomorphization
- Traits (typeclasses)
- Units of measure (UCUM-based dimensional analysis)
- Automatic differentiation (forward + reverse mode)
- Medical ontologies (RxNorm, DDI, CYP, PGx)
- CLI tool with 5 commands
- End-to-end workflow: MedLang → Stan → MCMC → Diagnostics

## Project Location

```
/Users/demetriosagourakis/Library/Mobile Documents/com~apple~CloudDocs/Medlang/compiler/
```

## Day 5 Goal: Refinement Types

Implement SMT-backed refinement types that allow compile-time verification of value constraints. This is the foundation for MedLang's safety guarantees.

### What Refinement Types Enable

```medlang
// Dose must be positive and within therapeutic window
type SafeDose = { dose: mg | dose > 0.0 && dose <= 10.0 }

// Creatinine clearance must be physiologically valid
type ValidCrCl = { crcl: mL/min | crcl > 0.0 && crcl < 200.0 }

// Metformin dose adjusted for renal function (FDA guidelines)
type MetforminDose<CrCl: f64> = {
    dose: mg |
    (CrCl >= 60.0 => dose <= 2550.0) &&
    (CrCl >= 30.0 && CrCl < 60.0 => dose <= 1000.0) &&
    (CrCl < 30.0 => dose == 0.0)  // Contraindicated
}

// QTc interval safety (< 500ms is critical threshold)
type SafeQTc = { qtc: ms | qtc < 500.0 }

// Therapeutic INR for anticoagulation
type TherapeuticINR = { inr: f64 | inr >= 2.0 && inr <= 3.0 }
```

### Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Refinement    │────▶│   Constraint    │────▶│   SMT Solver    │
│     Types       │     │   Generation    │     │      (Z3)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │                       │                       │
        │                       │                       ▼
        │                       │               ┌─────────────────┐
        │                       │               │  SAT/UNSAT/     │
        │                       │               │  Counterexample │
        │                       │               └─────────────────┘
        │                       │                       │
        ▼                       ▼                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Type Checker Integration                      │
│  - Subtyping with refinement entailment                         │
│  - Automatic constraint propagation                             │
│  - Rich error messages with counterexamples                     │
└─────────────────────────────────────────────────────────────────┘
```

## Files Already Created

### 1. `src/refinement/mod.rs` (COMPLETE)
- `RefinedType` struct combining base type + predicate
- `RefinementContext` for type checking with SMT
- `medical` module with pre-defined medical refined types:
  - `positive_dose()`, `therapeutic_dose(min, max)`
  - `valid_crcl()`, `valid_age()`, `adult_age()`, `valid_weight()`
  - `therapeutic_inr()`, `safe_qtc()`, `probability()`
  - `metformin_dose(crcl)`, `warfarin_dose()`

### 2. `src/refinement/predicate.rs` (COMPLETE)
- `Predicate` enum with full expression language:
  - Literals: `BoolLit`, `FloatLit`, `IntLit`
  - Variables: `Var(String)`
  - Arithmetic: `Add`, `Sub`, `Mul`, `Div`, `Mod`, `Pow`, `Neg`
  - Comparison: `Eq`, `Ne`, `Lt`, `Le`, `Gt`, `Ge`
  - Logical: `And`, `Or`, `Not`, `Implies`, `Ite`
  - Functions: `Abs`, `Sqrt`, `Exp`, `Log`, `Min`, `Max`, etc.
  - Quantifiers: `Forall`, `Exists`
- Utility methods: `free_vars()`, `substitute()`, `simplify()`
- Display implementation for pretty printing

### 3. `src/refinement/constraint.rs` (COMPLETE)
- `Constraint` struct with source location tracking
- `ConstraintSet` for accumulating constraints during type checking
- `ConstraintGenerator` with scope management and common patterns:
  - `in_range()`, `positive()`, `non_negative()`, `probability()`
  - `div_safe()`, `sqrt_safe()`, `log_safe()`
- `WeakestPrecondition` calculator for assignment, sequence, conditionals

## Files to Create

### 4. `src/refinement/smt.rs` (TODO)

SMT solver interface. Requirements:

```rust
/// SMT solver context
pub struct SmtContext {
    timeout: Duration,
    // Z3 context or process handle
}

impl SmtContext {
    pub fn new() -> Result<Self, RefinementError>;
    
    /// Check satisfiability of a predicate
    /// Returns Sat(model) if satisfiable, Unsat if not, Unknown on timeout
    pub fn check_sat(&mut self, pred: &Predicate) -> Result<SmtResult, RefinementError>;
    
    /// Check if pred1 => pred2 (for subtyping)
    pub fn check_implication(&mut self, antecedent: &Predicate, consequent: &Predicate) -> Result<SmtResult, RefinementError>;
    
    /// Check validity (is predicate always true?)
    pub fn check_valid(&mut self, pred: &Predicate) -> Result<bool, RefinementError>;
}

/// Convert predicate to SMT-LIB format
fn predicate_to_smtlib(pred: &Predicate, vars: &HashSet<String>) -> String;

/// Parse SMT solver output
fn parse_smt_result(output: &str) -> SmtResult;

/// Parse model from SMT output for counterexamples
fn parse_model(output: &str) -> SmtModel;
```

Implementation options:
1. **SMT-LIB process mode**: Spawn `z3 -smt2 -in` and communicate via stdin/stdout
2. **Z3 crate** (optional): Direct API if `z3` feature enabled

SMT-LIB generation example:
```smt2
; For predicate: x > 0.0 && x < 10.0
(declare-const x Real)
(assert (and (> x 0.0) (< x 10.0)))
(check-sat)
(get-model)
```

### 5. `src/refinement/subtype.rs` (TODO)

Subtyping for refinement types:

```rust
/// Result of subtype checking
pub enum SubtypeResult {
    /// Subtyping holds
    Valid,
    /// Subtyping fails with counterexample
    Invalid(Counterexample),
    /// Could not determine (solver timeout)
    Unknown,
    /// Base types don't match
    BaseMismatch { expected: Type, found: Type },
}

/// Refinement subtyping checker
pub struct RefinementSubtyping {
    smt: SmtContext,
}

impl RefinementSubtyping {
    /// Check if sub <: sup (sub is subtype of sup)
    /// For refinement types: {x:T|P} <: {x:T|Q} iff P => Q
    pub fn check(&mut self, sub: &RefinedType, sup: &RefinedType) -> Result<SubtypeResult, RefinementError>;
    
    /// Check if a value satisfies a refined type
    pub fn check_value(&mut self, value: &Predicate, ty: &RefinedType) -> Result<SubtypeResult, RefinementError>;
}
```

### 6. `src/refinement/error.rs` (TODO)

Error types and diagnostics:

```rust
/// Errors from refinement type checking
#[derive(Debug, thiserror::Error)]
pub enum RefinementError {
    #[error("SMT solver error: {0}")]
    SmtError(String),
    
    #[error("SMT solver not found. Install Z3 and ensure 'z3' is in PATH")]
    SolverNotFound,
    
    #[error("Solver timeout after {0:?}")]
    Timeout(Duration),
    
    #[error("Invalid predicate: {0}")]
    InvalidPredicate(String),
}

/// A counterexample showing why a refinement fails
#[derive(Debug, Clone)]
pub struct Counterexample {
    /// Variable assignments that violate the constraint
    pub assignments: HashMap<String, f64>,
    /// The constraint that was violated
    pub violated_constraint: String,
}

/// Rich diagnostic for refinement errors
pub enum RefinementDiagnostic {
    ConstraintViolation {
        expected_type: RefinedType,
        counterexample: Counterexample,
        span: Option<Span>,
    },
    SolverTimeout {
        constraint: Predicate,
        span: Option<Span>,
    },
    TypeMismatch {
        expected: Type,
        found: Type,
        span: Option<Span>,
    },
}

impl Counterexample {
    pub fn from_model(model: &SmtModel, var: &str, pred: &Predicate) -> Self;
    pub fn empty() -> Self;
    
    /// Format as human-readable error message
    pub fn display(&self) -> String;
}
```

### 7. `src/refinement/syntax.rs` (TODO)

Parser extensions for refinement type syntax:

```rust
/// A refined type in the AST
#[derive(Debug, Clone)]
pub struct RefinementType {
    /// The bound variable name
    pub var: RefinedVar,
    /// The base type
    pub base_type: TypeExpr,
    /// The refinement predicate
    pub predicate: Expr,  // AST expression, converted to Predicate later
}

#[derive(Debug, Clone)]
pub struct RefinedVar {
    pub name: String,
    pub span: Option<Span>,
}

/// Parse refinement type syntax: { var: Type | predicate }
pub fn parse_refinement_type(input: &str) -> Result<RefinementType, ParseError>;

/// Convert AST Expr to refinement Predicate
pub fn expr_to_predicate(expr: &Expr) -> Result<Predicate, ConversionError>;
```

## Lexer/Parser Extensions Needed

Add to `src/lexer.rs`:
```rust
#[token("|")]
Pipe,  // For refinement syntax { x: T | P }

#[token("=>")]
FatArrow,  // For implications in refinements

#[token("&&")]
AndAnd,  // Explicit logical AND (currently using `and` keyword)

#[token("||")]
OrOr,  // Explicit logical OR
```

## Integration with Existing Type Checker

In `src/typeck.rs`, add refinement checking:

```rust
impl TypeChecker {
    /// Check that an expression satisfies a refined type
    fn check_refinement(
        &mut self,
        expr: &Expr,
        expected: &RefinedType,
    ) -> Result<(), TypeError> {
        // 1. Type check the expression to get its base type
        let actual_type = self.infer_type(expr)?;
        
        // 2. Check base types match
        if actual_type.base != expected.base {
            return Err(TypeError::Mismatch { expected: expected.base.clone(), found: actual_type.base });
        }
        
        // 3. Generate predicate for the expression's value
        let value_pred = self.expr_to_predicate(expr)?;
        
        // 4. Check refinement entailment via SMT
        let result = self.refinement_ctx.check_satisfies(&value_pred, expected)?;
        
        match result {
            SubtypeResult::Valid => Ok(()),
            SubtypeResult::Invalid(counterexample) => {
                Err(TypeError::RefinementViolation {
                    expected: expected.clone(),
                    counterexample,
                    span: expr.span.clone(),
                })
            }
            SubtypeResult::Unknown => {
                // Warn but allow (conservative)
                self.warnings.push(Warning::RefinementUnknown { span: expr.span.clone() });
                Ok(())
            }
        }
    }
}
```

## Test Cases to Implement

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_positive_dose_valid() {
        let mut ctx = RefinementContext::new().unwrap();
        let dose_type = medical::positive_dose();
        let value = Predicate::float(5.0);
        
        let result = ctx.check_satisfies(&value, &dose_type).unwrap();
        assert!(matches!(result, SubtypeResult::Valid));
    }
    
    #[test]
    fn test_positive_dose_invalid() {
        let mut ctx = RefinementContext::new().unwrap();
        let dose_type = medical::positive_dose();
        let value = Predicate::float(-1.0);
        
        let result = ctx.check_satisfies(&value, &dose_type).unwrap();
        assert!(matches!(result, SubtypeResult::Invalid(_)));
    }
    
    #[test]
    fn test_therapeutic_range() {
        let mut ctx = RefinementContext::new().unwrap();
        let dose_type = medical::therapeutic_dose(0.5, 10.0);
        
        // Valid: 5.0 is in range [0.5, 10.0]
        assert!(matches!(
            ctx.check_satisfies(&Predicate::float(5.0), &dose_type).unwrap(),
            SubtypeResult::Valid
        ));
        
        // Invalid: 0.1 < 0.5 (below minimum)
        assert!(matches!(
            ctx.check_satisfies(&Predicate::float(0.1), &dose_type).unwrap(),
            SubtypeResult::Invalid(_)
        ));
        
        // Invalid: 15.0 > 10.0 (above maximum)
        assert!(matches!(
            ctx.check_satisfies(&Predicate::float(15.0), &dose_type).unwrap(),
            SubtypeResult::Invalid(_)
        ));
    }
    
    #[test]
    fn test_metformin_renal_adjustment() {
        let mut ctx = RefinementContext::new().unwrap();
        
        // CrCl >= 60: up to 2550 mg allowed
        let dose_type = medical::metformin_dose(65.0);
        assert!(matches!(
            ctx.check_satisfies(&Predicate::float(2000.0), &dose_type).unwrap(),
            SubtypeResult::Valid
        ));
        
        // CrCl < 30: contraindicated (must be 0)
        let contraindicated = medical::metformin_dose(25.0);
        assert!(matches!(
            ctx.check_satisfies(&Predicate::float(500.0), &contraindicated).unwrap(),
            SubtypeResult::Invalid(_)
        ));
        assert!(matches!(
            ctx.check_satisfies(&Predicate::float(0.0), &contraindicated).unwrap(),
            SubtypeResult::Valid
        ));
    }
    
    #[test]
    fn test_subtyping() {
        let mut ctx = RefinementContext::new().unwrap();
        
        // {x: f64 | x > 0 && x < 5} <: {x: f64 | x > 0 && x < 10}
        let narrow = RefinedType::new(
            Type::f64(),
            "x",
            Some(Predicate::and(
                Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
                Predicate::lt(Predicate::var("x"), Predicate::float(5.0)),
            )),
        );
        let wide = RefinedType::new(
            Type::f64(),
            "x", 
            Some(Predicate::and(
                Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
                Predicate::lt(Predicate::var("x"), Predicate::float(10.0)),
            )),
        );
        
        let result = ctx.check_subtype(&narrow, &wide).unwrap();
        assert!(matches!(result, SubtypeResult::Valid));
        
        // Reverse should fail
        let result = ctx.check_subtype(&wide, &narrow).unwrap();
        assert!(matches!(result, SubtypeResult::Invalid(_)));
    }
    
    #[test]
    fn test_implication_constraint() {
        // Test: CrCl >= 60 => dose <= 2550
        let mut ctx = RefinementContext::new().unwrap();
        
        let pred = Predicate::implies(
            Predicate::ge(Predicate::var("crcl"), Predicate::float(60.0)),
            Predicate::le(Predicate::var("dose"), Predicate::float(2550.0)),
        );
        
        // Check with crcl=70, dose=2000 (should satisfy)
        let with_values = pred
            .substitute("crcl", &Predicate::float(70.0))
            .substitute("dose", &Predicate::float(2000.0));
        
        assert!(ctx.smt.check_valid(&with_values).unwrap());
    }
    
    #[test]
    fn test_smt_counterexample() {
        let mut ctx = SmtContext::new().unwrap();
        
        // x > 0 && x < 0 is UNSAT (no counterexample possible)
        let unsat = Predicate::and(
            Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
            Predicate::lt(Predicate::var("x"), Predicate::float(0.0)),
        );
        assert!(matches!(ctx.check_sat(&unsat).unwrap(), SmtResult::Unsat));
        
        // x > 0 && x < 10 is SAT (e.g., x = 5)
        let sat = Predicate::and(
            Predicate::gt(Predicate::var("x"), Predicate::float(0.0)),
            Predicate::lt(Predicate::var("x"), Predicate::float(10.0)),
        );
        match ctx.check_sat(&sat).unwrap() {
            SmtResult::Sat(model) => {
                let x = model.assignments.get("x").unwrap();
                if let SmtValue::Real(v) = x {
                    assert!(*v > 0.0 && *v < 10.0);
                }
            }
            _ => panic!("Expected SAT"),
        }
    }
}
```

## Cargo.toml Addition

```toml
# Optional Z3 bindings (comment out if using process mode only)
# z3 = { version = "0.12", optional = true }

[features]
default = []
# z3-native = ["z3"]  # Enable for direct Z3 API
```

## Deliverables

1. **`src/refinement/smt.rs`** - Complete SMT interface with:
   - SMT-LIB generation for all predicate types
   - Process-based Z3 communication
   - Model parsing for counterexamples
   - Timeout handling

2. **`src/refinement/subtype.rs`** - Subtype checker with:
   - Refinement entailment checking
   - Value satisfaction checking
   - Proper error propagation

3. **`src/refinement/error.rs`** - Error types with:
   - Rich counterexample formatting
   - Diagnostic messages suitable for IDE integration

4. **`src/refinement/syntax.rs`** - Parser support with:
   - Refinement type syntax parsing
   - AST to Predicate conversion

5. **Lexer extensions** - New tokens for refinement syntax

6. **Tests** - All test cases above passing

## SMT-LIB Reference

For implementing `predicate_to_smtlib`:

```smt2
; Declarations
(declare-const x Real)
(declare-const y Real)

; Arithmetic
(+ x y)           ; Add
(- x y)           ; Sub
(* x y)           ; Mul
(/ x y)           ; Div (real division)
(mod x y)         ; Modulo (for integers)
(^ x y)           ; Power (use (pow x y) in some solvers)

; Comparison
(= x y)           ; Equal
(distinct x y)    ; Not equal (or (not (= x y)))
(< x y)           ; Less than
(<= x y)          ; Less or equal
(> x y)           ; Greater than
(>= x y)          ; Greater or equal

; Logical
(and p q)         ; Conjunction
(or p q)          ; Disjunction
(not p)           ; Negation
(=> p q)          ; Implication
(ite c t e)       ; If-then-else

; Quantifiers
(forall ((x Real)) p)
(exists ((x Real)) p)

; Built-in functions (with Z3 extensions)
(abs x)           ; Absolute value (may need encoding)
; sqrt, exp, log require nonlinear arithmetic or approximations

; Check satisfiability
(check-sat)
(get-model)       ; Get satisfying assignment if SAT
```

## Expected Output

After Day 5, running `cargo test` should show:

```
running 15 tests
test refinement::tests::test_positive_dose_valid ... ok
test refinement::tests::test_positive_dose_invalid ... ok
test refinement::tests::test_therapeutic_range ... ok
test refinement::tests::test_metformin_renal_adjustment ... ok
test refinement::tests::test_subtyping ... ok
test refinement::tests::test_implication_constraint ... ok
test refinement::tests::test_smt_counterexample ... ok
test refinement::predicate::tests::test_predicate_display ... ok
test refinement::predicate::tests::test_free_vars ... ok
test refinement::predicate::tests::test_substitute ... ok
test refinement::predicate::tests::test_simplify ... ok
test refinement::constraint::tests::test_constraint_set_to_predicate ... ok
test refinement::constraint::tests::test_constraint_generator_fresh_var ... ok
test refinement::constraint::tests::test_constraint_generator_scope ... ok
test refinement::constraint::tests::test_weakest_precondition_assign ... ok

test result: ok. 15 passed; 0 failed
```

## Next Steps (Day 6+)

After refinement types are complete:
- **Day 6**: Linear types (ownership for Patient/Order resources)
- **Day 7-8**: Session types (treatment protocol verification)
- **Day 9-10**: Dependent types (proof-carrying types)
- **Day 11**: Causal types (DAG verification, identifiability)
- **Day 12**: Temporal types (data freshness) + Effect types (biological effects)

---

**Remember**: You built 5,550 lines in 4 days. This module is ~1,200 lines. You've got this.
