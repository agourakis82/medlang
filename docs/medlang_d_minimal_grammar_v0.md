# MedLang-D Minimal Grammar v0.1

**Purpose:** Grammar specification for Vertical Slice 0 (1-compartment oral PK with NLME)  
**Notation:** EBNF-like syntax  
**Status:** V0 implementation target

---

## Lexical Elements

### Keywords
```
model population measure timeline cohort
state param obs rand input
dA_dt use_measure bind_params at dose observe to
```

### Identifiers
```
IDENT ::= [a-zA-Z_][a-zA-Z0-9_]*
```

### Literals
```
FLOAT_LIT ::= [0-9]+ ('.' [0-9]+)? ([eE] [+-]? [0-9]+)?
UNIT_LIT  ::= FLOAT_LIT '_' UNIT_SUFFIX
UNIT_SUFFIX ::= 'mg' | 'L' | 'h' | 'kg' | ...

STRING_LIT ::= '"' [^"]* '"'
```

### Operators
```
+ - * / ^ = ~ < > ( ) { } [ ] , ; : . |
```

### Comments
```
// Single-line comment
/* Multi-line comment */
```

---

## Top-Level Constructs

### Program
```ebnf
program ::= declaration*

declaration ::=
    | model_def
    | population_def
    | measure_def
    | timeline_def
    | cohort_def
```

---

## Model Definition

```ebnf
model_def ::= 'model' IDENT '{' model_item* '}'

model_item ::=
    | state_decl
    | param_decl
    | ode_equation
    | observable_decl

state_decl ::= 'state' IDENT ':' type_expr

param_decl ::= 'param' IDENT ':' type_expr

ode_equation ::= 'd' IDENT '/' 'dt' '=' expr

observable_decl ::= 'obs' IDENT ':' type_expr '=' expr
```

### Example
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
```

---

## Population Definition

```ebnf
population_def ::= 'population' IDENT '{' population_item* '}'

population_item ::=
    | 'model' IDENT                    // Reference to model
    | param_decl                       // Population parameter
    | input_decl                       // Covariate
    | random_effect_decl               // IIV
    | bind_params_block                // Individual parameter mapping
    | use_measure_stmt                 // Error model binding

input_decl ::= 'input' IDENT ':' type_expr

random_effect_decl ::= 'rand' IDENT ':' type_expr '~' distribution_expr

distribution_expr ::= 
    | 'Normal' '(' expr ',' expr ')'
    | 'LogNormal' '(' expr ',' expr ')'
    | 'Uniform' '(' expr ',' expr ')'

bind_params_block ::= 'bind_params' '(' IDENT ')' '{' statement* '}'

use_measure_stmt ::= 'use_measure' IDENT 'for' qualified_name
```

### Example
```medlang
population OneCompOralPop {
    model OneCompOral
    
    // Population parameters
    param CL_pop  : Clearance
    param V_pop   : Volume
    param Ka_pop  : RateConst
    param omega_CL : f64
    param omega_V  : f64
    param omega_Ka : f64
    
    // Covariate
    input WT : Quantity<kg, f64>
    
    // Random effects (IIV)
    rand eta_CL : f64 ~ Normal(0.0, omega_CL)
    rand eta_V  : f64 ~ Normal(0.0, omega_V)
    rand eta_Ka : f64 ~ Normal(0.0, omega_Ka)
    
    // Individual parameter mapping
    bind_params(patient) {
        let w = patient.WT / 70.0_kg
        model.CL = CL_pop * pow(w, 0.75) * exp(eta_CL)
        model.V  = V_pop * w * exp(eta_V)
        model.Ka = Ka_pop * exp(eta_Ka)
    }
    
    use_measure ConcPropError for model.C_plasma
}
```

---

## Measure Definition

```ebnf
measure_def ::= 'measure' IDENT '{' measure_item* '}'

measure_item ::=
    | 'pred' ':' type_expr
    | 'obs'  ':' type_expr
    | param_decl
    | loglik_stmt

loglik_stmt ::= 'log_likelihood' '=' expr
```

### Example
```medlang
measure ConcPropError {
    pred : ConcMass
    obs  : ConcMass
    param sigma_prop : f64
    
    log_likelihood = Normal_logpdf(
        x  = (obs / pred) - 1.0,
        mu = 0.0,
        sd = sigma_prop
    )
}
```

---

## Timeline Definition

```ebnf
timeline_def ::= 'timeline' IDENT '{' event* '}'

event ::=
    | dose_event
    | observe_event

dose_event ::= 'at' time_expr ':' 'dose' dose_spec

dose_spec ::= '{' dose_field* '}'

dose_field ::=
    | 'amount' '=' expr
    | 'to' '=' qualified_name

observe_event ::= 'at' time_expr ':' 'observe' qualified_name

time_expr ::= expr  // Must have Time units
```

### Example
```medlang
timeline OneCompOralTimeline {
    at 0.0_h:
        dose { amount = 100.0_mg; to = OneCompOral.A_gut }
    
    at 1.0_h:  observe OneCompOral.C_plasma
    at 2.0_h:  observe OneCompOral.C_plasma
    at 4.0_h:  observe OneCompOral.C_plasma
    at 8.0_h:  observe OneCompOral.C_plasma
}
```

---

## Cohort Definition

```ebnf
cohort_def ::= 'cohort' IDENT '{' cohort_item* '}'

cohort_item ::=
    | 'population' IDENT
    | 'timeline'   IDENT
    | 'data_file'  STRING_LIT
```

### Example
```medlang
cohort OneCompCohort {
    population OneCompOralPop
    timeline   OneCompOralTimeline
    data_file  "data/onecomp_synth.csv"
}
```

---

## Type Expressions

```ebnf
type_expr ::=
    | IDENT                                    // Simple type (e.g., f64)
    | unit_type                                // Built-in unit type
    | 'Quantity' '<' unit_expr ',' type_expr '>'  // Generic quantity

unit_type ::=
    | 'Mass' | 'Volume' | 'Time'               // Base units
    | 'DoseMass' | 'ConcMass'                  // Derived units
    | 'Clearance' | 'RateConst'                // Pharmacokinetic units

unit_expr ::=
    | IDENT                                    // Unit name (e.g., kg)
    | unit_expr '*' unit_expr                  // Product
    | unit_expr '/' unit_expr                  // Quotient
```

### Built-in Unit Types (V0)

| Type | Definition | Example |
|------|------------|---------|
| `Mass` | Base unit | `mg`, `g`, `kg` |
| `Volume` | Base unit | `mL`, `L` |
| `Time` | Base unit | `h`, `min`, `day` |
| `DoseMass` | Alias for `Mass` | `100.0_mg` |
| `ConcMass` | `Mass / Volume` | `5.0_mg/L` |
| `Clearance` | `Volume / Time` | `10.0_L/h` |
| `RateConst` | `1 / Time` | `1.5_1/h` |

---

## Expressions

```ebnf
expr ::=
    | literal
    | IDENT
    | qualified_name
    | unary_op expr
    | expr binary_op expr
    | function_call
    | '(' expr ')'

qualified_name ::= IDENT ('.' IDENT)*

literal ::=
    | FLOAT_LIT
    | UNIT_LIT

unary_op ::= '-' | '+'

binary_op ::=
    | '+' | '-' | '*' | '/' | '^'   // Arithmetic
    | '<' | '>' | '==' | '!='       // Comparison

function_call ::= IDENT '(' argument_list? ')'

argument_list ::= argument (',' argument)*

argument ::=
    | expr                          // Positional
    | IDENT '=' expr                // Named
```

### Built-in Functions (V0)

| Function | Signature | Description |
|----------|-----------|-------------|
| `exp(x: f64)` | `f64 -> f64` | Exponential (x must be dimensionless) |
| `log(x: f64)` | `f64 -> f64` | Natural log (x must be dimensionless) |
| `pow(x: f64, y: f64)` | `(f64, f64) -> f64` | Power (both dimensionless) |
| `sqrt(x: f64)` | `f64 -> f64` | Square root (x dimensionless) |
| `Normal_logpdf(...)` | Named args | Log-PDF of normal distribution |

---

## Statements

```ebnf
statement ::=
    | let_stmt
    | assign_stmt
    | expr_stmt

let_stmt ::= 'let' IDENT '=' expr

assign_stmt ::= qualified_name '=' expr

expr_stmt ::= expr
```

---

## Complete Grammar Summary (EBNF)

```ebnf
(* Top-level *)
program ::= declaration*
declaration ::= model_def | population_def | measure_def | timeline_def | cohort_def

(* Model *)
model_def ::= 'model' IDENT '{' model_item* '}'
model_item ::= state_decl | param_decl | ode_equation | observable_decl
state_decl ::= 'state' IDENT ':' type_expr
param_decl ::= 'param' IDENT ':' type_expr
ode_equation ::= 'd' IDENT '/' 'dt' '=' expr
observable_decl ::= 'obs' IDENT ':' type_expr '=' expr

(* Population *)
population_def ::= 'population' IDENT '{' population_item* '}'
population_item ::= 'model' IDENT | param_decl | input_decl | random_effect_decl 
                    | bind_params_block | use_measure_stmt
input_decl ::= 'input' IDENT ':' type_expr
random_effect_decl ::= 'rand' IDENT ':' type_expr '~' distribution_expr
bind_params_block ::= 'bind_params' '(' IDENT ')' '{' statement* '}'
use_measure_stmt ::= 'use_measure' IDENT 'for' qualified_name

(* Measure *)
measure_def ::= 'measure' IDENT '{' measure_item* '}'
measure_item ::= 'pred' ':' type_expr | 'obs' ':' type_expr | param_decl | loglik_stmt
loglik_stmt ::= 'log_likelihood' '=' expr

(* Timeline *)
timeline_def ::= 'timeline' IDENT '{' event* '}'
event ::= dose_event | observe_event
dose_event ::= 'at' time_expr ':' 'dose' dose_spec
dose_spec ::= '{' dose_field* '}'
dose_field ::= 'amount' '=' expr | 'to' '=' qualified_name
observe_event ::= 'at' time_expr ':' 'observe' qualified_name

(* Cohort *)
cohort_def ::= 'cohort' IDENT '{' cohort_item* '}'
cohort_item ::= 'population' IDENT | 'timeline' IDENT | 'data_file' STRING_LIT

(* Types *)
type_expr ::= IDENT | unit_type | 'Quantity' '<' unit_expr ',' type_expr '>'
unit_type ::= 'Mass' | 'Volume' | 'Time' | 'DoseMass' | 'ConcMass' | 'Clearance' | 'RateConst'

(* Expressions *)
expr ::= literal | IDENT | qualified_name | unary_op expr | expr binary_op expr 
         | function_call | '(' expr ')'
literal ::= FLOAT_LIT | UNIT_LIT
function_call ::= IDENT '(' argument_list? ')'
argument_list ::= argument (',' argument)*
argument ::= expr | IDENT '=' expr

(* Statements *)
statement ::= let_stmt | assign_stmt | expr_stmt
let_stmt ::= 'let' IDENT '=' expr
assign_stmt ::= qualified_name '=' expr
expr_stmt ::= expr
```

---

## Parsing Strategy

### Recommended Approach: Recursive Descent

1. **Tokenization** (using `logos` crate):
   - Keywords, identifiers, literals, operators
   - Preserve source locations for error messages

2. **Parser** (using `nom` combinators or hand-rolled):
   - Top-down recursive descent
   - Operator precedence for expressions
   - Error recovery at statement boundaries

3. **Error Handling**:
   - Informative messages with line/column
   - Context-aware suggestions
   - Partial AST construction for IDE support

---

## Type Checking Strategy

### Unit Inference Algorithm

1. **Assign units to literals:**
   - `100.0_mg` → `Quantity<mg, f64>`
   - `0.0` → `f64` (dimensionless)

2. **Assign units to variables:**
   - From declarations: `state A_gut : DoseMass` → `A_gut : Mass`
   - From type annotations: `param CL : Clearance` → `CL : Volume/Time`

3. **Infer units through expressions:**
   - `Ka * A_gut` → `(1/Time) * Mass = Mass/Time` ✓
   - `CL / V` → `(Volume/Time) / Volume = 1/Time` ✓
   - `A_central / V` → `Mass / Volume = ConcMass` ✓

4. **Check ODE constraints:**
   - `dA_gut/dt` must have units `Mass/Time`
   - RHS expression must match

5. **Check function constraints:**
   - `exp(x)`, `log(x)` require `x : f64` (dimensionless)
   - `pow(x, y)` requires both dimensionless

---

## Validation Rules

### Semantic Checks

1. **Name resolution:**
   - All referenced models/populations/measures must be defined
   - All state/param names must be unique within model
   - Qualified names must resolve correctly

2. **Type consistency:**
   - Observable units must match measure expectations
   - Dose amounts must have Mass units
   - Observation times must have Time units

3. **Population structure:**
   - `bind_params` must assign all model parameters
   - Random effects must be referenced in bind_params
   - Inputs (covariates) must be used

4. **Timeline validity:**
   - Dose targets must be states
   - Observe targets must be observables
   - Times must be non-negative and ordered

---

## Extensions for Future Versions

**Not in V0, but planned:**

- Multi-compartment models
- PBPK constructs (`compartment`, `flow`)
- QSP constructs (`species`, `reaction`)
- ML submodels (`mlmodel`, `train`)
- Track C operators (`QM_BindingFreeEnergy`, etc.)
- General function definitions
- Control flow (if/then/else, for loops)
- Matrix/vector operations

---

**End of Grammar Specification**

*This grammar defines the minimal surface syntax for MedLang V0. Implementation should follow the Rust parser structure outlined in PROMPT_V0_BASIC_COMPILER.md.*
