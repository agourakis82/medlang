# Week 27: Algebraic Data Types & Pattern Matching for Clinical States

**Status**: Core Infrastructure Complete (Parser & Codegen Integration Pending)

Week 27 extends MedLang with **algebraic data types (enums)** and **pattern matching**, enabling the type system to speak in discrete clinical concepts rather than just numbers.

## Motivation

Clinical medicine operates on discrete states as much as continuous measurements:
- **Response categories**: CR (Complete Response), PR (Partial Response), SD (Stable Disease), PD (Progressive Disease)
- **Toxicity grades**: G0, G1, G2, G3, G4
- **Performance status**: ECOG 0-4
- **Disease stages**: IA, IB, IIA, IIB, etc.

Before Week 27, these would be represented as raw integers or strings with no type safety. Week 27 makes them **first-class types** with compile-time checking.

## Implementation Summary

### 1. Enum Declarations AST (`compiler/src/ast/enum_decl.rs` - 158 lines)

Created AST structures for enum declarations:

```rust
pub struct EnumDecl {
    pub name: Ident,
    pub variants: Vec<EnumVariant>,
}

pub struct EnumVariant {
    pub name: Ident,
    // Week 27: nullary variants only
    // Future: pub fields: Vec<(Ident, Type)>
}
```

**Example**:
```medlang
enum Response {
  CR;
  PR;
  SD;
  PD;
}

enum ToxicityGrade {
  G0;
  G1;
  G2;
  G3;
  G4;
}
```

**Key Features**:
- Nullary variants (no payloads) for Week 27
- Extensible to sum types with fields in future weeks
- Integrated with Declaration enum in AST

### 2. Pattern Matching AST (`compiler/src/ast/match_expr.rs` - 126 lines)

Added match expressions and patterns to ExprKind:

```rust
pub enum ExprKind {
    // ... existing variants
    
    /// Enum variant constructor: Response::CR
    EnumVariant {
        enum_name: String,
        variant_name: String,
    },
    
    /// Pattern matching on enums
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
    },
}

pub struct MatchArm {
    pub pattern: MatchPattern,
    pub body: Expr,
}

pub enum MatchPattern {
    Variant { enum_name: String, variant_name: String },
    Wildcard,
}
```

**Example**:
```medlang
match resp {
  Response::CR => 1.0,
  Response::PR => 0.7,
  Response::SD => 0.0,
  Response::PD => 0.0,
}

// Or with wildcard:
match resp {
  Response::CR => 1.0,
  Response::PR => 0.7,
  _            => 0.0,
}
```

### 3. Enum Type System (`compiler/src/types/enum_types.rs` - 273 lines)

Comprehensive type-level representation:

```rust
pub struct EnumType {
    pub name: String,
    pub num_variants: usize,
}

pub struct EnumInfo {
    pub name: String,
    pub variants: Vec<String>,
}

pub struct EnumEnv {
    pub enums: HashMap<String, EnumInfo>,
    pub variant_codes: HashMap<(String, String), usize>,
}
```

**Key Features**:
- `EnumInfo`: Complete enum metadata (name, variants, indices)
- `EnumEnv`: Environment tracking all enum declarations
- Integer encoding: Each variant gets a stable code (CR=0, PR=1, SD=2, PD=3)
- Variant validation and lookup methods

**Extended CoreType**:
```rust
pub enum CoreType {
    // ... primitive types
    Enum(String), // Week 27: enum types
    // ... domain types
}
```

### 4. Type Checking (`compiler/src/typecheck/enum_check.rs` - 303 lines)

Full type checker with exhaustiveness checking:

```rust
pub fn typecheck_enum_variant(
    enum_env: &EnumEnv,
    enum_name: &str,
    variant_name: &str,
) -> Result<CoreType, TypeError>

pub fn typecheck_match(
    env: &mut TypeEnv,
    fn_env: &FnEnv,
    enum_env: &EnumEnv,
    scrutinee: &Expr,
    arms: &[MatchArm],
) -> Result<CoreType, TypeError>
```

**Type Checking Rules**:

1. **Enum Variant Constructor**:
   - Enum must exist in `EnumEnv`
   - Variant must exist in that enum
   - Result type: `Enum(enum_name)`

2. **Match Expression**:
   - Scrutinee must have enum type
   - All arm patterns must reference the same enum
   - All arm bodies must have the same type
   - Exhaustiveness: All variants covered OR wildcard present

**New Error Types**:
```rust
UnknownEnum(String)
UnknownEnumVariant { enum_name, variant_name, message }
MatchNonEnum { found }
MatchEnumMismatch { scrutinee_enum, arm_enum }
MatchArmTypeMismatch { expected, found }
NonExhaustiveMatch { enum_name, missing_variants }
```

## Example Usage

### Defining Clinical Enums

```medlang
enum Response {
  CR;
  PR;
  SD;
  PD;
}

enum ToxicityGrade {
  G0;
  G1;
  G2;
  G3;
  G4;
}
```

### Using Enums in Functions

```medlang
fn response_score(resp: Response) -> real {
  match resp {
    Response::CR => 1.0,
    Response::PR => 0.7,
    Response::SD => 0.0,
    Response::PD => 0.0,
  }
}

fn is_responder(resp: Response) -> bool {
  match resp {
    Response::CR => true,
    Response::PR => true,
    _            => false,
  }
}
```

### Pattern Matching in Policies

```medlang
policy DoseAdjustmentPolicy {
  inputs {
    tox: ToxicityGrade;
    dose_mg: real;
  }
  
  action_expr = match tox {
    ToxicityGrade::G0 => dose_mg * 1.0,
    ToxicityGrade::G1 => dose_mg * 1.0,
    ToxicityGrade::G2 => dose_mg * 0.75,
    ToxicityGrade::G3 => dose_mg * 0.50,
    ToxicityGrade::G4 => 0.0,
  };
}
```

## Backend Representation

### Integer Encoding

Enums are represented as **integers** at runtime:
- Each variant gets a stable integer code based on declaration order
- Example: `Response::CR → 0, PR → 1, SD → 2, PD → 3`
- Stored in `EnumEnv.variant_codes` map

### Match Expression Lowering

Match expressions lower to nested `if/else` cascades:

**MedLang**:
```medlang
match resp {
  Response::CR => 1.0,
  Response::PR => 0.7,
  _            => 0.0,
}
```

**Stan** (planned):
```stan
{
  real _tmp;
  if (resp == 0) {
    _tmp = 1.0;
  } else if (resp == 1) {
    _tmp = 0.7;
  } else {
    _tmp = 0.0;
  }
  _tmp;
}
```

**Julia** (planned):
```julia
if resp == 0
    1.0
elseif resp == 1
    0.7
else
    0.0
end
```

## Testing Summary

**Unit Tests**: 21 tests across 3 modules
- `ast/enum_decl.rs`: 4 tests (enum construction, variant lookup, indexing)
- `types/enum_types.rs`: 8 tests (EnumInfo, EnumEnv, variant codes, validation)
- `typecheck/enum_check.rs`: 9 tests (variant typechecking, match typechecking, error cases)

**Test Coverage**:
- ✅ Enum declaration construction
- ✅ Variant index assignment (0-based)
- ✅ EnumEnv registration and lookup
- ✅ Variant code mapping
- ✅ Enum variant type checking
- ✅ Match expression type checking
- ✅ Exhaustiveness checking
- ✅ Error detection (unknown enum, unknown variant, type mismatches)

**Known Limitations** (Pending Work):
- Parser integration not yet implemented
- IR lowering with integer codes not yet wired
- Stan codegen for match expressions pending
- Julia codegen for match expressions pending
- `med.clinical` stdlib module not yet created

## Integration Points

### Parser (Pending)

Need to implement parsing for:

1. **Enum declarations**:
   ```medlang
   enum Response {
     CR;
     PR;
     SD;
     PD;
   }
   ```

2. **Enum variant expressions**: `Response::CR`

3. **Match expressions**:
   ```medlang
   match resp {
     Response::CR => 1.0,
     Response::PR => 0.7,
     _ => 0.0,
   }
   ```

### IR Lowering (Pending)

Extend IR to include:

```rust
pub enum IRExpr {
    // ...
    EnumConst {
        enum_name: String,
        variant_name: String,
        code: i32, // Integer encoding
    },
    MatchEnum {
        scrutinee: Box<IRExpr>,
        arms: Vec<IRMatchArm>,
        default: Option<Box<IRExpr>>,
    },
}

pub struct IRMatchArm {
    pub variant_code: i32,
    pub body: IRExpr,
}
```

### Stan Codegen (Pending)

Emit:
- Enums as `int` variables
- Match as nested `if/else` chains
- Helper functions for variant checking

### Julia Codegen (Pending)

Emit:
- Enums as `Int` constants
- Match as `if/elseif/else` expressions
- Optionally use Julia `@enum` for better ergonomics

### Standard Library: `med.clinical` (Pending)

Create canonical clinical enums:

```medlang
module med.clinical;

enum Response {
  CR;
  PR;
  SD;
  PD;
}

enum ToxicityGrade {
  G0;
  G1;
  G2;
  G3;
  G4;
}

enum ECOG {
  ECOG0;
  ECOG1;
  ECOG2;
  ECOG3;
  ECOG4;
}

// Helper functions
fn is_responder(resp: Response) -> bool {
  match resp {
    Response::CR => true,
    Response::PR => true,
    _ => false,
  }
}

fn response_from_rel_change(pct_change: real) -> Response {
  if pct_change <= -0.30 {
    Response::PR
  } else if pct_change < 0.20 {
    Response::SD
  } else {
    Response::PD
  }
}
```

## Error Messages

### Unknown Enum
```
error: unknown enum `UnknownResponse`
 --> protocol.med:10:5
  |
10|   let r: UnknownResponse = ...;
  |          ^^^^^^^^^^^^^^^
```

### Unknown Variant
```
error: unknown variant `XX` in enum `Response`
 --> protocol.med:12:15
  |
12|   Response::XX => 0.0,
  |             ^^ Enum Response does not have variant XX. Valid variants: [CR, PR, SD, PD]
```

### Match Non-Enum
```
error: match scrutinee must be an enum type, found Int
 --> protocol.med:15:9
  |
15|   match x {
  |         ^ expected enum type, found Int
```

### Non-Exhaustive Match
```
error: non-exhaustive match on enum `Response`: missing variants [SD, PD]
 --> protocol.med:20:3
  |
20|   match resp {
  |   ^^^^^ missing coverage for: SD, PD
  |
help: add wildcard arm: `_ => ...` or handle all variants
```

### Match Arm Type Mismatch
```
error: match arm type mismatch: expected Float, found Int
 --> protocol.med:25:19
  |
25|     Response::CR => 1,
  |                     ^ expected Float (from first arm), found Int
```

## File Inventory

### New Files
- `compiler/src/ast/enum_decl.rs` (158 lines) - Enum declaration AST
- `compiler/src/ast/match_expr.rs` (126 lines) - Match expression utilities
- `compiler/src/types/enum_types.rs` (273 lines) - Enum type system
- `compiler/src/typecheck/enum_check.rs` (303 lines) - Enum type checker
- `WEEK_27_ENUMS_AND_PATTERN_MATCHING.md` (this file)

### Modified Files
- `compiler/src/ast/mod.rs` - Added EnumDecl to Declaration, EnumVariant/Match to ExprKind
- `compiler/src/types/mod.rs` - Registered enum_types module
- `compiler/src/types/core_lang.rs` - Extended CoreType with Enum variant
- `compiler/src/typecheck/mod.rs` - Registered enum_check module
- `compiler/src/typecheck/core_lang.rs` - Added enum-related TypeError variants

## What Week 27 Gives You

After Week 27, MedLang can express clinical logic in **domain language**:

**Before Week 27** (numbers):
```medlang
fn check_response(shrink_pct: real) -> int {
  if shrink_pct <= -0.30 {
    1  // What does 1 mean?
  } else if shrink_pct < 0.20 {
    2
  } else {
    3
  }
}
```

**After Week 27** (clinical states):
```medlang
fn classify_response(shrink_pct: real) -> Response {
  if shrink_pct <= -0.30 {
    Response::PR  // Clear clinical meaning
  } else if shrink_pct < 0.20 {
    Response::SD
  } else {
    Response::PD
  }
}

fn dose_reduction_factor(tox: ToxicityGrade) -> real {
  match tox {
    ToxicityGrade::G0 => 1.0,
    ToxicityGrade::G1 => 1.0,
    ToxicityGrade::G2 => 0.75,
    ToxicityGrade::G3 => 0.50,
    ToxicityGrade::G4 => 0.0,
  }
}
```

The compiler prevents mixing up response categories with toxicity grades, ensures exhaustive handling of all clinical states, and generates efficient integer-based backend code.

**Week 27 Status**: ✅ Core Infrastructure Complete (AST, Types, Typechecking)

**Remaining Work**: Parser integration, IR lowering, Stan/Julia codegen, `med.clinical` stdlib

**Next Steps**: 
1. Implement parser for enum declarations and match expressions
2. Extend IR with EnumConst and MatchEnum
3. Implement Stan codegen for match (if/else cascades)
4. Implement Julia codegen for match
5. Create `med.clinical` standard library module with canonical enums
6. Write integration tests
