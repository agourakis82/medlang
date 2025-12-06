//! Refinement Types for MedLang
//!
//! Refinement types add logical predicates to base types, enabling compile-time
//! verification of value constraints. This is the foundation for MedLang's
//! safety guarantees in medical computing.
//!
//! # Example
//!
//! ```text
//! type SafeDose = { dose: mg | dose >= 0.5 && dose <= 10.0 }
//! type ValidCrCl = { crcl: mL/min | crcl > 0 && crcl < 200 }
//! type PositiveInt = { n: Int | n > 0 }
//!
//! fn calculate_dose(weight: { kg | weight > 0 },
//!                   crcl: ValidCrCl) -> SafeDose {
//!     let dose = weight * 0.1;
//!     // Compiler verifies dose is in SafeDose range
//!     dose
//! }
//! ```
//!
//! # Architecture
//!
//! ```text
//! Source Code with Refinements
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  Parse          │  Extract refinement predicates
//! │  Refinements    │
//! └─────────────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  Constraint     │  Generate verification conditions
//! │  Generation     │
//! └─────────────────┘
//!          │
//!          ▼
//! ┌─────────────────┐
//! │  SMT Solving    │  Z3 checks satisfiability
//! │  (Z3)           │
//! └─────────────────┘
//!          │
//!          ├─── SAT: Type checks ✓
//!          └─── UNSAT: Generate counterexample
//! ```

pub mod check;
pub mod clinical; // Phase V1: Clinical refinement types (inspired by Demetrios)
pub mod constraint;
pub mod error;
pub mod smt;
pub mod subtype;
pub mod syntax;

// Keep predicate module for backwards compatibility but don't re-export
#[doc(hidden)]
pub mod predicate;

pub use check::{FunctionRefinement, RefinementChecker, RefinementEnv, VerificationStats};
pub use constraint::{
    Constraint, ConstraintGenerator as VCGenerator, ConstraintKind, ConstraintSet,
};
pub use error::{Counterexample, ErrorSeverity, RefinementError, RefinementErrorKind};
pub use smt::{Model, SmtContext, SmtLogic, SmtResult, SmtSolver, SmtSort};
pub use subtype::{CheckResults, SubtypeChecker, SubtypeResult};
pub use syntax::{
    ArithOp, BaseTypeRef, BuiltinFn, CompareOp, Predicate, PredicateOp, RefinedVar, RefinementExpr,
    RefinementType,
};
