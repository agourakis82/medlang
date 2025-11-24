// Week 26-28: Type Checking Module
//
// Contains type checkers for different MedLang language layers.

pub mod contract_check; // Week 28: Contracts & Assertions
pub mod core_lang;
pub mod enum_check;

pub use contract_check::{typecheck_assert, typecheck_fn_contract, typecheck_invariant_block};
pub use core_lang::{
    typecheck_block, typecheck_expr, typecheck_fn, DomainEnv, FnEnv, TypeEnv, TypeError,
};
pub use enum_check::{typecheck_enum_variant, typecheck_match};
