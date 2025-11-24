// Week 26-27: Type Checking Module
//
// Contains type checkers for different MedLang language layers.

pub mod core_lang;
pub mod enum_check;

pub use core_lang::{
    typecheck_block, typecheck_expr, typecheck_fn, DomainEnv, FnEnv, TypeEnv, TypeError,
};
pub use enum_check::{typecheck_enum_variant, typecheck_match};
