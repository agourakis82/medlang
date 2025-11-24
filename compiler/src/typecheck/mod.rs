// Week 26: Type Checking Module
//
// Contains type checkers for different MedLang language layers.

pub mod core_lang;

pub use core_lang::{
    typecheck_block, typecheck_expr, typecheck_fn, DomainEnv, FnEnv, TypeEnv, TypeError,
};
