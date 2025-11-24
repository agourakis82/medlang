// Week 26: Type System for MedLang
//
// This module contains type representations and type checking infrastructure.

pub mod core_lang;

pub use core_lang::{resolve_type_ann, CoreType, TypedFnSig};
