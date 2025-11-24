// Week 26-27: Type System for MedLang
//
// This module contains type representations and type checking infrastructure.

pub mod core_lang;
pub mod enum_types;

pub use core_lang::{resolve_type_ann, CoreType, TypedFnSig};
pub use enum_types::{EnumEnv, EnumInfo, EnumType};
