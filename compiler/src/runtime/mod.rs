// Week 29: Runtime Support for L₀ Core Language
//
// This module provides runtime execution support for L₀ built-in functions,
// particularly the Week 29 surrogate model built-ins.

pub mod builtins;
pub mod value;

pub use builtins::{call_builtin, BuiltinFn};
pub use value::{RuntimeError, RuntimeValue};
