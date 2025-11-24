// Week 29-30: ML/Surrogate Runtime Support
//
// This module provides runtime support for first-class surrogate models,
// backend selection, and surrogate evaluation in MedLang.

pub mod backend;
pub mod eval;
pub mod surrogate; // Week 30: Surrogate evaluation

pub use backend::{BackendError, BackendKind};
pub use eval::{evaluate_surrogate, SurrogateEvalConfig, SurrogateEvalError, SurrogateEvalReport};
pub use surrogate::{SurrogateError, SurrogateModelHandle, SurrogateTrainConfig};
