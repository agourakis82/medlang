//! Policy Layer for RL Integration
//!
//! This module provides the infrastructure for integrating RL-trained policies
//! back into MedLang as first-class clinical policies that can be:
//! - Evaluated using standard clinical endpoints (ORR, PFS, DLT)
//! - Compared against hand-crafted policies and standard trial arms
//! - Used in design optimization and PoS calculations
//! - Logged for policy distillation into interpretable MedLang expressions

pub mod expr_policy;
pub mod external_json;
pub mod logging;
pub mod plugin;

pub use expr_policy::ExprPolicy;
pub use external_json::{ExternalJsonPolicy, ExternalJsonPolicyConfig};
pub use logging::{PolicyLogger, PolicyStepLogRow};
pub use plugin::PolicyPlugin;
