//! Evidence Program Execution Layer
//!
//! Provides the runtime infrastructure for executing L₃ evidence programs.
//! This module acts as an interpreter that orchestrates:
//! - Multi-trial/multi-indication hierarchical model fitting
//! - MAP prior derivation
//! - Design evaluation and optimization
//!
//! Instead of manually calling multiple CLIs, users write declarative evidence
//! programs that are executed end-to-end by this runner.

pub mod runner;
pub mod validation;

pub use runner::{run_evidence_program, EvidenceRunConfig, EvidenceRunResult};
pub use validation::{check_evidence_program, EvidenceValidationError};
