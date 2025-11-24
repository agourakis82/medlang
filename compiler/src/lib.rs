//! MedLang compiler library.
//!
//! This crate implements the MedLang compiler for computational medicine,
//! targeting Vertical Slice 0: one-compartment oral PK with NLME.

pub mod ast;
pub mod codegen;
pub mod data;
pub mod datagen;
pub mod dataload;
pub mod design;
pub mod diagnostics;
pub mod endpoints;
pub mod interop;
pub mod ir;
pub mod lexer;
pub mod loader;
pub mod lower;
pub mod ml; // Week 29-30: ML/Surrogate runtime support
pub mod parser;
pub mod policy;
pub mod portfolio;
pub mod qm_stub;
pub mod registry; // Week 33: Artifact Registry for reproducible science
pub mod resolve;
pub mod rl; // Week 31: Reinforcement Learning for QSP-based policies
pub mod runtime; // Week 29: L₀ runtime execution support
pub mod stanrun;
pub mod typecheck;
pub mod typeck;
pub mod types;
