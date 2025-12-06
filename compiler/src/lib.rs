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
pub mod effects; // Phase V1: Effect System (Prob, IO, GPU)
pub mod endpoints;
pub mod epistemic; // Phase V1: Epistemic Computing (Knowledge<T> wrapper)
pub mod generics; // Week 52: Parametric Polymorphism
pub mod interop;
pub mod ir;
pub mod lexer;
pub mod loader;
pub mod lower;
pub mod mir; // MedLang Intermediate Representation (Low-level IR)
pub mod ml; // Week 29-30: ML/Surrogate runtime support
pub mod ontology;
pub mod parser;
pub mod parser_v1; // Phase V1: Parser extensions for effects, epistemic, refinements
pub mod policy;
pub mod portfolio;
pub mod qm_stub;
pub mod refinement; // Phase V1: Refinement Types with SMT verification (enhanced)
pub mod registry; // Week 33: Artifact Registry for reproducible science
pub mod resolve;
pub mod rl; // Week 31: Reinforcement Learning for QSP-based policies
pub mod runtime; // Week 29: L₀ runtime execution support
pub mod stanrun;
pub mod traits; // Week 53: Trait System (Typeclasses)
pub mod typecheck;
pub mod typeck;
pub mod typeck_v1; // Phase V1: Extended type checking with effects, epistemic, refinements
pub mod types;
pub mod units; // Week 54: Units of Measure (Dimensional Analysis)
