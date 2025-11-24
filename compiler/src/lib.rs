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
pub mod lower;
pub mod parser;
pub mod portfolio;
pub mod qm_stub;
pub mod stanrun;
pub mod typeck;
