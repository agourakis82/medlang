//! Deterministic Floating-Point Module
//!
//! This module provides reproducible floating-point computation for clinical computing.
//! Medical and statistical software requires bitwise reproducibility across:
//! - Different hardware platforms (x86, ARM, GPU)
//! - Different compiler versions and optimization levels
//! - Parallel execution with varying thread counts
//!
//! # Key Features
//!
//! - **Reproducible reductions**: Order-independent summation and products
//! - **Deterministic transcendentals**: Platform-independent exp, log, sin, cos, etc.
//! - **Controlled rounding**: Explicit rounding mode management
//! - **Error bounds**: Tracked error accumulation for numerical stability
//!
//! # Implementation Strategies
//!
//! 1. **Compensated summation**: Kahan/Neumaier algorithms for accurate sums
//! 2. **Reproducible BLAS**: Deterministic linear algebra operations
//! 3. **Fixed-order evaluation**: Canonical expression evaluation order
//! 4. **Software transcendentals**: Portable implementations of math functions

pub mod analysis;
pub mod rounding;
pub mod summation;
pub mod transcendental;
pub mod transform;

pub use analysis::{
    AccuracyRequirement, DeterminismAnalysis, DeterminismLevel, FPAnalysisResult,
    NumericalStabilityInfo, PrecisionLoss,
};
pub use rounding::{RoundingContext, RoundingMode, RoundingTransform};
pub use summation::{
    CompensatedSum, KahanSum, NeumaierSum, PairwiseSum, ReproducibleReduction, SummationAlgorithm,
    TreeSum,
};
pub use transcendental::{
    DeterministicMath, MathAccuracy, SoftwareTranscendental, TranscendentalImpl,
};
pub use transform::{DeterministicFPTransform, FPTransformConfig, FPTransformResult};
