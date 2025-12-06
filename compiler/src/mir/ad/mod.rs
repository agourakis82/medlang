//! Automatic Differentiation Transformation
//!
//! MedLang implements AD as a compile-time transformation, not a runtime library.
//! This provides zero-overhead differentiation with optimal code generation.
//!
//! Supported modes:
//! - Forward mode: Propagate tangents alongside primals (good for few inputs, many outputs)
//! - Reverse mode: Backpropagate adjoints (good for many inputs, few outputs - gradients)
//!
//! # Example
//!
//! ```text
//! @differentiable
//! fn log_likelihood(theta: Vector<f64>, data: Vector<f64>) -> f64 {
//!     let mu = theta[0];
//!     let sigma = theta[1];
//!     sum(map(data, |x| normal_lpdf(x, mu, sigma)))
//! }
//!
//! // Compiler generates:
//! // - log_likelihood_primal: Original computation
//! // - log_likelihood_tangent: Forward-mode derivative
//! // - log_likelihood_adjoint: Reverse-mode gradient
//! ```

pub mod activity;
pub mod forward;
pub mod reverse;
pub mod transform;

pub use activity::{ActivityAnalysis, ActivityResult};
pub use forward::ForwardModeTransform;
pub use reverse::ReverseModeTransform;
pub use transform::{ADError, ADMode, ADTransformResult, ADTransformer};
