//! Automatic Differentiation (AD) Module for MedLang
//!
//! This module provides a complete forward-mode automatic differentiation system
//! for MedLang, enabling gradient computation for optimization, sensitivity analysis,
//! and machine learning applications.
//!
//! ## Architecture
//!
//! The AD system is organized into several layers:
//!
//! 1. **Core Types** (`dual.rs`):
//!    - `DualNumber`: Scalar dual numbers (primal + tangent)
//!    - `DualVector`: Vectors of dual numbers for multi-variable functions
//!    - `DualValue`: Unified enum for scalar/vector operations
//!    - `DualRecord`: Named parameter gradients for structured differentiation
//!    - `AdContext`: Debugging and tracing context
//!    - `AdError`: Comprehensive error types
//!
//! 2. **Scalar Operations** (`ops_scalar.rs`):
//!    - 45+ differentiable scalar operations organized by category
//!    - Basic arithmetic (add, sub, mul, div, neg)
//!    - Power functions (sq, cube, sqrt, cbrt, pow)
//!    - Exponential/logarithmic (exp, ln, log10, log2)
//!    - Trigonometric (sin, cos, tan, etc.)
//!    - Hyperbolic (sinh, cosh, tanh, etc.)
//!    - Special functions (erf, sigmoid, softplus)
//!    - ML activations (relu, gelu, swish)
//!    - Smooth approximations for non-differentiable functions
//!    - PK/PD functions (hill, michaelis_menten, emax)
//!    - Statistical functions (log-likelihoods, losses)
//!
//! 3. **Vector Operations** (`ops_vector.rs`):
//!    - Element-wise operations
//!    - Reductions (sum, mean, norm, variance)
//!    - Linear algebra (dot, matvec, outer product)
//!    - Statistical operations (softmax, correlation)
//!    - Loss functions (MSE, cross-entropy, KL divergence)
//!    - Gradient/Jacobian/Hessian computation utilities
//!
//! 4. **High-Level Builtins** (`builtins.rs`):
//!    - `grad`: Compute gradient of scalar function
//!    - `jacobian`: Compute Jacobian of vector function
//!    - `hessian`: Compute Hessian of scalar function
//!    - `jvp`: Jacobian-vector product
//!    - `vjp_prepare`: Prepare for vector-Jacobian product
//!    - `partial`: Partial derivatives
//!    - `grad_check`: Gradient verification
//!    - Higher-order operators (laplacian, divergence, curl)
//!
//! ## Usage Examples
//!
//! ### Basic Gradient Computation
//!
//! ```ignore
//! use medlangc::runtime::ad::*;
//!
//! // f(x, y) = x² + y²
//! let f = |v: &DualVector| -> AdResult<DualNumber> {
//!     Ok(v.elements[0].sq().add(&v.elements[1].sq()))
//! };
//!
//! let gradient = grad(f, &[3.0, 4.0])?;
//! // gradient = [6.0, 8.0] since ∇f = [2x, 2y]
//! ```
//!
//! ### Named Parameter Gradients
//!
//! ```ignore
//! // f(alpha, beta) = alpha * exp(-beta * t)
//! let f = |r: &DualRecord| -> AdResult<DualNumber> {
//!     let alpha = r.get("alpha")?;
//!     let beta = r.get("beta")?;
//!     let t = r.get("t")?;
//!     Ok(alpha.mul(&beta.neg().mul(t).exp()))
//! };
//!
//! let params = [("alpha", 1.0), ("beta", 0.5), ("t", 2.0)];
//! let grads = grad_named(f, &params)?;
//! // grads["alpha"] = exp(-beta*t)
//! // grads["beta"] = -alpha * t * exp(-beta*t)
//! ```
//!
//! ### Jacobian Computation
//!
//! ```ignore
//! // f(x, y) = [x + y, x * y]
//! let f = |v: &DualVector| -> AdResult<DualVector> {
//!     Ok(DualVector::new(vec![
//!         v.elements[0].add(&v.elements[1]),
//!         v.elements[0].mul(&v.elements[1]),
//!     ]))
//! };
//!
//! let jac = jacobian(f, &[2.0, 3.0], 2)?;
//! // jac = [[1, 1], [3, 2]] (row-major)
//! ```
//!
//! ### Gradient Checking
//!
//! ```ignore
//! let f_dual = |v: &DualVector| Ok(v.elements[0].sq());
//! let f_scalar = |x: &[f64]| x[0] * x[0];
//!
//! let result = grad_check(f_dual, f_scalar, &[3.0], 1e-7, 1e-5)?;
//! assert!(result.passed);
//! ```
//!
//! ## Theory: Forward-Mode AD
//!
//! Forward-mode AD works by augmenting each value with its derivative,
//! represented as a dual number: x̂ = x + x'ε where ε² = 0.
//!
//! Key properties:
//! - (a + bε) + (c + dε) = (a+c) + (b+d)ε
//! - (a + bε) × (c + dε) = ac + (ad + bc)ε  (product rule!)
//! - f(a + bε) = f(a) + f'(a)bε  (chain rule!)
//!
//! This automatically computes derivatives through arbitrary compositions.
//!
//! ## Performance Characteristics
//!
//! - Gradient of f: R^n → R requires n forward passes
//! - Jacobian of f: R^n → R^m requires n forward passes
//! - JVP (J @ v) requires just 1 forward pass
//! - For m >> n, forward mode is efficient
//! - For n >> m, reverse mode (not implemented) would be better

pub mod builtins;
pub mod dual;
pub mod ops_scalar;
pub mod ops_vector;

// Re-export main types and functions for convenience
pub use builtins::{
    curl_3d, divergence, grad, grad_check, grad_check_simple, grad_named, hessian, hessian_2d,
    hessian_diag, jacobian, jacobian_2d, jvp, jvp_scalar, laplacian, newton_step, partial,
    partial_named, partials, value_and_grad, value_grad_hessian, vjp_prepare, GradCheckResult,
    VjpPrepared,
};

pub use dual::{AdContext, AdError, AdResult, DualNumber, DualRecord, DualValue, DualVector};

pub use ops_scalar::{
    // Smooth approximations
    abs_smooth,
    // Trigonometric
    acos,
    // Hyperbolic
    acosh,
    // Basic arithmetic
    add as scalar_add,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    // Statistics
    binary_cross_entropy,
    // Power functions
    cbrt,
    // Utilities
    ceil,
    cos,
    cosh,
    cot,
    coth,
    csc,
    csch,
    cube,
    div as scalar_div,
    // ML activations
    elu_smooth,
    // PK/PD
    emax,
    // Special functions
    erf,
    // Exponential/logarithmic
    exp,
    exp2,
    exp_m1,
    first_order_elimination,
    floor,
    fma,
    fract,
    gaussian_pdf,
    gelu,
    heaviside_smooth,
    hill,
    huber_loss,
    hypot,
    leaky_relu_smooth,
    lerp,
    ln,
    ln_1p,
    log,
    log10,
    log2,
    log_likelihood_exponential,
    log_likelihood_normal,
    log_likelihood_poisson,
    michaelis_menten,
    mse_loss as scalar_mse_loss,
    mul as scalar_mul,
    neg as scalar_neg,
    one_compartment_oral,
    pow,
    powf,
    recip,
    relu_smooth,
    round,
    sec,
    sech,
    sigmoid,
    sign_smooth,
    sin,
    sinh,
    smooth_clamp,
    smooth_gt,
    smooth_le,
    smooth_lt,
    smooth_max,
    smooth_min,
    softplus,
    sq,
    sqrt,
    sub as scalar_sub,
    swish,
    tan,
    tanh,
    trunc,
};

pub use ops_vector::{
    // Element-wise
    add as vector_add,
    // Gradient utilities
    check_gradient,
    // Statistics
    correlation,
    cos as vector_cos,
    covariance,
    // Losses
    cross_entropy_loss,
    directional_derivative,
    div_elementwise,
    // Linear algebra
    dot,
    exp as vector_exp,
    gelu as vector_gelu,
    gradient,
    hadamard,
    kl_divergence,
    ln as vector_ln,
    log_softmax,
    // Reductions
    log_sum_exp,
    mae_loss,
    matvec,
    mean,
    mse_loss as vector_mse_loss,
    neg as vector_neg,
    norm_inf,
    norm_l1,
    norm_l2,
    norm_lp,
    norm_sq,
    normalize,
    outer,
    powf as vector_powf,
    product,
    quadratic_form,
    relu_smooth as vector_relu_smooth,
    rmse_loss,
    sample_std_dev,
    sample_variance,
    scale,
    sigmoid as vector_sigmoid,
    sin as vector_sin,
    softmax,
    sq as vector_sq,
    sqrt as vector_sqrt,
    std_dev,
    sub as vector_sub,
    sum,
    swish as vector_swish,
    tan as vector_tan,
    tanh as vector_tanh,
    variance,
    vecmat,
    weighted_sum,
};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use super::builtins::*;
    pub use super::dual::*;
    pub use super::ops_scalar;
    pub use super::ops_vector;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_workflow() {
        // Demonstrate the typical AD workflow

        // 1. Define a function using dual numbers
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            // f(x, y) = x² + 2xy + y²
            let x = &v.elements[0];
            let y = &v.elements[1];
            let two = DualNumber::constant(2.0);
            Ok(x.sq().add(&two.mul(&x.mul(y))).add(&y.sq()))
        };

        // 2. Compute gradient
        let point = [3.0, 4.0];
        let gradient = grad(f, &point).unwrap();

        // ∇f = [2x + 2y, 2x + 2y] = [14, 14] at (3, 4)
        assert!((gradient[0] - 14.0).abs() < 1e-10);
        assert!((gradient[1] - 14.0).abs() < 1e-10);

        // 3. Compute value and gradient together
        let (value, grad2) = value_and_grad(f, &point).unwrap();
        assert!((value - 49.0).abs() < 1e-10); // (3+4)² = 49
        assert_eq!(gradient, grad2);
    }

    #[test]
    fn test_pk_model_gradient() {
        // Test gradient computation for a PK model

        // One-compartment first-order elimination: C(t) = C0 * exp(-k*t)
        let pk_model = |v: &DualVector| -> AdResult<DualNumber> {
            let c0 = &v.elements[0]; // Initial concentration
            let k = &v.elements[1]; // Elimination rate
            let t = &v.elements[2]; // Time

            Ok(c0.mul(&k.neg().mul(t).exp()))
        };

        // At C0=100, k=0.1, t=10: C = 100 * exp(-1) ≈ 36.79
        let point = [100.0, 0.1, 10.0];
        let (value, gradient) = value_and_grad(pk_model, &point).unwrap();

        let expected_c = 100.0 * (-1.0_f64).exp();
        assert!((value - expected_c).abs() < 1e-10);

        // ∂C/∂C0 = exp(-k*t)
        let expected_dc_dc0 = (-1.0_f64).exp();
        assert!((gradient[0] - expected_dc_dc0).abs() < 1e-10);

        // ∂C/∂k = -C0 * t * exp(-k*t)
        let expected_dc_dk = -100.0 * 10.0 * (-1.0_f64).exp();
        assert!((gradient[1] - expected_dc_dk).abs() < 1e-10);
    }

    #[test]
    fn test_ml_loss_gradient() {
        // Test gradient of MSE loss

        let mse = |params: &DualVector| -> AdResult<DualNumber> {
            // Simple linear model: y_pred = w * x + b
            // Loss = (y_true - y_pred)²
            let w = &params.elements[0];
            let b = &params.elements[1];

            // Fixed data point: x=2, y_true=5
            let x = DualNumber::constant(2.0);
            let y_true = DualNumber::constant(5.0);

            let y_pred = w.mul(&x).add(b);
            let error = y_true.sub(&y_pred);
            Ok(error.sq())
        };

        // At w=1, b=1: y_pred = 3, error = 2, loss = 4
        let point = [1.0, 1.0];
        let (loss, gradient) = value_and_grad(mse, &point).unwrap();

        assert!((loss - 4.0).abs() < 1e-10);

        // ∂L/∂w = 2 * (y_true - w*x - b) * (-x) = 2 * 2 * (-2) = -8
        assert!((gradient[0] - (-8.0)).abs() < 1e-10);

        // ∂L/∂b = 2 * (y_true - w*x - b) * (-1) = 2 * 2 * (-1) = -4
        assert!((gradient[1] - (-4.0)).abs() < 1e-10);
    }
}
