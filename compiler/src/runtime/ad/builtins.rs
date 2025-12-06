//! High-level AD builtins for MedLang runtime.
//!
//! This module provides the user-facing AD functions exposed to the MedLang language:
//! - `grad`: Compute gradient of scalar function
//! - `jacobian`: Compute Jacobian of vector function
//! - `hessian`: Compute Hessian of scalar function
//! - `jvp`: Jacobian-vector product
//! - `vjp_prepare`: Prepare for vector-Jacobian product (returns gradient function)
//! - `partial`: Partial derivative with respect to named parameter
//! - `grad_check`: Verify gradient correctness using finite differences

use super::dual::{AdContext, AdError, AdResult, DualNumber, DualRecord, DualVector};
use super::ops_vector;
use std::collections::HashMap;

// ============================================================================
// GRADIENT (grad)
// ============================================================================

/// Compute the gradient of a scalar function f: R^n → R.
///
/// # Arguments
/// * `f` - The function to differentiate
/// * `point` - The point at which to evaluate the gradient
///
/// # Returns
/// The gradient vector ∇f(point)
///
/// # Example
/// ```ignore
/// // f(x, y) = x² + y²
/// // ∇f(3, 4) = [6, 8]
/// let grad = grad(|v| v[0].sq().add(&v[1].sq()), &[3.0, 4.0])?;
/// assert_eq!(grad, vec![6.0, 8.0]);
/// ```
pub fn grad<F>(f: F, point: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    ops_vector::gradient(f, point)
}

/// Compute gradient with named parameters using DualRecord.
///
/// # Arguments
/// * `f` - The function to differentiate
/// * `params` - Named parameters as (name, value) pairs
///
/// # Returns
/// HashMap mapping parameter names to their partial derivatives
pub fn grad_named<F>(f: F, params: &[(&str, f64)]) -> AdResult<HashMap<String, f64>>
where
    F: Fn(&DualRecord) -> AdResult<DualNumber>,
{
    let mut gradients = HashMap::new();

    for (var_name, _) in params {
        // Create record with this variable seeded
        let record = DualRecord::seeded_at(params, var_name);
        let result = f(&record)?;
        gradients.insert(var_name.to_string(), result.tangent);
    }

    Ok(gradients)
}

// ============================================================================
// JACOBIAN
// ============================================================================

/// Compute the Jacobian matrix of a vector function f: R^n → R^m.
///
/// # Arguments
/// * `f` - The function to differentiate
/// * `point` - The point at which to evaluate the Jacobian
/// * `output_dim` - The output dimension m of the function
///
/// # Returns
/// The m×n Jacobian matrix in row-major order
///
/// # Example
/// ```ignore
/// // f(x, y) = [x + y, x * y]
/// // J(2, 3) = [[1, 1], [3, 2]]
/// let jac = jacobian(|v| ..., &[2.0, 3.0], 2)?;
/// ```
pub fn jacobian<F>(f: F, point: &[f64], output_dim: usize) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    ops_vector::jacobian(f, point, output_dim)
}

/// Jacobian as a 2D structure (vector of rows).
pub fn jacobian_2d<F>(f: F, point: &[f64], output_dim: usize) -> AdResult<Vec<Vec<f64>>>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    let flat = jacobian(f, point, output_dim)?;
    let n = point.len();
    let m = output_dim;

    let mut rows = Vec::with_capacity(m);
    for i in 0..m {
        let row: Vec<f64> = flat[i * n..(i + 1) * n].to_vec();
        rows.push(row);
    }
    Ok(rows)
}

// ============================================================================
// HESSIAN
// ============================================================================

/// Compute the Hessian matrix of a scalar function f: R^n → R.
///
/// # Arguments
/// * `f` - The function whose Hessian to compute
/// * `point` - The point at which to evaluate the Hessian
///
/// # Returns
/// The n×n symmetric Hessian matrix in row-major order
///
/// # Note
/// Uses finite differences for second derivatives. For large n,
/// this requires O(n²) function evaluations.
pub fn hessian<F>(f: F, point: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber> + Copy,
{
    ops_vector::hessian(f, point)
}

/// Hessian as a 2D structure (vector of rows).
pub fn hessian_2d<F>(f: F, point: &[f64]) -> AdResult<Vec<Vec<f64>>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber> + Copy,
{
    let flat = hessian(f, point)?;
    let n = point.len();

    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let row: Vec<f64> = flat[i * n..(i + 1) * n].to_vec();
        rows.push(row);
    }
    Ok(rows)
}

/// Compute only the diagonal of the Hessian (more efficient).
pub fn hessian_diag<F>(f: F, point: &[f64], epsilon: f64) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    let n = point.len();
    let mut diag = Vec::with_capacity(n);

    for i in 0..n {
        // Second derivative using central differences
        let mut p_plus = point.to_vec();
        let mut p_minus = point.to_vec();
        let mut p_center = point.to_vec();

        p_plus[i] += epsilon;
        p_minus[i] -= epsilon;

        let input_plus = DualVector::constants(&p_plus);
        let input_minus = DualVector::constants(&p_minus);
        let input_center = DualVector::constants(&p_center);

        let f_plus = f(&input_plus)?.primal;
        let f_minus = f(&input_minus)?.primal;
        let f_center = f(&input_center)?.primal;

        let second_deriv = (f_plus - 2.0 * f_center + f_minus) / (epsilon * epsilon);
        diag.push(second_deriv);
    }

    Ok(diag)
}

// ============================================================================
// JVP (Jacobian-Vector Product)
// ============================================================================

/// Compute the Jacobian-vector product: J(f)(x) @ v
///
/// This is computed efficiently in a single forward pass by seeding
/// the input with the tangent vector v.
///
/// # Arguments
/// * `f` - The function
/// * `point` - The point at which to evaluate
/// * `tangent_vec` - The vector v to multiply with the Jacobian
///
/// # Returns
/// The JVP J @ v
pub fn jvp<F>(f: F, point: &[f64], tangent_vec: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    ops_vector::jvp(f, point, tangent_vec)
}

/// Scalar function JVP (returns scalar since f: R^n → R).
pub fn jvp_scalar<F>(f: F, point: &[f64], tangent_vec: &[f64]) -> AdResult<f64>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    ops_vector::directional_derivative(f, point, tangent_vec)
}

// ============================================================================
// VJP PREPARATION
// ============================================================================

/// A prepared VJP (vector-Jacobian product) computation.
///
/// In forward-mode AD, we can't directly compute VJP (which is reverse-mode).
/// Instead, we prepare by computing the full Jacobian, then multiply.
#[derive(Debug, Clone)]
pub struct VjpPrepared {
    /// The Jacobian matrix (m×n) in row-major order
    pub jacobian: Vec<f64>,
    /// Number of outputs (m)
    pub output_dim: usize,
    /// Number of inputs (n)
    pub input_dim: usize,
}

impl VjpPrepared {
    /// Compute J^T @ cotangent (the actual VJP).
    pub fn apply(&self, cotangent: &[f64]) -> AdResult<Vec<f64>> {
        if cotangent.len() != self.output_dim {
            return Err(AdError::DimensionMismatch {
                expected: self.output_dim,
                got: cotangent.len(),
            });
        }

        let mut result = vec![0.0; self.input_dim];
        for j in 0..self.input_dim {
            for i in 0..self.output_dim {
                // J^T[j,i] = J[i,j]
                result[j] += self.jacobian[i * self.input_dim + j] * cotangent[i];
            }
        }
        Ok(result)
    }
}

/// Prepare for VJP computation by computing the Jacobian.
///
/// # Arguments
/// * `f` - The function
/// * `point` - The point at which to evaluate
/// * `output_dim` - The output dimension of f
///
/// # Returns
/// A VjpPrepared object that can compute J^T @ v for any cotangent v
pub fn vjp_prepare<F>(f: F, point: &[f64], output_dim: usize) -> AdResult<VjpPrepared>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    let jac = jacobian(f, point, output_dim)?;
    Ok(VjpPrepared {
        jacobian: jac,
        output_dim,
        input_dim: point.len(),
    })
}

// ============================================================================
// PARTIAL DERIVATIVES
// ============================================================================

/// Compute partial derivative with respect to a specific variable index.
pub fn partial<F>(f: F, point: &[f64], var_index: usize) -> AdResult<f64>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    if var_index >= point.len() {
        return Err(AdError::InvalidOperation(format!(
            "Variable index {} out of bounds (n={})",
            var_index,
            point.len()
        )));
    }

    let input = DualVector::seeded_at(point, var_index);
    let output = f(&input)?;
    Ok(output.tangent)
}

/// Compute partial derivative with respect to a named parameter.
pub fn partial_named<F>(f: F, params: &[(&str, f64)], var_name: &str) -> AdResult<f64>
where
    F: Fn(&DualRecord) -> AdResult<DualNumber>,
{
    let record = DualRecord::seeded_at(params, var_name);

    if !record.contains(var_name) {
        return Err(AdError::MissingParameter(var_name.to_string()));
    }

    let output = f(&record)?;
    Ok(output.tangent)
}

/// Compute multiple partial derivatives at once (more efficient than calling partial repeatedly).
pub fn partials<F>(f: F, point: &[f64], var_indices: &[usize]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    let mut results = Vec::with_capacity(var_indices.len());
    for &idx in var_indices {
        results.push(partial(&f, point, idx)?);
    }
    Ok(results)
}

// ============================================================================
// GRADIENT CHECKING
// ============================================================================

/// Result of gradient checking.
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    /// Whether the gradient check passed
    pub passed: bool,
    /// Computed (AD) gradient
    pub computed_gradient: Vec<f64>,
    /// Numerical (finite differences) gradient
    pub numerical_gradient: Vec<f64>,
    /// Relative errors for each component
    pub relative_errors: Vec<f64>,
    /// Maximum relative error
    pub max_relative_error: f64,
}

/// Check gradient correctness using finite differences.
///
/// # Arguments
/// * `f_dual` - The function using dual numbers
/// * `f_scalar` - The same function for scalar evaluation
/// * `point` - Point at which to check
/// * `epsilon` - Finite difference step size (default: 1e-7)
/// * `tolerance` - Relative error tolerance (default: 1e-5)
///
/// # Returns
/// Detailed gradient check result
pub fn grad_check<F, G>(
    f_dual: F,
    f_scalar: G,
    point: &[f64],
    epsilon: f64,
    tolerance: f64,
) -> AdResult<GradCheckResult>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
    G: Fn(&[f64]) -> f64,
{
    let computed = grad(&f_dual, point)?;

    // Compute numerical gradient
    let mut numerical = Vec::with_capacity(point.len());
    for i in 0..point.len() {
        let mut p_plus = point.to_vec();
        let mut p_minus = point.to_vec();
        p_plus[i] += epsilon;
        p_minus[i] -= epsilon;

        let deriv = (f_scalar(&p_plus) - f_scalar(&p_minus)) / (2.0 * epsilon);
        numerical.push(deriv);
    }

    // Compute relative errors
    let mut relative_errors = Vec::with_capacity(point.len());
    let mut max_error = 0.0_f64;

    for (c, n) in computed.iter().zip(numerical.iter()) {
        let scale = c.abs().max(n.abs()).max(1.0);
        let rel_err = (c - n).abs() / scale;
        relative_errors.push(rel_err);
        max_error = max_error.max(rel_err);
    }

    let passed = max_error < tolerance;

    Ok(GradCheckResult {
        passed,
        computed_gradient: computed,
        numerical_gradient: numerical,
        relative_errors,
        max_relative_error: max_error,
    })
}

/// Simple gradient check that just returns pass/fail.
pub fn grad_check_simple<F, G>(f_dual: F, f_scalar: G, point: &[f64]) -> AdResult<bool>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
    G: Fn(&[f64]) -> f64,
{
    let result = grad_check(f_dual, f_scalar, point, 1e-7, 1e-5)?;
    Ok(result.passed)
}

// ============================================================================
// HIGHER-ORDER UTILITIES
// ============================================================================

/// Compute the Laplacian (trace of Hessian): Δf = Σ ∂²f/∂xᵢ²
pub fn laplacian<F>(f: F, point: &[f64], epsilon: f64) -> AdResult<f64>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    let diag = hessian_diag(f, point, epsilon)?;
    Ok(diag.iter().sum())
}

/// Compute the divergence of a vector field: div(F) = Σ ∂Fᵢ/∂xᵢ
pub fn divergence<F>(f: F, point: &[f64]) -> AdResult<f64>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    let n = point.len();
    let mut div = 0.0;

    for i in 0..n {
        let input = DualVector::seeded_at(point, i);
        let output = f(&input)?;

        if output.len() != n {
            return Err(AdError::DimensionMismatch {
                expected: n,
                got: output.len(),
            });
        }

        div += output.elements[i].tangent;
    }

    Ok(div)
}

/// Compute the curl of a 3D vector field (for n=3 only).
/// curl(F) = [∂F₃/∂y - ∂F₂/∂z, ∂F₁/∂z - ∂F₃/∂x, ∂F₂/∂x - ∂F₁/∂y]
pub fn curl_3d<F>(f: F, point: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    if point.len() != 3 {
        return Err(AdError::InvalidOperation(
            "curl_3d requires 3D input".to_string(),
        ));
}

    // Compute full Jacobian
    let jac = jacobian(&f, point, 3)?;
    // jac is 3×3 in row-major: [[∂F₁/∂x, ∂F₁/∂y, ∂F₁/∂z], ...]

    let df1_dx = jac[0];
    let df1_dy = jac[1];
    let df1_dz = jac[2];
    let df2_dx = jac[3];
    let df2_dy = jac[4];
    let df2_dz = jac[5];
    let df3_dx = jac[6];
    let df3_dy = jac[7];
    let df3_dz = jac[8];

    Ok(vec![
        df3_dy - df2_dz, // ∂F₃/∂y - ∂F₂/∂z
        df1_dz - df3_dx, // ∂F₁/∂z - ∂F₃/∂x
        df2_dx - df1_dy, // ∂F₂/∂x - ∂F₁/∂y
    ])
}

// ============================================================================
// OPTIMIZATION HELPERS
// ============================================================================

/// Compute function value and gradient together (more efficient if both needed).
pub fn value_and_grad<F>(f: F, point: &[f64]) -> AdResult<(f64, Vec<f64>)>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    // First, compute value with constants
    let const_input = DualVector::constants(point);
    let value = f(&const_input)?.primal;

    // Then compute gradient
    let gradient = grad(&f, point)?;

    Ok((value, gradient))
}

/// Compute value, gradient, and Hessian together.
pub fn value_grad_hessian<F>(f: F, point: &[f64]) -> AdResult<(f64, Vec<f64>, Vec<f64>)>
where
    F: Fn(&DualVector) -> AdResult<DualNumber> + Copy,
{
    let (value, gradient) = value_and_grad(&f, point)?;
    let hess = hessian(f, point)?;
    Ok((value, gradient, hess))
}

/// Newton's method step: x_new = x - H⁻¹ @ ∇f
/// (Uses simple diagonal approximation for efficiency)
pub fn newton_step<F>(f: F, point: &[f64], epsilon: f64) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    let gradient = grad(&f, point)?;
    let hess_diag = hessian_diag(&f, point, epsilon)?;

    let mut step = Vec::with_capacity(point.len());
    for i in 0..point.len() {
        if hess_diag[i].abs() < 1e-10 {
            // If Hessian diagonal is near zero, use gradient descent
            step.push(-gradient[i]);
        } else {
            step.push(-gradient[i] / hess_diag[i]);
        }
    }

    Ok(step)
}

// ============================================================================
// AD CONTEXT INTEGRATION
// ============================================================================

/// Compute gradient with tracing enabled for debugging.
pub fn grad_with_trace<F>(f: F, point: &[f64]) -> AdResult<(Vec<f64>, AdContext)>
where
    F: Fn(&DualVector, &mut AdContext) -> AdResult<DualNumber>,
{
    let n = point.len();
    let mut gradient = Vec::with_capacity(n);
    let mut ctx = AdContext::with_tracing();

    for i in 0..n {
        let input = DualVector::seeded_at(point, i);
        let output = f(&input, &mut ctx)?;
        gradient.push(output.tangent);
    }

    Ok((gradient, ctx))
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON || (a - b).abs() / a.abs().max(b.abs()).max(1.0) < EPSILON
    }

    #[test]
    fn test_grad() {
        // f(x, y) = x² + y²
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].sq()))
        };

        let g = grad(f, &[3.0, 4.0]).unwrap();
        assert!(approx_eq(g[0], 6.0));
        assert!(approx_eq(g[1], 8.0));
    }

    #[test]
    fn test_grad_named() {
        // f(alpha, beta) = alpha² + 2*beta
        let f = |r: &DualRecord| -> AdResult<DualNumber> {
            let alpha = r
                .get("alpha")
                .ok_or(AdError::MissingParameter("alpha".to_string()))?;
            let beta = r
                .get("beta")
                .ok_or(AdError::MissingParameter("beta".to_string()))?;
            let two = DualNumber::constant(2.0);
            Ok(alpha.sq().add(&two.mul(beta)))
        };

        let params = [("alpha", 3.0), ("beta", 2.0)];
        let grads = grad_named(f, &params).unwrap();

        assert!(approx_eq(*grads.get("alpha").unwrap(), 6.0)); // 2*alpha
        assert!(approx_eq(*grads.get("beta").unwrap(), 2.0)); // 2
    }

    #[test]
    fn test_jacobian_2d() {
        // f(x, y) = [x + y, x * y]
        let f = |v: &DualVector| -> AdResult<DualVector> {
            Ok(DualVector::new(vec![
                v.elements[0].add(&v.elements[1]),
                v.elements[0].mul(&v.elements[1]),
            ]))
        };

        let jac = jacobian_2d(f, &[2.0, 3.0], 2).unwrap();

        // J = [[1, 1], [3, 2]]
        assert!(approx_eq(jac[0][0], 1.0));
        assert!(approx_eq(jac[0][1], 1.0));
        assert!(approx_eq(jac[1][0], 3.0));
        assert!(approx_eq(jac[1][1], 2.0));
    }

    #[test]
    fn test_partial() {
        // f(x, y, z) = x² + y³ + z
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0]
                .sq()
                .add(&v.elements[1].cube())
                .add(&v.elements[2]))
        };

        // ∂f/∂x = 2x
        let dx = partial(&f, &[2.0, 3.0, 4.0], 0).unwrap();
        assert!(approx_eq(dx, 4.0));

        // ∂f/∂y = 3y²
        let dy = partial(&f, &[2.0, 3.0, 4.0], 1).unwrap();
        assert!(approx_eq(dy, 27.0));

        // ∂f/∂z = 1
        let dz = partial(&f, &[2.0, 3.0, 4.0], 2).unwrap();
        assert!(approx_eq(dz, 1.0));
    }

    #[test]
    fn test_jvp_scalar() {
        // f(x, y) = x² + y²
        // ∇f(3, 4) = [6, 8]
        // JVP with v = [1, 0] = 6
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].sq()))
        };

        let jvp = jvp_scalar(f, &[3.0, 4.0], &[1.0, 0.0]).unwrap();
        assert!(approx_eq(jvp, 6.0));
    }

    #[test]
    fn test_vjp_prepare() {
        // f(x, y) = [x + y, x * y]
        // J(2, 3) = [[1, 1], [3, 2]]
        // J^T @ [1, 1] = [[1, 3], [1, 2]] @ [1, 1] = [4, 3]
        let f = |v: &DualVector| -> AdResult<DualVector> {
            Ok(DualVector::new(vec![
                v.elements[0].add(&v.elements[1]),
                v.elements[0].mul(&v.elements[1]),
            ]))
        };

        let vjp = vjp_prepare(f, &[2.0, 3.0], 2).unwrap();
        let result = vjp.apply(&[1.0, 1.0]).unwrap();

        assert!(approx_eq(result[0], 4.0)); // 1 + 3
        assert!(approx_eq(result[1], 3.0)); // 1 + 2
    }

    #[test]
    fn test_grad_check() {
        // f(x, y) = x² + y²
        let f_dual = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].sq()))
        };
        let f_scalar = |x: &[f64]| x[0] * x[0] + x[1] * x[1];

        let result = grad_check(f_dual, f_scalar, &[3.0, 4.0], 1e-7, 1e-5).unwrap();
        assert!(result.passed);
        assert!(result.max_relative_error < 1e-5);
    }

    #[test]
    fn test_laplacian() {
        // f(x, y) = x² + y²
        // Laplacian = 2 + 2 = 4
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].sq()))
        };

        let lap = laplacian(f, &[1.0, 1.0], 1e-5).unwrap();
        assert!((lap - 4.0).abs() < 0.01);
    }

    #[test]
    fn test_divergence() {
        // F(x, y) = [x, y]
        // div(F) = 1 + 1 = 2
        let f = |v: &DualVector| -> AdResult<DualVector> { Ok(v.clone()) };

        let div = divergence(f, &[1.0, 2.0]).unwrap();
        assert!(approx_eq(div, 2.0));
    }

    #[test]
    fn test_value_and_grad() {
        // f(x, y) = x² + y²
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].sq()))
        };

        let (value, gradient) = value_and_grad(f, &[3.0, 4.0]).unwrap();

        assert!(approx_eq(value, 25.0));
        assert!(approx_eq(gradient[0], 6.0));
        assert!(approx_eq(gradient[1], 8.0));
    }

    #[test]
    fn test_hessian_diag() {
        // f(x, y) = x² + y³
        // H = [[2, 0], [0, 6y]]
        // diag at (1, 2) = [2, 12]
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].cube()))
        };

        let diag = hessian_diag(f, &[1.0, 2.0], 1e-5).unwrap();
        assert!((diag[0] - 2.0).abs() < 0.01);
        assert!((diag[1] - 12.0).abs() < 0.1);
    }
}
