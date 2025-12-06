// Week 50: Builtin Functions for Automatic Differentiation
//
// This module provides the grad, value_and_grad, and gradient checking
// builtins for MedLang's automatic differentiation system.

use crate::runtime::dual::{AdError, AdResult, DualNumber};
use crate::runtime::value::{RuntimeError, RuntimeValue};
use std::collections::HashMap;

// =============================================================================
// GRADIENT COMPUTATION RESULT
// =============================================================================

/// Result of gradient computation
#[derive(Debug, Clone)]
pub struct GradResult {
    /// Value of the function at the point
    pub value: f64,
    /// Gradient (derivative) at the point
    pub gradient: f64,
}

impl GradResult {
    pub fn new(value: f64, gradient: f64) -> Self {
        Self { value, gradient }
    }
}

/// Result of gradient checking
#[derive(Debug, Clone)]
pub struct GradCheckResult {
    /// Gradient computed via AD
    pub ad_grad: f64,
    /// Gradient computed via finite differences
    pub fd_grad: f64,
    /// Relative error between the two
    pub relative_error: f64,
    /// Whether the check passed (rel error < tolerance)
    pub passed: bool,
}

impl GradCheckResult {
    pub fn to_runtime_value(&self) -> RuntimeValue {
        let mut map = HashMap::new();
        map.insert("ad_grad".to_string(), RuntimeValue::Float(self.ad_grad));
        map.insert("fd_grad".to_string(), RuntimeValue::Float(self.fd_grad));
        map.insert(
            "relative_error".to_string(),
            RuntimeValue::Float(self.relative_error),
        );
        map.insert("passed".to_string(), RuntimeValue::Bool(self.passed));
        RuntimeValue::Record(map)
    }
}

/// Result of range gradient check
#[derive(Debug, Clone)]
pub struct GradCheckRangeResult {
    /// Number of points tested
    pub n_points: usize,
    /// Number of points that passed
    pub n_passed: usize,
    /// Maximum relative error observed
    pub max_relative_error: f64,
    /// Whether all points passed
    pub all_passed: bool,
    /// Points where the check failed
    pub failure_points: Vec<f64>,
}

impl GradCheckRangeResult {
    pub fn to_runtime_value(&self) -> RuntimeValue {
        let mut map = HashMap::new();
        map.insert(
            "n_points".to_string(),
            RuntimeValue::Int(self.n_points as i64),
        );
        map.insert(
            "n_passed".to_string(),
            RuntimeValue::Int(self.n_passed as i64),
        );
        map.insert(
            "max_relative_error".to_string(),
            RuntimeValue::Float(self.max_relative_error),
        );
        map.insert(
            "all_passed".to_string(),
            RuntimeValue::Bool(self.all_passed),
        );

        // Convert failure points to vector
        let failures: Vec<RuntimeValue> = self
            .failure_points
            .iter()
            .map(|&x| RuntimeValue::Float(x))
            .collect();
        // Note: We don't have RuntimeValue::Vector, so we'll use a workaround
        // For now, store as a string representation
        let failures_str = format!("{:?}", self.failure_points);
        map.insert(
            "failure_points".to_string(),
            RuntimeValue::String(failures_str),
        );

        RuntimeValue::Record(map)
    }
}

// =============================================================================
// CORE GRADIENT FUNCTIONS
// =============================================================================

/// Compute gradient of a scalar function at a point
///
/// Given f: Float -> Float and x: Float, returns df/dx at x.
///
/// Implementation: Creates a dual number with x and tangent 1,
/// evaluates f(dual), and extracts the tangent.
pub fn compute_grad<F>(f: F, x: f64) -> AdResult<f64>
where
    F: Fn(DualNumber) -> DualNumber,
{
    let x_dual = DualNumber::variable(x);
    let result = f(x_dual);

    if result.is_nan() {
        return Err(AdError::NumericalInstability {
            operation: "grad".to_string(),
            message: "function returned NaN".to_string(),
        });
    }

    Ok(result.tangent)
}

/// Compute both value and gradient at a point
///
/// More efficient than computing value and gradient separately.
pub fn compute_value_and_grad<F>(f: F, x: f64) -> AdResult<GradResult>
where
    F: Fn(DualNumber) -> DualNumber,
{
    let x_dual = DualNumber::variable(x);
    let result = f(x_dual);

    if result.is_nan() {
        return Err(AdError::NumericalInstability {
            operation: "value_and_grad".to_string(),
            message: "function returned NaN".to_string(),
        });
    }

    Ok(GradResult::new(result.primal, result.tangent))
}

/// Compute second derivative using nested AD
///
/// grad2(f, x) = d²f/dx² at x
pub fn compute_grad2<F>(f: F, x: f64) -> AdResult<f64>
where
    F: Fn(DualNumber) -> DualNumber + Clone,
{
    // To compute the second derivative, we need to differentiate the derivative
    // Using forward-mode AD, we evaluate df/dx using dual numbers,
    // then differentiate that result
    //
    // For simple functions, we can use the chain rule directly
    // For a more general implementation, we would need nested dual numbers

    // Approximation using finite differences on the AD gradient
    let h = 1e-5;
    let grad_plus = compute_grad(f.clone(), x + h)?;
    let grad_minus = compute_grad(f, x - h)?;

    Ok((grad_plus - grad_minus) / (2.0 * h))
}

/// Compute n-th derivative
///
/// Uses recursive finite differences on the AD gradient
pub fn compute_grad_n<F>(f: F, x: f64, n: usize) -> AdResult<f64>
where
    F: Fn(DualNumber) -> DualNumber + Clone,
{
    if n == 0 {
        // 0th derivative is the function value
        return Ok(f(DualNumber::constant(x)).primal);
    }

    if n == 1 {
        return compute_grad(f, x);
    }

    // For higher derivatives, use finite differences on lower derivatives
    let h = 1e-4_f64.powf(1.0 / n as f64); // Adaptive step size
    let grad_plus = compute_grad_n(f.clone(), x + h, n - 1)?;
    let grad_minus = compute_grad_n(f, x - h, n - 1)?;

    Ok((grad_plus - grad_minus) / (2.0 * h))
}

// =============================================================================
// PARTIAL DERIVATIVES
// =============================================================================

/// Compute partial derivative with respect to one argument
///
/// For f(x1, x2, ..., xn), computes ∂f/∂xi at the given point.
pub fn compute_partial<F>(f: F, args: &[f64], arg_index: usize) -> AdResult<f64>
where
    F: Fn(&[DualNumber]) -> DualNumber,
{
    if arg_index >= args.len() {
        return Err(AdError::ArityMismatch {
            expected: args.len(),
            found: arg_index + 1,
            function: "partial".to_string(),
        });
    }

    // Create dual numbers: only the arg_index-th one is seeded
    let dual_args: Vec<DualNumber> = args
        .iter()
        .enumerate()
        .map(|(i, &x)| {
            if i == arg_index {
                DualNumber::variable(x)
            } else {
                DualNumber::constant(x)
            }
        })
        .collect();

    let result = f(&dual_args);

    if result.is_nan() {
        return Err(AdError::NumericalInstability {
            operation: "partial".to_string(),
            message: "function returned NaN".to_string(),
        });
    }

    Ok(result.tangent)
}

/// Compute full gradient (all partial derivatives)
pub fn compute_gradient<F>(f: F, args: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&[DualNumber]) -> DualNumber + Clone,
{
    let mut gradient = Vec::with_capacity(args.len());

    for i in 0..args.len() {
        let partial = compute_partial(f.clone(), args, i)?;
        gradient.push(partial);
    }

    Ok(gradient)
}

// =============================================================================
// GRADIENT CHECKING
// =============================================================================

/// Check gradient using finite differences
///
/// Uses central differences: (f(x+h) - f(x-h)) / (2h)
pub fn check_grad<F>(f: F, x: f64, epsilon: Option<f64>) -> AdResult<GradCheckResult>
where
    F: Fn(DualNumber) -> DualNumber + Fn(f64) -> f64,
{
    let h = epsilon.unwrap_or(1e-5);

    // Compute AD gradient
    let ad_grad = compute_grad(|d| f(d), x)?;

    // Compute finite difference gradient
    let f_plus = f(x + h);
    let f_minus = f(x - h);
    let fd_grad = (f_plus - f_minus) / (2.0 * h);

    // Compute relative error
    let abs_diff = (ad_grad - fd_grad).abs();
    let denom = ad_grad.abs().max(fd_grad.abs()).max(1e-8);
    let relative_error = abs_diff / denom;

    // Check if passed (relative error < 1e-4 is typical tolerance)
    let passed = relative_error < 1e-4;

    Ok(GradCheckResult {
        ad_grad,
        fd_grad,
        relative_error,
        passed,
    })
}

/// Check gradient with separate AD and value functions
pub fn check_grad_dual<F, G>(f_dual: F, f_val: G, x: f64, epsilon: Option<f64>) -> GradCheckResult
where
    F: Fn(DualNumber) -> DualNumber,
    G: Fn(f64) -> f64,
{
    let h = epsilon.unwrap_or(1e-5);

    // Compute AD gradient
    let ad_grad = f_dual(DualNumber::variable(x)).tangent;

    // Compute finite difference gradient
    let f_plus = f_val(x + h);
    let f_minus = f_val(x - h);
    let fd_grad = (f_plus - f_minus) / (2.0 * h);

    // Compute relative error
    let abs_diff = (ad_grad - fd_grad).abs();
    let denom = ad_grad.abs().max(fd_grad.abs()).max(1e-8);
    let relative_error = abs_diff / denom;

    let passed = relative_error < 1e-4;

    GradCheckResult {
        ad_grad,
        fd_grad,
        relative_error,
        passed,
    }
}

/// Comprehensive gradient check over a range of points
pub fn check_grad_range<F, G>(
    f_dual: F,
    f_val: G,
    x_min: f64,
    x_max: f64,
    n_points: usize,
) -> GradCheckRangeResult
where
    F: Fn(DualNumber) -> DualNumber,
    G: Fn(f64) -> f64,
{
    let mut max_error = 0.0f64;
    let mut n_passed = 0usize;
    let mut failures = Vec::new();

    for i in 0..n_points {
        let x = if n_points == 1 {
            x_min
        } else {
            x_min + (x_max - x_min) * (i as f64) / ((n_points - 1) as f64)
        };

        let check = check_grad_dual(&f_dual, &f_val, x, None);

        max_error = max_error.max(check.relative_error);
        if check.passed {
            n_passed += 1;
        } else {
            failures.push(x);
        }
    }

    GradCheckRangeResult {
        n_points,
        n_passed,
        max_relative_error: max_error,
        all_passed: n_passed == n_points,
        failure_points: failures,
    }
}

// =============================================================================
// DIRECTIONAL DERIVATIVE
// =============================================================================

/// Compute directional derivative
///
/// The directional derivative of f at x in direction v is:
/// D_v f(x) = ∇f(x) · v = Σ (∂f/∂xi) * vi
pub fn compute_directional_derivative<F>(f: F, point: &[f64], direction: &[f64]) -> AdResult<f64>
where
    F: Fn(&[DualNumber]) -> DualNumber + Clone,
{
    if point.len() != direction.len() {
        return Err(AdError::ArityMismatch {
            expected: point.len(),
            found: direction.len(),
            function: "directional_derivative".to_string(),
        });
    }

    // Compute gradient
    let grad = compute_gradient(f, point)?;

    // Dot product with direction
    let result: f64 = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();

    Ok(result)
}

// =============================================================================
// JACOBIAN (PREP FOR WEEK 51)
// =============================================================================

/// Compute Jacobian matrix for vector-valued function
///
/// For f: R^n -> R^m, the Jacobian is an m×n matrix
/// where J[i][j] = ∂fi/∂xj
pub fn compute_jacobian<F>(f: F, point: &[f64], output_dim: usize) -> AdResult<Vec<Vec<f64>>>
where
    F: Fn(&[DualNumber], usize) -> DualNumber + Clone,
{
    let input_dim = point.len();
    let mut jacobian = vec![vec![0.0; input_dim]; output_dim];

    for j in 0..input_dim {
        // Create dual numbers with j-th component seeded
        let dual_args: Vec<DualNumber> = point
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if i == j {
                    DualNumber::variable(x)
                } else {
                    DualNumber::constant(x)
                }
            })
            .collect();

        // Compute each output component
        for i in 0..output_dim {
            let result = f.clone()(&dual_args, i);
            jacobian[i][j] = result.tangent;
        }
    }

    Ok(jacobian)
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/// Convert RuntimeValue to f64 for AD
pub fn value_to_f64(v: &RuntimeValue) -> Result<f64, RuntimeError> {
    match v {
        RuntimeValue::Float(x) => Ok(*x),
        RuntimeValue::Int(n) => Ok(*n as f64),
        _ => Err(RuntimeError::TypeError {
            expected: "Float or Int".to_string(),
            found: v.runtime_type(),
            message: "Cannot convert to f64 for AD".to_string(),
        }),
    }
}

/// Convert DualNumber result to RuntimeValue
pub fn dual_to_runtime_value(d: DualNumber, extract_gradient: bool) -> RuntimeValue {
    if extract_gradient {
        RuntimeValue::Float(d.tangent)
    } else {
        RuntimeValue::Float(d.primal)
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ad_ops::*;

    const EPSILON: f64 = 1e-6;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_compute_grad_quadratic() {
        // f(x) = x², f'(x) = 2x
        let grad = compute_grad(|d| dual_mul(d, d), 3.0).unwrap();
        assert!(approx_eq(grad, 6.0));
    }

    #[test]
    fn test_compute_grad_exp() {
        // f(x) = exp(x), f'(x) = exp(x)
        let grad = compute_grad(dual_exp, 1.0).unwrap();
        assert!(approx_eq(grad, 1.0_f64.exp()));
    }

    #[test]
    fn test_compute_grad_sin() {
        // f(x) = sin(x), f'(x) = cos(x)
        let grad = compute_grad(dual_sin, 0.5).unwrap();
        assert!(approx_eq(grad, 0.5_f64.cos()));
    }

    #[test]
    fn test_compute_value_and_grad() {
        // f(x) = x², f(2) = 4, f'(2) = 4
        let result = compute_value_and_grad(|d| dual_mul(d, d), 2.0).unwrap();
        assert!(approx_eq(result.value, 4.0));
        assert!(approx_eq(result.gradient, 4.0));
    }

    #[test]
    fn test_compute_grad2() {
        // f(x) = x³, f'(x) = 3x², f''(x) = 6x
        // At x = 2: f''(2) = 12
        let f = |d: DualNumber| dual_mul(dual_mul(d, d), d);
        let grad2 = compute_grad2(f, 2.0).unwrap();
        assert!((grad2 - 12.0).abs() < 0.1); // Relaxed tolerance for finite diff
    }

    #[test]
    fn test_compute_partial() {
        // f(x, y) = x * y, ∂f/∂x = y, ∂f/∂y = x
        let f = |args: &[DualNumber]| dual_mul(args[0], args[1]);

        let partial_x = compute_partial(f, &[3.0, 4.0], 0).unwrap();
        assert!(approx_eq(partial_x, 4.0)); // ∂f/∂x = y = 4

        let partial_y = compute_partial(f, &[3.0, 4.0], 1).unwrap();
        assert!(approx_eq(partial_y, 3.0)); // ∂f/∂y = x = 3
    }

    #[test]
    fn test_compute_gradient() {
        // f(x, y) = x² + y², ∇f = (2x, 2y)
        let f =
            |args: &[DualNumber]| dual_add(dual_mul(args[0], args[0]), dual_mul(args[1], args[1]));

        let grad = compute_gradient(f, &[3.0, 4.0]).unwrap();
        assert_eq!(grad.len(), 2);
        assert!(approx_eq(grad[0], 6.0)); // 2 * 3
        assert!(approx_eq(grad[1], 8.0)); // 2 * 4
    }

    #[test]
    fn test_check_grad() {
        // Test with sin function
        let check = check_grad_dual(dual_sin, |x| x.sin(), 0.5, None);
        assert!(check.passed);
        assert!(check.relative_error < 1e-4);
    }

    #[test]
    fn test_check_grad_range() {
        // Test sin over [0, π]
        let result = check_grad_range(dual_sin, |x| x.sin(), 0.0, std::f64::consts::PI, 10);
        assert!(result.all_passed);
        assert_eq!(result.n_points, 10);
        assert_eq!(result.n_passed, 10);
    }

    #[test]
    fn test_directional_derivative() {
        // f(x, y) = x² + y², at (3, 4) in direction (1, 0)
        // D_(1,0) f = ∂f/∂x = 2x = 6
        let f =
            |args: &[DualNumber]| dual_add(dual_mul(args[0], args[0]), dual_mul(args[1], args[1]));

        let dir_deriv = compute_directional_derivative(f, &[3.0, 4.0], &[1.0, 0.0]).unwrap();
        assert!(approx_eq(dir_deriv, 6.0));
    }

    #[test]
    fn test_grad_composite_function() {
        // f(x) = exp(sin(x)), f'(x) = cos(x) * exp(sin(x))
        let f = |d: DualNumber| dual_exp(dual_sin(d));
        let x = 0.5;
        let grad = compute_grad(f, x).unwrap();

        let expected = x.cos() * x.sin().exp();
        assert!(approx_eq(grad, expected));
    }

    #[test]
    fn test_grad_polynomial() {
        // f(x) = 3x³ - 2x² + x - 5
        // f'(x) = 9x² - 4x + 1
        let f = |d: DualNumber| {
            let three = DualNumber::constant(3.0);
            let two = DualNumber::constant(2.0);
            let one = DualNumber::constant(1.0);
            let five = DualNumber::constant(5.0);

            let x2 = dual_mul(d, d);
            let x3 = dual_mul(x2, d);

            let term1 = dual_mul(three, x3);
            let term2 = dual_mul(two, x2);
            let term3 = dual_mul(one, d);

            dual_sub(dual_add(dual_sub(term1, term2), term3), five)
        };

        let x = 2.0;
        let grad = compute_grad(f, x).unwrap();

        // f'(2) = 9*4 - 4*2 + 1 = 36 - 8 + 1 = 29
        let expected = 9.0 * x * x - 4.0 * x + 1.0;
        assert!(approx_eq(grad, expected));
    }

    #[test]
    fn test_jacobian() {
        // f(x, y) = (x + y, x * y)
        // J = [[1, 1], [y, x]]
        let f = |args: &[DualNumber], i: usize| {
            if i == 0 {
                dual_add(args[0], args[1])
            } else {
                dual_mul(args[0], args[1])
            }
        };

        let jacobian = compute_jacobian(f, &[3.0, 4.0], 2).unwrap();

        assert_eq!(jacobian.len(), 2);
        assert_eq!(jacobian[0].len(), 2);

        // First row: [1, 1]
        assert!(approx_eq(jacobian[0][0], 1.0));
        assert!(approx_eq(jacobian[0][1], 1.0));

        // Second row: [y, x] = [4, 3]
        assert!(approx_eq(jacobian[1][0], 4.0));
        assert!(approx_eq(jacobian[1][1], 3.0));
    }
}
