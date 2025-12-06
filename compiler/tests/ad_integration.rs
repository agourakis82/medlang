//! Integration tests for Week 50-51 Automatic Differentiation System
//!
//! Tests the comprehensive AD implementation including:
//! - Core dual number operations
//! - Scalar operations (45+)
//! - Vector operations
//! - High-level builtins (grad, jacobian, hessian, jvp, vjp)
//! - Gradient checking
//! - PK/PD model differentiation

use medlangc::runtime::ad::*;
use std::f64::consts::PI;

const EPSILON: f64 = 1e-8;

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < EPSILON || (a - b).abs() / a.abs().max(b.abs()).max(1.0) < EPSILON
}

fn approx_eq_tol(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() < tol || (a - b).abs() / a.abs().max(b.abs()).max(1.0) < tol
}

// =============================================================================
// CORE DUAL NUMBER TESTS
// =============================================================================

#[test]
fn test_dual_constant() {
    let c = DualNumber::constant(5.0);
    assert_eq!(c.primal, 5.0);
    assert_eq!(c.tangent, 0.0);
}

#[test]
fn test_dual_variable() {
    let x = DualNumber::variable(3.0);
    assert_eq!(x.primal, 3.0);
    assert_eq!(x.tangent, 1.0);
}

#[test]
fn test_dual_arithmetic() {
    let x = DualNumber::variable(2.0);
    let y = DualNumber::constant(3.0);

    // x + 3 at x=2: value=5, deriv=1
    let sum = x.add(&y);
    assert!(approx_eq(sum.primal, 5.0));
    assert!(approx_eq(sum.tangent, 1.0));

    // x * 3 at x=2: value=6, deriv=3
    let prod = x.mul(&y);
    assert!(approx_eq(prod.primal, 6.0));
    assert!(approx_eq(prod.tangent, 3.0));

    // x / 3 at x=2: value=2/3, deriv=1/3
    let quot = x.div(&y).unwrap();
    assert!(approx_eq(quot.primal, 2.0 / 3.0));
    assert!(approx_eq(quot.tangent, 1.0 / 3.0));
}

#[test]
fn test_chain_rule() {
    // f(x) = sin(x²)
    // f'(x) = cos(x²) * 2x
    let x = DualNumber::variable(1.0);
    let x_sq = x.sq();
    let result = x_sq.sin();

    assert!(approx_eq(result.primal, 1.0_f64.sin()));
    assert!(approx_eq(result.tangent, 1.0_f64.cos() * 2.0));
}

#[test]
fn test_product_rule() {
    // f(x) = x * sin(x)
    // f'(x) = sin(x) + x * cos(x)
    let x = DualNumber::variable(PI / 4.0);
    let sin_x = x.sin();
    let result = x.mul(&sin_x);

    let expected_deriv = (PI / 4.0).sin() + (PI / 4.0) * (PI / 4.0).cos();
    assert!(approx_eq(result.tangent, expected_deriv));
}

// =============================================================================
// SCALAR OPERATIONS TESTS
// =============================================================================

#[test]
fn test_exp_ln() {
    let x = DualNumber::variable(2.0);

    // e^x at x=2
    let exp_x = x.exp();
    assert!(approx_eq(exp_x.primal, 2.0_f64.exp()));
    assert!(approx_eq(exp_x.tangent, 2.0_f64.exp()));

    // ln(x) at x=2
    let ln_x = x.ln().unwrap();
    assert!(approx_eq(ln_x.primal, 2.0_f64.ln()));
    assert!(approx_eq(ln_x.tangent, 0.5)); // 1/2
}

#[test]
fn test_trig_functions() {
    let x = DualNumber::variable(PI / 6.0);

    // sin(π/6) = 0.5, cos(π/6) = √3/2
    let sin_x = x.sin();
    assert!(approx_eq(sin_x.primal, 0.5));
    assert!(approx_eq(sin_x.tangent, (PI / 6.0).cos()));

    let cos_x = x.cos();
    assert!(approx_eq(cos_x.primal, (3.0_f64).sqrt() / 2.0));
    assert!(approx_eq(cos_x.tangent, -(PI / 6.0).sin()));
}

#[test]
fn test_hyperbolic_functions() {
    let x = DualNumber::variable(0.0);

    // tanh(0) = 0, tanh'(0) = 1
    let tanh_x = x.tanh();
    assert!(approx_eq(tanh_x.primal, 0.0));
    assert!(approx_eq(tanh_x.tangent, 1.0));

    // sinh(0) = 0, cosh(0) = 1
    let sinh_x = x.sinh();
    let cosh_x = x.cosh();
    assert!(approx_eq(sinh_x.primal, 0.0));
    assert!(approx_eq(cosh_x.primal, 1.0));
}

#[test]
fn test_power_functions() {
    let x = DualNumber::variable(2.0);

    // x² at x=2: value=4, deriv=4
    let sq_x = x.sq();
    assert!(approx_eq(sq_x.primal, 4.0));
    assert!(approx_eq(sq_x.tangent, 4.0));

    // x³ at x=2: value=8, deriv=12
    let cube_x = x.cube();
    assert!(approx_eq(cube_x.primal, 8.0));
    assert!(approx_eq(cube_x.tangent, 12.0));

    // √x at x=4
    let y = DualNumber::variable(4.0);
    let sqrt_y = y.sqrt().unwrap();
    assert!(approx_eq(sqrt_y.primal, 2.0));
    assert!(approx_eq(sqrt_y.tangent, 0.25)); // 1/(2√4)
}

#[test]
fn test_activation_functions() {
    let x = DualNumber::variable(0.0);

    // sigmoid(0) = 0.5, sigmoid'(0) = 0.25
    let sig = x.sigmoid();
    assert!(approx_eq(sig.primal, 0.5));
    assert!(approx_eq(sig.tangent, 0.25));

    // softplus(0) = ln(2), softplus'(0) = sigmoid(0) = 0.5
    let sp = x.softplus();
    assert!(approx_eq(sp.primal, 2.0_f64.ln()));
    assert!(approx_eq(sp.tangent, 0.5));
}

#[test]
fn test_smooth_functions() {
    let x = DualNumber::variable(0.1);

    // Smooth abs near zero
    let abs_smooth = x.abs_smooth(0.001);
    assert!(abs_smooth.primal > 0.0);
    assert!(abs_smooth.tangent > 0.0);

    // Smooth sign
    let sign_smooth = x.sign_smooth(0.1);
    assert!(sign_smooth.primal > 0.0);
    assert!(sign_smooth.primal < 1.0);
}

// =============================================================================
// VECTOR OPERATIONS TESTS
// =============================================================================

#[test]
fn test_dual_vector_basic() {
    let v = DualVector::variables(&[1.0, 2.0, 3.0]);
    assert_eq!(v.len(), 3);

    let primals = v.primals();
    assert!(approx_eq(primals[0], 1.0));
    assert!(approx_eq(primals[1], 2.0));
    assert!(approx_eq(primals[2], 3.0));
}

#[test]
fn test_dot_product() {
    let a = DualVector::new(vec![DualNumber::new(1.0, 1.0), DualNumber::new(2.0, 0.0)]);
    let b = DualVector::new(vec![DualNumber::new(3.0, 0.0), DualNumber::new(4.0, 1.0)]);

    // dot = 1*3 + 2*4 = 11
    // tangent = 1*3 + 2*1 = 5
    let dot = a.dot(&b).unwrap();
    assert!(approx_eq(dot.primal, 11.0));
    assert!(approx_eq(dot.tangent, 5.0));
}

#[test]
fn test_vector_norm() {
    let v = DualVector::variables(&[3.0, 4.0]);
    let norm = v.norm_l2().unwrap();
    assert!(approx_eq(norm.primal, 5.0)); // sqrt(9 + 16) = 5
}

#[test]
fn test_softmax() {
    let v = DualVector::constants(&[1.0, 2.0, 3.0]);
    let sm = v.softmax().unwrap();

    // Softmax sums to 1
    let sum: f64 = sm.primals().iter().sum();
    assert!(approx_eq(sum, 1.0));

    // Highest input has highest probability
    assert!(sm.primals()[2] > sm.primals()[1]);
    assert!(sm.primals()[1] > sm.primals()[0]);
}

// =============================================================================
// HIGH-LEVEL BUILTINS TESTS
// =============================================================================

#[test]
fn test_grad_scalar() {
    // f(x, y) = x² + y²
    // ∇f = [2x, 2y]
    let f = |v: &DualVector| -> AdResult<DualNumber> {
        Ok(v.elements[0].sq().add(&v.elements[1].sq()))
    };

    let gradient = grad(f, &[3.0, 4.0]).unwrap();
    assert!(approx_eq(gradient[0], 6.0)); // 2 * 3
    assert!(approx_eq(gradient[1], 8.0)); // 2 * 4
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
fn test_jacobian() {
    // f(x, y) = [x + y, x * y]
    // J = [[1, 1], [y, x]]
    let f = |v: &DualVector| -> AdResult<DualVector> {
        Ok(DualVector::new(vec![
            v.elements[0].add(&v.elements[1]),
            v.elements[0].mul(&v.elements[1]),
        ]))
    };

    let jac = jacobian(f, &[2.0, 3.0], 2).unwrap();

    // J = [[1, 1], [3, 2]] flattened row-major
    assert!(approx_eq(jac[0], 1.0)); // ∂(x+y)/∂x
    assert!(approx_eq(jac[1], 1.0)); // ∂(x+y)/∂y
    assert!(approx_eq(jac[2], 3.0)); // ∂(xy)/∂x = y
    assert!(approx_eq(jac[3], 2.0)); // ∂(xy)/∂y = x
}

#[test]
fn test_jvp() {
    // f(x, y) = [x², y²]
    // J = [[2x, 0], [0, 2y]]
    let f = |v: &DualVector| -> AdResult<DualVector> {
        Ok(DualVector::new(vec![
            v.elements[0].sq(),
            v.elements[1].sq(),
        ]))
    };

    let point = [2.0, 3.0];
    let tangent = [1.0, 1.0];
    let result = jvp(f, &point, &tangent).unwrap();

    // JVP = [[4, 0], [0, 6]] @ [1, 1] = [4, 6]
    assert!(approx_eq(result[0], 4.0));
    assert!(approx_eq(result[1], 6.0));
}

#[test]
fn test_vjp_prepare() {
    // f(x, y) = [x + y, x * y]
    // J = [[1, 1], [y, x]]
    // J^T = [[1, y], [1, x]]
    let f = |v: &DualVector| -> AdResult<DualVector> {
        Ok(DualVector::new(vec![
            v.elements[0].add(&v.elements[1]),
            v.elements[0].mul(&v.elements[1]),
        ]))
    };

    let vjp = vjp_prepare(f, &[2.0, 3.0], 2).unwrap();
    let result = vjp.apply(&[1.0, 1.0]).unwrap();

    // J^T @ [1, 1] = [1 + 3, 1 + 2] = [4, 3]
    assert!(approx_eq(result[0], 4.0));
    assert!(approx_eq(result[1], 3.0));
}

#[test]
fn test_partial_derivative() {
    // f(x, y, z) = x² + y³ + z
    let f = |v: &DualVector| -> AdResult<DualNumber> {
        Ok(v.elements[0]
            .sq()
            .add(&v.elements[1].cube())
            .add(&v.elements[2]))
    };

    let point = [2.0, 3.0, 4.0];

    // ∂f/∂x = 2x = 4
    let dx = partial(&f, &point, 0).unwrap();
    assert!(approx_eq(dx, 4.0));

    // ∂f/∂y = 3y² = 27
    let dy = partial(&f, &point, 1).unwrap();
    assert!(approx_eq(dy, 27.0));

    // ∂f/∂z = 1
    let dz = partial(&f, &point, 2).unwrap();
    assert!(approx_eq(dz, 1.0));
}

#[test]
fn test_value_and_grad() {
    // f(x, y) = x² + y²
    let f = |v: &DualVector| -> AdResult<DualNumber> {
        Ok(v.elements[0].sq().add(&v.elements[1].sq()))
    };

    let (value, gradient) = value_and_grad(f, &[3.0, 4.0]).unwrap();

    assert!(approx_eq(value, 25.0)); // 9 + 16
    assert!(approx_eq(gradient[0], 6.0));
    assert!(approx_eq(gradient[1], 8.0));
}

// =============================================================================
// GRADIENT CHECKING TESTS
// =============================================================================

#[test]
fn test_grad_check_pass() {
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
fn test_grad_check_complex() {
    // f(x, y) = sin(x) * exp(y)
    let f_dual = |v: &DualVector| -> AdResult<DualNumber> {
        Ok(v.elements[0].sin().mul(&v.elements[1].exp()))
    };
    let f_scalar = |x: &[f64]| x[0].sin() * x[1].exp();

    let result = grad_check(f_dual, f_scalar, &[1.0, 0.5], 1e-7, 1e-5).unwrap();
    assert!(result.passed);
}

// =============================================================================
// PK/PD MODEL TESTS
// =============================================================================

#[test]
fn test_pk_first_order_elimination() {
    // C(t) = C0 * exp(-k*t)
    // ∂C/∂C0 = exp(-k*t)
    // ∂C/∂k = -C0 * t * exp(-k*t)
    // ∂C/∂t = -C0 * k * exp(-k*t)

    let pk_model = |v: &DualVector| -> AdResult<DualNumber> {
        let c0 = &v.elements[0];
        let k = &v.elements[1];
        let t = &v.elements[2];
        Ok(c0.mul(&k.neg().mul(t).exp()))
    };

    let point = [100.0, 0.1, 10.0]; // C0=100, k=0.1, t=10
    let gradient = grad(pk_model, &point).unwrap();

    let exp_kt = (-0.1 * 10.0_f64).exp(); // exp(-1)

    // ∂C/∂C0 = exp(-k*t)
    assert!(approx_eq(gradient[0], exp_kt));

    // ∂C/∂k = -C0 * t * exp(-k*t) = -1000 * exp(-1)
    assert!(approx_eq(gradient[1], -100.0 * 10.0 * exp_kt));

    // ∂C/∂t = -C0 * k * exp(-k*t) = -10 * exp(-1)
    assert!(approx_eq(gradient[2], -100.0 * 0.1 * exp_kt));
}

#[test]
fn test_hill_function() {
    // Hill(x, K, n) = x^n / (K^n + x^n)
    // At x=K: Hill = K^n / (2*K^n) = 0.5

    let hill_model = |v: &DualVector| -> AdResult<DualNumber> {
        let x = &v.elements[0];
        let k = &v.elements[1];
        let n = &v.elements[2];

        let x_n = x.pow(n)?;
        let k_n = k.pow(n)?;
        x_n.div(&x_n.add(&k_n))
    };

    // At x=K=1, n=1: Hill = 0.5
    let point = [1.0, 1.0, 1.0];
    let (value, gradient) = value_and_grad(hill_model, &point).unwrap();

    assert!(approx_eq(value, 0.5));

    // ∂Hill/∂x at x=K=1, n=1 = K^n * n * x^(n-1) / (K^n + x^n)² = 1 * 1 * 1 / 4 = 0.25
    assert!(approx_eq_tol(gradient[0], 0.25, 1e-6));
}

#[test]
fn test_michaelis_menten() {
    // MM(S, Vmax, Km) = Vmax * S / (Km + S)

    let mm_model = |v: &DualVector| -> AdResult<DualNumber> {
        let s = &v.elements[0];
        let vmax = &v.elements[1];
        let km = &v.elements[2];
        vmax.mul(s).div(&km.add(s))
    };

    // At S=Km: V = Vmax * Km / (2*Km) = Vmax / 2
    let point = [10.0, 100.0, 10.0]; // S=10, Vmax=100, Km=10
    let (value, gradient) = value_and_grad(mm_model, &point).unwrap();

    assert!(approx_eq(value, 50.0)); // 100 * 10 / 20 = 50

    // ∂V/∂S = Vmax * Km / (Km + S)² = 100 * 10 / 400 = 2.5
    assert!(approx_eq(gradient[0], 2.5));

    // ∂V/∂Vmax = S / (Km + S) = 10 / 20 = 0.5
    assert!(approx_eq(gradient[1], 0.5));

    // ∂V/∂Km = -Vmax * S / (Km + S)² = -100 * 10 / 400 = -2.5
    assert!(approx_eq(gradient[2], -2.5));
}

// =============================================================================
// ML LOSS FUNCTION TESTS
// =============================================================================

#[test]
fn test_mse_loss() {
    // MSE = (1/n) Σ (y - ŷ)²
    let y_true = DualVector::constants(&[1.0, 2.0, 3.0]);
    let y_pred = DualVector::variables(&[1.0, 2.0, 4.0]);

    // MSE = (0 + 0 + 1) / 3 = 1/3
    let loss = ops_vector::mse_loss(&y_true, &y_pred).unwrap();
    assert!(approx_eq(loss.primal, 1.0 / 3.0));
}

#[test]
fn test_linear_regression_gradient() {
    // Loss = (y - (w*x + b))²
    // ∂L/∂w = -2 * (y - w*x - b) * x
    // ∂L/∂b = -2 * (y - w*x - b)

    let loss = |params: &DualVector| -> AdResult<DualNumber> {
        let w = &params.elements[0];
        let b = &params.elements[1];

        // Fixed data: x=2, y=5
        let x = DualNumber::constant(2.0);
        let y_true = DualNumber::constant(5.0);

        let y_pred = w.mul(&x).add(b);
        let error = y_true.sub(&y_pred);
        Ok(error.sq())
    };

    // At w=1, b=1: y_pred = 3, error = 2, loss = 4
    let point = [1.0, 1.0];
    let (value, gradient) = value_and_grad(loss, &point).unwrap();

    assert!(approx_eq(value, 4.0));

    // ∂L/∂w = -2 * 2 * 2 = -8
    assert!(approx_eq(gradient[0], -8.0));

    // ∂L/∂b = -2 * 2 = -4
    assert!(approx_eq(gradient[1], -4.0));
}

// =============================================================================
// HIGHER-ORDER DERIVATIVE TESTS
// =============================================================================

#[test]
fn test_laplacian() {
    // f(x, y) = x² + y²
    // Laplacian = ∂²f/∂x² + ∂²f/∂y² = 2 + 2 = 4
    let f = |v: &DualVector| -> AdResult<DualNumber> {
        Ok(v.elements[0].sq().add(&v.elements[1].sq()))
    };

    let lap = laplacian(f, &[1.0, 1.0], 1e-5).unwrap();
    assert!(approx_eq_tol(lap, 4.0, 0.01));
}

#[test]
fn test_divergence() {
    // F(x, y) = [x, y]
    // div(F) = ∂x/∂x + ∂y/∂y = 1 + 1 = 2
    let f = |v: &DualVector| -> AdResult<DualVector> { Ok(v.clone()) };

    let div = divergence(f, &[1.0, 2.0]).unwrap();
    assert!(approx_eq(div, 2.0));
}

// =============================================================================
// DUAL RECORD TESTS
// =============================================================================

#[test]
fn test_dual_record_basic() {
    let pairs = [("alpha", 0.5), ("beta", 0.3)];
    let record = DualRecord::from_constants(&pairs);

    assert_eq!(record.len(), 2);
    assert!(record.contains("alpha"));
    assert!(approx_eq(record.get("alpha").unwrap().primal, 0.5));
}

#[test]
fn test_dual_record_seeded() {
    let pairs = [("k", 1.0), ("v", 2.0)];
    let record = DualRecord::seeded_at(&pairs, "k");

    // k is seeded (tangent = 1), v is constant (tangent = 0)
    assert!(approx_eq(record.get("k").unwrap().tangent, 1.0));
    assert!(approx_eq(record.get("v").unwrap().tangent, 0.0));
}

// =============================================================================
// AD CONTEXT TESTS
// =============================================================================

#[test]
fn test_ad_context_tracing() {
    let mut ctx = AdContext::with_tracing();
    let x = DualNumber::variable(2.0);
    let y = x.sq();

    ctx.trace("square", &[&x], &y);
    assert_eq!(ctx.get_trace().len(), 1);
    assert_eq!(ctx.get_trace()[0].operation, "square");
}

#[test]
fn test_gradient_checking_context() {
    let ctx = AdContext::with_gradient_checking(1e-7);

    // Test f(x) = x² at x=3, f'(x) = 6
    let result = ctx.check_gradient(|x| x * x, 3.0, 6.0);
    assert!(result);

    // Wrong derivative should fail
    let result = ctx.check_gradient(|x| x * x, 3.0, 5.0);
    assert!(!result);
}

// =============================================================================
// ERROR HANDLING TESTS
// =============================================================================

#[test]
fn test_division_by_zero() {
    let x = DualNumber::variable(1.0);
    let zero = DualNumber::constant(0.0);
    assert!(x.div(&zero).is_err());
}

#[test]
fn test_log_domain_error() {
    let neg = DualNumber::constant(-1.0);
    assert!(neg.ln().is_err());
}

#[test]
fn test_sqrt_domain_error() {
    let neg = DualNumber::constant(-1.0);
    assert!(neg.sqrt().is_err());
}

#[test]
fn test_dimension_mismatch() {
    let a = DualVector::constants(&[1.0, 2.0]);
    let b = DualVector::constants(&[1.0, 2.0, 3.0]);
    assert!(a.add(&b).is_err());
}

// =============================================================================
// COMPLEX COMPOSITION TESTS
// =============================================================================

#[test]
fn test_neural_network_layer() {
    // Single neuron: y = sigmoid(w·x + b)
    let forward = |params: &DualVector| -> AdResult<DualNumber> {
        // params = [w1, w2, b], input = [0.5, 0.5]
        let w1 = &params.elements[0];
        let w2 = &params.elements[1];
        let b = &params.elements[2];

        let x1 = DualNumber::constant(0.5);
        let x2 = DualNumber::constant(0.5);

        // Linear: w1*x1 + w2*x2 + b
        let linear = w1.mul(&x1).add(&w2.mul(&x2)).add(b);
        // Activation: sigmoid
        Ok(linear.sigmoid())
    };

    let params = [1.0, 1.0, 0.0]; // w1=1, w2=1, b=0
    let (output, gradient) = value_and_grad(forward, &params).unwrap();

    // Linear output = 0.5 + 0.5 = 1.0
    // sigmoid(1.0) ≈ 0.731
    assert!(approx_eq_tol(
        output,
        1.0_f64 / (1.0 + (-1.0_f64).exp()),
        1e-6
    ));

    // All gradients should be non-zero
    assert!(gradient[0].abs() > 1e-10);
    assert!(gradient[1].abs() > 1e-10);
    assert!(gradient[2].abs() > 1e-10);
}

#[test]
fn test_rosenbrock_gradient() {
    // Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    // ∂f/∂x = -2(1-x) - 400x(y-x²)
    // ∂f/∂y = 200(y-x²)
    let rosenbrock = |v: &DualVector| -> AdResult<DualNumber> {
        let x = &v.elements[0];
        let y = &v.elements[1];
        let one = DualNumber::constant(1.0);
        let hundred = DualNumber::constant(100.0);

        let term1 = one.sub(x).sq();
        let term2 = hundred.mul(&y.sub(&x.sq()).sq());
        Ok(term1.add(&term2))
    };

    // At (1, 1): f = 0, ∇f = [0, 0] (minimum)
    let gradient = grad(rosenbrock, &[1.0, 1.0]).unwrap();
    assert!(approx_eq_tol(gradient[0], 0.0, 1e-6));
    assert!(approx_eq_tol(gradient[1], 0.0, 1e-6));

    // At (0, 0): f = 1, ∇f = [-2, 0]
    let gradient = grad(rosenbrock, &[0.0, 0.0]).unwrap();
    assert!(approx_eq_tol(gradient[0], -2.0, 1e-6));
    assert!(approx_eq_tol(gradient[1], 0.0, 1e-6));
}
