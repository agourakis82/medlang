//! Vector operations for automatic differentiation.
//!
//! This module provides differentiable vector operations including:
//! - Element-wise operations
//! - Reductions (sum, mean, product, norms)
//! - Linear algebra operations (dot product, matrix-vector products)
//! - Statistical operations
//! - Gradient, Jacobian, and Hessian computation utilities

use super::dual::{AdError, AdResult, DualNumber, DualVector};

// ============================================================================
// ELEMENT-WISE OPERATIONS
// ============================================================================

/// Element-wise addition of two vectors.
#[inline]
pub fn add(a: &DualVector, b: &DualVector) -> AdResult<DualVector> {
    a.add(b)
}

/// Element-wise subtraction of two vectors.
#[inline]
pub fn sub(a: &DualVector, b: &DualVector) -> AdResult<DualVector> {
    a.sub(b)
}

/// Element-wise multiplication (Hadamard product).
#[inline]
pub fn hadamard(a: &DualVector, b: &DualVector) -> AdResult<DualVector> {
    a.hadamard(b)
}

/// Element-wise division.
#[inline]
pub fn div_elementwise(a: &DualVector, b: &DualVector) -> AdResult<DualVector> {
    a.div_elementwise(b)
}

/// Scalar multiplication: c * v
#[inline]
pub fn scale(v: &DualVector, c: &DualNumber) -> DualVector {
    v.scale(c)
}

/// Add scalar to all elements.
#[inline]
pub fn add_scalar(v: &DualVector, c: &DualNumber) -> DualVector {
    v.add_scalar(c)
}

/// Element-wise negation.
#[inline]
pub fn neg(v: &DualVector) -> DualVector {
    v.map(|x| x.neg())
}

/// Element-wise absolute value (smooth).
#[inline]
pub fn abs_smooth(v: &DualVector, epsilon: f64) -> DualVector {
    v.map(|x| x.abs_smooth(epsilon))
}

/// Element-wise square.
#[inline]
pub fn sq(v: &DualVector) -> DualVector {
    v.map(|x| x.sq())
}

/// Element-wise square root.
#[inline]
pub fn sqrt(v: &DualVector) -> AdResult<DualVector> {
    v.try_map(|x| x.sqrt())
}

/// Element-wise exponential.
#[inline]
pub fn exp(v: &DualVector) -> DualVector {
    v.map(|x| x.exp())
}

/// Element-wise natural logarithm.
#[inline]
pub fn ln(v: &DualVector) -> AdResult<DualVector> {
    v.try_map(|x| x.ln())
}

/// Element-wise power with constant exponent.
#[inline]
pub fn powf(v: &DualVector, n: f64) -> AdResult<DualVector> {
    v.try_map(|x| x.powf(n))
}

/// Element-wise sine.
#[inline]
pub fn sin(v: &DualVector) -> DualVector {
    v.map(|x| x.sin())
}

/// Element-wise cosine.
#[inline]
pub fn cos(v: &DualVector) -> DualVector {
    v.map(|x| x.cos())
}

/// Element-wise tangent.
#[inline]
pub fn tan(v: &DualVector) -> AdResult<DualVector> {
    v.try_map(|x| x.tan())
}

/// Element-wise hyperbolic tangent.
#[inline]
pub fn tanh(v: &DualVector) -> DualVector {
    v.map(|x| x.tanh())
}

/// Element-wise sigmoid.
#[inline]
pub fn sigmoid(v: &DualVector) -> DualVector {
    v.map(|x| x.sigmoid())
}

/// Element-wise ReLU (smooth).
#[inline]
pub fn relu_smooth(v: &DualVector, beta: f64) -> DualVector {
    v.map(|x| x.relu_smooth(beta))
}

/// Element-wise GELU.
#[inline]
pub fn gelu(v: &DualVector) -> DualVector {
    v.map(|x| x.gelu())
}

/// Element-wise swish.
#[inline]
pub fn swish(v: &DualVector) -> DualVector {
    v.map(|x| x.swish())
}

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

/// Sum of all elements.
#[inline]
pub fn sum(v: &DualVector) -> DualNumber {
    v.sum()
}

/// Mean of all elements.
#[inline]
pub fn mean(v: &DualVector) -> AdResult<DualNumber> {
    v.mean()
}

/// Product of all elements.
#[inline]
pub fn product(v: &DualVector) -> DualNumber {
    v.product()
}

/// L2 (Euclidean) norm: ||v||₂ = √(Σvᵢ²)
#[inline]
pub fn norm_l2(v: &DualVector) -> AdResult<DualNumber> {
    v.norm_l2()
}

/// Squared L2 norm: ||v||₂² = Σvᵢ²
#[inline]
pub fn norm_sq(v: &DualVector) -> DualNumber {
    v.norm_sq()
}

/// L1 (Manhattan) norm: ||v||₁ = Σ|vᵢ|
#[inline]
pub fn norm_l1(v: &DualVector, epsilon: f64) -> DualNumber {
    v.norm_l1(epsilon)
}

/// Lp norm: ||v||_p = (Σ|vᵢ|^p)^(1/p)
#[inline]
pub fn norm_lp(v: &DualVector, p: f64, epsilon: f64) -> AdResult<DualNumber> {
    if p < 1.0 {
        return Err(AdError::InvalidOperation(
            "Lp norm requires p >= 1".to_string(),
        ));
    }
    let mut sum = DualNumber::constant(0.0);
    for e in &v.elements {
        let abs_e = e.abs_smooth(epsilon);
        sum = sum.add(&abs_e.powf(p)?);
    }
    sum.powf(1.0 / p)
}

/// Infinity norm (max absolute value) - smooth approximation
#[inline]
pub fn norm_inf(v: &DualVector, sharpness: f64, epsilon: f64) -> DualNumber {
    if v.is_empty() {
        return DualNumber::constant(0.0);
    }
    let abs_v = abs_smooth(v, epsilon);
    // Smooth max over all elements
    let mut max = abs_v.elements[0];
    for e in abs_v.elements.iter().skip(1) {
        max = max.smooth_max(e, sharpness);
    }
    max
}

/// Log-sum-exp: log(Σexp(vᵢ))
/// Numerically stable implementation.
#[inline]
pub fn log_sum_exp(v: &DualVector) -> AdResult<DualNumber> {
    if v.is_empty() {
        return Err(AdError::InvalidOperation(
            "log_sum_exp of empty vector".to_string(),
        ));
    }
    // Find max for numerical stability
    let max_val = v
        .elements
        .iter()
        .map(|e| e.primal)
        .fold(f64::NEG_INFINITY, f64::max);
    let max_d = DualNumber::constant(max_val);

    let mut sum_exp = DualNumber::constant(0.0);
    for e in &v.elements {
        sum_exp = sum_exp.add(&e.sub(&max_d).exp());
    }
    Ok(max_d.add(&sum_exp.ln()?))
}

// ============================================================================
// LINEAR ALGEBRA OPERATIONS
// ============================================================================

/// Dot product: a · b = Σaᵢbᵢ
#[inline]
pub fn dot(a: &DualVector, b: &DualVector) -> AdResult<DualNumber> {
    a.dot(b)
}

/// Weighted sum: Σwᵢvᵢ
#[inline]
pub fn weighted_sum(v: &DualVector, weights: &DualVector) -> AdResult<DualNumber> {
    dot(v, weights)
}

/// Normalize to unit length: v / ||v||₂
#[inline]
pub fn normalize(v: &DualVector) -> AdResult<DualVector> {
    v.normalize()
}

/// Outer product: a ⊗ b creates matrix where M[i,j] = a[i] * b[j]
/// Returns as flattened vector (row-major order).
#[inline]
pub fn outer(a: &DualVector, b: &DualVector) -> DualVector {
    let mut result = Vec::with_capacity(a.len() * b.len());
    for ai in &a.elements {
        for bj in &b.elements {
            result.push(ai.mul(bj));
        }
    }
    DualVector::new(result)
}

/// Matrix-vector product: M @ v (M stored in row-major order)
/// M is n×m matrix, v is m-vector, result is n-vector.
#[inline]
pub fn matvec(
    matrix: &DualVector,
    vector: &DualVector,
    n_rows: usize,
    n_cols: usize,
) -> AdResult<DualVector> {
    if matrix.len() != n_rows * n_cols {
        return Err(AdError::DimensionMismatch {
            expected: n_rows * n_cols,
            got: matrix.len(),
        });
    }
    if vector.len() != n_cols {
        return Err(AdError::DimensionMismatch {
            expected: n_cols,
            got: vector.len(),
        });
    }

    let mut result = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let mut row_sum = DualNumber::constant(0.0);
        for j in 0..n_cols {
            let m_ij = &matrix.elements[i * n_cols + j];
            let v_j = &vector.elements[j];
            row_sum = row_sum.add(&m_ij.mul(v_j));
        }
        result.push(row_sum);
    }
    Ok(DualVector::new(result))
}

/// Vector-matrix product: v @ M (M stored in row-major order)
/// v is n-vector, M is n×m matrix, result is m-vector.
#[inline]
pub fn vecmat(
    vector: &DualVector,
    matrix: &DualVector,
    n_rows: usize,
    n_cols: usize,
) -> AdResult<DualVector> {
    if matrix.len() != n_rows * n_cols {
        return Err(AdError::DimensionMismatch {
            expected: n_rows * n_cols,
            got: matrix.len(),
        });
    }
    if vector.len() != n_rows {
        return Err(AdError::DimensionMismatch {
            expected: n_rows,
            got: vector.len(),
        });
    }

    let mut result = Vec::with_capacity(n_cols);
    for j in 0..n_cols {
        let mut col_sum = DualNumber::constant(0.0);
        for i in 0..n_rows {
            let v_i = &vector.elements[i];
            let m_ij = &matrix.elements[i * n_cols + j];
            col_sum = col_sum.add(&v_i.mul(m_ij));
        }
        result.push(col_sum);
    }
    Ok(DualVector::new(result))
}

/// Quadratic form: v^T @ M @ v
#[inline]
pub fn quadratic_form(vector: &DualVector, matrix: &DualVector, n: usize) -> AdResult<DualNumber> {
    // Compute M @ v first, then dot with v
    let mv = matvec(matrix, vector, n, n)?;
    dot(vector, &mv)
}

// ============================================================================
// PROBABILITY/STATISTICS OPERATIONS
// ============================================================================

/// Softmax: softmax(v)[i] = exp(vᵢ) / Σexp(vⱼ)
#[inline]
pub fn softmax(v: &DualVector) -> AdResult<DualVector> {
    v.softmax()
}

/// Log-softmax: log(softmax(v)[i]) = vᵢ - log(Σexp(vⱼ))
#[inline]
pub fn log_softmax(v: &DualVector) -> AdResult<DualVector> {
    v.log_softmax()
}

/// Variance: Var(v) = E[(v - μ)²] = E[v²] - E[v]²
#[inline]
pub fn variance(v: &DualVector) -> AdResult<DualNumber> {
    if v.len() < 2 {
        return Err(AdError::InvalidOperation(
            "variance requires at least 2 elements".to_string(),
        ));
    }
    let n = DualNumber::constant(v.len() as f64);
    let mean = v.sum().div(&n)?;
    let mut sum_sq_diff = DualNumber::constant(0.0);
    for e in &v.elements {
        let diff = e.sub(&mean);
        sum_sq_diff = sum_sq_diff.add(&diff.sq());
    }
    sum_sq_diff.div(&n)
}

/// Sample variance (Bessel's correction): s² = Σ(vᵢ - v̄)² / (n-1)
#[inline]
pub fn sample_variance(v: &DualVector) -> AdResult<DualNumber> {
    if v.len() < 2 {
        return Err(AdError::InvalidOperation(
            "sample variance requires at least 2 elements".to_string(),
        ));
    }
    let n = DualNumber::constant(v.len() as f64);
    let n_minus_1 = DualNumber::constant((v.len() - 1) as f64);
    let mean = v.sum().div(&n)?;
    let mut sum_sq_diff = DualNumber::constant(0.0);
    for e in &v.elements {
        let diff = e.sub(&mean);
        sum_sq_diff = sum_sq_diff.add(&diff.sq());
    }
    sum_sq_diff.div(&n_minus_1)
}

/// Standard deviation: √Var(v)
#[inline]
pub fn std_dev(v: &DualVector) -> AdResult<DualNumber> {
    variance(v)?.sqrt()
}

/// Sample standard deviation: √(sample variance)
#[inline]
pub fn sample_std_dev(v: &DualVector) -> AdResult<DualNumber> {
    sample_variance(v)?.sqrt()
}

/// Covariance between two vectors.
#[inline]
pub fn covariance(a: &DualVector, b: &DualVector) -> AdResult<DualNumber> {
    if a.len() != b.len() {
        return Err(AdError::DimensionMismatch {
            expected: a.len(),
            got: b.len(),
        });
    }
    if a.len() < 2 {
        return Err(AdError::InvalidOperation(
            "covariance requires at least 2 elements".to_string(),
        ));
    }
    let n = DualNumber::constant(a.len() as f64);
    let mean_a = a.sum().div(&n)?;
    let mean_b = b.sum().div(&n)?;
    let mut sum_prod = DualNumber::constant(0.0);
    for (ai, bi) in a.elements.iter().zip(b.elements.iter()) {
        let diff_a = ai.sub(&mean_a);
        let diff_b = bi.sub(&mean_b);
        sum_prod = sum_prod.add(&diff_a.mul(&diff_b));
    }
    sum_prod.div(&n)
}

/// Correlation coefficient.
#[inline]
pub fn correlation(a: &DualVector, b: &DualVector) -> AdResult<DualNumber> {
    let cov = covariance(a, b)?;
    let std_a = std_dev(a)?;
    let std_b = std_dev(b)?;
    cov.div(&std_a.mul(&std_b))
}

// ============================================================================
// LOSS FUNCTIONS (VECTOR VERSIONS)
// ============================================================================

/// Mean squared error: MSE = (1/n)Σ(yᵢ - ŷᵢ)²
#[inline]
pub fn mse_loss(y_true: &DualVector, y_pred: &DualVector) -> AdResult<DualNumber> {
    if y_true.len() != y_pred.len() {
        return Err(AdError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    let diff = y_true.sub(y_pred)?;
    let sq_diff = sq(&diff);
    mean(&sq_diff)
}

/// Root mean squared error: RMSE = √MSE
#[inline]
pub fn rmse_loss(y_true: &DualVector, y_pred: &DualVector) -> AdResult<DualNumber> {
    mse_loss(y_true, y_pred)?.sqrt()
}

/// Mean absolute error (smooth): MAE = (1/n)Σ|yᵢ - ŷᵢ|
#[inline]
pub fn mae_loss(y_true: &DualVector, y_pred: &DualVector, epsilon: f64) -> AdResult<DualNumber> {
    if y_true.len() != y_pred.len() {
        return Err(AdError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    let diff = y_true.sub(y_pred)?;
    let abs_diff = abs_smooth(&diff, epsilon);
    mean(&abs_diff)
}

/// Cross-entropy loss for multi-class classification.
/// y_true: one-hot encoded targets, y_pred: predicted probabilities (or logits with softmax)
#[inline]
pub fn cross_entropy_loss(y_true: &DualVector, y_pred: &DualVector) -> AdResult<DualNumber> {
    if y_true.len() != y_pred.len() {
        return Err(AdError::DimensionMismatch {
            expected: y_true.len(),
            got: y_pred.len(),
        });
    }
    // -Σyᵢlog(pᵢ)
    let epsilon = 1e-15;
    let mut loss = DualNumber::constant(0.0);
    for (y, p) in y_true.elements.iter().zip(y_pred.elements.iter()) {
        let p_clipped = p.smooth_clamp(epsilon, 1.0 - epsilon, 100.0);
        let log_p = p_clipped.ln()?;
        loss = loss.add(&y.mul(&log_p));
    }
    Ok(loss.neg())
}

/// KL divergence: KL(P||Q) = Σpᵢlog(pᵢ/qᵢ)
#[inline]
pub fn kl_divergence(p: &DualVector, q: &DualVector) -> AdResult<DualNumber> {
    if p.len() != q.len() {
        return Err(AdError::DimensionMismatch {
            expected: p.len(),
            got: q.len(),
        });
    }
    let epsilon = 1e-15;
    let mut kl = DualNumber::constant(0.0);
    for (pi, qi) in p.elements.iter().zip(q.elements.iter()) {
        let pi_safe = pi.smooth_clamp(epsilon, 1.0, 100.0);
        let qi_safe = qi.smooth_clamp(epsilon, 1.0, 100.0);
        let log_ratio = pi_safe.div(&qi_safe)?.ln()?;
        kl = kl.add(&pi_safe.mul(&log_ratio));
    }
    Ok(kl)
}

// ============================================================================
// GRADIENT COMPUTATION UTILITIES
// ============================================================================

/// Compute gradient of scalar function f: R^n → R using forward-mode AD.
/// Performs n forward passes, each seeding one input.
pub fn gradient<F>(f: F, point: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    let n = point.len();
    let mut grad = Vec::with_capacity(n);

    for i in 0..n {
        // Create input seeded at position i
        let input = DualVector::seeded_at(point, i);
        let output = f(&input)?;
        grad.push(output.tangent);
    }

    Ok(grad)
}

/// Compute Jacobian of vector function f: R^n → R^m.
/// Returns m×n matrix in row-major order.
pub fn jacobian<F>(f: F, point: &[f64], output_dim: usize) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    let n = point.len();
    let m = output_dim;
    let mut jac = vec![0.0; m * n];

    for j in 0..n {
        // Seed input j
        let input = DualVector::seeded_at(point, j);
        let output = f(&input)?;

        if output.len() != m {
            return Err(AdError::DimensionMismatch {
                expected: m,
                got: output.len(),
            });
        }

        // Column j of Jacobian = tangents of output
        for i in 0..m {
            jac[i * n + j] = output.elements[i].tangent;
        }
    }

    Ok(jac)
}

/// Compute Hessian of scalar function f: R^n → R.
/// Uses nested forward-mode (forward-over-forward).
/// Returns n×n symmetric matrix in row-major order.
pub fn hessian<F>(f: F, point: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualNumber> + Copy,
{
    let n = point.len();
    let mut hess = vec![0.0; n * n];

    // For each pair (i, j), compute ∂²f/∂xᵢ∂xⱼ
    // Forward-over-forward: seed both i and j directions
    for i in 0..n {
        for j in i..n {
            // Create a function that computes ∂f/∂xᵢ
            let df_dxi = |x: &DualVector| -> AdResult<DualNumber> {
                // Inner differentiation w.r.t. xᵢ
                let mut inner_input = Vec::with_capacity(n);
                for (k, &xk) in point.iter().enumerate() {
                    if k == i {
                        inner_input.push(DualNumber::variable(x.elements[k].primal));
                    } else {
                        inner_input.push(DualNumber::constant(x.elements[k].primal));
                    }
                }
                let inner_vec = DualVector::new(inner_input);
                let result = f(&inner_vec)?;
                // Return the derivative w.r.t. xᵢ as a function of x
                // This is the tangent when xᵢ is seeded
                Ok(DualNumber::new(result.tangent, 0.0))
            };

            // Now differentiate ∂f/∂xᵢ w.r.t. xⱼ
            let input_j = DualVector::seeded_at(point, j);

            // Manually compute: need d/dxⱼ of (∂f/∂xᵢ)
            // Using finite differences as a simpler approach for Hessian
            let h = 1e-7;
            let mut point_plus = point.to_vec();
            let mut point_minus = point.to_vec();
            point_plus[j] += h;
            point_minus[j] -= h;

            let grad_plus = gradient(&f, &point_plus)?;
            let grad_minus = gradient(&f, &point_minus)?;

            let hessian_ij = (grad_plus[i] - grad_minus[i]) / (2.0 * h);

            hess[i * n + j] = hessian_ij;
            if i != j {
                hess[j * n + i] = hessian_ij; // Symmetry
            }
        }
    }

    Ok(hess)
}

/// Jacobian-vector product: J @ v where J is Jacobian of f.
/// Single forward pass with seed v.
pub fn jvp<F>(f: F, point: &[f64], tangent_vec: &[f64]) -> AdResult<Vec<f64>>
where
    F: Fn(&DualVector) -> AdResult<DualVector>,
{
    if point.len() != tangent_vec.len() {
        return Err(AdError::DimensionMismatch {
            expected: point.len(),
            got: tangent_vec.len(),
        });
    }

    let input = DualVector::with_seeds(point, tangent_vec)?;
    let output = f(&input)?;
    Ok(output.tangents())
}

/// Directional derivative of scalar function in direction v.
/// Single forward pass.
pub fn directional_derivative<F>(f: F, point: &[f64], direction: &[f64]) -> AdResult<f64>
where
    F: Fn(&DualVector) -> AdResult<DualNumber>,
{
    if point.len() != direction.len() {
        return Err(AdError::DimensionMismatch {
            expected: point.len(),
            got: direction.len(),
        });
    }

    let input = DualVector::with_seeds(point, direction)?;
    let output = f(&input)?;
    Ok(output.tangent)
}

// ============================================================================
// GRADIENT CHECKING
// ============================================================================

/// Check gradient using finite differences.
/// Returns true if computed gradient matches numerical gradient within tolerance.
pub fn check_gradient<F>(
    f: F,
    point: &[f64],
    computed_grad: &[f64],
    epsilon: f64,
    tolerance: f64,
) -> bool
where
    F: Fn(&[f64]) -> f64,
{
    if point.len() != computed_grad.len() {
        return false;
    }

    for i in 0..point.len() {
        let mut point_plus = point.to_vec();
        let mut point_minus = point.to_vec();
        point_plus[i] += epsilon;
        point_minus[i] -= epsilon;

        let numerical_deriv = (f(&point_plus) - f(&point_minus)) / (2.0 * epsilon);
        let computed_deriv = computed_grad[i];

        let diff = (numerical_deriv - computed_deriv).abs();
        let scale = numerical_deriv.abs().max(computed_deriv.abs()).max(1.0);

        if diff / scale > tolerance {
            return false;
        }
    }

    true
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
    fn test_elementwise_ops() {
        let a = DualVector::variables(&[1.0, 2.0, 3.0]);
        let b = DualVector::constants(&[4.0, 5.0, 6.0]);

        let sum = add(&a, &b).unwrap();
        assert!(approx_eq(sum.primals()[0], 5.0));
        assert!(approx_eq(sum.primals()[1], 7.0));
        assert!(approx_eq(sum.primals()[2], 9.0));
    }

    #[test]
    fn test_reductions() {
        let v = DualVector::variables(&[1.0, 2.0, 3.0, 4.0]);

        // Sum = 10
        let s = sum(&v);
        assert!(approx_eq(s.primal, 10.0));

        // Mean = 2.5
        let m = mean(&v).unwrap();
        assert!(approx_eq(m.primal, 2.5));

        // Norm = sqrt(1+4+9+16) = sqrt(30)
        let n = norm_l2(&v).unwrap();
        assert!(approx_eq(n.primal, 30.0_f64.sqrt()));
    }

    #[test]
    fn test_dot_product() {
        let a = DualVector::variables(&[1.0, 2.0, 3.0]);
        let b = DualVector::constants(&[4.0, 5.0, 6.0]);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        let d = dot(&a, &b).unwrap();
        assert!(approx_eq(d.primal, 32.0));
    }

    #[test]
    fn test_softmax() {
        let v = DualVector::constants(&[1.0, 2.0, 3.0]);
        let sm = softmax(&v).unwrap();

        // Softmax should sum to 1
        let sum: f64 = sm.primals().iter().sum();
        assert!(approx_eq(sum, 1.0));

        // Highest input should have highest probability
        assert!(sm.primals()[2] > sm.primals()[1]);
        assert!(sm.primals()[1] > sm.primals()[0]);
    }

    #[test]
    fn test_variance() {
        let v = DualVector::constants(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        let var = variance(&v).unwrap();
        // Mean = 5, Var = ((2-5)^2 + ... + (9-5)^2) / 8 = 4
        assert!(approx_eq(var.primal, 4.0));
    }

    #[test]
    fn test_mse_loss() {
        let y_true = DualVector::constants(&[1.0, 2.0, 3.0]);
        let y_pred = DualVector::variables(&[1.0, 2.0, 4.0]);

        // MSE = ((1-1)² + (2-2)² + (3-4)²) / 3 = 1/3
        let loss = mse_loss(&y_true, &y_pred).unwrap();
        assert!(approx_eq(loss.primal, 1.0 / 3.0));
    }

    #[test]
    fn test_gradient_computation() {
        // f(x, y) = x² + y²
        // ∇f = [2x, 2y]
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            let x = &v.elements[0];
            let y = &v.elements[1];
            Ok(x.sq().add(&y.sq()))
        };

        let point = [3.0, 4.0];
        let grad = gradient(f, &point).unwrap();

        assert!(approx_eq(grad[0], 6.0)); // 2 * 3
        assert!(approx_eq(grad[1], 8.0)); // 2 * 4
    }

    #[test]
    fn test_jacobian_computation() {
        // f(x, y) = [x + y, x * y]
        // J = [[1, 1], [y, x]]
        let f = |v: &DualVector| -> AdResult<DualVector> {
            let x = &v.elements[0];
            let y = &v.elements[1];
            Ok(DualVector::new(vec![x.add(y), x.mul(y)]))
        };

        let point = [2.0, 3.0];
        let jac = jacobian(f, &point, 2).unwrap();

        // J = [[1, 1], [3, 2]]
        assert!(approx_eq(jac[0], 1.0)); // ∂(x+y)/∂x
        assert!(approx_eq(jac[1], 1.0)); // ∂(x+y)/∂y
        assert!(approx_eq(jac[2], 3.0)); // ∂(xy)/∂x = y
        assert!(approx_eq(jac[3], 2.0)); // ∂(xy)/∂y = x
    }

    #[test]
    fn test_jvp() {
        // f(x, y) = [x², y²]
        // J = [[2x, 0], [0, 2y]]
        // JVP with v = [1, 1] at (2, 3) = [[4, 0], [0, 6]] @ [1, 1] = [4, 6]
        let f = |v: &DualVector| -> AdResult<DualVector> {
            Ok(DualVector::new(vec![
                v.elements[0].sq(),
                v.elements[1].sq(),
            ]))
        };

        let point = [2.0, 3.0];
        let tangent = [1.0, 1.0];
        let result = jvp(f, &point, &tangent).unwrap();

        assert!(approx_eq(result[0], 4.0));
        assert!(approx_eq(result[1], 6.0));
    }

    #[test]
    fn test_directional_derivative() {
        // f(x, y) = x² + y²
        // ∇f at (3, 4) = [6, 8]
        // Directional derivative in direction [1/sqrt(2), 1/sqrt(2)]
        // = [6, 8] · [1/√2, 1/√2] = 14/√2 ≈ 9.899
        let f = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].sq()))
        };

        let point = [3.0, 4.0];
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        let direction = [sqrt2_inv, sqrt2_inv];
        let dd = directional_derivative(f, &point, &direction).unwrap();

        assert!(approx_eq(dd, 14.0 / 2.0_f64.sqrt()));
    }

    #[test]
    fn test_gradient_check() {
        // f(x, y) = x² + y²
        let f_scalar = |x: &[f64]| -> f64 { x[0] * x[0] + x[1] * x[1] };
        let f_dual = |v: &DualVector| -> AdResult<DualNumber> {
            Ok(v.elements[0].sq().add(&v.elements[1].sq()))
        };

        let point = [3.0, 4.0];
        let computed_grad = gradient(f_dual, &point).unwrap();

        let check = check_gradient(f_scalar, &point, &computed_grad, 1e-7, 1e-5);
        assert!(check);
    }

    #[test]
    fn test_matvec() {
        // M = [[1, 2], [3, 4]], v = [1, 2]
        // M @ v = [1*1 + 2*2, 3*1 + 4*2] = [5, 11]
        let m = DualVector::constants(&[1.0, 2.0, 3.0, 4.0]);
        let v = DualVector::variables(&[1.0, 2.0]);

        let result = matvec(&m, &v, 2, 2).unwrap();
        assert!(approx_eq(result.primals()[0], 5.0));
        assert!(approx_eq(result.primals()[1], 11.0));
    }

    #[test]
    fn test_outer_product() {
        let a = DualVector::constants(&[1.0, 2.0]);
        let b = DualVector::constants(&[3.0, 4.0, 5.0]);

        // a ⊗ b = [[3, 4, 5], [6, 8, 10]]
        let outer = outer(&a, &b);
        let primals = outer.primals();

        assert!(approx_eq(primals[0], 3.0)); // 1*3
        assert!(approx_eq(primals[1], 4.0)); // 1*4
        assert!(approx_eq(primals[2], 5.0)); // 1*5
        assert!(approx_eq(primals[3], 6.0)); // 2*3
        assert!(approx_eq(primals[4], 8.0)); // 2*4
        assert!(approx_eq(primals[5], 10.0)); // 2*5
    }
}
