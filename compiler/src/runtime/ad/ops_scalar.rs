//! Scalar operations for automatic differentiation.
//!
//! This module provides a comprehensive collection of differentiable scalar operations
//! organized by category. All operations are implemented as free functions that work
//! with `DualNumber` types.

use super::dual::{AdError, AdResult, DualNumber};

// ============================================================================
// BASIC ARITHMETIC (5 operations)
// ============================================================================

/// Addition: d/dx[f + g] = f' + g'
#[inline]
pub fn add(a: &DualNumber, b: &DualNumber) -> DualNumber {
    a.add(b)
}

/// Subtraction: d/dx[f - g] = f' - g'
#[inline]
pub fn sub(a: &DualNumber, b: &DualNumber) -> DualNumber {
    a.sub(b)
}

/// Multiplication: d/dx[f * g] = f'g + fg'
#[inline]
pub fn mul(a: &DualNumber, b: &DualNumber) -> DualNumber {
    a.mul(b)
}

/// Division: d/dx[f / g] = (f'g - fg') / g²
#[inline]
pub fn div(a: &DualNumber, b: &DualNumber) -> AdResult<DualNumber> {
    a.div(b)
}

/// Negation: d/dx[-f] = -f'
#[inline]
pub fn neg(a: &DualNumber) -> DualNumber {
    a.neg()
}

// ============================================================================
// POWER FUNCTIONS (6 operations)
// ============================================================================

/// Square: d/dx[f²] = 2f·f'
#[inline]
pub fn sq(a: &DualNumber) -> DualNumber {
    a.sq()
}

/// Cube: d/dx[f³] = 3f²·f'
#[inline]
pub fn cube(a: &DualNumber) -> DualNumber {
    a.cube()
}

/// Square root: d/dx[√f] = f'/(2√f)
#[inline]
pub fn sqrt(a: &DualNumber) -> AdResult<DualNumber> {
    a.sqrt()
}

/// Cube root: d/dx[∛f] = f'/(3∛f²)
#[inline]
pub fn cbrt(a: &DualNumber) -> AdResult<DualNumber> {
    a.cbrt()
}

/// Power with constant exponent: d/dx[f^n] = n·f^(n-1)·f'
#[inline]
pub fn powf(base: &DualNumber, exponent: f64) -> AdResult<DualNumber> {
    base.powf(exponent)
}

/// Power with dual exponent: d/dx[f^g] = f^g·(g'·ln(f) + g·f'/f)
#[inline]
pub fn pow(base: &DualNumber, exponent: &DualNumber) -> AdResult<DualNumber> {
    base.pow(exponent)
}

/// Reciprocal: d/dx[1/f] = -f'/f²
#[inline]
pub fn recip(a: &DualNumber) -> AdResult<DualNumber> {
    a.recip()
}

// ============================================================================
// EXPONENTIAL & LOGARITHMIC (8 operations)
// ============================================================================

/// Natural exponential: d/dx[e^f] = e^f·f'
#[inline]
pub fn exp(a: &DualNumber) -> DualNumber {
    a.exp()
}

/// Base-2 exponential: d/dx[2^f] = 2^f·ln(2)·f'
#[inline]
pub fn exp2(a: &DualNumber) -> DualNumber {
    a.exp2()
}

/// Exponential minus one: d/dx[e^f - 1] = e^f·f'
#[inline]
pub fn exp_m1(a: &DualNumber) -> DualNumber {
    a.exp_m1()
}

/// Natural logarithm: d/dx[ln(f)] = f'/f
#[inline]
pub fn ln(a: &DualNumber) -> AdResult<DualNumber> {
    a.ln()
}

/// Natural log of (1+x): d/dx[ln(1+f)] = f'/(1+f)
#[inline]
pub fn ln_1p(a: &DualNumber) -> AdResult<DualNumber> {
    a.ln_1p()
}

/// Base-10 logarithm: d/dx[log₁₀(f)] = f'/(f·ln(10))
#[inline]
pub fn log10(a: &DualNumber) -> AdResult<DualNumber> {
    a.log10()
}

/// Base-2 logarithm: d/dx[log₂(f)] = f'/(f·ln(2))
#[inline]
pub fn log2(a: &DualNumber) -> AdResult<DualNumber> {
    a.log2()
}

/// Logarithm with arbitrary base: d/dx[log_b(f)] = f'/(f·ln(b))
#[inline]
pub fn log(a: &DualNumber, base: f64) -> AdResult<DualNumber> {
    a.log(base)
}

// ============================================================================
// TRIGONOMETRIC (6 operations)
// ============================================================================

/// Sine: d/dx[sin(f)] = cos(f)·f'
#[inline]
pub fn sin(a: &DualNumber) -> DualNumber {
    a.sin()
}

/// Cosine: d/dx[cos(f)] = -sin(f)·f'
#[inline]
pub fn cos(a: &DualNumber) -> DualNumber {
    a.cos()
}

/// Tangent: d/dx[tan(f)] = sec²(f)·f'
#[inline]
pub fn tan(a: &DualNumber) -> AdResult<DualNumber> {
    a.tan()
}

/// Cotangent: d/dx[cot(f)] = -csc²(f)·f'
#[inline]
pub fn cot(a: &DualNumber) -> AdResult<DualNumber> {
    a.cot()
}

/// Secant: d/dx[sec(f)] = sec(f)·tan(f)·f'
#[inline]
pub fn sec(a: &DualNumber) -> AdResult<DualNumber> {
    a.sec()
}

/// Cosecant: d/dx[csc(f)] = -csc(f)·cot(f)·f'
#[inline]
pub fn csc(a: &DualNumber) -> AdResult<DualNumber> {
    a.csc()
}

// ============================================================================
// INVERSE TRIGONOMETRIC (4 operations)
// ============================================================================

/// Arcsine: d/dx[asin(f)] = f'/√(1-f²)
#[inline]
pub fn asin(a: &DualNumber) -> AdResult<DualNumber> {
    a.asin()
}

/// Arccosine: d/dx[acos(f)] = -f'/√(1-f²)
#[inline]
pub fn acos(a: &DualNumber) -> AdResult<DualNumber> {
    a.acos()
}

/// Arctangent: d/dx[atan(f)] = f'/(1+f²)
#[inline]
pub fn atan(a: &DualNumber) -> DualNumber {
    a.atan()
}

/// Two-argument arctangent: atan2(y, x)
#[inline]
pub fn atan2(y: &DualNumber, x: &DualNumber) -> AdResult<DualNumber> {
    y.atan2(x)
}

// ============================================================================
// HYPERBOLIC (6 operations)
// ============================================================================

/// Hyperbolic sine: d/dx[sinh(f)] = cosh(f)·f'
#[inline]
pub fn sinh(a: &DualNumber) -> DualNumber {
    a.sinh()
}

/// Hyperbolic cosine: d/dx[cosh(f)] = sinh(f)·f'
#[inline]
pub fn cosh(a: &DualNumber) -> DualNumber {
    a.cosh()
}

/// Hyperbolic tangent: d/dx[tanh(f)] = sech²(f)·f'
#[inline]
pub fn tanh(a: &DualNumber) -> DualNumber {
    a.tanh()
}

/// Hyperbolic cotangent: d/dx[coth(f)] = -csch²(f)·f'
#[inline]
pub fn coth(a: &DualNumber) -> AdResult<DualNumber> {
    a.coth()
}

/// Hyperbolic secant: d/dx[sech(f)] = -sech(f)·tanh(f)·f'
#[inline]
pub fn sech(a: &DualNumber) -> DualNumber {
    a.sech()
}

/// Hyperbolic cosecant: d/dx[csch(f)] = -csch(f)·coth(f)·f'
#[inline]
pub fn csch(a: &DualNumber) -> AdResult<DualNumber> {
    a.csch()
}

// ============================================================================
// INVERSE HYPERBOLIC (3 operations)
// ============================================================================

/// Inverse hyperbolic sine: d/dx[asinh(f)] = f'/√(f²+1)
#[inline]
pub fn asinh(a: &DualNumber) -> DualNumber {
    a.asinh()
}

/// Inverse hyperbolic cosine: d/dx[acosh(f)] = f'/√(f²-1)
#[inline]
pub fn acosh(a: &DualNumber) -> AdResult<DualNumber> {
    a.acosh()
}

/// Inverse hyperbolic tangent: d/dx[atanh(f)] = f'/(1-f²)
#[inline]
pub fn atanh(a: &DualNumber) -> AdResult<DualNumber> {
    a.atanh()
}

// ============================================================================
// SPECIAL MATHEMATICAL FUNCTIONS (4 operations)
// ============================================================================

/// Error function: erf(x) = (2/√π)∫₀ˣ e^(-t²)dt
#[inline]
pub fn erf(a: &DualNumber) -> DualNumber {
    a.erf()
}

/// Gaussian/Normal PDF: (1/√(2π))·e^(-x²/2)
#[inline]
pub fn gaussian_pdf(a: &DualNumber) -> DualNumber {
    a.gaussian_pdf()
}

/// Logistic sigmoid: σ(x) = 1/(1+e^(-x))
#[inline]
pub fn sigmoid(a: &DualNumber) -> DualNumber {
    a.sigmoid()
}

/// Softplus: log(1 + e^x)
#[inline]
pub fn softplus(a: &DualNumber) -> DualNumber {
    a.softplus()
}

// ============================================================================
// ML ACTIVATION FUNCTIONS (4 operations)
// ============================================================================

/// ReLU with smooth approximation (softplus-based)
#[inline]
pub fn relu_smooth(a: &DualNumber, beta: f64) -> DualNumber {
    a.relu_smooth(beta)
}

/// GELU: Gaussian Error Linear Unit
#[inline]
pub fn gelu(a: &DualNumber) -> DualNumber {
    a.gelu()
}

/// Swish/SiLU: x·σ(x)
#[inline]
pub fn swish(a: &DualNumber) -> DualNumber {
    a.swish()
}

/// Leaky ReLU with smooth approximation
#[inline]
pub fn leaky_relu_smooth(a: &DualNumber, alpha: f64, beta: f64) -> DualNumber {
    // leaky_relu(x) ≈ max(x, αx) = x when x > 0, αx when x < 0
    // Smooth approximation using softplus
    let positive_part = a.relu_smooth(beta);
    let negative_part = a.neg().relu_smooth(beta).mul(&DualNumber::constant(alpha));
    positive_part
        .sub(&negative_part)
        .add(&negative_part.mul(&DualNumber::constant(1.0 + alpha)))
}

/// ELU: Exponential Linear Unit
/// elu(x) = x if x > 0, α(e^x - 1) if x ≤ 0
#[inline]
pub fn elu_smooth(a: &DualNumber, alpha: f64, sharpness: f64) -> DualNumber {
    // Smooth approximation using sigmoid as selector
    let selector = a.mul(&DualNumber::constant(sharpness)).sigmoid();
    let positive = a.clone();
    let negative = DualNumber::constant(alpha).mul(&a.exp_m1());
    // Mix: selector * positive + (1 - selector) * negative
    let one_minus_sel = DualNumber::constant(1.0).sub(&selector);
    selector.mul(&positive).add(&one_minus_sel.mul(&negative))
}

// ============================================================================
// SMOOTH APPROXIMATIONS (5 operations)
// ============================================================================

/// Smooth absolute value: √(x² + ε)
#[inline]
pub fn abs_smooth(a: &DualNumber, epsilon: f64) -> DualNumber {
    a.abs_smooth(epsilon)
}

/// Smooth sign function: tanh(x/ε)
#[inline]
pub fn sign_smooth(a: &DualNumber, epsilon: f64) -> DualNumber {
    a.sign_smooth(epsilon)
}

/// Smooth maximum: softmax-based approximation
#[inline]
pub fn smooth_max(a: &DualNumber, b: &DualNumber, sharpness: f64) -> DualNumber {
    a.smooth_max(b, sharpness)
}

/// Smooth minimum: softmin-based approximation
#[inline]
pub fn smooth_min(a: &DualNumber, b: &DualNumber, sharpness: f64) -> DualNumber {
    a.smooth_min(b, sharpness)
}

/// Smooth clamp between min and max
#[inline]
pub fn smooth_clamp(a: &DualNumber, min: f64, max: f64, sharpness: f64) -> DualNumber {
    a.smooth_clamp(min, max, sharpness)
}

/// Smooth Heaviside/step function using logistic sigmoid
#[inline]
pub fn heaviside_smooth(a: &DualNumber, sharpness: f64) -> DualNumber {
    a.heaviside_smooth(sharpness)
}

// ============================================================================
// UTILITY OPERATIONS (5 operations)
// ============================================================================

/// Linear interpolation: lerp(a, b, t) = a + t·(b - a)
#[inline]
pub fn lerp(a: &DualNumber, b: &DualNumber, t: &DualNumber) -> DualNumber {
    DualNumber::lerp(a, b, t)
}

/// Fused multiply-add: a·b + c
#[inline]
pub fn fma(a: &DualNumber, b: &DualNumber, c: &DualNumber) -> DualNumber {
    a.fma(b, c)
}

/// Hypotenuse: √(a² + b²)
#[inline]
pub fn hypot(a: &DualNumber, b: &DualNumber) -> AdResult<DualNumber> {
    a.hypot(b)
}

/// Floor (non-differentiable, derivative = 0)
#[inline]
pub fn floor(a: &DualNumber) -> DualNumber {
    a.floor()
}

/// Ceiling (non-differentiable, derivative = 0)
#[inline]
pub fn ceil(a: &DualNumber) -> DualNumber {
    a.ceil()
}

/// Round (non-differentiable, derivative = 0)
#[inline]
pub fn round(a: &DualNumber) -> DualNumber {
    a.round()
}

/// Truncate (non-differentiable, derivative = 0)
#[inline]
pub fn trunc(a: &DualNumber) -> DualNumber {
    a.trunc()
}

/// Fractional part: fract(x) = x - floor(x)
#[inline]
pub fn fract(a: &DualNumber) -> DualNumber {
    a.fract()
}

// ============================================================================
// COMPARISON OPERATIONS (smooth versions)
// ============================================================================

/// Smooth indicator for a > b: approaches 1 when a > b, 0 when a < b
#[inline]
pub fn smooth_gt(a: &DualNumber, b: &DualNumber, sharpness: f64) -> DualNumber {
    a.sub(b).heaviside_smooth(sharpness)
}

/// Smooth indicator for a < b: approaches 1 when a < b, 0 when a > b
#[inline]
pub fn smooth_lt(a: &DualNumber, b: &DualNumber, sharpness: f64) -> DualNumber {
    b.sub(a).heaviside_smooth(sharpness)
}

/// Smooth indicator for a ≥ b
#[inline]
pub fn smooth_ge(a: &DualNumber, b: &DualNumber, sharpness: f64) -> DualNumber {
    smooth_gt(a, b, sharpness)
}

/// Smooth indicator for a ≤ b
#[inline]
pub fn smooth_le(a: &DualNumber, b: &DualNumber, sharpness: f64) -> DualNumber {
    smooth_lt(a, b, sharpness)
}

// ============================================================================
// PHARMACOKINETIC/PHARMACODYNAMIC FUNCTIONS
// ============================================================================

/// Hill function: x^n / (K^n + x^n)
/// Used in PK/PD modeling for saturation kinetics.
#[inline]
pub fn hill(x: &DualNumber, k: &DualNumber, n: &DualNumber) -> AdResult<DualNumber> {
    let x_n = x.pow(n)?;
    let k_n = k.pow(n)?;
    x_n.div(&x_n.add(&k_n))
}

/// Emax model: E0 + Emax·C^n / (EC50^n + C^n)
#[inline]
pub fn emax(
    concentration: &DualNumber,
    e0: &DualNumber,
    emax: &DualNumber,
    ec50: &DualNumber,
    n: &DualNumber,
) -> AdResult<DualNumber> {
    let hill_val = hill(concentration, ec50, n)?;
    Ok(e0.add(&emax.mul(&hill_val)))
}

/// Michaelis-Menten kinetics: Vmax·S / (Km + S)
#[inline]
pub fn michaelis_menten(
    substrate: &DualNumber,
    vmax: &DualNumber,
    km: &DualNumber,
) -> AdResult<DualNumber> {
    vmax.mul(substrate).div(&km.add(substrate))
}

/// First-order elimination: C(t) = C0·e^(-k·t)
#[inline]
pub fn first_order_elimination(c0: &DualNumber, k: &DualNumber, t: &DualNumber) -> DualNumber {
    c0.mul(&k.neg().mul(t).exp())
}

/// One-compartment PK oral absorption:
/// C(t) = (F·D·ka / V·(ka - ke)) · (e^(-ke·t) - e^(-ka·t))
#[inline]
pub fn one_compartment_oral(
    dose: &DualNumber,
    bioavail: &DualNumber,
    ka: &DualNumber,
    ke: &DualNumber,
    volume: &DualNumber,
    t: &DualNumber,
) -> AdResult<DualNumber> {
    let ka_minus_ke = ka.sub(ke);
    if ka_minus_ke.primal.abs() < 1e-10 {
        return Err(AdError::NumericalInstability(
            "ka ≈ ke in one-compartment model".to_string(),
        ));
    }

    let coef = bioavail.mul(dose).mul(ka).div(&volume.mul(&ka_minus_ke))?;
    let exp_ke = ke.neg().mul(t).exp();
    let exp_ka = ka.neg().mul(t).exp();
    Ok(coef.mul(&exp_ke.sub(&exp_ka)))
}

// ============================================================================
// STATISTICAL FUNCTIONS
// ============================================================================

/// Log-likelihood for normal distribution: -0.5·((x-μ)/σ)² - log(σ) - 0.5·log(2π)
#[inline]
pub fn log_likelihood_normal(
    x: &DualNumber,
    mu: &DualNumber,
    sigma: &DualNumber,
) -> AdResult<DualNumber> {
    let z = x.sub(mu).div(sigma)?;
    let half = DualNumber::constant(0.5);
    let log_sigma = sigma.ln()?;
    let log_2pi = DualNumber::constant((2.0 * std::f64::consts::PI).ln());
    Ok(half
        .neg()
        .mul(&z.sq())
        .sub(&log_sigma)
        .sub(&half.mul(&log_2pi)))
}

/// Log-likelihood for exponential distribution: log(λ) - λ·x
#[inline]
pub fn log_likelihood_exponential(x: &DualNumber, lambda: &DualNumber) -> AdResult<DualNumber> {
    let log_lambda = lambda.ln()?;
    Ok(log_lambda.sub(&lambda.mul(x)))
}

/// Log-likelihood for Poisson distribution (ignoring factorial): k·log(λ) - λ
#[inline]
pub fn log_likelihood_poisson(k: &DualNumber, lambda: &DualNumber) -> AdResult<DualNumber> {
    let log_lambda = lambda.ln()?;
    Ok(k.mul(&log_lambda).sub(lambda))
}

/// Binary cross-entropy loss: -[y·log(p) + (1-y)·log(1-p)]
#[inline]
pub fn binary_cross_entropy(y: &DualNumber, p: &DualNumber) -> AdResult<DualNumber> {
    let one = DualNumber::constant(1.0);
    let epsilon = 1e-15;
    let p_clipped = p.smooth_clamp(epsilon, 1.0 - epsilon, 100.0);
    let log_p = p_clipped.ln()?;
    let log_1mp = one.sub(&p_clipped).ln()?;
    Ok(y.mul(&log_p).add(&one.sub(y).mul(&log_1mp)).neg())
}

/// Mean squared error contribution: (y - ŷ)²
#[inline]
pub fn mse_loss(y_true: &DualNumber, y_pred: &DualNumber) -> DualNumber {
    y_true.sub(y_pred).sq()
}

/// Huber loss: quadratic for small errors, linear for large
#[inline]
pub fn huber_loss(y_true: &DualNumber, y_pred: &DualNumber, delta: f64) -> DualNumber {
    let error = y_true.sub(y_pred);
    let abs_error = error.abs_smooth(1e-10);
    let delta_d = DualNumber::constant(delta);

    // Smooth transition between quadratic and linear
    let quadratic = DualNumber::constant(0.5).mul(&error.sq());
    let linear = delta_d.mul(&abs_error.sub(&DualNumber::constant(0.5 * delta)));

    // Use smooth_max/min to blend
    quadratic.smooth_min(&linear, 10.0)
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-8;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON || (a - b).abs() / a.abs().max(b.abs()).max(1.0) < EPSILON
    }

    #[test]
    fn test_basic_arithmetic() {
        let x = DualNumber::variable(3.0);
        let y = DualNumber::constant(2.0);

        // 3 + 2 = 5, d/dx = 1
        let sum = add(&x, &y);
        assert!(approx_eq(sum.primal, 5.0));
        assert!(approx_eq(sum.tangent, 1.0));

        // 3 * 2 = 6, d/dx = 2
        let prod = mul(&x, &y);
        assert!(approx_eq(prod.primal, 6.0));
        assert!(approx_eq(prod.tangent, 2.0));
    }

    #[test]
    fn test_power_functions() {
        let x = DualNumber::variable(2.0);

        // x³ at x=2: value=8, d/dx=12
        let cubed = cube(&x);
        assert!(approx_eq(cubed.primal, 8.0));
        assert!(approx_eq(cubed.tangent, 12.0));

        // √x at x=4
        let four = DualNumber::variable(4.0);
        let root = sqrt(&four).unwrap();
        assert!(approx_eq(root.primal, 2.0));
        assert!(approx_eq(root.tangent, 0.25)); // 1/(2√4) = 0.25
    }

    #[test]
    fn test_exp_log() {
        let x = DualNumber::variable(1.0);

        // e^1 ≈ 2.718, d/dx = e^1
        let exp_x = exp(&x);
        assert!(approx_eq(exp_x.primal, std::f64::consts::E));
        assert!(approx_eq(exp_x.tangent, std::f64::consts::E));

        // ln(e) = 1, d/dx = 1/e
        let e = DualNumber::variable(std::f64::consts::E);
        let ln_e = ln(&e).unwrap();
        assert!(approx_eq(ln_e.primal, 1.0));
        assert!(approx_eq(ln_e.tangent, 1.0 / std::f64::consts::E));
    }

    #[test]
    fn test_trig_functions() {
        let x = DualNumber::variable(0.0);

        // sin(0) = 0, d/dx = cos(0) = 1
        let sin_x = sin(&x);
        assert!(approx_eq(sin_x.primal, 0.0));
        assert!(approx_eq(sin_x.tangent, 1.0));

        // cos(0) = 1, d/dx = -sin(0) = 0
        let cos_x = cos(&x);
        assert!(approx_eq(cos_x.primal, 1.0));
        assert!(approx_eq(cos_x.tangent, 0.0));
    }

    #[test]
    fn test_hyperbolic() {
        let x = DualNumber::variable(0.0);

        // tanh(0) = 0, d/dx = sech²(0) = 1
        let tanh_x = tanh(&x);
        assert!(approx_eq(tanh_x.primal, 0.0));
        assert!(approx_eq(tanh_x.tangent, 1.0));
    }

    #[test]
    fn test_activations() {
        let x = DualNumber::variable(0.0);

        // sigmoid(0) = 0.5, d/dx = 0.25
        let sig = sigmoid(&x);
        assert!(approx_eq(sig.primal, 0.5));
        assert!(approx_eq(sig.tangent, 0.25));

        // softplus(0) = ln(2), d/dx = sigmoid(0) = 0.5
        let sp = softplus(&x);
        assert!(approx_eq(sp.primal, 2.0_f64.ln()));
        assert!(approx_eq(sp.tangent, 0.5));
    }

    #[test]
    fn test_smooth_functions() {
        let x = DualNumber::variable(0.5);
        let y = DualNumber::constant(0.3);

        // smooth_max(0.5, 0.3) should be close to 0.5
        let sm = smooth_max(&x, &y, 10.0);
        assert!(sm.primal > 0.4);
        assert!(sm.primal < 0.6);
    }

    #[test]
    fn test_hill_function() {
        let x = DualNumber::variable(1.0);
        let k = DualNumber::constant(1.0);
        let n = DualNumber::constant(1.0);

        // Hill(1, 1, 1) = 1/(1+1) = 0.5
        let h = hill(&x, &k, &n).unwrap();
        assert!(approx_eq(h.primal, 0.5));
    }

    #[test]
    fn test_michaelis_menten() {
        let s = DualNumber::variable(10.0);
        let vmax = DualNumber::constant(100.0);
        let km = DualNumber::constant(10.0);

        // MM(10, 100, 10) = 100*10/(10+10) = 50
        let v = michaelis_menten(&s, &vmax, &km).unwrap();
        assert!(approx_eq(v.primal, 50.0));
    }

    #[test]
    fn test_log_likelihood_normal() {
        let x = DualNumber::constant(0.0);
        let mu = DualNumber::variable(0.0);
        let sigma = DualNumber::constant(1.0);

        // Standard normal at mean: ll = -0.5*log(2π)
        let ll = log_likelihood_normal(&x, &mu, &sigma).unwrap();
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(approx_eq(ll.primal, expected));
    }

    #[test]
    fn test_mse_loss() {
        let y_true = DualNumber::constant(3.0);
        let y_pred = DualNumber::variable(2.0);

        // MSE = (3-2)² = 1, d/d(y_pred) = -2*(3-2) = -2
        let loss = mse_loss(&y_true, &y_pred);
        assert!(approx_eq(loss.primal, 1.0));
        assert!(approx_eq(loss.tangent, -2.0));
    }
}
