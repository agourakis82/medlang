//! Core dual number types for automatic differentiation.
//!
//! This module provides the foundational types for forward-mode AD:
//! - `DualNumber`: Scalar dual numbers (primal, tangent)
//! - `DualVector`: Vector of dual numbers for multi-variable gradients
//! - `DualValue`: Unified enum for scalar/vector duals
//! - `DualRecord`: Named parameter gradients for structured differentiation
//! - `AdContext`: Debugging and tracing context
//! - `AdError`: Error types for AD operations

use std::collections::HashMap;
use std::fmt;

/// Error types for automatic differentiation operations.
#[derive(Debug, Clone, PartialEq)]
pub enum AdError {
    /// Division by zero in dual arithmetic
    DivisionByZero,
    /// Invalid domain for operation (e.g., log of negative)
    DomainError(String),
    /// Dimension mismatch in vector operations
    DimensionMismatch { expected: usize, got: usize },
    /// Missing parameter in record
    MissingParameter(String),
    /// Invalid operation for dual type
    InvalidOperation(String),
    /// Numerical instability detected
    NumericalInstability(String),
    /// Operation not differentiable at this point
    NotDifferentiable(String),
}

impl fmt::Display for AdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdError::DivisionByZero => write!(f, "Division by zero in AD"),
            AdError::DomainError(msg) => write!(f, "Domain error: {}", msg),
            AdError::DimensionMismatch { expected, got } => {
                write!(f, "Dimension mismatch: expected {}, got {}", expected, got)
            }
            AdError::MissingParameter(name) => write!(f, "Missing parameter: {}", name),
            AdError::InvalidOperation(msg) => write!(f, "Invalid AD operation: {}", msg),
            AdError::NumericalInstability(msg) => write!(f, "Numerical instability: {}", msg),
            AdError::NotDifferentiable(msg) => write!(f, "Not differentiable: {}", msg),
        }
    }
}

impl std::error::Error for AdError {}

pub type AdResult<T> = Result<T, AdError>;

/// A scalar dual number representing a value and its derivative.
///
/// Dual numbers extend real numbers with an infinitesimal ε where ε² = 0.
/// A dual number a + bε encodes both:
/// - `primal`: the value a = f(x)
/// - `tangent`: the derivative b = f'(x)
///
/// # Examples
/// ```ignore
/// let x = DualNumber::variable(2.0);  // x = 2 + 1ε
/// let y = x.mul(&x);                   // y = 4 + 4ε (x² at x=2)
/// assert_eq!(y.primal, 4.0);          // f(2) = 4
/// assert_eq!(y.tangent, 4.0);         // f'(2) = 2x = 4
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DualNumber {
    /// The primal value f(x)
    pub primal: f64,
    /// The tangent (derivative) f'(x)
    pub tangent: f64,
}

impl DualNumber {
    /// Create a new dual number with specified primal and tangent.
    #[inline]
    pub fn new(primal: f64, tangent: f64) -> Self {
        Self { primal, tangent }
    }

    /// Create a constant dual number (tangent = 0).
    /// Constants have zero derivative.
    #[inline]
    pub fn constant(value: f64) -> Self {
        Self::new(value, 0.0)
    }

    /// Create a variable dual number (tangent = 1).
    /// This "seeds" differentiation with respect to this variable.
    #[inline]
    pub fn variable(value: f64) -> Self {
        Self::new(value, 1.0)
    }

    /// Create a dual number with custom seed (directional derivative).
    #[inline]
    pub fn seeded(value: f64, seed: f64) -> Self {
        Self::new(value, seed)
    }

    /// Check if this is effectively a constant (zero tangent).
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.tangent.abs() < 1e-15
    }

    /// Extract just the primal value.
    #[inline]
    pub fn value(&self) -> f64 {
        self.primal
    }

    /// Extract just the derivative.
    #[inline]
    pub fn derivative(&self) -> f64 {
        self.tangent
    }

    // ==================== Basic Arithmetic ====================

    /// Addition: d/dx[f + g] = f' + g'
    #[inline]
    pub fn add(&self, other: &DualNumber) -> DualNumber {
        DualNumber::new(self.primal + other.primal, self.tangent + other.tangent)
    }

    /// Subtraction: d/dx[f - g] = f' - g'
    #[inline]
    pub fn sub(&self, other: &DualNumber) -> DualNumber {
        DualNumber::new(self.primal - other.primal, self.tangent - other.tangent)
    }

    /// Multiplication (product rule): d/dx[f * g] = f' * g + f * g'
    #[inline]
    pub fn mul(&self, other: &DualNumber) -> DualNumber {
        DualNumber::new(
            self.primal * other.primal,
            self.tangent * other.primal + self.primal * other.tangent,
        )
    }

    /// Division (quotient rule): d/dx[f / g] = (f' * g - f * g') / g²
    #[inline]
    pub fn div(&self, other: &DualNumber) -> AdResult<DualNumber> {
        if other.primal.abs() < 1e-15 {
            return Err(AdError::DivisionByZero);
        }
        let denom = other.primal * other.primal;
        Ok(DualNumber::new(
            self.primal / other.primal,
            (self.tangent * other.primal - self.primal * other.tangent) / denom,
        ))
    }

    /// Negation: d/dx[-f] = -f'
    #[inline]
    pub fn neg(&self) -> DualNumber {
        DualNumber::new(-self.primal, -self.tangent)
    }

    /// Reciprocal: d/dx[1/f] = -f'/f²
    #[inline]
    pub fn recip(&self) -> AdResult<DualNumber> {
        if self.primal.abs() < 1e-15 {
            return Err(AdError::DivisionByZero);
        }
        let p2 = self.primal * self.primal;
        Ok(DualNumber::new(1.0 / self.primal, -self.tangent / p2))
    }

    // ==================== Power Functions ====================

    /// Square: d/dx[f²] = 2f * f'
    #[inline]
    pub fn sq(&self) -> DualNumber {
        DualNumber::new(self.primal * self.primal, 2.0 * self.primal * self.tangent)
    }

    /// Cube: d/dx[f³] = 3f² * f'
    #[inline]
    pub fn cube(&self) -> DualNumber {
        let p2 = self.primal * self.primal;
        DualNumber::new(p2 * self.primal, 3.0 * p2 * self.tangent)
    }

    /// Square root: d/dx[√f] = f'/(2√f)
    #[inline]
    pub fn sqrt(&self) -> AdResult<DualNumber> {
        if self.primal < 0.0 {
            return Err(AdError::DomainError("sqrt of negative number".to_string()));
        }
        if self.primal < 1e-15 {
            // At zero, derivative is undefined but we can use a large value
            return Err(AdError::NotDifferentiable("sqrt at zero".to_string()));
        }
        let sqrt_p = self.primal.sqrt();
        Ok(DualNumber::new(sqrt_p, self.tangent / (2.0 * sqrt_p)))
    }

    /// Cube root: d/dx[∛f] = f'/(3∛f²)
    #[inline]
    pub fn cbrt(&self) -> AdResult<DualNumber> {
        if self.primal.abs() < 1e-15 {
            return Err(AdError::NotDifferentiable("cbrt at zero".to_string()));
        }
        let cbrt_p = self.primal.cbrt();
        let cbrt_p2 = cbrt_p * cbrt_p;
        Ok(DualNumber::new(cbrt_p, self.tangent / (3.0 * cbrt_p2)))
    }

    /// Power with constant exponent: d/dx[f^n] = n * f^(n-1) * f'
    #[inline]
    pub fn powf(&self, n: f64) -> AdResult<DualNumber> {
        if self.primal < 0.0 && n.fract() != 0.0 {
            return Err(AdError::DomainError(
                "negative base with fractional exponent".to_string(),
            ));
        }
        if self.primal.abs() < 1e-15 && n < 1.0 {
            return Err(AdError::NotDifferentiable(
                "power at zero with exponent < 1".to_string(),
            ));
        }
        let pow_val = self.primal.powf(n);
        let deriv = n * self.primal.powf(n - 1.0) * self.tangent;
        Ok(DualNumber::new(pow_val, deriv))
    }

    /// Power with dual exponent: d/dx[f^g] = f^g * (g' * ln(f) + g * f'/f)
    #[inline]
    pub fn pow(&self, other: &DualNumber) -> AdResult<DualNumber> {
        if self.primal <= 0.0 {
            return Err(AdError::DomainError("non-positive base in pow".to_string()));
        }
        let pow_val = self.primal.powf(other.primal);
        let ln_base = self.primal.ln();
        let deriv = pow_val * (other.tangent * ln_base + other.primal * self.tangent / self.primal);
        Ok(DualNumber::new(pow_val, deriv))
    }

    // ==================== Exponential & Logarithmic ====================

    /// Exponential: d/dx[e^f] = e^f * f'
    #[inline]
    pub fn exp(&self) -> DualNumber {
        let exp_val = self.primal.exp();
        DualNumber::new(exp_val, exp_val * self.tangent)
    }

    /// Exponential minus one: d/dx[e^f - 1] = e^f * f'
    /// More accurate near zero.
    #[inline]
    pub fn exp_m1(&self) -> DualNumber {
        let exp_val = self.primal.exp();
        DualNumber::new(self.primal.exp_m1(), exp_val * self.tangent)
    }

    /// Base-2 exponential: d/dx[2^f] = 2^f * ln(2) * f'
    #[inline]
    pub fn exp2(&self) -> DualNumber {
        let exp_val = self.primal.exp2();
        DualNumber::new(exp_val, exp_val * std::f64::consts::LN_2 * self.tangent)
    }

    /// Natural logarithm: d/dx[ln(f)] = f'/f
    #[inline]
    pub fn ln(&self) -> AdResult<DualNumber> {
        if self.primal <= 0.0 {
            return Err(AdError::DomainError("ln of non-positive".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.ln(),
            self.tangent / self.primal,
        ))
    }

    /// Natural log of (1 + x): d/dx[ln(1+f)] = f'/(1+f)
    /// More accurate near zero.
    #[inline]
    pub fn ln_1p(&self) -> AdResult<DualNumber> {
        if self.primal <= -1.0 {
            return Err(AdError::DomainError("ln_1p of value <= -1".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.ln_1p(),
            self.tangent / (1.0 + self.primal),
        ))
    }

    /// Base-10 logarithm: d/dx[log10(f)] = f'/(f * ln(10))
    #[inline]
    pub fn log10(&self) -> AdResult<DualNumber> {
        if self.primal <= 0.0 {
            return Err(AdError::DomainError("log10 of non-positive".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.log10(),
            self.tangent / (self.primal * std::f64::consts::LN_10),
        ))
    }

    /// Base-2 logarithm: d/dx[log2(f)] = f'/(f * ln(2))
    #[inline]
    pub fn log2(&self) -> AdResult<DualNumber> {
        if self.primal <= 0.0 {
            return Err(AdError::DomainError("log2 of non-positive".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.log2(),
            self.tangent / (self.primal * std::f64::consts::LN_2),
        ))
    }

    /// Logarithm with arbitrary base: d/dx[log_b(f)] = f'/(f * ln(b))
    #[inline]
    pub fn log(&self, base: f64) -> AdResult<DualNumber> {
        if self.primal <= 0.0 {
            return Err(AdError::DomainError("log of non-positive".to_string()));
        }
        if base <= 0.0 || base == 1.0 {
            return Err(AdError::DomainError("invalid logarithm base".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.log(base),
            self.tangent / (self.primal * base.ln()),
        ))
    }

    // ==================== Trigonometric Functions ====================

    /// Sine: d/dx[sin(f)] = cos(f) * f'
    #[inline]
    pub fn sin(&self) -> DualNumber {
        DualNumber::new(self.primal.sin(), self.primal.cos() * self.tangent)
    }

    /// Cosine: d/dx[cos(f)] = -sin(f) * f'
    #[inline]
    pub fn cos(&self) -> DualNumber {
        DualNumber::new(self.primal.cos(), -self.primal.sin() * self.tangent)
    }

    /// Tangent: d/dx[tan(f)] = sec²(f) * f' = f'/cos²(f)
    #[inline]
    pub fn tan(&self) -> AdResult<DualNumber> {
        let cos_val = self.primal.cos();
        if cos_val.abs() < 1e-15 {
            return Err(AdError::DomainError("tan at π/2 + nπ".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.tan(),
            self.tangent / (cos_val * cos_val),
        ))
    }

    /// Cotangent: d/dx[cot(f)] = -csc²(f) * f'
    #[inline]
    pub fn cot(&self) -> AdResult<DualNumber> {
        let sin_val = self.primal.sin();
        if sin_val.abs() < 1e-15 {
            return Err(AdError::DomainError("cot at nπ".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.cos() / sin_val,
            -self.tangent / (sin_val * sin_val),
        ))
    }

    /// Secant: d/dx[sec(f)] = sec(f)*tan(f) * f'
    #[inline]
    pub fn sec(&self) -> AdResult<DualNumber> {
        let cos_val = self.primal.cos();
        if cos_val.abs() < 1e-15 {
            return Err(AdError::DomainError("sec at π/2 + nπ".to_string()));
        }
        let sec_val = 1.0 / cos_val;
        Ok(DualNumber::new(
            sec_val,
            sec_val * self.primal.tan() * self.tangent,
        ))
    }

    /// Cosecant: d/dx[csc(f)] = -csc(f)*cot(f) * f'
    #[inline]
    pub fn csc(&self) -> AdResult<DualNumber> {
        let sin_val = self.primal.sin();
        if sin_val.abs() < 1e-15 {
            return Err(AdError::DomainError("csc at nπ".to_string()));
        }
        let csc_val = 1.0 / sin_val;
        let cot_val = self.primal.cos() / sin_val;
        Ok(DualNumber::new(csc_val, -csc_val * cot_val * self.tangent))
    }

    // ==================== Inverse Trigonometric ====================

    /// Arcsine: d/dx[asin(f)] = f'/√(1-f²)
    #[inline]
    pub fn asin(&self) -> AdResult<DualNumber> {
        if self.primal.abs() > 1.0 {
            return Err(AdError::DomainError("asin of |x| > 1".to_string()));
        }
        let denom = (1.0 - self.primal * self.primal).sqrt();
        if denom < 1e-15 {
            return Err(AdError::NotDifferentiable("asin at ±1".to_string()));
        }
        Ok(DualNumber::new(self.primal.asin(), self.tangent / denom))
    }

    /// Arccosine: d/dx[acos(f)] = -f'/√(1-f²)
    #[inline]
    pub fn acos(&self) -> AdResult<DualNumber> {
        if self.primal.abs() > 1.0 {
            return Err(AdError::DomainError("acos of |x| > 1".to_string()));
        }
        let denom = (1.0 - self.primal * self.primal).sqrt();
        if denom < 1e-15 {
            return Err(AdError::NotDifferentiable("acos at ±1".to_string()));
        }
        Ok(DualNumber::new(self.primal.acos(), -self.tangent / denom))
    }

    /// Arctangent: d/dx[atan(f)] = f'/(1+f²)
    #[inline]
    pub fn atan(&self) -> DualNumber {
        DualNumber::new(
            self.primal.atan(),
            self.tangent / (1.0 + self.primal * self.primal),
        )
    }

    /// Two-argument arctangent: d/dx[atan2(y,x)]
    #[inline]
    pub fn atan2(&self, x: &DualNumber) -> AdResult<DualNumber> {
        let denom = self.primal * self.primal + x.primal * x.primal;
        if denom < 1e-15 {
            return Err(AdError::NotDifferentiable("atan2 at origin".to_string()));
        }
        Ok(DualNumber::new(
            self.primal.atan2(x.primal),
            (x.primal * self.tangent - self.primal * x.tangent) / denom,
        ))
    }

    // ==================== Hyperbolic Functions ====================

    /// Hyperbolic sine: d/dx[sinh(f)] = cosh(f) * f'
    #[inline]
    pub fn sinh(&self) -> DualNumber {
        DualNumber::new(self.primal.sinh(), self.primal.cosh() * self.tangent)
    }

    /// Hyperbolic cosine: d/dx[cosh(f)] = sinh(f) * f'
    #[inline]
    pub fn cosh(&self) -> DualNumber {
        DualNumber::new(self.primal.cosh(), self.primal.sinh() * self.tangent)
    }

    /// Hyperbolic tangent: d/dx[tanh(f)] = sech²(f) * f' = (1 - tanh²(f)) * f'
    #[inline]
    pub fn tanh(&self) -> DualNumber {
        let tanh_val = self.primal.tanh();
        DualNumber::new(tanh_val, (1.0 - tanh_val * tanh_val) * self.tangent)
    }

    /// Hyperbolic cotangent: d/dx[coth(f)] = -csch²(f) * f'
    #[inline]
    pub fn coth(&self) -> AdResult<DualNumber> {
        let sinh_val = self.primal.sinh();
        if sinh_val.abs() < 1e-15 {
            return Err(AdError::DomainError("coth at 0".to_string()));
        }
        let coth_val = self.primal.cosh() / sinh_val;
        Ok(DualNumber::new(
            coth_val,
            -self.tangent / (sinh_val * sinh_val),
        ))
    }

    /// Hyperbolic secant: d/dx[sech(f)] = -sech(f)*tanh(f) * f'
    #[inline]
    pub fn sech(&self) -> DualNumber {
        let cosh_val = self.primal.cosh();
        let sech_val = 1.0 / cosh_val;
        DualNumber::new(sech_val, -sech_val * self.primal.tanh() * self.tangent)
    }

    /// Hyperbolic cosecant: d/dx[csch(f)] = -csch(f)*coth(f) * f'
    #[inline]
    pub fn csch(&self) -> AdResult<DualNumber> {
        let sinh_val = self.primal.sinh();
        if sinh_val.abs() < 1e-15 {
            return Err(AdError::DomainError("csch at 0".to_string()));
        }
        let csch_val = 1.0 / sinh_val;
        let coth_val = self.primal.cosh() / sinh_val;
        Ok(DualNumber::new(
            csch_val,
            -csch_val * coth_val * self.tangent,
        ))
    }

    // ==================== Inverse Hyperbolic ====================

    /// Inverse hyperbolic sine: d/dx[asinh(f)] = f'/√(f²+1)
    #[inline]
    pub fn asinh(&self) -> DualNumber {
        let denom = (self.primal * self.primal + 1.0).sqrt();
        DualNumber::new(self.primal.asinh(), self.tangent / denom)
    }

    /// Inverse hyperbolic cosine: d/dx[acosh(f)] = f'/√(f²-1)
    #[inline]
    pub fn acosh(&self) -> AdResult<DualNumber> {
        if self.primal < 1.0 {
            return Err(AdError::DomainError("acosh of x < 1".to_string()));
        }
        let denom = (self.primal * self.primal - 1.0).sqrt();
        if denom < 1e-15 {
            return Err(AdError::NotDifferentiable("acosh at 1".to_string()));
        }
        Ok(DualNumber::new(self.primal.acosh(), self.tangent / denom))
    }

    /// Inverse hyperbolic tangent: d/dx[atanh(f)] = f'/(1-f²)
    #[inline]
    pub fn atanh(&self) -> AdResult<DualNumber> {
        if self.primal.abs() >= 1.0 {
            return Err(AdError::DomainError("atanh of |x| >= 1".to_string()));
        }
        let denom = 1.0 - self.primal * self.primal;
        Ok(DualNumber::new(self.primal.atanh(), self.tangent / denom))
    }

    // ==================== Special Functions ====================

    /// Absolute value with smooth approximation
    /// Uses sqrt(x² + ε) for smoothness near zero
    #[inline]
    pub fn abs_smooth(&self, epsilon: f64) -> DualNumber {
        let smooth = (self.primal * self.primal + epsilon).sqrt();
        DualNumber::new(smooth, self.primal * self.tangent / smooth)
    }

    /// Sign function with smooth approximation (tanh(x/ε))
    #[inline]
    pub fn sign_smooth(&self, epsilon: f64) -> DualNumber {
        let scaled = self.primal / epsilon;
        let tanh_val = scaled.tanh();
        let sech2 = 1.0 - tanh_val * tanh_val;
        DualNumber::new(tanh_val, sech2 * self.tangent / epsilon)
    }

    /// Smooth maximum: softmax(a, b) ≈ max(a, b)
    #[inline]
    pub fn smooth_max(&self, other: &DualNumber, sharpness: f64) -> DualNumber {
        let exp_a = (sharpness * self.primal).exp();
        let exp_b = (sharpness * other.primal).exp();
        let sum = exp_a + exp_b;
        let result = (exp_a * self.primal + exp_b * other.primal) / sum;
        let d_result = (exp_a * self.tangent + exp_b * other.tangent) / sum
            + sharpness
                * (exp_a * exp_b * (self.primal - other.primal) * (self.tangent - other.tangent))
                / (sum * sum);
        DualNumber::new(result, d_result)
    }

    /// Smooth minimum: softmin(a, b) ≈ min(a, b)
    #[inline]
    pub fn smooth_min(&self, other: &DualNumber, sharpness: f64) -> DualNumber {
        self.neg().smooth_max(&other.neg(), sharpness).neg()
    }

    /// Clamp with smooth boundaries
    #[inline]
    pub fn smooth_clamp(&self, min: f64, max: f64, sharpness: f64) -> DualNumber {
        let min_d = DualNumber::constant(min);
        let max_d = DualNumber::constant(max);
        self.smooth_max(&min_d, sharpness)
            .smooth_min(&max_d, sharpness)
    }

    /// Heaviside step with smooth approximation (logistic sigmoid)
    #[inline]
    pub fn heaviside_smooth(&self, sharpness: f64) -> DualNumber {
        self.mul(&DualNumber::constant(sharpness)).sigmoid()
    }

    /// Logistic sigmoid: σ(x) = 1/(1+e^(-x))
    /// d/dx[σ(f)] = σ(f)(1-σ(f)) * f'
    #[inline]
    pub fn sigmoid(&self) -> DualNumber {
        let sig = 1.0 / (1.0 + (-self.primal).exp());
        DualNumber::new(sig, sig * (1.0 - sig) * self.tangent)
    }

    /// Softplus: log(1 + e^x)
    /// d/dx[softplus(f)] = sigmoid(f) * f'
    #[inline]
    pub fn softplus(&self) -> DualNumber {
        // Numerically stable implementation
        if self.primal > 20.0 {
            // For large x, softplus(x) ≈ x
            DualNumber::new(self.primal, self.tangent)
        } else if self.primal < -20.0 {
            // For small x, softplus(x) ≈ e^x
            self.exp()
        } else {
            let exp_val = self.primal.exp();
            let softplus = (1.0 + exp_val).ln();
            let sig = exp_val / (1.0 + exp_val);
            DualNumber::new(softplus, sig * self.tangent)
        }
    }

    /// ReLU with smooth approximation (softplus)
    #[inline]
    pub fn relu_smooth(&self, beta: f64) -> DualNumber {
        self.mul(&DualNumber::constant(beta))
            .softplus()
            .mul(&DualNumber::constant(1.0 / beta))
    }

    /// GELU approximation: x * Φ(x) where Φ is standard normal CDF
    /// Using tanh approximation
    #[inline]
    pub fn gelu(&self) -> DualNumber {
        let k = 0.7978845608; // sqrt(2/π)
        let inner = DualNumber::constant(k)
            .mul(&self.add(&DualNumber::constant(0.044715).mul(&self.cube())));
        let tanh_inner = inner.tanh();
        let half = DualNumber::constant(0.5);
        let one = DualNumber::constant(1.0);
        half.mul(&self.mul(&one.add(&tanh_inner)))
    }

    /// Swish/SiLU: x * sigmoid(x)
    #[inline]
    pub fn swish(&self) -> DualNumber {
        self.mul(&self.sigmoid())
    }

    /// Error function approximation
    #[inline]
    pub fn erf(&self) -> DualNumber {
        // Using tanh approximation: erf(x) ≈ tanh(√π * x * (1 + 0.147x²)/(1 + 0.147x²))
        // Simplified: just use numerical erf with derivative 2/√π * e^(-x²)
        let erf_val = erf_approx(self.primal);
        let deriv = 2.0 / std::f64::consts::PI.sqrt() * (-self.primal * self.primal).exp();
        DualNumber::new(erf_val, deriv * self.tangent)
    }

    /// Gaussian/Normal PDF: (1/√(2π)) * e^(-x²/2)
    #[inline]
    pub fn gaussian_pdf(&self) -> DualNumber {
        let coef = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let exp_val = (-0.5 * self.primal * self.primal).exp();
        let pdf = coef * exp_val;
        DualNumber::new(pdf, -self.primal * pdf * self.tangent)
    }

    // ==================== Utility Operations ====================

    /// Linear interpolation: lerp(a, b, t) = a + t*(b-a)
    #[inline]
    pub fn lerp(a: &DualNumber, b: &DualNumber, t: &DualNumber) -> DualNumber {
        a.add(&t.mul(&b.sub(a)))
    }

    /// Fused multiply-add: a*b + c
    #[inline]
    pub fn fma(&self, b: &DualNumber, c: &DualNumber) -> DualNumber {
        self.mul(b).add(c)
    }

    /// Hypotenuse: √(a² + b²)
    #[inline]
    pub fn hypot(&self, other: &DualNumber) -> AdResult<DualNumber> {
        let sum_sq = self.primal * self.primal + other.primal * other.primal;
        if sum_sq < 1e-30 {
            return Err(AdError::NotDifferentiable("hypot at origin".to_string()));
        }
        let h = sum_sq.sqrt();
        Ok(DualNumber::new(
            h,
            (self.primal * self.tangent + other.primal * other.tangent) / h,
        ))
    }

    /// Floor (non-differentiable, derivative = 0)
    #[inline]
    pub fn floor(&self) -> DualNumber {
        DualNumber::new(self.primal.floor(), 0.0)
    }

    /// Ceil (non-differentiable, derivative = 0)
    #[inline]
    pub fn ceil(&self) -> DualNumber {
        DualNumber::new(self.primal.ceil(), 0.0)
    }

    /// Round (non-differentiable, derivative = 0)
    #[inline]
    pub fn round(&self) -> DualNumber {
        DualNumber::new(self.primal.round(), 0.0)
    }

    /// Truncate (non-differentiable, derivative = 0)
    #[inline]
    pub fn trunc(&self) -> DualNumber {
        DualNumber::new(self.primal.trunc(), 0.0)
    }

    /// Fractional part: fract(x) = x - floor(x)
    #[inline]
    pub fn fract(&self) -> DualNumber {
        DualNumber::new(self.primal.fract(), self.tangent)
    }
}

// Helper function for error function approximation
fn erf_approx(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

impl Default for DualNumber {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl fmt::Display for DualNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}+{}ε", self.primal, self.tangent)
    }
}

// ============================================================================
// DualVector - For multi-variable gradients
// ============================================================================

/// A vector of dual numbers for computing gradients of multi-input functions.
///
/// To compute the gradient of f: R^n → R, we run n forward passes,
/// each seeding one input variable with tangent=1 and others with tangent=0.
#[derive(Debug, Clone, PartialEq)]
pub struct DualVector {
    /// The elements of the dual vector
    pub elements: Vec<DualNumber>,
}

impl DualVector {
    /// Create a new dual vector from elements.
    pub fn new(elements: Vec<DualNumber>) -> Self {
        Self { elements }
    }

    /// Create a vector of constants.
    pub fn constants(values: &[f64]) -> Self {
        Self {
            elements: values.iter().map(|&v| DualNumber::constant(v)).collect(),
        }
    }

    /// Create a vector of variables for gradient computation.
    /// Each variable is seeded with tangent=1 at its index, 0 elsewhere.
    pub fn variables(values: &[f64]) -> Self {
        Self {
            elements: values.iter().map(|&v| DualNumber::variable(v)).collect(),
        }
    }

    /// Create a seeded vector for computing gradient w.r.t. index i.
    pub fn seeded_at(values: &[f64], index: usize) -> Self {
        Self {
            elements: values
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    if j == index {
                        DualNumber::variable(v)
                    } else {
                        DualNumber::constant(v)
                    }
                })
                .collect(),
        }
    }

    /// Create with custom seed vector (directional derivative).
    pub fn with_seeds(values: &[f64], seeds: &[f64]) -> AdResult<Self> {
        if values.len() != seeds.len() {
            return Err(AdError::DimensionMismatch {
                expected: values.len(),
                got: seeds.len(),
            });
        }
        Ok(Self {
            elements: values
                .iter()
                .zip(seeds.iter())
                .map(|(&v, &s)| DualNumber::seeded(v, s))
                .collect(),
        })
    }

    /// Get the length of the vector.
    #[inline]
    pub fn len(&self) -> usize {
        self.elements.len()
    }

    /// Check if the vector is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.elements.is_empty()
    }

    /// Get element at index.
    pub fn get(&self, index: usize) -> Option<&DualNumber> {
        self.elements.get(index)
    }

    /// Get mutable element at index.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut DualNumber> {
        self.elements.get_mut(index)
    }

    /// Extract primal values.
    pub fn primals(&self) -> Vec<f64> {
        self.elements.iter().map(|d| d.primal).collect()
    }

    /// Extract tangent values (partial derivatives).
    pub fn tangents(&self) -> Vec<f64> {
        self.elements.iter().map(|d| d.tangent).collect()
    }

    // ==================== Vector Operations ====================

    /// Element-wise addition.
    pub fn add(&self, other: &DualVector) -> AdResult<DualVector> {
        if self.len() != other.len() {
            return Err(AdError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        Ok(DualVector::new(
            self.elements
                .iter()
                .zip(other.elements.iter())
                .map(|(a, b)| a.add(b))
                .collect(),
        ))
    }

    /// Element-wise subtraction.
    pub fn sub(&self, other: &DualVector) -> AdResult<DualVector> {
        if self.len() != other.len() {
            return Err(AdError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        Ok(DualVector::new(
            self.elements
                .iter()
                .zip(other.elements.iter())
                .map(|(a, b)| a.sub(b))
                .collect(),
        ))
    }

    /// Element-wise multiplication (Hadamard product).
    pub fn hadamard(&self, other: &DualVector) -> AdResult<DualVector> {
        if self.len() != other.len() {
            return Err(AdError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        Ok(DualVector::new(
            self.elements
                .iter()
                .zip(other.elements.iter())
                .map(|(a, b)| a.mul(b))
                .collect(),
        ))
    }

    /// Element-wise division.
    pub fn div_elementwise(&self, other: &DualVector) -> AdResult<DualVector> {
        if self.len() != other.len() {
            return Err(AdError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        let mut result = Vec::with_capacity(self.len());
        for (a, b) in self.elements.iter().zip(other.elements.iter()) {
            result.push(a.div(b)?);
        }
        Ok(DualVector::new(result))
    }

    /// Scalar multiplication.
    pub fn scale(&self, scalar: &DualNumber) -> DualVector {
        DualVector::new(self.elements.iter().map(|e| e.mul(scalar)).collect())
    }

    /// Scalar addition (broadcast).
    pub fn add_scalar(&self, scalar: &DualNumber) -> DualVector {
        DualVector::new(self.elements.iter().map(|e| e.add(scalar)).collect())
    }

    /// Dot product: sum(a[i] * b[i])
    pub fn dot(&self, other: &DualVector) -> AdResult<DualNumber> {
        if self.len() != other.len() {
            return Err(AdError::DimensionMismatch {
                expected: self.len(),
                got: other.len(),
            });
        }
        let mut sum = DualNumber::constant(0.0);
        for (a, b) in self.elements.iter().zip(other.elements.iter()) {
            sum = sum.add(&a.mul(b));
        }
        Ok(sum)
    }

    /// L2 norm (Euclidean): sqrt(sum(x[i]²))
    pub fn norm_l2(&self) -> AdResult<DualNumber> {
        if self.is_empty() {
            return Ok(DualNumber::constant(0.0));
        }
        let mut sum_sq = DualNumber::constant(0.0);
        for e in &self.elements {
            sum_sq = sum_sq.add(&e.sq());
        }
        sum_sq.sqrt()
    }

    /// L1 norm (Manhattan): sum(|x[i]|)
    /// Uses smooth absolute value.
    pub fn norm_l1(&self, epsilon: f64) -> DualNumber {
        let mut sum = DualNumber::constant(0.0);
        for e in &self.elements {
            sum = sum.add(&e.abs_smooth(epsilon));
        }
        sum
    }

    /// Squared L2 norm: sum(x[i]²)
    pub fn norm_sq(&self) -> DualNumber {
        let mut sum = DualNumber::constant(0.0);
        for e in &self.elements {
            sum = sum.add(&e.sq());
        }
        sum
    }

    /// Sum of all elements.
    pub fn sum(&self) -> DualNumber {
        let mut sum = DualNumber::constant(0.0);
        for e in &self.elements {
            sum = sum.add(e);
        }
        sum
    }

    /// Mean of all elements.
    pub fn mean(&self) -> AdResult<DualNumber> {
        if self.is_empty() {
            return Err(AdError::InvalidOperation(
                "mean of empty vector".to_string(),
            ));
        }
        let n = DualNumber::constant(self.len() as f64);
        self.sum().div(&n)
    }

    /// Product of all elements.
    pub fn product(&self) -> DualNumber {
        let mut prod = DualNumber::constant(1.0);
        for e in &self.elements {
            prod = prod.mul(e);
        }
        prod
    }

    /// Element-wise application of a function.
    pub fn map<F>(&self, f: F) -> DualVector
    where
        F: Fn(&DualNumber) -> DualNumber,
    {
        DualVector::new(self.elements.iter().map(f).collect())
    }

    /// Element-wise application of a fallible function.
    pub fn try_map<F>(&self, f: F) -> AdResult<DualVector>
    where
        F: Fn(&DualNumber) -> AdResult<DualNumber>,
    {
        let mut result = Vec::with_capacity(self.len());
        for e in &self.elements {
            result.push(f(e)?);
        }
        Ok(DualVector::new(result))
    }

    /// Normalize to unit length.
    pub fn normalize(&self) -> AdResult<DualVector> {
        let norm = self.norm_l2()?;
        if norm.primal < 1e-15 {
            return Err(AdError::DivisionByZero);
        }
        let mut result = Vec::with_capacity(self.len());
        for e in &self.elements {
            result.push(e.div(&norm)?);
        }
        Ok(DualVector::new(result))
    }

    /// Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x[j]))
    pub fn softmax(&self) -> AdResult<DualVector> {
        if self.is_empty() {
            return Err(AdError::InvalidOperation(
                "softmax of empty vector".to_string(),
            ));
        }
        // For numerical stability, subtract max
        let max_val = self
            .elements
            .iter()
            .map(|e| e.primal)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_d = DualNumber::constant(max_val);
        let shifted: Vec<DualNumber> = self.elements.iter().map(|e| e.sub(&max_d)).collect();
        let exp_vals: Vec<DualNumber> = shifted.iter().map(|e| e.exp()).collect();
        let sum_exp = exp_vals
            .iter()
            .fold(DualNumber::constant(0.0), |acc, e| acc.add(e));
        let mut result = Vec::with_capacity(self.len());
        for e in &exp_vals {
            result.push(e.div(&sum_exp)?);
        }
        Ok(DualVector::new(result))
    }

    /// Log-softmax (numerically stable).
    pub fn log_softmax(&self) -> AdResult<DualVector> {
        if self.is_empty() {
            return Err(AdError::InvalidOperation(
                "log_softmax of empty vector".to_string(),
            ));
        }
        // log_softmax(x)[i] = x[i] - log(sum(exp(x[j])))
        // With numerical stability: x[i] - max - log(sum(exp(x[j] - max)))
        let max_val = self
            .elements
            .iter()
            .map(|e| e.primal)
            .fold(f64::NEG_INFINITY, f64::max);
        let max_d = DualNumber::constant(max_val);
        let shifted: Vec<DualNumber> = self.elements.iter().map(|e| e.sub(&max_d)).collect();
        let sum_exp = shifted
            .iter()
            .map(|e| e.exp())
            .fold(DualNumber::constant(0.0), |acc, e| acc.add(&e));
        let log_sum_exp = sum_exp.ln()?;
        let result: Vec<DualNumber> = shifted.iter().map(|e| e.sub(&log_sum_exp)).collect();
        Ok(DualVector::new(result))
    }
}

impl Default for DualVector {
    fn default() -> Self {
        Self {
            elements: Vec::new(),
        }
    }
}

impl fmt::Display for DualVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, e) in self.elements.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", e)?;
        }
        write!(f, "]")
    }
}

// ============================================================================
// DualValue - Unified scalar/vector type
// ============================================================================

/// Unified dual value type for flexible AD operations.
#[derive(Debug, Clone, PartialEq)]
pub enum DualValue {
    Scalar(DualNumber),
    Vector(DualVector),
}

impl DualValue {
    /// Create a scalar value.
    pub fn scalar(d: DualNumber) -> Self {
        DualValue::Scalar(d)
    }

    /// Create a vector value.
    pub fn vector(v: DualVector) -> Self {
        DualValue::Vector(v)
    }

    /// Check if this is a scalar.
    pub fn is_scalar(&self) -> bool {
        matches!(self, DualValue::Scalar(_))
    }

    /// Check if this is a vector.
    pub fn is_vector(&self) -> bool {
        matches!(self, DualValue::Vector(_))
    }

    /// Get as scalar if applicable.
    pub fn as_scalar(&self) -> Option<&DualNumber> {
        match self {
            DualValue::Scalar(d) => Some(d),
            _ => None,
        }
    }

    /// Get as vector if applicable.
    pub fn as_vector(&self) -> Option<&DualVector> {
        match self {
            DualValue::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Extract primal value(s).
    pub fn primals(&self) -> Vec<f64> {
        match self {
            DualValue::Scalar(d) => vec![d.primal],
            DualValue::Vector(v) => v.primals(),
        }
    }

    /// Extract tangent value(s).
    pub fn tangents(&self) -> Vec<f64> {
        match self {
            DualValue::Scalar(d) => vec![d.tangent],
            DualValue::Vector(v) => v.tangents(),
        }
    }
}

impl fmt::Display for DualValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DualValue::Scalar(d) => write!(f, "{}", d),
            DualValue::Vector(v) => write!(f, "{}", v),
        }
    }
}

// ============================================================================
// DualRecord - Named parameter gradients
// ============================================================================

/// A record of named dual numbers for structured parameter differentiation.
///
/// Useful for models with named parameters where you want to track
/// gradients for each parameter by name.
#[derive(Debug, Clone, PartialEq)]
pub struct DualRecord {
    /// Named dual number fields
    fields: HashMap<String, DualNumber>,
}

impl DualRecord {
    /// Create an empty record.
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    /// Create from a list of (name, value) pairs as constants.
    pub fn from_constants(pairs: &[(&str, f64)]) -> Self {
        let mut fields = HashMap::new();
        for (name, value) in pairs {
            fields.insert(name.to_string(), DualNumber::constant(*value));
        }
        Self { fields }
    }

    /// Create with one variable seeded for differentiation.
    pub fn seeded_at(pairs: &[(&str, f64)], variable_name: &str) -> Self {
        let mut fields = HashMap::new();
        for (name, value) in pairs {
            if *name == variable_name {
                fields.insert(name.to_string(), DualNumber::variable(*value));
            } else {
                fields.insert(name.to_string(), DualNumber::constant(*value));
            }
        }
        Self { fields }
    }

    /// Insert or update a field.
    pub fn insert(&mut self, name: &str, value: DualNumber) {
        self.fields.insert(name.to_string(), value);
    }

    /// Get a field by name.
    pub fn get(&self, name: &str) -> Option<&DualNumber> {
        self.fields.get(name)
    }

    /// Get a mutable field by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut DualNumber> {
        self.fields.get_mut(name)
    }

    /// Check if a field exists.
    pub fn contains(&self, name: &str) -> bool {
        self.fields.contains_key(name)
    }

    /// Get all field names.
    pub fn names(&self) -> Vec<&String> {
        self.fields.keys().collect()
    }

    /// Get the number of fields.
    pub fn len(&self) -> usize {
        self.fields.len()
    }

    /// Check if the record is empty.
    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    /// Extract all primal values as a map.
    pub fn primals(&self) -> HashMap<String, f64> {
        self.fields
            .iter()
            .map(|(k, v)| (k.clone(), v.primal))
            .collect()
    }

    /// Extract all tangent values as a map.
    pub fn tangents(&self) -> HashMap<String, f64> {
        self.fields
            .iter()
            .map(|(k, v)| (k.clone(), v.tangent))
            .collect()
    }

    /// Iterate over fields.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &DualNumber)> {
        self.fields.iter()
    }

    /// Convert to DualVector (ordering by sorted names for determinism).
    pub fn to_vector(&self) -> DualVector {
        let mut names: Vec<_> = self.fields.keys().collect();
        names.sort();
        DualVector::new(names.iter().map(|name| self.fields[*name]).collect())
    }

    /// Create from DualVector with given names.
    pub fn from_vector(names: &[&str], vector: &DualVector) -> AdResult<Self> {
        if names.len() != vector.len() {
            return Err(AdError::DimensionMismatch {
                expected: names.len(),
                got: vector.len(),
            });
        }
        let mut fields = HashMap::new();
        for (name, elem) in names.iter().zip(vector.elements.iter()) {
            fields.insert(name.to_string(), *elem);
        }
        Ok(Self { fields })
    }
}

impl Default for DualRecord {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for DualRecord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        let mut names: Vec<_> = self.fields.keys().collect();
        names.sort();
        for name in names {
            if !first {
                write!(f, ", ")?;
            }
            first = false;
            write!(f, "{}: {}", name, self.fields[name])?;
        }
        write!(f, "}}")
    }
}

// ============================================================================
// AdContext - Debugging and tracing
// ============================================================================

/// Context for AD operations providing debugging and tracing capabilities.
#[derive(Debug, Clone)]
pub struct AdContext {
    /// Enable detailed tracing of operations
    pub trace_enabled: bool,
    /// Collected trace entries
    trace: Vec<TraceEntry>,
    /// Tolerance for numerical comparisons
    pub tolerance: f64,
    /// Enable gradient checking mode
    pub check_gradients: bool,
    /// Finite difference step size for gradient checking
    pub fd_epsilon: f64,
}

/// A single trace entry recording an AD operation.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    pub operation: String,
    pub inputs: Vec<String>,
    pub output: String,
}

impl AdContext {
    /// Create a new AD context with default settings.
    pub fn new() -> Self {
        Self {
            trace_enabled: false,
            trace: Vec::new(),
            tolerance: 1e-10,
            check_gradients: false,
            fd_epsilon: 1e-7,
        }
    }

    /// Create a context with tracing enabled.
    pub fn with_tracing() -> Self {
        Self {
            trace_enabled: true,
            ..Self::new()
        }
    }

    /// Create a context with gradient checking enabled.
    pub fn with_gradient_checking(fd_epsilon: f64) -> Self {
        Self {
            check_gradients: true,
            fd_epsilon,
            ..Self::new()
        }
    }

    /// Record a trace entry.
    pub fn trace(&mut self, operation: &str, inputs: &[&DualNumber], output: &DualNumber) {
        if self.trace_enabled {
            self.trace.push(TraceEntry {
                operation: operation.to_string(),
                inputs: inputs.iter().map(|d| format!("{}", d)).collect(),
                output: format!("{}", output),
            });
        }
    }

    /// Get all trace entries.
    pub fn get_trace(&self) -> &[TraceEntry] {
        &self.trace
    }

    /// Clear the trace.
    pub fn clear_trace(&mut self) {
        self.trace.clear();
    }

    /// Check gradient using finite differences.
    pub fn check_gradient<F>(&self, f: F, x: f64, computed_derivative: f64) -> bool
    where
        F: Fn(f64) -> f64,
    {
        let fd_derivative =
            (f(x + self.fd_epsilon) - f(x - self.fd_epsilon)) / (2.0 * self.fd_epsilon);
        let diff = (computed_derivative - fd_derivative).abs();
        let scale = computed_derivative.abs().max(fd_derivative.abs()).max(1.0);
        diff / scale < self.tolerance * 100.0 // Allow some slack for FD approximation
    }
}

impl Default for AdContext {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON || (a - b).abs() / a.abs().max(b.abs()).max(1.0) < EPSILON
    }

    #[test]
    fn test_dual_constant() {
        let c = DualNumber::constant(5.0);
        assert_eq!(c.primal, 5.0);
        assert_eq!(c.tangent, 0.0);
        assert!(c.is_constant());
    }

    #[test]
    fn test_dual_variable() {
        let x = DualNumber::variable(3.0);
        assert_eq!(x.primal, 3.0);
        assert_eq!(x.tangent, 1.0);
        assert!(!x.is_constant());
    }

    #[test]
    fn test_arithmetic() {
        let x = DualNumber::variable(2.0);
        let y = DualNumber::constant(3.0);

        // x + 3 at x=2: value=5, deriv=1
        let sum = x.add(&y);
        assert_eq!(sum.primal, 5.0);
        assert_eq!(sum.tangent, 1.0);

        // x - 3 at x=2: value=-1, deriv=1
        let diff = x.sub(&y);
        assert_eq!(diff.primal, -1.0);
        assert_eq!(diff.tangent, 1.0);

        // x * 3 at x=2: value=6, deriv=3
        let prod = x.mul(&y);
        assert_eq!(prod.primal, 6.0);
        assert_eq!(prod.tangent, 3.0);

        // x / 3 at x=2: value=2/3, deriv=1/3
        let quot = x.div(&y).unwrap();
        assert!(approx_eq(quot.primal, 2.0 / 3.0));
        assert!(approx_eq(quot.tangent, 1.0 / 3.0));
    }

    #[test]
    fn test_product_rule() {
        // f(x) = x² = x * x
        // f'(x) = 2x
        let x = DualNumber::variable(3.0);
        let x_sq = x.mul(&x);
        assert_eq!(x_sq.primal, 9.0);
        assert_eq!(x_sq.tangent, 6.0); // 2 * 3 = 6
    }

    #[test]
    fn test_chain_rule() {
        // f(x) = sin(x²)
        // f'(x) = cos(x²) * 2x
        let x = DualNumber::variable(1.0);
        let x_sq = x.sq();
        let result = x_sq.sin();

        assert!(approx_eq(result.primal, 1.0_f64.sin())); // sin(1)
        assert!(approx_eq(result.tangent, 1.0_f64.cos() * 2.0)); // cos(1) * 2
    }

    #[test]
    fn test_exp_log() {
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
    fn test_trig() {
        let x = DualNumber::variable(PI / 4.0);

        let sin_x = x.sin();
        assert!(approx_eq(sin_x.primal, (PI / 4.0).sin()));
        assert!(approx_eq(sin_x.tangent, (PI / 4.0).cos()));

        let cos_x = x.cos();
        assert!(approx_eq(cos_x.primal, (PI / 4.0).cos()));
        assert!(approx_eq(cos_x.tangent, -(PI / 4.0).sin()));
    }

    #[test]
    fn test_sigmoid() {
        let x = DualNumber::variable(0.0);
        let sig = x.sigmoid();

        // sigmoid(0) = 0.5
        assert!(approx_eq(sig.primal, 0.5));
        // sigmoid'(0) = 0.5 * 0.5 = 0.25
        assert!(approx_eq(sig.tangent, 0.25));
    }

    #[test]
    fn test_tanh() {
        let x = DualNumber::variable(0.0);
        let tanh_x = x.tanh();

        // tanh(0) = 0
        assert!(approx_eq(tanh_x.primal, 0.0));
        // tanh'(0) = 1 - 0² = 1
        assert!(approx_eq(tanh_x.tangent, 1.0));
    }

    #[test]
    fn test_dual_vector() {
        let v = DualVector::variables(&[1.0, 2.0, 3.0]);
        assert_eq!(v.len(), 3);
        assert!(approx_eq(v.sum().primal, 6.0));
    }

    #[test]
    fn test_dot_product() {
        // Create vectors with custom seeds to test dot product derivative
        let a = DualVector::new(vec![DualNumber::new(1.0, 1.0), DualNumber::new(2.0, 0.0)]);
        let b = DualVector::new(vec![DualNumber::new(3.0, 0.0), DualNumber::new(4.0, 1.0)]);

        // dot = 1*3 + 2*4 = 11
        // d(dot)/d(a1) = 3, d(dot)/d(b2) = 2
        // tangent = 1*3 + 2*1 = 5
        let dot = a.dot(&b).unwrap();
        assert!(approx_eq(dot.primal, 11.0));
        assert!(approx_eq(dot.tangent, 5.0));
    }

    #[test]
    fn test_norm() {
        let v = DualVector::variables(&[3.0, 4.0]);
        let norm = v.norm_l2().unwrap();
        assert!(approx_eq(norm.primal, 5.0)); // sqrt(9 + 16) = 5
    }

    #[test]
    fn test_softmax() {
        let v = DualVector::constants(&[1.0, 2.0, 3.0]);
        let sm = v.softmax().unwrap();

        // Check that softmax sums to 1
        let sum: f64 = sm.primals().iter().sum();
        assert!(approx_eq(sum, 1.0));
    }

    #[test]
    fn test_dual_record() {
        let pairs = [("alpha", 0.5), ("beta", 0.3)];
        let record = DualRecord::from_constants(&pairs);

        assert_eq!(record.len(), 2);
        assert!(record.contains("alpha"));
        assert!(approx_eq(record.get("alpha").unwrap().primal, 0.5));
    }

    #[test]
    fn test_seeded_record() {
        let pairs = [("k", 1.0), ("v", 2.0)];
        let record = DualRecord::seeded_at(&pairs, "k");

        // k is seeded (tangent = 1), v is constant (tangent = 0)
        assert!(approx_eq(record.get("k").unwrap().tangent, 1.0));
        assert!(approx_eq(record.get("v").unwrap().tangent, 0.0));
    }

    #[test]
    fn test_ad_context_trace() {
        let mut ctx = AdContext::with_tracing();
        let x = DualNumber::variable(2.0);
        let y = x.sq();

        ctx.trace("square", &[&x], &y);
        assert_eq!(ctx.get_trace().len(), 1);
        assert_eq!(ctx.get_trace()[0].operation, "square");
    }

    #[test]
    fn test_gradient_check() {
        let ctx = AdContext::with_gradient_checking(1e-7);

        // Test f(x) = x² at x=3
        // f'(x) = 2x = 6
        let result = ctx.check_gradient(|x| x * x, 3.0, 6.0);
        assert!(result);

        // Wrong derivative should fail
        let result = ctx.check_gradient(|x| x * x, 3.0, 5.0);
        assert!(!result);
    }

    #[test]
    fn test_smooth_functions() {
        let x = DualNumber::variable(0.1);

        // Smooth abs near zero
        let abs_smooth = x.abs_smooth(0.001);
        assert!(abs_smooth.primal > 0.0);

        // Smooth sign
        let sign_smooth = x.sign_smooth(0.1);
        assert!(sign_smooth.primal > 0.0);
        assert!(sign_smooth.primal < 1.0);
    }

    #[test]
    fn test_activation_functions() {
        let x = DualNumber::variable(1.0);

        // ReLU smooth
        let relu = x.relu_smooth(1.0);
        assert!(relu.primal > 0.0);

        // GELU
        let gelu = x.gelu();
        assert!(gelu.primal > 0.0);

        // Swish
        let swish = x.swish();
        assert!(swish.primal > 0.0);
    }

    #[test]
    fn test_error_handling() {
        let x = DualNumber::constant(0.0);

        // Division by zero
        let y = DualNumber::variable(1.0);
        assert!(y.div(&x).is_err());

        // Log of non-positive
        let neg = DualNumber::constant(-1.0);
        assert!(neg.ln().is_err());

        // sqrt of negative
        assert!(neg.sqrt().is_err());
    }
}
