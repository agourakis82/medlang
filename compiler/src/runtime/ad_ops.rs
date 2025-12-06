// Week 50: Differentiable Operations for Automatic Differentiation
//
// All operations follow the chain rule:
// d/dx f(g(x)) = f'(g(x)) * g'(x)
//
// For dual numbers (a, a'):
// f((a, a')) = (f(a), f'(a) * a')

use crate::runtime::dual::{AdError, AdResult, DualNumber};

// =============================================================================
// BINARY ARITHMETIC OPERATIONS
// =============================================================================

/// Addition: d/dx (u + v) = u' + v'
#[inline]
pub fn dual_add(a: DualNumber, b: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal + b.primal,
        tangent: a.tangent + b.tangent,
    }
}

/// Subtraction: d/dx (u - v) = u' - v'
#[inline]
pub fn dual_sub(a: DualNumber, b: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal - b.primal,
        tangent: a.tangent - b.tangent,
    }
}

/// Multiplication: d/dx (u * v) = u' * v + u * v' (product rule)
#[inline]
pub fn dual_mul(a: DualNumber, b: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal * b.primal,
        tangent: a.tangent * b.primal + a.primal * b.tangent,
    }
}

/// Division: d/dx (u / v) = (u' * v - u * v') / v² (quotient rule)
#[inline]
pub fn dual_div(a: DualNumber, b: DualNumber) -> AdResult<DualNumber> {
    if b.primal == 0.0 {
        return Err(AdError::DomainError {
            operation: "div".into(),
            value: b.primal,
            message: "division by zero".into(),
        });
    }

    let denom = b.primal * b.primal;
    Ok(DualNumber {
        primal: a.primal / b.primal,
        tangent: (a.tangent * b.primal - a.primal * b.tangent) / denom,
    })
}

/// Modulo: d/dx (u % v) - derivative is u' when v is constant
/// Warning: not continuous, use with caution
pub fn dual_rem(a: DualNumber, b: DualNumber) -> AdResult<DualNumber> {
    if b.primal == 0.0 {
        return Err(AdError::DomainError {
            operation: "rem".into(),
            value: b.primal,
            message: "modulo by zero".into(),
        });
    }

    // Derivative of a % b when b is constant is a' (not continuous at multiples of b)
    if b.tangent != 0.0 {
        return Err(AdError::NotDifferentiable {
            reason: "modulo with non-constant divisor is not differentiable".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal % b.primal,
        tangent: a.tangent,
    })
}

/// Negation: d/dx (-u) = -u'
#[inline]
pub fn dual_neg(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: -a.primal,
        tangent: -a.tangent,
    }
}

// =============================================================================
// POWER FUNCTIONS
// =============================================================================

/// Power: d/dx (u^v) = u^v * (v' * ln(u) + v * u' / u)
/// Special cases:
/// - u^c where c is constant: d/dx = c * u^(c-1) * u'
/// - c^u where c is constant: d/dx = c^u * ln(c) * u'
pub fn dual_pow(base: DualNumber, exp: DualNumber) -> AdResult<DualNumber> {
    // Handle special cases for better numerical stability

    // Case 1: base is zero
    if base.primal == 0.0 {
        if exp.primal > 0.0 {
            // 0^(positive) = 0
            return Ok(DualNumber::constant(0.0));
        } else if exp.primal == 0.0 {
            // 0^0 = 1 (by convention), but derivative is undefined
            return Ok(DualNumber::constant(1.0));
        } else {
            return Err(AdError::DomainError {
                operation: "pow".into(),
                value: base.primal,
                message: "0 raised to negative power".into(),
            });
        }
    }

    // Case 2: base is negative
    if base.primal < 0.0 {
        // Only allow integer exponents
        if exp.primal.fract() != 0.0 || exp.tangent != 0.0 {
            return Err(AdError::DomainError {
                operation: "pow".into(),
                value: base.primal,
                message: "negative base with non-integer exponent".into(),
            });
        }

        let result = base.primal.powf(exp.primal);
        let tangent = exp.primal * base.primal.powf(exp.primal - 1.0) * base.tangent;
        return Ok(DualNumber {
            primal: result,
            tangent,
        });
    }

    // Case 3: general case (positive base)
    let result = base.primal.powf(exp.primal);

    // d/dx (u^v) = u^v * (v' * ln(u) + v * u' / u)
    let log_base = base.primal.ln();
    let tangent = result * (exp.tangent * log_base + exp.primal * base.tangent / base.primal);

    // Check for NaN/Inf
    if tangent.is_nan() || tangent.is_infinite() {
        return Err(AdError::NumericalInstability {
            operation: "pow".into(),
            message: format!(
                "unstable derivative at base={}, exp={}",
                base.primal, exp.primal
            ),
        });
    }

    Ok(DualNumber {
        primal: result,
        tangent,
    })
}

/// Square root: d/dx sqrt(u) = u' / (2 * sqrt(u))
pub fn dual_sqrt(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal < 0.0 {
        return Err(AdError::DomainError {
            operation: "sqrt".into(),
            value: a.primal,
            message: "square root of negative number".into(),
        });
    }

    if a.primal == 0.0 {
        if a.tangent != 0.0 {
            return Err(AdError::NonDifferentiable {
                operation: "sqrt".into(),
                point: 0.0,
                message: "derivative is infinite at x=0".into(),
            });
        }
        return Ok(DualNumber::constant(0.0));
    }

    let sqrt_a = a.primal.sqrt();
    Ok(DualNumber {
        primal: sqrt_a,
        tangent: a.tangent / (2.0 * sqrt_a),
    })
}

/// Cube root: d/dx cbrt(u) = u' / (3 * cbrt(u)²)
pub fn dual_cbrt(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal == 0.0 {
        if a.tangent != 0.0 {
            return Err(AdError::NonDifferentiable {
                operation: "cbrt".into(),
                point: 0.0,
                message: "derivative is infinite at x=0".into(),
            });
        }
        return Ok(DualNumber::constant(0.0));
    }

    let cbrt_a = a.primal.cbrt();
    Ok(DualNumber {
        primal: cbrt_a,
        tangent: a.tangent / (3.0 * cbrt_a * cbrt_a),
    })
}

/// Square: d/dx u² = 2u * u'
#[inline]
pub fn dual_square(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal * a.primal,
        tangent: 2.0 * a.primal * a.tangent,
    }
}

// =============================================================================
// EXPONENTIAL AND LOGARITHMIC FUNCTIONS
// =============================================================================

/// Exponential: d/dx exp(u) = exp(u) * u'
#[inline]
pub fn dual_exp(a: DualNumber) -> DualNumber {
    let exp_a = a.primal.exp();
    DualNumber {
        primal: exp_a,
        tangent: exp_a * a.tangent,
    }
}

/// Exponential base 2: d/dx exp2(u) = exp2(u) * ln(2) * u'
#[inline]
pub fn dual_exp2(a: DualNumber) -> DualNumber {
    let exp2_a = a.primal.exp2();
    DualNumber {
        primal: exp2_a,
        tangent: exp2_a * std::f64::consts::LN_2 * a.tangent,
    }
}

/// exp(x) - 1, more accurate for small x
/// d/dx expm1(u) = exp(u) * u'
#[inline]
pub fn dual_expm1(a: DualNumber) -> DualNumber {
    let exp_a = a.primal.exp();
    DualNumber {
        primal: a.primal.exp_m1(),
        tangent: exp_a * a.tangent,
    }
}

/// Natural logarithm: d/dx ln(u) = u' / u
pub fn dual_log(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal <= 0.0 {
        return Err(AdError::DomainError {
            operation: "log".into(),
            value: a.primal,
            message: "logarithm of non-positive number".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.ln(),
        tangent: a.tangent / a.primal,
    })
}

/// Logarithm base 2: d/dx log2(u) = u' / (u * ln(2))
pub fn dual_log2(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal <= 0.0 {
        return Err(AdError::DomainError {
            operation: "log2".into(),
            value: a.primal,
            message: "logarithm of non-positive number".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.log2(),
        tangent: a.tangent / (a.primal * std::f64::consts::LN_2),
    })
}

/// Logarithm base 10: d/dx log10(u) = u' / (u * ln(10))
pub fn dual_log10(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal <= 0.0 {
        return Err(AdError::DomainError {
            operation: "log10".into(),
            value: a.primal,
            message: "logarithm of non-positive number".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.log10(),
        tangent: a.tangent / (a.primal * std::f64::consts::LN_10),
    })
}

/// ln(1 + x), more accurate for small x
/// d/dx ln1p(u) = u' / (1 + u)
pub fn dual_ln1p(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal <= -1.0 {
        return Err(AdError::DomainError {
            operation: "ln1p".into(),
            value: a.primal,
            message: "ln1p of value <= -1".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.ln_1p(),
        tangent: a.tangent / (1.0 + a.primal),
    })
}

// =============================================================================
// TRIGONOMETRIC FUNCTIONS
// =============================================================================

/// Sine: d/dx sin(u) = cos(u) * u'
#[inline]
pub fn dual_sin(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal.sin(),
        tangent: a.primal.cos() * a.tangent,
    }
}

/// Cosine: d/dx cos(u) = -sin(u) * u'
#[inline]
pub fn dual_cos(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal.cos(),
        tangent: -a.primal.sin() * a.tangent,
    }
}

/// Tangent: d/dx tan(u) = sec²(u) * u' = u' / cos²(u)
pub fn dual_tan(a: DualNumber) -> AdResult<DualNumber> {
    let cos_a = a.primal.cos();
    if cos_a.abs() < 1e-15 {
        return Err(AdError::DomainError {
            operation: "tan".into(),
            value: a.primal,
            message: "tangent undefined (cos = 0)".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.tan(),
        tangent: a.tangent / (cos_a * cos_a),
    })
}

/// Arcsine: d/dx asin(u) = u' / sqrt(1 - u²)
pub fn dual_asin(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal.abs() > 1.0 {
        return Err(AdError::DomainError {
            operation: "asin".into(),
            value: a.primal,
            message: "asin of value outside [-1, 1]".into(),
        });
    }

    if a.primal.abs() == 1.0 && a.tangent != 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "asin".into(),
            point: a.primal,
            message: "derivative is infinite at ±1".into(),
        });
    }

    let denom = (1.0 - a.primal * a.primal).sqrt();
    Ok(DualNumber {
        primal: a.primal.asin(),
        tangent: if denom > 0.0 { a.tangent / denom } else { 0.0 },
    })
}

/// Arccosine: d/dx acos(u) = -u' / sqrt(1 - u²)
pub fn dual_acos(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal.abs() > 1.0 {
        return Err(AdError::DomainError {
            operation: "acos".into(),
            value: a.primal,
            message: "acos of value outside [-1, 1]".into(),
        });
    }

    if a.primal.abs() == 1.0 && a.tangent != 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "acos".into(),
            point: a.primal,
            message: "derivative is infinite at ±1".into(),
        });
    }

    let denom = (1.0 - a.primal * a.primal).sqrt();
    Ok(DualNumber {
        primal: a.primal.acos(),
        tangent: if denom > 0.0 { -a.tangent / denom } else { 0.0 },
    })
}

/// Arctangent: d/dx atan(u) = u' / (1 + u²)
#[inline]
pub fn dual_atan(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal.atan(),
        tangent: a.tangent / (1.0 + a.primal * a.primal),
    }
}

/// Two-argument arctangent: d/dx atan2(y, x)
/// ∂/∂y = x / (x² + y²)
/// ∂/∂x = -y / (x² + y²)
pub fn dual_atan2(y: DualNumber, x: DualNumber) -> AdResult<DualNumber> {
    let denom = x.primal * x.primal + y.primal * y.primal;
    if denom < 1e-30 {
        return Err(AdError::NonDifferentiable {
            operation: "atan2".into(),
            point: 0.0,
            message: "atan2 undefined at origin".into(),
        });
    }

    Ok(DualNumber {
        primal: y.primal.atan2(x.primal),
        tangent: (x.primal * y.tangent - y.primal * x.tangent) / denom,
    })
}

// =============================================================================
// HYPERBOLIC FUNCTIONS
// =============================================================================

/// Hyperbolic sine: d/dx sinh(u) = cosh(u) * u'
#[inline]
pub fn dual_sinh(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal.sinh(),
        tangent: a.primal.cosh() * a.tangent,
    }
}

/// Hyperbolic cosine: d/dx cosh(u) = sinh(u) * u'
#[inline]
pub fn dual_cosh(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal.cosh(),
        tangent: a.primal.sinh() * a.tangent,
    }
}

/// Hyperbolic tangent: d/dx tanh(u) = sech²(u) * u' = (1 - tanh²(u)) * u'
#[inline]
pub fn dual_tanh(a: DualNumber) -> DualNumber {
    let tanh_a = a.primal.tanh();
    DualNumber {
        primal: tanh_a,
        tangent: (1.0 - tanh_a * tanh_a) * a.tangent,
    }
}

/// Inverse hyperbolic sine: d/dx asinh(u) = u' / sqrt(u² + 1)
#[inline]
pub fn dual_asinh(a: DualNumber) -> DualNumber {
    DualNumber {
        primal: a.primal.asinh(),
        tangent: a.tangent / (a.primal * a.primal + 1.0).sqrt(),
    }
}

/// Inverse hyperbolic cosine: d/dx acosh(u) = u' / sqrt(u² - 1)
pub fn dual_acosh(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal < 1.0 {
        return Err(AdError::DomainError {
            operation: "acosh".into(),
            value: a.primal,
            message: "acosh of value < 1".into(),
        });
    }

    if a.primal == 1.0 && a.tangent != 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "acosh".into(),
            point: 1.0,
            message: "derivative is infinite at x=1".into(),
        });
    }

    let denom = (a.primal * a.primal - 1.0).sqrt();
    Ok(DualNumber {
        primal: a.primal.acosh(),
        tangent: if denom > 0.0 { a.tangent / denom } else { 0.0 },
    })
}

/// Inverse hyperbolic tangent: d/dx atanh(u) = u' / (1 - u²)
pub fn dual_atanh(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal.abs() >= 1.0 {
        return Err(AdError::DomainError {
            operation: "atanh".into(),
            value: a.primal,
            message: "atanh of |value| >= 1".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.atanh(),
        tangent: a.tangent / (1.0 - a.primal * a.primal),
    })
}

// =============================================================================
// SPECIAL FUNCTIONS
// =============================================================================

/// Absolute value: d/dx |u| = sign(u) * u'
/// Warning: not differentiable at u = 0
pub fn dual_abs(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal == 0.0 && a.tangent != 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "abs".into(),
            point: 0.0,
            message: "absolute value not differentiable at 0".into(),
        });
    }

    let sign = if a.primal >= 0.0 { 1.0 } else { -1.0 };
    Ok(DualNumber {
        primal: a.primal.abs(),
        tangent: sign * a.tangent,
    })
}

/// Smooth approximation to abs: sqrt(x² + ε)
/// Always differentiable, approaches |x| as ε → 0
#[inline]
pub fn dual_smooth_abs(a: DualNumber, epsilon: f64) -> DualNumber {
    let smoothed = (a.primal * a.primal + epsilon).sqrt();
    DualNumber {
        primal: smoothed,
        tangent: a.primal * a.tangent / smoothed,
    }
}

/// Sign function: returns -1, 0, or 1
/// Not differentiable (derivative is 0 everywhere except origin)
pub fn dual_sign(a: DualNumber) -> AdResult<DualNumber> {
    if a.tangent != 0.0 && a.primal == 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "sign".into(),
            point: 0.0,
            message: "sign function undefined derivative at origin".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.signum(),
        tangent: 0.0, // Zero derivative everywhere
    })
}

/// Floor: not differentiable at integers
pub fn dual_floor(a: DualNumber) -> AdResult<DualNumber> {
    if a.tangent != 0.0 && a.primal.fract() == 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "floor".into(),
            point: a.primal,
            message: "floor not differentiable at integers".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.floor(),
        tangent: 0.0,
    })
}

/// Ceiling: not differentiable at integers
pub fn dual_ceil(a: DualNumber) -> AdResult<DualNumber> {
    if a.tangent != 0.0 && a.primal.fract() == 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "ceil".into(),
            point: a.primal,
            message: "ceil not differentiable at integers".into(),
        });
    }

    Ok(DualNumber {
        primal: a.primal.ceil(),
        tangent: 0.0,
    })
}

/// Minimum of two values: d/dx min(u, v) = u' if u < v, v' if v < u
/// Warning: not differentiable when u = v
pub fn dual_min(a: DualNumber, b: DualNumber) -> AdResult<DualNumber> {
    if a.primal == b.primal && a.tangent != b.tangent {
        return Err(AdError::NonDifferentiable {
            operation: "min".into(),
            point: a.primal,
            message: "min not differentiable when arguments are equal".into(),
        });
    }

    if a.primal <= b.primal {
        Ok(a)
    } else {
        Ok(b)
    }
}

/// Maximum of two values
pub fn dual_max(a: DualNumber, b: DualNumber) -> AdResult<DualNumber> {
    if a.primal == b.primal && a.tangent != b.tangent {
        return Err(AdError::NonDifferentiable {
            operation: "max".into(),
            point: a.primal,
            message: "max not differentiable when arguments are equal".into(),
        });
    }

    if a.primal >= b.primal {
        Ok(a)
    } else {
        Ok(b)
    }
}

/// Clamp value to range
pub fn dual_clamp(x: DualNumber, min_val: DualNumber, max_val: DualNumber) -> AdResult<DualNumber> {
    let min_result = dual_max(x, min_val)?;
    dual_min(min_result, max_val)
}

/// Smooth minimum (softmin): -temperature * log(exp(-a/t) + exp(-b/t))
#[inline]
pub fn dual_smooth_min(a: DualNumber, b: DualNumber, temperature: f64) -> DualNumber {
    let t = temperature;
    let exp_a = (-a.primal / t).exp();
    let exp_b = (-b.primal / t).exp();
    let sum = exp_a + exp_b;

    let primal = -t * sum.ln();
    let tangent = (exp_a * a.tangent + exp_b * b.tangent) / sum;

    DualNumber { primal, tangent }
}

/// Smooth maximum (softmax): temperature * log(exp(a/t) + exp(b/t))
#[inline]
pub fn dual_smooth_max(a: DualNumber, b: DualNumber, temperature: f64) -> DualNumber {
    let t = temperature;
    let exp_a = (a.primal / t).exp();
    let exp_b = (b.primal / t).exp();
    let sum = exp_a + exp_b;

    let primal = t * sum.ln();
    let tangent = (exp_a * a.tangent + exp_b * b.tangent) / sum;

    DualNumber { primal, tangent }
}

// =============================================================================
// ACTIVATION FUNCTIONS (FOR ML)
// =============================================================================

/// Sigmoid: σ(x) = 1 / (1 + exp(-x))
/// d/dx σ(x) = σ(x) * (1 - σ(x))
#[inline]
pub fn dual_sigmoid(a: DualNumber) -> DualNumber {
    let sigmoid = 1.0 / (1.0 + (-a.primal).exp());
    DualNumber {
        primal: sigmoid,
        tangent: sigmoid * (1.0 - sigmoid) * a.tangent,
    }
}

/// ReLU: max(0, x)
/// d/dx ReLU(x) = 1 if x > 0, 0 if x < 0, undefined at 0
pub fn dual_relu(a: DualNumber) -> AdResult<DualNumber> {
    if a.primal == 0.0 && a.tangent != 0.0 {
        return Err(AdError::NonDifferentiable {
            operation: "relu".into(),
            point: 0.0,
            message: "ReLU not differentiable at 0".into(),
        });
    }

    if a.primal > 0.0 {
        Ok(a)
    } else {
        Ok(DualNumber::constant(0.0))
    }
}

/// Leaky ReLU: max(α*x, x) where α is small (e.g., 0.01)
#[inline]
pub fn dual_leaky_relu(a: DualNumber, alpha: f64) -> DualNumber {
    if a.primal >= 0.0 {
        a
    } else {
        DualNumber {
            primal: alpha * a.primal,
            tangent: alpha * a.tangent,
        }
    }
}

/// ELU: Exponential Linear Unit
/// f(x) = x if x >= 0, α(exp(x) - 1) if x < 0
#[inline]
pub fn dual_elu(a: DualNumber, alpha: f64) -> DualNumber {
    if a.primal >= 0.0 {
        a
    } else {
        let exp_a = a.primal.exp();
        DualNumber {
            primal: alpha * (exp_a - 1.0),
            tangent: alpha * exp_a * a.tangent,
        }
    }
}

/// GELU: Gaussian Error Linear Unit
/// Approximation: x * σ(1.702 * x)
#[inline]
pub fn dual_gelu(a: DualNumber) -> DualNumber {
    let k = 1.702;
    let sigmoid_kx = 1.0 / (1.0 + (-k * a.primal).exp());

    let primal = a.primal * sigmoid_kx;

    // d/dx (x * σ(kx)) = σ(kx) + kx * σ(kx) * (1 - σ(kx))
    let tangent = (sigmoid_kx + k * a.primal * sigmoid_kx * (1.0 - sigmoid_kx)) * a.tangent;

    DualNumber { primal, tangent }
}

/// Softplus: log(1 + exp(x))
/// d/dx = exp(x) / (1 + exp(x)) = sigmoid(x)
pub fn dual_softplus(a: DualNumber) -> DualNumber {
    // Use numerically stable formula
    let primal = if a.primal > 20.0 {
        a.primal // Asymptotically approaches x
    } else if a.primal < -20.0 {
        a.primal.exp() // Asymptotically approaches 0
    } else {
        (1.0 + a.primal.exp()).ln()
    };

    let sigmoid = 1.0 / (1.0 + (-a.primal).exp());

    DualNumber {
        primal,
        tangent: sigmoid * a.tangent,
    }
}

/// Swish/SiLU: x * sigmoid(x)
#[inline]
pub fn dual_swish(a: DualNumber) -> DualNumber {
    let sigmoid = 1.0 / (1.0 + (-a.primal).exp());
    let primal = a.primal * sigmoid;
    // d/dx (x * σ(x)) = σ(x) + x * σ(x) * (1 - σ(x)) = σ(x) * (1 + x * (1 - σ(x)))
    let tangent = sigmoid * (1.0 + a.primal * (1.0 - sigmoid)) * a.tangent;

    DualNumber { primal, tangent }
}

/// Mish: x * tanh(softplus(x))
pub fn dual_mish(a: DualNumber) -> DualNumber {
    let sp = dual_softplus(a);
    let tanh_sp = dual_tanh(sp);
    dual_mul(a, tanh_sp)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < EPSILON
    }

    fn finite_diff(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    #[test]
    fn test_dual_add() {
        let a = DualNumber::new(2.0, 1.0);
        let b = DualNumber::new(3.0, 0.0);
        let c = dual_add(a, b);
        assert!(approx_eq(c.primal, 5.0));
        assert!(approx_eq(c.tangent, 1.0));
    }

    #[test]
    fn test_dual_sub() {
        let a = DualNumber::new(5.0, 1.0);
        let b = DualNumber::new(3.0, 0.5);
        let c = dual_sub(a, b);
        assert!(approx_eq(c.primal, 2.0));
        assert!(approx_eq(c.tangent, 0.5));
    }

    #[test]
    fn test_dual_mul() {
        // d/dx (x * 3) = 3
        let a = DualNumber::new(2.0, 1.0); // x = 2
        let b = DualNumber::new(3.0, 0.0); // constant 3
        let c = dual_mul(a, b);
        assert!(approx_eq(c.primal, 6.0));
        assert!(approx_eq(c.tangent, 3.0));
    }

    #[test]
    fn test_dual_div() {
        // d/dx (x / 2) = 0.5
        let a = DualNumber::new(4.0, 1.0);
        let b = DualNumber::new(2.0, 0.0);
        let c = dual_div(a, b).unwrap();
        assert!(approx_eq(c.primal, 2.0));
        assert!(approx_eq(c.tangent, 0.5));
    }

    #[test]
    fn test_dual_exp() {
        // d/dx exp(x) = exp(x)
        let a = DualNumber::variable(1.0);
        let c = dual_exp(a);
        let e = 1.0_f64.exp();
        assert!(approx_eq(c.primal, e));
        assert!(approx_eq(c.tangent, e));
    }

    #[test]
    fn test_dual_log() {
        // d/dx log(x) = 1/x
        let a = DualNumber::variable(2.0);
        let c = dual_log(a).unwrap();
        assert!(approx_eq(c.primal, 2.0_f64.ln()));
        assert!(approx_eq(c.tangent, 0.5));
    }

    #[test]
    fn test_dual_sin_cos() {
        // d/dx sin(x) = cos(x)
        let a = DualNumber::variable(0.5);
        let s = dual_sin(a);
        assert!(approx_eq(s.primal, 0.5_f64.sin()));
        assert!(approx_eq(s.tangent, 0.5_f64.cos()));

        // d/dx cos(x) = -sin(x)
        let c = dual_cos(a);
        assert!(approx_eq(c.primal, 0.5_f64.cos()));
        assert!(approx_eq(c.tangent, -0.5_f64.sin()));
    }

    #[test]
    fn test_dual_pow() {
        // d/dx x^2 = 2x
        let x = DualNumber::variable(3.0);
        let two = DualNumber::constant(2.0);
        let c = dual_pow(x, two).unwrap();
        assert!(approx_eq(c.primal, 9.0));
        assert!(approx_eq(c.tangent, 6.0));
    }

    #[test]
    fn test_dual_sqrt() {
        // d/dx sqrt(x) = 1/(2*sqrt(x))
        let x = DualNumber::variable(4.0);
        let c = dual_sqrt(x).unwrap();
        assert!(approx_eq(c.primal, 2.0));
        assert!(approx_eq(c.tangent, 0.25));
    }

    #[test]
    fn test_dual_tanh() {
        // d/dx tanh(x) = 1 - tanh²(x)
        let x = DualNumber::variable(1.0);
        let c = dual_tanh(x);
        let t = 1.0_f64.tanh();
        assert!(approx_eq(c.primal, t));
        assert!(approx_eq(c.tangent, 1.0 - t * t));
    }

    #[test]
    fn test_dual_sigmoid() {
        // d/dx σ(x) = σ(x)(1-σ(x))
        let x = DualNumber::variable(0.0);
        let c = dual_sigmoid(x);
        assert!(approx_eq(c.primal, 0.5));
        assert!(approx_eq(c.tangent, 0.25)); // 0.5 * 0.5 = 0.25
    }

    #[test]
    fn test_chain_rule() {
        // f(x) = exp(x²), f'(x) = 2x * exp(x²)
        let x = DualNumber::variable(2.0);
        let x_sq = dual_mul(x, x);
        let result = dual_exp(x_sq);

        let expected_primal = (4.0_f64).exp();
        let expected_tangent = 4.0 * expected_primal; // 2 * 2 * exp(4)

        assert!(approx_eq(result.primal, expected_primal));
        assert!(approx_eq(result.tangent, expected_tangent));
    }

    #[test]
    fn test_complex_expression() {
        // f(x) = 3x² + 2x + 1, f'(x) = 6x + 2
        // At x = 2: f(2) = 17, f'(2) = 14
        let x = DualNumber::variable(2.0);
        let three = DualNumber::constant(3.0);
        let two = DualNumber::constant(2.0);
        let one = DualNumber::constant(1.0);

        let term1 = dual_mul(three, dual_mul(x, x)); // 3x²
        let term2 = dual_mul(two, x); // 2x
        let result = dual_add(dual_add(term1, term2), one);

        assert!(approx_eq(result.primal, 17.0));
        assert!(approx_eq(result.tangent, 14.0));
    }

    #[test]
    fn test_finite_diff_validation() {
        // Validate sin derivative
        let x_val = 0.5;
        let ad_grad = dual_sin(DualNumber::variable(x_val)).tangent;
        let fd_grad = finite_diff(|x| x.sin(), x_val, 1e-6);
        assert!((ad_grad - fd_grad).abs() < 1e-5);

        // Validate exp derivative
        let ad_grad = dual_exp(DualNumber::variable(x_val)).tangent;
        let fd_grad = finite_diff(|x| x.exp(), x_val, 1e-6);
        assert!((ad_grad - fd_grad).abs() < 1e-5);

        // Validate tanh derivative
        let ad_grad = dual_tanh(DualNumber::variable(x_val)).tangent;
        let fd_grad = finite_diff(|x| x.tanh(), x_val, 1e-6);
        assert!((ad_grad - fd_grad).abs() < 1e-5);
    }

    #[test]
    fn test_smooth_abs() {
        let x = DualNumber::variable(0.0);
        // Smooth abs should be differentiable at 0
        let result = dual_smooth_abs(x, 0.01);
        assert!(result.primal > 0.0);
        assert!(result.tangent.is_finite());
    }

    #[test]
    fn test_relu() {
        let pos = DualNumber::variable(2.0);
        let result_pos = dual_relu(pos).unwrap();
        assert!(approx_eq(result_pos.primal, 2.0));
        assert!(approx_eq(result_pos.tangent, 1.0));

        let neg = DualNumber::variable(-2.0);
        let result_neg = dual_relu(neg).unwrap();
        assert!(approx_eq(result_neg.primal, 0.0));
        assert!(approx_eq(result_neg.tangent, 0.0));
    }

    #[test]
    fn test_leaky_relu() {
        let pos = DualNumber::variable(2.0);
        let result_pos = dual_leaky_relu(pos, 0.01);
        assert!(approx_eq(result_pos.primal, 2.0));
        assert!(approx_eq(result_pos.tangent, 1.0));

        let neg = DualNumber::variable(-2.0);
        let result_neg = dual_leaky_relu(neg, 0.01);
        assert!(approx_eq(result_neg.primal, -0.02));
        assert!(approx_eq(result_neg.tangent, 0.01));
    }

    #[test]
    fn test_softplus() {
        // softplus(0) = ln(2) ≈ 0.693
        let x = DualNumber::variable(0.0);
        let result = dual_softplus(x);
        assert!((result.primal - 0.693).abs() < 0.01);
        assert!((result.tangent - 0.5).abs() < 0.01); // sigmoid(0) = 0.5
    }

    #[test]
    fn test_domain_errors() {
        // log of negative
        let neg = DualNumber::variable(-1.0);
        assert!(dual_log(neg).is_err());

        // sqrt of negative
        assert!(dual_sqrt(neg).is_err());

        // division by zero
        let zero = DualNumber::constant(0.0);
        let one = DualNumber::variable(1.0);
        assert!(dual_div(one, zero).is_err());
    }
}
