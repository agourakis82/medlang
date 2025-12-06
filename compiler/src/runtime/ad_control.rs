// Week 50: Handling of Comparisons and Control Flow in AD Context
//
// Key insight: Comparisons use primal values only.
// Derivatives don't affect control flow decisions.
//
// This module provides utilities for handling dual numbers in:
// - Comparison operations (==, <, >, etc.)
// - Control flow (if-then-else)
// - Smooth approximations for differentiable alternatives

use crate::runtime::ad_ops::{dual_add, dual_mul, dual_sigmoid};
use crate::runtime::dual::DualNumber;

// =============================================================================
// COMPARISON OPERATIONS
// =============================================================================

/// Compare two dual numbers for equality (uses primal only)
#[inline]
pub fn dual_eq(a: DualNumber, b: DualNumber) -> bool {
    (a.primal - b.primal).abs() < f64::EPSILON
}

/// Compare two dual numbers for inequality
#[inline]
pub fn dual_ne(a: DualNumber, b: DualNumber) -> bool {
    !dual_eq(a, b)
}

/// Less than comparison (uses primal only)
#[inline]
pub fn dual_lt(a: DualNumber, b: DualNumber) -> bool {
    a.primal < b.primal
}

/// Less than or equal (uses primal only)
#[inline]
pub fn dual_le(a: DualNumber, b: DualNumber) -> bool {
    a.primal <= b.primal
}

/// Greater than (uses primal only)
#[inline]
pub fn dual_gt(a: DualNumber, b: DualNumber) -> bool {
    a.primal > b.primal
}

/// Greater than or equal (uses primal only)
#[inline]
pub fn dual_ge(a: DualNumber, b: DualNumber) -> bool {
    a.primal >= b.primal
}

/// Compare dual number with scalar
#[inline]
pub fn dual_lt_scalar(a: DualNumber, b: f64) -> bool {
    a.primal < b
}

#[inline]
pub fn dual_le_scalar(a: DualNumber, b: f64) -> bool {
    a.primal <= b
}

#[inline]
pub fn dual_gt_scalar(a: DualNumber, b: f64) -> bool {
    a.primal > b
}

#[inline]
pub fn dual_ge_scalar(a: DualNumber, b: f64) -> bool {
    a.primal >= b
}

#[inline]
pub fn dual_eq_scalar(a: DualNumber, b: f64) -> bool {
    (a.primal - b).abs() < f64::EPSILON
}

// =============================================================================
// CONTROL FLOW THROUGH AD
// =============================================================================

/// Select one of two dual numbers based on condition (branch-free)
///
/// Returns: cond ? a : b
/// Useful when both branches must be evaluated (e.g., for vectorization)
#[inline]
pub fn dual_select(cond: bool, a: DualNumber, b: DualNumber) -> DualNumber {
    if cond {
        a
    } else {
        b
    }
}

/// Select based on dual number condition (uses primal)
#[inline]
pub fn dual_select_on_dual(cond: DualNumber, a: DualNumber, b: DualNumber) -> DualNumber {
    if cond.primal != 0.0 {
        a
    } else {
        b
    }
}

/// Blend two dual numbers smoothly
///
/// Returns: t * a + (1 - t) * b
/// Useful for differentiable interpolation between branches
#[inline]
pub fn dual_lerp(t: DualNumber, a: DualNumber, b: DualNumber) -> DualNumber {
    let one_minus_t = DualNumber {
        primal: 1.0 - t.primal,
        tangent: -t.tangent,
    };
    dual_add(dual_mul(t, a), dual_mul(one_minus_t, b))
}

/// Linear interpolation with scalar t
#[inline]
pub fn dual_lerp_scalar(t: f64, a: DualNumber, b: DualNumber) -> DualNumber {
    DualNumber {
        primal: t * a.primal + (1.0 - t) * b.primal,
        tangent: t * a.tangent + (1.0 - t) * b.tangent,
    }
}

/// Heaviside step function with smooth approximation
///
/// H(x) ≈ sigmoid(k * x) where k controls sharpness
/// As k → ∞, approaches true Heaviside
#[inline]
pub fn dual_smooth_heaviside(x: DualNumber, sharpness: f64) -> DualNumber {
    dual_sigmoid(DualNumber {
        primal: sharpness * x.primal,
        tangent: sharpness * x.tangent,
    })
}

/// Indicator function [x > threshold] with smooth approximation
#[inline]
pub fn dual_smooth_indicator_gt(x: DualNumber, threshold: f64, sharpness: f64) -> DualNumber {
    dual_smooth_heaviside(
        DualNumber {
            primal: x.primal - threshold,
            tangent: x.tangent,
        },
        sharpness,
    )
}

/// Indicator function [x < threshold] with smooth approximation
#[inline]
pub fn dual_smooth_indicator_lt(x: DualNumber, threshold: f64, sharpness: f64) -> DualNumber {
    dual_smooth_heaviside(
        DualNumber {
            primal: threshold - x.primal,
            tangent: -x.tangent,
        },
        sharpness,
    )
}

/// Indicator function [a < x < b] with smooth approximation
pub fn dual_smooth_indicator_between(x: DualNumber, a: f64, b: f64, sharpness: f64) -> DualNumber {
    let gt_a = dual_smooth_indicator_gt(x, a, sharpness);
    let lt_b = dual_smooth_indicator_lt(x, b, sharpness);
    dual_mul(gt_a, lt_b)
}

/// Smooth if-then-else using sigmoid blending
///
/// Approximates: if x > threshold then a else b
/// Uses sigmoid for smooth transition around threshold
pub fn dual_smooth_if_gt(
    x: DualNumber,
    threshold: f64,
    then_val: DualNumber,
    else_val: DualNumber,
    sharpness: f64,
) -> DualNumber {
    let t = dual_smooth_indicator_gt(x, threshold, sharpness);
    dual_lerp(t, then_val, else_val)
}

// =============================================================================
// LOGICAL OPERATIONS (ON PRIMAL VALUES)
// =============================================================================

/// Logical AND (both primals must be non-zero)
#[inline]
pub fn dual_and(a: DualNumber, b: DualNumber) -> bool {
    a.primal != 0.0 && b.primal != 0.0
}

/// Logical OR (at least one primal must be non-zero)
#[inline]
pub fn dual_or(a: DualNumber, b: DualNumber) -> bool {
    a.primal != 0.0 || b.primal != 0.0
}

/// Logical NOT (primal must be zero)
#[inline]
pub fn dual_not(a: DualNumber) -> bool {
    a.primal == 0.0
}

/// Convert bool to dual number (1.0 for true, 0.0 for false)
#[inline]
pub fn bool_to_dual(b: bool) -> DualNumber {
    DualNumber::constant(if b { 1.0 } else { 0.0 })
}

/// Convert dual number to bool (non-zero primal is true)
#[inline]
pub fn dual_to_bool(d: DualNumber) -> bool {
    d.primal != 0.0
}

// =============================================================================
// SMOOTH APPROXIMATIONS FOR PIECEWISE FUNCTIONS
// =============================================================================

/// Smooth ReLU approximation: softplus(x) = log(1 + exp(x))
/// Approaches ReLU as sharpness increases
pub fn dual_smooth_relu(x: DualNumber, _sharpness: f64) -> DualNumber {
    crate::runtime::ad_ops::dual_softplus(x)
}

/// Smooth absolute value using sqrt(x² + ε)
#[inline]
pub fn dual_smooth_abs(x: DualNumber, epsilon: f64) -> DualNumber {
    crate::runtime::ad_ops::dual_smooth_abs(x, epsilon)
}

/// Smooth sign function using tanh
/// sign(x) ≈ tanh(sharpness * x)
#[inline]
pub fn dual_smooth_sign(x: DualNumber, sharpness: f64) -> DualNumber {
    crate::runtime::ad_ops::dual_tanh(DualNumber {
        primal: sharpness * x.primal,
        tangent: sharpness * x.tangent,
    })
}

/// Smooth clamp using smooth min/max
pub fn dual_smooth_clamp(
    x: DualNumber,
    min_val: f64,
    max_val: f64,
    temperature: f64,
) -> DualNumber {
    let min_dual = DualNumber::constant(min_val);
    let max_dual = DualNumber::constant(max_val);
    let clamped_min = crate::runtime::ad_ops::dual_smooth_max(x, min_dual, temperature);
    crate::runtime::ad_ops::dual_smooth_min(clamped_min, max_dual, temperature)
}

// =============================================================================
// WHERE / CONDITIONAL OPERATIONS
// =============================================================================

/// Where operation: element-wise conditional selection
///
/// Returns values from `then_vals` where `condition` is true (non-zero),
/// otherwise values from `else_vals`.
///
/// Note: This is the branching version - derivatives flow through
/// the selected branch only.
pub fn dual_where(condition: DualNumber, then_val: DualNumber, else_val: DualNumber) -> DualNumber {
    if condition.primal != 0.0 {
        then_val
    } else {
        else_val
    }
}

/// Differentiable where using blending
///
/// Instead of hard branching, blends between then_val and else_val
/// based on how "true" the condition is (using sigmoid).
pub fn dual_where_smooth(
    condition: DualNumber,
    then_val: DualNumber,
    else_val: DualNumber,
    sharpness: f64,
) -> DualNumber {
    let t = dual_sigmoid(DualNumber {
        primal: sharpness * condition.primal,
        tangent: sharpness * condition.tangent,
    });
    dual_lerp(t, then_val, else_val)
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparisons() {
        let a = DualNumber::new(2.0, 1.0);
        let b = DualNumber::new(3.0, 0.5);

        assert!(dual_lt(a, b));
        assert!(!dual_gt(a, b));
        assert!(dual_le(a, b));
        assert!(!dual_ge(a, b));
        assert!(!dual_eq(a, b));
        assert!(dual_ne(a, b));
    }

    #[test]
    fn test_select() {
        let a = DualNumber::new(1.0, 1.0);
        let b = DualNumber::new(2.0, 2.0);

        let selected_a = dual_select(true, a, b);
        assert_eq!(selected_a.primal, 1.0);
        assert_eq!(selected_a.tangent, 1.0);

        let selected_b = dual_select(false, a, b);
        assert_eq!(selected_b.primal, 2.0);
        assert_eq!(selected_b.tangent, 2.0);
    }

    #[test]
    fn test_lerp() {
        let a = DualNumber::new(0.0, 1.0);
        let b = DualNumber::new(10.0, 0.0);

        // t = 0.5 should give midpoint
        let mid = dual_lerp_scalar(0.5, a, b);
        assert!((mid.primal - 5.0).abs() < 1e-10);
        assert!((mid.tangent - 0.5).abs() < 1e-10);

        // t = 0 should give b
        let at_0 = dual_lerp_scalar(0.0, a, b);
        assert!((at_0.primal - 10.0).abs() < 1e-10);

        // t = 1 should give a
        let at_1 = dual_lerp_scalar(1.0, a, b);
        assert!((at_1.primal - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_smooth_heaviside() {
        // At x = 0, sigmoid should give 0.5
        let x = DualNumber::variable(0.0);
        let h = dual_smooth_heaviside(x, 1.0);
        assert!((h.primal - 0.5).abs() < 1e-10);

        // At large positive x, should approach 1
        let x_pos = DualNumber::variable(10.0);
        let h_pos = dual_smooth_heaviside(x_pos, 1.0);
        assert!(h_pos.primal > 0.99);

        // At large negative x, should approach 0
        let x_neg = DualNumber::variable(-10.0);
        let h_neg = dual_smooth_heaviside(x_neg, 1.0);
        assert!(h_neg.primal < 0.01);
    }

    #[test]
    fn test_smooth_indicator() {
        let x = DualNumber::variable(5.0);

        // x > 3 should be close to 1
        let ind_gt = dual_smooth_indicator_gt(x, 3.0, 10.0);
        assert!(ind_gt.primal > 0.99);

        // x < 10 should be close to 1
        let ind_lt = dual_smooth_indicator_lt(x, 10.0, 10.0);
        assert!(ind_lt.primal > 0.99);

        // 3 < x < 10 should be close to 1
        let ind_between = dual_smooth_indicator_between(x, 3.0, 10.0, 10.0);
        assert!(ind_between.primal > 0.98);
    }

    #[test]
    fn test_smooth_if() {
        let x = DualNumber::variable(5.0);
        let then_val = DualNumber::constant(100.0);
        let else_val = DualNumber::constant(0.0);

        // x > 3, so result should be close to then_val
        let result = dual_smooth_if_gt(x, 3.0, then_val, else_val, 10.0);
        assert!(result.primal > 99.0);

        // x < 10, so result should be close to else_val
        let result2 = dual_smooth_if_gt(x, 10.0, then_val, else_val, 10.0);
        assert!(result2.primal < 1.0);
    }

    #[test]
    fn test_logical_ops() {
        let true_val = DualNumber::new(1.0, 0.0);
        let false_val = DualNumber::new(0.0, 0.0);

        assert!(dual_and(true_val, true_val));
        assert!(!dual_and(true_val, false_val));
        assert!(dual_or(true_val, false_val));
        assert!(!dual_or(false_val, false_val));
        assert!(!dual_not(true_val));
        assert!(dual_not(false_val));
    }

    #[test]
    fn test_smooth_abs() {
        let x = DualNumber::variable(0.0);
        // Smooth abs at 0 should still be differentiable
        let result = dual_smooth_abs(x, 0.01);
        assert!(result.primal > 0.0);
        assert!(result.tangent.is_finite());
    }

    #[test]
    fn test_smooth_clamp() {
        let x = DualNumber::variable(5.0);
        let clamped = dual_smooth_clamp(x, 0.0, 10.0, 0.1);
        assert!((clamped.primal - 5.0).abs() < 0.5); // Should be close to 5

        let x_low = DualNumber::variable(-5.0);
        let clamped_low = dual_smooth_clamp(x_low, 0.0, 10.0, 0.1);
        assert!(clamped_low.primal > -1.0); // Should be clamped towards 0

        let x_high = DualNumber::variable(15.0);
        let clamped_high = dual_smooth_clamp(x_high, 0.0, 10.0, 0.1);
        assert!(clamped_high.primal < 11.0); // Should be clamped towards 10
    }

    #[test]
    fn test_bool_dual_conversion() {
        let d_true = bool_to_dual(true);
        assert_eq!(d_true.primal, 1.0);
        assert!(d_true.is_constant());

        let d_false = bool_to_dual(false);
        assert_eq!(d_false.primal, 0.0);

        assert!(dual_to_bool(d_true));
        assert!(!dual_to_bool(d_false));
    }
}
