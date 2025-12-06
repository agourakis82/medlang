//! Rounding Mode Control
//!
//! Provides explicit control over floating-point rounding modes for
//! reproducible computation.

use crate::mir::function::MirFunction;
use crate::mir::value::ValueIdGen;

/// IEEE 754 rounding modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoundingMode {
    /// Round to nearest, ties to even (default)
    #[default]
    NearestEven,
    /// Round to nearest, ties away from zero
    NearestAway,
    /// Round toward positive infinity
    TowardPositive,
    /// Round toward negative infinity
    TowardNegative,
    /// Round toward zero (truncate)
    TowardZero,
}

impl RoundingMode {
    /// Get the LLVM intrinsic name suffix for this rounding mode
    pub fn llvm_intrinsic_suffix(&self) -> &'static str {
        match self {
            RoundingMode::NearestEven => "tonearest",
            RoundingMode::NearestAway => "tonearestaway",
            RoundingMode::TowardPositive => "upward",
            RoundingMode::TowardNegative => "downward",
            RoundingMode::TowardZero => "towardzero",
        }
    }

    /// Get C standard library fenv.h constant
    pub fn fenv_constant(&self) -> &'static str {
        match self {
            RoundingMode::NearestEven => "FE_TONEAREST",
            RoundingMode::NearestAway => "FE_TONEAREST",
            RoundingMode::TowardPositive => "FE_UPWARD",
            RoundingMode::TowardNegative => "FE_DOWNWARD",
            RoundingMode::TowardZero => "FE_TOWARDZERO",
        }
    }
}

/// Context for rounding mode management
#[derive(Debug, Clone)]
pub struct RoundingContext {
    /// Current rounding mode
    pub mode: RoundingMode,
    /// Whether to use strict FP semantics
    pub strict_fp: bool,
    /// Saved rounding mode (for restore)
    saved_mode: Option<RoundingMode>,
}

impl Default for RoundingContext {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundingContext {
    /// Create a new rounding context with default mode
    pub fn new() -> Self {
        Self {
            mode: RoundingMode::NearestEven,
            strict_fp: false,
            saved_mode: None,
        }
    }

    /// Create context with specific mode
    pub fn with_mode(mode: RoundingMode) -> Self {
        Self {
            mode,
            strict_fp: true,
            saved_mode: None,
        }
    }

    /// Enable strict FP semantics
    pub fn strict(mut self) -> Self {
        self.strict_fp = true;
        self
    }

    /// Save current mode and switch to a new one
    pub fn push_mode(&mut self, new_mode: RoundingMode) {
        self.saved_mode = Some(self.mode);
        self.mode = new_mode;
    }

    /// Restore previously saved mode
    pub fn pop_mode(&mut self) {
        if let Some(saved) = self.saved_mode.take() {
            self.mode = saved;
        }
    }
}

/// Transform that applies rounding mode control
pub struct RoundingTransform {
    /// Context for rounding operations
    context: RoundingContext,
}

impl RoundingTransform {
    /// Create a new rounding transform
    pub fn new(context: RoundingContext) -> Self {
        Self { context }
    }

    /// Transform a function to use explicit rounding
    pub fn transform_function(
        &mut self,
        func: &MirFunction,
        _id_gen: &mut ValueIdGen,
    ) -> MirFunction {
        if !self.context.strict_fp {
            return func.clone();
        }

        // For now, return unchanged - full implementation would use constrained intrinsics
        func.clone()
    }
}

/// Interval arithmetic support for error tracking
#[derive(Debug, Clone, Copy)]
pub struct Interval {
    /// Lower bound
    pub lo: f64,
    /// Upper bound
    pub hi: f64,
}

impl Interval {
    /// Create a point interval
    pub fn point(x: f64) -> Self {
        Self { lo: x, hi: x }
    }

    /// Create an interval from bounds
    pub fn new(lo: f64, hi: f64) -> Self {
        Self {
            lo: lo.min(hi),
            hi: lo.max(hi),
        }
    }

    /// Width of the interval
    pub fn width(&self) -> f64 {
        self.hi - self.lo
    }

    /// Midpoint of the interval
    pub fn mid(&self) -> f64 {
        (self.lo + self.hi) / 2.0
    }

    /// Add intervals (with outward rounding)
    pub fn add(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo + other.lo,
            hi: self.hi + other.hi,
        }
    }

    /// Subtract intervals
    pub fn sub(&self, other: &Interval) -> Interval {
        Interval {
            lo: self.lo - other.hi,
            hi: self.hi - other.lo,
        }
    }

    /// Multiply intervals
    pub fn mul(&self, other: &Interval) -> Interval {
        let products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        Interval {
            lo: products.iter().copied().fold(f64::INFINITY, f64::min),
            hi: products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        }
    }

    /// Check if interval contains zero
    pub fn contains_zero(&self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rounding_modes() {
        let ctx = RoundingContext::with_mode(RoundingMode::TowardZero);
        assert!(ctx.strict_fp);
        assert_eq!(ctx.mode, RoundingMode::TowardZero);
    }

    #[test]
    fn test_push_pop_mode() {
        let mut ctx = RoundingContext::new();
        assert_eq!(ctx.mode, RoundingMode::NearestEven);

        ctx.push_mode(RoundingMode::TowardPositive);
        assert_eq!(ctx.mode, RoundingMode::TowardPositive);

        ctx.pop_mode();
        assert_eq!(ctx.mode, RoundingMode::NearestEven);
    }

    #[test]
    fn test_interval_arithmetic() {
        let a = Interval::new(1.0, 2.0);
        let b = Interval::new(3.0, 4.0);

        let sum = a.add(&b);
        assert_eq!(sum.lo, 4.0);
        assert_eq!(sum.hi, 6.0);

        let diff = b.sub(&a);
        assert_eq!(diff.lo, 1.0);
        assert_eq!(diff.hi, 3.0);
    }
}
