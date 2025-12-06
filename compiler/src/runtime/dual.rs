// Week 50: Dual Numbers for Forward-Mode Automatic Differentiation
//
// A dual number (a, a') represents value a with derivative a'.
// Operations propagate derivatives via the chain rule.
//
// For multivariate functions, use multiple passes with different seeds.
// This enables efficient gradient computation for MedLang functions.

use std::fmt;

// =============================================================================
// CORE DUAL NUMBER
// =============================================================================

/// A dual number for forward-mode AD
///
/// Represents a value and its derivative with respect to some variable.
/// For multivariate functions, use multiple passes with different seeds.
#[derive(Clone, Copy, PartialEq)]
pub struct DualNumber {
    /// Primal value (the actual computed value)
    pub primal: f64,
    /// Tangent value (derivative with respect to seeded variable)
    pub tangent: f64,
}

impl DualNumber {
    /// Create a new dual number
    #[inline]
    pub fn new(primal: f64, tangent: f64) -> Self {
        Self { primal, tangent }
    }

    /// Create a constant (tangent = 0)
    #[inline]
    pub fn constant(x: f64) -> Self {
        Self {
            primal: x,
            tangent: 0.0,
        }
    }

    /// Create a variable (tangent = 1, the seed)
    #[inline]
    pub fn variable(x: f64) -> Self {
        Self {
            primal: x,
            tangent: 1.0,
        }
    }

    /// Create a variable with custom seed
    #[inline]
    pub fn variable_with_seed(x: f64, seed: f64) -> Self {
        Self {
            primal: x,
            tangent: seed,
        }
    }

    /// Check if this is effectively a constant
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.tangent == 0.0
    }

    /// Check for NaN in either component
    #[inline]
    pub fn is_nan(&self) -> bool {
        self.primal.is_nan() || self.tangent.is_nan()
    }

    /// Check for infinity in either component
    #[inline]
    pub fn is_infinite(&self) -> bool {
        self.primal.is_infinite() || self.tangent.is_infinite()
    }

    /// Check if dual number is finite
    #[inline]
    pub fn is_finite(&self) -> bool {
        self.primal.is_finite() && self.tangent.is_finite()
    }

    /// Get the value (alias for primal)
    #[inline]
    pub fn value(&self) -> f64 {
        self.primal
    }

    /// Get the derivative (alias for tangent)
    #[inline]
    pub fn derivative(&self) -> f64 {
        self.tangent
    }
}

impl Default for DualNumber {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl fmt::Debug for DualNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual({} + {}ε)", self.primal, self.tangent)
    }
}

impl fmt::Display for DualNumber {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.tangent >= 0.0 {
            write!(f, "{} + {}ε", self.primal, self.tangent)
        } else {
            write!(f, "{} - {}ε", self.primal, -self.tangent)
        }
    }
}

// =============================================================================
// FROM CONVERSIONS
// =============================================================================

impl From<f64> for DualNumber {
    fn from(x: f64) -> Self {
        Self::constant(x)
    }
}

impl From<i64> for DualNumber {
    fn from(x: i64) -> Self {
        Self::constant(x as f64)
    }
}

impl From<i32> for DualNumber {
    fn from(x: i32) -> Self {
        Self::constant(x as f64)
    }
}

// =============================================================================
// AD CONTEXT FOR DEBUGGING AND TRACING
// =============================================================================

/// AD computation trace entry
#[derive(Debug, Clone)]
pub struct AdTraceEntry {
    pub operation: String,
    pub inputs: Vec<DualNumber>,
    pub output: DualNumber,
    pub derivative_formula: String,
}

/// Context for AD computation (optional tracing)
#[derive(Debug, Default)]
pub struct AdContext {
    /// Whether to record trace
    pub trace_enabled: bool,
    /// Computation trace
    pub trace: Vec<AdTraceEntry>,
    /// Derivative depth (for higher-order)
    pub derivative_order: u32,
    /// Active variable name (for debugging)
    pub active_variable: Option<String>,
}

impl AdContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tracing() -> Self {
        Self {
            trace_enabled: true,
            ..Self::default()
        }
    }

    pub fn record(&mut self, op: &str, inputs: &[DualNumber], output: DualNumber, formula: &str) {
        if self.trace_enabled {
            self.trace.push(AdTraceEntry {
                operation: op.to_string(),
                inputs: inputs.to_vec(),
                output,
                derivative_formula: formula.to_string(),
            });
        }
    }

    pub fn clear_trace(&mut self) {
        self.trace.clear();
    }

    /// Generate human-readable trace
    pub fn trace_report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== AD Computation Trace ===\n");
        for (i, entry) in self.trace.iter().enumerate() {
            report.push_str(&format!(
                "Step {}: {} {:?} → {} [d/dx: {}]\n",
                i + 1,
                entry.operation,
                entry.inputs,
                entry.output,
                entry.derivative_formula
            ));
        }
        report
    }
}

// =============================================================================
// AD ERROR TYPES
// =============================================================================

/// Errors specific to automatic differentiation
#[derive(Debug, Clone, PartialEq)]
pub enum AdError {
    /// Domain error (e.g., log of negative number)
    DomainError {
        operation: String,
        value: f64,
        message: String,
    },

    /// Non-differentiable point (e.g., abs at 0)
    NonDifferentiable {
        operation: String,
        point: f64,
        message: String,
    },

    /// Numerical instability detected
    NumericalInstability { operation: String, message: String },

    /// Type error in AD computation
    TypeError { expected: String, found: String },

    /// Function not differentiable (contains non-differentiable ops)
    NotDifferentiable { reason: String },

    /// Division by zero
    DivisionByZero,

    /// Arity mismatch
    ArityMismatch {
        expected: usize,
        found: usize,
        function: String,
    },
}

impl std::fmt::Display for AdError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AdError::DomainError {
                operation,
                value,
                message,
            } => {
                write!(
                    f,
                    "AD domain error in {}: {} (value: {})",
                    operation, message, value
                )
            }
            AdError::NonDifferentiable {
                operation,
                point,
                message,
            } => {
                write!(
                    f,
                    "Non-differentiable point in {} at x={}: {}",
                    operation, point, message
                )
            }
            AdError::NumericalInstability { operation, message } => {
                write!(f, "Numerical instability in {}: {}", operation, message)
            }
            AdError::TypeError { expected, found } => {
                write!(f, "AD type error: expected {}, found {}", expected, found)
            }
            AdError::NotDifferentiable { reason } => {
                write!(f, "Function is not differentiable: {}", reason)
            }
            AdError::DivisionByZero => {
                write!(f, "AD error: division by zero")
            }
            AdError::ArityMismatch {
                expected,
                found,
                function,
            } => {
                write!(
                    f,
                    "AD arity mismatch in {}: expected {} args, got {}",
                    function, expected, found
                )
            }
        }
    }
}

impl std::error::Error for AdError {}

// =============================================================================
// RESULT TYPE ALIAS
// =============================================================================

pub type AdResult<T> = Result<T, AdError>;

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dual_creation() {
        let c = DualNumber::constant(5.0);
        assert_eq!(c.primal, 5.0);
        assert_eq!(c.tangent, 0.0);
        assert!(c.is_constant());

        let v = DualNumber::variable(3.0);
        assert_eq!(v.primal, 3.0);
        assert_eq!(v.tangent, 1.0);
        assert!(!v.is_constant());

        let d = DualNumber::new(2.0, 0.5);
        assert_eq!(d.primal, 2.0);
        assert_eq!(d.tangent, 0.5);
    }

    #[test]
    fn test_dual_display() {
        let d1 = DualNumber::new(3.0, 2.0);
        assert_eq!(format!("{}", d1), "3 + 2ε");

        let d2 = DualNumber::new(3.0, -2.0);
        assert_eq!(format!("{}", d2), "3 - 2ε");
    }

    #[test]
    fn test_dual_from() {
        let d1: DualNumber = 5.0.into();
        assert_eq!(d1.primal, 5.0);
        assert!(d1.is_constant());

        let d2: DualNumber = 42_i64.into();
        assert_eq!(d2.primal, 42.0);
        assert!(d2.is_constant());
    }

    #[test]
    fn test_dual_special_values() {
        let nan = DualNumber::new(f64::NAN, 1.0);
        assert!(nan.is_nan());

        let inf = DualNumber::new(f64::INFINITY, 1.0);
        assert!(inf.is_infinite());
        assert!(!inf.is_finite());

        let normal = DualNumber::new(1.0, 2.0);
        assert!(!normal.is_nan());
        assert!(!normal.is_infinite());
        assert!(normal.is_finite());
    }

    #[test]
    fn test_ad_context() {
        let mut ctx = AdContext::with_tracing();
        assert!(ctx.trace_enabled);

        let d1 = DualNumber::new(1.0, 1.0);
        let d2 = DualNumber::new(2.0, 0.0);
        let result = DualNumber::new(3.0, 1.0);

        ctx.record("add", &[d1, d2], result, "d1' + d2'");
        assert_eq!(ctx.trace.len(), 1);
        assert_eq!(ctx.trace[0].operation, "add");

        ctx.clear_trace();
        assert_eq!(ctx.trace.len(), 0);
    }

    #[test]
    fn test_ad_error_display() {
        let err = AdError::DomainError {
            operation: "log".to_string(),
            value: -1.0,
            message: "logarithm of negative number".to_string(),
        };
        let s = format!("{}", err);
        assert!(s.contains("log"));
        assert!(s.contains("-1"));
    }
}
