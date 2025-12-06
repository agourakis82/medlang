//! Error Types for Refinement Type Checking
//!
//! Provides detailed error information including counterexamples.

use std::fmt;

use super::syntax::{Predicate, RefinementType};
use crate::ast::Span;

/// A refinement type error
#[derive(Clone, Debug)]
pub struct RefinementError {
    /// The kind of error
    pub kind: RefinementErrorKind,
    /// Source location
    pub span: Option<Span>,
    /// Additional context/notes
    pub notes: Vec<String>,
    /// Suggested fixes
    pub suggestions: Vec<String>,
}

/// Kinds of refinement errors
#[derive(Clone, Debug)]
pub enum RefinementErrorKind {
    /// Subtype check failed
    SubtypeFailed {
        sub: RefinementType,
        sup: RefinementType,
        counterexample: Option<Counterexample>,
    },

    /// Precondition not satisfied
    PreconditionFailed {
        function: String,
        param: String,
        required: Predicate,
        counterexample: Option<Counterexample>,
    },

    /// Postcondition not satisfied
    PostconditionFailed {
        function: String,
        ensures: Predicate,
        counterexample: Option<Counterexample>,
    },

    /// Predicate unsatisfied
    PredicateUnsatisfied {
        expected: Predicate,
        counterexample: Option<Counterexample>,
    },

    /// Array bounds check failed
    BoundsCheckFailed {
        index: String,
        length: Option<String>,
        counterexample: Option<Counterexample>,
    },

    /// Division by zero possible
    DivisionByZeroPossible {
        divisor_expr: String,
        counterexample: Option<Counterexample>,
    },

    /// Dose range violation
    DoseRangeViolation {
        drug: Option<String>,
        computed_dose: String,
        min_dose: Option<String>,
        max_dose: Option<String>,
        counterexample: Option<Counterexample>,
    },

    /// Solver timeout
    SolverTimeout { constraint: String, timeout_ms: u64 },

    /// Solver error
    SolverError(String),
}

/// Error severity
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Warning,
    Error,
    Critical,
}

/// Counterexample showing why a constraint fails
#[derive(Clone, Debug, Default)]
pub struct Counterexample {
    /// Variable assignments in the counterexample
    pub assignments: Vec<(String, String)>,
}

impl Counterexample {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_assignment(mut self, var: impl Into<String>, val: impl Into<String>) -> Self {
        self.assignments.push((var.into(), val.into()));
        self
    }

    pub fn is_empty(&self) -> bool {
        self.assignments.is_empty()
    }
}

impl fmt::Display for Counterexample {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.assignments.is_empty() {
            write!(f, "(no counterexample)")
        } else {
            write!(f, "Counterexample: ")?;
            for (i, (var, val)) in self.assignments.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{} = {}", var, val)?;
            }
            Ok(())
        }
    }
}

impl RefinementError {
    pub fn new(kind: RefinementErrorKind) -> Self {
        Self {
            kind,
            span: None,
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    pub fn with_span(mut self, span: Span) -> Self {
        self.span = Some(span);
        self
    }

    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Subtype failed error
    pub fn subtype_failed(
        sub: RefinementType,
        sup: RefinementType,
        counterexample: Option<Counterexample>,
        span: Option<Span>,
    ) -> Self {
        Self {
            kind: RefinementErrorKind::SubtypeFailed {
                sub,
                sup,
                counterexample,
            },
            span,
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Precondition failed error
    pub fn precondition_failed(
        function: impl Into<String>,
        param: impl Into<String>,
        required: Predicate,
        counterexample: Option<Counterexample>,
        span: Option<Span>,
    ) -> Self {
        Self {
            kind: RefinementErrorKind::PreconditionFailed {
                function: function.into(),
                param: param.into(),
                required,
                counterexample,
            },
            span,
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Bounds check failed error
    pub fn bounds_check_failed(
        index: impl Into<String>,
        length: Option<String>,
        counterexample: Option<Counterexample>,
        span: Option<Span>,
    ) -> Self {
        Self {
            kind: RefinementErrorKind::BoundsCheckFailed {
                index: index.into(),
                length,
                counterexample,
            },
            span,
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Dose range violation error
    pub fn dose_range_violation(
        drug: Option<impl Into<String>>,
        computed_dose: impl Into<String>,
        min_dose: Option<impl Into<String>>,
        max_dose: Option<impl Into<String>>,
        counterexample: Option<Counterexample>,
        span: Option<Span>,
    ) -> Self {
        Self {
            kind: RefinementErrorKind::DoseRangeViolation {
                drug: drug.map(|d| d.into()),
                computed_dose: computed_dose.into(),
                min_dose: min_dose.map(|d| d.into()),
                max_dose: max_dose.map(|d| d.into()),
                counterexample,
            },
            span,
            notes: Vec::new(),
            suggestions: Vec::new(),
        }
    }

    /// Get error code
    pub fn code(&self) -> &'static str {
        match &self.kind {
            RefinementErrorKind::SubtypeFailed { .. } => "E0100",
            RefinementErrorKind::PreconditionFailed { .. } => "E0101",
            RefinementErrorKind::PostconditionFailed { .. } => "E0102",
            RefinementErrorKind::PredicateUnsatisfied { .. } => "E0103",
            RefinementErrorKind::BoundsCheckFailed { .. } => "E0104",
            RefinementErrorKind::DivisionByZeroPossible { .. } => "E0105",
            RefinementErrorKind::DoseRangeViolation { .. } => "E0106",
            RefinementErrorKind::SolverTimeout { .. } => "E0110",
            RefinementErrorKind::SolverError(_) => "E0111",
        }
    }

    /// Get severity
    pub fn severity(&self) -> ErrorSeverity {
        match &self.kind {
            RefinementErrorKind::DoseRangeViolation { .. } => ErrorSeverity::Critical,
            RefinementErrorKind::DivisionByZeroPossible { .. } => ErrorSeverity::Critical,
            RefinementErrorKind::SolverTimeout { .. } => ErrorSeverity::Warning,
            _ => ErrorSeverity::Error,
        }
    }
}

impl fmt::Display for RefinementError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] ", self.code())?;
        match &self.kind {
            RefinementErrorKind::SubtypeFailed {
                sub,
                sup,
                counterexample,
            } => {
                write!(f, "Type {} is not a subtype of {}", sub, sup)?;
                if let Some(ce) = counterexample {
                    write!(f, " ({})", ce)?;
                }
            }
            RefinementErrorKind::PreconditionFailed {
                function,
                param,
                required,
                counterexample,
            } => {
                write!(
                    f,
                    "Precondition for '{}' parameter '{}' not satisfied: {}",
                    function, param, required
                )?;
                if let Some(ce) = counterexample {
                    write!(f, " ({})", ce)?;
                }
            }
            RefinementErrorKind::PostconditionFailed {
                function,
                ensures,
                counterexample,
            } => {
                write!(
                    f,
                    "Postcondition for '{}' not satisfied: {}",
                    function, ensures
                )?;
                if let Some(ce) = counterexample {
                    write!(f, " ({})", ce)?;
                }
            }
            RefinementErrorKind::PredicateUnsatisfied {
                expected,
                counterexample,
            } => {
                write!(f, "Predicate not satisfied: {}", expected)?;
                if let Some(ce) = counterexample {
                    write!(f, " ({})", ce)?;
                }
            }
            RefinementErrorKind::BoundsCheckFailed {
                index,
                length,
                counterexample,
            } => {
                write!(f, "Array bounds check failed: index {}", index)?;
                if let Some(len) = length {
                    write!(f, " exceeds length {}", len)?;
                }
                if let Some(ce) = counterexample {
                    write!(f, " ({})", ce)?;
                }
            }
            RefinementErrorKind::DivisionByZeroPossible {
                divisor_expr,
                counterexample,
            } => {
                write!(f, "Possible division by zero: {}", divisor_expr)?;
                if let Some(ce) = counterexample {
                    write!(f, " ({})", ce)?;
                }
            }
            RefinementErrorKind::DoseRangeViolation {
                drug,
                computed_dose,
                min_dose,
                max_dose,
                counterexample,
            } => {
                write!(f, "Dose range violation")?;
                if let Some(d) = drug {
                    write!(f, " for {}", d)?;
                }
                write!(f, ": computed dose {}", computed_dose)?;
                if let (Some(min), Some(max)) = (min_dose, max_dose) {
                    write!(f, " outside range [{}, {}]", min, max)?;
                }
                if let Some(ce) = counterexample {
                    write!(f, " ({})", ce)?;
                }
            }
            RefinementErrorKind::SolverTimeout {
                constraint,
                timeout_ms,
            } => {
                write!(f, "Solver timeout after {}ms: {}", timeout_ms, constraint)?;
            }
            RefinementErrorKind::SolverError(msg) => {
                write!(f, "Solver error: {}", msg)?;
            }
        }
        Ok(())
    }
}

impl std::error::Error for RefinementError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counterexample() {
        let ce = Counterexample::new()
            .with_assignment("x", "5")
            .with_assignment("y", "10");
        assert_eq!(ce.assignments.len(), 2);
        assert!(!ce.is_empty());
    }

    #[test]
    fn test_error_display() {
        let err = RefinementError::new(RefinementErrorKind::SolverError("test".to_string()));
        let s = format!("{}", err);
        assert!(s.contains("E0111"));
        assert!(s.contains("test"));
    }
}
