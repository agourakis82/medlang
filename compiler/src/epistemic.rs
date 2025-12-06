//! Epistemic Computing for MedLang
//!
//! This module implements epistemic value tracking inspired by Demetrios,
//! adapted for clinical/medical computing. Values carry confidence scores
//! and provenance information, enabling uncertainty quantification and
//! regulatory compliance.
//!
//! ## Core Concepts
//!
//! - **Confidence**: Numerical score [0.0, 1.0] representing certainty
//! - **Provenance**: Source of the measurement/computation
//! - **Propagation**: Automatic confidence tracking through calculations
//!
//! ## Clinical Applications
//!
//! 1. **Measurement Quality**: Lab assay confidence, LLOQ issues
//! 2. **Imputation**: Track confidence of imputed vs. measured values
//! 3. **Model Predictions**: Uncertainty in pharmacokinetic predictions
//! 4. **Regulatory**: Document confidence in safety/efficacy claims

use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

// =============================================================================
// Provenance Tracking
// =============================================================================

/// Source of a value (for regulatory tracking)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provenance {
    /// Direct measurement from patient/device
    Measurement {
        source: String,     // e.g., "lab_assay_LC_MS_MS"
        timestamp: String,  // ISO 8601 timestamp
        subject_id: String, // Patient identifier
    },

    /// Computed from other values
    Computed {
        operation: String,   // e.g., "division", "exp", "solve_ode"
        inputs: Vec<String>, // Names of input variables
    },

    /// Imputed (missing data handling)
    Imputed {
        method: String, // e.g., "LOCF", "median", "model_based"
        original_missing: bool,
    },

    /// Parameter estimate from model fitting
    Estimated {
        model: String,     // Model name
        method: String,    // e.g., "MCMC", "MLE", "Bayes"
        iterations: usize, // Number of MCMC iterations, etc.
    },

    /// Literature value (external data)
    Literature {
        citation: String,   // DOI or publication reference
        population: String, // Population characteristics
    },

    /// Synthetic (simulated data)
    Synthetic {
        generator: String, // Random number generator
        seed: Option<u64>, // Seed for reproducibility
    },

    /// Unknown provenance
    Unknown,
}

impl fmt::Display for Provenance {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Provenance::Measurement { source, .. } => write!(f, "measured({})", source),
            Provenance::Computed { operation, .. } => write!(f, "computed({})", operation),
            Provenance::Imputed { method, .. } => write!(f, "imputed({})", method),
            Provenance::Estimated { model, method, .. } => {
                write!(f, "estimated({}, {})", model, method)
            }
            Provenance::Literature { citation, .. } => write!(f, "literature({})", citation),
            Provenance::Synthetic { generator, .. } => write!(f, "synthetic({})", generator),
            Provenance::Unknown => write!(f, "unknown"),
        }
    }
}

// =============================================================================
// Knowledge Wrapper
// =============================================================================

/// Epistemic value wrapper: value + confidence + provenance
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Knowledge<T> {
    /// The actual value
    pub value: T,

    /// Confidence score [0.0, 1.0]
    /// 1.0 = completely certain
    /// 0.0 = completely uncertain
    pub confidence: f64,

    /// Source/origin of this value
    pub provenance: Provenance,
}

impl<T> Knowledge<T> {
    /// Create a new epistemic value
    pub fn new(value: T, confidence: f64, provenance: Provenance) -> Result<Self, EpistemicError> {
        if !(0.0..=1.0).contains(&confidence) {
            return Err(EpistemicError::InvalidConfidence { value: confidence });
        }

        Ok(Self {
            value,
            confidence,
            provenance,
        })
    }

    /// Create with full confidence (certain value)
    pub fn certain(value: T, provenance: Provenance) -> Self {
        Self {
            value,
            confidence: 1.0,
            provenance,
        }
    }

    /// Create unknown/uncertain value
    pub fn uncertain(value: T) -> Self {
        Self {
            value,
            confidence: 0.0,
            provenance: Provenance::Unknown,
        }
    }

    /// Map the value while preserving confidence and provenance
    pub fn map<U, F>(self, f: F) -> Knowledge<U>
    where
        F: FnOnce(T) -> U,
    {
        Knowledge {
            value: f(self.value),
            confidence: self.confidence,
            provenance: self.provenance,
        }
    }

    /// Get the underlying value (discarding epistemic information)
    pub fn into_value(self) -> T {
        self.value
    }

    /// Check if confidence exceeds threshold
    pub fn is_reliable(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }
}

impl<T: fmt::Display> fmt::Display for Knowledge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [conf: {:.2}, prov: {}]",
            self.value, self.confidence, self.provenance
        )
    }
}

// =============================================================================
// Confidence Propagation Rules
// =============================================================================

/// Propagate confidence through binary operations
pub fn propagate_binary_confidence(conf1: f64, conf2: f64, operation: &str) -> f64 {
    match operation {
        // Conservative: take minimum confidence
        "add" | "sub" | "mul" | "div" => conf1.min(conf2),

        // Multiplicative: confidence degrades
        "pow" => conf1 * conf2,

        // Average for comparisons
        "eq" | "ne" | "lt" | "le" | "gt" | "ge" => (conf1 + conf2) / 2.0,

        // Unknown operation: minimum confidence
        _ => conf1.min(conf2),
    }
}

/// Propagate confidence through unary operations
pub fn propagate_unary_confidence(conf: f64, operation: &str) -> f64 {
    match operation {
        // Linear operations preserve confidence
        "neg" | "abs" => conf,

        // Nonlinear operations degrade confidence slightly
        "exp" | "log" | "sqrt" => conf * 0.95,

        // Trigonometric: moderate degradation
        "sin" | "cos" | "tan" => conf * 0.9,

        // Unknown operation: conservative
        _ => conf * 0.8,
    }
}

/// Propagate confidence through aggregations
pub fn propagate_aggregate_confidence(confidences: &[f64], operation: &str) -> f64 {
    if confidences.is_empty() {
        return 0.0;
    }

    match operation {
        // Mean: average confidence
        "mean" | "avg" => confidences.iter().sum::<f64>() / confidences.len() as f64,

        // Min/Max: take minimum input confidence
        "min" | "max" => confidences.iter().copied().fold(1.0, f64::min),

        // Sum/Product: geometric mean (penalizes low confidence)
        "sum" | "prod" => {
            let product: f64 = confidences.iter().product();
            product.powf(1.0 / confidences.len() as f64)
        }

        // Unknown: conservative (minimum)
        _ => confidences.iter().copied().fold(1.0, f64::min),
    }
}

// =============================================================================
// Epistemic Operations on Knowledge Values
// =============================================================================

impl Knowledge<f64> {
    /// Add two epistemic values
    pub fn add(&self, other: &Self) -> Self {
        let value = self.value + other.value;
        let confidence = propagate_binary_confidence(self.confidence, other.confidence, "add");
        let provenance = Provenance::Computed {
            operation: "add".to_string(),
            inputs: vec!["lhs".to_string(), "rhs".to_string()],
        };

        Knowledge {
            value,
            confidence,
            provenance,
        }
    }

    /// Subtract two epistemic values
    pub fn sub(&self, other: &Self) -> Self {
        let value = self.value - other.value;
        let confidence = propagate_binary_confidence(self.confidence, other.confidence, "sub");
        let provenance = Provenance::Computed {
            operation: "sub".to_string(),
            inputs: vec!["lhs".to_string(), "rhs".to_string()],
        };

        Knowledge {
            value,
            confidence,
            provenance,
        }
    }

    /// Multiply two epistemic values
    pub fn mul(&self, other: &Self) -> Self {
        let value = self.value * other.value;
        let confidence = propagate_binary_confidence(self.confidence, other.confidence, "mul");
        let provenance = Provenance::Computed {
            operation: "mul".to_string(),
            inputs: vec!["lhs".to_string(), "rhs".to_string()],
        };

        Knowledge {
            value,
            confidence,
            provenance,
        }
    }

    /// Divide two epistemic values
    pub fn div(&self, other: &Self) -> Result<Self, EpistemicError> {
        if other.value.abs() < 1e-10 {
            return Err(EpistemicError::DivisionByZero);
        }

        let value = self.value / other.value;
        let confidence = propagate_binary_confidence(self.confidence, other.confidence, "div");
        let provenance = Provenance::Computed {
            operation: "div".to_string(),
            inputs: vec!["lhs".to_string(), "rhs".to_string()],
        };

        Ok(Knowledge {
            value,
            confidence,
            provenance,
        })
    }

    /// Exponential function
    pub fn exp(&self) -> Self {
        let value = self.value.exp();
        let confidence = propagate_unary_confidence(self.confidence, "exp");
        let provenance = Provenance::Computed {
            operation: "exp".to_string(),
            inputs: vec!["x".to_string()],
        };

        Knowledge {
            value,
            confidence,
            provenance,
        }
    }

    /// Natural logarithm
    pub fn ln(&self) -> Result<Self, EpistemicError> {
        if self.value <= 0.0 {
            return Err(EpistemicError::InvalidOperation {
                operation: "ln".to_string(),
                reason: "logarithm of non-positive number".to_string(),
            });
        }

        let value = self.value.ln();
        let confidence = propagate_unary_confidence(self.confidence, "log");
        let provenance = Provenance::Computed {
            operation: "ln".to_string(),
            inputs: vec!["x".to_string()],
        };

        Ok(Knowledge {
            value,
            confidence,
            provenance,
        })
    }

    /// Power function
    pub fn pow(&self, exponent: f64) -> Self {
        let value = self.value.powf(exponent);
        // Confidence degrades with exponent magnitude
        let confidence = self.confidence * (1.0 / (1.0 + exponent.abs() * 0.1));
        let provenance = Provenance::Computed {
            operation: format!("pow({})", exponent),
            inputs: vec!["base".to_string()],
        };

        Knowledge {
            value,
            confidence,
            provenance,
        }
    }
}

// =============================================================================
// Epistemic Errors
// =============================================================================

#[derive(Debug, Error, Clone, PartialEq)]
pub enum EpistemicError {
    #[error("Invalid confidence value: {value} (must be in [0.0, 1.0])")]
    InvalidConfidence { value: f64 },

    #[error("Division by zero in epistemic computation")]
    DivisionByZero,

    #[error("Invalid operation '{operation}': {reason}")]
    InvalidOperation { operation: String, reason: String },

    #[error("Confidence threshold violation: {context} has confidence {actual:.2}, requires {required:.2}")]
    ConfidenceThresholdViolation {
        context: String,
        actual: f64,
        required: f64,
    },

    #[error("Missing provenance for {context}")]
    MissingProvenance { context: String },
}

// =============================================================================
// Epistemic Type System Integration
// =============================================================================

/// Type wrapper for epistemic values in type system
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EpistemicType {
    /// Regular non-epistemic type
    Plain(String),

    /// Epistemic-wrapped type: Knowledge<T>
    Epistemic {
        inner_type: String,
        min_confidence: Option<f64>, // Optional minimum confidence requirement
    },
}

impl fmt::Display for EpistemicType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EpistemicType::Plain(ty) => write!(f, "{}", ty),
            EpistemicType::Epistemic {
                inner_type,
                min_confidence: Some(conf),
            } => {
                write!(f, "Knowledge<{}> where conf >= {:.2}", inner_type, conf)
            }
            EpistemicType::Epistemic {
                inner_type,
                min_confidence: None,
            } => {
                write!(f, "Knowledge<{}>", inner_type)
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_creation() {
        let k = Knowledge::new(
            42.0,
            0.95,
            Provenance::Measurement {
                source: "test".to_string(),
                timestamp: "2024-01-01".to_string(),
                subject_id: "001".to_string(),
            },
        )
        .unwrap();

        assert_eq!(k.value, 42.0);
        assert_eq!(k.confidence, 0.95);
        assert!(k.is_reliable(0.9));
    }

    #[test]
    fn test_invalid_confidence() {
        let result = Knowledge::new(
            42.0,
            1.5, // Invalid: > 1.0
            Provenance::Unknown,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_knowledge_addition() {
        let k1 = Knowledge::certain(
            10.0,
            Provenance::Computed {
                operation: "test".to_string(),
                inputs: vec![],
            },
        );
        let k2 = Knowledge::new(20.0, 0.8, Provenance::Unknown).unwrap();

        let result = k1.add(&k2);
        assert_eq!(result.value, 30.0);
        assert_eq!(result.confidence, 0.8); // min(1.0, 0.8)
    }

    #[test]
    fn test_confidence_propagation_binary() {
        let conf = propagate_binary_confidence(0.9, 0.8, "add");
        assert_eq!(conf, 0.8); // Minimum

        let conf = propagate_binary_confidence(0.9, 0.8, "pow");
        assert_eq!(conf, 0.72); // Product
    }

    #[test]
    fn test_confidence_propagation_unary() {
        let conf = propagate_unary_confidence(0.9, "neg");
        assert_eq!(conf, 0.9); // Preserved

        let conf = propagate_unary_confidence(1.0, "exp");
        assert_eq!(conf, 0.95); // Slight degradation
    }

    #[test]
    fn test_confidence_propagation_aggregate() {
        let confs = vec![0.9, 0.8, 0.85];

        let mean_conf = propagate_aggregate_confidence(&confs, "mean");
        assert!((mean_conf - 0.85).abs() < 0.01);

        let min_conf = propagate_aggregate_confidence(&confs, "min");
        assert_eq!(min_conf, 0.8);
    }

    #[test]
    fn test_knowledge_exp() {
        let k = Knowledge::certain(1.0, Provenance::Unknown);
        let result = k.exp();

        assert!((result.value - std::f64::consts::E).abs() < 0.0001);
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn test_knowledge_division() {
        let k1 = Knowledge::certain(10.0, Provenance::Unknown);
        let k2 = Knowledge::new(2.0, 0.9, Provenance::Unknown).unwrap();

        let result = k1.div(&k2).unwrap();
        assert_eq!(result.value, 5.0);
        assert_eq!(result.confidence, 0.9);
    }

    #[test]
    fn test_knowledge_division_by_zero() {
        let k1 = Knowledge::certain(10.0, Provenance::Unknown);
        let k2 = Knowledge::certain(0.0, Provenance::Unknown);

        let result = k1.div(&k2);
        assert!(result.is_err());
    }
}
