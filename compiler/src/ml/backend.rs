// Week 29: Runtime Backend Kind Mapping
//
// Maps MedLang BackendKind enum values to Rust runtime representation.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Backend kind for evidence program execution
///
/// This mirrors the MedLang `BackendKind` enum defined in stdlib/med/ml/backend.medlang
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendKind {
    /// Full mechanistic model simulation (slow, accurate)
    Mechanistic,

    /// Neural network/ML surrogate (fast, approximate)
    Surrogate,

    /// Combination of mechanistic and surrogate (balanced)
    Hybrid,
}

impl BackendKind {
    /// Convert from MedLang enum variant name
    pub fn from_variant_name(name: &str) -> Result<Self, BackendError> {
        match name {
            "Mechanistic" => Ok(BackendKind::Mechanistic),
            "Surrogate" => Ok(BackendKind::Surrogate),
            "Hybrid" => Ok(BackendKind::Hybrid),
            _ => Err(BackendError::UnknownVariant(name.to_string())),
        }
    }

    /// Get the variant name as a string (for serialization/logging)
    pub fn variant_name(&self) -> &'static str {
        match self {
            BackendKind::Mechanistic => "Mechanistic",
            BackendKind::Surrogate => "Surrogate",
            BackendKind::Hybrid => "Hybrid",
        }
    }

    /// Check if this backend requires a surrogate model
    pub fn requires_surrogate(&self) -> bool {
        matches!(self, BackendKind::Surrogate | BackendKind::Hybrid)
    }

    /// Check if this backend requires mechanistic simulation
    pub fn requires_mechanistic(&self) -> bool {
        matches!(self, BackendKind::Mechanistic | BackendKind::Hybrid)
    }
}

impl fmt::Display for BackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.variant_name())
    }
}

/// Errors related to backend kind operations
#[derive(Debug, Clone, PartialEq)]
pub enum BackendError {
    /// Unknown BackendKind variant name
    UnknownVariant(String),

    /// Backend requires a surrogate model but none was provided
    SurrogateRequired,

    /// Backend configuration is invalid
    InvalidConfig(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::UnknownVariant(name) => {
                write!(f, "Unknown BackendKind variant: '{}'. Expected one of: Mechanistic, Surrogate, Hybrid", name)
            }
            BackendError::SurrogateRequired => {
                write!(
                    f,
                    "This backend requires a surrogate model, but none was provided"
                )
            }
            BackendError::InvalidConfig(msg) => {
                write!(f, "Invalid backend configuration: {}", msg)
            }
        }
    }
}

impl std::error::Error for BackendError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_variant_name() {
        assert_eq!(
            BackendKind::from_variant_name("Mechanistic").unwrap(),
            BackendKind::Mechanistic
        );
        assert_eq!(
            BackendKind::from_variant_name("Surrogate").unwrap(),
            BackendKind::Surrogate
        );
        assert_eq!(
            BackendKind::from_variant_name("Hybrid").unwrap(),
            BackendKind::Hybrid
        );

        assert!(BackendKind::from_variant_name("Unknown").is_err());
        assert!(BackendKind::from_variant_name("mechanistic").is_err()); // case-sensitive
    }

    #[test]
    fn test_variant_name() {
        assert_eq!(BackendKind::Mechanistic.variant_name(), "Mechanistic");
        assert_eq!(BackendKind::Surrogate.variant_name(), "Surrogate");
        assert_eq!(BackendKind::Hybrid.variant_name(), "Hybrid");
    }

    #[test]
    fn test_requires_surrogate() {
        assert!(!BackendKind::Mechanistic.requires_surrogate());
        assert!(BackendKind::Surrogate.requires_surrogate());
        assert!(BackendKind::Hybrid.requires_surrogate());
    }

    #[test]
    fn test_requires_mechanistic() {
        assert!(BackendKind::Mechanistic.requires_mechanistic());
        assert!(!BackendKind::Surrogate.requires_mechanistic());
        assert!(BackendKind::Hybrid.requires_mechanistic());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", BackendKind::Mechanistic), "Mechanistic");
        assert_eq!(format!("{}", BackendKind::Surrogate), "Surrogate");
        assert_eq!(format!("{}", BackendKind::Hybrid), "Hybrid");
    }

    #[test]
    fn test_backend_error_display() {
        let err = BackendError::UnknownVariant("Foo".to_string());
        assert!(format!("{}", err).contains("Unknown BackendKind variant: 'Foo'"));

        let err = BackendError::SurrogateRequired;
        assert!(format!("{}", err).contains("requires a surrogate model"));

        let err = BackendError::InvalidConfig("test".to_string());
        assert!(format!("{}", err).contains("Invalid backend configuration: test"));
    }
}
