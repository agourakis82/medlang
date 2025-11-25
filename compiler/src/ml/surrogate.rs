// Week 29: Surrogate Model Runtime Support
//
// Runtime representation and operations for surrogate models.

use super::backend::BackendKind;
use std::fmt;
use uuid::Uuid;
use serde::Serialize;

/// Handle to a trained surrogate model
///
/// This is an opaque handle that references a trained neural network
/// or other ML model that can approximate mechanistic simulations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SurrogateModelHandle {
    /// Unique identifier for this surrogate model
    pub id: Uuid,

    /// Optional human-readable name
    pub name: Option<String>,
}

impl SurrogateModelHandle {
    /// Create a new surrogate model handle with a fresh UUID
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            name: None,
        }
    }

    /// Create a surrogate model handle with a specific UUID (for testing/deserialization)
    pub fn with_id(id: Uuid) -> Self {
        Self { id, name: None }
    }

    /// Create a named surrogate model handle
    pub fn with_name(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: Some(name),
        }
    }
}

impl Default for SurrogateModelHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SurrogateModelHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = &self.name {
            write!(f, "SurrogateModel({}:{})", name, self.id)
        } else {
            write!(f, "SurrogateModel({})", self.id)
        }
    }
}

/// Configuration for training a surrogate model
///
/// This mirrors the MedLang `SurrogateTrainConfig` record type
/// defined in stdlib/med/ml/surrogate.medlang
#[derive(Debug, Clone, Serialize)]
pub struct SurrogateTrainConfig {
    /// Number of mechanistic simulations to generate for training
    pub n_train: i64,

    /// Backend to use for generating training data (typically Mechanistic)
    pub backend: BackendKind,

    /// Random seed for reproducibility
    pub seed: i64,

    /// Maximum training epochs for neural network
    pub max_epochs: i64,

    /// Batch size for training
    pub batch_size: i64,
}

impl SurrogateTrainConfig {
    /// Create a default configuration suitable for quick prototyping
    pub fn default_quick() -> Self {
        Self {
            n_train: 1000,
            backend: BackendKind::Mechanistic,
            seed: 42,
            max_epochs: 50,
            batch_size: 64,
        }
    }

    /// Create a configuration for production-quality surrogates
    pub fn default_production() -> Self {
        Self {
            n_train: 10000,
            backend: BackendKind::Mechanistic,
            seed: 42,
            max_epochs: 200,
            batch_size: 128,
        }
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<(), SurrogateError> {
        if self.n_train <= 0 {
            return Err(SurrogateError::InvalidConfig(
                "n_train must be positive".to_string(),
            ));
        }

        if self.max_epochs <= 0 {
            return Err(SurrogateError::InvalidConfig(
                "max_epochs must be positive".to_string(),
            ));
        }

        if self.batch_size <= 0 {
            return Err(SurrogateError::InvalidConfig(
                "batch_size must be positive".to_string(),
            ));
        }

        // Can't use Surrogate backend to generate training data (would be circular!)
        if self.backend == BackendKind::Surrogate {
            return Err(SurrogateError::InvalidConfig(
                "Cannot use Surrogate backend to generate training data".to_string(),
            ));
        }

        Ok(())
    }
}

/// Errors related to surrogate model operations
#[derive(Debug, Clone, PartialEq)]
pub enum SurrogateError {
    /// Training failed
    TrainingFailed(String),

    /// Invalid configuration
    InvalidConfig(String),

    /// Surrogate model not found
    NotFound(Uuid),

    /// Surrogate model is not yet trained
    NotTrained(Uuid),
}

impl fmt::Display for SurrogateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SurrogateError::TrainingFailed(msg) => {
                write!(f, "Surrogate training failed: {}", msg)
            }
            SurrogateError::InvalidConfig(msg) => {
                write!(f, "Invalid surrogate configuration: {}", msg)
            }
            SurrogateError::NotFound(id) => {
                write!(f, "Surrogate model not found: {}", id)
            }
            SurrogateError::NotTrained(id) => {
                write!(f, "Surrogate model {} has not been trained yet", id)
            }
        }
    }
}

impl std::error::Error for SurrogateError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surrogate_handle_creation() {
        let handle = SurrogateModelHandle::new();
        assert!(handle.name.is_none());

        let named = SurrogateModelHandle::with_name("TestSurrogate".to_string());
        assert_eq!(named.name.as_deref(), Some("TestSurrogate"));
    }

    #[test]
    fn test_surrogate_handle_display() {
        let handle = SurrogateModelHandle::with_name("MyModel".to_string());
        let display = format!("{}", handle);
        assert!(display.contains("MyModel"));
        assert!(display.contains("SurrogateModel"));
    }

    #[test]
    fn test_train_config_defaults() {
        let quick = SurrogateTrainConfig::default_quick();
        assert_eq!(quick.n_train, 1000);
        assert_eq!(quick.backend, BackendKind::Mechanistic);

        let prod = SurrogateTrainConfig::default_production();
        assert_eq!(prod.n_train, 10000);
        assert_eq!(prod.max_epochs, 200);
    }

    #[test]
    fn test_train_config_validation() {
        let mut cfg = SurrogateTrainConfig::default_quick();
        assert!(cfg.validate().is_ok());

        // Invalid n_train
        cfg.n_train = 0;
        assert!(cfg.validate().is_err());
        cfg.n_train = -1;
        assert!(cfg.validate().is_err());

        // Reset and test other fields
        cfg = SurrogateTrainConfig::default_quick();
        cfg.max_epochs = 0;
        assert!(cfg.validate().is_err());

        cfg = SurrogateTrainConfig::default_quick();
        cfg.batch_size = -10;
        assert!(cfg.validate().is_err());

        // Can't use Surrogate backend to generate training data
        cfg = SurrogateTrainConfig::default_quick();
        cfg.backend = BackendKind::Surrogate;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_surrogate_error_display() {
        let err = SurrogateError::TrainingFailed("test error".to_string());
        assert!(format!("{}", err).contains("training failed"));

        let err = SurrogateError::InvalidConfig("bad config".to_string());
        assert!(format!("{}", err).contains("Invalid surrogate configuration"));

        let id = Uuid::new_v4();
        let err = SurrogateError::NotFound(id);
        assert!(format!("{}", err).contains("not found"));

        let err = SurrogateError::NotTrained(id);
        assert!(format!("{}", err).contains("not been trained"));
    }
}
