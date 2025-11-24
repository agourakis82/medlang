//! IR configuration for neural surrogate / PINN backends

use serde::{Deserialize, Serialize};

/// Configuration for neural surrogate model training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRSurrogateConfig {
    /// Name of the mechanistic model to surrogate
    pub model_name: String,

    /// Input features for the neural model
    /// e.g., ["time", "dose_mg", "WT", "Kd_QM", "Kp_tumor_QM"]
    pub input_features: Vec<String>,

    /// Output observables to predict
    /// e.g., ["TumourVol"]
    pub output_observables: Vec<String>,

    /// Hidden layer sizes for neural network
    pub hidden_layers: Vec<usize>,

    /// Whether to include physics-informed (PINN) loss terms
    pub use_physics_loss: bool,
}

impl IRSurrogateConfig {
    /// Default configuration for the oncology PBPK-QSP-QM model
    pub fn default_oncology_qsp() -> Self {
        IRSurrogateConfig {
            model_name: "Oncology_PBPK_QSP_QM".to_string(),
            input_features: vec![
                "time".to_string(),
                "dose_mg".to_string(),
                "WT".to_string(),
                "Kd_QM".to_string(),
                "Kp_tumor_QM".to_string(),
            ],
            output_observables: vec!["TumourVol".to_string()],
            hidden_layers: vec![64, 64],
            use_physics_loss: false, // Scaffold for Week 15
        }
    }

    /// Get input dimension (number of features)
    pub fn input_dim(&self) -> usize {
        self.input_features.len()
    }

    /// Get output dimension (number of observables)
    pub fn output_dim(&self) -> usize {
        self.output_observables.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_oncology_config() {
        let cfg = IRSurrogateConfig::default_oncology_qsp();

        assert_eq!(cfg.model_name, "Oncology_PBPK_QSP_QM");
        assert_eq!(cfg.input_dim(), 5);
        assert_eq!(cfg.output_dim(), 1);
        assert_eq!(cfg.hidden_layers, vec![64, 64]);
        assert!(!cfg.use_physics_loss);
    }

    #[test]
    fn test_input_features() {
        let cfg = IRSurrogateConfig::default_oncology_qsp();

        assert!(cfg.input_features.contains(&"time".to_string()));
        assert!(cfg.input_features.contains(&"dose_mg".to_string()));
        assert!(cfg.input_features.contains(&"WT".to_string()));
        assert!(cfg.input_features.contains(&"Kd_QM".to_string()));
        assert!(cfg.input_features.contains(&"Kp_tumor_QM".to_string()));
    }
}
