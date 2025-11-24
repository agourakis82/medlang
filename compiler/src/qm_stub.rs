/// Quantum Mechanics Stub Module
///
/// Provides Track C integration for MedLang via JSON-based quantum stubs.
/// These stubs contain precomputed quantum properties (Kd, ΔG values) that
/// inform PK-PD parameters like EC50 and tissue partition coefficients.
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Quantum stub containing precomputed molecular properties
///
/// # Fields
/// - `drug_id`: Unique identifier for the drug molecule
/// - `target_id`: Biological target (e.g., receptor, enzyme)
/// - `Kd_M`: Dissociation constant in molar units [M]
/// - `dG_bind_kcal_per_mol`: Binding free energy [kcal/mol]
/// - `dG_part_plasma_tumor_kcal_per_mol`: Partition free energy plasma→tumor [kcal/mol]
/// - `T_ref_K`: Reference temperature [K], typically 310 K (37°C)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumStub {
    pub drug_id: String,
    pub target_id: String,
    pub Kd_M: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dG_bind_kcal_per_mol: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dG_part_plasma_tumor_kcal_per_mol: Option<f64>,
    #[serde(default = "default_temperature")]
    pub T_ref_K: f64,
}

fn default_temperature() -> f64 {
    310.0 // 37°C in Kelvin
}

/// Errors that can occur when loading or using quantum stubs
#[derive(Debug, thiserror::Error)]
pub enum QmStubError {
    #[error("IO error reading quantum stub: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON parse error in quantum stub: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Invalid quantum stub: {0}")]
    Validation(String),
}

impl QuantumStub {
    /// Load a quantum stub from a JSON file
    ///
    /// # Example
    /// ```no_run
    /// use medlangc::qm_stub::QuantumStub;
    ///
    /// let stub = QuantumStub::load("data/lig001_egfr_qm.json").unwrap();
    /// assert!(stub.Kd_M > 0.0);
    /// ```
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, QmStubError> {
        let bytes = std::fs::read(path)?;
        let stub: QuantumStub = serde_json::from_slice(&bytes)?;
        stub.validate()?;
        Ok(stub)
    }

    /// Validate the quantum stub values
    fn validate(&self) -> Result<(), QmStubError> {
        if self.Kd_M <= 0.0 {
            return Err(QmStubError::Validation(
                "Kd_M must be positive (dissociation constant)".into(),
            ));
        }

        if self.T_ref_K <= 0.0 {
            return Err(QmStubError::Validation(
                "T_ref_K must be positive (temperature in Kelvin)".into(),
            ));
        }

        // Reasonable bounds checking
        if self.Kd_M > 1.0 {
            eprintln!(
                "Warning: Kd_M = {:.2e} M is unusually high (> 1 M). \
                 Typical drug Kd values are nM to µM range.",
                self.Kd_M
            );
        }

        if self.T_ref_K < 273.0 || self.T_ref_K > 350.0 {
            eprintln!(
                "Warning: T_ref_K = {:.1} K is outside typical biological range (273-350 K)",
                self.T_ref_K
            );
        }

        Ok(())
    }

    /// Compute EC50 from quantum Kd with a scaling factor
    ///
    /// # Formula
    /// EC50 = alpha_EC50 * Kd_QM
    ///
    /// where alpha_EC50 accounts for in vivo vs in vitro differences
    /// (protein binding, active transport, etc.)
    ///
    /// # Arguments
    /// - `alpha_ec50`: Scaling factor (typically 0.1 to 10)
    ///
    /// # Returns
    /// EC50 value in the same units as Kd (molar)
    pub fn ec50_from_kd(&self, alpha_ec50: f64) -> f64 {
        alpha_ec50 * self.Kd_M
    }

    /// Compute tissue partition coefficient from free energy of partition
    ///
    /// # Formula
    /// Kp_tumor = exp(-ΔG_part / (R*T))
    ///
    /// where:
    /// - ΔG_part: free energy difference plasma→tumor [kcal/mol]
    /// - R: gas constant = 0.0019872041 kcal/(mol·K)
    /// - T: temperature [K]
    ///
    /// Negative ΔG indicates favorable partition into tumor (Kp > 1)
    ///
    /// # Returns
    /// - `Some(Kp)` if dG_part is available
    /// - `None` if dG_part was not provided in stub
    pub fn kp_tumor_from_dg(&self) -> Option<f64> {
        const R_KCAL: f64 = 0.0019872041; // kcal/(mol·K)

        self.dG_part_plasma_tumor_kcal_per_mol.map(|dg| {
            let exponent = -dg / (R_KCAL * self.T_ref_K);
            exponent.exp()
        })
    }

    /// Compute binding affinity from free energy (if available)
    ///
    /// # Formula
    /// Kd = exp(ΔG_bind / (R*T))
    ///
    /// This is the thermodynamic relationship. If both Kd and ΔG_bind
    /// are provided, they should be consistent. This method can be used
    /// to check consistency or compute Kd if only ΔG_bind is given.
    pub fn kd_from_binding_energy(&self) -> Option<f64> {
        const R_KCAL: f64 = 0.0019872041;

        self.dG_bind_kcal_per_mol.map(|dg| {
            let exponent = dg / (R_KCAL * self.T_ref_K);
            exponent.exp()
        })
    }

    /// Check thermodynamic consistency between Kd and ΔG_bind
    ///
    /// Returns relative error if both are present, None otherwise
    pub fn check_consistency(&self) -> Option<f64> {
        if let Some(kd_from_dg) = self.kd_from_binding_energy() {
            let rel_error = (self.Kd_M - kd_from_dg).abs() / self.Kd_M;
            Some(rel_error)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_ec50_calculation() {
        let stub = QuantumStub {
            drug_id: "TEST".into(),
            target_id: "TARGET".into(),
            Kd_M: 1e-9,
            dG_bind_kcal_per_mol: Some(-12.0),
            dG_part_plasma_tumor_kcal_per_mol: None,
            T_ref_K: 310.0,
        };

        let ec50 = stub.ec50_from_kd(1.0);
        assert!((ec50 - 1e-9).abs() < 1e-15);

        let ec50_scaled = stub.ec50_from_kd(2.5);
        assert!((ec50_scaled - 2.5e-9).abs() < 1e-15);
    }

    #[test]
    fn test_kp_calculation() {
        let stub = QuantumStub {
            drug_id: "TEST".into(),
            target_id: "TARGET".into(),
            Kd_M: 1e-9,
            dG_bind_kcal_per_mol: None,
            dG_part_plasma_tumor_kcal_per_mol: Some(-1.0),
            T_ref_K: 310.0,
        };

        let kp = stub.kp_tumor_from_dg().unwrap();
        // Negative ΔG means favorable partition, so Kp > 1
        assert!(kp > 1.0);

        // Check approximate value
        // exp(-(-1.0) / (0.00198720 * 310)) ≈ exp(1.62) ≈ 5.05
        assert!((kp - 5.05).abs() < 0.1);
    }

    #[test]
    fn test_kp_positive_dg() {
        let stub = QuantumStub {
            drug_id: "TEST".into(),
            target_id: "TARGET".into(),
            Kd_M: 1e-9,
            dG_bind_kcal_per_mol: None,
            dG_part_plasma_tumor_kcal_per_mol: Some(1.0),
            T_ref_K: 310.0,
        };

        let kp = stub.kp_tumor_from_dg().unwrap();
        // Positive ΔG means unfavorable partition, so Kp < 1
        assert!(kp < 1.0);
    }

    #[test]
    fn test_kp_none_when_no_dg() {
        let stub = QuantumStub {
            drug_id: "TEST".into(),
            target_id: "TARGET".into(),
            Kd_M: 1e-9,
            dG_bind_kcal_per_mol: None,
            dG_part_plasma_tumor_kcal_per_mol: None,
            T_ref_K: 310.0,
        };

        assert!(stub.kp_tumor_from_dg().is_none());
    }

    #[test]
    fn test_thermodynamic_consistency() {
        // Create a thermodynamically consistent stub
        let kd: f64 = 1e-9;
        let t: f64 = 310.0;
        let r: f64 = 0.0019872041;
        let dg = r * t * kd.ln(); // Should be about -12.4 kcal/mol

        let stub = QuantumStub {
            drug_id: "TEST".into(),
            target_id: "TARGET".into(),
            Kd_M: kd,
            dG_bind_kcal_per_mol: Some(dg),
            dG_part_plasma_tumor_kcal_per_mol: None,
            T_ref_K: t,
        };

        let kd_computed = stub.kd_from_binding_energy().unwrap();
        let rel_error = (kd - kd_computed).abs() / kd;
        assert!(rel_error < 1e-10);

        let consistency = stub.check_consistency().unwrap();
        assert!(consistency < 1e-10);
    }

    #[test]
    fn test_json_loading() {
        let json_content = r#"{
            "drug_id": "LIG001",
            "target_id": "EGFR",
            "Kd_M": 2.5e-9,
            "dG_bind_kcal_per_mol": -11.7,
            "dG_part_plasma_tumor_kcal_per_mol": -0.8,
            "T_ref_K": 310.0
        }"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(json_content.as_bytes()).unwrap();

        let stub = QuantumStub::load(temp_file.path()).unwrap();

        assert_eq!(stub.drug_id, "LIG001");
        assert_eq!(stub.target_id, "EGFR");
        assert!((stub.Kd_M - 2.5e-9).abs() < 1e-15);
        assert_eq!(stub.dG_bind_kcal_per_mol, Some(-11.7));
        assert_eq!(stub.T_ref_K, 310.0);
    }

    #[test]
    fn test_json_with_defaults() {
        let json_content = r#"{
            "drug_id": "LIG002",
            "target_id": "VEGFR",
            "Kd_M": 5.0e-10
        }"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(json_content.as_bytes()).unwrap();

        let stub = QuantumStub::load(temp_file.path()).unwrap();

        assert_eq!(stub.T_ref_K, 310.0); // Default temperature
        assert!(stub.dG_bind_kcal_per_mol.is_none());
        assert!(stub.dG_part_plasma_tumor_kcal_per_mol.is_none());
    }

    #[test]
    fn test_validation_negative_kd() {
        let json_content = r#"{
            "drug_id": "BAD",
            "target_id": "TARGET",
            "Kd_M": -1.0,
            "T_ref_K": 310.0
        }"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(json_content.as_bytes()).unwrap();

        let result = QuantumStub::load(temp_file.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Kd_M must be positive"));
    }

    #[test]
    fn test_validation_zero_temperature() {
        let json_content = r#"{
            "drug_id": "BAD",
            "target_id": "TARGET",
            "Kd_M": 1e-9,
            "T_ref_K": 0.0
        }"#;

        let mut temp_file = NamedTempFile::new().unwrap();
        temp_file.write_all(json_content.as_bytes()).unwrap();

        let result = QuantumStub::load(temp_file.path());
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("T_ref_K must be positive"));
    }
}
