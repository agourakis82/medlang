//! Data loading and conversion utilities
//!
//! Handles reading NONMEM-style CSV files and converting them to
//! Stan/Julia data formats.

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Structure representing a NONMEM-style dataset
#[derive(Debug, Clone)]
pub struct PKDataset {
    /// Subject IDs
    pub ids: Vec<usize>,

    /// Time points
    pub times: Vec<f64>,

    /// Dose amounts (0 for observation records)
    pub amounts: Vec<f64>,

    /// Observations (missing/0 for dose records)
    pub observations: Vec<f64>,

    /// Event IDs (0=obs, 1=dose)
    pub evids: Vec<usize>,

    /// Covariates (e.g., weight, age, etc.)
    pub covariates: HashMap<String, Vec<f64>>,

    /// Number of unique subjects
    pub n_subjects: usize,
}

impl PKDataset {
    /// Load dataset from CSV file
    pub fn from_csv(path: &Path) -> Result<Self> {
        let content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read CSV file: {}", path.display()))?;

        Self::parse_csv(&content)
    }

    /// Parse CSV content
    fn parse_csv(content: &str) -> Result<Self> {
        let mut lines = content.lines();

        // Parse header
        let header = lines.next().context("Empty CSV file")?;
        let columns: Vec<&str> = header.split(',').map(|s| s.trim()).collect();

        // Find required column indices
        let id_idx = columns
            .iter()
            .position(|&c| c == "ID")
            .context("Missing ID column")?;
        let time_idx = columns
            .iter()
            .position(|&c| c == "TIME")
            .context("Missing TIME column")?;
        let amt_idx = columns
            .iter()
            .position(|&c| c == "AMT")
            .context("Missing AMT column")?;
        let dv_idx = columns
            .iter()
            .position(|&c| c == "DV")
            .context("Missing DV column")?;
        let evid_idx = columns
            .iter()
            .position(|&c| c == "EVID")
            .context("Missing EVID column")?;

        // Find covariate columns (everything else)
        let standard_cols = vec!["ID", "TIME", "AMT", "DV", "EVID"];
        let covariate_cols: Vec<(usize, String)> = columns
            .iter()
            .enumerate()
            .filter(|(_, &col)| !standard_cols.contains(&col))
            .map(|(i, &col)| (i, col.to_string()))
            .collect();

        // Parse data rows
        let mut ids = Vec::new();
        let mut times = Vec::new();
        let mut amounts = Vec::new();
        let mut observations = Vec::new();
        let mut evids = Vec::new();
        let mut covariates: HashMap<String, Vec<f64>> = HashMap::new();

        for (_idx, &(_, ref name)) in covariate_cols.iter().enumerate() {
            covariates.insert(name.clone(), Vec::new());
        }

        for (line_num, line) in lines.enumerate() {
            if line.trim().is_empty() {
                continue;
            }

            let values: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

            if values.len() != columns.len() {
                bail!(
                    "Line {} has {} columns, expected {}",
                    line_num + 2,
                    values.len(),
                    columns.len()
                );
            }

            // Parse required columns
            ids.push(
                values[id_idx]
                    .parse()
                    .with_context(|| format!("Invalid ID at line {}", line_num + 2))?,
            );
            times.push(
                values[time_idx]
                    .parse()
                    .with_context(|| format!("Invalid TIME at line {}", line_num + 2))?,
            );

            // AMT and DV can be "." for missing
            let amt = if values[amt_idx] == "." {
                0.0
            } else {
                values[amt_idx]
                    .parse()
                    .with_context(|| format!("Invalid AMT at line {}", line_num + 2))?
            };
            amounts.push(amt);

            let dv = if values[dv_idx] == "." {
                0.0
            } else {
                values[dv_idx]
                    .parse()
                    .with_context(|| format!("Invalid DV at line {}", line_num + 2))?
            };
            observations.push(dv);

            evids.push(
                values[evid_idx]
                    .parse()
                    .with_context(|| format!("Invalid EVID at line {}", line_num + 2))?,
            );

            // Parse covariates
            for &(col_idx, ref name) in &covariate_cols {
                let val: f64 = values[col_idx]
                    .parse()
                    .with_context(|| format!("Invalid {} at line {}", name, line_num + 2))?;
                covariates.get_mut(name).unwrap().push(val);
            }
        }

        // Count unique subjects
        let mut unique_ids: Vec<usize> = ids.clone();
        unique_ids.sort();
        unique_ids.dedup();
        let n_subjects = unique_ids.len();

        Ok(PKDataset {
            ids,
            times,
            amounts,
            observations,
            evids,
            covariates,
            n_subjects,
        })
    }

    /// Convert to Stan data format (JSON)
    pub fn to_stan_data(&self) -> Result<String> {
        // Separate observation records
        let mut obs_indices = Vec::new();
        for (i, &evid) in self.evids.iter().enumerate() {
            if evid == 0 {
                obs_indices.push(i);
            }
        }

        let n_obs = obs_indices.len();

        // Extract observation-only data
        let subject_ids: Vec<usize> = obs_indices.iter().map(|&i| self.ids[i]).collect();
        let times: Vec<f64> = obs_indices.iter().map(|&i| self.times[i]).collect();
        let observations: Vec<f64> = obs_indices.iter().map(|&i| self.observations[i]).collect();

        // Get covariates by subject (assumes one row per subject for covariates)
        let mut subject_covariates: HashMap<String, Vec<f64>> = HashMap::new();
        for (name, values) in &self.covariates {
            let mut subject_vals = Vec::new();
            let mut seen_ids = Vec::new();

            for (i, &id) in self.ids.iter().enumerate() {
                if !seen_ids.contains(&id) {
                    seen_ids.push(id);
                    subject_vals.push(values[i]);
                }
            }

            subject_covariates.insert(name.clone(), subject_vals);
        }

        // Find dose information (assumes single dose at time 0)
        let dose_amount = self
            .amounts
            .iter()
            .find(|&&amt| amt > 0.0)
            .copied()
            .unwrap_or(100.0);

        // Build Stan data JSON
        let mut data = json!({
            "N": self.n_subjects,
            "n_obs": n_obs,
            "subject_id": subject_ids,
            "time": times,
            "observation": observations,
            "dose_amount": dose_amount,
            "dose_time": 0.0,
            "rtol": 1e-8,
            "atol": 1e-8,
            "max_steps": 100000,
        });

        // Add covariates
        let data_obj = data.as_object_mut().unwrap();
        for (name, values) in subject_covariates {
            data_obj.insert(name, json!(values));
        }

        serde_json::to_string_pretty(&data).context("Failed to serialize Stan data to JSON")
    }

    /// Get summary statistics
    pub fn summary(&self) -> String {
        let n_obs = self.evids.iter().filter(|&&e| e == 0).count();
        let n_dose = self.evids.iter().filter(|&&e| e == 1).count();

        format!(
            "Dataset Summary:\n\
             - Subjects: {}\n\
             - Total records: {}\n\
             - Observations: {}\n\
             - Dose events: {}\n\
             - Covariates: {}\n\
             - Time range: {:.2} - {:.2}",
            self.n_subjects,
            self.ids.len(),
            n_obs,
            n_dose,
            self.covariates
                .keys()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            self.times.iter().copied().fold(f64::INFINITY, f64::min),
            self.times.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_csv() {
        let csv = "\
ID,TIME,AMT,DV,EVID,WT
1,0,100,.,1,70
1,1,.,2.5,0,70
1,2,.,3.1,0,70
2,0,100,.,1,65
2,1,.,2.8,0,65
2,2,.,3.4,0,65";

        let dataset = PKDataset::parse_csv(csv).unwrap();

        assert_eq!(dataset.n_subjects, 2);
        assert_eq!(dataset.ids.len(), 6);
        assert_eq!(dataset.times, vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0]);
        assert_eq!(dataset.evids, vec![1, 0, 0, 1, 0, 0]);
        assert!(dataset.covariates.contains_key("WT"));
    }

    #[test]
    fn test_stan_data_conversion() {
        let csv = "\
ID,TIME,AMT,DV,EVID,WT
1,0,100,.,1,70
1,1,.,2.5,0,70
2,0,100,.,1,65
2,1,.,2.8,0,65";

        let dataset = PKDataset::parse_csv(csv).unwrap();
        let stan_data = dataset.to_stan_data().unwrap();

        assert!(stan_data.contains("\"N\": 2"));
        assert!(stan_data.contains("\"n_obs\": 2"));
        assert!(stan_data.contains("\"WT\""));
    }
}
