//! Policy Logging for Distillation
//!
//! Provides infrastructure for logging (observation, action) pairs during policy execution.
//! These logs can be used for:
//! - Policy distillation into interpretable MedLang expressions
//! - Training supervised models (decision trees, GLMs) that mimic RL policies
//! - Policy analysis and debugging

use crate::rl::{RlAction, RlObservation};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// A single logged step from policy execution.
///
/// Captures the complete state-action pair along with identifiers for
/// organizing the data (subject, episode, step).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyStepLogRow {
    /// Subject identifier (for multi-subject simulations)
    pub subject_id: usize,

    /// Episode identifier (for multiple episodes per subject)
    pub episode_id: usize,

    /// Step within episode (0-indexed)
    pub step: usize,

    /// Tumor volume (mm³)
    pub tumour: f64,

    /// Absolute neutrophil count (if available)
    pub anc: Option<f64>,

    /// Treatment cycle (0-indexed)
    pub cycle: usize,

    /// Time since treatment start (days)
    pub time_days: f64,

    /// Cumulative dose administered (mg)
    pub cumulative_dose_mg: f64,

    /// Latent model state (for mechanistic models)
    pub latent_state: Vec<f64>,

    /// Dose adjustment factor selected by policy
    pub dose_factor: f64,
}

impl PolicyStepLogRow {
    /// Create a log row from observation and action.
    pub fn from_obs_action(
        subject_id: usize,
        episode_id: usize,
        step: usize,
        obs: &RlObservation,
        action: &RlAction,
    ) -> Self {
        Self {
            subject_id,
            episode_id,
            step,
            tumour: obs.tumour,
            anc: obs.anc,
            cycle: obs.cycle,
            time_days: obs.time_days,
            cumulative_dose_mg: obs.cumulative_dose_mg,
            latent_state: obs.latent_state.clone(),
            dose_factor: action.dose_factor,
        }
    }
}

/// Policy logger that collects step logs in memory.
pub struct PolicyLogger {
    logs: Vec<PolicyStepLogRow>,
}

impl PolicyLogger {
    /// Create a new empty logger.
    pub fn new() -> Self {
        Self { logs: Vec::new() }
    }

    /// Log a step.
    pub fn log(&mut self, row: PolicyStepLogRow) {
        self.logs.push(row);
    }

    /// Get all logged steps.
    pub fn logs(&self) -> &[PolicyStepLogRow] {
        &self.logs
    }

    /// Get number of logged steps.
    pub fn len(&self) -> usize {
        self.logs.len()
    }

    /// Check if logger is empty.
    pub fn is_empty(&self) -> bool {
        self.logs.is_empty()
    }

    /// Clear all logs.
    pub fn clear(&mut self) {
        self.logs.clear();
    }

    /// Write logs to CSV file.
    ///
    /// Columns: SUBJID, EPISODE, STEP, TUMOUR, ANC, CYCLE, TIME_DAYS, CUM_DOSE_MG, DOSE_FACTOR, LATENT_STATE_JSON
    pub fn write_csv<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(
            writer,
            "SUBJID,EPISODE,STEP,TUMOUR,ANC,CYCLE,TIME_DAYS,CUM_DOSE_MG,DOSE_FACTOR,LATENT_STATE_JSON"
        )?;

        // Write rows
        for row in &self.logs {
            let anc_str = row
                .anc
                .map(|a| format!("{:.6}", a))
                .unwrap_or_else(|| "NA".to_string());

            let latent_json =
                serde_json::to_string(&row.latent_state).unwrap_or_else(|_| "[]".to_string());

            writeln!(
                writer,
                "{},{},{},{:.6},{},{},{:.6},{:.6},{:.6},\"{}\"",
                row.subject_id,
                row.episode_id,
                row.step,
                row.tumour,
                anc_str,
                row.cycle,
                row.time_days,
                row.cumulative_dose_mg,
                row.dose_factor,
                latent_json.replace("\"", "\"\"") // Escape quotes for CSV
            )?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Write logs to JSON file.
    pub fn write_json<P: AsRef<Path>>(&self, path: P) -> Result<(), std::io::Error> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.logs)?;
        Ok(())
    }
}

impl Default for PolicyLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_policy_logger_basic() {
        let mut logger = PolicyLogger::new();
        assert_eq!(logger.len(), 0);
        assert!(logger.is_empty());

        let obs = RlObservation {
            tumour: 50.0,
            anc: Some(1.5),
            cycle: 0,
            time_days: 0.0,
            cumulative_dose_mg: 0.0,
            latent_state: vec![1.0, 2.0, 3.0],
        };

        let action = RlAction { dose_factor: 1.0 };

        let row = PolicyStepLogRow::from_obs_action(1, 0, 0, &obs, &action);
        logger.log(row);

        assert_eq!(logger.len(), 1);
        assert!(!logger.is_empty());

        let logs = logger.logs();
        assert_eq!(logs[0].subject_id, 1);
        assert_eq!(logs[0].dose_factor, 1.0);
        assert_eq!(logs[0].tumour, 50.0);
    }

    #[test]
    fn test_policy_logger_clear() {
        let mut logger = PolicyLogger::new();

        let obs = RlObservation {
            tumour: 50.0,
            anc: Some(1.5),
            cycle: 0,
            time_days: 0.0,
            cumulative_dose_mg: 0.0,
            latent_state: vec![],
        };

        let action = RlAction { dose_factor: 1.0 };

        logger.log(PolicyStepLogRow::from_obs_action(1, 0, 0, &obs, &action));
        assert_eq!(logger.len(), 1);

        logger.clear();
        assert_eq!(logger.len(), 0);
        assert!(logger.is_empty());
    }

    #[test]
    fn test_log_row_creation() {
        let obs = RlObservation {
            tumour: 45.5,
            anc: Some(1.2),
            cycle: 3,
            time_days: 63.0,
            cumulative_dose_mg: 900.0,
            latent_state: vec![1.0, 2.0],
        };

        let action = RlAction { dose_factor: 0.8 };

        let row = PolicyStepLogRow::from_obs_action(5, 1, 10, &obs, &action);

        assert_eq!(row.subject_id, 5);
        assert_eq!(row.episode_id, 1);
        assert_eq!(row.step, 10);
        assert_eq!(row.tumour, 45.5);
        assert_eq!(row.anc, Some(1.2));
        assert_eq!(row.cycle, 3);
        assert_eq!(row.time_days, 63.0);
        assert_eq!(row.cumulative_dose_mg, 900.0);
        assert_eq!(row.dose_factor, 0.8);
        assert_eq!(row.latent_state, vec![1.0, 2.0]);
    }
}
