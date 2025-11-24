//! CDISC-like Export for MedLang Trial Data
//!
//! Generates ADSL (subject-level) and ADTR (tumor response) datasets
//! suitable for regulatory submissions (simplified eCTD structure)

use crate::data::trial::{TrialDataset, TrialRow};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// =============================================================================
// ADSL (Subject-Level Analysis Dataset)
// =============================================================================

/// ADSL row: subject-level characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdslRow {
    pub studyid: String,   // Study identifier
    pub subjid: u32,       // Subject identifier
    pub arm: String,       // Treatment arm
    pub dose_mg: f64,      // Dose in mg
    pub weight_kg: f64,    // Subject weight
    pub baseline_vol: f64, // Baseline tumor volume
    pub n_obs: u32,        // Number of observations
    pub last_obs_day: f64, // Last observation day
    pub status: String,    // "enrolled", "completed", "withdrew"
}

impl AdslRow {
    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{},{}",
            self.studyid,
            self.subjid,
            self.arm,
            self.dose_mg,
            self.weight_kg,
            self.baseline_vol,
            self.n_obs,
            self.last_obs_day,
            self.status
        )
    }
}

// =============================================================================
// ADTR (Tumor Response Analysis Dataset)
// =============================================================================

/// ADTR row: tumor measurement-level data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdtrRow {
    pub studyid: String,   // Study identifier
    pub subjid: u32,       // Subject identifier
    pub arm: String,       // Treatment arm
    pub time_day: f64,     // Days since baseline
    pub tumor_vol: f64,    // Tumor volume (mm³)
    pub baseline_vol: f64, // Baseline tumor volume (for percent change)
    pub pct_change: f64,   // Percent change from baseline
    pub response: u32,     // 1 = responder (≥30% reduction), 0 = non-responder
}

impl AdtrRow {
    pub fn to_csv_line(&self) -> String {
        format!(
            "{},{},{},{},{},{},{},{}",
            self.studyid,
            self.subjid,
            self.arm,
            self.time_day,
            self.tumor_vol,
            self.baseline_vol,
            self.pct_change,
            self.response
        )
    }
}

// =============================================================================
// Conversion Functions
// =============================================================================

/// Convert TrialDataset to ADSL + ADTR
pub fn trial_to_adsl_adtr(dataset: &TrialDataset, study_id: &str) -> (Vec<AdslRow>, Vec<AdtrRow>) {
    let mut adsl_map: HashMap<u32, AdslRow> = HashMap::new();
    let mut adtr_rows = Vec::new();

    // Group rows by subject
    let mut subjects: HashMap<u32, Vec<&TrialRow>> = HashMap::new();
    for row in &dataset.rows {
        subjects
            .entry(row.subject_id)
            .or_insert_with(Vec::new)
            .push(row);
    }

    // Process each subject
    for (subject_id, mut rows) in subjects {
        // Sort by time
        rows.sort_by(|a, b| a.time_days.partial_cmp(&b.time_days).unwrap());

        if rows.is_empty() {
            continue;
        }

        // Get baseline values
        let baseline_row = rows[0];
        let baseline_vol = baseline_row.dv;
        let arm = &baseline_row.arm;
        let dose_mg = baseline_row.dose_mg;
        let weight_kg = baseline_row.wt;
        let last_obs_day = rows[rows.len() - 1].time_days;

        // Create ADSL entry
        let adsl_entry = AdslRow {
            studyid: study_id.to_string(),
            subjid: subject_id,
            arm: arm.clone(),
            dose_mg,
            weight_kg,
            baseline_vol,
            n_obs: rows.len() as u32,
            last_obs_day,
            status: "completed".to_string(), // Assume all completed for now
        };

        adsl_map.insert(subject_id, adsl_entry);

        // Create ADTR entries for each observation
        for (obs_idx, row) in rows.iter().enumerate() {
            let pct_change = if baseline_vol > 0.0 {
                ((row.dv - baseline_vol) / baseline_vol) * 100.0
            } else {
                0.0
            };

            // Response: 1 if ≥30% reduction at any visit
            let response = if pct_change <= -30.0 { 1 } else { 0 };

            let adtr_entry = AdtrRow {
                studyid: study_id.to_string(),
                subjid: subject_id,
                arm: arm.clone(),
                time_day: row.time_days,
                tumor_vol: row.dv,
                baseline_vol,
                pct_change,
                response,
            };

            adtr_rows.push(adtr_entry);
        }
    }

    // Sort ADSL by subject ID
    let mut adsl_rows: Vec<AdslRow> = adsl_map.into_values().collect();
    adsl_rows.sort_by_key(|r| r.subjid);

    (adsl_rows, adtr_rows)
}

// =============================================================================
// CSV Export
// =============================================================================

/// Generate ADSL CSV string
pub fn adsl_to_csv(adsl_rows: &[AdslRow]) -> String {
    let mut csv =
        "STUDYID,SUBJID,ARM,DOSE_MG,WEIGHT_KG,BASELINE_VOL,N_OBS,LAST_OBS_DAY,STATUS\n".to_string();

    for row in adsl_rows {
        csv.push_str(&row.to_csv_line());
        csv.push('\n');
    }

    csv
}

/// Generate ADTR CSV string
pub fn adtr_to_csv(adtr_rows: &[AdtrRow]) -> String {
    let mut csv =
        "STUDYID,SUBJID,ARM,TIME_DAY,TUMOR_VOL,BASELINE_VOL,PCT_CHANGE,RESPONSE\n".to_string();

    for row in adtr_rows {
        csv.push_str(&row.to_csv_line());
        csv.push('\n');
    }

    csv
}

// =============================================================================
// Roundtrip: ADSL/ADTR → JSON
// =============================================================================

/// Serialize ADSL to JSON
pub fn adsl_to_json(adsl_rows: &[AdslRow]) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(adsl_rows)
}

/// Serialize ADTR to JSON
pub fn adtr_to_json(adtr_rows: &[AdtrRow]) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(adtr_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trial_to_adsl_adtr() {
        let dataset = TrialDataset {
            rows: vec![
                TrialRow {
                    subject_id: 1,
                    arm: "ArmA".to_string(),
                    time_days: 0.0,
                    dv: 100.0,
                    dose_mg: 0.0,
                    wt: 70.0,
                },
                TrialRow {
                    subject_id: 1,
                    arm: "ArmA".to_string(),
                    time_days: 84.0,
                    dv: 65.0, // 35% reduction
                    dose_mg: 0.0,
                    wt: 70.0,
                },
                TrialRow {
                    subject_id: 2,
                    arm: "ArmB".to_string(),
                    time_days: 0.0,
                    dv: 110.0,
                    dose_mg: 100.0,
                    wt: 75.0,
                },
                TrialRow {
                    subject_id: 2,
                    arm: "ArmB".to_string(),
                    time_days: 84.0,
                    dv: 70.0, // 36% reduction
                    dose_mg: 100.0,
                    wt: 75.0,
                },
            ],
        };

        let (adsl, adtr) = trial_to_adsl_adtr(&dataset, "TRIAL001");

        // Check ADSL
        assert_eq!(adsl.len(), 2);
        assert_eq!(adsl[0].subjid, 1);
        assert_eq!(adsl[0].baseline_vol, 100.0);
        assert_eq!(adsl[0].n_obs, 2);
        assert_eq!(adsl[0].dose_mg, 0.0);

        assert_eq!(adsl[1].subjid, 2);
        assert_eq!(adsl[1].baseline_vol, 110.0);
        assert_eq!(adsl[1].dose_mg, 100.0);

        // Check ADTR
        assert_eq!(adtr.len(), 4);

        // Subject 1, visit 1 (baseline)
        assert_eq!(adtr[0].subjid, 1);
        assert_eq!(adtr[0].time_day, 0.0);
        assert_eq!(adtr[0].tumor_vol, 100.0);
        assert_eq!(adtr[0].pct_change, 0.0);
        assert_eq!(adtr[0].response, 0);

        // Subject 1, visit 2 (response)
        assert_eq!(adtr[1].subjid, 1);
        assert_eq!(adtr[1].time_day, 84.0);
        assert_eq!(adtr[1].tumor_vol, 65.0);
        assert!((adtr[1].pct_change - (-35.0)).abs() < 0.01);
        assert_eq!(adtr[1].response, 1); // Responder: ≥30% reduction
    }

    #[test]
    fn test_adsl_to_csv() {
        let adsl_rows = vec![AdslRow {
            studyid: "TRIAL001".to_string(),
            subjid: 1,
            arm: "ArmA".to_string(),
            dose_mg: 0.0,
            weight_kg: 70.0,
            baseline_vol: 100.0,
            n_obs: 2,
            last_obs_day: 84.0,
            status: "completed".to_string(),
        }];

        let csv = adsl_to_csv(&adsl_rows);

        assert!(csv.contains("STUDYID,SUBJID,ARM,DOSE_MG"));
        assert!(csv.contains("TRIAL001,1,ArmA,0,70,100,2,84,completed"));
    }

    #[test]
    fn test_adtr_to_csv() {
        let adtr_rows = vec![AdtrRow {
            studyid: "TRIAL001".to_string(),
            subjid: 1,
            arm: "ArmA".to_string(),
            time_day: 84.0,
            tumor_vol: 65.0,
            baseline_vol: 100.0,
            pct_change: -35.0,
            response: 1,
        }];

        let csv = adtr_to_csv(&adtr_rows);

        assert!(csv.contains("STUDYID,SUBJID,ARM,TIME_DAY"));
        assert!(csv.contains("TRIAL001,1,ArmA,84,65,100,-35,1"));
    }

    #[test]
    fn test_adsl_json_serialization() {
        let adsl_rows = vec![AdslRow {
            studyid: "TRIAL001".to_string(),
            subjid: 1,
            arm: "ArmA".to_string(),
            dose_mg: 0.0,
            weight_kg: 70.0,
            baseline_vol: 100.0,
            n_obs: 2,
            last_obs_day: 84.0,
            status: "completed".to_string(),
        }];

        let json = adsl_to_json(&adsl_rows).unwrap();
        assert!(json.contains("TRIAL001"));
        assert!(json.contains("ArmA"));

        let deserialized: Vec<AdslRow> = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.len(), 1);
        assert_eq!(deserialized[0].subjid, 1);
    }

    #[test]
    fn test_response_classification() {
        let dataset = TrialDataset {
            rows: vec![
                TrialRow {
                    subject_id: 1,
                    arm: "ArmA".to_string(),
                    time_days: 0.0,
                    dv: 100.0,
                    dose_mg: 100.0,
                    wt: 70.0,
                },
                TrialRow {
                    subject_id: 1,
                    arm: "ArmA".to_string(),
                    time_days: 84.0,
                    dv: 70.0, // Exactly 30% reduction
                    dose_mg: 100.0,
                    wt: 70.0,
                },
            ],
        };

        let (_adsl, adtr) = trial_to_adsl_adtr(&dataset, "TRIAL001");

        // At day 84: -30% should be classified as responder (response=1)
        assert_eq!(adtr[1].response, 1);
    }
}
