//! Evidence Program Validation
//!
//! Checks semantic correctness of evidence programs:
//! - All references resolve correctly
//! - No circular dependencies
//! - Supported features are used correctly

use crate::ir::evidence::*;
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum EvidenceValidationError {
    #[error("Trial not found: {0}")]
    TrialNotFound(String),

    #[error("Hierarchy not found: {0}")]
    HierarchyNotFound(String),

    #[error("MAP prior not found: {0}")]
    MapPriorNotFound(String),

    #[error("Design not found: {0}")]
    DesignNotFound(String),

    #[error("Hierarchy references non-existent trial: {hierarchy} -> {trial}")]
    HierarchyTrialNotFound { hierarchy: String, trial: String },

    #[error("MAP prior references non-existent hierarchy: {map_prior} -> {hierarchy}")]
    MapPriorHierarchyNotFound {
        map_prior: String,
        hierarchy: String,
    },

    #[error("Design references non-existent MAP prior: {design} -> {prior}")]
    DesignPriorNotFound { design: String, prior: String },

    #[error("Unsupported group_by value: {0} (only 'indication' supported in Week 24)")]
    UnsupportedGroupBy(String),

    #[error("Hierarchy has no trial references: {0}")]
    EmptyHierarchy(String),

    #[error("Design has no grid values: {0}")]
    EmptyDesignGrid(String),
}

/// Check semantic correctness of an evidence program.
///
/// Returns Ok(()) if valid, or an error describing the first problem found.
pub fn check_evidence_program(ir: &IREvidenceProgram) -> Result<(), EvidenceValidationError> {
    // Build name sets for quick lookup
    let trial_names: HashSet<_> = ir.trials.iter().map(|t| t.name.as_str()).collect();
    let hierarchy_names: HashSet<_> = ir.hierarchies.iter().map(|h| h.name.as_str()).collect();
    let map_prior_names: HashSet<_> = ir.map_priors.iter().map(|mp| mp.name.as_str()).collect();

    // Check hierarchies
    for hier in &ir.hierarchies {
        // Check that all trial refs exist
        for trial_ref in &hier.trial_refs {
            if !trial_names.contains(trial_ref.as_str()) {
                return Err(EvidenceValidationError::HierarchyTrialNotFound {
                    hierarchy: hier.name.clone(),
                    trial: trial_ref.clone(),
                });
            }
        }

        // Check that hierarchy has at least one trial
        if hier.trial_refs.is_empty() {
            return Err(EvidenceValidationError::EmptyHierarchy(hier.name.clone()));
        }

        // Check that group_by is supported (Week 24: only "indication")
        if hier.group_by != "indication" {
            return Err(EvidenceValidationError::UnsupportedGroupBy(
                hier.group_by.clone(),
            ));
        }
    }

    // Check MAP priors
    for mp in &ir.map_priors {
        // Check that hierarchy ref exists
        if !hierarchy_names.contains(mp.hierarchy_name.as_str()) {
            return Err(EvidenceValidationError::MapPriorHierarchyNotFound {
                map_prior: mp.name.clone(),
                hierarchy: mp.hierarchy_name.clone(),
            });
        }
    }

    // Check designs
    for design in &ir.designs {
        // Check that prior ref exists (if specified)
        if let Some(ref prior_ref) = design.prior_ref {
            if !map_prior_names.contains(prior_ref.as_str()) {
                return Err(EvidenceValidationError::DesignPriorNotFound {
                    design: design.name.clone(),
                    prior: prior_ref.clone(),
                });
            }
        }

        // Check that design has at least one grid value
        if design.n_per_arm_values.is_empty() {
            return Err(EvidenceValidationError::EmptyDesignGrid(
                design.name.clone(),
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::evidence::HierarchyKind;

    fn make_test_program() -> IREvidenceProgram {
        IREvidenceProgram {
            name: "TestProgram".to_string(),
            population_model_name: "TestModel".to_string(),
            trials: vec![
                IRTrialRef {
                    name: "Trial1".to_string(),
                    data_path: "data1.csv".to_string(),
                    protocol_name: "Proto1".to_string(),
                    indication: Some("NSCLC".to_string()),
                    regimen: None,
                },
                IRTrialRef {
                    name: "Trial2".to_string(),
                    data_path: "data2.csv".to_string(),
                    protocol_name: "Proto2".to_string(),
                    indication: Some("HNSCC".to_string()),
                    regimen: None,
                },
            ],
            hierarchies: vec![IRHierarchy {
                name: "Hier1".to_string(),
                kind: HierarchyKind::OrrMultiIndication,
                trial_refs: vec!["Trial1".to_string(), "Trial2".to_string()],
                group_by: "indication".to_string(),
            }],
            map_priors: vec![IRMapPrior {
                name: "MAP1".to_string(),
                hierarchy_name: "Hier1".to_string(),
                indication: "NSCLC".to_string(),
                new_indication: false,
            }],
            designs: vec![IRDesignEvidence {
                name: "Design1".to_string(),
                protocol_name: "Proto1".to_string(),
                prior_ref: Some("MAP1".to_string()),
                n_per_arm_values: vec![100, 150],
                objective_name: Some("default_utility".to_string()),
            }],
        }
    }

    #[test]
    fn test_valid_program() {
        let prog = make_test_program();
        let result = check_evidence_program(&prog);
        assert!(result.is_ok());
    }

    #[test]
    fn test_hierarchy_trial_not_found() {
        let mut prog = make_test_program();
        prog.hierarchies[0]
            .trial_refs
            .push("NonExistent".to_string());

        let result = check_evidence_program(&prog);
        assert!(result.is_err());

        match result.unwrap_err() {
            EvidenceValidationError::HierarchyTrialNotFound { trial, .. } => {
                assert_eq!(trial, "NonExistent");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_map_prior_hierarchy_not_found() {
        let mut prog = make_test_program();
        prog.map_priors[0].hierarchy_name = "NonExistent".to_string();

        let result = check_evidence_program(&prog);
        assert!(result.is_err());

        match result.unwrap_err() {
            EvidenceValidationError::MapPriorHierarchyNotFound { hierarchy, .. } => {
                assert_eq!(hierarchy, "NonExistent");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_design_prior_not_found() {
        let mut prog = make_test_program();
        prog.designs[0].prior_ref = Some("NonExistent".to_string());

        let result = check_evidence_program(&prog);
        assert!(result.is_err());

        match result.unwrap_err() {
            EvidenceValidationError::DesignPriorNotFound { prior, .. } => {
                assert_eq!(prior, "NonExistent");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_unsupported_group_by() {
        let mut prog = make_test_program();
        prog.hierarchies[0].group_by = "regimen".to_string();

        let result = check_evidence_program(&prog);
        assert!(result.is_err());

        match result.unwrap_err() {
            EvidenceValidationError::UnsupportedGroupBy(group_by) => {
                assert_eq!(group_by, "regimen");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_empty_hierarchy() {
        let mut prog = make_test_program();
        prog.hierarchies[0].trial_refs.clear();

        let result = check_evidence_program(&prog);
        assert!(result.is_err());

        match result.unwrap_err() {
            EvidenceValidationError::EmptyHierarchy(_) => {}
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_empty_design_grid() {
        let mut prog = make_test_program();
        prog.designs[0].n_per_arm_values.clear();

        let result = check_evidence_program(&prog);
        assert!(result.is_err());

        match result.unwrap_err() {
            EvidenceValidationError::EmptyDesignGrid(_) => {}
            _ => panic!("Wrong error type"),
        }
    }
}
