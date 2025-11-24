//! Intermediate Representation for Evidence Programs (L₃)
//!
//! Provides a lower-level, type-checked representation of evidence programs
//! that can be executed by the evidence runner.

use crate::ast::evidence::HierarchyKind;
use serde::{Deserialize, Serialize};

/// IR for a complete evidence program.
///
/// This is the executable representation after AST lowering and validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IREvidenceProgram {
    pub name: String,
    pub population_model_name: String,
    pub trials: Vec<IRTrialRef>,
    pub hierarchies: Vec<IRHierarchy>,
    pub map_priors: Vec<IRMapPrior>,
    pub designs: Vec<IRDesignEvidence>,
}

/// Reference to a trial data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRTrialRef {
    pub name: String,
    pub data_path: String,
    pub protocol_name: String,
    pub indication: Option<String>,
    pub regimen: Option<String>,
}

/// Hierarchical model specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRHierarchy {
    pub name: String,
    pub kind: HierarchyKind,
    pub trial_refs: Vec<String>,
    pub group_by: String,
}

/// MAP prior derivation specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRMapPrior {
    pub name: String,
    pub hierarchy_name: String,
    pub indication: String,
    pub new_indication: bool,
}

/// Design evaluation/optimization specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IRDesignEvidence {
    pub name: String,
    pub protocol_name: String,
    pub prior_ref: Option<String>,
    pub n_per_arm_values: Vec<u32>,
    pub objective_name: Option<String>,
}

impl IREvidenceProgram {
    /// Find a trial by name.
    pub fn find_trial(&self, name: &str) -> Option<&IRTrialRef> {
        self.trials.iter().find(|t| t.name == name)
    }

    /// Find a hierarchy by name.
    pub fn find_hierarchy(&self, name: &str) -> Option<&IRHierarchy> {
        self.hierarchies.iter().find(|h| h.name == name)
    }

    /// Find a MAP prior by name.
    pub fn find_map_prior(&self, name: &str) -> Option<&IRMapPrior> {
        self.map_priors.iter().find(|mp| mp.name == name)
    }

    /// Find a design by name.
    pub fn find_design(&self, name: &str) -> Option<&IRDesignEvidence> {
        self.designs.iter().find(|d| d.name == name)
    }

    /// Get all trial names.
    pub fn trial_names(&self) -> Vec<&str> {
        self.trials.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get all hierarchy names.
    pub fn hierarchy_names(&self) -> Vec<&str> {
        self.hierarchies.iter().map(|h| h.name.as_str()).collect()
    }

    /// Get all MAP prior names.
    pub fn map_prior_names(&self) -> Vec<&str> {
        self.map_priors.iter().map(|mp| mp.name.as_str()).collect()
    }

    /// Get all design names.
    pub fn design_names(&self) -> Vec<&str> {
        self.designs.iter().map(|d| d.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_evidence_program_queries() {
        let prog = IREvidenceProgram {
            name: "TestEvidence".to_string(),
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
                name: "ORR_hier".to_string(),
                kind: HierarchyKind::OrrMultiIndication,
                trial_refs: vec!["Trial1".to_string(), "Trial2".to_string()],
                group_by: "indication".to_string(),
            }],
            map_priors: vec![IRMapPrior {
                name: "NSCLC_MAP".to_string(),
                hierarchy_name: "ORR_hier".to_string(),
                indication: "NSCLC".to_string(),
                new_indication: false,
            }],
            designs: vec![IRDesignEvidence {
                name: "Phase3".to_string(),
                protocol_name: "Phase3Proto".to_string(),
                prior_ref: Some("NSCLC_MAP".to_string()),
                n_per_arm_values: vec![100, 150, 200],
                objective_name: Some("default_utility".to_string()),
            }],
        };

        assert!(prog.find_trial("Trial1").is_some());
        assert!(prog.find_trial("NonExistent").is_none());

        assert!(prog.find_hierarchy("ORR_hier").is_some());
        assert!(prog.find_map_prior("NSCLC_MAP").is_some());
        assert!(prog.find_design("Phase3").is_some());

        assert_eq!(prog.trial_names(), vec!["Trial1", "Trial2"]);
        assert_eq!(prog.hierarchy_names(), vec!["ORR_hier"]);
        assert_eq!(prog.map_prior_names(), vec!["NSCLC_MAP"]);
        assert_eq!(prog.design_names(), vec!["Phase3"]);
    }

    #[test]
    fn test_ir_trial_ref() {
        let trial = IRTrialRef {
            name: "TestTrial".to_string(),
            data_path: "test.csv".to_string(),
            protocol_name: "TestProtocol".to_string(),
            indication: Some("TestIndication".to_string()),
            regimen: Some("standard".to_string()),
        };

        assert_eq!(trial.name, "TestTrial");
        assert_eq!(trial.indication, Some("TestIndication".to_string()));
        assert_eq!(trial.regimen, Some("standard".to_string()));
    }

    #[test]
    fn test_ir_hierarchy() {
        let hier = IRHierarchy {
            name: "TestHier".to_string(),
            kind: HierarchyKind::OrrMultiIndication,
            trial_refs: vec!["T1".to_string(), "T2".to_string()],
            group_by: "indication".to_string(),
        };

        assert_eq!(hier.trial_refs.len(), 2);
        assert_eq!(hier.group_by, "indication");
    }
}
