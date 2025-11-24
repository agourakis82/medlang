//! AST for Evidence Programs (L₃)
//!
//! Evidence programs are first-class MedLang constructs that orchestrate:
//! - Multi-trial data integration
//! - Multi-indication hierarchical modeling
//! - MAP prior derivation
//! - Design evaluation and optimization
//!
//! This provides a declarative, type-checked way to express complete evidence
//! generation workflows as source code, rather than manual CLI orchestration.

use crate::ast::Ident;
use serde::{Deserialize, Serialize};

/// Top-level evidence program declaration.
///
/// Example:
/// ```medlang
/// evidence_program OncologyEvidence {
///   population_model Oncology_PBPK_QSP_QM;
///   trials { ... }
///   hierarchies { ... }
///   map_priors { ... }
///   designs { ... }
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceProgram {
    pub name: Ident,
    pub population_model_name: Ident,
    pub body: EvidenceBody,
}

/// Body of an evidence program containing all declarations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBody {
    pub trials: Vec<TrialDecl>,
    pub hierarchies: Vec<HierarchyDecl>,
    pub map_priors: Vec<MapPriorDecl>,
    pub designs: Vec<DesignDecl>,
}

/// Trial data source declaration.
///
/// Example:
/// ```medlang
/// Phase2_NSCLC_A = trial("data/phase2_nsclc_A.csv",
///   protocol = Oncology_Phase2_NSCLC_A,
///   indication = "NSCLC",
///   regimen = "standard");
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrialDecl {
    pub name: Ident,
    pub data_path: String,
    pub protocol_name: Ident,
    pub indication: Option<String>,
    pub regimen: Option<String>,
}

/// Kind of hierarchical model to fit.
///
/// Week 24: Only ORR multi-indication hierarchy supported.
/// Future: PFS hierarchies, continuous endpoints, QSP parameter hierarchies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HierarchyKind {
    OrrMultiIndication,
    // Future: PfsMultiIndication, ToxicityHierarchy, etc.
}

/// Hierarchical model declaration.
///
/// Example:
/// ```medlang
/// ORR_multi_indication = orr_hierarchical(
///   trials = { Phase2_NSCLC_A, Phase2_HNSCC_A },
///   group_by = "indication"
/// );
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HierarchyDecl {
    pub name: Ident,
    pub kind: HierarchyKind,
    pub trial_refs: Vec<Ident>,
    pub group_by: String,
}

/// MAP prior derivation from a fitted hierarchy.
///
/// Example:
/// ```medlang
/// NSCLC_MAP = map_prior(ORR_multi_indication,
///   indication = "NSCLC");
///
/// NewTumor_MAP = map_prior(ORR_multi_indication,
///   indication = "NewTumor",
///   new_indication = true);
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapPriorDecl {
    pub name: Ident,
    pub hierarchy_ref: Ident,
    pub indication: String,
    pub new_indication: bool,
}

/// Design evaluation/optimization declaration.
///
/// Example:
/// ```medlang
/// Phase3_NSCLC = design {
///   protocol = Oncology_Phase3_NSCLC;
///   prior = NSCLC_MAP;
///   grid = { n_per_arm in [120, 160, 200] };
///   objective = default_utility();
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignDecl {
    pub name: Ident,
    pub protocol_name: Ident,
    pub prior_ref: Option<Ident>,
    pub grid: DesignGridSpec,
    pub objective: Option<String>,
}

/// Design grid specification.
///
/// Defines the parameter space to search over during design optimization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignGridSpec {
    pub n_per_arm_values: Vec<u32>,
    // Future: orr_margin_values, dlt_threshold_values, etc.
}

impl EvidenceProgram {
    /// Get all trial names declared in this program.
    pub fn trial_names(&self) -> Vec<&str> {
        self.body.trials.iter().map(|t| t.name.as_str()).collect()
    }

    /// Get all hierarchy names declared in this program.
    pub fn hierarchy_names(&self) -> Vec<&str> {
        self.body
            .hierarchies
            .iter()
            .map(|h| h.name.as_str())
            .collect()
    }

    /// Get all MAP prior names declared in this program.
    pub fn map_prior_names(&self) -> Vec<&str> {
        self.body
            .map_priors
            .iter()
            .map(|mp| mp.name.as_str())
            .collect()
    }

    /// Get all design names declared in this program.
    pub fn design_names(&self) -> Vec<&str> {
        self.body.designs.iter().map(|d| d.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchy_kind() {
        let kind = HierarchyKind::OrrMultiIndication;
        assert_eq!(kind, HierarchyKind::OrrMultiIndication);
    }

    #[test]
    fn test_evidence_program_queries() {
        let prog = EvidenceProgram {
            name: Ident::new("TestEvidence"),
            population_model_name: Ident::new("TestModel"),
            body: EvidenceBody {
                trials: vec![
                    TrialDecl {
                        name: Ident::new("Trial1"),
                        data_path: "data1.csv".to_string(),
                        protocol_name: Ident::new("Proto1"),
                        indication: Some("NSCLC".to_string()),
                        regimen: None,
                    },
                    TrialDecl {
                        name: Ident::new("Trial2"),
                        data_path: "data2.csv".to_string(),
                        protocol_name: Ident::new("Proto2"),
                        indication: Some("HNSCC".to_string()),
                        regimen: None,
                    },
                ],
                hierarchies: vec![HierarchyDecl {
                    name: Ident::new("ORR_hier"),
                    kind: HierarchyKind::OrrMultiIndication,
                    trial_refs: vec![Ident::new("Trial1"), Ident::new("Trial2")],
                    group_by: "indication".to_string(),
                }],
                map_priors: vec![MapPriorDecl {
                    name: Ident::new("NSCLC_MAP"),
                    hierarchy_ref: Ident::new("ORR_hier"),
                    indication: "NSCLC".to_string(),
                    new_indication: false,
                }],
                designs: vec![DesignDecl {
                    name: Ident::new("Phase3"),
                    protocol_name: Ident::new("Phase3Proto"),
                    prior_ref: Some(Ident::new("NSCLC_MAP")),
                    grid: DesignGridSpec {
                        n_per_arm_values: vec![100, 150, 200],
                    },
                    objective: Some("default_utility".to_string()),
                }],
            },
        };

        assert_eq!(prog.trial_names(), vec!["Trial1", "Trial2"]);
        assert_eq!(prog.hierarchy_names(), vec!["ORR_hier"]);
        assert_eq!(prog.map_prior_names(), vec!["NSCLC_MAP"]);
        assert_eq!(prog.design_names(), vec!["Phase3"]);
    }
}
