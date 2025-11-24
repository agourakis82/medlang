//! Lowering from AST to IR for Evidence Programs
//!
//! Converts high-level evidence program syntax into executable IR.

use crate::ast::evidence::*;
use crate::ir::evidence::*;

/// Lower an evidence program from AST to IR.
///
/// This performs:
/// - Name resolution (converting Ident to String)
/// - Basic structural validation
/// - Flattening of nested structures
pub fn lower_evidence_program(ast: &EvidenceProgram) -> IREvidenceProgram {
    IREvidenceProgram {
        name: ast.name.to_string(),
        population_model_name: ast.population_model_name.to_string(),
        trials: ast.body.trials.iter().map(lower_trial_decl).collect(),
        hierarchies: ast
            .body
            .hierarchies
            .iter()
            .map(lower_hierarchy_decl)
            .collect(),
        map_priors: ast
            .body
            .map_priors
            .iter()
            .map(lower_map_prior_decl)
            .collect(),
        designs: ast.body.designs.iter().map(lower_design_decl).collect(),
    }
}

fn lower_trial_decl(ast: &TrialDecl) -> IRTrialRef {
    IRTrialRef {
        name: ast.name.to_string(),
        data_path: ast.data_path.clone(),
        protocol_name: ast.protocol_name.to_string(),
        indication: ast.indication.clone(),
        regimen: ast.regimen.clone(),
    }
}

fn lower_hierarchy_decl(ast: &HierarchyDecl) -> IRHierarchy {
    IRHierarchy {
        name: ast.name.to_string(),
        kind: ast.kind.clone(),
        trial_refs: ast.trial_refs.iter().map(|id| id.to_string()).collect(),
        group_by: ast.group_by.clone(),
    }
}

fn lower_map_prior_decl(ast: &MapPriorDecl) -> IRMapPrior {
    IRMapPrior {
        name: ast.name.to_string(),
        hierarchy_name: ast.hierarchy_ref.to_string(),
        indication: ast.indication.clone(),
        new_indication: ast.new_indication,
    }
}

fn lower_design_decl(ast: &DesignDecl) -> IRDesignEvidence {
    IRDesignEvidence {
        name: ast.name.to_string(),
        protocol_name: ast.protocol_name.to_string(),
        prior_ref: ast.prior_ref.as_ref().map(|id| id.to_string()),
        n_per_arm_values: ast.grid.n_per_arm_values.clone(),
        objective_name: ast.objective.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Ident;

    #[test]
    fn test_lower_trial_decl() {
        let ast = TrialDecl {
            name: Ident::new("TestTrial"),
            data_path: "test.csv".to_string(),
            protocol_name: Ident::new("TestProtocol"),
            indication: Some("NSCLC".to_string()),
            regimen: None,
        };

        let ir = lower_trial_decl(&ast);

        assert_eq!(ir.name, "TestTrial");
        assert_eq!(ir.data_path, "test.csv");
        assert_eq!(ir.protocol_name, "TestProtocol");
        assert_eq!(ir.indication, Some("NSCLC".to_string()));
        assert_eq!(ir.regimen, None);
    }

    #[test]
    fn test_lower_hierarchy_decl() {
        let ast = HierarchyDecl {
            name: Ident::new("TestHier"),
            kind: HierarchyKind::OrrMultiIndication,
            trial_refs: vec![Ident::new("T1"), Ident::new("T2")],
            group_by: "indication".to_string(),
        };

        let ir = lower_hierarchy_decl(&ast);

        assert_eq!(ir.name, "TestHier");
        assert_eq!(ir.kind, HierarchyKind::OrrMultiIndication);
        assert_eq!(ir.trial_refs, vec!["T1", "T2"]);
        assert_eq!(ir.group_by, "indication");
    }

    #[test]
    fn test_lower_map_prior_decl() {
        let ast = MapPriorDecl {
            name: Ident::new("TestMAP"),
            hierarchy_ref: Ident::new("TestHier"),
            indication: "NSCLC".to_string(),
            new_indication: false,
        };

        let ir = lower_map_prior_decl(&ast);

        assert_eq!(ir.name, "TestMAP");
        assert_eq!(ir.hierarchy_name, "TestHier");
        assert_eq!(ir.indication, "NSCLC");
        assert!(!ir.new_indication);
    }

    #[test]
    fn test_lower_design_decl() {
        let ast = DesignDecl {
            name: Ident::new("TestDesign"),
            protocol_name: Ident::new("TestProtocol"),
            prior_ref: Some(Ident::new("TestMAP")),
            grid: DesignGridSpec {
                n_per_arm_values: vec![100, 150, 200],
            },
            objective: Some("default_utility".to_string()),
        };

        let ir = lower_design_decl(&ast);

        assert_eq!(ir.name, "TestDesign");
        assert_eq!(ir.protocol_name, "TestProtocol");
        assert_eq!(ir.prior_ref, Some("TestMAP".to_string()));
        assert_eq!(ir.n_per_arm_values, vec![100, 150, 200]);
        assert_eq!(ir.objective_name, Some("default_utility".to_string()));
    }

    #[test]
    fn test_lower_complete_evidence_program() {
        let ast = EvidenceProgram {
            name: Ident::new("TestEvidence"),
            population_model_name: Ident::new("TestModel"),
            body: EvidenceBody {
                trials: vec![TrialDecl {
                    name: Ident::new("Trial1"),
                    data_path: "data1.csv".to_string(),
                    protocol_name: Ident::new("Proto1"),
                    indication: Some("NSCLC".to_string()),
                    regimen: None,
                }],
                hierarchies: vec![HierarchyDecl {
                    name: Ident::new("Hier1"),
                    kind: HierarchyKind::OrrMultiIndication,
                    trial_refs: vec![Ident::new("Trial1")],
                    group_by: "indication".to_string(),
                }],
                map_priors: vec![MapPriorDecl {
                    name: Ident::new("MAP1"),
                    hierarchy_ref: Ident::new("Hier1"),
                    indication: "NSCLC".to_string(),
                    new_indication: false,
                }],
                designs: vec![DesignDecl {
                    name: Ident::new("Design1"),
                    protocol_name: Ident::new("Proto2"),
                    prior_ref: Some(Ident::new("MAP1")),
                    grid: DesignGridSpec {
                        n_per_arm_values: vec![100, 200],
                    },
                    objective: Some("default_utility".to_string()),
                }],
            },
        };

        let ir = lower_evidence_program(&ast);

        assert_eq!(ir.name, "TestEvidence");
        assert_eq!(ir.population_model_name, "TestModel");
        assert_eq!(ir.trials.len(), 1);
        assert_eq!(ir.hierarchies.len(), 1);
        assert_eq!(ir.map_priors.len(), 1);
        assert_eq!(ir.designs.len(), 1);

        assert_eq!(ir.trials[0].name, "Trial1");
        assert_eq!(ir.hierarchies[0].name, "Hier1");
        assert_eq!(ir.map_priors[0].name, "MAP1");
        assert_eq!(ir.designs[0].name, "Design1");
    }
}
