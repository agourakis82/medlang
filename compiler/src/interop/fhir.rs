//! FHIR R4 Export for MedLang Protocols
//!
//! Converts MedLang Protocol definitions to FHIR R4 resources:
//! - ResearchStudy: Overall trial metadata
//! - PlanDefinition: Study design with arms and visits
//! - Measure: Endpoint definitions
//! - Bundle: Patient + Observation data from TrialDataset

use crate::ast::{ArmDef, EndpointDef, EndpointKind, EndpointSpec, ProtocolDef, VisitDef};
use crate::data::trial::TrialDataset;
use serde::{Deserialize, Serialize};

// =============================================================================
// FHIR Resource Types
// =============================================================================

/// FHIR ResearchStudy resource (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirResearchStudy {
    #[serde(rename = "resourceType")]
    pub resource_type: String, // Always "ResearchStudy"

    pub id: String,
    pub status: String, // "active", "completed", "draft"
    pub title: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm: Option<Vec<FhirResearchStudyArm>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective: Option<Vec<FhirResearchStudyObjective>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirResearchStudyArm {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arm_type: Option<FhirCodeableConcept>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirResearchStudyObjective {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,

    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objective_type: Option<FhirCodeableConcept>,
}

/// FHIR PlanDefinition resource (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirPlanDefinition {
    #[serde(rename = "resourceType")]
    pub resource_type: String, // Always "PlanDefinition"

    pub id: String,
    pub status: String, // "active", "draft"
    pub title: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(rename = "type")]
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_type: Option<FhirCodeableConcept>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<Vec<FhirPlanDefinitionAction>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirPlanDefinitionAction {
    pub title: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing: Option<FhirTiming>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub action: Option<Vec<FhirPlanDefinitionAction>>, // Nested actions
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirTiming {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<Vec<String>>, // ISO8601 datetime strings

    #[serde(skip_serializing_if = "Option::is_none")]
    pub repeat: Option<FhirTimingRepeat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirTimingRepeat {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency: Option<u32>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub period: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub period_unit: Option<String>, // "d", "wk", "mo"
}

/// FHIR Measure resource (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirMeasure {
    #[serde(rename = "resourceType")]
    pub resource_type: String, // Always "Measure"

    pub id: String,
    pub status: String, // "active", "draft"
    pub title: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub group: Option<Vec<FhirMeasureGroup>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirMeasureGroup {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<FhirCodeableConcept>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub population: Option<Vec<FhirMeasurePopulation>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirMeasurePopulation {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<FhirCodeableConcept>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub criteria: Option<FhirExpression>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirExpression {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>, // "text/cql"

    #[serde(skip_serializing_if = "Option::is_none")]
    pub expression: Option<String>,
}

/// FHIR Bundle resource
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirBundle {
    #[serde(rename = "resourceType")]
    pub resource_type: String, // Always "Bundle"

    #[serde(rename = "type")]
    pub bundle_type: String, // "collection", "transaction"

    pub entry: Vec<FhirBundleEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirBundleEntry {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_url: Option<String>,

    pub resource: FhirResource,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FhirResource {
    Patient(FhirPatient),
    Observation(FhirObservation),
}

/// FHIR Patient resource (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirPatient {
    #[serde(rename = "resourceType")]
    pub resource_type: String, // Always "Patient"

    pub id: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub identifier: Option<Vec<FhirIdentifier>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirIdentifier {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

/// FHIR Observation resource (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirObservation {
    #[serde(rename = "resourceType")]
    pub resource_type: String, // Always "Observation"

    pub id: String,
    pub status: String, // "final", "preliminary"

    pub code: FhirCodeableConcept,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<FhirReference>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub effective_date_time: Option<String>, // ISO8601

    #[serde(skip_serializing_if = "Option::is_none")]
    pub value_quantity: Option<FhirQuantity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirCodeableConcept {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coding: Option<Vec<FhirCoding>>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirCoding {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirReference {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>, // e.g., "Patient/12345"

    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FhirQuantity {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>, // e.g., "http://unitsofmeasure.org"

    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>, // UCUM code
}

// =============================================================================
// Conversion Functions: Protocol → FHIR
// =============================================================================

/// Convert ProtocolDef to FHIR ResearchStudy
pub fn protocol_to_fhir_research_study(protocol: &ProtocolDef) -> FhirResearchStudy {
    FhirResearchStudy {
        resource_type: "ResearchStudy".to_string(),
        id: protocol.name.clone(),
        status: "active".to_string(),
        title: protocol.name.clone(),
        description: Some(format!(
            "MedLang protocol with {} arms and {} endpoints",
            protocol.arms.len(),
            protocol.endpoints.len()
        )),
        arm: Some(
            protocol
                .arms
                .iter()
                .map(|arm| arm_to_fhir_research_study_arm(arm))
                .collect(),
        ),
        objective: Some(
            protocol
                .endpoints
                .iter()
                .map(|ep| endpoint_to_fhir_objective(ep))
                .collect(),
        ),
    }
}

fn arm_to_fhir_research_study_arm(arm: &ArmDef) -> FhirResearchStudyArm {
    FhirResearchStudyArm {
        name: arm.name.clone(),
        description: Some(format!("{} ({}mg)", arm.label, arm.dose_mg)),
        arm_type: Some(FhirCodeableConcept {
            coding: None,
            text: Some("experimental".to_string()),
        }),
    }
}

fn endpoint_to_fhir_objective(endpoint: &EndpointDef) -> FhirResearchStudyObjective {
    FhirResearchStudyObjective {
        name: Some(endpoint.name.clone()),
        objective_type: Some(FhirCodeableConcept {
            coding: None,
            text: Some(format!("{}", endpoint.kind)), // Using Display trait
        }),
    }
}

/// Convert ProtocolDef to FHIR PlanDefinition
pub fn protocol_to_fhir_plan_definition(protocol: &ProtocolDef) -> FhirPlanDefinition {
    FhirPlanDefinition {
        resource_type: "PlanDefinition".to_string(),
        id: format!("{}_plan", protocol.name),
        status: "active".to_string(),
        title: format!("{} Study Plan", protocol.name),
        description: Some("Trial design with visit schedule".to_string()),
        plan_type: Some(FhirCodeableConcept {
            coding: None,
            text: Some("clinical-protocol".to_string()),
        }),
        action: Some(
            protocol
                .visits
                .iter()
                .map(|visit| visit_to_fhir_action(visit))
                .collect(),
        ),
    }
}

fn visit_to_fhir_action(visit: &VisitDef) -> FhirPlanDefinitionAction {
    FhirPlanDefinitionAction {
        title: visit.name.clone(),
        description: Some(format!("Visit at day {}", visit.time_days)),
        timing: Some(FhirTiming {
            event: Some(vec![format!("Day {}", visit.time_days)]),
            repeat: None,
        }),
        action: None,
    }
}

/// Convert endpoints to FHIR Measure resources
pub fn protocol_to_fhir_measures(protocol: &ProtocolDef) -> Vec<FhirMeasure> {
    protocol
        .endpoints
        .iter()
        .map(|ep| endpoint_to_fhir_measure(ep, &protocol.name))
        .collect()
}

fn endpoint_to_fhir_measure(endpoint: &EndpointDef, protocol_name: &str) -> FhirMeasure {
    let window_days = match &endpoint.spec {
        EndpointSpec::ResponseRate { window_end_days, .. } => *window_end_days,
        EndpointSpec::TimeToProgression { window_end_days, .. } => *window_end_days,
    };

    FhirMeasure {
        resource_type: "Measure".to_string(),
        id: format!("{}_{}", protocol_name, endpoint.name),
        status: "active".to_string(),
        title: endpoint.name.clone(),
        description: Some(format!("Endpoint: {}", endpoint.kind)),
        group: Some(vec![FhirMeasureGroup {
            code: Some(FhirCodeableConcept {
                coding: None,
                text: Some(format!("{}", endpoint.kind)), // Using Display trait
            }),
            description: Some(format!("Window: {} days", window_days)),
            population: None,
        }]),
    }
}

// =============================================================================
// Conversion Functions: TrialDataset → FHIR Bundle
// =============================================================================

/// Convert TrialDataset to FHIR Bundle with Patient and Observation resources
pub fn trial_to_fhir_bundle(dataset: &TrialDataset, study_id: &str) -> FhirBundle {
    let mut entries = Vec::new();

    // Create Patient resources for each unique subject
    let subject_ids: std::collections::HashSet<_> =
        dataset.rows.iter().map(|r| r.subject_id).collect();

    for subject_id in subject_ids {
        let patient = FhirPatient {
            resource_type: "Patient".to_string(),
            id: format!("patient_{}", subject_id),
            identifier: Some(vec![FhirIdentifier {
                system: Some(format!("urn:medlang:study:{}", study_id)),
                value: Some(subject_id.to_string()),
            }]),
        };

        entries.push(FhirBundleEntry {
            full_url: Some(format!("Patient/patient_{}", subject_id)),
            resource: FhirResource::Patient(patient),
        });
    }

    // Create Observation resources for each measurement
    for (obs_idx, row) in dataset.rows.iter().enumerate() {
        let observation = FhirObservation {
            resource_type: "Observation".to_string(),
            id: format!("obs_{}", obs_idx),
            status: "final".to_string(),
            code: FhirCodeableConcept {
                coding: Some(vec![FhirCoding {
                    system: Some("http://loinc.org".to_string()),
                    code: Some("33728-7".to_string()), // Tumor size code (placeholder)
                    display: Some("Tumor Volume".to_string()),
                }]),
                text: Some("Tumor Volume".to_string()),
            },
            subject: Some(FhirReference {
                reference: Some(format!("Patient/patient_{}", row.subject_id)),
                display: None,
            }),
            effective_date_time: Some(format!("Day {}", row.time_days)),
            value_quantity: Some(FhirQuantity {
                value: Some(row.dv),
                unit: Some("mm3".to_string()),
                system: Some("http://unitsofmeasure.org".to_string()),
                code: Some("mm3".to_string()),
            }),
        };

        entries.push(FhirBundleEntry {
            full_url: Some(format!("Observation/obs_{}", obs_idx)),
            resource: FhirResource::Observation(observation),
        });
    }

    FhirBundle {
        resource_type: "Bundle".to_string(),
        bundle_type: "collection".to_string(),
        entry: entries,
    }
}

// =============================================================================
// Helper: EndpointKind Display
// =============================================================================

impl std::fmt::Display for crate::ast::EndpointKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            crate::ast::EndpointKind::Binary => write!(f, "ORR (Objective Response Rate)"),
            crate::ast::EndpointKind::TimeToEvent => write!(f, "PFS (Progression-Free Survival)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{EndpointKind, InclusionDef};

    #[test]
    fn test_protocol_to_research_study() {
        let protocol = ProtocolDef {
            name: "TRIAL001".to_string(),
            population_model_name: "PopModel".to_string(),
            arms: vec![
                ArmDef {
                    name: "ArmA".to_string(),
                    label: "Control".to_string(),
                    dose_mg: 0.0,
                    span: None,
                },
                ArmDef {
                    name: "ArmB".to_string(),
                    label: "Treatment".to_string(),
                    dose_mg: 100.0,
                    span: None,
                },
            ],
            visits: vec![],
            inclusion: None,
            endpoints: vec![EndpointDef {
                name: "ORR".to_string(),
                kind: EndpointKind::Binary,
                spec: EndpointSpec::ResponseRate {
                    observable: "tumor_size".to_string(),
                    shrink_fraction: 0.3,
                    window_start_days: 0.0,
                    window_end_days: 84.0,
                },
                span: None,
            }],
            decisions: vec![],
            span: None,
        };

        let research_study = protocol_to_fhir_research_study(&protocol);

        assert_eq!(research_study.resource_type, "ResearchStudy");
        assert_eq!(research_study.id, "TRIAL001");
        assert_eq!(research_study.status, "active");
        assert!(research_study.arm.is_some());
        assert_eq!(research_study.arm.as_ref().unwrap().len(), 2);
        assert!(research_study.objective.is_some());
        assert_eq!(research_study.objective.as_ref().unwrap().len(), 1);
    }

    #[test]
    fn test_protocol_to_plan_definition() {
        let protocol = ProtocolDef {
            name: "TRIAL001".to_string(),
            population_model_name: "PopModel".to_string(),
            arms: vec![],
            visits: vec![
                VisitDef {
                    name: "Baseline".to_string(),
                    time_days: 0.0,
                    span: None,
                },
                VisitDef {
                    name: "Week12".to_string(),
                    time_days: 84.0,
                    span: None,
                },
            ],
            inclusion: None,
            endpoints: vec![],
            decisions: vec![],
            span: None,
        };

        let plan_def = protocol_to_fhir_plan_definition(&protocol);

        assert_eq!(plan_def.resource_type, "PlanDefinition");
        assert_eq!(plan_def.status, "active");
        assert!(plan_def.action.is_some());
        assert_eq!(plan_def.action.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_endpoint_to_measure() {
        let protocol = ProtocolDef {
            name: "TRIAL001".to_string(),
            population_model_name: "PopModel".to_string(),
            arms: vec![],
            visits: vec![],
            inclusion: None,
            endpoints: vec![EndpointDef {
                name: "PFS".to_string(),
                kind: EndpointKind::TimeToEvent,
                spec: EndpointSpec::TimeToProgression {
                    observable: "tumor_size".to_string(),
                    increase_fraction: 0.2,
                    window_start_days: 0.0,
                    window_end_days: 180.0,
                    ref_baseline: false,
                },
                span: None,
            }],
            decisions: vec![],
            span: None,
        };

        let measures = protocol_to_fhir_measures(&protocol);

        assert_eq!(measures.len(), 1);
        assert_eq!(measures[0].resource_type, "Measure");
        assert_eq!(measures[0].id, "TRIAL001_PFS");
        assert_eq!(measures[0].title, "PFS");
    }

    #[test]
    fn test_trial_to_bundle() {
        use crate::data::trial::TrialRow;

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
                    dv: 80.0,
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
            ],
        };

        let bundle = trial_to_fhir_bundle(&dataset, "TRIAL001");

        assert_eq!(bundle.resource_type, "Bundle");
        assert_eq!(bundle.bundle_type, "collection");

        // Should have 2 patients + 3 observations = 5 entries
        assert_eq!(bundle.entry.len(), 5);

        // Check first entry is a Patient
        match &bundle.entry[0].resource {
            FhirResource::Patient(p) => {
                assert_eq!(p.resource_type, "Patient");
            }
            _ => panic!("Expected Patient resource"),
        }
    }

    #[test]
    fn test_fhir_serialization() {
        let patient = FhirPatient {
            resource_type: "Patient".to_string(),
            id: "patient_1".to_string(),
            identifier: Some(vec![FhirIdentifier {
                system: Some("urn:medlang:study:TRIAL001".to_string()),
                value: Some("1".to_string()),
            }]),
        };

        let json = serde_json::to_string_pretty(&patient).unwrap();
        assert!(json.contains("Patient"));
        assert!(json.contains("patient_1"));
    }
}