//! Integration tests for Standards & Regulatory Interoperability
//!
//! Tests FHIR, CQL, and CDISC export functionality end-to-end

use medlangc::ast::{ArmDef, EndpointDef, EndpointKind, ProtocolDef, VisitDef};
use medlangc::data::trial::{TrialDataset, TrialRow};
use medlangc::interop::{
    adsl_to_csv, adtr_to_csv, endpoint_to_cql, protocol_endpoints_to_cql,
    protocol_to_fhir_measures, protocol_to_fhir_plan_definition, protocol_to_fhir_research_study,
    trial_to_adsl_adtr, trial_to_fhir_bundle,
};

// =============================================================================
// Test Data Fixtures
// =============================================================================

fn create_test_protocol() -> ProtocolDef {
    ProtocolDef {
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
        visits: vec![
            VisitDef {
                name: "Baseline".to_string(),
                day: 0,
                span: None,
            },
            VisitDef {
                name: "Week12".to_string(),
                day: 84,
                span: None,
            },
            VisitDef {
                name: "Week24".to_string(),
                day: 168,
                span: None,
            },
        ],
        inclusion: None,
        endpoints: vec![
            EndpointDef {
                name: "ORR".to_string(),
                kind: EndpointKind::BinaryResponse,
                window_days: 84,
                threshold: Some(0.3),
                span: None,
            },
            EndpointDef {
                name: "PFS".to_string(),
                kind: EndpointKind::TimeToEvent,
                window_days: 180,
                threshold: None,
                span: None,
            },
        ],
        decisions: vec![],
        span: None,
    }
}

fn create_test_trial_dataset() -> TrialDataset {
    TrialDataset {
        rows: vec![
            // Subject 1, Arm A (Control)
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
                dv: 95.0, // 5% reduction (non-responder)
                dose_mg: 0.0,
                wt: 70.0,
            },
            // Subject 2, Arm B (Treatment)
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
                dv: 70.0, // 36% reduction (responder)
                dose_mg: 100.0,
                wt: 75.0,
            },
            // Subject 3, Arm B (Treatment)
            TrialRow {
                subject_id: 3,
                arm: "ArmB".to_string(),
                time_days: 0.0,
                dv: 105.0,
                dose_mg: 100.0,
                wt: 72.0,
            },
            TrialRow {
                subject_id: 3,
                arm: "ArmB".to_string(),
                time_days: 84.0,
                dv: 73.0, // 30% reduction (responder - boundary)
                dose_mg: 100.0,
                wt: 72.0,
            },
        ],
    }
}

// =============================================================================
// FHIR Export Tests
// =============================================================================

#[test]
fn test_fhir_research_study_generation() {
    let protocol = create_test_protocol();
    let research_study = protocol_to_fhir_research_study(&protocol);

    // Verify resource type
    assert_eq!(research_study.resource_type, "ResearchStudy");
    assert_eq!(research_study.id, "TRIAL001");
    assert_eq!(research_study.status, "active");

    // Verify arms
    assert!(research_study.arm.is_some());
    let arms = research_study.arm.unwrap();
    assert_eq!(arms.len(), 2);
    assert_eq!(arms[0].name, "ArmA");
    assert_eq!(arms[1].name, "ArmB");

    // Verify objectives
    assert!(research_study.objective.is_some());
    let objectives = research_study.objective.unwrap();
    assert_eq!(objectives.len(), 2); // ORR and PFS
}

#[test]
fn test_fhir_plan_definition_generation() {
    let protocol = create_test_protocol();
    let plan_def = protocol_to_fhir_plan_definition(&protocol);

    assert_eq!(plan_def.resource_type, "PlanDefinition");
    assert_eq!(plan_def.status, "active");

    // Verify visit actions
    assert!(plan_def.action.is_some());
    let actions = plan_def.action.unwrap();
    assert_eq!(actions.len(), 3); // Baseline, Week12, Week24
    assert_eq!(actions[0].title, "Baseline");
    assert_eq!(actions[1].title, "Week12");
}

#[test]
fn test_fhir_measures_generation() {
    let protocol = create_test_protocol();
    let measures = protocol_to_fhir_measures(&protocol);

    assert_eq!(measures.len(), 2); // ORR and PFS

    // Verify ORR measure
    assert_eq!(measures[0].title, "ORR");
    assert_eq!(measures[0].resource_type, "Measure");
    assert_eq!(measures[0].id, "TRIAL001_ORR");

    // Verify PFS measure
    assert_eq!(measures[1].title, "PFS");
    assert_eq!(measures[1].id, "TRIAL001_PFS");
}

#[test]
fn test_fhir_bundle_generation() {
    let dataset = create_test_trial_dataset();
    let bundle = trial_to_fhir_bundle(&dataset, "TRIAL001");

    assert_eq!(bundle.resource_type, "Bundle");
    assert_eq!(bundle.bundle_type, "collection");

    // Should have 3 patients + 6 observations = 9 entries
    assert_eq!(bundle.entry.len(), 9);

    // Verify patients are first
    let mut patient_count = 0;
    for entry in &bundle.entry {
        match &entry.resource {
            medlangc::interop::FhirResource::Patient(_) => patient_count += 1,
            _ => {}
        }
    }
    assert_eq!(patient_count, 3);

    // Verify observations
    let obs_count = bundle.entry.len() - patient_count;
    assert_eq!(obs_count, 6);
}

#[test]
fn test_fhir_bundle_serialization() {
    let dataset = create_test_trial_dataset();
    let bundle = trial_to_fhir_bundle(&dataset, "TRIAL001");

    // Verify it serializes to JSON without error
    let json = serde_json::to_string_pretty(&bundle).expect("Failed to serialize bundle");
    assert!(json.contains("Bundle"));
    assert!(json.contains("Patient"));
    assert!(json.contains("Observation"));

    // Verify it can be deserialized
    let _deserialized: medlangc::interop::FhirBundle =
        serde_json::from_str(&json).expect("Failed to deserialize bundle");
}

// =============================================================================
// CQL Export Tests
// =============================================================================

#[test]
fn test_cql_orr_generation() {
    let protocol = create_test_protocol();
    let cql_libs = protocol_endpoints_to_cql(&protocol.endpoints, "TRIAL001");

    assert_eq!(cql_libs.len(), 2);

    // Find ORR library
    let (lib_name, cql_code) = &cql_libs[0];
    assert_eq!(lib_name, "TRIAL001_ORR");

    // Verify CQL structure
    assert!(cql_code.contains("library TRIAL001_ORR"));
    assert!(cql_code.contains("using FHIR version '4.0.1'"));
    assert!(cql_code.contains("Assessment Window"));
    assert!(cql_code.contains("84 days"));
    assert!(cql_code.contains("-0.3")); // 30% threshold
    assert!(cql_code.contains("context Patient"));
}

#[test]
fn test_cql_pfs_generation() {
    let protocol = create_test_protocol();
    let cql_libs = protocol_endpoints_to_cql(&protocol.endpoints, "TRIAL001");

    // Find PFS library (should be second)
    let (lib_name, cql_code) = &cql_libs[1];
    assert_eq!(lib_name, "TRIAL001_PFS");

    // Verify PFS-specific content
    assert!(cql_code.contains("library TRIAL001_PFS"));
    assert!(cql_code.contains("Progression Events"));
    assert!(cql_code.contains("Progression Time Days"));
    assert!(cql_code.contains("0.20")); // 20% progression threshold
    assert!(cql_code.contains("PFS Outcome"));
}

#[test]
fn test_single_endpoint_cql() {
    let endpoint = EndpointDef {
        name: "ORR".to_string(),
        kind: EndpointKind::BinaryResponse,
        window_days: 84,
        threshold: Some(0.3),
        span: None,
    };

    let cql = endpoint_to_cql(&endpoint, "MYTRIAL");

    assert!(cql.contains("library MYTRIAL_ORR"));
    assert!(cql.contains("using FHIR version '4.0.1'"));
    assert!(cql.contains("84 days"));
}

// =============================================================================
// CDISC Export Tests
// =============================================================================

#[test]
fn test_trial_to_adsl_adtr_conversion() {
    let dataset = create_test_trial_dataset();
    let (adsl, adtr) = trial_to_adsl_adtr(&dataset, "TRIAL001");

    // Verify ADSL
    assert_eq!(adsl.len(), 3); // 3 subjects
    assert_eq!(adsl[0].subjid, 1);
    assert_eq!(adsl[0].baseline_vol, 100.0);
    assert_eq!(adsl[0].n_obs, 2);

    // Verify ADTR
    assert_eq!(adtr.len(), 6); // 6 total observations

    // Check response classification
    // Subject 1: 5% reduction = non-responder
    assert_eq!(adtr[1].response, 0); // Second row for subject 1 at day 84
    assert_eq!(adtr[1].pct_change, -5.0);

    // Subject 2: 36% reduction = responder
    assert_eq!(adtr[3].response, 1); // Second row for subject 2 at day 84
    assert!((adtr[3].pct_change - (-36.36)).abs() < 0.1);

    // Subject 3: 30% reduction (boundary) = responder
    assert_eq!(adtr[5].response, 1);
}

#[test]
fn test_adsl_csv_generation() {
    let dataset = create_test_trial_dataset();
    let (adsl, _adtr) = trial_to_adsl_adtr(&dataset, "TRIAL001");

    let csv = adsl_to_csv(&adsl);

    // Verify header
    assert!(csv.contains("STUDYID,SUBJID,ARM,DOSE_MG,WEIGHT_KG,BASELINE_VOL"));

    // Verify data rows
    assert!(csv.contains("TRIAL001,1,ArmA,0,70,100"));
    assert!(csv.contains("TRIAL001,2,ArmB,100,75,110"));

    // Verify format (should be 3 subjects + 1 header = 4 lines)
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 4);
}

#[test]
fn test_adtr_csv_generation() {
    let dataset = create_test_trial_dataset();
    let (_adsl, adtr) = trial_to_adsl_adtr(&dataset, "TRIAL001");

    let csv = adtr_to_csv(&adtr);

    // Verify header
    assert!(csv.contains("STUDYID,SUBJID,ARM,TIME_DAY,TUMOR_VOL,BASELINE_VOL,PCT_CHANGE,RESPONSE"));

    // Verify data rows exist
    assert!(csv.contains("TRIAL001"));

    // Verify format (should be 6 observations + 1 header = 7 lines)
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines.len(), 7);
}

#[test]
fn test_cdisc_json_serialization() {
    let dataset = create_test_trial_dataset();
    let (adsl, adtr) = trial_to_adsl_adtr(&dataset, "TRIAL001");

    // Test ADSL JSON
    let adsl_json = medlangc::interop::adsl_to_json(&adsl).expect("ADSL JSON serialization failed");
    assert!(adsl_json.contains("TRIAL001"));
    assert!(adsl_json.contains("ArmA"));
    assert!(adsl_json.contains("ArmB"));

    // Deserialize and verify
    let deserialized: Vec<medlangc::interop::AdslRow> =
        serde_json::from_str(&adsl_json).expect("ADSL JSON deserialization failed");
    assert_eq!(deserialized.len(), 3);

    // Test ADTR JSON
    let adtr_json = medlangc::interop::adtr_to_json(&adtr).expect("ADTR JSON serialization failed");
    assert!(adtr_json.contains("TRIAL001"));

    let adtr_deserialized: Vec<medlangc::interop::AdtrRow> =
        serde_json::from_str(&adtr_json).expect("ADTR JSON deserialization failed");
    assert_eq!(adtr_deserialized.len(), 6);
}

// =============================================================================
// End-to-End Integration Tests
// =============================================================================

#[test]
fn test_complete_fhir_cql_workflow() {
    let protocol = create_test_protocol();

    // Generate FHIR resources
    let rs = protocol_to_fhir_research_study(&protocol);
    let pd = protocol_to_fhir_plan_definition(&protocol);
    let measures = protocol_to_fhir_measures(&protocol);

    assert_eq!(rs.resource_type, "ResearchStudy");
    assert_eq!(pd.resource_type, "PlanDefinition");
    assert_eq!(measures.len(), 2);

    // Generate CQL
    let cql_libs = protocol_endpoints_to_cql(&protocol.endpoints, &protocol.name);
    assert_eq!(cql_libs.len(), 2);

    // Verify all resources have correct IDs
    assert_eq!(rs.id, protocol.name);
    assert_eq!(pd.id, format!("{}_plan", protocol.name));
    for (idx, m) in measures.iter().enumerate() {
        assert_eq!(
            m.id,
            format!("{}_{}", protocol.name, protocol.endpoints[idx].name)
        );
    }
}

#[test]
fn test_complete_cdisc_workflow() {
    let dataset = create_test_trial_dataset();

    // Convert to ADSL/ADTR
    let (adsl, adtr) = trial_to_adsl_adtr(&dataset, "TEST_STUDY");

    // Export to CSV
    let adsl_csv = adsl_to_csv(&adsl);
    let adtr_csv = adtr_to_csv(&adtr);

    // Verify CSV format
    assert!(adsl_csv.contains("STUDYID"));
    assert!(adtr_csv.contains("STUDYID"));

    // Count records
    let adsl_records = adsl_csv.lines().count() - 1; // Exclude header
    let adtr_records = adtr_csv.lines().count() - 1;

    assert_eq!(adsl_records, 3);
    assert_eq!(adtr_records, 6);

    // Export to JSON
    let adsl_json = medlangc::interop::adsl_to_json(&adsl).unwrap();
    let adtr_json = medlangc::interop::adtr_to_json(&adtr).unwrap();

    // Verify roundtrip
    let adsl_rt: Vec<medlangc::interop::AdslRow> = serde_json::from_str(&adsl_json).unwrap();
    let adtr_rt: Vec<medlangc::interop::AdtrRow> = serde_json::from_str(&adtr_json).unwrap();

    assert_eq!(adsl_rt.len(), adsl.len());
    assert_eq!(adtr_rt.len(), adtr.len());
}

#[test]
fn test_all_interop_modules_basic() {
    // This test verifies that all three interop modules are working
    let protocol = create_test_protocol();
    let dataset = create_test_trial_dataset();

    // FHIR
    let _rs = protocol_to_fhir_research_study(&protocol);
    let _bundle = trial_to_fhir_bundle(&dataset, "TEST");

    // CQL
    let _cql = protocol_endpoints_to_cql(&protocol.endpoints, "TEST");

    // CDISC
    let (_adsl, _adtr) = trial_to_adsl_adtr(&dataset, "TEST");

    // All succeeded if we got here
    assert!(true);
}
