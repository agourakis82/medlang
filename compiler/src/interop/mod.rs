//! Standards & Regulatory Interoperability
//!
//! Modules for exporting MedLang protocols and data to standard formats:
//! - FHIR R4: ResearchStudy, PlanDefinition, Measure, Bundle
//! - CQL: Clinical Quality Language endpoint definitions
//! - CDISC: ADSL/ADTR regulatory datasets

pub mod cdisc;
pub mod cql;
pub mod fhir;

pub use cdisc::{trial_to_adsl_adtr, AdslRow, AdtrRow};
pub use cql::{endpoint_to_cql, protocol_endpoints_to_cql};
pub use fhir::{
    protocol_to_fhir_measures, protocol_to_fhir_plan_definition, protocol_to_fhir_research_study,
    trial_to_fhir_bundle, FhirBundle, FhirMeasure, FhirPlanDefinition, FhirResearchStudy,
};

// Re-export CSV/JSON functions for CLI
pub use cdisc::{adsl_to_csv, adsl_to_json, adtr_to_csv, adtr_to_json};
