//! CQL (Clinical Quality Language) Export for MedLang Endpoints
//!
//! Generates CQL library snippets encoding endpoint semantics
//! (ORR = objective response rate, PFS = progression-free survival)

use crate::ast::{EndpointDef, EndpointKind};

/// Generate CQL library for an endpoint
pub fn endpoint_to_cql(endpoint: &EndpointDef, protocol_name: &str) -> String {
    let library_name = format!("{}_{}", protocol_name, endpoint.name);

    match endpoint.kind {
        EndpointKind::BinaryResponse => generate_orr_cql(&library_name, endpoint),
        EndpointKind::TimeToEvent => generate_pfs_cql(&library_name, endpoint),
    }
}

/// Generate CQL for ORR (Objective Response Rate) endpoint
fn generate_orr_cql(library_name: &str, endpoint: &EndpointDef) -> String {
    let threshold = endpoint.threshold.unwrap_or(0.3);
    let window_days = endpoint.window_days;

    format!(
        r#"library {library_name} version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.0.1'

// Context: Patient-level evaluation
context Patient

// Define the assessment window (e.g., Week 12 = Day 84)
define "Assessment Window":
  Interval[@2024-01-01, @2024-01-01 + {window_days} days]

// Retrieve tumor volume observations within the window
define "Tumor Volume Observations":
  [Observation: "Tumor Volume"] O
    where O.status = 'final'
      and O.effective in "Assessment Window"

// Calculate baseline tumor volume (Day 0)
define "Baseline Tumor Volume":
  First(
    [Observation: "Tumor Volume"] O
      where O.status = 'final'
        and O.effective ~ @2024-01-01
      sort by effective
  ).value as Quantity

// Calculate tumor volume at assessment
define "Assessment Tumor Volume":
  Last(
    "Tumor Volume Observations"
      sort by effective
  ).value as Quantity

// Calculate percent change from baseline
define "Percent Change":
  if "Baseline Tumor Volume" is not null and "Assessment Tumor Volume" is not null
  then (("Assessment Tumor Volume".value - "Baseline Tumor Volume".value) / "Baseline Tumor Volume".value)
  else null

// Response criterion: ≥30% reduction from baseline
define "Has Response":
  "Percent Change" is not null
    and "Percent Change" <= -{threshold}

// ORR is the proportion of patients with response
define "Objective Response":
  if "Has Response" then 1 else 0
"#,
        library_name = library_name,
        window_days = window_days,
        threshold = threshold
    )
}

/// Generate CQL for PFS (Progression-Free Survival) endpoint
fn generate_pfs_cql(library_name: &str, endpoint: &EndpointDef) -> String {
    let window_days = endpoint.window_days;

    format!(
        r#"library {library_name} version '1.0.0'

using FHIR version '4.0.1'

include FHIRHelpers version '4.0.1'

// Context: Patient-level evaluation
context Patient

// Define the observation period
define "Observation Period":
  Interval[@2024-01-01, @2024-01-01 + {window_days} days]

// Retrieve all tumor volume observations
define "Tumor Volume Observations":
  [Observation: "Tumor Volume"] O
    where O.status = 'final'
      and O.effective in "Observation Period"
    sort by effective

// Calculate baseline tumor volume
define "Baseline Tumor Volume":
  First("Tumor Volume Observations").value as Quantity

// Detect progression: ≥20% increase from nadir
define "Progression Events":
  "Tumor Volume Observations" O
    let nadir: Minimum("Tumor Volume Observations" P
                        where P.effective <= O.effective
                        return P.value as Quantity),
        percent_change: if nadir is not null
                        then ((O.value as Quantity).value - nadir.value) / nadir.value
                        else null
    where percent_change is not null
      and percent_change >= 0.20
    return {{
      time: O.effective,
      volume: O.value as Quantity
    }}

// Time to first progression (in days)
define "Progression Time Days":
  if exists("Progression Events")
  then days between @2024-01-01 and First("Progression Events").time
  else null

// Event indicator: 1 = progressed, 0 = censored
define "Progression Event":
  if "Progression Time Days" is not null then 1 else 0

// Follow-up time (days from baseline to last observation or progression)
define "Follow Up Days":
  if "Progression Time Days" is not null
  then "Progression Time Days"
  else days between @2024-01-01 and Last("Tumor Volume Observations").effective

// PFS outcome: (time, event)
define "PFS Outcome":
  {{
    time_days: Coalesce("Progression Time Days", "Follow Up Days"),
    event: "Progression Event"
  }}
"#,
        library_name = library_name,
        window_days = window_days
    )
}

/// Generate CQL for all endpoints in a protocol
pub fn protocol_endpoints_to_cql(
    endpoints: &[EndpointDef],
    protocol_name: &str,
) -> Vec<(String, String)> {
    endpoints
        .iter()
        .map(|ep| {
            let library_name = format!("{}_{}", protocol_name, ep.name);
            let cql_code = endpoint_to_cql(ep, protocol_name);
            (library_name, cql_code)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orr_cql_generation() {
        let endpoint = EndpointDef {
            name: "ORR".to_string(),
            kind: EndpointKind::BinaryResponse,
            window_days: 84,
            threshold: Some(0.3),
            span: None,
        };

        let cql = endpoint_to_cql(&endpoint, "TRIAL001");

        assert!(cql.contains("library TRIAL001_ORR"));
        assert!(cql.contains("using FHIR version '4.0.1'"));
        assert!(cql.contains("Assessment Window"));
        assert!(cql.contains("Tumor Volume Observations"));
        assert!(cql.contains("Percent Change"));
        assert!(cql.contains("Has Response"));
        assert!(cql.contains("84 days"));
        assert!(cql.contains("-0.3"));
    }

    #[test]
    fn test_pfs_cql_generation() {
        let endpoint = EndpointDef {
            name: "PFS".to_string(),
            kind: EndpointKind::TimeToEvent,
            window_days: 180,
            threshold: None,
            span: None,
        };

        let cql = endpoint_to_cql(&endpoint, "TRIAL001");

        assert!(cql.contains("library TRIAL001_PFS"));
        assert!(cql.contains("using FHIR version '4.0.1'"));
        assert!(cql.contains("Observation Period"));
        assert!(cql.contains("Progression Events"));
        assert!(cql.contains("Progression Time Days"));
        assert!(cql.contains("PFS Outcome"));
        assert!(cql.contains("180 days"));
        assert!(cql.contains("0.20")); // 20% progression threshold
    }

    #[test]
    fn test_multiple_endpoints_cql() {
        let endpoints = vec![
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
        ];

        let cql_libraries = protocol_endpoints_to_cql(&endpoints, "TRIAL001");

        assert_eq!(cql_libraries.len(), 2);
        assert_eq!(cql_libraries[0].0, "TRIAL001_ORR");
        assert_eq!(cql_libraries[1].0, "TRIAL001_PFS");
        assert!(cql_libraries[0].1.contains("library TRIAL001_ORR"));
        assert!(cql_libraries[1].1.contains("library TRIAL001_PFS"));
    }

    #[test]
    fn test_cql_includes_fhir_helpers() {
        let endpoint = EndpointDef {
            name: "ORR".to_string(),
            kind: EndpointKind::BinaryResponse,
            window_days: 84,
            threshold: Some(0.3),
            span: None,
        };

        let cql = endpoint_to_cql(&endpoint, "TRIAL001");

        assert!(cql.contains("include FHIRHelpers"));
        assert!(cql.contains("context Patient"));
    }
}
