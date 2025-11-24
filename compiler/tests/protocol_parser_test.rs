//! Tests for Week 8 Protocol DSL parser

use medlangc::ast::*;
use medlangc::lexer::tokenize;
use medlangc::parser::parse_program;

#[test]
fn test_parse_simple_protocol() {
    let source = r#"
protocol TestProtocol {
    population model TestPop

    arms {
        ArmA {
            label = "Low Dose"
            dose = 100.0
        }
        ArmB {
            label = "High Dose"
            dose = 200.0
        }
    }

    visits {
        baseline at 0.0
        week4 at 28.0
    }

    endpoints {
        ORR {
            type = "binary"
            observable = TumourVol
            shrink_frac = 0.30
            window = [0.0, 56.0]
        }
    }
}
"#;

    let tokens = tokenize(source).expect("Tokenization should succeed");
    let program = parse_program(&tokens).expect("Parsing should succeed");

    assert_eq!(program.declarations.len(), 1);

    match &program.declarations[0] {
        Declaration::Protocol(proto) => {
            assert_eq!(proto.name, "TestProtocol");
            assert_eq!(proto.population_model_name, "TestPop");
            assert_eq!(proto.arms.len(), 2);
            assert_eq!(proto.visits.len(), 2);
            assert_eq!(proto.endpoints.len(), 1);

            // Check arms
            assert_eq!(proto.arms[0].name, "ArmA");
            assert_eq!(proto.arms[0].label, "Low Dose");
            assert_eq!(proto.arms[0].dose_mg, 100.0);

            assert_eq!(proto.arms[1].name, "ArmB");
            assert_eq!(proto.arms[1].label, "High Dose");
            assert_eq!(proto.arms[1].dose_mg, 200.0);

            // Check visits
            assert_eq!(proto.visits[0].name, "baseline");
            assert_eq!(proto.visits[0].time_days, 0.0);

            assert_eq!(proto.visits[1].name, "week4");
            assert_eq!(proto.visits[1].time_days, 28.0);

            // Check endpoint
            assert_eq!(proto.endpoints[0].name, "ORR");
            assert_eq!(proto.endpoints[0].kind, EndpointKind::Binary);

            match &proto.endpoints[0].spec {
                EndpointSpec::ResponseRate {
                    observable,
                    shrink_fraction,
                    window_start_days,
                    window_end_days,
                } => {
                    assert_eq!(observable, "TumourVol");
                    assert_eq!(*shrink_fraction, 0.30);
                    assert_eq!(*window_start_days, 0.0);
                    assert_eq!(*window_end_days, 56.0);
                }
                _ => panic!("Expected ResponseRate endpoint spec"),
            }
        }
        _ => panic!("Expected Protocol declaration"),
    }
}

#[test]
fn test_parse_protocol_with_inclusion() {
    let source = r#"
protocol InclusionTest {
    population model TestPop

    arms {
        Arm1 {
            label = "Test"
            dose = 50.0
        }
    }

    visits {
        v1 at 0.0
    }

    inclusion {
        age between 18 and 75
        ECOG in [0, 1, 2]
        baseline_tumour_volume >= 50.0
    }

    endpoints {
        E1 {
            type = "binary"
            observable = X
            shrink_frac = 0.5
            window = [0.0, 10.0]
        }
    }
}
"#;

    let tokens = tokenize(source).expect("Tokenization should succeed");
    let program = parse_program(&tokens).expect("Parsing should succeed");

    match &program.declarations[0] {
        Declaration::Protocol(proto) => {
            let inclusion = proto
                .inclusion
                .as_ref()
                .expect("Should have inclusion criteria");
            assert_eq!(inclusion.clauses.len(), 3);

            match &inclusion.clauses[0] {
                InclusionClause::AgeBetween {
                    min_years,
                    max_years,
                } => {
                    assert_eq!(*min_years, 18);
                    assert_eq!(*max_years, 75);
                }
                _ => panic!("Expected AgeBetween clause"),
            }

            match &inclusion.clauses[1] {
                InclusionClause::ECOGIn { allowed } => {
                    assert_eq!(allowed, &vec![0, 1, 2]);
                }
                _ => panic!("Expected ECOGIn clause"),
            }

            match &inclusion.clauses[2] {
                InclusionClause::BaselineTumourGe { volume_cm3 } => {
                    assert_eq!(*volume_cm3, 50.0);
                }
                _ => panic!("Expected BaselineTumourGe clause"),
            }
        }
        _ => panic!("Expected Protocol declaration"),
    }
}

#[test]
fn test_parse_protocol_with_pfs_endpoint() {
    let source = r#"
protocol PFSTest {
    population model TestPop

    arms {
        Arm1 {
            label = "Test"
            dose = 100.0
        }
    }

    visits {
        v1 at 0.0
    }

    endpoints {
        PFS {
            type = "time_to_event"
            observable = TumourVol
            progression_frac = 0.20
            ref_baseline = false
            window = [0.0, 84.0]
        }
    }
}
"#;

    let tokens = tokenize(source).expect("Tokenization should succeed");
    let program = parse_program(&tokens).expect("Parsing should succeed");

    match &program.declarations[0] {
        Declaration::Protocol(proto) => {
            assert_eq!(proto.endpoints.len(), 1);
            assert_eq!(proto.endpoints[0].name, "PFS");
            assert_eq!(proto.endpoints[0].kind, EndpointKind::TimeToEvent);

            match &proto.endpoints[0].spec {
                EndpointSpec::TimeToProgression {
                    observable,
                    increase_fraction,
                    window_start_days,
                    window_end_days,
                    ref_baseline,
                } => {
                    assert_eq!(observable, "TumourVol");
                    assert_eq!(*increase_fraction, 0.20);
                    assert_eq!(*window_start_days, 0.0);
                    assert_eq!(*window_end_days, 84.0);
                    assert_eq!(*ref_baseline, false);
                }
                _ => panic!("Expected TimeToProgression endpoint spec"),
            }
        }
        _ => panic!("Expected Protocol declaration"),
    }
}

#[test]
fn test_parse_protocol_with_multiple_endpoints() {
    let source = r#"
protocol MultiEndpoint {
    population model TestPop

    arms {
        Arm1 { label = "A"; dose = 100.0 }
    }

    visits {
        v1 at 0.0
    }

    endpoints {
        ORR {
            type = "binary"
            observable = TumourVol
            shrink_frac = 0.30
            window = [0.0, 56.0]
        }

        PFS {
            type = "time_to_event"
            observable = TumourVol
            progression_frac = 0.20
            ref_baseline = true
            window = [0.0, 84.0]
        }
    }
}
"#;

    let tokens = tokenize(source).expect("Tokenization should succeed");
    let program = parse_program(&tokens).expect("Parsing should succeed");

    match &program.declarations[0] {
        Declaration::Protocol(proto) => {
            assert_eq!(proto.endpoints.len(), 2);

            // First endpoint: ORR
            assert_eq!(proto.endpoints[0].name, "ORR");
            assert_eq!(proto.endpoints[0].kind, EndpointKind::Binary);

            // Second endpoint: PFS
            assert_eq!(proto.endpoints[1].name, "PFS");
            assert_eq!(proto.endpoints[1].kind, EndpointKind::TimeToEvent);

            match &proto.endpoints[1].spec {
                EndpointSpec::TimeToProgression { ref_baseline, .. } => {
                    assert_eq!(*ref_baseline, true);
                }
                _ => panic!("Expected TimeToProgression"),
            }
        }
        _ => panic!("Expected Protocol declaration"),
    }
}
