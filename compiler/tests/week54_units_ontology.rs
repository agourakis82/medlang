// Week 54: Units of Measure and Biomedical Ontology Integration Tests
//
// Comprehensive tests for the dimensional analysis and ontology systems.

use medlangc::units::{
    self, medical, parse_quantity, standard_units, ucum_parser, unit_type_registry, validate_ucum,
    Dimension, DimensionCheckResult, DimensionChecker, Quantity, QuantityBuilder, UnitType,
};

use medlangc::ontology::{
    self, diseases, drugs, example_ddi_checker, example_drugs, icd10, labs, loinc, parse_id,
    rxnorm, snomed, Concept, ConceptRef, DdiChecker, DdiSeverity, HierarchyBuilder,
    OntologyHierarchy, OntologyId, OntologySystem, RxDrug, RxTermType, SemanticType,
};

// =============================================================================
// Units of Measure Tests
// =============================================================================

mod units_tests {
    use super::*;

    #[test]
    fn test_medical_dose_calculations() {
        // Calculate dose per body weight
        let dose = medical::dose_mg(500.0);
        let weight = medical::weight_kg(70.0);

        let dose_per_kg = dose / weight;
        let expected = 500.0 / 70.0;
        assert!((dose_per_kg.value - expected).abs() < 1e-10);
    }

    #[test]
    fn test_concentration_calculations() {
        // Drug concentration after dilution
        let drug_amount = medical::dose_mg(100.0);
        let volume = medical::volume_ml(50.0);

        let concentration = drug_amount / volume;
        // 100 mg / 50 mL = 2 mg/mL
        assert!((concentration.value - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_infusion_rate() {
        let registry = standard_units();

        // 1000 mL over 8 hours
        let volume = Quantity::new(1000.0, registry.get("mL").unwrap().clone());
        let time = Quantity::new(8.0, registry.get("h").unwrap().clone());

        let rate = volume / time;
        // 1000 mL / 8 h = 125 mL/h
        assert!((rate.value - 125.0).abs() < 1e-10);
    }

    #[test]
    fn test_unit_conversion_chain() {
        let registry = standard_units();

        // Convert 1 g to mg
        let g = Quantity::new(1.0, registry.get("g").unwrap().clone());
        let mg = g.convert_to(registry.get("mg").unwrap()).unwrap();
        assert!((mg.value - 1000.0).abs() < 1e-10);

        // Convert back
        let g_again = mg.convert_to(registry.get("g").unwrap()).unwrap();
        assert!((g_again.value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clearance_units() {
        let registry = standard_units();

        // CL = 5 L/h
        let cl = Quantity::new(5.0, registry.get("L/h").unwrap().clone());

        // Convert to mL/min
        let ml_min = registry.get("mL/min").unwrap();
        let cl_ml_min = cl.convert_to(ml_min).unwrap();

        // 5 L/h = 5000 mL / 60 min ≈ 83.33 mL/min
        assert!((cl_ml_min.value - 83.333).abs() < 0.01);
    }

    #[test]
    fn test_auc_calculation() {
        // Trapezoidal rule: AUC = (C1 + C2) / 2 * Δt
        let c1 = medical::conc_ng_ml(100.0);
        let c2 = medical::conc_ng_ml(50.0);
        let dt = medical::time_h(2.0);

        // Average concentration
        let c_avg = Quantity::new((c1.value + c2.value) / 2.0, c1.unit.clone());

        let auc_segment = c_avg * dt;
        // (100 + 50) / 2 * 2 = 150 ng·h/mL
        assert!((auc_segment.value - 150.0).abs() < 1e-10);
    }

    #[test]
    fn test_ucum_parsing() {
        let parser = ucum_parser();

        // Parse various medical units
        assert!(parser.validate("mg").is_ok());
        assert!(parser.validate("mg/kg").is_ok());
        assert!(parser.validate("mL/min").is_ok());
        assert!(parser.validate("[IU]").is_ok());
        assert!(parser.validate("mol/L").is_ok());

        // Parse with annotation
        let unit = parser.parse_to_unit("mol/L{glucose}").unwrap();
        assert_eq!(unit.annotation, Some("glucose".to_string()));
    }

    #[test]
    fn test_dimension_type_checking() {
        let mut checker = DimensionChecker::new();
        let registry = unit_type_registry();

        let mass1 = registry.lookup("mg").unwrap();
        let mass2 = registry.lookup("kg").unwrap();
        let volume = registry.lookup("mL").unwrap();

        // Same dimension types should match
        assert_eq!(
            checker.check_compatible(&mass1, &mass2),
            DimensionCheckResult::Match
        );

        // Different dimensions should not match
        match checker.check_compatible(&mass1, &volume) {
            DimensionCheckResult::Mismatch { .. } => (),
            _ => panic!("Expected dimension mismatch"),
        }
    }

    #[test]
    fn test_quantity_string_parsing() {
        let builder = QuantityBuilder::standard();

        let q1 = builder.parse("100 mg").unwrap();
        assert!((q1.value - 100.0).abs() < 1e-10);
        assert_eq!(q1.unit.ucum_code, "mg");

        let q2 = builder.parse("5.5 mL/min").unwrap();
        assert!((q2.value - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_international_units_isolation() {
        let registry = standard_units();

        let iu = Quantity::new(100.0, registry.get("[IU]").unwrap().clone());
        let mg = registry.get("mg").unwrap();

        // IU should not convert to mg (different biological activities)
        assert!(iu.convert_to(mg).is_err());
    }

    #[test]
    fn test_quantity_arithmetic_chain() {
        // Complex calculation: clearance * time = volume eliminated
        let cl = medical::clearance_l_h(5.0);
        let time = medical::time_h(24.0);

        let volume_eliminated = cl * time;
        // 5 L/h * 24 h = 120 L (volume dimension, not dimensionless)
        assert!((volume_eliminated.value - 120.0).abs() < 1e-10);

        // dose / weight is dimensionless (mg/kg = mass/mass cancels)
        let dose = medical::dose_mg(500.0);
        let weight = medical::weight_kg(70.0);
        let dose_per_kg = dose / weight;

        // mg / kg = (mass) / (mass) = dimensionless ratio
        assert!((dose_per_kg.value - 500.0 / 70.0).abs() < 1e-10);
    }
}

// =============================================================================
// Ontology Tests
// =============================================================================

mod ontology_tests {
    use super::*;

    #[test]
    fn test_snomed_concept_creation() {
        let dm = diseases::diabetes_mellitus();
        assert_eq!(dm.system, OntologySystem::SnomedCt);
        assert_eq!(dm.code, "73211009");

        let uri = dm.to_uri();
        assert!(uri.contains("snomed.info"));
    }

    #[test]
    fn test_rxnorm_concept_creation() {
        let metformin = drugs::metformin();
        assert_eq!(metformin.system, OntologySystem::RxNorm);
        assert_eq!(metformin.code, "6809");
    }

    #[test]
    fn test_loinc_concept_creation() {
        let a1c = labs::hemoglobin_a1c();
        assert_eq!(a1c.system, OntologySystem::Loinc);
        assert_eq!(a1c.code, "4548-4");
    }

    #[test]
    fn test_concept_id_parsing() {
        let snomed_id = parse_id("SNOMEDCT:73211009").unwrap();
        assert_eq!(snomed_id.system, OntologySystem::SnomedCt);

        let rxnorm_id = parse_id("RXNORM:6809").unwrap();
        assert_eq!(rxnorm_id.system, OntologySystem::RxNorm);

        let icd_id = parse_id("ICD10:E11").unwrap();
        assert_eq!(icd_id.system, OntologySystem::Icd10Cm);
    }

    #[test]
    fn test_disease_hierarchy() {
        // Build a simple disease hierarchy
        let mut hierarchy = HierarchyBuilder::new(OntologySystem::SnomedCt)
            .add_concept(
                Concept::new(snomed("64572001"), "Disease (disorder)", "Disease")
                    .with_semantic_type(SemanticType::Disease),
            )
            .add_concept(
                Concept::new(
                    snomed("126877002"),
                    "Metabolic disease (disorder)",
                    "Metabolic disease",
                )
                .with_semantic_type(SemanticType::Disease)
                .with_parent(snomed("64572001")),
            )
            .add_concept(
                Concept::new(
                    diseases::diabetes_mellitus(),
                    "Diabetes mellitus (disorder)",
                    "Diabetes mellitus",
                )
                .with_semantic_type(SemanticType::Disease)
                .with_parent(snomed("126877002")),
            )
            .add_concept(
                Concept::new(
                    diseases::type_2_diabetes(),
                    "Type 2 diabetes mellitus (disorder)",
                    "Type 2 diabetes",
                )
                .with_semantic_type(SemanticType::Disease)
                .with_parent(diseases::diabetes_mellitus()),
            )
            .build();

        // Test subsumption
        assert!(hierarchy.is_subsumed_by("44054006", "73211009")); // T2DM is-a DM
        assert!(hierarchy.is_subsumed_by("73211009", "126877002")); // DM is-a Metabolic
        assert!(hierarchy.is_subsumed_by("44054006", "64572001")); // T2DM is-a Disease

        // Test non-subsumption
        assert!(!hierarchy.is_subsumed_by("73211009", "44054006")); // DM is NOT a T2DM
    }

    #[test]
    fn test_cross_system_mapping() {
        let concept = Concept::new(
            diseases::type_2_diabetes(),
            "Type 2 diabetes mellitus (disorder)",
            "Type 2 diabetes",
        )
        .with_semantic_type(SemanticType::Disease)
        .with_mapping(icd10("E11"))
        .with_mapping(OntologyId::umls("C0011860"));

        assert_eq!(concept.mappings.len(), 2);

        // Check ICD-10 mapping
        let icd_mapping = concept
            .mappings
            .iter()
            .find(|m| m.system == OntologySystem::Icd10Cm)
            .unwrap();
        assert_eq!(icd_mapping.code, "E11");
    }

    #[test]
    fn test_rxnorm_drug_registry() {
        let registry = example_drugs();

        // Find drug by RxCUI
        let metformin = registry.get_drug("861004").unwrap();
        assert!(metformin.name.contains("Metformin"));
        assert!(metformin.prescribable);

        // Find by ingredient
        let metformin_drugs = registry.find_by_ingredient("6809");
        assert!(!metformin_drugs.is_empty());

        // Find branded drug
        let lipitor_drugs = registry.find_by_brand("Lipitor");
        assert!(!lipitor_drugs.is_empty());
    }

    #[test]
    fn test_rx_term_types() {
        assert!(RxTermType::Ingredient.is_ingredient());
        assert!(!RxTermType::Ingredient.is_prescribable());
        assert!(RxTermType::ClinicalDrug.is_prescribable());
        assert!(RxTermType::BrandedDrug.is_prescribable());
    }

    #[test]
    fn test_concept_ref_display() {
        let id = diseases::diabetes_mellitus();
        let concept_ref = ConceptRef::with_display(id.clone(), "Diabetes mellitus");

        assert_eq!(concept_ref.display_or_code(), "Diabetes mellitus");

        let display = concept_ref.to_string();
        assert!(display.contains("Diabetes"));
        assert!(display.contains("SNOMEDCT"));
    }
}

// =============================================================================
// DDI Tests
// =============================================================================

mod ddi_tests {
    use super::*;

    #[test]
    fn test_ddi_severity_hierarchy() {
        assert!(DdiSeverity::Contraindicated > DdiSeverity::Major);
        assert!(DdiSeverity::Major > DdiSeverity::Moderate);
        assert!(DdiSeverity::Moderate > DdiSeverity::Minor);

        assert!(DdiSeverity::Contraindicated.is_hard_stop());
        assert!(DdiSeverity::Major.is_interruptive());
        assert!(!DdiSeverity::Moderate.is_interruptive());
    }

    #[test]
    fn test_warfarin_nsaid_interaction() {
        let checker = example_ddi_checker();

        let interactions = checker.check_pair(&drugs::warfarin(), &drugs::ibuprofen());

        assert!(!interactions.is_empty());

        let interaction = interactions[0];
        assert!(interaction.severity >= DdiSeverity::Major);
        assert!(interaction.description.to_lowercase().contains("bleeding"));
    }

    #[test]
    fn test_contraindicated_interaction() {
        let checker = example_ddi_checker();

        // Simvastatin + Clarithromycin is contraindicated
        let interactions = checker.check_pair(
            &rxnorm("21212"), // Clarithromycin
            &rxnorm("36567"), // Simvastatin
        );

        assert!(!interactions.is_empty());
        assert_eq!(interactions[0].severity, DdiSeverity::Contraindicated);
    }

    #[test]
    fn test_medication_list_screening() {
        let checker = example_ddi_checker();

        let meds = vec![drugs::warfarin(), drugs::ibuprofen(), drugs::metformin()];

        let interactions = checker.check_medication_list(&meds);

        // Should find warfarin + ibuprofen
        assert!(interactions
            .iter()
            .any(|i| { i.involves(&drugs::warfarin()) && i.involves(&drugs::ibuprofen()) }));

        // Should NOT find metformin + ibuprofen (no known interaction in our test data)
        assert!(!interactions
            .iter()
            .any(|i| { i.involves(&drugs::metformin()) && i.involves(&drugs::ibuprofen()) }));
    }

    #[test]
    fn test_alert_generation() {
        let checker = example_ddi_checker();

        let meds = vec![
            (drugs::warfarin(), "Warfarin 5mg".to_string()),
            (drugs::ibuprofen(), "Ibuprofen 400mg".to_string()),
        ];

        let alerts = checker.generate_alerts(&meds);

        assert!(!alerts.is_empty());

        let alert = &alerts[0];
        assert!(alert.message().contains("Warfarin"));
        assert!(alert.message().contains("Ibuprofen"));
    }

    #[test]
    fn test_high_priority_filtering() {
        let checker = example_ddi_checker();

        let meds = vec![
            rxnorm("4493"), // Fluoxetine
            rxnorm("6011"), // Phenelzine
        ];

        let high_priority = checker.check_high_priority(&meds);

        assert!(!high_priority.is_empty());
        assert!(high_priority
            .iter()
            .all(|i| i.severity >= DdiSeverity::Major));
    }

    #[test]
    fn test_bidirectional_interaction() {
        let checker = example_ddi_checker();

        // Check both directions
        let forward = checker.check_pair(&drugs::warfarin(), &drugs::ibuprofen());
        let reverse = checker.check_pair(&drugs::ibuprofen(), &drugs::warfarin());

        // Both directions should find the interaction
        assert!(!forward.is_empty());
        assert!(!reverse.is_empty());
    }
}

// =============================================================================
// Integration Tests - Combined Units + Ontology
// =============================================================================

mod combined_tests {
    use super::*;

    #[test]
    fn test_drug_dose_with_ontology() {
        // Get drug from ontology
        let registry = example_drugs();
        let metformin = registry.get_drug("861004").unwrap();

        // Verify it's the right drug
        assert!(metformin.name.contains("Metformin"));
        assert!(metformin.name.contains("500"));

        // Create a dose quantity
        let dose = medical::dose_mg(500.0);
        let weight = medical::weight_kg(70.0);
        let dose_per_kg = dose / weight;

        // 500 mg / 70 kg ≈ 7.14 mg/kg
        assert!((dose_per_kg.value - 7.14).abs() < 0.01);
    }

    #[test]
    fn test_lab_value_with_units() {
        // Reference a lab test from ontology
        let a1c = labs::hemoglobin_a1c();
        assert_eq!(a1c.system, OntologySystem::Loinc);

        // Create a lab value with proper units
        // A1c is typically reported as % (dimensionless)
        let registry = standard_units();
        let value = Quantity::new(7.2, registry.get("%").unwrap().clone());

        assert!((value.value - 7.2).abs() < 1e-10);
    }

    #[test]
    fn test_creatinine_clearance_calculation() {
        // Cockcroft-Gault formula components
        let age = 65.0; // years
        let weight = medical::weight_kg(70.0);
        let scr = 1.2; // mg/dL (serum creatinine)
        let is_female = false;

        // Reference the creatinine lab code
        let _scr_loinc = labs::creatinine_serum();

        // CrCl = ((140 - age) × weight) / (72 × SCr) × (0.85 if female)
        let numerator = (140.0 - age) * weight.value;
        let denominator = 72.0 * scr;
        let crcl_value = numerator / denominator * if is_female { 0.85 } else { 1.0 };

        let crcl = medical::clearance_ml_min(crcl_value);

        // Expected: (75 × 70) / (72 × 1.2) = 5250 / 86.4 ≈ 60.76 mL/min
        assert!((crcl.value - 60.76).abs() < 0.1);
    }

    #[test]
    fn test_clinical_scenario() {
        // Patient on warfarin getting ibuprofen prescription
        let checker = example_ddi_checker();

        // Check for interactions
        let current_meds = vec![drugs::warfarin()];
        let new_med = drugs::ibuprofen();

        // Combine for checking
        let all_meds: Vec<_> = current_meds
            .iter()
            .cloned()
            .chain(std::iter::once(new_med.clone()))
            .collect();

        let interactions = checker.check_medication_list(&all_meds);

        // Should flag the interaction
        assert!(!interactions.is_empty());
        let interaction = interactions[0];

        // Verify severity warrants attention
        assert!(interaction.severity >= DdiSeverity::Major);

        // Check that management guidance exists
        assert!(interaction.management.is_some());
    }
}
