// Week 54: Biomedical Ontology Infrastructure
//
// Comprehensive ontology integration for MedLang, supporting multiple
// terminology systems for type-safe biomedical programming.
//
// ## Supported Ontologies
//
// - **SNOMED CT**: Clinical terminology with 350K+ concepts
// - **ICD-10-CM/PCS**: Diagnosis and procedure coding
// - **RxNorm**: Drug terminology with normalized names
// - **ATC**: Anatomical Therapeutic Chemical classification
// - **LOINC**: Laboratory observations
// - **UMLS**: Meta-thesaurus integration hub
//
// ## Architecture
//
// 1. **Core** (`core.rs`)
//    - OntologyId: Universal concept identifier
//    - ConceptRef: Reference with display text
//    - Concept: Full concept definition
//    - Relationship: Inter-concept relationships
//
// 2. **Hierarchy** (`hierarchy.rs`)
//    - OntologyHierarchy: Efficient subsumption queries
//    - TreePosition: Preorder/postorder numbering
//    - Transitive closure for DAGs
//
// 3. **RxNorm** (`rxnorm.rs`)
//    - RxDrug: Drug concept with metadata
//    - RxTermType: IN, SCD, SBD, etc.
//    - DrugStrength: Dosage representation
//
// 4. **DDI** (`ddi.rs`)
//    - DrugInteraction: DDI records
//    - DdiChecker: Multi-source interaction detection
//    - DdiSeverity: Tiered alerting
//
// ## Example Usage
//
// ```medlang
// // Type-safe condition coding
// let diagnosis: Concept<SnomedCt> = SNOMEDCT:73211009; // Diabetes mellitus
//
// // Drug with RxNorm normalization
// let medication: Drug = RXNORM:861004; // Metformin 500 MG
//
// // DDI check at compile time
// let meds = [warfarin, ibuprofen]; // Warning: Major DDI
// ```

pub mod core;
pub mod cyp;
pub mod ddi;
pub mod hierarchy;
pub mod pgx;
pub mod rxnorm;
pub mod transporters;

// Re-exports
pub use core::{
    Concept, ConceptRef, OntologyId, OntologySystem, Relationship, RelationshipType, SemanticType,
};
pub use ddi::{
    example_ddi_checker, CypEnzyme, DdiAlert, DdiChecker, DdiMechanism, DdiSeverity,
    DrugInteraction, EvidenceLevel, InhibitorPotency, PdEffectType,
};
pub use hierarchy::{HierarchyBuilder, OntologyHierarchy, TreePosition};
pub use rxnorm::{
    example_drugs, DrugStrength, RxDrug, RxIngredient, RxNormRegistry, RxRelationType, RxTermType,
};

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a SNOMED CT concept ID
pub fn snomed(sctid: &str) -> OntologyId {
    OntologyId::snomed(sctid)
}

/// Create an ICD-10-CM code
pub fn icd10(code: &str) -> OntologyId {
    OntologyId::icd10(code)
}

/// Create an RxNorm concept ID
pub fn rxnorm(rxcui: &str) -> OntologyId {
    OntologyId::rxnorm(rxcui)
}

/// Create a LOINC code
pub fn loinc(code: &str) -> OntologyId {
    OntologyId::loinc(code)
}

/// Create a UMLS CUI
pub fn umls(cui: &str) -> OntologyId {
    OntologyId::umls(cui)
}

/// Parse an ontology ID from string (system:code format)
pub fn parse_id(s: &str) -> Option<OntologyId> {
    OntologyId::parse(s)
}

// =============================================================================
// Common Medical Concepts (compiled-in core)
// =============================================================================

/// Common SNOMED CT concepts for diseases
pub mod diseases {
    use super::*;

    pub fn diabetes_mellitus() -> OntologyId {
        snomed("73211009")
    }

    pub fn type_2_diabetes() -> OntologyId {
        snomed("44054006")
    }

    pub fn type_1_diabetes() -> OntologyId {
        snomed("46635009")
    }

    pub fn hypertension() -> OntologyId {
        snomed("38341003")
    }

    pub fn heart_failure() -> OntologyId {
        snomed("84114007")
    }

    pub fn atrial_fibrillation() -> OntologyId {
        snomed("49436004")
    }

    pub fn chronic_kidney_disease() -> OntologyId {
        snomed("709044004")
    }

    pub fn copd() -> OntologyId {
        snomed("13645005")
    }

    pub fn asthma() -> OntologyId {
        snomed("195967001")
    }

    pub fn breast_cancer() -> OntologyId {
        snomed("254837009")
    }

    pub fn lung_cancer() -> OntologyId {
        snomed("254637007")
    }

    pub fn colorectal_cancer() -> OntologyId {
        snomed("363406005")
    }
}

/// Common RxNorm drug concepts
pub mod drugs {
    use super::*;

    pub fn metformin() -> OntologyId {
        rxnorm("6809")
    }

    pub fn warfarin() -> OntologyId {
        rxnorm("11289")
    }

    pub fn atorvastatin() -> OntologyId {
        rxnorm("83367")
    }

    pub fn lisinopril() -> OntologyId {
        rxnorm("29046")
    }

    pub fn amlodipine() -> OntologyId {
        rxnorm("17767")
    }

    pub fn omeprazole() -> OntologyId {
        rxnorm("7646")
    }

    pub fn aspirin() -> OntologyId {
        rxnorm("1191")
    }

    pub fn ibuprofen() -> OntologyId {
        rxnorm("5640")
    }

    pub fn acetaminophen() -> OntologyId {
        rxnorm("161")
    }

    pub fn fluoxetine() -> OntologyId {
        rxnorm("4493")
    }

    pub fn sertraline() -> OntologyId {
        rxnorm("36437")
    }

    pub fn gabapentin() -> OntologyId {
        rxnorm("25480")
    }
}

/// Common LOINC lab observations
pub mod labs {
    use super::*;

    pub fn hemoglobin_a1c() -> OntologyId {
        loinc("4548-4")
    }

    pub fn fasting_glucose() -> OntologyId {
        loinc("1558-6")
    }

    pub fn creatinine_serum() -> OntologyId {
        loinc("2160-0")
    }

    pub fn egfr() -> OntologyId {
        loinc("33914-3")
    }

    pub fn total_cholesterol() -> OntologyId {
        loinc("2093-3")
    }

    pub fn ldl_cholesterol() -> OntologyId {
        loinc("2089-1")
    }

    pub fn hdl_cholesterol() -> OntologyId {
        loinc("2085-9")
    }

    pub fn triglycerides() -> OntologyId {
        loinc("2571-8")
    }

    pub fn inr() -> OntologyId {
        loinc("6301-6")
    }

    pub fn potassium_serum() -> OntologyId {
        loinc("2823-3")
    }

    pub fn sodium_serum() -> OntologyId {
        loinc("2951-2")
    }

    pub fn alt() -> OntologyId {
        loinc("1742-6")
    }

    pub fn ast() -> OntologyId {
        loinc("1920-8")
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_ontology_id_creation() {
        let dm = diseases::diabetes_mellitus();
        assert_eq!(dm.system, OntologySystem::SnomedCt);
        assert_eq!(dm.code, "73211009");
    }

    #[test]
    fn test_parse_id() {
        let id = parse_id("SNOMEDCT:73211009").unwrap();
        assert_eq!(id.system, OntologySystem::SnomedCt);

        let rx = parse_id("RXNORM:6809").unwrap();
        assert_eq!(rx.system, OntologySystem::RxNorm);
    }

    #[test]
    fn test_hierarchy_subsumption() {
        let mut hierarchy = HierarchyBuilder::new(OntologySystem::SnomedCt)
            .add_concept(Concept::new(
                diseases::diabetes_mellitus(),
                "Diabetes mellitus (disorder)",
                "Diabetes mellitus",
            ))
            .add_concept(
                Concept::new(
                    diseases::type_2_diabetes(),
                    "Type 2 diabetes mellitus (disorder)",
                    "Type 2 diabetes",
                )
                .with_parent(diseases::diabetes_mellitus()),
            )
            .build();

        assert!(hierarchy.is_subsumed_by("44054006", "73211009"));
    }

    #[test]
    fn test_rxnorm_registry() {
        let registry = example_drugs();

        let metformin = registry.get_drug("861004");
        assert!(metformin.is_some());
    }

    #[test]
    fn test_ddi_checker() {
        let checker = example_ddi_checker();

        let interactions = checker.check_pair(&drugs::warfarin(), &drugs::ibuprofen());
        assert!(!interactions.is_empty());
        assert!(interactions[0].severity >= DdiSeverity::Major);
    }

    #[test]
    fn test_common_diseases() {
        assert_eq!(diseases::diabetes_mellitus().code, "73211009");
        assert_eq!(diseases::hypertension().code, "38341003");
    }

    #[test]
    fn test_common_drugs() {
        assert_eq!(drugs::metformin().code, "6809");
        assert_eq!(drugs::warfarin().code, "11289");
    }

    #[test]
    fn test_common_labs() {
        assert_eq!(labs::hemoglobin_a1c().code, "4548-4");
        assert_eq!(labs::creatinine_serum().code, "2160-0");
    }

    #[test]
    fn test_cross_system_mapping() {
        // Create a concept with mapping
        let concept = Concept::new(
            diseases::type_2_diabetes(),
            "Type 2 diabetes mellitus (disorder)",
            "Type 2 diabetes",
        )
        .with_mapping(icd10("E11"));

        assert_eq!(concept.mappings.len(), 1);
        assert_eq!(concept.mappings[0].system, OntologySystem::Icd10Cm);
    }
}
