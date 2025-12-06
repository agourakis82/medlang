// Week 54: Biomedical Ontology Infrastructure - Core Types
//
// Foundation types for biomedical ontology integration in MedLang.
// Supports multiple terminology systems: SNOMED CT, ICD-10, RxNorm, etc.
//
// ## Design Principles
//
// 1. **Uniform identifiers**: All concepts use OntologyId with system + code
// 2. **Hierarchy support**: Efficient subsumption queries via preorder numbering
// 3. **Cross-references**: Map between terminology systems (UMLS-style)
// 4. **Lazy loading**: Core concepts compiled in, extended content on demand

use std::collections::HashMap;
use std::fmt;

/// Identifies a terminology/ontology system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OntologySystem {
    /// SNOMED CT - Clinical terminology
    SnomedCt,
    /// ICD-10-CM - Diagnosis coding
    Icd10Cm,
    /// ICD-10-PCS - Procedure coding
    Icd10Pcs,
    /// RxNorm - Drug terminology
    RxNorm,
    /// ATC - Anatomical Therapeutic Chemical classification
    Atc,
    /// LOINC - Laboratory observations
    Loinc,
    /// MeSH - Medical Subject Headings
    Mesh,
    /// UMLS - Unified Medical Language System (meta-thesaurus)
    Umls,
    /// ChEBI - Chemical Entities of Biological Interest
    Chebi,
    /// GO - Gene Ontology
    GeneOntology,
    /// HPO - Human Phenotype Ontology
    Hpo,
    /// DrugBank - Drug database
    DrugBank,
    /// ClinVar - Clinical variant database
    ClinVar,
    /// OMIM - Online Mendelian Inheritance in Man
    Omim,
    /// PharmGKB - Pharmacogenomics knowledge base
    PharmGkb,
    /// Custom/local terminology
    Custom(u32),
}

impl OntologySystem {
    /// Get the standard URI prefix for this system
    pub fn uri_prefix(&self) -> &'static str {
        match self {
            OntologySystem::SnomedCt => "http://snomed.info/sct",
            OntologySystem::Icd10Cm => "http://hl7.org/fhir/sid/icd-10-cm",
            OntologySystem::Icd10Pcs => "http://hl7.org/fhir/sid/icd-10-pcs",
            OntologySystem::RxNorm => "http://www.nlm.nih.gov/research/umls/rxnorm",
            OntologySystem::Atc => "http://www.whocc.no/atc",
            OntologySystem::Loinc => "http://loinc.org",
            OntologySystem::Mesh => "http://id.nlm.nih.gov/mesh",
            OntologySystem::Umls => "http://www.nlm.nih.gov/research/umls",
            OntologySystem::Chebi => "http://purl.obolibrary.org/obo/CHEBI_",
            OntologySystem::GeneOntology => "http://purl.obolibrary.org/obo/GO_",
            OntologySystem::Hpo => "http://purl.obolibrary.org/obo/HP_",
            OntologySystem::DrugBank => "https://www.drugbank.ca/drugs/",
            OntologySystem::ClinVar => "https://www.ncbi.nlm.nih.gov/clinvar/",
            OntologySystem::Omim => "https://omim.org/entry/",
            OntologySystem::PharmGkb => "https://www.pharmgkb.org/",
            OntologySystem::Custom(_) => "urn:custom:",
        }
    }

    /// Get short code for this system
    pub fn code(&self) -> &'static str {
        match self {
            OntologySystem::SnomedCt => "SNOMEDCT",
            OntologySystem::Icd10Cm => "ICD10CM",
            OntologySystem::Icd10Pcs => "ICD10PCS",
            OntologySystem::RxNorm => "RXNORM",
            OntologySystem::Atc => "ATC",
            OntologySystem::Loinc => "LOINC",
            OntologySystem::Mesh => "MSH",
            OntologySystem::Umls => "UMLS",
            OntologySystem::Chebi => "CHEBI",
            OntologySystem::GeneOntology => "GO",
            OntologySystem::Hpo => "HP",
            OntologySystem::DrugBank => "DRUGBANK",
            OntologySystem::ClinVar => "CLINVAR",
            OntologySystem::Omim => "OMIM",
            OntologySystem::PharmGkb => "PHARMGKB",
            OntologySystem::Custom(_) => "CUSTOM",
        }
    }
}

impl fmt::Display for OntologySystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// Unique identifier for an ontology concept
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OntologyId {
    /// The terminology system
    pub system: OntologySystem,
    /// The code within that system
    pub code: String,
}

impl OntologyId {
    pub fn new(system: OntologySystem, code: &str) -> Self {
        OntologyId {
            system,
            code: code.to_string(),
        }
    }

    /// Create a SNOMED CT concept ID
    pub fn snomed(sctid: &str) -> Self {
        Self::new(OntologySystem::SnomedCt, sctid)
    }

    /// Create an ICD-10-CM code
    pub fn icd10(code: &str) -> Self {
        Self::new(OntologySystem::Icd10Cm, code)
    }

    /// Create an RxNorm concept ID
    pub fn rxnorm(rxcui: &str) -> Self {
        Self::new(OntologySystem::RxNorm, rxcui)
    }

    /// Create an ATC code
    pub fn atc(code: &str) -> Self {
        Self::new(OntologySystem::Atc, code)
    }

    /// Create a LOINC code
    pub fn loinc(code: &str) -> Self {
        Self::new(OntologySystem::Loinc, code)
    }

    /// Create a UMLS CUI
    pub fn umls(cui: &str) -> Self {
        Self::new(OntologySystem::Umls, cui)
    }

    /// Get the full URI for this concept
    pub fn to_uri(&self) -> String {
        format!("{}{}", self.system.uri_prefix(), self.code)
    }

    /// Parse from system:code format
    pub fn parse(s: &str) -> Option<Self> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        if parts.len() != 2 {
            return None;
        }
        let system = match parts[0].to_uppercase().as_str() {
            "SNOMEDCT" | "SNOMED" | "SCT" => OntologySystem::SnomedCt,
            "ICD10CM" | "ICD10" | "ICD" => OntologySystem::Icd10Cm,
            "RXNORM" | "RXN" => OntologySystem::RxNorm,
            "ATC" => OntologySystem::Atc,
            "LOINC" => OntologySystem::Loinc,
            "UMLS" | "CUI" => OntologySystem::Umls,
            "CHEBI" => OntologySystem::Chebi,
            "GO" => OntologySystem::GeneOntology,
            "HP" | "HPO" => OntologySystem::Hpo,
            "DRUGBANK" | "DB" => OntologySystem::DrugBank,
            _ => return None,
        };
        Some(Self::new(system, parts[1]))
    }
}

impl fmt::Display for OntologyId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.system.code(), self.code)
    }
}

/// Reference to a concept with optional display text
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConceptRef {
    /// The concept identifier
    pub id: OntologyId,
    /// Human-readable display text
    pub display: Option<String>,
}

impl ConceptRef {
    pub fn new(id: OntologyId) -> Self {
        ConceptRef { id, display: None }
    }

    pub fn with_display(id: OntologyId, display: &str) -> Self {
        ConceptRef {
            id,
            display: Some(display.to_string()),
        }
    }

    /// Get display text or code as fallback
    pub fn display_or_code(&self) -> &str {
        self.display.as_deref().unwrap_or(&self.id.code)
    }
}

impl fmt::Display for ConceptRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref display) = self.display {
            write!(f, "{} ({})", display, self.id)
        } else {
            write!(f, "{}", self.id)
        }
    }
}

/// Semantic type/category for a concept
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SemanticType {
    // Clinical
    Disease,
    Finding,
    Procedure,
    BodyStructure,
    Substance,

    // Drugs/Chemicals
    Drug,
    Ingredient,
    ClinicalDrug,
    DrugClass,
    Chemical,

    // Genomics
    Gene,
    Variant,
    Protein,
    Pathway,

    // Administrative
    DiagnosticCode,
    ProcedureCode,
    LabTest,

    // Other
    Observable,
    Qualifier,
    Unknown,
}

impl SemanticType {
    pub fn as_str(&self) -> &'static str {
        match self {
            SemanticType::Disease => "Disease",
            SemanticType::Finding => "Finding",
            SemanticType::Procedure => "Procedure",
            SemanticType::BodyStructure => "Body Structure",
            SemanticType::Substance => "Substance",
            SemanticType::Drug => "Drug",
            SemanticType::Ingredient => "Ingredient",
            SemanticType::ClinicalDrug => "Clinical Drug",
            SemanticType::DrugClass => "Drug Class",
            SemanticType::Chemical => "Chemical",
            SemanticType::Gene => "Gene",
            SemanticType::Variant => "Variant",
            SemanticType::Protein => "Protein",
            SemanticType::Pathway => "Pathway",
            SemanticType::DiagnosticCode => "Diagnostic Code",
            SemanticType::ProcedureCode => "Procedure Code",
            SemanticType::LabTest => "Lab Test",
            SemanticType::Observable => "Observable",
            SemanticType::Qualifier => "Qualifier",
            SemanticType::Unknown => "Unknown",
        }
    }
}

impl fmt::Display for SemanticType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Full concept definition
#[derive(Debug, Clone)]
pub struct Concept {
    /// Primary identifier
    pub id: OntologyId,
    /// Fully specified name (unique)
    pub fsn: String,
    /// Preferred term (for display)
    pub preferred_term: String,
    /// Synonyms
    pub synonyms: Vec<String>,
    /// Semantic type
    pub semantic_type: SemanticType,
    /// Parent concepts (direct superclasses)
    pub parents: Vec<OntologyId>,
    /// Whether this is a leaf node
    pub is_leaf: bool,
    /// Whether this concept is active/current
    pub is_active: bool,
    /// Cross-references to other systems
    pub mappings: Vec<OntologyId>,
}

impl Concept {
    pub fn new(id: OntologyId, fsn: &str, preferred_term: &str) -> Self {
        Concept {
            id,
            fsn: fsn.to_string(),
            preferred_term: preferred_term.to_string(),
            synonyms: Vec::new(),
            semantic_type: SemanticType::Unknown,
            parents: Vec::new(),
            is_leaf: true,
            is_active: true,
            mappings: Vec::new(),
        }
    }

    pub fn with_semantic_type(mut self, sem_type: SemanticType) -> Self {
        self.semantic_type = sem_type;
        self
    }

    pub fn with_parent(mut self, parent: OntologyId) -> Self {
        self.parents.push(parent);
        self.is_leaf = false;
        self
    }

    pub fn with_parents(mut self, parents: Vec<OntologyId>) -> Self {
        self.parents = parents;
        self.is_leaf = false;
        self
    }

    pub fn with_synonym(mut self, synonym: &str) -> Self {
        self.synonyms.push(synonym.to_string());
        self
    }

    pub fn with_mapping(mut self, mapping: OntologyId) -> Self {
        self.mappings.push(mapping);
        self
    }

    /// Get a reference to this concept
    pub fn as_ref(&self) -> ConceptRef {
        ConceptRef::with_display(self.id.clone(), &self.preferred_term)
    }
}

/// Relationship type between concepts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelationshipType {
    /// Hierarchical: A is-a B (subsumption)
    IsA,
    /// Part-whole: A part-of B
    PartOf,
    /// SNOMED: Finding site
    FindingSite,
    /// SNOMED: Associated morphology
    AssociatedMorphology,
    /// SNOMED: Causative agent
    CausativeAgent,
    /// RxNorm: has ingredient
    HasIngredient,
    /// RxNorm: tradename of
    TradenameOf,
    /// Generic relationship
    RelatedTo,
    /// Mapping to another system
    MapsTo,
    /// Broader concept (for thesauri)
    Broader,
    /// Narrower concept
    Narrower,
}

impl RelationshipType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationshipType::IsA => "is-a",
            RelationshipType::PartOf => "part-of",
            RelationshipType::FindingSite => "finding-site",
            RelationshipType::AssociatedMorphology => "associated-morphology",
            RelationshipType::CausativeAgent => "causative-agent",
            RelationshipType::HasIngredient => "has-ingredient",
            RelationshipType::TradenameOf => "tradename-of",
            RelationshipType::RelatedTo => "related-to",
            RelationshipType::MapsTo => "maps-to",
            RelationshipType::Broader => "broader",
            RelationshipType::Narrower => "narrower",
        }
    }

    pub fn is_hierarchical(&self) -> bool {
        matches!(
            self,
            RelationshipType::IsA
                | RelationshipType::PartOf
                | RelationshipType::Broader
                | RelationshipType::Narrower
        )
    }
}

impl fmt::Display for RelationshipType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A relationship between two concepts
#[derive(Debug, Clone)]
pub struct Relationship {
    /// Source concept
    pub source: OntologyId,
    /// Relationship type
    pub rel_type: RelationshipType,
    /// Target concept
    pub target: OntologyId,
    /// Whether this relationship is active
    pub is_active: bool,
}

impl Relationship {
    pub fn new(source: OntologyId, rel_type: RelationshipType, target: OntologyId) -> Self {
        Relationship {
            source,
            rel_type,
            target,
            is_active: true,
        }
    }

    pub fn is_a(source: OntologyId, target: OntologyId) -> Self {
        Self::new(source, RelationshipType::IsA, target)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ontology_id_creation() {
        let id = OntologyId::snomed("73211009");
        assert_eq!(id.system, OntologySystem::SnomedCt);
        assert_eq!(id.code, "73211009");
        assert_eq!(id.to_string(), "SNOMEDCT:73211009");
    }

    #[test]
    fn test_ontology_id_parse() {
        let id = OntologyId::parse("SNOMEDCT:73211009").unwrap();
        assert_eq!(id.system, OntologySystem::SnomedCt);
        assert_eq!(id.code, "73211009");

        let rxn = OntologyId::parse("RXNORM:161").unwrap();
        assert_eq!(rxn.system, OntologySystem::RxNorm);
    }

    #[test]
    fn test_ontology_id_uri() {
        let id = OntologyId::snomed("73211009");
        assert!(id.to_uri().contains("snomed.info"));
        assert!(id.to_uri().contains("73211009"));
    }

    #[test]
    fn test_concept_ref() {
        let id = OntologyId::snomed("73211009");
        let concept_ref = ConceptRef::with_display(id.clone(), "Diabetes mellitus");

        assert_eq!(concept_ref.display_or_code(), "Diabetes mellitus");
        assert!(concept_ref.to_string().contains("Diabetes"));
    }

    #[test]
    fn test_concept_creation() {
        let concept = Concept::new(
            OntologyId::snomed("73211009"),
            "Diabetes mellitus (disorder)",
            "Diabetes mellitus",
        )
        .with_semantic_type(SemanticType::Disease)
        .with_parent(OntologyId::snomed("126877002")) // Metabolic disease
        .with_mapping(OntologyId::icd10("E11"));

        assert_eq!(concept.semantic_type, SemanticType::Disease);
        assert_eq!(concept.parents.len(), 1);
        assert_eq!(concept.mappings.len(), 1);
        assert!(!concept.is_leaf);
    }

    #[test]
    fn test_relationship() {
        let rel = Relationship::is_a(
            OntologyId::snomed("73211009"),  // Diabetes
            OntologyId::snomed("126877002"), // Metabolic disease
        );

        assert_eq!(rel.rel_type, RelationshipType::IsA);
        assert!(rel.rel_type.is_hierarchical());
    }

    #[test]
    fn test_semantic_type() {
        assert_eq!(SemanticType::Disease.as_str(), "Disease");
        assert_eq!(SemanticType::Drug.as_str(), "Drug");
    }
}
