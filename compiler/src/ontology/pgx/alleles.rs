//! Star Allele Definitions for Pharmacogenomics
//!
//! Implements the star allele nomenclature system used in pharmacogenomics
//! for naming variant alleles of pharmacogenes (e.g., CYP2D6*1, *4, *17).

use std::collections::HashMap;

/// Pharmacogene - genes with known pharmacogenomic significance
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pharmacogene {
    // CYP450 enzymes
    CYP2D6,
    CYP2C19,
    CYP2C9,
    CYP3A4,
    CYP3A5,
    CYP2B6,
    CYP1A2,
    CYP2A6,

    // Phase II enzymes
    UGT1A1,
    NAT2,
    TPMT,
    DPYD,

    // Transporters
    SLCO1B1,
    ABCB1,
    ABCG2,

    // Drug targets
    VKORC1,

    // HLA genes
    HLAB,
    HLAA,

    // Other
    NUDT15,
    IFNL3,
    G6PD,
    CYP4F2,
    Factor5Leiden,
}

impl Pharmacogene {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CYP2D6 => "CYP2D6",
            Self::CYP2C19 => "CYP2C19",
            Self::CYP2C9 => "CYP2C9",
            Self::CYP3A4 => "CYP3A4",
            Self::CYP3A5 => "CYP3A5",
            Self::CYP2B6 => "CYP2B6",
            Self::CYP1A2 => "CYP1A2",
            Self::CYP2A6 => "CYP2A6",
            Self::UGT1A1 => "UGT1A1",
            Self::NAT2 => "NAT2",
            Self::TPMT => "TPMT",
            Self::DPYD => "DPYD",
            Self::SLCO1B1 => "SLCO1B1",
            Self::ABCB1 => "ABCB1",
            Self::ABCG2 => "ABCG2",
            Self::VKORC1 => "VKORC1",
            Self::HLAB => "HLA-B",
            Self::HLAA => "HLA-A",
            Self::NUDT15 => "NUDT15",
            Self::IFNL3 => "IFNL3",
            Self::G6PD => "G6PD",
            Self::CYP4F2 => "CYP4F2",
            Self::Factor5Leiden => "F5",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "CYP2D6" => Some(Self::CYP2D6),
            "CYP2C19" => Some(Self::CYP2C19),
            "CYP2C9" => Some(Self::CYP2C9),
            "CYP3A4" => Some(Self::CYP3A4),
            "CYP3A5" => Some(Self::CYP3A5),
            "CYP2B6" => Some(Self::CYP2B6),
            "CYP1A2" => Some(Self::CYP1A2),
            "CYP2A6" => Some(Self::CYP2A6),
            "UGT1A1" => Some(Self::UGT1A1),
            "NAT2" => Some(Self::NAT2),
            "TPMT" => Some(Self::TPMT),
            "DPYD" => Some(Self::DPYD),
            "SLCO1B1" => Some(Self::SLCO1B1),
            "ABCB1" => Some(Self::ABCB1),
            "ABCG2" => Some(Self::ABCG2),
            "VKORC1" => Some(Self::VKORC1),
            "HLA-B" | "HLAB" => Some(Self::HLAB),
            "HLA-A" | "HLAA" => Some(Self::HLAA),
            "NUDT15" => Some(Self::NUDT15),
            "IFNL3" => Some(Self::IFNL3),
            "G6PD" => Some(Self::G6PD),
            "CYP4F2" => Some(Self::CYP4F2),
            "F5" | "FACTOR5" => Some(Self::Factor5Leiden),
            _ => None,
        }
    }
}

/// Allele function category based on clinical effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AlleleFunction {
    /// Normal function (wild-type activity)
    Normal,
    /// Increased function (greater than normal activity)
    Increased,
    /// Decreased function (reduced but present activity)
    Decreased,
    /// No function (null allele, no activity)
    NoFunction,
    /// Uncertain/unknown function
    Uncertain,
    /// Function depends on substrate
    SubstrateDependent,
}

impl AlleleFunction {
    /// Returns the typical activity score for this function category
    pub fn default_activity_score(&self) -> f64 {
        match self {
            Self::Normal => 1.0,
            Self::Increased => 1.5, // Can vary, some are 2.0
            Self::Decreased => 0.5,
            Self::NoFunction => 0.0,
            Self::Uncertain => 0.5, // Conservative estimate
            Self::SubstrateDependent => 0.5,
        }
    }
}

/// A star allele definition
#[derive(Debug, Clone)]
pub struct StarAllele {
    /// Gene this allele belongs to
    pub gene: Pharmacogene,
    /// Star allele designation (e.g., "*1", "*4", "*17")
    pub designation: String,
    /// Full allele name (e.g., "CYP2D6*4")
    pub full_name: String,
    /// Functional category
    pub function: AlleleFunction,
    /// Activity score (CPIC standardized)
    pub activity_score: f64,
    /// Common name if any (e.g., "*4" is often called "null")
    pub common_name: Option<String>,
    /// Associated rsIDs for key variants
    pub rs_ids: Vec<String>,
    /// Clinical significance notes
    pub clinical_notes: Option<String>,
    /// Population frequency data (population -> frequency)
    pub frequencies: HashMap<String, f64>,
}

impl StarAllele {
    pub fn new(
        gene: Pharmacogene,
        designation: &str,
        function: AlleleFunction,
        activity_score: f64,
    ) -> Self {
        let full_name = format!("{}{}", gene.as_str(), designation);
        Self {
            gene,
            designation: designation.to_string(),
            full_name,
            function,
            activity_score,
            common_name: None,
            rs_ids: Vec::new(),
            clinical_notes: None,
            frequencies: HashMap::new(),
        }
    }

    pub fn with_common_name(mut self, name: &str) -> Self {
        self.common_name = Some(name.to_string());
        self
    }

    pub fn with_rs_ids(mut self, ids: Vec<&str>) -> Self {
        self.rs_ids = ids.into_iter().map(String::from).collect();
        self
    }

    pub fn with_frequency(mut self, population: &str, freq: f64) -> Self {
        self.frequencies.insert(population.to_string(), freq);
        self
    }

    pub fn with_clinical_notes(mut self, notes: &str) -> Self {
        self.clinical_notes = Some(notes.to_string());
        self
    }
}

/// Registry of known star alleles
#[derive(Debug, Default)]
pub struct AlleleRegistry {
    alleles: HashMap<String, StarAllele>,
    by_gene: HashMap<Pharmacogene, Vec<String>>,
}

impl AlleleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a registry with common CYP2D6 alleles pre-populated
    pub fn with_cyp2d6_defaults() -> Self {
        let mut registry = Self::new();

        // CYP2D6 alleles
        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*1", AlleleFunction::Normal, 1.0)
                .with_common_name("wild-type")
                .with_frequency("European", 0.35)
                .with_frequency("African", 0.50),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*2", AlleleFunction::Normal, 1.0)
                .with_rs_ids(vec!["rs16947", "rs1135840"])
                .with_frequency("European", 0.25),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*3", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs35742686"])
                .with_clinical_notes("Frameshift mutation, non-functional")
                .with_frequency("European", 0.02),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*4", AlleleFunction::NoFunction, 0.0)
                .with_common_name("null")
                .with_rs_ids(vec!["rs3892097"])
                .with_clinical_notes("Most common null allele in Europeans")
                .with_frequency("European", 0.20)
                .with_frequency("Asian", 0.01),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*5", AlleleFunction::NoFunction, 0.0)
                .with_common_name("gene deletion")
                .with_clinical_notes("Whole gene deletion")
                .with_frequency("European", 0.03),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*6", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs5030655"])
                .with_frequency("European", 0.01),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*9", AlleleFunction::Decreased, 0.5)
                .with_rs_ids(vec!["rs5030656"]),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*10", AlleleFunction::Decreased, 0.25)
                .with_rs_ids(vec!["rs1065852"])
                .with_clinical_notes("Most common decreased function allele in Asians")
                .with_frequency("Asian", 0.40)
                .with_frequency("European", 0.02),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*17", AlleleFunction::Decreased, 0.5)
                .with_rs_ids(vec!["rs28371706"])
                .with_clinical_notes("Most common decreased function allele in Africans")
                .with_frequency("African", 0.20),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*29", AlleleFunction::Decreased, 0.5)
                .with_frequency("African", 0.10),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*41", AlleleFunction::Decreased, 0.5)
                .with_rs_ids(vec!["rs28371725"])
                .with_frequency("European", 0.08),
        );

        // Gene duplication/multiplication - increased function
        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*1xN", AlleleFunction::Increased, 2.0)
                .with_common_name("gene duplication")
                .with_clinical_notes("Multiple copies of functional allele"),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2D6, "*2xN", AlleleFunction::Increased, 2.0)
                .with_common_name("gene duplication"),
        );

        registry
    }

    /// Create a registry with common CYP2C19 alleles
    pub fn with_cyp2c19_defaults() -> Self {
        let mut registry = Self::new();

        registry.register(
            StarAllele::new(Pharmacogene::CYP2C19, "*1", AlleleFunction::Normal, 1.0)
                .with_common_name("wild-type"),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2C19, "*2", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs4244285"])
                .with_clinical_notes("Most common null allele")
                .with_frequency("European", 0.15)
                .with_frequency("Asian", 0.30),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2C19, "*3", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs4986893"])
                .with_frequency("Asian", 0.05),
        );

        registry.register(
            StarAllele::new(Pharmacogene::CYP2C19, "*17", AlleleFunction::Increased, 1.5)
                .with_rs_ids(vec!["rs12248560"])
                .with_clinical_notes("Increased transcription, ultrarapid metabolizer")
                .with_frequency("European", 0.20),
        );

        registry
    }

    /// Create a comprehensive registry with all common alleles
    pub fn comprehensive() -> Self {
        let mut registry = Self::with_cyp2d6_defaults();

        // Merge CYP2C19 alleles
        let cyp2c19 = Self::with_cyp2c19_defaults();
        for (name, allele) in cyp2c19.alleles {
            registry.alleles.insert(name.clone(), allele);
            registry
                .by_gene
                .entry(Pharmacogene::CYP2C19)
                .or_default()
                .push(name);
        }

        // Add CYP2C9 alleles
        registry.register(StarAllele::new(
            Pharmacogene::CYP2C9,
            "*1",
            AlleleFunction::Normal,
            1.0,
        ));
        registry.register(
            StarAllele::new(Pharmacogene::CYP2C9, "*2", AlleleFunction::Decreased, 0.5)
                .with_rs_ids(vec!["rs1799853"]),
        );
        registry.register(
            StarAllele::new(Pharmacogene::CYP2C9, "*3", AlleleFunction::Decreased, 0.25)
                .with_rs_ids(vec!["rs1057910"])
                .with_clinical_notes("Significantly reduced warfarin metabolism"),
        );

        // Add SLCO1B1 alleles
        registry.register(StarAllele::new(
            Pharmacogene::SLCO1B1,
            "*1A",
            AlleleFunction::Normal,
            1.0,
        ));
        registry.register(
            StarAllele::new(Pharmacogene::SLCO1B1, "*5", AlleleFunction::Decreased, 0.5)
                .with_rs_ids(vec!["rs4149056"])
                .with_clinical_notes("Associated with statin myopathy risk"),
        );
        registry.register(
            StarAllele::new(
                Pharmacogene::SLCO1B1,
                "*15",
                AlleleFunction::Decreased,
                0.25,
            )
            .with_clinical_notes("Compound effect with *1B"),
        );

        // Add DPYD alleles
        registry.register(StarAllele::new(
            Pharmacogene::DPYD,
            "*1",
            AlleleFunction::Normal,
            1.0,
        ));
        registry.register(
            StarAllele::new(Pharmacogene::DPYD, "*2A", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs3918290"])
                .with_clinical_notes("Critical for fluoropyrimidine dosing"),
        );

        // Add TPMT alleles
        registry.register(StarAllele::new(
            Pharmacogene::TPMT,
            "*1",
            AlleleFunction::Normal,
            1.0,
        ));
        registry.register(
            StarAllele::new(Pharmacogene::TPMT, "*2", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs1800462"]),
        );
        registry.register(
            StarAllele::new(Pharmacogene::TPMT, "*3A", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs1800460", "rs1142345"])
                .with_clinical_notes("Most common null allele"),
        );
        registry.register(
            StarAllele::new(Pharmacogene::TPMT, "*3B", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs1800460"]),
        );
        registry.register(
            StarAllele::new(Pharmacogene::TPMT, "*3C", AlleleFunction::NoFunction, 0.0)
                .with_rs_ids(vec!["rs1142345"]),
        );

        registry
    }

    /// Register a new star allele
    pub fn register(&mut self, allele: StarAllele) {
        let name = allele.full_name.clone();
        let gene = allele.gene;

        self.alleles.insert(name.clone(), allele);
        self.by_gene.entry(gene).or_default().push(name);
    }

    /// Look up an allele by full name (e.g., "CYP2D6*4")
    pub fn get(&self, full_name: &str) -> Option<&StarAllele> {
        self.alleles.get(full_name)
    }

    /// Get all alleles for a gene
    pub fn get_alleles_for_gene(&self, gene: Pharmacogene) -> Vec<&StarAllele> {
        self.by_gene
            .get(&gene)
            .map(|names| names.iter().filter_map(|n| self.alleles.get(n)).collect())
            .unwrap_or_default()
    }

    /// Get activity score for an allele
    pub fn get_activity_score(&self, full_name: &str) -> Option<f64> {
        self.alleles.get(full_name).map(|a| a.activity_score)
    }

    /// Find allele by rsID
    pub fn find_by_rsid(&self, rsid: &str) -> Vec<&StarAllele> {
        self.alleles
            .values()
            .filter(|a| a.rs_ids.iter().any(|r| r == rsid))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_star_allele_creation() {
        let allele = StarAllele::new(Pharmacogene::CYP2D6, "*4", AlleleFunction::NoFunction, 0.0);

        assert_eq!(allele.full_name, "CYP2D6*4");
        assert_eq!(allele.activity_score, 0.0);
    }

    #[test]
    fn test_allele_with_metadata() {
        let allele = StarAllele::new(Pharmacogene::CYP2D6, "*4", AlleleFunction::NoFunction, 0.0)
            .with_common_name("null")
            .with_rs_ids(vec!["rs3892097"])
            .with_frequency("European", 0.20);

        assert_eq!(allele.common_name, Some("null".to_string()));
        assert!(allele.rs_ids.contains(&"rs3892097".to_string()));
        assert_eq!(allele.frequencies.get("European"), Some(&0.20));
    }

    #[test]
    fn test_registry_cyp2d6() {
        let registry = AlleleRegistry::with_cyp2d6_defaults();

        // Check *1 (normal)
        let star1 = registry.get("CYP2D6*1").unwrap();
        assert_eq!(star1.activity_score, 1.0);
        assert_eq!(star1.function, AlleleFunction::Normal);

        // Check *4 (null)
        let star4 = registry.get("CYP2D6*4").unwrap();
        assert_eq!(star4.activity_score, 0.0);
        assert_eq!(star4.function, AlleleFunction::NoFunction);

        // Check *10 (decreased)
        let star10 = registry.get("CYP2D6*10").unwrap();
        assert_eq!(star10.activity_score, 0.25);
        assert_eq!(star10.function, AlleleFunction::Decreased);
    }

    #[test]
    fn test_get_alleles_for_gene() {
        let registry = AlleleRegistry::with_cyp2d6_defaults();
        let alleles = registry.get_alleles_for_gene(Pharmacogene::CYP2D6);

        // Should have multiple alleles
        assert!(alleles.len() > 5);

        // All should be CYP2D6
        for allele in &alleles {
            assert_eq!(allele.gene, Pharmacogene::CYP2D6);
        }
    }

    #[test]
    fn test_find_by_rsid() {
        let registry = AlleleRegistry::with_cyp2d6_defaults();
        let matches = registry.find_by_rsid("rs3892097");

        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].designation, "*4");
    }

    #[test]
    fn test_comprehensive_registry() {
        let registry = AlleleRegistry::comprehensive();

        // Should have CYP2D6
        assert!(registry.get("CYP2D6*4").is_some());

        // Should have CYP2C19
        assert!(registry.get("CYP2C19*2").is_some());
        assert!(registry.get("CYP2C19*17").is_some());

        // Should have CYP2C9
        assert!(registry.get("CYP2C9*2").is_some());
        assert!(registry.get("CYP2C9*3").is_some());

        // Should have SLCO1B1
        assert!(registry.get("SLCO1B1*5").is_some());

        // Should have TPMT
        assert!(registry.get("TPMT*3A").is_some());
    }

    #[test]
    fn test_pharmacogene_from_str() {
        assert_eq!(Pharmacogene::from_str("CYP2D6"), Some(Pharmacogene::CYP2D6));
        assert_eq!(Pharmacogene::from_str("cyp2d6"), Some(Pharmacogene::CYP2D6));
        assert_eq!(Pharmacogene::from_str("HLA-B"), Some(Pharmacogene::HLAB));
        assert_eq!(Pharmacogene::from_str("unknown"), None);
    }

    #[test]
    fn test_allele_function_activity_scores() {
        assert_eq!(AlleleFunction::Normal.default_activity_score(), 1.0);
        assert_eq!(AlleleFunction::NoFunction.default_activity_score(), 0.0);
        assert_eq!(AlleleFunction::Decreased.default_activity_score(), 0.5);
        assert_eq!(AlleleFunction::Increased.default_activity_score(), 1.5);
    }
}
