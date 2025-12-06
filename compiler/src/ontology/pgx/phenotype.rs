//! Metabolizer Phenotype Classification
//!
//! Implements the CPIC standardized phenotype classification system
//! for pharmacogenomics. Maps diplotypes to phenotype categories
//! using activity scores.

use super::alleles::{AlleleFunction, AlleleRegistry, Pharmacogene, StarAllele};
use std::collections::HashMap;

/// Metabolizer phenotype categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetabolizerPhenotype {
    /// Ultrarapid metabolizer - increased enzyme activity
    UltrarapidMetabolizer,
    /// Rapid metabolizer - between normal and ultrarapid
    RapidMetabolizer,
    /// Normal metabolizer - typical enzyme activity (formerly "extensive")
    NormalMetabolizer,
    /// Intermediate metabolizer - reduced enzyme activity
    IntermediateMetabolizer,
    /// Poor metabolizer - minimal or no enzyme activity
    PoorMetabolizer,
    /// Indeterminate - cannot be classified
    Indeterminate,
}

impl MetabolizerPhenotype {
    /// Standard abbreviation
    pub fn abbreviation(&self) -> &'static str {
        match self {
            Self::UltrarapidMetabolizer => "UM",
            Self::RapidMetabolizer => "RM",
            Self::NormalMetabolizer => "NM",
            Self::IntermediateMetabolizer => "IM",
            Self::PoorMetabolizer => "PM",
            Self::Indeterminate => "Indet",
        }
    }

    /// Full name
    pub fn full_name(&self) -> &'static str {
        match self {
            Self::UltrarapidMetabolizer => "Ultrarapid Metabolizer",
            Self::RapidMetabolizer => "Rapid Metabolizer",
            Self::NormalMetabolizer => "Normal Metabolizer",
            Self::IntermediateMetabolizer => "Intermediate Metabolizer",
            Self::PoorMetabolizer => "Poor Metabolizer",
            Self::Indeterminate => "Indeterminate",
        }
    }

    /// Clinical significance description
    pub fn clinical_significance(&self) -> &'static str {
        match self {
            Self::UltrarapidMetabolizer =>
                "Increased metabolism may result in reduced drug efficacy or increased toxicity of prodrugs",
            Self::RapidMetabolizer =>
                "Slightly increased metabolism, may require dosage adjustment for some drugs",
            Self::NormalMetabolizer =>
                "Expected drug response at standard doses",
            Self::IntermediateMetabolizer =>
                "Reduced metabolism may result in increased drug exposure and toxicity risk",
            Self::PoorMetabolizer =>
                "Significantly reduced metabolism, high risk of toxicity, may require major dose reduction or alternative drug",
            Self::Indeterminate =>
                "Phenotype cannot be determined from available genetic information",
        }
    }

    /// Typical dose adjustment factor (1.0 = standard dose)
    pub fn typical_dose_factor(&self) -> Option<f64> {
        match self {
            Self::UltrarapidMetabolizer => Some(1.5), // May need higher dose or avoid
            Self::RapidMetabolizer => Some(1.25),
            Self::NormalMetabolizer => Some(1.0),
            Self::IntermediateMetabolizer => Some(0.5),
            Self::PoorMetabolizer => Some(0.25),
            Self::Indeterminate => None,
        }
    }
}

/// A diplotype (pair of alleles) for a gene
#[derive(Debug, Clone, PartialEq)]
pub struct Diplotype {
    /// The gene
    pub gene: Pharmacogene,
    /// First allele (typically maternal)
    pub allele1: String,
    /// Second allele (typically paternal)
    pub allele2: String,
}

impl Diplotype {
    pub fn new(gene: Pharmacogene, allele1: &str, allele2: &str) -> Self {
        Self {
            gene,
            allele1: allele1.to_string(),
            allele2: allele2.to_string(),
        }
    }

    /// Standard notation (e.g., "CYP2D6 *1/*4")
    pub fn notation(&self) -> String {
        format!("{} {}/{}", self.gene.as_str(), self.allele1, self.allele2)
    }

    /// Calculate total activity score from allele registry
    pub fn activity_score(&self, registry: &AlleleRegistry) -> Option<f64> {
        let full1 = format!("{}{}", self.gene.as_str(), self.allele1);
        let full2 = format!("{}{}", self.gene.as_str(), self.allele2);

        let score1 = registry.get_activity_score(&full1)?;
        let score2 = registry.get_activity_score(&full2)?;

        Some(score1 + score2)
    }
}

/// Configuration for phenotype classification thresholds
#[derive(Debug, Clone)]
pub struct PhenotypeThresholds {
    /// Minimum activity score for ultrarapid
    pub ultrarapid_min: f64,
    /// Minimum activity score for rapid
    pub rapid_min: f64,
    /// Minimum activity score for normal
    pub normal_min: f64,
    /// Minimum activity score for intermediate
    pub intermediate_min: f64,
    /// Below this is poor metabolizer
    pub poor_max: f64,
}

impl Default for PhenotypeThresholds {
    /// Default CPIC thresholds for CYP2D6
    /// Based on CPIC CYP2D6 guideline activity score ranges:
    /// - UM: AS > 2.25 (gene duplications)
    /// - NM: AS 1.25-2.25 (includes *1/*1 = 2.0)
    /// - IM: AS 0.25-1.0
    /// - PM: AS 0
    fn default() -> Self {
        Self {
            ultrarapid_min: 2.5,    // > 2.25 for gene duplications (*1xN/*1 = 3.0)
            rapid_min: 2.26,        // Rarely used category
            normal_min: 1.25,       // 1.25-2.25 includes *1/*1 (2.0)
            intermediate_min: 0.25, // 0.25-1.0
            poor_max: 0.25,         // 0-0.25
        }
    }
}

impl PhenotypeThresholds {
    /// CPIC thresholds for CYP2D6
    pub fn cyp2d6() -> Self {
        Self::default()
    }

    /// CPIC thresholds for CYP2C19
    /// CYP2C19 activity scores: *1=1.0, *2=0.0, *17=1.5
    /// - UM: AS > 2.5 (*17/*17 = 3.0)
    /// - RM: AS 2.0-2.5 (*1/*17 = 2.5)
    /// - NM: AS 1.5-2.0 (*1/*1 = 2.0)
    /// - IM: AS 0.5-1.5 (*1/*2 = 1.0)
    /// - PM: AS 0 (*2/*2 = 0.0)
    pub fn cyp2c19() -> Self {
        Self {
            ultrarapid_min: 2.75,  // *17/*17 = 3.0
            rapid_min: 2.25,       // *1/*17 = 2.5
            normal_min: 1.5,       // *1/*1 = 2.0
            intermediate_min: 0.5, // *1/*2 = 1.0
            poor_max: 0.5,         // *2/*2 = 0.0
        }
    }

    /// Classify activity score to phenotype
    pub fn classify(&self, activity_score: f64) -> MetabolizerPhenotype {
        if activity_score >= self.ultrarapid_min {
            MetabolizerPhenotype::UltrarapidMetabolizer
        } else if activity_score >= self.rapid_min {
            MetabolizerPhenotype::RapidMetabolizer
        } else if activity_score >= self.normal_min {
            MetabolizerPhenotype::NormalMetabolizer
        } else if activity_score >= self.intermediate_min {
            MetabolizerPhenotype::IntermediateMetabolizer
        } else {
            MetabolizerPhenotype::PoorMetabolizer
        }
    }
}

/// Phenotype classification result
#[derive(Debug, Clone)]
pub struct PhenotypeResult {
    /// The diplotype that was analyzed
    pub diplotype: Diplotype,
    /// Calculated activity score
    pub activity_score: f64,
    /// Resulting phenotype
    pub phenotype: MetabolizerPhenotype,
    /// Individual allele activity scores
    pub allele_scores: (f64, f64),
    /// Any warnings or notes
    pub warnings: Vec<String>,
}

impl PhenotypeResult {
    /// Get dosing recommendation category
    pub fn dosing_category(&self) -> DosingCategory {
        match self.phenotype {
            MetabolizerPhenotype::UltrarapidMetabolizer => DosingCategory::ConsiderAlternative,
            MetabolizerPhenotype::RapidMetabolizer => DosingCategory::MonitorResponse,
            MetabolizerPhenotype::NormalMetabolizer => DosingCategory::StandardDose,
            MetabolizerPhenotype::IntermediateMetabolizer => DosingCategory::ReducedDose,
            MetabolizerPhenotype::PoorMetabolizer => DosingCategory::SignificantReduction,
            MetabolizerPhenotype::Indeterminate => DosingCategory::ClinicalJudgment,
        }
    }
}

/// Dosing recommendation categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DosingCategory {
    /// Use standard dosing
    StandardDose,
    /// Monitor response, may need adjustment
    MonitorResponse,
    /// Use reduced dose
    ReducedDose,
    /// Significant dose reduction needed
    SignificantReduction,
    /// Consider alternative drug
    ConsiderAlternative,
    /// Use clinical judgment
    ClinicalJudgment,
}

impl DosingCategory {
    pub fn description(&self) -> &'static str {
        match self {
            Self::StandardDose => "Standard dosing recommended",
            Self::MonitorResponse => "Use standard dose with enhanced monitoring",
            Self::ReducedDose => "Consider dose reduction (25-50%)",
            Self::SignificantReduction => "Significant dose reduction required (>50%) or avoid",
            Self::ConsiderAlternative => {
                "Consider alternative drug that is not affected by this enzyme"
            }
            Self::ClinicalJudgment => "Insufficient evidence; use clinical judgment",
        }
    }
}

/// Phenotype classifier for a specific gene
pub struct PhenotypeClassifier {
    gene: Pharmacogene,
    thresholds: PhenotypeThresholds,
    allele_registry: AlleleRegistry,
}

impl PhenotypeClassifier {
    pub fn new(
        gene: Pharmacogene,
        thresholds: PhenotypeThresholds,
        registry: AlleleRegistry,
    ) -> Self {
        Self {
            gene,
            thresholds,
            allele_registry: registry,
        }
    }

    /// Create a CYP2D6 classifier with default alleles
    pub fn cyp2d6() -> Self {
        Self::new(
            Pharmacogene::CYP2D6,
            PhenotypeThresholds::cyp2d6(),
            AlleleRegistry::with_cyp2d6_defaults(),
        )
    }

    /// Create a CYP2C19 classifier
    pub fn cyp2c19() -> Self {
        Self::new(
            Pharmacogene::CYP2C19,
            PhenotypeThresholds::cyp2c19(),
            AlleleRegistry::with_cyp2c19_defaults(),
        )
    }

    /// Classify a diplotype
    pub fn classify(&self, diplotype: &Diplotype) -> Result<PhenotypeResult, String> {
        if diplotype.gene != self.gene {
            return Err(format!(
                "Diplotype gene {} does not match classifier gene {}",
                diplotype.gene.as_str(),
                self.gene.as_str()
            ));
        }

        let full1 = format!("{}{}", self.gene.as_str(), diplotype.allele1);
        let full2 = format!("{}{}", self.gene.as_str(), diplotype.allele2);

        let score1 = self
            .allele_registry
            .get_activity_score(&full1)
            .ok_or_else(|| format!("Unknown allele: {}", full1))?;
        let score2 = self
            .allele_registry
            .get_activity_score(&full2)
            .ok_or_else(|| format!("Unknown allele: {}", full2))?;

        let total_score = score1 + score2;
        let phenotype = self.thresholds.classify(total_score);

        let mut warnings = Vec::new();

        // Add warnings for edge cases
        if let Some(allele1) = self.allele_registry.get(&full1) {
            if allele1.function == AlleleFunction::Uncertain {
                warnings.push(format!("{} has uncertain function", full1));
            }
        }
        if let Some(allele2) = self.allele_registry.get(&full2) {
            if allele2.function == AlleleFunction::Uncertain {
                warnings.push(format!("{} has uncertain function", full2));
            }
        }

        Ok(PhenotypeResult {
            diplotype: diplotype.clone(),
            activity_score: total_score,
            phenotype,
            allele_scores: (score1, score2),
            warnings,
        })
    }

    /// Classify from allele pair (convenience method)
    pub fn classify_alleles(
        &self,
        allele1: &str,
        allele2: &str,
    ) -> Result<PhenotypeResult, String> {
        let diplotype = Diplotype::new(self.gene, allele1, allele2);
        self.classify(&diplotype)
    }

    /// Get all possible phenotypes for this gene
    pub fn possible_phenotypes(&self) -> Vec<(Diplotype, PhenotypeResult)> {
        let alleles = self.allele_registry.get_alleles_for_gene(self.gene);
        let mut results = Vec::new();

        for (i, allele1) in alleles.iter().enumerate() {
            for allele2 in alleles.iter().skip(i) {
                let diplotype =
                    Diplotype::new(self.gene, &allele1.designation, &allele2.designation);
                if let Ok(result) = self.classify(&diplotype) {
                    results.push((diplotype, result));
                }
            }
        }

        results
    }
}

/// Multi-gene phenotype assessment
#[derive(Debug, Default)]
pub struct PgxProfile {
    /// Phenotype results by gene
    results: HashMap<Pharmacogene, PhenotypeResult>,
}

impl PgxProfile {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a phenotype result
    pub fn add_result(&mut self, result: PhenotypeResult) {
        self.results.insert(result.diplotype.gene, result);
    }

    /// Get result for a gene
    pub fn get(&self, gene: Pharmacogene) -> Option<&PhenotypeResult> {
        self.results.get(&gene)
    }

    /// Get phenotype for a gene
    pub fn phenotype(&self, gene: Pharmacogene) -> Option<MetabolizerPhenotype> {
        self.results.get(&gene).map(|r| r.phenotype)
    }

    /// Check if patient is a poor metabolizer for any tested gene
    pub fn is_poor_metabolizer_any(&self) -> bool {
        self.results
            .values()
            .any(|r| r.phenotype == MetabolizerPhenotype::PoorMetabolizer)
    }

    /// Get genes where patient is a poor metabolizer
    pub fn poor_metabolizer_genes(&self) -> Vec<Pharmacogene> {
        self.results
            .iter()
            .filter(|(_, r)| r.phenotype == MetabolizerPhenotype::PoorMetabolizer)
            .map(|(g, _)| *g)
            .collect()
    }

    /// Get summary of all results
    pub fn summary(&self) -> Vec<String> {
        self.results
            .iter()
            .map(|(gene, result)| {
                format!(
                    "{}: {} (AS: {:.1})",
                    gene.as_str(),
                    result.phenotype.abbreviation(),
                    result.activity_score
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diplotype_notation() {
        let diplotype = Diplotype::new(Pharmacogene::CYP2D6, "*1", "*4");
        assert_eq!(diplotype.notation(), "CYP2D6 *1/*4");
    }

    #[test]
    fn test_cyp2d6_poor_metabolizer() {
        let classifier = PhenotypeClassifier::cyp2d6();
        let result = classifier.classify_alleles("*4", "*4").unwrap();

        assert_eq!(result.phenotype, MetabolizerPhenotype::PoorMetabolizer);
        assert_eq!(result.activity_score, 0.0);
    }

    #[test]
    fn test_cyp2d6_normal_metabolizer() {
        let classifier = PhenotypeClassifier::cyp2d6();
        let result = classifier.classify_alleles("*1", "*1").unwrap();

        assert_eq!(result.phenotype, MetabolizerPhenotype::NormalMetabolizer);
        assert_eq!(result.activity_score, 2.0);
    }

    #[test]
    fn test_cyp2d6_intermediate_metabolizer() {
        let classifier = PhenotypeClassifier::cyp2d6();

        // *1/*4 should be intermediate (AS = 1.0)
        let result = classifier.classify_alleles("*1", "*4").unwrap();
        assert_eq!(
            result.phenotype,
            MetabolizerPhenotype::IntermediateMetabolizer
        );
        assert_eq!(result.activity_score, 1.0);

        // *41/*41 should be intermediate (AS = 1.0)
        let result2 = classifier.classify_alleles("*41", "*41").unwrap();
        assert_eq!(
            result2.phenotype,
            MetabolizerPhenotype::IntermediateMetabolizer
        );
    }

    #[test]
    fn test_cyp2d6_ultrarapid_metabolizer() {
        let classifier = PhenotypeClassifier::cyp2d6();
        let result = classifier.classify_alleles("*1xN", "*1").unwrap();

        assert_eq!(
            result.phenotype,
            MetabolizerPhenotype::UltrarapidMetabolizer
        );
        assert!(result.activity_score >= 2.25);
    }

    #[test]
    fn test_cyp2c19_classification() {
        let classifier = PhenotypeClassifier::cyp2c19();

        // *2/*2 is poor metabolizer
        let pm = classifier.classify_alleles("*2", "*2").unwrap();
        assert_eq!(pm.phenotype, MetabolizerPhenotype::PoorMetabolizer);

        // *17/*17 is ultrarapid
        let um = classifier.classify_alleles("*17", "*17").unwrap();
        assert_eq!(um.phenotype, MetabolizerPhenotype::UltrarapidMetabolizer);
    }

    #[test]
    fn test_phenotype_abbreviations() {
        assert_eq!(MetabolizerPhenotype::PoorMetabolizer.abbreviation(), "PM");
        assert_eq!(MetabolizerPhenotype::NormalMetabolizer.abbreviation(), "NM");
        assert_eq!(
            MetabolizerPhenotype::UltrarapidMetabolizer.abbreviation(),
            "UM"
        );
    }

    #[test]
    fn test_dosing_category() {
        let classifier = PhenotypeClassifier::cyp2d6();

        let pm = classifier.classify_alleles("*4", "*4").unwrap();
        assert_eq!(pm.dosing_category(), DosingCategory::SignificantReduction);

        let nm = classifier.classify_alleles("*1", "*1").unwrap();
        assert_eq!(nm.dosing_category(), DosingCategory::StandardDose);
    }

    #[test]
    fn test_pgx_profile() {
        let mut profile = PgxProfile::new();

        let cyp2d6 = PhenotypeClassifier::cyp2d6();
        let result = cyp2d6.classify_alleles("*4", "*4").unwrap();
        profile.add_result(result);

        assert!(profile.is_poor_metabolizer_any());
        assert_eq!(
            profile.phenotype(Pharmacogene::CYP2D6),
            Some(MetabolizerPhenotype::PoorMetabolizer)
        );

        let pm_genes = profile.poor_metabolizer_genes();
        assert!(pm_genes.contains(&Pharmacogene::CYP2D6));
    }

    #[test]
    fn test_activity_score_calculation() {
        let registry = AlleleRegistry::with_cyp2d6_defaults();
        let diplotype = Diplotype::new(Pharmacogene::CYP2D6, "*1", "*10");

        let score = diplotype.activity_score(&registry).unwrap();
        assert_eq!(score, 1.25); // 1.0 + 0.25
    }

    #[test]
    fn test_unknown_allele_error() {
        let classifier = PhenotypeClassifier::cyp2d6();
        let result = classifier.classify_alleles("*999", "*1");

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Unknown allele"));
    }

    #[test]
    fn test_thresholds_classification() {
        let thresholds = PhenotypeThresholds::cyp2d6();

        assert_eq!(
            thresholds.classify(0.0),
            MetabolizerPhenotype::PoorMetabolizer
        );
        assert_eq!(
            thresholds.classify(0.5),
            MetabolizerPhenotype::IntermediateMetabolizer
        );
        assert_eq!(
            thresholds.classify(1.5),
            MetabolizerPhenotype::NormalMetabolizer
        );
        assert_eq!(
            thresholds.classify(3.0),
            MetabolizerPhenotype::UltrarapidMetabolizer
        );
    }
}
