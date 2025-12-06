//! Pharmacogenomics Module
//!
//! Implements pharmacogenomic functionality for MedLang including:
//! - Star allele nomenclature (e.g., CYP2D6*4)
//! - Activity score calculation
//! - Diplotype to phenotype mapping (UM, RM, NM, IM, PM)
//! - CPIC guideline integration for dosing recommendations
//!
//! # Example
//!
//! ```
//! use medlang::ontology::pgx::{
//!     alleles::{AlleleRegistry, Pharmacogene},
//!     phenotype::{PhenotypeClassifier, Diplotype, MetabolizerPhenotype},
//!     cpic::{CpicDecisionSupport, ClinicalAction},
//! };
//!
//! // Classify a patient's CYP2D6 diplotype
//! let classifier = PhenotypeClassifier::cyp2d6();
//! let result = classifier.classify_alleles("*1", "*4").unwrap();
//!
//! // Check: *1/*4 gives activity score of 1.0 -> Intermediate Metabolizer
//! assert_eq!(result.phenotype, MetabolizerPhenotype::IntermediateMetabolizer);
//!
//! // Get CPIC recommendation for codeine
//! let support = CpicDecisionSupport::with_common_guidelines();
//! let decision = support.decide(&result, "codeine");
//!
//! // IM for codeine: monitor for reduced efficacy
//! println!("Action: {:?}", decision.action);
//! ```

pub mod alleles;
pub mod cpic;
pub mod phenotype;

// Re-export commonly used types
pub use alleles::{AlleleFunction, AlleleRegistry, Pharmacogene, StarAllele};

pub use phenotype::{
    Diplotype, DosingCategory, MetabolizerPhenotype, PgxProfile, PhenotypeClassifier,
    PhenotypeResult, PhenotypeThresholds,
};

pub use cpic::{
    ClinicalAction, CpicDecision, CpicDecisionSupport, CpicGuideline, CpicLevel, CpicRegistry,
    DoseAdjustment, DosingRecommendation, RecommendationStrength,
};

/// Quick helper to get a recommendation for a CYP2D6 diplotype and drug
pub fn cyp2d6_recommendation(
    allele1: &str,
    allele2: &str,
    drug: &str,
) -> Result<Option<DosingRecommendation>, String> {
    let classifier = PhenotypeClassifier::cyp2d6();
    let result = classifier.classify_alleles(allele1, allele2)?;

    let support = CpicDecisionSupport::with_common_guidelines();
    let decision = support.decide(&result, drug);

    Ok(decision.recommendation)
}

/// Quick helper to get phenotype for a CYP2D6 diplotype
pub fn cyp2d6_phenotype(allele1: &str, allele2: &str) -> Result<MetabolizerPhenotype, String> {
    let classifier = PhenotypeClassifier::cyp2d6();
    let result = classifier.classify_alleles(allele1, allele2)?;
    Ok(result.phenotype)
}

/// Quick helper to get phenotype for a CYP2C19 diplotype
pub fn cyp2c19_phenotype(allele1: &str, allele2: &str) -> Result<MetabolizerPhenotype, String> {
    let classifier = PhenotypeClassifier::cyp2c19();
    let result = classifier.classify_alleles(allele1, allele2)?;
    Ok(result.phenotype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyp2d6_recommendation_helper() {
        // Poor metabolizer + codeine -> avoid
        let rec = cyp2d6_recommendation("*4", "*4", "codeine").unwrap();
        assert!(rec.is_some());
        let rec = rec.unwrap();
        assert!(rec.recommendation.contains("Avoid"));
    }

    #[test]
    fn test_cyp2d6_phenotype_helper() {
        let phenotype = cyp2d6_phenotype("*4", "*4").unwrap();
        assert_eq!(phenotype, MetabolizerPhenotype::PoorMetabolizer);

        let phenotype = cyp2d6_phenotype("*1", "*1").unwrap();
        assert_eq!(phenotype, MetabolizerPhenotype::NormalMetabolizer);
    }

    #[test]
    fn test_cyp2c19_phenotype_helper() {
        let phenotype = cyp2c19_phenotype("*2", "*2").unwrap();
        assert_eq!(phenotype, MetabolizerPhenotype::PoorMetabolizer);

        let phenotype = cyp2c19_phenotype("*17", "*17").unwrap();
        assert_eq!(phenotype, MetabolizerPhenotype::UltrarapidMetabolizer);
    }

    #[test]
    fn test_full_workflow() {
        // Patient has CYP2D6 *1/*4 (IM) and CYP2C19 *2/*2 (PM)
        let mut profile = PgxProfile::new();

        // CYP2D6
        let cyp2d6_classifier = PhenotypeClassifier::cyp2d6();
        let cyp2d6_result = cyp2d6_classifier.classify_alleles("*1", "*4").unwrap();
        profile.add_result(cyp2d6_result);

        // CYP2C19
        let cyp2c19_classifier = PhenotypeClassifier::cyp2c19();
        let cyp2c19_result = cyp2c19_classifier.classify_alleles("*2", "*2").unwrap();
        profile.add_result(cyp2c19_result);

        // Check results
        assert_eq!(
            profile.phenotype(Pharmacogene::CYP2D6),
            Some(MetabolizerPhenotype::IntermediateMetabolizer)
        );
        assert_eq!(
            profile.phenotype(Pharmacogene::CYP2C19),
            Some(MetabolizerPhenotype::PoorMetabolizer)
        );

        // Patient is PM for at least one gene
        assert!(profile.is_poor_metabolizer_any());

        // Clopidogrel check (CYP2C19 PM -> avoid)
        let support = CpicDecisionSupport::with_common_guidelines();
        let clopidogrel_decision =
            support.decide(profile.get(Pharmacogene::CYP2C19).unwrap(), "clopidogrel");
        assert_eq!(clopidogrel_decision.action, ClinicalAction::Avoid);
    }
}
