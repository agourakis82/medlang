//! CPIC Guideline Integration
//!
//! Implements Clinical Pharmacogenetics Implementation Consortium (CPIC)
//! guidelines for gene-drug pairs. Provides standardized recommendations
//! based on phenotype classification.

use super::alleles::Pharmacogene;
use super::phenotype::{DosingCategory, MetabolizerPhenotype, PhenotypeResult};
use std::collections::HashMap;

/// CPIC evidence level for a guideline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CpicLevel {
    /// Level A: Genetic information should be used to change prescribing
    A,
    /// Level B: Genetic information could be used to change prescribing
    B,
    /// Level C: Evidence is weak, gene-drug association unclear
    C,
    /// Level D: Limited evidence, likely not clinically relevant
    D,
}

impl CpicLevel {
    pub fn description(&self) -> &'static str {
        match self {
            Self::A => "Prescribing action is recommended",
            Self::B => "Prescribing action is potentially beneficial",
            Self::C => "Evidence is weak or conflicting",
            Self::D => "Evidence is limited or not actionable",
        }
    }

    /// Whether this level warrants clinical action
    pub fn is_actionable(&self) -> bool {
        matches!(self, Self::A | Self::B)
    }
}

/// Strength of dosing recommendation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecommendationStrength {
    /// Strong recommendation
    Strong,
    /// Moderate recommendation
    Moderate,
    /// Optional/weak recommendation
    Optional,
    /// No recommendation possible
    None,
}

/// A specific dosing recommendation
#[derive(Debug, Clone)]
pub struct DosingRecommendation {
    /// The gene involved
    pub gene: Pharmacogene,
    /// Phenotype this applies to
    pub phenotype: MetabolizerPhenotype,
    /// Drug name
    pub drug: String,
    /// Drug class
    pub drug_class: Option<String>,
    /// Primary recommendation text
    pub recommendation: String,
    /// Dose adjustment if applicable
    pub dose_adjustment: Option<DoseAdjustment>,
    /// Strength of recommendation
    pub strength: RecommendationStrength,
    /// Alternative drugs to consider
    pub alternatives: Vec<String>,
    /// Additional clinical considerations
    pub considerations: Vec<String>,
    /// Source CPIC publication
    pub source: Option<String>,
}

/// Dose adjustment specification
#[derive(Debug, Clone)]
pub struct DoseAdjustment {
    /// Factor to multiply standard dose (1.0 = no change)
    pub factor: f64,
    /// Absolute maximum dose if applicable
    pub max_dose: Option<String>,
    /// Absolute minimum dose if applicable
    pub min_dose: Option<String>,
    /// Starting dose recommendation
    pub starting_dose: Option<String>,
    /// Special instructions
    pub instructions: Option<String>,
}

impl DoseAdjustment {
    pub fn reduce(factor: f64) -> Self {
        Self {
            factor,
            max_dose: None,
            min_dose: None,
            starting_dose: None,
            instructions: None,
        }
    }

    pub fn increase(factor: f64) -> Self {
        Self {
            factor,
            max_dose: None,
            min_dose: None,
            starting_dose: None,
            instructions: None,
        }
    }

    pub fn avoid() -> Self {
        Self {
            factor: 0.0,
            max_dose: None,
            min_dose: None,
            starting_dose: None,
            instructions: Some("Avoid use of this drug".to_string()),
        }
    }

    pub fn with_max_dose(mut self, dose: &str) -> Self {
        self.max_dose = Some(dose.to_string());
        self
    }

    pub fn with_starting_dose(mut self, dose: &str) -> Self {
        self.starting_dose = Some(dose.to_string());
        self
    }

    pub fn with_instructions(mut self, instructions: &str) -> Self {
        self.instructions = Some(instructions.to_string());
        self
    }
}

/// A CPIC guideline for a gene-drug pair
#[derive(Debug, Clone)]
pub struct CpicGuideline {
    /// Gene involved
    pub gene: Pharmacogene,
    /// Drug name
    pub drug: String,
    /// Drug class
    pub drug_class: Option<String>,
    /// CPIC evidence level
    pub level: CpicLevel,
    /// Recommendations by phenotype
    pub recommendations: HashMap<MetabolizerPhenotype, DosingRecommendation>,
    /// Publication reference
    pub publication: Option<String>,
    /// PMID if available
    pub pmid: Option<String>,
    /// Last update date
    pub last_updated: Option<String>,
}

impl CpicGuideline {
    pub fn new(gene: Pharmacogene, drug: &str, level: CpicLevel) -> Self {
        Self {
            gene,
            drug: drug.to_string(),
            drug_class: None,
            level,
            recommendations: HashMap::new(),
            publication: None,
            pmid: None,
            last_updated: None,
        }
    }

    pub fn with_drug_class(mut self, class: &str) -> Self {
        self.drug_class = Some(class.to_string());
        self
    }

    pub fn with_publication(mut self, pub_ref: &str, pmid: Option<&str>) -> Self {
        self.publication = Some(pub_ref.to_string());
        self.pmid = pmid.map(String::from);
        self
    }

    pub fn add_recommendation(&mut self, rec: DosingRecommendation) {
        self.recommendations.insert(rec.phenotype, rec);
    }

    /// Get recommendation for a phenotype
    pub fn get_recommendation(
        &self,
        phenotype: MetabolizerPhenotype,
    ) -> Option<&DosingRecommendation> {
        self.recommendations.get(&phenotype)
    }

    /// Get recommendation from a phenotype result
    pub fn recommend(&self, result: &PhenotypeResult) -> Option<&DosingRecommendation> {
        if result.diplotype.gene != self.gene {
            return None;
        }
        self.recommendations.get(&result.phenotype)
    }
}

/// Registry of CPIC guidelines
#[derive(Debug, Default)]
pub struct CpicRegistry {
    guidelines: Vec<CpicGuideline>,
    by_drug: HashMap<String, Vec<usize>>,
    by_gene: HashMap<Pharmacogene, Vec<usize>>,
}

impl CpicRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create registry with common CPIC guidelines
    pub fn with_common_guidelines() -> Self {
        let mut registry = Self::new();

        // CYP2D6 + Codeine
        let mut codeine = CpicGuideline::new(Pharmacogene::CYP2D6, "codeine", CpicLevel::A)
            .with_drug_class("opioid analgesic")
            .with_publication("CPIC Guideline for Codeine and CYP2D6", Some("23486447"));

        codeine.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2D6,
            phenotype: MetabolizerPhenotype::UltrarapidMetabolizer,
            drug: "codeine".to_string(),
            drug_class: Some("opioid".to_string()),
            recommendation: "Avoid codeine use due to potential for serious toxicity. Use alternative analgesic.".to_string(),
            dose_adjustment: Some(DoseAdjustment::avoid()),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["morphine".to_string(), "non-opioid analgesics".to_string()],
            considerations: vec!["Risk of life-threatening respiratory depression".to_string()],
            source: Some("CPIC".to_string()),
        });

        codeine.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2D6,
            phenotype: MetabolizerPhenotype::NormalMetabolizer,
            drug: "codeine".to_string(),
            drug_class: Some("opioid".to_string()),
            recommendation: "Use label-recommended age- or weight-specific dosing.".to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec![],
            source: Some("CPIC".to_string()),
        });

        codeine.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2D6,
            phenotype: MetabolizerPhenotype::IntermediateMetabolizer,
            drug: "codeine".to_string(),
            drug_class: Some("opioid".to_string()),
            recommendation: "Use label-recommended dosing. Monitor for reduced efficacy."
                .to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Moderate,
            alternatives: vec!["morphine".to_string()],
            considerations: vec!["May have reduced response".to_string()],
            source: Some("CPIC".to_string()),
        });

        codeine.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2D6,
            phenotype: MetabolizerPhenotype::PoorMetabolizer,
            drug: "codeine".to_string(),
            drug_class: Some("opioid".to_string()),
            recommendation: "Avoid codeine due to lack of efficacy. Use alternative analgesic.".to_string(),
            dose_adjustment: Some(DoseAdjustment::avoid()),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["morphine".to_string(), "non-opioid analgesics".to_string()],
            considerations: vec!["Codeine is a prodrug requiring CYP2D6 for activation".to_string()],
            source: Some("CPIC".to_string()),
        });

        registry.register(codeine);

        // CYP2D6 + Tramadol
        let mut tramadol = CpicGuideline::new(Pharmacogene::CYP2D6, "tramadol", CpicLevel::A)
            .with_drug_class("opioid analgesic");

        tramadol.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2D6,
            phenotype: MetabolizerPhenotype::UltrarapidMetabolizer,
            drug: "tramadol".to_string(),
            drug_class: Some("opioid".to_string()),
            recommendation: "Avoid tramadol. Use alternative analgesic.".to_string(),
            dose_adjustment: Some(DoseAdjustment::avoid()),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["non-opioid analgesics".to_string()],
            considerations: vec!["Risk of respiratory depression".to_string()],
            source: Some("CPIC".to_string()),
        });

        tramadol.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2D6,
            phenotype: MetabolizerPhenotype::PoorMetabolizer,
            drug: "tramadol".to_string(),
            drug_class: Some("opioid".to_string()),
            recommendation: "Avoid tramadol due to reduced efficacy. Consider alternative."
                .to_string(),
            dose_adjustment: Some(DoseAdjustment::avoid()),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["non-opioid analgesics".to_string()],
            considerations: vec![],
            source: Some("CPIC".to_string()),
        });

        registry.register(tramadol);

        // CYP2C19 + Clopidogrel
        let mut clopidogrel =
            CpicGuideline::new(Pharmacogene::CYP2C19, "clopidogrel", CpicLevel::A)
                .with_drug_class("antiplatelet")
                .with_publication(
                    "CPIC Guideline for Clopidogrel and CYP2C19",
                    Some("21716271"),
                );

        clopidogrel.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2C19,
            phenotype: MetabolizerPhenotype::UltrarapidMetabolizer,
            drug: "clopidogrel".to_string(),
            drug_class: Some("antiplatelet".to_string()),
            recommendation: "Use label-recommended dosage and administration.".to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec![],
            source: Some("CPIC".to_string()),
        });

        clopidogrel.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2C19,
            phenotype: MetabolizerPhenotype::NormalMetabolizer,
            drug: "clopidogrel".to_string(),
            drug_class: Some("antiplatelet".to_string()),
            recommendation: "Use label-recommended dosage.".to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec![],
            source: Some("CPIC".to_string()),
        });

        clopidogrel.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2C19,
            phenotype: MetabolizerPhenotype::IntermediateMetabolizer,
            drug: "clopidogrel".to_string(),
            drug_class: Some("antiplatelet".to_string()),
            recommendation: "Consider alternative antiplatelet therapy (prasugrel, ticagrelor)."
                .to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Moderate,
            alternatives: vec!["prasugrel".to_string(), "ticagrelor".to_string()],
            considerations: vec!["Reduced platelet inhibition with clopidogrel".to_string()],
            source: Some("CPIC".to_string()),
        });

        clopidogrel.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2C19,
            phenotype: MetabolizerPhenotype::PoorMetabolizer,
            drug: "clopidogrel".to_string(),
            drug_class: Some("antiplatelet".to_string()),
            recommendation: "Use alternative antiplatelet therapy (prasugrel, ticagrelor)."
                .to_string(),
            dose_adjustment: Some(DoseAdjustment::avoid()),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["prasugrel".to_string(), "ticagrelor".to_string()],
            considerations: vec!["Significantly reduced clopidogrel activation".to_string()],
            source: Some("CPIC".to_string()),
        });

        registry.register(clopidogrel);

        // CYP2C9 + VKORC1 + Warfarin (simplified, gene-specific)
        let mut warfarin_cyp2c9 =
            CpicGuideline::new(Pharmacogene::CYP2C9, "warfarin", CpicLevel::A)
                .with_drug_class("anticoagulant")
                .with_publication("CPIC Guideline for Warfarin", Some("28198005"));

        warfarin_cyp2c9.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::CYP2C9,
            phenotype: MetabolizerPhenotype::PoorMetabolizer,
            drug: "warfarin".to_string(),
            drug_class: Some("anticoagulant".to_string()),
            recommendation: "Decrease initial dose by 20-40%. More frequent INR monitoring."
                .to_string(),
            dose_adjustment: Some(
                DoseAdjustment::reduce(0.6).with_instructions("Start low, titrate slowly"),
            ),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["DOACs (if appropriate)".to_string()],
            considerations: vec![
                "Increased bleeding risk".to_string(),
                "Longer time to stable dose".to_string(),
            ],
            source: Some("CPIC".to_string()),
        });

        registry.register(warfarin_cyp2c9);

        // TPMT + Thiopurines
        let mut azathioprine = CpicGuideline::new(Pharmacogene::TPMT, "azathioprine", CpicLevel::A)
            .with_drug_class("immunosuppressant")
            .with_publication("CPIC Guideline for Thiopurines", Some("21270794"));

        azathioprine.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::TPMT,
            phenotype: MetabolizerPhenotype::NormalMetabolizer,
            drug: "azathioprine".to_string(),
            drug_class: Some("immunosuppressant".to_string()),
            recommendation: "Start with normal starting dose.".to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec![],
            source: Some("CPIC".to_string()),
        });

        azathioprine.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::TPMT,
            phenotype: MetabolizerPhenotype::IntermediateMetabolizer,
            drug: "azathioprine".to_string(),
            drug_class: Some("immunosuppressant".to_string()),
            recommendation:
                "Start with reduced dose (30-70% of normal). Allow 2-4 weeks to reach steady state."
                    .to_string(),
            dose_adjustment: Some(DoseAdjustment::reduce(0.5)),
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec!["Monitor for myelosuppression".to_string()],
            source: Some("CPIC".to_string()),
        });

        azathioprine.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::TPMT,
            phenotype: MetabolizerPhenotype::PoorMetabolizer,
            drug: "azathioprine".to_string(),
            drug_class: Some("immunosuppressant".to_string()),
            recommendation: "Consider alternative agent or drastically reduce dose (10-fold reduction). Frequent monitoring required.".to_string(),
            dose_adjustment: Some(DoseAdjustment::reduce(0.1).with_instructions("Thrice weekly dosing may be considered")),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["mycophenolate".to_string()],
            considerations: vec!["High risk of life-threatening myelosuppression".to_string()],
            source: Some("CPIC".to_string()),
        });

        registry.register(azathioprine);

        // SLCO1B1 + Simvastatin
        let mut simvastatin =
            CpicGuideline::new(Pharmacogene::SLCO1B1, "simvastatin", CpicLevel::A)
                .with_drug_class("statin")
                .with_publication(
                    "CPIC Guideline for Simvastatin and SLCO1B1",
                    Some("22617227"),
                );

        simvastatin.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::SLCO1B1,
            phenotype: MetabolizerPhenotype::NormalMetabolizer,
            drug: "simvastatin".to_string(),
            drug_class: Some("statin".to_string()),
            recommendation: "Use label-recommended dosing.".to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec![],
            source: Some("CPIC".to_string()),
        });

        simvastatin.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::SLCO1B1,
            phenotype: MetabolizerPhenotype::IntermediateMetabolizer,
            drug: "simvastatin".to_string(),
            drug_class: Some("statin".to_string()),
            recommendation: "Consider lower dose or alternative statin. Avoid >20mg simvastatin."
                .to_string(),
            dose_adjustment: Some(DoseAdjustment::reduce(0.5).with_max_dose("20mg")),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["pravastatin".to_string(), "rosuvastatin".to_string()],
            considerations: vec!["Increased myopathy risk".to_string()],
            source: Some("CPIC".to_string()),
        });

        simvastatin.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::SLCO1B1,
            phenotype: MetabolizerPhenotype::PoorMetabolizer,
            drug: "simvastatin".to_string(),
            drug_class: Some("statin".to_string()),
            recommendation: "Use alternative statin (pravastatin, rosuvastatin).".to_string(),
            dose_adjustment: Some(DoseAdjustment::avoid()),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["pravastatin".to_string(), "rosuvastatin".to_string()],
            considerations: vec!["High risk of myopathy/rhabdomyolysis".to_string()],
            source: Some("CPIC".to_string()),
        });

        registry.register(simvastatin);

        // DPYD + Fluoropyrimidines
        let mut fluorouracil = CpicGuideline::new(Pharmacogene::DPYD, "fluorouracil", CpicLevel::A)
            .with_drug_class("antineoplastic")
            .with_publication(
                "CPIC Guideline for Fluoropyrimidines and DPYD",
                Some("29152729"),
            );

        fluorouracil.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::DPYD,
            phenotype: MetabolizerPhenotype::NormalMetabolizer,
            drug: "fluorouracil".to_string(),
            drug_class: Some("fluoropyrimidine".to_string()),
            recommendation: "Use label-recommended dosing and administration.".to_string(),
            dose_adjustment: None,
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec![],
            source: Some("CPIC".to_string()),
        });

        fluorouracil.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::DPYD,
            phenotype: MetabolizerPhenotype::IntermediateMetabolizer,
            drug: "fluorouracil".to_string(),
            drug_class: Some("fluoropyrimidine".to_string()),
            recommendation: "Reduce starting dose by 50%. Titrate based on toxicity/efficacy."
                .to_string(),
            dose_adjustment: Some(DoseAdjustment::reduce(0.5)),
            strength: RecommendationStrength::Strong,
            alternatives: vec![],
            considerations: vec!["Increased severe toxicity risk".to_string()],
            source: Some("CPIC".to_string()),
        });

        fluorouracil.add_recommendation(DosingRecommendation {
            gene: Pharmacogene::DPYD,
            phenotype: MetabolizerPhenotype::PoorMetabolizer,
            drug: "fluorouracil".to_string(),
            drug_class: Some("fluoropyrimidine".to_string()),
            recommendation: "Avoid fluoropyrimidines. Select alternative drug.".to_string(),
            dose_adjustment: Some(DoseAdjustment::avoid()),
            strength: RecommendationStrength::Strong,
            alternatives: vec!["Consider non-fluoropyrimidine regimen".to_string()],
            considerations: vec!["High risk of potentially fatal toxicity".to_string()],
            source: Some("CPIC".to_string()),
        });

        registry.register(fluorouracil);

        registry
    }

    /// Register a guideline
    pub fn register(&mut self, guideline: CpicGuideline) {
        let idx = self.guidelines.len();
        let drug = guideline.drug.to_lowercase();
        let gene = guideline.gene;

        self.guidelines.push(guideline);
        self.by_drug.entry(drug).or_default().push(idx);
        self.by_gene.entry(gene).or_default().push(idx);
    }

    /// Get guidelines for a drug
    pub fn get_by_drug(&self, drug: &str) -> Vec<&CpicGuideline> {
        let drug = drug.to_lowercase();
        self.by_drug
            .get(&drug)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| self.guidelines.get(i))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get guidelines for a gene
    pub fn get_by_gene(&self, gene: Pharmacogene) -> Vec<&CpicGuideline> {
        self.by_gene
            .get(&gene)
            .map(|indices| {
                indices
                    .iter()
                    .filter_map(|&i| self.guidelines.get(i))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get specific gene-drug guideline
    pub fn get(&self, gene: Pharmacogene, drug: &str) -> Option<&CpicGuideline> {
        let drug = drug.to_lowercase();
        self.by_drug.get(&drug)?.iter().find_map(|&idx| {
            let g = self.guidelines.get(idx)?;
            if g.gene == gene {
                Some(g)
            } else {
                None
            }
        })
    }

    /// Get recommendation for a phenotype result and drug
    pub fn get_recommendation(
        &self,
        result: &PhenotypeResult,
        drug: &str,
    ) -> Option<&DosingRecommendation> {
        self.get(result.diplotype.gene, drug)?
            .get_recommendation(result.phenotype)
    }

    /// Get all actionable guidelines (Level A or B)
    pub fn actionable_guidelines(&self) -> Vec<&CpicGuideline> {
        self.guidelines
            .iter()
            .filter(|g| g.level.is_actionable())
            .collect()
    }

    /// Check if a gene-drug pair has a CPIC guideline
    pub fn has_guideline(&self, gene: Pharmacogene, drug: &str) -> bool {
        self.get(gene, drug).is_some()
    }

    /// Get all drugs with guidelines for a gene
    pub fn drugs_for_gene(&self, gene: Pharmacogene) -> Vec<&str> {
        self.get_by_gene(gene)
            .iter()
            .map(|g| g.drug.as_str())
            .collect()
    }
}

/// Clinical decision support result
#[derive(Debug)]
pub struct CpicDecision {
    /// The phenotype result used
    pub phenotype: PhenotypeResult,
    /// The drug being prescribed
    pub drug: String,
    /// The guideline (if found)
    pub guideline: Option<CpicLevel>,
    /// The recommendation
    pub recommendation: Option<DosingRecommendation>,
    /// Overall action category
    pub action: ClinicalAction,
}

/// Clinical action categories
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClinicalAction {
    /// Proceed with standard dosing
    Standard,
    /// Adjust dose
    AdjustDose,
    /// Consider alternative
    ConsiderAlternative,
    /// Avoid - high risk
    Avoid,
    /// Monitor closely
    Monitor,
    /// No guideline available
    NoGuideline,
}

impl ClinicalAction {
    pub fn severity(&self) -> u8 {
        match self {
            Self::Standard => 0,
            Self::Monitor => 1,
            Self::AdjustDose => 2,
            Self::ConsiderAlternative => 3,
            Self::Avoid => 4,
            Self::NoGuideline => 0,
        }
    }
}

/// Clinical decision support engine
pub struct CpicDecisionSupport {
    registry: CpicRegistry,
}

impl CpicDecisionSupport {
    pub fn new(registry: CpicRegistry) -> Self {
        Self { registry }
    }

    pub fn with_common_guidelines() -> Self {
        Self::new(CpicRegistry::with_common_guidelines())
    }

    /// Get clinical decision for a phenotype and drug
    pub fn decide(&self, phenotype: &PhenotypeResult, drug: &str) -> CpicDecision {
        let guideline = self.registry.get(phenotype.diplotype.gene, drug);
        let recommendation =
            guideline.and_then(|g| g.get_recommendation(phenotype.phenotype).cloned());

        let action = if let Some(ref rec) = recommendation {
            if rec
                .dose_adjustment
                .as_ref()
                .map(|d| d.factor == 0.0)
                .unwrap_or(false)
            {
                ClinicalAction::Avoid
            } else if rec.dose_adjustment.is_some() {
                ClinicalAction::AdjustDose
            } else if !rec.alternatives.is_empty() {
                ClinicalAction::ConsiderAlternative
            } else if rec.strength == RecommendationStrength::Moderate {
                ClinicalAction::Monitor
            } else {
                ClinicalAction::Standard
            }
        } else if guideline.is_some() {
            ClinicalAction::Standard // Guideline exists but no specific rec for this phenotype
        } else {
            ClinicalAction::NoGuideline
        };

        CpicDecision {
            phenotype: phenotype.clone(),
            drug: drug.to_string(),
            guideline: guideline.map(|g| g.level),
            recommendation,
            action,
        }
    }

    /// Check multiple drugs for a patient
    pub fn check_medications(
        &self,
        phenotype: &PhenotypeResult,
        drugs: &[&str],
    ) -> Vec<CpicDecision> {
        drugs
            .iter()
            .map(|drug| self.decide(phenotype, drug))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ontology::pgx::phenotype::{Diplotype, PhenotypeClassifier};

    #[test]
    fn test_cpic_level() {
        assert!(CpicLevel::A.is_actionable());
        assert!(CpicLevel::B.is_actionable());
        assert!(!CpicLevel::C.is_actionable());
        assert!(!CpicLevel::D.is_actionable());
    }

    #[test]
    fn test_dose_adjustment() {
        let reduce = DoseAdjustment::reduce(0.5);
        assert_eq!(reduce.factor, 0.5);

        let avoid = DoseAdjustment::avoid();
        assert_eq!(avoid.factor, 0.0);
        assert!(avoid.instructions.is_some());
    }

    #[test]
    fn test_registry_lookup() {
        let registry = CpicRegistry::with_common_guidelines();

        // Should find codeine by drug
        let codeine_guidelines = registry.get_by_drug("codeine");
        assert!(!codeine_guidelines.is_empty());

        // Should find codeine by gene
        let cyp2d6_guidelines = registry.get_by_gene(Pharmacogene::CYP2D6);
        assert!(!cyp2d6_guidelines.is_empty());

        // Specific lookup
        let codeine = registry.get(Pharmacogene::CYP2D6, "codeine");
        assert!(codeine.is_some());
        assert_eq!(codeine.unwrap().level, CpicLevel::A);
    }

    #[test]
    fn test_codeine_poor_metabolizer() {
        let registry = CpicRegistry::with_common_guidelines();
        let guideline = registry.get(Pharmacogene::CYP2D6, "codeine").unwrap();

        let rec = guideline
            .get_recommendation(MetabolizerPhenotype::PoorMetabolizer)
            .unwrap();
        assert!(rec.recommendation.contains("Avoid"));
        assert!(rec
            .dose_adjustment
            .as_ref()
            .map(|d| d.factor == 0.0)
            .unwrap_or(false));
    }

    #[test]
    fn test_codeine_ultrarapid_metabolizer() {
        let registry = CpicRegistry::with_common_guidelines();
        let guideline = registry.get(Pharmacogene::CYP2D6, "codeine").unwrap();

        let rec = guideline
            .get_recommendation(MetabolizerPhenotype::UltrarapidMetabolizer)
            .unwrap();
        assert!(rec.recommendation.contains("Avoid"));
        assert!(rec.alternatives.contains(&"morphine".to_string()));
    }

    #[test]
    fn test_clopidogrel_decision() {
        let classifier = PhenotypeClassifier::cyp2c19();
        let pm_result = classifier.classify_alleles("*2", "*2").unwrap();

        let support = CpicDecisionSupport::with_common_guidelines();
        let decision = support.decide(&pm_result, "clopidogrel");

        assert_eq!(decision.action, ClinicalAction::Avoid);
        assert!(decision.recommendation.is_some());

        let rec = decision.recommendation.unwrap();
        assert!(rec.alternatives.contains(&"prasugrel".to_string()));
    }

    #[test]
    fn test_simvastatin_slco1b1() {
        let registry = CpicRegistry::with_common_guidelines();
        let guideline = registry.get(Pharmacogene::SLCO1B1, "simvastatin").unwrap();

        // Poor metabolizer should avoid
        let pm_rec = guideline
            .get_recommendation(MetabolizerPhenotype::PoorMetabolizer)
            .unwrap();
        assert!(pm_rec.alternatives.contains(&"pravastatin".to_string()));

        // Intermediate should reduce dose
        let im_rec = guideline
            .get_recommendation(MetabolizerPhenotype::IntermediateMetabolizer)
            .unwrap();
        assert!(im_rec
            .dose_adjustment
            .as_ref()
            .map(|d| d.max_dose.as_ref().map(|m| m == "20mg").unwrap_or(false))
            .unwrap_or(false));
    }

    #[test]
    fn test_fluorouracil_dpyd() {
        let registry = CpicRegistry::with_common_guidelines();
        let guideline = registry.get(Pharmacogene::DPYD, "fluorouracil").unwrap();

        let pm_rec = guideline
            .get_recommendation(MetabolizerPhenotype::PoorMetabolizer)
            .unwrap();
        assert!(pm_rec.considerations.iter().any(|c| c.contains("fatal")));
    }

    #[test]
    fn test_azathioprine_tpmt() {
        let registry = CpicRegistry::with_common_guidelines();
        let guideline = registry.get(Pharmacogene::TPMT, "azathioprine").unwrap();

        let im_rec = guideline
            .get_recommendation(MetabolizerPhenotype::IntermediateMetabolizer)
            .unwrap();
        assert_eq!(im_rec.dose_adjustment.as_ref().map(|d| d.factor), Some(0.5));

        let pm_rec = guideline
            .get_recommendation(MetabolizerPhenotype::PoorMetabolizer)
            .unwrap();
        assert_eq!(pm_rec.dose_adjustment.as_ref().map(|d| d.factor), Some(0.1));
    }

    #[test]
    fn test_actionable_guidelines() {
        let registry = CpicRegistry::with_common_guidelines();
        let actionable = registry.actionable_guidelines();

        // All our default guidelines are Level A
        assert!(!actionable.is_empty());
        for g in actionable {
            assert!(g.level.is_actionable());
        }
    }

    #[test]
    fn test_drugs_for_gene() {
        let registry = CpicRegistry::with_common_guidelines();
        let drugs = registry.drugs_for_gene(Pharmacogene::CYP2D6);

        assert!(drugs.contains(&"codeine"));
        assert!(drugs.contains(&"tramadol"));
    }

    #[test]
    fn test_no_guideline_decision() {
        let classifier = PhenotypeClassifier::cyp2d6();
        let result = classifier.classify_alleles("*1", "*1").unwrap();

        let support = CpicDecisionSupport::with_common_guidelines();
        let decision = support.decide(&result, "nonexistent_drug");

        assert_eq!(decision.action, ClinicalAction::NoGuideline);
        assert!(decision.recommendation.is_none());
    }

    #[test]
    fn test_check_multiple_medications() {
        let classifier = PhenotypeClassifier::cyp2d6();
        let pm_result = classifier.classify_alleles("*4", "*4").unwrap();

        let support = CpicDecisionSupport::with_common_guidelines();
        let decisions = support.check_medications(&pm_result, &["codeine", "tramadol", "aspirin"]);

        assert_eq!(decisions.len(), 3);

        // Codeine and tramadol should have recommendations
        assert!(decisions[0].recommendation.is_some()); // codeine
        assert!(decisions[1].recommendation.is_some()); // tramadol
        assert!(decisions[2].recommendation.is_none()); // aspirin (no guideline)
    }
}
