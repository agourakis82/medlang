// Week 54: Drug-Drug Interaction (DDI) Checking System
//
// Implements multi-source DDI detection with mechanism-based classification
// and tiered alerting for clinical decision support.
//
// ## Severity Levels (ONC/FDA aligned)
//
// - Contraindicated: Never combine (hard stop)
// - Major: Avoid combination, high clinical significance
// - Moderate: Use with caution, may require adjustment
// - Minor: Monitor, low clinical significance
//
// ## Mechanism Classes
//
// - CYP Inhibition/Induction: Drug A affects metabolism of Drug B
// - Transporter Interactions: P-glycoprotein, OATP effects
// - Pharmacodynamic: Additive/synergistic effects (QT prolongation, bleeding)
// - Displacement: Protein binding displacement
//
// ## Evidence Levels
//
// - Level 1: Drug label + clinical studies
// - Level 2: Cohort/case studies, pharmacological inference
// - Level 3: Theoretical/in vitro only

use super::core::{ConceptRef, OntologyId};
use std::collections::{HashMap, HashSet};
use std::fmt;

/// DDI severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DdiSeverity {
    /// Monitor for effects
    Minor,
    /// Use with caution
    Moderate,
    /// Avoid combination if possible
    Major,
    /// Never combine (absolute contraindication)
    Contraindicated,
}

impl DdiSeverity {
    pub fn as_str(&self) -> &'static str {
        match self {
            DdiSeverity::Minor => "Minor",
            DdiSeverity::Moderate => "Moderate",
            DdiSeverity::Major => "Major",
            DdiSeverity::Contraindicated => "Contraindicated",
        }
    }

    /// Should this trigger an interruptive alert?
    pub fn is_interruptive(&self) -> bool {
        matches!(self, DdiSeverity::Contraindicated | DdiSeverity::Major)
    }

    /// Should this be a hard stop (require override)?
    pub fn is_hard_stop(&self) -> bool {
        matches!(self, DdiSeverity::Contraindicated)
    }

    /// Parse from string
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "minor" | "1" => Some(DdiSeverity::Minor),
            "moderate" | "2" => Some(DdiSeverity::Moderate),
            "major" | "3" | "severe" => Some(DdiSeverity::Major),
            "contraindicated" | "4" | "x" => Some(DdiSeverity::Contraindicated),
            _ => None,
        }
    }
}

impl fmt::Display for DdiSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Evidence level for DDI
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvidenceLevel {
    /// Theoretical/in vitro
    Theoretical,
    /// Case reports, pharmacological inference
    CaseReport,
    /// Cohort studies, case-control
    CohortStudy,
    /// Clinical trials, prospective studies
    ClinicalStudy,
    /// FDA label, CPIC guideline
    Established,
}

impl EvidenceLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            EvidenceLevel::Theoretical => "Theoretical",
            EvidenceLevel::CaseReport => "Case Report",
            EvidenceLevel::CohortStudy => "Cohort Study",
            EvidenceLevel::ClinicalStudy => "Clinical Study",
            EvidenceLevel::Established => "Established",
        }
    }

    /// Numeric score (higher = stronger evidence)
    pub fn score(&self) -> u8 {
        match self {
            EvidenceLevel::Theoretical => 1,
            EvidenceLevel::CaseReport => 2,
            EvidenceLevel::CohortStudy => 3,
            EvidenceLevel::ClinicalStudy => 4,
            EvidenceLevel::Established => 5,
        }
    }
}

impl fmt::Display for EvidenceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Mechanism type for DDI
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DdiMechanism {
    /// CYP enzyme inhibition (perpetrator inhibits object metabolism)
    CypInhibition {
        enzyme: CypEnzyme,
        potency: InhibitorPotency,
    },
    /// CYP enzyme induction (perpetrator induces object metabolism)
    CypInduction { enzyme: CypEnzyme },
    /// Object is CYP substrate affected by perpetrator
    CypSubstrate { enzyme: CypEnzyme },
    /// Transporter interaction
    TransporterInteraction {
        transporter: Transporter,
        effect: TransporterEffect,
    },
    /// Pharmacodynamic - additive/synergistic effects
    Pharmacodynamic { effect_type: PdEffectType },
    /// Protein binding displacement
    ProteinDisplacement,
    /// Renal elimination interaction
    RenalInteraction,
    /// GI absorption interaction
    AbsorptionInteraction,
    /// Unknown/unspecified mechanism
    Unknown,
}

impl DdiMechanism {
    pub fn description(&self) -> String {
        match self {
            DdiMechanism::CypInhibition { enzyme, potency } => {
                format!("{} {} inhibition", potency, enzyme)
            }
            DdiMechanism::CypInduction { enzyme } => {
                format!("{} induction", enzyme)
            }
            DdiMechanism::CypSubstrate { enzyme } => {
                format!("{} substrate", enzyme)
            }
            DdiMechanism::TransporterInteraction {
                transporter,
                effect,
            } => {
                format!("{} {} interaction", transporter, effect)
            }
            DdiMechanism::Pharmacodynamic { effect_type } => {
                format!("Pharmacodynamic: {}", effect_type)
            }
            DdiMechanism::ProteinDisplacement => "Protein binding displacement".to_string(),
            DdiMechanism::RenalInteraction => "Renal elimination interaction".to_string(),
            DdiMechanism::AbsorptionInteraction => "GI absorption interaction".to_string(),
            DdiMechanism::Unknown => "Unknown mechanism".to_string(),
        }
    }
}

/// CYP enzymes (major drug metabolizing enzymes)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CypEnzyme {
    Cyp1a2,
    Cyp2b6,
    Cyp2c8,
    Cyp2c9,
    Cyp2c19,
    Cyp2d6,
    Cyp2e1,
    Cyp3a4, // Metabolizes >50% of drugs
    Cyp3a5,
}

impl CypEnzyme {
    pub fn as_str(&self) -> &'static str {
        match self {
            CypEnzyme::Cyp1a2 => "CYP1A2",
            CypEnzyme::Cyp2b6 => "CYP2B6",
            CypEnzyme::Cyp2c8 => "CYP2C8",
            CypEnzyme::Cyp2c9 => "CYP2C9",
            CypEnzyme::Cyp2c19 => "CYP2C19",
            CypEnzyme::Cyp2d6 => "CYP2D6",
            CypEnzyme::Cyp2e1 => "CYP2E1",
            CypEnzyme::Cyp3a4 => "CYP3A4",
            CypEnzyme::Cyp3a5 => "CYP3A5",
        }
    }
}

impl fmt::Display for CypEnzyme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// CYP inhibitor potency (FDA classification)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InhibitorPotency {
    /// ≥5-fold AUC increase
    Strong,
    /// 2-5x AUC increase
    Moderate,
    /// 1.25-2x AUC increase
    Weak,
}

impl InhibitorPotency {
    pub fn as_str(&self) -> &'static str {
        match self {
            InhibitorPotency::Strong => "Strong",
            InhibitorPotency::Moderate => "Moderate",
            InhibitorPotency::Weak => "Weak",
        }
    }

    /// Expected fold-change in AUC
    pub fn expected_fold_change(&self) -> (f64, f64) {
        match self {
            InhibitorPotency::Strong => (5.0, f64::INFINITY),
            InhibitorPotency::Moderate => (2.0, 5.0),
            InhibitorPotency::Weak => (1.25, 2.0),
        }
    }
}

impl fmt::Display for InhibitorPotency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Drug transporters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Transporter {
    /// P-glycoprotein (MDR1)
    Pgp,
    /// OATP1B1 (hepatic uptake)
    Oatp1b1,
    /// OATP1B3
    Oatp1b3,
    /// OAT1 (renal)
    Oat1,
    /// OAT3
    Oat3,
    /// OCT2
    Oct2,
    /// BCRP (breast cancer resistance protein)
    Bcrp,
}

impl Transporter {
    pub fn as_str(&self) -> &'static str {
        match self {
            Transporter::Pgp => "P-gp",
            Transporter::Oatp1b1 => "OATP1B1",
            Transporter::Oatp1b3 => "OATP1B3",
            Transporter::Oat1 => "OAT1",
            Transporter::Oat3 => "OAT3",
            Transporter::Oct2 => "OCT2",
            Transporter::Bcrp => "BCRP",
        }
    }
}

impl fmt::Display for Transporter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Transporter interaction effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransporterEffect {
    Inhibition,
    Induction,
    Substrate,
}

impl fmt::Display for TransporterEffect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TransporterEffect::Inhibition => write!(f, "inhibition"),
            TransporterEffect::Induction => write!(f, "induction"),
            TransporterEffect::Substrate => write!(f, "substrate"),
        }
    }
}

/// Pharmacodynamic effect types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PdEffectType {
    /// QT interval prolongation risk
    QtProlongation,
    /// Bleeding risk (anticoagulants, antiplatelets)
    BleedingRisk,
    /// CNS depression (opioids, benzodiazepines)
    CnsDepression,
    /// Serotonin syndrome risk
    SerotoninSyndrome,
    /// Hypotension
    Hypotension,
    /// Hypertension
    Hypertension,
    /// Hyperkalemia
    Hyperkalemia,
    /// Hypoglycemia
    Hypoglycemia,
    /// Nephrotoxicity
    Nephrotoxicity,
    /// Hepatotoxicity
    Hepatotoxicity,
    /// Additive anticholinergic effects
    AnticholinergicBurden,
}

impl PdEffectType {
    pub fn as_str(&self) -> &'static str {
        match self {
            PdEffectType::QtProlongation => "QT prolongation",
            PdEffectType::BleedingRisk => "increased bleeding risk",
            PdEffectType::CnsDepression => "CNS depression",
            PdEffectType::SerotoninSyndrome => "serotonin syndrome risk",
            PdEffectType::Hypotension => "additive hypotension",
            PdEffectType::Hypertension => "hypertensive crisis risk",
            PdEffectType::Hyperkalemia => "hyperkalemia risk",
            PdEffectType::Hypoglycemia => "hypoglycemia risk",
            PdEffectType::Nephrotoxicity => "additive nephrotoxicity",
            PdEffectType::Hepatotoxicity => "additive hepatotoxicity",
            PdEffectType::AnticholinergicBurden => "anticholinergic burden",
        }
    }
}

impl fmt::Display for PdEffectType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// A drug-drug interaction record
#[derive(Debug, Clone)]
pub struct DrugInteraction {
    /// Drug A (perpetrator in PK interactions)
    pub drug_a: OntologyId,
    /// Drug B (object/victim in PK interactions)
    pub drug_b: OntologyId,
    /// Severity level
    pub severity: DdiSeverity,
    /// Evidence level
    pub evidence: EvidenceLevel,
    /// Mechanism(s)
    pub mechanisms: Vec<DdiMechanism>,
    /// Clinical description
    pub description: String,
    /// Management recommendation
    pub management: Option<String>,
    /// Is bidirectional (A affects B AND B affects A)
    pub bidirectional: bool,
    /// Source(s) of this interaction data
    pub sources: Vec<String>,
}

impl DrugInteraction {
    pub fn new(
        drug_a: OntologyId,
        drug_b: OntologyId,
        severity: DdiSeverity,
        description: &str,
    ) -> Self {
        DrugInteraction {
            drug_a,
            drug_b,
            severity,
            evidence: EvidenceLevel::Theoretical,
            mechanisms: Vec::new(),
            description: description.to_string(),
            management: None,
            bidirectional: false,
            sources: Vec::new(),
        }
    }

    pub fn with_evidence(mut self, evidence: EvidenceLevel) -> Self {
        self.evidence = evidence;
        self
    }

    pub fn with_mechanism(mut self, mechanism: DdiMechanism) -> Self {
        self.mechanisms.push(mechanism);
        self
    }

    pub fn with_management(mut self, management: &str) -> Self {
        self.management = Some(management.to_string());
        self
    }

    pub fn bidirectional(mut self) -> Self {
        self.bidirectional = true;
        self
    }

    pub fn with_source(mut self, source: &str) -> Self {
        self.sources.push(source.to_string());
        self
    }

    /// Check if this interaction involves a specific drug
    pub fn involves(&self, drug: &OntologyId) -> bool {
        self.drug_a == *drug || self.drug_b == *drug
    }

    /// Get the other drug in the interaction
    pub fn other_drug(&self, drug: &OntologyId) -> Option<&OntologyId> {
        if self.drug_a == *drug {
            Some(&self.drug_b)
        } else if self.drug_b == *drug {
            Some(&self.drug_a)
        } else {
            None
        }
    }
}

/// DDI alert for clinical decision support
#[derive(Debug, Clone)]
pub struct DdiAlert {
    /// The interaction
    pub interaction: DrugInteraction,
    /// Drug A display info
    pub drug_a_display: String,
    /// Drug B display info
    pub drug_b_display: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Has been acknowledged/overridden
    pub acknowledged: bool,
    /// Override reason (if acknowledged)
    pub override_reason: Option<String>,
}

impl DdiAlert {
    pub fn from_interaction(
        interaction: DrugInteraction,
        drug_a_display: &str,
        drug_b_display: &str,
    ) -> Self {
        let alert_type = match interaction.severity {
            DdiSeverity::Contraindicated => AlertType::HardStop,
            DdiSeverity::Major => AlertType::Interruptive,
            DdiSeverity::Moderate => AlertType::Passive,
            DdiSeverity::Minor => AlertType::Informational,
        };

        DdiAlert {
            interaction,
            drug_a_display: drug_a_display.to_string(),
            drug_b_display: drug_b_display.to_string(),
            alert_type,
            acknowledged: false,
            override_reason: None,
        }
    }

    /// Format alert message
    pub fn message(&self) -> String {
        format!(
            "[{}] {} + {}: {}",
            self.interaction.severity,
            self.drug_a_display,
            self.drug_b_display,
            self.interaction.description
        )
    }
}

/// Alert type for CDS
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertType {
    /// Hard stop - requires documented override
    HardStop,
    /// Interruptive - modal alert requiring acknowledgment
    Interruptive,
    /// Passive - displayed but doesn't interrupt workflow
    Passive,
    /// Informational - available on demand
    Informational,
}

/// DDI knowledge base / checker
pub struct DdiChecker {
    /// All known interactions
    interactions: Vec<DrugInteraction>,
    /// Index: drug -> interactions involving it
    drug_index: HashMap<String, Vec<usize>>,
    /// Index: drug pair -> interaction indices
    pair_index: HashMap<(String, String), Vec<usize>>,
    /// ONC high-priority DDI list
    high_priority_pairs: HashSet<(String, String)>,
}

impl DdiChecker {
    pub fn new() -> Self {
        DdiChecker {
            interactions: Vec::new(),
            drug_index: HashMap::new(),
            pair_index: HashMap::new(),
            high_priority_pairs: HashSet::new(),
        }
    }

    /// Add an interaction to the knowledge base
    pub fn add_interaction(&mut self, interaction: DrugInteraction) {
        let idx = self.interactions.len();

        // Index by drug A
        self.drug_index
            .entry(interaction.drug_a.code.clone())
            .or_default()
            .push(idx);

        // Index by drug B
        self.drug_index
            .entry(interaction.drug_b.code.clone())
            .or_default()
            .push(idx);

        // Index by pair (both orderings for bidirectional)
        let pair1 = (
            interaction.drug_a.code.clone(),
            interaction.drug_b.code.clone(),
        );
        let pair2 = (
            interaction.drug_b.code.clone(),
            interaction.drug_a.code.clone(),
        );

        self.pair_index.entry(pair1.clone()).or_default().push(idx);
        if interaction.bidirectional {
            self.pair_index.entry(pair2).or_default().push(idx);
        }

        // Track high-priority
        if interaction.severity >= DdiSeverity::Major {
            self.high_priority_pairs.insert(pair1);
        }

        self.interactions.push(interaction);
    }

    /// Check for interactions between two drugs
    pub fn check_pair(&self, drug_a: &OntologyId, drug_b: &OntologyId) -> Vec<&DrugInteraction> {
        let pair = (drug_a.code.clone(), drug_b.code.clone());

        self.pair_index
            .get(&pair)
            .map(|indices| indices.iter().map(|&i| &self.interactions[i]).collect())
            .unwrap_or_default()
    }

    /// Check a medication list for all interactions
    pub fn check_medication_list(&self, drugs: &[OntologyId]) -> Vec<&DrugInteraction> {
        let mut found = Vec::new();
        let mut seen_pairs = HashSet::new();

        for i in 0..drugs.len() {
            for j in (i + 1)..drugs.len() {
                let pair = if drugs[i].code < drugs[j].code {
                    (drugs[i].code.clone(), drugs[j].code.clone())
                } else {
                    (drugs[j].code.clone(), drugs[i].code.clone())
                };

                if seen_pairs.contains(&pair) {
                    continue;
                }
                seen_pairs.insert(pair);

                let interactions = self.check_pair(&drugs[i], &drugs[j]);
                found.extend(interactions);
            }
        }

        // Sort by severity (most severe first)
        found.sort_by(|a, b| b.severity.cmp(&a.severity));
        found
    }

    /// Get all interactions for a drug
    pub fn get_interactions_for(&self, drug: &OntologyId) -> Vec<&DrugInteraction> {
        self.drug_index
            .get(&drug.code)
            .map(|indices| indices.iter().map(|&i| &self.interactions[i]).collect())
            .unwrap_or_default()
    }

    /// Get high-priority (ONC list) interactions only
    pub fn check_high_priority(&self, drugs: &[OntologyId]) -> Vec<&DrugInteraction> {
        self.check_medication_list(drugs)
            .into_iter()
            .filter(|i| i.severity >= DdiSeverity::Major)
            .collect()
    }

    /// Generate alerts for a medication list
    pub fn generate_alerts(
        &self,
        drugs: &[(OntologyId, String)], // (id, display name)
    ) -> Vec<DdiAlert> {
        let ids: Vec<_> = drugs.iter().map(|(id, _)| id.clone()).collect();
        let name_map: HashMap<_, _> = drugs.iter().cloned().collect();

        self.check_medication_list(&ids)
            .into_iter()
            .map(|interaction| {
                let a_name = name_map
                    .get(&interaction.drug_a)
                    .map(|s| s.as_str())
                    .unwrap_or(&interaction.drug_a.code);
                let b_name = name_map
                    .get(&interaction.drug_b)
                    .map(|s| s.as_str())
                    .unwrap_or(&interaction.drug_b.code);

                DdiAlert::from_interaction(interaction.clone(), a_name, b_name)
            })
            .collect()
    }

    /// Number of interactions in knowledge base
    pub fn interaction_count(&self) -> usize {
        self.interactions.len()
    }
}

impl Default for DdiChecker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Example DDI Knowledge Base
// =============================================================================

/// Create example DDI checker with common interactions
pub fn example_ddi_checker() -> DdiChecker {
    let mut checker = DdiChecker::new();

    // Warfarin + NSAIDs (bleeding risk)
    checker.add_interaction(
        DrugInteraction::new(
            OntologyId::rxnorm("11289"), // Warfarin
            OntologyId::rxnorm("5640"),  // Ibuprofen
            DdiSeverity::Major,
            "Increased risk of bleeding due to antiplatelet effect of NSAIDs and anticoagulant effect of warfarin",
        )
        .with_evidence(EvidenceLevel::Established)
        .with_mechanism(DdiMechanism::Pharmacodynamic {
            effect_type: PdEffectType::BleedingRisk,
        })
        .with_management("Avoid combination if possible. If necessary, monitor INR closely and watch for signs of bleeding.")
        .bidirectional()
        .with_source("DrugBank")
    );

    // Warfarin + Fluconazole (CYP2C9 inhibition)
    checker.add_interaction(
        DrugInteraction::new(
            OntologyId::rxnorm("4083"),   // Fluconazole
            OntologyId::rxnorm("11289"),  // Warfarin
            DdiSeverity::Major,
            "Fluconazole inhibits CYP2C9, significantly increasing warfarin exposure and bleeding risk",
        )
        .with_evidence(EvidenceLevel::Established)
        .with_mechanism(DdiMechanism::CypInhibition {
            enzyme: CypEnzyme::Cyp2c9,
            potency: InhibitorPotency::Strong,
        })
        .with_management("Reduce warfarin dose by 25-50% when starting fluconazole. Monitor INR frequently.")
        .with_source("FDA Label")
    );

    // Simvastatin + Clarithromycin (CYP3A4 inhibition - contraindicated)
    checker.add_interaction(
        DrugInteraction::new(
            OntologyId::rxnorm("21212"),  // Clarithromycin
            OntologyId::rxnorm("36567"),  // Simvastatin
            DdiSeverity::Contraindicated,
            "Clarithromycin is a strong CYP3A4 inhibitor, greatly increasing simvastatin levels and risk of rhabdomyolysis",
        )
        .with_evidence(EvidenceLevel::Established)
        .with_mechanism(DdiMechanism::CypInhibition {
            enzyme: CypEnzyme::Cyp3a4,
            potency: InhibitorPotency::Strong,
        })
        .with_management("Do not use together. Consider azithromycin as alternative macrolide or use pravastatin/rosuvastatin.")
        .with_source("FDA Label")
    );

    // Metformin + Contrast dye (lactic acidosis risk)
    checker.add_interaction(
        DrugInteraction::new(
            OntologyId::rxnorm("6809"),   // Metformin
            OntologyId::rxnorm("100000"), // Iodinated contrast (placeholder)
            DdiSeverity::Major,
            "Risk of contrast-induced nephropathy leading to metformin accumulation and lactic acidosis",
        )
        .with_evidence(EvidenceLevel::Established)
        .with_mechanism(DdiMechanism::RenalInteraction)
        .with_management("Hold metformin before contrast procedures. Resume 48 hours after if renal function stable.")
        .with_source("ACR Guidelines")
    );

    // SSRIs + MAOIs (serotonin syndrome - contraindicated)
    checker.add_interaction(
        DrugInteraction::new(
            OntologyId::rxnorm("4493"),  // Fluoxetine (SSRI)
            OntologyId::rxnorm("6011"),  // Phenelzine (MAOI)
            DdiSeverity::Contraindicated,
            "Risk of fatal serotonin syndrome. 14-day washout required between agents.",
        )
        .with_evidence(EvidenceLevel::Established)
        .with_mechanism(DdiMechanism::Pharmacodynamic {
            effect_type: PdEffectType::SerotoninSyndrome,
        })
        .with_management("Never combine. Wait at least 14 days after stopping MAOI, 5 weeks after stopping fluoxetine.")
        .bidirectional()
        .with_source("DrugBank")
    );

    // QT prolonging drugs
    checker.add_interaction(
        DrugInteraction::new(
            OntologyId::rxnorm("6960"),  // Haloperidol
            OntologyId::rxnorm("18631"), // Ondansetron
            DdiSeverity::Major,
            "Both drugs prolong QT interval. Combination increases risk of torsades de pointes.",
        )
        .with_evidence(EvidenceLevel::ClinicalStudy)
        .with_mechanism(DdiMechanism::Pharmacodynamic {
            effect_type: PdEffectType::QtProlongation,
        })
        .with_management(
            "Monitor ECG. Consider alternative antiemetic. Correct electrolyte abnormalities.",
        )
        .bidirectional()
        .with_source("CredibleMeds"),
    );

    checker
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_severity_ordering() {
        assert!(DdiSeverity::Contraindicated > DdiSeverity::Major);
        assert!(DdiSeverity::Major > DdiSeverity::Moderate);
        assert!(DdiSeverity::Moderate > DdiSeverity::Minor);
    }

    #[test]
    fn test_severity_alert_type() {
        assert!(DdiSeverity::Contraindicated.is_hard_stop());
        assert!(DdiSeverity::Contraindicated.is_interruptive());
        assert!(DdiSeverity::Major.is_interruptive());
        assert!(!DdiSeverity::Moderate.is_interruptive());
    }

    #[test]
    fn test_cyp_mechanism() {
        let mech = DdiMechanism::CypInhibition {
            enzyme: CypEnzyme::Cyp3a4,
            potency: InhibitorPotency::Strong,
        };
        let desc = mech.description();
        assert!(desc.contains("CYP3A4"));
        assert!(desc.contains("Strong"));
    }

    #[test]
    fn test_interaction_creation() {
        let interaction = DrugInteraction::new(
            OntologyId::rxnorm("11289"),
            OntologyId::rxnorm("5640"),
            DdiSeverity::Major,
            "Test interaction",
        )
        .with_evidence(EvidenceLevel::Established)
        .with_mechanism(DdiMechanism::Pharmacodynamic {
            effect_type: PdEffectType::BleedingRisk,
        })
        .bidirectional();

        assert!(interaction.bidirectional);
        assert_eq!(interaction.severity, DdiSeverity::Major);
        assert_eq!(interaction.mechanisms.len(), 1);
    }

    #[test]
    fn test_checker_add_and_find() {
        let mut checker = DdiChecker::new();

        checker.add_interaction(DrugInteraction::new(
            OntologyId::rxnorm("A"),
            OntologyId::rxnorm("B"),
            DdiSeverity::Major,
            "Test",
        ));

        let found = checker.check_pair(&OntologyId::rxnorm("A"), &OntologyId::rxnorm("B"));
        assert_eq!(found.len(), 1);

        let not_found = checker.check_pair(&OntologyId::rxnorm("A"), &OntologyId::rxnorm("C"));
        assert!(not_found.is_empty());
    }

    #[test]
    fn test_medication_list_check() {
        let checker = example_ddi_checker();

        let meds = vec![
            OntologyId::rxnorm("11289"), // Warfarin
            OntologyId::rxnorm("5640"),  // Ibuprofen
            OntologyId::rxnorm("6809"),  // Metformin
        ];

        let interactions = checker.check_medication_list(&meds);
        assert!(!interactions.is_empty());

        // Should find warfarin + ibuprofen interaction
        assert!(interactions.iter().any(|i| {
            i.involves(&OntologyId::rxnorm("11289")) && i.involves(&OntologyId::rxnorm("5640"))
        }));
    }

    #[test]
    fn test_contraindicated_detection() {
        let checker = example_ddi_checker();

        let meds = vec![
            OntologyId::rxnorm("21212"), // Clarithromycin
            OntologyId::rxnorm("36567"), // Simvastatin
        ];

        let interactions = checker.check_medication_list(&meds);
        assert!(!interactions.is_empty());
        assert_eq!(interactions[0].severity, DdiSeverity::Contraindicated);
    }

    #[test]
    fn test_alert_generation() {
        let checker = example_ddi_checker();

        let meds = vec![
            (OntologyId::rxnorm("11289"), "Warfarin".to_string()),
            (OntologyId::rxnorm("5640"), "Ibuprofen".to_string()),
        ];

        let alerts = checker.generate_alerts(&meds);
        assert!(!alerts.is_empty());

        let alert = &alerts[0];
        assert!(alert.message().contains("Warfarin"));
        assert!(alert.message().contains("Ibuprofen"));
    }

    #[test]
    fn test_high_priority_filter() {
        let checker = example_ddi_checker();

        let meds = vec![
            OntologyId::rxnorm("4493"), // Fluoxetine
            OntologyId::rxnorm("6011"), // Phenelzine
        ];

        let high_priority = checker.check_high_priority(&meds);
        assert!(!high_priority.is_empty());
        assert!(high_priority
            .iter()
            .all(|i| i.severity >= DdiSeverity::Major));
    }
}
