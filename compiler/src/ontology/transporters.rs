//! Drug Transporter Interaction Module
//!
//! Implements comprehensive drug transporter modeling including:
//! - P-glycoprotein (P-gp/MDR1/ABCB1)
//! - OATP transporters (SLCO1B1, SLCO1B3)
//! - OAT/OCT transporters (renal)
//! - BCRP (ABCG2)
//!
//! ## Clinical Significance
//!
//! Drug transporters affect:
//! - Intestinal absorption (P-gp limits oral bioavailability)
//! - Hepatic uptake (OATP1B1 important for statin disposition)
//! - Renal elimination (OAT, OCT)
//! - Blood-brain barrier penetration (P-gp, BCRP)
//! - Drug distribution and tissue exposure
//!
//! ## FDA Guidance
//!
//! Per FDA DDI guidance, clinical studies recommended when drug is:
//! - P-gp inhibitor with [I1]/IC50 or [I2]/IC50 ≥ 0.1
//! - OATP1B1/1B3 inhibitor with R value ≥ 1.1
//! - Renal transporter inhibitor meeting threshold criteria

use super::ddi::{DdiSeverity, Transporter, TransporterEffect};
use std::collections::HashMap;

/// Transporter classification by location/function
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransporterClass {
    /// Efflux transporters (ABC family)
    Efflux,
    /// Uptake transporters (SLC family)
    Uptake,
    /// Bidirectional
    Bidirectional,
}

/// Extended transporter enumeration with gene names
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransporterType {
    // Efflux transporters
    /// P-glycoprotein (MDR1, ABCB1) - intestinal, hepatic, renal, BBB
    Pgp,
    /// Breast Cancer Resistance Protein (ABCG2)
    Bcrp,
    /// Multidrug Resistance Protein 2 (ABCC2)
    Mrp2,
    /// Multidrug Resistance Protein 4 (ABCC4)
    Mrp4,
    /// Bile Salt Export Pump (ABCB11)
    Bsep,

    // Hepatic uptake transporters
    /// OATP1B1 (SLCO1B1) - hepatic uptake
    Oatp1b1,
    /// OATP1B3 (SLCO1B3) - hepatic uptake
    Oatp1b3,
    /// OATP2B1 (SLCO2B1)
    Oatp2b1,

    // Renal transporters
    /// OAT1 (SLC22A6) - renal basolateral uptake
    Oat1,
    /// OAT3 (SLC22A8) - renal basolateral uptake
    Oat3,
    /// OCT2 (SLC22A2) - renal basolateral uptake
    Oct2,
    /// MATE1 (SLC47A1) - renal apical efflux
    Mate1,
    /// MATE2-K (SLC47A2) - renal apical efflux
    Mate2k,

    // Intestinal uptake
    /// PEPT1 (SLC15A1) - intestinal peptide transporter
    Pept1,
}

impl TransporterType {
    pub fn gene_name(&self) -> &'static str {
        match self {
            Self::Pgp => "ABCB1",
            Self::Bcrp => "ABCG2",
            Self::Mrp2 => "ABCC2",
            Self::Mrp4 => "ABCC4",
            Self::Bsep => "ABCB11",
            Self::Oatp1b1 => "SLCO1B1",
            Self::Oatp1b3 => "SLCO1B3",
            Self::Oatp2b1 => "SLCO2B1",
            Self::Oat1 => "SLC22A6",
            Self::Oat3 => "SLC22A8",
            Self::Oct2 => "SLC22A2",
            Self::Mate1 => "SLC47A1",
            Self::Mate2k => "SLC47A2",
            Self::Pept1 => "SLC15A1",
        }
    }

    pub fn common_name(&self) -> &'static str {
        match self {
            Self::Pgp => "P-glycoprotein",
            Self::Bcrp => "BCRP",
            Self::Mrp2 => "MRP2",
            Self::Mrp4 => "MRP4",
            Self::Bsep => "BSEP",
            Self::Oatp1b1 => "OATP1B1",
            Self::Oatp1b3 => "OATP1B3",
            Self::Oatp2b1 => "OATP2B1",
            Self::Oat1 => "OAT1",
            Self::Oat3 => "OAT3",
            Self::Oct2 => "OCT2",
            Self::Mate1 => "MATE1",
            Self::Mate2k => "MATE2-K",
            Self::Pept1 => "PEPT1",
        }
    }

    pub fn class(&self) -> TransporterClass {
        match self {
            Self::Pgp
            | Self::Bcrp
            | Self::Mrp2
            | Self::Mrp4
            | Self::Bsep
            | Self::Mate1
            | Self::Mate2k => TransporterClass::Efflux,
            Self::Oatp1b1
            | Self::Oatp1b3
            | Self::Oatp2b1
            | Self::Oat1
            | Self::Oat3
            | Self::Oct2
            | Self::Pept1 => TransporterClass::Uptake,
        }
    }

    /// Primary tissue location(s)
    pub fn tissue_expression(&self) -> Vec<&'static str> {
        match self {
            Self::Pgp => vec!["intestine", "liver", "kidney", "BBB", "placenta"],
            Self::Bcrp => vec!["intestine", "liver", "kidney", "BBB", "placenta"],
            Self::Mrp2 => vec!["liver", "kidney", "intestine"],
            Self::Mrp4 => vec!["kidney", "liver", "blood cells"],
            Self::Bsep => vec!["liver"],
            Self::Oatp1b1 => vec!["liver"],
            Self::Oatp1b3 => vec!["liver"],
            Self::Oatp2b1 => vec!["liver", "intestine"],
            Self::Oat1 => vec!["kidney"],
            Self::Oat3 => vec!["kidney"],
            Self::Oct2 => vec!["kidney"],
            Self::Mate1 => vec!["kidney", "liver"],
            Self::Mate2k => vec!["kidney"],
            Self::Pept1 => vec!["intestine", "kidney"],
        }
    }

    /// Convert from basic Transporter enum
    pub fn from_basic(t: Transporter) -> Self {
        match t {
            Transporter::Pgp => Self::Pgp,
            Transporter::Oatp1b1 => Self::Oatp1b1,
            Transporter::Oatp1b3 => Self::Oatp1b3,
            Transporter::Oat1 => Self::Oat1,
            Transporter::Oat3 => Self::Oat3,
            Transporter::Oct2 => Self::Oct2,
            Transporter::Bcrp => Self::Bcrp,
        }
    }
}

impl std::fmt::Display for TransporterType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.common_name())
    }
}

/// Inhibitor potency for transporters
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TransporterInhibitorPotency {
    /// Clinically significant inhibition demonstrated
    Strong,
    /// Moderate inhibition, may be clinically relevant
    Moderate,
    /// Weak inhibition, unlikely to be clinically significant
    Weak,
}

impl TransporterInhibitorPotency {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Strong => "Strong",
            Self::Moderate => "Moderate",
            Self::Weak => "Weak",
        }
    }
}

impl std::fmt::Display for TransporterInhibitorPotency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Transporter role for a drug
#[derive(Debug, Clone, PartialEq)]
pub enum TransporterRole {
    /// Drug is transported by this transporter
    Substrate {
        transporter: TransporterType,
        /// Clinical significance of this pathway
        significance: SubstrateSignificance,
    },
    /// Drug inhibits this transporter
    Inhibitor {
        transporter: TransporterType,
        potency: TransporterInhibitorPotency,
        /// IC50 in μM if known
        ic50: Option<f64>,
    },
    /// Drug induces this transporter
    Inducer { transporter: TransporterType },
}

impl TransporterRole {
    pub fn transporter(&self) -> TransporterType {
        match self {
            Self::Substrate { transporter, .. } => *transporter,
            Self::Inhibitor { transporter, .. } => *transporter,
            Self::Inducer { transporter } => *transporter,
        }
    }
}

/// Substrate significance classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateSignificance {
    /// Major route of elimination/distribution
    Major,
    /// Contributes to disposition but not sole pathway
    Moderate,
    /// Minor contribution
    Minor,
}

/// Drug transporter profile
#[derive(Debug, Clone)]
pub struct TransporterDrugProfile {
    /// Drug identifier
    pub drug_id: String,
    /// Drug name
    pub drug_name: String,
    /// All transporter roles
    pub roles: Vec<TransporterRole>,
    /// Clinical notes
    pub notes: Option<String>,
}

impl TransporterDrugProfile {
    pub fn new(drug_id: &str, drug_name: &str) -> Self {
        Self {
            drug_id: drug_id.to_string(),
            drug_name: drug_name.to_string(),
            roles: Vec::new(),
            notes: None,
        }
    }

    pub fn with_role(mut self, role: TransporterRole) -> Self {
        self.roles.push(role);
        self
    }

    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = Some(notes.to_string());
        self
    }

    /// Get transporters this drug is a substrate of
    pub fn substrate_of(&self) -> Vec<(TransporterType, SubstrateSignificance)> {
        self.roles
            .iter()
            .filter_map(|r| match r {
                TransporterRole::Substrate {
                    transporter,
                    significance,
                } => Some((*transporter, *significance)),
                _ => None,
            })
            .collect()
    }

    /// Get transporters this drug inhibits
    pub fn inhibits(&self) -> Vec<(TransporterType, TransporterInhibitorPotency)> {
        self.roles
            .iter()
            .filter_map(|r| match r {
                TransporterRole::Inhibitor {
                    transporter,
                    potency,
                    ..
                } => Some((*transporter, *potency)),
                _ => None,
            })
            .collect()
    }

    /// Check if drug is P-gp substrate
    pub fn is_pgp_substrate(&self) -> bool {
        self.roles.iter().any(|r| {
            matches!(
                r,
                TransporterRole::Substrate {
                    transporter: TransporterType::Pgp,
                    ..
                }
            )
        })
}

    /// Check if drug is OATP1B1 substrate
    pub fn is_oatp1b1_substrate(&self) -> bool {
        self.roles.iter().any(|r| {
            matches!(
                r,
                TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b1,
                    ..
                }
            )
        })
    }
}

/// Transporter interaction prediction
#[derive(Debug, Clone)]
pub struct TransporterInteractionPrediction {
    /// Perpetrator drug
    pub perpetrator: String,
    /// Victim drug
    pub victim: String,
    /// Affected transporter
    pub transporter: TransporterType,
    /// Type of interaction
    pub interaction_type: TransporterInteractionType,
    /// Predicted severity
    pub predicted_severity: DdiSeverity,
    /// Expected exposure change
    pub exposure_change: ExposureChange,
    /// Clinical recommendation
    pub recommendation: String,
}

/// Type of transporter interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransporterInteractionType {
    /// Inhibitor increases substrate exposure
    Inhibition,
    /// Inducer decreases substrate exposure
    Induction,
}

/// Expected change in drug exposure
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExposureChange {
    /// Large increase (>3-fold)
    LargeIncrease,
    /// Moderate increase (1.5-3 fold)
    ModerateIncrease,
    /// Small increase (1.25-1.5 fold)
    SmallIncrease,
    /// Large decrease (<30% of normal)
    LargeDecrease,
    /// Moderate decrease (30-60% of normal)
    ModerateDecrease,
    /// Small decrease (60-80% of normal)
    SmallDecrease,
    /// Unknown
    Unknown,
}

impl ExposureChange {
    pub fn description(&self) -> &'static str {
        match self {
            Self::LargeIncrease => ">3-fold increase in exposure",
            Self::ModerateIncrease => "1.5-3 fold increase in exposure",
            Self::SmallIncrease => "1.25-1.5 fold increase in exposure",
            Self::LargeDecrease => ">70% decrease in exposure",
            Self::ModerateDecrease => "40-70% decrease in exposure",
            Self::SmallDecrease => "20-40% decrease in exposure",
            Self::Unknown => "Unknown exposure change",
        }
    }
}

/// Transporter drug database
#[derive(Debug, Default)]
pub struct TransporterDatabase {
    profiles: HashMap<String, TransporterDrugProfile>,
    inhibitors_by_transporter: HashMap<TransporterType, Vec<String>>,
    substrates_by_transporter: HashMap<TransporterType, Vec<String>>,
}

impl TransporterDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create database with clinically relevant drugs
    pub fn with_clinical_drugs() -> Self {
        let mut db = Self::new();

        // P-gp inhibitors
        db.add_profile(
            TransporterDrugProfile::new("134748", "Ritonavir")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Pgp,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(2.0),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b1,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: None,
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Bcrp,
                    potency: TransporterInhibitorPotency::Moderate,
                    ic50: None,
                }),
        );

        db.add_profile(
            TransporterDrugProfile::new("114984", "Cyclosporine")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Pgp,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(1.7),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b1,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(0.3),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b3,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(0.07),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Bcrp,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: None,
                })
                .with_notes("Potent multi-transporter inhibitor"),
        );

        db.add_profile(
            TransporterDrugProfile::new("33738", "Verapamil")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Pgp,
                    potency: TransporterInhibitorPotency::Moderate,
                    ic50: Some(8.0),
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Pgp,
                    significance: SubstrateSignificance::Moderate,
                }),
        );

        db.add_profile(TransporterDrugProfile::new("27316", "Quinidine").with_role(
            TransporterRole::Inhibitor {
                transporter: TransporterType::Pgp,
                potency: TransporterInhibitorPotency::Strong,
                ic50: Some(3.2),
            },
        ));

        db.add_profile(
            TransporterDrugProfile::new("19484", "Dronedarone")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Pgp,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: None,
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b1,
                    potency: TransporterInhibitorPotency::Moderate,
                    ic50: None,
                }),
        );

        // OATP inhibitors
        db.add_profile(
            TransporterDrugProfile::new("301542", "Gemfibrozil")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b1,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(30.0),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b3,
                    potency: TransporterInhibitorPotency::Moderate,
                    ic50: None,
                })
                .with_notes("Contraindicated with simvastatin due to OATP1B1 + glucuronide CYP2C8 inhibition")
        );

        db.add_profile(
            TransporterDrugProfile::new("8183", "Rifampin")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b1,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(0.5),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b3,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(0.6),
                })
                .with_role(TransporterRole::Inducer {
                    transporter: TransporterType::Pgp,
                })
                .with_notes("Acute inhibition but chronic induction of transporters"),
        );

        // P-gp substrates
        db.add_profile(
            TransporterDrugProfile::new("32968", "Digoxin")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Pgp,
                    significance: SubstrateSignificance::Major,
                })
                .with_notes("Narrow therapeutic index - P-gp critical for clearance"),
        );

        db.add_profile(
            TransporterDrugProfile::new("114970", "Dabigatran")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Pgp,
                    significance: SubstrateSignificance::Major,
                })
                .with_notes("P-gp inhibitors can significantly increase dabigatran exposure"),
        );

        db.add_profile(
            TransporterDrugProfile::new("73494", "Fexofenadine")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Pgp,
                    significance: SubstrateSignificance::Major,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b1,
                    significance: SubstrateSignificance::Moderate,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b3,
                    significance: SubstrateSignificance::Moderate,
                })
                .with_notes("FDA recommended P-gp probe substrate"),
        );

        db.add_profile(
            TransporterDrugProfile::new("221118", "Aliskiren")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Pgp,
                    significance: SubstrateSignificance::Major,
                })
                .with_notes("Bioavailability significantly affected by P-gp"),
        );

        db.add_profile(
            TransporterDrugProfile::new("72625", "Colchicine")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Pgp,
                    significance: SubstrateSignificance::Major,
                })
                .with_notes("Fatal toxicity reported with P-gp inhibitors in renal impairment"),
        );

        // OATP substrates (statins)
        db.add_profile(
            TransporterDrugProfile::new("83367", "Atorvastatin")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b1,
                    significance: SubstrateSignificance::Major,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b3,
                    significance: SubstrateSignificance::Moderate,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Bcrp,
                    significance: SubstrateSignificance::Moderate,
                }),
        );

        db.add_profile(
            TransporterDrugProfile::new("83368", "Rosuvastatin")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b1,
                    significance: SubstrateSignificance::Major,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b3,
                    significance: SubstrateSignificance::Major,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Bcrp,
                    significance: SubstrateSignificance::Major,
                })
                .with_notes("Highly dependent on OATP transporters for hepatic uptake"),
        );

        db.add_profile(
            TransporterDrugProfile::new("42463", "Pravastatin")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b1,
                    significance: SubstrateSignificance::Major,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b3,
                    significance: SubstrateSignificance::Moderate,
                })
                .with_notes("FDA recommended OATP1B probe substrate"),
        );

        db.add_profile(
            TransporterDrugProfile::new("36567", "Simvastatin")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oatp1b1,
                    significance: SubstrateSignificance::Moderate,
                })
                .with_notes("Simvastatin acid is OATP1B1 substrate"),
        );

        // Renal transporter substrates
        db.add_profile(
            TransporterDrugProfile::new("6809", "Metformin")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Oct2,
                    significance: SubstrateSignificance::Major,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Mate1,
                    significance: SubstrateSignificance::Major,
                })
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Mate2k,
                    significance: SubstrateSignificance::Major,
                })
                .with_notes("Renal elimination via OCT2/MATE pathway"),
        );

        // OCT2 inhibitors
        db.add_profile(
            TransporterDrugProfile::new("18631", "Cimetidine")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oct2,
                    potency: TransporterInhibitorPotency::Moderate,
                    ic50: Some(100.0),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Mate1,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(1.1),
                }),
        );

        db.add_profile(
            TransporterDrugProfile::new("134517", "Dolutegravir")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oct2,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(1.9),
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Mate1,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: Some(6.3),
                })
                .with_notes("Raises serum creatinine via inhibition of creatinine secretion"),
        );

        // BCRP substrates/inhibitors
        db.add_profile(
            TransporterDrugProfile::new("26225", "Sulfasalazine")
                .with_role(TransporterRole::Substrate {
                    transporter: TransporterType::Bcrp,
                    significance: SubstrateSignificance::Major,
                })
                .with_notes("FDA recommended BCRP probe substrate"),
        );

        db.add_profile(
            TransporterDrugProfile::new("349308", "Eltrombopag")
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Bcrp,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: None,
                })
                .with_role(TransporterRole::Inhibitor {
                    transporter: TransporterType::Oatp1b1,
                    potency: TransporterInhibitorPotency::Strong,
                    ic50: None,
                }),
        );

        db
    }

    /// Add a profile to the database
    pub fn add_profile(&mut self, profile: TransporterDrugProfile) {
        let drug_id = profile.drug_id.clone();

        for role in &profile.roles {
            let transporter = role.transporter();
            match role {
                TransporterRole::Inhibitor { .. } | TransporterRole::Inducer { .. } => {
                    self.inhibitors_by_transporter
                        .entry(transporter)
                        .or_default()
                        .push(drug_id.clone());
                }
                TransporterRole::Substrate { .. } => {
                    self.substrates_by_transporter
                        .entry(transporter)
                        .or_default()
                        .push(drug_id.clone());
                }
            }
        }

        self.profiles.insert(drug_id, profile);
    }

    /// Get a drug profile
    pub fn get_profile(&self, drug_id: &str) -> Option<&TransporterDrugProfile> {
        self.profiles.get(drug_id)
    }

    /// Get P-gp inhibitors
    pub fn pgp_inhibitors(&self) -> Vec<&TransporterDrugProfile> {
        self.inhibitors_by_transporter
            .get(&TransporterType::Pgp)
            .map(|ids| ids.iter().filter_map(|id| self.profiles.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get P-gp substrates
    pub fn pgp_substrates(&self) -> Vec<&TransporterDrugProfile> {
        self.substrates_by_transporter
            .get(&TransporterType::Pgp)
            .map(|ids| ids.iter().filter_map(|id| self.profiles.get(id)).collect())
            .unwrap_or_default()
    }

    /// Predict transporter-mediated interactions
    pub fn predict_interactions(
        &self,
        drug_a: &str,
        drug_b: &str,
    ) -> Vec<TransporterInteractionPrediction> {
        let mut predictions = Vec::new();

        let profile_a = match self.profiles.get(drug_a) {
            Some(p) => p,
            None => return predictions,
        };

        let profile_b = match self.profiles.get(drug_b) {
            Some(p) => p,
            None => return predictions,
        };

        // Check: A inhibits transporter that B is substrate of
        for (transporter, potency) in profile_a.inhibits() {
            for (sub_transporter, significance) in profile_b.substrate_of() {
                if transporter == sub_transporter {
                    let (severity, exposure) = predict_transporter_interaction_severity(
                        potency,
                        significance,
                        transporter,
);

                    predictions.push(TransporterInteractionPrediction {
                        perpetrator: profile_a.drug_name.clone(),
                        victim: profile_b.drug_name.clone(),
                        transporter,
                        interaction_type: TransporterInteractionType::Inhibition,
                        predicted_severity: severity,
                        exposure_change: exposure,
                        recommendation: transporter_inhibition_recommendation(
                            transporter,
                            potency,
                            &profile_b.drug_name,
                        ),
                    });
                }
            }
        }

        // Check: B inhibits transporter that A is substrate of
        for (transporter, potency) in profile_b.inhibits() {
            for (sub_transporter, significance) in profile_a.substrate_of() {
                if transporter == sub_transporter {
                    let (severity, exposure) = predict_transporter_interaction_severity(
                        potency,
                        significance,
                        transporter,
);

                    predictions.push(TransporterInteractionPrediction {
                        perpetrator: profile_b.drug_name.clone(),
                        victim: profile_a.drug_name.clone(),
                        transporter,
                        interaction_type: TransporterInteractionType::Inhibition,
                        predicted_severity: severity,
                        exposure_change: exposure,
                        recommendation: transporter_inhibition_recommendation(
                            transporter,
                            potency,
                            &profile_a.drug_name,
                        ),
                    });
                }
            }
        }

        predictions
    }

    /// Check a medication list for interactions
    pub fn check_medication_list(
        &self,
        drug_ids: &[&str],
    ) -> Vec<TransporterInteractionPrediction> {
        let mut predictions = Vec::new();

        for i in 0..drug_ids.len() {
            for j in (i + 1)..drug_ids.len() {
                predictions.extend(self.predict_interactions(drug_ids[i], drug_ids[j]));
            }
        }

        predictions.sort_by(|a, b| b.predicted_severity.cmp(&a.predicted_severity));
        predictions
    }
}

/// Predict severity of transporter interaction
fn predict_transporter_interaction_severity(
    inhibitor_potency: TransporterInhibitorPotency,
    substrate_significance: SubstrateSignificance,
    transporter: TransporterType,
) -> (DdiSeverity, ExposureChange) {
    // P-gp and OATP interactions can be clinically significant
    let is_critical_transporter = matches!(
        transporter,
        TransporterType::Pgp | TransporterType::Oatp1b1 | TransporterType::Oatp1b3
    );

    match (
        inhibitor_potency,
        substrate_significance,
        is_critical_transporter,
    ) {
        (TransporterInhibitorPotency::Strong, SubstrateSignificance::Major, true) => {
            (DdiSeverity::Major, ExposureChange::LargeIncrease)
        }
        (TransporterInhibitorPotency::Strong, SubstrateSignificance::Major, false) => {
            (DdiSeverity::Moderate, ExposureChange::ModerateIncrease)
        }
        (TransporterInhibitorPotency::Strong, SubstrateSignificance::Moderate, _) => {
            (DdiSeverity::Moderate, ExposureChange::ModerateIncrease)
        }
        (TransporterInhibitorPotency::Moderate, SubstrateSignificance::Major, true) => {
            (DdiSeverity::Moderate, ExposureChange::ModerateIncrease)
        }
        (TransporterInhibitorPotency::Moderate, SubstrateSignificance::Major, false) => {
            (DdiSeverity::Minor, ExposureChange::SmallIncrease)
        }
        (TransporterInhibitorPotency::Moderate, SubstrateSignificance::Moderate, _) => {
            (DdiSeverity::Minor, ExposureChange::SmallIncrease)
        }
        _ => (DdiSeverity::Minor, ExposureChange::SmallIncrease),
    }
}

/// Generate recommendation for transporter interaction
fn transporter_inhibition_recommendation(
    transporter: TransporterType,
    potency: TransporterInhibitorPotency,
    victim_name: &str,
) -> String {
    match (transporter, potency) {
        (TransporterType::Pgp, TransporterInhibitorPotency::Strong) => {
            format!("Strong P-gp inhibition may significantly increase {} exposure. Consider dose reduction or avoid combination.", victim_name)
        }
        (
            TransporterType::Oatp1b1 | TransporterType::Oatp1b3,
            TransporterInhibitorPotency::Strong,
        ) => {
            format!(
                "OATP inhibition increases hepatic {} exposure. Consider lower {} dose.",
                victim_name, victim_name
            )
        }
        (TransporterType::Oct2 | TransporterType::Mate1, _) => {
            format!("Renal transporter inhibition may increase {} exposure. Monitor for adverse effects.", victim_name)
        }
        _ => {
            format!("Monitor for increased {} effects.", victim_name)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transporter_type_info() {
        assert_eq!(TransporterType::Pgp.gene_name(), "ABCB1");
        assert_eq!(TransporterType::Oatp1b1.gene_name(), "SLCO1B1");
        assert_eq!(TransporterType::Pgp.class(), TransporterClass::Efflux);
        assert_eq!(TransporterType::Oatp1b1.class(), TransporterClass::Uptake);
    }

    #[test]
    fn test_database_creation() {
        let db = TransporterDatabase::with_clinical_drugs();
        assert!(db.get_profile("134748").is_some()); // Ritonavir
        assert!(db.get_profile("32968").is_some()); // Digoxin
    }

    #[test]
    fn test_pgp_inhibitors() {
        let db = TransporterDatabase::with_clinical_drugs();
        let inhibitors = db.pgp_inhibitors();

        assert!(!inhibitors.is_empty());
        assert!(inhibitors.iter().any(|p| p.drug_name == "Ritonavir"));
        assert!(inhibitors.iter().any(|p| p.drug_name == "Cyclosporine"));
    }

    #[test]
    fn test_pgp_substrates() {
        let db = TransporterDatabase::with_clinical_drugs();
        let substrates = db.pgp_substrates();

        assert!(!substrates.is_empty());
        assert!(substrates.iter().any(|p| p.drug_name == "Digoxin"));
        assert!(substrates.iter().any(|p| p.drug_name == "Dabigatran"));
    }

    #[test]
    fn test_pgp_interaction_prediction() {
        let db = TransporterDatabase::with_clinical_drugs();

        // Ritonavir (P-gp inhibitor) + Digoxin (P-gp substrate)
        let predictions = db.predict_interactions("134748", "32968");

        assert!(!predictions.is_empty());
        let pred = predictions
            .iter()
            .find(|p| p.transporter == TransporterType::Pgp)
            .unwrap();

        assert_eq!(pred.perpetrator, "Ritonavir");
        assert_eq!(pred.victim, "Digoxin");
        assert!(pred.predicted_severity >= DdiSeverity::Moderate);
    }

    #[test]
    fn test_oatp_statin_interaction() {
        let db = TransporterDatabase::with_clinical_drugs();

        // Cyclosporine (OATP inhibitor) + Rosuvastatin (OATP substrate)
        let predictions = db.predict_interactions("114984", "83368");

        assert!(!predictions.is_empty());
        assert!(predictions
            .iter()
            .any(|p| p.transporter == TransporterType::Oatp1b1
                || p.transporter == TransporterType::Oatp1b3));
    }

    #[test]
    fn test_renal_transporter_interaction() {
        let db = TransporterDatabase::with_clinical_drugs();

        // Dolutegravir (OCT2 inhibitor) + Metformin (OCT2 substrate)
        let predictions = db.predict_interactions("134517", "6809");

        assert!(!predictions.is_empty());
        assert!(predictions
            .iter()
            .any(|p| p.transporter == TransporterType::Oct2
                || p.transporter == TransporterType::Mate1));
    }

    #[test]
    fn test_medication_list_check() {
        let db = TransporterDatabase::with_clinical_drugs();

        // Multiple drugs
        let meds = vec!["134748", "32968", "83368"]; // Ritonavir, Digoxin, Rosuvastatin
        let predictions = db.check_medication_list(&meds);

        // Should find multiple interactions
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_drug_profile_substrate_check() {
        let db = TransporterDatabase::with_clinical_drugs();
        let digoxin = db.get_profile("32968").unwrap();

        assert!(digoxin.is_pgp_substrate());
        assert!(!digoxin.is_oatp1b1_substrate());
    }

    #[test]
    fn test_exposure_change_description() {
        assert!(ExposureChange::LargeIncrease
            .description()
            .contains("3-fold"));
        assert!(ExposureChange::ModerateIncrease
            .description()
            .contains("1.5-3"));
    }
}
