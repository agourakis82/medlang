//! Enhanced CYP450 Drug Interaction Module
//!
//! Implements comprehensive CYP enzyme modeling with FDA potency classification.
//! This module extends the basic DDI system with detailed substrate-inhibitor-inducer
//! relationships for predicting pharmacokinetic drug interactions.
//!
//! ## FDA Classification
//!
//! Based on FDA Drug Development and Drug Interactions guidance:
//! - **Strong inhibitor**: ≥5-fold increase in AUC or >80% decrease in clearance
//! - **Moderate inhibitor**: 2-5x increase in AUC
//! - **Weak inhibitor**: 1.25-2x increase in AUC
//! - **Strong inducer**: ≥80% decrease in AUC
//! - **Moderate inducer**: 50-80% decrease in AUC
//! - **Weak inducer**: 20-50% decrease in AUC

use super::ddi::{CypEnzyme, DdiSeverity, InhibitorPotency};
use std::collections::{HashMap, HashSet};

/// FDA-defined inducer potency levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InducerPotency {
    /// ≥80% decrease in AUC
    Strong,
    /// 50-80% decrease in AUC
    Moderate,
    /// 20-50% decrease in AUC
    Weak,
}

impl InducerPotency {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Strong => "Strong",
            Self::Moderate => "Moderate",
            Self::Weak => "Weak",
        }
    }

    /// Expected fold-change in AUC (as a fraction)
    pub fn expected_auc_ratio(&self) -> (f64, f64) {
        match self {
            Self::Strong => (0.0, 0.2),   // <20% remaining
            Self::Moderate => (0.2, 0.5), // 20-50% remaining
            Self::Weak => (0.5, 0.8),     // 50-80% remaining
        }
    }
}

impl std::fmt::Display for InducerPotency {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Substrate sensitivity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SubstrateSensitivity {
    /// AUC increases ≥5-fold with strong inhibitor
    Sensitive,
    /// AUC increases 2-5x with strong inhibitor
    Moderate,
    /// Minor contribution from this CYP
    Minor,
}

impl SubstrateSensitivity {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Sensitive => "Sensitive",
            Self::Moderate => "Moderate",
            Self::Minor => "Minor",
        }
    }
}

/// CYP role for a drug
#[derive(Debug, Clone, PartialEq)]
pub enum CypRole {
    /// Drug is metabolized by this CYP
    Substrate {
        enzyme: CypEnzyme,
        sensitivity: SubstrateSensitivity,
        /// Fraction metabolized by this pathway (fm)
        fraction_metabolized: Option<f64>,
    },
    /// Drug inhibits this CYP
    Inhibitor {
        enzyme: CypEnzyme,
        potency: InhibitorPotency,
        /// Time-dependent (mechanism-based) inhibition
        time_dependent: bool,
        /// Ki value in μM if known
        ki: Option<f64>,
    },
    /// Drug induces this CYP
    Inducer {
        enzyme: CypEnzyme,
        potency: InducerPotency,
    },
}

impl CypRole {
    pub fn enzyme(&self) -> CypEnzyme {
        match self {
            Self::Substrate { enzyme, .. } => *enzyme,
            Self::Inhibitor { enzyme, .. } => *enzyme,
            Self::Inducer { enzyme, .. } => *enzyme,
        }
    }

    pub fn is_substrate(&self) -> bool {
        matches!(self, Self::Substrate { .. })
    }

    pub fn is_inhibitor(&self) -> bool {
        matches!(self, Self::Inhibitor { .. })
    }

    pub fn is_inducer(&self) -> bool {
        matches!(self, Self::Inducer { .. })
    }
}

/// Drug CYP profile containing all CYP-related properties
#[derive(Debug, Clone)]
pub struct CypDrugProfile {
    /// RxNorm CUI or drug identifier
    pub drug_id: String,
    /// Drug name
    pub drug_name: String,
    /// All CYP roles for this drug
    pub roles: Vec<CypRole>,
    /// Narrow therapeutic index (NTI) drug flag
    pub narrow_therapeutic_index: bool,
    /// Clinical notes
    pub notes: Option<String>,
}

impl CypDrugProfile {
    pub fn new(drug_id: &str, drug_name: &str) -> Self {
        Self {
            drug_id: drug_id.to_string(),
            drug_name: drug_name.to_string(),
            roles: Vec::new(),
            narrow_therapeutic_index: false,
            notes: None,
        }
    }

    pub fn with_role(mut self, role: CypRole) -> Self {
        self.roles.push(role);
        self
    }

    pub fn with_nti(mut self) -> Self {
        self.narrow_therapeutic_index = true;
        self
    }

    pub fn with_notes(mut self, notes: &str) -> Self {
        self.notes = Some(notes.to_string());
        self
    }

    /// Get all enzymes this drug is a substrate of
    pub fn substrate_of(&self) -> Vec<(CypEnzyme, SubstrateSensitivity)> {
        self.roles
            .iter()
            .filter_map(|r| match r {
                CypRole::Substrate {
                    enzyme,
                    sensitivity,
                    ..
                } => Some((*enzyme, *sensitivity)),
                _ => None,
            })
            .collect()
    }

    /// Get all enzymes this drug inhibits
    pub fn inhibits(&self) -> Vec<(CypEnzyme, InhibitorPotency)> {
        self.roles
            .iter()
            .filter_map(|r| match r {
                CypRole::Inhibitor {
                    enzyme, potency, ..
                } => Some((*enzyme, *potency)),
                _ => None,
            })
            .collect()
    }

    /// Get all enzymes this drug induces
    pub fn induces(&self) -> Vec<(CypEnzyme, InducerPotency)> {
        self.roles
            .iter()
            .filter_map(|r| match r {
                CypRole::Inducer { enzyme, potency } => Some((*enzyme, *potency)),
                _ => None,
            })
            .collect()
    }

    /// Check if drug is a strong inhibitor of any CYP
    pub fn is_strong_inhibitor(&self) -> bool {
        self.roles.iter().any(|r| {
            matches!(
                r,
                CypRole::Inhibitor {
                    potency: InhibitorPotency::Strong,
                    ..
                }
            )
        })
}

    /// Check if drug is a strong inducer of any CYP
    pub fn is_strong_inducer(&self) -> bool {
        self.roles.iter().any(|r| {
            matches!(
                r,
                CypRole::Inducer {
                    potency: InducerPotency::Strong,
                    ..
                }
            )
        })
    }
}

/// CYP drug interaction prediction result
#[derive(Debug, Clone)]
pub struct CypInteractionPrediction {
    /// Perpetrator drug
    pub perpetrator: String,
    /// Victim/object drug
    pub victim: String,
    /// Affected enzyme
    pub enzyme: CypEnzyme,
    /// Type of interaction
    pub interaction_type: CypInteractionType,
    /// Predicted severity
    pub predicted_severity: DdiSeverity,
    /// Expected AUC change (fold for inhibition, fraction for induction)
    pub expected_auc_change: Option<(f64, f64)>,
    /// Clinical recommendation
    pub recommendation: String,
}

/// Type of CYP-mediated interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CypInteractionType {
    /// Inhibitor increases substrate exposure
    Inhibition,
    /// Inducer decreases substrate exposure
    Induction,
    /// Competitive inhibition (bidirectional)
    Competition,
}

impl CypInteractionType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Inhibition => "CYP Inhibition",
            Self::Induction => "CYP Induction",
            Self::Competition => "CYP Competition",
        }
    }
}

/// CYP drug database for interaction prediction
#[derive(Debug, Default)]
pub struct CypDatabase {
    /// Drug profiles by ID
    profiles: HashMap<String, CypDrugProfile>,
    /// Index: enzyme -> inhibitors
    inhibitors_by_enzyme: HashMap<CypEnzyme, Vec<String>>,
    /// Index: enzyme -> inducers
    inducers_by_enzyme: HashMap<CypEnzyme, Vec<String>>,
    /// Index: enzyme -> substrates
    substrates_by_enzyme: HashMap<CypEnzyme, Vec<String>>,
}

impl CypDatabase {
    pub fn new() -> Self {
        Self::default()
    }

    /// Create database with FDA reference drugs
    pub fn with_fda_reference_drugs() -> Self {
        let mut db = Self::new();

        // CYP3A4 Strong Inhibitors (FDA list)
        db.add_profile(CypDrugProfile::new("21212", "Clarithromycin").with_role(
            CypRole::Inhibitor {
                enzyme: CypEnzyme::Cyp3a4,
                potency: InhibitorPotency::Strong,
                time_dependent: true,
                ki: Some(0.25),
            },
        ));

        db.add_profile(CypDrugProfile::new("114979", "Itraconazole").with_role(
            CypRole::Inhibitor {
                enzyme: CypEnzyme::Cyp3a4,
                potency: InhibitorPotency::Strong,
                time_dependent: false,
                ki: Some(0.002),
            },
        ));

        db.add_profile(CypDrugProfile::new("85762", "Ketoconazole").with_role(
            CypRole::Inhibitor {
                enzyme: CypEnzyme::Cyp3a4,
                potency: InhibitorPotency::Strong,
                time_dependent: false,
                ki: Some(0.015),
            },
        ));

        db.add_profile(
            CypDrugProfile::new("134748", "Ritonavir")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InhibitorPotency::Strong,
                    time_dependent: true,
                    ki: Some(0.019),
                })
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2d6,
                    potency: InhibitorPotency::Moderate,
                    time_dependent: false,
                    ki: None,
                }),
        );

        // CYP3A4 Moderate Inhibitors
        db.add_profile(
            CypDrugProfile::new("29046", "Diltiazem")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InhibitorPotency::Moderate,
                    time_dependent: true,
                    ki: None,
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Moderate,
                    fraction_metabolized: Some(0.7),
                }),
        );

        db.add_profile(
            CypDrugProfile::new("4083", "Fluconazole")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InhibitorPotency::Moderate,
                    time_dependent: false,
                    ki: None,
                })
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2c9,
                    potency: InhibitorPotency::Strong,
                    time_dependent: false,
                    ki: Some(7.0),
                })
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2c19,
                    potency: InhibitorPotency::Strong,
                    time_dependent: false,
                    ki: None,
                }),
        );

        db.add_profile(
            CypDrugProfile::new("33738", "Verapamil")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InhibitorPotency::Moderate,
                    time_dependent: false,
                    ki: None,
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.9),
                }),
        );

        // CYP3A4 Weak Inhibitors
        db.add_profile(
            CypDrugProfile::new("18631", "Cimetidine")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InhibitorPotency::Weak,
                    time_dependent: false,
                    ki: None,
                })
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2d6,
                    potency: InhibitorPotency::Weak,
                    time_dependent: false,
                    ki: None,
                }),
        );

        // CYP3A4 Strong Inducers
        db.add_profile(
            CypDrugProfile::new("8183", "Rifampin")
                .with_role(CypRole::Inducer {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InducerPotency::Strong,
                })
                .with_role(CypRole::Inducer {
                    enzyme: CypEnzyme::Cyp2c9,
                    potency: InducerPotency::Strong,
                })
                .with_role(CypRole::Inducer {
                    enzyme: CypEnzyme::Cyp2c19,
                    potency: InducerPotency::Strong,
                })
                .with_notes("Prototypical strong pan-CYP inducer"),
        );

        db.add_profile(
            CypDrugProfile::new("28439", "Phenytoin")
                .with_role(CypRole::Inducer {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InducerPotency::Strong,
                })
                .with_role(CypRole::Inducer {
                    enzyme: CypEnzyme::Cyp2c9,
                    potency: InducerPotency::Moderate,
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2c9,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.9),
                })
                .with_nti(),
        );

        db.add_profile(
            CypDrugProfile::new("2002", "Carbamazepine")
                .with_role(CypRole::Inducer {
                    enzyme: CypEnzyme::Cyp3a4,
                    potency: InducerPotency::Strong,
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.75),
                })
                .with_nti()
                .with_notes("Auto-induction occurs over 3-5 weeks"),
        );

        // CYP3A4 Sensitive Substrates (index drugs)
        db.add_profile(
            CypDrugProfile::new("6754", "Midazolam")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.95),
                })
                .with_notes("FDA recommended CYP3A4 index substrate"),
        );

        db.add_profile(
            CypDrugProfile::new("36567", "Simvastatin")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.85),
                })
                .with_notes("High risk of myopathy with CYP3A4 inhibitors"),
        );

        db.add_profile(
            CypDrugProfile::new("73178", "Lovastatin").with_role(CypRole::Substrate {
                enzyme: CypEnzyme::Cyp3a4,
                sensitivity: SubstrateSensitivity::Sensitive,
                fraction_metabolized: Some(0.90),
            }),
        );

        // CYP2D6 Inhibitors
        db.add_profile(
            CypDrugProfile::new("4493", "Fluoxetine")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2d6,
                    potency: InhibitorPotency::Strong,
                    time_dependent: false,
                    ki: Some(0.17),
                })
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2c19,
                    potency: InhibitorPotency::Moderate,
                    time_dependent: false,
                    ki: None,
                }),
        );

        db.add_profile(
            CypDrugProfile::new("32937", "Paroxetine").with_role(CypRole::Inhibitor {
                enzyme: CypEnzyme::Cyp2d6,
                potency: InhibitorPotency::Strong,
                time_dependent: true,
                ki: Some(0.15),
            }),
        );

        db.add_profile(
            CypDrugProfile::new("10689", "Quinidine")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2d6,
                    potency: InhibitorPotency::Strong,
                    time_dependent: false,
                    ki: Some(0.06),
                })
                .with_notes("Prototypical CYP2D6 inhibitor"),
        );

        db.add_profile(
            CypDrugProfile::new("1738", "Bupropion")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2d6,
                    potency: InhibitorPotency::Strong,
                    time_dependent: false,
                    ki: None,
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2b6,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.9),
                }),
        );

        // CYP2D6 Substrates
        db.add_profile(
            CypDrugProfile::new("2670", "Codeine")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2d6,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.10), // O-demethylation to morphine
                })
                .with_notes("Prodrug - CYP2D6 converts to active morphine"),
        );

        db.add_profile(
            CypDrugProfile::new("10689", "Tramadol")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2d6,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.30),
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Moderate,
                    fraction_metabolized: Some(0.20),
                })
                .with_notes("Prodrug - active metabolite M1 formed by CYP2D6"),
        );

        db.add_profile(
            CypDrugProfile::new("39786", "Atomoxetine").with_role(CypRole::Substrate {
                enzyme: CypEnzyme::Cyp2d6,
                sensitivity: SubstrateSensitivity::Sensitive,
                fraction_metabolized: Some(0.80),
            }),
        );

        // CYP2C9 Substrates
        db.add_profile(
            CypDrugProfile::new("11289", "Warfarin")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2c9,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.85), // S-warfarin
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Minor,
                    fraction_metabolized: Some(0.10), // R-warfarin
                })
                .with_nti()
                .with_notes("S-warfarin (active) mainly CYP2C9; R-warfarin CYP1A2, 3A4"),
        );

        db.add_profile(
            CypDrugProfile::new("28439", "Phenytoin")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2c9,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.90),
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2c19,
                    sensitivity: SubstrateSensitivity::Minor,
                    fraction_metabolized: Some(0.10),
                })
                .with_nti(),
        );

        // CYP2C19 Substrates/Inhibitors
        db.add_profile(
            CypDrugProfile::new("7646", "Omeprazole")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2c19,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.80),
                })
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2c19,
                    potency: InhibitorPotency::Moderate,
                    time_dependent: false,
                    ki: None,
                }),
        );

        db.add_profile(
            CypDrugProfile::new("32968", "Clopidogrel")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp2c19,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.45),
                })
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp3a4,
                    sensitivity: SubstrateSensitivity::Moderate,
                    fraction_metabolized: Some(0.40),
                })
                .with_notes("Prodrug - CYP2C19 crucial for activation"),
        );

        // CYP1A2
        db.add_profile(CypDrugProfile::new("2404", "Ciprofloxacin").with_role(
            CypRole::Inhibitor {
                enzyme: CypEnzyme::Cyp1a2,
                potency: InhibitorPotency::Strong,
                time_dependent: false,
                ki: None,
            },
        ));

        db.add_profile(
            CypDrugProfile::new("42347", "Fluvoxamine")
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp1a2,
                    potency: InhibitorPotency::Strong,
                    time_dependent: false,
                    ki: Some(0.2),
                })
                .with_role(CypRole::Inhibitor {
                    enzyme: CypEnzyme::Cyp2c19,
                    potency: InhibitorPotency::Strong,
                    time_dependent: false,
                    ki: None,
                }),
        );

        db.add_profile(
            CypDrugProfile::new("10438", "Theophylline")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp1a2,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.70),
                })
                .with_nti(),
        );

        db.add_profile(
            CypDrugProfile::new("2403", "Caffeine")
                .with_role(CypRole::Substrate {
                    enzyme: CypEnzyme::Cyp1a2,
                    sensitivity: SubstrateSensitivity::Sensitive,
                    fraction_metabolized: Some(0.95),
                })
                .with_notes("FDA recommended CYP1A2 index substrate"),
        );

        db
    }

    /// Add a drug profile to the database
    pub fn add_profile(&mut self, profile: CypDrugProfile) {
        let drug_id = profile.drug_id.clone();

        // Update enzyme indices
        for role in &profile.roles {
            let enzyme = role.enzyme();
            match role {
                CypRole::Inhibitor { .. } => {
                    self.inhibitors_by_enzyme
                        .entry(enzyme)
                        .or_default()
                        .push(drug_id.clone());
                }
                CypRole::Inducer { .. } => {
                    self.inducers_by_enzyme
                        .entry(enzyme)
                        .or_default()
                        .push(drug_id.clone());
                }
                CypRole::Substrate { .. } => {
                    self.substrates_by_enzyme
                        .entry(enzyme)
                        .or_default()
                        .push(drug_id.clone());
                }
            }
        }

        self.profiles.insert(drug_id, profile);
    }

    /// Get a drug profile by ID
    pub fn get_profile(&self, drug_id: &str) -> Option<&CypDrugProfile> {
        self.profiles.get(drug_id)
    }

    /// Get all strong inhibitors of a CYP enzyme
    pub fn strong_inhibitors_of(&self, enzyme: CypEnzyme) -> Vec<&CypDrugProfile> {
        self.inhibitors_by_enzyme
            .get(&enzyme)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.profiles.get(id))
                    .filter(|p| {
                        p.roles.iter().any(|r| matches!(r,
                        CypRole::Inhibitor { enzyme: e, potency: InhibitorPotency::Strong, .. }
                        if *e == enzyme
                    ))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all sensitive substrates of a CYP enzyme
    pub fn sensitive_substrates_of(&self, enzyme: CypEnzyme) -> Vec<&CypDrugProfile> {
        self.substrates_by_enzyme
            .get(&enzyme)
            .map(|ids| {
                ids.iter()
                    .filter_map(|id| self.profiles.get(id))
                    .filter(|p| p.roles.iter().any(|r| matches!(r,
                        CypRole::Substrate { enzyme: e, sensitivity: SubstrateSensitivity::Sensitive, .. }
                        if *e == enzyme
                    )))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Predict CYP-mediated interactions between two drugs
    pub fn predict_interactions(
        &self,
        drug_a: &str,
        drug_b: &str,
    ) -> Vec<CypInteractionPrediction> {
        let mut predictions = Vec::new();

        let profile_a = match self.profiles.get(drug_a) {
            Some(p) => p,
            None => return predictions,
        };

        let profile_b = match self.profiles.get(drug_b) {
            Some(p) => p,
            None => return predictions,
        };

        // Check: A inhibits enzyme that B is substrate of
        for (enzyme, potency) in profile_a.inhibits() {
            for (sub_enzyme, sensitivity) in profile_b.substrate_of() {
                if enzyme == sub_enzyme {
                    let severity = predict_inhibition_severity(
                        potency,
                        sensitivity,
                        profile_b.narrow_therapeutic_index,
                    );
                    let auc_change = potency.expected_fold_change();

                    predictions.push(CypInteractionPrediction {
                        perpetrator: profile_a.drug_name.clone(),
                        victim: profile_b.drug_name.clone(),
                        enzyme,
                        interaction_type: CypInteractionType::Inhibition,
                        predicted_severity: severity,
                        expected_auc_change: Some(auc_change),
                        recommendation: inhibition_recommendation(
                            potency,
                            sensitivity,
                            &profile_b.drug_name,
                        ),
                    });
                }
            }
}

        // Check: B inhibits enzyme that A is substrate of
        for (enzyme, potency) in profile_b.inhibits() {
            for (sub_enzyme, sensitivity) in profile_a.substrate_of() {
                if enzyme == sub_enzyme {
                    let severity = predict_inhibition_severity(
                        potency,
                        sensitivity,
                        profile_a.narrow_therapeutic_index,
                    );
                    let auc_change = potency.expected_fold_change();

                    predictions.push(CypInteractionPrediction {
                        perpetrator: profile_b.drug_name.clone(),
                        victim: profile_a.drug_name.clone(),
                        enzyme,
                        interaction_type: CypInteractionType::Inhibition,
                        predicted_severity: severity,
                        expected_auc_change: Some(auc_change),
                        recommendation: inhibition_recommendation(
                            potency,
                            sensitivity,
                            &profile_a.drug_name,
                        ),
                    });
                }
            }
}

        // Check: A induces enzyme that B is substrate of
        for (enzyme, potency) in profile_a.induces() {
            for (sub_enzyme, sensitivity) in profile_b.substrate_of() {
                if enzyme == sub_enzyme {
                    let severity = predict_induction_severity(
                        potency,
                        sensitivity,
                        profile_b.narrow_therapeutic_index,
                    );
                    let auc_change = potency.expected_auc_ratio();

                    predictions.push(CypInteractionPrediction {
                        perpetrator: profile_a.drug_name.clone(),
                        victim: profile_b.drug_name.clone(),
                        enzyme,
                        interaction_type: CypInteractionType::Induction,
                        predicted_severity: severity,
                        expected_auc_change: Some(auc_change),
                        recommendation: induction_recommendation(potency, &profile_b.drug_name),
                    });
                }
            }
        }

        // Check: B induces enzyme that A is substrate of
        for (enzyme, potency) in profile_b.induces() {
            for (sub_enzyme, sensitivity) in profile_a.substrate_of() {
                if enzyme == sub_enzyme {
                    let severity = predict_induction_severity(
                        potency,
                        sensitivity,
                        profile_a.narrow_therapeutic_index,
                    );
                    let auc_change = potency.expected_auc_ratio();

                    predictions.push(CypInteractionPrediction {
                        perpetrator: profile_b.drug_name.clone(),
                        victim: profile_a.drug_name.clone(),
                        enzyme,
                        interaction_type: CypInteractionType::Induction,
                        predicted_severity: severity,
                        expected_auc_change: Some(auc_change),
                        recommendation: induction_recommendation(potency, &profile_a.drug_name),
                    });
                }
            }
        }

        predictions
    }

    /// Check a medication list for all CYP interactions
    pub fn check_medication_list(&self, drug_ids: &[&str]) -> Vec<CypInteractionPrediction> {
        let mut predictions = Vec::new();

        for i in 0..drug_ids.len() {
            for j in (i + 1)..drug_ids.len() {
                predictions.extend(self.predict_interactions(drug_ids[i], drug_ids[j]));
            }
        }

        // Sort by severity
        predictions.sort_by(|a, b| b.predicted_severity.cmp(&a.predicted_severity));
        predictions
    }
}

/// Predict severity of CYP inhibition interaction
fn predict_inhibition_severity(
    inhibitor_potency: InhibitorPotency,
    substrate_sensitivity: SubstrateSensitivity,
    is_nti: bool,
) -> DdiSeverity {
    match (inhibitor_potency, substrate_sensitivity, is_nti) {
        // Strong inhibitor + sensitive substrate (or NTI) = Contraindicated/Major
        (InhibitorPotency::Strong, SubstrateSensitivity::Sensitive, true) => {
            DdiSeverity::Contraindicated
        }
        (InhibitorPotency::Strong, SubstrateSensitivity::Sensitive, false) => DdiSeverity::Major,
        (InhibitorPotency::Strong, SubstrateSensitivity::Moderate, true) => DdiSeverity::Major,
        (InhibitorPotency::Strong, SubstrateSensitivity::Moderate, false) => DdiSeverity::Moderate,

        // Moderate inhibitor
        (InhibitorPotency::Moderate, SubstrateSensitivity::Sensitive, true) => DdiSeverity::Major,
        (InhibitorPotency::Moderate, SubstrateSensitivity::Sensitive, false) => {
            DdiSeverity::Moderate
        }
        (InhibitorPotency::Moderate, SubstrateSensitivity::Moderate, _) => DdiSeverity::Moderate,

        // Weak inhibitor or minor substrate
        (InhibitorPotency::Weak, _, true) => DdiSeverity::Moderate,
        (InhibitorPotency::Weak, SubstrateSensitivity::Sensitive, false) => DdiSeverity::Minor,
        (_, SubstrateSensitivity::Minor, _) => DdiSeverity::Minor,

        _ => DdiSeverity::Minor,
    }
}

/// Predict severity of CYP induction interaction
fn predict_induction_severity(
    inducer_potency: InducerPotency,
    substrate_sensitivity: SubstrateSensitivity,
    is_nti: bool,
) -> DdiSeverity {
    match (inducer_potency, substrate_sensitivity, is_nti) {
        // Strong inducer + sensitive substrate = Major (loss of efficacy can be serious)
        (InducerPotency::Strong, SubstrateSensitivity::Sensitive, true) => {
            DdiSeverity::Contraindicated
        }
        (InducerPotency::Strong, SubstrateSensitivity::Sensitive, false) => DdiSeverity::Major,
        (InducerPotency::Strong, SubstrateSensitivity::Moderate, _) => DdiSeverity::Moderate,

        // Moderate inducer
        (InducerPotency::Moderate, SubstrateSensitivity::Sensitive, true) => DdiSeverity::Major,
        (InducerPotency::Moderate, SubstrateSensitivity::Sensitive, false) => DdiSeverity::Moderate,
        (InducerPotency::Moderate, SubstrateSensitivity::Moderate, _) => DdiSeverity::Minor,

        // Weak inducer
        (InducerPotency::Weak, _, _) => DdiSeverity::Minor,

        _ => DdiSeverity::Minor,
    }
}

/// Generate recommendation for inhibition interaction
fn inhibition_recommendation(
    potency: InhibitorPotency,
    sensitivity: SubstrateSensitivity,
    victim_name: &str,
) -> String {
    match (potency, sensitivity) {
        (InhibitorPotency::Strong, SubstrateSensitivity::Sensitive) => {
            format!(
                "Avoid combination. If unavoidable, reduce {} dose by ≥50% and monitor closely.",
                victim_name
            )
        }
        (InhibitorPotency::Strong, SubstrateSensitivity::Moderate) => {
            format!(
                "Consider {} dose reduction by 25-50%. Monitor for adverse effects.",
                victim_name
            )
        }
        (InhibitorPotency::Moderate, SubstrateSensitivity::Sensitive) => {
            format!(
                "Consider {} dose reduction. Monitor for adverse effects.",
                victim_name
            )
        }
        _ => {
            format!(
                "Monitor for increased {} effects. Dose adjustment may be needed.",
                victim_name
            )
        }
    }
}

/// Generate recommendation for induction interaction
fn induction_recommendation(potency: InducerPotency, victim_name: &str) -> String {
    match potency {
        InducerPotency::Strong => {
            format!("Avoid combination if possible. If used, may need to increase {} dose significantly. Monitor for reduced efficacy.", victim_name)
        }
        InducerPotency::Moderate => {
            format!(
                "Monitor {} efficacy. Dose increase may be needed.",
                victim_name
            )
        }
        InducerPotency::Weak => {
            format!(
                "Monitor {} efficacy. Unlikely to be clinically significant.",
                victim_name
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cyp_database_creation() {
        let db = CypDatabase::with_fda_reference_drugs();
        assert!(db.get_profile("21212").is_some()); // Clarithromycin
        assert!(db.get_profile("36567").is_some()); // Simvastatin
    }

    #[test]
    fn test_strong_inhibitors() {
        let db = CypDatabase::with_fda_reference_drugs();
        let strong_3a4 = db.strong_inhibitors_of(CypEnzyme::Cyp3a4);

        assert!(!strong_3a4.is_empty());
        assert!(strong_3a4.iter().any(|p| p.drug_name == "Clarithromycin"));
        assert!(strong_3a4.iter().any(|p| p.drug_name == "Ketoconazole"));
    }

    #[test]
    fn test_sensitive_substrates() {
        let db = CypDatabase::with_fda_reference_drugs();
        let sensitive_3a4 = db.sensitive_substrates_of(CypEnzyme::Cyp3a4);

        assert!(!sensitive_3a4.is_empty());
        assert!(sensitive_3a4.iter().any(|p| p.drug_name == "Simvastatin"));
        assert!(sensitive_3a4.iter().any(|p| p.drug_name == "Midazolam"));
    }

    #[test]
    fn test_interaction_prediction_inhibition() {
        let db = CypDatabase::with_fda_reference_drugs();

        // Clarithromycin (strong 3A4 inhibitor) + Simvastatin (sensitive 3A4 substrate)
        let predictions = db.predict_interactions("21212", "36567");

        assert!(!predictions.is_empty());
        let pred = &predictions[0];
        assert_eq!(pred.interaction_type, CypInteractionType::Inhibition);
        assert_eq!(pred.enzyme, CypEnzyme::Cyp3a4);
        assert!(pred.predicted_severity >= DdiSeverity::Major);
    }

    #[test]
    fn test_interaction_prediction_induction() {
        let db = CypDatabase::with_fda_reference_drugs();

        // Rifampin (strong inducer) + Simvastatin (sensitive substrate)
        let predictions = db.predict_interactions("8183", "36567");

        assert!(!predictions.is_empty());
        assert!(predictions
            .iter()
            .any(|p| p.interaction_type == CypInteractionType::Induction));
    }

    #[test]
    fn test_nti_drug_severity() {
        let db = CypDatabase::with_fda_reference_drugs();

        // Fluconazole (strong 2C9 inhibitor) + Warfarin (sensitive 2C9 substrate, NTI)
        let predictions = db.predict_interactions("4083", "11289");

        assert!(!predictions.is_empty());
        // Should be contraindicated due to NTI
        let cyp2c9_pred = predictions
            .iter()
            .find(|p| p.enzyme == CypEnzyme::Cyp2c9)
            .unwrap();
        assert_eq!(cyp2c9_pred.predicted_severity, DdiSeverity::Contraindicated);
    }

    #[test]
    fn test_medication_list_check() {
        let db = CypDatabase::with_fda_reference_drugs();

        let meds = vec!["21212", "36567", "4493"]; // Clarithromycin, Simvastatin, Fluoxetine
        let predictions = db.check_medication_list(&meds);

        // Should find clarithromycin + simvastatin interaction
        assert!(predictions
            .iter()
            .any(|p| p.perpetrator == "Clarithromycin" && p.victim == "Simvastatin"));
    }

    #[test]
    fn test_cyp2d6_interactions() {
        let db = CypDatabase::with_fda_reference_drugs();

        // Fluoxetine (strong 2D6 inhibitor) + Codeine (2D6 substrate prodrug)
        let predictions = db.predict_interactions("4493", "2670");

        assert!(!predictions.is_empty());
        let pred = &predictions[0];
        assert_eq!(pred.enzyme, CypEnzyme::Cyp2d6);
    }

    #[test]
    fn test_drug_profile_builder() {
        let profile = CypDrugProfile::new("12345", "TestDrug")
            .with_role(CypRole::Inhibitor {
                enzyme: CypEnzyme::Cyp3a4,
                potency: InhibitorPotency::Strong,
                time_dependent: true,
                ki: Some(0.1),
            })
            .with_role(CypRole::Substrate {
                enzyme: CypEnzyme::Cyp2d6,
                sensitivity: SubstrateSensitivity::Moderate,
                fraction_metabolized: Some(0.5),
            })
            .with_nti()
            .with_notes("Test drug");

        assert!(profile.narrow_therapeutic_index);
        assert!(profile.is_strong_inhibitor());
        assert_eq!(profile.inhibits().len(), 1);
        assert_eq!(profile.substrate_of().len(), 1);
    }

    #[test]
    fn test_inducer_potency_auc() {
        assert_eq!(InducerPotency::Strong.expected_auc_ratio(), (0.0, 0.2));
        assert_eq!(InducerPotency::Moderate.expected_auc_ratio(), (0.2, 0.5));
        assert_eq!(InducerPotency::Weak.expected_auc_ratio(), (0.5, 0.8));
    }

    #[test]
    fn test_bidirectional_interaction() {
        let db = CypDatabase::with_fda_reference_drugs();

        // Diltiazem is both CYP3A4 inhibitor AND substrate
        let profile = db.get_profile("29046").unwrap();

        assert!(!profile.inhibits().is_empty());
        assert!(!profile.substrate_of().is_empty());
    }
}
