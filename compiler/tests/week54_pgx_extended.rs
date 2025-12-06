//! Week 54 Extended Integration Tests
//!
//! Comprehensive tests for the pharmacogenomics, CYP modeling, and
//! transporter interaction systems.

use medlangc::ontology::pgx::{
    alleles::{AlleleFunction, AlleleRegistry, Pharmacogene, StarAllele},
    cpic::{
        ClinicalAction, CpicDecisionSupport, CpicLevel, CpicRegistry, DoseAdjustment,
        RecommendationStrength,
    },
    phenotype::{
        Diplotype, DosingCategory, MetabolizerPhenotype, PgxProfile, PhenotypeClassifier,
        PhenotypeThresholds,
    },
};

use medlangc::ontology::cyp::{
    CypDatabase, CypDrugProfile, CypInteractionType, CypRole, InducerPotency, SubstrateSensitivity,
};

use medlangc::ontology::transporters::{
    SubstrateSignificance, TransporterDatabase, TransporterDrugProfile,
    TransporterInhibitorPotency, TransporterRole, TransporterType,
};

use medlangc::ontology::ddi::{CypEnzyme, DdiSeverity, InhibitorPotency};

// =============================================================================
// Pharmacogenomics Tests
// =============================================================================

#[test]
fn test_comprehensive_allele_registry() {
    let registry = AlleleRegistry::comprehensive();

    // CYP2D6 alleles
    assert!(registry.get("CYP2D6*1").is_some());
    assert!(registry.get("CYP2D6*4").is_some());
    assert!(registry.get("CYP2D6*10").is_some());
    assert!(registry.get("CYP2D6*17").is_some());

    // CYP2C19 alleles
    assert!(registry.get("CYP2C19*1").is_some());
    assert!(registry.get("CYP2C19*2").is_some());
    assert!(registry.get("CYP2C19*17").is_some());

    // CYP2C9 alleles
    assert!(registry.get("CYP2C9*1").is_some());
    assert!(registry.get("CYP2C9*2").is_some());
    assert!(registry.get("CYP2C9*3").is_some());

    // TPMT alleles
    assert!(registry.get("TPMT*1").is_some());
    assert!(registry.get("TPMT*3A").is_some());

    // SLCO1B1 alleles
    assert!(registry.get("SLCO1B1*5").is_some());
}

#[test]
fn test_activity_score_ranges() {
    let registry = AlleleRegistry::with_cyp2d6_defaults();

    // Normal function = 1.0
    let star1 = registry.get("CYP2D6*1").unwrap();
    assert_eq!(star1.activity_score, 1.0);
    assert_eq!(star1.function, AlleleFunction::Normal);

    // No function = 0.0
    let star4 = registry.get("CYP2D6*4").unwrap();
    assert_eq!(star4.activity_score, 0.0);
    assert_eq!(star4.function, AlleleFunction::NoFunction);

    // Decreased function varies
    let star10 = registry.get("CYP2D6*10").unwrap();
    assert_eq!(star10.activity_score, 0.25);
    assert_eq!(star10.function, AlleleFunction::Decreased);

    let star41 = registry.get("CYP2D6*41").unwrap();
    assert_eq!(star41.activity_score, 0.5);

    // Increased function (gene duplication)
    let star1xn = registry.get("CYP2D6*1xN").unwrap();
    assert_eq!(star1xn.activity_score, 2.0);
    assert_eq!(star1xn.function, AlleleFunction::Increased);
}

#[test]
fn test_diplotype_activity_calculation() {
    let registry = AlleleRegistry::with_cyp2d6_defaults();

    // *1/*1 = 2.0 (normal)
    let diplotype = Diplotype::new(Pharmacogene::CYP2D6, "*1", "*1");
    assert_eq!(diplotype.activity_score(&registry), Some(2.0));

    // *1/*4 = 1.0 (IM)
    let diplotype = Diplotype::new(Pharmacogene::CYP2D6, "*1", "*4");
    assert_eq!(diplotype.activity_score(&registry), Some(1.0));

    // *4/*4 = 0.0 (PM)
    let diplotype = Diplotype::new(Pharmacogene::CYP2D6, "*4", "*4");
    assert_eq!(diplotype.activity_score(&registry), Some(0.0));

    // *1xN/*1 = 3.0 (UM)
    let diplotype = Diplotype::new(Pharmacogene::CYP2D6, "*1xN", "*1");
    assert_eq!(diplotype.activity_score(&registry), Some(3.0));
}

#[test]
fn test_phenotype_classification_cyp2d6() {
    let classifier = PhenotypeClassifier::cyp2d6();

    // Poor Metabolizer: *4/*4 (AS = 0)
    let pm = classifier.classify_alleles("*4", "*4").unwrap();
    assert_eq!(pm.phenotype, MetabolizerPhenotype::PoorMetabolizer);
    assert_eq!(pm.activity_score, 0.0);

    // Intermediate Metabolizer: *1/*4 (AS = 1.0)
    let im = classifier.classify_alleles("*1", "*4").unwrap();
    assert_eq!(im.phenotype, MetabolizerPhenotype::IntermediateMetabolizer);
    assert_eq!(im.activity_score, 1.0);

    // Normal Metabolizer: *1/*1 (AS = 2.0)
    let nm = classifier.classify_alleles("*1", "*1").unwrap();
    assert_eq!(nm.phenotype, MetabolizerPhenotype::NormalMetabolizer);
    assert_eq!(nm.activity_score, 2.0);

    // Ultrarapid Metabolizer: *1xN/*1 (AS = 3.0)
    let um = classifier.classify_alleles("*1xN", "*1").unwrap();
    assert_eq!(um.phenotype, MetabolizerPhenotype::UltrarapidMetabolizer);
    assert_eq!(um.activity_score, 3.0);
}

#[test]
fn test_phenotype_classification_cyp2c19() {
    let classifier = PhenotypeClassifier::cyp2c19();

    // Poor Metabolizer: *2/*2
    let pm = classifier.classify_alleles("*2", "*2").unwrap();
    assert_eq!(pm.phenotype, MetabolizerPhenotype::PoorMetabolizer);

    // Normal Metabolizer: *1/*1
    let nm = classifier.classify_alleles("*1", "*1").unwrap();
    assert_eq!(nm.phenotype, MetabolizerPhenotype::NormalMetabolizer);

    // Ultrarapid Metabolizer: *17/*17
    let um = classifier.classify_alleles("*17", "*17").unwrap();
    assert_eq!(um.phenotype, MetabolizerPhenotype::UltrarapidMetabolizer);

    // Rapid Metabolizer: *1/*17
    let rm = classifier.classify_alleles("*1", "*17").unwrap();
    assert!(
        rm.phenotype == MetabolizerPhenotype::RapidMetabolizer
            || rm.phenotype == MetabolizerPhenotype::NormalMetabolizer
    );
}

#[test]
fn test_pgx_profile_multi_gene() {
    let mut profile = PgxProfile::new();

    // Add CYP2D6 result
    let cyp2d6 = PhenotypeClassifier::cyp2d6();
    let cyp2d6_result = cyp2d6.classify_alleles("*4", "*4").unwrap();
    profile.add_result(cyp2d6_result);

    // Add CYP2C19 result
    let cyp2c19 = PhenotypeClassifier::cyp2c19();
    let cyp2c19_result = cyp2c19.classify_alleles("*1", "*1").unwrap();
    profile.add_result(cyp2c19_result);

    // Check results
    assert_eq!(
        profile.phenotype(Pharmacogene::CYP2D6),
        Some(MetabolizerPhenotype::PoorMetabolizer)
    );
    assert_eq!(
        profile.phenotype(Pharmacogene::CYP2C19),
        Some(MetabolizerPhenotype::NormalMetabolizer)
    );

    // Check aggregations
    assert!(profile.is_poor_metabolizer_any());
    let pm_genes = profile.poor_metabolizer_genes();
    assert!(pm_genes.contains(&Pharmacogene::CYP2D6));
    assert!(!pm_genes.contains(&Pharmacogene::CYP2C19));
}

#[test]
fn test_cpic_codeine_recommendations() {
    let registry = CpicRegistry::with_common_guidelines();
    let guideline = registry.get(Pharmacogene::CYP2D6, "codeine").unwrap();

    assert_eq!(guideline.level, CpicLevel::A);

    // UM: Avoid due to toxicity risk
    let um_rec = guideline
        .get_recommendation(MetabolizerPhenotype::UltrarapidMetabolizer)
        .unwrap();
    assert!(um_rec.recommendation.to_lowercase().contains("avoid"));
    assert!(um_rec.alternatives.contains(&"morphine".to_string()));

    // PM: Avoid due to lack of efficacy
    let pm_rec = guideline
        .get_recommendation(MetabolizerPhenotype::PoorMetabolizer)
        .unwrap();
    assert!(pm_rec.recommendation.to_lowercase().contains("avoid"));

    // NM: Standard dosing
    let nm_rec = guideline
        .get_recommendation(MetabolizerPhenotype::NormalMetabolizer)
        .unwrap();
    assert!(nm_rec.recommendation.to_lowercase().contains("label"));
}

#[test]
fn test_cpic_clopidogrel_recommendations() {
    let registry = CpicRegistry::with_common_guidelines();
    let guideline = registry.get(Pharmacogene::CYP2C19, "clopidogrel").unwrap();

    assert_eq!(guideline.level, CpicLevel::A);

    // PM: Use alternative antiplatelet
    let pm_rec = guideline
        .get_recommendation(MetabolizerPhenotype::PoorMetabolizer)
        .unwrap();
    assert!(pm_rec.alternatives.contains(&"prasugrel".to_string()));
    assert!(pm_rec.alternatives.contains(&"ticagrelor".to_string()));

    // IM: Consider alternative
    let im_rec = guideline
        .get_recommendation(MetabolizerPhenotype::IntermediateMetabolizer)
        .unwrap();
    assert!(im_rec.recommendation.to_lowercase().contains("alternative"));
}

#[test]
fn test_cpic_decision_support_workflow() {
    // Patient genotype: CYP2D6 *4/*4 (PM)
    let classifier = PhenotypeClassifier::cyp2d6();
    let result = classifier.classify_alleles("*4", "*4").unwrap();

    let support = CpicDecisionSupport::with_common_guidelines();

    // Check codeine
    let codeine_decision = support.decide(&result, "codeine");
    assert_eq!(codeine_decision.action, ClinicalAction::Avoid);
    assert!(codeine_decision.recommendation.is_some());

    // Check tramadol
    let tramadol_decision = support.decide(&result, "tramadol");
    assert_eq!(tramadol_decision.action, ClinicalAction::Avoid);

    // Check unknown drug
    let unknown_decision = support.decide(&result, "aspirin");
    assert_eq!(unknown_decision.action, ClinicalAction::NoGuideline);
}

#[test]
fn test_dosing_category_mapping() {
    let classifier = PhenotypeClassifier::cyp2d6();

    let pm = classifier.classify_alleles("*4", "*4").unwrap();
    assert_eq!(pm.dosing_category(), DosingCategory::SignificantReduction);

    let im = classifier.classify_alleles("*1", "*4").unwrap();
    assert_eq!(im.dosing_category(), DosingCategory::ReducedDose);

    let nm = classifier.classify_alleles("*1", "*1").unwrap();
    assert_eq!(nm.dosing_category(), DosingCategory::StandardDose);

    let um = classifier.classify_alleles("*1xN", "*1").unwrap();
    assert_eq!(um.dosing_category(), DosingCategory::ConsiderAlternative);
}

// =============================================================================
// CYP450 Modeling Tests
// =============================================================================

#[test]
fn test_cyp_database_fda_reference_drugs() {
    let db = CypDatabase::with_fda_reference_drugs();

    // Check strong CYP3A4 inhibitors
    let strong_3a4 = db.strong_inhibitors_of(CypEnzyme::Cyp3a4);
    assert!(strong_3a4.iter().any(|p| p.drug_name == "Clarithromycin"));
    assert!(strong_3a4.iter().any(|p| p.drug_name == "Ketoconazole"));
    assert!(strong_3a4.iter().any(|p| p.drug_name == "Ritonavir"));
    assert!(strong_3a4.iter().any(|p| p.drug_name == "Itraconazole"));

    // Check strong CYP2D6 inhibitors
    let strong_2d6 = db.strong_inhibitors_of(CypEnzyme::Cyp2d6);
    assert!(strong_2d6.iter().any(|p| p.drug_name == "Fluoxetine"));
    assert!(strong_2d6.iter().any(|p| p.drug_name == "Paroxetine"));
    assert!(strong_2d6.iter().any(|p| p.drug_name == "Bupropion"));
    // At least 3 strong 2D6 inhibitors in database
    assert!(strong_2d6.len() >= 3);
}

#[test]
fn test_cyp_sensitive_substrates() {
    let db = CypDatabase::with_fda_reference_drugs();

    // CYP3A4 sensitive substrates
    let sensitive_3a4 = db.sensitive_substrates_of(CypEnzyme::Cyp3a4);
    assert!(sensitive_3a4.iter().any(|p| p.drug_name == "Midazolam"));
    assert!(sensitive_3a4.iter().any(|p| p.drug_name == "Simvastatin"));
    assert!(sensitive_3a4.iter().any(|p| p.drug_name == "Lovastatin"));

    // CYP2D6 sensitive substrates
    let sensitive_2d6 = db.sensitive_substrates_of(CypEnzyme::Cyp2d6);
    assert!(sensitive_2d6.iter().any(|p| p.drug_name == "Codeine"));
    assert!(sensitive_2d6.iter().any(|p| p.drug_name == "Atomoxetine"));
}

#[test]
fn test_cyp3a4_inhibition_interaction() {
    let db = CypDatabase::with_fda_reference_drugs();

    // Clarithromycin + Simvastatin (contraindicated per FDA)
    let predictions = db.predict_interactions("21212", "36567");

    assert!(!predictions.is_empty());
    let cyp3a4_pred = predictions
        .iter()
        .find(|p| p.enzyme == CypEnzyme::Cyp3a4)
        .unwrap();

    assert_eq!(cyp3a4_pred.interaction_type, CypInteractionType::Inhibition);
    assert_eq!(cyp3a4_pred.perpetrator, "Clarithromycin");
    assert_eq!(cyp3a4_pred.victim, "Simvastatin");
    assert!(cyp3a4_pred.predicted_severity >= DdiSeverity::Major);
}

#[test]
fn test_cyp_induction_interaction() {
    let db = CypDatabase::with_fda_reference_drugs();

    // Rifampin (strong inducer) + Simvastatin (sensitive substrate)
    let predictions = db.predict_interactions("8183", "36567");

    assert!(!predictions.is_empty());
    let induction = predictions
        .iter()
        .find(|p| p.interaction_type == CypInteractionType::Induction)
        .unwrap();

    assert_eq!(induction.perpetrator, "Rifampin");
    assert_eq!(induction.victim, "Simvastatin");
    assert!(induction.predicted_severity >= DdiSeverity::Major);
}

#[test]
fn test_cyp2c9_warfarin_interaction() {
    let db = CypDatabase::with_fda_reference_drugs();

    // Fluconazole (strong 2C9 inhibitor) + Warfarin (sensitive 2C9 substrate, NTI)
    let predictions = db.predict_interactions("4083", "11289");

    let cyp2c9_pred = predictions
        .iter()
        .find(|p| p.enzyme == CypEnzyme::Cyp2c9)
        .unwrap();

    // Should be contraindicated due to NTI
    assert_eq!(cyp2c9_pred.predicted_severity, DdiSeverity::Contraindicated);
}

#[test]
fn test_medication_list_cyp_check() {
    let db = CypDatabase::with_fda_reference_drugs();

    // Complex medication list
    let meds = vec![
        "21212", // Clarithromycin (3A4 inhibitor)
        "36567", // Simvastatin (3A4 substrate)
        "4493",  // Fluoxetine (2D6 inhibitor)
        "2670",  // Codeine (2D6 substrate)
    ];

    let predictions = db.check_medication_list(&meds);

    // Should find clarithromycin + simvastatin
    assert!(predictions
        .iter()
        .any(|p| p.perpetrator == "Clarithromycin" && p.victim == "Simvastatin"));

    // Should find fluoxetine + codeine
    assert!(predictions
        .iter()
        .any(|p| p.perpetrator == "Fluoxetine" && p.victim == "Codeine"));
}

#[test]
fn test_drug_multiple_cyp_roles() {
    let db = CypDatabase::with_fda_reference_drugs();

    // Diltiazem is both CYP3A4 inhibitor AND substrate
    let diltiazem = db.get_profile("29046").unwrap();

    let inhibits = diltiazem.inhibits();
    let substrates = diltiazem.substrate_of();

    assert!(inhibits.iter().any(|(e, _)| *e == CypEnzyme::Cyp3a4));
    assert!(substrates.iter().any(|(e, _)| *e == CypEnzyme::Cyp3a4));
}

#[test]
fn test_pan_cyp_inducer() {
    let db = CypDatabase::with_fda_reference_drugs();

    // Rifampin induces multiple CYP enzymes
    let rifampin = db.get_profile("8183").unwrap();
    let induces = rifampin.induces();

    assert!(induces
        .iter()
        .any(|(e, p)| *e == CypEnzyme::Cyp3a4 && *p == InducerPotency::Strong));
    assert!(induces.iter().any(|(e, _)| *e == CypEnzyme::Cyp2c9));
    assert!(induces.iter().any(|(e, _)| *e == CypEnzyme::Cyp2c19));
}

// =============================================================================
// Transporter Interaction Tests
// =============================================================================

#[test]
fn test_transporter_database_coverage() {
    let db = TransporterDatabase::with_clinical_drugs();

    // P-gp inhibitors
    let pgp_inhibitors = db.pgp_inhibitors();
    assert!(pgp_inhibitors.iter().any(|p| p.drug_name == "Ritonavir"));
    assert!(pgp_inhibitors.iter().any(|p| p.drug_name == "Cyclosporine"));
    assert!(pgp_inhibitors.iter().any(|p| p.drug_name == "Verapamil"));

    // P-gp substrates
    let pgp_substrates = db.pgp_substrates();
    assert!(pgp_substrates.iter().any(|p| p.drug_name == "Digoxin"));
    assert!(pgp_substrates.iter().any(|p| p.drug_name == "Dabigatran"));
    assert!(pgp_substrates.iter().any(|p| p.drug_name == "Fexofenadine"));
}

#[test]
fn test_pgp_digoxin_interaction() {
    let db = TransporterDatabase::with_clinical_drugs();

    // Ritonavir (P-gp inhibitor) + Digoxin (P-gp substrate)
    let predictions = db.predict_interactions("134748", "32968");

    assert!(!predictions.is_empty());
    let pgp_pred = predictions
        .iter()
        .find(|p| p.transporter == TransporterType::Pgp)
        .unwrap();

    assert_eq!(pgp_pred.perpetrator, "Ritonavir");
    assert_eq!(pgp_pred.victim, "Digoxin");
    assert!(pgp_pred.predicted_severity >= DdiSeverity::Moderate);
}

#[test]
fn test_oatp_statin_interaction() {
    let db = TransporterDatabase::with_clinical_drugs();

    // Cyclosporine (OATP inhibitor) + Rosuvastatin (OATP substrate)
    let predictions = db.predict_interactions("114984", "83368");

    assert!(!predictions.is_empty());

    // Should find OATP1B1 and/or OATP1B3 interactions
    let oatp_interactions: Vec<_> = predictions
        .iter()
        .filter(|p| {
            matches!(
                p.transporter,
                TransporterType::Oatp1b1 | TransporterType::Oatp1b3
            )
        })
        .collect();

    assert!(!oatp_interactions.is_empty());
}

#[test]
fn test_renal_transporter_metformin() {
    let db = TransporterDatabase::with_clinical_drugs();

    // Dolutegravir (OCT2/MATE inhibitor) + Metformin (OCT2/MATE substrate)
    let predictions = db.predict_interactions("134517", "6809");

    assert!(!predictions.is_empty());

    // Should find OCT2 or MATE interaction
    let renal_interaction = predictions.iter().any(|p| {
        matches!(
            p.transporter,
            TransporterType::Oct2 | TransporterType::Mate1 | TransporterType::Mate2k
        )
    });

    assert!(renal_interaction);
}

#[test]
fn test_transporter_tissue_expression() {
    assert!(TransporterType::Pgp
        .tissue_expression()
        .contains(&"intestine"));
    assert!(TransporterType::Pgp.tissue_expression().contains(&"BBB"));
    assert!(TransporterType::Oatp1b1
        .tissue_expression()
        .contains(&"liver"));
    assert!(TransporterType::Oct2
        .tissue_expression()
        .contains(&"kidney"));
}

#[test]
fn test_transporter_classification() {
    use medlangc::ontology::transporters::TransporterClass;

    // Efflux transporters
    assert_eq!(TransporterType::Pgp.class(), TransporterClass::Efflux);
    assert_eq!(TransporterType::Bcrp.class(), TransporterClass::Efflux);

    // Uptake transporters
    assert_eq!(TransporterType::Oatp1b1.class(), TransporterClass::Uptake);
    assert_eq!(TransporterType::Oct2.class(), TransporterClass::Uptake);
}

// =============================================================================
// Integrated Clinical Scenario Tests
// =============================================================================

#[test]
fn test_clinical_scenario_codeine_pm() {
    // Patient: CYP2D6 *4/*4 (Poor Metabolizer)
    // Drug: Codeine

    // Step 1: Determine phenotype
    let classifier = PhenotypeClassifier::cyp2d6();
    let result = classifier.classify_alleles("*4", "*4").unwrap();
    assert_eq!(result.phenotype, MetabolizerPhenotype::PoorMetabolizer);

    // Step 2: Get CPIC recommendation
    let support = CpicDecisionSupport::with_common_guidelines();
    let decision = support.decide(&result, "codeine");

    // Step 3: Verify clinical action
    assert_eq!(decision.action, ClinicalAction::Avoid);

    // Step 4: Check alternatives available
    let rec = decision.recommendation.unwrap();
    assert!(!rec.alternatives.is_empty());
}

#[test]
fn test_clinical_scenario_warfarin_fluconazole() {
    // Patient on warfarin, needs antifungal treatment

    let cyp_db = CypDatabase::with_fda_reference_drugs();

    // Fluconazole (4083) + Warfarin (11289)
    let predictions = cyp_db.predict_interactions("4083", "11289");

    // Should predict contraindicated interaction due to NTI
    let cyp2c9_pred = predictions
        .iter()
        .find(|p| p.enzyme == CypEnzyme::Cyp2c9)
        .unwrap();

    assert_eq!(cyp2c9_pred.predicted_severity, DdiSeverity::Contraindicated);

    // Recommendation should mention dose reduction
    assert!(
        cyp2c9_pred.recommendation.to_lowercase().contains("reduce")
            || cyp2c9_pred
                .recommendation
                .to_lowercase()
                .contains("monitor")
    );
}

#[test]
fn test_clinical_scenario_statin_interactions() {
    // Patient on simvastatin, needs macrolide antibiotic

    let cyp_db = CypDatabase::with_fda_reference_drugs();
    let transporter_db = TransporterDatabase::with_clinical_drugs();

    // CYP3A4 inhibition: Clarithromycin + Simvastatin
    let cyp_predictions = cyp_db.predict_interactions("21212", "36567");
    assert!(cyp_predictions
        .iter()
        .any(|p| p.enzyme == CypEnzyme::Cyp3a4 && p.predicted_severity >= DdiSeverity::Major));

    // OATP inhibition: Cyclosporine + Rosuvastatin
    let transporter_predictions = transporter_db.predict_interactions("114984", "83368");
    assert!(!transporter_predictions.is_empty());
}

#[test]
fn test_clinical_scenario_anticoagulant_pgp() {
    // Patient on dabigatran, needs P-gp inhibitor

    let db = TransporterDatabase::with_clinical_drugs();

    // Ritonavir + Dabigatran
    let predictions = db.predict_interactions("134748", "114970");

    let pgp_interaction = predictions
        .iter()
        .find(|p| p.transporter == TransporterType::Pgp)
        .unwrap();

    // Should be significant due to narrow therapeutic index
    assert!(pgp_interaction.predicted_severity >= DdiSeverity::Moderate);
    assert!(pgp_interaction.recommendation.contains("P-gp"));
}

#[test]
fn test_comprehensive_medication_review() {
    // Simulate a comprehensive medication review

    // Patient profile: CYP2D6 *1/*4 (IM), CYP2C19 *2/*2 (PM)
    let mut pgx_profile = PgxProfile::new();

    let cyp2d6_classifier = PhenotypeClassifier::cyp2d6();
    let cyp2d6_result = cyp2d6_classifier.classify_alleles("*1", "*4").unwrap();
    pgx_profile.add_result(cyp2d6_result.clone());

    let cyp2c19_classifier = PhenotypeClassifier::cyp2c19();
    let cyp2c19_result = cyp2c19_classifier.classify_alleles("*2", "*2").unwrap();
    pgx_profile.add_result(cyp2c19_result.clone());

    // Medication list
    let medications = vec!["codeine", "clopidogrel", "omeprazole"];

    // Check each medication
    let cpic = CpicDecisionSupport::with_common_guidelines();

    // Codeine with CYP2D6 IM - reduced efficacy possible
    let codeine_decision = cpic.decide(&cyp2d6_result, "codeine");
    // IM for codeine typically results in reduced efficacy monitoring

    // Clopidogrel with CYP2C19 PM - should avoid
    let clopidogrel_decision = cpic.decide(&cyp2c19_result, "clopidogrel");
    assert_eq!(clopidogrel_decision.action, ClinicalAction::Avoid);

    // Generate summary
    let summary = pgx_profile.summary();
    assert!(!summary.is_empty());
}

#[test]
fn test_dose_adjustment_calculation() {
    // Test dose adjustment factors
    let reduce_50 = DoseAdjustment::reduce(0.5);
    assert_eq!(reduce_50.factor, 0.5);

    let avoid = DoseAdjustment::avoid();
    assert_eq!(avoid.factor, 0.0);
    assert!(avoid.instructions.is_some());

    let with_max = DoseAdjustment::reduce(0.5)
        .with_max_dose("20mg")
        .with_starting_dose("10mg");
    assert_eq!(with_max.max_dose, Some("20mg".to_string()));
    assert_eq!(with_max.starting_dose, Some("10mg".to_string()));
}

#[test]
fn test_phenotype_typical_dose_factors() {
    assert_eq!(
        MetabolizerPhenotype::NormalMetabolizer.typical_dose_factor(),
        Some(1.0)
    );
    assert_eq!(
        MetabolizerPhenotype::PoorMetabolizer.typical_dose_factor(),
        Some(0.25)
    );
    assert_eq!(
        MetabolizerPhenotype::UltrarapidMetabolizer.typical_dose_factor(),
        Some(1.5)
    );
    assert_eq!(
        MetabolizerPhenotype::Indeterminate.typical_dose_factor(),
        None
    );
}
