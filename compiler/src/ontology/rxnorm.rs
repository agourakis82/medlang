// Week 54: RxNorm Drug Terminology Types
//
// RxNorm provides normalized names for clinical drugs and links to vocabularies
// commonly used in pharmacy management and drug interaction software.
//
// ## RxNorm Term Types (TTY)
//
// - IN: Ingredient (base compound)
// - PIN: Precise Ingredient (specific salt form)
// - MIN: Multiple Ingredients
// - SCDC: Semantic Clinical Drug Component (ingredient + strength)
// - SCDF: Semantic Clinical Drug Form (ingredient + dose form)
// - SCD: Semantic Clinical Drug (full clinical drug)
// - BN: Brand Name
// - SBDC: Branded Drug Component
// - SBDF: Branded Drug Form
// - SBD: Semantic Branded Drug
// - GPCK: Generic Pack
// - BPCK: Branded Pack
//
// ## Example Structure
//
// Ingredient (IN): Fluoxetine
//   └─ Precise Ingredient (PIN): Fluoxetine Hydrochloride
//       └─ SCDC: Fluoxetine 20 MG
//           └─ SCD: Fluoxetine 20 MG Oral Capsule
//               └─ SBD: Fluoxetine 20 MG Oral Capsule [Prozac]

use super::core::{Concept, ConceptRef, OntologyId, OntologySystem, SemanticType};
use std::collections::HashMap;
use std::fmt;

/// RxNorm term types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RxTermType {
    /// Base ingredient
    Ingredient,
    /// Precise ingredient (salt form)
    PreciseIngredient,
    /// Multiple ingredients
    MultipleIngredients,
    /// Semantic Clinical Drug Component (ingredient + strength)
    ClinicalDrugComponent,
    /// Semantic Clinical Drug Form (ingredient + dose form)
    ClinicalDrugForm,
    /// Semantic Clinical Drug (full prescribable)
    ClinicalDrug,
    /// Brand name
    BrandName,
    /// Branded Drug Component
    BrandedDrugComponent,
    /// Branded Drug Form
    BrandedDrugForm,
    /// Semantic Branded Drug
    BrandedDrug,
    /// Generic pack
    GenericPack,
    /// Branded pack
    BrandedPack,
    /// Dose form
    DoseForm,
    /// Dose form group
    DoseFormGroup,
}

impl RxTermType {
    pub fn code(&self) -> &'static str {
        match self {
            RxTermType::Ingredient => "IN",
            RxTermType::PreciseIngredient => "PIN",
            RxTermType::MultipleIngredients => "MIN",
            RxTermType::ClinicalDrugComponent => "SCDC",
            RxTermType::ClinicalDrugForm => "SCDF",
            RxTermType::ClinicalDrug => "SCD",
            RxTermType::BrandName => "BN",
            RxTermType::BrandedDrugComponent => "SBDC",
            RxTermType::BrandedDrugForm => "SBDF",
            RxTermType::BrandedDrug => "SBD",
            RxTermType::GenericPack => "GPCK",
            RxTermType::BrandedPack => "BPCK",
            RxTermType::DoseForm => "DF",
            RxTermType::DoseFormGroup => "DFG",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            RxTermType::Ingredient => "Ingredient",
            RxTermType::PreciseIngredient => "Precise Ingredient",
            RxTermType::MultipleIngredients => "Multiple Ingredients",
            RxTermType::ClinicalDrugComponent => "Clinical Drug Component",
            RxTermType::ClinicalDrugForm => "Clinical Drug Form",
            RxTermType::ClinicalDrug => "Semantic Clinical Drug",
            RxTermType::BrandName => "Brand Name",
            RxTermType::BrandedDrugComponent => "Branded Drug Component",
            RxTermType::BrandedDrugForm => "Branded Drug Form",
            RxTermType::BrandedDrug => "Semantic Branded Drug",
            RxTermType::GenericPack => "Generic Pack",
            RxTermType::BrandedPack => "Branded Pack",
            RxTermType::DoseForm => "Dose Form",
            RxTermType::DoseFormGroup => "Dose Form Group",
        }
    }

    /// Is this a prescribable term type?
    pub fn is_prescribable(&self) -> bool {
        matches!(
            self,
            RxTermType::ClinicalDrug
                | RxTermType::BrandedDrug
                | RxTermType::GenericPack
                | RxTermType::BrandedPack
        )
    }

    /// Is this an ingredient-level type?
    pub fn is_ingredient(&self) -> bool {
        matches!(
            self,
            RxTermType::Ingredient
                | RxTermType::PreciseIngredient
                | RxTermType::MultipleIngredients
        )
    }

    /// Parse from TTY code
    pub fn from_code(code: &str) -> Option<Self> {
        match code {
            "IN" => Some(RxTermType::Ingredient),
            "PIN" => Some(RxTermType::PreciseIngredient),
            "MIN" => Some(RxTermType::MultipleIngredients),
            "SCDC" => Some(RxTermType::ClinicalDrugComponent),
            "SCDF" => Some(RxTermType::ClinicalDrugForm),
            "SCD" => Some(RxTermType::ClinicalDrug),
            "BN" => Some(RxTermType::BrandName),
            "SBDC" => Some(RxTermType::BrandedDrugComponent),
            "SBDF" => Some(RxTermType::BrandedDrugForm),
            "SBD" => Some(RxTermType::BrandedDrug),
            "GPCK" => Some(RxTermType::GenericPack),
            "BPCK" => Some(RxTermType::BrandedPack),
            "DF" => Some(RxTermType::DoseForm),
            "DFG" => Some(RxTermType::DoseFormGroup),
            _ => None,
        }
    }
}

impl fmt::Display for RxTermType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.code())
    }
}

/// RxNorm relationship types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RxRelationType {
    /// Has ingredient
    HasIngredient,
    /// Has precise ingredient
    HasPreciseIngredient,
    /// Has dose form
    HasDoseForm,
    /// Has dose form group
    HasDoseFormGroup,
    /// Consists of (for packs)
    ConsistsOf,
    /// Contains (pack contains drug)
    Contains,
    /// Tradename of
    TradenameOf,
    /// Has tradename
    HasTradename,
    /// Isa (hierarchical)
    Isa,
    /// Ingredient of
    IngredientOf,
    /// Reformulated to
    ReformulatedTo,
    /// Quantified form of
    QuantifiedFormOf,
}

impl RxRelationType {
    pub fn code(&self) -> &'static str {
        match self {
            RxRelationType::HasIngredient => "has_ingredient",
            RxRelationType::HasPreciseIngredient => "has_precise_ingredient",
            RxRelationType::HasDoseForm => "has_dose_form",
            RxRelationType::HasDoseFormGroup => "has_doseformgroup",
            RxRelationType::ConsistsOf => "consists_of",
            RxRelationType::Contains => "contains",
            RxRelationType::TradenameOf => "tradename_of",
            RxRelationType::HasTradename => "has_tradename",
            RxRelationType::Isa => "isa",
            RxRelationType::IngredientOf => "ingredient_of",
            RxRelationType::ReformulatedTo => "reformulated_to",
            RxRelationType::QuantifiedFormOf => "quantified_form_of",
        }
    }
}

/// Strength representation for drug components
#[derive(Debug, Clone, PartialEq)]
pub struct DrugStrength {
    /// Numeric value
    pub value: f64,
    /// Unit (mg, mL, etc.)
    pub unit: String,
    /// Per unit (for concentrations, e.g., "per mL")
    pub per_unit: Option<String>,
}

impl DrugStrength {
    pub fn new(value: f64, unit: &str) -> Self {
        DrugStrength {
            value,
            unit: unit.to_string(),
            per_unit: None,
        }
    }

    pub fn concentration(value: f64, unit: &str, per_unit: &str) -> Self {
        DrugStrength {
            value,
            unit: unit.to_string(),
            per_unit: Some(per_unit.to_string()),
        }
    }
}

impl fmt::Display for DrugStrength {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref per) = self.per_unit {
            write!(f, "{} {}/{}", self.value, self.unit, per)
        } else {
            write!(f, "{} {}", self.value, self.unit)
        }
    }
}

/// RxNorm drug concept with additional metadata
#[derive(Debug, Clone)]
pub struct RxDrug {
    /// RxCUI (RxNorm Concept Unique Identifier)
    pub rxcui: String,
    /// Term type
    pub tty: RxTermType,
    /// Display name
    pub name: String,
    /// Normalized name
    pub normalized_name: Option<String>,
    /// Ingredients (for clinical drugs)
    pub ingredients: Vec<RxIngredient>,
    /// Dose form
    pub dose_form: Option<String>,
    /// Brand name (for branded drugs)
    pub brand_name: Option<String>,
    /// NDC codes (National Drug Codes)
    pub ndc_codes: Vec<String>,
    /// Is prescribable
    pub prescribable: bool,
    /// Is active (not obsolete)
    pub is_active: bool,
}

impl RxDrug {
    pub fn new(rxcui: &str, tty: RxTermType, name: &str) -> Self {
        RxDrug {
            rxcui: rxcui.to_string(),
            tty,
            name: name.to_string(),
            normalized_name: None,
            ingredients: Vec::new(),
            dose_form: None,
            brand_name: None,
            ndc_codes: Vec::new(),
            prescribable: tty.is_prescribable(),
            is_active: true,
        }
    }

    pub fn with_ingredient(mut self, ingredient: RxIngredient) -> Self {
        self.ingredients.push(ingredient);
        self
    }

    pub fn with_dose_form(mut self, form: &str) -> Self {
        self.dose_form = Some(form.to_string());
        self
    }

    pub fn with_brand(mut self, brand: &str) -> Self {
        self.brand_name = Some(brand.to_string());
        self
    }

    pub fn with_ndc(mut self, ndc: &str) -> Self {
        self.ndc_codes.push(ndc.to_string());
        self
    }

    /// Get as OntologyId
    pub fn to_ontology_id(&self) -> OntologyId {
        OntologyId::rxnorm(&self.rxcui)
    }

    /// Get as ConceptRef
    pub fn to_concept_ref(&self) -> ConceptRef {
        ConceptRef::with_display(self.to_ontology_id(), &self.name)
    }

    /// Convert to generic Concept
    pub fn to_concept(&self) -> Concept {
        let semantic_type = if self.tty.is_ingredient() {
            SemanticType::Ingredient
        } else {
            SemanticType::ClinicalDrug
        };

        let mut concept = Concept::new(self.to_ontology_id(), &self.name, &self.name)
            .with_semantic_type(semantic_type);

        // Add synonyms
        if let Some(ref norm) = self.normalized_name {
            concept = concept.with_synonym(norm);
        }

        concept
    }
}

/// Ingredient within a drug
#[derive(Debug, Clone)]
pub struct RxIngredient {
    /// Ingredient RxCUI
    pub rxcui: String,
    /// Ingredient name
    pub name: String,
    /// Strength
    pub strength: Option<DrugStrength>,
    /// Is base ingredient (vs precise/salt form)
    pub is_base: bool,
}

impl RxIngredient {
    pub fn new(rxcui: &str, name: &str) -> Self {
        RxIngredient {
            rxcui: rxcui.to_string(),
            name: name.to_string(),
            strength: None,
            is_base: true,
        }
    }

    pub fn with_strength(mut self, strength: DrugStrength) -> Self {
        self.strength = Some(strength);
        self
    }

    pub fn precise(mut self) -> Self {
        self.is_base = false;
        self
    }
}

/// RxNorm drug registry
pub struct RxNormRegistry {
    /// Drugs by RxCUI
    drugs: HashMap<String, RxDrug>,
    /// Ingredients by RxCUI
    ingredients: HashMap<String, RxIngredient>,
    /// Index: ingredient -> drugs containing it
    ingredient_to_drugs: HashMap<String, Vec<String>>,
    /// Index: brand name -> branded drugs
    brand_to_drugs: HashMap<String, Vec<String>>,
    /// NDC to RxCUI mapping
    ndc_to_rxcui: HashMap<String, String>,
}

impl RxNormRegistry {
    pub fn new() -> Self {
        RxNormRegistry {
            drugs: HashMap::new(),
            ingredients: HashMap::new(),
            ingredient_to_drugs: HashMap::new(),
            brand_to_drugs: HashMap::new(),
            ndc_to_rxcui: HashMap::new(),
        }
    }

    /// Register a drug
    pub fn register_drug(&mut self, drug: RxDrug) {
        // Index by ingredient
        for ing in &drug.ingredients {
            self.ingredient_to_drugs
                .entry(ing.rxcui.clone())
                .or_default()
                .push(drug.rxcui.clone());
        }

        // Index by brand
        if let Some(ref brand) = drug.brand_name {
            self.brand_to_drugs
                .entry(brand.clone())
                .or_default()
                .push(drug.rxcui.clone());
        }

        // Index NDCs
        for ndc in &drug.ndc_codes {
            self.ndc_to_rxcui.insert(ndc.clone(), drug.rxcui.clone());
        }

        self.drugs.insert(drug.rxcui.clone(), drug);
    }

    /// Register an ingredient
    pub fn register_ingredient(&mut self, ingredient: RxIngredient) {
        self.ingredients
            .insert(ingredient.rxcui.clone(), ingredient);
    }

    /// Lookup drug by RxCUI
    pub fn get_drug(&self, rxcui: &str) -> Option<&RxDrug> {
        self.drugs.get(rxcui)
    }

    /// Lookup ingredient by RxCUI
    pub fn get_ingredient(&self, rxcui: &str) -> Option<&RxIngredient> {
        self.ingredients.get(rxcui)
    }

    /// Lookup by NDC
    pub fn get_by_ndc(&self, ndc: &str) -> Option<&RxDrug> {
        self.ndc_to_rxcui
            .get(ndc)
            .and_then(|cui| self.drugs.get(cui))
    }

    /// Find drugs containing an ingredient
    pub fn find_by_ingredient(&self, ingredient_rxcui: &str) -> Vec<&RxDrug> {
        self.ingredient_to_drugs
            .get(ingredient_rxcui)
            .map(|rxcuis| {
                rxcuis
                    .iter()
                    .filter_map(|cui| self.drugs.get(cui))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find drugs by brand name
    pub fn find_by_brand(&self, brand: &str) -> Vec<&RxDrug> {
        self.brand_to_drugs
            .get(brand)
            .map(|rxcuis| {
                rxcuis
                    .iter()
                    .filter_map(|cui| self.drugs.get(cui))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Find prescribable drugs
    pub fn find_prescribable(&self) -> Vec<&RxDrug> {
        self.drugs.values().filter(|d| d.prescribable).collect()
    }

    /// Search drugs by name (substring match)
    pub fn search_by_name(&self, query: &str) -> Vec<&RxDrug> {
        let query_lower = query.to_lowercase();
        self.drugs
            .values()
            .filter(|d| d.name.to_lowercase().contains(&query_lower))
            .collect()
    }
}

impl Default for RxNormRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Common Drug Examples (for testing/compilation)
// =============================================================================

/// Create common drug examples
pub fn example_drugs() -> RxNormRegistry {
    let mut registry = RxNormRegistry::new();

    // Metformin (common diabetes drug)
    let metformin_in = RxIngredient::new("6809", "Metformin");
    registry.register_ingredient(metformin_in.clone());

    let metformin_500 = RxDrug::new(
        "861004",
        RxTermType::ClinicalDrug,
        "Metformin 500 MG Oral Tablet",
    )
    .with_ingredient(
        RxIngredient::new("6809", "Metformin").with_strength(DrugStrength::new(500.0, "MG")),
    )
    .with_dose_form("Oral Tablet");
    registry.register_drug(metformin_500);

    // Atorvastatin (statin)
    let atorvastatin_in = RxIngredient::new("83367", "Atorvastatin");
    registry.register_ingredient(atorvastatin_in);

    let lipitor = RxDrug::new(
        "617312",
        RxTermType::BrandedDrug,
        "Atorvastatin 20 MG Oral Tablet [Lipitor]",
    )
    .with_ingredient(
        RxIngredient::new("83367", "Atorvastatin").with_strength(DrugStrength::new(20.0, "MG")),
    )
    .with_dose_form("Oral Tablet")
    .with_brand("Lipitor");
    registry.register_drug(lipitor);

    // Warfarin (anticoagulant - important for DDI)
    let warfarin_in = RxIngredient::new("11289", "Warfarin");
    registry.register_ingredient(warfarin_in);

    let warfarin = RxDrug::new(
        "855288",
        RxTermType::ClinicalDrug,
        "Warfarin Sodium 5 MG Oral Tablet",
    )
    .with_ingredient(
        RxIngredient::new("11289", "Warfarin").with_strength(DrugStrength::new(5.0, "MG")),
    )
    .with_dose_form("Oral Tablet");
    registry.register_drug(warfarin);

    registry
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rx_term_type() {
        assert_eq!(RxTermType::Ingredient.code(), "IN");
        assert!(RxTermType::ClinicalDrug.is_prescribable());
        assert!(!RxTermType::Ingredient.is_prescribable());
        assert!(RxTermType::Ingredient.is_ingredient());
    }

    #[test]
    fn test_drug_strength() {
        let s = DrugStrength::new(500.0, "MG");
        assert_eq!(s.to_string(), "500 MG");

        let conc = DrugStrength::concentration(10.0, "MG", "ML");
        assert_eq!(conc.to_string(), "10 MG/ML");
    }

    #[test]
    fn test_rx_drug_creation() {
        let drug = RxDrug::new(
            "861004",
            RxTermType::ClinicalDrug,
            "Metformin 500 MG Oral Tablet",
        )
        .with_ingredient(
            RxIngredient::new("6809", "Metformin").with_strength(DrugStrength::new(500.0, "MG")),
        )
        .with_dose_form("Oral Tablet");

        assert_eq!(drug.rxcui, "861004");
        assert!(drug.prescribable);
        assert_eq!(drug.ingredients.len(), 1);
        assert_eq!(drug.ingredients[0].name, "Metformin");
    }

    #[test]
    fn test_rx_drug_to_concept() {
        let drug = RxDrug::new(
            "861004",
            RxTermType::ClinicalDrug,
            "Metformin 500 MG Oral Tablet",
        );
        let concept = drug.to_concept();

        assert_eq!(concept.id.system, OntologySystem::RxNorm);
        assert_eq!(concept.id.code, "861004");
        assert_eq!(concept.semantic_type, SemanticType::ClinicalDrug);
    }

    #[test]
    fn test_registry() {
        let registry = example_drugs();

        // Find by RxCUI
        let metformin = registry.get_drug("861004").unwrap();
        assert!(metformin.name.contains("Metformin"));

        // Find by ingredient
        let metformin_drugs = registry.find_by_ingredient("6809");
        assert!(!metformin_drugs.is_empty());

        // Find by brand
        let lipitor_drugs = registry.find_by_brand("Lipitor");
        assert!(!lipitor_drugs.is_empty());
    }

    #[test]
    fn test_search() {
        let registry = example_drugs();

        let results = registry.search_by_name("metformin");
        assert!(!results.is_empty());
        assert!(results[0].name.to_lowercase().contains("metformin"));
    }

    #[test]
    fn test_prescribable() {
        let registry = example_drugs();

        let prescribable = registry.find_prescribable();
        assert!(!prescribable.is_empty());
        assert!(prescribable.iter().all(|d| d.prescribable));
    }
}
