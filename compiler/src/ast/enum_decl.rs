// Week 27: Algebraic Data Types (Enums) for Clinical States
//
// Enums represent discrete clinical concepts like:
// - Response = { CR, PR, SD, PD }
// - ToxicityGrade = { G0, G1, G2, G3, G4 }
// - ECOG = { ECOG0, ECOG1, ECOG2, ECOG3, ECOG4 }
//
// Week 27 supports nullary variants only (no payloads).
// Future extensions can add variant fields.

use crate::ast::Ident;
use serde::{Deserialize, Serialize};

/// Top-level enum declaration
///
/// Example:
/// ```medlang
/// enum Response {
///   CR;
///   PR;
///   SD;
///   PD;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumDecl {
    /// Enum name (e.g., "Response")
    pub name: Ident,
    /// List of variants (nullary constructors)
    pub variants: Vec<EnumVariant>,
}

/// A single enum variant (nullary constructor)
///
/// Example: `CR;` in `enum Response { CR; PR; SD; PD; }`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnumVariant {
    /// Variant name (e.g., "CR", "PR", "SD", "PD")
    pub name: Ident,
    // Week 27: no fields
    // Future: pub fields: Vec<(Ident, Type)>
}

impl EnumDecl {
    /// Create a new enum declaration
    pub fn new(name: Ident, variants: Vec<EnumVariant>) -> Self {
        EnumDecl { name, variants }
    }

    /// Get variant by name
    pub fn get_variant(&self, variant_name: &str) -> Option<&EnumVariant> {
        self.variants.iter().find(|v| v.name == variant_name)
    }

    /// Get variant index (for backend integer encoding)
    pub fn variant_index(&self, variant_name: &str) -> Option<usize> {
        self.variants.iter().position(|v| v.name == variant_name)
    }

    /// Number of variants
    pub fn num_variants(&self) -> usize {
        self.variants.len()
    }
}

impl EnumVariant {
    /// Create a new nullary variant
    pub fn new(name: Ident) -> Self {
        EnumVariant { name }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_decl_creation() {
        let decl = EnumDecl::new(
            "Response".to_string(),
            vec![
                EnumVariant::new("CR".to_string()),
                EnumVariant::new("PR".to_string()),
                EnumVariant::new("SD".to_string()),
                EnumVariant::new("PD".to_string()),
            ],
        );

        assert_eq!(decl.name, "Response");
        assert_eq!(decl.num_variants(), 4);
    }

    #[test]
    fn test_variant_lookup() {
        let decl = EnumDecl::new(
            "Response".to_string(),
            vec![
                EnumVariant::new("CR".to_string()),
                EnumVariant::new("PR".to_string()),
                EnumVariant::new("SD".to_string()),
                EnumVariant::new("PD".to_string()),
            ],
        );

        assert!(decl.get_variant("CR").is_some());
        assert!(decl.get_variant("PR").is_some());
        assert!(decl.get_variant("Unknown").is_none());
    }

    #[test]
    fn test_variant_index() {
        let decl = EnumDecl::new(
            "Response".to_string(),
            vec![
                EnumVariant::new("CR".to_string()),
                EnumVariant::new("PR".to_string()),
                EnumVariant::new("SD".to_string()),
                EnumVariant::new("PD".to_string()),
            ],
        );

        assert_eq!(decl.variant_index("CR"), Some(0));
        assert_eq!(decl.variant_index("PR"), Some(1));
        assert_eq!(decl.variant_index("SD"), Some(2));
        assert_eq!(decl.variant_index("PD"), Some(3));
        assert_eq!(decl.variant_index("Unknown"), None);
    }

    #[test]
    fn test_toxicity_grade_enum() {
        let decl = EnumDecl::new(
            "ToxicityGrade".to_string(),
            vec![
                EnumVariant::new("G0".to_string()),
                EnumVariant::new("G1".to_string()),
                EnumVariant::new("G2".to_string()),
                EnumVariant::new("G3".to_string()),
                EnumVariant::new("G4".to_string()),
            ],
        );

        assert_eq!(decl.name, "ToxicityGrade");
        assert_eq!(decl.num_variants(), 5);
        assert_eq!(decl.variant_index("G0"), Some(0));
        assert_eq!(decl.variant_index("G4"), Some(4));
    }
}
