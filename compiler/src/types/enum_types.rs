// Week 27: Type System Support for Algebraic Data Types (Enums)
//
// This module provides the type-level representation and environment
// for enum types used in clinical state modeling.

use std::collections::HashMap;

/// Type representation for enums
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct EnumType {
    /// Enum name (e.g., "Response", "ToxicityGrade")
    pub name: String,
    /// Number of variants
    pub num_variants: usize,
}

impl EnumType {
    pub fn new(name: String, num_variants: usize) -> Self {
        EnumType { name, num_variants }
    }
}

/// Complete information about an enum declaration
#[derive(Debug, Clone)]
pub struct EnumInfo {
    /// Enum name (e.g., "Response")
    pub name: String,
    /// Ordered list of variant names (e.g., ["CR", "PR", "SD", "PD"])
    pub variants: Vec<String>,
}

impl EnumInfo {
    pub fn new(name: String, variants: Vec<String>) -> Self {
        EnumInfo { name, variants }
    }

    /// Get the index of a variant (for backend integer encoding)
    pub fn variant_index(&self, variant_name: &str) -> Option<usize> {
        self.variants.iter().position(|v| v == variant_name)
    }

    /// Get the number of variants
    pub fn num_variants(&self) -> usize {
        self.variants.len()
    }

    /// Check if a variant exists
    pub fn has_variant(&self, variant_name: &str) -> bool {
        self.variants.iter().any(|v| v == variant_name)
    }

    /// Get variant name by index
    pub fn variant_name(&self, index: usize) -> Option<&str> {
        self.variants.get(index).map(|s| s.as_str())
    }
}

/// Environment tracking all enum declarations in a module
#[derive(Debug, Clone, Default)]
pub struct EnumEnv {
    /// Map from enum name to enum information
    pub enums: HashMap<String, EnumInfo>,

    /// Reverse map: (enum_name, variant_name) -> integer code
    /// This is used for backend code generation
    pub variant_codes: HashMap<(String, String), usize>,
}

impl EnumEnv {
    /// Create a new empty enum environment
    pub fn new() -> Self {
        EnumEnv {
            enums: HashMap::new(),
            variant_codes: HashMap::new(),
        }
    }

    /// Register an enum declaration
    pub fn register_enum(&mut self, info: EnumInfo) {
        // Build variant code map
        for (index, variant_name) in info.variants.iter().enumerate() {
            self.variant_codes
                .insert((info.name.clone(), variant_name.clone()), index);
        }

        self.enums.insert(info.name.clone(), info);
    }

    /// Look up enum by name
    pub fn get_enum(&self, name: &str) -> Option<&EnumInfo> {
        self.enums.get(name)
    }

    /// Check if an enum exists
    pub fn has_enum(&self, name: &str) -> bool {
        self.enums.contains_key(name)
    }

    /// Get the integer code for a variant
    pub fn variant_code(&self, enum_name: &str, variant_name: &str) -> Option<usize> {
        self.variant_codes
            .get(&(enum_name.to_string(), variant_name.to_string()))
            .copied()
    }

    /// Verify that a variant belongs to the specified enum
    pub fn verify_variant(&self, enum_name: &str, variant_name: &str) -> Result<(), String> {
        match self.get_enum(enum_name) {
            None => Err(format!("Unknown enum: {}", enum_name)),
            Some(info) => {
                if info.has_variant(variant_name) {
                    Ok(())
                } else {
                    Err(format!(
                        "Enum {} does not have variant {}. Valid variants: [{}]",
                        enum_name,
                        variant_name,
                        info.variants.join(", ")
                    ))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enum_info_creation() {
        let info = EnumInfo::new(
            "Response".to_string(),
            vec![
                "CR".to_string(),
                "PR".to_string(),
                "SD".to_string(),
                "PD".to_string(),
            ],
        );

        assert_eq!(info.name, "Response");
        assert_eq!(info.num_variants(), 4);
        assert!(info.has_variant("CR"));
        assert!(info.has_variant("PD"));
        assert!(!info.has_variant("Unknown"));
    }

    #[test]
    fn test_variant_index() {
        let info = EnumInfo::new(
            "Response".to_string(),
            vec![
                "CR".to_string(),
                "PR".to_string(),
                "SD".to_string(),
                "PD".to_string(),
            ],
        );

        assert_eq!(info.variant_index("CR"), Some(0));
        assert_eq!(info.variant_index("PR"), Some(1));
        assert_eq!(info.variant_index("SD"), Some(2));
        assert_eq!(info.variant_index("PD"), Some(3));
        assert_eq!(info.variant_index("Unknown"), None);
    }

    #[test]
    fn test_enum_env_registration() {
        let mut env = EnumEnv::new();

        let response_info = EnumInfo::new(
            "Response".to_string(),
            vec![
                "CR".to_string(),
                "PR".to_string(),
                "SD".to_string(),
                "PD".to_string(),
            ],
        );

        env.register_enum(response_info);

        assert!(env.has_enum("Response"));
        assert!(!env.has_enum("ToxicityGrade"));
    }

    #[test]
    fn test_variant_code_lookup() {
        let mut env = EnumEnv::new();

        let response_info = EnumInfo::new(
            "Response".to_string(),
            vec![
                "CR".to_string(),
                "PR".to_string(),
                "SD".to_string(),
                "PD".to_string(),
            ],
        );

        env.register_enum(response_info);

        assert_eq!(env.variant_code("Response", "CR"), Some(0));
        assert_eq!(env.variant_code("Response", "PR"), Some(1));
        assert_eq!(env.variant_code("Response", "SD"), Some(2));
        assert_eq!(env.variant_code("Response", "PD"), Some(3));
        assert_eq!(env.variant_code("Response", "Unknown"), None);
        assert_eq!(env.variant_code("UnknownEnum", "CR"), None);
    }

    #[test]
    fn test_verify_variant() {
        let mut env = EnumEnv::new();

        let response_info = EnumInfo::new(
            "Response".to_string(),
            vec![
                "CR".to_string(),
                "PR".to_string(),
                "SD".to_string(),
                "PD".to_string(),
            ],
        );

        env.register_enum(response_info);

        assert!(env.verify_variant("Response", "CR").is_ok());
        assert!(env.verify_variant("Response", "PR").is_ok());
        assert!(env.verify_variant("Response", "Unknown").is_err());
        assert!(env.verify_variant("UnknownEnum", "CR").is_err());
    }

    #[test]
    fn test_toxicity_grade_enum() {
        let mut env = EnumEnv::new();

        let toxicity_info = EnumInfo::new(
            "ToxicityGrade".to_string(),
            vec![
                "G0".to_string(),
                "G1".to_string(),
                "G2".to_string(),
                "G3".to_string(),
                "G4".to_string(),
            ],
        );

        env.register_enum(toxicity_info);

        assert_eq!(env.variant_code("ToxicityGrade", "G0"), Some(0));
        assert_eq!(env.variant_code("ToxicityGrade", "G4"), Some(4));

        let info = env.get_enum("ToxicityGrade").unwrap();
        assert_eq!(info.num_variants(), 5);
        assert_eq!(info.variant_name(0), Some("G0"));
        assert_eq!(info.variant_name(4), Some("G4"));
    }
}
