// Week 54: Units of Measure System
//
// Compile-time dimensional analysis for MedLang, following Kennedy's approach
// with UCUM (Unified Code for Units of Measure) for healthcare interoperability.
//
// ## Architecture
//
// 1. **Dimension** (`dimension.rs`)
//    - Base dimensions (Length, Mass, Time, etc.)
//    - Dimension algebra (multiplication, division, powers)
//    - Medical extensions (International Units)
//
// 2. **Unit** (`unit.rs`)
//    - Concrete units with conversion factors
//    - Unit registry with UCUM codes
//    - Compound unit construction
//
// 3. **Quantity** (`quantity.rs`)
//    - Runtime value + unit pairs
//    - Type-safe arithmetic operations
//    - Medical quantity helpers
//
// 4. **Types** (`types.rs`)
//    - Compile-time dimension types
//    - Dimension checker for type system
//    - Unit type annotations
//
// 5. **UCUM** (`ucum.rs`)
//    - UCUM parser and validator
//    - Standard prefixes and base units
//    - Healthcare unit interoperability
//
// ## Compilation Pipeline
//
// ```text
// Source with unit annotations
//         ↓ parse
// AST with UnitAnnotation nodes
//         ↓ type_check
// Dimensional analysis (DimensionChecker)
//         ↓ unit_erasure
// Plain numeric types (Float)
//         ↓ codegen
// Zero-cost runtime
// ```
//
// ## Example
//
// ```medlang
// // Type annotations with units
// let dose: Quantity<mg> = 100.0 mg;
// let weight: Quantity<kg> = 70.0 kg;
// let dose_per_kg: Quantity<mg/kg> = dose / weight;
//
// // Compile error: dimension mismatch
// // let bad = dose + weight;  // Error: cannot add mg + kg
//
// // At runtime, all units are erased:
// // dose_per_kg.value == 100.0 / 70.0
// ```
//
// ## Medical-Specific Features
//
// - **IU (International Units)**: Biologically standardized, non-convertible to mass
// - **Clearance normalization**: mL/min vs mL/min/1.73m² for GFR
// - **Dose normalization**: mg/kg (weight-based) vs mg/m² (BSA-based)
// - **Annotations**: mol/L{creatinine} for semantic distinction

pub mod dimension;
pub mod quantity;
pub mod types;
pub mod ucum;
pub mod unit;

// Re-exports
pub use dimension::{BaseDimension, Dimension};
pub use quantity::{medical, Quantity, QuantityBuilder, QuantityError};
pub use types::{
    DimensionCheckResult, DimensionChecker, DimensionError, DimensionType, UnitAnnotation,
    UnitType, UnitTypeRegistry,
};
pub use ucum::{ParsedUnit, UcumParseError, UcumParser, UcumRegistry};
pub use unit::{CompoundUnitBuilder, Unit, UnitParseError, UnitRegistry};

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a standard unit registry
pub fn standard_units() -> UnitRegistry {
    UnitRegistry::standard()
}

/// Create a standard UCUM parser
pub fn ucum_parser() -> UcumParser {
    UcumParser::new()
}

/// Create a standard unit type registry
pub fn unit_type_registry() -> UnitTypeRegistry {
    UnitTypeRegistry::standard()
}

/// Validate a UCUM unit string
pub fn validate_ucum(unit_str: &str) -> Result<(), UcumParseError> {
    UcumParser::new().validate(unit_str)
}

/// Parse a quantity string (e.g., "100 mg")
pub fn parse_quantity(s: &str) -> Result<Quantity, QuantityError> {
    QuantityBuilder::standard().parse(s)
}

// =============================================================================
// Standard Dimensions (re-export)
// =============================================================================

pub mod standard_dims {
    pub use super::dimension::standard::*;
    pub use super::types::standard_dims::*;
}

// =============================================================================
// Integration Tests
// =============================================================================

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_quantity() {
        // Parse a quantity
        let dose = parse_quantity("100 mg").unwrap();
        assert!((dose.value - 100.0).abs() < 1e-10);

        // Convert units
        let registry = standard_units();
        let dose_g = dose.convert_to(registry.get("g").unwrap()).unwrap();
        assert!((dose_g.value - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_dimension_checking() {
        let mut checker = DimensionChecker::new();
        let registry = unit_type_registry();

        let mg_type = registry.lookup("mg").unwrap();
        let ml_type = registry.lookup("mL").unwrap();

        // Same dimensions should match
        let kg_type = registry.lookup("kg").unwrap();
        assert_eq!(
            checker.check_compatible(&mg_type, &kg_type),
            DimensionCheckResult::Match
        );

        // Different dimensions should mismatch
        match checker.check_compatible(&mg_type, &ml_type) {
            DimensionCheckResult::Mismatch { .. } => (),
            _ => panic!("Expected mismatch"),
        }
    }

    #[test]
    fn test_ucum_validation() {
        assert!(validate_ucum("mg").is_ok());
        assert!(validate_ucum("mL/min").is_ok());
        assert!(validate_ucum("mg/kg").is_ok());
        assert!(validate_ucum("[IU]").is_ok());
        assert!(validate_ucum("mol/L{glucose}").is_ok());
        assert!(validate_ucum("invalid").is_err());
    }

    #[test]
    fn test_medical_quantities() {
        let dose = medical::dose_mg(500.0);
        let weight = medical::weight_kg(70.0);

        // Division produces correct dimension
        let dose_per_kg = dose / weight;
        assert!((dose_per_kg.value - 500.0 / 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_clearance_units() {
        let registry = standard_units();

        let cl_ml_min = registry.get("mL/min").unwrap();
        let cl_l_h = registry.get("L/h").unwrap();

        // Both are clearance units (same dimension)
        assert!(cl_ml_min.is_compatible(cl_l_h));

        // GFR unit is different (normalized to BSA)
        let gfr_unit = registry.get("mL/min/1.73m2").unwrap();
        assert!(!cl_ml_min.is_compatible(gfr_unit));
    }

    #[test]
    fn test_iu_non_convertible() {
        let registry = standard_units();

        let iu = registry.get("[IU]").unwrap();
        let mg = registry.get("mg").unwrap();

        // IU has different dimension than mass
        assert!(!iu.is_compatible(mg));

        // Cannot convert IU to mg
        let iu_quantity = Quantity::new(100.0, iu.clone());
        assert!(iu_quantity.convert_to(mg).is_err());
    }

    #[test]
    fn test_compound_unit_dimension_algebra() {
        let checker = DimensionChecker::new();
        let registry = unit_type_registry();

        let mass_type = registry.lookup("mg").unwrap();
        let time_type = registry.lookup("h").unwrap();

        // mg / h should produce mass/time dimension
        let rate_type = checker.infer_div(&mass_type, &time_type);
        assert!(matches!(rate_type, UnitType::Quantity(_)));
    }

    #[test]
    fn test_dimensionless_ratio() {
        let checker = DimensionChecker::new();
        let registry = unit_type_registry();

        let mg1 = registry.lookup("mg").unwrap();
        let mg2 = registry.lookup("kg").unwrap(); // Same dimension

        // mass / mass = dimensionless
        let ratio = checker.infer_div(&mg1, &mg2);
        assert!(ratio.is_dimensionless());
    }

    #[test]
    fn test_auc_dimension() {
        let conc = medical::conc_ng_ml(100.0);
        let time = medical::time_h(24.0);

        let auc = conc * time;

        // AUC has concentration × time dimension
        assert!(!auc.unit.dimension.is_dimensionless());
    }
}
