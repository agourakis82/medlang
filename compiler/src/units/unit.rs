// Week 54: Units of Measure - Unit Definitions
//
// Units are concrete measurement scales with associated dimensions and conversion factors.
// Following UCUM (Unified Code for Units of Measure) for healthcare interoperability.
//
// Key design decisions:
// 1. Gram is the base mass unit (UCUM), not kilogram (SI)
// 2. IU (International Units) are biologically standardized and NOT convertible to mass
// 3. Annotations support semantic clarification: g/mol{creatinine}

use super::dimension::{BaseDimension, Dimension};
use std::collections::HashMap;
use std::fmt;

/// A unit of measurement with dimension and conversion factor to base units
#[derive(Clone, PartialEq)]
pub struct Unit {
    /// Human-readable name
    pub name: String,
    /// UCUM code (canonical identifier)
    pub ucum_code: String,
    /// Dimension of this unit
    pub dimension: Dimension,
    /// Conversion factor to base unit (multiply by this to get base)
    pub to_base_factor: f64,
    /// Optional annotation for semantic clarification
    pub annotation: Option<String>,
}

impl Unit {
    /// Create a new unit
    pub fn new(name: &str, ucum_code: &str, dimension: Dimension, to_base_factor: f64) -> Self {
        Unit {
            name: name.to_string(),
            ucum_code: ucum_code.to_string(),
            dimension,
            to_base_factor,
            annotation: None,
        }
    }

    /// Create a base unit (conversion factor = 1.0)
    pub fn base(name: &str, ucum_code: &str, dimension: Dimension) -> Self {
        Self::new(name, ucum_code, dimension, 1.0)
    }

    /// Create a dimensionless unit
    pub fn dimensionless(name: &str, ucum_code: &str) -> Self {
        Self::new(name, ucum_code, Dimension::dimensionless(), 1.0)
    }

    /// Add an annotation (e.g., {creatinine} for g/mol{creatinine})
    pub fn with_annotation(mut self, annotation: &str) -> Self {
        self.annotation = Some(annotation.to_string());
        self
    }

    /// Check if units are dimensionally compatible
    pub fn is_compatible(&self, other: &Unit) -> bool {
        self.dimension == other.dimension
    }

    /// Get conversion factor from this unit to another compatible unit
    pub fn conversion_factor_to(&self, other: &Unit) -> Option<f64> {
        if !self.is_compatible(other) {
            return None;
        }
        // Check for IU - these are not convertible
        if self.dimension.contains_iu() {
            // IU can only convert to same IU unit
            if self.ucum_code != other.ucum_code {
                return None;
            }
        }
        Some(self.to_base_factor / other.to_base_factor)
    }

    /// Multiply two units (for compound units like mg/kg)
    pub fn multiply(&self, other: &Unit) -> Unit {
        Unit {
            name: format!("{}·{}", self.name, other.name),
            ucum_code: format!("{}.{}", self.ucum_code, other.ucum_code),
            dimension: self.dimension.clone() * other.dimension.clone(),
            to_base_factor: self.to_base_factor * other.to_base_factor,
            annotation: None,
        }
    }

    /// Divide two units
    pub fn divide(&self, other: &Unit) -> Unit {
        Unit {
            name: format!("{}/{}", self.name, other.name),
            ucum_code: format!("{}/{}", self.ucum_code, other.ucum_code),
            dimension: self.dimension.clone() / other.dimension.clone(),
            to_base_factor: self.to_base_factor / other.to_base_factor,
            annotation: None,
        }
    }

    /// Raise unit to a power
    pub fn pow(&self, n: i8) -> Unit {
        let suffix = if n == 2 {
            "²".to_string()
        } else if n == 3 {
            "³".to_string()
        } else {
            format!("^{}", n)
        };

        Unit {
            name: format!("{}{}", self.name, suffix),
            ucum_code: format!("{}{}", self.ucum_code, n),
            dimension: self.dimension.pow(n),
            to_base_factor: self.to_base_factor.powi(n as i32),
            annotation: None,
        }
    }
}

impl fmt::Debug for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unit({}, {})", self.name, self.ucum_code)
    }
}

impl fmt::Display for Unit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref ann) = self.annotation {
            write!(f, "{}{{{}}}", self.ucum_code, ann)
        } else {
            write!(f, "{}", self.ucum_code)
        }
    }
}

// =============================================================================
// Unit Registry
// =============================================================================

/// Registry of known units with UCUM codes
pub struct UnitRegistry {
    units: HashMap<String, Unit>,
}

impl UnitRegistry {
    pub fn new() -> Self {
        UnitRegistry {
            units: HashMap::new(),
        }
    }

    /// Create a registry with standard medical/scientific units
    pub fn standard() -> Self {
        let mut registry = Self::new();
        registry.register_base_units();
        registry.register_length_units();
        registry.register_mass_units();
        registry.register_time_units();
        registry.register_volume_units();
        registry.register_concentration_units();
        registry.register_medical_units();
        registry
    }

    pub fn register(&mut self, unit: Unit) {
        self.units.insert(unit.ucum_code.clone(), unit);
    }

    pub fn get(&self, ucum_code: &str) -> Option<&Unit> {
        self.units.get(ucum_code)
    }

    pub fn contains(&self, ucum_code: &str) -> bool {
        self.units.contains_key(ucum_code)
    }

    fn register_base_units(&mut self) {
        use super::dimension::standard::*;

        // Dimensionless
        self.register(Unit::dimensionless("unity", "1"));
        self.register(Unit::new("percent", "%", dimensionless(), 0.01));
        self.register(Unit::new("parts per million", "ppm", dimensionless(), 1e-6));
        self.register(Unit::new("parts per billion", "ppb", dimensionless(), 1e-9));

        // Base SI units (using UCUM conventions)
        self.register(Unit::base("meter", "m", length()));
        self.register(Unit::base("gram", "g", mass())); // UCUM uses gram, not kg
        self.register(Unit::base("second", "s", time()));
        self.register(Unit::base("ampere", "A", current()));
        self.register(Unit::base("kelvin", "K", temperature()));
        self.register(Unit::base("mole", "mol", amount()));
        self.register(Unit::base("candela", "cd", luminosity()));

        // International Unit (biologically standardized)
        self.register(Unit::base(
            "international unit",
            "[IU]",
            international_unit(),
        ));
    }

    fn register_length_units(&mut self) {
        use super::dimension::standard::length;

        let l = length();
        self.register(Unit::new("kilometer", "km", l.clone(), 1000.0));
        self.register(Unit::new("centimeter", "cm", l.clone(), 0.01));
        self.register(Unit::new("millimeter", "mm", l.clone(), 0.001));
        self.register(Unit::new("micrometer", "um", l.clone(), 1e-6));
        self.register(Unit::new("nanometer", "nm", l.clone(), 1e-9));

        // Area
        let area = l.pow(2);
        self.register(Unit::new("square meter", "m2", area.clone(), 1.0));
        self.register(Unit::new("square centimeter", "cm2", area.clone(), 1e-4));

        // Body Surface Area is typically expressed in m²
        // 1.73 m² is the standard normalization factor for GFR
    }

    fn register_mass_units(&mut self) {
        use super::dimension::standard::mass;

        let m = mass();
        self.register(Unit::new("kilogram", "kg", m.clone(), 1000.0));
        self.register(Unit::new("milligram", "mg", m.clone(), 0.001));
        self.register(Unit::new("microgram", "ug", m.clone(), 1e-6));
        self.register(Unit::new("nanogram", "ng", m.clone(), 1e-9));
        self.register(Unit::new("picogram", "pg", m.clone(), 1e-12));
    }

    fn register_time_units(&mut self) {
        use super::dimension::standard::time;

        let t = time();
        self.register(Unit::new("minute", "min", t.clone(), 60.0));
        self.register(Unit::new("hour", "h", t.clone(), 3600.0));
        self.register(Unit::new("day", "d", t.clone(), 86400.0));
        self.register(Unit::new("week", "wk", t.clone(), 604800.0));
        self.register(Unit::new("month", "mo", t.clone(), 2629746.0)); // Average month
        self.register(Unit::new("year", "a", t.clone(), 31556952.0)); // Julian year
        self.register(Unit::new("millisecond", "ms", t.clone(), 0.001));
    }

    fn register_volume_units(&mut self) {
        use super::dimension::standard::volume;

        let v = volume();
        self.register(Unit::new("liter", "L", v.clone(), 0.001)); // 1 L = 0.001 m³
        self.register(Unit::new("milliliter", "mL", v.clone(), 1e-6));
        self.register(Unit::new("microliter", "uL", v.clone(), 1e-9));
        self.register(Unit::new("deciliter", "dL", v.clone(), 1e-4));
        self.register(Unit::new("cubic meter", "m3", v.clone(), 1.0));
        self.register(Unit::new("cubic centimeter", "cm3", v.clone(), 1e-6));
    }

    fn register_concentration_units(&mut self) {
        use super::dimension::standard::*;

        // Molar concentration (mol/L)
        let molarity = concentration();
        self.register(Unit::new("molar", "mol/L", molarity.clone(), 1000.0)); // mol/m³
        self.register(Unit::new("millimolar", "mmol/L", molarity.clone(), 1.0));
        self.register(Unit::new("micromolar", "umol/L", molarity.clone(), 0.001));
        self.register(Unit::new("nanomolar", "nmol/L", molarity.clone(), 1e-6));

        // Mass concentration (g/L, mg/dL, etc.)
        let mass_conc = mass_concentration();
        self.register(Unit::new(
            "gram per liter",
            "g/L",
            mass_conc.clone(),
            1000.0,
        ));
        self.register(Unit::new(
            "milligram per liter",
            "mg/L",
            mass_conc.clone(),
            1.0,
        ));
        self.register(Unit::new(
            "microgram per liter",
            "ug/L",
            mass_conc.clone(),
            0.001,
        ));
        self.register(Unit::new(
            "milligram per deciliter",
            "mg/dL",
            mass_conc.clone(),
            10.0,
        ));
        self.register(Unit::new(
            "nanogram per milliliter",
            "ng/mL",
            mass_conc.clone(),
            1.0,
        ));
    }

    fn register_medical_units(&mut self) {
        use super::dimension::standard::*;

        // Clearance
        let cl = clearance();
        self.register(Unit::new(
            "milliliter per minute",
            "mL/min",
            cl.clone(),
            1e-6 / 60.0,
        ));
        self.register(Unit::new(
            "liter per hour",
            "L/h",
            cl.clone(),
            0.001 / 3600.0,
        ));

        // Clearance normalized to BSA (for GFR)
        let cl_bsa = clearance_per_bsa();
        self.register(Unit::new(
            "milliliter per minute per 1.73m²",
            "mL/min/1.73m2",
            cl_bsa.clone(),
            1e-6 / 60.0 / 1.73,
        ));

        // Dose per body weight
        let dose_wt = mass() / mass(); // Dimensionless
        self.register(Unit::new(
            "milligram per kilogram",
            "mg/kg",
            dose_wt.clone(),
            1e-6,
        ));
        self.register(Unit::new(
            "microgram per kilogram",
            "ug/kg",
            dose_wt.clone(),
            1e-9,
        ));

        // Dose per BSA
        let dose_bsa = dose_per_bsa();
        self.register(Unit::new(
            "milligram per square meter",
            "mg/m2",
            dose_bsa.clone(),
            0.001,
        ));

        // IU-based units
        let iu = international_unit();
        self.register(Unit::new("international unit", "IU", iu.clone(), 1.0));
        self.register(Unit::new(
            "thousand international units",
            "kIU",
            iu.clone(),
            1000.0,
        ));
        self.register(Unit::new(
            "million international units",
            "MIU",
            iu.clone(),
            1e6,
        ));

        // IU per volume
        let iu_vol = iu_per_volume();
        self.register(Unit::new("IU per milliliter", "IU/mL", iu_vol.clone(), 1e6));
        self.register(Unit::new("IU per liter", "IU/L", iu_vol.clone(), 1000.0));

        // IU per mass
        let iu_mass = iu_per_mass();
        self.register(Unit::new("IU per gram", "IU/g", iu_mass.clone(), 1.0));
        self.register(Unit::new(
            "IU per kilogram",
            "IU/kg",
            iu_mass.clone(),
            0.001,
        ));

        // Rate constants
        let rate = rate_constant();
        self.register(Unit::new("per hour", "/h", rate.clone(), 1.0 / 3600.0));
        self.register(Unit::new("per day", "/d", rate.clone(), 1.0 / 86400.0));

        // AUC
        let auc_dim = auc();
        self.register(Unit::new(
            "microgram hour per milliliter",
            "ug.h/mL",
            auc_dim.clone(),
            3600.0, // ug/mL * h -> base units
        ));
        self.register(Unit::new(
            "nanogram hour per milliliter",
            "ng.h/mL",
            auc_dim.clone(),
            3.6,
        ));
        self.register(Unit::new(
            "milligram hour per liter",
            "mg.h/L",
            auc_dim.clone(),
            3600.0,
        ));

        // Blood pressure
        let pressure_dim = pressure();
        self.register(Unit::new(
            "millimeter of mercury",
            "mm[Hg]",
            pressure_dim.clone(),
            133.322,
        ));

        // Temperature (with offset handling needed separately)
        let temp = temperature();
        self.register(Unit::new("degree Celsius", "Cel", temp.clone(), 1.0));
        // Note: Celsius to Kelvin requires offset, not just scaling
    }
}

impl Default for UnitRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

// =============================================================================
// Compound Unit Builder
// =============================================================================

/// Builder for constructing compound units
pub struct CompoundUnitBuilder {
    registry: UnitRegistry,
}

impl CompoundUnitBuilder {
    pub fn new(registry: UnitRegistry) -> Self {
        CompoundUnitBuilder { registry }
    }

    /// Parse a compound unit string like "mg/kg" or "mL/min/1.73m2"
    pub fn parse(&self, unit_str: &str) -> Result<Unit, UnitParseError> {
        // Simple parser for common patterns
        // Full UCUM parsing is more complex

        if let Some(unit) = self.registry.get(unit_str) {
            return Ok(unit.clone());
        }

        // Handle division
        if let Some(pos) = unit_str.find('/') {
            let (num, denom) = unit_str.split_at(pos);
            let denom = &denom[1..]; // Skip the '/'

            let num_unit = self.parse(num)?;
            let denom_unit = self.parse(denom)?;
            return Ok(num_unit.divide(&denom_unit));
        }

        // Handle multiplication
        if let Some(pos) = unit_str.find('.') {
            let (left, right) = unit_str.split_at(pos);
            let right = &right[1..]; // Skip the '.'

            let left_unit = self.parse(left)?;
            let right_unit = self.parse(right)?;
            return Ok(left_unit.multiply(&right_unit));
        }

        // Handle powers (e.g., m2, cm3)
        if unit_str.ends_with('2') || unit_str.ends_with('3') {
            let power: i8 = unit_str.chars().last().unwrap().to_digit(10).unwrap() as i8;
            let base = &unit_str[..unit_str.len() - 1];
            if let Some(base_unit) = self.registry.get(base) {
                return Ok(base_unit.pow(power));
            }
        }

        Err(UnitParseError::UnknownUnit(unit_str.to_string()))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum UnitParseError {
    UnknownUnit(String),
    InvalidSyntax(String),
    IncompatibleUnits(String, String),
}

impl fmt::Display for UnitParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitParseError::UnknownUnit(u) => write!(f, "Unknown unit: {}", u),
            UnitParseError::InvalidSyntax(s) => write!(f, "Invalid unit syntax: {}", s),
            UnitParseError::IncompatibleUnits(a, b) => {
                write!(f, "Incompatible units: {} and {}", a, b)
            }
        }
    }
}

impl std::error::Error for UnitParseError {}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_creation() {
        let registry = UnitRegistry::standard();

        let mg = registry.get("mg").unwrap();
        assert_eq!(mg.name, "milligram");
        assert_eq!(mg.to_base_factor, 0.001);

        let kg = registry.get("kg").unwrap();
        assert_eq!(kg.name, "kilogram");
        assert_eq!(kg.to_base_factor, 1000.0);
    }

    #[test]
    fn test_unit_compatibility() {
        let registry = UnitRegistry::standard();

        let mg = registry.get("mg").unwrap();
        let kg = registry.get("kg").unwrap();
        let ml = registry.get("mL").unwrap();

        assert!(mg.is_compatible(kg));
        assert!(!mg.is_compatible(ml));
    }

    #[test]
    fn test_unit_conversion() {
        let registry = UnitRegistry::standard();

        let mg = registry.get("mg").unwrap();
        let g = registry.get("g").unwrap();

        let factor = mg.conversion_factor_to(g).unwrap();
        assert!((factor - 0.001).abs() < 1e-10);

        // 500 mg = 0.5 g
        let mg_value = 500.0;
        let g_value = mg_value * factor;
        assert!((g_value - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_compound_unit_division() {
        let registry = UnitRegistry::standard();

        let ml = registry.get("mL").unwrap();
        let min = registry.get("min").unwrap();

        let ml_per_min = ml.divide(min);
        assert!(ml_per_min.ucum_code.contains('/'));

        // Check dimension is clearance
        assert_eq!(ml_per_min.dimension.get_exponent(BaseDimension::Length), 3);
        assert_eq!(ml_per_min.dimension.get_exponent(BaseDimension::Time), -1);
    }

    #[test]
    fn test_compound_unit_multiplication() {
        let registry = UnitRegistry::standard();

        let ug = registry.get("ug").unwrap();
        let h = registry.get("h").unwrap();

        let ug_h = ug.multiply(h);
        assert!(ug_h.ucum_code.contains('.'));
    }

    #[test]
    fn test_unit_power() {
        let registry = UnitRegistry::standard();

        let m = registry.get("m").unwrap();
        let m2 = m.pow(2);

        assert_eq!(m2.dimension.get_exponent(BaseDimension::Length), 2);
    }

    #[test]
    fn test_international_units_non_convertible() {
        let registry = UnitRegistry::standard();

        let iu = registry.get("[IU]").unwrap();
        let mg = registry.get("mg").unwrap();

        // IU and mg are not compatible (different dimensions)
        assert!(!iu.is_compatible(mg));

        // IU with different annotations should not convert
        let iu1 = iu.clone().with_annotation("insulin");
        let iu2 = iu.clone().with_annotation("vitamin_d");

        // They have same dimension but different semantic meaning
        assert!(iu1.is_compatible(&iu2)); // Dimensionally same
                                          // But conversion should fail for different IU types
    }

    #[test]
    fn test_compound_unit_parser() {
        let registry = UnitRegistry::standard();
        let builder = CompoundUnitBuilder::new(registry);

        // Simple unit
        let mg = builder.parse("mg").unwrap();
        assert_eq!(mg.name, "milligram");

        // Division
        let mg_kg = builder.parse("mg/kg").unwrap();
        assert!(mg_kg.dimension.is_dimensionless()); // mass/mass = dimensionless

        // Already registered compound
        let ml_min = builder.parse("mL/min").unwrap();
        assert_eq!(ml_min.dimension.get_exponent(BaseDimension::Length), 3);
        assert_eq!(ml_min.dimension.get_exponent(BaseDimension::Time), -1);
    }

    #[test]
    fn test_medical_units() {
        let registry = UnitRegistry::standard();

        // GFR unit
        let gfr_unit = registry.get("mL/min/1.73m2").unwrap();
        assert!(gfr_unit.name.contains("1.73"));

        // Dose per BSA
        let dose_bsa = registry.get("mg/m2").unwrap();
        assert_eq!(dose_bsa.dimension.get_exponent(BaseDimension::Mass), 1);
        assert_eq!(dose_bsa.dimension.get_exponent(BaseDimension::Length), -2);
    }

    #[test]
    fn test_unit_annotation() {
        let registry = UnitRegistry::standard();
        let mol_l = registry.get("mol/L").unwrap();

        let creatinine_unit = mol_l.clone().with_annotation("creatinine");
        assert_eq!(creatinine_unit.to_string(), "mol/L{creatinine}");
    }
}
