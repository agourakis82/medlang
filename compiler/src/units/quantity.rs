// Week 54: Units of Measure - Quantity Types
//
// A Quantity combines a numeric value with a Unit, enabling type-safe arithmetic
// with automatic dimensional checking. This is the runtime representation that
// corresponds to the compile-time dimensional types.
//
// ## Compile-time vs Runtime
//
// For zero-cost abstraction (Kennedy-style unit erasure), the compiler tracks
// dimensions at compile time and erases them for codegen. The Quantity type here
// is used for:
// 1. API boundaries where units must be preserved
// 2. User input validation
// 3. Database storage/retrieval
// 4. Serialization with unit information

use super::dimension::Dimension;
use super::unit::{Unit, UnitParseError, UnitRegistry};
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A quantity with value and unit
#[derive(Clone, PartialEq)]
pub struct Quantity {
    /// Numeric value in the specified unit
    pub value: f64,
    /// Unit of measurement
    pub unit: Unit,
}

impl Quantity {
    /// Create a new quantity
    pub fn new(value: f64, unit: Unit) -> Self {
        Quantity { value, unit }
    }

    /// Create a dimensionless quantity
    pub fn dimensionless(value: f64) -> Self {
        Quantity {
            value,
            unit: Unit::dimensionless("unity", "1"),
        }
    }

    /// Get the dimension of this quantity
    pub fn dimension(&self) -> &Dimension {
        &self.unit.dimension
    }

    /// Check if dimensionless
    pub fn is_dimensionless(&self) -> bool {
        self.unit.dimension.is_dimensionless()
    }

    /// Convert to another compatible unit
    pub fn convert_to(&self, target_unit: &Unit) -> Result<Quantity, QuantityError> {
        if let Some(factor) = self.unit.conversion_factor_to(target_unit) {
            Ok(Quantity {
                value: self.value * factor,
                unit: target_unit.clone(),
            })
        } else {
            Err(QuantityError::IncompatibleUnits {
                from: self.unit.ucum_code.clone(),
                to: target_unit.ucum_code.clone(),
            })
        }
    }

    /// Get value in base units
    pub fn to_base_value(&self) -> f64 {
        self.value * self.unit.to_base_factor
    }

    /// Raise quantity to integer power
    pub fn pow(&self, n: i32) -> Quantity {
        Quantity {
            value: self.value.powi(n),
            unit: self.unit.pow(n as i8),
        }
    }

    /// Square root (fails if dimension has odd exponents)
    pub fn sqrt(&self) -> Result<Quantity, QuantityError> {
        if let Some(new_dim) = self.unit.dimension.sqrt() {
            Ok(Quantity {
                value: self.value.sqrt(),
                unit: Unit::new(
                    &format!("sqrt({})", self.unit.name),
                    &format!("sqrt({})", self.unit.ucum_code),
                    new_dim,
                    self.unit.to_base_factor.sqrt(),
                ),
            })
        } else {
            Err(QuantityError::InvalidOperation(format!(
                "Cannot take sqrt of unit with odd exponents: {}",
                self.unit.ucum_code
            )))
        }
    }

    /// Absolute value
    pub fn abs(&self) -> Quantity {
        Quantity {
            value: self.value.abs(),
            unit: self.unit.clone(),
        }
    }

    /// Check if value is NaN
    pub fn is_nan(&self) -> bool {
        self.value.is_nan()
    }

    /// Check if value is infinite
    pub fn is_infinite(&self) -> bool {
        self.value.is_infinite()
    }

    /// Check if value is finite (not NaN or infinite)
    pub fn is_finite(&self) -> bool {
        self.value.is_finite()
    }
}

impl fmt::Debug for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Quantity({} {})", self.value, self.unit)
    }
}

impl fmt::Display for Quantity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit)
    }
}

/// Errors from quantity operations
#[derive(Debug, Clone, PartialEq)]
pub enum QuantityError {
    /// Cannot add/subtract quantities with different dimensions
    DimensionMismatch {
        left: Dimension,
        right: Dimension,
        operation: String,
    },
    /// Cannot convert between incompatible units
    IncompatibleUnits { from: String, to: String },
    /// Invalid mathematical operation
    InvalidOperation(String),
    /// Unit parsing error
    ParseError(UnitParseError),
}

impl fmt::Display for QuantityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantityError::DimensionMismatch {
                left,
                right,
                operation,
            } => {
                write!(
                    f,
                    "Cannot {} quantities with different dimensions: {} vs {}",
                    operation, left, right
                )
            }
            QuantityError::IncompatibleUnits { from, to } => {
                write!(f, "Cannot convert from {} to {}", from, to)
            }
            QuantityError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            QuantityError::ParseError(e) => write!(f, "Unit parse error: {}", e),
        }
    }
}

impl std::error::Error for QuantityError {}

impl From<UnitParseError> for QuantityError {
    fn from(e: UnitParseError) -> Self {
        QuantityError::ParseError(e)
    }
}

// =============================================================================
// Arithmetic Operations
// =============================================================================

/// Addition requires same dimension (converts to left unit)
impl Add for Quantity {
    type Output = Result<Quantity, QuantityError>;

    fn add(self, rhs: Self) -> Self::Output {
        if self.unit.dimension != rhs.unit.dimension {
            return Err(QuantityError::DimensionMismatch {
                left: self.unit.dimension.clone(),
                right: rhs.unit.dimension.clone(),
                operation: "add".to_string(),
            });
        }

        // Convert rhs to same unit as lhs
        let rhs_converted = rhs.convert_to(&self.unit)?;
        Ok(Quantity {
            value: self.value + rhs_converted.value,
            unit: self.unit,
        })
    }
}

/// Subtraction requires same dimension
impl Sub for Quantity {
    type Output = Result<Quantity, QuantityError>;

    fn sub(self, rhs: Self) -> Self::Output {
        if self.unit.dimension != rhs.unit.dimension {
            return Err(QuantityError::DimensionMismatch {
                left: self.unit.dimension.clone(),
                right: rhs.unit.dimension.clone(),
                operation: "subtract".to_string(),
            });
        }

        let rhs_converted = rhs.convert_to(&self.unit)?;
        Ok(Quantity {
            value: self.value - rhs_converted.value,
            unit: self.unit,
        })
    }
}

/// Multiplication creates compound unit
impl Mul for Quantity {
    type Output = Quantity;

    fn mul(self, rhs: Self) -> Self::Output {
        Quantity {
            value: self.value * rhs.value,
            unit: self.unit.multiply(&rhs.unit),
        }
    }
}

/// Scalar multiplication
impl Mul<f64> for Quantity {
    type Output = Quantity;

    fn mul(self, rhs: f64) -> Self::Output {
        Quantity {
            value: self.value * rhs,
            unit: self.unit,
        }
    }
}

impl Mul<Quantity> for f64 {
    type Output = Quantity;

    fn mul(self, rhs: Quantity) -> Self::Output {
        Quantity {
            value: self * rhs.value,
            unit: rhs.unit,
        }
    }
}

/// Division creates compound unit
impl Div for Quantity {
    type Output = Quantity;

    fn div(self, rhs: Self) -> Self::Output {
        Quantity {
            value: self.value / rhs.value,
            unit: self.unit.divide(&rhs.unit),
        }
    }
}

/// Scalar division
impl Div<f64> for Quantity {
    type Output = Quantity;

    fn div(self, rhs: f64) -> Self::Output {
        Quantity {
            value: self.value / rhs,
            unit: self.unit,
        }
    }
}

/// Negation
impl Neg for Quantity {
    type Output = Quantity;

    fn neg(self) -> Self::Output {
        Quantity {
            value: -self.value,
            unit: self.unit,
        }
    }
}

// =============================================================================
// Quantity Builder with Registry
// =============================================================================

/// Builder for creating quantities from strings
pub struct QuantityBuilder {
    registry: UnitRegistry,
}

impl QuantityBuilder {
    pub fn new(registry: UnitRegistry) -> Self {
        QuantityBuilder { registry }
    }

    pub fn standard() -> Self {
        QuantityBuilder {
            registry: UnitRegistry::standard(),
        }
    }

    /// Parse a quantity string like "100 mg" or "5.5 mL/min"
    pub fn parse(&self, s: &str) -> Result<Quantity, QuantityError> {
        let s = s.trim();

        // Find the split between value and unit
        let mut value_end = 0;
        for (i, c) in s.char_indices() {
            if c.is_alphabetic() || c == '[' || c == '/' || c == '%' {
                value_end = i;
                break;
            }
            value_end = i + c.len_utf8();
        }

        let value_str = s[..value_end].trim();
        let unit_str = s[value_end..].trim();

        let value: f64 = value_str.parse().map_err(|_| {
            QuantityError::InvalidOperation(format!("Invalid numeric value: {}", value_str))
        })?;

        if unit_str.is_empty() {
            return Ok(Quantity::dimensionless(value));
        }

        let unit = self.registry.get(unit_str).cloned().ok_or_else(|| {
            QuantityError::ParseError(UnitParseError::UnknownUnit(unit_str.to_string()))
        })?;

        Ok(Quantity::new(value, unit))
    }

    /// Create a quantity with a known unit
    pub fn quantity(&self, value: f64, unit_code: &str) -> Result<Quantity, QuantityError> {
        let unit = self.registry.get(unit_code).cloned().ok_or_else(|| {
            QuantityError::ParseError(UnitParseError::UnknownUnit(unit_code.to_string()))
        })?;
        Ok(Quantity::new(value, unit))
    }
}

// =============================================================================
// Medical Quantity Helpers
// =============================================================================

/// Common medical quantity constructors
pub mod medical {
    use super::*;

    /// Dose in milligrams
    pub fn dose_mg(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("mg").unwrap().clone())
    }

    /// Dose in micrograms
    pub fn dose_ug(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("ug").unwrap().clone())
    }

    /// Volume in milliliters
    pub fn volume_ml(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("mL").unwrap().clone())
    }

    /// Time in hours
    pub fn time_h(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("h").unwrap().clone())
    }

    /// Time in minutes
    pub fn time_min(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("min").unwrap().clone())
    }

    /// Concentration in ng/mL
    pub fn conc_ng_ml(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("ng/mL").unwrap().clone())
    }

    /// Concentration in mg/L
    pub fn conc_mg_l(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("mg/L").unwrap().clone())
    }

    /// Body weight in kilograms
    pub fn weight_kg(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("kg").unwrap().clone())
    }

    /// Clearance in mL/min
    pub fn clearance_ml_min(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("mL/min").unwrap().clone())
    }

    /// Clearance in L/h
    pub fn clearance_l_h(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("L/h").unwrap().clone())
    }

    /// GFR normalized to BSA
    pub fn gfr(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("mL/min/1.73m2").unwrap().clone())
    }

    /// International Units
    pub fn iu(value: f64) -> Quantity {
        let registry = UnitRegistry::standard();
        Quantity::new(value, registry.get("[IU]").unwrap().clone())
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantity_creation() {
        let registry = UnitRegistry::standard();
        let q = Quantity::new(100.0, registry.get("mg").unwrap().clone());

        assert_eq!(q.value, 100.0);
        assert_eq!(q.unit.ucum_code, "mg");
    }

    #[test]
    fn test_quantity_conversion() {
        let registry = UnitRegistry::standard();

        let mg = Quantity::new(1000.0, registry.get("mg").unwrap().clone());
        let g = mg.convert_to(registry.get("g").unwrap()).unwrap();

        assert!((g.value - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantity_addition() {
        let registry = UnitRegistry::standard();

        let q1 = Quantity::new(100.0, registry.get("mg").unwrap().clone());
        let q2 = Quantity::new(0.2, registry.get("g").unwrap().clone());

        let sum = (q1 + q2).unwrap();
        assert!((sum.value - 300.0).abs() < 1e-10); // 100 mg + 200 mg = 300 mg
        assert_eq!(sum.unit.ucum_code, "mg");
    }

    #[test]
    fn test_quantity_addition_dimension_mismatch() {
        let registry = UnitRegistry::standard();

        let mass = Quantity::new(100.0, registry.get("mg").unwrap().clone());
        let volume = Quantity::new(10.0, registry.get("mL").unwrap().clone());

        let result = mass + volume;
        assert!(result.is_err());
    }

    #[test]
    fn test_quantity_multiplication() {
        let registry = UnitRegistry::standard();

        let conc = Quantity::new(10.0, registry.get("ng/mL").unwrap().clone());
        let vol = Quantity::new(5.0, registry.get("mL").unwrap().clone());

        let amount = conc * vol;
        // 10 ng/mL * 5 mL = 50 ng
        assert!((amount.value - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantity_division() {
        let registry = UnitRegistry::standard();

        let amount = Quantity::new(500.0, registry.get("mg").unwrap().clone());
        let weight = Quantity::new(70.0, registry.get("kg").unwrap().clone());

        let dose_per_kg = amount / weight;
        assert!((dose_per_kg.value - 500.0 / 70.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantity_scalar_operations() {
        let q = medical::dose_mg(100.0);

        let doubled = q.clone() * 2.0;
        assert!((doubled.value - 200.0).abs() < 1e-10);

        let halved = q / 2.0;
        assert!((halved.value - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantity_power() {
        let registry = UnitRegistry::standard();
        let length = Quantity::new(2.0, registry.get("m").unwrap().clone());

        let area = length.pow(2);
        assert!((area.value - 4.0).abs() < 1e-10);
        assert_eq!(
            area.unit
                .dimension
                .get_exponent(super::super::dimension::BaseDimension::Length),
            2
        );
    }

    #[test]
    fn test_quantity_sqrt() {
        let registry = UnitRegistry::standard();
        let area = Quantity::new(4.0, registry.get("m2").unwrap().clone());

        let length = area.sqrt().unwrap();
        assert!((length.value - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_quantity_builder_parse() {
        let builder = QuantityBuilder::standard();

        let q = builder.parse("100 mg").unwrap();
        assert!((q.value - 100.0).abs() < 1e-10);
        assert_eq!(q.unit.ucum_code, "mg");

        let q2 = builder.parse("5.5 mL/min").unwrap();
        assert!((q2.value - 5.5).abs() < 1e-10);
    }

    #[test]
    fn test_medical_helpers() {
        let dose = medical::dose_mg(100.0);
        assert_eq!(dose.unit.ucum_code, "mg");

        let conc = medical::conc_ng_ml(50.0);
        assert_eq!(conc.unit.ucum_code, "ng/mL");

        let cl = medical::clearance_ml_min(120.0);
        assert_eq!(cl.unit.ucum_code, "mL/min");

        let gfr = medical::gfr(90.0);
        assert!(gfr.unit.ucum_code.contains("1.73m2"));
    }

    #[test]
    fn test_international_units_not_convertible() {
        let iu_insulin = medical::iu(100.0);
        let mass = medical::dose_mg(1.0);

        // Should not be able to convert IU to mass
        let result = iu_insulin.convert_to(&mass.unit);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantity_display() {
        let q = medical::dose_mg(250.0);
        let s = format!("{}", q);
        assert!(s.contains("250"));
        assert!(s.contains("mg"));
    }
}
