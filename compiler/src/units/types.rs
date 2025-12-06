// Week 54: Units of Measure - Compile-Time Types
//
// Kennedy-style compile-time dimensional analysis with unit erasure.
// The type system tracks dimensions at compile time, then erases them to
// achieve zero runtime overhead.
//
// ## Type Representation
//
// A quantity type is represented as `Quantity<D>` where D is a dimension type.
// At codegen time, `Quantity<D>` becomes simply `Float`.
//
// ## Type Checking Rules
//
// 1. Addition/Subtraction: Same dimension required
//    Quantity<D> + Quantity<D> → Quantity<D>
//
// 2. Multiplication: Dimensions multiply
//    Quantity<D1> * Quantity<D2> → Quantity<D1 * D2>
//
// 3. Division: Dimensions divide
//    Quantity<D1> / Quantity<D2> → Quantity<D1 / D2>
//
// 4. Assignment: Dimensions must match
//    let x: Quantity<Mass> = quantity_of_mass;
//
// 5. Function calls: Parameter dimensions must match declaration

use super::dimension::{BaseDimension, Dimension};
use std::collections::HashMap;
use std::fmt;

/// Compile-time unit type representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnitType {
    /// Dimensionless scalar (Float with no unit)
    Dimensionless,
    /// Quantity with known dimension
    Quantity(DimensionType),
    /// Type variable for generic unit types
    UnitVar(String),
    /// Unknown unit (needs inference)
    Unknown,
}

impl UnitType {
    pub fn dimensionless() -> Self {
        UnitType::Dimensionless
    }

    pub fn quantity(dim: DimensionType) -> Self {
        UnitType::Quantity(dim)
    }

    pub fn var(name: &str) -> Self {
        UnitType::UnitVar(name.to_string())
    }

    pub fn is_dimensionless(&self) -> bool {
        matches!(self, UnitType::Dimensionless)
    }

    pub fn is_known(&self) -> bool {
        !matches!(self, UnitType::Unknown | UnitType::UnitVar(_))
    }
}

impl fmt::Display for UnitType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnitType::Dimensionless => write!(f, "Float"),
            UnitType::Quantity(dim) => write!(f, "Quantity<{}>", dim),
            UnitType::UnitVar(name) => write!(f, "?{}", name),
            UnitType::Unknown => write!(f, "?"),
        }
    }
}

/// Compile-time dimension type (symbolic representation)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DimensionType {
    /// Symbolic name (e.g., "Mass", "Length/Time")
    pub name: String,
    /// Concrete dimension (if known)
    pub dimension: Option<Dimension>,
}

impl std::hash::Hash for DimensionType {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl DimensionType {
    pub fn new(name: &str, dimension: Dimension) -> Self {
        DimensionType {
            name: name.to_string(),
            dimension: Some(dimension),
        }
    }

    pub fn symbolic(name: &str) -> Self {
        DimensionType {
            name: name.to_string(),
            dimension: None,
        }
    }

    pub fn from_dimension(dimension: Dimension) -> Self {
        let name = Self::dimension_to_name(&dimension);
        DimensionType {
            name,
            dimension: Some(dimension),
        }
    }

    fn dimension_to_name(dim: &Dimension) -> String {
        if dim.is_dimensionless() {
            return "1".to_string();
        }

        let mut num_parts = Vec::new();
        let mut denom_parts = Vec::new();

        for base in BaseDimension::all() {
            let exp = dim.get_exponent(*base);
            if exp > 0 {
                if exp == 1 {
                    num_parts.push(base.symbol().to_string());
                } else {
                    num_parts.push(format!("{}^{}", base.symbol(), exp));
                }
            } else if exp < 0 {
                if exp == -1 {
                    denom_parts.push(base.symbol().to_string());
                } else {
                    denom_parts.push(format!("{}^{}", base.symbol(), -exp));
                }
            }
        }

        let num = if num_parts.is_empty() {
            "1".to_string()
        } else {
            num_parts.join("·")
        };

        if denom_parts.is_empty() {
            num
        } else {
            format!("{}/{}", num, denom_parts.join("·"))
        }
    }

    /// Multiply two dimension types
    pub fn multiply(&self, other: &DimensionType) -> DimensionType {
        match (&self.dimension, &other.dimension) {
            (Some(d1), Some(d2)) => DimensionType::from_dimension(d1.clone() * d2.clone()),
            _ => DimensionType::symbolic(&format!("{}·{}", self.name, other.name)),
        }
    }

    /// Divide two dimension types
    pub fn divide(&self, other: &DimensionType) -> DimensionType {
        match (&self.dimension, &other.dimension) {
            (Some(d1), Some(d2)) => DimensionType::from_dimension(d1.clone() / d2.clone()),
            _ => DimensionType::symbolic(&format!("{}/{}", self.name, other.name)),
        }
    }

    /// Raise to a power
    pub fn pow(&self, n: i8) -> DimensionType {
        match &self.dimension {
            Some(d) => DimensionType::from_dimension(d.pow(n)),
            None => DimensionType::symbolic(&format!("({})^{}", self.name, n)),
        }
    }
}

impl fmt::Display for DimensionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

// =============================================================================
// Standard Dimension Types
// =============================================================================

/// Pre-defined dimension types for common medical/scientific quantities
pub mod standard_dims {
    use super::super::dimension::standard as dim;
    use super::*;

    pub fn dimensionless() -> DimensionType {
        DimensionType::new("1", dim::dimensionless())
    }

    pub fn length() -> DimensionType {
        DimensionType::new("L", dim::length())
    }

    pub fn mass() -> DimensionType {
        DimensionType::new("M", dim::mass())
    }

    pub fn time() -> DimensionType {
        DimensionType::new("T", dim::time())
    }

    pub fn temperature() -> DimensionType {
        DimensionType::new("Θ", dim::temperature())
    }

    pub fn amount() -> DimensionType {
        DimensionType::new("N", dim::amount())
    }

    pub fn area() -> DimensionType {
        DimensionType::new("L²", dim::area())
    }

    pub fn volume() -> DimensionType {
        DimensionType::new("L³", dim::volume())
    }

    pub fn velocity() -> DimensionType {
        DimensionType::new("L/T", dim::velocity())
    }

    pub fn concentration() -> DimensionType {
        DimensionType::new("N/L³", dim::concentration())
    }

    pub fn mass_concentration() -> DimensionType {
        DimensionType::new("M/L³", dim::mass_concentration())
    }

    pub fn clearance() -> DimensionType {
        DimensionType::new("L³/T", dim::clearance())
    }

    pub fn clearance_per_weight() -> DimensionType {
        DimensionType::new("L³/T/M", dim::clearance_per_weight())
    }

    pub fn clearance_per_bsa() -> DimensionType {
        DimensionType::new("L/T", dim::clearance_per_bsa())
    }

    pub fn dose_per_bsa() -> DimensionType {
        DimensionType::new("M/L²", dim::dose_per_bsa())
    }

    pub fn auc() -> DimensionType {
        DimensionType::new("N·T/L³", dim::auc())
    }

    pub fn rate_constant() -> DimensionType {
        DimensionType::new("1/T", dim::rate_constant())
    }

    pub fn international_unit() -> DimensionType {
        DimensionType::new("IU", dim::international_unit())
    }
}

// =============================================================================
// Dimension Checking
// =============================================================================

/// Result of dimension checking
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionCheckResult {
    /// Dimensions match
    Match,
    /// Dimensions mismatch
    Mismatch {
        expected: DimensionType,
        found: DimensionType,
    },
    /// One or both dimensions are unknown (deferred check)
    Deferred,
}

/// Dimension checker for compile-time validation
pub struct DimensionChecker {
    /// Known dimension variables
    dim_vars: HashMap<String, DimensionType>,
    /// Collected errors
    errors: Vec<DimensionError>,
}

impl DimensionChecker {
    pub fn new() -> Self {
        DimensionChecker {
            dim_vars: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Define a dimension variable
    pub fn define_var(&mut self, name: &str, dim: DimensionType) {
        self.dim_vars.insert(name.to_string(), dim);
    }

    /// Lookup a dimension variable
    pub fn lookup_var(&self, name: &str) -> Option<&DimensionType> {
        self.dim_vars.get(name)
    }

    /// Check if two unit types are compatible
    pub fn check_compatible(
        &mut self,
        expected: &UnitType,
        found: &UnitType,
    ) -> DimensionCheckResult {
        match (expected, found) {
            (UnitType::Dimensionless, UnitType::Dimensionless) => DimensionCheckResult::Match,
            (UnitType::Quantity(d1), UnitType::Quantity(d2)) => self.check_dimensions_equal(d1, d2),
            (UnitType::Dimensionless, UnitType::Quantity(d)) => {
                if d.dimension
                    .as_ref()
                    .is_some_and(|dim| dim.is_dimensionless())
                {
                    DimensionCheckResult::Match
                } else {
                    DimensionCheckResult::Mismatch {
                        expected: standard_dims::dimensionless(),
                        found: d.clone(),
                    }
                }
            }
            (UnitType::Quantity(d), UnitType::Dimensionless) => {
                if d.dimension
                    .as_ref()
                    .is_some_and(|dim| dim.is_dimensionless())
                {
                    DimensionCheckResult::Match
                } else {
                    DimensionCheckResult::Mismatch {
                        expected: d.clone(),
                        found: standard_dims::dimensionless(),
                    }
                }
            }
            (UnitType::UnitVar(v), other) | (other, UnitType::UnitVar(v)) => {
                if let Some(dim) = self.dim_vars.get(v) {
                    let other_dim = match other {
                        UnitType::Dimensionless => standard_dims::dimensionless(),
                        UnitType::Quantity(d) => d.clone(),
                        UnitType::UnitVar(v2) => {
                            if let Some(d) = self.dim_vars.get(v2) {
                                d.clone()
                            } else {
                                return DimensionCheckResult::Deferred;
                            }
                        }
                        UnitType::Unknown => return DimensionCheckResult::Deferred,
                    };
                    self.check_dimensions_equal(dim, &other_dim)
                } else {
                    DimensionCheckResult::Deferred
                }
            }
            (UnitType::Unknown, _) | (_, UnitType::Unknown) => DimensionCheckResult::Deferred,
        }
    }

    fn check_dimensions_equal(
        &self,
        d1: &DimensionType,
        d2: &DimensionType,
    ) -> DimensionCheckResult {
        match (&d1.dimension, &d2.dimension) {
            (Some(dim1), Some(dim2)) => {
                if dim1 == dim2 {
                    DimensionCheckResult::Match
                } else {
                    DimensionCheckResult::Mismatch {
                        expected: d1.clone(),
                        found: d2.clone(),
                    }
                }
            }
            _ => {
                // Symbolic comparison
                if d1.name == d2.name {
                    DimensionCheckResult::Match
                } else {
                    DimensionCheckResult::Mismatch {
                        expected: d1.clone(),
                        found: d2.clone(),
                    }
                }
            }
        }
    }

    /// Infer the result type of addition/subtraction
    pub fn infer_add_sub(
        &mut self,
        left: &UnitType,
        right: &UnitType,
    ) -> Result<UnitType, DimensionError> {
        let check = self.check_compatible(left, right);
        match check {
            DimensionCheckResult::Match => Ok(left.clone()),
            DimensionCheckResult::Mismatch { expected, found } => {
                Err(DimensionError::AddSubMismatch { expected, found })
            }
            DimensionCheckResult::Deferred => Ok(left.clone()), // Assume OK, defer to runtime
        }
    }

    /// Infer the result type of multiplication
    pub fn infer_mul(&self, left: &UnitType, right: &UnitType) -> UnitType {
        match (left, right) {
            (UnitType::Dimensionless, other) | (other, UnitType::Dimensionless) => other.clone(),
            (UnitType::Quantity(d1), UnitType::Quantity(d2)) => {
                let result_dim = d1.multiply(d2);
                if result_dim
                    .dimension
                    .as_ref()
                    .is_some_and(|d| d.is_dimensionless())
                {
                    UnitType::Dimensionless
                } else {
                    UnitType::Quantity(result_dim)
                }
            }
            _ => UnitType::Unknown,
        }
    }

    /// Infer the result type of division
    pub fn infer_div(&self, left: &UnitType, right: &UnitType) -> UnitType {
        match (left, right) {
            (other, UnitType::Dimensionless) => other.clone(),
            (UnitType::Dimensionless, UnitType::Quantity(d)) => {
                UnitType::Quantity(standard_dims::dimensionless().divide(d))
            }
            (UnitType::Quantity(d1), UnitType::Quantity(d2)) => {
                let result_dim = d1.divide(d2);
                if result_dim
                    .dimension
                    .as_ref()
                    .is_some_and(|d| d.is_dimensionless())
                {
                    UnitType::Dimensionless
                } else {
                    UnitType::Quantity(result_dim)
                }
            }
            _ => UnitType::Unknown,
        }
    }

    /// Get collected errors
    pub fn errors(&self) -> &[DimensionError] {
        &self.errors
    }

    /// Add an error
    pub fn add_error(&mut self, error: DimensionError) {
        self.errors.push(error);
    }
}

impl Default for DimensionChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Dimension checking errors
#[derive(Debug, Clone, PartialEq)]
pub enum DimensionError {
    /// Addition/subtraction with mismatched dimensions
    AddSubMismatch {
        expected: DimensionType,
        found: DimensionType,
    },
    /// Assignment with mismatched dimensions
    AssignmentMismatch {
        variable: String,
        expected: DimensionType,
        found: DimensionType,
    },
    /// Function parameter dimension mismatch
    ParameterMismatch {
        function: String,
        param_index: usize,
        expected: DimensionType,
        found: DimensionType,
    },
    /// Return type dimension mismatch
    ReturnMismatch {
        function: String,
        expected: DimensionType,
        found: DimensionType,
    },
    /// Cannot take sqrt of dimension with odd exponents
    InvalidSqrt { dimension: DimensionType },
    /// IU units used in incompatible context
    IUConversionError { context: String },
}

impl fmt::Display for DimensionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimensionError::AddSubMismatch { expected, found } => {
                write!(
                    f,
                    "Dimension mismatch in addition/subtraction: expected {}, found {}",
                    expected, found
                )
            }
            DimensionError::AssignmentMismatch {
                variable,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Dimension mismatch in assignment to '{}': expected {}, found {}",
                    variable, expected, found
                )
            }
            DimensionError::ParameterMismatch {
                function,
                param_index,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Dimension mismatch in parameter {} of '{}': expected {}, found {}",
                    param_index, function, expected, found
                )
            }
            DimensionError::ReturnMismatch {
                function,
                expected,
                found,
            } => {
                write!(
                    f,
                    "Dimension mismatch in return type of '{}': expected {}, found {}",
                    function, expected, found
                )
            }
            DimensionError::InvalidSqrt { dimension } => {
                write!(
                    f,
                    "Cannot take sqrt of dimension with odd exponents: {}",
                    dimension
                )
            }
            DimensionError::IUConversionError { context } => {
                write!(
                    f,
                    "International Units cannot be converted to other units in context: {}",
                    context
                )
            }
        }
    }
}

impl std::error::Error for DimensionError {}

// =============================================================================
// Unit Type Annotations for AST
// =============================================================================

/// Unit annotation in source code
#[derive(Debug, Clone, PartialEq)]
pub struct UnitAnnotation {
    /// UCUM code or dimension name
    pub code: String,
    /// Optional annotation (e.g., {creatinine})
    pub annotation: Option<String>,
}

impl UnitAnnotation {
    pub fn new(code: &str) -> Self {
        UnitAnnotation {
            code: code.to_string(),
            annotation: None,
        }
    }

    pub fn with_annotation(code: &str, annotation: &str) -> Self {
        UnitAnnotation {
            code: code.to_string(),
            annotation: Some(annotation.to_string()),
        }
    }

    pub fn to_unit_type(&self, registry: &UnitTypeRegistry) -> UnitType {
        registry.lookup(&self.code).unwrap_or(UnitType::Unknown)
    }
}

impl fmt::Display for UnitAnnotation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref ann) = self.annotation {
            write!(f, "{}{{{}}}", self.code, ann)
        } else {
            write!(f, "{}", self.code)
        }
    }
}

/// Registry mapping unit codes to dimension types
pub struct UnitTypeRegistry {
    types: HashMap<String, UnitType>,
}

impl UnitTypeRegistry {
    pub fn new() -> Self {
        UnitTypeRegistry {
            types: HashMap::new(),
        }
    }

    pub fn standard() -> Self {
        let mut registry = Self::new();

        // Register standard units
        registry.register("1", UnitType::Dimensionless);
        registry.register("%", UnitType::Dimensionless);

        registry.register("m", UnitType::Quantity(standard_dims::length()));
        registry.register("cm", UnitType::Quantity(standard_dims::length()));
        registry.register("mm", UnitType::Quantity(standard_dims::length()));
        registry.register("m2", UnitType::Quantity(standard_dims::area()));
        registry.register("m3", UnitType::Quantity(standard_dims::volume()));

        registry.register("g", UnitType::Quantity(standard_dims::mass()));
        registry.register("kg", UnitType::Quantity(standard_dims::mass()));
        registry.register("mg", UnitType::Quantity(standard_dims::mass()));
        registry.register("ug", UnitType::Quantity(standard_dims::mass()));
        registry.register("ng", UnitType::Quantity(standard_dims::mass()));

        registry.register("s", UnitType::Quantity(standard_dims::time()));
        registry.register("min", UnitType::Quantity(standard_dims::time()));
        registry.register("h", UnitType::Quantity(standard_dims::time()));
        registry.register("d", UnitType::Quantity(standard_dims::time()));

        registry.register("L", UnitType::Quantity(standard_dims::volume()));
        registry.register("mL", UnitType::Quantity(standard_dims::volume()));
        registry.register("uL", UnitType::Quantity(standard_dims::volume()));

        registry.register("mol/L", UnitType::Quantity(standard_dims::concentration()));
        registry.register("mmol/L", UnitType::Quantity(standard_dims::concentration()));
        registry.register("umol/L", UnitType::Quantity(standard_dims::concentration()));

        registry.register(
            "mg/L",
            UnitType::Quantity(standard_dims::mass_concentration()),
        );
        registry.register(
            "ng/mL",
            UnitType::Quantity(standard_dims::mass_concentration()),
        );
        registry.register(
            "mg/dL",
            UnitType::Quantity(standard_dims::mass_concentration()),
        );

        registry.register("mL/min", UnitType::Quantity(standard_dims::clearance()));
        registry.register("L/h", UnitType::Quantity(standard_dims::clearance()));
        registry.register(
            "mL/min/1.73m2",
            UnitType::Quantity(standard_dims::clearance_per_bsa()),
        );

        registry.register("mg/m2", UnitType::Quantity(standard_dims::dose_per_bsa()));

        registry.register("/h", UnitType::Quantity(standard_dims::rate_constant()));
        registry.register("/d", UnitType::Quantity(standard_dims::rate_constant()));

        registry.register(
            "[IU]",
            UnitType::Quantity(standard_dims::international_unit()),
        );
        registry.register(
            "IU",
            UnitType::Quantity(standard_dims::international_unit()),
        );

        registry
    }

    pub fn register(&mut self, code: &str, unit_type: UnitType) {
        self.types.insert(code.to_string(), unit_type);
    }

    pub fn lookup(&self, code: &str) -> Option<UnitType> {
        self.types.get(code).cloned()
    }
}

impl Default for UnitTypeRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unit_type_display() {
        assert_eq!(format!("{}", UnitType::Dimensionless), "Float");

        let mass = UnitType::Quantity(standard_dims::mass());
        assert!(format!("{}", mass).contains("M"));
    }

    #[test]
    fn test_dimension_type_multiply() {
        let mass = standard_dims::mass();
        let accel = standard_dims::length().divide(&standard_dims::time().pow(2));

        let force = mass.multiply(&accel);
        assert!(force.name.contains("M"));
    }

    #[test]
    fn test_dimension_checker_compatible() {
        let mut checker = DimensionChecker::new();

        let mass1 = UnitType::Quantity(standard_dims::mass());
        let mass2 = UnitType::Quantity(standard_dims::mass());
        let length = UnitType::Quantity(standard_dims::length());

        assert_eq!(
            checker.check_compatible(&mass1, &mass2),
            DimensionCheckResult::Match
        );

        match checker.check_compatible(&mass1, &length) {
            DimensionCheckResult::Mismatch { .. } => (),
            _ => panic!("Expected mismatch"),
        }
    }

    #[test]
    fn test_dimension_checker_mul() {
        let checker = DimensionChecker::new();

        let mass = UnitType::Quantity(standard_dims::mass());
        let velocity = UnitType::Quantity(standard_dims::velocity());

        let momentum = checker.infer_mul(&mass, &velocity);
        assert!(matches!(momentum, UnitType::Quantity(_)));
    }

    #[test]
    fn test_dimension_checker_div_to_dimensionless() {
        let checker = DimensionChecker::new();

        let mass1 = UnitType::Quantity(standard_dims::mass());
        let mass2 = UnitType::Quantity(standard_dims::mass());

        let ratio = checker.infer_div(&mass1, &mass2);
        assert!(ratio.is_dimensionless());
    }

    #[test]
    fn test_unit_type_registry() {
        let registry = UnitTypeRegistry::standard();

        let mg_type = registry.lookup("mg").unwrap();
        assert!(matches!(mg_type, UnitType::Quantity(_)));

        let ml_min_type = registry.lookup("mL/min").unwrap();
        assert!(matches!(ml_min_type, UnitType::Quantity(_)));
    }

    #[test]
    fn test_unit_annotation() {
        let ann = UnitAnnotation::with_annotation("mol/L", "creatinine");
        assert_eq!(format!("{}", ann), "mol/L{creatinine}");
    }

    #[test]
    fn test_infer_add_sub_error() {
        let mut checker = DimensionChecker::new();

        let mass = UnitType::Quantity(standard_dims::mass());
        let length = UnitType::Quantity(standard_dims::length());

        let result = checker.infer_add_sub(&mass, &length);
        assert!(result.is_err());
    }
}
