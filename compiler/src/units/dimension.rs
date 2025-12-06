// Week 54: Dimensional Analysis - Dimension Types
//
// Kennedy-style compile-time dimensional analysis for MedLang.
// Dimensions are represented as exponent vectors over base dimensions.
//
// The seven SI base dimensions plus medical extensions:
// - Length (L), Mass (M), Time (T), Current (I), Temperature (Θ), Amount (N), Luminosity (J)
// - Plus: IU (International Units - biologically standardized, non-convertible)

use std::collections::HashMap;
use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// Base dimensions following SI with medical extensions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BaseDimension {
    /// Length (meter)
    Length,
    /// Mass (gram in UCUM, not kilogram)
    Mass,
    /// Time (second)
    Time,
    /// Electric current (ampere)
    Current,
    /// Thermodynamic temperature (kelvin)
    Temperature,
    /// Amount of substance (mole)
    Amount,
    /// Luminous intensity (candela)
    Luminosity,
    /// International Units - biologically standardized, NOT convertible to mass
    InternationalUnit,
}

impl BaseDimension {
    pub fn symbol(&self) -> &'static str {
        match self {
            BaseDimension::Length => "L",
            BaseDimension::Mass => "M",
            BaseDimension::Time => "T",
            BaseDimension::Current => "I",
            BaseDimension::Temperature => "Θ",
            BaseDimension::Amount => "N",
            BaseDimension::Luminosity => "J",
            BaseDimension::InternationalUnit => "IU",
        }
    }

    pub fn all() -> &'static [BaseDimension] {
        &[
            BaseDimension::Length,
            BaseDimension::Mass,
            BaseDimension::Time,
            BaseDimension::Current,
            BaseDimension::Temperature,
            BaseDimension::Amount,
            BaseDimension::Luminosity,
            BaseDimension::InternationalUnit,
        ]
    }
}

/// A dimension is a product of base dimensions raised to rational powers.
/// Represented as a map from base dimension to exponent (as i8 for simplicity).
///
/// Examples:
/// - Dimensionless: {} (empty map, all exponents zero)
/// - Length: {L: 1}
/// - Area: {L: 2}
/// - Velocity: {L: 1, T: -1}
/// - Force: {M: 1, L: 1, T: -2}
/// - Concentration: {N: 1, L: -3} (mol/L = mol/dm³)
#[derive(Clone, PartialEq, Eq, Default)]
pub struct Dimension {
    /// Exponents for each base dimension (only non-zero stored)
    exponents: HashMap<BaseDimension, i8>,
}

impl Dimension {
    /// Create a dimensionless quantity
    pub fn dimensionless() -> Self {
        Dimension {
            exponents: HashMap::new(),
        }
    }

    /// Create a dimension from a single base dimension
    pub fn from_base(base: BaseDimension) -> Self {
        let mut exponents = HashMap::new();
        exponents.insert(base, 1);
        Dimension { exponents }
    }

    /// Create a dimension with a specific exponent
    pub fn from_base_power(base: BaseDimension, power: i8) -> Self {
        if power == 0 {
            return Self::dimensionless();
        }
        let mut exponents = HashMap::new();
        exponents.insert(base, power);
        Dimension { exponents }
    }

    /// Get the exponent for a base dimension
    pub fn get_exponent(&self, base: BaseDimension) -> i8 {
        *self.exponents.get(&base).unwrap_or(&0)
    }

    /// Set an exponent (removes if zero)
    pub fn set_exponent(&mut self, base: BaseDimension, exp: i8) {
        if exp == 0 {
            self.exponents.remove(&base);
        } else {
            self.exponents.insert(base, exp);
        }
    }

    /// Check if dimensionless
    pub fn is_dimensionless(&self) -> bool {
        self.exponents.is_empty()
    }

    /// Raise dimension to a power
    pub fn pow(&self, n: i8) -> Self {
        if n == 0 {
            return Self::dimensionless();
        }
        let mut result = HashMap::new();
        for (&base, &exp) in &self.exponents {
            let new_exp = exp.saturating_mul(n);
            if new_exp != 0 {
                result.insert(base, new_exp);
            }
        }
        Dimension { exponents: result }
    }

    /// Take the nth root (returns None if not evenly divisible)
    pub fn root(&self, n: i8) -> Option<Self> {
        if n == 0 {
            return None;
        }
        let mut result = HashMap::new();
        for (&base, &exp) in &self.exponents {
            if exp % n != 0 {
                return None; // Not evenly divisible
            }
            let new_exp = exp / n;
            if new_exp != 0 {
                result.insert(base, new_exp);
            }
        }
        Some(Dimension { exponents: result })
    }

    /// Square root (convenience method)
    pub fn sqrt(&self) -> Option<Self> {
        self.root(2)
    }

    /// Check if this dimension contains IU (International Units)
    pub fn contains_iu(&self) -> bool {
        self.get_exponent(BaseDimension::InternationalUnit) != 0
    }
}

impl fmt::Debug for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_dimensionless() {
            return write!(f, "Dimensionless");
        }

        let mut parts: Vec<String> = Vec::new();
        for base in BaseDimension::all() {
            let exp = self.get_exponent(*base);
            if exp != 0 {
                if exp == 1 {
                    parts.push(base.symbol().to_string());
                } else {
                    parts.push(format!("{}^{}", base.symbol(), exp));
                }
            }
        }
        write!(f, "{}", parts.join("·"))
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// Multiplication of dimensions adds exponents
impl Mul for Dimension {
    type Output = Dimension;

    fn mul(self, rhs: Self) -> Self::Output {
        let mut result = self.exponents.clone();
        for (&base, &exp) in &rhs.exponents {
            let current = *result.get(&base).unwrap_or(&0);
            let new_exp = current.saturating_add(exp);
            if new_exp == 0 {
                result.remove(&base);
            } else {
                result.insert(base, new_exp);
            }
        }
        Dimension { exponents: result }
    }
}

impl Mul for &Dimension {
    type Output = Dimension;

    fn mul(self, rhs: Self) -> Self::Output {
        self.clone() * rhs.clone()
    }
}

/// Division of dimensions subtracts exponents
impl Div for Dimension {
    type Output = Dimension;

    fn div(self, rhs: Self) -> Self::Output {
        let mut result = self.exponents.clone();
        for (&base, &exp) in &rhs.exponents {
            let current = *result.get(&base).unwrap_or(&0);
            let new_exp = current.saturating_sub(exp);
            if new_exp == 0 {
                result.remove(&base);
            } else {
                result.insert(base, new_exp);
            }
        }
        Dimension { exponents: result }
    }
}

impl Div for &Dimension {
    type Output = Dimension;

    fn div(self, rhs: Self) -> Self::Output {
        self.clone() / rhs.clone()
    }
}

// =============================================================================
// Standard Dimensions
// =============================================================================

/// Common dimensions used in medical/scientific computing
pub mod standard {
    use super::*;

    /// Dimensionless (pure number, ratio, percentage)
    pub fn dimensionless() -> Dimension {
        Dimension::dimensionless()
    }

    // --- Base dimensions ---

    pub fn length() -> Dimension {
        Dimension::from_base(BaseDimension::Length)
    }

    pub fn mass() -> Dimension {
        Dimension::from_base(BaseDimension::Mass)
    }

    pub fn time() -> Dimension {
        Dimension::from_base(BaseDimension::Time)
    }

    pub fn current() -> Dimension {
        Dimension::from_base(BaseDimension::Current)
    }

    pub fn temperature() -> Dimension {
        Dimension::from_base(BaseDimension::Temperature)
    }

    pub fn amount() -> Dimension {
        Dimension::from_base(BaseDimension::Amount)
    }

    pub fn luminosity() -> Dimension {
        Dimension::from_base(BaseDimension::Luminosity)
    }

    pub fn international_unit() -> Dimension {
        Dimension::from_base(BaseDimension::InternationalUnit)
    }

    // --- Derived dimensions ---

    /// Area (L²)
    pub fn area() -> Dimension {
        Dimension::from_base_power(BaseDimension::Length, 2)
    }

    /// Volume (L³)
    pub fn volume() -> Dimension {
        Dimension::from_base_power(BaseDimension::Length, 3)
    }

    /// Velocity (L·T⁻¹)
    pub fn velocity() -> Dimension {
        length() / time()
    }

    /// Acceleration (L·T⁻²)
    pub fn acceleration() -> Dimension {
        length() / time().pow(2)
    }

    /// Force (M·L·T⁻²)
    pub fn force() -> Dimension {
        mass() * length() / time().pow(2)
    }

    /// Energy (M·L²·T⁻²)
    pub fn energy() -> Dimension {
        mass() * length().pow(2) / time().pow(2)
    }

    /// Power (M·L²·T⁻³)
    pub fn power() -> Dimension {
        energy() / time()
    }

    /// Pressure (M·L⁻¹·T⁻²)
    pub fn pressure() -> Dimension {
        force() / area()
    }

    /// Frequency (T⁻¹)
    pub fn frequency() -> Dimension {
        Dimension::from_base_power(BaseDimension::Time, -1)
    }

    // --- Medical/PK dimensions ---

    /// Concentration (amount per volume, N·L⁻³)
    pub fn concentration() -> Dimension {
        amount() / volume()
    }

    /// Mass concentration (mass per volume, M·L⁻³)
    pub fn mass_concentration() -> Dimension {
        mass() / volume()
    }

    /// Clearance (volume per time, L³·T⁻¹)
    pub fn clearance() -> Dimension {
        volume() / time()
    }

    /// Clearance per body weight (L³·T⁻¹·M⁻¹)
    pub fn clearance_per_weight() -> Dimension {
        clearance() / mass()
    }

    /// Clearance normalized to BSA (L³·T⁻¹·L⁻² = L·T⁻¹)
    /// Note: This is volume/time/area, used for GFR normalized to 1.73m²
    pub fn clearance_per_bsa() -> Dimension {
        clearance() / area()
    }

    /// Volume of distribution (L³·M⁻¹ when normalized to body weight)
    pub fn volume_per_weight() -> Dimension {
        volume() / mass()
    }

    /// Half-life (T)
    pub fn half_life() -> Dimension {
        time()
    }

    /// Rate constant (T⁻¹)
    pub fn rate_constant() -> Dimension {
        frequency()
    }

    /// Dose per body weight (M·M⁻¹ = dimensionless for mg/kg)
    /// Note: This becomes dimensionless because mass cancels
    pub fn dose_per_weight() -> Dimension {
        dimensionless()
    }

    /// Dose per BSA (M·L⁻²)
    pub fn dose_per_bsa() -> Dimension {
        mass() / area()
    }

    /// AUC - Area Under the Curve (concentration × time = N·L⁻³·T)
    pub fn auc() -> Dimension {
        concentration() * time()
    }

    /// IU per volume (for biological activity)
    pub fn iu_per_volume() -> Dimension {
        international_unit() / volume()
    }

    /// IU per mass (activity per weight)
    pub fn iu_per_mass() -> Dimension {
        international_unit() / mass()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimensionless() {
        let d = Dimension::dimensionless();
        assert!(d.is_dimensionless());
        assert_eq!(d.to_string(), "Dimensionless");
    }

    #[test]
    fn test_base_dimension() {
        let length = Dimension::from_base(BaseDimension::Length);
        assert_eq!(length.get_exponent(BaseDimension::Length), 1);
        assert_eq!(length.get_exponent(BaseDimension::Mass), 0);
        assert!(!length.is_dimensionless());
    }

    #[test]
    fn test_dimension_multiplication() {
        let length = standard::length();
        let time = standard::time();

        // Velocity = L / T = L · T⁻¹
        let velocity = length.clone() / time.clone();
        assert_eq!(velocity.get_exponent(BaseDimension::Length), 1);
        assert_eq!(velocity.get_exponent(BaseDimension::Time), -1);

        // Area = L × L = L²
        let area = length.clone() * length.clone();
        assert_eq!(area.get_exponent(BaseDimension::Length), 2);
    }

    #[test]
    fn test_dimension_cancellation() {
        let length = standard::length();
        let result = length.clone() / length.clone();
        assert!(result.is_dimensionless());
    }

    #[test]
    fn test_dimension_power() {
        let length = standard::length();
        let volume = length.pow(3);
        assert_eq!(volume.get_exponent(BaseDimension::Length), 3);

        let area = standard::area();
        let line = area.pow(0);
        assert!(line.is_dimensionless());
    }

    #[test]
    fn test_dimension_root() {
        let area = standard::area();
        let length = area.sqrt().unwrap();
        assert_eq!(length.get_exponent(BaseDimension::Length), 1);

        // Can't take sqrt of odd-powered dimension
        let cube = Dimension::from_base_power(BaseDimension::Length, 3);
        assert!(cube.sqrt().is_none());
    }

    #[test]
    fn test_force_dimension() {
        let force = standard::force();
        assert_eq!(force.get_exponent(BaseDimension::Mass), 1);
        assert_eq!(force.get_exponent(BaseDimension::Length), 1);
        assert_eq!(force.get_exponent(BaseDimension::Time), -2);
    }

    #[test]
    fn test_clearance_dimension() {
        let cl = standard::clearance();
        // mL/min = L³/T
        assert_eq!(cl.get_exponent(BaseDimension::Length), 3);
        assert_eq!(cl.get_exponent(BaseDimension::Time), -1);
    }

    #[test]
    fn test_concentration_dimension() {
        let conc = standard::concentration();
        // mol/L = N/L³
        assert_eq!(conc.get_exponent(BaseDimension::Amount), 1);
        assert_eq!(conc.get_exponent(BaseDimension::Length), -3);
    }

    #[test]
    fn test_auc_dimension() {
        let auc = standard::auc();
        // (N/L³) × T = N·T·L⁻³
        assert_eq!(auc.get_exponent(BaseDimension::Amount), 1);
        assert_eq!(auc.get_exponent(BaseDimension::Time), 1);
        assert_eq!(auc.get_exponent(BaseDimension::Length), -3);
    }

    #[test]
    fn test_international_units() {
        let iu = standard::international_unit();
        assert!(iu.contains_iu());
        assert!(!standard::mass().contains_iu());

        // IU cannot be converted to mass
        let iu_per_vol = standard::iu_per_volume();
        assert!(iu_per_vol.contains_iu());
    }

    #[test]
    fn test_dimension_display() {
        let velocity = standard::velocity();
        let display = format!("{}", velocity);
        assert!(display.contains("L"));
        assert!(display.contains("T"));
    }
}
